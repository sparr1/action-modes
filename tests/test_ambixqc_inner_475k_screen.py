"""Pinned checkpoint, controlled adaptation dose, evidence and publication gates."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import run_ambixqc_inner_475k_screen as campaign
import publish_ambixqc_inner_475k_screen as publication
from test_ambixqc_backbone_mppi_evaluation import inventory, write


def test_matrix_resolves_four_conditions_and_controls_actor_dose():
    from RL.AMBIXQC import AMBIXQC
    from utils.ambi_research import load_preset_matrix, resolve_preset
    from utils.checkpoint_context import CheckpointContext
    base = campaign.ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_aux_shared_utd2.json"
    context = CheckpointContext(json.loads(base.read_text()), {"env_params": {}}, base)
    matrix = load_preset_matrix(campaign.MATRIX)
    assert matrix["evaluation"]["default_presets"] == list(campaign.SELECTORS)
    assert matrix["evaluation"]["seeds"] == campaign.SEEDS
    for index, selector in enumerate(campaign.SELECTORS):
        resolved = resolve_preset(campaign.MATRIX, selector, matrix, checkpoint_context=context)
        model = object.__new__(AMBIXQC)
        model.env = SimpleNamespace(
            observation_space=gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32),
            action_space=gym.spaces.Box(-1, 1, (21,), dtype=np.float32),
            spec=SimpleNamespace(max_episode_steps=500))
        model.run_params = resolved["algorithm_config"]
        model.experiment_params = {}
        model.custom_params = model.run_params["alg_params"]
        cfg = model._build_cfg({**model.custom_params, "device": "cpu"})
        for key, value in campaign.expected_settings(index).items():
            assert getattr(cfg, key) == value, key
        assert cfg.inner_actor_lr * cfg.inner_rounds == 5e-5
        assert cfg.inner_temperature_lr == cfg.inner_actor_lr
        assert cfg.inner_actor_updates_per_action == cfg.inner_rounds
        assert cfg.inner_critic_updates_per_action == 3 * cfg.inner_rounds
        assert cfg.inner_model_step_budget == 256 * cfg.inner_rounds
        assert cfg.xqc_utd == 2 and cfg.utd == 1


@pytest.mark.parametrize("index", [-1, 4, True, 1.5])
def test_invalid_condition_rejected(index):
    with pytest.raises(ValueError, match="index"):
        campaign.selector_for(index)


def test_only_selected_checkpoint_is_allowed(inventory, monkeypatch):
    row = inventory[2]["checkpoints"][98]
    with pytest.raises(ValueError, match="pinned"):
        campaign.select_checkpoint(inventory[1])
    monkeypatch.setattr(campaign, "CHECKPOINT_SHA", row["sha256"])
    selected = campaign.select_checkpoint(inventory[1])
    assert (selected["cell"], selected["step"]) == ("aux_shared_utd2", 475000)
    assert campaign.INVENTORY_INDEX == 98


def make_bundle(path, index=0, *, seeds=(101, 102), steps=3):
    path.mkdir(parents=True)
    j = campaign.expected_settings(index)["inner_rounds"]
    episodes = [{"seed": seed, "episode_id": f"seed-{seed}", "length": steps,
                 "return": float(seed), "capped": steps < 500} for seed in seeds]
    trace_files = []
    for seed in seeds:
        relative = f"seed-{seed}.jsonl.gz"
        trace_files.append(relative)
        with gzip.open(path / relative, "wt") as stream:
            for decision in range(steps):
                event = {"episode_id": f"seed-{seed}", "decision_index": decision, "phase": "decision",
                         "nonfinite": [], "critic_updates": 3*j, "actor_updates": j, "temperature_updates": j,
                         "metrics": {"decision/inner_model_steps": 256*j,
                                     "decision/inner_reward_scale_delta": 0,
                                     "decision/inner_reward_normalizer_imagined_updates": 0,
                                     "decision/inner_diagnostics_sampled": 1,
                                     "decision/inner_critic_source_aux_return": float(index % 2 == 0),
                                     "decision/inner_horizon_critic_source_aux_return": float(index % 2 == 0),
                                     "decision/inner_critic_target_reward_only": float(index % 2 == 0)}}
                # The real producer omits all routing metrics for pure soft
                # XQC rather than emitting three redundant zero values.
                if index % 2:
                    for key in ("decision/inner_critic_source_aux_return",
                                "decision/inner_horizon_critic_source_aux_return",
                                "decision/inner_critic_target_reward_only"):
                        event["metrics"].pop(key)
                stream.write(json.dumps(event) + "\n")
    result = {"action_rule": "tanh_mean", "outer_state_unchanged": True,
              "outer_updates_before": 9, "outer_updates_after": 9,
              "nonfinite_model_metrics": [], "nonfinite_trace_metrics": [],
              "resolved_config": campaign.expected_settings(index),
              "checkpoint_evaluation_provenance": {"evaluated_semantic_signature": {"inner_actor_bn_mode": "running"}}}
    manifest = {"schema_version": 1, "status": "complete", "code": {"commit": "a"*40, "dirty": False},
                "checkpoint": {"sha256": campaign.CHECKPOINT_SHA}, "protocol": {**campaign.PROTOCOL, "max_steps": steps},
                "runs": [{"selector": campaign.selector_for(index), "status": "complete", "result": result,
                          "episodes": episodes, "trace_files": trace_files}]}
    write(path / "manifest.json", manifest)
    return manifest


@pytest.mark.parametrize("index", range(4))
def test_validation_accepts_exact_work_and_frozen_state(tmp_path, index):
    make_bundle(tmp_path / "bundle", index)
    result = campaign.validate_bundle(tmp_path / "bundle", index, seeds=[101, 102], max_steps=3, source_sha="a"*40)
    assert result["decisions"] == 6
    assert result["actor_updates_per_decision"] == (1 if index < 2 else 4)
    assert len(result["trace_sha256"]) == 2


@pytest.mark.parametrize("problem", ["checkpoint", "bn", "outer", "settings", "protocol", "seeds", "code", "nonfinite"])
def test_validation_rejects_wrong_science(tmp_path, problem):
    path = tmp_path / "bundle"
    saved = make_bundle(path)
    result = saved["runs"][0]["result"]
    if problem == "checkpoint": saved["checkpoint"]["sha256"] = "f"*64
    elif problem == "bn": result["checkpoint_evaluation_provenance"]["evaluated_semantic_signature"]["inner_actor_bn_mode"] = "batch_update"
    elif problem == "outer": result["outer_state_unchanged"] = False
    elif problem == "settings": result["resolved_config"]["inner_critic_source"] = "xqc"
    elif problem == "protocol": saved["protocol"]["controller_seed"] = 9
    elif problem == "seeds": saved["runs"][0]["episodes"].reverse()
    elif problem == "code": saved["code"]["dirty"] = True
    else: result["nonfinite_trace_metrics"] = ["loss"]
    write(path / "manifest.json", saved)
    with pytest.raises(ValueError):
        campaign.validate_bundle(path, 0, seeds=[101, 102], max_steps=3)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "work", "scale", "nonfinite", "route"])
def test_trace_rejects_incomplete_or_invalid_decisions(tmp_path, problem):
    path = tmp_path / "bundle"
    make_bundle(path)
    trace = path / "seed-101.jsonl.gz"
    with gzip.open(trace, "rt") as stream: rows = [json.loads(line) for line in stream]
    if problem == "duplicate": rows.append(deepcopy(rows[0]))
    elif problem == "missing": rows.pop()
    elif problem == "work": rows[0]["critic_updates"] = 4
    elif problem == "scale": rows[0]["metrics"]["decision/inner_reward_scale_delta"] = .1
    elif problem == "route": rows[0]["metrics"]["decision/inner_critic_source_aux_return"] = 0
    else: rows[0]["metrics"]["bad"] = float("nan")
    with gzip.open(trace, "wt") as stream:
        for row in rows: stream.write(json.dumps(row)+"\n")
    with pytest.raises(ValueError):
        campaign.validate_bundle(path, 0, seeds=[101, 102], max_steps=3)


@pytest.mark.parametrize("index,problem", [(0, "missing_return"), (1, "wrong_soft"),
                                         (1, "missing_scale"), (3, "missing_work")])
def test_only_omitted_soft_routing_metrics_default_to_zero(tmp_path, index, problem):
    path = tmp_path / "bundle"
    make_bundle(path, index)
    trace = path / "seed-101.jsonl.gz"
    with gzip.open(trace, "rt") as stream: rows = [json.loads(line) for line in stream]
    metrics = rows[0]["metrics"]
    if problem == "missing_return": metrics.pop("decision/inner_critic_source_aux_return")
    elif problem == "wrong_soft": metrics["decision/inner_critic_source_aux_return"] = 1
    elif problem == "missing_scale": metrics.pop("decision/inner_reward_scale_delta")
    else: metrics.pop("decision/inner_model_steps")
    with gzip.open(trace, "wt") as stream:
        for row in rows: stream.write(json.dumps(row)+"\n")
    with pytest.raises(ValueError):
        campaign.validate_bundle(path, index, seeds=[101, 102], max_steps=3)


def prior_reference(tmp_path, inventory):
    path = tmp_path / "prior"
    saved = make_bundle(path, seeds=campaign.SEEDS, steps=500)
    prior = saved["runs"][0]
    prior.update(selector="controller/prior", kind="episodes", action_rule="tanh_mean",
                 config={"alg": "AMBIXQC/AMBIXQC", "alg_params": {"inner_operator": "none"}})
    prior["result"]["resolved_config"] = {"inner_operator": "none"}
    saved["checkpoint"]["metadata_sha256"] = "b"*64
    write(path / "manifest.json", saved)
    pointer = tmp_path / "reference.json"
    write(pointer, {"schema": campaign.REFERENCE_SCHEMA, "checkpoint_manifest_sha256": campaign.file_sha256(inventory[1]),
                    "checkpoint_sha256": campaign.CHECKPOINT_SHA, "bundle_path": str(path),
                    "manifest_sha256": campaign.file_sha256(path / "manifest.json")})
    return pointer, path, saved


@pytest.mark.parametrize("problem", [None, "hash", "protocol", "checkpoint", "length", "state"])
def test_prior_reference_contract_is_checked_before_reuse(tmp_path, inventory, problem):
    pointer, path, saved = prior_reference(tmp_path, inventory)
    data = json.loads(pointer.read_text())
    if problem == "hash": data["manifest_sha256"] = "e"*64
    elif problem == "protocol": saved["protocol"]["controller_seed"] = 8
    elif problem == "checkpoint": saved["checkpoint"]["sha256"] = "f"*64
    elif problem == "length": saved["runs"][0]["episodes"][0]["length"] = 4
    elif problem == "state": saved["runs"][0]["result"]["outer_state_unchanged"] = False
    if problem not in (None, "hash"):
        write(path / "manifest.json", saved)
        data["manifest_sha256"] = campaign.file_sha256(path / "manifest.json")
    write(pointer, data)
    if problem:
        with pytest.raises(ValueError): campaign.select_reference(pointer, {"metadata_sha256": "b"*64}, inventory[1])
    else:
        assert campaign.select_reference(pointer, {"metadata_sha256": "b"*64}, inventory[1]) == path


def test_only_inner_is_evaluated_and_outputs_never_overwritten(tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    from utils import ambi_benchmark as storage
    row = {"cell": campaign.CELL, "step": campaign.STEP, "path": "checkpoint",
           "sha256": campaign.CHECKPOINT_SHA, "source_run": campaign.banks.SOURCE_RUNS[campaign.CELL]}
    manifest = tmp_path / "inventory.json"
    write(manifest, {})
    monkeypatch.setattr(campaign, "select_checkpoint", lambda *a, **kw: row)
    monkeypatch.setattr(campaign, "resolve_run_map", lambda *a, **kw: {campaign.SELECTORS[0]: "/curve"})
    monkeypatch.setattr(campaign, "select_reference", lambda *a, **kw: tmp_path / "prior")
    monkeypatch.setattr(storage, "code_identity", lambda: {"commit": "a"*40, "dirty": False})
    monkeypatch.setattr(storage, "stage_completed_bundle", lambda *a, **kw: {campaign.SELECTORS[0]: {"status": "queued"}})
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return {"checkpoint_sha256": campaign.CHECKPOINT_SHA}
    monkeypatch.setattr(evaluator, "evaluate_matrix", evaluate)
    monkeypatch.setattr(campaign, "validate_bundle", lambda *a, **kw: {"decisions": 2500})
    out = campaign.run(manifest, 0, tmp_path / "results", reference_index=manifest)
    assert (out / "PASS").is_file()
    assert calls[0]["selectors"] == [campaign.SELECTORS[0]]
    assert calls[0]["reference_bundle"] == tmp_path / "prior"
    assert calls[0]["seeds"] == campaign.SEEDS and calls[0]["max_steps"] == 500
    assert calls[0]["stage_results"] is False
    with pytest.raises(FileExistsError): campaign.run(manifest, 0, tmp_path / "results", reference_index=manifest)


def fake_publisher_setup(tmp_path, monkeypatch, *, index=0, accepted_after=1, wrong=False, acknowledge=True):
    run = tmp_path / "curve"
    (run / "records").mkdir(parents=True)
    write(run / "publication.json", {"records": {}})
    from utils.eval_series_data import planner_identity
    planner = planner_identity(campaign.expected_settings(index), {}, "AMBIXQC/AMBIXQC", "tanh_mean")
    registry = {"run_id": "run", "identity": {"backbone": campaign.banks.SOURCE_RUNS[campaign.CELL],
                  "planner": planner}}
    monkeypatch.setattr(publication, "load_run", lambda _: registry)
    class FakePublisher:
        def __init__(self, *args, **kwargs): self.scans = 0
        def __enter__(self): return self
        def publish_pending(self):
            self.scans += 1
            if accepted_after is not None and self.scans >= accepted_after:
                write(run / "records/record.json", {"selector": campaign.SELECTORS[(index+1)%4 if wrong else index]})
                write(run / "publication.json", {"records": {"record": {"status": "queued",
                    "checkpoint_step": campaign.STEP, "checkpoint_sha256": campaign.CHECKPOINT_SHA}}})
            return {"scans": self.scans}
        def __exit__(self, *args):
            data = json.loads((run / "publication.json").read_text())
            if acknowledge:
                for entry in data["records"].values(): entry["status"] = "published"
            write(run / "publication.json", data)
    clock = [0]
    def sleep(seconds): clock[0] += seconds
    kwargs = {"publisher_factory": FakePublisher, "scheduler": lambda *args: True,
              "monotonic": lambda: clock[0], "sleep": sleep,
              "poll_seconds": 1, "visibility_seconds": 3}
    return run, kwargs


def test_publisher_retries_delayed_final_visibility_and_requires_ack(tmp_path, monkeypatch):
    run, kwargs = fake_publisher_setup(tmp_path, monkeypatch, accepted_after=3)
    result = publication.publish(run, 0, ["123"], **kwargs)
    assert result["accepted"] == result["published"] == 1
    assert (run / "screen-publication-verified.json").is_file()


@pytest.mark.parametrize("index", range(4))
def test_publisher_accepts_real_canonical_planner_identities(tmp_path, monkeypatch, index):
    run, kwargs = fake_publisher_setup(tmp_path, monkeypatch, index=index)
    if index % 2:
        settings = publication.load_run(run)["identity"]["planner"]["settings"]
        assert not {"inner_critic_source", "inner_horizon_critic_source", "inner_critic_target"} & settings.keys()
    result = publication.publish(run, index, ["123"], **kwargs)
    assert result["selector"] == campaign.SELECTORS[index]
    assert result["accepted"] == result["published"] == 1


@pytest.mark.parametrize("problem", ["empty", "wrong", "unacknowledged"])
def test_publisher_cannot_succeed_empty_wrong_or_unacknowledged(tmp_path, monkeypatch, problem):
    run, kwargs = fake_publisher_setup(tmp_path, monkeypatch,
        accepted_after=None if problem == "empty" else 1,
        wrong=problem == "wrong", acknowledge=problem != "unacknowledged")
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        publication.publish(run, 0, ["123"], **kwargs)
    assert not (run / "screen-publication-verified.json").exists()


def test_launchers_require_locked_clean_source_smokes_and_no_gpu_publication():
    path = campaign.ROOT / "slurm/run_ambixqc_inner_475k_screen_oscar.sbatch"
    text = path.read_text()
    for required in ("#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=6", "#SBATCH --mem=32G",
                     "SLURM_ARRAY_TASK_ID < 4", "--verify-smoke-root", "--reference-index",
                     "test_ambixqc_actor_bn.py", "test_ambixqc_actor_bn_identity.py",
                     "LOCK_SHA", "--untracked-files=all", "WANDB_MODE=disabled", "--color=no"):
        assert required in text
    publisher = campaign.ROOT / "slurm/publish_ambixqc_inner_475k_screen_oscar.sbatch"
    assert "--visibility-seconds 180" in publisher.read_text()
    for script in (path, publisher): subprocess.run(["bash", "-n", str(script)], check=True, close_fds=False)
