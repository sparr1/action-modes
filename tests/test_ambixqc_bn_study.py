"""Immutable staged selection, one-axis controls, complete evidence and GPU gates."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import run_ambixqc_bn_study as study
from test_ambixqc_inner_475k_screen import make_bundle, write

SHA = "a" * 40


@pytest.fixture
def selection(tmp_path, monkeypatch):
    import run_ambixqc_bn_probe as probe
    path = tmp_path / "selection.json"
    write(path, {"selected_critic_bn_mode": "running", "source_sha": SHA})
    calls = []
    def validate(path, source_sha=None):
        calls.append((str(path), source_sha))
        value = study.read(path)
        if value["source_sha"] != source_sha:
            raise ValueError("Probe source differs")
        return value
    monkeypatch.setattr(probe, "validate_selection", validate)
    return path, calls


def plan(tmp_path, stage="smoke", **kwargs):
    path = tmp_path / f"{stage}.json"
    value = study.prepare(stage, path, source_sha=SHA, **kwargs)
    return path, value


def test_smoke_covers_all_modes_routes_and_independent_temperature(tmp_path):
    path, value = plan(tmp_path)
    assert study.load_plan(path) == value
    assert len(value["conditions"]) == 6
    assert {(c["critic_bn_mode"], c["route"]) for c in value["conditions"]} == set(
        (mode, route) for mode in study.MODES for route in study.ROUTES)
    assert all(c["rounds"] == 4 and c["actor_lr"] == 1.25e-5
               and c["settings"]["inner_temperature_lr"] == 5e-5 for c in value["conditions"])
    assert study.read(value["matrix"]["path"])["evaluation"]["default_presets"] == []
    # Exact repeat is idempotent; a different proposed payload may not replace it.
    assert study.prepare("smoke", path, source_sha=SHA) == value
    with pytest.raises(FileExistsError):
        study.prepare("smoke", path, source_sha="b" * 40)


def test_stage2_uses_only_validated_alternative_and_fixed_settings(tmp_path, selection):
    path, calls = selection
    output, value = plan(tmp_path, "stage2", selection=path)
    assert calls == [(str(path), SHA)]
    assert study.load_plan(output) == value
    assert len(value["conditions"]) == 4
    assert [c["critic_bn_mode"] for c in value["conditions"]] == ["batch_update"] * 2 + ["running"] * 2
    assert all(c["rounds"] == 1 and c["actor_lr"] == c["settings"]["inner_temperature_lr"] == 5e-5
               for c in value["conditions"])
    write(path, {"selected_critic_bn_mode": "batch_no_update", "source_sha": SHA})
    with pytest.raises(ValueError, match="changed"):
        study.load_plan(output)


@pytest.mark.parametrize("field", ["source_sha", "checkpoint_sha256", "conditions", "matrix"])
def test_modified_plan_fails_digest(tmp_path, field):
    path, value = plan(tmp_path)
    value[field] = None
    write(path, value)
    with pytest.raises(ValueError): study.load_plan(path)


@pytest.mark.parametrize("index", [True, -1, 6, 1.5])
def test_invalid_condition_index_is_rejected(tmp_path, index):
    _, value = plan(tmp_path)
    with pytest.raises(ValueError, match="index"):
        study.select_condition(value, index)


def test_all_conditions_resolve_with_only_actor_rate_changed_at_fixed_temperature(tmp_path):
    from RL.AMBIXQC import AMBIXQC
    from utils.ambi_research import load_preset_matrix, resolve_preset
    from utils.checkpoint_context import CheckpointContext
    base = study.ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_aux_shared_utd2.json"
    context = CheckpointContext(study.read(base), {"env_params": {}}, base)
    cells = [study.condition(mode, route, j, lr) for mode in study.MODES
             for route in study.ROUTES for j in (1, 4) for lr in (1.25e-5, 5e-5)]
    path = tmp_path / "matrix.json"
    write(path, study.matrix_for(cells))
    matrix = load_preset_matrix(path)
    for cell in cells:
        resolved = resolve_preset(path, cell["selector"], matrix, checkpoint_context=context)
        model = object.__new__(AMBIXQC)
        model.env = SimpleNamespace(observation_space=gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32),
            action_space=gym.spaces.Box(-1, 1, (21,), dtype=np.float32), spec=SimpleNamespace(max_episode_steps=500))
        model.run_params = resolved["algorithm_config"]
        model.custom_params = model.run_params["alg_params"]
        cfg = model._build_cfg({**model.custom_params, "device": "cpu"})
        for key, value in study.expected_settings(cell).items():
            assert getattr(cfg, key) == value, key
        assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == cell["rounds"]
        assert cfg.inner_critic_updates_per_action == 3 * cell["rounds"]
        assert cfg.inner_temperature_lr == 5e-5


def complete(tmp_path, plan_path, value, index, monkeypatch, mean=200):
    from utils import ambi_benchmark
    monkeypatch.setattr(ambi_benchmark, "reference_returns", lambda *a: {seed: float(seed) for seed in study.SEEDS})
    cell = value["conditions"][index]
    job = tmp_path / f"job123-task{index}"
    output = job / cell["selector"].split("/")[1]
    bundle = output / "bundle"
    old_index = (0 if cell["route"] == "return_return" else 1) + (2 if cell["rounds"] == 4 else 0)
    saved = make_bundle(bundle, old_index, seeds=value["environment_seeds"], steps=value["max_steps"])
    result = saved["runs"][0]
    result["selector"] = cell["selector"]
    result["result"]["resolved_config"] = study.expected_settings(cell)
    result["result"]["checkpoint_evaluation_provenance"]["evaluated_semantic_signature"]["inner_critic_bn_mode"] = cell["critic_bn_mode"]
    for episode in result["episodes"]:
        episode["return"] = float(mean + episode["seed"] - 103)
        episode["paired_return_delta"] = episode["return"] - episode["seed"]
    write(bundle / "manifest.json", saved)
    inventory = tmp_path / "inventory.json"
    if not inventory.exists(): write(inventory, {})
    reference_index = tmp_path / "reference.json"
    if not reference_index.exists(): write(reference_index, {})
    reference = None if value["stage"] == "smoke" else str(tmp_path / "reference")
    actual = study.screen.validate_bundle(bundle, 0, seeds=value["environment_seeds"],
        max_steps=value["max_steps"], reference_bundle=reference, source_sha=SHA,
        expected_selector=cell["selector"], expected_config=study.expected_settings(cell))
    write(output / "validation.json", {**actual, "study_stage": value["stage"], "plan": study.bind(plan_path),
        "source_sha": SHA, "plan_sha256": value["plan_sha256"], "index": index,
        "inventory": study.bind(inventory), "reference_bundle": reference,
        "reference_index": study.bind(reference_index) if reference else None})
    (output / "PASS").write_text("PASS\n")
    (job / "PASS").write_text("PASS\n")
    write(job / "runtime.json", {"gpu": "test GPU", "torch": "2.3.1", "cuda_device_count": 1})
    if index == 0: (job / "pytest.log").write_text("20 passed\n")
    return output


def test_collect_requires_complete_results_and_stage3_reuses_tie_winner(tmp_path, selection, monkeypatch):
    plan_path, value = plan(tmp_path, "stage2", selection=selection[0])
    results = tmp_path / "results"
    result_path = tmp_path / "stage2-results.json"
    for index in range(3): complete(results, plan_path, value, index, monkeypatch)
    with pytest.raises(ValueError, match="All planned"):
        study.collect(plan_path, results, result_path)
    complete(results, plan_path, value, 3, monkeypatch)
    collected = study.collect(plan_path, results, result_path)
    assert collected["winner"]["index"] == 0  # All returns equal; stable predeclared order.
    assert study.validate_result_index(result_path, source_sha=SHA) == collected
    output, next_stage = plan(tmp_path, "stage3", stage2_results=result_path)
    assert study.load_plan(output) == next_stage
    assert len(next_stage["conditions"]) == 3 and len(next_stage["reused"]) == 1
    assert {(c["rounds"], c["actor_lr"]) for c in next_stage["conditions"]} == {
        (1, 1.25e-5), (4, 1.25e-5), (4, 5e-5)}
    assert next_stage["reused"][0]["result"] == collected["winner"]
    assert all(c["settings"]["inner_temperature_lr"] == c["settings"]["inner_critic_lr"] == 5e-5
               for c in next_stage["conditions"])


def test_winner_is_raw_return_and_corrupt_results_prevent_stage3(tmp_path, selection, monkeypatch):
    plan_path, value = plan(tmp_path, "stage2", selection=selection[0])
    for index, score in enumerate([100, 120, 90, 110]):
        complete(tmp_path / "results", plan_path, value, index, monkeypatch, mean=score)
    result_path = tmp_path / "result-index.json"
    index = study.collect(plan_path, tmp_path / "results", result_path)
    assert index["winner"]["index"] == 1
    entry = index["entries"][0]
    path = Path(entry["output_path"]) / "bundle/seed-101.jsonl.gz"
    path.write_bytes(b"corrupted")
    with pytest.raises((ValueError, OSError)):
        study.prepare("stage3", tmp_path / "stage3.json", source_sha=SHA, stage2_results=result_path)


@pytest.mark.parametrize("problem", ["missing", "wrong_source", "gpu", "pytest", "incomplete_job"])
def test_all_modes_smoke_gate_requires_matching_cuda_evidence(tmp_path, monkeypatch, problem):
    plan_path, value = plan(tmp_path)
    for index in range(6): complete(tmp_path / "smokes", plan_path, value, index, monkeypatch)
    manifest = tmp_path / "smokes/inventory.json"
    gate = study.verify_smokes(tmp_path / "smokes", manifest, SHA)
    assert gate["validated_indices"] == list(range(6))
    job = tmp_path / "smokes/job123-task0"
    if problem == "missing": (job / value["conditions"][0]["selector"].split("/")[1] / "validation.json").unlink()
    elif problem == "wrong_source": SHA_ARG = "b"*40
    elif problem == "gpu": write(job / "runtime.json", {"gpu": "", "torch": "2.3.1", "cuda_device_count": 0})
    elif problem == "pytest": (job / "pytest.log").write_text("1 failed")
    else: (job / "PASS").unlink()
    with pytest.raises(ValueError):
        study.verify_smokes(tmp_path / "smokes", manifest, "b"*40 if problem == "wrong_source" else SHA)


def test_run_maps_cannot_mix_plans_or_reuse_curves(tmp_path, selection, monkeypatch):
    _, value = plan(tmp_path, "stage2", selection=selection[0])
    path = tmp_path / "map.json"
    mapping = {c["selector"]: str(tmp_path / f"registry-{i}") for i, c in enumerate(value["conditions"])}
    data = {"schema": study.MAP_SCHEMA, "plan_sha256": value["plan_sha256"], "runs": mapping}
    write(path, data)
    from utils import ambi_benchmark
    monkeypatch.setattr(ambi_benchmark, "resolve_eval_run_map", lambda selectors, run_map: run_map)
    assert study.run_map(path, value, value["conditions"][0]) == {value["conditions"][0]["selector"]: mapping[value["conditions"][0]["selector"]]}
    data["plan_sha256"] = "b"*64
    write(path, data)
    with pytest.raises(ValueError): study.run_map(path, value, value["conditions"][0])


def test_preset_identity_distinguishes_modes_and_fixed_temperature_without_changing_defaults():
    from utils.eval_series_data import planner_identity
    config = {**study.expected_settings(study.condition("batch_update", "soft_soft", 1, 5e-5))}
    original = dict(config)
    original.pop("inner_critic_bn_mode")
    identities = [planner_identity(c, {}, "AMBIXQC/AMBIXQC", "tanh_mean") for c in (original, config)]
    assert identities[0] == identities[1]
    for mode in ("batch_no_update", "running"):
        altered = planner_identity({**config, "inner_critic_bn_mode": mode}, {}, "AMBIXQC/AMBIXQC", "tanh_mean")
        assert altered != identities[0]
    low = planner_identity({**config, "inner_actor_lr": 1.25e-5}, {}, "AMBIXQC/AMBIXQC", "tanh_mean")
    assert low["settings"]["inner_temperature_lr"] == 5e-5


def test_launcher_requires_exact_source_lock_explicit_plan_and_all_mode_smokes():
    path = study.ROOT / "slurm/run_ambixqc_bn_study_oscar.sbatch"
    text = path.read_text()
    for required in ("AMBIXQC_BN_PLAN", "AMBIXQC_STUDY_STAGE", "verify-smokes", "LOCK_SHA",
                     "--untracked-files=all", "WANDB_MODE=disabled", "test_ambixqc_critic_bn.py",
                     "test_ambixqc_critic_bn_checkpoint.py", "cuda_device_count", "--color=no",
                     "SLURM_ARRAY_TASK_ID < 6", "SLURM_ARRAY_TASK_ID < 4", "SLURM_ARRAY_TASK_ID < 3"):
        assert required in text
    subprocess.run(["bash", "-n", str(path)], check=True, close_fds=False)
