"""Portable trace boundaries, compatible references, and publication cleanup."""

import gzip
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils import ambi_benchmark as storage


CHECKPOINT = {"sha256": "a" * 64, "path": "checkpoint.pt"}


@pytest.fixture(autouse=True)
def _fixed_code_identity(monkeypatch):
    monkeypatch.setattr(storage, "code_identity", lambda: {
        "commit": "fixture", "dirty": False, "diff_sha256": "fixture-diff",
    })


def _resolved(variant="sac"):
    return {
        "selector": f"inner_budget/{variant}",
        "algorithm_config": {
            "alg": "AMBITDMPC2/AMBITDMPC2", "env": "DMControl-v0",
            "alg_params": {"obs": "state", "inner_operator": "none" if variant == "prior" else "sac"},
        },
        "environment": {"id": "DMControl-v0", "params": {"task": "humanoid-walk", "obs": "state"}},
    }


def _protocol():
    return storage.protocol_for(_resolved(), 55, 500)


def _episode(seed=101, value=50.0):
    return {
        "seed": seed, "return": value, "length": 1, "terminated": False,
        "truncated": True, "truncated_by_evaluator": True,
        "control_seconds": 0.01, "model_metrics": {"inner/critic_loss": 0.25},
    }


def _event(**kwargs):
    return {
        "episode_id": "seed-101", "decision_index": 0, "event_index": 0,
        "phase": "critic_update", "round_index": 0, "critic_updates": 1,
        "actor_updates": 0, "temperature_updates": 0,
        "metrics": {"critic_loss": 0.25}, **kwargs,
    }


def _bundle(tmp_path, **kwargs):
    return storage.BenchmarkBundle(tmp_path / "bundle", checkpoint=CHECKPOINT,
                                   protocol=_protocol(), **kwargs)


def _trace_rows(bundle, run):
    return [json.loads(line) for relative in run["trace_files"]
            for line in gzip.decompress((bundle.path / relative).read_bytes()).decode().splitlines()]


def _label_inputs(bootstrap="inner_target", operator="sac"):
    checkpoint = {**CHECKPOINT, "source_run": "rwgao_b-brown-university/ambi/u13m14st",
                  "metadata": {"checkpoint": {"step": 100_000}}}
    config = {"alg_params": {"inner_operator": operator, "inner_rounds": 6,
              "inner_rollouts_per_round": 512, "inner_rollout_horizon": 3,
              "inner_updates_per_round": 3, "inner_batch_size": 512,
              "inner_bootstrap_source": bootstrap, "inner_temperature_mode": "auto",
              "inner_finite_horizon": False}}
    return checkpoint, _protocol(), config


@pytest.mark.parametrize("bootstrap", ["inner_target", "outer_target"])
def test_benchmark_labels_expose_actual_checkpoint_schedule_and_bootstrap(bootstrap):
    checkpoint, protocol, config = _label_inputs(bootstrap)
    original = deepcopy((checkpoint, protocol, config))
    labels = storage.benchmark_run_labels(checkpoint, protocol, config, "episodes",
                                          selector="named_run/d512_4_j6")
    assert labels["name"] == (
        f"humanoid-walk | ckpt 100k | SAC J6 N512 H3 G3 | Q {bootstrap.replace('_', '-')} | episodes"
    )
    assert {"source-run:u13m14st", "checkpoint-step:100000", "controller:sac",
            f"bootstrap:{bootstrap}", "J:6", "N:512", "H:3", "G:3", "schedule:joint",
            "kind:episodes", "preset:named_run/d512_4_j6", "finite-horizon:false",
            "action:tanh_mean"} <= set(labels["tags"])
    assert (checkpoint, protocol, config) == original
    assert storage.benchmark_run_labels(checkpoint, protocol, config, "episodes",
                                         selector="named_run/d512_4_j6") == labels


def test_prior_labels_do_not_describe_inherited_inactive_sac_settings():
    labels = storage.benchmark_run_labels(*_label_inputs(operator="none"), "episodes")
    assert labels["name"] == "humanoid-walk | ckpt 100k | prior only | episodes"
    assert {"controller:prior", "bootstrap:none"} <= set(labels["tags"])
    assert not any(tag.startswith(("J:", "N:", "H:", "G:", "schedule:", "temperature:"))
                   for tag in labels["tags"])
    assert "controller:sac" not in labels["tags"]
    assert "bootstrap:inner_target" not in labels["tags"]
    assert "action:tanh_mean" in labels["tags"]


def test_labels_do_not_invent_finite_horizon_or_action_metadata():
    checkpoint, protocol, config = _label_inputs()
    config["alg_params"]["inner_finite_horizon"] = True
    assert "finite-horizon:true" in storage.benchmark_run_labels(checkpoint, protocol, config, "bank")["tags"]
    config["alg_params"].pop("inner_finite_horizon")
    protocol.pop("action_rule")
    labels = storage.benchmark_run_labels(checkpoint, protocol, config, "bank")
    assert not any(tag.startswith(("finite-horizon:", "action:")) for tag in labels["tags"])


@pytest.mark.parametrize("source", [
    "rwgao_b-brown-university/ambi/u13m14st",
    "https://wandb.ai/rwgao_b-brown-university/ambi/runs/u13m14st?view=overview",
    "u13m14st", {"id": "u13m14st"},
])
def test_source_run_tag_is_stable_across_saved_source_formats(source):
    checkpoint, protocol, config = _label_inputs()
    checkpoint["source_run"] = source
    labels = storage.benchmark_run_labels(checkpoint, protocol, config, "bank")
    assert "source-run:u13m14st" in labels["tags"]
    assert labels["name"].endswith(" | bank")
    assert "kind:bank" in labels["tags"]


def test_missing_step_uses_hash_without_guessing_from_path_or_selector():
    checkpoint, protocol, config = _label_inputs()
    checkpoint.update(metadata=None, path="/checkpoint_500000_steps.pt", source_run=None)
    labels = storage.benchmark_run_labels(checkpoint, protocol, config, "episodes",
                                          selector="misleading/ckpt_1m")
    assert "ckpt aaaaaaaaaaaa" in labels["name"]
    assert not any(tag.startswith(("checkpoint-step:", "source-run:")) for tag in labels["tags"])
    assert "Q inner-target" in labels["name"]


def test_separate_and_transition_schedules_are_not_labeled_joint_updates():
    checkpoint, protocol, config = _label_inputs()
    params = config["alg_params"]
    params.update(inner_updates_per_round=None, inner_critic_updates_per_round=6,
                  inner_actor_updates_per_round=2)
    labels = storage.benchmark_run_labels(checkpoint, protocol, config, "both")
    assert "SAC J6 N512 H3 C6 A2" in labels["name"]
    assert {"schedule:separate", "C:6", "A:2", "kind:both"} <= set(labels["tags"])
    assert not any(tag.startswith("G:") for tag in labels["tags"])
    params.update(inner_critic_updates_per_round=None, inner_actor_updates_per_round=None,
                  inner_steps_per_update=256)
    labels = storage.benchmark_run_labels(checkpoint, protocol, config, "bank")
    assert "update/256 transitions" in labels["name"]
    assert {"schedule:transitions", "steps-per-update:256"} <= set(labels["tags"])


def _legacy_label_inputs(bootstrap="inner_target", critic_total=36):
    checkpoint, protocol, config = _label_inputs(bootstrap)
    config["alg_params"].update(
        inner_rollouts_per_round=None, inner_updates_per_round=None,
        inner_critic_updates_per_round=None, inner_actor_updates_per_round=None,
        inner_model_step_budget=9216, inner_critic_updates_per_action=critic_total,
        inner_actor_updates_per_action=18, inner_temperature_updates_per_action=18,
    )
    return checkpoint, protocol, config


@pytest.mark.parametrize("bootstrap", ["inner_target", "outer_target"])
@pytest.mark.parametrize("critic_total,critic_per_round", [(36, 6), (72, 12)])
def test_legacy_budget_labels_identify_uniform_counts_and_joint_then_critic_order(
    bootstrap, critic_total, critic_per_round,
):
    inputs = _legacy_label_inputs(bootstrap, critic_total)
    original = deepcopy(inputs)
    labels = storage.benchmark_run_labels(*inputs, "bank")
    assert labels["name"] == (
        f"humanoid-walk | ckpt 100k | SAC J6 N512 H3 C{critic_per_round} A3 T3 "
        f"(joint then critic) | Q {bootstrap.replace('_', '-')} | bank"
    )
    assert {"J:6", "N:512", "H:3", f"C:{critic_per_round}", "A:3", "T:3",
            f"C-per-action:{critic_total}", "A-per-action:18", "T-per-action:18",
            "schedule:legacy-total-budget", "update-order:joint-then-critic",
            f"bootstrap:{bootstrap}"} <= set(labels["tags"])
    assert "schedule:separate" not in labels["tags"]
    assert not any(tag.startswith("G:") for tag in labels["tags"])
    assert inputs == original


@pytest.mark.parametrize("component,total", [("critic", 37), ("actor", 19), ("temperature", 19)])
def test_uneven_legacy_allocations_are_labeled_as_totals_not_uniform_round_counts(component, total):
    checkpoint, protocol, config = _legacy_label_inputs()
    config["alg_params"][f"inner_{component}_updates_per_action"] = total
    labels = storage.benchmark_run_labels(checkpoint, protocol, config, "bank")
    assert "N512" in labels["name"]
    assert "(overlapping slots)" in labels["name"]
    assert "joint then critic" not in labels["name"]
    for symbol, name in (("C", "critic"), ("A", "actor"), ("T", "temperature")):
        value = config["alg_params"][f"inner_{name}_updates_per_action"]
        assert f"{symbol}/action{value}" in labels["name"]
        assert f"{symbol}-per-action:{value}" in labels["tags"]
    assert not any(tag.startswith(("C:", "A:", "T:")) for tag in labels["tags"])


@pytest.mark.parametrize("changed", [
    {"inner_rounds": None}, {"inner_rounds": 0}, {"inner_rollout_horizon": None},
    {"inner_model_step_budget": None}, {"inner_model_step_budget": 9217},
])
def test_legacy_rollout_count_is_not_inferred_without_an_exact_budget(changed):
    checkpoint, protocol, config = _legacy_label_inputs()
    config["alg_params"].update(changed)
    labels = storage.benchmark_run_labels(checkpoint, protocol, config, "bank")
    assert not any(tag.startswith("N:") for tag in labels["tags"])
    assert "N512" not in labels["name"]


@pytest.mark.parametrize("critic_total,actor_total,temperature_total,order", [
    (18, 18, 18, "joint"), (18, 36, 18, "overlapping slots"),
    (36, 18, 12, "overlapping slots"),
])
def test_legacy_order_label_respects_all_three_component_totals(
    critic_total, actor_total, temperature_total, order,
):
    checkpoint, protocol, config = _legacy_label_inputs(critic_total=critic_total)
    config["alg_params"].update(inner_actor_updates_per_action=actor_total,
                                inner_temperature_updates_per_action=temperature_total)
    labels = storage.benchmark_run_labels(checkpoint, protocol, config, "bank")
    assert f"({order})" in labels["name"]
    assert "(joint then critic)" not in labels["name"]


def test_new_run_persists_and_publishes_labels_without_changing_groups(tmp_path, monkeypatch):
    calls = []

    def initialize(params, *, default_project, run_name, config):
        calls.append({"params": params, "name": run_name, "config": config})
        return SimpleNamespace(path="entity/ambi-inner-bench/new-run", define_metric=lambda *args, **kwargs: None)

    monkeypatch.setattr("utils.wandb_utils.init_wandb", initialize)
    checkpoint, protocol, config = _label_inputs("outer_target")
    bundle = storage.BenchmarkBundle(tmp_path / "labels", checkpoint=checkpoint, protocol=protocol,
                                    wandb={"project": "ambi-inner-bench", "entity": "entity", "mode": "offline"})
    resolved = {**_resolved(), "selector": "named_run/d512_4_j6_outer_target", "algorithm_config": config}
    run = bundle.start_run(resolved, "episodes")
    expected = storage.benchmark_run_labels(checkpoint, protocol, config, "episodes", selector=resolved["selector"])
    saved = storage.read_json(bundle.path / "manifest.json")["runs"][0]
    assert saved["wandb_name"] == calls[0]["name"] == expected["name"]
    assert saved["wandb_tags"] == calls[0]["params"]["wandb_tags"] == expected["tags"]
    assert calls[0]["params"]["wandb_group"] == f"{'a' * 12}-named_run__d512_4_j6_outer_target"
    assert calls[0]["config"]["inner_config"] == config
    assert run["wandb_path"] == "entity/ambi-inner-bench/new-run"


def test_atomic_outputs_preserve_existing_data_and_clean_temporary_files(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    storage.atomic_json(target, {"first": True})
    with pytest.raises(FileExistsError):
        storage.atomic_json(target, {"replacement": True})
    assert storage.read_json(target) == {"first": True}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.json"]

    def broken_replace(*args):
        raise OSError("simulated atomic publish failure")
    monkeypatch.setattr(storage.os, "replace", broken_replace)
    with pytest.raises(OSError, match="publish failure"):
        storage.atomic_json(target, {"replacement": True}, overwrite=True)
    assert storage.read_json(target) == {"first": True}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.json"]


def test_trace_roundtrip_distinguishes_nonfinite_from_missing_and_retains_failed_run(tmp_path):
    bundle = _bundle(tmp_path)
    run = bundle.start_run(_resolved(), "episodes")
    event = _event(metrics={"critic_loss": float("inf"), "actor_loss": float("nan"), "missing": None})
    bundle.episode(run, _episode(), [event])
    error = RuntimeError("later episode failed")
    bundle.finish_run(run, error=error)
    bundle.finish(error=error)

    manifest = storage.read_json(bundle.path / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["episodes"][0]["return"] == 50.0
    row = _trace_rows(bundle, run)[0]
    assert row["metrics"] == {"critic_loss": None, "actor_loss": None, "missing": None}
    assert row["nonfinite"] == {"critic_loss": "inf", "actor_loss": "nan"}
    assert run["nonfinite_trace_metrics"] == {"critic_loss": 1, "actor_loss": 1}
    assert np.isinf(event["metrics"]["critic_loss"])
    assert np.isnan(event["metrics"]["actor_loss"])


def test_actual_storage_bundle_loads_in_report_without_schema_translation(tmp_path):
    from report_ambi_benchmark import load_bundles

    bundle = _bundle(tmp_path)
    run = bundle.start_run(_resolved(), "episodes")
    bundle.episode(run, _episode(), [_event()])
    bundle.finish_run(run)
    bundle.finish()
    report = load_bundles([bundle.path])
    assert report["runs"][0]["traces"][0]["metrics"]["critic_loss"] == [0.25]
    semantic = report["metric_catalog"]["critic_loss"]
    assert semantic["preferred_axis"] == "critic_updates"
    assert "before" in semantic["definition"].lower() or "pre-update" in semantic["definition"].lower()


def test_shared_bank_hash_validation_protocol_matching_and_order_independent_seeds(tmp_path):
    roots = [storage.capture_root(np.array([0.5, 1.0], dtype=np.float32), seed, decision, 0.0)
             for seed, decision in ((101, 0), (102, 100))]
    bank = storage.make_bank(CHECKPOINT["sha256"], _protocol(), roots, complete=True)
    path = tmp_path / "bank.json"
    storage.atomic_json(path, bank)
    assert storage.load_bank(path, CHECKPOINT["sha256"], _protocol()) == bank
    seeds = {root["root_id"]: storage.solver_seed(55, "root", root["root_id"], 0) for root in roots}
    assert seeds == {root["root_id"]: storage.solver_seed(55, "root", root["root_id"], 0)
                     for root in reversed(roots)}
    assert seeds[roots[0]["root_id"]] != storage.solver_seed(55, "root", roots[0]["root_id"], 1)
    assert seeds[roots[0]["root_id"]] != storage.solver_seed(55, "probe", roots[0]["root_id"], 0)

    with pytest.raises(ValueError, match="checkpoint"):
        storage.load_bank(path, "different-checkpoint", _protocol())
    wrong_protocol = _protocol()
    wrong_protocol["environment"]["params"]["task"] = "walker-walk"
    with pytest.raises(ValueError, match="protocol"):
        storage.load_bank(path, CHECKPOINT["sha256"], wrong_protocol)
    bank["roots"][0]["observation"][0] = 0.75
    storage.atomic_json(path, bank, overwrite=True)
    with pytest.raises(ValueError, match="corrupted"):
        storage.load_bank(path, CHECKPOINT["sha256"], _protocol())


@pytest.mark.parametrize("invalid", ["duplicate", "shape", "dtype", "incomplete"])
def test_shared_bank_rejects_invalid_observations_even_with_valid_hash(tmp_path, invalid):
    root = storage.capture_root(np.array([1.0, 2.0], dtype=np.float32), 101, 0, 0.0)
    roots = [root]
    if invalid == "duplicate":
        roots.append(deepcopy(root))
    elif invalid == "shape":
        root["shape"] = [1]
    elif invalid == "dtype":
        root["dtype"] = "float64"
    bank = storage.make_bank(CHECKPOINT["sha256"], _protocol(), roots, complete=invalid != "incomplete")
    path = tmp_path / "bank.json"
    storage.atomic_json(path, bank)
    with pytest.raises(ValueError):
        storage.load_bank(path, CHECKPOINT["sha256"], _protocol())


def test_prior_reference_checks_protocol_and_pairs_by_seed(tmp_path):
    bundle = _bundle(tmp_path)
    run = bundle.start_run(_resolved("prior"), "episodes")
    bundle.episode(run, _episode(102, 70.0), [])
    bundle.episode(run, _episode(101, 50.0), [])
    bundle.finish_run(run)
    bundle.finish()
    assert storage.reference_returns(bundle.path, CHECKPOINT["sha256"], _protocol()) == {102: 70.0, 101: 50.0}
    for key, changed in (("max_steps", 200), ("controller_seed", 56), ("action_rule", "sample")):
        protocol = {**_protocol(), key: changed}
        with pytest.raises(ValueError, match="protocol"):
            storage.reference_returns(bundle.path, CHECKPOINT["sha256"], protocol)
    with pytest.raises(ValueError, match="checkpoint"):
        storage.reference_returns(bundle.path, "different-checkpoint", _protocol())
    protocol_with_bank = {**_protocol(), "root_bank_id": "screen-bank"}
    assert storage.reference_returns(bundle.path, CHECKPOINT["sha256"], protocol_with_bank)[101] == 50.0


def test_episode_deltas_are_explicit_seed_matches(tmp_path):
    bundle = _bundle(tmp_path, reference={102: 70.0, 101: 50.0})
    run = bundle.start_run(_resolved(), "episodes")
    bundle.episode(run, _episode(101, 65.0), [])
    bundle.episode(run, _episode(103, 80.0), [])
    assert run["episodes"][0]["paired_return_delta"] == 15.0
    assert "paired_return_delta" not in run["episodes"][1]
    bundle.finish_run(run)
    bundle.finish()


def _fake_wandb(monkeypatch, *, artifact_error=None, finish_error=None, define_error=None):
    events, runs = [], []

    class Artifact:
        def __init__(self, name, type):
            self.name, self.type, self.files = name, type, []

        def add_file(self, path, name):
            self.files.append((path, name))

    class Run:
        def __init__(self, name):
            self.name, self.path = name, f"entity/project/{name}"
            self.summary, self.logs, self.artifacts = {}, [], []
            self.closed = False
            # The generic helper initially associates both metric families
            # with cumulative environment steps. Benchmark setup overrides it.
            self.definitions = {
                "episode/*": {"step_metric": "env_step"},
                "eval/*": {"step_metric": "env_step"},
            }

        def define_metric(self, name, **kwargs):
            assert not self.closed
            if define_error:
                raise define_error
            self.definitions.setdefault(name, {}).update(kwargs)

        def log(self, value):
            assert not self.closed
            self.logs.append(value)

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)
            events.append((self.name, "artifact"))
            if artifact_error:
                raise artifact_error

        def finish(self, exit_code):
            self.closed = True
            self.exit_code = exit_code
            events.append((self.name, "finish"))
            if finish_error:
                raise finish_error

    def initialize(params, *, default_project, run_name, config):
        assert not any(not run.closed for run in runs), "W&B runs must be finished sequentially"
        assert params["wandb_project"] == "ambi-inner-bench"
        run = Run(run_name)
        events.append((run_name, "init"))
        runs.append(run)
        return run

    monkeypatch.setattr("utils.wandb_utils.init_wandb", initialize)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=Artifact))
    return runs, events


def test_wandb_runs_are_sequential_and_artifacts_independently_readable(tmp_path, monkeypatch):
    from report_ambi_benchmark import load_bundles

    remotes, events = _fake_wandb(monkeypatch)
    options = {"project": "ambi-inner-bench", "entity": "entity", "mode": "offline"}
    bundle = _bundle(tmp_path, wandb=options)
    for variant in ("prior", "sac"):
        run = bundle.start_run(_resolved(variant), "episodes")
        bundle.episode(run, _episode(), [_event()])
        bundle.finish_run(run)
    bundle.finish()
    assert [phase for _, phase in events] == ["init", "artifact", "finish", "init", "artifact", "finish"]
    assert all(remote.exit_code == 0 for remote in remotes)
    for remote in remotes:
        assert remote.logs[0]["episode/return"] == 50.0
        artifact = remote.artifacts[0]
        destination = tmp_path / remote.name
        for source, name in artifact.files:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source).read_bytes())
        report = load_bundles([destination])
        assert len(report["runs"]) == 1
        assert report["runs"][0]["status"] == "complete"


def test_episode_metric_definitions_hide_seed_observations_without_changing_history(tmp_path, monkeypatch):
    remotes, _ = _fake_wandb(monkeypatch)
    bundle = _bundle(tmp_path, reference={101: 40.0, 102: 80.0},
                     wandb={"project": "ambi-inner-bench", "entity": "entity", "mode": "offline"})
    run = bundle.start_run(_resolved(), "episodes")
    remote = remotes[0]
    assert remote.definitions["episode/*"] == {"step_metric": "episode/index", "hidden": True}
    assert remote.definitions["eval/paired_return_delta"] == {
        "step_metric": "episode/index", "hidden": True,
    }
    assert remote.definitions["episode/index"] == {"hidden": True}
    assert remote.definitions["env_step"] == {"hidden": True}
    assert remote.definitions["eval/*"] == {"step_metric": "env_step"}
    bundle.episode(run, {**_episode(101, 50.0), "length": 3}, [])
    bundle.episode(run, {**_episode(102, 60.0), "length": 5}, [])
    assert [row["episode/index"] for row in remote.logs] == [1, 2]
    assert [row["episode/seed"] for row in remote.logs] == [101, 102]
    assert [row["episode/return"] for row in remote.logs] == [50.0, 60.0]
    assert [row["env_step"] for row in remote.logs] == [3, 8]
    assert [row["eval/paired_return_delta"] for row in remote.logs] == [10.0, -20.0]
    bundle.finish_run(run)
    bundle.finish()


def test_metric_definition_failure_keeps_remote_registered_for_cleanup(tmp_path, monkeypatch):
    error = RuntimeError("benchmark metric definition failed")
    remotes, _ = _fake_wandb(monkeypatch, define_error=error)
    bundle = _bundle(tmp_path, wandb={"project": "ambi-inner-bench", "entity": "entity", "mode": "offline"})
    with pytest.raises(RuntimeError, match="metric definition failed"):
        bundle.start_run(_resolved(), "episodes")
    bundle.finish(error=error)
    assert remotes[0].closed and remotes[0].exit_code == 1
    assert not bundle.remote_runs
    assert storage.read_json(bundle.path / "manifest.json")["runs"][0]["status"] == "failed"


def test_wandb_publication_failure_closes_as_failed_and_retains_local_data(tmp_path, monkeypatch):
    publication_error = RuntimeError("artifact upload failed")
    remotes, _ = _fake_wandb(monkeypatch, artifact_error=publication_error)
    bundle = _bundle(tmp_path, wandb={"project": "ambi-inner-bench", "entity": "entity", "mode": "offline"})
    run = bundle.start_run(_resolved(), "episodes")
    bundle.episode(run, _episode(), [_event()])
    with pytest.raises(RuntimeError, match="artifact upload failed"):
        bundle.finish_run(run)
    assert remotes[0].closed and remotes[0].exit_code == 1
    assert not bundle.remote_runs
    manifest = storage.read_json(bundle.path / "manifest.json")
    assert manifest["runs"][0]["episodes"][0]["return"] == 50.0
    assert len(_trace_rows(bundle, run)) == 1


def test_wandb_finish_failure_does_not_mask_publication_error(tmp_path, monkeypatch):
    publication_error = RuntimeError("artifact upload failed")
    _fake_wandb(monkeypatch, artifact_error=publication_error, finish_error=OSError("finish failed"))
    bundle = _bundle(tmp_path, wandb={"project": "ambi-inner-bench", "entity": "entity", "mode": "offline"})
    run = bundle.start_run(_resolved(), "episodes")
    with pytest.raises(RuntimeError, match="artifact upload failed") as caught:
        bundle.finish_run(run)
    assert caught.value is publication_error
    assert any("finish failed" in note for note in getattr(caught.value, "__notes__", []))


def test_failed_wandb_initialization_marks_local_run_failed(tmp_path, monkeypatch):
    initialization_error = RuntimeError("W&B initialization failed")

    def initialize(*args, **kwargs):
        raise initialization_error

    monkeypatch.setattr("utils.wandb_utils.init_wandb", initialize)
    bundle = _bundle(tmp_path, wandb={"project": "ambi-inner-bench", "entity": "entity", "mode": "offline"})
    with pytest.raises(RuntimeError, match="initialization failed"):
        bundle.start_run(_resolved(), "episodes")
    bundle.finish(error=initialization_error)
    manifest = storage.read_json(bundle.path / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["status"] == "failed"
    assert "initialization failed" in manifest["runs"][0]["error"]
