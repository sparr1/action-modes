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
_CODE_IDENTITY = storage.code_identity


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


def test_new_run_persists_labels_without_initializing_wandb(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.wandb_utils.init_wandb", lambda *a, **k: pytest.fail("GPU W&B initialization"))
    checkpoint, protocol, config = _label_inputs("outer_target")
    bundle = storage.BenchmarkBundle(tmp_path / "labels", checkpoint=checkpoint, protocol=protocol)
    resolved = {**_resolved(), "selector": "named_run/d512_4_j6_outer_target", "algorithm_config": config}
    run = bundle.start_run(resolved, "episodes")
    expected = storage.benchmark_run_labels(checkpoint, protocol, config, "episodes", selector=resolved["selector"])
    saved = storage.read_json(bundle.path / "manifest.json")["runs"][0]
    assert saved["wandb_name"] == expected["name"]
    assert saved["wandb_tags"] == expected["tags"]
    assert "wandb_path" not in run


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


def _fake_series(monkeypatch, *, error=None):
    calls = []
    def stage(run_dir, path, **kwargs):
        calls.append((run_dir, Path(path), kwargs, storage.read_json(path)))
        if error:
            raise error
    monkeypatch.setitem(sys.modules, "utils.eval_series", SimpleNamespace(
        stage_result=stage, load_run=lambda path: {"run_id": Path(path).name}))
    monkeypatch.setattr("utils.wandb_utils.init_wandb", lambda *a, **k: pytest.fail("GPU W&B initialization"))
    return calls


def test_finished_bundle_queues_only_after_all_results_are_durable(tmp_path, monkeypatch):
    calls = _fake_series(monkeypatch)
    mapping = {"inner_budget/prior": str(tmp_path / "prior"), "inner_budget/sac": str(tmp_path / "sac")}
    bundle = _bundle(tmp_path, eval_run_map=mapping)
    for variant in ("prior", "sac"):
        run = bundle.start_run(_resolved(variant), "episodes")
        bundle.episode(run, _episode(), [_event()])
        bundle.finish_run(run, result={"outer_state_unchanged": True})
        assert calls == []
    bundle.finish()
    assert len(calls) == 2
    assert all(call[3]["status"] == "complete" for call in calls)
    assert [call[2]["selector"] for call in calls] == list(mapping)
    assert all(call[2]["format"] == "ambi-bundle" for call in calls)
    from report_ambi_benchmark import load_bundles
    assert len(load_bundles([bundle.path])["runs"]) == 2


def test_staging_failure_retains_complete_science_and_trace_data(tmp_path, monkeypatch):
    _fake_series(monkeypatch, error=OSError("registry unavailable"))
    bundle = _bundle(tmp_path, eval_run_map={"inner_budget/sac": str(tmp_path / "series")})
    run = bundle.start_run(_resolved(), "episodes")
    bundle.episode(run, _episode(), [_event()])
    bundle.finish_run(run, result={"outer_state_unchanged": True})
    bundle.finish()
    manifest = storage.read_json(bundle.path / "manifest.json")
    assert manifest["status"] == manifest["runs"][0]["status"] == "complete"
    assert manifest["runs"][0]["episodes"][0]["return"] == 50
    assert len(_trace_rows(bundle, run)) == 1
    publication = storage.read_json(bundle.path / ".series-staging.json")
    assert publication["inner_budget/sac"]["status"] == "failed"
    assert "registry unavailable" in publication["inner_budget/sac"]["error"]


def test_later_config_failure_still_queues_completed_config(tmp_path, monkeypatch):
    calls = _fake_series(monkeypatch)
    bundle = _bundle(tmp_path, eval_run_map={key: str(tmp_path / key.replace("/", "_"))
                                           for key in ("inner_budget/prior", "inner_budget/sac")})
    prior = bundle.start_run(_resolved("prior"), "episodes")
    bundle.episode(prior, _episode(), [])
    bundle.finish_run(prior, result={"outer_state_unchanged": True})
    failed = bundle.start_run(_resolved(), "episodes")
    bundle.finish(error=RuntimeError("second planner failed"))
    assert len(calls) == 1 and calls[0][2]["selector"] == "inner_budget/prior"
    assert failed["status"] == "failed" and prior["status"] == "complete"


def test_explicit_selection_is_required_before_creating_output(tmp_path, monkeypatch):
    _fake_series(monkeypatch)
    with pytest.raises(ValueError, match="explicit"):
        _bundle(tmp_path, wandb={"project": "test"})
    assert not (tmp_path / "bundle").exists()
    with pytest.raises(ValueError, match="explicit"):
        storage.resolve_eval_run_map(["a"], wandb=True)
    with pytest.raises(ValueError, match="exactly one"):
        storage.resolve_eval_run_map(["a", "b"], run_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="every selected"):
        storage.resolve_eval_run_map(["a", "b"], run_map={"a": "one"})
    with pytest.raises(ValueError, match="distinct"):
        storage.resolve_eval_run_map(["a", "b"], run_map={"a": "one", "b": "one"})
    assert storage.resolve_eval_run_map(["a"], run_dir=tmp_path / "run") == {"a": str(tmp_path / "run")}


def _xqc_resolved(operator="xqc"):
    resolved = _resolved("prior" if operator == "none" else "xqc")
    resolved["algorithm_config"].update(alg="AMBIXQC/AMBIXQC")
    resolved["algorithm_config"]["alg_params"].update(
        inner_operator=operator, inner_rounds=2, inner_rollouts_per_round=32,
        inner_rollout_horizon=3, inner_updates_per_round=4, xqc_policy_delay=3,
        inner_reward_normalization="frozen_real_scale", inner_batch_size=64,
    )
    return resolved


def test_xqc_labels_and_decision_bundle_use_measured_delayed_counts_and_reward_units(tmp_path):
    from report_ambi_benchmark import load_bundles, render_html

    bundle = _bundle(tmp_path, reference={101: 40.0})
    run = bundle.start_run(_xqc_resolved(), "episodes")
    assert "XQC J2 N32 H3 G4 policy delay 3" in run["wandb_name"]
    assert "schedule:xqc-slots" in run["wandb_tags"]
    assert not any(tag.startswith("bootstrap:") for tag in run["wandb_tags"])
    # G is critic slots, not accepted actor/temperature updates. The second
    # action is intentionally a smaller actual solve to forbid static totals.
    events = [_event(phase="decision", decision_index=index, round_index=2,
                     critic_updates=critic, actor_updates=actor, temperature_updates=actor,
                     metrics={"decision/reward": 25.0, "decision/inner_q1_mean": 0.5,
                              "decision/inner_behavior_reward_sum_mean": 1.5,
                              "decision/inner_actor_loss": -0.2,
                              "decision/inner_critic_loss": 0.7,
                              "decision/control_seconds": 0.01})
              for index, critic, actor in ((0, 8, 3), (1, 4, 2))]
    bundle.episode(run, {**_episode(), "length": 2}, events)
    bundle.finish_run(run, result={})
    bundle.finish()
    assert run["actual_optimizer_steps"] == {"critic": 12, "actor": 5, "temperature": 5}
    assert "completed episodes: totalC12 totalA5 totalT5" in run["display_name"]
    assert run["episodes"][0]["paired_return_delta"] == 10
    assert len(_trace_rows(bundle, run)) == 2
    catalog = bundle.manifest["metric_catalog"]
    assert catalog["decision/reward"]["unit"] == "raw_environment_reward"
    assert catalog["decision/inner_q1_mean"]["unit"] == "normalized_value"
    assert catalog["decision/inner_behavior_reward_sum_mean"]["unit"] == "raw_predicted_reward"
    assert catalog["decision/inner_actor_loss"]["unit"] == "normalized_objective"
    assert catalog["decision/inner_critic_loss"]["unit"] == "cross_entropy"
    assert all(semantic["preferred_axis"] == "decision_index" for semantic in catalog.values())
    data = load_bundles([bundle.path])
    assert data["runs"][0]["diagnostic_capabilities"] == {
        "decision_metrics": True, "optimizer_traces": False, "shared_observation_probes": False,
    }
    html = render_html(data)
    assert '<option value="decision_index">' in html
    assert '<option value="bank">' not in html
    assert '<option value="actor_updates">' not in html
    assert "Shared-root overlays require" not in html


def test_xqc_completed_episode_remains_complete_after_later_failure_and_reference_reuse(tmp_path):
    bundle = storage.BenchmarkBundle(tmp_path / "prior", checkpoint=CHECKPOINT, protocol=_protocol())
    prior = bundle.start_run(_xqc_resolved("none"), "episodes")
    bundle.episode(prior, _episode(), [_event(phase="decision", critic_updates=0,
        metrics={"decision/inner_q1_mean": None, "decision/reward": 50.0})])
    bundle.finish_run(prior)
    bundle.finish()
    reference = storage.reference_returns(bundle.path, CHECKPOINT["sha256"], _protocol())
    candidate = storage.BenchmarkBundle(tmp_path / "inner", checkpoint=CHECKPOINT,
                                       protocol=_protocol(), reference=reference)
    run = candidate.start_run(_xqc_resolved(), "episodes")
    candidate.episode(run, _episode(value=60.0), [_event(phase="decision", critic_updates=8,
        actor_updates=3, temperature_updates=3,
        metrics={"decision/inner_q1_mean": float("nan"), "decision/reward": 60.0})])
    run["episodes"].append({**_episode(seed=102, value=20.0), "episode_id": "seed-102",
                           "status": "failed", "actual_optimizer_steps": {
                               "critic": 4, "actor": 2, "temperature": 2,
                           }})
    candidate.write_trace(run, "seed-102-partial", [_event(
        episode_id="seed-102", phase="decision", critic_updates=4,
        actor_updates=2, temperature_updates=2, metrics={"decision/reward": 20.0},
    )])
    error = RuntimeError("second episode failed")
    candidate.finish_run(run, error=error)
    candidate.finish(error=error)
    from report_ambi_benchmark import load_bundles
    report = load_bundles([bundle.path, candidate.path])
    saved = report["runs"][1]
    assert saved["status"] == "failed"
    assert saved["episodes"][0]["status"] == "complete"
    assert saved["episodes"][0]["paired_return_delta"] == 10.0
    assert saved["traces"][0]["nonfinite"]["decision/inner_q1_mean"] == ["nan"]
    assert saved["actual_optimizer_steps"] == {"critic": 8, "actor": 3, "temperature": 3}
    assert saved["actual_optimizer_steps_scope"] == "completed_episodes"
    assert "completed episodes: totalC8 totalA3 totalT3" in saved["label"]
    assert len(saved["traces"]) == 2  # Partial work still has its measured decision row.


def test_code_identity_uses_posix_spawn_compatible_git_invocation(monkeypatch):
    calls = []
    monkeypatch.setattr(storage.shutil, "which", lambda name: "/usr/bin/git")

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return b"abc123\n" if command[3] == "rev-parse" else b""

    monkeypatch.setattr(storage.subprocess, "check_output", check_output)
    result = _CODE_IDENTITY()
    assert result["commit"] == "abc123"
    assert result["dirty"] is False
    assert len(calls) == 3
    for command, kwargs in calls:
        assert Path(command[0]).is_absolute()
        assert command[1:3] == ["-C", str(Path(storage.__file__).resolve().parents[1])]
        assert kwargs["close_fds"] is False
        assert "cwd" not in kwargs


def _mppi_resolved():
    resolved = _xqc_resolved("none")
    resolved["selector"] = "controller/mppi"
    resolved["algorithm_config"]["evaluation_controller"] = {"type": "mppi", "params": {
        "horizon": 3, "iterations": 6, "num_samples": 512, "num_elites": 64,
        "num_pi_trajs": 24, "min_std": 0.05, "max_std": 2.0, "temperature": 0.5,
    }}
    return resolved


def _mppi_controller():
    return {"type": "mppi", "settings": {
        **_mppi_resolved()["algorithm_config"]["evaluation_controller"]["params"], "effective_iterations": 8,
    }, "protocol": {
        "action_rule": storage.MPPI_ACTION_RULE, "terminal_value_source": "online_xqc_twin_mean",
        "terminal_value_units": "normalized_xqc_soft_q_times_frozen_real_reward_scale", "reward_scale": 2.5,
    }}


def test_mppi_bundle_reuses_old_prior_reference_and_reports_actual_search(tmp_path):
    from report_ambi_benchmark import load_bundles, render_html

    prior = storage.BenchmarkBundle(tmp_path / "prior", checkpoint=CHECKPOINT, protocol=_protocol())
    prior_run = prior.start_run(_xqc_resolved("none"), "episodes")
    prior.episode(prior_run, _episode(), [_event(phase="decision", critic_updates=0,
                                               metrics={"decision/reward": 50.0})])
    prior.finish_run(prior_run)
    prior.finish()
    # Existing bundles lack explicit candidate-action metadata and still pair.
    prior.manifest.pop("protocol_semantics")
    prior_run.pop("action_rule")
    prior.save()
    references = storage.reference_returns(prior.path, CHECKPOINT["sha256"], _protocol())
    bundle = _bundle(tmp_path, reference=references)
    run = bundle.start_run(_mppi_resolved(), "episodes")
    assert storage.run_controller_type(run) == "mppi"
    assert "configured iterations 6" in run["wandb_name"]
    bundle.set_evaluation_controller(run, _mppi_controller())
    assert "MPPI H3 N512 E64 pi24 J8" in run["wandb_name"]
    assert "weighted elite" in run["wandb_name"]
    assert not any(tag.startswith(("G:", "policy-delay:", "bootstrap:")) for tag in run["wandb_tags"])
    assert {"J:8", "configured-iterations:6", "C:0", "A:0", "T:0"} <= set(run["wandb_tags"])
    bundle.episode(run, _episode(value=60), [_event(phase="decision", critic_updates=0, metrics={
        "decision/reward": 60.0, "decision/planner_value_mean": 1.25,
        "decision/inner_model_steps": 12336, "decision/planner_candidate_model_steps": 12288,
    })])
    bundle.finish_run(run)
    bundle.finish()
    assert run["episodes"][0]["paired_return_delta"] == 10
    assert run["actual_optimizer_steps"] == {"critic": 0, "actor": 0, "temperature": 0}
    with pytest.raises(ValueError, match="exactly one completed prior"):
        storage.reference_returns(bundle.path, CHECKPOINT["sha256"], _protocol())
    data = load_bundles([prior.path, bundle.path])
    assert data["runs"][0]["action_rule"] == "tanh_mean"
    assert data["runs"][1]["action_rule"] == storage.MPPI_ACTION_RULE
    assert data["metric_catalog"]["decision/planner_value_mean"]["unit"] == "raw_return_score"
    assert data["metric_catalog"]["decision/reward"]["unit"] == "raw_environment_reward"
    assert data["runs"][1]["traces"][0]["metrics"]["decision/inner_model_steps"] == [12336]
    html = render_html(data)
    assert "Executed action rule" in html
    assert '<option value="bank">' not in html
    assert '<option value="actor_updates">' not in html


@pytest.mark.parametrize("change,match", [
    (lambda c: c["settings"].update(num_samples=256), "authored configuration"),
    (lambda c: c["protocol"].update(action_rule="tanh_mean"), "action rule"),
    (lambda c: c["protocol"].update(reward_scale=float("nan")), "finite positive"),
])
def test_mppi_runtime_metadata_must_match_authored_controller(tmp_path, change, match):
    bundle = _bundle(tmp_path)
    run = bundle.start_run(_mppi_resolved(), "episodes")
    controller = _mppi_controller()
    change(controller)
    with pytest.raises(ValueError, match=match):
        bundle.set_evaluation_controller(run, controller)
    assert "evaluation_controller" not in run


@pytest.mark.parametrize("change,match", [
    (lambda r: r.update(action_rule="tanh_mean"), "action rule"),
    (lambda r: r["evaluation_controller"]["settings"].update(horizon=4), "authored configuration"),
    (lambda r: r["evaluation_controller"]["protocol"].update(reward_scale=1.0), "controller hash"),
])
def test_report_rejects_misrecorded_mppi_controller(tmp_path, change, match):
    from report_ambi_benchmark import load_bundles

    bundle = _bundle(tmp_path)
    run = bundle.start_run(_mppi_resolved(), "episodes")
    bundle.set_evaluation_controller(run, _mppi_controller())
    bundle.finish_run(run)
    bundle.finish()
    change(run)
    bundle.save()
    with pytest.raises(ValueError, match=match):
        load_bundles([bundle.path])


def test_mppi_runtime_settings_are_saved_without_remote_access(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.wandb_utils.init_wandb", lambda *a, **k: pytest.fail("GPU W&B initialization"))
    bundle = _bundle(tmp_path)
    run = bundle.start_run(_mppi_resolved(), "episodes")
    controller = _mppi_controller()
    bundle.set_evaluation_controller(run, controller)
    assert "J8" in run["wandb_name"]
    assert "action:weighted_elite_gumbel_no_execution_noise" in run["wandb_tags"]
    assert run["evaluation_controller"] == controller


def test_xqc_outer_terminal_bootstrap_has_distinct_labels_without_changing_native_defaults():
    native = _xqc_resolved()["algorithm_config"]
    standard = storage.benchmark_run_labels(CHECKPOINT, _protocol(), native, "episodes")
    outer = deepcopy(native)
    outer["alg_params"]["inner_terminal_bootstrap"] = "outer"
    labels = storage.benchmark_run_labels(CHECKPOINT, _protocol(), outer, "episodes")
    assert "outer terminal bootstrap" not in standard["name"]
    assert "outer terminal bootstrap" in labels["name"]
    assert {"terminal-bootstrap:outer", "terminal-policy:frozen-outer", "terminal-q:online-outer", "terminal-alpha:inner"} <= set(labels["tags"])
    metrics = storage.decision_metric_catalog(["decision/inner_outer_terminal_bootstrap_rows", "decision/inner_outer_terminal_q_evaluations"], xqc=True)
    assert all(item["unit"] == "count" for item in metrics.values())


def test_xqc_step_schedule_labels_and_measurements_are_truthful():
    config = _xqc_resolved()["algorithm_config"]
    config["alg_params"].update(inner_terminal_bootstrap="outer")
    original = storage.benchmark_run_labels(CHECKPOINT, _protocol(), config, "episodes")
    assert "round updates" in original["name"]
    config["alg_params"].update(inner_update_timing="step",inner_policy_delay=1)
    stepped = storage.benchmark_run_labels(CHECKPOINT, _protocol(), config, "episodes")
    assert "step updates" in stepped["name"] and "outer terminal bootstrap" in stepped["name"]
    assert "update-timing:step" in stepped["tags"]
    assert "policy delay 1" in stepped["name"] and config["alg_params"]["xqc_policy_delay"] == 3
    metrics = storage.decision_metric_catalog(["decision/inner_update_timing_step",
        "decision/inner_updates_per_rollout_step", "decision/inner_collection_steps"], xqc=True)
    assert metrics["decision/inner_update_timing_step"]["unit"] == "indicator"
    assert metrics["decision/inner_collection_steps"]["unit"] == "count"
    assert metrics["decision/inner_updates_per_rollout_step"]["unit"] == "count"


def test_prepare_specs_resolves_before_writing_and_never_initializes_wandb(tmp_path, monkeypatch):
    calls = []
    def identity(checkpoint, resolved, protocol, seeds, code, **kwargs):
        calls.append((resolved["selector"], seeds, kwargs))
        return {"backbone": "entity/project/backbone", "planner": {"type": "prior" if resolved["selector"].endswith("prior") else "sac"}}
    monkeypatch.setitem(sys.modules, "utils.eval_series_data", SimpleNamespace(identity_for_ambi_checkpoint=identity, descriptive_label=lambda identity, selector: "Policy prior" if identity["planner"]["type"] == "prior" else selector))
    monkeypatch.setattr("utils.wandb_utils.init_wandb", lambda *a, **k: pytest.fail("spec initialization contacted W&B"))
    directory = tmp_path / "specs"
    checkpoint = {**CHECKPOINT, "path": str(tmp_path / "checkpoint")}
    prepared = storage.write_eval_series_specs(directory, checkpoint, [_resolved("prior"), _resolved()],
                                               _protocol(), [101, 102], inventory_path="inventory.json")
    assert list(prepared["specs"]) == ["inner_budget/prior", "inner_budget/sac"]
    assert storage.read_json(Path(prepared["specs"]["inner_budget/prior"]))["label"] == "Policy prior"
    assert all(call[2]["inventory_path"] == "inventory.json" for call in calls)
    with pytest.raises(FileExistsError):
        storage.write_eval_series_specs(directory, checkpoint, [_resolved()], _protocol(), [101])


def test_spec_error_leaves_no_partial_output_and_dirty_sources_are_rejected(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("incompatible checkpoint")
    monkeypatch.setitem(sys.modules, "utils.eval_series_data", SimpleNamespace(identity_for_ambi_checkpoint=fail, descriptive_label=lambda *a: "unused"))
    with pytest.raises(ValueError, match="incompatible checkpoint"):
        storage.write_eval_series_specs(tmp_path / "specs", CHECKPOINT, [_resolved()], _protocol(), [101])
    assert not (tmp_path / "specs").exists()
    monkeypatch.setattr(storage, "code_identity", lambda: {"dirty": True})
    with pytest.raises(ValueError, match="clean checkout"):
        storage.write_eval_series_specs(tmp_path / "specs", CHECKPOINT, [_resolved()], _protocol(), [101])


def test_preflight_rejects_incompatible_assigned_identity_before_staging(tmp_path, monkeypatch):
    calls = []
    expected = {"backbone": "source", "planner": {"type": "sac", "updates": 3}}
    from utils.eval_series import validate_identity
    def check(run_dir, actual):
        calls.append(run_dir)
        return validate_identity({"identity": expected}, actual)
    monkeypatch.setitem(sys.modules, "utils.eval_series", SimpleNamespace(validate_identity=check))
    monkeypatch.setitem(sys.modules, "utils.eval_series_data", SimpleNamespace(
        identity_for_ambi_checkpoint=lambda *a, **k: {**expected, "planner": {"type": "sac", "updates": 12}}))
    with pytest.raises(ValueError, match="planner"):
        storage.preflight_eval_runs({"inner_budget/sac": str(tmp_path / "selected")}, CHECKPOINT,
                                     [_resolved()], _protocol(), [101], result_path=tmp_path / "manifest.json")
    assert calls == [str(tmp_path / "selected")]
    assert not (tmp_path / "manifest.json").exists()
