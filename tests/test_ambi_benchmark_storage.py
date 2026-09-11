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
