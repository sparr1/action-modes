"""Three-seed curve provenance, immutable reuse and publication contracts."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_prior_curve as curve
from utils import eval_series as series
from utils import eval_series_data as data


def inventory():
    return {"schema_version": 1, "campaign_id": curve.ATTEMPT,
        "seeds": curve.SEEDS, "expected_steps": curve.STEPS,
        "entries": [{"source_run": curve.SOURCE, "source_run_id": curve.SOURCE.rsplit("/", 1)[-1],
            "source_run_name": "training", "step": step, "J": 8, "K": 1,
            "sha256": str(i + 1) * 64, "metadata_sha256": "a" * 64,
            **({"reuse_output": "/original/500k"} if step == 500000 else {})}
            for i, step in enumerate(curve.STEPS)]}


def spec(arm):
    return {"selector": "comparison/" + arm, "label": arm, "identity": {
        "backbone": curve.SOURCE, "planner": {"type": arm, "action_rule": "tanh_mean",
            **({"settings": {"inner_rounds": 8, "inner_rollout_horizon": 3}} if arm == "sac" else {})},
        "protocol": {"environment": "humanoid-walk", "environment_seeds": curve.SEEDS,
                     "max_steps": 500, "controller_seed": 55, "action_rule": "tanh_mean"},
        "science": {"algorithm": "AMBITDMPC2/AMBITDMPC2", "source_sha256": "b" * 64}}}


def record(tmp_path, arm, seeds=curve.SEEDS, step=500000):
    artifact = tmp_path / (arm + ".json")
    artifact.write_text("{}")
    identity = copy.deepcopy(spec(arm)["identity"])
    identity["protocol"]["environment_seeds"] = list(seeds)
    episodes = [{"seed": seed, "solver_seed": seed + 1000, "return": float(seed - 100 + (10 if arm == "sac" else 0)),
        "length": 500, "terminated": False, "truncated": True, "truncated_by_evaluator": False,
        "control_seconds": seed - 100, "model_metrics": {"inner_model_steps": 0 if arm == "prior" else 10}}
        for seed in seeds]
    return data._record(identity, {"step": step, "sha256": "7" * 64}, episodes,
                        {"source.json": str(artifact)}, {"source_result_sha256": "d" * 64}, artifact,
                        arm, "comparison/" + arm, arm)


def prepare(tmp_path):
    inv = inventory()
    for index in range(len(curve.STEPS)):
        for arm in curve.ARMS:
            series._atomic_json(curve.output_dir(tmp_path, index, "prepare") / "specs" / f"comparison__{arm}.json", spec(arm))
    return inv, curve.prepare_publication(inv, tmp_path)


def test_inventory_exact_scope_and_reuse_only_500k(tmp_path):
    path = tmp_path / "inventory.json"
    inv = inventory()
    series._atomic_json(path, inv)
    assert curve.load_inventory(path) == inv
    for mutation in (lambda d: d["entries"][0].update(step=150000),
                     lambda d: d["entries"][-1].pop("reuse_output"),
                     lambda d: d["entries"][0].update(reuse_output="/unexpected"),
                     lambda d: d.update(seeds=[101, 102]),
                     lambda d: d["entries"][0].update(K=3)):
        changed = copy.deepcopy(inv)
        mutation(changed)
        series._atomic_json(path, changed)
        with pytest.raises(ValueError):
            curve.load_inventory(path)


def test_subset_recomputes_statistics_and_preserves_source(tmp_path):
    old = record(tmp_path, "sac", [101, 102, 103, 104, 105])
    original = copy.deepcopy(old)
    derived = curve.subset_record(old, spec("sac")["identity"])
    assert old == original
    assert [e["seed"] for e in derived["episodes"]] == curve.SEEDS
    assert derived["metrics"]["eval/return_mean"] == 12
    assert derived["metrics"]["eval/return_sample_std"] == 1
    assert derived["metrics"]["work/environment_decisions"] == 1500
    assert derived["metrics"]["runtime/control_seconds"] == 6
    assert derived["record_id"] != old["record_id"]
    reuse = derived["provenance"]["seed_subset_reuse"]
    assert reuse["source_record_id"] == old["record_id"] and reuse["new_environment_decisions"] == 0
    assert derived["artifact_files"] == old["artifact_files"]
    assert derived["provenance"]["source_result_sha256"] == "d" * 64


@pytest.mark.parametrize("field", ["science", "planner", "protocol", "backbone"])
def test_subset_rejects_any_identity_change_beyond_seeds(tmp_path, field):
    old = record(tmp_path, "sac", [101, 102, 103, 104, 105])
    desired = copy.deepcopy(spec("sac")["identity"])
    if field == "backbone":
        desired[field] = "different/source/run"
    else:
        desired[field]["changed"] = True
    with pytest.raises(ValueError, match="beyond"):
        curve.subset_record(old, desired)


def test_subset_rejects_missing_or_short_or_wrong_checkpoint(tmp_path):
    old = record(tmp_path, "sac", [101, 102, 103, 104, 105])
    for key, value in (("length", 499), ("terminated", True), ("truncated_by_evaluator", True)):
        changed = copy.deepcopy(old)
        changed["episodes"][0][key] = value
        with pytest.raises(ValueError, match="complete"):
            curve.subset_record(changed, spec("sac")["identity"])
    old["checkpoint"]["step"] = 450000
    with pytest.raises(ValueError, match="500k"):
        curve.subset_record(old, spec("sac")["identity"])


def test_pairing_recomputes_exact_three_seed_gain(tmp_path):
    prior, sac = record(tmp_path, "prior"), record(tmp_path, "sac")
    sac["episodes"].reverse()
    paired = curve.paired_record(sac, prior)
    assert paired["metrics"]["eval/paired_gain_mean"] == 10
    assert paired["metrics"]["eval/paired_gain_sample_std"] == 0
    assert paired["metrics"]["eval/paired_win_fraction"] == 1
    assert paired["metrics"]["eval/paired_episodes"] == 3
    assert paired["metrics"]["eval/base_return_mean"] == 2
    assert all("paired_return_delta" not in e for e in sac["episodes"])
    sac["episodes"][0]["seed"] = 102
    with pytest.raises(ValueError, match="three unique"):
        curve.paired_record(sac, prior)


def test_two_new_curves_shared_across_all_checkpoints_and_idempotent(tmp_path):
    inv, first = prepare(tmp_path)
    assert curve.prepare_publication(inv, tmp_path) == first
    assert set(first["curves"]) == {"prior", "sac"}
    assert len({r["run_id"] for r in first["curves"].values()}) == 2
    for row in first["curves"].values():
        registry = series.load_run(row["run_dir"])
        assert registry["identity"]["protocol"]["environment_seeds"] == curve.SEEDS
        assert registry["attempt_label"] == curve.ATTEMPT and registry["owner"] == curve.OWNER
        assert series._read(Path(row["run_dir"]) / "publication.json")["records"] == {}
    changed = spec("sac")
    changed["identity"]["science"]["source_sha256"] = "f" * 64
    series._atomic_json(curve.output_dir(tmp_path, 3, "prepare") / "specs" / "comparison__sac.json", changed)
    with pytest.raises(ValueError, match="Checkpoint-dependent"):
        curve.prepare_publication(inv, tmp_path)


def test_500k_cannot_be_run_or_validated_as_new_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(curve, "verify_source", lambda *_: pytest.fail("Must reject before source or GPU work"))
    for smoke in (True, False):
        with pytest.raises(ValueError, match="reuse-only"):
            curve.run_cell(inventory(), tmp_path, 6, "a" * 40, smoke=smoke)
    with pytest.raises(ValueError, match="reuse-only"):
        curve.validate_completed(inventory(), tmp_path, 6)


def test_reused_500k_pair_stages_without_rerunning_or_overwriting_sources(tmp_path, monkeypatch):
    inv, publication = prepare(tmp_path)
    source = {arm: record(tmp_path, arm, [101, 102, 103, 104, 105]) for arm in curve.ARMS}
    original = copy.deepcopy(source)
    monkeypatch.setattr(curve.previous, "completed_records", lambda *_: copy.deepcopy(source))
    result = curve.completed_records(inv, tmp_path, 6)
    assert source == original
    for arm in curve.ARMS:
        directory = publication["curves"][arm]["run_dir"]
        assert series.stage_record(directory, result[arm])["status"] == "staged"
        assert series.stage_record(directory, curve.completed_records(inv, tmp_path, 6)[arm])["status"] == "already_staged"
        assert len(series._read(Path(directory) / "publication.json")["records"]) == 1


def test_incomplete_results_stay_pending_and_workspace_gate_precedes_publication(tmp_path, monkeypatch):
    inv, _ = prepare(tmp_path)
    assert curve.validate_completed(inv, tmp_path, 0) is None
    monkeypatch.setattr(series, "Publisher", lambda *_, **__: pytest.fail("No publication without panels"))
    with pytest.raises(ValueError, match="Install native"):
        curve.publish_once(inv, tmp_path)


def test_pending_progress_has_no_invented_return(tmp_path, monkeypatch):
    inv, _ = prepare(tmp_path)
    monkeypatch.setattr(curve, "completed_records", lambda *_: None)
    sessions = []
    class Publisher:
        def __init__(self, *_, **__):
            self.run = SimpleNamespace(config={}, summary={}, define_metric=lambda *a, **k: None)
            sessions.append(self)
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def publish_pending(self): return None
    monkeypatch.setattr(series, "Publisher", Publisher)
    result = curve.publish_once(inv, tmp_path, workspace_verifier=lambda *_: {"url": "verified"})
    assert len(sessions) == 2
    assert all(row["complete"] == 0 and row["expected"] == 7 for row in result["runs"])
    assert all(s.run.summary["progress/status"] == "pending results" for s in sessions)
    assert all("eval/return_mean" not in s.run.summary for s in sessions)
