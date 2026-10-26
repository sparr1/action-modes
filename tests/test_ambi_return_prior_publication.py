"""Scientific pairing, durable identity and visible publication gates."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_prior_publish as publish
from utils import eval_series as series


def inventory():
    return {"schema_version": 1, "campaign_id": publish.ATTEMPT, "entries": [
        {"source_run": f"entity/project/run{index}", "sha256": str(index + 1) * 64,
         "metadata_sha256": str(index + 4) * 64, "step": step, "J": rounds, "K": interval}
        for index, (step, rounds, interval) in enumerate(((500000, 8, 1), (500000, 8, 3), (325000, 10, 3)))]}


def spec(entry, arm):
    settings = {"inner_rounds": entry["J"], "inner_rollout_horizon": 3}
    if entry["K"] != 1:
        settings["inner_solve_interval"] = entry["K"]
    return {"selector": "comparison/" + arm, "label": arm, "identity": {
        "backbone": entry["source_run"],
        "planner": {"type": "prior" if arm == "prior" else "sac", "action_rule": "tanh_mean",
                    **({"settings": settings} if arm == "sac" else {})},
        "protocol": {"environment": "humanoid-walk", "action_rule": "tanh_mean",
                     "environment_seeds": publish.SEEDS, "controller_seed": 55, "max_steps": 500},
        "science": {"source_sha256": "a" * 64, "algorithm": "AMBI"}}}


def prepared(tmp_path):
    data = inventory()
    for index, entry in enumerate(data["entries"]):
        for arm in publish.ARMS:
            series._atomic_json(tmp_path / "prepared" / f"cell-{index}" / "specs" / f"comparison__{arm}.json", spec(entry, arm))
    return data, publish.prepare(data, tmp_path)


def record(tmp_path, entry, arm, returns):
    artifact = tmp_path / (arm + ".json")
    artifact.write_text("{}")
    return {**spec(entry, arm), "checkpoint": {"step": entry["step"], "sha256": entry["sha256"]},
        "record_id": arm, "metrics": {"eval/return_mean": sum(returns) / 5}, "provenance": {},
        "episodes": [{"seed": seed, "solver_seed": seed + (1000 if arm == "sac" else 0),
                      "return": value, "length": 500, "truncated": True, "truncated_by_evaluator": False}
                     for seed, value in zip(publish.SEEDS, returns)],
        "artifact_files": {"source.json": str(artifact)}, "source_result_path": str(artifact)}


def test_prepare_allocates_six_explicit_new_runs_and_recovers_idempotently(tmp_path):
    data, first = prepared(tmp_path)
    second = publish.prepare(data, tmp_path)
    assert first == second
    assert len({curve["run_id"] for curve in first["curves"].values()}) == 6
    for index in range(3):
        mapping = series._read(first["cell_run_maps"][str(index)])
        assert set(mapping) == {"comparison/prior", "comparison/sac"}
        for directory in mapping.values():
            run = series.load_run(directory)
            assert (run["owner"], run["attempt_label"]) == ("oscar-rgao48", publish.ATTEMPT)
            assert not series._read(Path(directory) / "publication.json")["records"]
    # A checkpoint change cannot silently reuse the prepared attempt.
    data["entries"][0]["step"] = 550000
    with pytest.raises(ValueError, match="inventory changed"):
        publish.prepare(data, tmp_path)


def test_spec_checks_original_solve_interval_not_horizon():
    entry = inventory()["entries"][1]
    value = spec(entry, "sac")
    publish.validate_spec(entry, "sac", value)
    value["identity"]["planner"]["settings"]["inner_solve_interval"] = 1
    with pytest.raises(ValueError, match="original J/K"):
        publish.validate_spec(entry, "sac", value)
    publish.validate_spec(inventory()["entries"][0], "sac", spec(inventory()["entries"][0], "sac"))


def test_pairing_uses_environment_seeds_and_allows_distinct_solver_streams(tmp_path):
    entry = inventory()["entries"][0]
    prior = record(tmp_path, entry, "prior", [10, 20, 30, 40, 50])
    inner = record(tmp_path, entry, "sac", [12, 18, 30, 44, 60])
    inner["episodes"].reverse()
    original = copy.deepcopy(inner)
    paired = publish.paired_record(inner, prior)
    assert inner == original  # An existing immutable unpaired object is not modified.
    assert paired["metrics"]["eval/paired_gain_mean"] == pytest.approx(2.8)
    assert paired["metrics"]["eval/paired_win_fraction"] == .6
    assert paired["metrics"]["eval/paired_episodes"] == 5
    assert {ep["seed"]: ep["paired_return_delta"] for ep in paired["episodes"]} == dict(zip(publish.SEEDS, [2, -2, 0, 4, 10]))
    assert paired["record_id"] != inner["record_id"]
    prior["checkpoint"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint/source"):
        publish.paired_record(inner, prior)


def test_pairing_rejects_duplicate_seeds_and_different_action_rule(tmp_path):
    entry = inventory()["entries"][0]
    prior = record(tmp_path, entry, "prior", [1] * 5)
    inner = record(tmp_path, entry, "sac", [2] * 5)
    inner["episodes"][0]["seed"] = 102
    with pytest.raises(ValueError, match="unique matching"):
        publish.paired_record(inner, prior)
    inner["identity"]["protocol"]["action_rule"] = "tanh_sample"
    with pytest.raises(ValueError, match="protocol differs"):
        publish.paired_record(inner, prior)


def test_no_points_without_production_receipt_and_changed_files_rejected(tmp_path):
    entry = inventory()["entries"][0]
    output = tmp_path / "output"
    output.mkdir()
    assert publish.completed_records(entry, output) is None
    results = output / "results.json"
    results.write_text("{}")
    manifest = output / "bundle" / "manifest.json"
    series._atomic_json(manifest, {"status": "complete", "runs": []})
    receipt = {"validated": True, "mode": "run", "source_run": entry["source_run"],
        "step": entry["step"], "J": entry["J"], "K": entry["K"],
        "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"],
        "results_sha256": series._file_digest(results), "manifest_sha256": series._file_digest(manifest)}
    series._atomic_json(output / "receipt.json", receipt)
    results.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed after receipt"):
        publish.completed_records(entry, output)
    receipt["mode"] = "smoke"
    series._atomic_json(output / "receipt.json", receipt)
    with pytest.raises(ValueError, match="smoke"):
        publish.completed_records(entry, output)


def test_complete_pair_is_hash_bound_paired_and_staged_once(tmp_path, monkeypatch):
    from utils import eval_series_data
    data, publication = prepared(tmp_path)
    entry = data["entries"][0]
    output = tmp_path / "production" / "cell-0"
    output.mkdir(parents=True)
    results = output / "results.json"
    results.write_text("{}")
    manifest = output / "bundle" / "manifest.json"
    series._atomic_json(manifest, {"status": "complete", "runs": [
        {"selector": "comparison/" + arm, "status": "complete"} for arm in publish.ARMS]})
    receipt = {"validated": True, "mode": "run", "source_run": entry["source_run"],
        "step": entry["step"], "J": entry["J"], "K": entry["K"],
        "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"],
        "results_sha256": series._file_digest(results), "manifest_sha256": series._file_digest(manifest)}
    series._atomic_json(output / "receipt.json", receipt)
    raw = [record(tmp_path, entry, "prior", [10] * 5), record(tmp_path, entry, "sac", [11, 12, 13, 14, 15])]
    monkeypatch.setattr(eval_series_data, "load_records", lambda *_, **__: copy.deepcopy(raw))
    records = publish.completed_records(entry, output)
    assert records["sac"]["metrics"]["eval/paired_gain_mean"] == 3
    for arm, value in records.items():
        directory = publication["curves"][f"cell-0_{arm}"]["run_dir"]
        assert series.stage_record(directory, value)["status"] == "staged"
        assert series.stage_record(directory, publish.completed_records(entry, output)[arm])["status"] == "already_staged"
        assert len(series._read(Path(directory) / "publication.json")["records"]) == 1
    raw[1]["episodes"][0]["length"] = 32
    with pytest.raises(ValueError, match="shortened episode"):
        publish.completed_records(entry, output)


def test_workspace_live_verification_rejects_missing_native_panels(tmp_path, monkeypatch):
    _, receipt = prepared(tmp_path)
    path = tmp_path / "workspace.json"
    with pytest.raises(ValueError, match="Install native"):
        publish.verify_workspace(receipt, path)
    ids = [curve["run_id"] for curve in receipt["curves"].values()]
    series._atomic_json(path, {"campaign_id": publish.ATTEMPT, "api_verified": True,
        "run_ids": ids, "url": f"https://wandb.ai/{publish.ENTITY}/{publish.PROJECT}?nw=example"})
    wire = ids + ["progress/checkpoints_complete", "progress/failed_cells"] + [metric for group in publish.METRICS.values() for _, metric in group]
    response = {"project": {"allViews": {"edges": [{"node": {"spec": {"tokens": wire}}}]}}}
    fake = SimpleNamespace(Api=lambda **_: SimpleNamespace(client=SimpleNamespace(execute=lambda *_, **__: response)))
    monkeypatch.setattr(publish.importlib, "import_module", lambda _: SimpleNamespace(gql=lambda value: value))
    assert publish.verify_workspace(receipt, path, fake)["api_verified"]
    wire.remove("eval/paired_gain_mean")
    with pytest.raises(ValueError, match="lost expected"):
        publish.verify_workspace(receipt, path, fake)


def test_pending_publication_exposes_progress_without_inventing_return_points(tmp_path, monkeypatch):
    data, receipt = prepared(tmp_path)
    sessions = []
    class FakePublisher:
        def __init__(self, directory, owner, **kwargs):
            assert owner == publish.OWNER
            self.run = SimpleNamespace(config={}, summary={}, define_metric=lambda *args, **kw: None)
            sessions.append(self)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def publish_pending(self): return None
    monkeypatch.setattr(series, "Publisher", FakePublisher)
    status = publish.publish_once(data, tmp_path, workspace_verifier=lambda *_: {"url": "verified-view"})
    assert len(sessions) == len(status["runs"]) == 6
    assert all(session.run.summary["progress/status"] == "pending results" for session in sessions)
    assert all("eval/return_mean" not in session.run.summary for session in sessions)
    assert all(row["complete"] == row["published"] == 0 for row in status["runs"])
    assert all(not series._read(Path(curve["run_dir"]) / "publication.json")["records"] for curve in receipt["curves"].values())


def test_publication_refuses_missing_workspace_before_starting_any_runs(tmp_path, monkeypatch):
    data, _ = prepared(tmp_path)
    monkeypatch.setattr(series, "Publisher", lambda *_, **__: pytest.fail("Must not publish without panels"))
    with pytest.raises(ValueError, match="Install native"):
        publish.publish_once(data, tmp_path)
