"""K4 campaign configuration preserves K1 behavior and immutable publication."""
import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_h4_k4_publish as publish
from slurm import ambi_return_h4_publish as workflow
from utils import eval_series as series
import test_ambi_return_h4_publication as h4_test
import test_ambi_return_prior_publication as original_test


def k4_spec(entry):
    value = h4_test.h4_spec(entry)
    value["selector"] = publish.SELECTOR
    value["identity"]["planner"]["settings"].update(inner_solve_interval=4, inner_hold_encoder="solve_snapshot")
    return value


def fixture(tmp_path, monkeypatch):
    data, h4_root, reference, swaps, originals, _ = h4_test.fixture(tmp_path, monkeypatch)
    h4_receipt = workflow.prepare(data, h4_root, reference, swaps)
    root = tmp_path / "k4"
    for index, entry in enumerate(data["entries"][:2]):
        series._atomic_json(root / f"prepare/cell-{index}/specs/horizon__h4_k4.json", k4_spec(entry))
    return data, root, reference, h4_root, h4_receipt


def output(root, entry, index, pins):
    path = root / f"production/cell-{index}"
    series._atomic_json(path / "results.json", {})
    series._atomic_json(path / "bundle/manifest.json", {"status": "complete", "runs": [{"status": "complete"}]})
    receipt = {"validated": True, "mode": "run", "campaign_id": publish.ATTEMPT,
        "source_run": entry["source_run"], "step": 500000, "J": 8, "source_K": entry["K"], "eval_K": 4,
        "source_H": 3, "eval_H": 4, "selector": publish.SELECTOR, "checkpoint_sha256": entry["sha256"],
        "metadata_sha256": entry["metadata_sha256"], "reference": {
            "receipt_sha256": pins["prior"]["receipt.json"], "results_sha256": pins["prior"]["results.json"],
            "manifest_sha256": pins["prior"]["bundle/manifest.json"]},
        "results_sha256": series._file_digest(path / "results.json"),
        "manifest_sha256": series._file_digest(path / "bundle/manifest.json")}
    series._atomic_json(path / "receipt.json", receipt)
    return path, receipt


def test_new_config_is_immutable_and_old_default_is_unchanged():
    with pytest.raises(FrozenInstanceError):
        publish.CAMPAIGN.eval_K = 1
    assert workflow.DEFAULT.eval_K == 1 and not workflow.DEFAULT.context_h4
    assert workflow.ATTEMPT == "return-prior-h4-k1-20261026"
    assert publish.CAMPAIGN.eval_K == 4 and publish.CAMPAIGN.root_key == "h4_root"
    assert len(publish.METRICS) == 6


def test_prepare_reuses_four_ids_without_waiting_for_h4_k1_or_mutating_them(tmp_path, monkeypatch):
    data, root, reference, h4_root, h4_receipt = fixture(tmp_path, monkeypatch)
    before = {str(path): path.read_bytes() for tree in (reference, h4_root) for path in tree.rglob("*") if path.is_file()}
    receipt = publish.prepare(data, root, reference, h4_root)
    assert publish.prepare(data, root, reference, h4_root) == receipt
    assert before == {str(path): path.read_bytes() for tree in (reference, h4_root) for path in tree.rglob("*") if path.is_file()}
    assert len(receipt["curves"]) == 2 and len(receipt["context_curves"]) == 4
    assert len({row["run_id"] for row in workflow.cadence.layout_curves(receipt).values()}) == 6
    for index in (0, 1):
        assert receipt["context_curves"][f"cell-{index}_h4_k1"]["run_id"] == h4_receipt["curves"][f"cell-{index}_h4"]["run_id"]
        assert "h4_k1_result" not in receipt["references"][str(index)]
        assert set(series._read(receipt["cell_run_maps"][str(index)])) == {publish.SELECTOR}


def test_context_registry_wrong_checkpoint_step_or_changed_identity_is_rejected(tmp_path, monkeypatch):
    data, root, reference, h4_root, h4_receipt = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, h4_root)
    curve = copy.deepcopy(h4_receipt["curves"]["cell-0_h4"])
    curve["step"] = 450000
    with pytest.raises(ValueError, match="different source or attempt"):
        workflow.pending_h4_control(data["entries"][0], 0, curve)
    registry = Path(h4_receipt["curves"]["cell-0_h4"]["run_dir"]) / "run.json"
    value = series._read(registry)
    value["created_at"] = "changed"
    series._atomic_json(registry, value)
    with pytest.raises(ValueError, match="context changed"):
        publish.prepare(data, root, reference, h4_root)


def test_k4_requires_fresh_solve_encoder_hold_and_does_not_change_k1_validation():
    entry = original_test.inventory()["entries"][0]
    publish.validate_spec(entry, 0, k4_spec(entry))
    workflow.validate_spec(entry, 0, h4_test.h4_spec(entry))
    wrong = k4_spec(entry)
    wrong["identity"]["planner"]["settings"].pop("inner_hold_encoder")
    with pytest.raises(ValueError, match="solve-time encoder"):
        publish.validate_spec(entry, 0, wrong)


def test_k4_publishes_before_k1_completion_and_remains_immutable_afterwards(tmp_path, monkeypatch):
    from utils import eval_series_data
    data, root, reference, h4_root, h4_receipt = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, h4_root)
    entry, pins = data["entries"][0], receipt["references"]["0"]
    path, production_receipt = output(root, entry, 0, pins)
    raw = original_test.record(path, entry, "sac", [11, 12, 13, 14, 15])
    raw.update(k4_spec(entry))
    monkeypatch.setattr(eval_series_data, "load_records", lambda *_, **__: [copy.deepcopy(raw)])
    args = (entry, 0, path, reference, h4_root, pins)
    record = publish.completed_record(*args)
    assert record["metrics"]["eval/paired_gain_mean"] == 3
    assert not any("horizon_" in key or "cadence_" in key for key in record["metrics"])
    registry = receipt["curves"]["cell-0_h4"]["run_dir"]
    assert series.stage_record(registry, record)["status"] == "staged"
    # The comparator can acquire completed results without changing this point.
    series._atomic_json(h4_root / "production/cell-0/receipt.json", {"now_complete": True})
    assert publish.completed_record(*args) == record
    assert series.stage_record(registry, publish.completed_record(*args))["status"] == "already_staged"
    production_receipt["eval_K"] = 1
    series._atomic_json(path / "receipt.json", production_receipt)
    with pytest.raises(ValueError, match="authorized checkpoint/controller"):
        publish.completed_record(*args)


def test_already_complete_h4_k1_context_is_validated_and_pinned(tmp_path, monkeypatch):
    data, root, reference, h4_root, h4_receipt = fixture(tmp_path, monkeypatch)
    real_completed = workflow.completed_record
    for index, entry in enumerate(data["entries"][:2]):
        path = h4_root / f"production/cell-{index}"
        path.mkdir(parents=True)
        record = original_test.record(path, entry, "sac", [20] * 5)
        record.update(h4_test.h4_spec(entry))
        h4_test.publish_locally(h4_receipt["curves"][f"cell-{index}_h4"], record)
        for name in ("receipt.json", "results.json", "bundle/manifest.json"):
            series._atomic_json(path / name, {"complete": index})
    records = {index: series._read(Path(h4_receipt["curves"][f"cell-{index}_h4"]["run_dir"]) / "records/sac.json") for index in (0, 1)}
    def completed(entry, index, *args, **kwargs):
        if kwargs.get("campaign", workflow.DEFAULT) == workflow.DEFAULT:
            return copy.deepcopy(records[index])
        return real_completed(entry, index, *args, **kwargs)
    monkeypatch.setattr(workflow, "completed_record", completed)
    receipt = publish.prepare(data, root, reference, h4_root)
    assert all("h4_k1_result" in pins for pins in receipt["references"].values())
    series._atomic_json(h4_root / "production/cell-1/results.json", {"changed": True})
    with pytest.raises(ValueError, match="H4/K1 result changed"):
        publish.prepare(data, root, reference, h4_root)


def test_pending_publication_only_opens_new_k4_runs(tmp_path, monkeypatch):
    data, root, reference, h4_root, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, h4_root)
    allowed = {row["run_dir"] for row in receipt["curves"].values()}
    sessions = []
    class FakePublisher:
        def __init__(self, directory, **kwargs):
            assert directory in allowed
            self.run = SimpleNamespace(config={}, summary={}, define_metric=lambda *args, **kw: None)
            sessions.append(self)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def publish_pending(self): return None
    monkeypatch.setattr(series, "Publisher", FakePublisher)
    result = publish.publish_once(data, root, reference, h4_root, workspace_verifier=lambda *_: {"url": "verified"})
    assert len(sessions) == len(result["runs"]) == 2
    assert all(session.run.config["evaluation_K"] == 4 for session in sessions)
    assert all(session.run.summary["progress/status"] == "pending results" for session in sessions)


def test_workspace_verifier_uses_k4_campaign_and_metrics(tmp_path, monkeypatch):
    data, root, reference, h4_root, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, h4_root)
    ids = [row["run_id"] for row in workflow.cadence.layout_curves(receipt).values()]
    path = root / "publication/workspace.json"
    series._atomic_json(path, {"campaign_id": publish.ATTEMPT, "api_verified": True, "run_ids": ids,
        "url": f"https://wandb.ai/{workflow.prior.ENTITY}/{workflow.prior.PROJECT}?nw=k4"})
    tokens = ids + ["progress/checkpoints_complete", "progress/failed_cells"] + [metric for _, metric in publish.METRICS]
    response = {"project": {"allViews": {"edges": [{"node": {"spec": tokens}}]}}}
    fake = SimpleNamespace(Api=lambda **_: SimpleNamespace(client=SimpleNamespace(execute=lambda *_, **__: response)))
    monkeypatch.setattr(workflow.importlib, "import_module", lambda _: SimpleNamespace(gql=lambda value: value))
    assert publish.verify_workspace(receipt, path, fake)["api_verified"]
    with pytest.raises(ValueError, match="six selected runs"):
        workflow.verify_workspace(receipt, path, fake)
    tokens.remove("eval/paired_gain_sample_std")
    with pytest.raises(ValueError, match="lost expected"):
        publish.verify_workspace(receipt, path, fake)
