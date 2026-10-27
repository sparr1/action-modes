"""J10 comparisons pair within checkpoints and keep the trained reference separate."""
import copy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from slurm import ambi_return_j10_publish as publish
from utils import eval_series as series
import test_ambi_return_h4_publication as h4_test
import test_ambi_return_prior_publication as original_test


def spec(entry, index, arm):
    actual = "sac" if index < 2 else arm
    value = original_test.spec({**entry, "J": 10}, actual)
    value["selector"] = publish.selector(index, arm)
    if actual == "sac" and entry["K"] > 1:
        value["identity"]["planner"]["settings"]["inner_hold_encoder"] = "solve_snapshot"
    return value


def fixture(tmp_path, monkeypatch):
    old, _, reference, _, _, _ = h4_test.fixture(tmp_path, monkeypatch)
    data = copy.deepcopy(old)
    data["campaign_id"] = publish.ATTEMPT
    data["entries"][2].update(step=500000, sha256="a" * 64, metadata_sha256="b" * 64)
    root = tmp_path / "j10"
    for index, entry in enumerate(data["entries"]):
        for arm in publish.arms(index):
            selected = publish.selector(index, arm)
            series._atomic_json(root / f"prepare/cell-{index}/specs" / (selected.replace("/", "__") + ".json"), spec(entry, index, arm))
    return data, root, reference


def production(root, entry, index, pins=None):
    output = root / f"production/cell-{index}"
    series._atomic_json(output / "results.json", {})
    series._atomic_json(output / "bundle/manifest.json", {"status": "complete", "runs": [
        {"selector": publish.selector(index, arm), "status": "complete"} for arm in publish.arms(index)]})
    receipt = {"validated": True, "mode": "run", "campaign_id": publish.ATTEMPT,
        "source_run": entry["source_run"], "step": 500000, "J": 10, "K": entry["K"],
        "source_J": entry["J"], "eval_J": 10, "source_K": entry["K"], "eval_K": entry["K"],
        "source_H": 3, "eval_H": 3, "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"],
        "results_sha256": series._file_digest(output / "results.json"),
        "manifest_sha256": series._file_digest(output / "bundle/manifest.json")}
    if index < 2:
        receipt.update(selector=publish.SELECTOR, reference={"receipt_sha256": pins["receipt.json"],
            "results_sha256": pins["results.json"], "manifest_sha256": pins["bundle/manifest.json"]})
    series._atomic_json(output / "receipt.json", receipt)
    return output, receipt


def raw_record(output, entry, index, arm, returns):
    value = original_test.record(output, {**entry, "J": 10}, "sac" if index < 2 else arm, returns)
    value.update(spec(entry, index, arm))
    return value


def test_inventory_is_new_campaign_all_500k_and_rejects_old_325k(tmp_path, monkeypatch):
    data, root, _ = fixture(tmp_path, monkeypatch)
    path = root / "inventory.json"
    series._atomic_json(path, data)
    assert publish.load_inventory(path) == data
    data["entries"][2]["step"] = 325000
    series._atomic_json(path, data)
    with pytest.raises(ValueError, match="500k training J/K"):
        publish.load_inventory(path)


def test_prepare_creates_four_new_and_reuses_four_unchanged_despite_third_inventory_change(tmp_path, monkeypatch):
    data, root, reference = fixture(tmp_path, monkeypatch)
    before = {str(path): path.read_bytes() for path in reference.rglob("*") if path.is_file()}
    receipt = publish.prepare(data, root, reference)
    assert publish.prepare(data, root, reference) == receipt
    assert before == {str(path): path.read_bytes() for path in reference.rglob("*") if path.is_file()}
    assert len(receipt["curves"]) == len(receipt["context_curves"]) == 4
    assert len({row["run_id"] for row in publish.cadence.layout_curves(receipt).values()}) == 8
    assert set(receipt["references"]) == {"0", "1"}
    assert set(series._read(receipt["cell_run_maps"]["2"])) == {"comparison/prior", "comparison/sac"}
    assert "Train J8 K3" in receipt["curves"]["cell-1_j10"]["name"]
    assert "eval J10 H3 K3" in receipt["curves"]["cell-1_j10"]["name"]
    assert "Train J10 K3 @ 500k" in receipt["curves"]["cell-2_sac"]["name"]
    assert "eval J8 H3 K3" in receipt["context_curves"]["cell-1_j8"]["name"]


def test_spec_checks_eval_j10_and_preserves_original_k(tmp_path, monkeypatch):
    data, _, _ = fixture(tmp_path, monkeypatch)
    entry = data["entries"][1]
    value = spec(entry, 1, "j10")
    publish.validate_spec(entry, 1, "j10", value)
    value["identity"]["planner"]["settings"]["inner_rounds"] = 8
    with pytest.raises(ValueError, match="original J/K"):
        publish.validate_spec(entry, 1, "j10", value)


def test_same_checkpoint_j10_minus_j8_pairs_seeds_and_rejects_third_backbone(tmp_path):
    entry = original_test.inventory()["entries"][0]
    j8 = original_test.record(tmp_path, entry, "sac", [10, 20, 30, 40, 50])
    j10 = original_test.record(tmp_path, {**entry, "J": 10}, "sac", [12, 18, 30, 44, 60])
    j10["episodes"].reverse()
    before = copy.deepcopy(j10)
    result = publish.add_rounds_delta(j10, j8)
    assert before == j10
    assert result["metrics"]["eval/rounds_delta_mean"] == pytest.approx(2.8)
    assert result["metrics"]["eval/rounds_win_fraction"] == .6
    assert {ep["seed"]: ep["rounds_return_delta"] for ep in result["episodes"]} == dict(zip(publish.prior.SEEDS, [2, -2, 0, 4, 10]))
    j8["identity"]["backbone"] = "entity/project/third"
    with pytest.raises(ValueError, match="same checkpoint"):
        publish.add_rounds_delta(j10, j8)


def test_complete_new_j10_arm_pairs_both_reused_controls_and_stages_once(tmp_path, monkeypatch):
    from utils import eval_series_data
    data, root, reference = fixture(tmp_path, monkeypatch)
    publication = publish.prepare(data, root, reference)
    entry, pins = data["entries"][0], publication["references"]["0"]
    output, receipt = production(root, entry, 0, pins)
    raw = raw_record(output, entry, 0, "j10", [21, 22, 23, 24, 25])
    monkeypatch.setattr(eval_series_data, "load_records", lambda *_, **__: [copy.deepcopy(raw)])
    args = (entry, 0, output, reference, pins)
    record = publish.completed_records(*args)["j10"]
    assert record["metrics"]["eval/paired_gain_mean"] == 13
    assert record["metrics"]["eval/rounds_delta_mean"] == 3
    registry = publication["curves"]["cell-0_j10"]["run_dir"]
    assert series.stage_record(registry, record)["status"] == "staged"
    assert series.stage_record(registry, publish.completed_records(*args)["j10"])["status"] == "already_staged"
    raw["episodes"][0]["length"] = 499
    with pytest.raises(ValueError, match="complete environment-limited"):
        publish.completed_records(*args)
    receipt["eval_J"] = 8
    series._atomic_json(output / "receipt.json", receipt)
    with pytest.raises(ValueError, match="authorized checkpoint/controller"):
        publish.completed_records(*args)


def test_third_new500k_pair_uses_own_prior_and_has_no_rounds_delta(tmp_path, monkeypatch):
    from utils import eval_series_data
    data, root, reference = fixture(tmp_path, monkeypatch)
    publish.prepare(data, root, reference)
    entry = data["entries"][2]
    output, receipt = production(root, entry, 2)
    raw = [raw_record(output, entry, 2, arm, [10 if arm == "prior" else 30] * 5) for arm in publish.prior.ARMS]
    monkeypatch.setattr(eval_series_data, "load_records", lambda *_, **__: copy.deepcopy(raw))
    records = publish.completed_records(entry, 2, output, reference)
    assert set(records) == {"prior", "sac"}
    assert records["sac"]["metrics"]["eval/paired_gain_mean"] == 20
    assert not any("rounds_" in key for record in records.values() for key in record["metrics"])
    receipt["step"] = 325000
    series._atomic_json(output / "receipt.json", receipt)
    with pytest.raises(ValueError, match="authorized checkpoint/controller"):
        publish.completed_records(entry, 2, output, reference)


def test_pending_publisher_owns_four_new_runs_only(tmp_path, monkeypatch):
    data, root, reference = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference)
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
    status = publish.publish_once(data, root, reference, workspace_verifier=lambda *_: {"url": "verified"})
    assert len(sessions) == len(status["runs"]) == 4
    assert all(session.run.summary["progress/status"] == "pending results" for session in sessions)
    assert all("eval/return_mean" not in session.run.summary for session in sessions)


def test_shared_native_layout_displays_all_eight_runs(tmp_path, monkeypatch):
    data, root, reference = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference)
    package, ws, reports, wr = [ModuleType(name) for name in ("wandb_workspaces", "wandb_workspaces.workspaces", "wandb_workspaces.reports", "wandb_workspaces.reports.v2")]
    latest = []
    class Workspace(SimpleNamespace):
        def save(self):
            self.url = "https://wandb.ai/entity/project?nw=test"
            latest[:] = [self]
        @classmethod
        def from_url(cls, url): return latest[0]
    ws.Workspace = Workspace
    for name in ("Section", "WorkspaceSettings", "RunsetSettings", "RunSettings", "SectionLayoutSettings"):
        setattr(ws, name, SimpleNamespace)
    for name in ("BarPlot", "MarkdownPanel", "Layout"):
        setattr(wr, name, SimpleNamespace)
    package.workspaces = ws
    package.reports = reports
    reports.v2 = wr
    package.expr = SimpleNamespace(Metric=lambda _: SimpleNamespace(isin=lambda values: values))
    for module in (package, ws, reports, wr):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    saved = publish.install_workspace(receipt, root / "publication/workspace.json")
    assert saved["api_verified"] and saved["panel_count"] == 9 and len(saved["run_ids"]) == 8
    workspace = latest[0]
    assert workspace.settings.max_runs == 8
    bars = [panel for section in workspace.sections for panel in section.panels if hasattr(panel, "metrics")]
    assert all(bar.max_bars_to_show == bar.max_runs_to_show == 8 for bar in bars)
