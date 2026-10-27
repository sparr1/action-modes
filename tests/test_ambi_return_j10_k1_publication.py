"""Complete K1 coverage without mutating controls or pairing different checkpoints."""

import copy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from slurm import ambi_return_j10_k1_publish as publish
from utils import eval_series as series
import test_ambi_return_j10_publication as j10_test
import test_ambi_return_h4_publication as h4_test
import test_ambi_return_prior_publication as original_test


def spec(entry):
    value = original_test.spec({**entry, "J": 10, "K": 1}, "sac")
    value["selector"] = publish.SELECTOR
    return value


def fixture(tmp_path, monkeypatch):
    from utils import eval_series_data

    old, j10_root, reference = j10_test.fixture(tmp_path, monkeypatch)
    publication = publish.j10.prepare(old, j10_root, reference)
    series._atomic_json(j10_root / "inventory.json", old)
    raw_records = {}
    monkeypatch.setattr(eval_series_data, "load_records", lambda path, **kwargs: copy.deepcopy(raw_records[str(path)]))
    for index, entry in enumerate(old["entries"]):
        pins = publication["references"].get(str(index))
        output, _ = j10_test.production(j10_root, entry, index, pins)
        raw_records[str(output / "bundle/manifest.json")] = [
            j10_test.raw_record(output, entry, index, arm, [10 if arm == "prior" else 30 + index] * 5)
            for arm in publish.j10.arms(index)]
        records = publish.j10.completed_records(entry, index, output, reference, pins)
        for arm, record in records.items():
            h4_test.publish_locally(publication["curves"][f"cell-{index}_{arm}"], record)
    data = copy.deepcopy(old)
    data["campaign_id"] = publish.ATTEMPT
    root = tmp_path / "j10-k1"
    for index in publish.INDICES:
        series._atomic_json(root / f"prepare/cell-{index}/specs/rounds__j10_k1.json", spec(data["entries"][index]))
    publish._CONTEXT_CACHE.clear()
    return data, root, reference, j10_root, raw_records


def production(root, entry, index, pins):
    output = root / f"production/cell-{index}"
    series._atomic_json(output / "results.json", {})
    series._atomic_json(output / "bundle/manifest.json", {"status": "complete", "runs": [
        {"selector": publish.SELECTOR, "status": "complete"}]})
    receipt = {"validated": True, "mode": "run", "campaign_id": publish.ATTEMPT,
        "source_run": entry["source_run"], "step": 500000, "J": 10, "source_J": entry["J"], "eval_J": 10,
        "source_K": 3, "eval_K": 1, "source_H": 3, "eval_H": 3, "selector": publish.SELECTOR,
        "episodes": 5, "decisions": 2500, "solves": 2500,
        "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"],
        "reference": {"receipt_sha256": pins["receipt.json"], "results_sha256": pins["results.json"],
                      "manifest_sha256": pins["bundle/manifest.json"]},
        "results_sha256": series._file_digest(output / "results.json"),
        "manifest_sha256": series._file_digest(output / "bundle/manifest.json")}
    series._atomic_json(output / "receipt.json", receipt)
    return output, receipt


def test_inventory_and_specs_reject_old_step_existing_index_and_wrong_controller(tmp_path, monkeypatch):
    data, root, _, _, _ = fixture(tmp_path, monkeypatch)
    path = root / "inventory.json"
    series._atomic_json(path, data)
    assert publish.load_inventory(path) == data
    for index in (1, 2):
        entry = data["entries"][index]
        publish.validate_spec(entry, index, spec(entry))
        for field, value in (("inner_rounds", 8), ("inner_rollout_horizon", 4), ("inner_solve_interval", 3)):
            wrong = spec(entry)
            wrong["identity"]["planner"]["settings"][field] = value
            with pytest.raises(ValueError, match="original J/K"):
                publish.validate_spec(entry, index, wrong)
    with pytest.raises(ValueError, match="Only the two"):
        publish.validate_spec(data["entries"][0], 0, spec(data["entries"][0]))
    data["entries"][2]["step"] = 325000
    series._atomic_json(path, data)
    with pytest.raises(ValueError, match="500k training J/K"):
        publish.load_inventory(path)


def test_prepare_only_two_new_runs_reuses_eight_identical_controls(tmp_path, monkeypatch):
    data, root, reference, j10_root, _ = fixture(tmp_path, monkeypatch)
    before = {str(path): path.read_bytes() for tree in (reference, j10_root) for path in tree.rglob("*") if path.is_file()}
    receipt = publish.prepare(data, root, reference, j10_root)
    assert publish.prepare(data, root, reference, j10_root) == receipt
    assert before == {str(path): path.read_bytes() for tree in (reference, j10_root) for path in tree.rglob("*") if path.is_file()}
    assert set(receipt["curves"]) == {"cell-1_k1", "cell-2_k1"}
    old = series._read(j10_root / "publication/publication-map.json")
    assert receipt["context_curves"] == publish.cadence.layout_curves(old)
    assert len({row["run_id"] for row in publish.cadence.layout_curves(receipt).values()}) == 10
    assert receipt["references"]["1"] == publish.shared.file_pins(reference / "production/cell-1")
    assert receipt["references"]["2"] == publish.shared.file_pins(j10_root / "production/cell-2")
    assert set(receipt["cell_run_maps"]) == {"1", "2"}
    for path in receipt["cell_run_maps"].values():
        assert set(series._read(path)) == {publish.SELECTOR}


@pytest.mark.parametrize("change", ["output", "journal"])
def test_changed_frozen_context_pins_are_rejected(tmp_path, monkeypatch, change):
    data, root, reference, j10_root, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, j10_root)
    path = (j10_root / "production/cell-2/results.json" if change == "output" else
            Path(receipt["context_curves"]["cell-0_j10"]["run_dir"]) / "publication.json")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen J10 comparison control changed"):
        publish.prepare(data, root, reference, j10_root)


@pytest.mark.parametrize("index", [1, 2])
def test_new_result_pairs_own_prior_and_j10_k3_and_rejects_mismatches(tmp_path, monkeypatch, index):
    data, root, reference, j10_root, raw_records = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, j10_root)
    entry, pins = data["entries"][index], receipt["references"][str(index)]
    controls, _, _ = publish.context_records(data, reference, j10_root, receipt["context_pins"])
    output, result_receipt = production(root, entry, index, pins)
    raw = original_test.record(output, {**entry, "J": 10, "K": 1}, "sac", [40, 41, 42, 43, 44])
    raw.update(spec(entry))
    raw["episodes"].reverse()
    raw_records[str(output / "bundle/manifest.json")] = [raw]
    original_controls = copy.deepcopy(controls)
    result = publish.completed_record(entry, index, output, controls, pins)
    assert controls == original_controls
    assert result["metrics"]["eval/paired_gain_mean"] == 32
    assert result["metrics"]["eval/cadence_delta_mean"] == 12 - index
    assert result["metrics"]["eval/cadence_win_fraction"] == 1
    assert {ep["seed"]: ep["cadence_return_delta"] for ep in result["episodes"]} == {
        seed: value - (30 + index) for seed, value in zip(publish.prior.SEEDS, [40, 41, 42, 43, 44])}
    registry = receipt["curves"][f"cell-{index}_k1"]["run_dir"]
    assert series.stage_record(registry, result)["status"] == "staged"
    assert series.stage_record(registry, publish.completed_record(entry, index, output, controls, pins))["status"] == "already_staged"
    key = f"cell-{index}_{'j10' if index == 1 else 'sac'}"
    controls[key]["checkpoint"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="same checkpoint"):
        publish.completed_record(entry, index, output, controls, pins)
    controls = copy.deepcopy(original_controls)
    controls[key]["episodes"][0]["seed"] = 102
    with pytest.raises(ValueError, match="same five seeds"):
        publish.completed_record(entry, index, output, controls, pins)
    raw["episodes"][0]["truncated_by_evaluator"] = True
    with pytest.raises(ValueError, match="complete environment-limited"):
        publish.completed_record(entry, index, output, original_controls, pins)
    result_receipt["solves"] = 835
    series._atomic_json(output / "receipt.json", result_receipt)
    with pytest.raises(ValueError, match="authorized checkpoint/controller"):
        publish.completed_record(entry, index, output, original_controls, pins)


def test_pending_publisher_only_owns_two_new_runs(tmp_path, monkeypatch):
    data, root, reference, j10_root, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, j10_root)
    sessions = []
    allowed = {curve["run_dir"] for curve in receipt["curves"].values()}
    def publish_curve(row, failed, **kwargs):
        assert row["run_dir"] in allowed and not failed and row["complete"] == 0
        sessions.append(row)
    monkeypatch.setattr(publish, "publish_curve", publish_curve)
    status = publish.publish_once(data, root, reference, j10_root, workspace_verifier=lambda *_: {"url": "verified"})
    assert len(sessions) == len(status["runs"]) == 2
    assert {row["index"] for row in sessions} == {1, 2}
    assert not status["failures"] and not status["publication_failures"]


def test_native_workspace_displays_all_ten_runs(tmp_path, monkeypatch):
    data, root, reference, j10_root, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, j10_root)
    package, ws, reports, wr = [ModuleType(name) for name in (
        "wandb_workspaces", "wandb_workspaces.workspaces", "wandb_workspaces.reports", "wandb_workspaces.reports.v2")]
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
    package.workspaces, package.reports, reports.v2 = ws, reports, wr
    package.expr = SimpleNamespace(Metric=lambda _: SimpleNamespace(isin=lambda values: values))
    for module in (package, ws, reports, wr):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    saved = publish.install_workspace(receipt, root / "publication/workspace.json")
    assert saved["api_verified"] and saved["panel_count"] == 9 and len(saved["run_ids"]) == 10
    assert latest[0].settings.max_runs == 10
    bars = [panel for section in latest[0].sections for panel in section.panels if hasattr(panel, "metrics")]
    assert all(bar.max_bars_to_show == bar.max_runs_to_show == 10 for bar in bars)
