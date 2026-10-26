"""H4 publication reuses exact H3/K1 controls and creates two new identities."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_h4_publish as publish
from utils import eval_series as series
import test_ambi_return_prior_publication as original_test


def h4_spec(entry):
    spec = original_test.spec({**entry, "K": 1}, "sac")
    spec["selector"] = publish.SELECTOR
    spec["identity"]["planner"]["settings"]["inner_rollout_horizon"] = 4
    return spec


def publish_locally(curve, record):
    series.stage_record(curve["run_dir"], record)
    path = Path(curve["run_dir"]) / "publication.json"
    journal = series._read(path)
    journal["records"][record["record_id"]]["status"] = "published"
    series._atomic_json(path, journal)


def fixture(tmp_path, monkeypatch):
    root, reference, swaps = (tmp_path / name for name in ("h4", "original", "cadence"))
    reference.mkdir()
    data, original_map = original_test.prepared(reference)
    originals, h3s, original_pins = {}, {}, {}
    for index, entry in enumerate(data["entries"][:2]):
        output = reference / f"production/cell-{index}"
        output.mkdir(parents=True)
        prior = original_test.record(output, entry, "prior", [10] * 5)
        sac = original_test.record(output, entry, "sac", [20] * 5)
        originals[index] = {"prior": prior, "sac": publish.prior.paired_record(sac, prior)}
        for name in ("receipt.json", "results.json", "bundle/manifest.json"):
            series._atomic_json(output / name, {"original": index})
        original_pins[index] = publish.file_pins(output)
        for arm in ("prior", "sac"):
            publish_locally(original_map["curves"][f"cell-{index}_{arm}"], originals[index][arm])
        series._atomic_json(root / f"prepare/cell-{index}/specs/horizon__h4_k1.json", h4_spec(entry))
    h3_output = swaps / "production/cell-1"
    h3_output.mkdir(parents=True)
    h3 = original_test.record(h3_output, {**data["entries"][1], "K": 1}, "sac", [30] * 5)
    h3["selector"] = "cadence/k1"
    h3 = publish.cadence.add_cadence_delta(publish.prior.paired_record(h3, originals[1]["prior"]), originals[1]["sac"])
    registry = series.create_run(swaps / "publication/runs/cell-1_swap", h3, publish.cadence.ATTEMPT,
                                publish.prior.PROJECT, publish.prior.ENTITY, publish.prior.OWNER)
    curve = {"run_id": registry["run_id"], "run_dir": registry["run_dir"], "color": "#D946EF",
             "index": 1, "source_run": data["entries"][1]["source_run"], "name": "Original swap"}
    publish_locally(curve, h3)
    series._atomic_json(swaps / "publication/publication-map.json", {
        "inventory_sha256": series._digest(data), "curves": {"cell-1_swap": curve}})
    for name in ("receipt.json", "results.json", "bundle/manifest.json"):
        series._atomic_json(h3_output / name, {"swap": 1})
    def references(entry, index, reference_root, pins=None):
        observed = publish.file_pins(Path(reference_root) / f"production/cell-{index}")
        if pins is not None and pins != observed:
            raise ValueError("Original reference changed")
        return copy.deepcopy(originals[index]), observed
    monkeypatch.setattr(publish.cadence, "reference_records", references)
    monkeypatch.setattr(publish.cadence, "completed_record", lambda entry, index, output, *args: copy.deepcopy(h3))
    publish._H3_CACHE.clear()
    return data, root, reference, swaps, originals, h3


def test_prepare_creates_two_new_runs_and_reuses_exact_four_controls_unchanged(tmp_path, monkeypatch):
    data, root, reference, swaps, originals, h3 = fixture(tmp_path, monkeypatch)
    before = {str(path): path.read_bytes() for tree in (reference, swaps) for path in tree.rglob("*") if path.is_file()}
    receipt = publish.prepare(data, root, reference, swaps)
    assert publish.prepare(data, root, reference, swaps) == receipt
    assert before == {str(path): path.read_bytes() for tree in (reference, swaps) for path in tree.rglob("*") if path.is_file()}
    assert len(receipt["curves"]) == 2 and len(receipt["context_curves"]) == 4
    assert len({curve["run_id"] for curve in publish.cadence.layout_curves(receipt).values()}) == 6
    assert receipt["context_curves"]["cell-1_h3"]["run_dir"].startswith(str(swaps))
    assert receipt["context_curves"]["cell-0_h3"]["run_dir"].startswith(str(reference))
    for index in (0, 1):
        mapping = series._read(receipt["cell_run_maps"][str(index)])
        assert set(mapping) == {publish.SELECTOR}
        registry = series.load_run(next(iter(mapping.values())))
        assert registry["attempt_label"] == publish.ATTEMPT
        assert registry["identity"]["planner"]["settings"]["inner_rollout_horizon"] == 4
        assert registry["identity"]["planner"]["settings"].get("inner_solve_interval", 1) == 1


def test_changed_h3_result_and_unpublished_control_cannot_be_reused(tmp_path, monkeypatch):
    data, root, reference, swaps, _, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, swaps)
    series._atomic_json(swaps / "production/cell-1/results.json", {"changed": True})
    with pytest.raises(ValueError, match="H3/K1 comparison changed"):
        publish.prepare(data, root, reference, swaps)
    curve = receipt["context_curves"]["cell-0_h3"]
    path = Path(curve["run_dir"]) / "publication.json"
    journal = series._read(path)
    next(iter(journal["records"].values()))["status"] = "staged"
    series._atomic_json(path, journal)
    with pytest.raises(ValueError, match="exact validated published"):
        publish.prepare(data, root, reference, swaps)


def test_h4_identity_rejects_h3_or_eval_k3():
    entry = original_test.inventory()["entries"][1]
    spec = h4_spec(entry)
    publish.validate_spec(entry, 1, spec)
    for field, value in (("inner_rollout_horizon", 3), ("inner_solve_interval", 3)):
        wrong = copy.deepcopy(spec)
        wrong["identity"]["planner"]["settings"][field] = value
        with pytest.raises(ValueError, match="J8/H4/K1"):
            publish.validate_spec(entry, 1, wrong)


def test_horizon_delta_pairs_by_seed_preserves_inputs_and_rejects_other_checkpoint(tmp_path):
    entry = original_test.inventory()["entries"][0]
    h3 = original_test.record(tmp_path, entry, "sac", [10, 20, 30, 40, 50])
    h4 = original_test.record(tmp_path, entry, "sac", [12, 18, 30, 44, 60])
    h4["episodes"].reverse()
    before = copy.deepcopy(h4)
    record = publish.add_horizon_delta(h4, h3)
    assert before == h4
    assert record["metrics"]["eval/horizon_delta_mean"] == pytest.approx(2.8)
    assert record["metrics"]["eval/horizon_win_fraction"] == .6
    assert "eval/cadence_delta_mean" not in record["metrics"]
    assert {ep["seed"]: ep["horizon_return_delta"] for ep in record["episodes"]} == dict(zip(publish.prior.SEEDS, [2, -2, 0, 4, 10]))
    h3["checkpoint"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="same checkpoint"):
        publish.add_horizon_delta(h4, h3)


def test_completed_h4_pairs_reused_prior_and_h3_and_stages_idempotently(tmp_path, monkeypatch):
    from utils import eval_series_data
    data, root, reference, swaps, _, _ = fixture(tmp_path, monkeypatch)
    publication = publish.prepare(data, root, reference, swaps)
    entry, pins = data["entries"][1], publication["references"]["1"]
    output = root / "production/cell-1"
    series._atomic_json(output / "results.json", {})
    series._atomic_json(output / "bundle/manifest.json", {"status": "complete", "runs": [{"status": "complete"}]})
    receipt = {"validated": True, "mode": "run", "campaign_id": publish.ATTEMPT,
        "source_run": entry["source_run"], "step": 500000, "J": 8, "source_K": 3, "eval_K": 1,
        "source_H": 3, "eval_H": 4, "selector": publish.SELECTOR, "checkpoint_sha256": entry["sha256"],
        "metadata_sha256": entry["metadata_sha256"], "reference": {
            "receipt_sha256": pins["prior"]["receipt.json"], "results_sha256": pins["prior"]["results.json"],
            "manifest_sha256": pins["prior"]["bundle/manifest.json"]},
        "results_sha256": series._file_digest(output / "results.json"),
        "manifest_sha256": series._file_digest(output / "bundle/manifest.json")}
    series._atomic_json(output / "receipt.json", receipt)
    raw = original_test.record(output, entry, "sac", [31, 32, 33, 34, 35])
    raw.update(h4_spec(entry))
    monkeypatch.setattr(eval_series_data, "load_records", lambda *_, **__: [copy.deepcopy(raw)])
    args = (entry, 1, output, reference, swaps, pins)
    result = publish.completed_record(*args)
    assert result["metrics"]["eval/paired_gain_mean"] == 23
    assert result["metrics"]["eval/horizon_delta_mean"] == 3
    registry = publication["curves"]["cell-1_h4"]["run_dir"]
    assert series.stage_record(registry, result)["status"] == "staged"
    assert series.stage_record(registry, publish.completed_record(*args))["status"] == "already_staged"
    raw["episodes"][0]["length"] = 499
    with pytest.raises(ValueError, match="complete environment-limited"):
        publish.completed_record(*args)
    raw["episodes"][0]["length"] = 500
    receipt["eval_H"] = 3
    series._atomic_json(output / "receipt.json", receipt)
    with pytest.raises(ValueError, match="authorized checkpoint/controller"):
        publish.completed_record(*args)


def test_pending_publisher_only_opens_two_new_runs_without_returns(tmp_path, monkeypatch):
    data, root, reference, swaps, _, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, swaps)
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
    status = publish.publish_once(data, root, reference, swaps, workspace_verifier=lambda *_: {"url": "verified"})
    assert len(sessions) == len(status["runs"]) == 2
    assert all(session.run.summary["progress/status"] == "pending results" for session in sessions)
    assert all("eval/return_mean" not in session.run.summary for session in sessions)


def test_native_workspace_gate_requires_horizon_metrics_and_all_six_ids(tmp_path, monkeypatch):
    data, root, reference, swaps, _, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference, swaps)
    path = root / "publication/workspace.json"
    with pytest.raises(ValueError, match="Install H4"):
        publish.verify_workspace(receipt, path)
    ids = [row["run_id"] for row in publish.cadence.layout_curves(receipt).values()]
    series._atomic_json(path, {"campaign_id": publish.ATTEMPT, "api_verified": True, "run_ids": ids,
        "url": f"https://wandb.ai/{publish.prior.ENTITY}/{publish.prior.PROJECT}?nw=example"})
    tokens = ids + ["progress/checkpoints_complete", "progress/failed_cells"] + [metric for _, metric in publish.METRICS]
    response = {"project": {"allViews": {"edges": [{"node": {"spec": tokens}}]}}}
    fake = SimpleNamespace(Api=lambda **_: SimpleNamespace(client=SimpleNamespace(execute=lambda *_, **__: response)))
    monkeypatch.setattr(publish.importlib, "import_module", lambda _: SimpleNamespace(gql=lambda value: value))
    assert publish.verify_workspace(receipt, path, fake)["api_verified"]
    tokens.remove("eval/horizon_delta_mean")
    with pytest.raises(ValueError, match="lost expected"):
        publish.verify_workspace(receipt, path, fake)
