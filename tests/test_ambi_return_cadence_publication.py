"""Cadence crossover reuses immutable controls and publishes only two new arms."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_return_cadence_publish as publish
from utils import eval_series as series
import test_ambi_return_prior_publication as original_test


def swap_spec(entry, index):
    value = original_test.spec({**entry, "K": publish.TARGETS[index]}, "sac")
    value["selector"] = f"cadence/k{publish.TARGETS[index]}"
    return value


def fixture(tmp_path, monkeypatch):
    root, reference = tmp_path / "swap", tmp_path / "reference"
    reference.mkdir()
    data, original = original_test.prepared(reference)
    originals = {}
    for index, entry in enumerate(data["entries"][:2]):
        artifacts = reference / "production" / f"cell-{index}"
        artifacts.mkdir(parents=True)
        originals[index] = {arm: original_test.record(artifacts, entry, arm, [10 if arm == "prior" else 20] * 5)
                            for arm in publish.prior.ARMS}
        for name in ("receipt.json", "results.json", "bundle/manifest.json"):
            series._atomic_json(artifacts / name, {"original": index})
        for arm in publish.prior.ARMS:
            path = Path(original["curves"][f"cell-{index}_{arm}"]["run_dir"]) / "publication.json"
            journal = series._read(path)
            journal["records"] = {arm: {"status": "published"}}
            series._atomic_json(path, journal)
        series._atomic_json(root / "prepare" / f"cell-{index}" / "specs" /
                            f"cadence__k{publish.TARGETS[index]}.json", swap_spec(entry, index))
    monkeypatch.setattr(publish.prior, "completed_records", lambda entry, output:
                        copy.deepcopy(originals[int(Path(output).name.split("-")[-1])]))
    publish._REFERENCE_CACHE.clear()
    return data, root, reference, originals


def completed_output(root, entry, index, pins):
    output = root / "production" / f"cell-{index}"
    series._atomic_json(output / "results.json", {})
    series._atomic_json(output / "bundle/manifest.json", {"status": "complete", "runs": [{"status": "complete"}]})
    receipt = {"validated": True, "mode": "run", "campaign_id": publish.ATTEMPT, "source_run": entry["source_run"],
        "step": entry["step"], "J": 8, "source_K": entry["K"], "eval_K": publish.TARGETS[index],
        "selector": f"cadence/k{publish.TARGETS[index]}", "checkpoint_sha256": entry["sha256"],
        "metadata_sha256": entry["metadata_sha256"],
        "reference": {"receipt_sha256": pins["receipt.json"], "results_sha256": pins["results.json"],
                      "manifest_sha256": pins["bundle/manifest.json"]},
        "results_sha256": series._file_digest(output / "results.json"),
        "manifest_sha256": series._file_digest(output / "bundle/manifest.json")}
    series._atomic_json(output / "receipt.json", receipt)
    return output, receipt


def test_prepare_allocates_only_two_and_leaves_originals_untouched(tmp_path, monkeypatch):
    data, root, reference, _ = fixture(tmp_path, monkeypatch)
    before = {str(path): path.read_bytes() for path in reference.rglob("*") if path.is_file()}
    receipt = publish.prepare(data, root, reference)
    assert publish.prepare(data, root, reference) == receipt
    assert {str(path): path.read_bytes() for path in reference.rglob("*") if path.is_file()} == before
    assert len(receipt["curves"]) == 2 and len(receipt["context_curves"]) == 4
    assert len({curve["run_id"] for curve in publish.layout_curves(receipt).values()}) == 6
    for index in (0, 1):
        mapping = series._read(receipt["cell_run_maps"][str(index)])
        assert set(mapping) == {f"cadence/k{publish.TARGETS[index]}"}
        registry = series.load_run(next(iter(mapping.values())))
        assert (registry["owner"], registry["attempt_label"]) == (publish.prior.OWNER, publish.ATTEMPT)
        assert receipt["curves"][f"cell-{index}_swap"]["name"] == (
            f"Train K{data['entries'][index]['K']} @ 500k | SAC eval K{publish.TARGETS[index]}")


def test_prepare_rejects_unpublished_or_changed_references(tmp_path, monkeypatch):
    data, root, reference, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference)
    series._atomic_json(reference / "production/cell-0/results.json", {"changed": True})
    with pytest.raises(ValueError, match="reference changed"):
        publish.prepare(data, root, reference)
    assert series._read(root / "publication/publication-map.json") == receipt


def test_reference_cache_rechecks_hashes_without_rehashing_traces(tmp_path, monkeypatch):
    data, _, reference, _ = fixture(tmp_path, monkeypatch)
    entry = data["entries"][0]
    first, pins = publish.reference_records(entry, 0, reference)
    monkeypatch.setattr(publish.prior, "completed_records", lambda *_: pytest.fail("Already validated"))
    assert publish.reference_records(entry, 0, reference, pins)[0] == first
    first["prior"]["episodes"][0]["return"] = -1
    assert publish.reference_records(entry, 0, reference, pins)[0]["prior"]["episodes"][0]["return"] == 10
    series._atomic_json(reference / "production/cell-0/receipt.json", {"changed": True})
    with pytest.raises(ValueError, match="reference changed"):
        publish.reference_records(entry, 0, reference, pins)


def test_spec_requires_swap_at_same_frozen_J8_checkpoint():
    entry = original_test.inventory()["entries"][0]
    spec = swap_spec(entry, 0)
    publish.validate_spec(entry, 0, spec)
    assert entry["K"] == 1 and spec["selector"] == "cadence/k3"
    spec["identity"]["planner"]["settings"]["inner_solve_interval"] = 1
    with pytest.raises(ValueError, match="original J/K"):
        publish.validate_spec(entry, 0, spec)
    with pytest.raises(ValueError, match="J8 500k"):
        publish.validate_spec({**entry, "step": 450000}, 0, spec)


def test_delta_uses_seed_pairing_not_episode_order_and_preserves_inputs(tmp_path):
    entry = original_test.inventory()["entries"][0]
    old = original_test.record(tmp_path, entry, "sac", [10, 20, 30, 40, 50])
    new = original_test.record(tmp_path, entry, "sac", [12, 18, 30, 44, 60])
    new["episodes"].reverse()
    untouched = copy.deepcopy(new)
    result = publish.add_cadence_delta(new, old)
    assert new == untouched
    assert result["metrics"]["eval/cadence_delta_mean"] == pytest.approx(2.8)
    assert result["metrics"]["eval/cadence_win_fraction"] == .6
    assert {ep["seed"]: ep["cadence_return_delta"] for ep in result["episodes"]} == dict(zip(publish.prior.SEEDS, [2, -2, 0, 4, 10]))
    old["identity"]["protocol"]["action_rule"] = "tanh_sample"
    with pytest.raises(ValueError, match="same checkpoint and episode protocol"):
        publish.add_cadence_delta(new, old)


def test_delta_rejects_duplicate_seeds(tmp_path):
    entry = original_test.inventory()["entries"][0]
    old = original_test.record(tmp_path, entry, "sac", [10] * 5)
    new = copy.deepcopy(old)
    new["episodes"][0]["seed"] = 102
    with pytest.raises(ValueError, match="same five seeds"):
        publish.add_cadence_delta(new, old)


def test_completed_swap_is_paired_with_both_reused_controls_and_staged_once(tmp_path, monkeypatch):
    from utils import eval_series_data
    data, root, reference, _ = fixture(tmp_path, monkeypatch)
    publication = publish.prepare(data, root, reference)
    entry = data["entries"][0]
    output, receipt = completed_output(root, entry, 0, publication["references"]["0"])
    raw = original_test.record(output, {**entry, "K": 3}, "sac", [11, 12, 13, 14, 15])
    raw["selector"] = "cadence/k3"
    monkeypatch.setattr(eval_series_data, "load_records", lambda *_, **__: [copy.deepcopy(raw)])
    args = (entry, 0, output, reference, publication["references"]["0"])
    result = publish.completed_record(*args)
    assert result["metrics"]["eval/paired_gain_mean"] == 3
    assert result["metrics"]["eval/cadence_delta_mean"] == -7
    registry = publication["curves"]["cell-0_swap"]["run_dir"]
    assert series.stage_record(registry, result)["status"] == "staged"
    assert series.stage_record(registry, publish.completed_record(*args))["status"] == "already_staged"
    raw["episodes"][0]["length"] = 499
    with pytest.raises(ValueError, match="complete environment-limited"):
        publish.completed_record(*args)
    raw["episodes"][0]["length"] = 500
    receipt["eval_K"] = 1
    series._atomic_json(output / "receipt.json", receipt)
    with pytest.raises(ValueError, match="authorized checkpoint/controller"):
        publish.completed_record(*args)
    receipt["eval_K"] = 3
    receipt["reference"]["receipt_sha256"] = "0" * 64
    series._atomic_json(output / "receipt.json", receipt)
    with pytest.raises(ValueError, match="different original reference"):
        publish.completed_record(*args)


def test_pending_publication_never_opens_original_runs_or_invents_returns(tmp_path, monkeypatch):
    data, root, reference, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference)
    sessions = []
    allowed = {curve["run_dir"] for curve in receipt["curves"].values()}
    class FakePublisher:
        def __init__(self, directory, owner, **kwargs):
            assert directory in allowed and owner == publish.prior.OWNER
            self.run = SimpleNamespace(config={}, summary={}, define_metric=lambda *args, **kw: None)
            sessions.append(self)
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def publish_pending(self): return None
    monkeypatch.setattr(series, "Publisher", FakePublisher)
    result = publish.publish_once(data, root, reference, workspace_verifier=lambda *_: {"url": "verified"})
    assert len(sessions) == len(result["runs"]) == 2
    assert all(session.run.summary["progress/status"] == "pending results" for session in sessions)
    assert all("eval/return_mean" not in session.run.summary for session in sessions)


def test_live_workspace_gate_requires_all_six_runs_and_native_metrics(tmp_path, monkeypatch):
    data, root, reference, _ = fixture(tmp_path, monkeypatch)
    receipt = publish.prepare(data, root, reference)
    path = root / "publication/workspace.json"
    with pytest.raises(ValueError, match="Install cadence"):
        publish.verify_workspace(receipt, path)
    ids = [curve["run_id"] for curve in publish.layout_curves(receipt).values()]
    series._atomic_json(path, {"campaign_id": publish.ATTEMPT, "api_verified": True,
        "run_ids": ids, "url": f"https://wandb.ai/{publish.prior.ENTITY}/{publish.prior.PROJECT}?nw=example"})
    tokens = ids + ["progress/checkpoints_complete", "progress/failed_cells"] + [metric for _, metric in publish.METRICS]
    response = {"project": {"allViews": {"edges": [{"node": {"spec": tokens}}]}}}
    fake = SimpleNamespace(Api=lambda **_: SimpleNamespace(client=SimpleNamespace(execute=lambda *_, **__: response)))
    monkeypatch.setattr(publish.importlib, "import_module", lambda _: SimpleNamespace(gql=lambda value: value))
    assert publish.verify_workspace(receipt, path, fake)["api_verified"]
    tokens.remove("eval/cadence_delta_mean")
    with pytest.raises(ValueError, match="lost expected"):
        publish.verify_workspace(receipt, path, fake)


def test_publication_checks_panels_before_any_sdk_session(tmp_path, monkeypatch):
    data, root, reference, _ = fixture(tmp_path, monkeypatch)
    publish.prepare(data, root, reference)
    monkeypatch.setattr(series, "Publisher", lambda *_, **__: pytest.fail("Must install panels first"))
    with pytest.raises(ValueError, match="Install cadence"):
        publish.publish_once(data, root, reference)
