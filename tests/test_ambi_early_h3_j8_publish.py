"""Publication safety: immutable curves, validated points, no placeholder returns."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_early_h3_j8_publish as pub
from utils import eval_series as series
from utils import eval_series_data as data


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    science = {"version": 1, "algorithm": "AMBITDMPC2/AMBITDMPC2", "source_sha256": "b" * 64}
    monkeypatch.setattr(data, "scientific_identity", lambda *a, **k: science)
    campaign = {"campaign_id": "early-h3-j8-20261018", "source_commit": "a" * 40,
                "seeds": [101, 102, 103, 104, 105], "controller_seed": 55, "max_steps": 500, "cells": []}
    specs = tmp_path / "specs"
    for backbone, source in pub.BACKBONES.items():
        for execution in ("mean", "sample"):
            selector = "early_h3_j8/" + execution
            identity = {"backbone": source, "science": science,
                "planner": {"type": "sac", "action_rule": "tanh_" + execution,
                            "settings": {"inner_rollout_horizon": 3, "inner_rounds": 8}},
                "protocol": {"action_rule": "tanh_" + execution, "environment_seeds": campaign["seeds"],
                             "max_steps": 500, "controller_seed": 55}}
            series._atomic_json(specs / (backbone + "_" + execution) / (selector.replace("/", "__") + ".json"),
                                {"identity": identity, "selector": selector, "label": "H3 J8 " + execution})
            for step in pub.STEPS:
                index = len(campaign["cells"])
                campaign["cells"].append({"index": index, "backbone_id": backbone, "source_run": source,
                    "step": step, "checkpoint": str(tmp_path / str(step) / "checkpoint.pt"),
                    "metadata": str(tmp_path / str(step) / "metadata.json"), "checkpoint_sha256": "c" * 64,
                    "metadata_sha256": "d" * 64, "execution": execution, "selector": selector,
                    "output_dir": str(tmp_path / "cells" / str(index))})
    campaign_path = tmp_path / "campaign.json"
    series._atomic_json(campaign_path, campaign)
    pub.load_campaign(campaign_path)
    root = tmp_path / "publication"
    receipt = pub.publication_prepare(campaign, root, specs)
    return campaign, root, specs, receipt


def test_prepare_reuses_exactly_four_runs(prepared):
    campaign, root, specs, receipt = prepared
    assert pub.publication_prepare(campaign, root, specs) == receipt
    assert len(receipt["cell_run_dirs"]) == 16
    assert len(set(receipt["cell_run_dirs"].values())) == 4
    assert len(list(root.glob("runs/*/*/run.json"))) == 4
    assert not any(series._read(path)["records"] for path in root.glob("runs/*/*/publication.json"))
    changed = copy.deepcopy(campaign)
    changed["cells"][0]["checkpoint_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="changed"):
        pub.publication_prepare(changed, root, specs)


def test_sample_spec_cannot_claim_mean(prepared):
    campaign, root, specs, _ = prepared
    path = specs / "mppi_aux_sample" / "early_h3_j8__sample.json"
    spec = series._read(path)
    spec["identity"]["planner"]["action_rule"] = "tanh_mean"
    series._atomic_json(path, spec)
    with pytest.raises(ValueError, match="execution action"):
        pub.publication_prepare(campaign, root, specs)


def make_completed(campaign, receipt, monkeypatch):
    cell = campaign["cells"][0]
    output = Path(cell["output_dir"])
    manifest = output / "bundle" / "manifest.json"
    trace = output / "bundle" / "trace.jsonl.gz"
    series._atomic_json(manifest, {"runs": [{"selector": cell["selector"], "status": "complete", "trace_files": [trace.name]}]})
    trace.write_bytes(b"large trace placeholder")
    results = output / "results.json"
    series._atomic_json(results, {"episodes": [101, 102, 103, 104, 105]})
    validated = {"validated": True, "cell_index": cell["index"], "smoke": False,
                 "manifest_sha256": series._file_digest(manifest), "results_sha256": series._file_digest(results)}
    series._atomic_json(output / "receipt.json", validated)
    record = {"identity": series.load_run(receipt["cell_run_dirs"]["0"])["identity"],
        "checkpoint": {"step": cell["step"], "sha256": cell["checkpoint_sha256"]},
        "metrics": {"eval/return_mean": 5., "eval/episodes": 5}, "record_id": "actual-result",
        "episodes": [{"seed": seed} for seed in campaign["seeds"]], "provenance": {},
        "artifact_files": {"manifest.json": str(manifest), trace.name: str(trace)},
        "source_result_path": str(manifest), "label": "actual", "selector": cell["selector"]}
    monkeypatch.setattr(data, "load_records", lambda *a, **k: [copy.deepcopy(record)])
    return cell, validated, record, trace


def test_requires_validation_and_omits_trace_uploads(prepared, monkeypatch):
    campaign, root, _, receipt = prepared
    cell, validated, record, trace = make_completed(campaign, receipt, monkeypatch)
    validation_path = Path(cell["output_dir"]) / "receipt.json"
    validation_path.unlink()
    assert pub.stage_completed(campaign, receipt)["0"] == "pending"
    series._atomic_json(validation_path, {**validated, "validated": False})
    with pytest.raises(ValueError, match="explicitly validated"):
        pub.stage_completed(campaign, receipt)
    series._atomic_json(validation_path, validated)
    assert pub.stage_completed(campaign, receipt)["0"] == "complete"
    staged = series._read(Path(receipt["cell_run_dirs"]["0"]) / "records" / "actual-result.json")
    assert trace.name not in staged["artifact_files"]
    assert staged["provenance"]["remote_trace_files"][0]["sha256"] == series._file_digest(trace)
    assert len(staged["artifact_files"]) == 3
    assert staged["metrics"] == record["metrics"]
    # Later polls do not reread potentially huge trace files.
    monkeypatch.setattr(data, "load_records", lambda *a, **k: pytest.fail("repeat normalization"))
    assert pub.stage_completed(campaign, receipt)["0"] == "complete"
    series._atomic_json(validation_path, {**validated, "edited": True})
    with pytest.raises(ValueError, match="Previously accepted"):
        pub.stage_completed(campaign, receipt)


def test_rejects_wrong_checkpoint_and_missing_seed(prepared, monkeypatch):
    campaign, _, _, receipt = prepared
    _, _, record, _ = make_completed(campaign, receipt, monkeypatch)
    record["checkpoint"]["step"] = 500000
    with pytest.raises(ValueError, match="checkpoint"):
        pub.stage_completed(campaign, receipt)
    record["checkpoint"]["step"] = 100000
    record["episodes"].pop()
    with pytest.raises(ValueError, match="all five seeds"):
        pub.stage_completed(campaign, receipt)


def test_pending_publication_only_writes_summary(prepared, monkeypatch):
    campaign, root, _, receipt = prepared
    sessions = []
    class PendingPublisher:
        def __init__(self, *a, **k):
            self.run = SimpleNamespace(name=None, config={}, summary={}, define_metric=lambda *a, **k: None)
            sessions.append(self)
        def __enter__(self):
            return self
        def publish_pending(self):
            return {}
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(series, "Publisher", PendingPublisher)
    verifier = lambda *a: {"url": "https://wandb.ai/test", "api_verified": True}
    result = pub.publish_once(campaign, root, workspace_verifier=verifier)
    assert len(sessions) == 4
    assert all(session.run.summary["progress/checkpoints_complete"] == 0 for session in sessions)
    assert all(not any(key.startswith("eval/") for key in session.run.summary) for session in sessions)
    assert all(row["status"] == "pending results" for row in result["runs"])
    assert all(row["published"] == 0 for row in result["runs"])


def test_missing_layout_fails_before_wandb_mutation(prepared):
    campaign, root, _, _ = prepared
    with pytest.raises(ValueError, match="Install native panels"):
        pub.publish_once(campaign, root)


@pytest.fixture
def paired(prepared):
    old, old_root, _, old_receipt = prepared
    campaign = copy.deepcopy(old)
    campaign.update(campaign_id="original-bank-h3-j8-20261018", steps=[200000, 400000, 475000, 500000],
                    sources={"original_aux": {"source_run": "rwgao_b-brown-university/ambi/aux6428346x0",
                                              "label": "Original prior-trained backbone", "inner_operator": "none"}},
                    baseline_root=str(old_root.parent / "baselines"))
    campaign["cells"] = campaign["cells"][:8]
    source = campaign["sources"]["original_aux"]["source_run"]
    for cell in campaign["cells"]:
        cell["backbone_id"], cell["source_run"] = "original_aux", source
        cell["step"] = dict(zip(pub.STEPS, campaign["steps"]))[cell["step"]]
    specs = old_root.parent / "new-specs"
    for key, definition in pub.curve_definitions(campaign).items():
        identity = copy.deepcopy(series.load_run(old_receipt["curves"]["mppi_aux_" + definition["execution"]]["run_dir"])["identity"])
        identity["backbone"] = source
        prior = definition["role"] == "base"
        selector = "early_h3_j8/" + definition["execution"]
        if prior:
            identity["planner"] = {"type": "prior", "action_rule": "tanh_" + definition["execution"]}
            selector = "reference/prior" if definition["execution"] == "mean" else "early_prior/sample"
            identity["science"]["source_sha256"] = "e" * 64  # Audited historical control remains distinct.
        series._atomic_json(specs / key / (selector.replace("/", "__") + ".json"),
                            {"identity": identity, "selector": selector, "label": definition["name"]})
    root = old_root.parent / "paired-publication"
    path = root.parent / "new-campaign.json"
    series._atomic_json(path, campaign)
    pub.load_campaign(path)
    receipt = pub.publication_prepare(campaign, root, specs)
    return campaign, root, specs, receipt


def paired_result(cell, run_dir, returns, record_id):
    output = Path(cell["output_dir"])
    selector = "reference/prior" if cell.get("role") == "base" else cell["selector"]
    manifest = output / "bundle" / "manifest.json"
    series._atomic_json(manifest, {"runs": [{"selector": selector, "status": "complete", "trace_files": []}]})
    result_path = output / "results.json"
    series._atomic_json(result_path, {"returns": returns})
    series._atomic_json(output / "receipt.json", {"validated": True, "smoke": False, "cell_index": cell["index"],
        "manifest_sha256": series._file_digest(manifest), "results_sha256": series._file_digest(result_path)})
    return {"identity": series.load_run(run_dir)["identity"],
        "checkpoint": {"step": cell["step"], "sha256": cell["checkpoint_sha256"]}, "record_id": record_id,
        "metrics": {"eval/return_mean": sum(returns) / len(returns), "eval/episodes": 5},
        "episodes": [{"seed": 101 + i, "return": value, "length": 500, "solver_seed": i + (100 if cell.get("role") == "base" else 200)}
                     for i, value in enumerate(returns)], "artifact_files": {"manifest.json": str(manifest)},
        "source_result_path": str(manifest), "provenance": {}, "selector": selector, "label": record_id}


def test_manifest_driven_four_curves_include_historical_base(paired):
    campaign, root, specs, receipt = paired
    assert len(receipt["cell_run_dirs"]) == len(receipt["baseline_run_dirs"]) == 8
    assert set(receipt["curves"]) == {"original_aux_mean", "original_aux_sample", "original_aux_base_mean", "original_aux_base_sample"}
    assert len({spec["color"] for spec in receipt["curves"].values()}) == 4
    base = series.load_run(receipt["curves"]["original_aux_base_mean"]["run_dir"])
    assert base["identity"]["science"]["source_sha256"] == "e" * 64
    assert pub.publication_prepare(campaign, root, specs) == receipt
    assert "Paired improvement over matching base" in pub.metric_sections(campaign)


def test_gain_waits_for_base_then_publishes_exact_paired_statistics(paired, monkeypatch):
    campaign, _, _, receipt = paired
    cell = campaign["cells"][0]
    inner = paired_result(cell, receipt["cell_run_dirs"]["0"], [2, 0, 4, 4, 5], "inner")
    records = {inner["source_result_path"]: inner}
    monkeypatch.setattr(data, "load_records", lambda path, **k: [copy.deepcopy(records[str(path)])])
    status = pub.stage_completed(campaign, receipt)
    assert status["0"] == "waiting for matched base"
    assert not series._read(Path(receipt["cell_run_dirs"]["0"]) / "publication.json")["records"]
    base_cell = pub.baseline_cell(campaign, cell)
    base = paired_result(base_cell, receipt["baseline_run_dirs"]["0"], [1, 2, 3, 4, 4], "base")
    records[base["source_result_path"]] = base
    status = pub.stage_completed(campaign, receipt)
    assert status["0"] == status["base:0"] == "complete"
    rows = list((Path(receipt["cell_run_dirs"]["0"]) / "records").glob("*.json"))
    assert len(rows) == 1
    paired_record = series._read(rows[0])
    assert [ep["paired_return_delta"] for ep in paired_record["episodes"]] == [1, -2, 1, 0, 1]
    assert paired_record["metrics"]["eval/paired_gain_mean"] == pytest.approx(.2)
    assert paired_record["metrics"]["eval/paired_gain_sample_std"] == pytest.approx(1.303840481)
    assert paired_record["metrics"]["eval/paired_win_fraction"] == .6
    assert paired_record["metrics"]["eval/paired_episodes"] == 5
    assert paired_record["provenance"]["paired_baseline"]["identity"]["science"] == base["identity"]["science"]
    assert "paired-base/manifest.json" in paired_record["artifact_files"]
    monkeypatch.setattr(data, "load_records", lambda *a, **k: pytest.fail("repeated normalization"))
    assert pub.stage_completed(campaign, receipt) == status
    assert len(list((Path(receipt["cell_run_dirs"]["0"]) / "records").glob("*.json"))) == 1


@pytest.mark.parametrize("mutation", ["action", "seed", "checkpoint", "source", "nonprior"])
def test_pairing_rejects_incompatible_base(paired, mutation):
    campaign, _, _, receipt = paired
    cell = campaign["cells"][0]
    inner = paired_result(cell, receipt["cell_run_dirs"]["0"], [2, 0, 4, 4, 5], "inner")
    base = paired_result(pub.baseline_cell(campaign, cell), receipt["baseline_run_dirs"]["0"], [1, 2, 3, 4, 4], "base")
    if mutation == "action":
        base["identity"]["protocol"]["action_rule"] = "tanh_sample"
    elif mutation == "seed":
        base["episodes"][0]["seed"] = 999
    elif mutation == "checkpoint":
        base["checkpoint"]["step"] = 575000
    elif mutation == "source":
        base["identity"]["backbone"] = "wrong-source"
    else:
        base["identity"]["planner"]["type"] = "sac"
    with pytest.raises(ValueError, match="baseline"):
        pub.pair_with_baseline(inner, base)
