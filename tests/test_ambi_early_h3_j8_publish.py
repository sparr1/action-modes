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
