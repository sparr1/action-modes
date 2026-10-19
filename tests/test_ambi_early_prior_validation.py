from copy import deepcopy
import json

import pytest

from slurm import ambi_early_prior_baselines as baseline
from slurm.ambi_early_prior_baselines import validate_result
from utils.ambi_benchmark import solver_seed


def artifacts(mode="mean"):
    campaign = {"seeds": [101, 102, 103, 104, 105], "controller_seed": 55}
    cell = {"checkpoint_sha256": "pinned", "execution": mode}
    episodes = [{"seed": seed, "solver_seed": solver_seed(55, "episode", seed),
        "length": 500, "return": 2.0, "truncated": True, "truncated_by_evaluator": False,
        "model_metrics": {key: 0.0 for key in (
            "inner_actor_optimizer_steps", "inner_critic_optimizer_steps",
            "inner_temperature_optimizer_steps", "inner_model_steps")}}
        for seed in campaign["seeds"]]
    row = {"selector": "early_prior/" + mode, "execution_action": mode,
        "resolved_config": {"inner_operator": "none"}, "outer_state_unchanged": True,
        "outer_updates_before": 123, "outer_updates_after": 123,
        "episodes": episodes, "return": {"mean": 2.0},
        "model_metrics": {key: {"max": 0.0} for key in episodes[0]["model_metrics"]}}
    result = {"checkpoint_sha256": "pinned", "results": [row]}
    manifest = {"status": "complete", "protocol": {"action_rule": "tanh_" + mode}}
    return result, manifest, cell, campaign


@pytest.mark.parametrize("mode", ["mean", "sample"])
def test_complete_matched_prior_is_accepted(mode):
    assert validate_result(*artifacts(mode))["passed"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(checkpoint_sha256="other"),
    lambda r: r["results"][0].update(execution_action="sample"),
    lambda r: r["results"][0]["episodes"].pop(),
    lambda r: r["results"][0]["episodes"][0].update(solver_seed=0),
    lambda r: r["results"][0]["episodes"][0].update(truncated_by_evaluator=True),
    lambda r: r["results"][0].update(outer_state_unchanged=False),
    lambda r: r["results"][0]["episodes"][0]["model_metrics"].update(inner_model_steps=1.0),
    lambda r: r["results"][0]["model_metrics"]["inner_temperature_optimizer_steps"].update(max=1.0),
])
def test_mismatched_or_adapting_reference_is_rejected(mutation):
    result, manifest, cell, campaign = deepcopy(artifacts())
    mutation(result)
    with pytest.raises(ValueError):
        validate_result(result, manifest, cell, campaign)


def historical_reference(tmp_path, monkeypatch):
    result, manifest, cell, campaign = artifacts()
    metadata = {"checkpoint": {"step": 200000}, "trial_run_params": {"seed": 55}}
    sidecar = tmp_path / "checkpoint.metadata.json"
    sidecar.write_text(json.dumps(metadata))
    source = tmp_path / "historical"
    bundle = source / "bundle"
    bundle.mkdir(parents=True)
    trace = bundle / "reference__prior/seed-101.jsonl.gz"
    trace.parent.mkdir()
    trace.write_bytes(b"immutable historical trace")
    row = result["results"][0]
    row["selector"] = "reference/prior"
    row.pop("execution_action")
    result.update(deterministic_execution=True, matrix_sha256="m" * 64)
    manifest.update(
        checkpoint={"sha256": "pinned", "source_run": "entity/project/original",
                    "metadata": metadata},
        code={"commit": "a" * 40, "dirty": False, "source_sha256": "b" * 64},
        runs=[{"selector": "reference/prior", "status": "complete",
               "config": {"alg_params": {"inner_operator": "none"}},
               "result": deepcopy(row), "trace_files": [str(trace.relative_to(bundle))]}],
    )
    result_path = source / "results.json"
    manifest_path = bundle / "manifest.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    cell.update(index=0, execution="mean", step=200000, metadata=str(sidecar),
                metadata_sha256=baseline.sha256(sidecar), source_run="entity/project/original",
                prior_reference=str(bundle), prior_pin={
                    "manifest_sha256": baseline.sha256(manifest_path),
                    "checkpoint_step": 200000, "checkpoint_sha256": "pinned",
                    "selector": "reference/prior"})
    campaign.update(cells=[cell], source_commit="c" * 40, max_steps=500)
    monkeypatch.setattr(baseline, "load_campaign", lambda path: campaign)
    monkeypatch.setattr(baseline, "verify_source", lambda: "d" * 40)
    verified = []
    monkeypatch.setattr(baseline, "verify_checkpoint", lambda row: verified.append(row))
    monkeypatch.setattr(baseline, "matched_protocol", lambda row, study: manifest["protocol"])
    return result_path, manifest_path, cell, campaign, verified


def test_historical_mean_reuse_preserves_original_science_and_all_file_bytes(tmp_path, monkeypatch):
    result_path, manifest_path, cell, campaign, verified = historical_reference(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in result_path.parent.rglob("*") if path.is_file()}
    output_root = tmp_path / "baselines"
    receipt = baseline.reuse_mean("campaign.json", 0, output_root)
    output = output_root / "cell-00"
    assert verified == [cell]
    assert receipt["validated"] and receipt["reused"] and not receipt["smoke"]
    assert receipt["source_commit"] == "a" * 40
    assert receipt["importer_source_commit"] == "d" * 40
    assert receipt["adapted_source_commit"] == campaign["source_commit"]
    assert receipt["reused_from"]["selector"] == "reference/prior"
    assert receipt["manifest_sha256"] == cell["prior_pin"]["manifest_sha256"]
    for path, data in before.items():
        assert path.read_bytes() == data
        assert (output / path.relative_to(result_path.parent)).read_bytes() == data
    row = json.loads((output / "results.json").read_text())["results"][0]
    assert row["selector"] == "reference/prior" and "execution_action" not in row
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        baseline.reuse_mean("campaign.json", 0, output_root)


@pytest.mark.parametrize("corruption", ["pin", "sample", "source", "sidecar", "protocol", "missing_trace"])
def test_historical_reuse_rejects_mismatch_before_creating_output(tmp_path, monkeypatch, corruption):
    result_path, manifest_path, cell, campaign, _ = historical_reference(tmp_path, monkeypatch)
    if corruption == "pin":
        cell["prior_pin"]["manifest_sha256"] = "wrong"
    elif corruption == "sample":
        cell["execution"] = "sample"
    elif corruption == "source":
        cell["source_run"] = "entity/project/different"
    elif corruption == "sidecar":
        from pathlib import Path
        Path(cell["metadata"]).write_text('{"checkpoint": {"step": 500000}}')
    elif corruption == "protocol":
        monkeypatch.setattr(baseline, "matched_protocol", lambda row, study: {"action_rule": "tanh_sample"})
    elif corruption == "missing_trace":
        (manifest_path.parent / "reference__prior/seed-101.jsonl.gz").unlink()
    output = tmp_path / "baselines"
    with pytest.raises(ValueError):
        baseline.reuse_mean("campaign.json", 0, output)
    assert not output.exists()


def test_old_selector_cannot_bypass_fresh_or_sample_validation():
    result, manifest, cell, campaign = artifacts()
    result["results"][0]["selector"] = "reference/prior"
    result["results"][0].pop("execution_action")
    result["deterministic_execution"] = True
    with pytest.raises(ValueError):
        validate_result(result, manifest, cell, campaign)
    assert validate_result(result, manifest, cell, campaign, historical_mean=True)["passed"]
    cell["execution"] = "sample"
    with pytest.raises(ValueError):
        validate_result(result, manifest, cell, campaign, historical_mean=True)
