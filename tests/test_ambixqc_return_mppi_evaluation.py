"""Return-tail MPPI preserves pairing and declares its distinct curve identity."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import evaluate_ambi_checkpoint as evaluator
from test_ambixqc_checkpoint_evaluation import checkpoint_case
from utils import ambi_benchmark as storage
from utils import eval_series_data as data
from utils.ambi_research import PresetMatrixError, load_preset_matrix


MATRIX = Path(__file__).resolve().parents[1] / "configs/research/ambixqc_humanoid_aux_return_mppi_benchmark.json"


def _matrix(path):
    value = load_preset_matrix(MATRIX)
    value["source_run"] = "entity/training/auxiliary"
    value["evaluation"].update(seeds=[101, 102], max_steps=2)
    for name in ("mppi_soft", "mppi_return"):
        value["comparisons"]["bootstrap"]["variants"][name]["evaluation_controller"]["params"].update(
            horizon=2, iterations=2, num_samples=8, num_elites=3, num_pi_trajs=2)
    path.write_text(json.dumps(value))
    return value


@pytest.mark.parametrize("checkpoint_case", [
    {"aux_return_mode": "xqc", "aux_return_detach_representation": True},
    {"aux_return_mode": "xqc", "aux_return_detach_representation": False},
], indirect=True)
def test_return_tail_reuses_prior_and_preflight_matches_executed_identity(checkpoint_case, tmp_path, monkeypatch):
    matrix, checkpoint = checkpoint_case
    _matrix(matrix)
    prior_bundle = tmp_path / "prior"
    prior = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["bootstrap/prior"],
                                     bundle_dir=prior_bundle)["results"][0]
    inventory = tmp_path / "inventory.json"
    side = Path(str(checkpoint) + ".metadata.json")
    inventory.write_text(json.dumps({"source_run": "entity/training/auxiliary", "checkpoints": [{
        "step": 4, "path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(side.read_bytes()).hexdigest(),
    }]}))
    monkeypatch.setattr(data, "scientific_identity", lambda *a, **kw: {"fixture": "science"})
    # This fixture is Pendulum; metadata-only space reconstruction is a
    # DMControl contract. Reuse the already resolved tiny fixture config.
    monkeypatch.setattr(data, "resolved_checkpoint_config", lambda *a, **kw: deepcopy(prior["resolved_config"]))
    monkeypatch.setattr(storage, "code_identity", lambda: {"commit": "fixture", "dirty": False})
    original_make_env = evaluator._make_env
    monkeypatch.setattr(evaluator, "_make_env", lambda *a: pytest.fail("specification created an environment"))
    prepared = evaluator.evaluate_matrix(matrix, checkpoint, checkpoint_inventory=inventory,
                                        eval_series_spec_dir=tmp_path / "specs")
    spec = json.loads(Path(prepared["specs"]["bootstrap/mppi_return"]).read_text())
    monkeypatch.setattr(evaluator, "_make_env", original_make_env)
    candidate_bundle = tmp_path / "return"
    result = evaluator.evaluate_matrix(matrix, checkpoint, reference_bundle=prior_bundle,
                                       bundle_dir=candidate_bundle, checkpoint_inventory=inventory)["results"][0]
    assert result["outer_state_unchanged"]
    assert result["outer_updates_before"] == result["outer_updates_after"] == 4
    assert result["evaluation_controller"]["protocol"]["terminal_value_source"] == "online_aux_return_twin_mean"
    deltas = [a["return"] - b["return"] for a, b in zip(result["episodes"], prior["episodes"])]
    assert result["paired_return_delta_vs_prior"]["mean"] == pytest.approx(sum(deltas) / len(deltas))
    records = data.load_records(candidate_bundle, checkpoint_inventory=inventory)
    assert len(records) == 1
    assert records[0]["identity"] == spec["identity"]
    assert "return Q" in data.descriptive_label(spec["identity"])
    manifest = json.loads((candidate_bundle / "manifest.json").read_text())
    assert "terminal-q:online-aux-return-twin-mean" in manifest["runs"][0]["wandb_tags"]
    assert len(manifest["runs"]) == 1  # Reusing prior never executes it again.


def test_return_tail_without_auxiliary_rejected_before_environment(checkpoint_case, tmp_path, monkeypatch):
    matrix, checkpoint = checkpoint_case
    _matrix(matrix)
    monkeypatch.setattr(evaluator, "_make_env", lambda *a: pytest.fail("invalid checkpoint constructed an environment"))
    with pytest.raises(PresetMatrixError, match="aux_return_mode"):
        evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("source", [None, True, [], "target", "return"])
def test_invalid_terminal_source_rejected(source, tmp_path):
    matrix = tmp_path / "matrix.json"
    value = _matrix(matrix)
    value["comparisons"]["bootstrap"]["variants"]["mppi_return"]["evaluation_controller"]["params"]["terminal_value_source"] = source
    matrix.write_text(json.dumps(value))
    with pytest.raises(PresetMatrixError, match="terminal_value_source"):
        load_preset_matrix(matrix)
