"""Curve assignment fails before compute and queues only durable results."""
import json
from pathlib import Path

import pytest

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambixqc_checkpoint_evaluation import checkpoint_case
from utils import ambi_benchmark as storage
from utils import eval_series as core
from utils import eval_series_data as data


def identity():
    return {"backbone": "entity/train/prior", "planner": {"type": "prior", "action_rule": "tanh_mean"},
            "protocol": {"max_steps": 3, "seeds": [101, 102]}, "science": {"evaluator": "fixture"}}


def test_mismatched_curve_assignment_fails_before_environment(checkpoint_case, tmp_path, monkeypatch):
    matrix, checkpoint = checkpoint_case
    registry = core.create_run(tmp_path / "registry", {"identity": identity(), "label": "prior"},
                               "first", "project", "entity", "owner")
    changed = identity()
    changed["protocol"]["max_steps"] = 500
    monkeypatch.setattr(data, "identity_for_ambi_checkpoint", lambda *args, **kwargs: changed)
    monkeypatch.setattr(evaluator, "_make_env", lambda *args: pytest.fail("incompatible assignment created environment"))
    with pytest.raises(core.SeriesError, match="Incompatible append: protocol"):
        evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=tmp_path / "output", eval_run_dir=registry["run_dir"])
    assert not (tmp_path / "output").exists()


def test_matching_curve_assignment_queues_after_frozen_evaluation(checkpoint_case, tmp_path, monkeypatch):
    matrix, checkpoint = checkpoint_case
    registry = core.create_run(tmp_path / "registry", {"identity": identity(), "label": "prior"},
                               "first", "project", "entity", "owner")
    monkeypatch.setattr(data, "identity_for_ambi_checkpoint", lambda *args, **kwargs: identity())
    monkeypatch.setattr("utils.wandb_utils.init_wandb", lambda *args, **kwargs: pytest.fail("GPU W&B initialization"))
    output = tmp_path / "output"
    result = evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=output, eval_run_dir=registry["run_dir"])
    assert result["results"][0]["outer_state_unchanged"]
    manifest = storage.read_json(output / "manifest.json")
    assert manifest["status"] == "complete"
    assert len(manifest["runs"][0]["episodes"]) == 2
    pointers = list((Path(registry["run_dir"]) / "incoming").glob("*.json"))
    assert len(pointers) == 1
    pointer = json.loads(pointers[0].read_text())
    assert pointer["result_path"] == str(output / "manifest.json")
    assert pointer["selector"] == "controller/prior"
    assert json.loads((Path(registry["run_dir"]) / "publication.json").read_text())["records"] == {}


def test_prelaunch_specification_does_not_create_environment(checkpoint_case, tmp_path, monkeypatch):
    matrix, checkpoint = checkpoint_case
    monkeypatch.setattr(storage, "code_identity", lambda: {"commit": "fixture", "dirty": False})
    monkeypatch.setattr(data, "identity_for_ambi_checkpoint", lambda *args, **kwargs: identity())
    monkeypatch.setattr(evaluator, "_make_env", lambda *args: pytest.fail("specification created environment"))
    result = evaluator.evaluate_matrix(matrix, checkpoint, eval_series_spec_dir=tmp_path / "specs")
    assert result["mode"] == "evaluation_series_specifications"
    spec = json.loads(Path(result["specs"]["controller/prior"]).read_text())
    assert spec["identity"] == identity()


def test_banked_diagnostics_reject_curve_assignment_before_compute(checkpoint_case, tmp_path, monkeypatch):
    matrix, checkpoint = checkpoint_case
    registry = core.create_run(tmp_path / "registry", {"identity": identity(), "label": "prior"},
                               "first", "project", "entity", "owner")
    monkeypatch.setattr(evaluator, "_make_env", lambda *args: pytest.fail("bank rejection created environment"))
    with pytest.raises(ValueError, match="Observation-bank diagnostics remain in local bundles"):
        evaluator.evaluate_matrix(matrix, checkpoint, bank_only=True, root_bank_path=tmp_path / "missing-bank.json",
                                  bundle_dir=tmp_path / "output", eval_run_dir=registry["run_dir"])
    assert not (tmp_path / "output").exists()
