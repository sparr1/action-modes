"""Paired native MPPI evaluation uses a frozen XQC checkpoint and its own RNG."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

import evaluate_ambi_checkpoint as evaluator
from test_ambixqc_checkpoint_evaluation import checkpoint_case, _events, _scientific_episodes
from utils.ambi_research import (
    PresetMatrixError, load_preset_matrix, materialize_presets, normalize_selectors, resolve_preset,
)
from utils.checkpoint_context import load_checkpoint_context


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambixqc_humanoid_mppi_benchmark.json"


@pytest.fixture
def mppi_case(checkpoint_case):
    path, checkpoint = checkpoint_case
    matrix = json.loads(MATRIX.read_text())
    matrix["evaluation"].update(seeds=[101, 102], max_steps=2)
    matrix["comparisons"]["controller"]["variants"]["mppi"]["evaluation_controller"]["params"].update(
        horizon=2, iterations=2, num_samples=8, num_elites=3, num_pi_trajs=2)
    path.write_text(json.dumps(matrix))
    return path, checkpoint


def test_production_matrix_is_prior_and_mppi_with_native_settings():
    matrix = load_preset_matrix(MATRIX)
    assert normalize_selectors(matrix) == ["controller/prior", "controller/mppi"]
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["controller_seed"] == 12345
    assert matrix["evaluation"]["max_steps"] == 500
    variants = matrix["comparisons"]["controller"]["variants"]
    assert set(variants) == {"prior", "mppi"}
    assert variants["mppi"]["evaluation_controller"] == {
        "type": "mppi", "params": {"horizon": 3, "iterations": 6, "num_samples": 512,
            "num_elites": 64, "num_pi_trajs": 24, "min_std": 0.05, "max_std": 2.0, "temperature": 0.5},
    }


def test_mppi_resolution_preserves_checkpoint_and_cannot_materialize_for_training(mppi_case, tmp_path):
    matrix, checkpoint = mppi_case
    context = load_checkpoint_context(checkpoint)
    before = deepcopy(context)
    resolved = resolve_preset(matrix, "controller/mppi", checkpoint_context=context)
    assert context == before
    assert resolved["algorithm_config"]["alg"] == "AMBIXQC/AMBIXQC"
    assert resolved["algorithm_config"]["alg_params"] == context.trial_run_params["alg_params"]
    assert resolved["evaluation_controller"] == resolved["algorithm_config"]["evaluation_controller"]
    assert resolved["saved_algorithm_config"] == context.trial_run_params
    with pytest.raises(PresetMatrixError, match="cannot be materialized"):
        materialize_presets(matrix, tmp_path / "bad", checkpoint_context=context)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("override", [{"horizon": 0}, {"num_samples": True}, {"num_elites": 999},
                                      {"num_pi_trajs": -1}, {"temperature": 0}, {"min_std": 3},
                                      {"terminal_q_source": "target"}])
def test_bad_mppi_settings_fail_before_env_or_output(mppi_case, tmp_path, monkeypatch, override):
    path, checkpoint = mppi_case
    matrix = json.loads(path.read_text())
    matrix["comparisons"]["controller"]["variants"]["mppi"]["evaluation_controller"]["params"].update(override)
    path.write_text(json.dumps(matrix))
    monkeypatch.setattr(evaluator, "_make_env", lambda resolved: pytest.fail("late MPPI preflight"))
    with pytest.raises(PresetMatrixError):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_mppi_does_not_accept_ignored_xqc_update_overrides(mppi_case):
    path, checkpoint = mppi_case
    matrix = json.loads(path.read_text())
    matrix["comparisons"]["controller"]["variants"]["mppi"]["alg_params"] = {"inner_rounds": 5}
    path.write_text(json.dumps(matrix))
    with pytest.raises(PresetMatrixError, match="cannot request inner XQC settings"):
        evaluator.evaluate_matrix(path, checkpoint)


def test_paired_mppi_file_checkpoint_bundle_and_freeze(mppi_case, tmp_path):
    matrix, checkpoint = mppi_case
    bundle = tmp_path / "paired"
    payload = evaluator.evaluate_matrix(matrix, checkpoint, bundle_dir=bundle)
    assert not payload["deterministic_execution"]
    assert [result["controller"] for result in payload["results"]] == ["prior", "mppi"]
    prior, mppi = payload["results"]
    assert prior["deterministic_execution"]
    assert prior["action_rule"] == "tanh_mean"
    assert mppi["action_rule"] == "weighted_elite_gumbel_no_execution_noise"
    assert mppi["evaluation_controller"]["settings"]["effective_iterations"] == 2
    assert mppi["evaluation_controller"]["protocol"]["terminal_value_source"] == "online_xqc_twin_mean"
    for result in payload["results"]:
        assert result["outer_state_unchanged"]
        assert result["outer_updates_before"] == result["outer_updates_after"] == 4
        assert result["paired_return_delta_vs_reference"]["count"] == 2
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert "MPPI" in manifest["runs"][1]["wandb_name"]
    events = _events(bundle, manifest["runs"][1])
    assert len(events) == 4
    for event in events:
        assert event["metrics"]["decision/inner_model_steps"] == 34
        assert tuple(event[key] for key in ("critic_updates", "actor_updates", "temperature_updates")) == (0, 0, 0)


def test_mppi_seed_order_and_warmup_independence(mppi_case, tmp_path):
    matrix, checkpoint = mppi_case
    standalone = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/mppi"])["results"][0]
    paired = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/mppi", "controller/prior"])["results"][0]
    warm = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/mppi"], bundle_dir=tmp_path / "warm")["results"][0]
    reverse = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/mppi"], seeds=[102, 101])["results"][0]
    assert _scientific_episodes(standalone) == _scientific_episodes(paired)
    assert _scientific_episodes(standalone) == _scientific_episodes(warm)
    assert _scientific_episodes(standalone) == list(reversed(_scientific_episodes(reverse)))


def test_prior_reference_reuse_does_not_mistake_mppi_for_prior(mppi_case, tmp_path, monkeypatch):
    matrix, checkpoint = mppi_case
    prior_bundle, mppi_bundle = tmp_path / "prior", tmp_path / "mppi"
    prior = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/prior"], bundle_dir=prior_bundle)["results"][0]
    mppi = evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/mppi"], bundle_dir=mppi_bundle,
                                     reference_bundle=prior_bundle)["results"][0]
    deltas = [candidate["return"] - baseline["return"] for candidate, baseline in zip(mppi["episodes"], prior["episodes"])]
    assert mppi["paired_return_delta_vs_prior"]["mean"] == pytest.approx(sum(deltas) / len(deltas))
    monkeypatch.setattr(evaluator, "_make_env", lambda resolved: pytest.fail("late reference validation"))
    with pytest.raises(ValueError, match="prior-only"):
        evaluator.evaluate_matrix(matrix, checkpoint, selectors=["controller/mppi"], bundle_dir=tmp_path / "wrong",
                                  reference_bundle=mppi_bundle)
    assert not (tmp_path / "wrong").exists()
