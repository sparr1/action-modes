"""MPPI evaluation reuses a sampled prior without relabeling either policy."""

from copy import deepcopy
import json

import numpy as np
import pytest

from utils.ambi_benchmark import (
    AMBI_SAMPLED_PRIOR_REFERENCE, BenchmarkBundle, protocol_for,
    reference_protocol_for, reference_returns,
)
from utils.eval_series_data import _bundle_prior_reference


SOURCE = "entity/project/backbone"
DIGEST = "a" * 64


def _resolved():
    return {
        "algorithm_config": {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": {
            "inner_operator": "mppi", "inner_mppi_eval_action": "weighted_elite"}},
        "environment": {"id": "Pendulum-v1"},
        "evaluation": {"execution_action": "mean",
                       "prior_reference": AMBI_SAMPLED_PRIOR_REFERENCE},
    }


def _prior(tmp_path):
    resolved = _resolved()
    resolved["algorithm_config"]["alg_params"] = {"inner_operator": "none"}
    resolved["evaluation"] = {"execution_action": "sample"}
    manifest = {
        "schema_version": 1, "status": "complete",
        "checkpoint": {"sha256": DIGEST, "source_run": SOURCE,
                       "metadata": {"checkpoint": {"step": 400000}}},
        "protocol": protocol_for(resolved, 55, 500),
        "runs": [{"config": resolved["algorithm_config"], "status": "complete",
                  "episodes": [{"seed": 101, "return": 10.}, {"seed": 102, "return": 20.}],
                  "result": {"outer_state_unchanged": True,
                             "outer_updates_before": 123, "outer_updates_after": 123}}],
    }
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_pairing_is_explicit_and_preserves_both_action_rules(tmp_path):
    resolved = _resolved()
    protocol = reference_protocol_for(resolved, 55, 500)
    assert protocol["action_rule"] == "mppi_weighted_elite"
    assert "prior_reference" not in protocol_for(resolved, 55, 500)
    path, manifest = _prior(tmp_path)
    with pytest.raises(ValueError, match="protocol does not match"):
        reference_returns(path, DIGEST, protocol_for(resolved, 55, 500))
    reference = reference_returns(path, DIGEST, protocol, source_run=SOURCE, seeds=[101, 102])
    assert reference == {101: 10., 102: 20.}
    assert reference.provenance["source_protocol"]["action_rule"] == "tanh_sample"
    assert reference.provenance["evaluation_protocol"]["action_rule"] == "mppi_weighted_elite"
    assert json.loads(path.read_text()) == manifest


@pytest.mark.parametrize("field,value", [
    ("controller_seed", 56), ("max_steps", 499), ("action_rule", "tanh_mean"),
    ("environment", {"id": "different"}), ("seed_scheme", "different"),
])
def test_reference_pairing_rejects_other_protocol_differences(tmp_path, field, value):
    path, manifest = _prior(tmp_path)
    manifest["protocol"][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol does not match"):
        reference_returns(path, DIGEST, reference_protocol_for(_resolved(), 55, 500))


def test_reference_requires_exact_checkpoint_source_seeds_and_frozen_state(tmp_path):
    path, manifest = _prior(tmp_path)
    protocol = reference_protocol_for(_resolved(), 55, 500)
    with pytest.raises(ValueError, match="checkpoint does not match"):
        reference_returns(path, "b" * 64, protocol)
    with pytest.raises(ValueError, match="backbone does not match"):
        reference_returns(path, DIGEST, protocol, source_run="other/project/run")
    with pytest.raises(ValueError, match="seeds do not match"):
        reference_returns(path, DIGEST, protocol, seeds=[101])
    manifest["runs"][0]["result"]["outer_state_unchanged"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="frozen-state validation"):
        reference_returns(path, DIGEST, protocol)


def test_portable_reference_is_copied_and_revalidated_before_publication(tmp_path, monkeypatch):
    monkeypatch.setattr("utils.ambi_benchmark.code_identity", lambda: {})
    path, manifest = _prior(tmp_path)
    protocol = reference_protocol_for(_resolved(), 55, 500)
    reference = reference_returns(path, DIGEST, protocol, source_run=SOURCE, seeds=[101, 102])
    bundle = BenchmarkBundle(tmp_path / "mppi", checkpoint=manifest["checkpoint"],
                             protocol=protocol, reference=reference)
    values, files = _bundle_prior_reference(
        bundle.manifest, bundle.path / "manifest.json", {"sha256": DIGEST, "step": 400000}, SOURCE)
    assert values == reference and set(files) == {"reference/manifest.json"}
    assert (bundle.path / "reference/manifest.json").read_bytes() == path.read_bytes()
    (bundle.path / "reference/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        _bundle_prior_reference(bundle.manifest, bundle.path / "manifest.json",
                                {"sha256": DIGEST, "step": 400000}, SOURCE)


def test_reference_marker_cannot_be_attached_to_other_controllers():
    resolved = _resolved()
    resolved["algorithm_config"]["alg_params"]["inner_operator"] = "sac"
    with pytest.raises(ValueError, match="weighted-elite"):
        reference_protocol_for(resolved, 55, 500)


def test_publication_validates_paired_deltas_against_portable_prior(tmp_path, monkeypatch):
    from utils.eval_series_data import normalize_bundle

    monkeypatch.setattr("utils.ambi_benchmark.code_identity", lambda: {})
    monkeypatch.setattr("utils.eval_series_data.scientific_identity", lambda *args: {"version": 1})
    path, manifest = _prior(tmp_path)
    resolved = {**_resolved(), "selector": "mppi/aux"}
    protocol = reference_protocol_for(resolved, 55, 500)
    reference = reference_returns(path, DIGEST, protocol, source_run=SOURCE, seeds=[101, 102])
    checkpoint = {**manifest["checkpoint"], "source_run_verified": True}
    bundle = BenchmarkBundle(tmp_path / "mppi", checkpoint=checkpoint,
                             protocol=protocol, reference=reference)
    run = bundle.start_run(resolved, "episodes")
    episodes = [{"seed": seed, "return": value, "length": 500,
                 "terminated": False, "truncated": True, "truncated_by_evaluator": False,
                 "control_seconds": 10., "model_metrics": {}}
                for seed, value in ((101, 30.), (102, 50.))]
    for episode in episodes:
        bundle.episode(run, episode, [])
    bundle.finish_run(run, result={"resolved_config": resolved["algorithm_config"]["alg_params"],
                                  "environment_seeds": [101, 102], "episodes": episodes,
                                  "outer_state_unchanged": True,
                                  "outer_updates_before": 123, "outer_updates_after": 123})
    bundle.finish()
    record, = normalize_bundle(bundle.path)
    assert record["metrics"]["eval/paired_gain_mean"] == 25.
    assert record["metrics"]["runtime/control_seconds_per_decision"] == .02
    assert record["identity"]["protocol"]["action_rule"] == "mppi_weighted_elite"
    run["episodes"][0]["paired_return_delta"] += 1.
    bundle.save()
    with pytest.raises(ValueError, match="Paired return delta"):
        normalize_bundle(bundle.path)


def test_real_frozen_mppi_bundle_is_repeatable_and_has_no_optimizer_updates(tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    from tests.test_ambi_early_h3_j8 import _tiny_checkpoint, _evaluate

    checkpoint, resolved = _tiny_checkpoint(tmp_path, "mppi")
    params = resolved["algorithm_config"]["alg_params"]
    for key in ("inner_rounds", "inner_rollouts_per_round", "inner_critic_updates_per_round",
                "inner_actor_updates_per_round"):
        params.pop(key, None)
    params.update(inner_operator="mppi", inner_finite_horizon=False,
                  inner_rollout_horizon=2, inner_mppi_iterations=1,
                  inner_model_step_budget=8, inner_mppi_num_elites=2,
                  inner_mppi_num_pi_trajs=0, inner_mppi_eval_action="weighted_elite",
                  inner_mppi_warm_start_scope="episode")
    # Exercise the new wrapper through the existing small-environment helper.
    native = evaluator.evaluate_preset
    wrapper = evaluator.evaluate_mppi_preset
    def wrapped(*args, **kwargs):
        monkeypatch.setattr(evaluator, "evaluate_preset", native)
        try:
            return wrapper(*args, **kwargs)
        finally:
            monkeypatch.setattr(evaluator, "evaluate_preset", wrapped)
    monkeypatch.setattr(evaluator, "evaluate_preset", wrapped)
    first, actions, bundle, run = _evaluate(
        monkeypatch, checkpoint, resolved, bundle_dir=tmp_path / "first", steps=3)
    second, repeated, _, _ = _evaluate(
        monkeypatch, checkpoint, deepcopy(resolved), bundle_dir=tmp_path / "second", steps=3)
    np.testing.assert_array_equal(actions, repeated)
    assert first["return"] == second["return"]
    assert first["outer_state_unchanged"]
    assert first["outer_updates_before"] == first["outer_updates_after"]
    assert first["model_metrics"]["inner_actor_optimizer_steps"]["mean"] == 0
    assert first["model_metrics"]["inner_critic_optimizer_steps"]["mean"] == 0
    assert first["model_metrics"]["inner_model_steps"]["count"] == 3
    assert first["model_metrics"]["inner_model_steps"]["sum"] == 24
    assert first["episodes"][0]["control_seconds"] > 0
    assert first["timing_convention"] == run["timing_convention"]
    assert first["timing_convention"]["first_call_startup_included"] is True
    assert first["timing_convention"]["unscored_warmup_actions"] == 0
    assert first["timing_convention"]["model_initialization_included"] is False
    assert bundle.manifest["protocol"]["action_rule"] == "mppi_weighted_elite"
    assert len(run["episodes"]) == 1 and run["trace_files"] == []
