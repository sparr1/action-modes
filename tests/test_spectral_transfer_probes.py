"""Spectral probes use common fixed targets without changing the learner."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params
from tests.test_aux_critic_transfer import critic_params
from utils.spectral_transfer_probes import (
    build_spectral_context, evaluate_spectral_handoff, evaluate_spectral_objectives,
    spectral_probe_settings,
)


SETTINGS = dict(state_count=6, action_count=3, mc_rollouts=2)
OBSERVATION = np.array([1., .2, -.1], dtype=np.float32)


@pytest.fixture
def wrapped():
    model = _model_from_params(critic_params("return", inner_critic_scope="action", inner_rollout_horizon=2))
    # A newly initialized return critic may be identically zero; make the
    # reference action-sensitive so gradient and heldout tests are substantive.
    generator = torch.Generator().manual_seed(271)
    with torch.no_grad():
        for head in model.agent.inner_engine._critic_base.modules_list:
            head[-1].weight.copy_(.1 * torch.randn(head[-1].weight.shape, generator=generator))
    yield model
    model.close()


def context(wrapped, **kwargs):
    return build_spectral_context(wrapped, OBSERVATION, controller_seed=55,
        episode_seed=101, decision=1, settings=SETTINGS, **kwargs)


def test_context_matrix_names_shapes_and_targets(wrapped):
    result = context(wrapped)
    assert result["states"].shape == (6, wrapped.cfg.latent_dim)
    assert result["actions"].shape == (6, 3, wrapped.cfg.action_dim)
    assert result["samples"].shape == (2, 6, 3)
    assert result["labels"].shape == (6, 3)
    torch.testing.assert_close(result["labels"], result["samples"].mean(0), rtol=0, atol=0)
    assert len(result["inputs"]["actor"]) == 3
    assert len(result["inputs"]["critic"]) == 6
    for component in ("actor", "critic"):
        prior = getattr(wrapped.agent.inner_engine, f"_{component}_base")
        matrices = {name: value for name, value in prior.named_parameters() if value.ndim == 2}
        assert matrices.keys() == result["inputs"][component].keys() == result["gradients"][component].keys()
        for name, matrix in matrices.items():
            inputs, gradient = result["inputs"][component][name], result["gradients"][component][name]
            assert inputs.shape == (6 if component == "actor" else 18, matrix.shape[1])
            assert gradient.shape == matrix.shape
            assert not inputs.requires_grad and not gradient.requires_grad
            assert torch.isfinite(inputs).all() and torch.isfinite(gradient).all()
    # The mean-policy surrogate deliberately does not score variance rows.
    assert torch.count_nonzero(result["gradients"]["actor"]["2.weight"][wrapped.cfg.action_dim:]) == 0


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_all_requested_horizons_have_finite_fixed_reference_probes(horizon):
    wrapped = _model_from_params(critic_params("return", inner_critic_scope="action",
        train_unroll_horizon=3, inner_rollout_horizon=horizon))
    try:
        result = context(wrapped)
        assert result["metadata"]["horizon"] == horizon
        json.dumps(result["metadata"], allow_nan=False)
    finally:
        wrapped.close()


def test_distributional_five_head_critic_uses_decoded_per_head_fixed_errors():
    wrapped = _model_from_params(critic_params("return", inner_critic_scope="action",
        q_representation="distributional", num_q=5))
    try:
        bank = context(wrapped)
        assert len(bank["inputs"]["critic"]) == 15
        assert len(bank["gradients"]["critic"]) == 15
        for head in range(5):
            output = bank["gradients"]["critic"][f"modules_list.{head}.2.weight"]
            assert output.shape[0] == wrapped.cfg.q_num_bins
            assert torch.isfinite(output).all()
        values = evaluate_spectral_objectives(wrapped, bank["prior_states"], bank)
        assert values["critic_loss"] == pytest.approx(bank["metadata"]["reference_losses"]["critic"])
    finally:
        wrapped.close()


def test_context_exactly_preserves_weights_modes_gradients_and_all_rng(wrapped):
    engine, model = wrapped.agent.inner_engine, wrapped.agent.model
    with engine.diagnostic_initialization():
        wrapped.predict(OBSERVATION, deterministic=True, episode_start=True)
    for index, module in enumerate(model.modules()):
        module.training = index % 2 == 0
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, .125) if index % 2 else None
    outer, learner = _clone_tree(model.state_dict()), engine.export_diagnostic_state()
    rng = _clone_tree(engine.rng.training_state_dict())
    modes = [module.training for module in model.modules()]
    grads = [None if parameter.grad is None else parameter.grad.clone() for parameter in model.parameters()]
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    with torch.no_grad():
        scoring = context(wrapped)
        heldout = context(wrapped, purpose="heldout")
        evaluate_spectral_handoff(wrapped, donor=learner, initial_states=heldout["prior_states"],
                                  final_states=learner["modules"], context=heldout)
    _assert_tree_equal(outer, model.state_dict())
    _assert_tree_equal(learner, engine.export_diagnostic_state())
    _assert_tree_equal(rng, engine.rng.training_state_dict())
    assert modes == [module.training for module in model.modules()]
    assert requires_grad == [parameter.requires_grad for parameter in model.parameters()]
    for before, parameter in zip(grads, model.parameters()):
        if before is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(before, parameter.grad, rtol=0, atol=0)
    torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)
    np.testing.assert_equal(numpy_rng, np.random.get_state())
    assert python_rng == random.getstate()
    assert scoring["metadata"]["seed"] != heldout["metadata"]["seed"]


def test_bank_is_repeatable_independent_of_candidate_and_heldout_is_separate(wrapped):
    scoring = context(wrapped)
    altered = deepcopy(scoring["prior_states"])
    altered["actor"]["2.weight"].add_(.1)
    altered["critic"]["modules_list.0.2.weight"].sub_(.1)
    evaluate_spectral_objectives(wrapped, altered, scoring)
    repeated, heldout = context(wrapped), context(wrapped, purpose="heldout")
    for name in ("states", "actions", "samples", "labels", "inputs", "gradients", "metadata"):
        _assert_tree_equal(scoring[name], repeated[name])
    assert scoring["metadata"]["action_sha256"] != heldout["metadata"]["action_sha256"]
    assert scoring["metadata"]["state_sha256"] != heldout["metadata"]["state_sha256"]
    assert scoring["metadata"]["target_sha256"] != heldout["metadata"]["target_sha256"]
    # Actual current root is shared; only independent sampled support differs.
    torch.testing.assert_close(scoring["states"][0], heldout["states"][0], rtol=0, atol=0)
    with pytest.raises(ValueError, match="heldout"):
        evaluate_spectral_handoff(wrapped, donor=None, initial_states=altered, context=scoring)


@pytest.mark.parametrize("component", ["actor", "critic"])
def test_gradient_scores_match_local_fixed_objective_finite_difference(wrapped, component):
    bank = context(wrapped)
    name, gradient = max(bank["gradients"][component].items(), key=lambda pair: float(pair[1].norm()))
    norm = float(gradient.norm())
    assert norm > 1e-5
    direction, epsilon = gradient / norm, 1e-3
    plus, minus = deepcopy(bank["prior_states"]), deepcopy(bank["prior_states"])
    plus[component][name].add_(epsilon * direction)
    minus[component][name].sub_(epsilon * direction)
    losses = [evaluate_spectral_objectives(wrapped, states, bank)[f"{component}_loss"] for states in (plus, minus)]
    derivative = (losses[0] - losses[1]) / (2 * epsilon)
    assert derivative == pytest.approx(norm, rel=.015, abs=2e-5)
    assert losses[1] < losses[0]


def test_handoff_reports_spectrum_true_norm_alignment_and_post_j_losses(wrapped):
    bank = context(wrapped, purpose="heldout")
    prior, donor = bank["prior_states"], deepcopy(bank["prior_states"])
    initial = deepcopy(prior)
    for component in ("actor", "critic"):
        for name in bank["inputs"][component]:
            gradient = bank["gradients"][component][name]
            delta = -.02 * gradient
            donor[component][name].add_(delta)
            initial[component][name].add_(.5 * delta)
    result = evaluate_spectral_handoff(wrapped, donor={"modules": donor}, initial_states=initial,
                                      final_states=donor, context=bank, ranks=(1, 2, 4))
    assert set(result["objectives"]) == {"prior", "donor", "initial", "final"}
    for component in ("actor", "critic"):
        summary = result["summary"]
        assert summary[f"{component}_transferred_energy_ratio"] == pytest.approx(.25, rel=2e-4)
        assert summary[f"{component}_donor_initial_cosine"] == pytest.approx(1., abs=1e-5)
        assert summary[f"{component}_initial_first_order_benefit"] > 0
        assert f"{component}_post_j_loss_gain_vs_prior" in summary
        assert f"{component}_mean_energy_at_rank_2" in summary
        for name, row in result["layers"][component].items():
            assert row["spectrum"]["singular_values"]
            assert row["spectrum"]["component_benefits"]
            if row["donor_squared_norm"] > 1e-10:
                assert row["transferred_energy_ratio"] == pytest.approx(.25, rel=.02)
                assert row["prior_input_output_relative_mse"] == pytest.approx(.25, rel=.02)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("component", ["actor", "critic"])
def test_global_projection_handoff_marks_nonzero_over_zero_layer_ratios_undefined(wrapped, component):
    from utils.spectral_transfer import spectral_state
    bank = context(wrapped, purpose="heldout")
    prior = bank["prior_states"]
    donor, initial = deepcopy(prior), deepcopy(prior)
    gradients = bank["gradients"][component]
    source, gradient = max(gradients.items(), key=lambda pair: float(pair[1].norm()))
    assert gradient.norm() > 1e-5
    donor[component][source].add_(.2 * gradient / gradient.norm())
    initial[component] = spectral_state(prior[component], donor[component],
        parameter_names=gradients.keys(), gradients=gradients,
        spec={"method": "gradient_projection"}, include_layer_metrics=False)
    result = evaluate_spectral_handoff(wrapped, donor={"modules": donor},
        initial_states=initial, final_states=initial, context=bank, ranks=(1, 2))
    introduced = [row for row in result["layers"][component].values()
                  if row["donor_squared_norm"] == 0 and row["initial_squared_norm"] > 0]
    assert introduced
    assert any(row["prior_input_output_mse"] > 0 for row in introduced)
    for row in introduced:
        assert row["transferred_energy_ratio"] is None
        assert row["parameter_residual_energy_ratio"] is None
        if row["prior_input_output_mse"] > 0:
            assert row["prior_input_output_relative_mse"] is None
    # Component totals still have a nonzero donor denominator.
    assert np.isfinite(result["summary"][f"{component}_transferred_energy_ratio"])
    json.dumps(result, allow_nan=False)


def test_first_decision_zero_donor_and_component_only_activation_context(wrapped):
    inputs_only = context(wrapped, components=("actor",), compute_gradients=False)
    assert set(inputs_only["inputs"]) == {"actor"}
    assert inputs_only["gradients"] == {}
    bank = context(wrapped, purpose="heldout")
    result = evaluate_spectral_handoff(wrapped, donor=None, initial_states=bank["prior_states"], context=bank)
    assert result["donor_available"] is False
    assert set(result["objectives"]) == {"prior", "initial"}
    for component in ("actor", "critic"):
        assert result["summary"][f"{component}_transferred_energy_ratio"] == 0.
        for row in result["layers"][component].values():
            assert row["zero_donor"]
            assert row["spectrum"]["stable_rank"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("purpose,components,gradients,has_actions,has_returns", [
    ("scoring", ("actor",), True, False, False),
    ("scoring", ("actor",), False, False, False),
    ("scoring", ("critic",), False, True, False),
    ("scoring", ("actor", "critic"), False, True, False),
    ("scoring", ("critic",), True, True, True),
    ("heldout", ("actor",), False, True, True),
    ("heldout", ("actor", "critic"), True, True, True),
])
def test_scoring_skips_unused_mc_preserves_exact_inputs_gradients_and_rng(
        wrapped, monkeypatch, purpose, components, gradients, has_actions, has_returns):
    from utils.spectral_transfer_probes import Reference
    complete = context(wrapped, purpose=purpose)
    calls = []
    original = Reference.returns
    def count_returns(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(Reference, "returns", count_returns)
    global_rng = torch.get_rng_state().clone()
    engine_rng = _clone_tree(wrapped.agent.inner_engine.rng.training_state_dict())
    result = context(wrapped, purpose=purpose, components=components, compute_gradients=gradients)
    assert len(calls) == int(has_returns)
    torch.testing.assert_close(global_rng, torch.get_rng_state(), rtol=0, atol=0)
    _assert_tree_equal(engine_rng, wrapped.agent.inner_engine.rng.training_state_dict())
    torch.testing.assert_close(complete["states"], result["states"], rtol=0, atol=0)
    for component in components:
        _assert_tree_equal(result["inputs"][component], complete["inputs"][component])
        if gradients:
            _assert_tree_equal(result["gradients"][component], complete["gradients"][component])
    assert result["metadata"]["actions_available"] == has_actions
    assert result["metadata"]["return_labels_available"] == has_returns
    assert result["metadata"]["gradients_available"] == gradients
    if has_actions:
        torch.testing.assert_close(result["actions"], complete["actions"], rtol=0, atol=0)
        assert result["metadata"]["action_sha256"] == complete["metadata"]["action_sha256"]
    else:
        assert result["actions"] is None and result["metadata"]["action_sha256"] is None
    if has_returns:
        for name in ("labels", "samples"):
            torch.testing.assert_close(result[name], complete[name], rtol=0, atol=0)
        assert result["metadata"]["target_sha256"] == complete["metadata"]["target_sha256"]
    else:
        assert result["labels"] is None and result["samples"] is None
        assert result["metadata"]["target_sha256"] is None
        with pytest.raises(ValueError, match="complete"):
            evaluate_spectral_objectives(wrapped, result["prior_states"], result)
    assert set(result["metadata"]["reference_losses"]) == (set(components) if gradients or purpose == "heldout" else set())
    json.dumps(result["metadata"], allow_nan=False)


@pytest.mark.parametrize("settings", [dict(state_count=1), dict(mc_rollouts=True), dict(action_count=1), dict(unrecognized=2)])
def test_bad_settings_rejected(settings):
    with pytest.raises(ValueError):
        spectral_probe_settings(settings)


def test_unsupported_soft_return_semantics_rejected():
    model = _model_from_params(critic_params("soft", inner_critic_scope="action"))
    try:
        with pytest.raises(ValueError, match="return/return"):
            context(model)
    finally:
        model.close()
