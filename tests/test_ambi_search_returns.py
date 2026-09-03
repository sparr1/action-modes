import pytest
import torch

from RL.tdmpc2_core.common.search_returns import (
    full_suffix_target,
    importance_sampling_ratios,
    lambda_return_target,
    n_step_target,
    resimulated_suffix_target,
    retrace_target,
    soft_value,
    td0_target,
    vtrace_actor_loss,
    vtrace_targets,
)


def _sequence(values):
    return torch.tensor(values, dtype=torch.float64).reshape(1, -1, 1)


def test_td0_uses_soft_successor_value_and_excludes_current_action_entropy():
    reward = _sequence([1.0])
    next_q = torch.tensor([[4.0]], dtype=torch.float64)
    next_log_prob = torch.tensor([[-0.2]], dtype=torch.float64)
    bootstrap = soft_value(next_q, next_log_prob, entropy_coefficient=0.5)

    target = td0_target(reward, bootstrap, discount=0.5)
    torch.testing.assert_close(target, torch.tensor([[3.05]], dtype=torch.float64))

    # There is no API slot for the current log-probability in TD(0).  For a
    # multi-step expansion index zero is likewise ignored explicitly.
    target_with_current_logp = n_step_target(
        _sequence([1.0, 999.0]),
        bootstrap,
        steps=1,
        discount=0.5,
        action_log_probs=_sequence([-100.0, 20.0]),
        entropy_coefficient=0.5,
    )
    torch.testing.assert_close(target_with_current_logp, target)


def test_h3_n_step_and_full_suffix_have_correct_entropy_and_leaf_discount():
    rewards = _sequence([1.0, 2.0, 3.0])
    log_probs = _sequence([-9.0, 0.2, 0.4])

    n2 = n_step_target(
        rewards,
        torch.tensor([[8.0]], dtype=torch.float64),
        steps=2,
        discount=0.5,
        action_log_probs=log_probs,
        entropy_coefficient=0.5,
    )
    full, diagnostics = full_suffix_target(
        rewards,
        torch.tensor([[10.0]], dtype=torch.float64),
        discount=0.5,
        action_log_probs=log_probs,
        entropy_coefficient=0.5,
        return_diagnostics=True,
    )

    torch.testing.assert_close(n2, torch.tensor([[3.95]], dtype=torch.float64))
    torch.testing.assert_close(full, torch.tensor([[3.90]], dtype=torch.float64))
    torch.testing.assert_close(
        diagnostics["leaf_contribution"],
        torch.tensor([[1.25]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics["effective_return_length"],
        torch.tensor([[3.0]], dtype=torch.float64),
    )


def test_termination_and_padding_mask_rewards_and_outer_leaf():
    rewards = _sequence([1.0, 2.0, 100.0])
    terminal = torch.tensor([[[False], [True], [False]]])
    terminated_target = full_suffix_target(
        rewards,
        torch.tensor([[10.0]], dtype=torch.float64),
        horizon=3,
        discount=0.5,
        terminated=terminal,
    )
    torch.testing.assert_close(
        terminated_target, torch.tensor([[2.0]], dtype=torch.float64)
    )

    valid = torch.tensor([[[True], [True], [False]]])
    shortened_target = full_suffix_target(
        rewards,
        torch.tensor([[10.0]], dtype=torch.float64),
        horizon=torch.tensor([2]),
        discount=0.5,
        valid=valid,
    )
    torch.testing.assert_close(
        shortened_target, torch.tensor([[4.5]], dtype=torch.float64)
    )


def test_pdis_conditions_on_anchor_and_uses_w_n_minus_one_for_bootstrap():
    rewards = _sequence([1.0, 2.0, 3.0])
    ratios = _sequence([100.0, 2.0, 3.0])
    target, diagnostics = n_step_target(
        rewards,
        torch.tensor([[8.0]], dtype=torch.float64),
        steps=2,
        discount=0.5,
        importance_ratios=ratios,
        return_diagnostics=True,
    )

    # r0 + gamma*rho1*r1 + gamma^2*rho1*bootstrap.  rho0 is
    # conditioned away and rho2 must not enter the W_(n-1) bootstrap weight.
    torch.testing.assert_close(target, torch.tensor([[7.0]], dtype=torch.float64))
    torch.testing.assert_close(
        diagnostics["cumulative_importance_weight"],
        torch.tensor([[2.0]], dtype=torch.float64),
    )


def test_pdis_reports_ess_of_cumulative_target_weights():
    rewards = torch.zeros(2, 3, 1, dtype=torch.float64)
    ratios = torch.tensor(
        [[[100.0], [2.0], [3.0]], [[100.0], [0.5], [0.5]]],
        dtype=torch.float64,
    )
    _, diagnostics = n_step_target(
        rewards,
        torch.ones(2, 1, dtype=torch.float64),
        steps=3,
        discount=0.5,
        importance_ratios=ratios,
        return_diagnostics=True,
    )

    # The conditioned anchor ratio is absent, leaving final PDIS weights
    # W_2=[2*3, .5*.5].  Their sample ESS is distinct from an ESS over the
    # four individual future-action ratios.
    weights = torch.tensor([6.0, 0.25], dtype=torch.float64)
    expected_ess = weights.sum().square() / weights.square().sum()
    torch.testing.assert_close(
        diagnostics["pdis_weight_mean"], weights.mean()
    )
    torch.testing.assert_close(
        diagnostics["pdis_weight_max"], weights.max()
    )
    torch.testing.assert_close(
        diagnostics["pdis_weight_ess"], expected_ess
    )
    torch.testing.assert_close(
        diagnostics["pdis_weight_normalized_ess"], expected_ess / 2
    )


@pytest.mark.parametrize(
    ("trace_lambda", "expected"),
    [(0.0, 3.0), (0.5, 3.5), (1.0, 4.0)],
)
def test_h3_finite_lambda_return_includes_exact_endpoints(trace_lambda, expected):
    target, diagnostics = lambda_return_target(
        _sequence([1.0, 2.0, 3.0]),
        _sequence([4.0, 8.0, 10.0]),
        trace_lambda=trace_lambda,
        discount=0.5,
        return_diagnostics=True,
    )
    torch.testing.assert_close(
        target, torch.tensor([[expected]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        diagnostics["mixture_weights"].sum(dim=1),
        torch.ones(1, 1, dtype=torch.float64),
    )


def test_retrace_is_td_error_control_variate_with_depth_aware_successors():
    target, diagnostics = retrace_target(
        _sequence([1.0, 2.0, 3.0]),
        _sequence([4.0, 5.0, 6.0]),
        _sequence([5.0, 6.0, 10.0]),
        torch.log(_sequence([100.0, 0.5, 2.0])),
        discount=0.5,
        trace_lambda=0.8,
        return_diagnostics=True,
    )

    # deltas=(-.5,0,2); c1=.4,c2=.8, hence
    # 4 + delta0 + .5*c1*delta1 + .25*c1*c2*delta2 = 3.66.
    torch.testing.assert_close(target, torch.tensor([[3.66]], dtype=torch.float64))
    torch.testing.assert_close(
        diagnostics["trace_c"], _sequence([0.8, 0.4, 0.8])
    )
    assert diagnostics["ratio_clipped_fraction"].item() == pytest.approx(0.5)


def test_resimulation_uses_same_on_policy_return_algebra_without_is():
    kwargs = {
        "rewards": _sequence([1.0, 2.0, 3.0]),
        "leaf_value": torch.tensor([[10.0]], dtype=torch.float64),
        "discount": 0.5,
    }
    torch.testing.assert_close(
        resimulated_suffix_target(**kwargs), full_suffix_target(**kwargs)
    )
    with pytest.raises(ValueError, match="do not use importance"):
        resimulated_suffix_target(**kwargs, importance_ratios=_sequence([1, 1, 1]))


def test_importance_ratios_are_exact_and_masked_padding_is_neutral():
    ratios = importance_sampling_ratios(
        _sequence([-1.0, -2.0, 50.0]),
        _sequence([-2.0, -2.5, -50.0]),
        valid=torch.tensor([[[True], [True], [False]]]),
    )
    torch.testing.assert_close(
        ratios,
        _sequence([torch.exp(torch.tensor(1.0)).item(), torch.exp(torch.tensor(0.5)).item(), 1.0]),
    )


def test_h3_vtrace_targets_clipped_ratios_and_actor_advantages():
    result = vtrace_targets(
        _sequence([1.0, 2.0, 3.0]),
        _sequence([4.0, 5.0, 6.0]),
        _sequence([5.0, 6.0, 10.0]),
        torch.log(_sequence([2.0, 0.5, 3.0])),
        discount=0.5,
        trace_lambda=0.8,
        rho_clip=1.0,
        c_clip=1.0,
        pg_rho_clip=1.0,
    )

    torch.testing.assert_close(
        result["value_target"], _sequence([3.66, 5.4, 8.0])
    )
    torch.testing.assert_close(
        result["pg_advantage"], _sequence([-0.3, 0.5, 2.0])
    )
    torch.testing.assert_close(result["clipped_rho"], _sequence([1.0, 0.5, 1.0]))
    torch.testing.assert_close(result["trace_c"], _sequence([0.8, 0.4, 0.8]))
    assert result["ratio_clipped_fraction"].item() == pytest.approx(2 / 3)
    expected_ess = 5.5**2 / (2.0**2 + 0.5**2 + 3.0**2)
    assert result["ess"].item() == pytest.approx(expected_ess)
    assert result["normalized_ess"].item() == pytest.approx(expected_ess / 3)
    # gamma^3 * c0 * c1 * clipped_rho2 * V_outer(z3)
    torch.testing.assert_close(
        result["leaf_contribution"],
        torch.tensor([[0.4]], dtype=torch.float64),
    )


def test_vtrace_actor_loss_keeps_entropy_as_a_separate_regularizer():
    log_prob = _sequence([-0.2, -0.4]).requires_grad_()
    loss, diagnostics = vtrace_actor_loss(
        log_prob,
        _sequence([1.0, 2.0]),
        entropy_coefficient=0.1,
        return_diagnostics=True,
    )

    torch.testing.assert_close(loss, torch.tensor(0.47, dtype=torch.float64))
    torch.testing.assert_close(
        diagnostics["policy_gradient_loss"],
        torch.tensor(0.5, dtype=torch.float64),
    )
    torch.testing.assert_close(
        diagnostics["entropy_loss"],
        torch.tensor(-0.03, dtype=torch.float64),
    )
    loss.backward()
    assert log_prob.grad is not None
