"""Numerical contracts for the finite-horizon SAC Retrace return operator."""

import pytest
import torch

from RL.tdmpc2_core.common.retrace import retrace_targets


def _rows(values, *, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype).unsqueeze(-1)


def _three_step(*, dtype=torch.float64):
    return (
        _rows([[1, 2, 3], [4, 5, 6]], dtype=dtype),
        _rows([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=dtype),
        _rows([[10, 20, 30], [7, 8, 9]], dtype=dtype),
        _rows([[99, 4, 6], [99, 1, 2]], dtype=dtype),
        _rows([[0.123, 0.5, 0.25], [0.987, 1, 1]], dtype=dtype),
        torch.ones(2, 3, 1, dtype=torch.bool),
    )


def test_three_step_hand_oracle_returns_every_suffix_and_trace_length():
    targets, lengths, corrections = retrace_targets(*_three_step())
    # First trajectory: G2=18; G1=12 + .5*.25*(18-6)=13.5;
    # G0=6 + .5*.5*(13.5-4)=8.375. The root's own c is irrelevant.
    torch.testing.assert_close(targets, _rows([[8.375, 13.5, 18], [13.625, 13.25, 10.5]]))
    torch.testing.assert_close(lengths, _rows([[1.625, 1.25, 1], [3, 2, 1]]))
    torch.testing.assert_close(corrections, _rows([[2.375, 1.5, 0], [6.125, 4.25, 0]]))


@pytest.mark.parametrize("terminal", [False, True])
def test_horizon_one_is_one_step_and_ignores_own_q_and_coefficient(terminal):
    reward = _rows([[3]])
    discount = _rows([[0 if terminal else 0.9]])
    unused = _rows([[float("nan")]])
    value = unused if terminal else _rows([[11]])
    targets, lengths, corrections = retrace_targets(
        reward, discount, value, unused, unused, torch.ones_like(reward, dtype=torch.bool),
    )
    torch.testing.assert_close(targets, _rows([[3 if terminal else 12.9]]))
    torch.testing.assert_close(lengths, _rows([[1]]))
    torch.testing.assert_close(corrections, _rows([[0]]))


def test_zero_trace_coefficients_recover_one_step_even_with_nan_recorded_q():
    reward, discount, value, q, c, valid = _three_step()
    targets, lengths, corrections = retrace_targets(
        reward, discount, value, torch.full_like(q, float("nan")), torch.zeros_like(c), valid,
    )
    torch.testing.assert_close(targets, reward + discount * value, rtol=0, atol=0)
    torch.testing.assert_close(lengths, torch.ones_like(lengths), rtol=0, atol=0)
    torch.testing.assert_close(corrections, torch.zeros_like(corrections), rtol=0, atol=0)


def test_on_policy_lambda_one_matches_three_reward_return_at_all_suffixes():
    reward = _rows([[1, 2, 3]])
    discount = _rows([[0.5, 0.5, 0.5]])
    # With matching on-policy Q/V samples and no entropy, the intermediate
    # baselines cancel, leaving the finite reward return plus frozen tail.
    value = _rows([[4, 6, 30]])
    q = _rows([[999, 4, 6]])
    targets, lengths, _ = retrace_targets(
        reward, discount, value, q, torch.ones_like(q), torch.ones_like(q, dtype=torch.bool),
    )
    torch.testing.assert_close(targets, _rows([[6.5, 11, 18]]))
    torch.testing.assert_close(lengths, _rows([[3, 2, 1]]))


def test_soft_values_include_entropy_at_every_interior_step():
    reward = _rows([[1, 2, 3]])
    discount = _rows([[0.5, 0.5, 0.5]])
    # Interior -alpha*log_pi bonuses are .2 and .4. The frozen terminal value
    # is supplied directly by the caller, without another inner entropy term.
    value = _rows([[4.2, 6.4, 30]])
    q = _rows([[999, 4, 6]])
    targets, _, _ = retrace_targets(
        reward, discount, value, q, torch.ones_like(q), torch.ones_like(q, dtype=torch.bool),
    )
    torch.testing.assert_close(targets, _rows([[6.7, 11.2, 18]]))


def test_clipped_importance_weights_use_next_action_and_match_expanded_td_sum():
    reward, discount, value, q, _, valid = _three_step()
    # A ratio above one is clipped; the smaller ratio shortens the trace.
    ratios = _rows([[99, 4, 0.25], [99, 0, 0.5]])
    c = 0.8 * ratios.clamp(max=1)
    actual, _, _ = retrace_targets(reward, discount, value, q, c, valid)
    td_error = reward + discount * value - q
    expected = []
    for start in range(3):
        estimate = q[:, start] + td_error[:, start]
        weight = torch.ones_like(estimate)
        for future in range(start + 1, 3):
            weight = weight * discount[:, future - 1] * c[:, future]
            estimate = estimate + weight * td_error[:, future]
        expected.append(estimate)
    torch.testing.assert_close(actual, torch.stack(expected, dim=1))


def test_termination_and_padding_mask_nan_before_arithmetic_and_keep_cutoff_tail():
    nan = float("nan")
    reward = _rows([[1, 2, nan], [1, 2, nan], [nan, nan, nan]])
    discount = _rows([[0.5, 0, nan], [0.5, 0.5, nan], [nan, nan, nan]])
    value = _rows([[4, nan, nan], [4, 10, nan], [nan, nan, nan]])
    q = _rows([[nan, 4, nan], [nan, 4, nan], [nan, nan, nan]])
    c = _rows([[nan, 1, nan], [nan, 1, nan], [nan, nan, nan]])
    valid = _rows([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.bool)
    targets, lengths, corrections = retrace_targets(reward, discount, value, q, c, valid)
    torch.testing.assert_close(targets, _rows([[2, 2, 0], [4.5, 7, 0], [0, 0, 0]]))
    torch.testing.assert_close(lengths, _rows([[2, 1, 0], [2, 1, 0], [0, 0, 0]]))
    torch.testing.assert_close(corrections, _rows([[-1, 0, 0], [1.5, 0, 0], [0, 0, 0]]))
    assert all(torch.isfinite(output).all() for output in (targets, lengths, corrections))


def test_terminal_does_not_use_a_valid_successor_or_its_nan_weight():
    args = list(_three_step())
    args[1][:, 0] = 0
    args[2][:, 0] = float("nan")
    args[3][:, 1] = float("nan")
    args[4][:, 1] = float("nan")
    targets, lengths, corrections = retrace_targets(*args)
    torch.testing.assert_close(targets[:, 0], args[0][:, 0])
    torch.testing.assert_close(lengths[:, 0], torch.ones_like(lengths[:, 0]))
    torch.testing.assert_close(corrections[:, 0], torch.zeros_like(corrections[:, 0]))
    assert torch.isfinite(targets).all()


def test_trajectories_are_independent_and_inputs_and_autograd_are_untouched():
    args = [value.requires_grad_() if value.is_floating_point() else value for value in _three_step()]
    before = [value.clone() for value in args]
    expected = retrace_targets(*args)
    for batch_index in range(2):
        individual = retrace_targets(*(value[batch_index:batch_index + 1] for value in args))
        for combined, single in zip(expected, individual):
            torch.testing.assert_close(combined[batch_index:batch_index + 1], single)
    for original, snapshot in zip(args, before):
        torch.testing.assert_close(original, snapshot, rtol=0, atol=0)
        assert original.grad is None
    assert all(not output.requires_grad and output.grad_fn is None for output in expected)


@pytest.mark.parametrize("horizon", [1, 3])
def test_fixed_horizon_compile_eager_backend_matches_eager(horizon):
    args = tuple(value[:, :horizon] for value in _three_step(dtype=torch.float32))
    compiled = torch.compile(retrace_targets, backend="eager", fullgraph=True)
    expected = retrace_targets(*args)
    actual = compiled(*args)
    for eager_output, compiled_output in zip(expected, actual):
        torch.testing.assert_close(compiled_output, eager_output, rtol=0, atol=0)


def test_rejects_incompatible_static_shapes():
    args = list(_three_step())
    args[-1] = args[-1][:, :1]
    with pytest.raises(ValueError, match="same.*shape"):
        retrace_targets(*args)
    with pytest.raises(ValueError, match="positive horizon"):
        retrace_targets(*(value[:, :0] for value in _three_step()))
