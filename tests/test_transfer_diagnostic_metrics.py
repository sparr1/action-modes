import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from utils.transfer_diagnostic_metrics import (
    action_value_metrics,
    aggregate_paired_root_metrics,
    feature_metrics,
    fit_stationary_targets,
    forced_action_mc_returns,
    paired_action_difference,
    paired_directional_metrics,
)


def mc_arguments(horizon=5, samples=4, batch=2):
    return dict(
        roots=torch.arange(batch, dtype=torch.float64)[:, None],
        first_actions=torch.ones(batch, 1, dtype=torch.float64),
        horizons=[1, horizon] if horizon != 1 else [1], discount=.7,
        policy_noise=torch.arange((horizon - 1) * samples * batch, dtype=torch.float64).reshape(horizon - 1, samples, batch, 1) / 10,
        tail_noise=torch.arange(horizon * samples * batch, dtype=torch.float64).reshape(horizon, samples, batch, 1) / 20,
        transition=lambda state, action: (state[:, 0] + 2 * action[:, 0], state + action, torch.zeros(len(state), dtype=torch.bool)),
        policy=lambda state, noise: (noise, -3 * torch.ones(len(state), dtype=state.dtype)),
        tail=lambda state, noise: 5 + state[:, 0] + noise[:, 0],
    )


@pytest.mark.parametrize("critic_target,alpha", [("reward_only", 9.), ("entropy_augmented", .2)])
def test_forced_first_action_mc_matches_analytic_all_horizons_and_noise(critic_target, alpha):
    arguments = mc_arguments()
    arguments["horizons"] = [5, 1, 3]
    before = {name: tensor.clone() for name, tensor in arguments.items() if torch.is_tensor(tensor)}
    rng = torch.get_rng_state().clone()
    result = forced_action_mc_returns(**arguments, critic_target=critic_target, entropy_coefficient=alpha)
    assert list(result) == [5, 1, 3]
    for horizon, actual in result.items():
        expected = torch.zeros(4, 2, dtype=torch.float64)
        for sample in range(4):
            for query in range(2):
                state, total = float(query), 0.
                for depth in range(horizon):
                    action = 1. if depth == 0 else float(arguments["policy_noise"][depth - 1, sample, query, 0])
                    total += .7 ** depth * (state + 2 * action)
                    if depth and critic_target == "entropy_augmented":
                        total += .7 ** depth * alpha * 3
                    state += action
                total += .7 ** horizon * (5 + state + arguments["tail_noise"][horizon - 1, sample, query, 0])
                expected[sample, query] = total
        torch.testing.assert_close(actual, expected)
    for name, tensor in before.items():
        torch.testing.assert_close(arguments[name], tensor)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(not tensor.requires_grad for tensor in result.values())


def test_horizon_one_does_not_sample_current_policy_or_add_entropy_at_tail():
    arguments = mc_arguments(horizon=1)
    def forbidden_policy(*args):
        raise AssertionError("H1 never samples the continuation policy")
    arguments["policy"] = forbidden_policy
    reward = forced_action_mc_returns(**arguments)[1]
    soft = forced_action_mc_returns(**arguments, critic_target="entropy_augmented", entropy_coefficient=100)[1]
    torch.testing.assert_close(reward, soft)


def test_termination_suppresses_later_reward_entropy_and_tail_without_calling_dead_states():
    arguments = mc_arguments(horizon=4, samples=3)
    arguments.update(discount=.5, horizons=[1, 2, 4])
    def transition(state, action):
        assert (state < 2).all()
        successor = state + 1
        return torch.ones(len(state)), successor, successor[:, 0] >= 2
    def policy(state, noise):
        assert (state < 2).all()
        return torch.ones_like(noise), -2 * torch.ones(len(state))
    def tail(state, noise):
        assert (state < 2).all()
        return 10 * torch.ones(len(state))
    arguments.update(transition=transition, policy=policy, tail=tail)
    result = forced_action_mc_returns(**arguments, critic_target="entropy_augmented", entropy_coefficient=.25)
    torch.testing.assert_close(result[1], torch.tensor([[6., 1.]] * 3, dtype=torch.float64))
    for horizon in (2, 4):
        # Second action gets gamma*entropy=.25 before terminating with reward.
        torch.testing.assert_close(result[horizon], torch.tensor([[1.75, 1.]] * 3, dtype=torch.float64))


@pytest.mark.parametrize("overrides,match", [
    ({"horizons": [1, 1]}, "unique positive"),
    ({"horizons": [0]}, "unique positive"),
    ({"horizons": [True]}, "unique positive"),
    ({"discount": 1.1}, "discount"),
    ({"policy_noise": torch.zeros(2, 4, 2, 1)}, "policy_noise"),
    ({"tail_noise": torch.zeros(5, 0, 2, 1)}, "tail_noise"),
    ({"entropy_coefficient": -1}, "entropy_coefficient"),
    ({"critic_target": "ambiguous"}, "critic_target"),
])
def test_mc_rejects_ambiguous_reference_contract(overrides, match):
    arguments = mc_arguments()
    arguments.update(overrides)
    with pytest.raises(ValueError, match=match):
        forced_action_mc_returns(**arguments)


def test_action_metrics_separate_common_value_offset_from_action_error_and_pair_noise():
    noise = np.array([-100., 0., 100.])[:, None]
    references = noise + np.array([0., 1., 2.])[None, :]
    metrics = action_value_metrics([10., 11., 12.], references)
    assert metrics["bias"] == metrics["rmse"] == 10
    assert metrics["centered_rmse"] == metrics["relative_rmse"] == 0
    assert metrics["spearman"] == metrics["pearson"] == pytest.approx(1)
    assert metrics["top_action_index"] == 2 and metrics["top_action_regret"] == 0
    assert metrics["selected_action_gain"] == 2 and metrics["selected_action_gain_se"] == 0
    assert metrics["reference_se"][0] > 50
    assert metrics["reference_relative_se"] == [0, 0, 0]
    assert paired_action_difference(references, 2, 0)["se"] == 0
    json.dumps(metrics, allow_nan=False)


def test_action_metrics_find_wrong_winner_and_handle_ties_and_single_draw():
    metrics = action_value_metrics([[2, 1, 0], [4, 3, 2]], [[0, 1, 2]])
    assert metrics["top_action_index"] == 0
    assert metrics["reference_top_action_index"] == 2
    assert metrics["top_action_regret"] == 2
    assert metrics["spearman"] == pytest.approx(-1)
    assert metrics["top_action_regret_se"] is None
    assert metrics["prediction_draw_std"] == pytest.approx([2 ** .5] * 3)
    constant = action_value_metrics([1, 1, 1], [[0, 1, 2], [0, 1, 2]])
    assert constant["spearman"] is constant["pearson"] is None
    tied = action_value_metrics([0, 0, 1], [[1, 1, 2]])
    assert tied["spearman"] == pytest.approx(1)


def test_directional_derivative_pairs_noise_before_computing_se():
    common_noise = np.array([-100, 0, 100])[:, None]
    roots = np.array([2., -1.])[None, :]
    step = .01
    metrics = paired_directional_metrics((roots + step) ** 2 + common_noise,
                                         (roots - step) ** 2 + common_noise, step)
    assert metrics["reference_slope"] == pytest.approx([4, -2])
    assert metrics["reference_slope_se"] == pytest.approx([0, 0], abs=1e-12)
    assert metrics["fraction_positive"] == [1, 0]
    assert paired_directional_metrics([2], [1], 1)["gain_se"] is None
    with pytest.raises(ValueError, match="positive"):
        paired_directional_metrics([2], [1], 0)


def test_paired_aggregation_weights_episodes_not_roots_or_mc_replicates():
    records = [{"episode": "a", "root": 0, "candidate": 100., "reference": 100.}] * 20
    records += [{"episode": "a", "root": 1, "candidate": 102., "reference": 100.},
                {"episode": "b", "root": 0, "candidate": 310., "reference": 300.}]
    result = aggregate_paired_root_metrics(records, value_key="candidate", reference_key="reference")
    assert result["mean"] == 5.5  # episode means 1 and 10
    assert result["episode_se"] == 4.5
    assert result["episodes"] == 2 and result["roots"] == 3 and result["records"] == 22
    assert result["uncertainty_unit"] == "episode"
    single = aggregate_paired_root_metrics(records[:20], value_key="candidate", reference_key="reference")
    assert single["episode_se"] is None
    with pytest.raises(KeyError):
        aggregate_paired_root_metrics([{"episode": 0, "root": 0, "candidate": 1}],
                                     value_key="candidate", reference_key="reference")


def fitting_data():
    x = torch.linspace(-1, 1, 31)[:, None]
    holdout = torch.linspace(-.97, .93, 19)[:, None]
    return x, 2 * x[:, 0] - .5, holdout, 2 * holdout[:, 0] - .5


def test_stationary_fit_fresh_clone_learns_known_mapping_and_broadcasts_heads():
    model = nn.Linear(1, 2)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    before = copy.deepcopy(model.state_dict())
    inputs, targets, heldout, heldout_targets = fitting_data()
    inputs.requires_grad_(True)
    targets.requires_grad_(True)
    batches = torch.arange(len(inputs)).expand(180, -1)
    rng = torch.get_rng_state().clone()
    result = fit_stationary_targets(model, lambda m, x: m(x), inputs, targets,
                                    heldout, heldout_targets, batch_indices=batches,
                                    learning_rate=.05, evaluation_steps=[20, 100])
    assert [row["step"] for row in result["curve"]] == [0, 20, 100, 180]
    assert result["curve"][-1]["heldout_mse"] < 1e-5
    assert result["curve"][-1]["gradient_norm"] is not None
    assert result["trainable_parameters"] == 4
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value)
    assert model.training and not result["model"].training
    assert inputs.grad is targets.grad is None
    assert torch.equal(torch.get_rng_state(), rng)


def test_fit_matches_dropout_rng_and_evaluation_does_not_change_training():
    model = nn.Sequential(nn.Linear(1, 4), nn.Dropout(.5), nn.Linear(4, 1))
    inputs, targets, heldout, heldout_targets = fitting_data()
    arguments = dict(batch_indices=torch.arange(len(inputs)).expand(8, -1), seed=37,
                     feature_fn=lambda m, x: torch.cat((m[0](x), torch.rand(len(x), 1)), dim=1))
    rng = torch.get_rng_state().clone()
    every = fit_stationary_targets(model, lambda m, x: m(x), inputs, targets, heldout, heldout_targets,
                                   **arguments)
    sparse = fit_stationary_targets(model, lambda m, x: m(x), inputs, targets, heldout, heldout_targets,
                                    evaluation_steps=[0, 8], **arguments)
    for left, right in zip(every["model"].parameters(), sparse["model"].parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert every["curve"][-1] == sparse["curve"][-1]
    assert torch.equal(torch.get_rng_state(), rng)


def test_fit_preserves_frozen_features_and_reports_optional_feature_statistics():
    model = nn.Sequential(nn.Linear(1, 4), nn.Tanh(), nn.Linear(4, 1))
    inputs, targets, heldout, heldout_targets = fitting_data()
    result = fit_stationary_targets(
        model, lambda m, x: m(x), inputs, targets, heldout, heldout_targets,
        batch_indices=torch.arange(len(inputs)).expand(3, -1),
        trainable_selector=lambda name, parameter: name.startswith("2."),
        feature_fn=lambda m, x: m[:2](x),
    )
    torch.testing.assert_close(result["model"][0].weight, model[0].weight)
    assert result["trainable_parameters"] == 5
    assert result["curve"][0]["features"]["features"] == 4
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_feature_rank_known_orthogonal_and_inactive_cases():
    values = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    metrics = feature_metrics(values)
    assert metrics["effective_rank"] == pytest.approx(2)
    assert metrics["stable_rank"] == pytest.approx(2)
    assert metrics["inactive_fraction"] == metrics["constant_fraction"] == 0
    zero = feature_metrics(torch.zeros(3, 2))
    assert zero["effective_rank"] == zero["stable_rank"] == 0
    assert zero["inactive_fraction"] == zero["constant_fraction"] == 1


def test_fit_rejects_cross_sample_broadcast_and_noninteger_batches():
    model = nn.Linear(1, 1)
    inputs, targets, heldout, heldout_targets = fitting_data()
    with pytest.raises(ValueError, match="integer"):
        fit_stationary_targets(model, lambda m, x: m(x), inputs, targets, heldout, heldout_targets,
                               batch_indices=torch.zeros(2, 3))
    with pytest.raises(ValueError, match="incompatible"):
        fit_stationary_targets(model, lambda m, x: m(x).squeeze(-1), inputs,
                               targets[:, None].expand(-1, 31), heldout, heldout_targets,
                               batch_indices=torch.arange(31).expand(1, -1))
