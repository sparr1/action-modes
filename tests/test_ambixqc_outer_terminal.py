"""Horizon-only outer XQC targets preserve earlier Bellman/BatchNorm paths."""

from copy import deepcopy
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from RL.tdmpc2_core.inner_xqc import InnerXQCEngine
from RL.tdmpc2_core.xqc_controller import LatentXQCBatch, LatentXQCConfig, LatentXQCController
from test_ambixqc_core import _batch, _tree_equal
from test_ambixqc_inner import _agent
from test_ambixqc_inner_j6 import deterministic_xqc_numerics
from test_ambixqc_prior_checkpoint import _wrapper


def _controller():
    return LatentXQCController(2, 1, LatentXQCConfig(
        actor_net_arch=(8,), critic_net_arch=(8,), num_atoms=7,
        vmin=-2, vmax=2, init_temperature=0.3, target_entropy=-0.5,
        optimizer_backend="single_tensor",
    ))


def _target_oracle(log_q, reward, mask, discount, alpha, log_prob, support):
    """Independent scalar-loop C51 projection, including entropy and reward scale."""
    probabilities = log_q.exp()
    expectations = (probabilities * support).sum(-1)
    heads = expectations.argmin(0)
    result = torch.zeros_like(log_q[0])
    spacing = float((support[-1] - support[0]) / (support.numel() - 1))
    clipped = 0
    for row in range(reward.numel()):
        for atom, value in enumerate(support):
            target = float(reward[row] + discount * mask[row] * (value - alpha * log_prob[row]))
            target = min(max(target, float(support[0])), float(support[-1]))
            clipped += target == float(support[0]) or target == float(support[-1])
            position = min(max((target - float(support[0])) / spacing, 0), support.numel() - 1)
            lower, upper = math.floor(position), math.ceil(position)
            probability = probabilities[heads[row], row, atom]
            result[row, lower] += probability * (upper + (lower == upper) - position)
            result[row, upper] += probability * (position - lower)
    return result, expectations[heads, torch.arange(reward.numel())], heads, clipped / result.numel()


@pytest.mark.parametrize("requested", [[False, True, True, False], [False] * 4, [True] * 4])
def test_mixed_target_oracle_keeps_earlier_targets_and_all_inner_bn_inputs_exact(monkeypatch, requested):
    torch.manual_seed(14)
    base = _controller()
    inner = deepcopy(base)
    outer = _controller()
    with torch.no_grad():
        outer.actor.mean.bias.add_(0.7)
        outer.log_temperature.fill_(math.log(1.7))  # Must never replace inner alpha.
    data = LatentXQCBatch(
        latents=torch.tensor([[.1, .3], [-.2, .7], [.9, -.4], [.5, .5]]),
        actions=torch.tensor([[.2], [-.8], [.5], [0.]]),
        rewards=torch.tensor([.4, -.6, 1., -.1]),
        next_latents=torch.tensor([[.4, .1], [.1, .8], [-.6, .2], [.3, -.1]]),
        bootstrap_mask=torch.tensor([1., 1., 0., 1.]), discount=.9,
    )
    noise = torch.tensor([[.2], [-.9], [.1], [.8]])
    mask = torch.tensor(requested) & data.bootstrap_mask.bool()
    outer_before = deepcopy(outer.state_dict())
    global_rng = torch.get_rng_state().clone()
    baseline = base.critic_objective(data, next_noise=noise, reward_scale=2.5)
    actor_calls, critic_calls = [], []
    actor_sample = outer.actor.sample
    critic_log_probs = outer.critic.log_probs

    def sample(z, **kwargs):
        assert kwargs["bn_mode"] == "running"
        assert torch.equal(kwargs["noise"], noise)
        assert not kwargs.get("deterministic", False)
        actor_calls.append(z.clone())
        return actor_sample(z, **kwargs)

    def log_probs(z, action, **kwargs):
        assert kwargs["bn_mode"] == "running"
        result = critic_log_probs(z, action, **kwargs)
        critic_calls.append((action.clone(), result.clone()))
        return result

    monkeypatch.setattr(outer.actor, "sample", sample)
    monkeypatch.setattr(outer.critic, "log_probs", log_probs)
    monkeypatch.setattr(outer.critic_target, "log_probs", lambda *a, **k: pytest.fail("Used outer target Q"))
    actual = inner.critic_objective(
        data, next_noise=noise, reward_scale=2.5,
        outer_terminal_mask=torch.tensor(requested), outer_controller=outer,
    )
    assert len(actor_calls) == len(critic_calls) == 1
    with torch.no_grad():
        mean, log_std = outer.actor.distribution(data.next_latents, bn_mode="running")
        pre_tanh = mean + log_std.exp() * noise
        log_prob = (-.5 * noise.square() - log_std - .5 * math.log(2 * math.pi)
                    - 2 * (math.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))).sum(-1)
        torch.testing.assert_close(critic_calls[0][0], pre_tanh.tanh(), rtol=0, atol=0)
        expected_outer, values, heads, _ = _target_oracle(
            critic_calls[0][1], data.rewards / 2.5, data.bootstrap_mask,
            .9, inner.temperature, log_prob, outer.critic.support,
        )
    expected = torch.where(mask[:, None], expected_outer, baseline.target_probabilities)
    torch.testing.assert_close(actual.target_probabilities, expected, atol=2e-7, rtol=1e-6)
    assert torch.equal(actual.target_probabilities[~mask], baseline.target_probabilities[~mask])
    assert torch.equal(actual.current_log_probs, baseline.current_log_probs)
    assert torch.equal(actual.current_values, baseline.current_values)
    torch.testing.assert_close(actual.target_values, torch.where(mask, values, baseline.target_values), atol=1e-7, rtol=1e-6)
    assert torch.equal(actual.target_head, torch.where(mask, heads, baseline.target_head))
    expected_loss = -(expected.unsqueeze(0) * actual.current_log_probs).sum(-1).sum(0).mean()
    torch.testing.assert_close(actual.loss, expected_loss)
    assert _tree_equal(inner.state_dict(), base.state_dict())  # BN writes exactly match baseline.
    actual.loss.backward()
    assert all(parameter.grad is None for parameter in outer.parameters())
    assert _tree_equal(outer_before, outer.state_dict())
    assert torch.equal(global_rng, torch.get_rng_state())


@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("episodic", [False, True])
def test_boundary_sidecar_matches_dense_and_dynamic_order_across_rounds_and_reset(horizon, episodic):
    agent, _ = _agent(episodic=episodic)
    agent.cfg.inner_terminal_bootstrap = "outer"
    agent.cfg.inner_rollout_horizon = horizon
    agent.cfg.inner_replay_capacity = 2 * 2 * horizon
    engine = InnerXQCEngine(agent)
    engine._prepare_action()
    root_z = torch.zeros(1, 2)
    for _ in range(2):
        engine._collect_round(root_z)
    expected = torch.tensor(([False] * (2 * (horizon - 1)) + [True] * 2) * 2)
    assert torch.equal(engine.state.outer_terminal_flags, expected)
    assert engine.state.replay.next_sample_id == expected.numel()
    assert engine.state.outer_terminal_boundary_rows == 4
    engine._collect_diagnostics = False
    for _ in range(4):
        sampled = engine._sample_batch()
        assert torch.equal(sampled["outer_terminal_mask"], expected[sampled["sample_ids"]])
    assert engine.state.sampled_ids == []
    engine._release_action()
    engine.reset_for_evaluation(77, reuse_action_pool=True)
    engine._prepare_action()
    assert engine.state.replay.size == engine.state.replay.next_sample_id == 0
    assert not engine.state.outer_terminal_flags.any()
    assert engine.state.outer_terminal_boundary_rows == engine.state.outer_terminal_bootstrap_rows == 0


@pytest.mark.parametrize("horizon", [1, 3])
def test_true_termination_does_not_become_outer_bootstrap_at_horizon(horizon):
    agent, _ = _agent(episodic=True)
    agent.cfg.inner_terminal_bootstrap = "outer"
    agent.cfg.inner_rollout_horizon = horizon
    agent.cfg.inner_replay_capacity = 2 * 2 * horizon
    # Branch 0 terminates at the first step; branch 1 survives to the horizon.
    calls = []

    def termination(z):
        result = z.new_zeros(z.shape[0], 1)
        if not calls:
            result[0] = 1
        calls.append(z.shape[0])
        return result

    agent.model.termination = termination
    engine = InnerXQCEngine(agent)
    engine._prepare_action()
    engine._collect_round(torch.zeros(1, 2))
    assert engine.state.replay.size == horizon + 1
    expected = torch.tensor([False] * horizon + [True])
    assert torch.equal(engine.state.outer_terminal_flags[:horizon + 1], expected)
    assert engine.state.outer_terminal_boundary_rows == 1
    assert engine.state.replay.terminated[0] == 1


def test_boundary_sidecar_rejects_eviction_instead_of_mislabeling_sample_ids():
    agent, _ = _agent()
    agent.cfg.inner_terminal_bootstrap = "outer"
    agent.cfg.inner_replay_capacity = 3
    engine = InnerXQCEngine(agent)
    engine._prepare_action()
    with pytest.raises(ValueError, match="without eviction"):
        engine._collect_round(torch.zeros(1, 2))
    assert engine.state.replay.size == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware is unavailable"))])
def test_frozen_outer_terminal_real_updates_preserve_state_rng_and_repeat_actions(
    device, tmp_path, monkeypatch, deterministic_xqc_numerics
):
    source = _wrapper(device=device, inner_operator="none", xqc_optimizer_backend="auto")
    target = _wrapper(device=device, xqc_optimizer_backend="auto", inner_rounds=2,
                      inner_terminal_bootstrap="outer",
                      inner_rollouts_per_round=4, inner_rollout_horizon=2,
                      inner_updates_per_round=4, inner_batch_size=4, inner_replay_capacity=16)
    try:
        source.agent._update(*(tensor.to(device) for tensor in _batch(source.agent)))
        source.agent.observe_reward(2., False, False)
        path = tmp_path / "prior.pt"
        source.agent.save(str(path))
        target.load(str(path), frozen_evaluation=True)
        agent, engine = target.agent, target.agent.inner_engine
        before = agent.frozen_outer_state()
        observation, _ = target.env.reset(seed=101)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(agent.device).clone() if device == "cuda" else None
        batches = []
        sample = engine._sample_batch

        def sampled():
            batch = sample()
            batches.append(batch["outer_terminal_mask"].clone())
            return batch

        monkeypatch.setattr(engine, "_sample_batch", sampled)

        def episode():
            target.reset_for_evaluation(77, reuse_action_pool=True)
            actions = []
            for _ in range(2):
                batches.clear()
                actions.append(target.predict(observation, deterministic=True)[0])
                metrics = agent.last_inner_metrics
                assert metrics["inner_model_steps"] == 16
                assert tuple(metrics[f"inner_{name}_optimizer_steps"] for name in ("critic", "actor", "temperature")) == (8, 3, 3)
                assert metrics["inner_outer_terminal_boundary_rows"] == 8
                assert metrics["inner_outer_terminal_bootstrap_rows"] == sum(int(mask.sum()) for mask in batches)
                assert metrics["inner_outer_terminal_policy_evaluations"] == 32
                assert metrics["inner_outer_terminal_q_evaluations"] == 32
                assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
                assert _tree_equal(before, agent.frozen_outer_state())
            return np.stack(actions)

        first = episode()
        np.testing.assert_array_equal(first, episode())
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(agent.device))
    finally:
        source.env.close()
        target.env.close()
