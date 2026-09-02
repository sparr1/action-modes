from copy import deepcopy
import math

import gymnasium as gym
import pytest
import torch
import torch.nn.functional as F

from RL.AMBIXQC import AMBIXQC
from RL.tdmpc2_core.common import math as td_math


def _tiny_params(**overrides):
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 16,
        "mlp_dim": 16,
        "latent_dim": 8,
        "num_enc_layers": 2,
        "simnorm_dim": 4,
        "num_bins": 5,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "train_unroll_horizon": 2,
        "buffer_size": 32,
        "seed_steps": 4,
        "pretrain_steps": 1,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "wandb": False,
        "xqc_actor_net_arch": [8, 8],
        "xqc_critic_net_arch": [8, 8],
        "xqc_num_atoms": 11,
        "xqc_vmin": -2,
        "xqc_vmax": 2,
        "xqc_optimizer_backend": "single_tensor",
        "inner_rounds": 1,
        "inner_rollouts_per_round": 2,
        "inner_rollout_horizon": 2,
        "inner_updates_per_round": 1,
        "inner_batch_size": 2,
        "inner_replay_capacity": 4,
    }
    params.update(overrides)
    return params


def _tiny_model(**overrides):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    return AMBIXQC(
        "AMBIXQC",
        env,
        _tiny_params(**overrides),
        {"seed": 3, "device": "cpu", "env": "test", "total_steps": 10},
        {},
    )


def _batch(agent):
    horizon = int(agent.cfg.train_unroll_horizon)
    batch_size = int(agent.cfg.batch_size)
    obs_dim = int(agent.cfg.obs_shape["state"][0])
    torch.manual_seed(31)
    return (
        torch.randn(horizon + 1, batch_size, obs_dim),
        torch.randn(horizon, batch_size, agent.cfg.action_dim).tanh(),
        torch.randn(horizon, batch_size, 1),
        torch.zeros(horizon, batch_size, 1),
    )


def _tree_equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def test_deepest_xqc_value_loss_keeps_told_bptt_and_isolates_targets_and_actor():
    wrapper = _tiny_model()
    agent = wrapper.agent
    obs, action, reward, terminated = _batch(agent)
    with torch.no_grad():
        next_targets = agent.model.encode(obs[1:])
    target_buffers = {
        name: value.detach().clone()
        for name, value in agent.xqc_controller.critic_target.named_buffers()
    }
    online_buffers = {
        name: value.detach().clone()
        for name, value in agent.xqc_controller.critic.named_buffers()
    }

    losses = agent._recurrent_world_and_value_losses(
        obs, action, reward, terminated, next_targets
    )
    deepest = losses["critic"].per_sample_loss[-1].mean()
    agent.zero_grad(set_to_none=True)
    deepest.backward()

    encoder_grad = sum(
        parameter.grad.abs().sum()
        for parameter in agent.model._encoder.parameters()
        if parameter.grad is not None
    )
    dynamics_grad = sum(
        parameter.grad.abs().sum()
        for parameter in agent.model._dynamics.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad > 0
    assert dynamics_grad > 0
    assert all(
        parameter.grad is None
        for parameter in agent.xqc_controller.actor.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in agent.xqc_controller.critic_target.parameters()
    )
    for name, value in agent.xqc_controller.critic_target.named_buffers():
        assert torch.equal(value, target_buffers[name])
    assert any(
        not torch.equal(value, online_buffers[name])
        for name, value in agent.xqc_controller.critic.named_buffers()
        if name.endswith(("running_mean", "running_var"))
    )
    wrapper.env.close()


def test_episodic_termination_loss_uses_the_told_temporal_weights():
    wrapper = _tiny_model(episodic=True, rho=0.25)
    agent = wrapper.agent
    obs, action, reward, terminated = _batch(agent)
    terminated = terminated.clone()
    terminated[1].fill_(1.0)
    with torch.no_grad():
        next_targets = agent.model.encode(obs[1:])

    losses = agent._recurrent_world_and_value_losses(
        obs, action, reward, terminated, next_targets
    )
    per_sample = F.binary_cross_entropy_with_logits(
        losses["termination_prediction"], terminated, reduction="none"
    )
    per_time = per_sample.mean(dim=tuple(range(1, per_sample.ndim)))
    expected = td_math.reduce_temporal_loss(
        per_time,
        agent.cfg.rho,
        normalization=agent.cfg.temporal_loss_normalization,
        reference_horizon=agent.cfg.temporal_loss_reference_horizon,
        legacy_order="vector_sum_divide",
        weights=agent._transition_temporal_weights,
    )

    torch.testing.assert_close(losses["termination_loss"], expected)
    assert not torch.isclose(losses["termination_loss"], per_time.mean())
    wrapper.env.close()


def test_outer_update_is_finite_uses_xqc_order_and_keeps_target_bn_frozen():
    wrapper = _tiny_model()
    agent = wrapper.agent
    obs, action, reward, terminated = _batch(agent)
    target_buffers = {
        name: value.detach().clone()
        for name, value in agent.xqc_controller.critic_target.named_buffers()
    }
    actor_before = deepcopy(agent.xqc_controller.actor.state_dict())

    metrics = agent._update(obs, action, reward, terminated)

    assert agent.num_updates == agent.outer_version == 1
    assert agent.xqc_workspace.update_step == 1
    assert agent.xqc_workspace.actor_optimizer_steps == 1
    assert agent.xqc_workspace.temperature_optimizer_steps == 1
    assert metrics["actor_update_accepted"] == 1.0
    assert all(
        torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values()
    )
    for name, value in agent.xqc_controller.critic_target.named_buffers():
        assert torch.equal(value, target_buffers[name])
    assert any(
        not torch.equal(value, actor_before[name])
        for name, value in agent.xqc_controller.actor.state_dict().items()
        if name in actor_before and value.is_floating_point()
    )
    for module in (agent.xqc_controller.actor, agent.xqc_controller.critic):
        for child in module.modules():
            if isinstance(child, torch.nn.Linear):
                residual = (child.weight.norm(dim=1) - 1.0).abs().max()
                assert residual < 1e-6
    wrapper.env.close()


def test_outer_xqc_delay_runs_actor_and_temperature_at_zero_and_three():
    wrapper = _tiny_model()
    agent = wrapper.agent
    accepted = []
    for _ in range(4):
        accepted.append(agent._update(*_batch(agent))["actor_update_accepted"])

    assert accepted == [1.0, 0.0, 0.0, 1.0]
    assert agent.xqc_workspace.update_step == 4
    assert agent.xqc_workspace.actor_optimizer_steps == 2
    assert agent.xqc_workspace.temperature_optimizer_steps == 2
    wrapper.env.close()


def test_frozen_real_reward_scale_resets_on_timeout_and_imagination_does_not_touch_it():
    wrapper = _tiny_model()
    agent = wrapper.agent
    agent.observe_reward(2.0, False, False)
    agent.observe_reward(3.0, False, True)
    # Match the released recurrence: a boundary removes the preceding return
    # before adding the boundary transition's own reward.
    assert agent.reward_normalizer.return_accumulator == 3.0
    state = deepcopy(agent.reward_normalizer.state_dict())

    obs, _ = wrapper.env.reset(seed=3)
    action, _ = wrapper.predict(obs, deterministic=False)

    assert action.shape == wrapper.env.action_space.shape
    assert _tree_equal(agent.reward_normalizer.state_dict(), state)
    assert agent.last_inner_metrics["inner_reward_scale"] == pytest.approx(
        agent.reward_normalizer.scale
    )
    assert agent.last_inner_metrics["inner_reward_scale_initial"] == pytest.approx(
        agent.reward_normalizer.scale
    )
    assert agent.last_inner_metrics["inner_reward_scale_final"] == pytest.approx(
        agent.reward_normalizer.scale
    )
    assert agent.last_inner_metrics["inner_reward_scale_delta"] == pytest.approx(0.0)
    assert agent.last_inner_metrics["inner_reward_normalizer_count_initial"] == 2.0
    assert agent.last_inner_metrics["inner_reward_normalizer_count_final"] == 2.0
    assert agent.last_inner_metrics[
        "inner_reward_normalizer_imagined_updates"
    ] == 0.0
    wrapper.env.close()


def test_action_local_imagined_reward_statistics_are_fresh_and_never_write_back():
    wrapper = _tiny_model(
        inner_reward_normalization="action_local_imagined"
    )
    agent = wrapper.agent
    agent.observe_reward(2.0, False, False)
    agent.observe_reward(3.0, False, True)
    outer_state = deepcopy(agent.reward_normalizer.state_dict())

    obs, _ = wrapper.env.reset(seed=3)
    action, _ = wrapper.predict(obs, deterministic=False)

    assert action.shape == wrapper.env.action_space.shape
    assert _tree_equal(agent.reward_normalizer.state_dict(), outer_state)
    metrics = agent.last_inner_metrics
    assert metrics["inner_reward_scale_initial"] == pytest.approx(
        agent.reward_normalizer.scale
    )
    assert metrics["inner_reward_scale"] == pytest.approx(
        metrics["inner_reward_scale_final"]
    )
    assert metrics["inner_reward_scale_delta"] == pytest.approx(
        metrics["inner_reward_scale_final"] - metrics["inner_reward_scale_initial"]
    )
    assert metrics["inner_reward_normalizer_count_initial"] == 0.0
    assert metrics["inner_reward_normalizer_count_final"] == 4.0
    assert metrics["inner_reward_normalizer_imagined_updates"] == 4.0
    assert metrics["inner_reward_normalizer_imagined_updates"] == metrics[
        "inner_realized_model_steps"
    ]
    wrapper.env.close()


def test_portable_checkpoint_round_trip_and_bad_state_is_atomic():
    wrapper = _tiny_model()
    source = wrapper.agent
    source.observe_reward(1.5, False, False)
    for _ in range(4):
        source._update(*_batch(source))
    saved = deepcopy(source.checkpoint_state())

    restored_wrapper = _tiny_model()
    restored = restored_wrapper.agent
    restored.load(saved)
    assert restored.num_updates == 4
    assert _tree_equal(restored.checkpoint_state(), saved)

    before = deepcopy(restored.checkpoint_state())
    invalid = deepcopy(saved)
    invalid["semantic_signature"]["policy_delay"] = 99
    with pytest.raises(ValueError, match="semantics"):
        restored.load(invalid)
    assert _tree_equal(restored.checkpoint_state(), before)

    invalid = deepcopy(saved)
    invalid["xqc_workspace"]["actor_optimizer_steps"] = 0
    with pytest.raises(ValueError, match="delayed actor/temperature counters"):
        restored.load(invalid)
    assert _tree_equal(restored.checkpoint_state(), before)

    invalid = deepcopy(saved)
    invalid["outer_generator"] = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="outer generator state"):
        restored.load(invalid)
    assert _tree_equal(restored.checkpoint_state(), before)

    invalid = deepcopy(saved)
    critic_state = next(
        iter(invalid["xqc_workspace"]["critic_optimizer"]["state"].values())
    )
    critic_state["step"].zero_()
    with pytest.raises(ValueError, match="critic optimizer step"):
        restored.load(invalid)
    assert _tree_equal(restored.checkpoint_state(), before)

    invalid = deepcopy(saved)
    critic_state = next(
        iter(invalid["xqc_workspace"]["critic_optimizer"]["state"].values())
    )
    critic_state["exp_avg"].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="critic optimizer state must be finite"):
        restored.load(invalid)
    assert _tree_equal(restored.checkpoint_state(), before)

    invalid = deepcopy(saved)
    module_key = next(
        key
        for key, value in invalid["module"].items()
        if value.is_floating_point() and value.numel()
    )
    invalid["module"][module_key].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="module tensor.*must be finite"):
        restored.load(invalid)
    assert _tree_equal(restored.checkpoint_state(), before)
    wrapper.env.close()
    restored_wrapper.env.close()


def test_inner_reward_normalization_mode_is_checkpoint_semantic():
    frozen_wrapper = _tiny_model()
    adaptive_wrapper = _tiny_model(
        inner_reward_normalization="action_local_imagined"
    )
    frozen = frozen_wrapper.agent
    adaptive = adaptive_wrapper.agent

    assert frozen.semantic_signature()["reward_normalization"] == (
        "real_discounted_return_only"
    )
    assert adaptive.semantic_signature()["reward_normalization"] == (
        "real_discounted_return_plus_fresh_action_local_imagined_returns"
    )
    with pytest.raises(ValueError, match="semantics"):
        frozen.load(deepcopy(adaptive.checkpoint_state()))

    frozen_wrapper.env.close()
    adaptive_wrapper.env.close()


def test_inner_action_is_logically_fresh_and_does_not_mutate_outer_prior():
    wrapper = _tiny_model()
    agent = wrapper.agent
    outer_before = deepcopy(agent.xqc_controller.state_dict())
    obs, _ = wrapper.env.reset(seed=9)

    action, _ = wrapper.predict(obs, deterministic=False)

    assert action.shape == wrapper.env.action_space.shape
    assert _tree_equal(agent.xqc_controller.state_dict(), outer_before)
    assert agent.last_inner_metrics["inner_updates"] == 1.0
    assert agent.last_inner_metrics["inner_actor_optimizer_steps"] == 1.0
    assert math.isfinite(agent.last_inner_metrics["inner_critic_loss"])
    wrapper.env.close()


def test_short_shared_training_loop_collects_raw_replay_and_runs_both_learners():
    wrapper = _tiny_model(inner_diagnostics_every=1)

    wrapper.learn(total_timesteps=10)

    assert wrapper._global_step == 10
    assert wrapper._num_updates == wrapper.agent.num_updates == 6
    assert wrapper.agent.xqc_workspace.update_step == 6
    # The first five interactions are warm-up; every later action performs one
    # logically fresh inner solve under this tiny configuration.
    assert wrapper.agent.inner_engine.action_index == 5
    assert wrapper.agent.reward_normalizer.count == 10
    assert wrapper.buffer.total_transitions == 10
    wrapper.env.close()
