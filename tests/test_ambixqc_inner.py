from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from RL.tdmpc2_core.inner_xqc import InnerXQCEngine
from RL.tdmpc2_core.xqc_controller import LatentXQCConfig, LatentXQCController


def _assert_tree_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    else:
        assert left == right


class _FakeActor(nn.Module):
    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.log_std = nn.Parameter(torch.tensor(0.0))
        self.bn_modes = []

    def distribution(self, z, bn_mode="running"):
        self.bn_modes.append(bn_mode)
        return (
            self.bias.expand(z.shape[0], 1),
            self.log_std.expand(z.shape[0], 1),
        )

    def sample(
        self,
        z,
        *,
        deterministic=False,
        bn_mode="running",
        noise=None,
        **_,
    ):
        pre_tanh, _ = self.distribution(z, bn_mode=bn_mode)
        # Keep the test action deterministic while requiring the engine to
        # supply correctly shaped private noise for every stochastic call.
        if not deterministic:
            assert noise is not None and noise.shape == pre_tanh.shape
            pre_tanh = pre_tanh + noise * 0.0
        return torch.tanh(pre_tanh), torch.zeros(z.shape[0], device=z.device)


class _FakeLocalController:
    def __init__(self, actor):
        self.actor = actor
        self.log_temperature = nn.Parameter(torch.tensor(-2.0))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def sample_action(self, z, *, deterministic=False, noise=None):
        return self.actor.sample(
            z,
            deterministic=deterministic,
            bn_mode="running",
            noise=noise,
        )


class _FakeWorkspace:
    def __init__(self, outer, rules):
        self._records = outer.records
        self.controller = _FakeLocalController(deepcopy(outer.actor))
        self.controller.config = rules
        self.update_step = 0
        self.actor_optimizer_steps = 0
        self.temperature_optimizer_steps = 0

    def reset_from_(self, outer):
        outer.reset_calls += 1
        self.controller.actor.load_state_dict(outer.actor.state_dict())
        self.controller.log_temperature.data.copy_(outer.log_temperature.data)
        self.update_step = 0
        self.actor_optimizer_steps = 0
        self.temperature_optimizer_steps = 0

    def update(
        self,
        batch,
        *,
        next_noise,
        actor_noise,
        reward_scale,
    ):
        assert next_noise.shape == actor_noise.shape == batch.actions.shape
        self._records.append(
            {
                "latents": batch.latents.detach().clone(),
                "actions": batch.actions.detach().clone(),
                "rewards": batch.rewards.detach().clone(),
                "next_latents": batch.next_latents.detach().clone(),
                "bootstrap_mask": batch.bootstrap_mask.detach().clone(),
                "discount": batch.discount.detach().clone(),
                "reward_scale": float(reward_scale),
            }
        )
        accepted = self.update_step % self.controller.config.policy_delay == 0
        if accepted:
            with torch.no_grad():
                self.controller.actor.bias.add_(0.1)
            self.actor_optimizer_steps += 1
            self.temperature_optimizer_steps += 1
        self.update_step += 1
        return {
            "critic_loss": float(self.update_step),
            "actor_update_accepted": float(accepted),
        }


class _FakeOuterController:
    def __init__(self):
        self.actor = _FakeActor()
        self.log_temperature = nn.Parameter(torch.tensor(-2.0))
        self.records = []
        self.clone_calls = 0
        self.reset_calls = 0

    def clone_for_inner(self, **kwargs):
        self.clone_calls += 1
        assert kwargs == {
            "actor_lr": 5e-5,
            "critic_lr": 7e-5,
            "transition_steps": 4,
        }
        return _FakeWorkspace(
            self,
            SimpleNamespace(policy_delay=3, target_update_interval=1),
        )


class _FakeTOLD(nn.Module):
    def __init__(self, *, terminate=False):
        super().__init__()
        self.terminate = terminate

    @staticmethod
    def joint_input(z, action):
        return torch.cat((z, action), dim=-1)

    @staticmethod
    def reward_from_joint(joint):
        return torch.ones(joint.shape[0], 1, device=joint.device)

    @staticmethod
    def next_from_joint(joint):
        return joint[:, :2] + 0.25

    def termination(self, z):
        value = 1.0 if self.terminate else 0.0
        return torch.full((z.shape[0], 1), value, device=z.device)


def _agent(*, episodic=False, terminate=False):
    cfg = SimpleNamespace(
        seed=11,
        latent_dim=2,
        action_dim=1,
        episodic=episodic,
        num_bins=0,
        vmin=-10.0,
        vmax=10.0,
        inner_rounds=2,
        inner_rollouts_per_round=2,
        inner_rollout_horizon=2,
        inner_updates_per_round=2,
        inner_batch_size=2,
        inner_replay_capacity=8,
        inner_replay_sampling="with_replacement",
        inner_actor_lr=5e-5,
        inner_critic_lr=7e-5,
        xqc_adam_eps=1e-8,
        xqc_policy_delay=3,
        xqc_tau=0.005,
        xqc_target_update_interval=1,
        inner_termination_threshold=0.5,
        inner_model_step_budget=8,
    )
    controller = _FakeOuterController()
    agent = SimpleNamespace(
        cfg=cfg,
        device=torch.device("cpu"),
        model=_FakeTOLD(terminate=terminate),
        xqc_controller=controller,
        discount=0.9,
        reward_normalizer=SimpleNamespace(scale=2.5),
    )
    return agent, controller


def test_action_local_xqc_uses_raw_imagination_and_frozen_real_reward_scale():
    agent, outer = _agent()
    engine = InnerXQCEngine(agent)
    root = torch.zeros(1, 2)
    outer_state = deepcopy(outer.actor.state_dict())
    global_rng = torch.random.get_rng_state().clone()

    action, metrics, lengths = engine.act(root, eval_mode=False)

    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert torch.allclose(action, torch.tanh(torch.tensor([0.2])))
    assert lengths == [2, 2, 2, 2]
    assert metrics["inner_model_steps"] == 8
    assert metrics["inner_update_slots"] == 4
    assert metrics["inner_actor_optimizer_steps"] == 2
    assert metrics["inner_temperature_optimizer_steps"] == 2
    assert metrics["inner_final_outer_policy_kl"] == pytest.approx(0.02)
    assert metrics["inner_reward_scale"] == pytest.approx(2.5)
    assert metrics["inner_buffer_size"] == 8
    assert all(record["reward_scale"] == 2.5 for record in outer.records)
    assert all(torch.equal(record["rewards"], torch.ones(2, 1)) for record in outer.records)
    assert all(
        torch.equal(record["bootstrap_mask"], torch.ones(2, 1))
        for record in outer.records
    )
    assert all(float(record["discount"]) == pytest.approx(0.9) for record in outer.records)
    assert outer.actor.state_dict().keys() == outer_state.keys()
    assert all(
        torch.equal(outer.actor.state_dict()[key], value)
        for key, value in outer_state.items()
    )
    assert engine.state.workspace is None
    assert engine.state.replay is None
    assert set(engine._workspace_pool.controller.actor.bn_modes) == {"running"}
    assert torch.equal(engine._replay_pool.reward[:8], torch.ones(8, 1))


def test_final_policy_kl_uses_closed_form_direction_and_running_bn_without_rng():
    agent, outer = _agent()
    engine = InnerXQCEngine(agent)
    root = torch.zeros(1, 2)
    with engine.rng.fork("initialization"):
        engine._prepare_action()

    inner_actor = engine.state.workspace.controller.actor
    with torch.no_grad():
        outer.actor.bias.fill_(0.0)
        outer.actor.log_std.fill_(0.0)
        inner_actor.bias.fill_(1.0)
        inner_actor.log_std.fill_(math.log(2.0))
    outer_before = deepcopy(outer.actor.state_dict())
    inner_before = deepcopy(inner_actor.state_dict())
    global_rng = torch.random.get_rng_state().clone()

    metrics = engine._final_policy_diagnostics(root)

    assert metrics["inner_final_outer_policy_kl"] == pytest.approx(
        2.0 - math.log(2.0)
    )
    assert outer.actor.bn_modes[-1] == "running"
    assert inner_actor.bn_modes[-1] == "running"
    assert engine.state.policy_evaluations == 2
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    _assert_tree_equal(outer.actor.state_dict(), outer_before)
    _assert_tree_equal(inner_actor.state_dict(), inner_before)
    engine._release_action()


def test_unsampled_action_omits_final_policy_kl_and_diagnostic_work():
    agent, _ = _agent()
    engine = InnerXQCEngine(agent)

    _, metrics, _ = engine.act(torch.zeros(1, 2), collect_diagnostics=False)
    metrics = engine.finalize_timing_metrics(metrics)

    assert "inner_final_outer_policy_kl" not in metrics
    assert "inner_diagnostic_seconds" not in metrics
    assert metrics["inner_policy_evaluations"] == 25.0


def test_second_action_reuses_allocations_but_is_logically_fresh():
    agent, outer = _agent()
    engine = InnerXQCEngine(agent)
    root = torch.zeros(1, 2)

    first_action, first_metrics, _ = engine.act(root)
    workspace_id = id(engine._workspace_pool)
    second_action, second_metrics, _ = engine.act(root)

    assert id(engine._workspace_pool) == workspace_id
    assert outer.clone_calls == 1
    assert outer.reset_calls == 1
    assert torch.equal(first_action, second_action)
    assert first_metrics["inner_actor_optimizer_steps"] == 2
    assert second_metrics["inner_actor_optimizer_steps"] == 2


def test_true_imagined_termination_masks_bootstrap_but_horizon_does_not():
    agent, outer = _agent(episodic=True, terminate=True)
    engine = InnerXQCEngine(agent)

    _, metrics, lengths = engine.act(torch.zeros(1, 2))

    assert lengths.tolist() == [1, 1, 1, 1]
    assert metrics["inner_model_steps"] == 4
    assert metrics["inner_termination_rate"] == pytest.approx(1.0)
    assert all(
        torch.equal(record["bootstrap_mask"], torch.zeros(2, 1))
        for record in outer.records
    )


def test_inner_boundary_state_round_trip_restores_only_persistent_rng_and_indices():
    agent, _ = _agent()
    source = InnerXQCEngine(agent)
    source.act(torch.zeros(1, 2))
    source.reset_episode()
    state = deepcopy(source.training_state_dict())

    restored = InnerXQCEngine(agent)
    restored.load_training_state_dict(state)

    assert restored.action_index == 1
    assert restored.episode_index == 1
    assert restored.state.workspace is None
    assert restored._workspace_pool is None
    _assert_tree_equal(restored.training_state_dict(), state)


def test_bad_inner_boundary_state_fails_before_mutation():
    agent, _ = _agent()
    engine = InnerXQCEngine(agent)
    engine.act(torch.zeros(1, 2))
    before = deepcopy(engine.training_state_dict())
    invalid = deepcopy(before)
    invalid["action_index"] = -1

    with pytest.raises(ValueError, match="action_index"):
        engine.load_training_state_dict(invalid)

    _assert_tree_equal(engine.training_state_dict(), before)


def test_real_inner_xqc_update_is_finite_isolated_and_keeps_target_bn_frozen():
    agent, _ = _agent()
    torch.manual_seed(19)
    controller = LatentXQCController(
        latent_dim=2,
        action_dim=1,
        config=LatentXQCConfig(
            actor_net_arch=(8,),
            critic_net_arch=(8,),
            num_atoms=11,
            target_entropy=-0.5,
            optimizer_backend="single_tensor",
        ),
    )
    agent.xqc_controller = controller
    outer_before = deepcopy(controller.state_dict())
    target_buffer_start = {
        name: value.detach().clone()
        for name, value in controller.critic.named_buffers()
    }
    engine = InnerXQCEngine(agent)
    global_rng = torch.random.get_rng_state().clone()

    action, metrics, _ = engine.act(torch.zeros(1, 2))

    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert torch.isfinite(action).all()
    assert all(
        torch.isfinite(torch.as_tensor(value)).all()
        for value in metrics.values()
    )
    _assert_tree_equal(controller.state_dict(), outer_before)
    inner = engine._workspace_pool.controller
    assert engine._workspace_pool.update_step == 4
    assert engine._workspace_pool.actor_optimizer_steps == 2
    assert engine._workspace_pool.temperature_optimizer_steps == 2
    for name, value in inner.critic_target.named_buffers():
        assert torch.equal(value, target_buffer_start[name])
    assert any(
        not torch.equal(value, target_buffer_start[name])
        for name, value in inner.critic.named_buffers()
        if name.endswith(("running_mean", "running_var"))
    )


def test_failed_update_releases_all_logically_live_action_state():
    agent, _ = _agent()
    engine = InnerXQCEngine(agent)
    engine.act(torch.zeros(1, 2))

    def fail_update(*args, **kwargs):
        raise RuntimeError("synthetic update failure")

    engine._workspace_pool.update = fail_update
    with pytest.raises(RuntimeError, match="synthetic update failure"):
        engine.act(torch.zeros(1, 2))

    assert engine.state.workspace is None
    assert engine.state.replay is None
    assert engine._workspace_pool is not None
    assert engine._replay_pool is not None
