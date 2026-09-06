from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.mppi import MPPIModelCallbacks, mppi_plan
from RL.tdmpc2_core.xqc_mppi import FrozenXQCMPPIController, resolve_mppi_settings
from RL.tdmpc2_core.common import math as td_math
from test_ambixqc_core import _tiny_model, _batch, _tree_equal


def _linear_callbacks():
    return MPPIModelCallbacks(
        action_dim=1, dynamics=lambda z, a: z,
        reward=lambda z, a: a,
        policy=lambda z, *, generator: z.new_zeros((z.shape[0], 1)),
        terminal_q=lambda z, a, *, reduction, generator: z.new_zeros((z.shape[0], 1)),
    )


def _native_reference(seed):
    """Direct upstream equations for a two-step linear-reward callback model."""
    generator = torch.Generator().manual_seed(seed)
    mean, std = torch.zeros(2, 1), torch.full((2, 1), 2.0)
    for _ in range(2):
        actions = (mean[:, None] + std[:, None] * torch.randn(2, 8, 1, generator=generator)).clamp(-1, 1)
        values = actions[0] + 0.99 * actions[1]
        indices = values.squeeze(1).topk(3).indices
        elite_values, elite_actions = values[indices], actions[:, indices]
        score = (0.5 * (elite_values - elite_values.max(0).values)).exp()
        score = score / score.sum(0)
        mean = (score[None] * elite_actions).sum(1) / (score.sum(0) + 1e-9)
        std = ((score[None] * (elite_actions - mean[:, None]).square()).sum(1) / (score.sum(0) + 1e-9)).sqrt().clamp(0.05, 2)
    gumbels = -torch.empty_like(score.squeeze(1)).exponential_(generator=generator).log()
    selected = (score.squeeze(1).log() + gumbels).softmax(0).argmax(-1)
    return elite_actions[0, selected], mean, generator.get_state()


@pytest.mark.parametrize("seed", [1, 7, 49])
def test_native_evaluation_matches_upstream_gumbel_and_consumes_no_execution_noise(seed):
    expected_action, expected_mean, expected_rng = _native_reference(seed)
    generator = torch.Generator().manual_seed(seed)
    outer_rng = torch.get_rng_state().clone()
    result = mppi_plan(
        torch.zeros(1, 2), callbacks=_linear_callbacks(), horizon=2,
        iterations=2, num_samples=8, num_elites=3, num_pi_trajs=0,
        temperature=0.5, min_std=0.05, max_std=2, discount=0.99,
        q_reduction="mean_all", generator=generator, eval_mode=True,
        action_selection="tdmpc2",
    )
    torch.testing.assert_close(result.action, expected_action, rtol=0, atol=0)
    torch.testing.assert_close(result.next_mean, expected_mean, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), expected_rng)
    assert torch.equal(torch.get_rng_state(), outer_rng)


def test_native_evaluation_preserves_nan_to_zero_value_guard():
    callbacks = _linear_callbacks()
    callbacks = MPPIModelCallbacks(
        action_dim=1, dynamics=callbacks.dynamics,
        reward=lambda z, a: torch.full_like(a, float("nan")),
        policy=callbacks.policy, terminal_q=callbacks.terminal_q,
    )
    result = mppi_plan(
        torch.zeros(1, 2), callbacks=callbacks, horizon=2, iterations=2,
        num_samples=8, num_elites=3, num_pi_trajs=0, temperature=0.5,
        min_std=0.05, max_std=2, discount=0.99, q_reduction="mean_all",
        generator=torch.Generator().manual_seed(5), eval_mode=True,
        action_selection="tdmpc2",
    )
    assert torch.isfinite(result.action).all()
    assert result.metrics["planner_value_mean"] == 0


def test_humanoid_defaults_keep_authored_and_effective_iterations_distinct():
    settings = resolve_mppi_settings(None, action_dim=21)
    assert settings == {
        "horizon": 3, "iterations": 6, "effective_iterations": 8,
        "num_samples": 512, "num_elites": 64, "num_pi_trajs": 24,
        "min_std": 0.05, "max_std": 2.0, "temperature": 0.5,
    }
    assert resolve_mppi_settings({"iterations": 2}, action_dim=21)["effective_iterations"] == 4
    assert resolve_mppi_settings(None, action_dim=19)["effective_iterations"] == 6


@pytest.mark.parametrize("settings", [
    {"iterations": True}, {"iterations": 0}, {"horizon": 1.2},
    {"num_samples": 2, "num_elites": 3}, {"num_pi_trajs": 513},
    {"temperature": float("nan")}, {"min_std": 4}, {"max_std": 0},
    {"effective_iterations": 8}, {"q_reduction": "min_all"},
])
def test_settings_reject_invalid_or_unsupported_semantics(settings):
    with pytest.raises(ValueError):
        resolve_mppi_settings(settings, action_dim=21)


@pytest.fixture
def frozen_model():
    model = _tiny_model(inner_operator="none")
    model.agent.observe_reward(2.0, False, False)
    model.agent._update(*_batch(model.agent))
    model.load(deepcopy(model.agent.checkpoint_state()), frozen_evaluation=True)
    yield model
    model.env.close()


SMALL = {"horizon": 2, "num_samples": 8, "num_elites": 3, "num_pi_trajs": 2, "iterations": 2}


def test_adapter_preserves_all_outer_state_and_resets_episode_warmstarts(frozen_model):
    model = frozen_model
    agent = model.agent
    planner = FrozenXQCMPPIController(agent, SMALL)
    observation, _ = model.env.reset(seed=17)
    before = agent.frozen_outer_state()
    outer_rng = torch.get_rng_state().clone()

    def episode(seed):
        planner.reset(seed)
        assert planner.previous_mean is None and planner.action_index == 0
        return torch.stack([planner.act(observation) for _ in range(3)])

    first = episode(101)
    assert planner.previous_mean is not None and planner.action_index == 3
    episode(500)  # Warmup/other episodes must not contaminate the seeded solve.
    repeated = episode(101)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert _tree_equal(before, agent.frozen_outer_state())
    assert torch.equal(outer_rng, torch.get_rng_state())
    metrics = agent.last_inner_metrics
    assert metrics["inner_model_steps"] == 2 * (2 - 1) + 2 * 8 * 2 == 34
    assert metrics["planner_policy_model_steps"] == 2
    assert metrics["planner_candidate_model_steps"] == 32
    assert metrics["inner_mppi_iterations"] == 2
    assert metrics["inner_policy_evaluations"] == 2 * 2 + 2 * 8
    assert metrics["inner_q_evaluations"] == 2 * 8
    assert metrics["planner_q_head_evaluations"] == 2 * 2 * 8
    assert metrics["inner_active"] == metrics["inner_algorithm_mppi"] == 1
    for key in ("inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                "inner_temperature_optimizer_steps", "inner_replay_draws"):
        assert metrics[key] == 0
    assert agent.inner_engine._workspace_pool is None
    assert all(torch.isfinite(torch.as_tensor(value)) for value in metrics.values())


def test_terminal_q_uses_running_online_twin_mean_and_frozen_scale(frozen_model, monkeypatch):
    agent = frozen_model.agent
    planner = FrozenXQCMPPIController(agent, SMALL)
    calls = []

    def values(z, action, *, bn_mode):
        calls.append(bn_mode)
        return torch.stack((z.new_full((z.shape[0],), 2), z.new_full((z.shape[0],), 4)))

    monkeypatch.setattr(agent.xqc_controller.critic, "values", values)
    monkeypatch.setattr(agent.xqc_controller.critic_target, "values", lambda *a, **k: pytest.fail("Must not use target Q"))
    z = torch.zeros(5, agent.cfg.latent_dim)
    result = planner._terminal_q(z, torch.zeros(5, 1), reduction="mean_all", generator=planner.generator)
    torch.testing.assert_close(result, torch.full((5, 1), 3 * agent.reward_normalizer.scale))
    assert calls == ["running"]
    assert planner.protocol["terminal_value_semantics"] == "learned_soft_q_tail_without_entropy_correction"
    assert planner.protocol["reward_units"] == "raw_environment_reward"
    assert planner.protocol["reward_scale"] == agent.reward_normalizer.scale


def test_planner_combines_raw_model_rewards_with_scaled_soft_q_tail(frozen_model, monkeypatch):
    agent = frozen_model.agent
    planner = FrozenXQCMPPIController(agent, SMALL)

    def reward_logits(joint):
        logits = joint.new_full((joint.shape[0], agent.cfg.num_bins), -100)
        logits[:, -2] = 100
        return logits

    monkeypatch.setattr(agent.model, "reward_from_joint", reward_logits)
    monkeypatch.setattr(agent.xqc_controller.critic, "values", lambda z, a, *, bn_mode:
                        torch.stack((z.new_full((z.shape[0],), 2), z.new_full((z.shape[0],), 4))))
    logits = reward_logits(torch.zeros(1, 1))
    raw_reward = float(td_math.two_hot_inv(logits, agent.cfg).item())
    observation, _ = frozen_model.env.reset(seed=13)
    planner.act(observation)
    expected = raw_reward * (1 + agent.discount) + agent.discount ** 2 * 3 * planner.reward_scale
    assert agent.last_inner_metrics["planner_value_mean"] == pytest.approx(expected, rel=1e-6)


def test_adapter_rejects_training_agent_and_invalid_reward_scale(monkeypatch):
    model = _tiny_model(inner_operator="none")
    try:
        with pytest.raises(ValueError, match="frozen-evaluation"):
            FrozenXQCMPPIController(model.agent)
        model.reset_for_evaluation(101)
        monkeypatch.setattr(type(model.agent.reward_normalizer), "scale", property(lambda self: float("nan")))
        with pytest.raises(ValueError, match="positive finite"):
            FrozenXQCMPPIController(model.agent)
    finally:
        model.env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_adapter_preserves_outer_and_global_rng_and_repeats_seeded_episodes():
    model = _tiny_model(
        device="cuda", inner_operator="none", xqc_optimizer_backend="auto"
    )
    try:
        agent = model.agent
        agent.observe_reward(2.0, False, False)
        agent._update(*(tensor.to(agent.device) for tensor in _batch(agent)))
        model.load(deepcopy(agent.checkpoint_state()), frozen_evaluation=True)
        planner = FrozenXQCMPPIController(agent, SMALL)
        assert planner.generator.device.type == "cuda"
        observation, _ = model.env.reset(seed=17)
        outer_before = agent.frozen_outer_state()
        cpu_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(agent.device).clone()
        planner.act(observation)  # An unscored warmup must be erased by reset.

        def episode(seed):
            planner.reset(seed)
            assert planner.previous_mean is None and planner.action_index == 0
            actions = torch.stack([planner.act(observation) for _ in range(3)])
            assert actions.device.type == "cpu"
            assert planner.previous_mean.device.type == "cuda"
            assert planner.action_index == 3
            assert agent.last_inner_metrics["inner_model_steps"] == 34
            return actions

        first = episode(101)
        episode(500)
        repeated = episode(101)
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        assert _tree_equal(outer_before, agent.frozen_outer_state())
        assert torch.equal(cpu_rng_before, torch.get_rng_state())
        assert torch.equal(cuda_rng_before, torch.cuda.get_rng_state(agent.device))
        for key in (
            "inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
            "inner_temperature_optimizer_steps", "inner_replay_draws",
        ):
            assert agent.last_inner_metrics[key] == 0
    finally:
        model.env.close()
