"""Frozen AMBI adaptation of the native TD-MPC2 planning protocol."""

from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.ambi_mppi import FrozenAMBIMPPIController, resolve_mppi_settings
from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.mppi import MPPIModelCallbacks, mppi_plan
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model


SMALL = {"horizon": 2, "num_samples": 8, "num_elites": 3,
         "num_pi_trajs": 2, "iterations": 2}


def _prior(**overrides):
    model = _tiny_model(inner_operator="none", inner_rounds=None,
                        inner_rollouts_per_round=None, inner_updates_per_round=None,
                        inner_temperature_mode="inherit_outer", **overrides)
    model.agent.model.eval()
    return model


@pytest.fixture
def prior():
    model = _prior()
    yield model
    model.env.close()


@pytest.mark.parametrize("seed", [1, 7, 49])
def test_native_action_matches_direct_upstream_equations(seed):
    """Verify weighted elite sampling, including its private RNG consumption."""
    expected_generator = torch.Generator().manual_seed(seed)
    mean, std = torch.zeros(2, 1), torch.full((2, 1), 2.0)
    for _ in range(2):
        actions = (mean[:, None] + std[:, None] *
                   torch.randn(2, 8, 1, generator=expected_generator)).clamp(-1, 1)
        values = actions[0] + 0.99 * actions[1]
        indices = values.squeeze(1).topk(3).indices
        elite_values, elite_actions = values[indices], actions[:, indices]
        score = (0.5 * (elite_values - elite_values.max(0).values)).exp()
        score = score / score.sum(0)
        mean = (score[None] * elite_actions).sum(1) / (score.sum(0) + 1e-9)
        std = ((score[None] * (elite_actions - mean[:, None]).square()).sum(1) /
               (score.sum(0) + 1e-9)).sqrt().clamp(0.05, 2)
    gumbels = -torch.empty_like(score.squeeze(1)).exponential_(
        generator=expected_generator).log()
    selected = (score.squeeze(1).log() + gumbels).softmax(0).argmax(-1)

    callbacks = MPPIModelCallbacks(
        action_dim=1, dynamics=lambda z, a: z, reward=lambda z, a: a,
        policy=lambda z, *, generator: z.new_zeros((z.shape[0], 1)),
        terminal_q=lambda z, a, *, reduction, generator: z.new_zeros((z.shape[0], 1)),
    )
    generator = torch.Generator().manual_seed(seed)
    result = mppi_plan(torch.zeros(1, 2), callbacks=callbacks, horizon=2,
                       iterations=2, num_samples=8, num_elites=3, num_pi_trajs=0,
                       temperature=0.5, min_std=0.05, max_std=2, discount=0.99,
                       q_reduction="mean_pair", generator=generator, eval_mode=True,
                       action_selection="tdmpc2")
    torch.testing.assert_close(result.action, elite_actions[0, selected], rtol=0, atol=0)
    torch.testing.assert_close(result.next_mean, mean, rtol=0, atol=0)
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_humanoid_defaults_keep_authored_and_effective_iterations_distinct():
    assert resolve_mppi_settings(None, action_dim=21) == {
        "horizon": 3, "iterations": 6, "effective_iterations": 8,
        "num_samples": 512, "num_elites": 64, "num_pi_trajs": 24,
        "min_std": 0.05, "max_std": 2.0, "temperature": 0.5,
    }
    assert resolve_mppi_settings({"iterations": 2}, action_dim=20)["effective_iterations"] == 4
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


@pytest.mark.parametrize(("representation", "num_q"), [("scalar", 2), ("distributional", 5)])
def test_frozen_planning_preserves_outer_inner_modes_and_every_global_rng(representation, num_q):
    model = _prior(q_representation=representation, num_q=num_q, dropout=0.2)
    try:
        agent = model.agent
        planner = FrozenAMBIMPPIController(agent, SMALL)
        observation, _ = model.env.reset(seed=17)
        outer_before = deepcopy(agent.checkpoint_state())
        inner_before = deepcopy(agent.inner_engine.training_state_dict())
        modes_before = [module.training for module in agent.model.modules()]
        python_before, numpy_before = random.getstate(), np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        def episode(seed):
            planner.reset(seed)
            assert planner.previous_mean is None and planner.action_index == 0
            actions = torch.stack([planner.act(observation) for _ in range(3)])
            assert planner.previous_mean is not None and planner.action_index == 3
            return actions

        first = episode(101)
        episode(500)
        repeated = episode(101)
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        assert first.device.type == "cpu" and first.shape == (3, 1)
        assert torch.isfinite(first).all() and (first.abs() <= 1).all()
        _assert_tree_equal(agent.checkpoint_state(), outer_before)
        _assert_tree_equal(agent.inner_engine.training_state_dict(), inner_before)
        assert [module.training for module in agent.model.modules()] == modes_before
        assert random.getstate() == python_before
        assert np.random.get_state()[0] == numpy_before[0]
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        assert np.random.get_state()[2:] == numpy_before[2:]
        assert torch.equal(torch.get_rng_state(), torch_before)
        metrics = agent.last_inner_metrics
        assert metrics["inner_model_steps"] == 2 * (2 - 1) + 2 * 8 * 2 == 34
        assert metrics["planner_policy_model_steps"] == 2
        assert metrics["planner_candidate_model_steps"] == 32
        assert metrics["inner_mppi_iterations"] == 2
        assert metrics["inner_policy_evaluations"] == 2 * 2 + 2 * 8
        assert metrics["inner_q_evaluations"] == 2 * 8
        for key in ("inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                    "inner_temperature_optimizer_steps", "inner_critic_target_updates"):
            assert metrics[key] == 0
        assert agent.last_inner_rollout_lengths == []
        assert all(np.isfinite(value) for value in metrics.values())
    finally:
        model.env.close()


def test_terminal_q_uses_online_pair_mean_sampled_prior_and_raw_reward_without_entropy(prior, monkeypatch):
    agent = prior.agent
    planner = FrozenAMBIMPPIController(agent, SMALL)
    q_calls, policy_calls = [], []

    def reward_logits(joint):
        logits = joint.new_full((joint.shape[0], agent.cfg.num_bins), -100)
        logits[:, -2] = 100
        return logits

    def sampled_policy(z, task=None, *, generator=None, deterministic=False, **kwargs):
        assert not deterministic
        assert generator is planner.generator
        policy_calls.append(z.shape[0])
        return z.new_full((z.shape[0], agent.cfg.action_dim), 0.375)

    def q(z, action, task=None, *, reduction=None, target=False, generator=None, **kwargs):
        assert reduction == "mean_pair" and target is False
        assert generator is planner.generator
        torch.testing.assert_close(action, torch.full_like(action, 0.375))
        q_calls.append(z.shape[0])
        return z.new_full((z.shape[0], 1), 3)

    monkeypatch.setattr(agent.model, "reward_from_joint", reward_logits)
    monkeypatch.setattr(agent.model, "pi_action", sampled_policy)
    monkeypatch.setattr(agent.model, "pi", lambda *a, **k: pytest.fail("No entropy/log-probability evaluation"))
    monkeypatch.setattr(agent.model, "Q", q)
    raw_reward = float(td_math.two_hot_inv(reward_logits(torch.zeros(1, 1)), agent.cfg).item())
    observation, _ = prior.env.reset(seed=13)
    planner.act(observation)
    expected = raw_reward * (1 + agent.discount) + agent.discount ** 2 * 3
    assert agent.last_inner_metrics["planner_value_mean"] == pytest.approx(expected, rel=1e-6)
    assert policy_calls == [2, 2, 8, 8]
    assert q_calls == [8, 8]
    assert planner.protocol["terminal_value_source"] == "online_ambi_q_mean_pair"
    assert planner.protocol["terminal_value_semantics"] == "learned_soft_q_tail_without_entropy_correction"
    assert planner.protocol["reward_units"] == "raw_environment_reward"
    assert planner.protocol["terminal_action"] == "frozen_prior_sample"


def test_controller_guards_training_submodules_and_observation_shape(prior):
    planner = FrozenAMBIMPPIController(prior.agent, SMALL)
    with pytest.raises(ValueError, match="shape"):
        planner.act(torch.zeros(2))
    child = next(module for module in prior.agent.model.modules() if isinstance(module, torch.nn.Linear))
    child.train()
    with pytest.raises(ValueError, match="eval mode"):
        FrozenAMBIMPPIController(prior.agent, SMALL)
    with pytest.raises(RuntimeError, match="training-mode"):
        planner.act(torch.zeros(3))
    assert child.training  # Reject rather than silently change caller modes.
    prior.agent.model.eval()
    prior.agent.cfg.inner_operator = "sac"
    with pytest.raises(ValueError, match="frozen prior"):
        FrozenAMBIMPPIController(prior.agent, SMALL)


@pytest.mark.parametrize("seed", [True, 1.5, "101"])
def test_controller_rejects_invalid_episode_seeds(prior, seed):
    with pytest.raises(ValueError, match="integer"):
        FrozenAMBIMPPIController(prior.agent, SMALL).reset(seed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_planning_preserves_device_rng_and_repeatability():
    model = _prior(device="cuda", dropout=0.2)
    try:
        planner = FrozenAMBIMPPIController(model.agent, SMALL)
        observation, _ = model.env.reset(seed=17)
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        planner.reset(101)
        first = torch.stack([planner.act(observation) for _ in range(3)])
        planner.reset(101)
        repeated = torch.stack([planner.act(observation) for _ in range(3)])
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        assert first.device.type == "cpu"
        assert planner.previous_mean.device.type == "cuda"
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    finally:
        model.env.close()


@pytest.mark.parametrize("case", [
    "reward_qscale", "entropy_qscale", "entropy_autotemp", "reward_autotemp",
])
def test_four_saved_bank_objectives_keep_raw_q_and_checkpoint_state(tmp_path, case):
    """Reload nontrivial saved scalars, then prove search cannot use or adapt them."""
    from tests.test_td_ambi_study_execution import _agent

    source = _agent("prior", case)
    with torch.no_grad():
        if source.actor_loss_scale_enabled:
            source.actor_loss_scale.fill_(17.25)
        if source.log_ent_coef is not None:
            source.log_ent_coef.fill_(np.log(0.004))
    checkpoint = tmp_path / "bank.pt"
    source.save(checkpoint)
    restored = _agent("prior", case)
    restored.load(checkpoint)
    restored.model.eval()
    _assert_tree_equal(restored.checkpoint_state(), source.checkpoint_state())
    before = deepcopy(restored.checkpoint_state())
    inner_before = deepcopy(restored.inner_engine.training_state_dict())
    planner = FrozenAMBIMPPIController(restored, SMALL)
    observation = np.linspace(-0.2, 0.2, 67, dtype=np.float32)
    planner.reset(42)
    expected = torch.stack([planner.act(observation) for _ in range(2)])
    expected_mean = planner.previous_mean.clone()
    _assert_tree_equal(restored.checkpoint_state(), before)
    _assert_tree_equal(restored.inner_engine.training_state_dict(), inner_before)
    assert restored.last_inner_metrics["inner_temperature_optimizer_steps"] == 0
    assert restored.last_inner_metrics["inner_model_steps"] == 2 + 4 * 8 * 2
    soft_q = case.startswith("entropy_")
    assert planner.protocol["source_critic_target"] == (
        "entropy_augmented" if soft_q else "reward_only"
    )
    assert planner.protocol["terminal_value_units"] == (
        "raw_reward_soft_q" if soft_q else "raw_reward_q"
    )
    assert planner.protocol["terminal_value_semantics"] == (
        "learned_soft_q_tail_without_entropy_correction" if soft_q
        else "learned_reward_only_q_tail"
    )

    # Alter scalar state only, preserving every learned network. Raw-Q MPPI
    # actions and identities must remain identical even for soft-Q checkpoints.
    protocol_before = planner.protocol
    with torch.no_grad():
        if restored.actor_loss_scale_enabled:
            restored.actor_loss_scale.fill_(125.0)
        if restored.log_ent_coef is not None:
            restored.log_ent_coef.fill_(np.log(0.5))
        else:
            restored.fixed_ent_coef.fill_(0.5)
    changed = deepcopy(restored.checkpoint_state())
    planner.reset(42)
    actual = torch.stack([planner.act(observation) for _ in range(2)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(planner.previous_mean, expected_mean, rtol=0, atol=0)
    assert planner.protocol == protocol_before
    _assert_tree_equal(restored.checkpoint_state(), changed)


def test_unknown_native_selection_is_rejected():
    with pytest.raises(ValueError, match="action_selection"):
        mppi_plan(None, horizon=2, iterations=1, num_samples=8, num_elites=2,
                  num_pi_trajs=0, temperature=0.5, min_std=0.05, max_std=2,
                  discount=0.99, q_reduction="mean_pair", generator=None,
                  action_selection="unsupported")
