"""Evaluation sampling changes only the final action, using private policy RNG."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_component_model, _tiny_model


def _model(**overrides):
    options = dict(
        inner_rounds=2,
        inner_rollouts_per_round=2,
        inner_rollout_horizon=2,
        inner_critic_updates_per_round=2,
        inner_actor_updates_per_round=1,
        inner_replay_capacity=8,
        inner_finite_horizon=True,
        inner_eval_execution_action="policy_sample",
        log_std_mapping="direct_clamp",
    )
    options.update(overrides)
    return _tiny_component_model(**options)


def _capture_execution(monkeypatch, engine):
    """Observe the final learner before its action-local workspace expires."""
    captured = []
    original = engine._execute_policy

    def execute(root_z, policy, **kwargs):
        modes = [(module, module.training) for module in policy.modules()]
        try:
            policy.eval()
            bounds = (dict(
                log_std_mapping=engine.cfg.inner_log_std_mapping,
                log_std_min=engine.cfg.inner_log_std_min,
                log_std_max=engine.cfg.inner_log_std_max,
            ) if kwargs.get("inner_bounds", True) else engine._actor_options)
            with torch.no_grad():
                stats = _clone_tree(engine.model.policy_stats(root_z, policy=policy, **bounds))
        finally:
            for module, training in modes:
                module.training = training
        generator = torch.Generator(device=engine.device)
        generator.set_state(engine.rng.generator("execution").get_state())
        epsilon = torch.randn(stats["pre_tanh_mean"].shape, generator=generator,
                              device=root_z.device, dtype=root_z.dtype)
        adaptation = {}
        for name in ("actor", "critic", "critic_target", "actor_optim", "critic_optim",
                     "temperature_optim", "replay"):
            value = getattr(engine.state, name)
            adaptation[name] = _clone_tree(value.state_dict()) if value is not None else None
        adaptation["log_alpha"] = _clone_tree(engine.state.log_alpha)
        captured.append(dict(
            mean=stats["mean"][0],
            sample=torch.tanh(stats["pre_tanh_mean"] + stats["log_std"].exp() * epsilon)[0],
            execution_rng_after_sample=generator.get_state().clone(),
            adaptation=adaptation,
            eval_mode=kwargs["eval_mode"],
        ))
        return original(root_z, policy, **kwargs)

    monkeypatch.setattr(engine, "_execute_policy", execute)
    return captured


def test_evaluation_execution_defaults_to_mean():
    assert _build_cfg().inner_eval_execution_action == "mean"


@pytest.mark.parametrize("operator", ["none", "sac"])
@pytest.mark.parametrize("mode", ["mean", "policy_sample"])
def test_supported_evaluation_execution_configuration(operator, mode):
    cfg = _build_cfg(inner_operator=operator, inner_eval_execution_action=mode)
    assert cfg.inner_eval_execution_action == mode


@pytest.mark.parametrize("mode", [None, True, 2, "sample", "mean_plus_gaussian"])
def test_evaluation_execution_rejects_unknown_modes(mode):
    with pytest.raises(ValueError, match="inner_eval_execution_action"):
        _build_cfg(inner_eval_execution_action=mode)


@pytest.mark.parametrize("operator", ["td3", "mppi"])
def test_sampled_evaluation_rejects_unsupported_operators(operator):
    with pytest.raises(ValueError, match="inner_eval_execution_action"):
        _build_cfg(inner_operator=operator, inner_eval_execution_action="policy_sample")


@pytest.mark.parametrize("extra", [
    {"inner_explorer_mode": "frozen_random"},
    {"inner_explorer_mode": "shared_mixture"},
    {"inner_explorer_mode": "separate_critics"},
    {"inner_explorer_mode": "frozen_random", "inner_execution_policy_source": "explorer"},
])
def test_sampled_evaluation_rejects_explorer_execution(extra):
    with pytest.raises(ValueError, match="inner_eval_execution_action"):
        _build_cfg(inner_eval_execution_action="policy_sample", **extra)


@pytest.mark.parametrize("operator", ["mppi", "tdambi"])
def test_runtime_rejection_does_not_advance_action_state(operator):
    holder = _model()
    try:
        engine = holder.agent.inner_engine
        engine.reset_for_evaluation(919)
        holder.cfg.inner_operator = operator
        state_before = engine.state
        rng_before = _clone_tree(engine.rng.training_state_dict())
        with pytest.raises(ValueError, match="Sampled evaluation requires"):
            engine.act(torch.zeros(1, holder.cfg.latent_dim), t0=True, eval_mode=True)
        assert engine.state is state_before
        assert engine.state.actor is None and engine.state.replay is None
        assert engine.action_index == 0
        assert engine._active_trace is None
        _assert_tree_equal(engine.rng.training_state_dict(), rng_before)
    finally:
        holder.env.close()


@pytest.mark.parametrize("training_execution", [
    {"inner_execution_action": "mean"},
    {"inner_execution_action": "policy_sample", "inner_execution_std_scale": 0.0},
    {"inner_execution_action": "policy_sample", "inner_execution_std_scale": 0.2},
    {"inner_execution_action": "mean_plus_gaussian", "inner_execution_noise_std": 0.7},
])
def test_sampled_evaluation_uses_exact_final_gaussian_not_training_noise(monkeypatch, training_execution):
    holder = _model(**training_execution)
    try:
        agent, engine = holder.agent, holder.agent.inner_engine
        engine.reset_for_evaluation(919)
        captured = _capture_execution(monkeypatch, engine)
        outer_before = deepcopy(agent.checkpoint_state())
        global_before = torch.random.get_rng_state().clone()

        action = agent.act(torch.zeros(3), t0=True, eval_mode=True)

        final, = captured
        assert final["eval_mode"] is True
        torch.testing.assert_close(action, final["sample"], rtol=0, atol=0)
        assert not torch.equal(action, final["mean"])
        torch.testing.assert_close(engine.rng.generator("execution").get_state(),
                                   final["execution_rng_after_sample"], rtol=0, atol=0)
        metrics = agent.last_inner_metrics
        assert metrics["inner_eval_execution_sampled"] == 1
        assert metrics["inner_eval_execution_mean_action_l2"] == pytest.approx(
            float(torch.linalg.vector_norm(action - final["mean"])))
        assert metrics["inner_critic_optimizer_steps"] == 4
        assert metrics["inner_actor_optimizer_steps"] == 2
        _assert_tree_equal(agent.checkpoint_state(), outer_before)
        torch.testing.assert_close(torch.random.get_rng_state(), global_before, rtol=0, atol=0)
    finally:
        holder.env.close()


def test_mean_and_sampled_execution_have_identical_adaptation(monkeypatch):
    holder = _model()
    try:
        agent, engine = holder.agent, holder.agent.inner_engine
        captured = _capture_execution(monkeypatch, engine)
        actions, metrics = [], []
        for mode in ("mean", "policy_sample"):
            holder.cfg.inner_eval_execution_action = mode
            engine.reset_for_evaluation(919)
            outer_before = deepcopy(agent.checkpoint_state())
            actions.append(agent.act(torch.zeros(3), t0=True, eval_mode=True))
            metrics.append(_clone_tree(agent.last_inner_metrics))
            _assert_tree_equal(agent.checkpoint_state(), outer_before)

        assert len(captured) == 2
        _assert_tree_equal(captured[0]["adaptation"], captured[1]["adaptation"])
        torch.testing.assert_close(actions[0], captured[0]["mean"], rtol=0, atol=0)
        torch.testing.assert_close(actions[1], captured[1]["sample"], rtol=0, atol=0)
        assert not torch.equal(actions[0], actions[1])
        assert metrics[0]["inner_eval_execution_sampled"] == 0
        assert metrics[0]["inner_eval_execution_mean_action_l2"] == 0
        assert metrics[1]["inner_eval_execution_sampled"] == 1
    finally:
        holder.env.close()


def test_sampled_prior_execution_uses_same_seeded_gaussian(monkeypatch):
    holder = _tiny_model(inner_operator="none", inner_rounds=0,
                         inner_rollouts_per_round=0, inner_updates_per_round=0,
                         inner_eval_execution_action="policy_sample")
    try:
        agent, engine = holder.agent, holder.agent.inner_engine
        engine.reset_for_evaluation(919)
        captured = _capture_execution(monkeypatch, engine)
        action = agent.act(torch.zeros(3), t0=True, eval_mode=True)
        final, = captured
        torch.testing.assert_close(action, final["sample"], rtol=0, atol=0)
        assert agent.last_inner_metrics["inner_eval_execution_sampled"] == 1
        assert not torch.equal(action, final["mean"])
    finally:
        holder.env.close()


def test_training_execution_ignores_evaluation_override(monkeypatch):
    holder = _model(inner_execution_action="mean")
    try:
        agent, engine = holder.agent, holder.agent.inner_engine
        captured = _capture_execution(monkeypatch, engine)
        actions, streams = [], []
        for mode in ("mean", "policy_sample"):
            holder.cfg.inner_eval_execution_action = mode
            engine.reset_for_evaluation(919)
            actions.append(agent.act(torch.zeros(3), t0=True, eval_mode=False))
            streams.append(_clone_tree(engine.rng.training_state_dict()))
        for action, final in zip(actions, captured):
            assert final["eval_mode"] is False
            torch.testing.assert_close(action, final["mean"], rtol=0, atol=0)
        torch.testing.assert_close(actions[0], actions[1], rtol=0, atol=0)
        _assert_tree_equal(captured[0]["adaptation"], captured[1]["adaptation"])
        _assert_tree_equal(streams[0], streams[1])
    finally:
        holder.env.close()


def test_sampled_evaluation_blocks_explicit_prior_writeback():
    holder = _model(inner_actor_writeback_coef=0.5, inner_critic_writeback_coef=0.5)
    try:
        agent = holder.agent
        agent.inner_engine.reset_for_evaluation(919)
        outer_before = deepcopy(agent.checkpoint_state())

        agent.act(torch.zeros(3), t0=True, eval_mode=True, apply_inner_writeback=True)

        metrics = agent.last_inner_metrics
        assert metrics["inner_eval_execution_sampled"] == 1
        assert metrics["inner_actor_optimizer_steps"] == 2
        assert metrics["inner_critic_optimizer_steps"] == 4
        assert metrics["inner_actor_writeback_applied"] == 0
        assert metrics["inner_critic_writeback_applied"] == 0
        _assert_tree_equal(agent.checkpoint_state(), outer_before)
    finally:
        holder.env.close()


def test_sampled_action_sequence_is_seeded_and_probe_independent():
    holder = _model()
    try:
        agent, engine = holder.agent, holder.agent.inner_engine
        outer_before = deepcopy(agent.checkpoint_state())

        def sequence(seed, probes):
            engine.reset_for_evaluation(seed)
            global_before = torch.random.get_rng_state().clone()
            actions = []
            for decision in range(3):
                trace = (InnerActionTrace(probes=True, probe_mode="outer_tail",
                                          probe_seed=700 + decision, probe_rollouts=3,
                                          probe_horizon=2) if probes else None)
                actions.append(agent.act(torch.zeros(3), t0=decision == 0,
                                         eval_mode=True, trace=trace))
                if probes:
                    assert sum(event["phase"] == "probe" for event in trace.events) == 3
            torch.testing.assert_close(torch.random.get_rng_state(), global_before, rtol=0, atol=0)
            _assert_tree_equal(agent.checkpoint_state(), outer_before)
            return torch.stack(actions), _clone_tree(engine.rng.training_state_dict())

        plain, plain_rng = sequence(919, False)
        probed, probed_rng = sequence(919, True)
        repeated, repeated_rng = sequence(919, False)
        changed, _ = sequence(920, False)
        torch.testing.assert_close(probed, plain, rtol=0, atol=0)
        torch.testing.assert_close(repeated, plain, rtol=0, atol=0)
        _assert_tree_equal(probed_rng, plain_rng)
        _assert_tree_equal(repeated_rng, plain_rng)
        assert not torch.equal(changed, plain)
        assert not torch.equal(plain[0], plain[1])
    finally:
        holder.env.close()
