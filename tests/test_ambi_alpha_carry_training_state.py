"""Online alpha carry follows episode boundaries despite a changing backbone."""

from copy import deepcopy
import sys

import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from tests.resume_test_support import BoundaryEnv, _FakeWandb, _session
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _tiny_params
from tests.test_ambi_temperature_adaptation import _model
from utils.resume_training import RESUME_COMPLETE, RESUME_HANDOFF


def _training_model(scope, **overrides):
    return _model(**{
        "aux_return_mode": "sac",
        "inner_actor_source": "sac",
        "inner_critic_source": "aux_return",
        "inner_horizon_actor_source": "sac",
        "inner_horizon_critic_source": "aux_return",
        "inner_sac_critic_target": "reward_only",
        "inner_temperature_scope": scope,
        "inner_temperature_optimizer_scope": scope,
        "inner_temperature_initialization": "inherit_outer",
        "inner_target_entropy": "inherit_outer",
        "ent_coef": "auto_1.0",
        **overrides,
    })


@pytest.mark.parametrize("scope", ["episode", "run"])
@pytest.mark.parametrize("detach", [False, True])
@pytest.mark.parametrize("target", [-21.0, -10.5])
def test_online_training_carries_only_temperature_and_refreshes_current_priors(
    scope, detach, target, monkeypatch,
):
    model = _training_model(scope, target_entropy=target,
                            aux_return_detach_representation=detach)
    try:
        engine = model.agent.inner_engine
        prepare, record = engine._prepare_workspace, model._record_action_metrics
        previous = None
        observed = []

        def checked_prepare(*, t0):
            prepare(t0=t0)
            state = engine.state
            # Actual outer optimizer updates have occurred since the last act.
            if previous is not None:
                assert model.agent.outer_version > previous["outer_version"]
            _assert_tree_equal(state.actor.state_dict(), model.agent.model._pi.state_dict())
            _assert_tree_equal(state.critic.state_dict(), model.agent.model._aux_return_Qs.state_dict())
            assert state.replay.size == 0
            assert state.actor_lifetime_steps == state.critic_lifetime_steps == 0
            assert engine._resolved_inner_target_entropy() == model.agent.target_entropy == target
            should_reset = previous is None or (scope == "episode" and t0)
            if should_reset:
                torch.testing.assert_close(engine.alpha, model.agent.alpha.reshape(()), rtol=1e-6, atol=0)
                assert not state.temperature_optim.state
                assert state.temperature_lifetime_steps == 0
            else:
                assert state.log_alpha is previous["parameter"]
                assert state.temperature_optim is previous["optimizer"]
                torch.testing.assert_close(engine.alpha, previous["alpha"], rtol=0, atol=0)
                _assert_tree_equal(state.temperature_optim.state_dict(), previous["adam"])
            observed.append((t0, should_reset, model.agent.outer_version))

        def checked_record(*, planned, action_seconds):
            nonlocal previous
            record(planned=planned, action_seconds=action_seconds)
            if planned:
                state = engine.state
                previous = {
                    "parameter": state.log_alpha,
                    "optimizer": state.temperature_optim,
                    "alpha": engine.alpha.detach().clone(),
                    "adam": _clone_tree(state.temperature_optim.state_dict()),
                    "outer_version": model.agent.outer_version,
                }

        monkeypatch.setattr(engine, "_prepare_workspace", checked_prepare)
        monkeypatch.setattr(model, "_record_action_metrics", checked_record)
        # First five decisions seed replay, followed by two learned episodes.
        model.learn(total_timesteps=15)
        assert len(observed) == 10
        assert sum(t0 for t0, _, _ in observed) == 2
        assert sum(reset for _, reset, _ in observed) == (2 if scope == "episode" else 1)
        assert model.agent.outer_version == model.agent.num_updates == 11
    finally:
        model.close()
        model.env.close()


@pytest.mark.parametrize("scope", ["episode", "run"])
def test_online_carry_exact_resume_retains_alpha_adam_and_next_action(scope):
    source = _training_model(scope)
    restored = _training_model(scope)
    try:
        source.learn(total_timesteps=15)
        source.agent.prepare_training_resume_boundary()
        saved = deepcopy(source.agent.training_state_dict())
        inner = saved["inner"]["workspace"]
        assert (inner["log_alpha"] is not None) == (scope == "run")
        assert (inner["temperature_optim"] is not None) == (scope == "run")
        for field in ("actor", "critic", "actor_optim", "critic_optim", "replay"):
            assert inner[field] is None
        restored.agent.load_training_state_dict(saved)
        _assert_tree_equal(restored.agent.training_state_dict(), saved)
        source.agent.reset()
        restored.agent.reset()
        observation = torch.tensor([.1, .2, .3])
        torch.testing.assert_close(
            source.agent.act(observation, t0=True, eval_mode=False),
            restored.agent.act(observation, t0=True, eval_mode=False),
            rtol=0, atol=0,
        )
        left, right = source.agent.inner_engine, restored.agent.inner_engine
        torch.testing.assert_close(left.alpha, right.alpha, rtol=0, atol=0)
        _assert_tree_equal(left.state.temperature_optim.state_dict(),
                           right.state.temperature_optim.state_dict())
    finally:
        source.close()
        restored.close()
        source.env.close()
        restored.env.close()


def test_million_decision_run_carry_resume_accepts_adam_float32_step_saturation():
    source, restored = _training_model("run"), _training_model("run")
    try:
        source.agent.act(torch.zeros(3), t0=True)
        engine = source.agent.inner_engine
        # J8/A4 supplies 32 million alpha updates over one million decisions.
        # Adam's float32 scalar stops increasing at 2**24; the scientific
        # integer lifetime remains authoritative for continuation and logging.
        engine.state.temperature_lifetime_steps = 32_000_000
        engine.state.temperature_optim.state[engine.state.log_alpha]["step"].fill_(2**24)
        source.agent.prepare_training_resume_boundary()
        saved = deepcopy(source.agent.training_state_dict())
        restored.agent.load_training_state_dict(saved)
        _assert_tree_equal(restored.agent.training_state_dict(), saved)
        assert restored.agent.inner_engine.state.temperature_lifetime_steps == 32_000_000
    finally:
        source.close()
        restored.close()
        source.env.close()
        restored.env.close()


def test_auxiliary_detach_blocks_representation_loss_but_keeps_critic_and_action_gradients():
    model = _training_model("run")
    try:
        agent = model.agent
        # Fresh final weights are zero. Nonzero weights exercise the learned
        # auxiliary critic's actual representation and action derivatives.
        with torch.no_grad():
            for head in agent.model._aux_return_Qs:
                head[-1].weight.fill_(.1)
        actions = torch.full((2, 2, 1), .2)
        z0 = agent.model.encode(torch.full((2, 3), .3))
        z1 = agent.model.next(z0, actions[0])
        latents = torch.stack((z0, z1, agent.model.next(z1, actions[1])))
        targets = torch.ones((2, 2, 1))
        representation = tuple(agent.model._encoder.parameters()) + tuple(agent.model._dynamics.parameters())
        critic = tuple(agent.model._aux_return_Qs.parameters())
        action_gradients = []
        for detach in (False, True):
            agent.aux_return.cfg.aux_return_detach_representation = detach
            loss, _ = agent.aux_return.critic_loss(latents, actions, targets)
            gradients = torch.autograd.grad(loss, representation + critic,
                                           allow_unused=True, retain_graph=True)
            rep_gradients, critic_gradients = gradients[:len(representation)], gradients[len(representation):]
            if detach:
                assert all(gradient is None for gradient in rep_gradients)
            else:
                assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in rep_gradients)
            assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in critic_gradients)
            action = actions[0].clone().requires_grad_(True)
            value = agent.model.aux_return_Q(z0.detach(), action, detach=True,
                                            reduction="mean_all")
            action_gradients.append(torch.autograd.grad(value.sum(), action)[0])
        assert torch.count_nonzero(action_gradients[0])
        torch.testing.assert_close(*action_gradients, rtol=0, atol=0)
    finally:
        model.close()
        model.env.close()


@pytest.mark.parametrize("scope,interval", [("episode", 1), ("run", 1), ("run", 3)])
def test_full_trainer_continuation_matches_uninterrupted_temperature_carry(
    scope, interval, tmp_path, monkeypatch,
):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def learner(env):
        params = _tiny_params(
            seed_steps=1, train_unroll_horizon=1, inner_rollout_horizon=1, inner_rounds=1,
            inner_updates_per_round=1,
            inner_solve_interval=interval,
            inner_temperature_scope=scope, inner_temperature_optimizer_scope=scope,
            aux_return_mode="sac", inner_actor_source="sac",
            inner_critic_source="aux_return", inner_horizon_actor_source="sac",
            inner_horizon_critic_source="aux_return", inner_sac_critic_target="reward_only",
            ent_coef="auto_1.0", target_entropy=-10.5,
            wandb=True, wandb_mode="online", wandb_step_every=2,
        )
        result = AMBITDMPC2("AMBITDMPC2", env, params,
            {"seed": 13, "device": "cpu", "env": "BoundaryResume-v0", "total_steps": 8}, {})
        return result.enable_training_resume(total_timesteps=8)

    continuous = learner(BoundaryEnv())
    continuous_session = _session(tmp_path / "continuous", mode="new",
                                  segment="whole", total_steps=8)
    split = None
    split_session = None
    restored = None
    resumed_session = None
    try:
        assert continuous.learn(total_timesteps=8, resume_session=continuous_session) == RESUME_COMPLETE
        split = learner(BoundaryEnv())
        split_session = _session(tmp_path / "split", mode="new",
                                 segment="first", total_steps=8)
        original_step = split.env.step

        def drain_after_adaptation(action):
            result = original_step(action)
            if len(split.env.trace) == 4:
                split_session._drain_requested = True
            return result

        monkeypatch.setattr(split.env, "step", drain_after_adaptation)
        assert split.learn(total_timesteps=8, resume_session=split_session) == RESUME_HANDOFF
        assert split._global_step == 4
        assert split.agent.inner_engine.action_index > 0
        split_session.close()
        restored = learner(BoundaryEnv())
        resumed_session = _session(tmp_path / "split", mode="required",
                                    segment="second", total_steps=8)
        assert restored.learn(total_timesteps=8, resume_session=resumed_session) == RESUME_COMPLETE
        trace = split.env.trace + restored.env.trace
        assert len(trace) == len(continuous.env.trace) == 8
        for actual, expected in zip(trace, continuous.env.trace):
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1] == expected[1]
            np.testing.assert_array_equal(actual[2], expected[2])
            assert actual[3] == expected[3]
        _assert_tree_equal(restored.agent.training_state_dict(),
                           continuous.agent.training_state_dict())
        def root_history(session):
            run = fake_wandb.runs[session.lineage_metadata["initial_wandb_run_id"]]
            return [{key: value for key, value in row.items()
                     if key.startswith("train/inner_root_entropy")}
                    for row in run.history]

        continuous_entropy = root_history(continuous_session)
        assert any("train/inner_root_entropy_final" in row for row in continuous_entropy)
        assert root_history(resumed_session) == continuous_entropy
    finally:
        continuous_session.close()
        if split_session is not None:
            split_session.close()
        if resumed_session is not None:
            resumed_session.close()
        for model in (continuous, split, restored):
            if model is not None:
                model.close()
                model.env.close()
