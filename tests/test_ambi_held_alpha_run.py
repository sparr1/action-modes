"""K3 preserves run-carried alpha/Adam while refreshing the other priors."""

from copy import deepcopy

import pytest
import torch

from tests.test_ambi_alpha_carry_training_state import _training_model
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_pave_integration import _batch


@pytest.fixture
def models():
    opened = []

    def create():
        model = _training_model(
            "run", inner_solve_interval=3, inner_rollout_horizon=3, train_unroll_horizon=3,
            inner_rounds=8, inner_critic_updates_per_round=16,
            inner_actor_updates_per_round=4, inner_replay_capacity=48,
            aux_return_detach_representation=True, target_entropy=-10.5,
            inner_temperature_lr=3e-4, inner_execution_action="policy_sample",
        )
        opened.append(model)
        return model

    yield create
    for model in opened:
        model.close()
        model.env.close()


def temperature(engine):
    return (engine.state.log_alpha.detach().clone(),
            _clone_tree(engine.state.temperature_optim.state_dict()),
            engine.state.temperature_lifetime_steps)


def test_j8_alpha_and_adam_update_only_at_solves_and_survive_episode_reset(models):
    agent = models().agent
    engine = agent.inner_engine
    previous = None
    parameter = optimizer = None
    solves = 0
    for decision in range(8):
        if decision == 5:
            before_reset = temperature(engine)
            agent.reset()
            _assert_tree_equal(temperature(engine), before_reset)
            assert engine._held_actor is engine._held_encoder is None
        agent.act(torch.ones(3) * decision / 10, t0=decision in (0, 5))
        metrics = agent.last_inner_metrics
        solve = decision in (0, 3, 5)
        assert metrics["inner_solve_performed"] == float(solve)
        assert metrics["inner_policy_held"] == float(not solve)
        if solve:
            assert metrics["inner_temperature_optimizer_step_initial"] == 32 * solves
            solves += 1
            assert metrics["inner_temperature_optimizer_step_final"] == 32 * solves
            assert metrics["inner_temperature_optimizer_steps"] == 32
            assert metrics["inner_critic_optimizer_steps"] == 128
            assert metrics["inner_actor_optimizer_steps"] == 32
            assert metrics["inner_temperature_episode_reset"] == 0
        else:
            _assert_tree_equal(temperature(engine), previous)
            assert metrics["inner_temperature_optimizer_steps"] == 0
            assert "inner_alpha_initial" not in metrics
            assert "inner_alpha_final" not in metrics
        if parameter is None:
            parameter, optimizer = engine.state.log_alpha, engine.state.temperature_optim
        assert engine.state.log_alpha is parameter
        assert engine.state.temperature_optim is optimizer
        assert engine.state.temperature_lifetime_steps == 32 * solves
        for field in ("actor", "critic", "actor_optim", "critic_optim", "replay"):
            assert getattr(engine.state, field) is None
        previous = temperature(engine)
        # Every real decision still allows the ordinary outer update. It must
        # not mutate the independent run-scoped temperature or optimizer.
        agent._update(*_batch(agent))
        _assert_tree_equal(temperature(engine), previous)


def test_j8_run_alpha_exact_resume_reproduces_next_solve_and_held_actions(models):
    source, restored = models().agent, models().agent
    for decision in range(5):
        source.act(torch.ones(3) * decision / 10, t0=decision == 0,
                   collect_diagnostics=False)
        source._update(*_batch(source))
    before_boundary = temperature(source.inner_engine)
    source.prepare_training_resume_boundary()
    _assert_tree_equal(temperature(source.inner_engine), before_boundary)
    saved = deepcopy(source.training_state_dict())
    assert saved["inner"]["workspace"]["log_alpha"] is not None
    assert saved["inner"]["workspace"]["temperature_optim"] is not None
    restored.load_training_state_dict(saved)
    _assert_tree_equal(restored.training_state_dict(), saved)
    for agent in (source, restored):
        agent.reset()
    for decision in range(4):
        observation = torch.tensor([.1 * decision, -.2, .3])
        expected = source.act(observation, t0=decision == 0, collect_diagnostics=False)
        actual = restored.act(observation, t0=decision == 0, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(temperature(restored.inner_engine), temperature(source.inner_engine))
        assert restored.last_inner_metrics["inner_solve_performed"] == float(decision % 3 == 0)
        rng = torch.random.get_rng_state().clone()
        for agent in (source, restored):
            torch.random.set_rng_state(rng)
            agent._update(*_batch(agent))
    for agent in (source, restored):
        agent.prepare_training_resume_boundary()
    _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())


@pytest.mark.parametrize("interval,expected", [(1, [1000, 2000]), (3, [1001, 2001])])
def test_training_diagnostics_follow_first_solve_after_each_1000_decisions(
    monkeypatch, interval, expected,
):
    model = _training_model("run", inner_solve_interval=interval, inner_diagnostics_every=1000)
    try:
        engine = model.agent.inner_engine
        sampled = []
        episode_decision = 0

        def prepare(*, t0, eval_mode):
            nonlocal episode_decision
            if t0:
                episode_decision = 0
            return episode_decision % interval != 0

        def act(obs, *, t0, eval_mode, collect_diagnostics, **kwargs):
            nonlocal episode_decision
            if t0:
                episode_decision = 0
            solve = episode_decision % interval == 0
            assert not collect_diagnostics or solve
            model.agent.last_inner_metrics = {}
            if collect_diagnostics:
                sampled.append(model._global_step + 1)
                model.agent.last_inner_metrics = {
                    "inner_diagnostics_step": 123., "inner_root_entropy_final": 2.,
                }
            episode_decision += 1
            return torch.zeros(1)

        monkeypatch.setattr(engine, "prepare_control", prepare)
        monkeypatch.setattr(model.agent, "act", act)
        for step in range(1, 2003):
            model._global_step = step - 1
            model._act_agent(torch.zeros(3), t0=(step - 1) % 500 == 0, eval_mode=False)
            metrics = model.agent.last_inner_metrics
            if step in expected:
                assert metrics["inner_diagnostics_step"] == step
                assert "inner_root_entropy_final" in metrics
            else:
                assert "inner_diagnostics_step" not in metrics
                assert "inner_root_entropy_final" not in metrics
        assert sampled == expected
        state = model._training_resume_algorithm_state()
        assert ("held_diagnostics_bucket" in state) == (interval > 1)
        if interval > 1:
            assert state["held_diagnostics_bucket"] == 2
    finally:
        model.close()
        model.env.close()


def test_pending_diagnostic_at_episode_boundary_survives_exact_resume():
    models = [_training_model("run", inner_solve_interval=3, inner_diagnostics_every=1000)
              for _ in range(2)]
    source, restored = models
    try:
        for step in (998, 999, 1000):
            source._global_step = step - 1
            source._act_agent(torch.zeros(3), t0=step == 998, eval_mode=False)
            assert "inner_root_entropy_final" not in source.agent.last_inner_metrics
        source.agent.prepare_training_resume_boundary()
        wrapper = source._training_resume_algorithm_state()
        assert wrapper["held_diagnostics_bucket"] == 0
        restored.agent.load_training_state_dict(deepcopy(source.agent.training_state_dict()))
        restored._load_training_resume_algorithm_state(wrapper)
        for model in models:
            model.agent.reset()
            model._global_step = 1000
        actions = [model._act_agent(torch.zeros(3), t0=True, eval_mode=False) for model in models]
        torch.testing.assert_close(*actions, rtol=0, atol=0)
        for model in models:
            metrics = model.agent.last_inner_metrics
            assert metrics["inner_diagnostics_step"] == 1001
            assert "inner_root_entropy_final" in metrics
            assert model._held_diagnostics_bucket == 1
            model._global_step = 1001
            model._act_agent(torch.zeros(3), t0=False, eval_mode=False)
            assert "inner_root_entropy_final" not in model.agent.last_inner_metrics
        _assert_tree_equal(temperature(source.agent.inner_engine), temperature(restored.agent.inner_engine))
    finally:
        for model in models:
            model.close()
            model.env.close()
