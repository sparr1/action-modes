"""Collection-graph teacher, isolation, and exact-continuation contracts."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.whole_inner_graph import (
    WholeInnerGraph, training_graph_act,
)
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_reanalyze import _assert_globals, _globals
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_params


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a scheduled CUDA GPU.")


def make_agent(*, device="cpu", compiled=False, graph=False, full=False):
    params = _tiny_params(
        outer_actor_objective="distillation", reanalyze_interval=0,
        inner_cuda_graph=graph, compile=compiled, compile_strict=compiled,
        critic_ensemble_backend="vmap", q_representation="distributional",
        num_q=5, q_num_bins=101, num_bins=101, q_pair_size=2, dropout=.01,
        q_vmin=-10, q_vmax=10, inner_critic_dropout_enabled=True,
        outer_q_actor_reduction="mean_pair", inner_q_actor_reduction="mean_pair",
        outer_q_target_reduction="min_pair", inner_q_target_reduction="min_pair",
        outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range", sac_actor_loss_scale_tau=.01,
        inner_actor_loss_scale_update="per_update", inner_finite_horizon=True,
        outer_actor_entropy_mode="squashed", inner_actor_entropy_mode="squashed",
        log_std_mapping="tdmpc2_tanh", inner_log_std_mapping="tdmpc2_tanh",
        log_std_min=-10, log_std_max=2, inner_log_std_min=-10, inner_log_std_max=2,
        ent_coef=.0021, inner_temperature_mode="inherit_outer", inner_temperature=.0021,
        inner_actor_lr=.0003, inner_critic_lr=.0003, inner_adam_eps=1e-8,
        inner_actor_adam_eps=1e-5, inner_critic_target_tau=.01,
        inner_actor_grad_clip_norm=20., inner_critic_grad_clip_norm=20.,
        inner_behavior_action="policy_sample", inner_execution_action="policy_sample",
        inner_rounds=2, inner_rollouts_per_round=4, inner_rollout_horizon=2,
        inner_batch_size=4, inner_replay_capacity=16,
        inner_critic_updates_per_round=3, inner_actor_updates_per_round=2,
        inner_temperature_updates_per_round=0, inner_diagnostic_rollouts=0,
    )
    params.pop("inner_updates_per_round")
    if full:
        params.update(enc_dim=256, mlp_dim=512, latent_dim=512, simnorm_dim=8,
                      train_unroll_horizon=3, outer_planning_horizon=3,
                      inner_rounds=8, inner_rollouts_per_round=128, inner_rollout_horizon=3,
                      inner_batch_size=256, inner_replay_capacity=3072,
                      inner_critic_updates_per_round=16, inner_actor_updates_per_round=4)
    cfg = _build_cfg(**params)
    cfg.device = device
    cfg.obs_shape = {"state": (67,)} if full else {"state": (5,)}
    cfg.obs_shapes = [cfg.obs_shape]
    cfg.action_dim = 21 if full else 3
    cfg.action_dims = [cfg.action_dim]
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    return agent


def _assert_close(actual, expected, *, exact=False):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0 if exact else 1e-4,
                                   atol=0 if exact else 2e-6)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_close(actual[key], expected[key], exact=exact)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, target in zip(actual, expected):
            _assert_close(value, target, exact=exact)
    else:
        assert actual == expected


def _inner_snapshot(engine):
    pool = engine._action_pool
    return {
        name: None if getattr(pool, name) is None else _clone_tree(getattr(pool, name).state_dict())
        for name in ("actor", "critic", "critic_target", "actor_optim", "critic_optim")
    }


def test_unused_graph_state_roundtrips_and_rejects_cross_resume_before_mutation():
    agent = make_agent(compiled=True, graph=True)
    engine = agent.inner_engine
    state = deepcopy(engine.training_state_dict())
    assert state["whole_graph"]["calls"] == 0 and state["whole_graph"]["rng_state"] is None
    engine.load_training_state_dict(state)
    _assert_tree_equal(engine.training_state_dict(), state)
    for change in ({"calls": 1}, {"protocol": "different"}, {"seed": -1}, {"rng_state": torch.ones(3)}):
        bad = deepcopy(state)
        bad["whole_graph"].update(change)
        with pytest.raises((ValueError, RuntimeError)):
            engine.load_training_state_dict(bad)
        _assert_tree_equal(engine.training_state_dict(), state)
    native = make_agent().inner_engine
    before = deepcopy(native.training_state_dict())
    assert "whole_graph" not in before
    with pytest.raises(ValueError, match="lineage"):
        native.load_training_state_dict(state)
    _assert_tree_equal(native.training_state_dict(), before)
    with pytest.raises(ValueError, match="complete RNG"):
        engine.load_training_state_dict(before)
    _assert_tree_equal(engine.training_state_dict(), state)


def test_graph_teacher_outputs_own_storage_and_count_one_logical_action():
    agent = make_agent()
    engine = agent.inner_engine
    values = {"pre_tanh_mean": torch.ones(3), "log_std": torch.zeros(3)}

    class Runner:
        capture_teacher_policy = True
        calls = 0

        def __call__(self, root):
            self.calls += 1
            return root.new_zeros(3), {"inner_actor_optimizer_steps": 4.}, [], values

    engine._whole_graph_runner = Runner()
    before = engine.action_index
    _, metrics, _ = training_graph_act(engine, torch.zeros(1, agent.cfg.latent_dim), capture_teacher_policy=True)
    assert engine.action_index == before + 1 and engine._whole_graph_runner.calls == 1
    assert engine.state.outer_version == agent.outer_version
    assert metrics["inner_cuda_graph_used"] == 1
    retained = _clone_tree(engine.last_teacher_policy)
    values["pre_tanh_mean"].fill_(9.)
    _assert_tree_equal(engine.last_teacher_policy, retained)
    engine.finalize_timing_metrics(metrics)


def test_teacher_only_workers_bypass_collection_graph_dispatch(monkeypatch):
    from RL.tdmpc2_core import whole_inner_graph
    agent = make_agent(compiled=True, graph=True)
    engine = agent.inner_engine
    monkeypatch.setattr(whole_inner_graph, "validate_training_graph",
                        lambda *args: pytest.fail("Reanalysis must not route through collection graphs"))
    monkeypatch.setattr(engine, "_act", lambda *args, **kwargs: (torch.zeros(3), {}, []))
    engine._teacher_only = True
    engine.act(torch.zeros(1, agent.cfg.latent_dim), collect_diagnostics=False, capture_teacher_policy=True)
    assert engine._whole_graph_runner is None


def test_collection_graph_failure_restores_mixed_model_modes():
    agent = make_agent(compiled=True, graph=True)
    agent.model.train()
    agent.model._pi[0].eval()
    modes = [(module, module.training) for module in agent.model.modules()]
    with pytest.raises(ValueError, match="requires CUDA"):
        agent.act(torch.zeros(5), collect_diagnostics=False, capture_teacher_policy=True)
    assert all(module.training == expected for module, expected in modes)
    assert agent.inner_engine.last_teacher_policy is None


def test_prepared_boundary_is_consumed_by_direct_action_like_explicit_reset():
    torch.manual_seed(519)
    agent = make_agent()
    observation = torch.zeros(agent.cfg.obs_shape["state"])
    agent.act(observation, collect_diagnostics=True, capture_teacher_policy=True)
    agent.prepare_training_resume_boundary()
    saved = deepcopy(agent.training_state_dict())
    episode = agent.inner_engine.episode_index
    attempts = []
    for explicit_reset in (False, True):
        agent.load_training_state_dict(saved)
        assert agent._resume_boundary_prepared
        if explicit_reset:
            agent.reset()
        host = _globals()
        action = agent.act(observation + .1, t0=True, collect_diagnostics=True,
                           capture_teacher_policy=True)
        _assert_globals(host)
        assert not agent._resume_boundary_prepared
        assert agent.inner_engine.episode_index == episode
        assert agent.inner_engine._nondefault_workspace_fields(agent.inner_engine._action_pool)
        teacher = _clone_tree(agent.last_teacher_policy)
        metrics = {key: value for key, value in agent.last_inner_metrics.items()
                   if not key.endswith("seconds")}
        agent.prepare_training_resume_boundary()
        assert agent.inner_engine.episode_index == episode + 1
        assert not agent.inner_engine._nondefault_workspace_fields(agent.inner_engine._action_pool)
        attempts.append((action.clone(), teacher, metrics, deepcopy(agent.training_state_dict())))
    _assert_close(attempts[0], attempts[1], exact=True)


@CUDA
@pytest.mark.parametrize("full,precision", [
    pytest.param(False, "highest", id="small"),
    pytest.param(True, "highest", id="production_budget"),
    pytest.param(True, "high", id="production_budget_tf32"),
])
def test_collection_graph_matches_uncaptured_protocol_teachers_and_optimizer_state(full, precision):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    torch._dynamo.reset()
    try:
        torch.manual_seed(473)
        reference = make_agent(device="cuda", compiled=True, full=full)
        captured = make_agent(device="cuda", compiled=True, full=full)
        captured.load(deepcopy(reference.checkpoint_state()))
        root = torch.randn(1, reference.cfg.latent_dim, device="cuda")
        control = WholeInnerGraph(reference.inner_engine, root, capture=False, capture_teacher_policy=True)
        graph = WholeInnerGraph(captured.inner_engine, root, capture=True, capture_teacher_policy=True)
        for iteration in range(2):
            with torch.no_grad():
                for agent in (reference, captured):
                    agent.actor_loss_scale.fill_(2. + iteration)
                    if iteration:
                        next(agent.model._pi.parameters()).add_(.001)
            before = [_clone_tree(agent.model.state_dict()) for agent in (reference, captured)]
            host, cuda = _globals(), torch.cuda.get_rng_state().clone()
            expected = _clone_tree(control(root))
            actual = _clone_tree(graph(root))
            torch.cuda.synchronize()
            _assert_close(actual, expected)
            _assert_close(_inner_snapshot(captured.inner_engine), _inner_snapshot(reference.inner_engine))
            torch.testing.assert_close(graph.rng_state, control.rng_state, rtol=0, atol=0)
            _assert_globals(host)
            torch.testing.assert_close(torch.cuda.get_rng_state(), cuda, rtol=0, atol=0)
            for agent, state in zip((reference, captured), before):
                _assert_tree_equal(agent.model.state_dict(), state)
                assert all(parameter.grad is None for parameter in agent.model.parameters())
            assert actual[1]["inner_actor_optimizer_steps"] == captured.cfg.inner_actor_updates_per_action
            assert actual[1]["inner_critic_optimizer_steps"] == captured.cfg.inner_critic_updates_per_action
            torch.testing.assert_close(actual[1]["inner_actor_loss_scale_initial"], root.new_tensor(2. + iteration))
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(previous)


@CUDA
def test_teacher_capture_does_not_change_collection_action_or_cuda_rng():
    torch._dynamo.reset()
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        torch.manual_seed(825)
        agent = make_agent(device="cuda", compiled=True)
        root = torch.randn(1, agent.cfg.latent_dim, device="cuda")
        plain = WholeInnerGraph(InnerImprovementEngine(agent), root, capture_teacher_policy=False)
        teacher = WholeInnerGraph(InnerImprovementEngine(agent), root, capture_teacher_policy=True)
        for _ in range(2):
            expected = _clone_tree(plain(root))
            actual = _clone_tree(teacher(root))
            torch.cuda.synchronize()
            _assert_close(actual[:3], expected, exact=True)
            torch.testing.assert_close(teacher.rng_state, plain.rng_state, rtol=0, atol=0)
            assert set(actual[3]) == {"pre_tanh_mean", "log_std"}
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(previous)


@CUDA
@pytest.mark.parametrize("full", [False, True], ids=["small", "production_budget"])
def test_collection_graph_checkpoint_replays_next_teachers_actions_and_native_diagnostics_exactly(full):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()
    try:
        torch.manual_seed(238)
        agent = make_agent(device="cuda", compiled=True, graph=True, full=full)
        observation = torch.zeros(agent.cfg.obs_shape["state"])
        agent.act(observation, collect_diagnostics=False, capture_teacher_policy=True)
        agent.prepare_training_resume_boundary()
        saved = deepcopy(agent.training_state_dict())
        original = agent.inner_engine._whole_graph_runner
        attempts = []
        for attempt in range(2):
            if attempt:
                agent.load_training_state_dict(saved)
                assert agent.inner_engine._whole_graph_runner is None
            sequence = []
            for index, diagnostic in enumerate((False, True, False)):
                with torch.no_grad():
                    agent.actor_loss_scale.fill_(4. + index)
                    if index:
                        next(agent.model._pi.parameters()).add_(.0001)
                host, cuda = _globals(), torch.cuda.get_rng_state().clone()
                action = agent.act(observation + index * .1, t0=index == 0,
                                   collect_diagnostics=diagnostic, capture_teacher_policy=True)
                assert agent.last_inner_metrics["inner_cuda_graph_used"] == float(not diagnostic)
                assert agent.last_inner_metrics["inner_actor_loss_scale_initial"] == pytest.approx(4. + index)
                _assert_globals(host)
                torch.testing.assert_close(torch.cuda.get_rng_state(), cuda, rtol=0, atol=0)
                sequence.append((action.clone(), _clone_tree(agent.last_teacher_policy), {
                    key: value for key, value in agent.last_inner_metrics.items() if not key.endswith("seconds")
                }))
            agent.prepare_training_resume_boundary()
            attempts.append((sequence, deepcopy(agent.training_state_dict())))
        assert agent.inner_engine._whole_graph_runner is not original
        _assert_close(attempts[0], attempts[1], exact=True)
        assert attempts[0][1]["inner"]["whole_graph"]["calls"] == 3
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(previous)
