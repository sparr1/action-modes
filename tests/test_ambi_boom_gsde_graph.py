"""BOOM behavior labels across the live-gSDE collection graph boundary."""

from copy import deepcopy
from itertools import product

import pytest
import torch

from RL.tdmpc2_core import whole_inner_graph
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.whole_inner_graph import (
    WholeInnerGraph, _gsde_inputs_for_action, training_graph_act,
)
from tests.test_ambi_boom_gsde import _agent
from tests.test_ambi_distillation_collection_graph import _assert_close, _inner_snapshot
from tests.test_ambi_gsde_graph import _record_controller
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a scheduled CUDA GPU.")


class _MetadataRunner:
    """Adapter test double with reusable outputs; does not simulate CUDA."""

    def __init__(self, worker, root, *, capture_teacher_policy=False,
                 return_behavior_policy=False, **unused):
        self.capture_teacher_policy = capture_teacher_policy
        self.return_behavior_policy = return_behavior_policy
        self.calls = 0
        self.rng_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        self.behavior = {"pre_tanh_mean": root.new_ones(1), "log_std": root.new_zeros(1)}
        self.teacher = {"pre_tanh_mean": root.new_ones(1), "log_std": root.new_full((1,), 2.)}

    def __call__(self, root, **inputs):
        self.calls += 1
        for metadata in (self.behavior, self.teacher):
            metadata["pre_tanh_mean"].add_(1.)
        result = (root.new_zeros(1), {}, [])
        if self.return_behavior_policy:
            result = (*result, self.behavior)
        if self.capture_teacher_policy:
            result = (*result, self.teacher)
        return result


@pytest.mark.parametrize("teacher,behavior", list(product((False, True), repeat=2)))
def test_solve_forwards_independent_metadata_controls_without_changing_noise(teacher, behavior):
    reference = _agent(inner_execution_std_scale=.5)
    target = _agent(inner_execution_std_scale=.5)
    target.load(deepcopy(reference.checkpoint_state()))
    root = reference.model.encode(torch.zeros(1, 3)).detach()
    expected = reference.inner_engine.act(root, t0=True, collect_diagnostics=False)
    runner = object.__new__(WholeInnerGraph)
    runner.engine = target.inner_engine
    runner._engine_act = target.inner_engine.act
    runner.root = root
    runner.capture_teacher_policy = teacher
    runner.return_behavior_policy = behavior
    actual = runner._solve()
    assert len(actual) == 3 + int(teacher) + int(behavior)
    torch.testing.assert_close(actual[0], expected[0])
    _assert_tree_equal(reference.inner_engine.rng.training_state_dict(),
                       target.inner_engine.rng.training_state_dict())
    if teacher and behavior:
        torch.testing.assert_close(actual[3]["pre_tanh_mean"], actual[4]["pre_tanh_mean"])
        torch.testing.assert_close(actual[3]["log_std"], actual[4]["log_std"] + torch.tensor(.5).log())


@pytest.mark.parametrize("teacher", [False, True])
@pytest.mark.parametrize("behavior", [False, True])
def test_adapter_keeps_behavior_and_teacher_owned_and_public_arity(teacher, behavior):
    agent = _agent()
    engine = agent.inner_engine
    root = torch.zeros(1, agent.cfg.latent_dim)
    runner = _MetadataRunner(engine, root, capture_teacher_policy=teacher, return_behavior_policy=True)
    engine._whole_graph_runner = runner
    before = engine.action_index
    first = training_graph_act(engine, root, t0=True, capture_teacher_policy=teacher,
                               return_behavior_policy=behavior)
    assert len(first) == 3 + int(behavior)
    assert first[1]["inner_cuda_graph_used"] == 1.
    retained_behavior = _clone_tree(first[3]) if behavior else None
    retained_teacher = _clone_tree(engine.last_teacher_policy) if teacher else None
    escaped_teacher = engine.last_teacher_policy
    training_graph_act(engine, root, capture_teacher_policy=teacher, return_behavior_policy=behavior)
    assert engine.action_index == before + 2 and runner.calls == 2
    assert engine._gsde_execution_uses == 2
    if behavior:
        _assert_tree_equal(first[3], retained_behavior)
        assert first[3]["pre_tanh_mean"].data_ptr() != runner.behavior["pre_tanh_mean"].data_ptr()
    if teacher:
        _assert_tree_equal(escaped_teacher, retained_teacher)
        assert escaped_teacher["log_std"].item() != runner.behavior["log_std"].item()


def test_adapter_rebuild_preserves_rng_and_calls_for_either_metadata_flag(monkeypatch):
    agent = _agent(outer_actor_objective="q_maximization", store_behavior_policy=False)
    engine = agent.inner_engine
    monkeypatch.setattr(whole_inner_graph, "WholeInnerGraph", _MetadataRunner)
    root = torch.zeros(1, agent.cfg.latent_dim)
    previous = None
    saved_rng = torch.tensor([9, 8, 7], dtype=torch.uint8)
    for calls, (teacher, behavior) in enumerate(((False, False), (False, True), (True, True), (True, False)), 1):
        result = training_graph_act(engine, root, t0=calls == 1,
                                    capture_teacher_policy=teacher, return_behavior_policy=behavior)
        runner = engine._whole_graph_runner
        assert runner is not previous and runner.calls == calls
        assert len(result) == 3 + int(behavior)
        if previous is not None:
            torch.testing.assert_close(runner.rng_state, saved_rng, rtol=0, atol=0)
        runner.rng_state = saved_rng.clone()
        previous = runner


def test_engine_dispatches_behavior_and_teacher_capture_to_collection_graph(monkeypatch):
    agent = _agent(inner_cuda_graph=True, compile=True, compile_strict=True)
    received = {}

    def dispatch(engine, root, **kwargs):
        received.update(kwargs)
        return root.new_zeros(1), {}, [], {"pre_tanh_mean": root.new_zeros(1)}

    monkeypatch.setattr(whole_inner_graph, "validate_training_graph", lambda engine: None)
    monkeypatch.setattr(whole_inner_graph, "training_graph_act", dispatch)
    result = agent.inner_engine.act(
        torch.zeros(1, agent.cfg.latent_dim), t0=True, collect_diagnostics=False,
        return_behavior_policy=True, capture_teacher_policy=True,
    )
    assert len(result) == 4
    assert received == dict(t0=True, return_behavior_policy=True, capture_teacher_policy=True)


@pytest.mark.parametrize("scope", ["round", "solve", "real_steps"])
def test_boom_graph_native_interleaving_keeps_behavior_noise_lifetimes(scope, monkeypatch):
    options = dict(inner_gsde_noise_scope=scope, inner_rounds=2,
                   inner_replay_capacity=16, inner_gsde_resample_steps=2,
                   gsde_execution_resample_steps=2)
    native, mixed = _agent(**options), _agent(**options)
    a, b = native.inner_engine, mixed.inner_engine
    root = native.model.encode(torch.zeros(1, 3)).detach()
    a_draws, b_draws = _record_controller(a, monkeypatch), _record_controller(b, monkeypatch)
    b._whole_graph_runner = _MetadataRunner(b, root, return_behavior_policy=True)
    for index, diagnostic in enumerate((False, True, False, False)):
        kwargs = dict(t0=index in {0, 3}, return_behavior_policy=True)
        a.act(root, collect_diagnostics=diagnostic, **kwargs)
        if diagnostic:
            b.act(root, collect_diagnostics=True, **kwargs)
        else:
            training_graph_act(b, root, **kwargs)
        assert a._gsde_execution_uses == b._gsde_execution_uses
        assert a._gsde_decision_index == b._gsde_decision_index
    _assert_tree_equal(a_draws, b_draws)
    for stream in ("collection", "execution"):
        torch.testing.assert_close(a.rng.generator(stream).get_state(),
                                   b.rng.generator(stream).get_state(), rtol=0, atol=0)


def _cuda_agent(scope):
    cfg = deepcopy(_agent(
        inner_cuda_graph=True, compile=True, compile_strict=True,
        critic_ensemble_backend="vmap", inner_gsde_noise_scope=scope,
        inner_gsde_resample_steps=2, gsde_execution_resample_steps=2,
        inner_rounds=2, inner_replay_capacity=16, inner_execution_std_scale=.5,
    ).cfg)
    cfg.device = "cuda"
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    return agent


@CUDA
@pytest.mark.parametrize("scope", ["round", "solve", "real_steps"])
def test_cuda_boom_gsde_metadata_matches_uncaptured_solve_and_optimizer_state(scope):
    torch._dynamo.reset()
    try:
        source, target = _cuda_agent(scope), _cuda_agent(scope)
        target.load(deepcopy(source.checkpoint_state()))
        root = source.model.encode(torch.zeros(1, 3, device="cuda")).detach()
        a, b = InnerImprovementEngine(source), InnerImprovementEngine(target)
        eager = WholeInnerGraph(a, root, capture=False, return_behavior_policy=True,
                               capture_teacher_policy=True)
        graph = WholeInnerGraph(b, root, capture=True, return_behavior_policy=True,
                               capture_teacher_policy=True)
        for index in range(3):
            controller = source.inner_engine
            controller._begin_gsde_action(t0=index == 0, eval_mode=False)
            collection, execution = _gsde_inputs_for_action(controller, root)
            cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            expected = _clone_tree(eager(root, gsde_collection=collection, gsde_execution=execution))
            actual = _clone_tree(graph(root, gsde_collection=collection, gsde_execution=execution))
            torch.cuda.synchronize()
            _assert_close(actual, expected)
            _assert_close(_inner_snapshot(b), _inner_snapshot(a))
            torch.testing.assert_close(actual[3]["log_std"], actual[4]["log_std"] + root.new_tensor(.5).log())
            torch.testing.assert_close(graph.rng_state, eager.rng_state, rtol=0, atol=0)
            torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng, rtol=0, atol=0)
            torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
            controller._gsde_execution_uses += 1
            controller._finish_gsde_action()
            with torch.no_grad():
                source.model._pi.log_scale.add_(.01)
                target.model._pi.log_scale.add_(.01)
    finally:
        torch._dynamo.reset()


@CUDA
def test_cuda_boom_gsde_boundary_restore_repeats_actions_and_behavior_labels():
    torch._dynamo.reset()
    try:
        agent = _cuda_agent("real_steps")
        observation = torch.zeros(3, device="cuda")
        agent.act(observation, t0=True, collect_diagnostics=False, return_behavior_policy=True)
        agent.prepare_training_resume_boundary()
        saved = deepcopy(agent.training_state_dict())
        attempts = []
        for attempt in range(2):
            if attempt:
                agent.load_training_state_dict(saved)
            sequence = []
            for index, diagnostic in enumerate((False, True, False)):
                sequence.append(_clone_tree(agent.act(
                    observation, t0=index == 0, collect_diagnostics=diagnostic,
                    return_behavior_policy=True, capture_teacher_policy=True,
                )))
            agent.prepare_training_resume_boundary()
            attempts.append((sequence, deepcopy(agent.training_state_dict())))
        _assert_close(attempts[0], attempts[1], exact=True)
    finally:
        torch._dynamo.reset()
