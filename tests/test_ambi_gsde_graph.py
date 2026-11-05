"""Explicit gSDE graph inputs, native interleaving, and CUDA replay gates."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.teacher_graph import validate_teacher_graph_config
from RL.tdmpc2_core.whole_inner_graph import (
    WholeInnerGraph, _GsdeGraphInputs, _gsde_inputs_for_action,
    training_graph_act, training_graph_protocol, validate_collection_graph_config,
)
from tests.test_ambi_gsde_lifecycle import _model
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_params


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU; replay is not CPU-validated.")


@pytest.mark.parametrize("scope", ["round", "solve", "real_steps"])
@pytest.mark.parametrize("timing", ["round", "step"])
@pytest.mark.parametrize("target", ["reward_only", "entropy_augmented"])
def test_gsde_collection_validator_accepts_static_schedules_and_entropy_targets(scope, timing, target):
    options = dict(
        actor_distribution="gsde", compile=True, compile_strict=True, inner_cuda_graph=True,
        outer_actor_entropy_mode="pre_tanh", inner_actor_entropy_mode="pre_tanh",
        ent_coef=.2, inner_target_entropy=-1., inner_gsde_noise_scope=scope,
        outer_critic_target=target, inner_sac_critic_target=target,
        inner_rounds=2, inner_rollouts_per_round=3, inner_updates_per_round=1,
    )
    if timing == "step":
        options.update(inner_update_timing="step", inner_steps_per_update=3,
                       inner_updates_per_round=None)
    cfg = _build_cfg(**options)
    validate_collection_graph_config(cfg)
    with pytest.raises(ValueError, match="live collection noise"):
        validate_teacher_graph_config(cfg)


@pytest.mark.parametrize("scope", ["round", "solve", "real_steps"])
def test_graph_buffers_are_owned_and_share_only_requested_rounds(scope):
    model = _model(inner_gsde_noise_scope=scope)
    engine = model.agent.inner_engine
    root = engine.model.encode(torch.zeros(1, 3)).detach()
    try:
        inputs = _GsdeGraphInputs(engine, root)
        engine._begin_gsde_action(t0=True, eval_mode=False)
        collection, execution = _gsde_inputs_for_action(engine, root)
        inputs.copy_(collection, execution)
        assert len(inputs.collection) == (2 if scope == "round" else 1)
        retained = tuple(tensor.clone() for tensor in inputs.collection)
        for supplied in collection:
            supplied.fill_(27)
        for actual, expected in zip(inputs.collection, retained):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        inputs.begin(t0=True, eval_mode=False)
        first = inputs.next_round(root, 3, None)
        second = inputs.next_round(root, 3, None)
        assert (first.data_ptr() == second.data_ptr()) is (scope != "round")
        with pytest.raises(RuntimeError, match="more rollout rounds"):
            inputs.next_round(root, 3, None)
        before = tuple(tensor.clone() for tensor in inputs.collection)
        with pytest.raises(ValueError, match="shape, dtype and device"):
            inputs.copy_(collection, execution.double())
        for actual, expected in zip(inputs.collection, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        model.env.close()


class _InputRunner:
    """CPU contract double: receives real inputs but does not emulate CUDA."""

    capture_teacher_policy = False
    calls = 0

    def __init__(self, engine, root):
        self.inputs = _GsdeGraphInputs(engine, root)

    def __call__(self, root, *, gsde_collection, gsde_execution):
        self.inputs.copy_(gsde_collection, gsde_execution)
        self.calls += 1
        return root.new_zeros(1), {}, []


def _record_controller(engine, monkeypatch):
    collection, execution = [], []
    old_collection = engine._gsde_noise_for_round
    old_execution = engine._gsde_noise_for_execution

    def collect(*args, **kwargs):
        result = old_collection(*args, **kwargs)
        collection.append(None if result is None else result.clone())
        return result

    def execute(*args, **kwargs):
        result = old_execution(*args, **kwargs)
        execution.append(None if result is None else result.clone())
        return result

    monkeypatch.setattr(engine, "_gsde_noise_for_round", collect)
    monkeypatch.setattr(engine, "_gsde_noise_for_execution", execute)
    return collection, execution


@pytest.mark.parametrize("scope", ["round", "solve", "real_steps"])
@pytest.mark.parametrize("timing", ["round", "step"])
def test_graph_and_native_interleaving_keep_identical_behavior_matrix_streams(scope, timing, monkeypatch):
    options = dict(inner_gsde_noise_scope=scope, inner_gsde_resample_steps=2,
                   gsde_execution_resample_steps=2)
    if timing == "step":
        options.update(inner_update_timing="step", inner_steps_per_update=3,
                       inner_updates_per_round=None)
    native, mixed = _model(**options), _model(**options)
    a, b = native.agent.inner_engine, mixed.agent.inner_engine
    root = a.model.encode(torch.zeros(1, 3)).detach()
    a_draws, b_draws = _record_controller(a, monkeypatch), _record_controller(b, monkeypatch)
    b._whole_graph_runner = _InputRunner(b, root)
    try:
        globals_before = torch.random.get_rng_state().clone()
        for index, diagnostic in enumerate((False, True, False, False)):
            t0 = index in {0, 3}
            _, values, _ = a._act(root, t0=t0, collect_diagnostics=diagnostic)
            a.finalize_timing_metrics(values)
            if diagnostic:
                _, values, _ = b._act(root, t0=t0, collect_diagnostics=True)
            else:
                _, values, _ = training_graph_act(b, root, t0=t0)
            b.finalize_timing_metrics(values)
            assert a._gsde_execution_uses == b._gsde_execution_uses
            assert a._gsde_decision_index == b._gsde_decision_index
        _assert_tree_equal(a_draws, b_draws)
        for stream in ("collection", "execution"):
            torch.testing.assert_close(a.rng.generator(stream).get_state(),
                                       b.rng.generator(stream).get_state(), rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), globals_before, rtol=0, atol=0)
    finally:
        native.env.close()
        mixed.env.close()


def test_unused_graph_boundary_records_new_protocol_and_rejects_legacy_graph_lineage():
    model = _model(inner_cuda_graph=True, compile=True, compile_strict=True)
    engine = model.agent.inner_engine
    try:
        before = deepcopy(engine.training_state_dict())
        graph = before["whole_graph"]
        assert graph["protocol"] == training_graph_protocol(model.cfg)
        assert graph["protocol"] == "isolated_cuda_graph_with_live_gsde_inputs_v1"
        engine.load_training_state_dict(before)
        _assert_tree_equal(engine.training_state_dict(), before)
        incompatible = deepcopy(before)
        incompatible["whole_graph"]["protocol"] = "isolated_cuda_graph_with_native_diagnostics_v1"
        with pytest.raises(ValueError, match="protocol"):
            engine.load_training_state_dict(incompatible)
        _assert_tree_equal(engine.training_state_dict(), before)
    finally:
        model.env.close()


def _cuda_agent(scope, timing):
    params = _tiny_params(
        actor_distribution="gsde", compile=True, compile_strict=True, inner_cuda_graph=True,
        inner_gsde_noise_scope=scope, inner_gsde_resample_steps=2,
        gsde_execution_resample_steps=2, outer_actor_entropy_mode="pre_tanh",
        inner_actor_entropy_mode="pre_tanh", ent_coef=.2, inner_target_entropy=-1.,
        inner_rounds=2, inner_rollouts_per_round=3, inner_updates_per_round=1,
        inner_replay_capacity=12, outer_critic_target="entropy_augmented",
        inner_sac_critic_target="entropy_augmented",
    )
    if timing == "step":
        params.update(inner_update_timing="step", inner_steps_per_update=3,
                      inner_updates_per_round=None)
    cfg = _build_cfg(**params)
    cfg.device = "cuda"
    cfg.obs_shape = {"state": (3,)}
    cfg.obs_shapes = [cfg.obs_shape]
    cfg.action_dim, cfg.action_dims = 1, [1]
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    return agent


@CUDA
@pytest.mark.parametrize("scope", ["round", "solve", "real_steps"])
@pytest.mark.parametrize("timing", ["round", "step"])
def test_cuda_replay_matches_uncaptured_solve_with_explicit_current_matrices(scope, timing):
    from tests.test_ambi_distillation_collection_graph import _assert_close, _inner_snapshot

    torch._dynamo.reset()
    try:
        source, target = _cuda_agent(scope, timing), _cuda_agent(scope, timing)
        target.load(deepcopy(source.checkpoint_state()))
        source_controller = source.inner_engine
        target_controller = target.inner_engine
        root = source.model.encode(torch.zeros(1, 3, device="cuda")).detach()
        source_worker, target_worker = InnerImprovementEngine(source), InnerImprovementEngine(target)
        eager = WholeInnerGraph(source_worker, root, capture=False)
        graph = WholeInnerGraph(target_worker, root, capture=True)
        for index in range(3):
            for controller in (source_controller, target_controller):
                controller._begin_gsde_action(t0=index == 0, eval_mode=False)
            source_noise = _gsde_inputs_for_action(source_controller, root)
            target_noise = _gsde_inputs_for_action(target_controller, root)
            before_cpu = torch.random.get_rng_state().clone()
            before_cuda = torch.cuda.get_rng_state().clone()
            expected = _clone_tree(eager(root, gsde_collection=source_noise[0], gsde_execution=source_noise[1]))
            actual = _clone_tree(graph(root, gsde_collection=target_noise[0], gsde_execution=target_noise[1]))
            torch.cuda.synchronize()
            _assert_close(actual, expected)
            _assert_close(_inner_snapshot(target_worker), _inner_snapshot(source_worker))
            torch.testing.assert_close(eager.rng_state, graph.rng_state, rtol=0, atol=0)
            torch.testing.assert_close(torch.random.get_rng_state(), before_cpu, rtol=0, atol=0)
            torch.testing.assert_close(torch.cuda.get_rng_state(), before_cuda, rtol=0, atol=0)
            for controller in (source_controller, target_controller):
                controller._gsde_execution_uses += 1
                controller._finish_gsde_action()
            with torch.no_grad():
                source.model._pi.log_scale.add_(.01)
                target.model._pi.log_scale.add_(.01)
    finally:
        torch._dynamo.reset()


@CUDA
def test_cuda_boundary_restore_repeats_graph_native_graph_sequence():
    from tests.test_ambi_distillation_collection_graph import _assert_close

    torch._dynamo.reset()
    try:
        agent = _cuda_agent("real_steps", "round")
        observation = torch.zeros(3, device="cuda")
        agent.act(observation, t0=True, collect_diagnostics=False)
        agent.prepare_training_resume_boundary()
        saved = deepcopy(agent.training_state_dict())
        attempts = []
        for attempt in range(2):
            if attempt:
                agent.load_training_state_dict(saved)
            sequence = []
            for index, diagnostics in enumerate((False, True, False)):
                action = agent.act(observation, t0=index == 0, collect_diagnostics=diagnostics)
                engine = agent.inner_engine
                sequence.append((action.clone(), engine._gsde_execution_noise.clone(),
                                 engine._gsde_collection_noise.clone()))
            agent.prepare_training_resume_boundary()
            attempts.append((sequence, deepcopy(agent.training_state_dict())))
        _assert_close(attempts[0], attempts[1], exact=True)
    finally:
        torch._dynamo.reset()
