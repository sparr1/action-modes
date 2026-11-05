"""Behavior matrix ownership, persistence, and isolated gSDE randomness."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.common import math as td_math
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model


def _model(**overrides):
    options = dict(
        actor_distribution="gsde", inner_rounds=2, inner_updates_per_round=1,
        inner_rollouts_per_round=3, inner_rollout_horizon=2,
        inner_replay_capacity=12, outer_actor_entropy_mode="pre_tanh",
        inner_actor_entropy_mode="pre_tanh", target_entropy=-1.0,
        inner_target_entropy=-1.0,
    )
    options.update(overrides)
    return _tiny_model(**options)


def _record_matrices(engine, monkeypatch):
    records = []
    original = engine._policy_action

    def record(z, policy, **kwargs):
        matrix = kwargs.get("sde_noise")
        if matrix is not None:
            records.append((int(z.shape[0]), matrix.detach().clone()))
        return original(z, policy, **kwargs)

    monkeypatch.setattr(engine, "_policy_action", record)
    return records


@pytest.mark.parametrize("feature", ["actor", "latent"])
@pytest.mark.parametrize("scope", ["round", "solve", "real_steps"])
def test_collection_lifetimes_and_independent_trajectory_draws(feature, scope, monkeypatch):
    model = _model(gsde_feature_source=feature, inner_gsde_noise_scope=scope,
                   inner_gsde_resample_steps=2, gsde_execution_resample_steps=2)
    engine = model.agent.inner_engine
    records = _record_matrices(engine, monkeypatch)
    try:
        global_rng = torch.random.get_rng_state().clone()
        decisions = []
        executions = []
        for index in range(3):
            records.clear()
            model.agent.act(torch.zeros(3), t0=index == 0, collect_diagnostics=False)
            matrices = [matrix for count, matrix in records if count == 3]
            executions.append([matrix for count, matrix in records if count == 1][-1])
            assert len(matrices) == 4
            assert not torch.equal(matrices[0][0], matrices[0][1])
            torch.testing.assert_close(matrices[0], matrices[1], rtol=0, atol=0)
            torch.testing.assert_close(matrices[2], matrices[3], rtol=0, atol=0)
            if scope == "round":
                assert not torch.equal(matrices[0], matrices[2])
            else:
                torch.testing.assert_close(matrices[0], matrices[2], rtol=0, atol=0)
            decisions.append(matrices[0])
        if scope == "real_steps":
            torch.testing.assert_close(decisions[0], decisions[1], rtol=0, atol=0)
        else:
            assert not torch.equal(decisions[0], decisions[1])
        assert not torch.equal(decisions[1], decisions[2])
        torch.testing.assert_close(executions[0], executions[1], rtol=0, atol=0)
        assert not torch.equal(executions[1], executions[2])
        assert not torch.equal(executions[0][0], decisions[0][0])
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        model.env.close()


def test_step_updates_keep_the_matrix_and_use_current_actor(monkeypatch):
    model = _model(inner_rounds=1, inner_rollout_horizon=3, train_unroll_horizon=3,
                   inner_updates_per_round=None, inner_steps_per_update=3,
                   inner_update_timing="step")
    engine = model.agent.inner_engine
    records = _record_matrices(engine, monkeypatch)
    actor_versions = []
    original = engine._rollout_step_kernel

    def record_step(*args, **kwargs):
        actor_versions.append(torch.cat([p.detach().flatten() for p in engine.state.actor.parameters()]))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_rollout_step_kernel", record_step)
    # The eager CompileRegion captured its callable during construction.
    engine._initialize_compile_regions()
    try:
        model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        matrices = [matrix for count, matrix in records if count == 3]
        assert len(matrices) == 3
        for matrix in matrices[1:]:
            torch.testing.assert_close(matrix, matrices[0], rtol=0, atol=0)
        assert len(actor_versions) == 3
        assert not torch.equal(actor_versions[0], actor_versions[1])
        assert model.agent.last_inner_metrics["inner_actor_optimizer_steps"] == 3
    finally:
        model.env.close()


@pytest.mark.parametrize("timing", ["round", "step"])
def test_episodic_compaction_retains_original_matrix_identity(timing, monkeypatch):
    options = dict(episodic=True, inner_rounds=1, inner_rollout_horizon=3,
                   train_unroll_horizon=3)
    if timing == "step":
        options.update(inner_update_timing="step", inner_updates_per_round=None,
                       inner_steps_per_update=3)
    model = _model(**options)
    engine = model.agent.inner_engine
    records = _record_matrices(engine, monkeypatch)
    calls = 0

    def termination(z):
        nonlocal calls
        result = z.new_zeros((z.shape[0], 1))
        if calls == 0:
            result[0] = 1
        calls += 1
        return result

    monkeypatch.setattr(engine.model, "termination", termination)
    try:
        model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        collection = [matrix for count, matrix in records if count != 1]
        assert [matrix.shape[0] for matrix in collection] == [3, 2, 2]
        for matrix in collection[1:]:
            torch.testing.assert_close(matrix, collection[0][1:], rtol=0, atol=0)
    finally:
        model.env.close()


@pytest.mark.parametrize("inactive", ["none", "zero_rounds"])
def test_prior_only_execution_persists_and_mean_eval_uses_no_noise(inactive):
    options = (dict(inner_operator="none", inner_rounds=0, inner_actor_entropy_mode="squashed",
                    inner_rollouts_per_round=0, inner_updates_per_round=0)
               if inactive == "none" else {"inner_rounds": 0})
    model = _model(gsde_execution_resample_steps=2, **options)
    engine = model.agent.inner_engine
    try:
        first = model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        draw = engine._gsde_execution_noise.clone()
        second = model.agent.act(torch.zeros(3), collect_diagnostics=False)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(engine._gsde_execution_noise, draw, rtol=0, atol=0)
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert not torch.equal(engine._gsde_execution_noise, draw)
        state_before_eval = engine.rng.generator("execution").get_state().clone()
        model.agent.act(torch.zeros(3), eval_mode=True, collect_diagnostics=False)
        assert engine._gsde_execution_noise is None
        assert engine._gsde_execution_uses == 0
        torch.testing.assert_close(engine.rng.generator("execution").get_state(),
                                   state_before_eval, rtol=0, atol=0)
        model.agent.act(torch.zeros(3), eval_mode=False, collect_diagnostics=False)
        assert engine._gsde_execution_uses == 1
        assert engine._gsde_execution_noise is not None
    finally:
        model.env.close()


def test_gradient_noise_and_diagnostics_cannot_replace_behavior_matrices(monkeypatch):
    model = _model(inner_gsde_noise_scope="real_steps")
    engine = model.agent.inner_engine
    calls = []
    original = engine.model.pi

    def record(z, **kwargs):
        if kwargs.get("noise") is not None:
            calls.append((kwargs["noise"].clone(), kwargs.get("sde_noise")))
        return original(z, **kwargs)

    monkeypatch.setattr(engine.model, "pi", record)
    try:
        model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        collection, execution = engine._gsde_collection_noise.clone(), engine._gsde_execution_noise.clone()
        assert len(calls) >= 4
        assert all(matrix is None and noise.shape[-1] == model.cfg.action_dim
                   for noise, matrix in calls)
        assert not torch.equal(calls[0][0], calls[1][0])
        model.agent.act(torch.zeros(3), collect_diagnostics=True)
        torch.testing.assert_close(engine._gsde_collection_noise, collection, rtol=0, atol=0)
        torch.testing.assert_close(engine._gsde_execution_noise, execution, rtol=0, atol=0)
    finally:
        model.env.close()


def test_boundary_transport_resets_noise_and_validates_protocol():
    source = _model(inner_gsde_noise_scope="real_steps")
    target = _model(inner_gsde_noise_scope="real_steps")
    mismatch = _model(inner_gsde_noise_scope="solve")
    try:
        engine = source.agent.inner_engine
        source.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        with pytest.raises(RuntimeError, match="gSDE"):
            engine.training_state_dict()
        source.agent.prepare_training_resume_boundary()
        state = deepcopy(engine.training_state_dict())
        target.agent.model.load_state_dict(source.agent.model.state_dict())
        target.agent.inner_engine.load_training_state_dict(state)
        assert target.agent.inner_engine._gsde_execution_noise is None
        assert target.agent.inner_engine._gsde_collection_noise is None
        with pytest.raises(ValueError, match="gSDE"):
            mismatch.agent.inner_engine.load_training_state_dict(state)
        for index in range(2):
            actual = source.agent.act(torch.zeros(3), t0=index == 0, collect_diagnostics=False)
            expected = target.agent.act(torch.zeros(3), t0=index == 0, collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(engine.rng.training_state_dict(),
                           target.agent.inner_engine.rng.training_state_dict())
        target.agent.inner_engine.reset_for_evaluation(29, reuse_action_pool=True)
        assert target.agent.inner_engine._gsde_decision_index == 0
        assert target.agent.inner_engine._gsde_execution_noise is None
        assert target.agent.inner_engine._gsde_collection_noise is None
    finally:
        for model in (source, target, mismatch):
            model.env.close()


@pytest.mark.parametrize("fails", [False, True])
def test_online_evaluation_restores_noise_free_boundary_even_on_failure(monkeypatch, fails):
    from RL.AMBITDMPC2 import AMBITDMPC2

    model = _model(inner_gsde_noise_scope="real_steps")
    engine = model.agent.inner_engine
    try:
        model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        model.agent.prepare_training_resume_boundary()
        before = deepcopy(engine.training_state_dict())

        def evaluation(self, step, *, initial_obs=None):
            self.agent.act(torch.zeros(3), t0=True, eval_mode=True, collect_diagnostics=False)
            assert self.agent.inner_engine._gsde_collection_noise is not None
            assert self.agent.inner_engine._gsde_execution_noise is None
            if fails:
                raise RuntimeError("evaluation failure")
            return 7.0

        monkeypatch.setattr(AMBITDMPC2.__bases__[0], "_evaluate_policy", evaluation)
        if fails:
            with pytest.raises(RuntimeError, match="evaluation failure"):
                model._evaluate_policy(5)
        else:
            assert model._evaluate_policy(5) == 7.0
        assert engine._gsde_collection_noise is None
        assert engine._gsde_execution_noise is None
        assert engine._gsde_decision_index == 0
        _assert_tree_equal(engine.training_state_dict(), before)
    finally:
        model.env.close()


@pytest.mark.parametrize("fails", [False, True])
def test_paired_evaluation_preserves_live_training_matrices(fails):
    from tests.test_ambi_paired_controller_integration import _AuxEnvFactory, _paired_evaluator

    model = _model(inner_gsde_noise_scope="real_steps")
    evaluator = _paired_evaluator(model, _AuxEnvFactory(fail_inner_step=fails))
    engine = model.agent.inner_engine
    try:
        model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        collection = engine._gsde_collection_noise.clone()
        execution = engine._gsde_execution_noise.clone()
        counters = (engine._gsde_decision_index, engine._gsde_execution_uses)
        rng = deepcopy(engine.rng.training_state_dict())
        if fails:
            with pytest.raises(RuntimeError, match="injected auxiliary"):
                evaluator.evaluate()
        else:
            evaluator.evaluate()
        assert model.agent.inner_engine is engine
        torch.testing.assert_close(engine._gsde_collection_noise, collection, rtol=0, atol=0)
        torch.testing.assert_close(engine._gsde_execution_noise, execution, rtol=0, atol=0)
        assert (engine._gsde_decision_index, engine._gsde_execution_uses) == counters
        _assert_tree_equal(engine.rng.training_state_dict(), rng)
    finally:
        evaluator.close()
        model.env.close()


@pytest.mark.parametrize("kernel_name", ["_dense_rollout_kernel", "_rollout_step_kernel"])
def test_explicit_matrices_work_in_fullgraph_tensor_kernels(kernel_name):
    model = _model()
    engine = model.agent.inner_engine
    try:
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        engine.state.actor.eval()
        engine.model.eval()
        root = engine.model.encode(torch.zeros(1, 3)).detach()
        count = model.cfg.inner_rollouts_per_round
        matrix = engine._draw_gsde_noise(root, engine.state.actor, count,
                                        engine.rng.generator("collection"))
        support = td_math.categorical_support(root, model.cfg)
        inputs = (root, root.new_empty((0, count, model.cfg.action_dim)), support)
        if kernel_name == "_rollout_step_kernel":
            inputs = (root.expand(count, -1), root.new_empty(0), support)
        function = getattr(engine, kernel_name)
        compiled = torch.compile(function, backend="eager", fullgraph=True)
        with torch.no_grad():
            expected = function(*inputs, sde_noise=matrix)
            actual = compiled(*inputs, sde_noise=matrix)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    finally:
        model.env.close()
        torch._dynamo.reset()


@pytest.mark.parametrize("timing,feature", [("round", "actor"), ("step", "latent")])
def test_ordinary_compiled_solve_keeps_matrix_lifetimes(timing, feature, monkeypatch):
    torch._dynamo.reset()
    compiler = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: compiler(fn, backend="eager", **kwargs))
    options = dict(compile=True, compile_strict=True, gsde_feature_source=feature,
                   inner_gsde_noise_scope="real_steps", inner_gsde_resample_steps=2)
    if timing == "step":
        options.update(inner_update_timing="step", inner_steps_per_update=3,
                       inner_updates_per_round=None)
    model = _model(**options)
    engine = model.agent.inner_engine
    try:
        model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        matrix = engine._gsde_collection_noise.clone()
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        torch.testing.assert_close(engine._gsde_collection_noise, matrix, rtol=0, atol=0)
        assert model.agent.last_inner_metrics["inner_compile_fallback"] == 0
        assert model.agent.last_inner_metrics["inner_actor_optimizer_steps"] > 0
    finally:
        model.env.close()
        torch._dynamo.reset()
