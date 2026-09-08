"""Causal ordering and scientific invariants of interleaved inner SAC."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace
from RL.tdmpc2_core.common import math as td_math
from tests.test_ambi_latency_contract import _assert_tree_equal, _clone_tree, _pool_snapshot
from tests.test_ambi_outer_replay import _episode
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_model
from utils.resume_identity import scientific_trial_parameters


def _snapshot(agent):
    result = _pool_snapshot(agent)
    pool = agent.inner_engine._action_pool
    result["log_alpha"] = _clone_tree(pool.log_alpha)
    result["temperature_optim"] = _clone_tree(pool.temperature_optim.state_dict())
    result["outer"] = _clone_tree(agent.model.state_dict())
    result["rng"] = _clone_tree(agent.inner_engine.rng.training_state_dict())
    return result


def _step_model(**overrides):
    options = dict(inner_update_timing="step", inner_steps_per_update=2,
                   inner_updates_per_round=None)
    options.update(overrides)
    return _tiny_model(**options)


@pytest.mark.parametrize("value", [None, True, 1, "", "episode"])
def test_invalid_update_timing_is_rejected(value):
    with pytest.raises(ValueError, match="inner_update_timing"):
        _build_cfg(inner_update_timing=value)


def test_step_timing_requires_explicit_interval():
    with pytest.raises(ValueError, match="explicit inner_steps_per_update"):
        _build_cfg(inner_update_timing="step")


@pytest.mark.parametrize("mode", ["frozen_random", "shared_mixture", "separate_critics", "adaptive_param_noise"])
def test_step_timing_rejects_unsupported_populations(mode):
    with pytest.raises(ValueError, match="inner_explorer_mode='none'"):
        _build_cfg(inner_update_timing="step", inner_steps_per_update=2,
                   inner_explorer_mode=mode)


@pytest.mark.parametrize("operator", ["none", "mppi", "td3"])
def test_step_timing_requires_sac(operator):
    with pytest.raises(ValueError, match="inner_update_timing"):
        _build_cfg(inner_update_timing="step", inner_steps_per_update=2,
                   inner_operator=operator)


@pytest.mark.parametrize("interval,expected", [
    (2, [1, 1, 1, 1, 1, 1]),
    (5, [0, 0, 1, 0, 1, 0]),
    (2.5, [0, 1, 1, 1, 1, 0]),
    (.5, [4, 4, 4, 4, 4, 4]),
])
def test_interval_credit_carries_across_steps_and_rounds_but_resets_per_action(interval, expected, monkeypatch):
    model = _step_model(inner_steps_per_update=interval)
    engine = model.agent.inner_engine
    counts = []
    original = engine._run_update_counts

    def record(**kwargs):
        counts.append((engine.state.replay.next_sample_id, kwargs["critic_count"],
                       kwargs["actor_count"], kwargs["temperature_count"]))
        return original(**kwargs)

    monkeypatch.setattr(engine, "_run_update_counts", record)
    try:
        for _ in range(2):
            trace = InnerActionTrace()
            model.agent.act(torch.zeros(3), collect_diagnostics=False, trace=trace)
            assert counts[-6:] == [(2 * (i + 1), n, n, n) for i, n in enumerate(expected)]
            phases = ["initial"]
            for n in expected:
                phases.extend(["collection"] + ["update"] * n)
            assert [e["phase"] for e in trace.events] == phases
            collections = [e for e in trace.events if e["phase"] == "collection"]
            assert [e["metrics"]["collection_rollout_step"] for e in collections] == [1, 2] * 3
            assert model.agent.last_inner_rollout_lengths == [2] * 6
            assert model.agent.last_inner_metrics["inner_model_steps"] == 12
            assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] == sum(expected)
            assert model.agent.last_inner_metrics["inner_requested_update_slots"] == sum(expected)
        assert model.cfg.inner_expected_update_slots == sum(expected)
    finally:
        model.env.close()


def test_next_imagined_action_uses_updated_actor_and_continues_same_branch(monkeypatch):
    model = _step_model(inner_rounds=2)
    engine = model.agent.inner_engine
    calls = []
    original = engine._policy_action

    def record(z, actor, **kwargs):
        result = original(z, actor, **kwargs)
        if z.shape[0] == 2:  # Collection; root action execution has one row.
            calls.append((engine.state.actor_steps, z.clone(), result[0].clone(),
                          [p.detach().clone() for p in actor.parameters()]))
        return result

    monkeypatch.setattr(engine, "_policy_action", record)
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        replay = engine._action_pool.replay
        assert [entry[0] for entry in calls] == [0, 1, 2, 3]
        assert any(not torch.equal(a, b) for a, b in zip(calls[0][3], calls[1][3]))
        torch.testing.assert_close(replay.z[2:4], replay.next_z[:2], rtol=0, atol=0)
        torch.testing.assert_close(replay.z[6:8], replay.next_z[4:6], rtol=0, atol=0)
        torch.testing.assert_close(calls[2][1], calls[0][1], rtol=0, atol=0)
        torch.testing.assert_close(replay.action[:8], torch.cat([c[2] for c in calls]), rtol=0, atol=0)
    finally:
        model.env.close()


def test_termination_compaction_and_horizon_flags_survive_interleaving(monkeypatch):
    model = _step_model(episodic=True, inner_rounds=2, inner_rollout_horizon=3,
                        train_unroll_horizon=3, inner_finite_horizon=True)
    engine = model.agent.inner_engine

    def terminate_one_branch(z):
        done = z.new_zeros((z.shape[0], 1))
        if z.shape[0] == 2:
            done[0] = 1
        return done

    monkeypatch.setattr(engine.model, "termination", terminate_one_branch)
    try:
        trace = InnerActionTrace()
        model.agent.act(torch.zeros(3), collect_diagnostics=False, trace=trace)
        assert model.agent.last_inner_rollout_lengths == [1, 3, 1, 3]
        assert model.agent.last_inner_metrics["inner_model_steps"] == 8
        assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] == 4
        replay = engine._action_pool.replay
        assert replay.terminated[:8, 0].tolist() == [1, 0, 0, 0] * 2
        assert replay.horizon_end[:8, 0].tolist() == [0, 0, 0, 1] * 2
        collections = [e for e in trace.events if e["phase"] == "collection"]
        assert [e["metrics"]["collection_transitions"] for e in collections] == [2, 1, 1] * 2
    finally:
        model.env.close()


def test_fully_terminated_rounds_stop_collecting_but_keep_earned_updates(monkeypatch):
    model = _step_model(episodic=True)
    monkeypatch.setattr(model.agent.model, "termination", lambda z: z.new_ones((z.shape[0], 1)))
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert model.agent.last_inner_rollout_lengths == [1] * 6
        assert model.agent.last_inner_metrics["inner_model_steps"] == 6
        assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] == 3
    finally:
        model.env.close()


def test_replay_wraparound_does_not_reset_transition_credit():
    model = _step_model(inner_steps_per_update=3, inner_replay_capacity=3,
                        inner_replay_scope="run")
    try:
        for decision in range(2):
            model.agent.act(torch.zeros(3), collect_diagnostics=False)
            replay = model.agent.inner_engine.state.replay
            assert replay.size == 3
            assert replay.next_sample_id == (decision + 1) * 12
            assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] == 4
    finally:
        model.env.close()


def test_target_cadence_and_fixed_temperature_cross_collection_boundaries():
    model = _step_model(inner_critic_target_update_interval=4,
                        inner_temperature_mode="inherit_outer")
    try:
        for _ in range(2):
            model.agent.act(torch.zeros(3), collect_diagnostics=False)
            metrics = model.agent.last_inner_metrics
            assert metrics["inner_critic_optimizer_steps"] == 6
            assert metrics["inner_actor_optimizer_steps"] == 6
            assert metrics["inner_critic_target_updates"] == 1
            assert metrics["inner_temperature_optimizer_steps"] == 0
    finally:
        model.env.close()


def test_without_replacement_validates_first_eligible_step():
    with pytest.raises(ValueError, match="before the first update"):
        _build_cfg(inner_update_timing="step", inner_steps_per_update=2,
                   inner_rollouts_per_round=2, inner_batch_size=4,
                   inner_replay_sampling="without_replacement")
    model = _step_model(inner_steps_per_update=3.5, inner_replay_sampling="without_replacement")
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] == 3
    finally:
        model.env.close()


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_step_tracing_preserves_rng_updates_and_outer_state(adaptation):
    options = dict(inner_actor_adaptation="clone", inner_critic_adaptation=adaptation,
                   dropout=.2, inner_critic_lora_rank=4)
    ordinary, traced = _step_model(**options), _step_model(**options)
    outer = deepcopy(ordinary.agent.model.state_dict())
    rng = torch.random.get_rng_state().clone()
    try:
        for _ in range(2):
            first = ordinary.agent.act(torch.zeros(3), collect_diagnostics=False)
            second = traced.agent.act(torch.zeros(3), collect_diagnostics=False,
                                      trace=InnerActionTrace(probes=True, probe_rollouts=2))
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            _assert_tree_equal(_snapshot(ordinary.agent), _snapshot(traced.agent))
            _assert_tree_equal(ordinary.agent.model.state_dict(), outer)
            torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
            assert all(p.grad is None for p in ordinary.agent.model.parameters())
    finally:
        ordinary.env.close()
        traced.env.close()


def test_update_timing_has_backward_compatible_identity_and_exact_resume_checks():
    base = {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": {}}
    explicit = {**base, "alg_params": {"inner_update_timing": "round"}}
    assert scientific_trial_parameters(base) == scientific_trial_parameters(explicit)
    assert "inner_update_timing" not in scientific_trial_parameters(explicit)["alg_params"]
    assert scientific_trial_parameters(base) != scientific_trial_parameters(
        {**base, "alg_params": {"inner_update_timing": "step"}})
    source, matching = _step_model(), _step_model()
    target = _tiny_model(inner_steps_per_update=2, inner_updates_per_round=None)
    try:
        source.agent.prepare_training_resume_boundary()
        saved = deepcopy(source.agent.training_state_dict())
        assert saved["outer"]["critic_target_spec"]["inner_solve"]["update_timing"] == "step"
        assert "update_timing" not in target.agent._critic_target_spec()["inner_solve"]
        matching.agent.load_training_state_dict(saved)
        before = deepcopy(target.agent.model.state_dict())
        with pytest.raises(ValueError, match="critic-target specification"):
            target.agent.load_training_state_dict(saved)
        _assert_tree_equal(target.agent.model.state_dict(), before)
    finally:
        source.env.close()
        matching.env.close()
        target.env.close()


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_step_graphs_capture_and_reuse_with_horizon_and_real_replay(monkeypatch, representation):
    torch._dynamo.reset()
    graphs = []
    real_compile = torch.compile

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    monkeypatch.setattr(torch, "compile", lambda function, **kw: real_compile(function, backend=backend, **kw))
    model = _step_model(compile=True, compile_strict=True, inner_finite_horizon=True,
                        q_representation=representation, inner_outer_replay_fraction=.5)
    try:
        model.buffer.add(_episode())
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        count = len(graphs)
        assert count > 0
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert len(graphs) == count
        assert model.agent.last_inner_metrics["inner_compile_fallback"] == 0
        assert model.agent.last_inner_metrics["inner_outer_replay_samples"] == 12
        assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] == 6
    finally:
        model.env.close()
        torch._dynamo.reset()


def test_inductor_parallel_step_matches_eager():
    torch._dynamo.reset()
    model = _step_model(compile=True, compile_strict=True)
    engine = model.agent.inner_engine
    try:
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        engine.state.actor.eval()
        engine.model.eval()
        generator = torch.Generator().manual_seed(17)
        z = torch.rand(2, model.cfg.latent_dim, generator=generator)
        noise = torch.randn(2, model.cfg.action_dim, generator=generator)
        support = td_math.categorical_support(z, model.cfg)
        compiled = torch.compile(engine._rollout_step_kernel, backend="inductor", fullgraph=True)
        with torch.no_grad():
            expected = engine._rollout_step_kernel(z, noise, support)
            actual = compiled(z, noise, support)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=3e-5, atol=3e-6)
    finally:
        model.env.close()
        torch._dynamo.reset()

def test_checkpoint_weights_load_for_stepwise_prediction(tmp_path):
    source = _tiny_model(inner_rounds=2, inner_updates_per_round=1)
    adapted = _step_model(inner_rounds=2)
    try:
        checkpoint = tmp_path / "round-prior.pt"
        source.agent.save(str(checkpoint))
        adapted.agent.load(str(checkpoint))
        outer = deepcopy(adapted.agent.model.state_dict())
        trace = InnerActionTrace()
        adapted.predict([0., 0., 0.], deterministic=True,
                        collect_diagnostics=False, trace=trace)
        _assert_tree_equal(adapted.agent.model.state_dict(), outer)
        assert [event["phase"] for event in trace.events] == ["initial"] + ["collection", "update"] * 4
    finally:
        source.env.close()
        adapted.env.close()
