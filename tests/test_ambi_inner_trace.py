"""Scientific non-interference and fidelity checks for inner trace recording."""

import json
import math
import os
from pathlib import Path
import statistics
import time

import pytest
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace, metric_catalog, metric_definitions
from tests.test_ambi_latency_contract import _assert_tree_equal, _clone_tree, _pool_snapshot
from tests.test_ambi_root_local_sac import _tiny_component_model, _tiny_model


def _snapshot(agent):
    result = _pool_snapshot(agent)
    pool = agent.inner_engine._action_pool
    result["log_alpha"] = _clone_tree(pool.log_alpha)
    result["temperature_optim"] = _clone_tree(pool.temperature_optim.state_dict())
    result["outer"] = _clone_tree(agent.model.state_dict())
    result["rng"] = _clone_tree(agent.inner_engine.rng.training_state_dict())
    return result


@pytest.mark.parametrize("probes", [False, True])
@pytest.mark.parametrize("adaptation", ["clone", "lora"])
def test_trace_preserves_updates_actions_all_rng_and_modes(probes, adaptation):
    params = dict(inner_rounds=2, inner_critic_updates_per_round=2,
                  inner_actor_updates_per_round=1, q_representation="distributional",
                  num_q=3, dropout=0.2, inner_actor_adaptation=adaptation,
                  inner_critic_adaptation=adaptation,
                  inner_actor_lora_dropout=0.3, inner_critic_lora_dropout=0.3)
    ordinary = _tiny_component_model(**params)
    observed = _tiny_component_model(**params)
    global_rng = torch.random.get_rng_state().clone()
    try:
        for decision in range(2):
            obs = torch.full((3,), decision * 0.1)
            first = ordinary.agent.act(obs, t0=decision == 0, collect_diagnostics=False)
            trace = InnerActionTrace(probes=probes, probe_seed=19, probe_rollouts=2, probe_horizon=2)
            second = observed.agent.act(obs, t0=decision == 0, collect_diagnostics=False, trace=trace)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            _assert_tree_equal(_snapshot(observed.agent), _snapshot(ordinary.agent))
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
            assert [m.training for m in observed.agent.model.modules()] == [m.training for m in ordinary.agent.model.modules()]
            assert observed.agent.inner_engine._active_trace is None
            assert trace._noise is None
            assert all(isinstance(value, float) for event in trace.events for value in event["metrics"].values())
            assert all(math.isfinite(value) for event in trace.events for value in event["metrics"].values())
    finally:
        ordinary.env.close()
        observed.env.close()


def test_trace_records_critic_first_sequence_and_reconstructs_solve_metrics():
    model = _tiny_component_model(inner_rounds=2, inner_critic_updates_per_round=3, inner_actor_updates_per_round=1)
    trace = InnerActionTrace()
    try:
        model.predict([0.1, 0.2, 0.3], deterministic=True, collect_diagnostics=False, trace=trace)
        assert [event["phase"] for event in trace.events] == ["initial"] + ["collection", "update", "update", "update", "update"] * 2
        updates = [event for event in trace.events if event["phase"] == "update"]
        assert [(e["round_index"], e["critic_updates"], e["actor_updates"], e["temperature_updates"]) for e in updates] == [
            (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 3, 1, 1),
            (2, 4, 1, 1), (2, 5, 1, 1), (2, 6, 1, 1), (2, 6, 2, 2),
        ]
        assert [e["event_index"] for e in trace.events] == list(range(len(trace.events)))
        assert all(e["measurement"] == "pre_update_minibatch" for e in updates)
        assert all(e["updated_critic"] != e["updated_actor"] for e in updates)
        reconstructed = model.agent.inner_engine._average_update_metrics([e["metrics"] for e in updates])
        for key, value in reconstructed.items():
            if key.startswith("inner_alpha_used"):
                continue
            assert model.agent.last_inner_metrics[key] == pytest.approx(float(value), abs=1e-6)
        assert "critic_loss" not in updates[3]["metrics"]
        assert "actor_loss" not in updates[0]["metrics"]
        json.dumps(trace.events, allow_nan=False)
        assert "pre-update" in metric_definitions()["critic_loss"]
        catalog = metric_catalog({"additional_metric"})
        assert catalog["critic_loss"]["preferred_axis"] == "critic_updates"
        assert catalog["actor_entropy"]["preferred_axis"] == "actor_updates"
        assert catalog["temperature_loss"]["preferred_axis"] == "temperature_updates"
        assert catalog["fixed_target_q_action_gain"]["preferred_axis"] == "round_index"
        assert "additional_metric" in catalog
    finally:
        model.env.close()


def test_trace_only_adds_no_model_calls_or_rng_forks(monkeypatch):
    model = _tiny_model(inner_rounds=2, inner_updates_per_round=1)
    counts = {}
    for name in ("pi", "pi_action", "policy_stats", "Q", "q_predictions", "next_from_joint", "reward_from_joint"):
        original = getattr(model.agent.model, name)
        def counted(*args, _original=original, _name=name, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(model.agent.model, name, counted)
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        expected = dict(counts)
        counts.clear()
        model.agent.act(torch.zeros(3), collect_diagnostics=False, trace=InnerActionTrace())
        assert counts == expected
    finally:
        model.env.close()


def test_packing_trace_uses_one_host_transfer_and_preserves_other_payloads(monkeypatch):
    model = _tiny_model()
    trace = InnerActionTrace()
    trace.begin()
    trace.record("initial", model.agent.inner_engine.state, {"alpha": torch.tensor(0.2)})
    calls = []
    original = torch.Tensor.cpu
    def counted(tensor, *args, **kwargs):
        calls.append(tensor.numel())
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", counted)
    try:
        action, metrics, lengths, behavior = model.agent._materialize_action_metrics(
            torch.tensor([0.1]), {"loss": torch.tensor(0.5)}, torch.tensor([2, 1]),
            behavior_policy={"pre_tanh_mean": torch.tensor([0.3]), "log_std": torch.tensor([-1.0])}, trace=trace,
        )
        assert calls == [7]
        assert lengths == [2, 1]
        assert action.tolist() == pytest.approx([0.1])
        assert metrics["loss"] == 0.5
        assert behavior["log_std"].tolist() == [-1.0]
        assert trace.events[0]["metrics"]["alpha"] == pytest.approx(0.2)
    finally:
        model.env.close()


def test_none_baseline_has_initial_probe_with_zero_gains():
    model = _tiny_model(inner_operator="none", inner_rounds=0,
                        inner_rollouts_per_round=0, inner_updates_per_round=0,
                        inner_temperature_mode="inherit_outer")
    trace = InnerActionTrace(probes=True, probe_rollouts=2)
    try:
        model.agent.act(torch.zeros(3), eval_mode=True, trace=trace, collect_diagnostics=False)
        assert [event["phase"] for event in trace.events] == ["initial", "probe"]
        probe = trace.events[-1]["metrics"]
        assert probe["probe_model_steps"] == 12
        assert probe["probe_seconds"] >= 0
        assert all(value == 0 for key, value in probe.items() if key.endswith("_gain"))
    finally:
        model.env.close()


def test_probe_failure_restores_modules_and_rng(monkeypatch):
    model = _tiny_model()
    before = torch.random.get_rng_state().clone()
    modes = [module.training for module in model.agent.model.modules()]
    def broken(*args, **kwargs):
        raise RuntimeError("probe failed")
    monkeypatch.setattr(model.agent.model, "policy_stats", broken)
    try:
        with pytest.raises(RuntimeError, match="probe failed"):
            model.agent.act(torch.zeros(3), trace=InnerActionTrace(probes=True))
        assert model.agent.inner_engine._active_trace is None
        assert [module.training for module in model.agent.model.modules()] == modes
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    finally:
        model.env.close()


def test_probe_scores_separate_reward_bootstrap_and_fixed_alpha_entropy(monkeypatch):
    model = _tiny_model()
    engine = model.agent.inner_engine
    trace = InnerActionTrace(probes=True, probe_rollouts=2, probe_horizon=3)
    trace.begin()
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    monkeypatch.setattr(engine.model, "pi", lambda z, **kw: (z.new_zeros(z.shape[0], 1), {"log_prob": z.new_full((z.shape[0], 1), -0.5)}))
    monkeypatch.setattr(engine.model, "reward_from_joint", lambda joint: joint.new_full((joint.shape[0], 1), 2.0))
    monkeypatch.setattr(engine.model, "next_from_joint", lambda joint: joint[:, :model.cfg.latent_dim])
    monkeypatch.setattr(engine.model, "Q", lambda z, action, **kw: z.new_full((z.shape[0], 1), 7.0))
    monkeypatch.setattr("RL.tdmpc2_core.inner_trace.td_math.two_hot_inv", lambda value, cfg: value)
    try:
        trace.probe(engine, torch.zeros(1, model.cfg.latent_dim), engine.state.actor)
        model.agent._materialize_action_metrics(torch.zeros(1), {}, [], trace=trace)
        metrics = trace.events[0]["metrics"]
        discount = float(model.agent.discount)
        reward = 2 * sum(discount ** step for step in range(3))
        bootstrap = 7 * discount ** 3
        entropy = float(model.agent.alpha) * 0.5 * sum(discount ** step for step in range(4))
        assert metrics["discounted_reward_inner"] == pytest.approx(reward)
        assert metrics["discounted_terminal_q_inner"] == pytest.approx(bootstrap)
        assert metrics["fixed_alpha_entropy_bonus_inner"] == pytest.approx(entropy)
        assert metrics["fixed_alpha_soft_score_inner"] == pytest.approx(reward + bootstrap + entropy)
    finally:
        model.env.close()


def test_recorder_reuse_and_invalid_options_fail_early():
    with pytest.raises(ValueError, match="probe_rollouts"):
        InnerActionTrace(probe_rollouts=0)
    with pytest.raises(ValueError, match="probe_seed"):
        InnerActionTrace(probe_seed=True)
    trace = InnerActionTrace()
    trace.begin()
    with pytest.raises(ValueError, match="only one action"):
        trace.begin()


def test_trace_does_not_recompile_or_change_kernel_shapes(monkeypatch):
    model = _tiny_component_model(inner_rounds=2, inner_critic_updates_per_round=2,
                                  inner_actor_updates_per_round=1)
    engine = model.agent.inner_engine
    compilations, shapes = [], []
    def fake_compile(function, **kwargs):
        compilations.append(function.__name__)
        def compiled(*args, **call_kwargs):
            shapes.append((function.__name__, tuple(args[0].shape)))
            return function(*args, **call_kwargs)
        return compiled
    monkeypatch.setattr(torch, "compile", fake_compile)
    engine._compile_regions["critic"].enabled = True
    engine._compile_regions["actor"].enabled = True
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        expected_compilations, expected_shapes = list(compilations), list(shapes)
        shapes.clear()
        model.agent.act(torch.zeros(3), collect_diagnostics=False, trace=InnerActionTrace())
        assert compilations == expected_compilations
        assert shapes == expected_shapes
    finally:
        model.env.close()


def test_root_reset_makes_probes_independent_of_root_order():
    model = _tiny_model(inner_rounds=2, inner_updates_per_round=1)
    def solve(root_id):
        model.agent.inner_engine.reset_for_evaluation(900 + root_id)
        trace = InnerActionTrace(probes=True, probe_seed=800 + root_id, probe_rollouts=2)
        action = model.agent.act(torch.full((3,), root_id * 0.1), t0=True,
                                 collect_diagnostics=False, trace=trace)
        for event in trace.events:
            event["metrics"].pop("probe_seconds", None)
        return action, trace.events
    try:
        forward = {root: solve(root) for root in (1, 2)}
        reverse = {root: solve(root) for root in (2, 1)}
        _assert_tree_equal(forward, reverse)
    finally:
        model.env.close()


@pytest.mark.parametrize("adaptation", ["clone", "lora"])
def test_evaluation_pool_reuse_matches_discarded_scientific_state(adaptation):
    params = dict(inner_rounds=2, inner_critic_updates_per_round=2,
                  inner_actor_updates_per_round=1, inner_actor_adaptation=adaptation,
                  inner_critic_adaptation=adaptation, dropout=0.2,
                  inner_actor_lora_dropout=0.2, inner_critic_lora_dropout=0.2)
    discarded = _tiny_component_model(**params)
    reused = _tiny_component_model(**params)
    try:
        for model in (discarded, reused):
            model.agent.act(torch.zeros(3), collect_diagnostics=False)
        pool = reused.agent.inner_engine._action_pool
        actor_identity, critic_identity = id(pool.actor), id(pool.critic)
        for seed in (909, 910):
            discarded.agent.inner_engine.reset_for_evaluation(seed)
            reused.agent.inner_engine.reset_for_evaluation(seed, reuse_action_pool=True)
            first = discarded.agent.act(torch.ones(3), t0=True, collect_diagnostics=False)
            second = reused.agent.act(torch.ones(3), t0=True, collect_diagnostics=False)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            _assert_tree_equal(_snapshot(reused.agent), _snapshot(discarded.agent))
            assert id(pool.actor) == actor_identity
            assert id(pool.critic) == critic_identity
    finally:
        discarded.env.close()
        reused.env.close()


def test_evaluation_pool_reuse_discards_persistent_components():
    model = _tiny_model(inner_actor_scope="episode")
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        engine = model.agent.inner_engine
        pool, actor = engine._action_pool, engine.state.actor
        engine.reset_for_evaluation(99, reuse_action_pool=True)
        assert engine._action_pool is not pool
        assert engine.state.actor is None
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert engine.state.actor is not actor
        with pytest.raises(TypeError, match="reuse_action_pool"):
            engine.reset_for_evaluation(99, reuse_action_pool=1)
    finally:
        model.env.close()


def test_evaluation_pool_reuse_preserves_real_dynamo_graphs(monkeypatch):
    # Count real Dynamo backend invocations, not just calls to torch.compile.
    # Eager graph execution avoids expensive platform-specific code generation.
    torch._dynamo.reset()
    graphs = []
    original_compile = torch.compile
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    def compile_counted(function, **kwargs):
        return original_compile(function, backend=backend, **kwargs)
    monkeypatch.setattr(torch, "compile", compile_counted)
    model = _tiny_component_model(compile=True, inner_rounds=1,
                                  inner_critic_updates_per_round=1,
                                  inner_actor_updates_per_round=1)
    engine = model.agent.inner_engine
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        initial_count = len(graphs)
        assert initial_count > 0
        for seed in (101, 102):
            engine.reset_for_evaluation(seed, reuse_action_pool=True)
            model.agent.act(torch.zeros(3), collect_diagnostics=False,
                            trace=InnerActionTrace())
            assert len(graphs) == initial_count
    finally:
        model.env.close()
        torch._dynamo.reset()


def _cuda_trace_benchmark_setup():
    """Use the normal checkpoint/sidecar evaluator path when explicitly supplied."""
    checkpoint_value = os.environ.get("AMBI_TRACE_BENCHMARK_CHECKPOINT")
    if checkpoint_value:
        from evaluate_ambi_checkpoint import (
            _close_resources, _file_sha256, _initialize_frozen_model, _jsonable, _make_env,
        )
        from utils.ambi_research import resolve_preset
        from utils.checkpoint_context import load_checkpoint_context

        checkpoint = Path(checkpoint_value).expanduser().resolve()
        matrix = Path(__file__).resolve().parents[1] / "configs/research/ambi_humanoid_inner_benchmark.json"
        context = load_checkpoint_context(checkpoint)
        resolved = resolve_preset(
            matrix, "inner_budget/sac_1x", checkpoint_context=context,
        )
        env = _make_env(resolved)
        model = None
        try:
            model, _ = _initialize_frozen_model(
                resolved, env, checkpoint, controller_seed=55, device="cuda",
            )
            observation, _ = env.reset(seed=101)
            return model, model._obs_to_tensor(observation), {
                "architecture_source": "checkpoint",
                "checkpoint": str(checkpoint), "checkpoint_sha256": _file_sha256(checkpoint),
                "metadata": str(context.source), "metadata_sha256": _file_sha256(context.source),
                "matrix": str(matrix), "matrix_sha256": _file_sha256(matrix),
                "selector": "inner_budget/sac_1x",
                "resolved_config": _jsonable(vars(model.cfg)),
            }
        except BaseException as error:
            _close_resources(model, env, primary_error=error)
            raise

    model = _tiny_component_model(
        device="cuda", compile=True, enc_dim=256, mlp_dim=512, latent_dim=256,
        inner_rounds=8, inner_rollouts_per_round=512, inner_rollout_horizon=3,
        inner_batch_size=512, inner_replay_capacity=12288,
        inner_critic_updates_per_round=3, inner_actor_updates_per_round=1,
    )
    return model, torch.zeros(3), {"architecture_source": "synthetic_proxy"}


def test_actual_checkpoint_benchmark_uses_sidecar_and_research_preset(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import evaluate_ambi_checkpoint as evaluation
    import utils.ambi_research as research
    import utils.checkpoint_context as checkpoint_context

    checkpoint = tmp_path / "checkpoint.pt"
    sidecar = tmp_path / "checkpoint.pt.metadata.json"
    checkpoint.write_bytes(b"checkpoint fixture")
    sidecar.write_text("{}")
    monkeypatch.setenv("AMBI_TRACE_BENCHMARK_CHECKPOINT", str(checkpoint))
    context = SimpleNamespace(source=sidecar)
    calls = {}
    def load_context(path):
        assert path == checkpoint
        return context
    def resolve(matrix, selector, **kwargs):
        assert kwargs["checkpoint_context"] is context
        calls.update(matrix=matrix, selector=selector)
        return {"fixture": True}
    env = SimpleNamespace(reset=lambda seed: ([0.1, 0.2, 0.3], {}))
    model = SimpleNamespace(cfg=SimpleNamespace(latent_dim=256),
                            _obs_to_tensor=torch.as_tensor)
    def initialize(resolved, actual_env, actual_checkpoint, **kwargs):
        assert resolved == {"fixture": True}
        assert actual_env is env and actual_checkpoint == checkpoint
        assert kwargs == {"controller_seed": 55, "device": "cuda"}
        return model, {}
    monkeypatch.setattr(checkpoint_context, "load_checkpoint_context", load_context)
    monkeypatch.setattr(research, "resolve_preset", resolve)
    monkeypatch.setattr(evaluation, "_make_env", lambda resolved: env)
    monkeypatch.setattr(evaluation, "_initialize_frozen_model", initialize)
    actual, observation, provenance = _cuda_trace_benchmark_setup()
    assert actual is model and observation.shape == (3,)
    assert calls["selector"] == "inner_budget/sac_1x"
    assert calls["matrix"].name == "ambi_humanoid_inner_benchmark.json"
    assert provenance["architecture_source"] == "checkpoint"
    assert provenance["metadata"] == str(sidecar)
    assert len(provenance["checkpoint_sha256"]) == 64


@pytest.mark.skipif(
    os.environ.get("AMBI_RUN_TRACE_CUDA_BENCHMARK") != "1" or not torch.cuda.is_available(),
    reason="opt-in CUDA trace overhead measurement",
)
def test_cuda_trace_overhead_measurement(capsys):
    """Measure the <=5% target without a noisy wall-clock CI assertion.

    Set AMBI_RUN_TRACE_CUDA_BENCHMARK=1 and optionally
    AMBI_TRACE_BENCHMARK_CHECKPOINT=/path/to/humanoid.pt (adjacent saved sidecar
    required) and AMBI_TRACE_BENCHMARK_OUTPUT=/tmp/trace-overhead.json. Without
    a checkpoint this explicitly reports a synthetic proxy architecture.
    """
    from evaluate_ambi_checkpoint import _close_resources, _outer_state_digest

    output_value = os.environ.get("AMBI_TRACE_BENCHMARK_OUTPUT")
    output_path = Path(output_value).expanduser().resolve() if output_value else None
    if output_path is not None:
        if output_path.is_relative_to(Path(__file__).resolve().parents[1]):
            raise ValueError("Write benchmark artifacts outside the source repository.")
        if output_path.exists():
            raise FileExistsError(f"Benchmark output already exists: {output_path}")
        if not output_path.parent.is_dir():
            raise FileNotFoundError(f"Benchmark output parent does not exist: {output_path.parent}")

    model, observation, provenance = _cuda_trace_benchmark_setup()
    elapsed, control_elapsed, append_elapsed = ({False: [], True: []} for _ in range(3))
    traces = []
    engine = model.agent.inner_engine
    try:
        digest_before = _outer_state_digest(model)
        warmup_started = time.perf_counter()
        for repeat in range(5):
            engine.reset_for_evaluation(90000 + repeat, reuse_action_pool=True)
            model.agent.act(observation, t0=True, eval_mode=True, collect_diagnostics=False)
        warmup_seconds = time.perf_counter() - warmup_started
        for repeat in range(10):
            for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                engine.reset_for_evaluation(90100 + repeat, reuse_action_pool=True)
                trace = InnerActionTrace() if enabled else None
                started = time.perf_counter()
                model.agent.act(observation, t0=True, eval_mode=True, trace=trace,
                                collect_diagnostics=False)
                control_done = time.perf_counter()
                if trace is not None:
                    traces.append(trace.events)
                finished = time.perf_counter()
                control_elapsed[enabled].append(control_done - started)
                append_elapsed[enabled].append(finished - control_done)
                elapsed[enabled].append(finished - started)
        assert _outer_state_digest(model) == digest_before
        started = time.perf_counter()
        payload = json.dumps(traces, allow_nan=False)
        serialization = time.perf_counter() - started
        ordinary, traced = (statistics.median(elapsed[value]) for value in (False, True))
        result = {
            **provenance, "device": torch.cuda.get_device_name(model.agent.device),
            "torch_version": torch.__version__, "outer_state_unchanged": True,
            "warmup_including_compile_seconds": warmup_seconds,
            "warmup_actions": 5, "paired_repetitions": 10,
            "reuse_action_pool": True, "ordinary_seconds": ordinary,
            "trace_seconds": traced, "trace_overhead_fraction": traced / ordinary - 1,
            "target_overhead_fraction": 0.05,
            "serialization_seconds": serialization, "serialized_bytes": len(payload.encode()),
            "samples": {
                "ordinary_control_seconds": control_elapsed[False],
                "traced_control_seconds": control_elapsed[True],
                "traced_cpu_append_seconds": append_elapsed[True],
                "ordinary_total_seconds": elapsed[False],
                "traced_total_seconds": elapsed[True],
            },
        }
        if output_path is not None:
            with output_path.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        with capsys.disabled():
            print(json.dumps(result, allow_nan=False))
    finally:
        _close_resources(model, model.env)
