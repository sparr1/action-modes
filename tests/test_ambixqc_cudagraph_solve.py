"""Verify staged solve RNG, fresh resets and actual one-launch CUDA replay."""

from contextlib import contextmanager
import json
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from test_ambixqc_core import _batch, _tree_equal
from test_ambixqc_prior_checkpoint import _wrapper

PATH = Path(__file__).parent / "benchmarks/ambixqc_cudagraph_solve.py"
SPEC = importlib.util.spec_from_file_location("captured_solve", PATH)
capture_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_module)


def compare_state(a, b, *, atol=1e-6, rtol=1e-5):
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=atol, rtol=rtol, msg=lambda msg: f"{key}: {msg}")


@pytest.mark.parametrize("execution", ["cpu", "cuda-staged", "cuda-graph"])
def test_whole_solve_matches_reference_through_prior_changes(execution):
    cuda = execution != "cpu"
    capture = execution == "cuda-graph"
    if cuda and (not torch.cuda.is_available() or os.getenv("AMBI_RUN_XQC_SOLVE_CUDA_GATE") != "1"):
        pytest.skip("Opt-in CUDA graph gate requires a scheduled CUDA allocation")
    device = "cuda:0" if cuda else "cpu"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    options = dict(device=device, xqc_optimizer_backend="fused" if cuda else "single_tensor",
                   inner_rounds=2, inner_rollouts_per_round=3,
                   inner_updates_per_round=4, inner_policy_delay=2,
                   inner_batch_size=4, inner_replay_capacity=12)
    reference, graphed = _wrapper(**options), _wrapper(**options)
    graphed.agent.load_state_dict(reference.agent.state_dict())
    engine = capture_module.CapturedXQCSolve(graphed.agent, capture=capture)
    graphed.agent.inner_engine = engine
    try:
        for index in range(3):
            if index:
                batch = tuple(t.to(device) for t in _batch(reference.agent))
                reference.agent._update(*batch)
                graphed.agent._update(*batch)
            obs = torch.tensor([0.25 + index, -0.75, 0.5])
            outer = graphed.agent.frozen_outer_state()
            global_cpu = torch.get_rng_state().clone()
            global_cuda = torch.cuda.get_rng_state().clone() if cuda else None
            expected = reference.agent.act(obs, collect_diagnostics=False)
            actual = graphed.agent.act(obs, collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            assert _tree_equal(outer, graphed.agent.frozen_outer_state())
            assert torch.equal(global_cpu, torch.get_rng_state())
            if cuda:
                assert torch.equal(global_cuda, torch.cuda.get_rng_state())
            assert _tree_equal(reference.agent.inner_engine.rng.training_state_dict(), engine.rng.training_state_dict())
            a = reference.agent.inner_engine._workspace_pool
            b = engine._workspace_pool
            compare_state(a.controller.state_dict(), b.controller.state_dict(), atol=2e-6, rtol=2e-5)
            assert (b.update_step, b.actor_optimizer_steps, b.temperature_optimizer_steps) == (8, 4, 4)
            for key in ("inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                        "inner_temperature_optimizer_steps", "inner_model_steps"):
                assert graphed.agent.last_inner_metrics[key] == reference.agent.last_inner_metrics[key]
            for key in ("inner_critic_loss", "inner_actor_loss", "inner_temperature_loss", "inner_alpha_final"):
                assert graphed.agent.last_inner_metrics[key] == pytest.approx(reference.agent.last_inner_metrics[key], rel=2e-5, abs=2e-6)
            assert engine.action_index == index + 1
        if capture:
            assert engine.capture_count == 1 and engine.replay_count == 3
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                graphed.agent.act(obs, collect_diagnostics=False)
                torch.cuda.synchronize()
            launches = sum(event.count for event in prof.key_averages() if event.key == "cudaGraphLaunch")
            assert launches == 1, [(e.key, e.count) for e in prof.key_averages() if "Graph" in e.key]
    finally:
        for model in (reference, graphed):
            model.flush_checkpoints()
            model.agent.inner_engine.clear_all()
            model.env.close()
        torch.set_num_threads(threads)


def test_capture_rejects_unsupported_semantics():
    model = _wrapper(inner_reward_normalization="action_local_imagined")
    try:
        with pytest.raises(ValueError, match="frozen reward scale"):
            capture_module.CapturedXQCSolve(model.agent, capture=False)
    finally:
        model.flush_checkpoints()
        model.env.close()


cuda_gate = pytest.mark.skipif(
    not torch.cuda.is_available() or os.getenv("AMBI_RUN_XQC_SOLVE_CUDA_GATE") != "1",
    reason="Production-shape graph gate requires scheduled CUDA",
)


@contextmanager
def production_pair(*, capture):
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBIXQC import AMBIXQC

    spec = importlib.util.spec_from_file_location("j8_timing_gate", PATH.with_name("ambixqc_j8_training_throughput.py"))
    timing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing)
    run, manifest = timing.recipe("j8", True, 21, 108, 3)
    torch.set_num_threads(1)
    torch._dynamo.reset()
    models = []
    try:
        for name in ("reference", "candidate"):
            env = gym.make(run["env"], **manifest["env_params"])
            models.append(AMBIXQC(name, env, run["alg_params"], run, manifest))
        reference, candidate = models
        candidate.agent.load_state_dict(reference.agent.state_dict())
        engine = capture_module.CapturedXQCSolve(candidate.agent, capture=capture)
        candidate.agent.inner_engine = engine
        observation, _ = reference.env.reset(seed=55)
        yield reference, candidate, engine, reference._obs_to_tensor(observation), timing
    finally:
        for model in models:
            model.flush_checkpoints()
            model.agent.inner_engine.clear_all()
            model.env.close()


def update_matching_priors(reference, candidate):
    batch = tuple(t.to(reference.agent.device) for t in _batch(reference.agent))
    reference.agent._update(*batch)
    candidate.agent._update(*batch)
    # Isolate inner-solve parity from nondeterministic outer reductions. Copy
    # in place, preserving every address already used by the captured graph.
    candidate.agent.load_state_dict(reference.agent.state_dict())


def assert_solve_invariants(reference, candidate, engine, outer, cpu_rng, cuda_rng, timing):
    assert _tree_equal(outer, candidate.agent.frozen_outer_state())
    assert _tree_equal(reference.agent.inner_engine.rng.training_state_dict(), engine.rng.training_state_dict())
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert [candidate.agent.last_inner_metrics[k] for k in timing.WORK_KEYS] == [128, 32, 32, 3072]


@cuda_gate
def test_production_j8_staging_matches_exactly_with_deterministic_kernels():
    # PyTorch 2.3's deterministic scatter fallback synchronizes and cannot be
    # captured. It can still prove exact RNG/input/reset equivalence of the
    # full staged body, independently of normal-kernel numerical variation.
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with production_pair(capture=False) as (reference, candidate, engine, obs, timing):
            for index in range(2):
                if index:
                    update_matching_priors(reference, candidate)
                outer = candidate.agent.frozen_outer_state()
                cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                expected = reference.agent.act(obs + index * 0.01, collect_diagnostics=False)
                actual = candidate.agent.act(obs + index * 0.01, collect_diagnostics=False)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                compare_state(reference.agent.inner_engine._workspace_pool.controller.state_dict(),
                              engine._workspace_pool.controller.state_dict(), atol=0, rtol=0)
                assert_solve_invariants(reference, candidate, engine, outer, cpu_rng, cuda_rng, timing)
            print("PRODUCTION_STAGED_EXACT_GATE_PASSED", flush=True)
    finally:
        torch.use_deterministic_algorithms(deterministic)


def compare_reference_envelope(actual, samples, name):
    """Bound capture error by measured reference repeat variation, per tensor.

    Keep the original 2e-5/2e-4 floor. For nondeterministic tensors allow three
    observed reference ranges; do not replace all state tolerances with one
    permissive bound from a noisy BatchNorm bias. Integer buffers stay exact.
    """
    if not actual.is_floating_point():
        for sample in samples:
            torch.testing.assert_close(actual.cpu(), sample, atol=0, rtol=0)
        return {"spread": 0.0, "error": 0.0}
    stack = torch.stack(samples)
    center = stack.mean(0)
    spread = float((stack.max(0).values - stack.min(0).values).abs().max())
    error = float((actual.cpu() - center).abs().max())
    torch.testing.assert_close(actual.cpu(), center, atol=2e-5 + 3 * spread, rtol=2e-4,
                               msg=lambda msg: f"{name}: reference spread={spread}, capture error={error}: {msg}")
    return {"spread": spread, "error": error}


@cuda_gate
def test_production_j8_compiled_capture_matches_and_launches_once():
    assert not torch.are_deterministic_algorithms_enabled()
    with production_pair(capture=True) as (reference, candidate, engine, obs, timing):
        for index in range(2):
            if index:
                update_matching_priors(reference, candidate)
            outer = candidate.agent.frozen_outer_state()
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            original = reference.agent.inner_engine
            before_rng, before_index = original.rng.training_state_dict(), original.action_index
            actions, states = [], []
            for _ in range(4):
                original.rng.load_training_state_dict(before_rng)
                original.action_index = before_index
                actions.append(reference.agent.act(obs + index * 0.01, collect_diagnostics=False))
                states.append({k: v.detach().cpu().clone()
                               for k, v in original._workspace_pool.controller.state_dict().items()})
            actual = candidate.agent.act(obs + index * 0.01, collect_diagnostics=False)
            action_error = compare_reference_envelope(actual, actions, "action")
            # Also cap both reference variation and capture action error to
            # 0.1% of the normalized action range; wide envelopes must fail.
            assert action_error["spread"] < 0.002 and action_error["error"] < 0.002
            state_errors = {key: compare_reference_envelope(value, [s[key] for s in states], key)
                            for key, value in engine._workspace_pool.controller.state_dict().items()}
            assert_solve_invariants(reference, candidate, engine, outer, cpu_rng, cuda_rng, timing)
            print("PRODUCTION_CAPTURE_PARITY " + json.dumps(
                {"solve": index, "reference_repeats": 4, "action": action_error, "state": state_errors}), flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            candidate.agent.act(obs, collect_diagnostics=False)
            torch.cuda.synchronize()
        launches = sum(e.count for e in prof.key_averages() if e.key == "cudaGraphLaunch")
        assert launches == 1
        assert engine.capture_count == 1 and engine.replay_count == 3
        print("PRODUCTION_GRAPH_GATE_PASSED", engine.capture_status, "cudaGraphLaunch", launches, flush=True)
