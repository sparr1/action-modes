"""Verify staged solve RNG, fresh resets and actual one-launch CUDA replay."""

from copy import deepcopy
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


@pytest.mark.skipif(not torch.cuda.is_available() or os.getenv("AMBI_RUN_XQC_SOLVE_CUDA_GATE") != "1",
                    reason="Production-shape graph gate requires scheduled CUDA")
def test_production_j8_compiled_capture_matches_and_launches_once():
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBIXQC import AMBIXQC

    spec = importlib.util.spec_from_file_location("j8_timing_gate", PATH.with_name("ambixqc_j8_training_throughput.py"))
    timing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing)
    run, manifest = timing.recipe("j8", True, 21, 108, 3)
    torch.set_num_threads(1)
    models = []
    try:
        for name in ("reference", "captured"):
            env = gym.make(run["env"], **manifest["env_params"])
            models.append(AMBIXQC(name, env, run["alg_params"], run, manifest))
        reference, graphed = models
        graphed.agent.load_state_dict(reference.agent.state_dict())
        engine = capture_module.CapturedXQCSolve(graphed.agent)
        graphed.agent.inner_engine = engine
        observation, _ = reference.env.reset(seed=55)
        obs = reference._obs_to_tensor(observation)
        for index in range(2):
            if index:
                batch = tuple(t.to(reference.agent.device) for t in _batch(reference.agent))
                reference.agent._update(*batch)
                graphed.agent._update(*batch)
            outer = graphed.agent.frozen_outer_state()
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            if index == 0:
                before_reference_rng = reference.agent.inner_engine.rng.training_state_dict()
            expected = reference.agent.act(obs, collect_diagnostics=False)
            if index == 0:
                expected_state = deepcopy(reference.agent.inner_engine._workspace_pool.controller.state_dict())
                after_reference_rng = reference.agent.inner_engine.rng.training_state_dict()
                reference.agent.inner_engine.rng.load_training_state_dict(before_reference_rng)
                repeat = reference.agent.act(obs, collect_diagnostics=False)
                print("REFERENCE_REPEAT_ACTION_MAX", float((repeat-expected).abs().max()), flush=True)
                print("REFERENCE_REPEAT_STATE_MAX", max(float((value-reference.agent.inner_engine._workspace_pool.controller.state_dict()[key]).abs().max())
                      for key,value in expected_state.items() if value.is_floating_point()), flush=True)
                reference.agent.inner_engine.rng.load_training_state_dict(after_reference_rng)
                staged = capture_module.CapturedXQCSolve(graphed.agent, capture=False)
                graphed.agent.inner_engine = staged
                staged_action = graphed.agent.act(obs, collect_diagnostics=False)
                print("STAGED_ACTION_MAX", float((staged_action-expected).abs().max()), flush=True)
                print("STAGED_STATE_MAX", max(float((value-staged._workspace_pool.controller.state_dict()[key]).abs().max())
                      for key,value in expected_state.items() if value.is_floating_point()), flush=True)
                print("STAGED_RNG_EQUAL", _tree_equal(after_reference_rng, staged.rng.training_state_dict()), flush=True)
                graphed.agent.inner_engine = engine
            actual = graphed.agent.act(obs, collect_diagnostics=False)
            print("CAPTURE_ACTION_MAX", index, float((actual-expected).abs().max()), flush=True)
            print("CAPTURE_RNG_EQUAL", _tree_equal(reference.agent.inner_engine.rng.training_state_dict(), engine.rng.training_state_dict()), flush=True)
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            compare_state(reference.agent.inner_engine._workspace_pool.controller.state_dict(),
                          engine._workspace_pool.controller.state_dict(), atol=2e-5, rtol=2e-4)
            assert _tree_equal(outer, graphed.agent.frozen_outer_state())
            assert _tree_equal(reference.agent.inner_engine.rng.training_state_dict(), engine.rng.training_state_dict())
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
            assert [graphed.agent.last_inner_metrics[k] for k in timing.WORK_KEYS] == [128,32,32,3072]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            graphed.agent.act(obs, collect_diagnostics=False)
            torch.cuda.synchronize()
        launches = sum(e.count for e in prof.key_averages() if e.key == "cudaGraphLaunch")
        assert launches == 1
        assert engine.capture_count == 1 and engine.replay_count == 3
        print("PRODUCTION_GRAPH_GATE_PASSED", engine.capture_status, "cudaGraphLaunch", launches, flush=True)
    finally:
        for model in models:
            model.flush_checkpoints()
            model.agent.inner_engine.clear_all()
            model.env.close()
