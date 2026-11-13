"""BOOM Dog transfer, task dynamics, and scheduled CUDA validation."""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task", ("walk", "trot", "run"))
def test_dog_boom_preserves_recipe_with_dimension_specific_strength(task):
    from slurm.boom_dog_oscar import verify_recipe
    from tests.test_ambi_root_local_sac import _build_cfg
    verify_recipe(task)
    run = json.loads((ROOT / f"configs/dmcontrol/algs/ambi_boom_x0p01_dog_{task}.json").read_text())
    cfg = _build_cfg(**run["alg_params"])
    assert cfg.outer_actor_objective == "boom" and cfg.outer_boom_coef == pytest.approx(.01 * 38 / 1000)
    assert cfg.inner_execution_action == "policy_sample" and cfg.inner_behavior_action == "policy_sample"
    assert cfg.ent_coef == cfg.inner_temperature == .0021
    assert cfg.inner_actor_loss_scale_scope == "action" and cfg.inner_actor_loss_scale_update == "per_update"
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
    assert run["total_steps"] == 1000000 and run["checkpoint_every"] == 25000 and run["save_strat"] == "all"


def test_plan_includes_observed_training_overhead_and_smallest_useful_cpu_count():
    from slurm.boom_dog_oscar import select_plan
    rows = [{"index": 0, "physical_cpus": n, "measured": {"seconds_per_decision": seconds},
             "checkpoint_bytes": 74000000, "peak_host_memory_bytes": 5 * 2**30}
            for n, seconds in [(1, .18), (2, .177)]]
    plan = select_plan(rows, {"end_to_end_seconds_per_decision": .20, "benchmark_seconds_per_decision": .18})
    assert plan["cpus"] == 1 and plan["memory_gib"] == 8
    assert plan["wall_hours"] == 65
    assert plan["projected_end_to_end_seconds_per_decision"] == pytest.approx(.20)


@pytest.mark.skipif(os.environ.get("AMBI_RUN_REAL_DMCONTROL_TESTS") != "1", reason="Opt-in real DMControl check")
@pytest.mark.parametrize("task", ("walk", "trot", "run"))
def test_real_dog_matches_two_control_steps(task):
    import gymnasium as gym
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale
    import domains  # noqa: F401
    env = gym.make("DMControl-v0", task="dog-" + task, obs="state", render_mode=None)
    raw = action_scale.Wrapper(suite.load("dog", task, task_kwargs={"random": 55}, visualize_reward=False), minimum=-1., maximum=1.)
    def flatten(values):
        return np.concatenate([np.asarray(v).reshape(-1) for v in values.values()]).astype(np.float32)
    try:
        observation, _ = env.reset(seed=55)
        np.testing.assert_array_equal(observation, flatten(raw.reset().observation))
        assert observation.shape == (223,) and env.action_space.shape == (38,)
        assert env.unwrapped.action_repeat == 2 and env.spec.max_episode_steps == 500
        generator = np.random.default_rng(55)
        for _ in range(5):
            action = generator.uniform(-1., 1., size=(38,)).astype(env.action_space.dtype)
            reward = 0.
            for repeat in range(2):
                step = raw.step(action); reward += float(step.reward)
            observed, actual, terminated, truncated, _ = env.step(action)
            np.testing.assert_array_equal(observed, flatten(step.observation))
            assert actual == pytest.approx(reward, rel=0, abs=1e-12)
            assert not terminated and not truncated
    finally:
        env.close(); raw.physics.free()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires scheduled CUDA GPU")
def test_cuda_dog_capture_matches_uncaptured(monkeypatch):
    from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
    from tests import test_inner_entropy_graph as graph_tests
    original = graph_tests.make_agent
    def dog_agent(entropy, scope, update, *, device, compiled):
        cfg = deepcopy(original(entropy, scope, update).cfg)
        cfg.action_dim = 38; cfg.action_dims = [38]
        cfg.obs_shape = {"state": (223,)}; cfg.obs_shapes = [{"state": (223,)}]
        cfg.ent_coef = .0021; cfg.inner_temperature = .0021
        cfg.target_entropy = -38.; cfg.inner_target_entropy = -38.
        cfg.device = device; cfg.compile = compiled; cfg.compile_strict = compiled
        cfg.outer_actor_objective = "boom"; cfg.outer_boom_coef = .00038
        cfg.outer_boom_min_std = .05; cfg.outer_boom_compile_mode = "reduce-overhead"
        cfg.store_behavior_policy = True
        agent = AMBITDMPC2Agent(cfg); agent.model.eval(); agent.actor_loss_scale.fill_(4.)
        return agent
    monkeypatch.setattr(graph_tests, "make_agent", dog_agent)
    graph_tests.test_cuda_capture_matches_uncaptured_normalized_solve("squashed", "action", "per_update")
