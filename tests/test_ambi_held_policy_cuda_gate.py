"""Bounded production-shape gate, opt in only on an allocated CUDA GPU.

AMBI_RUN_HELD_POLICY_CUDA_GATE=1 MUJOCO_GL=egl python -m pytest -q -s \
    tests/test_ambi_held_policy_cuda_gate.py

This checks strict CUDA compilation, held decisions and interleaved real outer
updates. It uses a fixed batch of real transitions, not a full training run or
the production warmup. Timings separate cold solves, subsequent solves, held
actions and outer updates; they are smoke observations, not speedup claims.
"""

from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from tests.test_ambi_held_policy_training import HELD_ZERO_WORK
from tests.test_ambi_prior_sac_cuda_gate import _FixedRealReplay
from tests.test_ambi_latency_contract import _assert_tree_equal
from slurm.ambi_transfer_training import binding, runtime, write_new


ROOT = Path(__file__).resolve().parents[1]
HELD_MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_held_training.json"
CASES = (
    ("ambi_full_h3_j8_k3_fresh", 3, 0.),
    ("ambi_full_h3_j8_fresh", 2, 0.),
    ("ambi_full_h3_j8_bernoulli50", 3, .5),
    ("ambi_full_h3_j8_soft", 3, 0.),
    ("ambi_full_h3_j8_soft", 2, .5),
)
CASE_IDS = ("aux-fresh-k3", "aux-fresh-k2", "aux-bernoulli-k3", "soft-fresh-k3", "soft-bernoulli-k2")


def _gate_recipe(name, interval, probability):
    config = json.loads((ROOT / "configs/dmcontrol/algs" / f"{name}.json").read_text())
    config["alg_params"].update(
        wandb=False, inner_solve_interval=interval, inner_critic_bernoulli_p=probability,
    )
    return config


def test_gate_recipes_preserve_full_network_and_optimizer_budgets():
    for name, interval, probability in CASES:
        recipe = _gate_recipe(name, interval, probability)
        params = recipe["alg_params"]
        assert params["model_size"] == 5
        assert params["inner_rollout_horizon"] == 3
        assert params["inner_rounds"] == 8
        assert params["inner_rollouts_per_round"] == 128
        assert params["inner_batch_size"] == 256
        assert params["inner_critic_updates_per_round"] == 16
        assert params["inner_actor_updates_per_round"] == 4
        assert params["compile"] and params["compile_strict"]
        assert params["inner_solve_interval"] == interval
        assert params["inner_critic_bernoulli_p"] == probability
        assert params["wandb"] is False
        if name == "ambi_full_h3_j8_k3_fresh":
            assert params["aux_return_joint_vmap"] is False
            assert params["inner_critic_source"] == params["inner_horizon_critic_source"] == "aux_return"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="held-policy gate requires CUDA")
@pytest.mark.skipif(os.environ.get("AMBI_RUN_HELD_POLICY_CUDA_GATE") != "1",
                    reason="set AMBI_RUN_HELD_POLICY_CUDA_GATE=1 on an allocated GPU")
@pytest.mark.parametrize("name,interval,probability", CASES, ids=CASE_IDS)
def test_production_shape_held_policy_cuda_gate(tmp_path, tmp_path_factory, name, interval, probability):
    pytest.importorskip("dm_control")
    import domains  # noqa: F401: register actual Humanoid Walk adapter
    from torch._dynamo.utils import counters

    config = _gate_recipe(name, interval, probability)
    run = {"name": name, **config}
    experiment = {"env_params": {"task": "humanoid-walk", "obs": "state", "render_mode": None}}
    env = gym.make(run["env"], **experiment["env_params"])
    learner = None
    rows = []
    started = time.perf_counter()
    torch._dynamo.reset()
    counters.clear()
    try:
        learner = AMBITDMPC2(name, env, run["alg_params"], run_params=run,
                            experiment_params=experiment)
        agent, cfg = learner.agent, learner.cfg
        assert agent.device.type == "cuda" and cfg.compile and cfg.compile_strict
        assert (cfg.model_size, cfg.enc_dim, cfg.mlp_dim, cfg.latent_dim, cfg.num_q) == (5, 256, 512, 512, 5)
        assert (cfg.inner_rollout_horizon, cfg.inner_rounds, cfg.action_dim) == (3, 8, 21)
        assert (cfg.train_unroll_horizon, cfg.batch_size) == (3, 256)
        assert cfg.inner_hold_encoder == "solve_snapshot"
        assert cfg.inner_execution_action == "policy_sample"
        observation, _ = env.reset(seed=55)
        env.action_space.seed(55)
        observations, actions, rewards = [observation.copy()], [], []
        for _ in range(32):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
            observations.append(observation.copy())
            actions.append(action.copy())
            rewards.append(reward)
        replay = _FixedRealReplay(observations, actions, rewards, agent.device)
        # Warm the real outer update independently so held-only graph counts
        # cannot be confused with the first capture of outer training.
        initial_update = agent.update(replay)
        torch.cuda.synchronize()
        assert initial_update["compile_fallback"] == 0
        for decision in range(2 * interval + 1):
            graphs_before = counters["stats"]["unique_graphs"]
            torch.cuda.synchronize()
            action_started = time.perf_counter()
            action, behavior = agent.act(
                learner._obs_to_tensor(observation), t0=decision == 0,
                collect_diagnostics=True, return_behavior_policy=True,
            )
            torch.cuda.synchronize()
            action_seconds = time.perf_counter() - action_started
            metrics = deepcopy(agent.last_inner_metrics)
            solve = decision % interval == 0
            assert metrics["inner_solve_performed"] == float(solve)
            assert metrics["inner_policy_held"] == float(not solve)
            assert metrics["inner_action_age"] == decision % interval
            assert all(float(value) == 0 for key, value in metrics.items()
                       if "compile" in key and "fallback" in key)
            assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
            assert torch.isfinite(action).all() and action.abs().max() <= 1
            assert behavior is not None
            assert behavior["pre_tanh_mean"].shape == behavior["log_std"].shape == (21,)
            assert torch.isfinite(behavior["pre_tanh_mean"]).all()
            assert torch.isfinite(behavior["log_std"]).all()
            if solve:
                assert metrics["inner_critic_optimizer_steps"] == 128
                assert metrics["inner_actor_optimizer_steps"] == 32
                assert metrics["inner_temperature_optimizer_steps"] == 32
                assert metrics["inner_model_steps"] == 3072
                if probability:
                    assert metrics["inner_critic_bernoulli_applied"] == float(decision > 0)
            else:
                assert all(metrics[key] == 0 for key in HELD_ZERO_WORK)
                assert metrics["inner_policy_evaluations"] == 1
                assert counters["stats"]["unique_graphs"] == graphs_before, "Held decision captured a graph"
            observation, reward, terminated, truncated, _ = env.step(
                learner._unscale_action(action.numpy())
            )
            assert not terminated and not truncated
            assert np.isfinite(observation).all() and np.isfinite(reward)
            update_started = time.perf_counter()
            update = agent.update(replay)
            torch.cuda.synchronize()
            update_seconds = time.perf_counter() - update_started
            assert agent.num_updates == replay.draws == decision + 2
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in update.values())
            assert update["compile_fallback"] == 0
            rows.append({
                "decision": decision, "solve": solve, "action_seconds": action_seconds,
                "outer_update_seconds": update_seconds,
                "unique_graphs": counters["stats"]["unique_graphs"],
                "outer_updates": agent.num_updates,
                "metrics": {key: float(value) for key, value in metrics.items()},
            })
        assert sum(row["solve"] for row in rows) == 3
        assert learner.buffer.total_transitions == 0, "Gate replay is a bounded fixed real batch"
        # Verify the exact recipe's portable save/load and the agent portion of
        # boundary resume. Full replay/lineage round trips have separate tests.
        agent.prepare_training_resume_boundary()
        checkpoint = tmp_path / "portable.pt"
        expected_checkpoint = deepcopy(agent.checkpoint_state())
        agent.save(checkpoint)
        with torch.no_grad():
            next(agent.model.parameters()).add_(1.)
        agent.load(checkpoint)
        _assert_tree_equal(agent.checkpoint_state(), expected_checkpoint)
        agent.prepare_training_resume_boundary()
        boundary_state = deepcopy(agent.training_state_dict())
        agent.load_training_state_dict(boundary_state)
        _assert_tree_equal(agent.training_state_dict(), boundary_state)
        assert agent.inner_engine._held_actor is agent.inner_engine._held_encoder is None
        torch.cuda.synchronize()
        report = {
            "passed": True, "runtime": runtime(),
            "binding": binding(HELD_MANIFEST) if name == "ambi_full_h3_j8_k3_fresh" else None,
            "config": name, "interval": interval, "bernoulli_probability": probability,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "compile_strict": True, "fallbacks": 0,
            "fixture_random_decisions": 32, "controlled_decisions": len(rows),
            "outer_updates": agent.num_updates, "rows": rows,
            "elapsed_seconds": time.perf_counter() - started,
            "production_warmup_and_pretraining_executed": False,
            "checkpoint_roundtrip": True, "boundary_training_state_roundtrip": True,
            "full_replay_resume_roundtrip": False,
            "timing_scope": "bounded smoke with cold compilation; not a full-training speed estimate",
        }
        report_path = (tmp_path_factory.getbasetemp() / "held-policy-cuda-gate.json"
                       if name == "ambi_full_h3_j8_k3_fresh" else
                       tmp_path / f"{name}-k{interval}-p{probability}-gate.json")
        write_new(report_path, report)
        print("HELD_POLICY_CUDA_GATE_REPORT " + json.dumps(report, sort_keys=True))
    finally:
        env.close()
        if learner is not None:
            learner.close()
        del learner
        gc.collect()
        torch._dynamo.reset()
        torch.cuda.empty_cache()
