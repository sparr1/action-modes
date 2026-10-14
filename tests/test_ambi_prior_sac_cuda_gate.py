"""Opt-in, bounded production-shape CUDA gate; never starts a training run.

Run on an allocated GPU with AMBI_RUN_REAL_DMCONTROL_TESTS=1 and MUJOCO_GL=egl.
The actual study config supplies model/optimizer shapes and compile settings.
Only W&B is disabled. A separate diagnostic fixture compresses bank collection
to 32 consecutive real observations; it does not simulate the production
2,501-decision warmup or 2,500-update pretraining. Two updates use deterministic
repeated real H3 transition slices at the full batch size of 256. The existing
test_ambi_outer_policy_diagnostics.py supplies paired enabled/disabled training
equivalence and strict compilation coverage across the scientific axes.
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
from tests.test_ambi_outer_policy_diagnostics import _equal, _rng_state
from utils.outer_policy_diagnostics import OuterPolicyDiagnostics


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("ambi_prior_sac_clip_target21", "direct_clamp", -21.0),
    ("ambi_prior_sac_smooth_target10p5", "tdmpc2_tanh", -10.5),
)


def _scientific_state(agent):
    return {
        "weights": deepcopy(agent.state_dict()),
        "optimizers": {
            key: deepcopy(getattr(agent, key).state_dict())
            for key in ("optim", "pi_optim", "ent_coef_optim")
        },
        "gradients": [None if p.grad is None else p.grad.clone()
                      for p in agent.parameters()],
        "rng": _rng_state(),
        "inner_rng": deepcopy(agent.inner_engine.rng.training_state_dict()),
        "modes": [module.training for module in agent.modules()],
        "updates": agent.num_updates,
    }


def _unchanged_probe(recorder, agent, **kwargs):
    before = _scientific_state(agent)
    recorder.probe(agent, run=None, **kwargs)
    _equal(_scientific_state(agent), before)


class _FixedRealReplay:
    """One deterministic real-transition batch, without a second replay draw."""

    def __init__(self, observations, actions, rewards, device):
        starts = np.arange(256) % (len(actions) - 2)
        observation_indices = np.arange(4)[:, None] + starts[None, :]
        transition_indices = np.arange(3)[:, None] + starts[None, :]
        self.batch = (
            torch.as_tensor(np.asarray(observations, dtype=np.float32)[observation_indices], device=device),
            # The raw DMControl action spec is float64. Ordinary training stages
            # normalized actions as float32 before adding them to replay.
            torch.as_tensor(np.asarray(actions, dtype=np.float32)[transition_indices], device=device),
            torch.as_tensor(np.asarray(rewards, dtype=np.float32)[transition_indices, None], device=device),
            torch.zeros(3, 256, 1, device=device),
            None,
        )
        self.draws = 0

    def sample(self):
        self.draws += 1
        return self.batch


def test_gate_replay_repeats_complete_real_slices_without_extra_random_draws():
    observations = np.arange(33 * 67, dtype=np.float32).reshape(33, 67)
    actions = np.arange(32 * 21, dtype=np.float64).reshape(32, 21)
    rewards = np.arange(32, dtype=np.float32)
    before = _rng_state()
    replay = _FixedRealReplay(observations, actions, rewards, "cpu")
    obs, act, reward, terminated, task = replay.sample()
    _equal(_rng_state(), before)
    assert obs.shape == (4, 256, 67) and act.shape == (3, 256, 21)
    assert reward.shape == terminated.shape == (3, 256, 1)
    assert obs.dtype == act.dtype == reward.dtype == terminated.dtype == torch.float32
    assert replay.draws == 1 and task is None
    for column in (0, 29, 30, 255):
        start = column % 30
        np.testing.assert_array_equal(obs[:, column], observations[start:start + 4])
        np.testing.assert_array_equal(act[:, column], actions[start:start + 3])
        np.testing.assert_array_equal(reward[:, column, 0], rewards[start:start + 3])
    assert not terminated.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="production-shape gate requires CUDA")
@pytest.mark.skipif(os.environ.get("AMBI_RUN_REAL_DMCONTROL_TESTS") != "1",
                    reason="set AMBI_RUN_REAL_DMCONTROL_TESTS=1 on an allocated GPU")
@pytest.mark.parametrize("name,mapping,target", CASES)
def test_humanoid_production_shape_cuda_gate(tmp_path, name, mapping, target):
    pytest.importorskip("dm_control")
    import domains  # noqa: F401: register the actual adapter

    manifest = json.loads((ROOT / "configs/dmcontrol/experiments/ambi_prior_sac_parameterization_study.json").read_text())
    config = json.loads((ROOT / "configs/dmcontrol/algs" / (name + ".json")).read_text())
    run = {"name": name, **config, **manifest["overrides_alg"]}
    params = {**run["alg_params"], "wandb": False}
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name, env, params, run_params=run, experiment_params=manifest)
        agent, cfg = learner.agent, learner.cfg
        assert cfg.device.startswith("cuda") and cfg.compile and not cfg.compile_strict
        assert (cfg.model_size, cfg.enc_dim, cfg.mlp_dim, cfg.latent_dim, cfg.num_q) == (5, 256, 512, 512, 5)
        assert (cfg.train_unroll_horizon, cfg.batch_size, cfg.action_dim) == (3, 256, 21)
        assert tuple(cfg.obs_shape["state"]) == (67,)
        assert cfg.discount == pytest.approx(.99)
        assert cfg.seed_steps == cfg.pretrain_steps == 2500
        assert cfg.buffer_size == 1_000_000 and cfg.steps == 2_000_000
        assert cfg.inner_operator == "none" and not cfg.mpc
        assert cfg.log_std_mapping == mapping and agent.target_entropy == target
        assert cfg.outer_q_actor_reduction == "mean_pair"
        assert cfg.outer_q_target_reduction == "min_pair"
        assert cfg.outer_critic_target == "entropy_augmented"
        assert cfg.outer_actor_entropy_mode == "squashed"
        assert cfg.sac_actor_loss_scale_mode == "none"
        assert agent.alpha.item() == 1.

        fixture_cfg = deepcopy(cfg)
        fixture_cfg.seed_steps = 31
        fixture_cfg.diagnostic_fixture = "32 consecutive real observations; compressed bank only"
        recorder = OuterPolicyDiagnostics(fixture_cfg, tmp_path / name)
        observation, _ = env.reset(seed=55)
        env.action_space.seed(55)
        assert observation.shape == (67,) and observation.dtype == np.float32
        _unchanged_probe(recorder, agent, observation=observation,
                         env_step=0, updates=0, phase="initialization")
        observations, actions, rewards = [observation.copy()], [], []
        for decision in range(32):
            recorder.observe(observation, decision)
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
            assert np.isfinite(observation).all() and np.isfinite(reward)
            assert 0. <= reward <= 2.
            recorder.action(action, prior=False)
            observations.append(observation.copy())
            actions.append(action.copy())
            rewards.append(reward)
        assert len(recorder.bank) == 32
        _unchanged_probe(recorder, agent, env_step=32, updates=0, phase="pretrain_before")
        replay = _FixedRealReplay(observations, actions, rewards, agent.device)
        compile_status = []
        for completed in (1, 2):
            agent._outer_policy_diagnostics_force = True
            try:
                metrics = agent.update(replay)
            finally:
                agent._outer_policy_diagnostics_force = False
            torch.cuda.synchronize()
            assert agent.num_updates == replay.draws == completed
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
            packet = agent.drain_outer_policy_diagnostics()
            assert packet is not None
            assert (packet["actor_updates_before"], packet["actor_updates_after"]) == (completed - 1, completed)
            torch.testing.assert_close(packet["metrics"]["alpha_after"], agent.alpha.reshape(()))
            torch.testing.assert_close(packet["metrics"]["entropy_shortfall"],
                                       target - packet["metrics"]["entropy_rho_mean"])
            recorder.learner(packet, env_step=32, updates=completed, phase="pretraining", run=None)
            _unchanged_probe(recorder, agent, env_step=32, updates=completed, phase="pretraining")
            bank = [row for row in recorder.rows if row["source"] == "reference_bank"][-1]
            assert bank["observation_count"] == bank["samples_per_state"] == 32
            assert bank["parameter_coordinate_count"] == 32 * 21
            assert bank["sampled_coordinate_count"] == 32 * 32 * 21
            assert sum(bank["histograms"]["log_std_pooled"]["counts"]) == 32 * 21
            compile_status.append({
                "update": completed,
                "any_fallback": bool(metrics["compile_fallback"]),
                "outer_update_fallback": bool(metrics["compile_outer_update_fallback"]),
                "online_critic_fallback": agent.model._Qs.compile_failed,
                "target_critic_fallback": agent.model._target_Qs.compile_failed,
            })

        # Exercise the exact ordinary stochastic no-inner collection hook.
        for decision in range(4):
            action = learner._act_agent(learner._obs_to_tensor(observation), t0=False, eval_mode=False).numpy()
            assert action.shape == (21,) and np.isfinite(action).all()
            assert np.abs(action).max() <= 1.
            observation, reward, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
            assert np.isfinite(observation).all() and 0. <= reward <= 2.
            recorder.action(action, prior=True)
            assert agent.num_updates == 2
        recorder.finish(agent, env_step=36, updates=2, run=None, publish_artifact=False)
        assert {"executed_uniform", "executed_prior", "learner", "reference_bank", "initial_observation"} <= {
            row["source"] for row in recorder.rows
        }
        assert learner.buffer.total_transitions == 0
        report = {
            "config": name, "device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "real_decisions": 36, "optimizer_updates": 2, "replay_draws": replay.draws,
            "model_size": cfg.model_size, "batch_size": cfg.batch_size,
            "compile_requested": cfg.compile, "compile_strict": cfg.compile_strict,
            "compile_status": compile_status, "diagnostic_timing": recorder.timing,
            "elapsed_seconds": time.perf_counter() - started,
            "production_recipe_overrides": {"wandb": False},
            "fixture_only_bank_seed_steps": fixture_cfg.seed_steps,
            "production_warmup_and_pretraining_executed": False,
        }
        (tmp_path / (name + "_gate.json")).write_text(json.dumps(report, indent=2) + "\n")
        print("SAC_PRIOR_CUDA_GATE_REPORT " + json.dumps(report, sort_keys=True))
    finally:
        env.close()
        if learner is not None:
            learner.agent._outer_policy_diagnostics_force = False
        del learner
        gc.collect()
        torch.cuda.empty_cache()
