"""Opt-in real-Humanoid CUDA validation for the auxiliary-return SAC campaign."""

from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from slurm import ambi_aux_return_sac_campaign as campaign
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_prior_sac_cuda_gate import _FixedRealReplay


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / campaign.MANIFEST
CASES = campaign.CASES
pytestmark = pytest.mark.skipif(
    os.environ.get("AMBI_RUN_AUX_RETURN_SAC_CUDA_GATE") != "1",
    reason="set AMBI_RUN_AUX_RETURN_SAC_CUDA_GATE=1 on an allocated CUDA GPU",
)


def _recipe(name):
    manifest = json.loads(MANIFEST.read_text())
    config_path = ROOT / "configs/dmcontrol/algs" / (name + ".json")
    config = json.loads(config_path.read_text())
    run = {"name": name, **config, **manifest["overrides_alg"]}
    return manifest, run, hashlib.sha256(config_path.read_bytes()).hexdigest()


def _flags(agent, metrics):
    return {
        "aggregate": bool(metrics["compile_fallback"]),
        "outer_update": bool(metrics["compile_outer_update_fallback"]),
        "sac_online": bool(agent.model._Qs.compile_failed),
        "sac_target": bool(agent.model._target_Qs.compile_failed),
        "aux_online": bool(agent.model._aux_return_Qs.compile_failed),
        "aux_target": bool(agent.model._target_aux_return_Qs.compile_failed),
    }


def _report(name, payload):
    root = Path(os.environ["AMBI_AUX_GATE_OUTPUT_ROOT"])
    with (root / (name + ".json")).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("AUX_RETURN_SAC_GATE_REPORT " + json.dumps(payload, sort_keys=True))


@pytest.mark.parametrize("name", CASES)
def test_production_shape_cold_cuda_updates_checkpoint_and_gradient_routing(tmp_path, name):
    assert torch.cuda.is_available(), "Requested CUDA gate requires a GPU."
    import domains  # noqa: F401

    manifest, run, config_sha256 = _recipe(name)
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name, env, {**run["alg_params"], "wandb": False},
                            run_params=run, experiment_params=manifest)
        agent, cfg = learner.agent, learner.cfg
        assert cfg.compile and cfg.compile_strict and cfg.aux_return_mode == "sac"
        assert cfg.critic_value_mode == "single" and cfg.inner_operator == "none"
        assert not hasattr(agent.model, "_return_pi") and agent.aux_return.pi_optim is None
        assert not agent.aux_return.has_actor
        assert agent.aux_return.ent_coef_optim is None and agent.aux_return.log_ent_coef is None
        assert cfg.inner_actor_source == "sac" and cfg.inner_execution_action == "policy_sample"
        assert cfg.inner_execution_std_scale == 1.
        assert not agent.actor_loss_scale_enabled and not agent.aux_return.actor_loss_scale_enabled
        assert (cfg.enc_dim, cfg.mlp_dim, cfg.latent_dim, cfg.num_q) == (256, 512, 512, 5)
        assert (cfg.train_unroll_horizon, cfg.batch_size, cfg.action_dim) == (3, 256, 21)
        assert tuple(cfg.obs_shape["state"]) == (67,)
        assert cfg.seed_steps == cfg.pretrain_steps == 2500 and cfg.steps == run["total_steps"]
        assert cfg.critic_coef == cfg.aux_return_critic_coef == .1
        assert cfg.actor_lr == cfg.critic_lr == cfg.aux_return_critic_lr == 3e-4
        assert cfg.outer_q_actor_reduction == run["alg_params"]["outer_q_actor_reduction"]
        assert cfg.outer_q_target_reduction == agent.aux_return.cfg.outer_q_target_reduction == "min_pair"
        assert cfg.log_std_mapping == "direct_clamp"
        assert cfg.log_std_min == run["alg_params"]["log_std_min"] and cfg.log_std_max == 2
        assert agent.alpha.item() == 1. and agent.ent_coef_optim is not None
        assert agent.target_entropy == run["alg_params"]["target_entropy"]

        # Keep all model/Q kernels cold until the first full production-shape update.
        observation, _ = env.reset(seed=55)
        env.action_space.seed(55)
        observations, actions, rewards = [observation.copy()], [], []
        for _ in range(32):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)
            assert not terminated and not truncated
            assert np.isfinite(observation).all() and np.isfinite(reward)
            observations.append(observation.copy())
            actions.append(action.copy())
            rewards.append(reward)
        replay = _FixedRealReplay(observations, actions, rewards, agent.device)
        replay_digest = hashlib.sha256()
        for tensor in replay.batch[:-1]:
            array = tensor.detach().cpu().contiguous().numpy()
            replay_digest.update(str(array.dtype).encode())
            replay_digest.update(str(array.shape).encode())
            replay_digest.update(array.tobytes())
        compile_status = []
        update_seconds = []
        for completed in (1, 2):
            agent._outer_policy_diagnostics_force = True
            update_started = time.perf_counter()
            try:
                metrics = agent.update(replay)
            finally:
                agent._outer_policy_diagnostics_force = False
            torch.cuda.synchronize()
            update_seconds.append(time.perf_counter() - update_started)
            assert agent.num_updates == replay.draws == completed
            assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
            assert "aux_return_critic_loss" in metrics and "aux_return_q_target_clip_fraction" in metrics
            flags = _flags(agent, metrics)
            assert not any(flags.values()), flags
            compile_status.append(flags)
            assert agent._outer_update_region._compiled is not None
            assert agent.drain_outer_policy_diagnostics() is not None

        # After learning opens the initially zero output weights, isolate only
        # the auxiliary loss: attached gradients must reach encoder AND dynamics.
        obs, action, reward, _, _ = replay.batch
        z = agent.model.encode(obs[0])
        zs = [z]
        for a in action:
            z = agent.model.next(z, a)
            zs.append(z)
        loss, _ = agent.aux_return.critic_loss(torch.stack(zs), action, reward)
        groups = (agent.model._encoder, agent.model._dynamics, agent.model._aux_return_Qs)
        parameters = [p for group in groups for p in group.parameters()]
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        norms, offset = [], 0
        for group in groups:
            count = len(list(group.parameters()))
            norms.append(sum(g.abs().sum().item() for g in gradients[offset:offset+count] if g is not None))
            offset += count
        assert (norms[0] > 0) is (not cfg.aux_return_detach_representation)
        assert (norms[1] > 0) is (not cfg.aux_return_detach_representation)
        assert norms[2] > 0

        saved = deepcopy(agent.checkpoint_state())
        assert saved["aux_return_spec"]["mode"] == "sac"
        checkpoint = tmp_path / (name + ".pt")
        torch.save(saved, checkpoint)
        with torch.no_grad():
            agent.model._aux_return_Qs[0][-1].bias.add_(1.)
        agent.load(checkpoint)
        _assert_tree_equal(agent.checkpoint_state(), saved)

        # The private auxiliary RNG and optimizer state must resume the same
        # stochastic learner, not merely reload the same initial weights.
        agent.prepare_training_resume_boundary()
        boundary = deepcopy(agent.training_state_dict())
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(agent.device).clone()
        next_metrics = agent.update(replay)
        torch.cuda.synchronize()
        expected = deepcopy(agent.training_state_dict())
        expected_cpu_rng = torch.random.get_rng_state().clone()
        expected_cuda_rng = torch.cuda.get_rng_state(agent.device).clone()
        agent.load_training_state_dict(boundary)
        _assert_tree_equal(agent.training_state_dict(), boundary)
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, agent.device)
        resumed_metrics = agent.update(replay)
        torch.cuda.synchronize()
        _assert_tree_equal(agent.training_state_dict(), expected)
        assert torch.equal(torch.random.get_rng_state(), expected_cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(agent.device), expected_cuda_rng)
        for row in (next_metrics, resumed_metrics):
            assert all(torch.isfinite(torch.as_tensor(v)).all() for v in row.values())
            flags = _flags(agent, row)
            assert not any(flags.values()), flags
            compile_status.append(flags)
        assert agent.num_updates == agent.aux_return.num_updates == 3

        # Repeated observations should sample the SAC actor without adapting any
        # actor/critic, drawing model rollouts, or advancing outer learner state.
        updates_before_collection = agent.num_updates
        collected = []
        for _ in range(2):
            result = learner._act_agent(learner._obs_to_tensor(observation), t0=False, eval_mode=False)
            assert result.shape == (21,) and torch.isfinite(result).all() and result.abs().max() <= 1.
            for key in (
                "inner_active", "inner_updates", "inner_total_model_steps",
                "inner_actor_optimizer_steps", "inner_critic_optimizer_steps",
                "inner_temperature_optimizer_steps",
            ):
                assert agent.last_inner_metrics[key] == 0., key
            collected.append(result.clone())
        assert not torch.equal(collected[0], collected[1])
        assert agent.num_updates == agent.aux_return.num_updates == updates_before_collection
        _report(name, {
            "config": name, "config_sha256": config_sha256, "passed": True,
            "replay_seed": 55, "replay_sha256": replay_digest.hexdigest(),
            "source_commit": os.environ["EXPECTED_ACTION_MODES_SHA"],
            "compile_strict": True, "compile_status": compile_status,
            "checkpoint_roundtrip": True, "exact_checkpoint_roundtrip": True,
            "next_update_reproducible": True, "gradient_routing": True,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "auxiliary_gradient_l1": dict(zip(("encoder", "dynamics", "critic"), norms)),
            "detach_representation": cfg.aux_return_detach_representation,
            "no_return_actor": True, "no_inner_updates": True,
            "optimizer_updates": agent.num_updates, "real_decisions": 32,
            "cold_production_shape_updates": 2, "continuation_validation_executions": 2,
            "stochastic_sac_collection": True,
            "device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "elapsed_seconds": time.perf_counter() - started,
            "cold_update_seconds": update_seconds[0], "warm_update_seconds": update_seconds[1],
            "production_overrides": {"wandb": False},
            "production_warmup_and_pretraining_executed": False,
        })
    finally:
        if learner is not None:
            learner.close()
        env.close()
        del learner
        gc.collect()
        torch.cuda.empty_cache()


def test_short_fresh_humanoid_training_and_scheduled_checkpoint_roundtrip(tmp_path):
    assert torch.cuda.is_available(), "Requested CUDA gate requires a GPU."
    import domains  # noqa: F401

    name = CASES[0]
    manifest, run, _ = _recipe(name)
    overrides = {"wandb": False, "seed_steps": 500, "pretrain_steps": 2, "buffer_size": 4096}
    params = {**run["alg_params"], **overrides}
    run = {**run, "total_steps": 512, "alg_params": params}
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    try:
        learner = AMBITDMPC2(name + "-fresh-smoke", env, params, run_params=run, experiment_params=manifest)
        learner.set_checkpointing(256, str(tmp_path / "models"), "fresh", save_strat="all")
        learner.learn(total_timesteps=512)
        learner.flush_checkpoints()
        assert learner._global_step == 512 and learner._num_updates >= 2
        assert learner._pretrained and learner.buffer.total_transitions >= 500
        flags = _flags(learner.agent, learner._last_train_metrics)
        assert not any(flags.values()), flags
        checkpoint = tmp_path / "models/fresh_512"
        assert checkpoint.is_file() and Path(str(checkpoint) + ".metadata.json").is_file()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert any(key.startswith("_aux_return_Qs.") for key in saved["model"])
        learner.agent.load(checkpoint)
        _report("fresh-training", {
            "passed": True, "source_commit": os.environ["EXPECTED_ACTION_MODES_SHA"],
            "config": name, "real_decisions": 512, "optimizer_updates": learner._num_updates,
            "checkpoint_roundtrip": True, "checkpoint_sidecar": True,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "compile_status": [flags], "production_overrides": {**overrides, "total_steps": 512, "checkpoint_every": 256},
        })
    finally:
        if learner is not None:
            learner.close()
        env.close()
        del learner
        gc.collect()
        torch.cuda.empty_cache()
