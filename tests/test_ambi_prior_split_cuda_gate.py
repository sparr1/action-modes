"""Opt-in production-shape CUDA gate for the three split-prior backbones.

This runs two updates on repeated real Humanoid transition slices, not the
production warmup or training budget. Reward codecs remain cold until the
first compiled outer update. The validation launcher requires all three JSON
reports before issuing its successful gate receipt.
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
from tests.test_ambi_prior_sac_cuda_gate import _FixedRealReplay


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "ambi_prior_split_clip_fixed0p0021",
    "ambi_prior_split_clip_target10p5",
    "ambi_prior_split_clip_target21",
)


@pytest.mark.skipif(
    os.environ.get("AMBI_RUN_SPLIT_PRIOR_CUDA_GATE") != "1",
    reason="set AMBI_RUN_SPLIT_PRIOR_CUDA_GATE=1 on an allocated CUDA GPU",
)
@pytest.mark.parametrize("name", CASES)
def test_split_prior_production_shape_cuda_gate(tmp_path, name):
    assert torch.cuda.is_available(), "The requested CUDA gate requires a GPU."
    import domains  # noqa: F401: register the real DMControl adapter

    manifest = json.loads((ROOT / "configs/dmcontrol/experiments/ambi_prior_split_study.json").read_text())
    config = json.loads((ROOT / "configs/dmcontrol/algs" / (name + ".json")).read_text())
    run = {"name": name, **config, **manifest["overrides_alg"]}
    params = {**run["alg_params"], "wandb": False}
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name, env, params, run_params=run, experiment_params=manifest)
        agent, cfg = learner.agent, learner.cfg
        assert cfg.device.startswith("cuda") and cfg.compile and cfg.compile_strict
        assert cfg.critic_value_mode == "return_entropy"
        assert tuple(agent.model.value_spec.components) == ("return", "entropy")
        assert (cfg.model_size, cfg.enc_dim, cfg.mlp_dim, cfg.latent_dim, cfg.num_q) == (5, 256, 512, 512, 5)
        assert (cfg.train_unroll_horizon, cfg.batch_size, cfg.action_dim) == (3, 256, 21)
        assert tuple(cfg.obs_shape["state"]) == (67,)
        assert cfg.q_num_bins == 101 and cfg.critic_coef == pytest.approx(.2)
        assert cfg.log_std_mapping == "direct_clamp"
        assert (cfg.log_std_min, cfg.log_std_max) == (-10, 2)
        assert cfg.inner_operator == "none" and not cfg.mpc
        assert cfg.outer_q_actor_reduction == cfg.outer_q_target_reduction == "min_pair"
        assert cfg.buffer_size == 1_000_000 and cfg.steps == 2_000_000
        assert cfg.seed_steps == cfg.pretrain_steps == 2500
        if name.endswith("fixed0p0021"):
            assert agent.alpha.item() == pytest.approx(.0021)
            assert cfg.sac_actor_loss_scale_mode == "tdmpc2_percentile_range"
            assert agent.ent_coef_optim is None
        else:
            assert agent.alpha.item() == pytest.approx(.005)
            assert cfg.sac_actor_loss_scale_mode == "none"
            assert agent.target_entropy == (-10.5 if name.endswith("target10p5") else -21.)
            assert agent.ent_coef_optim is not None

        # No model Q/reward/actor forward, eager loss, or codec invocation occurs
        # before update 1. Only real environment transitions populate this fixture.
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
        fallback_status = []
        for completed in (1, 2):
            agent._outer_policy_diagnostics_force = True
            try:
                metrics = agent.update(replay)
            finally:
                agent._outer_policy_diagnostics_force = False
            torch.cuda.synchronize()
            assert agent.num_updates == replay.draws == completed
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
            for component in ("return", "entropy"):
                assert f"critic_{component}_loss" in metrics
                assert f"q_{component}_mean" in metrics
                assert f"q_{component}_target_clip_fraction" in metrics
            flags = (
                bool(metrics["compile_fallback"]),
                bool(metrics["compile_outer_update_fallback"]),
                bool(agent.model._Qs.compile_failed),
                bool(agent.model._target_Qs.compile_failed),
            )
            fallback_status.extend(flags)
            assert not any(flags), "Production-shape compilation fell back to eager."
            assert agent._outer_update_region._compiled is not None
            packet = agent.drain_outer_policy_diagnostics()
            assert packet is not None
            assert packet["actor_updates_after"] == completed

        # Both categorical output blocks receive actual TD learning, including
        # the unweighted future-entropy component.
        for head in agent.model._Qs:
            assert torch.count_nonzero(head[-1].weight[:cfg.q_num_bins]).item() > 0
            assert torch.count_nonzero(head[-1].weight[cfg.q_num_bins:]).item() > 0

        # Only after the cold compile do we exercise the explicit numeric APIs.
        target = torch.tensor([-13., -.1, 0., .4, 27.], device=agent.device)[:, None]
        for codec in (agent.model.reward_codec, agent.model.q_backend.value_codec):
            encoded = codec.encode_target(target)
            torch.testing.assert_close(codec.decode(encoded.log()), target, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(codec.decode(encoded.mean(0).log()), target.mean(0), rtol=1e-5, atol=1e-5)

        saved = deepcopy(agent.checkpoint_state())
        checkpoint = tmp_path / (name + ".pt")
        torch.save(saved, checkpoint)
        with torch.no_grad():
            agent.model._Qs[0][-1].bias.add_(.125)
        agent.load(checkpoint)
        for key, value in saved["model"].items():
            torch.testing.assert_close(agent.model.state_dict()[key], value, rtol=0, atol=0)
        assert agent.model.critic_signature == saved["critic_spec"]
        assert agent.num_updates == 2
        assert learner.buffer.total_transitions == 0

        # The ordinary stochastic no-inner acting path must work after reload.
        action = learner._act_agent(learner._obs_to_tensor(observation), t0=False, eval_mode=False).numpy()
        assert action.shape == (21,) and np.isfinite(action).all()
        assert np.abs(action).max() <= 1.
        report = {
            "config": name,
            "source_commit": os.environ["EXPECTED_ACTION_MODES_SHA"],
            "passed": True,
            "compile_strict": bool(cfg.compile_strict),
            "compile_fallbacks": fallback_status,
            "checkpoint_roundtrip": True,
            "optimizer_updates": 2,
            "replay_draws": replay.draws,
            "real_decisions": 32,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "elapsed_seconds": time.perf_counter() - started,
            "production_overrides": {"wandb": False},
            "production_warmup_and_pretraining_executed": False,
        }
        output = Path(os.environ["AMBI_SPLIT_GATE_OUTPUT_ROOT"])
        with (output / (name + ".json")).open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print("SPLIT_PRIOR_CUDA_GATE_REPORT " + json.dumps(report, sort_keys=True))
    finally:
        if learner is not None:
            learner.close()
        env.close()
        del learner
        gc.collect()
        torch.cuda.empty_cache()
