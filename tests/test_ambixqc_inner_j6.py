"""Requested J6 XQC budget, inherited checkpoint semantics, and real frozen solves."""

from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBIXQC import AMBIXQC
from test_ambixqc_core import _batch, _tree_equal
from test_ambixqc_prior_checkpoint import _wrapper
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset
from utils.checkpoint_context import CheckpointContext


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambixqc_humanoid_inner_j6_benchmark.json"
BANK = ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m.json"
BUDGET = {
    "inner_operator": "xqc", "inner_rounds": 6,
    "inner_rollouts_per_round": 512, "inner_rollout_horizon": 3,
    "inner_updates_per_round": 3, "inner_batch_size": 512,
    "inner_replay_capacity": 9216, "inner_replay_sampling": "with_replacement",
}


def test_j6_matrix_borrows_only_budget_and_defaults_to_inner_xqc():
    matrix = load_preset_matrix(MATRIX)  # Also rejects duplicate/nonfinite JSON.
    assert matrix["base_alg_config"] == "checkpoint"
    assert "environment" not in matrix and "source_run" not in matrix
    assert matrix.get("shared_alg_params", {}) == {}
    assert normalize_selectors(matrix) == ["controller/xqc"]
    assert matrix["evaluation"] == {
        "controller_seed": 12345, "seeds": [101, 102, 103, 104, 105],
        "max_steps": 500, "default_presets": ["controller/xqc"],
        "wandb_project": "ambi-inner-bench",
    }
    comparison = matrix["comparisons"]["controller"]
    assert comparison["reference"] == "prior"
    assert set(comparison["variants"]) == {"prior", "xqc"}
    assert comparison["variants"]["prior"]["alg_params"] == {"inner_operator": "none"}
    assert comparison["variants"]["xqc"]["alg_params"] == BUDGET
    source = matrix["budget_source"]
    assert source["run_name"] == "AMBITDMPC2-humanoid-walk-base-v2-d512-4-j6-seed55"
    assert source["source_commit"] == "698bd074551bc48128cb34f78adaf8caaab1f188"
    assert source["config_sha256"] == "96f16f50207e20b118b5c30f913c89c853e71369a02464777c73062ff6e34ae5"


def test_j6_resolution_preserves_checkpoint_outer_semantics_and_inner_defaults():
    saved = json.loads(BANK.read_text())
    context = CheckpointContext(saved, {"env_params": {"task": "humanoid-walk", "obs": "state"}}, BANK)
    before = deepcopy(context)
    resolved = resolve_preset(MATRIX, "controller/xqc", checkpoint_context=context)
    expected = deepcopy(saved)
    expected["alg_params"].update(BUDGET)
    assert resolved["algorithm_config"] == expected
    assert resolved["saved_algorithm_config"] == saved
    assert context == before
    assert resolved["environment"] == {
        "id": "DMControl-v0", "params": {"task": "humanoid-walk", "obs": "state"},
    }
    wrapper = object.__new__(AMBIXQC)
    wrapper.env = gym.make("Pendulum-v1")
    wrapper.run_params = {**expected, "device": "cpu"}
    wrapper.experiment_params = {}
    wrapper.custom_params = expected["alg_params"]
    try:
        cfg = wrapper._build_cfg({**expected["alg_params"], "device": "cpu"})
    finally:
        wrapper.env.close()
    assert cfg.xqc_policy_delay == 3
    assert cfg.inner_actor_lr == cfg.inner_critic_lr == 5e-5
    assert cfg.inner_reward_normalization == "frozen_real_scale"
    assert cfg.inner_model_step_budget == 9216
    assert cfg.inner_expected_update_slots == cfg.inner_critic_updates_per_action == 18
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 6
    assert cfg.compile is False


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA hardware is unavailable")),
])
def test_full_j6_budget_has_fresh_learners_exact_steps_and_seeded_frozen_actions(
    device, tmp_path, monkeypatch
):
    # Keep the production rollout/update budget; only the learned network sizes
    # and outer training batch are reduced for this local integration test.
    source = _wrapper(device=device, inner_operator="none", xqc_optimizer_backend="auto",
                      train_unroll_horizon=3)
    target = _wrapper(device=device, xqc_optimizer_backend="auto", train_unroll_horizon=3,
                      **BUDGET)
    try:
        source.agent.observe_reward(2.0, False, False)
        source.agent.observe_reward(3.0, False, True)
        source.agent._update(*(tensor.to(device) for tensor in _batch(source.agent)))
        checkpoint = tmp_path / "learned-prior.pt"
        source.agent.save(str(checkpoint))
        target.load(str(checkpoint), frozen_evaluation=True)
        agent, engine = target.agent, target.agent.inner_engine
        before = agent.frozen_outer_state()
        observation, _ = target.env.reset(seed=101)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(agent.device).clone() if device == "cuda" else None
        preparations, observed_steps, model_rows = [], [], []
        instrumented = set()
        prepare = engine._prepare_action
        dynamics = agent.model.next_from_joint

        def checked_prepare():
            prepare()
            state = engine.state
            workspace, local = state.workspace, state.workspace.controller
            assert workspace.update_step == workspace.actor_optimizer_steps == workspace.temperature_optimizer_steps == 0
            assert state.replay.size == state.replay.next_sample_id == 0
            assert state.reward_normalizer is None
            assert state.reward_scale == agent.reward_normalizer.scale
            for name in ("actor", "critic"):
                assert _tree_equal(getattr(local, name).state_dict(), getattr(agent.xqc_controller, name).state_dict())
            assert _tree_equal(local.critic_target.state_dict(), agent.xqc_controller.critic.state_dict())
            assert torch.equal(local.log_temperature, agent.xqc_controller.log_temperature)
            preparations.append(id(workspace))
            for component, optimizer in (
                ("critic", workspace.critic_optimizer), ("actor", workspace.actor_optimizer),
                ("temperature", workspace.temperature_optimizer),
            ):
                assert not optimizer.state
                if id(optimizer) not in instrumented:
                    instrumented.add(id(optimizer))
                    original = optimizer.step

                    def counted_step(*args, _step=original, _component=component, **kwargs):
                        observed_steps.append(_component)
                        return _step(*args, **kwargs)

                    monkeypatch.setattr(optimizer, "step", counted_step)

        def counted_dynamics(joint):
            model_rows.append(joint.shape[0])
            return dynamics(joint)

        monkeypatch.setattr(engine, "_prepare_action", checked_prepare)
        monkeypatch.setattr(agent.model, "next_from_joint", counted_dynamics)

        def decision():
            observed_steps.clear()
            model_rows.clear()
            action, _ = target.predict(observation, deterministic=True)
            metrics = agent.last_inner_metrics
            assert sum(model_rows) == metrics["inner_model_steps"] == 9216
            assert metrics["inner_rounds"] == 6
            assert metrics["inner_rollouts"] == 3072
            assert metrics["inner_replay_draws"] == metrics["inner_buffer_size"] == 9216
            expected_order = []
            for slot in range(18):
                expected_order.append("critic")
                if slot % 3 == 0:
                    expected_order.extend(("actor", "temperature"))
            assert observed_steps == expected_order
            assert tuple(metrics[f"inner_{name}_optimizer_steps"] for name in ("critic", "actor", "temperature")) == (18, 6, 6)
            assert metrics["inner_reward_normalizer_imagined_updates"] == 0
            assert metrics["inner_reward_scale_delta"] == 0
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
            pool = engine._workspace_pool
            assert pool.update_step == 18 and pool.actor_optimizer_steps == pool.temperature_optimizer_steps == 6
            assert engine.state.workspace is None and engine.state.replay is None
            assert not _tree_equal(pool.controller.actor.state_dict(), agent.xqc_controller.actor.state_dict())
            with torch.no_grad():
                z = agent.model.encode(target._obs_to_tensor(observation).unsqueeze(0))
                mean, _ = pool.controller.actor.distribution(z, bn_mode="running")
            np.testing.assert_array_equal(action, target._unscale_action(mean.tanh()[0].cpu().numpy()))
            agent.observe_reward(100.0, True, False)
            assert _tree_equal(before, agent.frozen_outer_state())
            return action

        decision()  # Warmup allocation must not affect scored seeded actions.

        def episode(seed, reuse=True):
            target.reset_for_evaluation(seed, reuse_action_pool=reuse)
            assert engine.rng.generator("collection").device.type == device
            assert engine.action_index == 0
            actions = np.stack([decision(), decision()])
            assert engine.action_index == 2
            return actions

        first = episode(12345)
        assert len(set(preparations)) == 1  # Allocation reuse, with fresh state each action.
        alternate = episode(12346)
        assert not np.array_equal(first, alternate)  # Inner adaptation is stochastic.
        np.testing.assert_array_equal(first, episode(12345))
        np.testing.assert_array_equal(first, episode(12345, reuse=False))
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(agent.device))
    finally:
        source.env.close()
        target.env.close()
