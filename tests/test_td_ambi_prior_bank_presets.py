"""Frozen bank presets preserve checkpoint objectives and execute J5 C1/A1."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset


ROOT = Path(__file__).resolve().parents[1]
BANKS = {
    "reward_qscale": "mey3rxj8",
    "entropy_qscale": "zdnoyhbt",
    "entropy_autotemp": "hw0nlj29",
    "reward_autotemp": "jirflxz1",
}
SELECTORS = ("controller/prior", "controller/mppi", "inner/fixed", "inner/adaptive")


def _resolved(bank, selector):
    source = ROOT / f"configs/dmcontrol/algs/td_ambi_prior_{bank}.json"
    run = json.loads(source.read_text())
    context = SimpleNamespace(
        source=source,
        trial_run_params=run,
        experiment_params={"env_params": {"task": "humanoid-walk", "obs": "state"}},
    )
    resolved = resolve_preset(
        ROOT / f"configs/research/td_ambi_prior_bank_{bank}.json",
        selector,
        checkpoint_context=context,
    )
    assert context.trial_run_params == json.loads(source.read_text())
    return resolved


def _cfg(resolved, *, small=False):
    run = deepcopy(resolved["algorithm_config"])
    params = run["alg_params"]
    params.update(device="cpu", compile=False, wandb=False)
    if small:
        params.update(model_size=None, enc_dim=32, mlp_dim=32, latent_dim=16,
                      num_enc_layers=2, simnorm_dim=8)
    env = gym.Env()
    env.observation_space = gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32)
    env.action_space = gym.spaces.Box(-1.0, 1.0, (21,), dtype=np.float32)
    env.spec = SimpleNamespace(max_episode_steps=500)
    wrapper = object.__new__(AMBITDMPC2)
    wrapper.env = env
    wrapper.run_params = {**run, "device": "cpu"}
    wrapper.custom_params = params
    return wrapper._build_cfg(params)


@pytest.mark.parametrize("bank,run_id", BANKS.items())
def test_each_bank_resolves_from_its_own_checkpoint_without_changing_outer_objective(bank, run_id):
    path = ROOT / f"configs/research/td_ambi_prior_bank_{bank}.json"
    matrix = load_preset_matrix(path)
    assert matrix["source_run"] == f"rwgao_b-brown-university/ambi/{run_id}"
    assert normalize_selectors(matrix) == ["controller/prior", "controller/mppi"]
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["controller_seed"] == 55
    assert matrix["evaluation"]["max_steps"] == 500
    outer = json.loads((ROOT / f"configs/dmcontrol/algs/td_ambi_prior_{bank}.json").read_text())
    for selector in SELECTORS:
        resolved = _resolved(bank, selector)
        cfg = _cfg(resolved)
        params = resolved["algorithm_config"]["alg_params"]
        assert {key: value for key, value in params.items() if not key.startswith("inner_")} == {
            key: value for key, value in outer["alg_params"].items() if not key.startswith("inner_")
        }
        assert cfg.action_dim == 21
        assert cfg.outer_critic_target == ("entropy_augmented" if bank.startswith("entropy") else "reward_only")
        assert cfg.sac_actor_loss_scale_mode == ("tdmpc2_percentile_range" if bank.endswith("qscale") else "none")
        if selector.startswith("controller/"):
            assert cfg.inner_operator == "none"
            assert cfg.inner_update_timing == "round"
            assert cfg.inner_steps_per_update is None
            assert cfg.inner_critic_target_initialization == "online"
            assert cfg.inner_critic_loss_coef == 1.0
        else:
            assert cfg.inner_operator == "sac"
            assert cfg.inner_sac_critic_target == cfg.outer_critic_target
            assert cfg.inner_actor_entropy_mode == cfg.outer_actor_entropy_mode == "tdmpc2_scaled"
            assert cfg.inner_log_std_mapping == cfg.log_std_mapping == "tdmpc2_tanh"
            assert (cfg.inner_log_std_min, cfg.inner_log_std_max) == (-10, 2)
            assert cfg.inner_critic_target_initialization == "outer_target"
            assert cfg.inner_critic_loss_coef == cfg.critic_coef == 0.1
            assert cfg.inner_actor_adam_eps == cfg.actor_adam_eps == 1e-5
            assert cfg.inner_adam_eps == cfg.adam_eps == 1e-8


@pytest.mark.parametrize("bank", BANKS)
def test_fixed_and_adaptive_presets_change_only_the_requested_local_scalar(bank):
    fixed = _resolved(bank, "inner/fixed")["algorithm_config"]["alg_params"]
    adaptive = _resolved(bank, "inner/adaptive")["algorithm_config"]["alg_params"]
    changed = {key for key in fixed.keys() | adaptive.keys() if fixed.get(key) != adaptive.get(key)}
    cfg = _cfg(_resolved(bank, "inner/adaptive"))
    if bank.endswith("qscale"):
        assert changed == {"inner_actor_loss_scale_update"}
        assert cfg.inner_actor_loss_scale_update == "per_update"
        assert cfg.inner_temperature_mode == "inherit_outer"
        assert cfg.sac_actor_loss_scale_tau == 0.01
    else:
        assert changed == {"inner_temperature_mode"}
        assert cfg.inner_temperature_mode == "auto"
        assert cfg.inner_target_entropy == -441
        assert cfg.inner_temperature_grad_clip_norm is None
        assert cfg.inner_actor_loss_scale_update == "per_action"
    assert cfg.inner_temperature_initialization == "inherit_outer"


@pytest.mark.parametrize("bank", BANKS)
@pytest.mark.parametrize("mode", ("fixed", "adaptive"))
def test_bank_sac_executes_15_paired_updates_and_discards_private_learner(bank, mode, monkeypatch):
    cfg = _cfg(_resolved(bank, f"inner/{mode}"), small=True)
    assert (cfg.inner_rounds, cfg.inner_rollouts_per_round, cfg.inner_rollout_horizon) == (5, 512, 3)
    assert cfg.inner_batch_size == cfg.inner_steps_per_update == 512
    assert cfg.inner_replay_capacity == 7680
    assert cfg.inner_update_timing == "step"
    assert cfg.inner_actor_adaptation == cfg.inner_critic_adaptation == "clone"
    assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0
    agent = AMBITDMPC2Agent(cfg)
    engine = agent.inner_engine
    if agent.actor_loss_scale_enabled:
        agent.actor_loss_scale.fill_(17.5)
    if agent.log_ent_coef is not None:
        with torch.no_grad():
            agent.log_ent_coef.fill_(np.log(9e-8))
    before = deepcopy(agent.checkpoint_state())
    events = []
    capture = {}
    original_critic = engine._sac_critic_step
    original_policy = engine._sac_policy_step
    original_target = engine._maybe_update_targets

    def critic(batch, *args, **kwargs):
        events.append("critic")
        capture["batch"] = batch
        assert batch["z"].shape == (512, cfg.latent_dim)
        assert engine.state.replay.size == (engine.state.critic_steps + 1) * 512
        return original_critic(batch, *args, **kwargs)

    def policy(batch, **kwargs):
        events.append("actor")
        assert batch is capture["batch"]
        assert kwargs["update_actor"] is True
        assert kwargs["update_temperature"] is (mode == "adaptive" and bank.endswith("autotemp"))
        return original_policy(batch, **kwargs)

    def target(**kwargs):
        events.append("target")
        result = original_target(**kwargs)
        if engine.state.actor_steps == 15:
            capture["final"] = SimpleNamespace(**vars(engine.state))
        return result

    monkeypatch.setattr(engine, "_sac_critic_step", critic)
    monkeypatch.setattr(engine, "_sac_policy_step", policy)
    monkeypatch.setattr(engine, "_maybe_update_targets", target)
    action = agent.act(torch.zeros(67), t0=True, eval_mode=True, collect_diagnostics=False)
    assert action.shape == (21,) and torch.isfinite(action).all() and (action.abs() <= 1).all()
    assert events == ["critic", "actor", "target"] * 15
    final = capture["final"]
    assert final.critic_steps == final.actor_steps == final.critic_target_steps == 15
    assert final.temperature_steps == (15 if mode == "adaptive" and bank.endswith("autotemp") else 0)
    assert final.replay.size == 7680
    assert engine.state.actor is engine.state.critic is engine.state.replay is None
    _assert_tree_equal(agent.checkpoint_state(), before)
