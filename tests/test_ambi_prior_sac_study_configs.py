"""Freeze the two-seed SAC prior screen and its native-initialization contrast."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel


ROOT = Path(__file__).resolve().parents[1] / "configs/dmcontrol"
MANIFEST = "ambi_prior_sac_parameterization_study"
CASES = {
    "ambi_prior_sac_clip_target21": ("direct_clamp", -21.0),
    "ambi_prior_sac_clip_target10p5": ("direct_clamp", -10.5),
    "ambi_prior_sac_smooth_target21": ("tdmpc2_tanh", -21.0),
    "ambi_prior_sac_smooth_target10p5": ("tdmpc2_tanh", -10.5),
}
COMMON = {
    "obs": "state", "model_size": 5, "episodic": False,
    "discount": 0.99, "buffer_size": 1_000_000, "batch_size": 256,
    "utd": 1, "train_unroll_horizon": 3, "outer_planning_horizon": 3,
    "rho": 0.5, "seed_steps": 2500, "pretrain_steps": 2500,
    "compile": True, "compile_strict": False, "mpc": False,
    "q_representation": "distributional", "num_q": 5, "q_pair_size": 2,
    "q_num_bins": 101, "q_vmin": -10, "q_vmax": 10,
    "outer_q_actor_reduction": "mean_pair",
    "outer_q_target_reduction": "min_pair",
    "outer_critic_target": "entropy_augmented",
    "outer_actor_entropy_mode": "squashed",
    "sac_actor_loss_scale_mode": "none",
    "ent_coef": "auto_1.0", "ent_coef_lr": 3e-4,
    "log_std_min": -10, "log_std_max": 2,
    "actor_lr": 3e-4, "critic_lr": 3e-4, "lr": 3e-4,
    "enc_lr_scale": 0.3, "adam_eps": 1e-8, "actor_adam_eps": 1e-5,
    "tau": 0.01, "target_update_interval": 1, "dropout": 0.01,
    "grad_clip_norm": 20.0, "reward_coef": 0.1, "critic_coef": 0.1,
    "consistency_coef": 20.0, "termination_coef": 1.0,
    "inner_operator": "none", "inner_rounds": 0,
    "inner_rollouts_per_round": 0, "inner_updates_per_round": 0,
    "inner_diagnostic_rollouts": 0, "inner_explorer_mode": "none",
    "inner_actor_writeback_coef": 0.0, "inner_critic_writeback_coef": 0.0,
    "outer_behavior_policy_kl_schedule": "none",
    "value_equivalence_loss_coef": 0.0, "value_equivalence_diagnostics": False,
    "outer_policy_episode_probability": 0.0,
    "eval_freq": None, "eval_inner_comparison": False, "eval_value": False,
    "outer_policy_diagnostics": True,
    "outer_policy_diagnostics_early_every": 100,
    "outer_policy_diagnostics_early_until": 10_000,
    "outer_policy_diagnostics_every": 1000,
    "outer_policy_diagnostics_states": 32,
    "outer_policy_diagnostics_samples": 32,
    "outer_policy_diagnostics_seed": 12345,
    "wandb_event_indexed": True,
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"Duplicate JSON key: {key}"
        result[key] = value
    return result


def _load(kind, name):
    return json.loads(
        (ROOT / kind / f"{name}.json").read_text(),
        object_pairs_hook=_unique_object,
    )


class _HumanoidSpaces(gym.Env):
    """Resolve the real state/action dimensions without constructing MuJoCo."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (21,), dtype=np.float32)
    spec = SimpleNamespace(max_episode_steps=500)


def _resolve(name, trial=0):
    config = _load("algs", name)
    manifest = _load("experiments", MANIFEST)
    # Match main.py's top-level merge and seed resolution. No learner is built.
    run = {"name": name, **config, **manifest["overrides_alg"]}
    run["seed"] += trial
    run["device"] = "cpu"
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = _HumanoidSpaces()
    algorithm.run_params = run
    algorithm.custom_params = deepcopy(run["alg_params"])
    algorithm.experiment_params = manifest
    algorithm.cfg = algorithm._build_cfg({"device": "cpu", **run["alg_params"]})
    return algorithm


@pytest.mark.parametrize("name", CASES)
def test_prior_sac_settings_resolve_with_the_humanoid_contract(name):
    mapping, target = CASES[name]
    config = _load("algs", name)
    params = config["alg_params"]
    algorithm = _resolve(name)
    cfg = algorithm.cfg
    assert config["alg"] == "AMBITDMPC2/AMBITDMPC2"
    for key, value in COMMON.items():
        assert params[key] == value, key
        assert getattr(cfg, key) == value, key
    assert cfg.action_dim == 21
    assert cfg.obs_shape == {"state": (67,)}
    assert cfg.episode_length == 500
    assert cfg.latent_dim == cfg.mlp_dim == 512
    assert cfg.temporal_loss_normalization == "divide_horizon"
    assert cfg.log_std_mapping == cfg.inner_log_std_mapping == mapping
    assert params["inner_log_std_mapping"] is None
    assert cfg.target_entropy == target
    assert cfg.inner_model_step_budget == 0
    assert cfg.inner_actor_updates_per_action == 0
    assert cfg.inner_critic_updates_per_action == 0
    assert cfg.inner_temperature_updates_per_action == 0
    assert not any("initial_std" in key for key in params)
    assert params["wandb_group"] == "ambi-prior-sac-parameterization-20260913"
    assert params["wandb_project"] == "ambi"
    assert params["wandb_mode"] == "online"


def test_manifest_covers_exactly_four_configurations_and_eight_seeded_runs():
    manifest = _load("experiments", MANIFEST)
    assert len(manifest["configs"]) == len(set(manifest["configs"])) == 4
    assert set(manifest["configs"]) == set(CASES)
    assert set(CASES.values()) == {
        (mapping, target)
        for mapping in ("direct_clamp", "tdmpc2_tanh")
        for target in (-21.0, -10.5)
    }
    assert manifest["trials"] == 2
    assert "alg_params" not in manifest["overrides_alg"]
    assert manifest["overrides_alg"]["seed"] == 55
    assert manifest["env_params"] == {
        "task": "humanoid-walk", "obs": "state", "render_mode": None,
    }
    assert manifest["logs"] == "timestamp"
    assert manifest["save_trials"] == "none"
    assert manifest["save_strat"] == "all"
    assert manifest["checkpoint_every"] == 25_000
    cells = []
    for name in manifest["configs"]:
        config = _load("algs", name)
        assert config["total_steps"] == manifest["overrides_alg"]["total_steps"] == 2_000_000
        assert config["episodes"] is None
        assert config["checkpoint_every"] == 25_000
        assert config["save_strat"] == "all"
        assert config["seed"] == 55
        for trial in range(manifest["trials"]):
            cells.append((name, config["seed"] + trial))
    assert len(set(cells)) == 8
    assert {seed for _, seed in cells} == {55, 56}
    assert len(cells) * 2_000_000 // 25_000 == 640


def test_only_mapping_target_and_descriptive_tags_differ_between_cells():
    ignored = {"log_std_mapping", "target_entropy", "wandb_tags"}
    normalized = []
    for name in CASES:
        config = _load("algs", name)
        params = config["alg_params"]
        assert "wandb_run_name" not in params
        assert not {"seed55", "single-seed", "entropy-tdmpc2-scaled"}.intersection(
            params["wandb_tags"]
        )
        normalized.append({
            **config,
            "alg_params": {key: value for key, value in params.items() if key not in ignored},
        })
    assert all(config == normalized[0] for config in normalized[1:])


def test_full_resolved_recipe_differs_only_on_approved_axes_and_seed_identity():
    ignored = {"log_std_mapping", "inner_log_std_mapping", "target_entropy", "wandb_tags", "seed"}
    resolved = [
        {key: value for key, value in vars(_resolve(name, trial).cfg).items()
         if key not in ignored}
        for name in CASES for trial in (0, 1)
    ]
    assert all(config == resolved[0] for config in resolved[1:])


def test_wandb_receives_eight_distinct_seed_correct_names_and_configs(monkeypatch):
    calls = []
    run = SimpleNamespace(finish=lambda: None, log=lambda *args, **kwargs: None)
    fake = SimpleNamespace(
        init=lambda **kwargs: calls.append(kwargs) or run,
        define_metric=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    expected_names = set()
    for name in CASES:
        for trial in range(2):
            algorithm = _resolve(name, trial)
            assert algorithm._init_wandb().raw_run is run
            expected_name = f"AMBITDMPC2-{name}-seed{55 + trial}"
            expected_names.add(expected_name)
            assert calls[-1]["name"] == expected_name
            assert calls[-1]["config"]["config"]["seed"] == 55 + trial
            assert calls[-1]["config"]["run_params"]["seed"] == 55 + trial
            assert calls[-1]["group"] == "ambi-prior-sac-parameterization-20260913"
    assert len(expected_names) == 8
    assert {call["name"] for call in calls} == expected_names


@pytest.mark.parametrize("trial", [0, 1])
def test_checkpoint_sidecar_keeps_resolved_diagnostic_settings(trial):
    from main import _resolved_runtime_metadata
    from utils.checkpointing import checkpoint_metadata

    algorithm = _resolve("ambi_prior_sac_clip_target10p5", trial)
    run_params = deepcopy(algorithm.run_params)
    run_params["resolved_runtime"] = _resolved_runtime_metadata(
        algorithm, trial_run_params=run_params,
    )
    sidecar = checkpoint_metadata(
        kind="scheduled", step=25_000, episode=50, best_score=None,
        best_window=100, trial_run_params=run_params,
        experiment_params=algorithm.experiment_params,
    )
    restored = json.loads(json.dumps(sidecar, allow_nan=False))["trial_run_params"]
    assert restored["seed"] == 55 + trial
    expected = {
        key: value for key, value in COMMON.items()
        if key.startswith("outer_policy_diagnostics") or key == "wandb_event_indexed"
    }
    assert restored["resolved_runtime"]["outer_policy_diagnostics"] == expected
    for key, value in expected.items():
        assert restored["alg_params"][key] == value


@pytest.mark.parametrize("trial", [0, 1])
def test_native_initialization_pairs_raw_weights_without_matching_policy_noise(trial):
    states = []
    policies = {}
    rng_states = []
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        for name, (mapping, target) in CASES.items():
            cfg = deepcopy(_resolve(name, trial).cfg)
            # Tiny network, same factory and 21-action head initialization.
            cfg.model_size = None
            cfg.enc_dim = cfg.mlp_dim = 32
            cfg.latent_dim = 16
            cfg.num_enc_layers = 2
            cfg.simnorm_dim = 8
            cfg.num_bins = cfg.q_num_bins = 11
            torch.manual_seed(cfg.seed)
            model = SoftWorldModel(cfg).eval()
            states.append(deepcopy(model.state_dict()))
            rng_states.append(torch.random.get_rng_state().clone())
            # No special constant-sigma initialization was inserted into either arm.
            assert torch.count_nonzero(model._pi[-1].weight[21:]) > 0
            obs = torch.linspace(-0.2, 0.4, 2 * 67).reshape(2, 67)
            z = model.encode(obs)
            mean, log_std = model._policy_parameters(z)
            policies[(mapping, target)] = (mean, log_std)
    for state, rng in zip(states[1:], rng_states[1:]):
        assert state.keys() == states[0].keys()
        for key in state:
            torch.testing.assert_close(state[key], states[0][key], rtol=0, atol=0)
        torch.testing.assert_close(rng, rng_states[0], rtol=0, atol=0)
    for mapping in ("direct_clamp", "tdmpc2_tanh"):
        for actual, expected in zip(policies[(mapping, -21.0)], policies[(mapping, -10.5)]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    clipped_mean, clipped_log_std = policies[("direct_clamp", -21.0)]
    smooth_mean, smooth_log_std = policies[("tdmpc2_tanh", -21.0)]
    torch.testing.assert_close(clipped_mean, smooth_mean, rtol=0, atol=0)
    assert not torch.allclose(clipped_log_std, smooth_log_std)
    assert not torch.allclose(clipped_log_std.exp(), smooth_log_std.exp())
