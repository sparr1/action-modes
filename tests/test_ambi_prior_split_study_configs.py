"""Resolve the three split-value backbone recipes before cluster submission."""

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
MANIFEST = "ambi_prior_split_study"
GROUP = "ambi-prior-split-value-20260915"
CASES = {
    "ambi_prior_split_clip_fixed0p0021": (0.0021, -21.0, "tdmpc2_percentile_range"),
    "ambi_prior_split_clip_target10p5": ("auto_0.005", -10.5, "none"),
    "ambi_prior_split_clip_target21": ("auto_0.005", -21.0, "none"),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"Duplicate JSON key: {key}"
        result[key] = value
    return result


def _load(kind, name):
    return json.loads((ROOT / kind / f"{name}.json").read_text(),
                      object_pairs_hook=_unique_object)


class _HumanoidSpaces(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (21,), dtype=np.float32)
    spec = SimpleNamespace(max_episode_steps=500)


def _resolve(name):
    config = _load("algs", name)
    manifest = _load("experiments", MANIFEST)
    # Follow main.py's shallow top-level replacement without creating a learner.
    run = {"name": name, **config, **manifest["overrides_alg"]}
    run["device"] = "cpu"
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = _HumanoidSpaces()
    algorithm.run_params = run
    algorithm.custom_params = deepcopy(run["alg_params"])
    algorithm.experiment_params = manifest
    algorithm.cfg = algorithm._build_cfg({"device": "cpu", **run["alg_params"]})
    return algorithm


@pytest.mark.parametrize("name", CASES)
def test_production_split_recipe_resolves_and_disables_inner_work(name):
    config = _load("algs", name)
    cfg = _resolve(name).cfg
    alpha, target, scale = CASES[name]
    assert config["device"] == "cuda"
    assert cfg.compile and cfg.compile_strict
    assert cfg.action_dim == 21 and cfg.obs_shape == {"state": (67,)}
    assert cfg.episode_length == 500 and cfg.latent_dim == cfg.mlp_dim == 512
    assert cfg.critic_value_mode == "return_entropy"
    assert cfg.critic_value_spec["value_components"] == ["return", "entropy"]
    assert cfg.q_representation == "distributional"
    assert (cfg.num_q, cfg.q_pair_size, cfg.q_num_bins) == (5, 2, 101)
    assert (cfg.q_vmin, cfg.q_vmax) == (-10, 10)
    assert cfg.critic_coef == 0.2  # The model averages the two component losses.
    assert cfg.critic_coef / len(cfg.critic_value_spec["value_components"]) == 0.1
    for field in ("outer_q_target_reduction", "outer_q_actor_reduction",
                  "inner_q_target_reduction", "inner_q_actor_reduction",
                  "mppi_terminal_q_reduction"):
        assert getattr(cfg, field) == "min_pair"
    assert cfg.outer_critic_target == "entropy_augmented"
    assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.log_std_mapping == cfg.inner_log_std_mapping == "direct_clamp"
    assert (cfg.log_std_min, cfg.log_std_max) == (-10, 2)
    assert cfg.ent_coef == alpha and cfg.target_entropy == target
    assert cfg.sac_actor_loss_scale_mode == scale
    assert cfg.sac_actor_loss_scale_tau == 0.01
    assert cfg.inner_operator == "none" and cfg.mpc is False
    assert cfg.inner_entropy_enabled is False
    assert cfg.inner_value_initialization == "return"
    assert cfg.inner_sac_critic_target == "reward_only"
    assert cfg.inner_finite_horizon is cfg.inner_rebase_persistent is False
    assert cfg.inner_temperature_mode == "inherit_outer"
    for field in ("inner_rounds", "inner_rollouts_per_round", "inner_updates_per_round",
                  "inner_model_step_budget", "inner_actor_updates_per_action",
                  "inner_critic_updates_per_action", "inner_temperature_updates_per_action",
                  "inner_actor_writeback_coef", "inner_critic_writeback_coef",
                  "value_equivalence_loss_coef"):
        assert getattr(cfg, field) == 0
    assert cfg.inner_explorer_mode == "none"
    assert cfg.eval_freq is None and not cfg.eval_inner_comparison and not cfg.eval_value
    assert (cfg.seed, cfg.seed_steps, cfg.pretrain_steps) == (55, 2500, 2500)
    assert (cfg.train_unroll_horizon, cfg.batch_size, cfg.utd, cfg.rho) == (3, 256, 1, 0.5)
    assert cfg.buffer_size == 1_000_000 and cfg.discount == 0.99
    assert cfg.outer_policy_diagnostics and cfg.wandb_event_indexed
    assert (cfg.outer_policy_diagnostics_early_every,
            cfg.outer_policy_diagnostics_early_until,
            cfg.outer_policy_diagnostics_every) == (100, 10_000, 1000)
    assert (cfg.outer_policy_diagnostics_states,
            cfg.outer_policy_diagnostics_samples,
            cfg.outer_policy_diagnostics_seed) == (32, 32, 12345)


def test_manifest_has_three_distinct_seed55_banks_and_preserves_checkpoint_cadence():
    manifest = _load("experiments", MANIFEST)
    assert manifest["configs"] == list(CASES)
    assert manifest["trials"] == 1 and "alg_params" not in manifest["overrides_alg"]
    assert manifest["env_params"] == {
        "task": "humanoid-walk", "obs": "state", "render_mode": None,
    }
    assert manifest["logs"] == "timestamp" and manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 25_000 and manifest["save_strat"] == "all"
    for name in CASES:
        config = _load("algs", name)
        merged = {**config, **manifest["overrides_alg"]}
        assert merged["seed"] == 55 and merged["total_steps"] == 2_000_000
        assert merged["episodes"] is None and merged["checkpoint_every"] == 25_000
        assert merged["save_strat"] == "all"
        assert merged["total_steps"] // merged["checkpoint_every"] == 80


def test_cells_differ_only_in_coefficient_system_entropy_target_and_tags():
    ignored = {"ent_coef", "target_entropy", "sac_actor_loss_scale_mode", "wandb_tags"}
    recipes = []
    for name in CASES:
        config = _load("algs", name)
        config["alg_params"] = {
            key: value for key, value in config["alg_params"].items() if key not in ignored
        }
        recipes.append(config)
    assert recipes[0] == recipes[1] == recipes[2]


def test_wandb_names_and_semantics_identify_each_arm(monkeypatch):
    calls = []
    run = SimpleNamespace(finish=lambda: None, log=lambda *args, **kwargs: None)
    fake = SimpleNamespace(init=lambda **kwargs: calls.append(kwargs) or run,
                           define_metric=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    for name in CASES:
        algorithm = _resolve(name)
        assert algorithm._init_wandb().raw_run is run
        call = calls[-1]
        assert call["name"] == f"AMBITDMPC2-{name}-seed55"
        assert call["group"] == GROUP
        assert call["project"] == "ambi" and call["entity"] == "rwgao_b-brown-university"
        config = call["config"]["config"]
        assert config["critic_value_mode"] == "return_entropy" and config["critic_coef"] == 0.2
        assert config["ent_coef"] == CASES[name][0]
        assert {"split-value-prior", "return-entropy-critics", "critic-loss0p1-per-component",
                "actor-q-min-pair", "single-seed", "seed55"}.issubset(call["tags"])
        assert not {"reward-critic-prior", "actor-q-mean-pair", "outer-alpha-auto-initial1",
                    "outer-alpha-fixed0p005"}.intersection(call["tags"])
    assert len({call["name"] for call in calls}) == 3


@pytest.mark.parametrize("name", CASES)
def test_checkpoint_sidecar_retains_split_semantics_and_temperature_regime(name):
    from main import _resolved_runtime_metadata
    from utils.checkpointing import checkpoint_metadata

    algorithm = _resolve(name)
    # Build the production network to obtain its actual strict metadata. No
    # optimizer, environment simulation, or compilation is started here.
    with torch.random.fork_rng():
        model = SoftWorldModel(algorithm.cfg)
    algorithm.agent = SimpleNamespace(model=model)
    params = deepcopy(algorithm.run_params)
    params["resolved_runtime"] = _resolved_runtime_metadata(algorithm, trial_run_params=params)
    sidecar = checkpoint_metadata(
        kind="scheduled", step=25_000, episode=50, best_score=None,
        best_window=100, trial_run_params=params, experiment_params=algorithm.experiment_params,
    )
    restored = json.loads(json.dumps(sidecar, allow_nan=False))["trial_run_params"]
    signature = restored["resolved_runtime"]["critic"]
    assert signature["critic_value_mode"] == "return_entropy"
    assert signature["value_components"] == ["return", "entropy"]
    assert signature["q_layout"] == "packed_component_bins_v1"
    assert signature["q_value_codec"] == signature["reward_value_codec"] == "symexp_two_hot_mean_v1"
    assert (signature["reward_num_bins"], signature["reward_vmin"], signature["reward_vmax"]) == (101, -10, 10)
    entropy = restored["resolved_runtime"]["actor_entropy"]["outer"]
    fixed = isinstance(CASES[name][0], float)
    assert entropy["temperature_mode"] == ("fixed" if fixed else "auto")
    assert entropy["target_active"] is not fixed
    assert entropy["target_entropy_semantics"] == "squashed_action_entropy"
    assert restored["alg_params"]["critic_coef"] == 0.2
    assert restored["alg_params"]["wandb_group"] == GROUP
