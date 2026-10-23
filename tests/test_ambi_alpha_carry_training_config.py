"""Protect the factorial comparison and the requested full-training budget."""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.test_ambi_root_local_sac import _build_cfg


ROOT = Path(__file__).resolve().parents[1]
GROUP = "ambi-alpha-carry-full-20261023"


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"Duplicate JSON key: {key}"
        result[key] = value
    return result


def _read(path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_keys)


MANIFEST = _read(ROOT / "configs/dmcontrol/experiments/ambi_alpha_carry_training.json")
NAMES = MANIFEST["configs"]
FACTORS = list(product(("episode", "run"), (False, True), (-10.5, -21.0)))


@pytest.mark.parametrize("name,factors", zip(NAMES, FACTORS))
def test_resolved_cell_lifetimes_targets_and_update_budget(name, factors):
    scope, detach, target = factors
    config = _read(ROOT / "configs/dmcontrol/algs" / f"{name}.json")
    params = config["alg_params"]
    cfg = _build_cfg(**params)
    assert cfg.inner_temperature_scope == cfg.inner_temperature_optimizer_scope == scope
    assert cfg.aux_return_detach_representation is detach
    assert cfg.target_entropy == target
    assert cfg.inner_target_entropy == "inherit_outer"
    # Exercise the actual target resolver, including the selected actor owner.
    engine = object.__new__(InnerImprovementEngine)
    engine.cfg = cfg
    engine.agent = SimpleNamespace(cfg=cfg, target_entropy=target)
    assert engine._resolved_inner_target_entropy() == target
    assert cfg.inner_temperature_initialization == "inherit_outer"
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_entropy_enabled
    for component in ("actor", "critic", "replay", "actor_optimizer", "critic_optimizer"):
        assert getattr(cfg, f"inner_{component}_scope") == "action"
    assert (cfg.train_unroll_horizon, cfg.outer_planning_horizon, cfg.inner_rollout_horizon) == (3, 3, 3)
    assert (cfg.inner_rounds, cfg.inner_rollouts_per_round, cfg.inner_batch_size) == (8, 128, 256)
    assert (cfg.inner_critic_updates_per_round, cfg.inner_actor_updates_per_round) == (16, 4)
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
    assert cfg.inner_replay_capacity == 3072
    assert cfg.inner_actor_source == cfg.inner_horizon_actor_source == "sac"
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "aux_return"
    assert cfg.aux_return_mode == "sac"
    assert cfg.outer_critic_target == "entropy_augmented"
    assert cfg.inner_sac_critic_target == "reward_only" and cfg.inner_finite_horizon
    assert cfg.inner_sac_return_estimator == "one_step"
    assert cfg.inner_outer_policy_kl_coef == cfg.inner_critic_bernoulli_p == 0
    assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0
    assert cfg.inner_behavior_action == cfg.inner_execution_action == "policy_sample"
    assert cfg.inner_actor_initialization == cfg.inner_critic_initialization == "prior"
    assert cfg.seed_steps == cfg.pretrain_steps == 2500 and cfg.utd == 1
    assert cfg.critic_ensemble_backend == "vmap" and cfg.compile and cfg.compile_strict
    assert cfg.inner_root_entropy_samples == 128 and cfg.inner_diagnostics_every == 1000
    assert cfg.eval_freq is None and not cfg.eval_inner_comparison
    assert params["ent_coef_lr"] == cfg.inner_temperature_lr == 3e-4
    assert config["total_steps"] == 1_000_000 and config["seed"] == 55
    assert config["checkpoint_every"] == 25_000 and config["save_strat"] == "all"
    assert not params.get("pretrained_model_path")
    assert params["wandb_project"] == "ambi" and params["wandb_group"] == GROUP
    assert params["wandb"] and params["wandb_event_indexed"]
    assert f"alpha {scope}" in params["wandb_run_name"]
    assert f"target {target:g}" in params["wandb_run_name"]


def test_eight_unique_factor_cells_with_no_other_algorithm_changes():
    assert len(NAMES) == len(set(NAMES)) == len(FACTORS) == 8
    source = _read(ROOT / "configs/dmcontrol/algs/ambi_full_h3_j8_fresh.json")
    labels = set()
    for name, (scope, detach, target) in zip(NAMES, FACTORS):
        config = _read(ROOT / "configs/dmcontrol/algs" / f"{name}.json")
        expected = deepcopy(source)
        expected["total_steps"] = 1_000_000
        expected["alg_params"].update(
            inner_temperature_scope=scope,
            inner_temperature_optimizer_scope=scope,
            aux_return_detach_representation=detach,
            target_entropy=target,
            inner_root_entropy_samples=128,
        )
        labels.add(config["alg_params"]["wandb_run_name"])
        for key in ("wandb_run_name", "wandb_group", "wandb_tags"):
            config["alg_params"].pop(key)
            expected["alg_params"].pop(key)
        assert config == expected, name
    assert len(labels) == 8


def test_manifest_cannot_shallow_replace_algorithm_parameters_or_budget():
    assert MANIFEST["trials"] == 1
    assert MANIFEST["overrides_alg"] == {
        "seed": 55, "device": "cuda", "env": "DMControl-v0",
        "total_steps": 1_000_000, "episodes": None,
    }
    assert MANIFEST["env_params"] == {
        "task": "humanoid-walk", "obs": "state", "render_mode": None,
    }
    assert MANIFEST["checkpoint_every"] == 25_000
    assert MANIFEST["save_strat"] == "all" and MANIFEST["save_trials"] == "none"
    assert MANIFEST["logs"] == "timestamp" and MANIFEST["log_type"] == "summary"
