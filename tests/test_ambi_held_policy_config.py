"""Resolve held-policy training independently of imagined horizon and budget."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_ambi_root_local_sac import _build_cfg


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("interval", [1, 2, 3, 5])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_solve_interval_is_independent_of_imagination_horizon(interval, horizon):
    cfg = _build_cfg(inner_solve_interval=interval, inner_rollout_horizon=horizon)
    assert cfg.inner_solve_interval == interval
    assert cfg.inner_rollout_horizon == horizon
    if interval > 1:
        assert cfg.inner_hold_encoder == "solve_snapshot"
    else:
        assert not hasattr(cfg, "inner_hold_encoder")


@pytest.mark.parametrize("interval", [True, False, 0, -1, 1.5, "3", None])
def test_solve_interval_rejects_invalid_values(interval):
    with pytest.raises(ValueError, match="inner_solve_interval"):
        _build_cfg(inner_solve_interval=interval)


@pytest.mark.parametrize("overrides", [
    {"inner_operator": "none"},
    {"inner_rounds": 0},
    {"inner_actor_scope": "episode"},
    {"inner_critic_scope": "run"},
    {"inner_replay_scope": "episode"},
    {"inner_actor_optimizer_scope": "episode"},
    {"inner_actor_initialization": "random"},
    {"inner_critic_initialization": "random"},
    {"outer_policy_episode_probability": .5},
    {"inner_actor_writeback_coef": .5},
    {"inner_critic_writeback_coef": .5},
    {"inner_outer_replay_fraction": .5},
])
def test_unsupported_hold_combinations_fail_during_configuration(overrides):
    with pytest.raises(ValueError):
        _build_cfg(inner_solve_interval=3, **overrides)


@pytest.mark.parametrize("name", [
    "ambi_full_h1_j6_fresh", "ambi_full_h2_j6_bernoulli50",
    "ambi_full_h3_j8_fresh", "ambi_full_h3_j8_bernoulli50",
    "ambi_full_h3_j8_soft",
])
def test_existing_full_training_recipes_accept_only_the_cadence_addition(name):
    params = json.loads((ROOT / "configs/dmcontrol/algs" / (name + ".json")).read_text())["alg_params"]
    baseline = _build_cfg(**params)
    held = _build_cfg(**{**params, "inner_solve_interval": 3})
    expected = vars(baseline).copy()
    expected.update(inner_solve_interval=3, inner_hold_encoder="solve_snapshot")
    assert vars(held) == expected


def test_runtime_metadata_distinguishes_solve_work_from_real_decisions():
    from main import _resolved_runtime_metadata

    cfg = _build_cfg(inner_solve_interval=3, inner_rollout_horizon=3)
    model = SimpleNamespace(cfg=cfg)
    metadata = _resolved_runtime_metadata(model, trial_run_params={"seed": 3})["inner_budget"]
    assert metadata["inner_solve_interval"] == 3
    assert metadata["inner_hold_encoder"] == "solve_snapshot"
    assert metadata["budget_scope"] == "per_solve"
    assert "transitions_per_action" not in metadata
    assert metadata["transitions_per_solve"] == cfg.inner_rounds * cfg.inner_rollouts_per_round * 3
    assert metadata["nominal_transitions_per_decision"] == metadata["transitions_per_solve"] / 3
