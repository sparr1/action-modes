"""Opt-in collection CUDA-graph configuration and lineage compatibility."""

import pytest

from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_ambi_distillation_config import _identity


def _graph_cfg(**overrides):
    options = dict(
        inner_cuda_graph=True, compile=True, compile_strict=True,
        outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
        inner_execution_action="policy_sample",
        inner_rounds=1, inner_updates_per_round=1,
    )
    options.update(overrides)
    return _build_cfg(**options)


def test_collection_graph_default_preserves_existing_identity():
    assert _build_cfg().inner_cuda_graph is False
    for objective in ("q_maximization", "distillation"):
        base = dict(outer_actor_objective=objective)
        assert _identity(base) == _identity(dict(base, inner_cuda_graph=False))
        assert "inner_cuda_graph" not in _identity(dict(base, inner_cuda_graph=False))["alg_params"]
        enabled = _identity(dict(base, inner_cuda_graph=True))
        assert enabled != _identity(base)
        assert enabled["alg_params"]["inner_cuda_graph"] is True


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_collection_graph_requires_strict_boolean(value):
    with pytest.raises(ValueError, match="inner_cuda_graph"):
        _build_cfg(inner_cuda_graph=value)


@pytest.mark.parametrize("objective", ["q_maximization", "distillation"])
def test_collection_graph_accepts_fresh_posttanh_outer_adapt_recipe(objective):
    cfg = _graph_cfg(
        outer_actor_objective=objective, outer_actor_entropy_mode="squashed",
        inner_actor_entropy_mode="squashed", ent_coef=0.0021,
        inner_temperature_mode="inherit_outer", sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        inner_actor_loss_scale_update="per_update", inner_actor_loss_scale_scope="action",
        inner_solve_interval=1,
    )
    assert cfg.inner_cuda_graph is True
    assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.ent_coef == 0.0021
    assert cfg.inner_actor_loss_scale_update == "per_update"
    assert cfg.store_teacher_policy is (objective == "distillation")
    assert cfg.store_behavior_policy is False


@pytest.mark.parametrize("override,match", [
    ({"compile": False}, "compile"),
    ({"compile_strict": False}, "compile_strict"),
    ({"inner_operator": "none"}, "inner_operator"),
    ({"inner_actor_adaptation": "frozen"}, "inner_actor_adaptation"),
    ({"inner_actor_initialization": "random"}, "inner_actor_initialization"),
    ({"inner_critic_initialization": "random"}, "inner_critic_initialization"),
    ({"inner_actor_scope": "episode"}, "inner_actor_scope"),
    ({"inner_replay_scope": "run"}, "inner_replay_scope"),
    ({"inner_explorer_mode": "frozen_random"}, "inner_explorer_mode"),
    ({"inner_actor_writeback_coef": 0.1}, "inner_actor_writeback_coef"),
    ({"inner_critic_writeback_coef": 0.1}, "inner_critic_writeback_coef"),
    ({"inner_outer_replay_fraction": 0.5}, "inner_outer_replay_fraction"),
    ({"outer_behavior_policy_kl_schedule": "smooth"}, "outer_behavior_policy_kl_schedule"),
    ({"outer_critic_target": "entropy_augmented"}, "outer_critic_target"),
    ({"inner_sac_critic_target": "entropy_augmented"}, "inner_sac_critic_target"),
    ({"inner_solve_interval": 3}, "inner_solve_interval"),
    ({"inner_actor_loss_scale_scope": "run"}, "inner_actor_loss_scale_scope"),
    ({"inner_root_entropy_samples": 1}, "inner_root_entropy_samples"),
    ({"inner_execution_action": "mean"}, "inner_execution_action"),
])
def test_collection_graph_rejects_unsupported_workloads(override, match):
    with pytest.raises(ValueError, match=match):
        _graph_cfg(**override)


def test_disabled_collection_graph_does_not_restrict_native_sac():
    cfg = _build_cfg(inner_cuda_graph=False, inner_actor_initialization="random")
    assert cfg.inner_actor_initialization == "random"
    assert cfg.inner_cuda_graph is False


def test_collection_graph_usage_is_logged_once_per_action():
    from tests.test_ambi_root_local_sac import _tiny_model

    model = _tiny_model(inner_rounds=1, inner_updates_per_round=1)
    try:
        model._wandb_train_window.snapshot(clear=True)
        for used, updates in ((1.0, 128), (0.0, 1)):
            model.agent.last_inner_rollout_lengths = []
            model.agent.last_inner_metrics = {
                "inner_cuda_graph_used": used,
                "inner_actor_optimizer_steps": updates,
                "inner_critic_optimizer_steps": updates,
            }
            model._record_action_metrics(planned=True, action_seconds=0.01)
        metrics = model._wandb_train_window.snapshot()
        assert metrics["train/inner_cuda_graph_used"] == 0.5
        assert metrics["train/inner_cuda_graph_used_count"] == 2
    finally:
        model.env.close()
