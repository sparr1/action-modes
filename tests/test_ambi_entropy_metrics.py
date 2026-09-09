"""Scaled statistics reach training logs without relabeling action entropy."""

import pytest

from tests.test_ambi_inner_decoupling import _model


def test_scaled_inner_and_explorer_metrics_reach_the_weighted_training_window():
    holder = _model()
    try:
        holder.agent.last_inner_metrics = {
            "inner_actor_optimizer_steps": 3,
            "inner_explorer_actor_optimizer_steps": 5,
            "inner_actor_entropy": -0.2,
            "inner_actor_scaled_entropy": 4.0,
            "inner_actor_entropy_bonus": 0.04,
            "inner_explorer_actor_entropy": -1.0,
            "inner_explorer_actor_scaled_entropy": 6.0,
            "inner_explorer_actor_entropy_bonus": 0.12,
        }
        holder._record_action_metrics(planned=True, action_seconds=0.0)
        snapshot = holder._wandb_train_window.snapshot()
        for prefix, count in (("inner_", 3), ("inner_explorer_", 5)):
            for name in ("actor_entropy", "actor_scaled_entropy", "actor_entropy_bonus"):
                key = prefix + name
                assert snapshot[f"train/{key}"] == pytest.approx(holder.agent.last_inner_metrics[key])
                assert snapshot[f"train/{key}_count"] == count
    finally:
        holder.close()


def test_default_training_window_does_not_add_scaled_metrics():
    holder = _model()
    try:
        holder.agent.last_inner_metrics = {"inner_actor_entropy": 0.2}
        holder._record_action_metrics(planned=True, action_seconds=0.0)
        snapshot = holder._wandb_train_window.snapshot()
        assert snapshot["train/inner_actor_entropy"] == 0.2
        assert not any("scaled_entropy" in key or "entropy_bonus" in key for key in snapshot)
    finally:
        holder.close()
