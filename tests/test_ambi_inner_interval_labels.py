"""Interval campaigns expose their changed update dose and learning rate."""

from copy import deepcopy

import pytest

from utils.ambi_benchmark import benchmark_run_labels


def _labels(interval=128, actor_lr=5e-5, **overrides):
    params = dict(inner_operator="sac", inner_rounds=6, inner_rollouts_per_round=512,
                  inner_rollout_horizon=3, inner_steps_per_update=interval,
                  inner_temperature_mode="auto", inner_actor_lr=actor_lr,
                  inner_bootstrap_source="inner_target", **overrides)
    checkpoint = {"metadata": {"checkpoint": {"step": 300000}}, "sha256": "a" * 64}
    protocol = {"environment": {"params": {"task": "humanoid-walk"}}}
    config = {"alg_params": params}
    before = deepcopy(config)
    result = benchmark_run_labels(checkpoint, protocol, config, "episodes")
    assert config == before
    return result


@pytest.mark.parametrize("interval,total", [(512, 18), (256, 36), (128, 72)])
def test_interval_labels_expose_nominal_joint_temperature_budget(interval, total):
    result = _labels(interval)
    assert f"update/{interval} transitions" in result["name"]
    assert f"nominal C{total}/A{total}/T{total} per action (joint)" in result["name"]
    assert {"schedule:transitions", "update-order:joint", f"steps-per-update:{interval}",
            f"C-nominal-per-action:{total}", f"A-nominal-per-action:{total}",
            f"T-nominal-per-action:{total}"} <= set(result["tags"])


def test_half_actor_learning_rate_cannot_collide_with_standard_interval_name():
    standard, half = _labels(), _labels(actor_lr=2.5e-5)
    assert standard["name"] != half["name"]
    assert "actor LR 5e-05" in standard["name"]
    assert "actor LR 2.5e-05" in half["name"]
    assert "actor-lr:5e-05" in standard["tags"]
    assert "actor-lr:2.5e-05" in half["tags"]


def test_episodic_totals_remain_explicitly_nominal():
    result = _labels(episodic=True)
    assert "nominal C72/A72/T72 per action" in result["name"]
    assert "C-per-action:72" not in result["tags"]


def test_explorer_schedule_does_not_claim_primary_totals_are_all_optimizer_work():
    result = _labels(inner_explorer_mode="separate_critics")
    assert "nominal C" not in result["name"]
    assert "update-order:joint" not in result["tags"]
    assert "actor LR 5e-05" in result["name"]
