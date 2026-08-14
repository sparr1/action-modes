import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from utils.wandb_utils import WandbAccumulator, extract_reward_components, init_wandb


def test_accumulator_combines_weighted_sums_and_last_values_then_clears():
    metrics = WandbAccumulator()
    metrics.add_weighted("train/loss", 2.0, weight=1)
    metrics.add_weighted("train/loss", 6.0, weight=3)
    metrics.update_sums({"train/updates_since_log": 0, "train/inner_steps": 2})
    metrics.add_sum("train/inner_steps", 3)
    metrics.set_last("train/n_updates", 4)
    metrics.set_last("train/n_updates", 7)

    expected = {
        "train/loss": 5.0,
        "train/updates_since_log": 0.0,
        "train/inner_steps": 5.0,
        "train/n_updates": 7.0,
    }
    assert metrics.snapshot() == expected
    assert metrics.snapshot() == expected
    assert not metrics.empty
    assert metrics.pop() == expected
    assert metrics.empty
    assert metrics.snapshot() == {}


def test_accumulator_pools_raw_and_precomputed_population_moments():
    metrics = WandbAccumulator()
    metrics.add_stats("rollout/reward", [1.0, 3.0, np.nan])
    metrics.add_stats(
        "rollout/reward",
        count=2,
        mean=6.0,
        std=2.0,
        min_value=4.0,
        max_value=8.0,
    )

    payload = metrics.snapshot(clear=True)
    assert payload["rollout/reward_count"] == 4.0
    assert payload["rollout/reward_mean"] == pytest.approx(4.0)
    assert payload["rollout/reward_std"] == pytest.approx(math.sqrt(6.5))
    assert payload["rollout/reward_min"] == 1.0
    assert payload["rollout/reward_max"] == 8.0
    assert metrics.empty


def test_accumulator_omits_empty_and_nonfinite_observations_but_keeps_zero_sum():
    metrics = WandbAccumulator()
    metrics.add_weighted("train/loss", np.nan)
    metrics.add_weighted("train/zero_weight", 1.0, weight=0)
    metrics.add_sum("train/invalid_sum", np.inf)
    metrics.set_last("train/invalid_last", "not numeric")
    metrics.add_stats("rollout/reward", [np.nan, np.inf])
    assert metrics.empty
    assert metrics.snapshot() == {}

    metrics.add_sum("train/inactive_actions", 0)
    assert metrics.snapshot() == {"train/inactive_actions": 0.0}
    metrics.clear()
    assert metrics.empty


def test_accumulator_rejects_mixing_aggregation_modes_for_one_output_key():
    metrics = WandbAccumulator()
    metrics.add_sum("train/work", 1)
    with pytest.raises(ValueError, match="already using sum aggregation"):
        metrics.set_last("train/work", 2)

    metrics.clear()
    metrics.add_stats("rollout/reward", [1.0])
    with pytest.raises(ValueError, match="pooled statistics aggregation"):
        metrics.add_weighted("rollout/reward_mean", 1.0)


def test_extract_reward_components_uses_finite_numeric_scalars_and_normalized_names():
    info = {
        "reward_forward": 1.25,
        "reward_ctrl_cost": np.nan,
        "reward_duplicate": 4,
        "reward_boolean": True,
        "unrelated": 99,
        "reward_info": {
            "control cost": np.float32(-2.0),
            "duplicate": 3.0,
            "reward Healthy/Bonus": 0.5,
            "nonfinite": np.inf,
            "label": "1.0",
            "flag": False,
            "vector": np.array([1.0]),
        },
    }

    assert extract_reward_components(info) == {
        "rollout/reward_control_cost": -2.0,
        "rollout/reward_duplicate": 4.0,
        "rollout/reward_healthy_bonus": 0.5,
        "rollout/reward_forward": 1.25,
    }
    assert extract_reward_components(None) == {}
    assert extract_reward_components([info]) == {}


def test_init_wandb_forwards_online_group_and_defines_eval_metrics(monkeypatch):
    calls = {}
    defined = []
    run = object()

    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: calls.update(kwargs) or run,
        define_metric=lambda *args, **kwargs: defined.append((args, kwargs)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    result = init_wandb(
        {
            "wandb": True,
            "wandb_project": "ambi",
            "wandb_entity": "brown",
            "wandb_mode": "online",
            "wandb_group": "humanoid",
            "wandb_tags": ["benchmark"],
        },
        default_project="unused",
        run_name="seed-1",
        config={"seed": 1},
    )

    assert result is run
    assert calls == {
        "project": "ambi",
        "entity": "brown",
        "name": "seed-1",
        "config": {"seed": 1},
        "mode": "online",
        "tags": ["benchmark"],
        "group": "humanoid",
    }
    assert (("eval/*",), {"step_metric": "env_step"}) in defined
