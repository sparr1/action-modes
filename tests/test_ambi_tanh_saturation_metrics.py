from types import SimpleNamespace

import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from tests.test_ambi_random_explorer_engine_invariants import _explorer_model
from tests.test_ambi_root_local_sac import _tiny_component_model
from utils.wandb_utils import WandbAccumulator


_SATURATION_SUFFIXES = (
    "pre_tanh_abs_mean",
    "pre_tanh_abs_max",
    "pre_tanh_abs_ge_7p6_fraction",
    "action_exact_saturation_fraction",
)


def _saturation_keys(prefix):
    return {f"{prefix}{suffix}" for suffix in _SATURATION_SUFFIXES}


def _prepare_engine(model):
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    return engine


def test_tanh_saturation_statistics_match_crafted_coordinates_and_detach():
    pre_tanh_action = torch.tensor(
        [[-8.0, -7.5, 0.0, 7.7, 8.0]], requires_grad=True
    )
    action = torch.tensor(
        [[-1.0, -0.999, 0.0, 0.999, 1.0]], requires_grad=True
    )

    mean, maximum, jacobian_floor_fraction, exact_fraction = (
        td_math.tanh_saturation_statistics(pre_tanh_action, action)
    )

    assert mean == pytest.approx(6.24)
    assert maximum == pytest.approx(8.0)
    assert jacobian_floor_fraction == pytest.approx(3 / 5)
    assert exact_fraction == pytest.approx(2 / 5)
    assert not any(
        value.requires_grad
        for value in (mean, maximum, jacobian_floor_fraction, exact_fraction)
    )


def test_outer_actor_saturation_metrics_route_through_wandb_update_window():
    model = _tiny_component_model()
    try:
        agent = model.agent
        latent_states = torch.zeros(
            int(agent.cfg.train_unroll_horizon) + 1,
            int(agent.cfg.batch_size),
            int(agent.cfg.latent_dim),
            device=agent.device,
        )

        actor_metrics = agent._update_actor(latent_states)
        model._accumulate_train_metrics(actor_metrics)
        payload = model._wandb_update_window.floats(include_stats=True)

        for key in _saturation_keys("actor_"):
            wandb_key = f"train/{key}"
            expected = float(actor_metrics[key])
            assert payload[wandb_key] == pytest.approx(expected)
            assert payload[f"{wandb_key}_count"] == pytest.approx(1)
            assert payload[f"{wandb_key}_mean"] == pytest.approx(expected)
            assert payload[f"{wandb_key}_std"] == pytest.approx(0)
            assert payload[f"{wandb_key}_min"] == pytest.approx(expected)
            assert payload[f"{wandb_key}_max"] == pytest.approx(expected)
    finally:
        model.env.close()


@pytest.mark.parametrize("collect_diagnostics", [False, True])
@pytest.mark.parametrize(
    ("explorer_mode", "expected_policy_forwards"),
    [(None, 1), ("shared_mixture", 2), ("separate_critics", 2)],
)
def test_inner_actor_saturation_metrics_are_gated_and_reuse_policy_samples(
    monkeypatch,
    explorer_mode,
    expected_policy_forwards,
    collect_diagnostics,
):
    if explorer_mode is None:
        model = _tiny_component_model(
            inner_rounds=1,
            inner_rollouts_per_round=2,
            inner_rollout_horizon=1,
            inner_critic_updates_per_round=0,
            inner_actor_updates_per_round=1,
            inner_temperature_updates_per_round=0,
        )
    else:
        model = _explorer_model(
            explorer_mode,
            inner_critic_updates_per_round=0,
            inner_actor_updates_per_round=1,
            inner_temperature_updates_per_round=0,
        )
    try:
        engine = _prepare_engine(model)
        engine._collect_diagnostics = collect_diagnostics
        batch = {
            "z": torch.zeros(
                int(engine.cfg.inner_batch_size),
                int(engine.cfg.latent_dim),
                device=engine.device,
            )
        }
        policy_forwards = 0
        original_pi = engine.model.pi

        def counted_pi(*args, **kwargs):
            nonlocal policy_forwards
            policy_forwards += 1
            return original_pi(*args, **kwargs)

        monkeypatch.setattr(engine.model, "pi", counted_pi)
        if explorer_mode is None:
            metrics = engine._sac_policy_step(
                batch,
                update_temperature=False,
                update_actor=True,
                alpha=engine.alpha.detach(),
            )
        elif explorer_mode == "shared_mixture":
            metrics = engine._shared_mixture_policy_step(
                batch,
                update_actor=True,
                update_temperature=False,
            )
        else:
            metrics = engine._separate_policy_step(
                batch,
                update_primary_actor=True,
                update_explorer_actor=True,
                update_primary_temperature=False,
                update_explorer_temperature=False,
            )

        assert policy_forwards == expected_policy_forwards
        expected_keys = _saturation_keys("actor_")
        if explorer_mode is not None:
            expected_keys |= _saturation_keys("explorer_actor_")
        if collect_diagnostics:
            assert expected_keys <= metrics.keys()
            assert all(torch.isfinite(metrics[key]) for key in expected_keys)
        else:
            assert expected_keys.isdisjoint(metrics)
    finally:
        model.env.close()


def test_inner_saturation_prefixes_average_per_slot_and_route_by_actor_group():
    first = {}
    second = {}
    expected_means = {}
    for index, suffix in enumerate(_SATURATION_SUFFIXES):
        primary_key = f"actor_{suffix}"
        explorer_key = f"explorer_actor_{suffix}"
        first[primary_key] = 1.0 + index
        second[primary_key] = 3.0 + index
        first[explorer_key] = 10.0 + index
        second[explorer_key] = 14.0 + index
        expected_means[f"inner_{primary_key}"] = 2.0 + index
        expected_means[f"inner_{explorer_key}"] = 12.0 + index

    aggregated = InnerImprovementEngine._average_update_metrics([first, second])
    for key, expected_mean in expected_means.items():
        assert aggregated[key] == pytest.approx(expected_mean)
        expected_std = 2.0 if "explorer_actor_" in key else 1.0
        assert aggregated[f"{key}_std"] == pytest.approx(expected_std)
        assert aggregated[f"{key}_min"] == pytest.approx(
            expected_mean - expected_std
        )
        assert aggregated[f"{key}_max"] == pytest.approx(
            expected_mean + expected_std
        )

    algorithm = object.__new__(AMBITDMPC2)
    algorithm.agent = SimpleNamespace(
        last_inner_rollout_lengths=[],
        last_inner_metrics={
            "inner_active": 1.0,
            "inner_rollouts": 0,
            "inner_steps": 0,
            "inner_updates": 0,
            "inner_actor_optimizer_steps": 3,
            "inner_explorer_actor_optimizer_steps": 5,
            **aggregated,
        },
    )
    algorithm._wandb_train_window = WandbAccumulator()
    algorithm._wandb_inner_seconds = 0.0
    algorithm._wandb_inner_actions = 0
    algorithm._wandb_inner_steps = 0
    algorithm._wandb_outer_policy_seconds = 0.0
    algorithm._wandb_outer_policy_actions = 0
    algorithm._inner_steps_total = 0
    algorithm._inner_updates_total = 0
    algorithm._outer_policy_episode_selected = False

    algorithm._record_action_metrics(planned=True, action_seconds=0.0)
    payload = algorithm._wandb_train_window.pop()

    for key in _saturation_keys("inner_actor_"):
        assert payload[f"train/{key}"] == pytest.approx(aggregated[key])
        assert payload[f"train/{key}_count"] == pytest.approx(3)
    for key in _saturation_keys("inner_explorer_actor_"):
        assert payload[f"train/{key}"] == pytest.approx(aggregated[key])
        assert payload[f"train/{key}_count"] == pytest.approx(5)
