"""Configuration contract for action-local adaptive parameter noise."""

import math
from types import SimpleNamespace

import gymnasium as gym
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2
from utils.resume_identity import scientific_trial_parameters
from utils.wandb_utils import WandbAccumulator


def _build_cfg(**params):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 12,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def _active(**overrides):
    params = {
        "inner_explorer_mode": "adaptive_param_noise",
        "inner_param_noise_actor_count": 8,
    }
    params.update(overrides)
    return _build_cfg(**params)


def _identity_trial(**algorithm_overrides):
    return {
        "alg": "AMBITDMPC2/AMBITDMPC2",
        "alg_params": dict(algorithm_overrides),
        "seed": 7,
    }


def test_adaptive_param_noise_defaults_are_inert():
    cfg = _build_cfg()

    assert cfg.inner_param_noise_active is False
    assert cfg.inner_param_noise_actor_count is None
    assert cfg.inner_param_noise_rollouts_per_actor == 0


def test_adaptive_param_noise_resolves_exact_policy_population():
    cfg = _active(inner_explorer_mode="ADAPTIVE_PARAM_NOISE")

    assert cfg.inner_explorer_mode == "adaptive_param_noise"
    assert cfg.inner_explorer_active is True
    assert cfg.inner_param_noise_active is True
    assert cfg.inner_explorer_trainable is False
    assert cfg.inner_explorer_has_separate_critic is False
    assert cfg.inner_primary_rollouts_per_round == 32
    assert cfg.inner_explorer_rollouts_per_round == 32
    assert cfg.inner_param_noise_actor_count == 8
    assert cfg.inner_param_noise_rollouts_per_actor == 4
    assert cfg.inner_primary_rollout_fraction == pytest.approx(0.5)
    assert cfg.inner_explorer_rollout_fraction == pytest.approx(0.5)
    assert cfg.inner_behavior_action == "policy_sample"
    assert cfg.inner_execution_policy_source == "primary"
    for component in ("actor", "critic", "temperature"):
        assert getattr(cfg, f"inner_explorer_{component}_updates_per_round") == 0
        assert getattr(cfg, f"inner_explorer_{component}_updates_per_action") == 0


def test_adaptive_param_noise_resolves_nondefault_exact_p_e_r_partition():
    cfg = _active(
        inner_rollouts_per_round=12,
        inner_prior_rollout_weight=0.25,
        inner_param_noise_actor_count=3,
    )

    assert cfg.inner_primary_rollouts_per_round == 3
    assert cfg.inner_explorer_rollouts_per_round == 9
    assert cfg.inner_param_noise_actor_count == 3
    assert cfg.inner_param_noise_rollouts_per_actor == 3


def test_adaptive_param_noise_requires_explicit_positive_actor_count():
    with pytest.raises(ValueError, match="explicit positive.*actor_count"):
        _build_cfg(inner_explorer_mode="adaptive_param_noise")

    for value in (True, False, 0, -1, 1.5, "8"):
        with pytest.raises(ValueError, match="inner_param_noise_actor_count"):
            _build_cfg(
                inner_explorer_mode="adaptive_param_noise",
                inner_param_noise_actor_count=value,
            )


def test_param_noise_actor_count_is_rejected_for_other_modes():
    for mode in ("none", "frozen_random", "shared_mixture", "separate_critics"):
        with pytest.raises(ValueError, match="only valid.*adaptive_param_noise"):
            _build_cfg(
                inner_explorer_mode=mode,
                inner_param_noise_actor_count=1,
            )


def test_adaptive_param_noise_actor_allocation_must_be_exact():
    with pytest.raises(ValueError, match="cannot exceed.*explorer_rollouts"):
        _active(
            inner_rollouts_per_round=12,
            inner_param_noise_actor_count=7,
        )
    with pytest.raises(ValueError, match="exactly divisible"):
        _active(
            inner_rollouts_per_round=12,
            inner_param_noise_actor_count=4,
        )


@pytest.mark.parametrize("weight", [1e-12, 1.0 - 1e-12])
def test_adaptive_param_noise_rejects_rounded_empty_populations(weight):
    with pytest.raises(ValueError, match="non-empty primary and explorer"):
        _active(
            inner_rollouts_per_round=64,
            inner_prior_rollout_weight=weight,
            inner_param_noise_actor_count=1,
        )


def test_adaptive_param_noise_requires_sampled_behavior_and_primary_execution():
    with pytest.raises(ValueError, match="inner_behavior_action='policy_sample'"):
        _active(inner_behavior_action="mean")
    with pytest.raises(ValueError, match="positive inner_behavior_std_scale"):
        _active(inner_behavior_std_scale=0.0)
    with pytest.raises(ValueError, match="inner_execution_policy_source='primary'"):
        _active(inner_execution_policy_source="explorer")


@pytest.mark.parametrize(
    "key",
    [
        "inner_explorer_actor_updates_per_round",
        "inner_explorer_critic_updates_per_round",
        "inner_explorer_temperature_updates_per_round",
    ],
)
def test_adaptive_param_noise_has_no_explorer_optimizers(key):
    with pytest.raises(ValueError, match="has no explorer optimizers"):
        _active(**{key: 1})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("inner_param_noise_target_action_rms", True),
        ("inner_param_noise_target_action_rms", "0.1"),
        ("inner_param_noise_target_action_rms", -0.1),
        ("inner_param_noise_target_action_rms", 0.0),
        ("inner_param_noise_target_action_rms", 2.0),
        ("inner_param_noise_target_action_rms", math.inf),
        ("inner_param_noise_target_action_rms", math.nan),
    ],
)
def test_adaptive_param_noise_target_action_rms_is_strict(key, value):
    with pytest.raises(ValueError, match=key):
        _active(**{key: value})


@pytest.mark.parametrize(
    ("sigma_min", "sigma_init", "sigma_max"),
    [
        (0.0, 1e-3, 0.1),
        (1e-3, 1e-4, 0.1),
        (1e-6, 0.2, 0.1),
        (1e-6, math.inf, math.inf),
    ],
)
def test_adaptive_param_noise_sigma_interval_is_ordered(
    sigma_min, sigma_init, sigma_max
):
    with pytest.raises(
        ValueError, match="inner_param_noise_sigma|scales must satisfy"
    ):
        _active(
            inner_param_noise_sigma_min=sigma_min,
            inner_param_noise_sigma_init=sigma_init,
            inner_param_noise_sigma_max=sigma_max,
        )


@pytest.mark.parametrize(
    "key",
    [
        "inner_param_noise_calibration_directions",
        "inner_param_noise_calibration_batch_size",
        "inner_param_noise_calibration_max_probes",
    ],
)
@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "8"])
def test_adaptive_param_noise_calibration_counts_are_positive_integers(key, value):
    with pytest.raises(ValueError, match=key):
        _active(**{key: value})


def test_param_noise_defaults_participate_in_scientific_identity():
    omitted = scientific_trial_parameters(_identity_trial())
    explicit = scientific_trial_parameters(
        _identity_trial(
            inner_param_noise_actor_count=None,
            inner_param_noise_target_action_rms=0.1,
            inner_param_noise_sigma_init=1e-3,
            inner_param_noise_sigma_min=1e-6,
            inner_param_noise_sigma_max=0.1,
            inner_param_noise_calibration_directions=8,
            inner_param_noise_calibration_batch_size=32,
            inner_param_noise_calibration_max_probes=8,
        )
    )
    assert omitted == explicit

    for key, value in (
        ("inner_param_noise_actor_count", 4),
        ("inner_param_noise_target_action_rms", 0.2),
        ("inner_param_noise_sigma_init", 2e-3),
        ("inner_param_noise_sigma_min", 2e-6),
        ("inner_param_noise_sigma_max", 0.2),
        ("inner_param_noise_calibration_directions", 4),
        ("inner_param_noise_calibration_batch_size", 16),
        ("inner_param_noise_calibration_max_probes", 4),
    ):
        changed = scientific_trial_parameters(_identity_trial(**{key: value}))
        assert changed != omitted, key


def test_param_noise_identity_canonicalizes_behavior_and_inner_logstd():
    canonical = scientific_trial_parameters(
        _identity_trial(
            inner_explorer_mode="adaptive_param_noise",
            inner_param_noise_actor_count=4,
            inner_behavior_action="policy_sample",
            inner_behavior_std_scale=1.0,
            inner_log_std_mapping="direct_clamp",
            inner_log_std_min=-20,
            inner_log_std_max=2,
        )
    )
    equivalent = scientific_trial_parameters(
        _identity_trial(
            inner_explorer_mode="ADAPTIVE_PARAM_NOISE",
            inner_param_noise_actor_count=4,
        )
    )
    assert equivalent == canonical
    assert scientific_trial_parameters(
        _identity_trial(
            inner_explorer_mode="adaptive_param_noise",
            inner_param_noise_actor_count=4,
            inner_behavior_std_scale=2.0,
        )
    ) != canonical
    assert scientific_trial_parameters(
        _identity_trial(
            inner_explorer_mode="adaptive_param_noise",
            inner_param_noise_actor_count=4,
            inner_log_std_mapping="tdmpc2_tanh",
        )
    ) != canonical


def test_parameter_noise_metrics_use_explicit_wandb_aggregation():
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.agent = SimpleNamespace(
        last_inner_rollout_lengths=[],
        last_inner_metrics={
            "inner_active": 1.0,
            "inner_rollouts": 8,
            "inner_steps": 24,
            "inner_param_noise_actor_count": 4,
            "inner_param_noise_rollouts_per_actor": 64,
            "inner_param_noise_target_action_rms": 0.1,
            "inner_param_noise_sigma_initial": 0.001,
            "inner_param_noise_sigma_final": 0.004,
            "inner_param_noise_sigma_mean": 0.003,
            "inner_param_noise_sigma_min": 0.001,
            "inner_param_noise_sigma_max": 0.004,
            "inner_param_noise_calibration_probes": 5,
            "inner_param_noise_calibration_policy_evaluations": 320,
            "inner_param_noise_calibration_rounds": 4,
            "inner_param_noise_calibration_action_rms_count": 5,
            "inner_param_noise_calibration_action_rms_mean": 0.098,
            "inner_param_noise_calibration_action_rms_std": 0.01,
            "inner_param_noise_calibration_action_rms_min": 0.08,
            "inner_param_noise_calibration_action_rms_max": 0.11,
            "inner_param_noise_calibration_target_hit_fraction": 0.75,
            "inner_param_noise_sigma_bound_hit_fraction": 0.0,
            "inner_param_noise_behavior_action_rms_count": 192,
            "inner_param_noise_behavior_action_rms_mean": 0.105,
            "inner_param_noise_behavior_action_rms_std": 0.02,
            "inner_param_noise_behavior_action_rms_min": 0.06,
            "inner_param_noise_behavior_action_rms_max": 0.15,
            "inner_param_noise_mean_action_saturation_count": 384,
            "inner_param_noise_mean_action_saturation_fraction": 0.01,
            "inner_param_noise_calibration_seconds": 0.02,
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

    assert payload["train/inner_param_noise_calibration_probes"] == 5
    assert payload[
        "train/inner_param_noise_calibration_policy_evaluations"
    ] == 320
    assert payload["train/inner_param_noise_actor_count"] == pytest.approx(4)
    assert payload["train/inner_param_noise_rollouts_per_actor"] == pytest.approx(
        64
    )
    assert payload["train/inner_param_noise_sigma_final"] == pytest.approx(0.004)
    assert payload[
        "train/inner_param_noise_behavior_action_rms_mean"
    ] == pytest.approx(0.105)
    assert payload["time/inner_param_noise_calibration_seconds"] == pytest.approx(
        0.02
    )

    # Population diagnostics pool by their actual row/probe denominators,
    # rather than averaging unequal per-action summaries with unit weight.
    for count, mean in ((1, 0.0), (9, 1.0)):
        metrics = algorithm.agent.last_inner_metrics
        metrics["inner_param_noise_behavior_action_rms_count"] = count
        metrics["inner_param_noise_behavior_action_rms_mean"] = mean
        metrics["inner_param_noise_behavior_action_rms_std"] = 0.0
        metrics["inner_param_noise_behavior_action_rms_min"] = mean
        metrics["inner_param_noise_behavior_action_rms_max"] = mean
        metrics["inner_param_noise_mean_action_saturation_count"] = count
        metrics["inner_param_noise_mean_action_saturation_fraction"] = mean
        algorithm._record_action_metrics(planned=True, action_seconds=0.0)
    pooled = algorithm._wandb_train_window.pop()
    assert pooled[
        "train/inner_param_noise_behavior_action_rms_mean"
    ] == pytest.approx(0.9)
    assert pooled[
        "train/inner_param_noise_mean_action_saturation_fraction"
    ] == pytest.approx(0.9)
