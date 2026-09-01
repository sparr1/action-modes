from types import SimpleNamespace

import gymnasium as gym
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2
from utils.wandb_utils import WandbAccumulator


def _build_cfg(**params):
    """Resolve AMBI config without constructing networks or replay storage."""

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


def test_random_explorer_defaults_are_inert_and_resolved():
    cfg = _build_cfg()

    assert cfg.inner_explorer_mode == "none"
    assert cfg.inner_explorer_active is False
    assert cfg.inner_explorer_trainable is False
    assert cfg.inner_explorer_has_separate_critic is False
    assert cfg.inner_prior_rollout_weight == pytest.approx(0.5)
    assert cfg.inner_mixture_target_estimator == "stratified"
    assert cfg.inner_execution_policy_source == "primary"
    assert cfg.inner_execution_handoff_samples == 8
    assert cfg.inner_primary_rollouts_per_round == cfg.inner_rollouts_per_round
    assert cfg.inner_explorer_rollouts_per_round == 0
    assert cfg.inner_primary_rollout_fraction == pytest.approx(1.0)
    assert cfg.inner_explorer_rollout_fraction == pytest.approx(0.0)
    assert cfg.inner_primary_target_rows_per_batch is None
    assert cfg.inner_explorer_target_rows_per_batch is None
    assert cfg.inner_primary_updates_per_round_is_auto is True
    for component in ("actor", "critic", "temperature"):
        assert getattr(cfg, f"inner_explorer_{component}_updates_per_round") == 0
        assert getattr(cfg, f"inner_explorer_{component}_updates_per_action") == 0
        assert (
            getattr(
                cfg,
                f"inner_explorer_{component}_updates_inherit_primary",
            )
            is False
        )
    assert (
        cfg.inner_total_optimizer_steps_per_action
        == cfg.inner_primary_optimizer_steps_per_action
    )


@pytest.mark.parametrize(
    "requested",
    ["FROZEN_RANDOM", "SHARED_MIXTURE", "SEPARATE_CRITICS"],
)
def test_supported_explorer_modes_normalize_case(requested):
    cfg = _build_cfg(inner_explorer_mode=requested)
    assert cfg.inner_explorer_mode == requested.lower()
    assert cfg.inner_explorer_active is True


@pytest.mark.parametrize("value", [None, True, 1, "random", "mixture"])
def test_explorer_mode_is_strict(value):
    with pytest.raises(ValueError, match="inner_explorer_mode"):
        _build_cfg(inner_explorer_mode=value)


@pytest.mark.parametrize(
    ("key", "supported"),
    [
        (
            "inner_mixture_target_estimator",
            ["STRATIFIED", "WEIGHTED"],
        ),
        (
            "inner_execution_policy_source",
            [
                "PRIMARY",
                "EXPLORER",
                "MIXTURE_SAMPLE",
                "OUTER_Q_GATE",
                "OUTER_SOFT_HANDOFF",
            ],
        ),
    ],
)
def test_explorer_choice_controls_normalize_supported_values(key, supported):
    for requested in supported:
        overrides = {key: requested}
        if key == "inner_execution_policy_source" and requested != "PRIMARY":
            overrides["inner_explorer_mode"] = "frozen_random"
        cfg = _build_cfg(**overrides)
        assert getattr(cfg, key) == requested.lower()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("inner_mixture_target_estimator", None),
        ("inner_mixture_target_estimator", True),
        ("inner_mixture_target_estimator", "sampled"),
        ("inner_execution_policy_source", None),
        ("inner_execution_policy_source", 1),
        ("inner_execution_policy_source", "average"),
    ],
)
def test_explorer_choice_controls_are_strict(key, value):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key: value})


def test_frozen_random_splits_rollouts_but_has_no_explorer_optimizers():
    cfg = _build_cfg(
        inner_explorer_mode="frozen_random",
        inner_prior_rollout_weight=0.25,
        inner_rollouts_per_round=12,
    )

    assert cfg.inner_primary_rollouts_per_round == 3
    assert cfg.inner_explorer_rollouts_per_round == 9
    assert cfg.inner_primary_rollout_fraction == pytest.approx(0.25)
    assert cfg.inner_explorer_rollout_fraction == pytest.approx(0.75)
    assert cfg.inner_explorer_trainable is False
    assert cfg.inner_explorer_optimizer_steps_per_action == 0


def test_shared_mixture_inherits_only_primary_actor_dose():
    cfg = _build_cfg(
        inner_explorer_mode="shared_mixture",
        inner_rounds=2,
        inner_rollouts_per_round=6,
        inner_rollout_horizon=2,
        inner_critic_updates_per_round=3,
        inner_actor_updates_per_round=2,
    )

    assert cfg.inner_primary_actor_updates_per_round == 2
    assert cfg.inner_primary_critic_updates_per_round == 3
    assert cfg.inner_primary_temperature_updates_per_round == 2
    assert cfg.inner_primary_actor_updates_per_action == 4
    assert cfg.inner_primary_critic_updates_per_action == 6
    assert cfg.inner_primary_temperature_updates_per_action == 4
    assert cfg.inner_explorer_actor_updates_per_round == 2
    assert cfg.inner_explorer_critic_updates_per_round == 0
    assert cfg.inner_explorer_temperature_updates_per_round == 0
    assert cfg.inner_explorer_actor_updates_per_action == 4
    assert cfg.inner_explorer_optimizer_steps_per_action == 4
    assert cfg.inner_explorer_actor_updates_inherit_primary is True
    assert cfg.inner_explorer_critic_updates_inherit_primary is False
    assert cfg.inner_explorer_temperature_updates_inherit_primary is False


def test_separate_critics_inherit_all_primary_update_doses():
    cfg = _build_cfg(
        inner_explorer_mode="separate_critics",
        inner_rounds=3,
        inner_rollouts_per_round=6,
        inner_rollout_horizon=2,
        inner_critic_updates_per_round=4,
        inner_actor_updates_per_round=2,
    )

    assert cfg.inner_explorer_has_separate_critic is True
    assert cfg.inner_explorer_actor_updates_per_round == 2
    assert cfg.inner_explorer_critic_updates_per_round == 4
    assert cfg.inner_explorer_temperature_updates_per_round == 2
    assert cfg.inner_explorer_actor_updates_per_action == 6
    assert cfg.inner_explorer_critic_updates_per_action == 12
    assert cfg.inner_explorer_temperature_updates_per_action == 6
    assert cfg.inner_explorer_actor_updates_inherit_primary is True
    assert cfg.inner_explorer_critic_updates_inherit_primary is True
    assert cfg.inner_explorer_temperature_updates_inherit_primary is True


def test_separate_critics_accept_explicit_independent_update_doses():
    cfg = _build_cfg(
        inner_explorer_mode="separate_critics",
        inner_explorer_actor_updates_per_round=1,
        inner_explorer_critic_updates_per_round=2,
        inner_explorer_temperature_updates_per_round=3,
    )

    assert cfg.inner_explorer_actor_updates_per_round == 1
    assert cfg.inner_explorer_critic_updates_per_round == 2
    assert cfg.inner_explorer_temperature_updates_per_round == 3
    assert cfg.inner_explorer_actor_updates_per_action == 4
    assert cfg.inner_explorer_critic_updates_per_action == 8
    assert cfg.inner_explorer_temperature_updates_per_action == 12
    assert cfg.inner_explorer_actor_updates_inherit_primary is False
    assert cfg.inner_explorer_critic_updates_inherit_primary is False
    assert cfg.inner_explorer_temperature_updates_inherit_primary is False


def test_separate_critic_slot_budget_includes_larger_explorer_doses():
    joint = _build_cfg(
        inner_explorer_mode="separate_critics",
        inner_rounds=2,
        inner_rollouts_per_round=4,
        inner_rollout_horizon=2,
        inner_updates_per_round=1,
        inner_explorer_actor_updates_per_round=3,
        inner_explorer_critic_updates_per_round=4,
        inner_explorer_temperature_updates_per_round=2,
    )
    phased = _build_cfg(
        inner_explorer_mode="separate_critics",
        inner_rounds=2,
        inner_rollouts_per_round=4,
        inner_rollout_horizon=2,
        inner_critic_updates_per_round=1,
        inner_actor_updates_per_round=1,
        inner_explorer_actor_updates_per_round=3,
        inner_explorer_critic_updates_per_round=4,
        inner_explorer_temperature_updates_per_round=2,
    )

    assert joint.inner_nominal_updates_per_round == 4
    assert joint.inner_expected_update_slots == 8
    assert phased.inner_nominal_updates_per_round == 7
    assert phased.inner_expected_update_slots == 14


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("inner_explorer_actor_updates_per_round", True),
        ("inner_explorer_actor_updates_per_round", -1),
        ("inner_explorer_actor_updates_per_round", 1.5),
        ("inner_explorer_actor_updates_per_round", "1"),
        ("inner_explorer_critic_updates_per_round", False),
        ("inner_explorer_critic_updates_per_round", -1),
        ("inner_explorer_temperature_updates_per_round", 0.5),
        ("inner_explorer_temperature_updates_per_round", "auto"),
    ],
)
def test_explorer_update_counts_are_strict_nonnegative_integers(key, value):
    with pytest.raises(ValueError, match=key):
        _build_cfg(inner_explorer_mode="separate_critics", **{key: value})


@pytest.mark.parametrize("mode", ["none", "frozen_random"])
def test_modes_without_explorer_optimizers_reject_nonzero_doses(mode):
    with pytest.raises(ValueError, match="no explorer optimizer|optimizer updates"):
        _build_cfg(
            inner_explorer_mode=mode,
            inner_explorer_actor_updates_per_round=1,
        )


def test_shared_mixture_requires_equal_actor_counts_and_shared_q_alpha_counts():
    with pytest.raises(ValueError, match="equal primary and explorer actor"):
        _build_cfg(
            inner_explorer_mode="shared_mixture",
            inner_updates_per_round=2,
            inner_explorer_actor_updates_per_round=1,
        )
    with pytest.raises(ValueError, match="one shared critic"):
        _build_cfg(
            inner_explorer_mode="shared_mixture",
            inner_explorer_critic_updates_per_round=1,
        )
    with pytest.raises(ValueError, match="one shared temperature"):
        _build_cfg(
            inner_explorer_mode="shared_mixture",
            inner_explorer_temperature_updates_per_round=1,
        )


def test_shared_mixture_auto_actor_dose_must_inherit_realized_transitions():
    inherited = _build_cfg(inner_explorer_mode="shared_mixture")
    assert inherited.inner_primary_updates_per_round_is_auto is True
    assert inherited.inner_explorer_actor_updates_inherit_primary is True

    with pytest.raises(ValueError, match="follow the realized transition count"):
        _build_cfg(
            inner_explorer_mode="shared_mixture",
            inner_explorer_actor_updates_per_round=(
                inherited.inner_primary_actor_updates_per_round
            ),
        )


def test_explorer_temperature_updates_require_auto_temperature():
    fixed = _build_cfg(
        inner_explorer_mode="separate_critics",
        inner_temperature_mode="fixed",
    )
    assert fixed.inner_explorer_temperature_updates_per_round == 0

    with pytest.raises(ValueError, match="inner_temperature_mode='auto'"):
        _build_cfg(
            inner_explorer_mode="separate_critics",
            inner_temperature_mode="fixed",
            inner_explorer_temperature_updates_per_round=1,
        )


@pytest.mark.parametrize("weight", [0.25, 0.5, 0.75])
def test_rollout_and_stratified_target_partitions_are_exact(weight):
    cfg = _build_cfg(
        inner_explorer_mode="shared_mixture",
        inner_prior_rollout_weight=weight,
        inner_rollouts_per_round=8,
        inner_batch_size=12,
    )

    assert cfg.inner_primary_rollouts_per_round == round(8 * weight)
    assert cfg.inner_explorer_rollouts_per_round == 8 - round(8 * weight)
    assert cfg.inner_primary_target_rows_per_batch == round(12 * weight)
    assert cfg.inner_explorer_target_rows_per_batch == 12 - round(12 * weight)


def test_active_explorer_rejects_nonintegral_rollout_partition():
    with pytest.raises(ValueError, match="inner_rollouts_per_round.*integral"):
        _build_cfg(
            inner_explorer_mode="frozen_random",
            inner_rollouts_per_round=3,
            inner_prior_rollout_weight=0.5,
        )


def test_only_stratified_mixture_requires_integral_target_partition():
    with pytest.raises(ValueError, match="inner_batch_size.*integral"):
        _build_cfg(
            inner_explorer_mode="shared_mixture",
            inner_mixture_target_estimator="stratified",
            inner_rollouts_per_round=4,
            inner_batch_size=3,
            inner_prior_rollout_weight=0.5,
        )

    weighted = _build_cfg(
        inner_explorer_mode="shared_mixture",
        inner_mixture_target_estimator="weighted",
        inner_rollouts_per_round=4,
        inner_batch_size=3,
        inner_prior_rollout_weight=0.5,
    )
    assert weighted.inner_primary_target_rows_per_batch is None
    assert weighted.inner_explorer_target_rows_per_batch is None


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_active_explorer_requires_strictly_interior_prior_weight(weight):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        _build_cfg(
            inner_explorer_mode="frozen_random",
            inner_prior_rollout_weight=weight,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"inner_operator": "td3"}, "inner_operator='sac'"),
        (
            {
                "inner_model_step_budget": 192,
                "inner_critic_updates_per_action": 1,
                "inner_actor_updates_per_action": 1,
                "inner_temperature_updates_per_action": 0,
            },
            "canonical per-round schedule",
        ),
        ({"inner_actor_adaptation": "lora"}, "clone adaptation"),
        ({"inner_critic_adaptation": "lora"}, "clone adaptation"),
        ({"inner_actor_scope": "episode"}, "action-local state"),
        ({"inner_replay_scope": "run"}, "action-local state"),
        ({"inner_actor_optimizer_scope": "episode"}, "cannot outlive"),
        ({"inner_bootstrap_source": "outer_target"}, "inner_bootstrap_source"),
        ({"inner_actor_writeback_coef": 0.1}, "writeback"),
        ({"inner_critic_writeback_coef": 0.1}, "writeback"),
        ({"inner_outer_policy_kl_coef": 0.1}, "regularization coefficients"),
    ],
)
def test_active_explorer_rejects_incompatible_inner_contracts(overrides, match):
    with pytest.raises(ValueError, match=match):
        _build_cfg(inner_explorer_mode="frozen_random", **overrides)


def test_shared_mixture_requires_entropy_augmented_inner_target():
    with pytest.raises(ValueError, match="inner_sac_critic_target='entropy_augmented'"):
        _build_cfg(
            inner_explorer_mode="shared_mixture",
            inner_sac_critic_target="reward_only",
        )


def test_outer_soft_handoff_requires_entropy_augmented_outer_target():
    with pytest.raises(ValueError, match="outer_critic_target='entropy_augmented'"):
        _build_cfg(
            inner_explorer_mode="frozen_random",
            inner_execution_policy_source="outer_soft_handoff",
            outer_critic_target="reward_only",
        )


@pytest.mark.parametrize(
    "source",
    ["mixture_sample", "outer_soft_handoff"],
)
def test_stochastic_population_selectors_reject_active_behavior_policy_kl(source):
    with pytest.raises(ValueError, match="behavior-policy regularizer.*incompatible"):
        _build_cfg(
            inner_explorer_mode="frozen_random",
            inner_execution_policy_source=source,
            outer_behavior_policy_kl_schedule="smooth",
        )


@pytest.mark.parametrize("source", ["primary", "explorer", "outer_q_gate"])
def test_exact_component_execution_supports_active_behavior_policy_kl(source):
    cfg = _build_cfg(
        inner_explorer_mode="frozen_random",
        inner_execution_policy_source=source,
        outer_behavior_policy_kl_schedule="smooth",
    )
    assert cfg.store_behavior_policy is True


def test_nonprimary_execution_requires_an_active_explorer():
    with pytest.raises(ValueError, match="must be 'primary'"):
        _build_cfg(inner_execution_policy_source="explorer")


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "8"])
def test_execution_handoff_samples_is_strict_positive_integer(value):
    with pytest.raises(ValueError, match="inner_execution_handoff_samples"):
        _build_cfg(inner_execution_handoff_samples=value)


def test_explorer_metrics_are_routed_with_source_specific_weights():
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.agent = SimpleNamespace(
        last_inner_rollout_lengths=[],
        last_inner_metrics={
            "inner_active": 1.0,
            "inner_rollouts": 4,
            "inner_steps": 12,
            "inner_primary_rollouts": 2,
            "inner_explorer_rollouts": 2,
            "inner_primary_transitions": 5,
            "inner_explorer_transitions": 7,
            "inner_optimization_model_steps": 12,
            "inner_selector_model_steps": 16,
            "inner_explorer_actor_optimizer_steps": 2,
            "inner_explorer_critic_optimizer_steps": 3,
            "inner_explorer_temperature_optimizer_steps": 1,
            "inner_explorer_critic_target_updates": 2,
            "inner_selector_primary_wins": 1,
            "inner_selector_explorer_wins": 0,
            "inner_explorer_actor_grad_norm": 4.0,
            "inner_explorer_critic_loss": 5.0,
            "inner_explorer_temperature_loss": 6.0,
            "inner_primary_reward_mean": 1.5,
            "inner_primary_reward_std": 0.25,
            "inner_primary_reward_min": 1.0,
            "inner_primary_reward_max": 2.0,
            "inner_explorer_termination_rate_mean": 0.2,
            "inner_primary_replay_fraction": 0.4,
            "inner_explorer_replay_fraction": 0.6,
            "inner_primary_replay_samples": 9,
            "inner_explorer_replay_samples": 11,
            "inner_primary_replay_sample_fraction": 0.45,
            "inner_explorer_replay_sample_fraction": 0.55,
            "inner_primary_td_error_abs_mean": 0.25,
            "inner_primary_td_error_abs_count": 9,
            "inner_selector_score_margin": 0.75,
            "inner_primary_explorer_action_l2": 0.5,
            "inner_explorer_actor_trainable_params": 123,
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

    assert payload["train/inner_primary_rollouts"] == pytest.approx(2)
    assert payload["train/inner_explorer_transitions"] == pytest.approx(7)
    assert payload["train/inner_selector_model_steps"] == pytest.approx(16)
    assert payload["train/inner_explorer_actor_grad_norm_count"] == pytest.approx(2)
    assert payload["train/inner_explorer_critic_loss_count"] == pytest.approx(3)
    assert payload["train/inner_explorer_temperature_loss_count"] == pytest.approx(1)
    assert payload["train/inner_primary_reward_count"] == pytest.approx(5)
    assert payload["train/inner_explorer_termination_rate_count"] == pytest.approx(7)
    assert payload["train/inner_primary_replay_fraction"] == pytest.approx(0.4)
    assert payload["train/inner_primary_replay_samples"] == pytest.approx(9)
    assert payload["train/inner_primary_replay_sample_fraction"] == pytest.approx(
        0.45
    )
    assert payload["train/inner_primary_td_error_abs_mean"] == pytest.approx(0.25)
    assert payload["train/inner_primary_td_error_abs_count"] == pytest.approx(9)
    assert payload["train/inner_selector_score_margin"] == pytest.approx(0.75)
    assert payload["train/inner_primary_explorer_action_l2"] == pytest.approx(0.5)


def test_fixed_q_counterfactual_metrics_aggregate_per_action():
    algorithm = object.__new__(AMBITDMPC2)
    first = {
        "inner_active": 1.0,
        "inner_rollouts": 0,
        "inner_steps": 0,
        "inner_fixed_q_counterfactual_policy_evaluations": 2,
        "inner_fixed_q_counterfactual_q_evaluations": 2,
        "inner_fixed_q_counterfactual_primary_wins": 0,
        "inner_fixed_q_counterfactual_explorer_wins": 1,
        "inner_fixed_q_counterfactual_explorer_rate": 1.0,
        "inner_fixed_q_counterfactual_primary_q": 1.0,
        "inner_fixed_q_counterfactual_explorer_q": 3.0,
        "inner_fixed_q_counterfactual_margin": 2.0,
        "inner_fixed_q_counterfactual_execution_agreement": 0.0,
        "inner_fixed_q_counterfactual_action_l2_to_executed": 1.0,
        "inner_fixed_q_counterfactual_action_0": 0.5,
        "inner_fixed_q_counterfactual_action_1": -0.5,
    }
    second = {
        **first,
        "inner_fixed_q_counterfactual_primary_wins": 1,
        "inner_fixed_q_counterfactual_explorer_wins": 0,
        "inner_fixed_q_counterfactual_explorer_rate": 0.0,
        "inner_fixed_q_counterfactual_primary_q": 4.0,
        "inner_fixed_q_counterfactual_explorer_q": 2.0,
        "inner_fixed_q_counterfactual_margin": -2.0,
        "inner_fixed_q_counterfactual_execution_agreement": 1.0,
        "inner_fixed_q_counterfactual_action_l2_to_executed": 0.0,
        "inner_fixed_q_counterfactual_action_0": -0.5,
        "inner_fixed_q_counterfactual_action_1": 0.25,
    }
    algorithm.agent = SimpleNamespace(
        last_inner_rollout_lengths=[], last_inner_metrics=first
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
    algorithm.agent.last_inner_metrics = second
    algorithm._record_action_metrics(planned=True, action_seconds=0.0)
    payload = algorithm._wandb_train_window.pop()

    assert payload[
        "train/inner_fixed_q_counterfactual_policy_evaluations"
    ] == pytest.approx(4)
    assert payload[
        "train/inner_fixed_q_counterfactual_q_evaluations"
    ] == pytest.approx(4)
    assert payload[
        "train/inner_fixed_q_counterfactual_primary_wins"
    ] == pytest.approx(1)
    assert payload[
        "train/inner_fixed_q_counterfactual_explorer_wins"
    ] == pytest.approx(1)
    assert payload[
        "train/inner_fixed_q_counterfactual_explorer_rate"
    ] == pytest.approx(0.5)
    assert payload[
        "train/inner_fixed_q_counterfactual_primary_q"
    ] == pytest.approx(2.5)
    assert payload[
        "train/inner_fixed_q_counterfactual_explorer_q"
    ] == pytest.approx(2.5)
    assert payload[
        "train/inner_fixed_q_counterfactual_margin"
    ] == pytest.approx(0.0)
    assert payload[
        "train/inner_fixed_q_counterfactual_execution_agreement"
    ] == pytest.approx(0.5)
    assert payload[
        "train/inner_fixed_q_counterfactual_action_l2_to_executed"
    ] == pytest.approx(0.5)
    assert payload[
        "train/inner_fixed_q_counterfactual_action_0"
    ] == pytest.approx(0.0)
    assert payload[
        "train/inner_fixed_q_counterfactual_action_1"
    ] == pytest.approx(-0.125)
    for key in (
        "inner_fixed_q_counterfactual_explorer_rate",
        "inner_fixed_q_counterfactual_primary_q",
        "inner_fixed_q_counterfactual_explorer_q",
        "inner_fixed_q_counterfactual_margin",
        "inner_fixed_q_counterfactual_execution_agreement",
        "inner_fixed_q_counterfactual_action_l2_to_executed",
        "inner_fixed_q_counterfactual_action_0",
        "inner_fixed_q_counterfactual_action_1",
    ):
        assert payload[f"train/{key}_count"] == pytest.approx(2)
