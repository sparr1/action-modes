import gymnasium as gym
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2


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


def test_q_representation_is_independent_of_model_size_expansion():
    scalar = _build_cfg(model_size=5, q_representation="scalar")
    assert scalar.num_q == 2

    inherited_ensemble = _build_cfg(model_size=5, q_representation="distributional")
    assert inherited_ensemble.num_q == 5

    explicit_ensemble = _build_cfg(
        model_size=5,
        q_representation="distributional",
        num_q=7,
        q_num_bins=51,
        q_vmin=-8,
        q_vmax=12,
    )
    assert explicit_ensemble.num_q == 7
    assert explicit_ensemble.q_num_bins == 51
    assert explicit_ensemble.q_bin_size == pytest.approx(0.4)


def test_sac_actor_loss_scale_defaults_and_normalizes_explicit_mode():
    default = _build_cfg()
    assert default.sac_actor_loss_scale_mode == "none"
    assert default.sac_actor_loss_scale_tau == pytest.approx(0.01)

    enabled = _build_cfg(
        sac_actor_loss_scale_mode="TDMPC2_PERCENTILE_RANGE",
        sac_actor_loss_scale_tau=0.2,
    )
    assert enabled.sac_actor_loss_scale_mode == "tdmpc2_percentile_range"
    assert enabled.sac_actor_loss_scale_tau == pytest.approx(0.2)


def test_critic_target_modes_default_to_entropy_and_normalize_independently():
    default = _build_cfg()
    assert default.outer_critic_target == "entropy_augmented"
    assert default.inner_sac_critic_target == "entropy_augmented"

    reward_return = _build_cfg(
        outer_critic_target="REWARD_ONLY",
        inner_sac_critic_target="reward_only",
    )
    assert reward_return.outer_critic_target == "reward_only"
    assert reward_return.inner_sac_critic_target == "reward_only"

    mixed = _build_cfg(
        outer_critic_target="reward_only",
        inner_sac_critic_target="ENTROPY_AUGMENTED",
    )
    assert mixed.outer_critic_target == "reward_only"
    assert mixed.inner_sac_critic_target == "entropy_augmented"


@pytest.mark.parametrize("key", ["outer_critic_target", "inner_sac_critic_target"])
@pytest.mark.parametrize("mode", [None, True, 1, "soft", "unknown"])
def test_critic_target_modes_are_strict(key, mode):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key: mode})


def test_all_head_q_reductions_are_configurable_independently():
    cfg = _build_cfg(
        q_representation="distributional",
        num_q=5,
        outer_q_target_reduction="MIN_ALL",
        outer_q_actor_reduction="mean_all",
        inner_q_target_reduction="min_all",
        inner_q_actor_reduction="MEAN_ALL",
    )

    assert cfg.outer_q_target_reduction == "min_all"
    assert cfg.outer_q_actor_reduction == "mean_all"
    assert cfg.inner_q_target_reduction == "min_all"
    assert cfg.inner_q_actor_reduction == "mean_all"


def test_canonical_inner_sac_can_adapt_only_the_actor():
    cfg = _build_cfg(
        inner_rounds=2,
        inner_rollouts_per_round=3,
        inner_rollout_horizon=2,
        inner_updates_per_round=4,
        inner_actor_adaptation="clone",
        inner_critic_adaptation="frozen",
        inner_temperature_mode="inherit_outer",
    )

    assert cfg.inner_actor_updates_per_action == 8
    assert cfg.inner_critic_updates_per_action == 0
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.inner_expected_update_slots == 8


@pytest.mark.parametrize(
    "mode",
    [None, True, 1, "percentile_range", "unknown"],
)
def test_sac_actor_loss_scale_mode_is_strict(mode):
    with pytest.raises(ValueError, match="sac_actor_loss_scale_mode"):
        _build_cfg(sac_actor_loss_scale_mode=mode)


@pytest.mark.parametrize(
    "tau",
    [None, True, "invalid", float("nan"), float("inf"), 0.0, -0.1, 1.1],
)
def test_sac_actor_loss_scale_tau_is_strict(tau):
    with pytest.raises(ValueError, match="sac_actor_loss_scale_tau"):
        _build_cfg(sac_actor_loss_scale_tau=tau)


def test_legacy_ant_compute_and_lora_resolve_exactly():
    with pytest.warns(DeprecationWarning):
        cfg = _build_cfg(
            model_size=5,
            inner_adaptation="lora",
            inner_iterations=2,
            inner_rollouts=32,
            inner_horizon=3,
            inner_updates_per_iteration=2,
            inner_buffer_size=192,
            inner_tau=1.0,
            inner_target_update_interval=1,
            lora_rank=8,
            lora_alpha=8.0,
            lora_dropout=0.0,
        )

    assert cfg.inner_model_step_budget == 192
    assert cfg.inner_rounds == 2
    assert cfg.inner_rollouts_per_round == 32
    assert cfg.inner_actor_updates_per_action == 4
    assert cfg.inner_critic_updates_per_action == 4
    assert cfg.inner_actor_adaptation == "lora"
    assert cfg.inner_critic_adaptation == "lora"
    assert cfg.inner_actor_lora_scale == 1.0
    assert cfg.inner_critic_lora_scale == 1.0
    # One-release aliases preserve old integrations.
    assert cfg.inner_iterations == 2
    assert cfg.inner_rollouts == 32
    assert cfg.inner_updates_per_iteration == 2


def test_conflicting_legacy_and_canonical_keys_fail_before_resolution():
    with pytest.raises(ValueError, match="Cannot mix legacy and canonical compute"):
        _build_cfg(inner_iterations=2, inner_rounds=2)
    with pytest.raises(ValueError, match="inner_adaptation"):
        _build_cfg(inner_adaptation="clone", inner_actor_adaptation="lora")
    with pytest.raises(ValueError, match="lora_alpha"):
        _build_cfg(lora_alpha=8.0, inner_actor_lora_scale=1.0)


def test_compute_and_component_controls_validate_independently():
    cfg = _build_cfg(
        inner_model_step_budget=48,
        inner_rounds=2,
        inner_rollout_horizon=3,
        inner_actor_adaptation="frozen",
        inner_actor_updates_per_action=0,
        inner_critic_adaptation="clone",
        inner_critic_updates_per_action=3,
    )
    assert cfg.inner_rollouts_per_round == 8
    assert cfg.inner_actor_updates_per_action == 0
    assert cfg.inner_critic_updates_per_action == 3

    with pytest.raises(ValueError, match="divisible"):
        _build_cfg(
            inner_model_step_budget=47,
            inner_rounds=2,
            inner_rollout_horizon=3,
        )
    with pytest.raises(ValueError, match="Positive actor updates"):
        _build_cfg(inner_actor_adaptation="frozen", inner_actor_updates_per_action=1)
    with pytest.raises(ValueError, match="Temperature updates require"):
        _build_cfg(inner_temperature_updates_per_action=1, inner_temperature_mode="fixed")


def test_long_control_horizon_is_an_explicit_warning_not_a_gate():
    with pytest.warns(UserWarning, match="model-bias"):
        cfg = _build_cfg(
            train_unroll_horizon=2,
            outer_planning_horizon=2,
            inner_rollout_horizon=3,
            inner_model_step_budget=96,
            inner_rounds=1,
        )
    assert cfg.inner_horizon_ratio == 1.5


def test_mppi_candidates_include_policy_prior_compute_overhead():
    cfg = _build_cfg(
        inner_operator="mppi",
        inner_model_step_budget=188,
        inner_rounds=2,
        inner_rollout_horizon=3,
        inner_mppi_num_pi_trajs=4,
        inner_mppi_num_elites=8,
    )
    # Four policy-prior trajectories cost 4 * (H-1) = 8 transitions to
    # generate. The remaining 180 fund 2 * 30 * H candidate evaluations.
    assert cfg.inner_mppi_num_samples == 30


def test_optimizer_scope_cannot_outlive_component_parameters():
    with pytest.raises(ValueError, match="cannot outlive"):
        _build_cfg(
            inner_actor_scope="action",
            inner_actor_optimizer_scope="episode",
        )


def test_actor_and_critic_target_controls_resolve_independently():
    cfg = _build_cfg(
        inner_actor_target_tau=0.2,
        inner_actor_target_update_interval=3,
        inner_critic_target_tau=0.7,
        inner_critic_target_update_interval=5,
    )
    assert cfg.inner_actor_target_tau == pytest.approx(0.2)
    assert cfg.inner_actor_target_update_interval == 3
    assert cfg.inner_critic_target_tau == pytest.approx(0.7)
    assert cfg.inner_critic_target_update_interval == 5


def test_actor_adam_epsilon_defaults_to_main_and_validates_independent_override():
    inherited = _build_cfg(adam_eps=2e-8)
    assert inherited.adam_eps == pytest.approx(2e-8)
    assert inherited.actor_adam_eps == pytest.approx(2e-8)

    split = _build_cfg(adam_eps=1e-8, actor_adam_eps=1e-5)
    assert split.adam_eps == pytest.approx(1e-8)
    assert split.actor_adam_eps == pytest.approx(1e-5)

    with pytest.raises(ValueError, match="actor_adam_eps"):
        _build_cfg(actor_adam_eps=float("nan"))
    with pytest.raises(ValueError, match="must be positive"):
        _build_cfg(actor_adam_eps=0.0)


def test_inert_noise_knob_combinations_and_nonfinite_values_fail_early():
    with pytest.raises(ValueError, match="noise_std requires"):
        _build_cfg(inner_behavior_action="mean", inner_behavior_noise_std=0.1)
    with pytest.raises(ValueError, match="std_scale only affects"):
        _build_cfg(inner_execution_action="mean", inner_execution_std_scale=0.5)
    for key in (
        "inner_actor_lr",
        "inner_temperature",
        "inner_behavior_noise_std",
        "inner_mppi_temperature",
        "inner_critic_target_tau",
    ):
        with pytest.raises(ValueError, match="finite"):
            _build_cfg(**{key: float("nan")})


def test_without_replacement_rejects_impossible_replay_capacity():
    with pytest.raises(ValueError, match="inner_replay_capacity"):
        _build_cfg(
            inner_replay_sampling="without_replacement",
            inner_batch_size=16,
            inner_replay_capacity=8,
        )


def test_operator_specific_noop_controls_are_rejected():
    with pytest.raises(ValueError, match="no imagined work"):
        _build_cfg(inner_operator="none", inner_model_step_budget=12)
    with pytest.raises(ValueError, match="no entropy temperature"):
        _build_cfg(inner_operator="td3", inner_temperature_mode="auto")
    with pytest.raises(ValueError, match="TD3-only"):
        _build_cfg(inner_outer_action_l2_coef=0.1)
    with pytest.raises(ValueError, match="SAC-only"):
        _build_cfg(
            inner_operator="td3",
            inner_temperature_mode="inherit_outer",
            inner_outer_policy_kl_coef=0.1,
        )
