from types import SimpleNamespace

from main import _resolved_runtime_metadata
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from utils.resume_identity import scientific_trial_parameters


def _identity_trial(**algorithm_overrides):
    return {
        "alg": "AMBITDMPC2/AMBITDMPC2",
        "alg_params": dict(algorithm_overrides),
        "seed": 7,
    }


def test_omitted_random_explorer_controls_match_explicit_defaults():
    omitted = scientific_trial_parameters(_identity_trial())
    explicit = scientific_trial_parameters(
        _identity_trial(
            inner_explorer_mode="none",
            inner_prior_rollout_weight=0.5,
            inner_mixture_target_estimator="stratified",
            inner_explorer_actor_updates_per_round=None,
            inner_explorer_critic_updates_per_round=None,
            inner_explorer_temperature_updates_per_round=None,
            inner_execution_policy_source="primary",
            inner_execution_handoff_samples=8,
        )
    )

    assert omitted == explicit


def test_random_explorer_controls_change_scientific_identity():
    baseline = scientific_trial_parameters(_identity_trial())
    explorer = scientific_trial_parameters(
        _identity_trial(inner_explorer_mode="frozen_random")
    )

    assert baseline != explorer


def _agent_stub(mode):
    agent = object.__new__(AMBITDMPC2Agent)
    agent.cfg = SimpleNamespace(
        outer_critic_target="entropy_augmented",
        inner_sac_critic_target="entropy_augmented",
        inner_explorer_mode=mode,
        inner_prior_rollout_weight=0.5,
        inner_mixture_target_estimator="stratified",
        inner_primary_rollouts_per_round=64 if mode == "none" else 32,
        inner_explorer_rollouts_per_round=0 if mode == "none" else 32,
        inner_primary_target_rows_per_batch=None,
        inner_explorer_target_rows_per_batch=None,
        inner_rounds=4,
        inner_batch_size=32,
        inner_component_update_schedule=False,
        inner_nominal_updates_per_round=0,
        inner_expected_update_slots=0,
        inner_primary_actor_updates_per_round=0,
        inner_primary_critic_updates_per_round=0,
        inner_primary_temperature_updates_per_round=0,
        inner_explorer_actor_updates_per_round=0,
        inner_explorer_critic_updates_per_round=0,
        inner_explorer_temperature_updates_per_round=0,
        inner_primary_optimizer_steps_per_action=0,
        inner_explorer_optimizer_steps_per_action=0,
        inner_total_optimizer_steps_per_action=0,
        inner_execution_policy_source="primary",
        inner_execution_handoff_samples=8,
    )
    return agent


def test_legacy_exact_target_spec_migrates_only_when_explorer_is_disabled():
    legacy = {
        "outer_critic_target": "entropy_augmented",
        "inner_sac_critic_target": "entropy_augmented",
    }
    disabled = _agent_stub("none")
    active = _agent_stub("frozen_random")

    assert disabled._normalize_saved_critic_target_spec(legacy) == (
        disabled._critic_target_spec()
    )
    assert active._normalize_saved_critic_target_spec(legacy) == legacy


def test_exact_population_spec_records_resolved_primary_and_target_row_doses():
    baseline = _agent_stub("shared_mixture")
    changed_primary = _agent_stub("shared_mixture")
    changed_primary.cfg.inner_primary_actor_updates_per_round = 3
    changed_rows = _agent_stub("shared_mixture")
    changed_rows.cfg.inner_primary_target_rows_per_batch = 17
    changed_rows.cfg.inner_explorer_target_rows_per_batch = 15

    assert baseline._critic_target_spec() != changed_primary._critic_target_spec()
    assert baseline._critic_target_spec() != changed_rows._critic_target_spec()


def test_resolved_runtime_metadata_records_population_and_compute_contract():
    cfg = SimpleNamespace(
        inner_explorer_mode="separate_critics",
        inner_prior_rollout_weight=0.5,
        inner_mixture_target_estimator="weighted",
        inner_explorer_actor_updates_per_round=3,
        inner_explorer_critic_updates_per_round=2,
        inner_explorer_temperature_updates_per_round=1,
        inner_execution_policy_source="outer_q_gate",
        inner_execution_handoff_samples=8,
        inner_explorer_active=True,
        inner_explorer_trainable=True,
        inner_explorer_has_separate_critic=True,
        inner_primary_rollouts_per_round=32,
        inner_explorer_rollouts_per_round=32,
        inner_primary_rollout_fraction=0.5,
        inner_explorer_rollout_fraction=0.5,
        inner_primary_target_rows_per_batch=None,
        inner_explorer_target_rows_per_batch=None,
        inner_primary_actor_updates_per_round=3,
        inner_primary_critic_updates_per_round=2,
        inner_primary_temperature_updates_per_round=1,
        inner_primary_actor_updates_per_action=12,
        inner_primary_critic_updates_per_action=8,
        inner_primary_temperature_updates_per_action=4,
        inner_explorer_actor_updates_per_action=12,
        inner_explorer_critic_updates_per_action=8,
        inner_explorer_temperature_updates_per_action=4,
        inner_primary_optimizer_steps_per_action=24,
        inner_explorer_optimizer_steps_per_action=24,
        inner_total_optimizer_steps_per_action=48,
    )
    model = SimpleNamespace(cfg=cfg, agent=None, env=None)

    metadata = _resolved_runtime_metadata(
        model,
        trial_run_params={"alg": "AMBITDMPC2/AMBITDMPC2", "seed": 7},
    )
    inner = metadata["inner_budget"]

    assert inner["inner_explorer_mode"] == "separate_critics"
    assert inner["inner_primary_rollouts_per_round"] == 32
    assert inner["inner_explorer_rollouts_per_round"] == 32
    assert inner["inner_total_optimizer_steps_per_action"] == 48
