"""Checkpoint interval presets preserve collection and clear inherited schedules."""

from copy import deepcopy
from pathlib import Path

import pytest

from tests.test_checkpoint_research_configs import _build_cfg, checkpoint_context
from utils.ambi_benchmark import protocol_for
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_humanoid_inner_interval_sweep.json"
BASE = ROOT / "configs/research/ambi_humanoid_inner_benchmark.json"
SETTINGS = ((512, 5e-5, "s512"), (256, 5e-5, "s256"),
            (128, 5e-5, "s128"), (128, 2.5e-5, "s128_half_actor_lr"))
COMPETING_KEYS = {
    "inner_updates_per_round", "inner_critic_updates_per_round",
    "inner_actor_updates_per_round", "inner_iterations", "inner_rollouts",
    "inner_horizon", "inner_updates_per_iteration", "inner_model_step_budget",
    "inner_critic_updates_per_action", "inner_actor_updates_per_action",
    "inner_temperature_updates_per_action", "inner_explorer_actor_updates_per_round",
    "inner_explorer_critic_updates_per_round", "inner_explorer_temperature_updates_per_round",
}


def test_interval_matrix_keeps_paired_protocol_and_changes_only_selected_controls(
    checkpoint_context,
):
    matrix = load_preset_matrix(MATRIX)
    before = deepcopy(matrix)
    context_before = deepcopy(checkpoint_context)
    expected_selectors = [
        f"interval_budget/{bootstrap}_{suffix}"
        for bootstrap in ("inner_target", "outer_target")
        for _, _, suffix in SETTINGS
    ]
    assert normalize_selectors(matrix) == expected_selectors
    assert matrix["source_run"] == "rwgao_b-brown-university/ambi/u13m14st"
    assert matrix["base_alg_config"] == "checkpoint"
    base_matrix = load_preset_matrix(BASE)
    assert {key: value for key, value in matrix["evaluation"].items()
            if key != "default_presets"} == {
                key: value for key, value in base_matrix["evaluation"].items()
                if key != "default_presets"}
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["max_steps"] == 500
    assert matrix["evaluation"]["controller_seed"] == 55

    base = resolve_preset(BASE, "named_run/d512_4_j6", checkpoint_context=checkpoint_context)
    for bootstrap in ("inner_target", "outer_target"):
        for interval, actor_lr, suffix in SETTINGS:
            selected = resolve_preset(
                MATRIX, f"interval_budget/{bootstrap}_{suffix}", matrix,
                checkpoint_context=checkpoint_context,
            )
            expected = deepcopy(base["algorithm_config"])
            expected["alg_params"].pop("inner_updates_per_round")
            expected["alg_params"].update(
                inner_steps_per_update=interval, inner_actor_lr=actor_lr,
                inner_bootstrap_source=bootstrap,
            )
            assert selected["algorithm_config"] == expected
            assert selected["environment"] == base["environment"]
            assert protocol_for(selected, 55, 500) == protocol_for(base, 55, 500)
            assert protocol_for(selected, 55, 500)["action_rule"] == "tanh_mean"
    assert matrix == before
    assert checkpoint_context == context_before


@pytest.mark.parametrize("bootstrap", ["inner_target", "outer_target"])
@pytest.mark.parametrize("interval,actor_lr,suffix", SETTINGS)
def test_interval_presets_remove_conflicting_checkpoint_budgets_and_resolve_joint_totals(
    checkpoint_context, bootstrap, interval, actor_lr, suffix,
):
    # Source snapshots can preserve obsolete schedule fields, including zeroes.
    # None of these may silently switch the interval sweep to a legacy schedule.
    checkpoint_context.trial_run_params["alg_params"].update(
        {key: 99 for key in COMPETING_KEYS}
    )
    context_before = deepcopy(checkpoint_context)
    resolved = resolve_preset(
        MATRIX, f"interval_budget/{bootstrap}_{suffix}",
        checkpoint_context=checkpoint_context,
    )
    params = resolved["algorithm_config"]["alg_params"]
    assert not COMPETING_KEYS.intersection(params)
    cfg = _build_cfg(resolved["algorithm_config"])
    count = 9216 // interval
    assert cfg.inner_schedule_mode == "canonical"
    assert cfg.inner_component_update_schedule is False
    assert cfg.inner_steps_per_update == interval
    assert cfg.inner_rounds == 6
    assert cfg.inner_rollouts_per_round == cfg.inner_batch_size == 512
    assert cfg.inner_rollout_horizon == 3
    assert cfg.inner_model_step_budget == cfg.inner_replay_capacity == 9216
    assert cfg.inner_expected_update_slots == count
    assert cfg.inner_nominal_updates_per_round == count // 6
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
            cfg.inner_temperature_updates_per_action) == (count, count, count)
    assert cfg.inner_temperature_mode == "auto"
    assert cfg.inner_temperature_initialization == "inherit_outer"
    assert cfg.inner_target_entropy == "inherit_outer"
    assert cfg.inner_actor_lr == actor_lr
    assert cfg.inner_critic_lr == 1e-4
    assert cfg.inner_temperature_lr == 3e-4
    assert cfg.inner_bootstrap_source == bootstrap
    assert cfg.inner_actor_writeback_coef == cfg.inner_critic_writeback_coef == 0
    assert cfg.inner_finite_horizon is False
    assert cfg.inner_outer_replay_fraction == 0
    for component in ("actor", "critic", "temperature", "replay",
                      "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
        assert getattr(cfg, f"inner_{component}_scope") == "action"
    assert checkpoint_context == context_before
