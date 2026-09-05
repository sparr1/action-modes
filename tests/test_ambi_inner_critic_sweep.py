"""Critic-dose screens retain the existing joint SAC schedule and actor dose."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_ambi_latency_contract import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _tiny_legacy_model, _tiny_model
from tests.test_checkpoint_research_configs import _build_cfg, checkpoint_context
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_humanoid_inner_critic_sweep.json"
BASE = ROOT / "configs/research/ambi_humanoid_inner_benchmark.json"


def _snapshot(agent):
    pool = agent.inner_engine._action_pool
    state = {}
    for name in ("actor", "critic", "critic_target", "actor_optim", "critic_optim",
                 "temperature_optim", "replay"):
        component = getattr(pool, name)
        state[name] = None if component is None else _clone_tree(component.state_dict())
    state["log_alpha"] = _clone_tree(pool.log_alpha)
    state["outer"] = _clone_tree(agent.model.state_dict())
    state["rng"] = _clone_tree(agent.inner_engine.rng.training_state_dict())
    return state


def test_matrix_selects_only_higher_critic_doses_and_preserves_named_settings(checkpoint_context):
    matrix = load_preset_matrix(MATRIX)
    assert normalize_selectors(matrix) == [
        "critic_budget/inner_target_c6", "critic_budget/inner_target_c12",
        "critic_budget/outer_target_c6", "critic_budget/outer_target_c12",
    ]
    assert matrix["source_run"] == "rwgao_b-brown-university/ambi/u13m14st"
    assert matrix["evaluation"]["bank_repetitions"] == 3
    assert matrix["evaluation"]["diagnostic_rollouts"] == 8
    assert matrix["evaluation"]["diagnostic_horizon"] == 3
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["max_steps"] == 500
    assert matrix["evaluation"]["wandb_project"] == "ambi-inner-bench"
    base = resolve_preset(BASE, "named_run/d512_4_j6", checkpoint_context=checkpoint_context)
    before = deepcopy(checkpoint_context)
    for bootstrap in ("inner_target", "outer_target"):
        for critic_count in (3, 6, 12):
            selected = resolve_preset(
                MATRIX, f"critic_budget/{bootstrap}_c{critic_count}",
                checkpoint_context=checkpoint_context,
            )
            expected = deepcopy(base["algorithm_config"])
            params = expected["alg_params"]
            params.pop("inner_rollouts_per_round")
            params.pop("inner_updates_per_round")
            params.update(
                inner_model_step_budget=9216,
                inner_critic_updates_per_action=critic_count * 6,
                inner_actor_updates_per_action=18,
                inner_temperature_updates_per_action=18,
                inner_bootstrap_source=bootstrap,
            )
            assert selected["algorithm_config"] == expected
            assert selected["environment"] == base["environment"]
            with pytest.warns(DeprecationWarning):
                cfg = _build_cfg(selected["algorithm_config"])
            assert cfg.inner_schedule_mode == "legacy"
            assert cfg.inner_component_update_schedule is False
            assert cfg.inner_rounds == 6
            assert cfg.inner_rollouts_per_round == cfg.inner_batch_size == 512
            assert cfg.inner_rollout_horizon == 3
            assert cfg.inner_model_step_budget == cfg.inner_replay_capacity == 9216
            assert cfg.inner_expected_update_slots == critic_count * 6
            assert cfg.inner_critic_updates_per_action == critic_count * 6
            assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 18
            assert cfg.inner_temperature_initialization == "inherit_outer"
            assert cfg.inner_temperature_mode == "auto"
    assert checkpoint_context == before


@pytest.mark.parametrize("bootstrap", ["inner_target", "outer_target"])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_equal_total_budget_is_exactly_equivalent_to_canonical_joint_g3(bootstrap, dropout):
    shared = dict(inner_rounds=2, inner_rollout_horizon=2,
                  inner_replay_capacity=8, inner_bootstrap_source=bootstrap,
                  dropout=dropout, inner_actor_lr=5e-5,
                  inner_critic_lr=1e-4, inner_temperature_lr=3e-4)
    canonical = _tiny_model(**shared, inner_rollouts_per_round=2, inner_updates_per_round=3)
    with pytest.warns(DeprecationWarning):
        totals = _tiny_legacy_model(
            **shared, inner_model_step_budget=8,
            inner_critic_updates_per_action=6, inner_actor_updates_per_action=6,
            inner_temperature_updates_per_action=6,
        )
    try:
        for seed in (55, 56):
            for model in (canonical, totals):
                model.agent.inner_engine.reset_for_evaluation(seed, reuse_action_pool=True)
            rng_before = torch.random.get_rng_state().clone()
            first = canonical.agent.act(torch.ones(3), t0=True, eval_mode=True, collect_diagnostics=False)
            second = totals.agent.act(torch.ones(3), t0=True, eval_mode=True, collect_diagnostics=False)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            _assert_tree_equal(_snapshot(canonical.agent), _snapshot(totals.agent))
            torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    finally:
        canonical.env.close()
        totals.env.close()


@pytest.mark.parametrize("bootstrap", ["inner_target", "outer_target"])
@pytest.mark.parametrize("critic_count", [6, 12])
def test_higher_critic_dose_keeps_three_joint_slots_then_only_critic_extras(bootstrap, critic_count):
    with pytest.warns(DeprecationWarning):
        model = _tiny_legacy_model(
            inner_rounds=2, inner_rollout_horizon=2, inner_model_step_budget=8,
            inner_replay_capacity=8, inner_critic_updates_per_action=critic_count * 2,
            inner_actor_updates_per_action=6, inner_temperature_updates_per_action=6,
            inner_bootstrap_source=bootstrap,
        )
    try:
        trace = InnerActionTrace()
        model.agent.act(torch.zeros(3), t0=True, eval_mode=True, collect_diagnostics=False, trace=trace)
        for round_index in (1, 2):
            updates = [event for event in trace.events
                       if event["phase"] == "update" and event["round_index"] == round_index]
            assert len(updates) == critic_count
            for slot, event in enumerate(updates, 1):
                assert event["critic_updates"] == (round_index - 1) * critic_count + slot
                assert event["actor_updates"] == (round_index - 1) * 3 + min(slot, 3)
                assert event["temperature_updates"] == event["actor_updates"]
                assert ("actor_loss" in event["metrics"]) is (slot <= 3)
                assert "critic_loss" in event["metrics"]
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_model_steps"] == 8
        assert metrics["inner_critic_optimizer_steps"] == critic_count * 2
        assert metrics["inner_actor_optimizer_steps"] == metrics["inner_temperature_optimizer_steps"] == 6
        assert metrics["inner_update_slots"] == critic_count * 2
    finally:
        model.env.close()
