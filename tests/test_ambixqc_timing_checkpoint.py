"""Step scheduling remains explicit in configuration and portable checkpoints."""

from copy import deepcopy

import pytest

from test_ambixqc_core import _batch, _tree_equal
from test_ambixqc_prior_checkpoint import wrappers


@pytest.mark.parametrize("selection", [None, True, 1, [], "interval", "per_transition"])
def test_invalid_update_timing_is_rejected(wrappers, selection):
    with pytest.raises(ValueError, match="inner_update_timing"):
        wrappers(inner_update_timing=selection)


@pytest.mark.parametrize("delay", [True, 0, -1, 1.5, "not-a-delay"])
def test_invalid_inner_policy_delay_is_rejected(wrappers, delay):
    with pytest.raises(ValueError, match="inner_policy_delay"):
        wrappers(inner_policy_delay=delay)


def test_inner_policy_delay_only_changes_action_local_update_budget(wrappers):
    default = wrappers(inner_updates_per_round=6)
    every_step = wrappers(inner_update_timing="step", inner_updates_per_round=6,
                          inner_policy_delay=1)
    assert default.cfg.inner_policy_delay == default.cfg.xqc_policy_delay == 3
    assert every_step.cfg.xqc_policy_delay == every_step.agent.xqc_controller.config.policy_delay == 3
    assert every_step.cfg.inner_policy_delay == 1
    assert every_step.cfg.inner_actor_updates_per_action == 6
    assert default.cfg.inner_actor_updates_per_action == 2


@pytest.mark.parametrize("updates", [1, 3, 5])
def test_step_timing_rejects_unequal_depth_budgets(wrappers, updates):
    with pytest.raises(ValueError, match="divisible"):
        wrappers(inner_update_timing="step", inner_rollout_horizon=2,
                 inner_updates_per_round=updates)


def test_step_timing_compile_request_is_explicitly_unsupported(wrappers):
    with pytest.raises(ValueError, match="compile=false"):
        wrappers(inner_update_timing="step", inner_updates_per_round=2, compile=True)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_legacy_round_checkpoints_support_explicit_step_evaluation(wrappers, version):
    source = wrappers(inner_operator="none", inner_updates_per_round=2)
    source.agent._update(*_batch(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    saved["checkpoint_version"] = version
    if version < 4:
        saved["semantic_signature"].pop("inner_update_timing")
        saved["semantic_signature"].pop("inner_policy_delay")
    if version < 3:
        saved["semantic_signature"].pop("inner_terminal_bootstrap")
    if version == 1:
        saved["semantic_signature"].pop("collection_operator")
        saved["semantic_signature"].pop("action_contract")
    target = wrappers(inner_update_timing="step", inner_updates_per_round=2, inner_policy_delay=1,
                      inner_terminal_bootstrap="outer")
    before = target.agent.frozen_outer_state()
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    assert _tree_equal(before, target.agent.frozen_outer_state())
    target.load(saved, frozen_evaluation=True)
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["saved_semantic_signature"]["inner_update_timing"] == "round"
    assert provenance["evaluated_semantic_signature"]["inner_update_timing"] == "step"
    assert provenance["saved_semantic_signature"]["inner_policy_delay"] == source.cfg.xqc_policy_delay
    assert provenance["evaluated_semantic_signature"]["inner_policy_delay"] == 1
    assert _tree_equal(source.agent.frozen_outer_state(), target.agent.frozen_outer_state())


@pytest.mark.parametrize("delay", [1, 3])
def test_step_checkpoint_round_trip_and_reverse_frozen_override(wrappers, delay):
    source = wrappers(inner_update_timing="step", inner_updates_per_round=2,
                      inner_terminal_bootstrap="outer", inner_policy_delay=delay)
    source.agent._update(*_batch(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    assert saved["checkpoint_version"] == 4
    assert saved["semantic_signature"]["inner_update_timing"] == "step"
    restored = wrappers(inner_update_timing="step", inner_updates_per_round=2,
                        inner_terminal_bootstrap="outer", inner_policy_delay=delay).load(saved)
    assert _tree_equal(saved, restored.agent.checkpoint_state())
    target = wrappers(inner_updates_per_round=2, inner_terminal_bootstrap="outer")
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    target.load(saved, frozen_evaluation=True)
    assert _tree_equal(source.agent.frozen_outer_state(), target.agent.frozen_outer_state())


@pytest.mark.parametrize("mutation", ["missing", "invalid", "legacy_extra", "budget", "outer_lr", "missing_delay", "invalid_delay"])
def test_timing_preflight_rejects_invalid_state_atomically(wrappers, mutation):
    saved = deepcopy(wrappers(inner_update_timing="step", inner_updates_per_round=2).agent.checkpoint_state())
    signature = saved["semantic_signature"]
    if mutation == "missing":
        signature.pop("inner_update_timing")
    elif mutation == "invalid":
        signature["inner_update_timing"] = []
    elif mutation == "legacy_extra":
        saved["checkpoint_version"] = 3
    elif mutation == "budget":
        signature["inner_schedule"]["updates"] = 3
    elif mutation == "missing_delay":
        signature.pop("inner_policy_delay")
    elif mutation == "invalid_delay":
        signature["inner_policy_delay"] = 0
    else:
        signature["actor_lr"] *= 2
    target = wrappers(inner_updates_per_round=2)
    before = target.agent.frozen_outer_state()
    with pytest.raises(ValueError):
        target.load(saved, frozen_evaluation=True)
    assert _tree_equal(before, target.agent.frozen_outer_state())
