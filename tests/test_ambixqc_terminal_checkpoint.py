"""Terminal bootstrap selection is explicit and portable across old checkpoints."""

from copy import deepcopy

import pytest

from test_ambixqc_core import _batch, _tree_equal
from test_ambixqc_prior_checkpoint import wrappers


@pytest.mark.parametrize("selection", [None, True, "all", "outer_target", 1])
def test_terminal_selection_rejects_ambiguous_values(wrappers, selection):
    with pytest.raises(ValueError, match="inner_terminal_bootstrap"):
        wrappers(inner_terminal_bootstrap=selection)


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_legacy_and_current_checkpoints_load_with_terminal_outer_evaluation(wrappers, version):
    source = wrappers()
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
    native = wrappers().load(saved)
    assert _tree_equal(source.agent.frozen_outer_state(), native.agent.frozen_outer_state())
    target = wrappers(inner_terminal_bootstrap="outer")
    before = target.agent.frozen_outer_state()
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    assert _tree_equal(before, target.agent.frozen_outer_state())
    target.load(saved, frozen_evaluation=True)
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["checkpoint_version"] == version
    assert provenance["saved_semantic_signature"]["inner_terminal_bootstrap"] == "inner"
    assert provenance["evaluated_semantic_signature"]["inner_terminal_bootstrap"] == "outer"
    assert _tree_equal(source.agent.frozen_outer_state(), target.agent.frozen_outer_state())


def test_terminal_outer_checkpoint_round_trip_and_reverse_override(wrappers):
    source = wrappers(inner_terminal_bootstrap="outer")
    saved = deepcopy(source.agent.checkpoint_state())
    assert saved["checkpoint_version"] == 4
    assert saved["semantic_signature"]["inner_terminal_bootstrap"] == "outer"
    target = wrappers(inner_terminal_bootstrap="outer").load(saved)
    assert _tree_equal(source.agent.frozen_outer_state(), target.agent.frozen_outer_state())
    native = wrappers()
    with pytest.raises(ValueError, match="semantics"):
        native.load(saved)
    native.load(saved, frozen_evaluation=True)
    assert native.agent.checkpoint_evaluation_provenance["saved_semantic_signature"]["inner_terminal_bootstrap"] == "outer"


@pytest.mark.parametrize("mutation", ["missing", "invalid", "legacy_extra", "outer_lr"])
def test_terminal_checkpoint_preflight_stays_strict_and_atomic(wrappers, mutation):
    saved = deepcopy(wrappers().agent.checkpoint_state())
    if mutation == "missing":
        saved["semantic_signature"].pop("inner_terminal_bootstrap")
    elif mutation == "invalid":
        saved["semantic_signature"]["inner_terminal_bootstrap"] = "all_outer"
    elif mutation == "legacy_extra":
        saved["checkpoint_version"] = 2
    else:
        saved["semantic_signature"]["actor_lr"] *= 2
    target = wrappers(inner_terminal_bootstrap="outer")
    before = target.agent.frozen_outer_state()
    with pytest.raises((ValueError, TypeError)):
        target.load(saved, frozen_evaluation=True)
    assert _tree_equal(before, target.agent.frozen_outer_state())
