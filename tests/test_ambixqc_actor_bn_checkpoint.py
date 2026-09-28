"""Actor BatchNorm semantics survive portable loads and explicit evaluation overrides."""

from copy import deepcopy

import pytest

from test_ambixqc_core import _batch, _tree_equal
from test_ambixqc_prior_checkpoint import wrappers


def _legacy_checkpoint(agent, version):
    """Produce the actual version-specific schema, not a relabeled v6 state."""
    state = deepcopy(agent.checkpoint_state())
    state["checkpoint_version"] = version
    signature = state["semantic_signature"]
    signature.pop("inner_actor_bn_mode")
    if version < 5:
        for key in ("aux_return", "inner_critic_source", "inner_horizon_critic_source",
                    "inner_critic_target"):
            signature.pop(key)
    if version < 4:
        signature.pop("inner_update_timing")
        signature.pop("inner_policy_delay")
    if version < 3:
        signature.pop("inner_terminal_bootstrap")
    if version == 1:
        signature.pop("collection_operator")
        signature.pop("action_contract")
    return state


@pytest.mark.parametrize("mode", ["batch_update", "running"])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_v6_actor_bn_mode_round_trip_and_next_outer_update(wrappers, mode, auxiliary):
    settings = {"inner_actor_bn_mode": mode, "aux_return_mode": "xqc" if auxiliary else "off"}
    source = wrappers(**settings)
    source.agent._update(*_batch(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    assert saved["checkpoint_version"] == 6
    assert saved["semantic_signature"]["inner_actor_bn_mode"] == mode
    restored = wrappers(**settings).load(saved)
    assert _tree_equal(saved, restored.agent.checkpoint_state())
    source.agent._update(*_batch(source.agent))
    restored.agent._update(*_batch(restored.agent))
    assert _tree_equal(source.agent.checkpoint_state(), restored.agent.checkpoint_state())


@pytest.mark.parametrize("saved_mode,evaluated_mode", [
    ("batch_update", "running"), ("running", "batch_update"),
])
@pytest.mark.parametrize("source", ["xqc", "aux_return"])
def test_actor_bn_override_is_frozen_only_and_records_both_modes(
    wrappers, saved_mode, evaluated_mode, source,
):
    settings = {
        "aux_return_mode": "xqc" if source == "aux_return" else "off",
        "inner_critic_source": source, "inner_horizon_critic_source": source,
        "inner_terminal_bootstrap": "outer",
    }
    original = wrappers(inner_actor_bn_mode=saved_mode, **settings)
    original.agent._update(*_batch(original.agent))
    saved = deepcopy(original.agent.checkpoint_state())
    untouched = deepcopy(saved)
    target = wrappers(inner_actor_bn_mode=evaluated_mode, **settings)
    before = deepcopy(target.agent.checkpoint_state())
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    assert _tree_equal(before, target.agent.checkpoint_state())
    assert target.agent.checkpoint_evaluation_provenance is None
    target.load(saved, frozen_evaluation=True)
    assert _tree_equal(original.agent.frozen_outer_state(), target.agent.frozen_outer_state())
    assert _tree_equal(saved, untouched)
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["checkpoint_version"] == 6
    assert provenance["saved_semantic_signature"]["inner_actor_bn_mode"] == saved_mode
    assert provenance["evaluated_semantic_signature"]["inner_actor_bn_mode"] == evaluated_mode
    assert provenance["frozen_evaluation"] is True
    provenance["saved_semantic_signature"]["inner_actor_bn_mode"] = "tampered"
    assert target.agent.checkpoint_evaluation_provenance["saved_semantic_signature"]["inner_actor_bn_mode"] == saved_mode


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_legacy_checkpoint_defaults_to_batch_update_without_mutating_saved_schema(wrappers, version):
    source = wrappers()
    source.agent._update(*_batch(source.agent))
    saved = _legacy_checkpoint(source.agent, version)
    untouched = deepcopy(saved)
    batch = wrappers().load(saved)
    assert _tree_equal(source.agent.frozen_outer_state(), batch.agent.frozen_outer_state())
    assert batch.agent.checkpoint_evaluation_provenance["saved_semantic_signature"]["inner_actor_bn_mode"] == "batch_update"
    running = wrappers(inner_actor_bn_mode="running")
    before = deepcopy(running.agent.checkpoint_state())
    with pytest.raises(ValueError, match="semantics"):
        running.load(saved)
    assert _tree_equal(before, running.agent.checkpoint_state())
    running.load(saved, frozen_evaluation=True)
    provenance = running.agent.checkpoint_evaluation_provenance
    assert provenance["checkpoint_version"] == version
    assert provenance["saved_semantic_signature"]["inner_actor_bn_mode"] == "batch_update"
    assert provenance["evaluated_semantic_signature"]["inner_actor_bn_mode"] == "running"
    assert _tree_equal(source.agent.frozen_outer_state(), running.agent.frozen_outer_state())
    assert _tree_equal(saved, untouched)


def test_real_v5_auxiliary_schema_can_use_running_actor_bn_in_frozen_evaluation(wrappers):
    source = wrappers(aux_return_mode="xqc", inner_operator="none")
    source.agent._update(*_batch(source.agent))
    saved = _legacy_checkpoint(source.agent, 5)
    assert "aux_return" in saved
    assert "inner_actor_bn_mode" not in saved["semantic_signature"]
    target = wrappers(
        aux_return_mode="xqc", inner_actor_bn_mode="running",
        inner_critic_source="aux_return", inner_horizon_critic_source="aux_return",
        inner_terminal_bootstrap="outer",
    ).load(saved, frozen_evaluation=True)
    assert _tree_equal(source.agent.frozen_outer_state(), target.agent.frozen_outer_state())
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["checkpoint_version"] == 5
    assert provenance["saved_semantic_signature"]["inner_actor_bn_mode"] == "batch_update"
    assert provenance["evaluated_semantic_signature"]["inner_actor_bn_mode"] == "running"
    assert provenance["evaluated_semantic_signature"]["inner_critic_source"] == "aux_return"


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("mutation", [
    "missing", "unknown", "null", "boolean", "list", "case", "legacy_extra", "outer_lr",
])
def test_actor_bn_checkpoint_preflight_rejects_invalid_semantics_before_mutation(
    wrappers, frozen, mutation,
):
    source = wrappers(inner_actor_bn_mode="running")
    source.agent._update(*_batch(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    signature = saved["semantic_signature"]
    if mutation == "missing":
        signature.pop("inner_actor_bn_mode")
    elif mutation == "legacy_extra":
        saved["checkpoint_version"] = 5  # v5 must not claim v6-only semantics.
    elif mutation == "outer_lr":
        signature["actor_lr"] *= 2
    else:
        signature["inner_actor_bn_mode"] = {
            "unknown": "batch_no_update", "null": None, "boolean": True,
            "list": [], "case": "RUNNING",
        }[mutation]
    target = wrappers(inner_actor_bn_mode="running")
    target.agent._update(*_batch(target.agent))
    target.agent._update(*_batch(target.agent))
    # An existing load also makes rejection preserve prior provenance and the
    # evaluation guard, rather than just leaving their default values intact.
    target.load(deepcopy(target.agent.checkpoint_state()), frozen_evaluation=frozen)
    before = deepcopy(target.agent.checkpoint_state())
    outer_before = target.agent.frozen_outer_state()
    provenance_before = target.agent.checkpoint_evaluation_provenance
    with pytest.raises(ValueError):
        target.load(saved, frozen_evaluation=frozen)
    assert _tree_equal(before, target.agent.checkpoint_state())
    assert _tree_equal(outer_before, target.agent.frozen_outer_state())
    assert target.agent.checkpoint_evaluation_provenance == provenance_before
    assert target.agent._frozen_evaluation is frozen
