"""Optional v7 inner BN/rate semantics remain explicit and backward compatible."""
from copy import deepcopy

import pytest

from test_ambixqc_actor_bn_checkpoint import _legacy_checkpoint
from test_ambixqc_core import _batch, _tree_equal
from test_ambixqc_prior_checkpoint import wrappers


def test_default_signature_preserves_existing_v7_schema(wrappers):
    signature = wrappers().agent.semantic_signature()
    assert "inner_critic_bn_mode" not in signature
    assert "inner_temperature_lr" not in signature
    explicit = wrappers(inner_critic_bn_mode="batch_update",
                        inner_temperature_lr=signature["inner_schedule"]["actor_lr"])
    assert _tree_equal(signature, explicit.agent.semantic_signature())


@pytest.mark.parametrize("mode", ["batch_update", "batch_no_update", "running"])
def test_v7_extensions_round_trip_and_next_outer_update(wrappers, mode):
    settings = dict(inner_critic_bn_mode=mode, inner_actor_lr=1.25e-5,
                    inner_temperature_lr=5e-5, aux_return_mode="xqc")
    source = wrappers(**settings)
    source.agent._update(*_batch(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    assert saved["checkpoint_version"] == 7
    assert saved["semantic_signature"].get("inner_critic_bn_mode", "batch_update") == mode
    assert saved["semantic_signature"]["inner_temperature_lr"] == 5e-5
    restored = wrappers(**settings).load(saved)
    assert _tree_equal(saved, restored.agent.checkpoint_state())
    source.agent._update(*_batch(source.agent))
    restored.agent._update(*_batch(restored.agent))
    assert _tree_equal(source.agent.checkpoint_state(), restored.agent.checkpoint_state())


@pytest.mark.parametrize("saved_mode,evaluated_mode", [
    ("batch_update", "running"), ("running", "batch_no_update"),
    ("batch_no_update", "batch_update"),
])
@pytest.mark.parametrize("change", ["mode", "temperature", "both"])
def test_frozen_override_records_saved_and_evaluated_semantics_before_mutation(
    wrappers, saved_mode, evaluated_mode, change,
):
    original = wrappers(inner_critic_bn_mode=saved_mode, inner_actor_lr=1.25e-5,
                        inner_temperature_lr=5e-5, aux_return_mode="xqc")
    original.agent._update(*_batch(original.agent))
    saved = deepcopy(original.agent.checkpoint_state())
    untouched = deepcopy(saved)
    mode = evaluated_mode if change != "temperature" else saved_mode
    rate = 1.25e-5 if change != "mode" else 5e-5
    target = wrappers(inner_critic_bn_mode=mode, inner_actor_lr=1.25e-5,
                      inner_temperature_lr=rate, aux_return_mode="xqc")
    before = deepcopy(target.agent.checkpoint_state())
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    assert _tree_equal(before, target.agent.checkpoint_state())
    assert target.agent.checkpoint_evaluation_provenance is None
    target.load(saved, frozen_evaluation=True)
    assert _tree_equal(original.agent.frozen_outer_state(), target.agent.frozen_outer_state())
    assert _tree_equal(saved, untouched)
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["saved_semantic_signature"]["inner_critic_bn_mode"] == saved_mode
    assert provenance["saved_semantic_signature"]["inner_temperature_lr"] == 5e-5
    assert provenance["evaluated_semantic_signature"]["inner_critic_bn_mode"] == mode
    assert provenance["evaluated_semantic_signature"]["inner_temperature_lr"] == rate
    provenance["saved_semantic_signature"]["inner_temperature_lr"] = 123.0
    assert target.agent.checkpoint_evaluation_provenance["saved_semantic_signature"]["inner_temperature_lr"] == 5e-5


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, "6_utd", "6_actor_bn", 7])
def test_historical_checkpoints_resolve_old_defaults_and_allow_frozen_overrides(wrappers, version):
    original = wrappers(inner_actor_lr=1.25e-5)
    original.agent._update(*_batch(original.agent))
    if isinstance(version, int) and version < 6:
        saved = _legacy_checkpoint(original.agent, version)
    else:
        saved = deepcopy(original.agent.checkpoint_state())
        if isinstance(version, str):
            saved["checkpoint_version"] = 6
            saved["semantic_signature"].pop(
                "inner_actor_bn_mode" if version == "6_utd" else "xqc_utd"
            )
    untouched = deepcopy(saved)
    restored = wrappers(inner_actor_lr=1.25e-5).load(saved)
    assert _tree_equal(original.agent.frozen_outer_state(), restored.agent.frozen_outer_state())
    target = wrappers(inner_actor_lr=1.25e-5, inner_temperature_lr=5e-5,
                      inner_critic_bn_mode="running")
    before = deepcopy(target.agent.checkpoint_state())
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    assert _tree_equal(before, target.agent.checkpoint_state())
    target.load(saved, frozen_evaluation=True)
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["saved_semantic_signature"]["inner_critic_bn_mode"] == "batch_update"
    assert provenance["saved_semantic_signature"]["inner_temperature_lr"] == 1.25e-5
    assert provenance["evaluated_semantic_signature"]["inner_critic_bn_mode"] == "running"
    assert provenance["evaluated_semantic_signature"]["inner_temperature_lr"] == 5e-5
    assert _tree_equal(original.agent.frozen_outer_state(), target.agent.frozen_outer_state())
    assert _tree_equal(saved, untouched)


@pytest.mark.parametrize("key,value", [
    ("inner_critic_bn_mode", None), ("inner_critic_bn_mode", True),
    ("inner_critic_bn_mode", "eval"), ("inner_critic_bn_mode", []),
    ("inner_temperature_lr", None), ("inner_temperature_lr", True),
    ("inner_temperature_lr", 0), ("inner_temperature_lr", float("nan")),
    ("inner_temperature_lr", float("inf")),
])
@pytest.mark.parametrize("frozen", [False, True])
def test_malformed_optional_extensions_fail_before_mutation(wrappers, key, value, frozen):
    target = wrappers()
    before = deepcopy(target.agent.checkpoint_state())
    saved = deepcopy(before)
    saved["semantic_signature"][key] = value
    with pytest.raises(ValueError, match=key):
        target.load(saved, frozen_evaluation=frozen)
    assert _tree_equal(before, target.agent.checkpoint_state())
    assert target.agent.checkpoint_evaluation_provenance is None


@pytest.mark.parametrize("key,value", [
    ("inner_critic_bn_mode", "running"), ("inner_temperature_lr", 5e-5),
])
def test_legacy_schema_cannot_claim_later_extensions(wrappers, key, value):
    target = wrappers()
    before = deepcopy(target.agent.checkpoint_state())
    saved = _legacy_checkpoint(target.agent, 5)
    saved["semantic_signature"][key] = value
    with pytest.raises(ValueError, match="extensions require v7"):
        target.load(saved, frozen_evaluation=True)
    assert _tree_equal(before, target.agent.checkpoint_state())
