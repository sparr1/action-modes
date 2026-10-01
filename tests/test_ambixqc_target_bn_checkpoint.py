"""Target BN is an optional v7 semantic extension, with historical defaults."""
from copy import deepcopy

import pytest

from test_ambixqc_actor_bn_checkpoint import _legacy_checkpoint
from test_ambixqc_core import _tree_equal
from test_ambixqc_update_ratio import _Replay
from test_ambixqc_prior_checkpoint import wrappers


KEY = "inner_critic_target_bn_mode"


def test_default_signature_preserves_v7_checkpoint_identity(wrappers):
    implicit = wrappers().agent.semantic_signature()
    explicit = wrappers(**{KEY: "batch_no_update"}).agent.semantic_signature()
    assert KEY not in implicit
    assert _tree_equal(implicit, explicit)


@pytest.mark.parametrize("mode", ["batch_no_update", "running"])
def test_target_bn_roundtrip_and_outer_continuation(wrappers, mode):
    settings = {KEY: mode, "aux_return_mode": "xqc", "xqc_utd": 2}
    source = wrappers(**settings)
    source.agent.update(_Replay(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    assert saved["checkpoint_version"] == 7
    assert saved["semantic_signature"].get(KEY, "batch_no_update") == mode
    restored = wrappers(**settings).load(saved)
    assert _tree_equal(saved, restored.agent.checkpoint_state())
    source.agent.update(_Replay(source.agent))
    restored.agent.update(_Replay(restored.agent))
    assert _tree_equal(source.agent.checkpoint_state(), restored.agent.checkpoint_state())


@pytest.mark.parametrize("saved_mode,evaluated_mode", [("batch_no_update", "running"), ("running", "batch_no_update")])
def test_target_bn_override_is_explicit_frozen_only_and_records_semantics(wrappers, saved_mode, evaluated_mode):
    original = wrappers(**{KEY: saved_mode}, aux_return_mode="xqc")
    original.agent.update(_Replay(original.agent))
    saved = deepcopy(original.agent.checkpoint_state())
    untouched = deepcopy(saved)
    target = wrappers(**{KEY: evaluated_mode}, aux_return_mode="xqc")
    before = deepcopy(target.agent.checkpoint_state())
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    assert _tree_equal(before, target.agent.checkpoint_state())
    assert target.agent.checkpoint_evaluation_provenance is None
    target.load(saved, frozen_evaluation=True)
    assert _tree_equal(original.agent.frozen_outer_state(), target.agent.frozen_outer_state())
    assert _tree_equal(saved, untouched)
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["saved_semantic_signature"][KEY] == saved_mode
    assert provenance["evaluated_semantic_signature"][KEY] == evaluated_mode
    provenance["saved_semantic_signature"][KEY] = "tampered"
    assert target.agent.checkpoint_evaluation_provenance["saved_semantic_signature"][KEY] == saved_mode


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, "6_utd", "6_actor_bn", 7])
def test_legacy_checkpoint_defaults_and_explicit_frozen_override(wrappers, version):
    settings = {"xqc_utd": 2} if version == "6_utd" else {}
    original = wrappers(**settings)
    original.agent.update(_Replay(original.agent))
    if isinstance(version, int) and version < 6:
        saved = _legacy_checkpoint(original.agent, version)
    else:
        saved = deepcopy(original.agent.checkpoint_state())
        if isinstance(version, str):
            saved["checkpoint_version"] = 6
            saved["semantic_signature"].pop("inner_actor_bn_mode" if version == "6_utd" else "xqc_utd")
    untouched = deepcopy(saved)
    restored = wrappers(**settings).load(saved)
    assert _tree_equal(original.agent.frozen_outer_state(), restored.agent.frozen_outer_state())
    target = wrappers(**settings, **{KEY: "running"})
    before = deepcopy(target.agent.checkpoint_state())
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    assert _tree_equal(before, target.agent.checkpoint_state())
    target.load(saved, frozen_evaluation=True)
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["saved_semantic_signature"][KEY] == "batch_no_update"
    assert provenance["evaluated_semantic_signature"][KEY] == "running"
    assert _tree_equal(original.agent.frozen_outer_state(), target.agent.frozen_outer_state())
    assert _tree_equal(saved, untouched)


@pytest.mark.parametrize("value", [None, True, [], "eval", "batch_update"])
@pytest.mark.parametrize("frozen", [False, True])
def test_malformed_target_bn_fails_preflight_before_mutation(wrappers, value, frozen):
    target = wrappers()
    before = deepcopy(target.agent.checkpoint_state())
    saved = deepcopy(before)
    saved["semantic_signature"][KEY] = value
    with pytest.raises(ValueError, match=KEY):
        target.load(saved, frozen_evaluation=frozen)
    assert _tree_equal(before, target.agent.checkpoint_state())
    assert target.agent.checkpoint_evaluation_provenance is None


def test_legacy_checkpoint_cannot_claim_new_target_bn_semantics(wrappers):
    target = wrappers()
    before = deepcopy(target.agent.checkpoint_state())
    saved = _legacy_checkpoint(target.agent, 5)
    saved["semantic_signature"][KEY] = "running"
    with pytest.raises(ValueError, match="extensions require v7"):
        target.load(saved, frozen_evaluation=True)
    assert _tree_equal(before, target.agent.checkpoint_state())
