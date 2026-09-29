"""Actor BatchNorm semantics survive portable loads and explicit evaluation overrides."""

from copy import deepcopy

import pytest

from test_ambixqc_core import _batch, _tree_equal
from test_ambixqc_prior_checkpoint import wrappers


def _legacy_checkpoint(agent, version):
    """Produce the actual version-specific schema, not a relabeled v7 state."""
    state = deepcopy(agent.checkpoint_state())
    state["checkpoint_version"] = version
    signature = state["semantic_signature"]
    signature.pop("inner_actor_bn_mode")
    signature.pop("xqc_utd")
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
def test_v7_actor_bn_mode_round_trip_and_next_outer_update(wrappers, mode, auxiliary):
    settings = {"inner_actor_bn_mode": mode, "aux_return_mode": "xqc" if auxiliary else "off"}
    source = wrappers(**settings)
    source.agent._update(*_batch(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    assert saved["checkpoint_version"] == 7
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
    assert provenance["checkpoint_version"] == 7
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


@pytest.mark.parametrize("mode", ["batch_update", "running"])
def test_historical_actor_bn_v6_loads_as_utd_one_and_continues(wrappers, mode):
    """The actor-BN branch predated xqc_utd, despite also using version six."""
    source = wrappers(aux_return_mode="xqc", inner_actor_bn_mode=mode)
    source.agent._update(*_batch(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    saved["checkpoint_version"] = 6
    saved["semantic_signature"].pop("xqc_utd")
    untouched = deepcopy(saved)
    restored = wrappers(aux_return_mode="xqc", inner_actor_bn_mode=mode).load(saved)
    assert _tree_equal(source.agent.frozen_outer_state(), restored.agent.frozen_outer_state())
    provenance = restored.agent.checkpoint_evaluation_provenance
    assert provenance["checkpoint_version"] == 6
    assert provenance["saved_semantic_signature"]["xqc_utd"] == 1
    assert provenance["saved_semantic_signature"]["inner_actor_bn_mode"] == mode
    assert _tree_equal(saved, untouched)
    source.agent._update(*_batch(source.agent))
    restored.agent._update(*_batch(restored.agent))
    assert _tree_equal(source.agent.checkpoint_state(), restored.agent.checkpoint_state())
    with pytest.raises(ValueError, match="matching xqc_utd"):
        wrappers(aux_return_mode="xqc", xqc_utd=2, inner_actor_bn_mode=mode).load(
            untouched, frozen_evaluation=True,
        )


def test_utd_two_v6_checkpoint_supports_frozen_running_bn_with_auxiliary_and_resets(wrappers):
    from test_ambixqc_update_ratio import _Replay

    source = wrappers(xqc_utd=2, aux_return_mode="xqc", inner_operator="none",
                      aux_return_detach_representation=False)
    source.agent.update(_Replay(source.agent))
    saved = deepcopy(source.agent.checkpoint_state())
    saved["checkpoint_version"] = 6
    saved["semantic_signature"].pop("inner_actor_bn_mode")
    untouched = deepcopy(saved)
    target = wrappers(
        xqc_utd=2, aux_return_mode="xqc", inner_actor_bn_mode="running",
        aux_return_detach_representation=False,
        inner_critic_source="aux_return", inner_horizon_critic_source="aux_return",
        inner_critic_target="reward_only", inner_terminal_bootstrap="outer",
        inner_rollout_horizon=1, inner_rounds=2, inner_updates_per_round=3,
        inner_replay_capacity=4,
    )
    before = deepcopy(target.agent.checkpoint_state())
    with pytest.raises(ValueError, match="semantics"):
        target.load(saved)
    assert _tree_equal(before, target.agent.checkpoint_state())
    target.load(saved, frozen_evaluation=True)
    provenance = target.agent.checkpoint_evaluation_provenance
    assert provenance["checkpoint_version"] == 6
    assert provenance["saved_semantic_signature"]["inner_actor_bn_mode"] == "batch_update"
    assert provenance["evaluated_semantic_signature"]["inner_actor_bn_mode"] == "running"
    assert provenance["evaluated_semantic_signature"]["xqc_utd"] == 2
    assert provenance["evaluated_semantic_signature"]["inner_critic_target"] == "reward_only"
    outer = source.agent.frozen_outer_state()
    assert _tree_equal(outer, target.agent.frozen_outer_state())
    outputs = []
    for _ in range(2):
        target.reset_for_evaluation(101, reuse_action_pool=True)
        observation, _ = target.env.reset(seed=101)
        outputs.append(target.predict(observation, deterministic=True)[0].copy())
        local = target.agent.inner_engine._workspace_pool
        assert local.update_step == 6
        assert local.actor_optimizer_steps == local.temperature_optimizer_steps == 2
        assert _tree_equal(
            dict(source.agent.xqc_controller.actor.named_buffers()),
            dict(local.controller.actor.named_buffers()),
        )
        assert _tree_equal(
            dict(source.agent.aux_return.critic.named_buffers()),
            dict(local.controller.critic_target.named_buffers()),
        )
        assert not _tree_equal(
            dict(source.agent.aux_return.critic.named_buffers()),
            dict(local.controller.critic.named_buffers()),
        )
        assert _tree_equal(outer, target.agent.frozen_outer_state())
    assert (outputs[0] == outputs[1]).all()
    assert _tree_equal(saved, untouched)


@pytest.mark.parametrize("dialect", ["both_features", "neither_feature"])
@pytest.mark.parametrize("frozen", [False, True])
def test_v6_schema_discriminators_must_be_unambiguous_before_mutation(wrappers, dialect, frozen):
    saved = deepcopy(wrappers().agent.checkpoint_state())
    saved["checkpoint_version"] = 6
    if dialect == "neither_feature":
        saved["semantic_signature"].pop("xqc_utd")
        saved["semantic_signature"].pop("inner_actor_bn_mode")
    target = wrappers()
    before = deepcopy(target.agent.checkpoint_state())
    with pytest.raises(ValueError, match="exactly one historical schema"):
        target.load(saved, frozen_evaluation=frozen)
    assert _tree_equal(before, target.agent.checkpoint_state())


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
