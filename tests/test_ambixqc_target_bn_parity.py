"""The short parity diagnostic reports differences without weakening the launch gate."""
from copy import deepcopy
import json

import pytest

import diagnose_ambixqc_target_bn_parity as diagnostic


def sample():
    episodes = [{"seed": seed, "return": .95} for seed in (101, 102)]
    events = {(f"seed-{seed}", step): {
        "phase": "decision", "event_index": 0, "round_index": 2,
        "critic_updates": 12, "actor_updates": 4, "temperature_updates": 4,
        "metrics": {"decision/inner_prior_kl": 1., "decision/reward": .3,
                    "decision/inner_reward_normalizer_count": 475000., "decision/control_seconds": .1}}
        for seed in (101, 102) for step in range(3)}
    return {"episodes": episodes}, events


def test_parent_and_child_syntax_and_no_publication_or_numerical_mode_change():
    compile(diagnostic.CHILD, "diagnostic-child", "exec")
    assert "stage_results=False" in diagnostic.CHILD
    assert "max_steps=3" in diagnostic.CHILD and "seeds=[101,102]" in diagnostic.CHILD
    assert "use_deterministic_algorithms(" not in diagnostic.CHILD
    assert "configure_compile(" not in diagnostic.CHILD
    assert "compile=cfg.get('compile')" in diagnostic.CHILD


def test_identical_science_ignores_timing_and_new_observational_flag():
    left = sample(); right = deepcopy(left)
    for event in right[1].values():
        event["metrics"]["decision/control_seconds"] = 99.
        event["metrics"]["decision/inner_critic_target_bn_running"] = 0.
    result = diagnostic.compare(left, right)
    assert result["passed"] and result["first_decisions_passed"] and not result["failures"]


def test_later_drift_remains_failed_but_first_decisions_are_reported_separately():
    left = sample(); right = deepcopy(left)
    right[1][("seed-101", 2)]["metrics"]["decision/inner_prior_kl"] += .0008
    right[0]["episodes"][0]["return"] += .00006
    result = diagnostic.compare(left, right)
    assert not result["passed"] and result["first_decisions_passed"]
    assert len(result["failures"]) == 2
    assert result["rtol"] == 1e-5 and result["atol"] == 1e-6


def test_counts_remain_exact_even_when_relative_tolerance_would_allow_a_difference():
    left = sample(); right = deepcopy(left)
    right[1][("seed-101", 0)]["metrics"]["decision/inner_reward_normalizer_count"] += 1
    result = diagnostic.compare(left, right)
    assert not result["passed"] and not result["first_decisions_passed"]
    assert result["failures"][0]["exact"] is True


@pytest.mark.parametrize("change", ["counter", "missing_decision", "missing_metric"])
def test_misaligned_or_incomplete_comparison_is_rejected(change):
    left = sample(); right = deepcopy(left)
    event = right[1][("seed-101", 0)]
    if change == "counter": event["actor_updates"] = 5
    elif change == "missing_decision": right[1].pop(("seed-101", 0))
    else: event["metrics"].pop("decision/reward")
    with pytest.raises(ValueError): diagnostic.compare(left, right)


def signed_plan(tmp_path):
    previous = tmp_path / "stage10-plan.json"
    previous.write_text(json.dumps({"source_sha": diagnostic.OLD_SOURCE,
        "execution": {"root": "/old", "commit": diagnostic.OLD_SOURCE}}))
    manifest = tmp_path / "inventory.json"; manifest.write_text("{}")
    value = {"schema": "ambixqc-target-bn-study-v1", "stage": "stage11",
        "source_sha": "a"*40, "execution": {"root": "/new", "commit": "a"*40},
        "checkpoint_sha256": diagnostic.CHECKPOINT_SHA, "smoke_seeds": [101, 102],
        "smoke_max_steps": 3, "controller_seed": 12345,
        "reused": [{"condition": {"selector": diagnostic.SELECTOR, "settings": deepcopy(diagnostic.SETTINGS)}}],
        "parent": {"stage10": {"plan": diagnostic.bind(previous)}},
        "inputs": {"manifest": diagnostic.bind(manifest)}}
    return value


@pytest.mark.parametrize("change", [None, "horizon", "seed", "source", "hash", "input"])
def test_exact_plan_and_historical_input_binding(tmp_path, change):
    value = signed_plan(tmp_path)
    if change == "horizon": value["reused"][0]["condition"]["settings"]["inner_rollout_horizon"] = 1
    elif change == "seed": value["smoke_seeds"] = [102, 103]
    elif change == "source": value["source_sha"] = diagnostic.OLD_SOURCE
    value["plan_sha256"] = diagnostic.digest(value)
    if change == "hash": value["plan_sha256"] = "0"*64
    path = tmp_path / "plan.json"; path.write_text(json.dumps(value))
    if change == "input": (tmp_path / "inventory.json").write_text("changed")
    if change is None:
        plan, old = diagnostic.inputs(path)
        assert plan == value and old["commit"] == diagnostic.OLD_SOURCE
    else:
        with pytest.raises(ValueError): diagnostic.inputs(path)
