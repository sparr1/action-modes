"""Independent identity coverage; no evaluator, learner, or publisher imports."""

import json

import pytest

from utils.lora_identity import normalize_lora_rl_identity, publication_lora_identity


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("normalize", [normalize_lora_rl_identity, publication_lora_identity])
def test_lora_rl_identity_defaults_and_scalar_types_are_canonical(normalize):
    omitted = {"inner_critic_adaptation": "lora_rl"}
    explicit = {
        "inner_actor_adaptation": "CLONE",
        "inner_critic_adaptation": "LORA_RL",
        "inner_critic_lora_layers": "INPUT_HIDDEN",
        "inner_critic_lora_rank": 96,
        "inner_critic_lora_scale": 1,
        "inner_critic_lora_weight_decay": 0.0002,
    }
    expected = {
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "lora_rl",
        "inner_critic_lora_layers": "input_hidden",
        "inner_critic_lora_rank": 96,
        "inner_critic_lora_scale": 1.0,
        "inner_critic_lora_weight_decay": 0.0002,
    }
    assert _canonical(normalize(omitted)) == _canonical(expected)
    assert _canonical(normalize(explicit)) == _canonical(expected)
    assert omitted == {"inner_critic_adaptation": "lora_rl"}
    assert explicit["inner_actor_adaptation"] == "CLONE"


@pytest.mark.parametrize("normalize", [normalize_lora_rl_identity, publication_lora_identity])
@pytest.mark.parametrize("field,value", [
    ("inner_critic_adaptation", "clone"),
    ("inner_critic_adaptation", "lora"),
    ("inner_critic_lora_layers", "hidden"),
    ("inner_critic_lora_rank", 48),
    ("inner_critic_lora_scale", 2.0),
    ("inner_critic_lora_weight_decay", 0.0),
])
def test_lora_rl_scientific_choices_are_distinct(normalize, field, value):
    baseline = {"inner_critic_adaptation": "lora_rl"}
    assert _canonical(normalize(baseline)) != _canonical(normalize({**baseline, field: value}))


@pytest.mark.parametrize("adaptation", ["clone", "lora"])
def test_historical_raw_lineage_inputs_are_preserved(adaptation):
    historical = {
        "inner_operator": "sac",
        "inner_actor_adaptation": adaptation,
        "inner_critic_adaptation": adaptation,
        "inner_actor_lora_rank": 8,
        "inner_actor_lora_scale": 2.0,
        "inner_actor_lora_dropout": 0.0,
        "inner_critic_lora_rank": 16,
        "inner_critic_lora_scale": 2.0,
        "inner_critic_lora_dropout": 0.0,
    }
    assert _canonical(normalize_lora_rl_identity(historical)) == _canonical(historical)
    if adaptation == "lora":
        assert _canonical(publication_lora_identity(historical)) == _canonical(historical)


def test_dense_publication_ignores_old_and_new_inactive_adapter_fields():
    dense = {"inner_actor_adaptation": "clone", "inner_critic_adaptation": "clone"}
    resolved = {
        **dense,
        "inner_actor_lora_rank": 8,
        "inner_actor_lora_scale": 2.0,
        "inner_actor_lora_dropout": 0.0,
        "inner_critic_lora_rank": 96,
        "inner_critic_lora_layers": "input_hidden",
        "inner_critic_lora_scale": 1.0,
        "inner_critic_lora_weight_decay": 0.0002,
        "inner_critic_lora_dropout": 0.0,
    }
    assert publication_lora_identity(resolved) == publication_lora_identity(dense)
    assert "inner_actor_lora_rank" in resolved
