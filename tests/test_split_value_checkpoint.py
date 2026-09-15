"""Semantic identity is stricter than tensor shape for split-value weights."""

from copy import deepcopy

import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_split_value_outer import models, batch


@pytest.mark.parametrize("automatic", [False, True])
def test_portable_and_exact_checkpoint_round_trips(models, automatic):
    options = {"ent_coef": "auto_0.5" if automatic else .5,
               "inner_value_initialization": "soft"}
    source = models(**options)
    source._update(*batch(source))
    portable = deepcopy(source.checkpoint_state())
    restored = models(**options)
    restored.load(portable)
    _assert_tree_equal(restored.checkpoint_state(), portable)
    source.prepare_training_resume_boundary()
    exact = deepcopy(source.training_state_dict())
    restored.load_training_state_dict(exact)
    _assert_tree_equal(restored.training_state_dict(), exact)
    for observation in (torch.zeros(3), torch.tensor([.1, .2, -.1])):
        expected = source.act(observation, collect_diagnostics=False)
        actual = restored.act(observation, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [
    ("critic_value_mode", "return_soft"),
    ("value_components", ["return", "soft"]),
    ("value_components", ["entropy", "return"]),
    ("q_value_codec", "symlog_mean"),
    ("reward_value_codec", "symlog_mean"),
    ("q_layout", "separate_networks"),
    ("reward_vmax", 8.),
])
@pytest.mark.parametrize("exact", [False, True])
def test_semantic_mismatches_rejected_without_live_or_rng_mutation(models, field, value, exact):
    source, target = models(), models()
    target.act(torch.ones(3), collect_diagnostics=False)
    target.prepare_training_resume_boundary()
    before = deepcopy(target.training_state_dict())
    payload = deepcopy(source.training_state_dict() if exact else source.checkpoint_state())
    outer = payload["outer"] if exact else payload
    outer["critic_spec"][field] = value
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="critic specification"):
        (target.load_training_state_dict if exact else target.load)(payload)
    _assert_tree_equal(target.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("raw_weights", [False, True])
def test_new_variant_requires_metadata_even_when_shapes_match(models, raw_weights):
    source, target = models(), models()
    payload = deepcopy(source.model.state_dict() if raw_weights else source.checkpoint_state())
    if not raw_weights:
        payload.pop("critic_spec")
    before = deepcopy(target.training_state_dict())
    with pytest.raises(ValueError, match="semantic metadata"):
        target.load(payload)
    _assert_tree_equal(target.training_state_dict(), before)


def test_evaluator_architecture_identity_separates_value_and_reward_semantics():
    from evaluate_ambi_checkpoint import _critic_architecture_key

    def identity(**kwargs):
        return _critic_architecture_key({"algorithm_config": {"alg_params": kwargs}})

    legacy = identity()
    assert legacy == identity(critic_value_mode="single")
    split = identity(critic_value_mode="return_entropy")
    assert split != legacy
    assert split != identity(critic_value_mode="return_soft")
    assert split != identity(critic_value_mode="return_entropy", q_vmax=10, vmax=8)


def test_scientific_identity_distinguishes_split_entropy_and_resolves_defaults():
    from utils.resume_identity import scientific_trial_parameters

    def identity(**kwargs):
        return scientific_trial_parameters({"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": kwargs})

    legacy = identity()
    assert legacy == identity(critic_value_mode="single", inner_entropy_enabled=True,
                              inner_value_initialization="return")
    split = identity(critic_value_mode="return_entropy")
    assert split != legacy
    assert split == identity(critic_value_mode="return_entropy", inner_entropy_enabled=False,
                             inner_value_initialization="return", inner_finite_horizon=True)
    assert split != identity(critic_value_mode="return_entropy", inner_entropy_enabled=True)
    assert split != identity(critic_value_mode="return_entropy", inner_value_initialization="soft")
    assert identity(critic_value_mode="return_entropy", inner_operator="none") == identity(
        critic_value_mode="return_entropy", inner_operator="none", inner_finite_horizon=False)
