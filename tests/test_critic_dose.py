"""Actual C32/C64 optimizer work and C16 compatibility on small frozen networks."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from evaluate_ambi_checkpoint import _outer_state_digest
from tests.test_critic_bypass import controller, call
from utils.critic_bypass import validate_bypass_controller


def test_explicit_c16_matches_unchanged_default_bitwise(controller):
    default = call(controller, arm='learned_q', shadow=True)
    weights = deepcopy(controller.agent.inner_engine._action_pool.critic.state_dict())
    explicit = call(controller, arm='learned_q', shadow=True, critic_updates_per_round=16)
    for key in ('action', 'normalized_action'):
        np.testing.assert_array_equal(default[key], explicit[key])
    assert default['diagnostics'] == explicit['diagnostics']
    assert default['work'] == explicit['work'] == dict(actor_updates=24, critic_updates=96, model_steps=768)
    for key, value in weights.items():
        assert torch.equal(value, controller.agent.inner_engine._action_pool.critic.state_dict()[key])


@pytest.mark.parametrize('dose', [32, 64])
def test_dose_executes_exact_updates_with_same_within_dose_bank_and_frozen_outer(controller, dose):
    controller.cfg.inner_critic_updates_per_round = dose
    before = _outer_state_digest(controller)
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match='fixed fresh'):
        validate_bypass_controller(controller)  # Larger doses must be explicit.
    actor = call(controller, arm='actor_mean', shadow=True, critic_updates_per_round=dose)
    selected = call(controller, arm='learned_q', shadow=True, critic_updates_per_round=dose)
    assert actor['diagnostics']['bank_sha256'] == selected['diagnostics']['bank_sha256']
    assert actor['work'] == selected['work'] == dict(actor_updates=24, critic_updates=6*dose, model_steps=768)
    assert actor['bank_count'] == selected['bank_count'] == 770
    assert actor['replay_count'] == selected['replay_count'] == 768
    assert _outer_state_digest(controller) == before
    assert torch.equal(torch.random.get_rng_state(), rng)
    altered_validation = call(controller, arm='learned_q', shadow=True,
        critic_updates_per_round=dose, validation_seed=9999)
    np.testing.assert_array_equal(altered_validation['action'], selected['action'])
    assert altered_validation['diagnostics']['choices']['prior_mean']['validation_model']['draws'] != (
        selected['diagnostics']['choices']['prior_mean']['validation_model']['draws'])


@pytest.mark.parametrize('dose', [True, 0, 1, 31, 128, 32.0])
def test_unsupported_doses_fail_before_learning(controller, dose):
    with pytest.raises(ValueError):
        call(controller, critic_updates_per_round=dose)
