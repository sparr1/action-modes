"""Auxiliary learned state is explicit and checkpoint loading is transactional."""

from copy import deepcopy

import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_split_value_outer import batch


@pytest.fixture
def models():
    wrappers = []
    def create(**overrides):
        options = dict(aux_return_mode="return_actor", inner_rounds=1,
                       inner_updates_per_round=1, ent_coef=.1,
                       aux_return_ent_coef="auto_0.1")
        options.update(overrides)
        wrapper = _tiny_model(**options)
        wrappers.append(wrapper)
        return wrapper.agent
    yield create
    for wrapper in wrappers:
        wrapper.env.close()


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
@pytest.mark.parametrize("exact", [False, True])
def test_auxiliary_checkpoint_round_trip(models, mode, exact):
    source = models(aux_return_mode=mode)
    source._update(*batch(source))
    source.prepare_training_resume_boundary()
    payload = deepcopy(source.training_state_dict() if exact else source.checkpoint_state())
    target = models(aux_return_mode=mode)
    (target.load_training_state_dict if exact else target.load)(payload)
    actual = target.training_state_dict() if exact else target.checkpoint_state()
    _assert_tree_equal(actual, payload)


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("corruption", ["missing", "mode", "rng", "updates", "temperature", "optimizer", "shared_optimizer"])
def test_auxiliary_load_rejects_bad_state_before_mutating_live_agent(models, exact, corruption):
    source, target = models(), models()
    source._update(*batch(source))
    target.prepare_training_resume_boundary()
    payload = deepcopy(source.training_state_dict() if exact else source.checkpoint_state())
    outer = payload["outer"] if exact else payload
    if corruption == "missing":
        outer.pop("aux_return_state")
    elif corruption == "mode":
        outer["aux_return_spec"]["mode"] = "sac"
    elif corruption == "rng":
        outer["aux_return_state"]["rng_cpu"] = torch.zeros(3, dtype=torch.uint8)
    elif corruption == "updates":
        outer["aux_return_state"]["num_updates"] += 1
    elif corruption == "temperature":
        outer["aux_return_state"]["log_ent_coef"].fill_(float("nan"))
    elif corruption == "shared_optimizer":
        outer.pop("optim")
    else:
        outer["aux_return_state"]["pi_optim"]["param_groups"][0]["params"].pop()
    before = deepcopy(target.training_state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError)):
        (target.load_training_state_dict if exact else target.load)(payload)
    _assert_tree_equal(target.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_portable_routing_can_change_but_exact_resume_cannot(models):
    source = models()
    source.prepare_training_resume_boundary()
    target = models(inner_actor_source="return_actor", inner_critic_source="aux_return",
                    inner_horizon_actor_source="return_actor", inner_horizon_critic_source="aux_return")
    target.load(deepcopy(source.checkpoint_state()))
    with pytest.raises(ValueError):
        target.load_training_state_dict(deepcopy(source.training_state_dict()))


def test_enabled_checkpoint_requires_auxiliary_semantics_even_for_raw_weights(models):
    source, target = models(), models()
    with pytest.raises(ValueError, match="semantic"):
        target.load(deepcopy(source.model.state_dict()))
    off = models(aux_return_mode="off")
    with pytest.raises(ValueError, match="auxiliary"):
        off.load(deepcopy(source.checkpoint_state()))
    with pytest.raises(ValueError, match="semantic"):
        target.load(deepcopy(off.checkpoint_state()))


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("schedule", ["none", "smooth", "quantile_gate", "dual"])
def test_auxiliary_scaler_and_regularizer_state_round_trip(models, exact, schedule):
    options = dict(aux_return_ent_coef=0.,
                   aux_return_sac_actor_loss_scale_mode="tdmpc2_percentile_range",
                   aux_return_outer_behavior_policy_kl_schedule=schedule,
                   aux_return_outer_behavior_policy_kl_min_valid_count=1)
    source = models(**options)
    obs, action, reward, terminated = batch(source)
    source._update(obs, action, reward, terminated,
                   behavior_pre_tanh_mean=torch.zeros_like(action),
                   behavior_log_std=torch.zeros_like(action),
                   behavior_policy_valid=torch.ones_like(action, dtype=torch.bool))
    source.aux_return.actor_loss_scale.fill_(7.)
    source.prepare_training_resume_boundary()
    payload = deepcopy(source.training_state_dict() if exact else source.checkpoint_state())
    target = models(**options)
    (target.load_training_state_dict if exact else target.load)(payload)
    _assert_tree_equal(target.training_state_dict() if exact else target.checkpoint_state(), payload)


@pytest.mark.parametrize("prefix", ["_aux_return_Qs.", "_target_aux_return_Qs.", "_return_pi."])
def test_missing_auxiliary_network_weights_rejected_before_mutation(models, prefix):
    source, target = models(), models()
    payload = deepcopy(source.checkpoint_state())
    key = next(key for key in payload["model"] if key.startswith(prefix))
    payload["model"].pop(key)
    before = deepcopy(target.checkpoint_state())
    with pytest.raises(ValueError):
        target.load(payload)
    _assert_tree_equal(target.checkpoint_state(), before)
