"""LoRA-RL integration at the decision-time SAC lifecycle boundary."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.common.lora import LoRARLLinear, dense_lora_rl_critic
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_ambi_latency_contract import _assert_tree_equal


def _model(**overrides):
    params = {
        "inner_critic_adaptation": "lora_rl",
        "inner_critic_lora_rank": 4,
        "inner_critic_lora_layers": "input_hidden",
        "inner_rounds": 1,
        "inner_rollouts_per_round": 4,
        "inner_updates_per_round": 3,
        "inner_critic_target_tau": 1.0,
    }
    params.update(overrides)
    return _tiny_model(**params)


def _assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)


@pytest.mark.parametrize("representation,num_q", [("scalar", 2), ("distributional", 5)])
@pytest.mark.parametrize("layers", ["input_hidden", "hidden"])
def test_sac_solve_updates_selected_critic_without_mutating_outer_or_global_rng(
    representation, num_q, layers,
):
    model = _model(q_representation=representation, num_q=num_q, inner_critic_lora_layers=layers)
    outer = model.agent.model
    before = deepcopy(outer.state_dict())
    rng_before = torch.random.get_rng_state().clone()
    model.agent.act(torch.zeros(3), collect_diagnostics=False)
    state = model.agent.inner_engine._action_pool
    counters = model.agent.inner_engine.state
    _assert_state_equal(outer.state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    assert all(value.grad is None for value in outer.parameters())
    assert type(state.actor_optim) is torch.optim.Adam
    assert type(state.critic_optim) is torch.optim.AdamW
    assert [group["weight_decay"] for group in state.critic_optim.param_groups] == [2e-4, 0.0]
    assert not any(isinstance(layer, LoRARLLinear) for layer in state.actor.modules())
    assert not any(isinstance(layer, LoRARLLinear) for layer in state.critic_target.modules())
    assert counters.critic_steps == counters.actor_steps == 3
    assert counters.critic_target_steps == 3
    assert any(not torch.equal(value, outer._pi.state_dict()[name]) for name, value in state.actor.state_dict().items())
    assert any(torch.count_nonzero(layer.lora_B) for layer in state.critic.modules() if isinstance(layer, LoRARLLinear))
    for path, layer in state.critic.named_modules():
        if isinstance(layer, LoRARLLinear):
            torch.testing.assert_close(layer.base.weight, outer._Qs.get_submodule(path).weight, rtol=0, atol=0)
            assert layer.base.weight.grad is None
    assert any(not torch.equal(inner[-1].weight, prior[-1].weight) for inner, prior in zip(state.critic, outer._Qs))
    _assert_state_equal(state.critic_target.state_dict(), dense_lora_rl_critic(state.critic).state_dict())


def test_next_root_restores_latest_dense_priors_and_reuses_optimizer_allocations():
    model = _model()
    model.agent.act(torch.zeros(3), collect_diagnostics=False)
    engine = model.agent.inner_engine
    state = engine._action_pool
    actor, critic, target, replay = state.actor, state.critic, state.critic_target, state.replay
    actor_optim, critic_optim = state.actor_optim, state.critic_optim
    parameter_ids = tuple(id(value) for value in critic.parameters())
    moments = [value for entry in critic_optim.state.values() for value in entry.values() if torch.is_tensor(value)]
    pointers = [value.data_ptr() for value in moments]
    assert moments and any(torch.count_nonzero(value) for value in moments)
    assert replay.size > 0
    with torch.no_grad():
        for module in (model.agent.model._pi, model.agent.model._Qs):
            for value in module.parameters():
                value.add_(0.02)
    model.agent.outer_version += 1
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=False)
    assert engine.state.actor is actor
    assert engine.state.critic is critic
    assert engine.state.critic_target is target
    assert engine.state.replay is replay
    assert engine.state.actor_optim is actor_optim
    assert engine.state.critic_optim is critic_optim
    assert tuple(id(value) for value in critic.parameters()) == parameter_ids
    assert [value.data_ptr() for value in moments] == pointers
    assert all(not torch.count_nonzero(value) for value in moments)
    assert replay.size == 0
    _assert_state_equal(actor.state_dict(), model.agent.model._pi.state_dict())
    _assert_state_equal(dense_lora_rl_critic(critic).state_dict(), model.agent.model._Qs.state_dict())
    _assert_state_equal(target.state_dict(), model.agent.model._Qs.state_dict())
    assert all(not torch.count_nonzero(layer.lora_B) for layer in critic.modules() if isinstance(layer, LoRARLLinear))
    torch.testing.assert_close(engine.state.log_alpha.exp(), model.agent.alpha.detach().reshape(()))


def test_engine_target_schedule_uses_effective_weight_ema():
    model = _model(inner_critic_target_tau=0.3, inner_critic_target_update_interval=2)
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    state = engine.state
    before = deepcopy(state.critic_target.state_dict())
    with torch.no_grad():
        for value in state.critic_params:
            value.add_(0.2)
    effective = dense_lora_rl_critic(state.critic).state_dict()
    state.critic_lifetime_steps = 1
    engine._maybe_update_targets(critic_updated=True, actor_updated=False)
    _assert_state_equal(state.critic_target.state_dict(), before)
    assert state.critic_target_steps == 0
    state.critic_lifetime_steps = 2
    engine._maybe_update_targets(critic_updated=True, actor_updated=False)
    for name, value in state.critic_target.state_dict().items():
        torch.testing.assert_close(value, before[name].lerp(effective[name], 0.3))
    assert state.critic_target_steps == 1


def test_lora_rl_retains_outer_bootstrap_without_allocating_inner_target():
    model = _model(inner_bootstrap_source="outer_target")
    model.agent.act(torch.zeros(3), collect_diagnostics=False)
    state = model.agent.inner_engine.state
    assert state.critic_target is None
    assert state.critic_target_steps == 0
    assert state.critic_steps == 3


def test_direct_inner_checkpoint_roundtrip_identifies_action_local_protocol():
    source = _model()
    source.agent.act(torch.zeros(3), collect_diagnostics=False)
    source.agent.prepare_training_resume_boundary()
    payload = deepcopy(source.agent.inner_engine.training_state_dict())
    assert payload["version"] == 3
    assert payload["lora_rl_spec"] == {
        "method": "lora_rl", "protocol_version": 1, "layers": "input_hidden",
        "rank": 4, "scale": 1.0, "weight_decay": 2e-4,
    }
    assert payload["workspace"]["critic"] is None
    target = _model()
    target.agent.inner_engine.load_training_state_dict(payload)
    _assert_tree_equal(target.agent.inner_engine.training_state_dict(), payload)


@pytest.mark.parametrize("key,value", [
    ("method", "legacy_lora"), ("protocol_version", 2), ("layers", "hidden"),
    ("rank", 3), ("scale", 0.5), ("weight_decay", 6e-4), ("rank", True),
])
def test_direct_inner_checkpoint_rejects_protocol_mismatch_transactionally(key, value):
    model = _model()
    engine = model.agent.inner_engine
    before = deepcopy(engine.training_state_dict())
    payload = deepcopy(before)
    payload["lora_rl_spec"][key] = value
    with pytest.raises(ValueError, match="LoRA-RL.*incompatible"):
        engine.load_training_state_dict(payload)
    _assert_tree_equal(engine.training_state_dict(), before)


@pytest.mark.parametrize("source_lora", [False, True])
def test_direct_inner_checkpoint_rejects_cross_method_state(source_lora):
    source = _model() if source_lora else _tiny_model()
    target = _tiny_model() if source_lora else _model()
    before = deepcopy(target.agent.inner_engine.training_state_dict())
    with pytest.raises(ValueError, match="LoRA-RL.*incompatible"):
        target.agent.inner_engine.load_training_state_dict(source.agent.inner_engine.training_state_dict())
    _assert_tree_equal(target.agent.inner_engine.training_state_dict(), before)
