"""Independent scratch initialization obeys the action-local SAC lifecycle."""

from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from RL.tdmpc2_core.common.lora import LoRARLLinear, dense_lora_rl_critic
from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model


INITIALIZATIONS = [(actor, critic) for actor in ("prior", "random")
                   for critic in ("prior", "random")]


def _model(**overrides):
    params = {
        "inner_actor_initialization": "random",
        "inner_critic_initialization": "random",
        "inner_critic_target_initialization": "online",
        "inner_rounds": 1,
        "inner_rollouts_per_round": 4,
        "inner_updates_per_round": 3,
        "inner_replay_capacity": 8,
        "inner_critic_target_tau": 1.0,
        **overrides,
    }
    if params.get("inner_critic_adaptation") == "lora_rl":
        params.setdefault("inner_critic_lora_rank", 4)
    return _tiny_model(**params)


@pytest.fixture
def make_model():
    models = []

    def make(**overrides):
        model = _model(**overrides)
        models.append(model)
        return model

    yield make
    for model in models:
        model.env.close()


def _prepare(engine, *, t0=True):
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=t0)
    return engine.state


def _dense(critic):
    return (dense_lora_rl_critic(critic)
            if any(isinstance(layer, LoRARLLinear) for layer in critic.modules())
            else critic)


def _assert_equal(left, right):
    _assert_tree_equal(left, right)


def _poison(module, value=0.73):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def _snapshot(engine):
    pool = engine._action_pool
    result = {name: deepcopy(getattr(pool, name).state_dict()) for name in (
        "actor", "critic", "critic_target", "actor_optim", "critic_optim",
        "temperature_optim", "replay",
    )}
    result["log_alpha"] = pool.log_alpha.detach().clone()
    result["rng"] = engine.rng.training_state_dict()
    return result


@pytest.mark.parametrize("representation,num_q", [("scalar", 2), ("distributional", 5)])
@pytest.mark.parametrize("actor_initialization,critic_initialization", INITIALIZATIONS)
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_actor_and_critic_choose_prior_or_scratch_independently(
    make_model, representation, num_q, actor_initialization,
    critic_initialization, adaptation,
):
    model = make_model(q_representation=representation, num_q=num_q,
                       inner_actor_initialization=actor_initialization,
                       inner_critic_initialization=critic_initialization,
                       inner_critic_adaptation=adaptation)
    outer, engine = model.agent.model, model.agent.inner_engine
    _poison(outer._pi)
    _poison(outer._Qs)
    outer_before = deepcopy(outer.state_dict())
    global_rng = torch.random.get_rng_state().clone()
    state = _prepare(engine)
    critic = _dense(state.critic)
    for inner, prior, initialization in (
        (state.actor, outer._pi, actor_initialization),
        (critic, outer._Qs, critic_initialization),
    ):
        assert inner is not prior
        assert all(value.data_ptr() != dict(prior.named_parameters())[name].data_ptr()
                   for name, value in inner.named_parameters())
        if initialization == "prior":
            _assert_equal(inner.state_dict(), prior.state_dict())
        else:
            # Biases and LayerNorm tensors must also stop inheriting learned values.
            assert all(not torch.any(value == 0.73) for value in inner.parameters())
            for layer in inner.modules():
                if isinstance(layer, nn.LayerNorm):
                    torch.testing.assert_close(layer.weight, torch.ones_like(layer.weight), rtol=0, atol=0)
                    torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias), rtol=0, atol=0)
    if actor_initialization == "random":
        for layer in state.actor.modules():
            if isinstance(layer, nn.Linear):
                assert torch.count_nonzero(layer.weight)
                torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias), rtol=0, atol=0)
    if critic_initialization == "random":
        for head in critic:
            assert torch.count_nonzero(head[0].weight)
            assert not torch.count_nonzero(head[-1].weight)
            assert torch.count_nonzero(head[-1].bias)
    _assert_equal(state.critic_target.state_dict(), critic.state_dict())
    assert all(not parameter.requires_grad for parameter in state.critic_target.parameters())
    _assert_equal(outer.state_dict(), outer_before)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("representation,num_q", [("scalar", 2), ("distributional", 5)])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_scratch_preserves_actor_and_critic_initialization_rules_on_every_reset(
    make_model, monkeypatch, representation, num_q, adaptation,
):
    model = make_model(q_representation=representation, num_q=num_q,
                       inner_critic_adaptation=adaptation)
    engine = model.agent.inner_engine
    original_reset = nn.Linear.reset_parameters
    original_trunc_normal = nn.init.trunc_normal_
    actor_samples = {}

    def record_reset(module):
        original_reset(module)
        # Deep-copied dense bases preserve this evidence when LoRA wraps them.
        module._scratch_test_constructor_values = (
            module.weight.detach().clone(), module.bias.detach().clone(),
        )

    def record_trunc_normal(tensor, *args, **kwargs):
        result = original_trunc_normal(tensor, *args, **kwargs)
        actor_samples[tensor.data_ptr()] = (result.detach().clone(), kwargs)
        return result

    monkeypatch.setattr(nn.Linear, "reset_parameters", record_reset)
    monkeypatch.setattr(nn.init, "trunc_normal_", record_trunc_normal)
    for decision in range(2):
        state = _prepare(engine, t0=decision == 0)
        for layer in state.actor.modules():
            if isinstance(layer, nn.Linear):
                expected, kwargs = actor_samples[layer.weight.data_ptr()]
                assert kwargs["std"] == 0.02
                torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)
                assert not torch.count_nonzero(layer.bias)
        for head in state.critic:
            for index, wrapped in enumerate(head):
                layer = wrapped.base if isinstance(wrapped, LoRARLLinear) else wrapped
                expected_weight, expected_bias = layer._scratch_test_constructor_values
                if index == len(head) - 1:
                    expected_weight = torch.zeros_like(expected_weight)
                torch.testing.assert_close(layer.weight, expected_weight, rtol=0, atol=0)
                torch.testing.assert_close(layer.bias, expected_bias, rtol=0, atol=0)
        for module in (state.actor, state.critic):
            for layer in module.modules():
                if isinstance(layer, nn.LayerNorm):
                    assert torch.all(layer.weight == 1)
                    assert not torch.count_nonzero(layer.bias)
            _poison(module)


@pytest.mark.parametrize("placement", ["input_hidden", "hidden"])
def test_random_critic_draws_all_base_weights_before_any_lora_adapters(make_model, placement):
    dense = make_model()
    adapted = make_model(inner_critic_adaptation="lora_rl", inner_critic_lora_layers=placement)
    plain_state, lora_state = (_prepare(model.agent.inner_engine) for model in (dense, adapted))
    _assert_equal(plain_state.actor.state_dict(), lora_state.actor.state_dict())
    _assert_equal(plain_state.critic.state_dict(), _dense(lora_state.critic).state_dict())
    _assert_equal(lora_state.critic_target.state_dict(), plain_state.critic.state_dict())
    for layer in lora_state.critic.modules():
        if isinstance(layer, LoRARLLinear):
            assert torch.count_nonzero(layer.lora_A)
            assert not torch.count_nonzero(layer.lora_B)


@pytest.mark.parametrize("actor_initialization,critic_initialization", INITIALIZATIONS[1:])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_new_decisions_draw_fresh_reproducible_private_weights(
    make_model, actor_initialization, critic_initialization, adaptation,
):
    models = [make_model(inner_actor_initialization=actor_initialization,
                         inner_critic_initialization=critic_initialization,
                         inner_critic_adaptation=adaptation) for _ in range(2)]
    previous = {}
    global_rng = torch.random.get_rng_state().clone()
    for decision in range(3):
        states = [_prepare(model.agent.inner_engine, t0=decision == 0) for model in models]
        _assert_equal(models[0].agent.inner_engine.rng.training_state_dict(),
                      models[1].agent.inner_engine.rng.training_state_dict())
        for component, initialization in (("actor", actor_initialization), ("critic", critic_initialization)):
            modules = [getattr(state, component) for state in states]
            values = [_dense(module).state_dict() if component == "critic" else module.state_dict()
                      for module in modules]
            _assert_equal(*values)
            if decision and initialization == "random":
                assert any(not torch.equal(value, previous[component][name])
                           for name, value in values[0].items())
            previous[component] = deepcopy(values[0])
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("representation,num_q", [("scalar", 2), ("distributional", 5)])
@pytest.mark.parametrize("adaptation,placement", [("clone", "input_hidden"),
                                                   ("lora_rl", "input_hidden"),
                                                   ("lora_rl", "hidden")])
def test_scratch_sac_updates_trainable_parameters_and_preserves_outer_state(
    make_model, monkeypatch, representation, num_q, adaptation, placement,
):
    model = make_model(q_representation=representation, num_q=num_q,
                       inner_critic_adaptation=adaptation,
                       **({"inner_critic_lora_layers": placement} if adaptation == "lora_rl" else {}))
    outer, engine = model.agent.model, model.agent.inner_engine
    outer_before = deepcopy(outer.state_dict())
    global_rng = torch.random.get_rng_state().clone()
    initial = {}
    prepare = engine._prepare_workspace

    def capture_initial(**kwargs):
        prepare(**kwargs)
        for name in ("actor", "critic"):
            initial[name] = deepcopy(getattr(engine.state, name).state_dict())

    monkeypatch.setattr(engine, "_prepare_workspace", capture_initial)
    model.agent.act(torch.zeros(3), collect_diagnostics=False)
    pool = engine._action_pool
    assert engine.state.actor_steps == engine.state.critic_steps == 3
    assert any(not torch.equal(value, initial["actor"][name])
               for name, value in pool.actor.state_dict().items())
    assert any(not torch.equal(value, initial["critic"][name])
               for name, value in pool.critic.state_dict().items())
    assert all(value.requires_grad for value in pool.actor.parameters())
    assert not any(isinstance(layer, LoRARLLinear) for layer in pool.actor.modules())
    if adaptation == "lora_rl":
        assert type(pool.critic_optim) is torch.optim.AdamW
        assert [group["weight_decay"] for group in pool.critic_optim.param_groups] == [2e-4, 0.0]
        for name, parameter in pool.critic.named_parameters():
            if not parameter.requires_grad:
                torch.testing.assert_close(parameter, initial["critic"][name], rtol=0, atol=0)
                assert parameter.grad is None
        for layer in pool.critic.modules():
            if isinstance(layer, LoRARLLinear):
                assert not layer.base.weight.requires_grad
                assert layer.base.bias.requires_grad
                assert layer.lora_A.requires_grad and layer.lora_B.requires_grad
                assert torch.count_nonzero(layer.lora_B)
                if hasattr(layer.base, "ln"):
                    assert all(value.requires_grad for value in layer.base.ln.parameters())
        assert all(head[-1].weight.requires_grad for head in pool.critic)
    else:
        assert type(pool.critic_optim) is torch.optim.Adam
        assert all(value.requires_grad for value in pool.critic.parameters())
    _assert_equal(pool.critic_target.state_dict(), _dense(pool.critic).state_dict())
    _assert_equal(outer.state_dict(), outer_before)
    assert all(parameter.grad is None for parameter in outer.parameters())
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_scratch_resets_moments_replay_temperature_and_preserves_storage(make_model, adaptation):
    model = make_model(inner_critic_adaptation=adaptation)
    model.agent.act(torch.zeros(3), collect_diagnostics=False)
    engine, outer = model.agent.inner_engine, model.agent.model
    pool = engine._action_pool
    names = ("actor", "critic", "critic_target", "actor_optim", "critic_optim",
             "temperature_optim", "replay", "log_alpha")
    objects = {name: getattr(pool, name) for name in names}
    parameters = [parameter for name in ("actor", "critic", "critic_target")
                  for parameter in objects[name].parameters()]
    pointers = [parameter.data_ptr() for parameter in parameters]
    moments = [value for name in ("actor_optim", "critic_optim", "temperature_optim")
               for entry in objects[name].state.values() for value in entry.values()
               if torch.is_tensor(value)]
    moment_pointers = [value.data_ptr() for value in moments]
    replay_pointer = objects["replay"]._storage.data_ptr()
    before = deepcopy(objects["critic"].state_dict())
    assert moments and any(torch.count_nonzero(value) for value in moments)
    assert objects["replay"].size == 8
    _poison(outer._pi)
    _poison(outer._Qs)
    with torch.no_grad():
        model.agent.log_ent_coef.fill_(math.log(0.37))
    model.agent.outer_version += 1
    state = _prepare(engine, t0=False)
    for name, value in objects.items():
        assert getattr(state, name) is value, name
    current_parameters = [parameter for name in ("actor", "critic", "critic_target")
                          for parameter in getattr(state, name).parameters()]
    assert [id(parameter) for parameter in current_parameters] == [id(parameter) for parameter in parameters]
    assert [parameter.data_ptr() for parameter in current_parameters] == pointers
    current_moments = [value for name in ("actor_optim", "critic_optim", "temperature_optim")
                       for entry in getattr(state, name).state.values() for value in entry.values()
                       if torch.is_tensor(value)]
    assert [value.data_ptr() for value in current_moments] == moment_pointers
    assert all(not torch.count_nonzero(value) for value in current_moments)
    assert state.replay._storage.data_ptr() == replay_pointer
    assert state.replay.size == state.replay.next_sample_id == state.replay.pos == 0
    assert not state.replay.full
    assert state.actor_lifetime_steps == state.critic_lifetime_steps == state.temperature_lifetime_steps == 0
    assert state.log_alpha.exp().item() == pytest.approx(0.37)
    assert any(not torch.equal(value, before[name]) for name, value in state.critic.state_dict().items())
    assert all(not torch.any(value == 0.73) for value in state.actor.parameters())
    assert all(not torch.any(value == 0.73) for value in _dense(state.critic).parameters())
    _assert_equal(state.critic_target.state_dict(), _dense(state.critic).state_dict())
    if adaptation == "lora_rl":
        assert all(not torch.count_nonzero(layer.lora_B) for layer in state.critic.modules()
                   if isinstance(layer, LoRARLLinear))


@pytest.mark.parametrize("actor_initialization,critic_initialization", INITIALIZATIONS[1:])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_cold_and_pooled_evaluation_scratch_solves_are_identical(
    make_model, actor_initialization, critic_initialization, adaptation,
):
    discarded, reused = [make_model(inner_actor_initialization=actor_initialization,
                                    inner_critic_initialization=critic_initialization,
                                    inner_critic_adaptation=adaptation, dropout=0.2)
                         for _ in range(2)]
    for model in (discarded, reused):
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
    pool = reused.agent.inner_engine._action_pool
    identities = (id(pool.actor), id(pool.critic))
    for seed in (909, 910):
        discarded.agent.inner_engine.reset_for_evaluation(seed)
        reused.agent.inner_engine.reset_for_evaluation(seed, reuse_action_pool=True)
        for decision in range(2):
            expected = discarded.agent.act(torch.ones(3), t0=decision == 0, collect_diagnostics=False)
            actual = reused.agent.act(torch.ones(3), t0=decision == 0, collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _assert_equal(_snapshot(reused.agent.inner_engine), _snapshot(discarded.agent.inner_engine))
            assert (id(pool.actor), id(pool.critic)) == identities


@pytest.mark.parametrize("component", ["actor", "critic"])
def test_frozen_scratch_component_keeps_its_weights_during_learning(make_model, monkeypatch, component):
    model = make_model(**{f"inner_{component}_adaptation": "frozen"})
    engine = model.agent.inner_engine
    initial = {}
    prepare = engine._prepare_workspace

    def capture_initial(**kwargs):
        prepare(**kwargs)
        initial.update(deepcopy(getattr(engine.state, component).state_dict()))

    monkeypatch.setattr(engine, "_prepare_workspace", capture_initial)
    model.agent.act(torch.zeros(3), collect_diagnostics=False)
    module = getattr(engine._action_pool, component)
    _assert_equal(module.state_dict(), initial)
    assert not any(parameter.requires_grad for parameter in module.parameters())
    assert getattr(engine.state, f"{component}_steps") == 0
    other = "critic" if component == "actor" else "actor"
    assert getattr(engine.state, f"{other}_steps") == 3


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_scratch_keeps_outer_bootstrap_anchor_and_fixed_temperature_choices(make_model, adaptation):
    model = make_model(inner_critic_adaptation=adaptation,
                       inner_bootstrap_source="outer_target",
                       inner_outer_policy_kl_coef=0.1,
                       inner_temperature_mode="fixed",
                       inner_temperature_initialization="fixed",
                       inner_temperature=0.37)
    state = _prepare(model.agent.inner_engine)
    assert state.critic_target is None
    assert state.actor_anchor is model.agent.model._pi
    assert state.alpha_fixed.item() == pytest.approx(0.37)
    assert state.temperature_optim is None


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_random_targets_follow_online_effective_weight_averaging(make_model, adaptation):
    model = make_model(inner_critic_adaptation=adaptation,
                       inner_critic_target_tau=0.3,
                       inner_critic_target_update_interval=2)
    engine = model.agent.inner_engine
    state = _prepare(engine)
    before = deepcopy(state.critic_target.state_dict())
    with torch.no_grad():
        for parameter in state.critic_params:
            parameter.add_(0.2)
    effective = deepcopy(_dense(state.critic).state_dict())
    state.critic_lifetime_steps = 1
    engine._maybe_update_targets(critic_updated=True, actor_updated=False)
    _assert_equal(state.critic_target.state_dict(), before)
    assert state.critic_target_steps == 0
    state.critic_lifetime_steps = 2
    engine._maybe_update_targets(critic_updated=True, actor_updated=False)
    for name, value in state.critic_target.state_dict().items():
        torch.testing.assert_close(value, before[name].lerp(effective[name], 0.3), rtol=0, atol=0)
    assert state.critic_target_steps == 1


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_zero_rounds_keeps_outer_policy_bypass_without_scratch_allocation(make_model, adaptation):
    scratch = make_model(inner_rounds=0, inner_critic_adaptation=adaptation)
    prior = make_model(inner_rounds=0, inner_critic_adaptation=adaptation,
                       inner_actor_initialization="prior", inner_critic_initialization="prior")
    engine = scratch.agent.inner_engine
    before = engine.rng.training_state_dict()
    expected = prior.agent.act(torch.zeros(3), collect_diagnostics=False)
    actual = scratch.agent.act(torch.zeros(3), collect_diagnostics=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    after = engine.rng.training_state_dict()
    for key in ("streams", "phase_streams"):
        torch.testing.assert_close(after[key]["initialization"], before[key]["initialization"], rtol=0, atol=0)
    _assert_equal(after, prior.agent.inner_engine.rng.training_state_dict())
    for workspace in (engine.state, engine._action_pool):
        assert workspace.actor is workspace.critic is workspace.critic_target is None
        assert workspace.replay is None
    assert scratch.agent.last_inner_metrics["inner_model_steps"] == 0
