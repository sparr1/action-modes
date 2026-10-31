"""Scientific invariants for the historical detached auxiliary-critic backport."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_model, _tiny_params


@pytest.fixture
def agents():
    wrappers = []

    def create(**overrides):
        params = {
            "inner_operator": "none",
            "inner_rounds": None,
            "inner_rollouts_per_round": None,
            "inner_updates_per_round": None,
            "aux_return_mode": "sac",
            "aux_return_detach_representation": True,
            "aux_return_critic_coef": 0.1,
            "ent_coef": "auto",
            "target_entropy": -21.0,
        }
        params.update(overrides)
        wrapper = _tiny_model(**params)
        wrappers.append(wrapper)
        return wrapper.agent

    yield create
    for wrapper in wrappers:
        wrapper.env.close()


def batch(agent):
    h, b = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    return (
        torch.randn(h + 1, b, 3),
        torch.randn(h, b, agent.cfg.action_dim).tanh(),
        torch.randn(h, b, 1),
        torch.zeros(h, b, 1),
    )


def expose_auxiliary_input_gradients(agent):
    # Historical output weights start at zero, which otherwise masks whether
    # input gradients are connected at all.
    with torch.no_grad():
        for head in agent.model._aux_return_Qs:
            head[-1].weight.fill_(0.1)


def test_auxiliary_construction_preserves_primary_parameters_and_rng(agents):
    torch.manual_seed(55)
    original = agents(aux_return_mode="off")
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(55)
    augmented = agents()
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    for name, value in original.model.state_dict().items():
        torch.testing.assert_close(augmented.model.state_dict()[name], value, rtol=0, atol=0)
    assert not any("aux_return" in name for name in original.model.state_dict())
    assert "aux_return_spec" not in original.checkpoint_state()
    assert "aux_return_state" not in original.checkpoint_state()
    assert not hasattr(augmented.model, "_return_pi")
    assert augmented.target_entropy == -21.0
    assert augmented.cfg.aux_return_critic_lr == augmented.cfg.critic_lr


def test_auxiliary_uses_historical_zero_bias_initialization_and_frozen_targets(agents):
    agent = agents()
    for head in agent.model._aux_return_Qs:
        for layer in head.modules():
            if isinstance(layer, torch.nn.Linear) and layer.bias is not None:
                assert torch.count_nonzero(layer.bias) == 0
        assert torch.count_nonzero(head[-1].weight) == 0
    assert not {id(p) for p in agent.model._Qs.parameters()}.intersection(
        id(p) for p in agent.model._aux_return_Qs.parameters()
    )
    agent.model.train()
    assert not agent.model._target_aux_return_Qs.training
    assert all(not p.requires_grad for p in agent.model._target_aux_return_Qs.parameters())
    before = deepcopy(agent.model._target_aux_return_Qs.state_dict())
    with torch.no_grad():
        for parameter in agent.model._aux_return_Qs.parameters():
            parameter.add_(2.0)
    agent.model.soft_update_target_Q(tau=0.25)
    for name, value in agent.model._target_aux_return_Qs.state_dict().items():
        torch.testing.assert_close(value, 0.75 * before[name] + 0.25 * agent.model._aux_return_Qs.state_dict()[name])


def test_auxiliary_target_is_reward_only_under_original_actor(agents, monkeypatch):
    agent = agents()
    calls = []

    def action(z, **kwargs):
        calls.append("original_actor")
        return torch.full_like(z[..., :agent.cfg.action_dim], 0.25)

    def value(z, action, **kwargs):
        assert kwargs["target"] is True
        assert kwargs["reduction"] == agent.cfg.outer_q_target_reduction
        assert torch.all(action == 0.25)
        return z.new_tensor([[3.0], [float("nan")]])

    monkeypatch.setattr(agent.model, "pi_action", action)
    monkeypatch.setattr(agent.model, "aux_return_Q", value)
    target = agent.aux_return.td_target(
        torch.zeros(2, agent.cfg.latent_dim),
        torch.tensor([[2.0], [4.0]]),
        torch.tensor([[0.0], [1.0]]),
    )
    torch.testing.assert_close(target, torch.tensor([[2.0 + agent.discount * 3.0], [4.0]]))
    assert calls == ["original_actor"]
    assert not target.requires_grad


def test_auxiliary_loss_detaches_representation_but_trains_critic(agents):
    agent = agents()
    expose_auxiliary_input_gradients(agent)
    obs, action, reward, _ = batch(agent)
    z = agent.model.encode(obs[0])
    latents = [z]
    for a in action:
        z = agent.model.next(z, a)
        latents.append(z)
    loss, _ = agent.aux_return.critic_loss(torch.stack(latents), action, reward)
    loss.backward()
    for module in (
        agent.model._encoder, agent.model._dynamics, agent.model._reward,
        agent.model._Qs, agent.model._pi, agent.model._target_aux_return_Qs,
    ):
        assert all(p.grad is None for p in module.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in agent.model._aux_return_Qs.parameters())


def test_auxiliary_samples_and_dropout_advance_only_private_rng(agents):
    agent = agents(dropout=0.2)
    agent.model.train()
    obs, action, reward, terminated = batch(agent)
    latents = torch.randn(obs.shape[0], obs.shape[1], agent.cfg.latent_dim)
    global_before = torch.random.get_rng_state().clone()
    private_before = agent.aux_return.checkpoint_state()["rng_cpu"].clone()
    target = agent.aux_return.td_target(latents[1:], reward, terminated)
    agent.aux_return.critic_loss(latents, action, target)
    assert torch.equal(torch.random.get_rng_state(), global_before)
    assert not torch.equal(agent.aux_return.checkpoint_state()["rng_cpu"], private_before)


@pytest.mark.parametrize("h", [2, 3])
@pytest.mark.parametrize("normalization", ["divide_horizon", "reference_weighted_mean"])
def test_auxiliary_loss_respects_historical_temporal_normalization(agents, monkeypatch, h, normalization):
    agent = agents(train_unroll_horizon=h, outer_planning_horizon=h, rho=0.5,
                   temporal_loss_normalization=normalization, temporal_loss_reference_horizon=3)
    b = agent.cfg.batch_size
    predictions = torch.arange(1.0, h + 1).view(1, h, 1, 1).expand(agent.cfg.num_q, h, b, 1)
    monkeypatch.setattr(agent.model, "aux_return_q_predictions", lambda *a, **kw: predictions)
    loss, _ = agent.aux_return.critic_loss(
        torch.zeros(h + 1, b, agent.cfg.latent_dim),
        torch.zeros(h, b, agent.cfg.action_dim), torch.zeros(h, b, 1),
    )
    weighted = sum(0.5 ** depth * (depth + 1) ** 2 for depth in range(h))
    scale = (1.0 / h if normalization == "divide_horizon" else
             ((1.0 + 0.5 + 0.25) / 3.0) / sum(0.5 ** depth for depth in range(h)))
    torch.testing.assert_close(loss, torch.tensor(weighted * scale))


def test_one_replay_sample_updates_auxiliary_once_with_explicit_loss_weight(agents):
    agent = agents(aux_return_critic_coef=0.17)
    data = batch(agent)
    draws = []

    def sample():
        draws.append(1)
        return (*data, None)

    info = agent.update(SimpleNamespace(sample=sample))
    expected = (
        agent.cfg.consistency_coef * info["consistency_loss"]
        + agent.cfg.reward_coef * info["reward_loss"]
        + agent.cfg.termination_coef * info["termination_loss"]
        + agent.cfg.critic_coef * info["critic_loss"]
        + 0.17 * info["aux_return_critic_loss"]
    )
    torch.testing.assert_close(info["total_loss"], expected)
    assert len(draws) == 1
    assert agent.num_updates == agent.aux_return.num_updates == 1
    for parameter in agent.model._aux_return_Qs.parameters():
        assert agent.optim.state[parameter]["step"].item() == 1
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in info.values())


def test_zero_weight_auxiliary_preserves_primary_update_and_rng(agents):
    torch.manual_seed(55)
    original = agents(aux_return_mode="off", dropout=0.1)
    torch.manual_seed(55)
    augmented = agents(aux_return_critic_coef=0.0, dropout=0.1)
    data = batch(original)
    before = torch.random.get_rng_state().clone()
    original._update(*data)
    after = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(before)
    augmented._update(*data)
    assert torch.equal(torch.random.get_rng_state(), after)
    for name, value in original.model.state_dict().items():
        torch.testing.assert_close(augmented.model.state_dict()[name], value, rtol=0, atol=0)


def test_auxiliary_checkpoint_round_trip_preserves_weights_optimizer_and_private_rng(agents):
    source = agents()
    source._update(*batch(source))
    payload = deepcopy(source.checkpoint_state())
    assert payload["aux_return_spec"]["target"] == "reward_only"
    assert payload["aux_return_spec"]["detach_representation"] is True
    assert any(key.startswith("_aux_return_Qs.") for key in payload["model"])
    assert any(key.startswith("_target_aux_return_Qs.") for key in payload["model"])
    target = agents()
    target.load(deepcopy(payload))
    _assert_tree_equal(target.checkpoint_state(), payload)


@pytest.mark.parametrize("corruption", ["missing_state", "wrong_spec", "bad_rng", "missing_weight"])
def test_auxiliary_checkpoint_rejects_corruption_before_mutation(agents, corruption):
    source, target = agents(), agents()
    source._update(*batch(source))
    payload = deepcopy(source.checkpoint_state())
    if corruption == "missing_state":
        payload.pop("aux_return_state")
    elif corruption == "wrong_spec":
        payload["aux_return_spec"]["detach_representation"] = False
    elif corruption == "bad_rng":
        payload["aux_return_state"]["rng_cpu"] = torch.zeros(3, dtype=torch.uint8)
    else:
        key = next(key for key in payload["model"] if key.startswith("_aux_return_Qs."))
        payload["model"].pop(key)
    before = deepcopy(target.checkpoint_state())
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        target.load(payload)
    _assert_tree_equal(target.checkpoint_state(), before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_auxiliary_checkpoint_requires_matching_enabled_configuration(agents):
    enabled, disabled = agents(), agents(aux_return_mode="off")
    with pytest.raises((ValueError, RuntimeError)):
        enabled.load(deepcopy(disabled.checkpoint_state()))
    with pytest.raises((ValueError, RuntimeError)):
        disabled.load(deepcopy(enabled.checkpoint_state()))


def test_auxiliary_ensembles_compile_fullgraph_and_match_eager_update(agents, monkeypatch):
    real_compile = torch.compile
    compile_calls = []

    def compile_eager(function, **kwargs):
        compile_calls.append(kwargs)
        return real_compile(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", compile_eager)
    eager = agents(q_representation="distributional")
    compiled = agents(q_representation="distributional")
    compiled.load(deepcopy(eager.checkpoint_state()))
    # Exercise the added graph regions; the historical actor's detached
    # parameter-view path itself does not support strict fullgraph on PyTorch 2.3.
    compiled.model._aux_return_Qs.enable_compile(strict=True)
    compiled.model._target_aux_return_Qs.enable_compile(strict=True)
    data = batch(eager)
    rng = torch.random.get_rng_state().clone()
    expected = eager._update(*data)
    torch.random.set_rng_state(rng)
    actual = compiled._update(*data)
    assert compile_calls and all(call["fullgraph"] for call in compile_calls)
    assert actual["compile_fallback"] == 0.0
    assert not compiled.model._aux_return_Qs.compile_failed
    assert not compiled.model._target_aux_return_Qs.compile_failed
    for name in ("aux_return_critic_loss", "critic_loss", "actor_loss", "total_loss"):
        torch.testing.assert_close(actual[name], expected[name])
    for name, value in eager.model.state_dict().items():
        torch.testing.assert_close(compiled.model.state_dict()[name], value)
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in actual.values())


@pytest.mark.parametrize("invalid", [
    {"aux_return_mode": "return_actor"},
    {"aux_return_detach_representation": 1},
    {"aux_return_critic_coef": -0.1},
    {"aux_return_critic_lr": 0.0},
    {"inner_operator": "sac"},
])
def test_auxiliary_backport_rejects_unsupported_or_invalid_configuration(invalid):
    options = _tiny_params(inner_operator="none", aux_return_mode="sac")
    options.update(invalid)
    if options["inner_operator"] == "none":
        for name in ("inner_rounds", "inner_rollouts_per_round", "inner_updates_per_round"):
            options[name] = None
    with pytest.raises((ValueError, TypeError)):
        _build_cfg(**options)
