"""Production ensemble routing through complete learner and inner updates."""

from copy import deepcopy

import gymnasium as gym
import pytest
import torch

from RL.TDMPC2 import TDMPC2Baseline
from RL.tdmpc2_core.common import layers
from RL.tdmpc2_core.common.lora import LoRARLLinear
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_pave_integration import _batch, _nonconstant_critics, _weights
from tests.test_tdambi_checkpoint import make_tdambi, tiny_native_params


class _ResidentBatch:
    def __init__(self, batch):
        self.batch = batch
        self.samples = 0

    def sample(self):
        self.samples += 1
        return (*self.batch, None)


@pytest.fixture
def wrappers():
    instances = []

    def make(**overrides):
        options = dict(inner_rounds=1, inner_updates_per_round=2)
        options.update(overrides)
        wrapper = _tiny_model(**options)
        instances.append(wrapper)
        return wrapper

    yield make
    for wrapper in instances:
        wrapper.close()
        wrapper.env.close()


def _assert_state_unchanged(module, before):
    assert module.state_dict().keys() == before.keys()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def _assert_optimizer_steps(optimizer, parameters, count):
    for parameter in parameters:
        assert optimizer.state[parameter]["step"].item() == count


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("pave_enabled", [False, True])
def test_default_two_public_updates_with_actual_aot_autograd(
    wrappers, monkeypatch, representation, pave_enabled,
):
    """AOT owns ordinary losses; higher-order PAVE uses production eager heads."""
    actual_compile = torch.compile
    requested_regions = []

    def aot_compile(function, **options):
        assert options.get("fullgraph") is True
        requested_regions.append(getattr(function, "__name__", str(function)))
        return actual_compile(function, backend="aot_eager", **options)

    monkeypatch.setattr(torch, "compile", aot_compile)
    torch._dynamo.reset()
    try:
        wrapper = wrappers(
            compile=True, compile_strict=True, aux_return_mode="sac", dropout=.2,
            q_representation=representation,
            num_q=5 if representation == "distributional" else 2,
            **(_weights("aux_return_pave") if pave_enabled else {}),
        )
        agent = wrapper.agent
        assert agent.cfg.critic_ensemble_backend == "vmap"
        for name in ("_Qs", "_target_Qs", "_aux_return_Qs", "_target_aux_return_Qs"):
            assert getattr(agent.model, name).backend == "vmap"
        _nonconstant_critics(agent)
        parameter_ids = tuple(id(p) for p in agent.model.parameters())
        critic_keys = tuple(agent.model._Qs.state_dict())
        actor_before = deepcopy(agent.model._pi.state_dict())
        target_before = deepcopy(agent.model._target_Qs.state_dict())
        buffer = _ResidentBatch(_batch(agent))
        for step in (1, 2):
            info = agent.update(buffer)
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in info.values())
            assert agent.num_updates == agent.aux_return.num_updates == step
            assert not agent._outer_update_region.failed
            assert ("aux_return_pave_weighted_loss" in info) is pave_enabled
            if pave_enabled:
                assert info["aux_return_pave_sample_count"] == 4
            for name in ("_Qs", "_aux_return_Qs"):
                _assert_optimizer_steps(agent.optim, getattr(agent.model, name).parameters(), step)
            _assert_optimizer_steps(agent.pi_optim, agent.model._pi.parameters(), step)
            _assert_optimizer_steps(agent.ent_coef_optim, [agent.log_ent_coef], step)
        assert buffer.samples == 2
        assert tuple(id(p) for p in agent.model.parameters()) == parameter_ids
        assert tuple(agent.model._Qs.state_dict()) == critic_keys
        assert "_outer_update_kernel" in requested_regions
        assert agent._outer_update_region._compiled is not None
        assert any(not torch.equal(value, actor_before[name])
                   for name, value in agent.model._pi.state_dict().items())
        assert any(not torch.equal(value, target_before[name])
                   for name, value in agent.model._target_Qs.state_dict().items())
        for name in ("_Qs", "_target_Qs", "_aux_return_Qs", "_target_aux_return_Qs"):
            assert not getattr(agent.model, name).compile_failed
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("representation,source,split", [
    ("scalar", "sac", False),
    ("distributional", "aux_return", False),
    ("distributional", "sac", True),
])
@pytest.mark.parametrize("backend", ["vmap", "loop"])
def test_factories_targets_and_dense_inner_copies_keep_selected_backend(
    wrappers, representation, source, split, backend,
):
    overrides = dict(q_representation=representation, inner_critic_source=source)
    if split:
        overrides.update(critic_value_mode="return_entropy", ent_coef=.2,
                         inner_temperature_mode="inherit_outer")
    else:
        overrides["aux_return_mode"] = "sac"
    # Exercise the default without redundantly setting the new control.
    if backend == "loop":
        overrides["critic_ensemble_backend"] = backend
    wrapper = wrappers(**overrides)
    model, engine = wrapper.agent.model, wrapper.agent.inner_engine
    assert wrapper.cfg.critic_ensemble_backend == backend
    assert model._Qs.backend == model._target_Qs.backend == backend
    if not split:
        assert model._aux_return_Qs.backend == model._target_aux_return_Qs.backend == backend
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    assert engine.state.critic.backend == engine.state.critic_target.backend == backend
    prior = model._aux_return_Qs if source == "aux_return" else model._Qs
    assert engine.state.critic is not prior
    assert set(map(id, engine.state.critic.parameters())).isdisjoint(map(id, prior.parameters()))
    inputs = torch.randn(4, wrapper.cfg.latent_dim + wrapper.cfg.action_dim)
    torch.testing.assert_close(engine.state.critic(inputs), prior(inputs))


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_full_inner_adaptation_preserves_outer_and_lora_falls_back(wrappers, adaptation):
    from RL.tdmpc2_core.common.ensemble_vmap import ensemble_vmap_spec

    wrapper = wrappers(inner_critic_adaptation=adaptation,
                       inner_critic_target_tau=1., aux_return_mode="sac",
                       inner_critic_source="aux_return",
                       **({"inner_critic_lora_rank": 2} if adaptation == "lora_rl" else {}))
    agent = wrapper.agent
    _nonconstant_critics(agent)
    before = deepcopy(agent.model.state_dict())
    rng = torch.random.get_rng_state().clone()
    action = agent.act(torch.tensor([.2, -.3, .7]), collect_diagnostics=False)
    assert torch.isfinite(action).all()
    engine, state = agent.inner_engine, agent.inner_engine._action_pool
    assert state.critic.backend == state.critic_target.backend == "vmap"
    assert engine.state.critic_steps == engine.state.actor_steps == 2
    assert engine.state.critic_target_steps == 2
    assert not any(isinstance(layer, LoRARLLinear) for layer in state.critic_target.modules())
    spec = ensemble_vmap_spec(state.critic.modules_list, layers.NormedLinear)
    assert (spec is None) is (adaptation == "lora_rl")
    if adaptation == "lora_rl":
        assert any(torch.count_nonzero(layer.lora_B)
                   for layer in state.critic.modules() if isinstance(layer, LoRARLLinear))
    _assert_state_unchanged(agent.model, before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in agent.model.parameters())


def test_native_checkpoint_transfer_uses_default_vmap_with_numerical_parity():
    """Native baseline stays loop; TDAMBI's transferred dense networks use vmap."""
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    native = adapted = None
    try:
        native = TDMPC2Baseline(
            "TDMPC2", env, tiny_native_params(),
            {"seed": 3, "device": "cpu", "env": "Pendulum-v1", "total_steps": 10}, {},
        )
        adapted = make_tdambi()
        source, target = native.agent.model, adapted.agent.model
        with torch.no_grad():
            for critic in (source._Qs, source._target_Qs):
                for head in critic:
                    head[-1].weight.normal_(std=.1)
                    head[-1].bias.normal_(std=.1)
        adapted.agent.load(deepcopy(native.agent.checkpoint_state()))
        source.eval()
        target.eval()
        assert source._Qs.backend == source._target_Qs.backend == "loop"
        assert adapted.cfg.critic_ensemble_backend == "vmap"
        assert target._Qs.backend == target._target_Qs.backend == "vmap"
        observations = torch.tensor([[.2, -.3, .7], [-.1, .0, .4]])
        z = source.encode(observations, None)
        torch.testing.assert_close(target.encode(observations), z, rtol=0, atol=0)
        torch.manual_seed(71)
        action, _ = source.pi(z, None)
        torch.manual_seed(71)
        copied_action, _ = target.pi_tdmpc2(z)
        torch.testing.assert_close(copied_action, action, rtol=0, atol=0)
        inputs = torch.cat((z, action), dim=-1)
        for name in ("_Qs", "_target_Qs"):
            torch.testing.assert_close(getattr(target, name)(inputs), getattr(source, name)(inputs),
                                       rtol=1e-5, atol=1e-6)
        with adapted.agent.inner_engine.rng.fork("initialization"):
            adapted.agent.inner_engine._prepare_workspace(t0=True)
        state = adapted.agent.inner_engine.state
        assert state.critic.backend == state.critic_target.backend == "vmap"
        torch.testing.assert_close(state.critic_target(inputs), target._target_Qs(inputs),
                                   rtol=1e-5, atol=1e-6)
    finally:
        if native is not None:
            native.close()
        if adapted is not None:
            adapted.close()
            adapted.env.close()
        env.close()
