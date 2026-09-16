"""Independent reward targets, shared representation training and actor state."""

from copy import deepcopy

import pytest
import torch

from tests.test_ambi_root_local_sac import _tiny_model


@pytest.fixture
def agents():
    models = []

    def create(**kwargs):
        options = {"aux_return_mode": "return_actor", "aux_return_ent_coef": .1}
        options.update(kwargs)
        model = _tiny_model(**options)
        models.append(model)
        return model.agent

    yield create
    for model in models:
        model.env.close()


def batch(agent):
    h, b = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    return (torch.randn(h + 1, b, 3), torch.randn(h, b, 1).tanh(),
            torch.randn(h, b, 1), torch.zeros(h, b, 1))


def nonzero_outputs(agent):
    with torch.no_grad():
        for member in agent.model._aux_return_Qs:
            member[-1].weight.normal_(std=.1)
        agent.model._target_aux_return_Qs.load_state_dict(agent.model._aux_return_Qs.state_dict())


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
def test_auxiliary_target_uses_selected_actor_and_reward_only(agents, monkeypatch, mode):
    agent = agents(aux_return_mode=mode)
    aux = agent.aux_return
    calls = []
    def selected(z, **kwargs):
        calls.append("return")
        return torch.full_like(z[..., :1], .75), {"log_prob": z[..., :1] * 0 - 1000.}
    def sac(z, **kwargs):
        calls.append("sac")
        return torch.full_like(z[..., :1], -.25)
    def value(z, action, **kwargs):
        assert kwargs == {"target": True, "reduction": aux.cfg.outer_q_target_reduction}
        assert action[0].item() == (.75 if mode == "return_actor" else -.25)
        return z.new_tensor([[3.], [float("nan")]])
    monkeypatch.setattr(aux.model, "pi", selected)
    monkeypatch.setattr(agent.model, "pi_action", sac)
    monkeypatch.setattr(aux.model, "Q", value)
    target = aux.td_target(torch.zeros(2, agent.cfg.latent_dim), torch.tensor([[2.], [4.]]), torch.tensor([[0.], [1.]]))
    torch.testing.assert_close(target, torch.tensor([[2. + agent.discount * 3.], [4.]]))
    assert calls == ["return" if mode == "return_actor" else "sac"]
    assert not target.requires_grad


@pytest.mark.parametrize("detach", [False, True])
def test_auxiliary_loss_gradient_routes_and_detach_option(agents, detach):
    agent = agents(aux_return_detach_representation=detach)
    nonzero_outputs(agent)
    obs, action, reward, _ = batch(agent)
    z = agent.model.encode(obs[0])
    latents = [z]
    for a in action:
        z = agent.model.next(z, a)
        latents.append(z)
    loss, _ = agent.aux_return.critic_loss(torch.stack(latents), action, reward)
    loss.backward()
    for module in (agent.model._encoder, agent.model._dynamics):
        total = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
        assert (total > 0) is (not detach)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.model._aux_return_Qs.parameters())
    for module in (agent.model._Qs, agent.model._pi, agent.model._return_pi, agent.model._target_aux_return_Qs):
        assert all(p.grad is None for p in module.parameters())


@pytest.mark.parametrize("coefficient", [0., .1, "auto_0.2"])
def test_actor_and_temperature_are_independent(agents, coefficient):
    agent = agents(aux_return_ent_coef=coefficient)
    nonzero_outputs(agent)
    zs = torch.randn(3, 2, agent.cfg.latent_dim, requires_grad=True)
    primary = deepcopy(agent.model._pi.state_dict())
    primary_alpha = agent.alpha.detach().clone()
    auxiliary = deepcopy(agent.model._return_pi.state_dict())
    metrics = agent.aux_return.update_actor(zs)
    assert metrics["aux_return_actor_grad_norm"] > 0
    assert zs.grad is None
    torch.testing.assert_close(agent.alpha, primary_alpha)
    for key, value in primary.items():
        torch.testing.assert_close(agent.model._pi.state_dict()[key], value, rtol=0, atol=0)
    assert any(not torch.equal(value, agent.model._return_pi.state_dict()[key]) for key, value in auxiliary.items())
    for module in (agent.model._Qs, agent.model._aux_return_Qs, agent.model._encoder, agent.model._dynamics):
        assert all(p.grad is None for p in module.parameters())
    if isinstance(coefficient, str):
        assert agent.aux_return.ent_coef_optim is not agent.ent_coef_optim
        assert agent.aux_return.alpha.item() != pytest.approx(.2)
    else:
        assert agent.aux_return.ent_coef_optim is None
        assert agent.aux_return.alpha.item() == pytest.approx(coefficient)


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
def test_joint_update_adds_auxiliary_loss_and_preserves_update_dose(agents, mode):
    agent = agents(aux_return_mode=mode, aux_return_critic_coef=.17)
    info = agent._update(*batch(agent))
    expected = (agent.cfg.consistency_coef * info["consistency_loss"]
                + agent.cfg.reward_coef * info["reward_loss"]
                + agent.cfg.termination_coef * info["termination_loss"]
                + agent.cfg.critic_coef * info["critic_loss"]
                + .17 * info["aux_return_critic_loss"])
    torch.testing.assert_close(info["total_loss"], expected)
    assert agent.num_updates == agent.aux_return.num_updates == 1
    for param in agent.model._aux_return_Qs.parameters():
        assert agent.optim.state[param]["step"].item() == 1
    assert (agent.aux_return.pi_optim is not None) == (mode == "return_actor")
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in info.values())


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
def test_auxiliary_sampling_and_dropout_preserve_primary_rng(agents, mode):
    agent = agents(aux_return_mode=mode, dropout=.2)
    agent.model.train()
    obs, action, reward, terminated = batch(agent)
    zs = torch.randn(3, 2, agent.cfg.latent_dim)
    before = torch.random.get_rng_state().clone()
    target = agent.aux_return.td_target(zs[1:], reward, terminated)
    loss, _ = agent.aux_return.critic_loss(zs, action, target)
    loss.backward()
    agent.aux_return.update_actor(zs)
    assert torch.equal(before, torch.random.get_rng_state())


def test_sac_bootstrap_tracks_independent_return_scale_without_actor(agents):
    agent = agents(aux_return_mode="sac", aux_return_sac_actor_loss_scale_mode="tdmpc2_percentile_range",
                   aux_return_sac_actor_loss_scale_tau=1.)
    aux = agent.aux_return
    assert not aux.has_actor and aux.pi_optim is None
    assert aux.actor_loss_scale_enabled
    aux.model.Q = lambda *a, **k: torch.tensor([[0.], [20.]])
    aux.update_actor(torch.zeros(3, 2, agent.cfg.latent_dim))
    assert aux.actor_loss_scale.item() == pytest.approx(18.)
    assert not agent.actor_loss_scale_enabled


@pytest.mark.parametrize("schedule", ["smooth", "quantile_gate", "dual"])
def test_return_actor_behavior_regularizer_has_independent_state(agents, schedule):
    agent = agents(aux_return_outer_behavior_policy_kl_schedule=schedule,
                   aux_return_outer_behavior_policy_kl_min_valid_count=1)
    obs, action, reward, terminated = batch(agent)
    metadata = dict(behavior_pre_tanh_mean=torch.zeros_like(action),
                    behavior_log_std=torch.zeros_like(action),
                    behavior_policy_valid=torch.ones_like(action, dtype=torch.bool))
    result = agent._update(obs, action, reward, terminated, **metadata)
    assert "aux_return_behavior_policy_kl" in result
    assert not agent.behavior_policy_kl_enabled
    assert agent.cfg.store_behavior_policy
    if schedule == "dual":
        assert agent.aux_return.behavior_policy_kl_dual_updates == 1
        assert agent.behavior_policy_kl_optim is None
    if schedule == "smooth":
        assert agent.aux_return.behavior_policy_kl_eligible_updates == 1
        assert agent.behavior_policy_kl_eligible_updates == 0
    state = deepcopy(agent.aux_return.checkpoint_state())
    agent.aux_return.preflight_state(state, exact=True, expected_updates=1)
    agent.aux_return.load_state(state, exact=True)


def test_helper_rejects_invalid_checkpoint_before_mutation(agents):
    agent = agents(aux_return_ent_coef="auto_0.2")
    agent._update(*batch(agent))
    aux = agent.aux_return
    before = aux.alpha.detach().clone()
    state = deepcopy(aux.checkpoint_state())
    state["log_ent_coef"].fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        aux.preflight_state(state, exact=True, expected_updates=1)
    torch.testing.assert_close(aux.alpha, before)
    state = deepcopy(aux.checkpoint_state())
    state["num_updates"] = True
    with pytest.raises(ValueError, match="nonnegative integer"):
        aux.preflight_state(state)


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
def test_zero_weight_auxiliary_preserves_primary_training_and_rng(agents, mode):
    original = agents(aux_return_mode="off")
    augmented = agents(aux_return_mode=mode, aux_return_critic_coef=0.)
    inputs = batch(original)
    initial_rng = torch.random.get_rng_state().clone()
    original._update(*inputs)
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(initial_rng)
    augmented._update(*inputs)
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    for name, value in original.model.state_dict().items():
        torch.testing.assert_close(value, augmented.model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [("exp_avg", float("nan")), ("exp_avg_sq", -1.)])
def test_helper_rejects_corrupt_optimizer_moments(agents, field, value):
    agent = agents()
    agent._update(*batch(agent))
    state = deepcopy(agent.aux_return.checkpoint_state())
    first = next(iter(state["pi_optim"]["state"].values()))
    first[field].fill_(value)
    with pytest.raises(ValueError, match="finite|nonnegative"):
        agent.aux_return.preflight_state(state, exact=True, expected_updates=1)


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
def test_compiled_joint_auxiliary_update_is_finite_without_fallback(agents, mode):
    agent = agents(aux_return_mode=mode, compile=True, compile_strict=True,
                   q_representation="distributional")
    metrics = agent._update(*batch(agent))
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
    assert metrics["compile_fallback"] == 0.
    assert not agent.model._aux_return_Qs.compile_failed
    assert not agent.model._target_aux_return_Qs.compile_failed
    assert agent.num_updates == agent.aux_return.num_updates == 1


@pytest.mark.parametrize("target", [False, True])
def test_auxiliary_compile_failures_are_reported_in_aggregate(agents, monkeypatch, target):
    from RL.tdmpc2_core.common.layers import Ensemble
    agent = agents(aux_return_mode="sac")
    failed = agent.model._target_aux_return_Qs if target else agent.model._aux_return_Qs
    original = Ensemble.compile_failed
    monkeypatch.setattr(Ensemble, "compile_failed", property(
        lambda ensemble: ensemble is failed or original.fget(ensemble),
    ))
    metrics = agent._update(*batch(agent))
    assert metrics["compile_fallback"] == 1.
    assert metrics["compile_aux_return_target_fallback"] == float(target)
    assert metrics["compile_aux_return_online_fallback"] == float(not target)
