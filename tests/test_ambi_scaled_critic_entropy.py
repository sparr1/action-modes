"""Raw-return Bellman entropy coefficients under Q-only actor normalization."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree, _model
from tests.test_ambi_inner_entropy_modes import _controlled_policy, _prepare
from tests.test_td_ambi_inner_objectives import (
    _engine, _literal_critic_loss, _literal_policy, _literal_values,
)


@pytest.mark.parametrize("mode", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("target_mode", ["reward_only", "entropy_augmented"])
def test_outer_target_uses_current_scale_in_raw_return_units(monkeypatch, mode, target_mode):
    model = _model(
        ent_coef=0.25, outer_actor_entropy_mode=mode,
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        outer_critic_target=target_mode, inner_sac_critic_target=target_mode,
    )
    try:
        agent = model.agent
        agent.actor_loss_scale.fill_(7.0)
        requests = []

        def policy(z, **kwargs):
            requests.append(kwargs.get("include_scaled_entropy", False))
            info = {"log_prob": z[..., :1] * 0 - 0.4}
            if kwargs.get("include_scaled_entropy"):
                info["scaled_entropy"] = z[..., :1] * 0 - 2.0
            return z[..., :1] * 0, info

        monkeypatch.setattr(agent.model, "pi", policy)
        monkeypatch.setattr(agent.model, "Q", lambda z, a, **kw: z[..., :1] * 0 + 3.0)
        z = torch.randn(2, 4, agent.cfg.latent_dim, requires_grad=True)
        reward = torch.ones(2, 4, 1, requires_grad=True)
        terminated = torch.zeros_like(reward)
        terminated[:, 1] = 1.0
        entropy = 0.4 if mode == "squashed" else -2.0
        for scale in (7.0, 11.0):
            agent.actor_loss_scale.fill_(scale)
            actual = agent._soft_td_target(z, reward, terminated)
            bonus = 0.25 * scale * entropy if target_mode == "entropy_augmented" else 0.0
            expected = reward + agent.discount * (1 - terminated) * (3.0 + bonus)
            torch.testing.assert_close(actual, expected)
            assert not actual.requires_grad
            assert agent.actor_loss_scale.item() == scale
        assert requests == [mode == "tdmpc2_scaled" and target_mode == "entropy_augmented"] * 2
    finally:
        model.env.close()


def _soft_engine(**overrides):
    return _engine(
        ent_coef=0.25, outer_critic_target="entropy_augmented",
        inner_sac_critic_target="entropy_augmented",
        inner_q_target_reduction="min_pair", **overrides,
    )


def _batch(engine, size=8):
    cfg = engine.cfg
    return {
        "z": torch.randn(size, cfg.latent_dim),
        "action": torch.randn(size, cfg.action_dim).tanh(),
        "reward": torch.linspace(-0.25, 0.75, size).unsqueeze(-1),
        "next_z": torch.randn(size, cfg.latent_dim),
        "terminated": (torch.arange(size) % 3 == 0).float().unsqueeze(-1),
    }


@pytest.mark.parametrize("saturated", [False, True])
def test_inner_scaled_target_loss_and_gradient_match_literal_reference(saturated):
    engine = _soft_engine()
    state, cfg = engine.state, engine.cfg
    if saturated:
        with torch.no_grad():
            state.actor[-1].bias[0] = 20.0
    batch = _batch(engine)
    noise = torch.randn(8, cfg.action_dim)
    pair = torch.tensor([0, 3])
    alpha = torch.tensor(0.25, requires_grad=True)
    local_scale = torch.tensor([9.0], requires_grad=True)
    outer_scale = engine.agent.actor_loss_scale.clone()
    reference_critic = deepcopy(state.critic)
    with torch.no_grad():
        next_action, entropy = _literal_policy(state.actor, batch["next_z"], noise, cfg)
        next_q = _literal_values(state.critic_target, batch["next_z"], next_action, cfg)[pair].min(0).values
        target = batch["reward"] + engine.agent.discount * (1 - batch["terminated"]) * (
            next_q + 0.25 * 9.0 * entropy
        )
    expected_loss = _literal_critic_loss(
        reference_critic, batch["z"], batch["action"], target, cfg,
    )
    actual = engine._sac_critic_kernel(
        *batch.values(), alpha, noise, pair, actor_loss_scale=local_scale,
    )
    torch.testing.assert_close(actual[2], target)
    torch.testing.assert_close(actual[0], expected_loss)
    actual[0].backward()
    expected_loss.backward()
    assert max(p.grad.abs().max() for p in state.critic.parameters()) > 1e-6
    for parameter, reference in zip(state.critic.parameters(), reference_critic.parameters()):
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=5e-5, atol=2e-7)
    assert not actual[2].requires_grad
    assert local_scale.grad is None and alpha.grad is None
    torch.testing.assert_close(engine.agent.actor_loss_scale, outer_scale, rtol=0, atol=0)


@pytest.mark.parametrize("scale_update", ["per_action", "per_update"])
def test_paired_updates_use_private_scale_before_each_actor_ema(monkeypatch, scale_update):
    engine = _soft_engine(inner_actor_loss_scale_update=scale_update, sac_actor_loss_scale_tau=0.5)
    batch = _batch(engine, 16)
    engine.state.replay.add_batch(*batch.values())
    local_scale = torch.tensor([9.0])
    outer_scale = engine.agent.actor_loss_scale.clone()
    outer_parameters = [p.detach().clone() for p in engine.model.parameters()]
    original_kernel = engine._sac_critic_kernel
    original_policy_step = engine._sac_policy_step
    critic_scales, actor_scales = [], []
    pair = torch.tensor([0, 3])

    def critic_kernel(*args, **kwargs):
        scale = kwargs["actor_loss_scale"]
        assert scale is local_scale
        critic_scales.append(scale.clone())
        z, action, reward, next_z, terminated, alpha, noise, _ = args
        with torch.no_grad():
            next_action, entropy = _literal_policy(engine.state.actor, next_z, noise, engine.cfg)
            q = _literal_values(engine.state.critic_target, next_z, next_action, engine.cfg)[pair].min(0).values
            expected = reward + engine.agent.discount * (1 - terminated) * (q + alpha * scale * entropy)
        outputs = original_kernel(*args[:-1], pair, **kwargs)
        torch.testing.assert_close(outputs[2], expected)
        return outputs

    def actor_step(*args, **kwargs):
        before = local_scale.clone()
        result = original_policy_step(*args, **kwargs)
        actor_scales.append((before, local_scale.clone()))
        return result

    monkeypatch.setitem(engine._compile_regions, "critic", critic_kernel)
    monkeypatch.setattr(engine, "_sac_policy_step", actor_step)
    engine._run_update_counts(critic_count=3, actor_count=3, temperature_count=0, actor_loss_scale=local_scale)
    assert len(critic_scales) == len(actor_scales) == 3
    torch.testing.assert_close(critic_scales[0], torch.tensor([9.0]))
    for index, (before, after) in enumerate(actor_scales):
        torch.testing.assert_close(critic_scales[index], before)
        if index < 2:
            torch.testing.assert_close(critic_scales[index + 1], after)
    if scale_update == "per_action":
        torch.testing.assert_close(local_scale, torch.tensor([9.0]), rtol=0, atol=0)
    else:
        assert not torch.equal(local_scale, torch.tensor([9.0]))
    torch.testing.assert_close(engine.agent.actor_loss_scale, outer_scale, rtol=0, atol=0)
    for parameter, original in zip(engine.model.parameters(), outer_parameters):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)


def test_scaled_inner_target_compiles_with_detached_scale_and_backward():
    engine = _soft_engine()
    batch = _batch(engine)
    scale = torch.tensor([8.0], requires_grad=True)
    args = (*batch.values(), engine.alpha.detach(), torch.randn(8, engine.cfg.action_dim), torch.tensor([0, 3]))
    eager = engine._sac_critic_kernel(*args, actor_loss_scale=scale)
    eager[0].backward()
    gradients = [p.grad.clone() for p in engine.state.critic_params]
    engine.state.critic_optim.zero_grad(set_to_none=True)
    compiled = torch.compile(engine._sac_critic_kernel, backend="aot_eager", fullgraph=True)
    actual = compiled(*args, actor_loss_scale=scale)
    actual[0].backward()
    for value, expected in zip(actual, eager):
        torch.testing.assert_close(value, expected)
    for parameter, gradient in zip(engine.state.critic_params, gradients):
        torch.testing.assert_close(parameter.grad, gradient, rtol=5e-5, atol=2e-7)
    assert scale.grad is None and scale.item() == 8.0


@pytest.mark.parametrize("clip", [None, 20.0])
def test_temperature_none_is_unclipped_and_preserves_norm_metric(monkeypatch, clip):
    model = _model(
        ent_coef="auto_0.0001", outer_actor_entropy_mode="tdmpc2_scaled",
        target_entropy=-441, inner_actor_entropy_mode="tdmpc2_scaled",
        inner_target_entropy=-441, inner_temperature_mode="auto",
        inner_temperature_initialization="inherit_outer",
        inner_temperature_updates_per_action=1, inner_temperature_grad_clip_norm=clip,
    )
    try:
        engine = _prepare(model)
        parameter = torch.nn.Parameter(torch.tensor([0.0, -3.0, -541.0, 0.0]))
        monkeypatch.setattr(engine.model, "pi", _controlled_policy(engine, parameter, []))
        alpha_before = engine.alpha.detach().clone()
        engine.state.temperature_optim = torch.optim.SGD([engine.state.log_alpha], lr=0.001)
        metrics = engine._sac_policy_step(
            {"z": torch.zeros(4, engine.cfg.latent_dim)},
            update_actor=False, update_temperature=True, alpha=alpha_before,
        )
        expected_gradient = -100.0 if clip is None else -20.0
        torch.testing.assert_close(engine.state.log_alpha.grad, torch.tensor(expected_gradient))
        torch.testing.assert_close(metrics["temperature_grad_norm"], torch.tensor(100.0))
        torch.testing.assert_close(engine.alpha, alpha_before * torch.exp(torch.tensor(-0.001 * expected_gradient)))
        assert engine.alpha > alpha_before
        assert parameter.grad is None
    finally:
        model.env.close()


@pytest.mark.parametrize("corruption", ["missing", "wrong"])
def test_scaled_target_units_are_exact_resume_state_but_portable_provenance(corruption):
    engine = _soft_engine()
    # Resume snapshots require a genuine action boundary, rather than the
    # manually prepared inner workspace used by the numerical kernel fixtures.
    agent = AMBITDMPC2Agent(deepcopy(engine.cfg))
    agent.prepare_training_resume_boundary()
    pristine = _clone_tree(agent.training_state_dict())
    bad = _clone_tree(pristine)
    spec = bad["outer"]["critic_target_spec"]["entropy_semantics"]
    if corruption == "missing":
        spec.pop("coefficient_units")
    else:
        spec["coefficient_units"]["inner"] = "alpha"
    with pytest.raises(ValueError, match="critic-target specification"):
        agent.load_training_state_dict(bad)
    _assert_tree_equal(agent.training_state_dict(), pristine)
    agent.load(bad["outer"])


def test_fresh_inner_value_probe_inherits_outer_target_initialization():
    engine = _soft_engine()
    assert engine.agent._value_equivalence_reference_critic() is engine.model._target_Qs
    engine.cfg.inner_critic_target_initialization = "online"
    assert engine.agent._value_equivalence_reference_critic() is engine.model._Qs
