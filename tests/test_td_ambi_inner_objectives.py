"""TD-MPC2 loss formulas on AMBI's retained transition-minibatch inner loop."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.common.compile_regions import CompileRegion
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _model


def _engine(**overrides):
    settings = dict(
        q_representation="distributional", num_q=5, q_pair_size=2,
        log_std_mapping="tdmpc2_tanh", log_std_min=-10.0, log_std_max=2.0,
        outer_actor_entropy_mode="tdmpc2_scaled", inner_actor_entropy_mode="tdmpc2_scaled",
        outer_q_actor_reduction="mean_pair", inner_q_actor_reduction="mean_pair",
        outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
        ent_coef=0.0001, inner_temperature_mode="inherit_outer",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        sac_actor_loss_scale_tau=0.01, inner_actor_loss_scale_update="per_update",
        inner_critic_loss_coef=0.1, inner_actor_adam_eps=1e-5, inner_adam_eps=1e-8,
        inner_actor_lr=3e-4, inner_critic_lr=3e-4,
        inner_critic_target_initialization="outer_target",
        inner_critic_target_tau=0.01,
    )
    settings.update(overrides)
    holder = _model(**settings)
    try:
        cfg = deepcopy(holder.cfg)
        # Three action coordinates allow a saturated coordinate and nonzero Q
        # action gradients to coexist in the same literal-reference test.
        cfg.action_dim = 3
        agent = AMBITDMPC2Agent(cfg)
    finally:
        holder.env.close()
    agent.actor_loss_scale.fill_(3.75)
    with torch.no_grad():
        for head, critic in enumerate(agent.model._Qs):
            output = critic[-1]
            bins = torch.linspace(-1.0, 1.0, output.out_features)
            features = torch.linspace(-1.0, 1.0, output.in_features)
            output.weight.copy_((2.0 + 0.1 * head) * torch.outer(bins, features))
            output.bias.copy_((0.1 + 0.03 * head) * bins)
        agent.model._target_Qs.load_state_dict(agent.model._Qs.state_dict())
        for critic in agent.model._target_Qs:
            critic[-1].bias.add_(torch.linspace(-0.3, 0.3, critic[-1].out_features))
    engine = agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    return engine


def _literal_policy(actor, z, noise, cfg):
    mean, raw = actor(z).chunk(2, dim=-1)
    log_std = cfg.inner_log_std_min + 0.5 * (
        cfg.inner_log_std_max - cfg.inner_log_std_min
    ) * (torch.tanh(raw) + 1.0)
    action = torch.tanh(mean + noise * log_std.exp())
    gaussian = (-0.5 * noise.square() - log_std - 0.9189385175704956).sum(-1, keepdim=True)
    log_prob = gaussian - torch.log(torch.relu(1.0 - action.square()) + 1e-6).sum(-1, keepdim=True)
    entropy = -log_prob * (cfg.action_dim * gaussian / (log_prob + 1e-8))
    return action, entropy


def _literal_values(critic, z, action, cfg):
    logits = critic(torch.cat((z, action), dim=-1))
    support = torch.linspace(cfg.q_vmin, cfg.q_vmax, cfg.q_num_bins)
    transformed = (logits.softmax(-1) * support).sum(-1, keepdim=True)
    return transformed.sign() * (transformed.abs().exp() - 1.0)


def _literal_critic_loss(critic, z, action, target, cfg):
    # Dense two-hot construction is independent of AMBI's sparse CE backend.
    transformed = (target.sign() * torch.log(1.0 + target.abs())).clamp(cfg.q_vmin, cfg.q_vmax)
    position = (transformed - cfg.q_vmin) / ((cfg.q_vmax - cfg.q_vmin) / (cfg.q_num_bins - 1))
    lower = position.floor().long()
    upper = (lower + 1).clamp(max=cfg.q_num_bins - 1)
    fraction = position - lower
    labels = target.new_zeros(target.shape[0], cfg.q_num_bins)
    labels.scatter_add_(-1, lower, 1.0 - fraction)
    labels.scatter_add_(-1, upper, fraction)
    logits = critic(torch.cat((z, action), dim=-1))
    return cfg.inner_critic_loss_coef * torch.stack([
        -(labels * F.log_softmax(head, dim=-1)).sum(-1).mean()
        for head in logits
    ]).mean()


def _reference_optimizer(actual, parameters):
    # Nonempty moments make loss scaling/epsilon mistakes observable even when
    # first-step Adam would approximately cancel a constant gradient multiplier.
    for group in actual.param_groups:
        for parameter in group["params"]:
            actual.state[parameter] = {
                "step": torch.tensor(3.0),
                "exp_avg": torch.full_like(parameter, 0.001),
                "exp_avg_sq": torch.full_like(parameter, 0.002),
            }
    group = actual.param_groups[0]
    reference = torch.optim.Adam(parameters, lr=group["lr"], eps=group["eps"])
    reference.load_state_dict(deepcopy(actual.state_dict()))
    return reference


def _assert_updated_module(actual, expected, actual_optim, expected_optim):
    for parameter, reference in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=5e-5, atol=2e-7)
        torch.testing.assert_close(parameter, reference, rtol=1e-5, atol=1e-7)
    for state, reference in zip(actual_optim.state.values(), expected_optim.state.values()):
        for key in state:
            torch.testing.assert_close(state[key], reference[key], rtol=5e-5, atol=2e-8)


@pytest.mark.parametrize("saturated", [False, True])
@pytest.mark.parametrize("clip", [20.0, 0.0001])
def test_td_ambi_inner_losses_gradients_and_adam_steps_match_literal_tdmpc2(monkeypatch, saturated, clip):
    engine = _engine(inner_actor_grad_clip_norm=clip, inner_critic_grad_clip_norm=clip)
    state, cfg = engine.state, engine.cfg
    if saturated:
        with torch.no_grad():
            state.actor[-1].bias[0] = 20.0
    actor = deepcopy(state.actor)
    critic = deepcopy(state.critic)
    target_critic = deepcopy(state.critic_target)
    actor_optim = _reference_optimizer(state.actor_optim, actor.parameters())
    critic_optim = _reference_optimizer(state.critic_optim, critic.parameters())
    assert state.actor_optim.param_groups[0]["eps"] == 1e-5
    assert state.critic_optim.param_groups[0]["eps"] == 1e-8
    rng = torch.Generator().manual_seed(417)
    batch = {
        "z": torch.randn(8, cfg.latent_dim, generator=rng),
        "action": torch.randn(8, cfg.action_dim, generator=rng).tanh(),
        "next_z": torch.randn(8, cfg.latent_dim, generator=rng),
        "reward": torch.linspace(-2.0, 3.0, 8).unsqueeze(-1),
        "terminated": torch.tensor([0., 0., 1., 0., 0., 1., 0., 0.]).unsqueeze(-1),
    }
    bootstrap_noise = torch.randn(8, cfg.action_dim, generator=rng)
    actor_noise = torch.randn(8, cfg.action_dim, generator=rng)
    target_pair, actor_pair = torch.tensor([1, 4]), torch.tensor([0, 3])
    with torch.no_grad():
        next_action, _ = _literal_policy(actor, batch["next_z"], bootstrap_noise, cfg)
        bootstrap = _literal_values(target_critic, batch["next_z"], next_action, cfg)[target_pair].min(0).values
        target = batch["reward"] + engine.agent.discount * (1.0 - batch["terminated"]) * bootstrap
    expected_loss = _literal_critic_loss(critic, batch["z"], batch["action"], target, cfg)
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), clip)
    critic_optim.step()

    def controlled_critic(z, action, reward, next_z, terminated, alpha, noise, pair):
        return engine._sac_critic_kernel(z, action, reward, next_z, terminated, alpha, bootstrap_noise, target_pair)

    monkeypatch.setitem(engine._compile_regions, "critic", controlled_critic)
    actual = engine._sac_critic_step(batch, engine.alpha.detach())
    torch.testing.assert_close(actual["critic_loss"], expected_loss.detach())
    torch.testing.assert_close(actual["critic_grad_norm"], expected_norm)
    _assert_updated_module(state.critic, critic, state.critic_optim, critic_optim)

    outer_scale = engine.agent.actor_loss_scale.detach().clone()
    local_scale = outer_scale.clone()
    reference_scale = outer_scale.clone()
    calls = {"pi": 0, "Q": 0}
    original_pi, original_q = engine.model.pi, engine.model.Q

    def counted_pi(*args, **kwargs):
        calls["pi"] += 1
        return original_pi(*args, **kwargs)

    def counted_q(*args, **kwargs):
        calls["Q"] += 1
        return original_q(*args, **kwargs)

    def controlled_actor(z, alpha, scale, noise, pair, update):
        return engine._scaled_sac_actor_kernel(z, alpha, scale, actor_noise, actor_pair, update)

    monkeypatch.setattr(engine.model, "pi", counted_pi)
    monkeypatch.setattr(engine.model, "Q", counted_q)
    monkeypatch.setitem(engine._compile_regions, "actor", controlled_actor)
    # Two successive updates verify that the local running scale advances at
    # every actor step, using the exact Q sample in that step's loss.
    for _ in range(2):
        actor_optim.zero_grad(set_to_none=True)
        action, entropy = _literal_policy(actor, batch["z"], actor_noise, cfg)
        if saturated:
            assert (action[:, 0] == 1.0).all()
        q = _literal_values(critic, batch["z"], action, cfg)[actor_pair].mean(0)
        q_gradient = torch.autograd.grad(q.sum(), action, retain_graph=True)[0]
        assert q_gradient[:, 1:].abs().max() > 1e-5
        quantiles = torch.quantile(q.detach(), torch.tensor([0.05, 0.95]), dim=0)
        robust_range = (quantiles[1] - quantiles[0]).clamp(min=1.0)
        reference_scale.lerp_(robust_range, cfg.sac_actor_loss_scale_tau)
        expected_loss = -(q / reference_scale.clone() + 0.0001 * entropy).mean()
        expected_loss.backward()
        expected_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), clip)
        actor_optim.step()
        before_rng = torch.random.get_rng_state().clone()
        actual = engine._sac_policy_step(
            batch, update_temperature=False, update_actor=True,
            alpha=engine.alpha.detach(), actor_loss_scale=local_scale,
        )
        torch.testing.assert_close(torch.random.get_rng_state(), before_rng, rtol=0, atol=0)
        torch.testing.assert_close(actual["actor_loss"], expected_loss.detach(), rtol=3e-5, atol=1e-6)
        torch.testing.assert_close(actual["actor_grad_norm"], expected_norm, rtol=3e-5, atol=1e-6)
        torch.testing.assert_close(local_scale, reference_scale)
        torch.testing.assert_close(engine.agent.actor_loss_scale, outer_scale, rtol=0, atol=0)
        _assert_updated_module(state.actor, actor, state.actor_optim, actor_optim)
    assert calls == {"pi": 2, "Q": 2}


@pytest.mark.parametrize("initialization", ["online", "outer_target"])
def test_td_ambi_inner_target_initialization_is_restored_when_action_pool_is_reused(initialization):
    engine = _engine(inner_critic_target_initialization=initialization)
    state = engine.state
    expected = engine.model._target_Qs if initialization == "outer_target" else engine.model._Qs
    _assert_tree_equal(state.critic_target.state_dict(), expected.state_dict())
    target_id = id(state.critic_target)
    with torch.no_grad():
        for parameter in state.critic_target.parameters():
            parameter.add_(7.0)
        for parameter in engine.model._target_Qs.parameters():
            parameter.add_(0.1)
        for parameter in engine.model._Qs.parameters():
            parameter.sub_(0.2)
    engine._clear_expired(t0=False, include_action=True)
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=False)
    assert id(engine.state.critic_target) == target_id
    _assert_tree_equal(engine.state.critic_target.state_dict(), expected.state_dict())
    assert not any(parameter.requires_grad for parameter in engine.state.critic_target.parameters())


def test_td_ambi_per_action_scale_mode_preserves_the_frozen_divisor():
    engine = _engine(inner_actor_loss_scale_update="per_action")
    z = torch.linspace(-2.0, 2.0, 8 * engine.cfg.latent_dim).reshape(8, -1)
    scale = engine.agent.actor_loss_scale.detach().clone()
    before = scale.clone()
    outputs = engine._sac_actor_kernel(
        z, engine.alpha.detach(), torch.zeros(8, 3), torch.tensor([0, 3]), True,
        q_scale=scale,
    )
    outputs[2].backward()
    torch.testing.assert_close(scale, before, rtol=0, atol=0)


def test_td_ambi_per_update_scale_snapshots_are_safe_for_delayed_backward():
    engine = _engine()
    z = torch.linspace(-2.0, 2.0, 8 * engine.cfg.latent_dim).reshape(8, -1)
    scale = engine.agent.actor_loss_scale.detach().clone()
    losses = []
    for factor in (1.0, 1.5):
        before = scale.clone()
        result = engine._sac_actor_kernel(
            z * factor, engine.alpha.detach(), torch.zeros(8, 3), torch.tensor([0, 3]), True,
            q_scale=scale,
        )
        torch.testing.assert_close(scale, before, rtol=0, atol=0)
        scale.copy_(result[-1])
        losses.append(result[2])
    sum(losses).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in engine.state.actor_params)
    assert scale.grad is None


def test_td_ambi_per_update_scale_proposals_and_backward_match_compiled_execution():
    engine = _engine()
    kernel = engine._scaled_sac_actor_kernel
    compiled = torch.compile(kernel, backend="aot_eager", fullgraph=True, dynamic=False)
    z = torch.linspace(-2.0, 2.0, 8 * engine.cfg.latent_dim).reshape(8, -1)
    noise = torch.linspace(-1.0, 1.0, 24).reshape(8, 3)
    eager_scale = engine.agent.actor_loss_scale.detach().clone()
    compiled_scale = eager_scale.clone()
    eager_losses, compiled_losses = [], []
    for factor in (1.0, 1.5):
        args = (z * factor, engine.alpha.detach())
        tail = (noise, torch.tensor([0, 3]), True)
        before_eager, before_compiled = eager_scale.clone(), compiled_scale.clone()
        expected = kernel(*args, eager_scale, *tail)
        actual = compiled(*args, compiled_scale, *tail)
        torch.testing.assert_close(eager_scale, before_eager, rtol=0, atol=0)
        torch.testing.assert_close(compiled_scale, before_compiled, rtol=0, atol=0)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference)
        eager_scale.copy_(expected[-1])
        compiled_scale.copy_(actual[-1])
        torch.testing.assert_close(compiled_scale, eager_scale)
        eager_losses.append(expected[2])
        compiled_losses.append(actual[2])
    expected_grad = torch.autograd.grad(sum(eager_losses), engine.state.actor_params)
    actual_grad = torch.autograd.grad(sum(compiled_losses), engine.state.actor_params)
    for value, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(value, reference)
    torch.testing.assert_close(engine.agent.actor_loss_scale, torch.tensor([3.75]), rtol=0, atol=0)


def test_td_ambi_action_updates_only_one_private_scale_copy(monkeypatch):
    engine = _engine()
    outer_scale = engine.agent.actor_loss_scale.detach().clone()
    original = engine._sac_policy_step
    updates = []

    def record_scale(*args, **kwargs):
        scale = kwargs["actor_loss_scale"]
        assert scale.data_ptr() != engine.agent.actor_loss_scale.data_ptr()
        before = scale.clone()
        result = original(*args, **kwargs)
        updates.append((scale.data_ptr(), before, scale.clone()))
        return result

    monkeypatch.setattr(engine, "_sac_policy_step", record_scale)
    engine.agent.act(torch.zeros(3), t0=True, eval_mode=False)
    assert len(updates) == engine.cfg.inner_actor_updates_per_action
    assert len({pointer for pointer, _, _ in updates}) == 1
    torch.testing.assert_close(updates[0][1], outer_scale, rtol=0, atol=0)
    assert not torch.equal(updates[0][1], updates[-1][2])
    torch.testing.assert_close(engine.agent.actor_loss_scale, outer_scale, rtol=0, atol=0)


def test_td_ambi_compile_fallback_commits_scale_once_after_success(monkeypatch):
    engine = _engine()
    scale = engine.agent.actor_loss_scale.detach().clone()
    before = scale.clone()
    z = torch.linspace(-2.0, 2.0, 8 * engine.cfg.latent_dim).reshape(8, -1)
    proposals = []
    calls = []
    kernel = engine._scaled_sac_actor_kernel

    def eager(*args):
        calls.append("eager")
        result = kernel(*args)
        proposals.append(result[-1].clone())
        return result

    def fail_after_proposal(*args):
        calls.append("compiled")
        result = kernel(*args)
        proposals.append(result[-1].clone())
        torch.testing.assert_close(scale, before, rtol=0, atol=0)
        raise RuntimeError("backend failed after calculating a scale proposal")

    region = CompileRegion("TD-AMBI test actor", eager, enabled=True, strict=False)
    region._compiled = fail_after_proposal
    monkeypatch.setitem(engine._compile_regions, "actor", region)
    with pytest.warns(RuntimeWarning, match="Falling back to eager"):
        engine._sac_policy_step(
            {"z": z}, update_temperature=False, update_actor=True,
            alpha=engine.alpha.detach(), actor_loss_scale=scale,
        )
    assert calls == ["compiled", "eager"]
    torch.testing.assert_close(proposals[0], proposals[1], rtol=0, atol=0)
    torch.testing.assert_close(scale, proposals[1], rtol=0, atol=0)
    assert not torch.equal(scale, before)
    torch.testing.assert_close(engine.agent.actor_loss_scale, before, rtol=0, atol=0)
