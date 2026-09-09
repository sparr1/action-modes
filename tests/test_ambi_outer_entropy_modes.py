"""Outer entropy objective and continuation contract against literal oracles."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree, _model


def _agent(**overrides):
    holder = _model(**overrides)
    holder.env.close()
    return holder.agent


def _scaled_agent(**overrides):
    params = {
        "outer_actor_entropy_mode": "tdmpc2_scaled",
        "ent_coef": 0.0001,
        "inner_target_entropy": "auto",
    }
    params.update(overrides)
    return _agent(**params)


@pytest.mark.parametrize("action_dim", [1, 3])
@pytest.mark.parametrize("grad_clip_norm", [20.0, 0.0001])
def test_complete_scaled_outer_step_matches_literal_tdmpc2(
    monkeypatch, action_dim, grad_clip_norm,
):
    holder = _scaled_agent(
        q_representation="distributional",
        num_q=5,
        q_pair_size=2,
        outer_q_actor_reduction="mean_pair",
        log_std_mapping="tdmpc2_tanh",
        log_std_min=-10,
        log_std_max=2,
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
        sac_actor_loss_scale_tau=0.01,
        rho=0.5,
        actor_lr=0.0003,
        actor_adam_eps=0.00001,
        grad_clip_norm=grad_clip_norm,
    )
    cfg = deepcopy(holder.cfg)
    cfg.action_dim = action_dim
    agent = AMBITDMPC2Agent(cfg)
    # Fresh critics have zero output weights, which would make the Q term's
    # action gradient vanish. Represent learned critics with deterministic,
    # nonzero output layers so parity exercises both terms of the objective.
    with torch.no_grad():
        for head, critic in enumerate(agent.model._Qs):
            output = critic[-1]
            bins = torch.linspace(-1.0, 1.0, output.out_features)
            features = torch.linspace(-1.0, 1.0, output.in_features)
            output.weight.copy_((2.0 + 0.1 * head) * torch.outer(bins, features))
            output.bias.copy_((0.1 + 0.03 * head) * bins)
    reference_pi = deepcopy(agent.model._pi)
    reference_optim = torch.optim.Adam(
        reference_pi.parameters(), lr=cfg.actor_lr, eps=cfg.actor_adam_eps,
    )
    # Exercise a continuation step, including nonzero Adam moments.
    for parameter in agent.model._pi.parameters():
        agent.pi_optim.state[parameter] = {
            "step": torch.tensor(3.0),
            "exp_avg": torch.full_like(parameter, 0.001),
            "exp_avg_sq": torch.full_like(parameter, 0.002),
        }
    reference_optim.load_state_dict(deepcopy(agent.pi_optim.state_dict()))
    agent.actor_loss_scale.fill_(3.75)
    zs = torch.randn(3, 4, cfg.latent_dim)
    noise = torch.randn(3, 4, action_dim)
    actual_pi = agent.model.pi

    def fixed_sample(z, **kwargs):
        assert kwargs == {"include_scaled_entropy": True}
        return actual_pi(z, noise=noise, **kwargs)

    monkeypatch.setattr(agent.model, "pi", fixed_sample)
    mean, raw_log_std = reference_pi(zs).chunk(2, dim=-1)
    log_std = -10.0 + 0.5 * 12.0 * (torch.tanh(raw_log_std) + 1.0)
    action = torch.tanh(mean + noise * log_std.exp())
    # Independent literal expression from upstream world_model.py/math.py.
    gaussian_log_prob = (
        -0.5 * noise.square() - log_std - 0.9189385175704956
    ).sum(-1, keepdim=True)
    td_log_prob = gaussian_log_prob - torch.log(
        torch.relu(1.0 - action.square()) + 1e-6
    ).sum(-1, keepdim=True)
    entropy_scale = (gaussian_log_prob * action_dim) / (td_log_prob + 1e-8)
    scaled_entropy = -td_log_prob * entropy_scale
    rng = torch.random.get_rng_state()
    q_all = agent.model.Q(zs, action, reduction="all", detach=True)
    pair = torch.randperm(5)[:2]
    q_values = q_all[pair].mean(0)
    q_action_gradient = torch.autograd.grad(
        q_values.sum(), action, retain_graph=True,
    )[0]
    assert q_action_gradient.abs().max() > 1e-4
    quantiles = torch.quantile(
        q_values[0].detach(), torch.tensor([0.05, 0.95]), dim=0,
    )
    assert (quantiles[1] - quantiles[0]).item() > 1.0
    scale = torch.tensor([3.75]).lerp_(
        (quantiles[1] - quantiles[0]).clamp(min=1.0), 0.01,
    )
    expected_loss = (
        -(0.0001 * scaled_entropy + q_values / scale).mean((1, 2))
        * torch.tensor([1.0, 0.5, 0.25])
    ).mean()
    reference_optim.zero_grad(set_to_none=True)
    expected_loss.backward()
    expected_gradients = [p.grad.clone() for p in reference_pi.parameters()]
    expected_norm = torch.nn.utils.clip_grad_norm_(
        reference_pi.parameters(), cfg.grad_clip_norm,
    )
    if grad_clip_norm == 0.0001:
        assert expected_norm > cfg.grad_clip_norm
    reference_optim.step()
    captured_gradients = []
    clip = agent._clip_actor_grad_norm_

    def capture_and_clip():
        captured_gradients.extend(p.grad.clone() for p in agent.model._pi.parameters())
        return clip()

    monkeypatch.setattr(agent, "_clip_actor_grad_norm_", capture_and_clip)
    torch.random.set_rng_state(rng)
    metrics = agent._update_actor(zs)
    torch.testing.assert_close(metrics["actor_loss"], expected_loss.detach())
    torch.testing.assert_close(metrics["actor_grad_norm"], expected_norm)
    torch.testing.assert_close(agent.actor_loss_scale, scale)
    torch.testing.assert_close(metrics["actor_scaled_entropy"], scaled_entropy.detach().mean())
    torch.testing.assert_close(metrics["actor_entropy_bonus"], 0.0001 * scaled_entropy.detach().mean())
    for actual, expected in zip(captured_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-8)
    for actual, expected in zip(agent.model._pi.parameters(), reference_pi.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=1e-9)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    for actual, expected in zip(
        agent.pi_optim.state.values(), reference_optim.state.values(),
    ):
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key])


@pytest.mark.parametrize("mode", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("below_target", [True, False])
def test_outer_temperature_uses_selected_entropy_and_normalized_weights(
    monkeypatch, mode, below_target,
):
    agent = _agent(
        outer_actor_entropy_mode=mode, ent_coef="auto_0.5",
        target_entropy=1.0, inner_target_entropy="auto", rho=0.5,
    )
    chosen = torch.tensor([-1.0, 0.0, 2.0]) if below_target else torch.tensor([3.0, 4.0, 2.0])
    other = -chosen + 10.0
    anchor = next(agent.model._pi.parameters()).reshape(-1)[0]

    def policy(z, **kwargs):
        assert kwargs == ({"include_scaled_entropy": True} if mode == "tdmpc2_scaled" else {})
        action = z[..., :1] * 0.0 + anchor * 0.0
        selected = chosen.reshape(3, 1, 1).expand(3, 4, 1) + anchor * 0.0
        actual = selected if mode == "squashed" else other.reshape(3, 1, 1).expand_as(selected) + anchor * 0.0
        info = {"log_prob": -actual, "entropy": actual}
        if mode == "tdmpc2_scaled":
            info["scaled_entropy"] = selected
        return action, info

    monkeypatch.setattr(agent.model, "pi", policy)
    before = agent.log_ent_coef.detach().clone()
    zs = torch.zeros(3, 4, agent.cfg.latent_dim)
    metrics = agent._update_actor(zs)
    weights = torch.tensor([1.0, 0.5, 0.25]) / 1.75
    expected = (before * ((chosen - 1.0) * weights).sum()).mean()
    torch.testing.assert_close(metrics["ent_coef_loss"], expected)
    assert bool(agent.log_ent_coef.detach() > before) == below_target
    torch.testing.assert_close(metrics["ent_coef"], before.exp())
    expected_actual = chosen if mode == "squashed" else other
    torch.testing.assert_close(metrics["actor_entropy"], expected_actual.mean())


@pytest.mark.parametrize("mode", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("inner_mode", ["squashed", "tdmpc2_scaled"])
def test_entropy_checkpoint_portable_and_exact_roundtrip(mode, inner_mode):
    settings = {
        "outer_actor_entropy_mode": mode,
        "inner_actor_entropy_mode": inner_mode,
        "ent_coef": 0.001,
        "inner_target_entropy": 2.0,
    }
    source = _agent(**settings)
    payload = _clone_tree(source.checkpoint_state())
    assert payload["entropy_spec"]["actor_entropy_mode"] == mode
    assert payload["entropy_spec"]["inner"]["actor_entropy_mode"] == inner_mode
    restored = _agent(**settings)
    restored.load(payload)
    _assert_tree_equal(restored.checkpoint_state(), payload)
    source.prepare_training_resume_boundary()
    exact = _clone_tree(source.training_state_dict())
    restored.load_training_state_dict(exact)
    _assert_tree_equal(restored.training_state_dict(), exact)


@pytest.mark.parametrize("exact", [False, True])
def test_outer_entropy_mismatch_rejects_before_mutation(exact):
    source = _agent(ent_coef=0.001, inner_target_entropy="auto")
    target = _scaled_agent(ent_coef=0.001)
    source.prepare_training_resume_boundary()
    incoming = source.training_state_dict() if exact else source.checkpoint_state()
    pristine = _clone_tree(target.training_state_dict())
    load = target.load_training_state_dict if exact else target.load
    with pytest.raises(ValueError, match="entropy specification"):
        load(_clone_tree(incoming))
    _assert_tree_equal(target.training_state_dict(), pristine)


def test_inner_entropy_may_change_for_outer_weight_loading_but_not_exact_resume():
    source = _agent(ent_coef=0.001, inner_target_entropy=2.0)
    target = _agent(
        ent_coef=0.001, inner_actor_entropy_mode="tdmpc2_scaled",
        inner_target_entropy=2.0,
    )
    target.load(_clone_tree(source.checkpoint_state()))
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())
    pristine = _clone_tree(target.training_state_dict())
    with pytest.raises(ValueError, match="inner entropy specification"):
        target.load_training_state_dict(_clone_tree(source.training_state_dict()))
    _assert_tree_equal(target.training_state_dict(), pristine)


def test_exact_resume_rejects_changed_inner_target_before_mutation():
    source = _agent(
        ent_coef=0.001, inner_actor_entropy_mode="tdmpc2_scaled",
        inner_target_entropy=2.0,
    )
    target = _agent(
        ent_coef=0.001, inner_actor_entropy_mode="tdmpc2_scaled",
        inner_target_entropy=3.0,
    )
    pristine = _clone_tree(target.training_state_dict())
    with pytest.raises(ValueError, match="inner entropy specification"):
        target.load_training_state_dict(_clone_tree(source.training_state_dict()))
    _assert_tree_equal(target.training_state_dict(), pristine)


@pytest.mark.parametrize("exact", [False, True])
def test_missing_historical_entropy_fields_mean_squashed(exact):
    source = _agent(ent_coef=0.001)
    source.prepare_training_resume_boundary()
    legacy = _clone_tree(source.training_state_dict() if exact else source.checkpoint_state())
    outer = legacy["outer"] if exact else legacy
    outer["entropy_spec"] = {key: outer["entropy_spec"][key] for key in ("mode", "target_entropy")}
    restored = _agent(ent_coef=0.001)
    (restored.load_training_state_dict if exact else restored.load)(legacy)
    scaled = _scaled_agent(ent_coef=0.001)
    pristine = _clone_tree(scaled.training_state_dict())
    with pytest.raises(ValueError, match="entropy specification"):
        (scaled.load_training_state_dict if exact else scaled.load)(legacy)
    _assert_tree_equal(scaled.training_state_dict(), pristine)
    if exact:
        scaled_inner = _agent(
            ent_coef=0.001, inner_actor_entropy_mode="tdmpc2_scaled",
            inner_target_entropy=2.0,
        )
        with pytest.raises(ValueError, match="inner entropy specification"):
            scaled_inner.load_training_state_dict(legacy)


def test_raw_weight_transfer_allows_outer_entropy_change():
    source = _agent(ent_coef=0.001)
    target = _scaled_agent(ent_coef=0.001)
    target.load(_clone_tree(source.model.state_dict()))
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())


@pytest.mark.parametrize("mode", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("target_mode", ["entropy_augmented", "reward_only"])
@pytest.mark.parametrize("coefficient", [0.25, "auto_0.25"])
def test_outer_bellman_target_uses_selected_entropy(monkeypatch, mode, target_mode, coefficient):
    agent = _agent(
        outer_actor_entropy_mode=mode, ent_coef=coefficient, target_entropy=1.0,
        inner_target_entropy="auto", outer_critic_target=target_mode,
    )
    requests = []

    def policy(z, **kwargs):
        requests.append(kwargs)
        info = {"log_prob": z[..., :1] * 0 - 0.4}
        if kwargs.get("include_scaled_entropy"):
            info["scaled_entropy"] = z[..., :1] * 0 + 2.0
        return z[..., :1] * 0, info

    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.model, "Q", lambda z, action, **kwargs: z[..., :1] * 0 + 7.0)
    z = torch.zeros(2, 4, agent.cfg.latent_dim, requires_grad=True)
    reward = torch.full((2, 4, 1), 2.0)
    terminated = torch.zeros_like(reward)
    terminated[:, 0] = 1
    target = agent._soft_td_target(z, reward, terminated)
    entropy = 0.4 if mode == "squashed" else 2.0
    bonus = 0.25 * entropy if target_mode == "entropy_augmented" else 0.0
    torch.testing.assert_close(target, reward + agent.discount * (1 - terminated) * (7.0 + bonus))
    assert not target.requires_grad
    assert requests == ([{"include_scaled_entropy": True}]
                        if mode == "tdmpc2_scaled" and target_mode == "entropy_augmented" else [{}])


@pytest.mark.parametrize("target_mode", ["reward_only", "entropy_augmented"])
def test_critic_entropy_selection_preserves_outer_sampling_rng(target_mode):
    agent = _scaled_agent(outer_critic_target=target_mode)
    z = torch.randn(2, 4, agent.cfg.latent_dim)
    reward = torch.zeros(2, 4, 1)
    rng = torch.random.get_rng_state()
    scaled = agent._soft_td_target(z, reward, reward)
    after_scaled = torch.random.get_rng_state()
    agent._actor_entropy_mode = "squashed"
    torch.random.set_rng_state(rng)
    ordinary = agent._soft_td_target(z, reward, reward)
    torch.testing.assert_close(torch.random.get_rng_state(), after_scaled, rtol=0, atol=0)
    if target_mode == "reward_only":
        torch.testing.assert_close(scaled, ordinary, rtol=0, atol=0)


@pytest.mark.parametrize("learner", ["outer", "inner"])
@pytest.mark.parametrize("mode", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("target_mode", ["reward_only", "entropy_augmented"])
def test_legacy_bellman_semantics_preflight_is_transactional(learner, mode, target_mode):
    target_key = "outer_critic_target" if learner == "outer" else "inner_sac_critic_target"
    agent = _agent(**{
        f"{learner}_actor_entropy_mode": mode, target_key: target_mode,
        "ent_coef": 0.25, "inner_target_entropy": 1.0,
    })
    agent.prepare_training_resume_boundary()
    pristine = _clone_tree(agent.training_state_dict())
    legacy = _clone_tree(pristine)
    legacy["outer"]["critic_target_spec"].pop("entropy_semantics")
    if mode == "tdmpc2_scaled" and target_mode == "entropy_augmented":
        with pytest.raises(ValueError, match="critic-target specification"):
            agent.load_training_state_dict(legacy)
        _assert_tree_equal(agent.training_state_dict(), pristine)
        # Portable target metadata is provenance: explicit weight transfer is allowed.
        agent.load(legacy["outer"])
    else:
        agent.load_training_state_dict(legacy)
        _assert_tree_equal(agent.training_state_dict(), pristine)
