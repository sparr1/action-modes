"""gSDE learning equations, checkpoint semantics, and Gaussian compatibility."""

from copy import deepcopy
import math

import pytest
import torch

from RL.tdmpc2_core.common.entropy import policy_entropy
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _tiny_model
from utils.resume_identity import scientific_trial_parameters


@pytest.fixture
def agents():
    opened = []

    def create(**kwargs):
        params = dict(
            actor_distribution="gsde", inner_rounds=1,
            inner_rollouts_per_round=4, inner_updates_per_round=1,
            ent_coef=0.2, target_entropy=1.0, inner_target_entropy=1.0,
        )
        params.update(kwargs)
        holder = _tiny_model(**params)
        opened.append(holder)
        return holder.agent

    yield create
    for holder in opened:
        holder.env.close()


@pytest.mark.parametrize("distribution", ["gaussian", "gsde"])
@pytest.mark.parametrize("feature", ["actor", "latent"])
def test_pre_tanh_outer_objective_matches_gaussian_entropy(agents, monkeypatch, distribution, feature):
    agent = agents(actor_distribution=distribution, gsde_feature_source=feature,
                   outer_actor_entropy_mode="pre_tanh", inner_actor_entropy_mode="pre_tanh")
    z = agent.model.encode(torch.randn(3, 4, 3)).detach()
    noise = torch.randn(3, 4, 1)
    sample = agent.model.pi
    action, info = sample(z, noise=noise)
    expected_entropy = torch.distributions.Normal(
        info["pre_tanh_mean"], info["log_std"].exp(),
    ).entropy().sum(-1, keepdim=True)
    q = agent.model.Q(z, action, reduction=agent.cfg.outer_q_actor_reduction, detach=True)
    expected_loss = ((-0.2 * expected_entropy - q).mean((1, 2))
                     * torch.tensor([1.0, 0.5, 0.25])).mean()
    expected_gradients = torch.autograd.grad(expected_loss, tuple(agent.model._pi.parameters()),
                                             allow_unused=True)
    captured = []
    clip = agent._clip_actor_grad_norm_

    def capture():
        captured.extend(None if p.grad is None else p.grad.clone() for p in agent.model._pi.parameters())
        return clip()

    monkeypatch.setattr(agent.model, "pi", lambda states, **kw: sample(states, noise=noise, **kw))
    monkeypatch.setattr(agent, "_clip_actor_grad_norm_", capture)
    metrics = agent._update_actor(z)
    torch.testing.assert_close(metrics["actor_loss"], expected_loss.detach())
    torch.testing.assert_close(metrics["actor_selected_entropy"], expected_entropy.detach().mean())
    torch.testing.assert_close(metrics["actor_entropy_bonus"], .2 * expected_entropy.detach().mean())
    for actual, expected in zip(captured, expected_gradients):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    assert all(p.grad is None for p in agent.model._encoder.parameters())


@pytest.mark.parametrize("below_target", [True, False])
def test_pre_tanh_temperature_uses_analytic_entropy(agents, monkeypatch, below_target):
    agent = agents(ent_coef="auto_0.5", outer_actor_entropy_mode="pre_tanh")
    chosen = torch.tensor([-1., 0., 2.]) if below_target else torch.tensor([3., 4., 2.])
    anchor = next(agent.model._pi.parameters()).flatten()[0]

    def policy(z, **kwargs):
        entropy = chosen[:, None, None].expand(3, 4, 1) + 0 * anchor
        return z[..., :1] * 0 + 0 * anchor, {
            "log_prob": entropy * 0 - 100,
            "entropy": entropy * 0 + 100,
            "log_std": entropy - .5 * math.log(2 * math.pi * math.e),
        }

    monkeypatch.setattr(agent.model, "pi", policy)
    before = agent.log_ent_coef.detach().clone()
    metrics = agent._update_actor(torch.randn(3, 4, agent.cfg.latent_dim))
    expected = before * (((chosen - 1) * torch.tensor([1., .5, .25])).sum() / 1.75)
    torch.testing.assert_close(metrics["ent_coef_loss"], expected.mean())
    assert bool(agent.log_ent_coef > before) == below_target
    assert metrics["actor_entropy"].item() == 100


@pytest.mark.parametrize("target", ["reward_only", "entropy_augmented"])
@pytest.mark.parametrize("coefficient", [0.25, "auto_0.25"])
def test_pre_tanh_outer_target_uses_exact_entropy(agents, monkeypatch, target, coefficient):
    agent = agents(outer_actor_entropy_mode="pre_tanh", outer_critic_target=target, ent_coef=coefficient)
    def policy(z, **kwargs):
        assert kwargs == {}
        return z[..., :1] * 0, {"log_prob": z[..., :1] * 0 - 100,
                               "log_std": z[..., :1] * 0 + .3}
    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.model, "Q", lambda z, a, **kw: z[..., :1] * 0 + 7)
    z = torch.randn(2, 4, agent.cfg.latent_dim, requires_grad=True)
    reward = torch.full((2, 4, 1), 2.)
    terminated = torch.zeros_like(reward)
    terminated[:, 0] = 1
    actual = agent._soft_td_target(z, reward, terminated)
    bonus = .25 * (.3 + .5 * math.log(2 * math.pi * math.e)) if target == "entropy_augmented" else 0
    torch.testing.assert_close(actual, reward + agent.discount * (1 - terminated) * (7 + bonus))
    assert not actual.requires_grad


@pytest.mark.parametrize("feature", ["actor", "latent"])
@pytest.mark.parametrize("mode", ["squashed", "tdmpc2_scaled", "pre_tanh"])
@pytest.mark.parametrize("auxiliary", ["off", "sac"])
def test_complete_outer_and_inner_updates_are_finite(agents, feature, mode, auxiliary):
    agent = agents(gsde_feature_source=feature, outer_actor_entropy_mode=mode,
                   inner_actor_entropy_mode=mode, ent_coef="auto_0.2", aux_return_mode=auxiliary)
    before = deepcopy(agent.model._pi.state_dict())
    h, b = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    metrics = agent._update(torch.randn(h + 1, b, 3), torch.randn(h, b, 1).tanh(),
                            torch.randn(h, b, 1), torch.zeros(h, b, 1))
    for key in ("actor_loss", "critic_loss", "total_loss", "ent_coef_loss"):
        assert torch.isfinite(metrics[key]).all(), key
    assert any(not torch.equal(value, agent.model._pi.state_dict()[key]) for key, value in before.items())
    action = agent.act(torch.tensor([.2, -.1, .3]), collect_diagnostics=False)
    assert torch.isfinite(action).all() and (action.abs() <= 1).all()


@pytest.mark.parametrize("mode", ["squashed", "tdmpc2_scaled", "pre_tanh"])
@pytest.mark.parametrize("feature", ["actor", "latent"])
def test_gsde_checkpoint_roundtrip(agents, mode, feature):
    settings = dict(outer_actor_entropy_mode=mode, inner_actor_entropy_mode=mode, gsde_feature_source=feature)
    source = agents(**settings)
    payload = _clone_tree(source.checkpoint_state())
    assert payload["policy_spec"]["actor_distribution"] == "gsde"
    assert payload["policy_spec"]["gsde_feature_source"] == feature
    if mode == "pre_tanh":
        assert payload["entropy_spec"]["target_entropy_semantics"] == "pre_tanh_gaussian_entropy"
        assert payload["critic_target_spec"]["entropy_semantics"]["outer"] == "pre_tanh_gaussian_entropy"
    restored = agents(**settings)
    restored.load(payload)
    _assert_tree_equal(restored.checkpoint_state(), payload)
    source.prepare_training_resume_boundary()
    exact = _clone_tree(source.training_state_dict())
    restored.load_training_state_dict(exact)
    _assert_tree_equal(restored.training_state_dict(), exact)


@pytest.mark.parametrize("change", [
    {"actor_distribution": "gaussian"}, {"gsde_feature_source": "latent"},
    {"gsde_mean_clip": None}, {"outer_actor_entropy_mode": "pre_tanh"},
])
@pytest.mark.parametrize("exact", [False, True])
def test_gsde_checkpoint_mismatch_is_atomic(agents, change, exact):
    # Equal feature dimensions ensure feature identity cannot rely on shapes.
    source = agents(mlp_dim=8)
    target = agents(mlp_dim=8, **change)
    before = _clone_tree(target.training_state_dict())
    payload = source.training_state_dict() if exact else source.checkpoint_state()
    with pytest.raises(ValueError, match="specification|architecture|gSDE|gsde"):
        (target.load_training_state_dict if exact else target.load)(_clone_tree(payload))
    _assert_tree_equal(target.training_state_dict(), before)


def test_gsde_weights_require_feature_metadata(agents):
    source, target = agents(), agents()
    before = _clone_tree(target.model.state_dict())
    with pytest.raises(ValueError, match="policy specification"):
        target.load(_clone_tree(source.model.state_dict()))
    _assert_tree_equal(target.model.state_dict(), before)


@pytest.mark.parametrize("change", [
    {"inner_gsde_noise_scope": "solve"}, {"inner_gsde_resample_steps": 3},
    {"gsde_execution_resample_steps": 3},
])
def test_gsde_protocol_changes_only_allow_portable_transfer(agents, change):
    source, target = agents(), agents(**change)
    target.load(_clone_tree(source.checkpoint_state()))
    before = _clone_tree(target.training_state_dict())
    with pytest.raises(ValueError, match="specification|gSDE|gsde"):
        target.load_training_state_dict(_clone_tree(source.training_state_dict()))
    _assert_tree_equal(target.training_state_dict(), before)


def test_gsde_identity_defaults_and_inactive_options():
    defaults = dict(gsde_feature_source="actor", gsde_log_std_init=-3., gsde_mean_clip=2.,
                    inner_gsde_noise_scope="round", inner_gsde_resample_steps=8,
                    gsde_execution_resample_steps=8)
    def identity(**params):
        return scientific_trial_parameters({"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": params})
    assert identity() == identity(actor_distribution="gaussian", **defaults)
    assert identity(actor_distribution="gsde") == identity(actor_distribution="gsde", **defaults)
    for key, value in dict(gsde_feature_source="latent", gsde_mean_clip=None,
                           inner_gsde_noise_scope="solve", inner_gsde_resample_steps=3,
                           gsde_execution_resample_steps=3, gsde_log_std_init=-2.).items():
        assert identity(actor_distribution="gsde", **{key: value}) != identity(actor_distribution="gsde")
    assert identity(actor_distribution="gsde", log_std_mapping="tdmpc2_tanh", log_std_min=-10) == identity(actor_distribution="gsde")


def test_analytic_entropy_gradient_matches_normal_distribution():
    log_std = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    actual = policy_entropy({"log_std": log_std}, "pre_tanh")
    expected = torch.distributions.Normal(torch.zeros_like(log_std), log_std.exp()).entropy().sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), log_std)[0], torch.ones_like(log_std))


def test_gsde_graph_protocol_is_explicit_and_checkpointed(agents):
    from RL.tdmpc2_core.whole_inner_graph import training_graph_protocol

    agent = agents(inner_cuda_graph=True, compile=True, compile_strict=True,
                   outer_actor_entropy_mode="pre_tanh", inner_actor_entropy_mode="pre_tanh")
    # Empty boundary metadata can be checked on CPU; execution still requires CUDA.
    state = _clone_tree(agent.training_state_dict())
    protocol = training_graph_protocol(agent.cfg)
    assert state["outer"]["critic_target_spec"]["gsde"]["graph_protocol"] == protocol
    assert state["inner"]["whole_graph"]["protocol"] == protocol
    bad = _clone_tree(state)
    bad["inner"]["whole_graph"]["protocol"] = "isolated_cuda_graph_with_native_diagnostics_v1"
    with pytest.raises(ValueError, match="protocol"):
        agent.load_training_state_dict(bad)
    _assert_tree_equal(agent.training_state_dict(), state)
