"""BOOM's saved-mean alignment with a state-dependent Gaussian actor."""

from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.common.buffer import Buffer
from tests.test_ambi_boom import _batch, _labels
from tests.test_ambi_boom_acceleration import (
    _actor_step,
    _assert_actor_state_equal,
    _assert_tree_close,
    _avoid_unrelated_macos_compiler_worker_forks,
    _inputs,
    _nonzero_critic,
    _trace_actor_region,
)
from tests.test_ambi_boom_config import _boom_cfg
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _tiny_model


def _model(**overrides):
    return _tiny_model(**{
        "actor_distribution": "gsde",
        "outer_actor_objective": "boom",
        "outer_boom_coef": 0.0042,
        "outer_boom_min_std": 0.05,
        "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
        "sac_actor_loss_scale_tau": 0.1,
        "ent_coef": 0.0021,
        "inner_temperature_mode": "inherit_outer",
        "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only",
        "outer_q_actor_reduction": "min_all",
        "grad_clip_norm": 100.0,
        "batch_size": 4,
        "inner_rounds": 1,
        "inner_rollouts_per_round": 4,
        "inner_updates_per_round": 1,
        **overrides,
    })


def _agent(**overrides):
    wrapped = _model(**overrides)
    wrapped.env.close()
    return wrapped.agent


@pytest.mark.parametrize("scope", ["real_steps", "solve", "round"])
def test_boom_gsde_resolves_accelerated_recipe_and_enables_behavior_replay(scope):
    cfg = _boom_cfg(
        actor_distribution="gsde", inner_gsde_noise_scope=scope,
        inner_gsde_resample_steps=8, gsde_execution_resample_steps=8,
        outer_boom_coef=0.0042, ent_coef=0.0021,
        outer_actor_entropy_mode="squashed", inner_actor_entropy_mode="squashed",
        inner_rounds=8, inner_rollouts_per_round=128, inner_rollout_horizon=3,
        inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
        inner_batch_size=256, compile=True, compile_strict=True,
        critic_ensemble_backend="vmap", inner_cuda_graph=True,
    )
    assert cfg.outer_actor_objective == "boom"
    assert cfg.actor_distribution == "gsde"
    assert cfg.inner_gsde_noise_scope == scope
    assert cfg.store_behavior_policy is True
    assert cfg.store_teacher_policy is False
    assert cfg.outer_boom_coef == pytest.approx(0.0042)
    assert cfg.inner_model_step_budget == 8 * 128 * 3
    assert cfg.inner_actor_updates_per_action == 8 * 4
    assert cfg.inner_critic_updates_per_action == 8 * 16
    assert cfg.inner_cuda_graph and cfg.compile and cfg.compile_strict
    assert cfg.critic_ensemble_backend == "vmap"


@pytest.mark.parametrize("overrides,match", [
    ({"aux_return_mode": "sac"}, "aux_return_mode"),
    ({"outer_behavior_policy_kl_schedule": "smooth"}, "outer_behavior_policy_kl_schedule"),
    ({"inner_actor_adaptation": "frozen"}, "inner_actor_adaptation"),
    ({"inner_operator": "none"}, "inner_operator"),
])
def test_boom_gsde_keeps_both_methods_compatibility_guards(overrides, match):
    with pytest.raises(ValueError, match=match):
        _boom_cfg(actor_distribution="gsde", **overrides)


@pytest.mark.parametrize("feature", ["actor", "latent"])
@pytest.mark.parametrize("horizon,rho", [(1, 0.0), (3, 0.5)])
def test_boom_gsde_actor_matches_independent_marginal_gradient_oracle(
    monkeypatch, feature, horizon, rho,
):
    cfg = deepcopy(_agent(gsde_feature_source=feature, train_unroll_horizon=horizon, rho=rho).cfg)
    cfg.action_dim = 3
    agent = AMBITDMPC2Agent(cfg)
    reference = deepcopy(agent.model._pi)
    optimizer = torch.optim.Adam(reference.parameters(), lr=cfg.actor_lr, eps=cfg.actor_adam_eps)
    agent.actor_loss_scale.fill_(1.95)
    zs = torch.randn(horizon + 1, cfg.batch_size, cfg.latent_dim, requires_grad=True)
    noise = torch.linspace(-1.2, 1.4, horizon * cfg.batch_size * cfg.action_dim).reshape(
        horizon, cfg.batch_size, cfg.action_dim,
    )
    behavior_mean, behavior_log_std, valid = _labels(agent)
    valid[:, -1] = False
    behavior_mean[~valid.expand_as(behavior_mean)] = float("nan")
    behavior_mean.requires_grad_()
    behavior_log_std.requires_grad_()
    offsets = torch.tensor([-12.0, -2.0, 6.0, 17.0]).reshape(1, cfg.batch_size, 1)

    def q_value(action):
        return offsets + 0.7 * action.sum(-1, keepdim=True) + 0.2 * action.square().sum(-1, keepdim=True)

    # Compute the gSDE marginal directly from features and learned matrix
    # scales, independently of policy sampling and BOOM production helpers.
    hidden = reference.trunk(zs[:-1].detach())
    features = hidden if feature == "actor" else zs[:-1].detach()
    mean = reference.mean_head(hidden).clamp(-cfg.gsde_mean_clip, cfg.gsde_mean_clip)
    std = (features.square() @ reference.log_scale.exp().square()).sqrt()
    pre_tanh = mean + std * noise
    action = pre_tanh.tanh()
    log_prob = (
        torch.distributions.Normal(mean, std).log_prob(pre_tanh)
        - torch.distributions.TanhTransform().log_abs_det_jacobian(pre_tanh, action)
    ).sum(-1, keepdim=True)
    q = q_value(action)
    quantiles = torch.quantile(q[0].detach(), torch.tensor([0.05, 0.95]), dim=0)
    scale = torch.tensor([1.95]).lerp_((quantiles[1] - quantiles[0]).clamp_min(1), 0.1)
    assert scale.item() > 2  # The gate must use this update's shared scale.
    temporal = rho ** torch.arange(horizon)
    sac = ((cfg.ent_coef * log_prob - q / scale).mean((1, 2)) * temporal).mean()
    target = torch.where(valid, behavior_mean.detach(), 0.0).tanh()
    alignment_std = std.detach().clamp_min(cfg.outer_boom_min_std)
    nll = -torch.distributions.Normal(target, alignment_std).log_prob(action).sum(-1)
    weights = (q.detach().squeeze(-1) / scale).masked_fill(~valid.squeeze(-1), -torch.inf).softmax(1)
    alignment = cfg.outer_boom_coef * ((weights * nll).sum(1) * temporal).mean() / scale.squeeze()
    expected_loss = sac + alignment
    expected_loss.backward()
    expected_grads = [parameter.grad.clone() for parameter in reference.parameters()]
    assert reference.log_scale.grad.norm() > 0  # Reparameterized samples still train scales.
    torch.nn.utils.clip_grad_norm_(reference.parameters(), cfg.grad_clip_norm)
    optimizer.step()

    sample = agent.model.pi
    calls = {"policy": 0, "critic": 0}

    def policy(states, **kwargs):
        calls["policy"] += 1
        assert states.shape[0] == horizon  # BOOM excludes the terminal H+1 state.
        return sample(states, noise=noise, **kwargs)

    def critic(states, actions, **kwargs):
        calls["critic"] += 1
        assert states.shape[0] == horizon
        assert kwargs == {"reduction": "all", "detach": True}
        return torch.stack([q_value(actions) + head for head in range(cfg.num_q)])

    captured = []
    clip = agent._clip_actor_grad_norm_

    def capture_gradients():
        captured.extend(parameter.grad.clone() for parameter in agent.model._pi.parameters())
        return clip()

    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.model, "Q", critic)
    monkeypatch.setattr(agent, "_clip_actor_grad_norm_", capture_gradients)
    metrics = agent._update_actor(
        zs.detach(), behavior_pre_tanh_mean=behavior_mean,
        behavior_log_std=behavior_log_std, behavior_policy_valid=valid,
    )
    assert calls == {"policy": 1, "critic": 1}
    assert zs.grad is None and behavior_mean.grad is None and behavior_log_std.grad is None
    assert metrics["boom_gate_active"] == 1
    torch.testing.assert_close(agent.actor_loss_scale, scale)
    torch.testing.assert_close(metrics["actor_loss"], expected_loss.detach())
    torch.testing.assert_close(metrics["boom_alignment_loss"], alignment.detach())
    for actual, expected in zip(captured, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=4e-5, atol=2e-7)
    for actual, expected in zip(agent.model._pi.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("scope", ["real_steps", "solve", "round"])
def test_boom_gsde_collection_metadata_reaches_replay_and_outer_update(scope):
    wrapped = _model(inner_gsde_noise_scope=scope)
    try:
        obs = torch.tensor([0.2, -0.1, 0.3])
        wrapped._start_episode_staging(obs)
        action = wrapped._act_agent(obs, t0=True, eval_mode=False)
        wrapped._stage_transition(1, obs, action, 1.0, False)
        staged = wrapped._episode_staging
        assert staged["behavior_policy_valid"][1]
        assert torch.isfinite(staged["behavior_pre_tanh_mean"][1]).all()
        assert torch.isfinite(staged["behavior_log_std"][1]).all()
        assert not torch.allclose(action, staged["behavior_pre_tanh_mean"][1].tanh())
        wrapped._random_action_norm()
        wrapped._stage_transition(2, obs, torch.tensor([0.1]), 0.0, False)
        assert not staged["behavior_policy_valid"][2]

        rows = 7
        buffer = Buffer(wrapped.agent.cfg)
        buffer.add(TensorDict({
            "obs": torch.randn(rows, 3), "action": action.repeat(rows, 1),
            "reward": torch.ones(rows), "terminated": torch.zeros(rows),
            "behavior_pre_tanh_mean": staged["behavior_pre_tanh_mean"][1].repeat(rows, 1),
            "behavior_log_std": staged["behavior_log_std"][1].repeat(rows, 1),
            "behavior_policy_valid": torch.ones(rows, dtype=torch.bool),
        }, batch_size=[rows]))
        wrapped.agent.actor_loss_scale.fill_(3.2)
        metrics = wrapped.agent.update(buffer)
        assert metrics["boom_valid_fraction"] == 1
        assert metrics["boom_gate_active"] == 1
        assert torch.isfinite(torch.as_tensor(metrics["actor_loss"]))
    finally:
        wrapped.env.close()


@pytest.mark.parametrize("scope", ["real_steps", "solve", "round"])
def test_boom_gsde_checkpoint_roundtrip_preserves_next_actor_update(scope):
    source = _agent(inner_gsde_noise_scope=scope)
    data = _batch(source)
    mean, log_std, valid = _labels(source)
    labels = dict(behavior_pre_tanh_mean=mean, behavior_log_std=log_std, behavior_policy_valid=valid)
    source.actor_loss_scale.fill_(3.2)
    source._update(*data, **labels)
    portable = _clone_tree(source.checkpoint_state())
    assert portable["policy_spec"]["actor_distribution"] == "gsde"
    assert portable["boom_spec"] == source._boom_spec()
    restored = _agent(inner_gsde_noise_scope=scope)
    restored.load(portable)
    _assert_tree_equal(restored.checkpoint_state(), portable)
    training = _clone_tree(source.training_state_dict())
    restored.load_training_state_dict(training)
    _assert_tree_equal(restored.training_state_dict(), training)
    rng = torch.random.get_rng_state().clone()
    expected_metrics = source._update(*data, **labels)
    expected_state = _clone_tree(source.training_state_dict())
    torch.random.set_rng_state(rng)
    actual_metrics = restored._update(*data, **labels)
    _assert_tree_equal(restored.training_state_dict(), expected_state)
    torch.testing.assert_close(actual_metrics["actor_loss"], expected_metrics["actor_loss"])


@pytest.mark.parametrize("override", [{"outer_boom_coef": 0.042}, {"gsde_mean_clip": None}])
@pytest.mark.parametrize("exact", [False, True])
def test_boom_gsde_checkpoint_mismatch_is_atomic(override, exact):
    source, receiving = _agent(), _agent(**override)
    before = _clone_tree(receiving.training_state_dict())
    payload = source.training_state_dict() if exact else source.checkpoint_state()
    with pytest.raises(ValueError, match="(?i)boom|specification"):
        (receiving.load_training_state_dict if exact else receiving.load)(_clone_tree(payload))
    _assert_tree_equal(receiving.training_state_dict(), before)


@pytest.mark.parametrize("feature", ["actor", "latent"])
def test_boom_gsde_strict_fullgraph_actor_updates_match_eager_across_gate(feature):
    eager = _agent(
        gsde_feature_source=feature,
        q_representation="distributional", num_q=5, q_pair_size=2,
        outer_q_actor_reduction="min_pair", dropout=0.2,
    )
    _nonzero_critic(eager)
    compiled = AMBITDMPC2Agent(deepcopy(eager.cfg))
    compiled.load_training_state_dict(_clone_tree(eager.training_state_dict()))
    eager.model.train()
    compiled.model.train()
    eager.actor_loss_scale.fill_(2.15)
    compiled.actor_loss_scale.copy_(eager.actor_loss_scale)
    graphs = _trace_actor_region(compiled)
    gates = []
    for index, all_invalid in enumerate((False, True, False)):
        if index:
            # The compiled detached critic must read later outer updates.
            with torch.no_grad():
                for agent in (eager, compiled):
                    for head in agent.model._Qs.modules_list:
                        head[-1].weight[0].add_(0.002)
        inputs = _inputs(eager, all_invalid=all_invalid)
        before = eager.actor_loss_scale.clone()
        rng = torch.get_rng_state().clone()
        expected = _actor_step(eager, inputs)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        actual = _actor_step(compiled, inputs)
        _assert_tree_close(actual, expected)
        _assert_actor_state_equal(compiled, eager, compiled=True)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        torch.testing.assert_close(
            actual["actor_loss_scale"],
            before.lerp(actual["actor_q_range_instant"], eager.cfg.sac_actor_loss_scale_tau),
        )
        assert torch.isfinite(compiled.model._pi.log_scale.grad).all()
        assert not compiled._boom_actor_region.failed
        gates.append(actual["boom_gate_active"].item())
    assert graphs
    assert gates[0] == 1 and gates[-1] == 0
