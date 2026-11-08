"""BOOM actor, replay, and checkpoint integration against independent oracles."""

from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.common.buffer import Buffer
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree, _model


def _boom_model(**overrides):
    settings = {
        "outer_actor_objective": "boom",
        "outer_boom_coef": 0.17,
        "outer_boom_min_std": 0.05,
        "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
        "sac_actor_loss_scale_tau": 0.1,
        "ent_coef": 0.2,
        "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only",
        "outer_q_actor_reduction": "min_all",
        "grad_clip_norm": 100.0,
    }
    settings.update(overrides)
    return _model(**settings)


def _agent(**overrides):
    wrapped = _boom_model(**overrides)
    wrapped.env.close()
    return wrapped.agent


def _labels(agent, *, valid=True):
    shape = (agent.cfg.train_unroll_horizon, agent.cfg.batch_size, agent.cfg.action_dim)
    mean = torch.linspace(-0.9, 0.8, torch.tensor(shape).prod().item()).reshape(shape)
    return (
        mean,
        torch.full(shape, 1.7),
        torch.full((*shape[:-1], 1), valid, dtype=torch.bool),
    )


def _batch(agent):
    horizon, batch = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    return (
        torch.randn(horizon + 1, batch, 3),
        torch.randn(horizon, batch, agent.cfg.action_dim).tanh(),
        torch.randn(horizon, batch, 1),
        torch.zeros(horizon, batch, 1),
    )


@pytest.mark.parametrize("coefficient", [0.0, 0.17])
@pytest.mark.parametrize("horizon,rho", [(1, 0.0), (2, 0.5)])
@pytest.mark.parametrize("valid_labels", [True, False])
def test_boom_actor_step_matches_corrected_reference_and_reuses_sampled_q(
    monkeypatch, horizon, rho, valid_labels, coefficient,
):
    base = _agent(train_unroll_horizon=horizon, rho=rho, outer_boom_coef=coefficient)
    cfg = deepcopy(base.cfg)
    cfg.action_dim = 3
    agent = AMBITDMPC2Agent(cfg)
    reference_pi = deepcopy(agent.model._pi)
    reference_optim = torch.optim.Adam(
        reference_pi.parameters(), lr=cfg.actor_lr, eps=cfg.actor_adam_eps,
    )
    # Starting below the gate makes use of the just-updated shared S observable.
    agent.actor_loss_scale.fill_(1.95)
    zs = torch.randn(horizon + 1, cfg.batch_size, cfg.latent_dim)
    noise = torch.linspace(-1.2, 1.4, horizon * cfg.batch_size * cfg.action_dim).reshape(
        horizon, cfg.batch_size, cfg.action_dim,
    )
    behavior_mean, behavior_log_std, valid = _labels(agent, valid=valid_labels)
    if valid_labels:
        valid[:, -1] = False
    behavior_mean[~valid.expand_as(behavior_mean)] = float("nan")
    behavior_mean.requires_grad_()
    # Collection-time variance is intentionally unrelated to the current actor.
    behavior_log_std.requires_grad_()
    offsets = torch.tensor([-12.0, -2.0, 6.0, 17.0]).reshape(1, cfg.batch_size, 1)

    def q_expression(action):
        return offsets + 0.7 * action.sum(-1, keepdim=True) + 0.2 * action.square().sum(-1, keepdim=True)

    original_pi = agent.model.pi
    action_ref, info_ref = original_pi(zs[:-1], policy=reference_pi, noise=noise)
    raw_q = q_expression(action_ref)
    quantiles = torch.quantile(raw_q[0].detach(), torch.tensor([0.05, 0.95]), dim=0)
    scale = torch.tensor([1.95]).lerp_((quantiles[1] - quantiles[0]).clamp_min(1.0), 0.1)
    assert scale.item() > 2
    temporal = rho ** torch.arange(horizon)
    q_entropy_loss = ((cfg.ent_coef * info_ref["log_prob"] - raw_q / scale).mean((1, 2)) * temporal).mean()
    if valid_labels:
        std = info_ref["log_std"].detach().exp().clamp_min(cfg.outer_boom_min_std)
        target = torch.where(valid, behavior_mean.detach(), 0.0).tanh()
        nll = -torch.distributions.Normal(target, std).log_prob(action_ref).sum(-1)
        logits = (raw_q.detach() / scale).squeeze(-1).masked_fill(~valid.squeeze(-1), -torch.inf)
        weights = logits.softmax(dim=1)
        alignment = cfg.outer_boom_coef * ((weights * nll).sum(1) * temporal).mean() / scale.reshape(())
    else:
        alignment = q_entropy_loss.new_zeros(())
    expected_loss = q_entropy_loss + alignment
    expected_loss.backward()
    expected_gradients = [p.grad.clone() for p in reference_pi.parameters()]
    torch.nn.utils.clip_grad_norm_(reference_pi.parameters(), cfg.grad_clip_norm)
    reference_optim.step()

    calls = {"policy": 0, "critic": 0}

    def fixed_policy(z, **kwargs):
        calls["policy"] += 1
        assert z.shape[0] == horizon
        return original_pi(z, noise=noise, **kwargs)

    def critic(z, action, **kwargs):
        calls["critic"] += 1
        assert z.shape[0] == horizon
        assert kwargs == {"reduction": "all", "detach": True}
        q = q_expression(action)
        return torch.stack([q + head for head in range(cfg.num_q)], dim=0)

    monkeypatch.setattr(agent.model, "pi", fixed_policy)
    monkeypatch.setattr(agent.model, "Q", critic)
    captured_gradients = []
    original_clip = agent._clip_actor_grad_norm_

    def capture_and_clip():
        captured_gradients.extend(p.grad.clone() for p in agent.model._pi.parameters())
        return original_clip()

    monkeypatch.setattr(agent, "_clip_actor_grad_norm_", capture_and_clip)
    metrics = agent._update_actor(
        zs, behavior_pre_tanh_mean=behavior_mean,
        behavior_log_std=behavior_log_std, behavior_policy_valid=valid,
    )
    assert calls == {"policy": 1, "critic": 1}
    torch.testing.assert_close(agent.actor_loss_scale, scale)
    torch.testing.assert_close(metrics["actor_loss"], expected_loss.detach())
    torch.testing.assert_close(metrics["boom_alignment_loss"], alignment.detach())
    assert metrics["boom_gate_active"] == 1
    assert behavior_mean.grad is None
    assert behavior_log_std.grad is None
    for actual, expected in zip(captured_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-7)
    for actual, expected in zip(agent.model._pi.parameters(), reference_pi.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("missing", ["mean", "valid"])
def test_missing_boom_metadata_fails_before_state_or_rng_changes(missing):
    agent = _agent()
    mean, log_std, valid = _labels(agent)
    zs = torch.zeros(agent.cfg.train_unroll_horizon + 1, agent.cfg.batch_size, agent.cfg.latent_dim)
    before = _clone_tree(agent.checkpoint_state())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="(?i)boom|behavior"):
        agent._update_actor(
            zs, behavior_pre_tanh_mean=None if missing == "mean" else mean,
            behavior_log_std=log_std, behavior_policy_valid=None if missing == "valid" else valid,
        )
    _assert_tree_equal(agent.checkpoint_state(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_boom_replay_mean_capture_and_random_mask_reach_training():
    wrapped = _boom_model(inner_execution_action="mean")
    try:
        obs = torch.zeros(3)
        assert wrapped._start_episode_staging(obs) == 1
        assert not wrapped._episode_staging["behavior_policy_valid"][0]
        actual_act = wrapped.agent.act
        captured = []

        def capture_act(*args, **kwargs):
            result = actual_act(*args, **kwargs)
            captured.append(result[1])
            return result

        wrapped.agent.act = capture_act
        action = wrapped._act_agent(obs, t0=True, eval_mode=False)
        assert captured[0] is not None
        expected_mean = captured[0]["pre_tanh_mean"].clone()
        wrapped._stage_transition(1, obs, action, 1.0, False)
        torch.testing.assert_close(wrapped._episode_staging["behavior_pre_tanh_mean"][1], expected_mean)
        assert wrapped._episode_staging["behavior_policy_valid"][1]
        # Exploration/seed actions have no trustworthy adapted-policy target.
        wrapped._random_action_norm()
        wrapped._stage_transition(2, obs, torch.tensor([0.1]), 0.0, False)
        assert not wrapped._episode_staging["behavior_policy_valid"][2]

        buffer = Buffer(wrapped.agent.cfg)
        rows = 7
        means = torch.linspace(-1, 1, rows).reshape(rows, 1)
        buffer.add(TensorDict({
            "obs": torch.randn(rows, 3), "action": torch.zeros(rows, 1),
            "reward": torch.ones(rows), "terminated": torch.zeros(rows),
            "behavior_pre_tanh_mean": means,
            "behavior_log_std": torch.full((rows, 1), -1.2),
            "behavior_policy_valid": torch.ones(rows, dtype=torch.bool),
        }, batch_size=[rows]))
        metrics = wrapped.agent.update(buffer)
        assert wrapped.agent.num_updates == 1
        assert metrics["boom_valid_fraction"] == 1
        assert torch.isfinite(torch.as_tensor(metrics["actor_loss"]))
    finally:
        wrapped.env.close()


def test_boom_checkpoint_roundtrip_and_exact_next_update():
    source = _agent()
    data = _batch(source)
    mean, log_std, valid = _labels(source)
    labels = dict(behavior_pre_tanh_mean=mean, behavior_log_std=log_std, behavior_policy_valid=valid)
    source.actor_loss_scale.fill_(3.2)
    source._update(*data, **labels)
    portable = _clone_tree(source.checkpoint_state())
    assert portable["checkpoint_version"] == 9
    assert portable["boom_spec"] == source._boom_spec()
    restored = _agent()
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


@pytest.mark.parametrize("override", [{"outer_boom_coef": 0.4}, {"outer_boom_min_std": 0.1}])
def test_boom_checkpoint_rejects_changed_alignment_before_mutation(override):
    source = _agent()
    receiving = _agent(**override)
    before = _clone_tree(receiving.checkpoint_state())
    with pytest.raises(ValueError, match="(?i)boom"):
        receiving.load(_clone_tree(source.checkpoint_state()))
    _assert_tree_equal(receiving.checkpoint_state(), before)


@pytest.mark.parametrize("receiving_boom", [False, True])
def test_boom_and_q_actor_checkpoints_require_explicit_raw_weight_transfer(receiving_boom):
    boom = _agent()
    ordinary = _agent(outer_actor_objective="q_maximization")
    source, receiving = (ordinary, boom) if receiving_boom else (boom, ordinary)
    before = _clone_tree(receiving.checkpoint_state())
    with pytest.raises(ValueError, match="(?i)boom|objective"):
        receiving.load(_clone_tree(source.checkpoint_state()))
    _assert_tree_equal(receiving.checkpoint_state(), before)
    receiving.actor_loss_scale.fill_(9)
    receiving.load(_clone_tree(source.model.state_dict()))
    torch.testing.assert_close(receiving.actor_loss_scale, torch.ones(1))
    _assert_tree_equal(receiving.model.state_dict(), source.model.state_dict())
