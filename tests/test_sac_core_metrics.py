import numpy as np
import pytest
import torch

from RL.sac_core import SACAgent, SACConfig, _grad_norm


class StaticBatch:
    def __init__(self, batch):
        self.batch = batch
        self.sample_calls = 0

    def sample(self, *_args):
        self.sample_calls += 1
        return self.batch


def _batch(batch_size=8, obs_dim=3, action_dim=2):
    generator = torch.Generator().manual_seed(811)
    return {
        "obs": torch.randn(batch_size, obs_dim, generator=generator),
        "actions": torch.tanh(torch.randn(batch_size, action_dim, generator=generator)),
        "rewards": torch.randn(batch_size, 1, generator=generator),
        "next_obs": torch.randn(batch_size, obs_dim, generator=generator),
        "dones": (torch.rand(batch_size, 1, generator=generator) < 0.25).float(),
    }


def test_auto_entropy_update_reports_diagnostics_from_existing_forwards(monkeypatch):
    agent = SACAgent(
        obs_dim=3,
        action_dim=2,
        config=SACConfig(net_arch=(8,), ent_coef="auto", seed=17, device="cpu"),
    )
    replay = StaticBatch(_batch())

    actor_outputs = []
    critic_outputs = []
    target_critic_outputs = []
    original_action_log_prob = agent.actor.action_log_prob
    original_critic_forward = agent.critic.forward
    original_target_critic_forward = agent.critic_target.forward

    def capture_action_log_prob(obs):
        output = original_action_log_prob(obs)
        actor_outputs.append(tuple(value.detach().clone() for value in output))
        return output

    def capture_critic(obs, actions):
        output = original_critic_forward(obs, actions)
        critic_outputs.append(tuple(value.detach().clone() for value in output))
        return output

    def capture_target_critic(obs, actions):
        output = original_target_critic_forward(obs, actions)
        target_critic_outputs.append(tuple(value.detach().clone() for value in output))
        return output

    monkeypatch.setattr(agent.actor, "action_log_prob", capture_action_log_prob)
    monkeypatch.setattr(agent.critic, "forward", capture_critic)
    monkeypatch.setattr(agent.critic_target, "forward", capture_target_critic)

    torch.manual_seed(29)
    metrics = agent.update(replay, gradient_steps=1, batch_size=8)

    expected_keys = {
        "actor_loss",
        "critic_loss",
        "ent_coef",
        "ent_coef_loss",
        "policy_log_prob",
        "policy_entropy",
        "q1_mean",
        "q2_mean",
        "q_target_mean",
        "q_policy_mean",
        "q_disagreement_mean",
        "td_error_abs_mean",
        "actor_grad_norm",
        "critic_grad_norm",
        "ent_coef_grad_norm",
    }
    assert set(metrics) == expected_keys
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["policy_entropy"] == pytest.approx(-metrics["policy_log_prob"])
    assert metrics["actor_grad_norm"] > 0.0
    assert metrics["critic_grad_norm"] > 0.0
    assert metrics["ent_coef_grad_norm"] > 0.0

    # Diagnostics reuse the two actor calls, two online-critic calls, and one
    # target-critic call already required by a single SAC gradient step.
    assert replay.sample_calls == 1
    assert len(actor_outputs) == 2
    assert len(critic_outputs) == 2
    assert len(target_critic_outputs) == 1

    _, policy_log_prob = actor_outputs[0]
    _, next_log_prob = actor_outputs[1]
    current_q1, current_q2 = critic_outputs[0]
    policy_q1, policy_q2 = critic_outputs[1]
    next_q1, next_q2 = target_critic_outputs[0]
    target_q = replay.batch["rewards"] + (1.0 - replay.batch["dones"]) * agent.config.gamma * (
        torch.min(next_q1, next_q2) - metrics["ent_coef"] * next_log_prob
    )

    assert metrics["policy_log_prob"] == pytest.approx(float(policy_log_prob.mean()))
    assert metrics["q1_mean"] == pytest.approx(float(current_q1.mean()))
    assert metrics["q2_mean"] == pytest.approx(float(current_q2.mean()))
    assert metrics["q_target_mean"] == pytest.approx(float(target_q.mean()))
    assert metrics["q_policy_mean"] == pytest.approx(float(torch.min(policy_q1, policy_q2).mean()))
    assert metrics["q_disagreement_mean"] == pytest.approx(float((current_q1 - current_q2).abs().mean()))
    expected_td_error = 0.5 * ((current_q1 - target_q).abs() + (current_q2 - target_q).abs())
    assert metrics["td_error_abs_mean"] == pytest.approx(float(expected_td_error.mean()))


def test_fixed_entropy_update_omits_automatic_tuning_metrics():
    agent = SACAgent(
        obs_dim=3,
        action_dim=2,
        config=SACConfig(net_arch=(8,), ent_coef=0.2, seed=23, device="cpu"),
    )
    metrics = agent.update(StaticBatch(_batch()), gradient_steps=1, batch_size=8)

    assert metrics["ent_coef"] == pytest.approx(0.2)
    assert "ent_coef_loss" not in metrics
    assert "ent_coef_grad_norm" not in metrics
    assert all(np.isfinite(value) for value in metrics.values())


def test_grad_norm_does_not_modify_gradients():
    layer = torch.nn.Linear(3, 2)
    layer(torch.ones(4, 3)).square().mean().backward()
    before = [parameter.grad.detach().clone() for parameter in layer.parameters()]
    expected = torch.stack([gradient.norm(2) for gradient in before]).norm(2)

    actual = _grad_norm(layer.parameters())

    torch.testing.assert_close(actual, expected)
    for parameter, original_gradient in zip(layer.parameters(), before):
        torch.testing.assert_close(parameter.grad, original_gradient, rtol=0, atol=0)
