"""Whole outer updates against independent pinned TD-MPC2 equations.

The oracle shares only copied network layers, not AMBI's policy, Q decoder,
target builder, CE, percentile estimator, temporal reducer or update methods.
It transcribes official 8bbc14e's update order and optimizer equations. Inner
transition batching and real-environment protocol are outside this contract.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_config_decoupling import _build_cfg


def _policy(network, z):
    mean, raw_std = network(z).chunk(2, dim=-1)
    log_std = -10 + 6 * (raw_std.tanh() + 1)
    eps = torch.randn_like(mean)
    gaussian_log_prob = (-0.5 * eps.square() - log_std - 0.9189385175704956).sum(-1, keepdim=True)
    action = (mean + eps * log_std.exp()).tanh()
    log_prob = gaussian_log_prob - torch.log(F.relu(1 - action.square()) + 1e-6).sum(-1, keepdim=True)
    entropy = -log_prob * ((gaussian_log_prob * mean.shape[-1]) / (log_prob + 1e-8))
    return action, entropy


def _predictions(critics, z, action):
    joint = torch.cat((z, action), -1)
    return torch.stack([head(joint) for head in critics])


def _values(critics, z, action):
    logits = _predictions(critics, z, action)
    bins = torch.linspace(-10, 10, 101, dtype=logits.dtype)
    symlog = (logits.softmax(-1) * bins).sum(-1, keepdim=True)
    return symlog.sign() * (symlog.abs().exp() - 1)


def _ce(logits, targets):
    # Official dense two-hot encoding, including the endpoint's zero-weight wrap.
    x = (targets.sign() * torch.log(1 + targets.abs())).clamp(-10, 10)
    position = (x + 10) / 0.2
    lower = position.floor().long()
    offset = position - lower
    weights = targets.new_zeros(*targets.shape[:-1], 101)
    weights.scatter_(-1, lower, 1 - offset)
    weights.scatter_(-1, (lower + 1) % 101, offset)
    return -(weights * logits.log_softmax(-1)).sum(-1, keepdim=True)


def _oracle_step(model, optim, pi_optim, scale, obs, actions, rewards, terminated, cfg):
    horizon = actions.shape[0]
    with torch.no_grad():
        next_z = model._encoder['state'](obs[1:])
        next_action, _ = _policy(model._pi, next_z)
        values = _values(model._target_Qs, next_z, next_action)
        pair = torch.randperm(5)[:2]
        targets = rewards + 0.99 * (1 - terminated) * values[pair].min(0).values

    model.train()
    z = model._encoder['state'](obs[0])
    zs, consistency = [z], 0
    for t in range(horizon):
        z = model._dynamics(torch.cat((z, actions[t]), -1))
        consistency = consistency + F.mse_loss(z, next_z[t]) * 0.5 ** t
        zs.append(z)
    zs = torch.stack(zs)
    logits = _predictions(model._Qs, zs[:-1], actions)
    reward_logits = model._reward(torch.cat((zs[:-1], actions), -1))
    reward_loss, critic_loss = 0, 0
    for t in range(horizon):
        reward_loss = reward_loss + _ce(reward_logits[t], rewards[t]).mean() * 0.5 ** t
        for head in range(5):
            critic_loss = critic_loss + _ce(logits[head, t], targets[t]).mean() * 0.5 ** t
    consistency = consistency / horizon
    reward_loss = reward_loss / horizon
    critic_loss = critic_loss / (horizon * 5)
    total = 20 * consistency + 0.1 * reward_loss + 0.1 * critic_loss
    optim.zero_grad(set_to_none=True)
    total.backward()
    world_parameters = [p for group in optim.param_groups for p in group['params']]
    grad_norm = torch.nn.utils.clip_grad_norm_(world_parameters, cfg.grad_clip_norm)
    optim.step()

    # Use the detached *pre-model-step* latent rollout, but post-step critics.
    zs = zs.detach()
    action, entropy = _policy(model._pi, zs)
    model._Qs.requires_grad_(False)
    values = _values(model._Qs, zs, action)
    model._Qs.requires_grad_(True)
    pair = torch.randperm(5)[:2]
    actor_q = values[pair].sum(0) / 2
    ordered = actor_q[0].detach().sort(dim=0).values
    position = torch.tensor([5., 95.]) * (len(ordered) - 1) / 100
    low, high = position.floor().long(), position.ceil().long()
    fraction = (position - low).unsqueeze(-1)
    percentiles = ordered[low] * (1 - fraction) + ordered[high] * fraction
    scale.lerp_((percentiles[1] - percentiles[0]).clamp(min=1), 0.01)
    actor_loss = (-(1e-4 * entropy + actor_q / scale).mean((1, 2)) * 0.5 ** torch.arange(horizon + 1)).mean()
    pi_optim.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_grad_norm = torch.nn.utils.clip_grad_norm_(model._pi.parameters(), cfg.grad_clip_norm)
    pi_optim.step()
    with torch.no_grad():
        for target, online in zip(model._target_Qs.parameters(), model._Qs.parameters()):
            target.lerp_(online, 0.01)
    model.eval()
    return {
        'total_loss': total, 'critic_loss': critic_loss, 'reward_loss': reward_loss,
        'consistency_loss': consistency, 'grad_norm': grad_norm,
        'actor_loss': actor_loss, 'actor_grad_norm': actor_grad_norm,
    }


@pytest.mark.parametrize('dropout,clip,action_dim', [(0., 20., 3), (0.01, 20., 21), (0.01, 0.001, 3)])
def test_complete_outer_update_matches_independent_upstream_equations(dropout, clip, action_dim):
    params = json.loads((Path(__file__).parents[1] / 'configs/dmcontrol/algs/td_ambi_prior_reward_qscale.json').read_text())['alg_params']
    cfg = _build_cfg(**{**params, 'compile': False, 'wandb': False, 'dropout': dropout, 'grad_clip_norm': clip})
    cfg.enc_dim = cfg.mlp_dim = 32
    cfg.latent_dim = 16
    cfg.action_dim = action_dim
    cfg.batch_size = 8
    torch.manual_seed(835)
    actual = AMBITDMPC2Agent(cfg)
    # Nonzero learned Q output weights ensure the actor tests Q gradients.
    with torch.no_grad():
        for index, head in enumerate(actual.model._Qs):
            bins = torch.linspace(-1., 1., 101)
            features = torch.linspace(-1., 1., 32)
            head[-1].weight.copy_(0.2 * torch.outer(bins, features))
            head[-1].bias.add_((index + 1) * 0.01 * bins)
        actual.model.soft_update_target_Q(tau=0.4)
        actual.actor_loss_scale.fill_(2.7)
    reference = deepcopy(actual.model)
    optim = torch.optim.Adam([
        {'params': reference._encoder.parameters(), 'lr': 9e-5},
        {'params': reference._dynamics.parameters()},
        {'params': reference._reward.parameters()},
        {'params': reference._Qs.parameters()},
    ], lr=3e-4, eps=1e-8)
    pi_optim = torch.optim.Adam(reference._pi.parameters(), lr=3e-4, eps=1e-5)
    scale = torch.tensor([2.7])
    assert actual.discount == 0.99
    obs = torch.randn(4, 8, 3)
    actions = torch.randn(3, 8, action_dim).tanh()
    rewards = torch.linspace(-2, 3, 24).reshape(3, 8, 1)
    terminated = (torch.arange(24).reshape(3, 8, 1) % 4 == 0).float()
    # Two consecutive updates exercise optimizer moments and target history.
    for _ in range(2):
        rng = torch.random.get_rng_state()
        expected = _oracle_step(reference, optim, pi_optim, scale, obs, actions, rewards, terminated, cfg)
        expected_rng = torch.random.get_rng_state()
        torch.random.set_rng_state(rng)
        observed = actual._update(obs, actions, rewards, terminated)
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        for name, value in expected.items():
            torch.testing.assert_close(observed[name], value.detach(), rtol=1e-4, atol=2e-6, msg=name)
        torch.testing.assert_close(actual.actor_loss_scale, scale, rtol=1e-6, atol=1e-7)
        for name, parameter in actual.model.named_parameters():
            reference_parameter = dict(reference.named_parameters())[name]
            # AMBI's log1p/expm1 and sparse CE preserve the equations, with
            # sub-micro-unit FP32 differences from upstream's dense/log path.
            torch.testing.assert_close(parameter, reference_parameter, rtol=2e-5, atol=2e-7, msg=lambda detail: f'{name}: {detail}')
            if not name.startswith('_target_Qs.'):
                assert parameter.grad is not None, name
                torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-3, atol=2e-7, msg=name)
        for optimizer, reference_optimizer in ((actual.optim, optim), (actual.pi_optim, pi_optim)):
            for state, expected_state in zip(optimizer.state.values(), reference_optimizer.state.values()):
                for key in state:
                    torch.testing.assert_close(state[key], expected_state[key], rtol=1e-3, atol=2e-8)
        assert not any(p.grad is not None for p in actual.model._target_Qs.parameters())
        assert all(not module.training for module in actual.model.modules())
