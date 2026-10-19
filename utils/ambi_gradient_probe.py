"""Frozen reward-critic gradients and paired simulator interventions.

This measures the inherited prior, before any inner learning. It neither
implements MAGE nor estimates a full closed-loop refinement return.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import numpy as np
import torch

from utils.ambi_real_calibration import evaluate_real_branches


@contextmanager
def frozen_parameters(module):
    """Freeze weights, retaining derivatives through actions and latent states."""
    modes = [(m, m.training) for m in module.modules()]
    flags = [(p, p.requires_grad) for p in module.parameters()]
    try:
        module.eval().requires_grad_(False)
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        for m, mode in modes:
            m.training = mode


class PriorRewardObjective:
    """Q^prior and differentiable H-step reward + frozen Q^prior objectives.

    Only single-value critics evaluating the primary SAC prior are supported.
    An auxiliary SAC-return critic takes precedence over the soft primary Q.
    The same explicit ensemble pair is used at the root and all endpoints.
    """
    def __init__(self, wrapper, *, reduction, seed):
        self.model, self.cfg = wrapper.agent.model, wrapper.cfg
        self.device, self.discount = wrapper.agent.device, float(wrapper.agent.discount)
        if self.cfg.obs != 'state' or self.cfg.episodic:
            raise ValueError('Gradient calibration requires continuing state observations.')
        if self.cfg.critic_value_mode != 'single':
            raise ValueError('Gradient calibration currently requires a single-value critic.')
        auxiliary = self.cfg.aux_return_mode
        if auxiliary == 'sac':
            self.critic, self.source = self.model._aux_return_Qs, 'aux_return'
        elif auxiliary == 'off' and self.cfg.outer_critic_target == 'reward_only':
            self.critic, self.source = self.model._Qs, 'sac'
        else:
            raise ValueError('Need aux_return_mode=sac or a reward-only primary critic; '
                             'soft Q cannot be calibrated against raw reward returns.')
        if reduction not in {'mean_pair', 'min_pair', 'mean_all', 'min_all'}:
            raise ValueError('Select an explicit scalar ensemble reduction.')
        if not 0 < self.discount < 1:
            raise ValueError('Continuing calibration requires 0 < discount < 1.')
        self.reduction = reduction
        self.pair_indices = None
        if reduction.endswith('_pair'):
            generator = torch.Generator(device=self.device).manual_seed(seed)
            self.pair_indices = self.model.q_backend.sample_pair_indices(self.device, generator=generator)

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def q(self, z, action):
        # Eager forwarding prevents diagnostic batch sizes compiling learner kernels.
        return self.model.Q(z, action, qs=self.critic._forward_eager,
                            reduction=self.reduction, pair_indices=self.pair_indices,
                            trusted_pair_indices=True)

    @torch.no_grad()
    def anchor(self, observation):
        z = self.model.encode(self.tensor(observation)[None])
        mean, _ = self.model._policy_parameters(z)
        return mean[0].cpu().numpy()

    @torch.no_grad()
    def actor(self, observations, noise):
        z = self.model.encode(self.tensor(observations))
        action, _ = self.model.pi(z, noise=self.tensor(noise))
        return action.cpu().numpy()

    @torch.no_grad()
    def real_q(self, observations, actions):
        z = self.model.encode(self.tensor(observations))
        return self.q(z, self.tensor(actions)).reshape(-1).cpu().numpy()

    def scores(self, observation, first_action, noise, horizon):
        """Fixed first action, then feedback prior, including its sampled Q action.

        Noise is indexed by decision: noise[0] is unused, noise[H] samples the
        terminal action. Returns one value per rollout, without reducing noise.
        """
        if noise.ndim != 3 or len(noise) <= horizon or horizon < 1:
            raise ValueError('Need noise[0:H+1, rollouts, action_dim] and H >= 1.')
        z0 = self.model.encode(self.tensor(observation)[None]).detach()
        action = first_action.reshape(1, -1)
        critic = self.q(z0, action).reshape(())
        count = noise.shape[1]
        z, action = z0.expand(count, -1), action.expand(count, -1)
        reward = z.new_zeros(count, 1)
        for step in range(horizon):
            if step:
                action, _ = self.model.pi(z, noise=noise[step])
            reward = reward + self.discount ** step * self.model.decode_reward(self.model.reward(z, action))
            z = self.model.next(z, action)
        tail_action, _ = self.model.pi(z, noise=noise[horizon])
        bootstrap = self.discount ** horizon * self.q(z, tail_action)
        return dict(critic=critic, reward=reward.reshape(-1),
                    bootstrap=bootstrap.reshape(-1), total=(reward + bootstrap).reshape(-1))


def cosine(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator > 1e-12 else None


def gradient_directions(objective, observation, anchor_u, noise, horizon, *, random_directions, seed):
    """Report action gradients; use their pre-tanh pullbacks for feasible steps."""
    anchor_u = np.asarray(anchor_u, dtype=np.float32)
    action = objective.tensor(anchor_u).tanh().detach().requires_grad_(True)
    gradients = {}
    for name, h in [('critic', 1), ('model_h1', 1), (f'model_h{horizon}', horizon)]:
        if name in gradients:
            continue
        scores = objective.scores(observation, action, noise, h)
        value = scores['critic'] if name == 'critic' else scores['total'].mean()
        derivative = torch.autograd.grad(value, action)[0]
        if not torch.isfinite(derivative).all():
            raise ValueError(f'Nonfinite {name} gradient.')
        gradients[name] = derivative.detach().cpu().numpy()
    a = action.detach().cpu().numpy()
    pulled_back = {name: grad * (1 - a * a) for name, grad in gradients.items()}
    directions, skipped = {}, []
    for name, grad in pulled_back.items():
        norm = float(np.linalg.norm(grad))
        if norm <= 1e-10:
            skipped.append(name)
        else:
            directions[name] = grad / norm
    rng = np.random.default_rng(seed)
    for index in range(random_directions):
        value = rng.normal(size=anchor_u.shape)
        directions[f'random_{index}'] = (value / np.linalg.norm(value)).astype(np.float32)
    details = dict(action_gradients={name: g.tolist() for name, g in gradients.items()},
                   pretanh_gradients={name: g.tolist() for name, g in pulled_back.items()},
                   action_gradient_norms={name: float(np.linalg.norm(g)) for name, g in gradients.items()},
                   pretanh_gradient_norms={name: float(np.linalg.norm(g)) for name, g in pulled_back.items()},
                   cosine_action={name: cosine(gradients['critic'], g) for name, g in gradients.items()
                                  if name != 'critic'},
                   cosine_pretanh={name: cosine(pulled_back['critic'], g) for name, g in pulled_back.items()
                                   if name != 'critic'},
                   skipped_zero_directions=skipped)
    return directions, details


def probe_root(objective, envs, snapshot, observation, anchor_u, *, horizon,
               gradient_noise, evaluation_noise, epsilons, random_directions,
               direction_seed, original_remaining_steps):
    """Pair every intervention with one shared baseline and held-out policy noise."""
    directions, gradients = gradient_directions(
        objective, observation, anchor_u, objective.tensor(gradient_noise), horizon,
        random_directions=random_directions, seed=direction_seed)
    candidates = [('baseline', 0., 0, np.asarray(anchor_u, np.float32))]
    for name, direction in directions.items():
        for epsilon in epsilons:
            for sign in (-1, 1):
                candidates.append((name, epsilon, sign, anchor_u + sign * epsilon * direction))
    rows, measured, work = [], {}, {'simulator_decisions': 0}
    base_action = objective.tensor(anchor_u).tanh().cpu().numpy()
    for name, epsilon, sign, u in candidates:
        action = objective.tensor(u).tanh().detach()
        with torch.no_grad():
            model = objective.scores(observation, action, objective.tensor(evaluation_noise), horizon)
            one_step = objective.scores(observation, action, objective.tensor(evaluation_noise), 1)
        calls = 0

        def prefix_actor(observations, noise):
            nonlocal calls
            calls += 1
            if calls == 1:
                return np.repeat(action.cpu().numpy()[None], len(observations), axis=0)
            return objective.actor(observations, noise)

        real = evaluate_real_branches(
            envs, snapshot, prefix_actor, objective.actor, objective.real_q,
            evaluation_noise[:horizon], evaluation_noise[horizon:],
            discount=objective.discount, horizon=horizon,
            original_remaining_steps=original_remaining_steps, reward_bound=2.)
        if not all(row['mc_complete'] for row in real['rows']):
            raise ValueError('Unexpected simulator truncation: refusing an incomplete gradient comparison.')
        work['simulator_decisions'] += real['work']['simulator_decisions']
        candidate_rows = []
        for i, actual in enumerate(real['rows']):
            row = dict(direction=name, epsilon=float(epsilon), sign=sign,
                       action=action.cpu().tolist(), pretanh_action=np.asarray(u).tolist(),
                       action_displacement=float(np.linalg.norm(action.cpu().numpy() - base_action)),
                       critic=float(model['critic']), model_h1=float(one_step['total'][i]),
                       model_return=float(model['total'][i]), model_prefix_reward=float(model['reward'][i]),
                       model_bootstrap=float(model['bootstrap'][i]), **actual)
            rows.append(row)
            candidate_rows.append(row)
        measured[name, epsilon, sign] = candidate_rows
    baseline = measured['baseline', 0., 0]
    fields = ('critic', 'model_h1', 'model_return', 'real_bootstrapped_return', 'real_mc_return')
    contrasts = []
    for name in directions:
        for epsilon in epsilons:
            minus, plus = measured[name, epsilon, -1], measured[name, epsilon, 1]
            contrast = dict(direction=name, epsilon=float(epsilon))
            for field in fields:
                gains = [p[field] - b[field] for p, b in zip(plus, baseline)]
                derivatives = [(p[field] - m[field]) / (2 * epsilon) for p, m in zip(plus, minus)]
                contrast[field + '_gain'] = float(np.mean(gains))
                contrast[field + '_central_difference'] = float(np.mean(derivatives))
            contrast['mc_gain_omitted_tail_bound'] = float(np.mean([
                p['return_truncation_bound'] + b['return_truncation_bound']
                for p, b in zip(plus, baseline)]))
            contrast['mc_central_difference_omitted_tail_bound'] = float(np.mean([
                (p['return_truncation_bound'] + m['return_truncation_bound']) / (2 * epsilon)
                for p, m in zip(plus, minus)]))
            contrasts.append(contrast)
    calibration = {field + '_bias': float(np.mean([r[field] - r['real_mc_return'] for r in baseline]))
                   for field in ('critic', 'model_h1', 'model_return', 'real_bootstrapped_return')}
    calibration['model_error_with_common_endpoint_critic'] = float(np.mean([
        r['model_return'] - r['real_bootstrapped_return'] for r in baseline]))
    return dict(gradients=gradients, directions={n: d.tolist() for n, d in directions.items()},
                calibration=calibration, rows=rows, contrasts=contrasts, work=work)


def summarize_roots(roots):
    """Average rollouts within roots, roots within source episodes, then episodes.

    SE is across source episodes, not roots or rollout repetitions. This is a
    conditional checkpoint diagnostic, not uncertainty across training seeds.
    """
    groups = {}
    for root in roots:
        for contrast in root['contrasts']:
            key = (contrast['direction'], contrast['epsilon'])
            groups.setdefault(key, {}).setdefault(root['episode_id'], []).append(contrast)
    summaries = []
    for (direction, epsilon), episodes in groups.items():
        metric_names = [key for key in next(iter(episodes.values()))[0] if key not in ('direction', 'epsilon')]
        episode_means = {name: [float(np.mean([row[name] for row in rows])) for rows in episodes.values()]
                         for name in metric_names}
        values = episode_means['real_mc_return_gain']
        summaries.append(dict(direction=direction, epsilon=epsilon, source_episodes=len(episodes),
            roots=sum(len(rows) for rows in episodes.values()),
            means={name: float(np.mean(values)) for name, values in episode_means.items()},
            mc_gain_se_across_source_episodes=(float(np.std(values, ddof=1) / math.sqrt(len(values)))
                                              if len(values) > 1 else None),
            source_episodes_with_positive_mc_gain=sum(value > 0 for value in values)))
    return summaries


def summarize_calibration(roots):
    """Episode-balanced baseline bias and gradient-cosine summaries."""
    grouped = {}
    for root in roots:
        metrics = dict(root['calibration'])
        for coordinates in ('action', 'pretanh'):
            metrics.update({f'cosine_{coordinates}_{name}': value
                            for name, value in root['gradients'][f'cosine_{coordinates}'].items()})
        for name, value in metrics.items():
            if value is not None:
                grouped.setdefault(name, {}).setdefault(root['episode_id'], []).append(value)
    result = {}
    for name, episodes in grouped.items():
        values = [float(np.mean(rows)) for rows in episodes.values()]
        result[name] = dict(mean=float(np.mean(values)), source_episodes=len(values),
            roots=sum(len(rows) for rows in episodes.values()),
            se_across_source_episodes=(float(np.std(values, ddof=1) / math.sqrt(len(values)))
                                      if len(values) > 1 else None))
    return result
