"""Frozen Dog Stand diagnosis: sampled episodes, common roots, and real returns.

Writes immutable local JSON; never trains or publishes. See the protocol in
configs/research/DOG_CHECKPOINT_DIAGNOSTICS.md.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_ambi_calibration import FrozenCallbacks
from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _make_env,
    _outer_state_digest, _seed_spaces,
)
from probe_ambi_gradients import _resolved
from RL.tdmpc2_core.inner_trace import InnerActionTrace
from utils.ambi_benchmark import atomic_json, canonical_hash, read_json, solver_seed
from utils.ambi_gradient_probe import PriorRewardObjective, frozen_parameters, probe_root
from utils.ambi_real_calibration import (
    SimulatorSnapshot, capture_simulator_snapshot, evaluate_real_branches,
)
from utils.checkpoint_context import load_checkpoint_context


def checkpoint_path(root, seed, step):
    paths = list(Path(root).glob(f'seed{seed}/*/models/model:*_seed{seed}_0_{step}'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one checkpoint for seed {seed}, step {step}: {paths}')
    return paths[0]


def load_model(checkpoint, *, graph=False, device='cuda'):
    context = load_checkpoint_context(checkpoint)
    resolved = _resolved(context)
    if resolved['environment']['params'].get('task') != 'dog-stand':
        raise ValueError('This campaign requires Dog Stand.')
    params = resolved['algorithm_config']['alg_params']
    params.update(compile=graph, compile_strict=graph, inner_cuda_graph=graph, wandb=False)
    if (params.get('inner_actor_writeback_coef', 0) or params.get('inner_critic_writeback_coef', 0)
            or params.get('inner_outer_replay_fraction', 0)):
        raise ValueError('Frozen diagnosis requires zero writeback and no outer replay mixing.')
    env = _make_env(resolved)
    model, _ = _initialize_frozen_model(resolved, env, checkpoint, 55, device=device)
    return model, env, resolved


def reset_episode(model, seed):
    """Reset private noise, retaining only the graph's disposable allocation.

    The graph resets actor, critic, targets, optimizers and action-scoped scale
    on every call. Reusing its allocation avoids a fresh compilation per episode.
    Its isolated RNG is explicitly reset, independent of warmup and checkpoints.
    """
    engine = model.agent.inner_engine
    runner = engine._whole_graph_runner
    engine.reset_for_evaluation(seed, reuse_action_pool=True)
    if runner is not None:
        runner.rng_state = torch.Generator(device=engine.device).manual_seed(seed).get_state()
        runner.calls = 0
        engine._whole_graph_runner = runner


@torch.no_grad()
def head_values(objective, observation, actions):
    actions = objective.tensor(actions).reshape(-1, objective.cfg.action_dim)
    z = objective.model.encode(objective.tensor(observation)[None]).expand(len(actions), -1)
    values = objective.model.Q(z, actions, qs=objective.critic._forward_eager, return_type='all')
    return values.squeeze(-1).transpose(0, 1).cpu().numpy().tolist()


def run_episodes(args, checkpoint, output):
    model, env, resolved = load_model(checkpoint, graph=args.device == 'cuda', device=args.device)
    before = _outer_state_digest(model)
    objective = PriorRewardObjective(model, reduction='mean_all', seed=55)
    roots, episodes = [], []
    started = time.perf_counter()
    try:
        warm_obs = env.reset(seed=args.seeds[0])[0]
        warm_start = time.perf_counter()
        model.predict(warm_obs, deterministic=False, episode_start=True, collect_diagnostics=False)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        warmup = time.perf_counter() - warm_start
        for controller in ('prior', 'inner'):
            for seed in args.seeds:
                seed_value = solver_seed(55, 'dog-diagnosis-episode', seed)
                reset_episode(model, seed_value)
                noise = np.random.default_rng(seed_value).standard_normal(
                    (args.steps, 1, model.cfg.action_dim)).astype(np.float32)
                _seed_spaces(env, seed)
                obs, _ = env.reset(seed=seed)
                rewards, q_rows = [], []
                episode_start = time.perf_counter()
                for decision in range(args.steps):
                    if decision in args.decisions:
                        roots.append(dict(root_id=f'{args.label}-{controller}-{seed}-{decision}',
                            episode_id=f'{args.label}-{controller}-{seed}', source=args.label,
                            controller=controller, environment_seed=seed, decision_index=decision,
                            observation=np.asarray(obs).tolist(),
                            anchor_pretanh_action=objective.anchor(obs).tolist(),
                            snapshot=capture_simulator_snapshot(env).to_dict()))
                    if controller == 'prior':
                        action = objective.actor(np.asarray(obs)[None], noise[decision])[0]
                    else:
                        action, _ = model.predict(obs, deterministic=False,
                            episode_start=decision == 0, collect_diagnostics=False)
                    if decision % 25 == 0:
                        q_rows.append(dict(decision=decision, heads=head_values(objective, obs, [action])[0]))
                    obs, reward, terminal, truncated, _ = env.step(action)
                    rewards.append(float(reward))
                    if terminal or truncated:
                        break
                if args.steps == 500 and len(rewards) != 500:
                    raise RuntimeError('Dog Stand episode ended before its 500-decision horizon.')
                row = dict(controller=controller, seed=seed, return_=sum(rewards), length=len(rewards),
                    rewards=rewards, q_samples=q_rows, seconds=time.perf_counter() - episode_start)
                episodes.append(row)
                atomic_json(output / f'episode-{controller}-{seed}.json', row)
                print(json.dumps(dict(event='episode', label=args.label, **{k:v for k,v in row.items()
                      if k not in ('rewards', 'q_samples')})), flush=True)
        after = _outer_state_digest(model)
        if before != after:
            raise RuntimeError('Frozen outer model/optimizer/scale changed during episodes.')
        result = dict(schema=1, complete=True, label=args.label, checkpoint=str(checkpoint),
            checkpoint_sha256=_file_sha256(checkpoint), resolved=resolved,
            execution_action='sampled', pairing='environment seeds; private controller noise across checkpoints',
            full_episodes=args.steps == 500, outer_state_before=before, outer_state_after=after,
            warmup_seconds=warmup, total_seconds=time.perf_counter() - started,
            episodes=episodes, roots=roots)
        atomic_json(output / 'episodes.json', result)
        return result
    finally:
        _close_resources(model, env)


def make_bank(episodes_dir, output):
    # Balanced successful and collapsed behavior; selection is fixed before
    # looking at checkpoint-probe outcomes. Reuse the exact roots in every arm.
    selection = [('s55-675000', 'prior'), ('s55-700000', 'inner'), ('s56-575000', 'inner')]
    roots, sources = [], []
    environment = None
    for label, controller in selection:
        data = read_json(Path(episodes_dir) / label / 'episodes.json')
        if not data['complete'] or not data['full_episodes']:
            raise ValueError('Common banks require completed full-episode source shards.')
        env = data['resolved']['environment']
        if environment is not None and environment != env:
            raise ValueError('Source environments differ.')
        environment = env
        sources.append(dict(label=label, checkpoint_sha256=data['checkpoint_sha256'], controller=controller))
        selected = [r for r in data['roots'] if r['controller'] == controller
                    and r['environment_seed'] in [101, 102] and r['decision_index'] == 200]
        if len(selected) != 2:
            raise ValueError(f'Incomplete root selection for {label}.')
        roots.extend(selected)
    bank = dict(schema=1, kind='dog_checkpoint_common_roots_v1', environment=environment,
                sources=sources, roots=roots)
    bank['id'] = canonical_hash(bank)
    if Path(output).exists():
        raise FileExistsError(output)
    atomic_json(output, bank)


def load_bank(path, environment):
    bank = read_json(path)
    digest = bank.pop('id')
    if canonical_hash(bank) != digest or bank['kind'] != 'dog_checkpoint_common_roots_v1':
        raise ValueError('Invalid common bank identity.')
    bank['id'] = digest
    if bank['environment'] != environment:
        raise ValueError('Common bank environment mismatch.')
    ids = [r['root_id'] for r in bank['roots']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Common bank needs unique roots.')
    for root in bank['roots']:
        SimulatorSnapshot.from_dict(root['snapshot'])
        if not np.isfinite(root['observation']).all() or not np.isfinite(root['anchor_pretanh_action']).all():
            raise ValueError('Nonfinite common root.')
    return bank


@torch.no_grad()
def same_actions(objective, envs, snapshot, observation, actions, noise, remaining):
    """Replay one identical open-loop action sequence in model and simulator.

    Use the *actual* endpoint prior action in both critics. This separates
    reward/dynamics error from terminal critic bias without mixing encoder spaces.
    No Q bootstrap is inserted at the end of the Monte Carlo continuation.
    """
    horizon, count, _ = actions.shape
    z = objective.model.encode(objective.tensor(observation)[None]).expand(count, -1)
    reward = z.new_zeros(count, 1)
    for step in range(horizon):
        action = objective.tensor(actions[step])
        reward += objective.discount ** step * objective.model.decode_reward(objective.model.reward(z, action))
        z = objective.model.next(z, action)
    calls, endpoint = 0, {}
    def prefix(obs, epsilon):
        nonlocal calls
        action = actions[calls]
        calls += 1
        return action
    def q(obs, action):
        endpoint['action'] = action.copy()
        endpoint['observation'] = obs.copy()
        return objective.real_q(obs, action)
    real = evaluate_real_branches(envs, snapshot, prefix, objective.actor, q,
        noise[:horizon], noise[horizon:], horizon=horizon, discount=objective.discount,
        original_remaining_steps=remaining, reward_bound=2.)
    if calls != horizon or 'action' not in endpoint or any(not r['mc_complete'] for r in real['rows']):
        raise RuntimeError('Incomplete open-loop diagnostic.')
    model_q = objective.q(z, objective.tensor(endpoint['action'])).reshape(-1).cpu().numpy()
    encoded_real = objective.model.encode(objective.tensor(endpoint['observation']))
    latent_rmse = (z - encoded_real).square().mean(-1).sqrt().cpu().numpy()
    rows = []
    for index, row in enumerate(real['rows']):
        model_reward = float(reward[index].item())
        model_bootstrap = objective.discount ** horizon * float(model_q[index])
        rows.append(dict(**row, model_prefix_reward=model_reward, model_bootstrap=model_bootstrap,
            model_return=model_reward + model_bootstrap,
            reward_error=model_reward - row['real_prefix_reward'],
            model_endpoint_value_error=model_bootstrap - row['real_bootstrap'],
            terminal_critic_error=row['bootstrap_prediction_error'],
            total_prediction_error=model_reward + model_bootstrap - row['real_mc_return'],
            latent_rmse=float(latent_rmse[index])))
    return dict(rows=rows, actions=actions.tolist(), work=real['work'], timing=real['timing'])


@torch.no_grad()
def imagined_actions(objective, policy, bounds, observation, noise, horizon):
    z = objective.model.encode(objective.tensor(observation)[None]).expand(noise.shape[1], -1)
    actions = []
    for step in range(horizon):
        action, _ = objective.model.pi(z, policy=policy, noise=objective.tensor(noise[step]), **bounds)
        actions.append(action.cpu().numpy())
        z = objective.model.next(z, action)
    return np.stack(actions)


def measure_first_actions(objective, envs, root, actions, noise):
    snapshot = SimulatorSnapshot.from_dict(root['snapshot'])
    calls = 0
    def prefix(obs, epsilon):
        nonlocal calls
        calls += 1
        return actions if calls == 1 else objective.actor(obs, epsilon)
    real = evaluate_real_branches(envs, snapshot, prefix, objective.actor, objective.real_q,
        noise[:3], noise[3:], horizon=3, discount=objective.discount,
        original_remaining_steps=500-root['decision_index'], reward_bound=2.)
    if any(not row['mc_complete'] for row in real['rows']):
        raise RuntimeError('Incomplete first-action return.')
    values = objective.real_q(np.repeat(np.asarray(root['observation'])[None], len(actions), axis=0), actions)
    heads = head_values(objective, root['observation'], actions)
    for row, q, head, action in zip(real['rows'], values, heads, actions):
        row.update(critic=float(q), critic_bias=float(q)-row['real_mc_return'],
                   heads=head, action=action.tolist())
    return real


def run_probes(args, checkpoint, output):
    model, env, resolved = load_model(checkpoint, device=args.device)
    before = _outer_state_digest(model)
    bank = load_bank(args.bank, resolved['environment'])
    objective = PriorRewardObjective(model, reduction='mean_all', seed=55)
    partner, partner_env, _ = load_model(Path(args.partner), device=args.device)
    partner_before = _outer_state_digest(partner)
    other = PriorRewardObjective(partner, reduction='mean_all', seed=55)
    envs = [_make_env(resolved) for _ in range(args.rollouts)]
    started = time.perf_counter()
    results = []
    try:
        roots = bank['roots'][:args.root_limit] if args.root_limit else bank['roots']
        for root in roots:
            root_start = time.perf_counter()
            obs = np.asarray(root['observation'], np.float32)
            seed = solver_seed(55, 'dog-diagnostic-noise', root['root_id'])
            noise = np.random.default_rng(seed).standard_normal(
                (3+args.tail_steps, args.rollouts, model.cfg.action_dim)).astype(np.float32)
            gradient_noise = np.random.default_rng(seed+1).standard_normal(
                (4, 32, model.cfg.action_dim)).astype(np.float32)
            with frozen_parameters(objective.model):
                gradient = probe_root(objective, envs, SimulatorSnapshot.from_dict(root['snapshot']),
                    obs, np.asarray(root['anchor_pretanh_action'], np.float32), horizon=3,
                    gradient_noise=gradient_noise, evaluation_noise=noise, epsilons=[.1],
                    random_directions=1, direction_seed=seed+2,
                    original_remaining_steps=500-root['decision_index'])
            cross_actions = []
            for label, source in [('self', objective), ('partner', other)]:
                action = np.tanh(source.anchor(obs))
                measured = measure_first_actions(objective, envs, root,
                    np.repeat(action[None], args.rollouts, axis=0), noise)
                cross_actions.append(dict(action_source=label, **measured))
            model.agent.inner_engine.reset_for_evaluation(seed, reuse_action_pool=True)
            rounds = list(range(int(model.cfg.inner_rounds)+1))
            trace = InnerActionTrace(capture_actors=True, actor_rounds=rounds,
                probes=True, probe_mode='outer_tail', probe_rollouts=32,
                probe_horizon=3, probe_seed=seed+3)
            model.predict(obs, deterministic=False, episode_start=True, trace=trace)
            if [a.round_index for a in trace.actor_snapshots] != rounds:
                raise RuntimeError('Incomplete inner-round actor capture.')
            round_rows, open_loop = [], []
            for actor in trace.actor_snapshots:
                policy = actor.make_policy(model.agent.device)
                callback = FrozenCallbacks(model, policy, actor.policy_bounds)
                actions = callback.actor(np.repeat(obs[None], args.rollouts, axis=0), noise[0])
                measured = measure_first_actions(objective, envs, root, actions, noise)
                round_rows.append(dict(round=actor.round_index, actor_sha256=actor.sha256,
                    actor_updates=actor.actor_updates, critic_updates=actor.critic_updates, **measured))
                if actor.round_index in (0, rounds[-1]):
                    sequence = imagined_actions(objective, policy, actor.policy_bounds, obs, noise, 3)
                    compared = same_actions(objective, envs, SimulatorSnapshot.from_dict(root['snapshot']),
                        obs, sequence, noise, 500-root['decision_index'])
                    open_loop.append(dict(round=actor.round_index, **compared))
                del policy
            result = dict(root_id=root['root_id'], episode_id=root['episode_id'],
                source=root['source'], controller=root['controller'], noise_seed=seed,
                gradient=gradient, cross_actions=cross_actions, rounds=round_rows,
                same_action_sequences=open_loop, trace_events=trace.events,
                seconds=time.perf_counter()-root_start)
            atomic_json(output / (root['root_id']+'.json'), result)
            results.append(result)
            print(json.dumps(dict(event='root', label=args.label, root=root['root_id'],
                                  seconds=result['seconds'])), flush=True)
        if before != _outer_state_digest(model) or partner_before != _outer_state_digest(partner):
            raise RuntimeError('Frozen outer model/optimizer/scale changed during probes.')
        result = dict(schema=1, complete=True, label=args.label, checkpoint=str(checkpoint),
            checkpoint_sha256=_file_sha256(checkpoint), partner=str(args.partner),
            partner_sha256=_file_sha256(args.partner), bank_id=bank['id'], resolved=resolved,
            outer_state_before=before, outer_state_after=_outer_state_digest(model),
            diagnostic_q_reduction='mean_all (expectation of actor mean_pair)',
            continuation='frozen checkpoint prior; sampled; no final Q bootstrap',
            tail_steps=args.tail_steps, rollouts=args.rollouts, roots=results,
            total_seconds=time.perf_counter()-started)
        atomic_json(output / 'probes.json', result)
        return result
    finally:
        for branch in envs:
            branch.close()
        _close_resources(partner, partner_env)
        _close_resources(model, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['episodes', 'probes', 'bank'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label')
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 102, 103])
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--decisions', type=int, nargs='+', default=[200])
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--partner', type=Path)
    parser.add_argument('--episodes-dir', type=Path)
    parser.add_argument('--rollouts', type=int, default=2)
    parser.add_argument('--tail-steps', type=int, default=1000)
    parser.add_argument('--root-limit', type=int)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.mode == 'bank':
        make_bank(args.episodes_dir, args.output)
        return
    if not args.checkpoint or not args.label:
        parser.error('checkpoint and label are required')
    if not 0 < args.steps <= 500 or min(args.rollouts, args.tail_steps, args.threads) < 1:
        parser.error('invalid work budget')
    if args.mode == 'probes' and (args.bank is None or args.partner is None):
        parser.error('probes require bank and partner')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    atomic_json(args.output / 'request.json', {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    if args.mode == 'episodes':
        run_episodes(args, args.checkpoint, args.output)
    else:
        run_probes(args, args.checkpoint, args.output)


if __name__ == '__main__':
    main()
