"""Separate continuing-simulator calibration of frozen, captured inner policies.

No training job or W&B session starts implicitly. ``run`` writes an immutable
local diagnostic bundle; ``publish`` is a separate, explicit CPU operation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _jsonable, _make_env,
    _outer_state_digest, _seed_spaces, _validate_checkpoint_contract,
    _validate_frozen_selection,
)
from utils.ambi_benchmark import atomic_json, canonical_hash, capture_root, read_json, solver_seed
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset
from utils.checkpoint_context import load_checkpoint_context

DEFAULT_MATRIX = Path(__file__).parent / 'configs/research/ambi_scratch_togo_reference.json'


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return value


def calibration_options(matrix, cfg, **overrides):
    """Resolve coverage independently of the learning and evaluation budgets."""
    options = copy.deepcopy(matrix.get('real_calibration', {}))
    coverage = ('decisions', 'every_n', 'every_decision')
    if any(overrides.get(key) is not None for key in coverage):
        for key in coverage:
            options.pop(key, None)
    options.update({key: value for key, value in overrides.items() if value is not None})
    defaults = dict(solver_repetitions=3, rollout_repetitions=4, tail_steps=1000,
                    bootstrap_resamples=2000, reward_bound=2.0, max_steps=500,
                    model_probe_rollouts=32, benchmark_repetitions=0)
    for key, value in defaults.items():
        options.setdefault(key, value)
    for key in ('solver_repetitions', 'rollout_repetitions', 'tail_steps', 'bootstrap_resamples', 'max_steps'):
        _positive(options[key], key)
    for key in ('model_probe_rollouts', 'benchmark_repetitions'):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer.')
    active = [key for key in coverage if options.get(key) is not None and options.get(key) is not False]
    if len(active) > 1:
        raise ValueError('Select only one of decisions, every_n, or every_decision.')
    if options.get('every_decision'):
        decisions = list(range(options['max_steps']))
    elif options.get('every_n') is not None:
        decisions = list(range(0, options['max_steps'], _positive(options['every_n'], 'every_n')))
    else:
        decisions = options.get('decisions', [0, 100, 200, 300, 400])
    if (not isinstance(decisions, list) or not decisions or len(set(decisions)) != len(decisions)
            or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < options['max_steps'] for x in decisions)):
        raise ValueError('Decisions must be unique integer indices within the original episode cutoff.')
    options['decisions'] = sorted(decisions)
    rounds = options.get('rounds', list(range(int(cfg.inner_rounds) + 1)))
    if (not isinstance(rounds, list) or not rounds or len(set(rounds)) != len(rounds)
            or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= int(cfg.inner_rounds) for x in rounds)):
        raise ValueError('Rounds must select initialization (0) or completed rounds up to inner_rounds.')
    options['rounds'] = sorted(rounds)
    horizon = int(getattr(cfg, 'inner_rollout_horizon', 3))
    if horizon + options['tail_steps'] < options['max_steps'] - min(options['decisions']):
        raise ValueError('The continuation is too short to reach the original episode cutoff. '
                         'Increase tail_steps, shorten max_steps, or select later roots.')
    if not math.isfinite(options['reward_bound']) or options['reward_bound'] <= 0:
        raise ValueError('reward_bound must be finite and positive.')
    return options


def _science_identity():
    """Pin calibration semantics as well as the underlying model implementation."""
    root = Path(__file__).parent
    files = [Path(__file__), root / 'evaluate_ambi_checkpoint.py', root / 'domains/dmcontrol.py',
             root / 'utils/ambi_real_calibration.py', root / 'utils/ambi_diagnostic_series.py',
             root / 'RL/AMBITDMPC2.py', *sorted((root / 'RL/tdmpc2_core').rglob('*.py'))]
    return canonical_hash({str(path.relative_to(root)): _file_sha256(path) for path in files})


def _synchronize(model):
    if model.agent.device.type == 'cuda':
        torch.cuda.synchronize(model.agent.device)


class FrozenCallbacks:
    """Batch new simulator observations through the frozen encoder at every step."""
    def __init__(self, model, policy=None, bounds=None, pair_indices=None):
        self.model = model.agent.model
        self.device = model.agent.device
        self.policy = policy
        self.bounds = bounds or {}
        self.reduction = model.cfg.mppi_terminal_q_reduction
        self.pair_indices = pair_indices

    @torch.no_grad()
    def actor(self, observations, noise):
        z = self.model.encode(torch.as_tensor(observations, device=self.device, dtype=torch.float32))
        actions, _ = self.model.pi(z, policy=self.policy,
            noise=torch.as_tensor(noise, device=self.device, dtype=z.dtype), **self.bounds)
        return actions.cpu().numpy()

    @torch.no_grad()
    def q(self, observations, actions):
        from RL.tdmpc2_core.inner_trace import evaluate_frozen_outer_q
        z = self.model.encode(torch.as_tensor(observations, device=self.device, dtype=torch.float32))
        value = evaluate_frozen_outer_q(self.model, z,
            torch.as_tensor(actions, device=self.device, dtype=z.dtype),
            reduction=self.reduction, pair_indices=self.pair_indices)
        return value.reshape(-1).cpu().numpy()


def _bank_protocol(resolved, seeds, controller_seed, options, science):
    return dict(version=1, environment=resolved['environment'],
        env_wrappers=resolved['algorithm_config'].get('env_wrappers', []),
        env_wrapper=resolved['algorithm_config'].get('env_wrapper'),
        seeds=seeds, controller_seed=controller_seed, action_rule='tanh_mean',
        max_steps=options['max_steps'], decisions=options['decisions'], science= science)


def collect_root_bank(env, model, *, checkpoint_sha256, protocol):
    from utils.ambi_real_calibration import capture_simulator_snapshot
    roots, episodes = [], []
    callback = FrozenCallbacks(model)
    started = time.perf_counter()
    for seed in protocol['seeds']:
        _seed_spaces(env, seed)
        obs, _ = env.reset(seed=seed)
        total = 0.0
        for decision in range(protocol['max_steps']):
            if decision in protocol['decisions']:
                root = capture_root(obs, seed, decision, total)
                root['snapshot'] = capture_simulator_snapshot(env).to_dict()
                roots.append(root)
            action = callback.actor(np.asarray(obs)[None], np.zeros((1, model.cfg.action_dim), np.float32))[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        episodes.append({'episode_id': f'seed-{seed}', 'length': decision + 1, 'return': total})
    expected = {(f'seed-{seed}', decision) for seed in protocol['seeds'] for decision in protocol['decisions']}
    actual = {(root['episode_id'], root['decision_index']) for root in roots}
    if actual != expected:
        raise ValueError('Source episodes ended before every requested root was captured; choose valid coverage.')
    bank = dict(schema_version=1, kind='humanoid_integration_state_bank', complete=True,
                checkpoint_sha256=checkpoint_sha256, protocol=protocol, roots=roots, episodes=episodes)
    bank['id'] = canonical_hash(bank)
    return bank, dict(root_collection_seconds=time.perf_counter() - started,
                      root_collection_simulator_decisions=sum(row['length'] for row in episodes))


def load_root_bank(path, checkpoint_sha256, protocol):
    from utils.ambi_real_calibration import SimulatorSnapshot
    bank = read_json(path)
    if (bank.get('kind') != 'humanoid_integration_state_bank' or bank.get('schema_version') != 1
            or not bank.get('complete') or bank.get('id') != canonical_hash({k:v for k,v in bank.items() if k != 'id'})
            or bank.get('checkpoint_sha256') != checkpoint_sha256 or bank.get('protocol') != protocol):
        raise ValueError('Simulator bank is incomplete, corrupted, or incompatible with this checkpoint/root protocol.')
    expected = {(f'seed-{seed}', decision) for seed in protocol['seeds'] for decision in protocol['decisions']}
    actual = [(row['episode_id'], row['decision_index']) for row in bank['roots']]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Simulator bank is missing requested roots or contains duplicates.')
    for root in bank['roots']:
        state = SimulatorSnapshot.from_dict(root['snapshot']).state()['environment']
        seed, decision = root['seed'], root['decision_index']
        observation = np.asarray(root['observation'], dtype=np.float32)
        if (root['episode_id'] != f'seed-{seed}' or root['root_id'] != f'seed-{seed}-decision-{decision}'
                or root['dtype'] != 'float32' or list(observation.shape) != root['shape']
                or observation.ndim != 1 or not np.isfinite(observation).all()
                or not np.array_equal(observation, state['observation'])
                or state['step_count'] != state['runtime']['action_repeat'] * decision):
            raise ValueError('Simulator bank root metadata or observation differs from its integration snapshot.')
    return bank


def paired_noise(controller_seed, root_id, *, horizon, tail_steps, rollouts, action_dim):
    seed = solver_seed(controller_seed, 'real-calibration-noise', root_id)
    rng = np.random.default_rng(seed)
    prefix = rng.standard_normal((horizon, rollouts, action_dim)).astype(np.float32)
    tail = rng.standard_normal((tail_steps, rollouts, action_dim)).astype(np.float32)
    return prefix, tail, seed


def _pair_indices(model, seed):
    if not model.cfg.mppi_terminal_q_reduction.endswith('_pair'):
        return None
    generator = torch.Generator(device=model.agent.device).manual_seed(seed)
    return model.agent.model.q_backend.sample_pair_indices(model.agent.device, generator=generator)


@torch.no_grad()
def _model_branches(model, observation, policy, bounds, prefix_noise, tail_noise, pair_indices):
    from RL.tdmpc2_core.inner_trace import evaluate_outer_tail
    z = model.agent.model.encode(torch.as_tensor(observation, device=model.agent.device, dtype=torch.float32)[None])
    noise = torch.as_tensor(np.concatenate((prefix_noise, tail_noise[:1])), device=z.device, dtype=z.dtype)
    scores = evaluate_outer_tail(model.agent.inner_engine, z, policy,
        noise=noise, pair_indices=pair_indices, policy_bounds=bounds)
    return [dict(model_prefix_reward=float(r), model_bootstrap=float(b), model_return=float(t))
            for r, b, t in zip(scores['reward'].reshape(-1).cpu(), scores['bootstrap'].reshape(-1).cpu(),
                               scores['total'].reshape(-1).cpu())]


def _cache_reference(path, identity, evaluate):
    """Reuse only complete, content-verified prior continuations with identical semantics."""
    key = canonical_hash(identity)
    path = Path(path) / f'{key}.json'
    if path.exists():
        record = read_json(path)
        if (record.get('identity') != identity or record.get('complete') is not True
                or record.get('sha256') != canonical_hash(record['result'])):
            raise ValueError(f'Incompatible or corrupted prior reference: {path}')
        return record['result'], True, path
    result = evaluate()
    atomic_json(path, dict(identity=identity, complete=True, result=result, sha256=canonical_hash(result)))
    return result, False, path


def _benchmark_model_probes(model, root, seed, rollouts, repetitions):
    from RL.tdmpc2_core.inner_trace import InnerActionTrace
    if repetitions == 0:
        return []
    rows = []
    for repeat in range(repetitions + 1):
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            model.agent.inner_engine.reset_for_evaluation(seed, reuse_action_pool=True)
            trace = InnerActionTrace(probes=enabled, probe_mode='outer_tail', probe_rollouts=rollouts or 32,
                probe_horizon=model.cfg.inner_rollout_horizon, probe_seed=seed)
            _synchronize(model)
            start = time.perf_counter()
            action, _ = model.predict(np.asarray(root['observation'], np.float32), deterministic=True,
                                      episode_start=True, trace=trace)
            _synchronize(model)
            elapsed = time.perf_counter() - start
            if repeat:
                rows.append(dict(repetition=repeat - 1, model_probes=enabled,
                                 elapsed_seconds=elapsed, action=np.asarray(action).tolist()))
    for repeat in range(repetitions):
        paired = [row for row in rows if row['repetition'] == repeat]
        if paired[0]['action'] != paired[1]['action']:
            raise RuntimeError('Probe benchmark changed controller actions.')
    return rows


def run_calibration(matrix_path, checkpoint, *, preset=None, bundle_dir, attempt_label,
                    root_bank=None, save_root_bank=None, reference_cache=None, metadata=None,
                    device=None, seeds=None, controller_seed=None, **overrides):
    from RL.tdmpc2_core.inner_trace import InnerActionTrace
    from utils.ambi_real_calibration import (
        SimulatorSnapshot, enable_continuing_calibration, evaluate_real_branches,
    )
    from utils.ambi_diagnostic_series import build_diagnostic_record, write_diagnostic_bundle, read_diagnostic_bundle
    if not attempt_label or not attempt_label.strip():
        raise ValueError('An explicit nonempty attempt label is required.')
    bundle_dir = Path(bundle_dir).resolve()
    work = bundle_dir.with_name(bundle_dir.name + '.work')
    if bundle_dir.exists() or work.exists():
        raise FileExistsError('Choose a fresh bundle directory and attempt; existing results are immutable.')
    if root_bank and save_root_bank:
        raise ValueError('Load an existing root bank or save a new one, not both.')
    if save_root_bank and Path(save_root_bank).exists():
        raise FileExistsError(f'Root bank already exists: {save_root_bank}')
    matrix_path, checkpoint = Path(matrix_path).resolve(), Path(checkpoint).resolve()
    matrix = load_preset_matrix(matrix_path)
    selectors = normalize_selectors(matrix, [preset] if preset else None)
    if len(selectors) != 1:
        raise ValueError('Real calibration requires one explicitly selected inner setting per bundle.')
    context = load_checkpoint_context(checkpoint, metadata_path=metadata)
    resolved = resolve_preset(matrix_path, selectors[0], matrix, checkpoint_context=context)
    _validate_frozen_selection(matrix, [resolved])
    _validate_checkpoint_contract(matrix, checkpoint, context, [resolved])
    params = resolved['algorithm_config']['alg_params']
    if (resolved['algorithm_config']['alg'] != 'AMBITDMPC2/AMBITDMPC2'
            or params.get('inner_operator', 'sac') != 'sac'
            or params.get('obs', 'state') != 'state'
            or params.get('inner_rounds', 0) < 1):
        raise ValueError('Calibration requires an active state-observation AMBI SAC solve.')
    if any(params.get(key, 0) for key in ('inner_actor_writeback_coef', 'inner_critic_writeback_coef')):
        raise ValueError('Calibration requires frozen outer priors with writeback disabled.')
    evaluation = matrix.get('evaluation', {})
    seeds = list(seeds if seeds is not None else evaluation.get('seeds', [101,102,103,104,105]))
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError('Seeds must be distinct NumPy seed integers.')
    controller_seed = evaluation.get('controller_seed', 55) if controller_seed is None else controller_seed
    if isinstance(controller_seed, bool) or not isinstance(controller_seed, int) or not 0 <= controller_seed < 2**32:
        raise ValueError('controller_seed must be a NumPy seed integer.')
    if overrides.get('max_steps') is None:
        overrides['max_steps'] = evaluation.get('max_steps', 500)
    if overrides.get('model_probe_rollouts') is None:
        overrides['model_probe_rollouts'] = evaluation.get('togo_return_rollouts', 32)
    env, model, branch_envs = None, None, []
    error = None
    started = time.perf_counter()
    work.mkdir(parents=True)
    rows, artifacts = [], {'matrix.json': matrix_path, 'checkpoint.metadata.json': context.source}
    timing = dict(optimization_seconds=0.0, model_probe_seconds=0.0, simulator_seconds=0.0,
                  serialization_seconds=0.0, publication_seconds=0.0, frozen_policy_seconds=0.0,
                  q_seconds=0.0, real_calibration_seconds=0.0)
    try:
        env = _make_env(resolved)
        model, _ = _initialize_frozen_model(resolved, env, checkpoint, controller_seed, device=device)
        for component in ('actor', 'critic', 'temperature', 'replay', 'actor_optimizer',
                          'critic_optimizer', 'temperature_optimizer'):
            if getattr(model.cfg, f'inner_{component}_scope') != 'action':
                raise ValueError('Independent calibration solves require every inner state scope to be action-local.')
        if model.cfg.inner_explorer_mode != 'none':
            raise ValueError('Captured-policy calibration currently requires a single inner actor.')
        options = calibration_options(matrix, model.cfg, **overrides)
        # Integration snapshots intentionally support only this continuing task.
        from utils.ambi_real_calibration import capture_simulator_snapshot
        env.reset(seed=seeds[0])
        capture_simulator_snapshot(env)
        if model.cfg.outer_critic_target != 'reward_only':
            raise ValueError('Real reward calibration requires a reward-only outer critic.')
        if not 0 < float(model.agent.discount) < 1:
            raise ValueError('Continuing-return calibration requires 0 < checkpoint discount < 1.')
        expected_scale = matrix.get('checkpoint_contract', {}).get('saved_q_scale')
        if expected_scale is not None and not math.isclose(float(model.agent.actor_loss_scale), expected_scale, rel_tol=1e-7):
            raise ValueError('Loaded Q scale does not match the pinned reference.')
        frozen_digest = _outer_state_digest(model)
        checkpoint_hash = _file_sha256(checkpoint)
        science = _science_identity()
        protocol = _bank_protocol(resolved, seeds, controller_seed, options, science)
        if root_bank:
            bank = load_root_bank(root_bank, checkpoint_hash, protocol)
            timing.update(root_collection_seconds=0.0, root_collection_simulator_decisions=0)
            bank_path = Path(root_bank)
        else:
            bank, collection_timing = collect_root_bank(env, model, checkpoint_sha256=checkpoint_hash, protocol=protocol)
            timing.update(collection_timing)
            bank_path = Path(save_root_bank) if save_root_bank else work / 'root-bank.json'
            atomic_json(bank_path, bank)
        artifacts['root-bank.json'] = bank_path
        horizon, discount = int(model.cfg.inner_rollout_horizon), float(model.agent.discount)
        cache = Path(reference_cache) if reference_cache else work / 'prior-references'
        for _ in range(options['rollout_repetitions']):
            branch = _make_env(resolved)
            branch_envs.append(branch)
            branch.reset(seed=seeds[0])
            enable_continuing_calibration(branch)
        _synchronize(model)
        warm_start = time.perf_counter()
        model.predict(np.asarray(bank['roots'][0]['observation'], np.float32), deterministic=True, episode_start=True)
        _synchronize(model)
        timing['warmup_seconds'] = time.perf_counter() - warm_start
        before = time.perf_counter()
        benchmarks = _benchmark_model_probes(model, bank['roots'][0],
            solver_seed(controller_seed, 'timing'), options['model_probe_rollouts'], options['benchmark_repetitions'])
        timing['benchmark_seconds'] = time.perf_counter() - before if benchmarks else 0.0
        if benchmarks:
            atomic_json(work / 'timing-benchmark.json', benchmarks)
            artifacts['timing-benchmark.json'] = work / 'timing-benchmark.json'
        cache_hits = 0
        for root in bank['roots']:
            snapshot = SimulatorSnapshot.from_dict(root['snapshot'])
            prefix_noise, tail_noise, noise_seed = paired_noise(controller_seed, root['root_id'],
                horizon=horizon, tail_steps=options['tail_steps'], rollouts=options['rollout_repetitions'],
                action_dim=model.cfg.action_dim)
            pair_seed = solver_seed(controller_seed, 'real-calibration-q-pair', root['root_id'])
            indices = _pair_indices(model, pair_seed)
            prior = FrozenCallbacks(model, pair_indices=indices)
            real_kwargs = dict(prefix_noise=prefix_noise, tail_noise=tail_noise, discount=discount,
                horizon=horizon, original_remaining_steps=options['max_steps'] - root['decision_index'],
                reward_bound=options['reward_bound'])
            def measure(policy, bounds):
                before = time.perf_counter()
                predicted = _model_branches(model, root['observation'], policy, bounds,
                                            prefix_noise, tail_noise, indices)
                model_seconds = time.perf_counter() - before
                actor = FrozenCallbacks(model, policy, bounds, indices)
                real = evaluate_real_branches(branch_envs, snapshot, actor.actor, prior.actor, prior.q, **real_kwargs)
                if any(not row['mc_complete'] for row in real['rows']):
                    raise RuntimeError('A continuing branch unexpectedly truncated; calibration is incomplete.')
                return dict(model_rows=predicted, real=real, model_seconds=model_seconds,
                            model_work={'transition_rows': horizon * len(predicted),
                                        'policy_rows': (horizon + 1) * len(predicted),
                                        'q_rows': len(predicted)})
            reference_identity = dict(version=1, checkpoint=checkpoint_hash, bank_id=bank['id'],
                root_id=root['root_id'], snapshot_hash=canonical_hash(root['snapshot']), science=science,
                noise_seed=noise_seed, q_pair_seed=pair_seed, horizon=horizon, tail_steps=options['tail_steps'],
                rollouts=options['rollout_repetitions'], discount=discount, reward_bound=options['reward_bound'],
                q_reduction=model.cfg.mppi_terminal_q_reduction,
                outer_policy=dict(log_std_mapping=model.agent.model._log_std_mapping,
                    log_std_min=model.agent.model._log_std_min_value,
                    log_std_max=model.agent.model._log_std_max_value))
            reference, hit, reference_path = _cache_reference(cache, reference_identity,
                                                             lambda: measure(model.agent.model._pi, {}))
            if (len(reference['model_rows']) != options['rollout_repetitions']
                    or len(reference['real']['rows']) != options['rollout_repetitions']
                    or any(row['rollout_index'] != index or not row['mc_complete']
                           or not row['episode_cutoff_complete']
                           for index, row in enumerate(reference['real']['rows']))):
                raise ValueError('Prior reference does not contain the complete requested rollout panel.')
            cache_hits += int(hit)
            artifacts[f"references/{root['root_id']}.json"] = reference_path
            if not hit:
                _accumulate_timing(timing, reference)
            for repeat in range(options['solver_repetitions']):
                solve_seed = solver_seed(controller_seed, 'real-calibration-solve', root['root_id'], repeat)
                model.agent.inner_engine.reset_for_evaluation(solve_seed, reuse_action_pool=True)
                trace = InnerActionTrace(probes=bool(options['model_probe_rollouts']), probe_mode='outer_tail',
                    probe_rollouts=options['model_probe_rollouts'] or 32, probe_horizon=horizon,
                    probe_seed=solver_seed(controller_seed, 'real-model-probe', root['root_id']),
                    capture_actors=True, actor_rounds=options['rounds'])
                before = time.perf_counter()
                model.predict(np.asarray(root['observation'], np.float32), deterministic=True,
                              episode_start=True, trace=trace)
                _synchronize(model)
                elapsed = time.perf_counter() - before
                probe_seconds = sum(event['metrics'].get('probe_seconds', 0) for event in trace.events)
                snapshot_seconds = sum(event['metrics'].get('actor_snapshot_seconds', 0) for event in trace.events)
                timing['optimization_seconds'] += max(0.0, elapsed - probe_seconds - snapshot_seconds)
                timing['model_probe_seconds'] += probe_seconds
                timing['serialization_seconds'] += snapshot_seconds
                for metric, total in [('probe_model_steps', 'model_transition_rows'),
                                      ('probe_policy_evaluations', 'model_policy_rows'),
                                      ('probe_q_evaluations', 'model_q_rows')]:
                    timing[total] = timing.get(total, 0) + sum(
                        int(event['metrics'].get(metric, 0)) for event in trace.events)
                probe_rows = [dict(episode_id=root['episode_id'], root_id=root['root_id'], solver_repeat=repeat, **event)
                              for event in trace.events if event['phase'] == 'probe']
                if [actor.round_index for actor in trace.actor_snapshots] != options['rounds']:
                    raise RuntimeError('The solve did not capture every requested actor boundary.')
                solve_rows = []
                for actor_snapshot in trace.actor_snapshots:
                    before = time.perf_counter()
                    policy = actor_snapshot.make_policy(model.agent.device)
                    _synchronize(model)
                    timing['serialization_seconds'] += time.perf_counter() - before
                    measured = measure(policy, actor_snapshot.policy_bounds)
                    _accumulate_timing(timing, measured)
                    for rollout, (predicted, real) in enumerate(zip(measured['model_rows'], measured['real']['rows'])):
                        ref_pred = reference['model_rows'][rollout]
                        ref_real = reference['real']['rows'][rollout]
                        metrics = _branch_metrics(predicted, real, ref_pred, ref_real)
                        row = dict(episode_id=root['episode_id'], root_id=root['root_id'],
                            decision_index=root['decision_index'], solver_repeat=repeat, rollout_repeat=rollout,
                            round_index=actor_snapshot.round_index, actor_updates=actor_snapshot.actor_updates,
                            critic_updates=actor_snapshot.critic_updates, solver_seed=solve_seed,
                            actor_sha256=actor_snapshot.sha256,
                            policy_noise_seed=noise_seed, q_pair_seed=pair_seed,
                            endpoint_action=real['endpoint_action'],
                            terminated=real['terminated'], truncated=real['truncated'],
                            mc_complete=real['mc_complete'], metrics=metrics)
                        solve_rows.append(row)
                    del policy
                before = time.perf_counter()
                shard_path = work / f"{root['root_id']}-solver-{repeat}.json"
                atomic_json(shard_path, dict(rows=solve_rows, model_probes=probe_rows,
                                            trace_events=trace.events))
                artifacts[f"solves/{shard_path.name}"] = shard_path
                timing['serialization_seconds'] += time.perf_counter() - before
                rows.extend(solve_rows)
                del trace
        if _outer_state_digest(model) != frozen_digest:
            raise RuntimeError('Calibration changed frozen outer weights or optimizer state.')
        identity = dict(checkpoint=dict(sha256=checkpoint_hash, step=context.metadata['checkpoint']['step']),
            backbone=matrix.get('source_run'), setting=_jsonable(vars(model.cfg)), attempt=attempt_label,
            scope='common_prior_roots', code={'source_sha256': science}, protocol=dict(root_bank_id=bank['id'], root_protocol=protocol,
                **options, horizon=horizon, discount=discount, action_rule='sampled',
                tail_actor='outer', tail_critic='outer_online', q_reduction=model.cfg.mppi_terminal_q_reduction,
                q_pair_rule='private_fixed_per_root', tail_bootstrap=False, entropy_bonus=False,
                actor_snapshot_rule='fixed_policy_for_H_real_steps_then_prior'))
        expected = dict(roots=[dict(episode_id=r['episode_id'], root_id=r['root_id']) for r in bank['roots']],
            solver_repeats=options['solver_repetitions'], rollout_repeats=options['rollout_repetitions'],
            rounds=options['rounds'])
        timing.update(total_elapsed_seconds=time.perf_counter() - started, prior_reference_cache_hits=cache_hits,
                      prior_reference_count=len(bank['roots']), outer_state_unchanged=True)
        record = build_diagnostic_record(identity, rows, expected, timing=timing,
            bootstrap_resamples=options['bootstrap_resamples'], bootstrap_seed=solver_seed(controller_seed, 'intervals'))
        write_diagnostic_bundle(bundle_dir, record, artifact_files=artifacts,
                                elapsed_before_bundle=time.perf_counter() - started)
        record = read_diagnostic_bundle(bundle_dir)
        atomic_json(work / 'status.json', dict(status='complete', bundle=str(bundle_dir)))
        return record
    except BaseException as exc:
        error = exc
        atomic_json(work / 'status.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                    completed_rows=len(rows), timing=timing))
        raise
    finally:
        _close_resources(model, env, *branch_envs, primary_error=error)


def _accumulate_timing(timing, measured):
    timing['model_probe_seconds'] += measured['model_seconds']
    values = measured['real']['timing']
    timing['real_calibration_seconds'] += values['total_seconds']
    for source, target in [('simulator_seconds','simulator_seconds'), ('policy_seconds','frozen_policy_seconds'),
                           ('q_seconds','q_seconds'), ('serialization_seconds','serialization_seconds')]:
        timing[target] += values.get(source, 0.0)
    timing['branch_simulator_decisions'] = timing.get('branch_simulator_decisions', 0) + measured['real']['work']['simulator_decisions']
    for name in ('policy_rows', 'q_rows'):
        timing[f'real_{name}'] = timing.get(f'real_{name}', 0) + measured['real']['work'][name]
    for name, count in measured['model_work'].items():
        timing[f'model_{name}'] = timing.get(f'model_{name}', 0) + count


def _branch_metrics(predicted, real, ref_pred, ref_real):
    if not real['episode_cutoff_complete'] or not real['mc_complete']:
        raise ValueError('A branch did not complete its requested continuation and episode cutoff.')
    metrics = {**predicted, **{key: float(value) for key, value in real.items()
                              if key != 'rollout_index' and isinstance(value, (int, float, np.number))
                              and not isinstance(value, bool)}}
    for key, value in ref_pred.items():
        metrics[f'prior_{key}'] = value
    for key in ('real_mc_return', 'real_bootstrapped_return', 'real_prefix_reward', 'real_bootstrap',
                'episode_cutoff_discounted_return', 'episode_cutoff_undiscounted_return'):
        metrics[f'prior_{key}'] = ref_real[key]
    metrics.update(bootstrap_prediction_error=real['real_bootstrap'] - real['real_tail_contribution'],
                   total_prediction_error=predicted['model_return'] - real['real_mc_return'],
                   model_dynamics_prediction_error=predicted['model_return'] - real['real_bootstrapped_return'],
                   model_gain_vs_prior=predicted['model_return'] - ref_pred['model_return'],
                   real_bootstrapped_gain_vs_prior=real['real_bootstrapped_return'] - ref_real['real_bootstrapped_return'],
                   real_mc_gain_vs_prior=real['real_mc_return'] - ref_real['real_mc_return'],
                   episode_cutoff_complete=float(real['episode_cutoff_complete']))
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Create a fresh local simulator-calibration bundle.')
    run.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX)
    run.add_argument('--checkpoint', type=Path, required=True)
    run.add_argument('--metadata', type=Path)
    run.add_argument('--preset')
    run.add_argument('--bundle-dir', type=Path, required=True)
    run.add_argument('--attempt-label', required=True)
    run.add_argument('--root-bank', type=Path)
    run.add_argument('--save-root-bank', type=Path)
    run.add_argument('--reference-cache', type=Path)
    run.add_argument('--device')
    run.add_argument('--seeds', nargs='+', type=int)
    run.add_argument('--controller-seed', type=int)
    coverage = run.add_mutually_exclusive_group()
    coverage.add_argument('--decisions', nargs='+', type=int)
    coverage.add_argument('--every-n', type=int)
    coverage.add_argument('--every-decision', action='store_true', default=None)
    run.add_argument('--rounds', nargs='+', type=int)
    for name in ('solver-repetitions','rollout-repetitions','tail-steps','bootstrap-resamples',
                 'max-steps','model-probe-rollouts','benchmark-repetitions'):
        run.add_argument(f'--{name}', type=int)
    export = commands.add_parser('export-model', help='Create a separate diagnostic series from saved model probes.')
    export.add_argument('--bundle', type=Path, required=True)
    export.add_argument('--selector', required=True)
    export.add_argument('--attempt-label', required=True)
    export.add_argument('--output', type=Path, required=True)
    report = commands.add_parser('report', help='Render complete paired rows and summaries without evaluation.')
    report.add_argument('--bundle', type=Path, required=True)
    report.add_argument('--output', type=Path, required=True)
    publish = commands.add_parser('publish', help='Explicitly publish a diagnostic bundle; never a checkpoint curve.')
    publish.add_argument('--bundle', type=Path, required=True)
    publish.add_argument('--entity', default='rwgao_b-brown-university')
    publish.add_argument('--project', default='ambi-inner-bench')
    publish.add_argument('--mode', choices=['online','offline'], default='offline')
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    command = args.pop('command')
    if command == 'run':
        args['matrix_path'] = args.pop('matrix')
        record = run_calibration(**args)
        print(json.dumps({'status':record.get('status'), 'bundle':str(args['bundle_dir'])}))
    elif command == 'export-model':
        from utils.ambi_diagnostic_series import record_from_model_bundle, write_diagnostic_bundle, render_diagnostic_html
        record = record_from_model_bundle(args['bundle'], args['selector'], args['attempt_label'])
        write_diagnostic_bundle(args['output'], record)
    elif command == 'report':
        from utils.ambi_diagnostic_series import render_diagnostic_html
        render_diagnostic_html(args['bundle'], args['output'])
    else:
        from utils.ambi_diagnostic_series import publish_diagnostic_bundle
        publish_diagnostic_bundle(args.pop('bundle'), **args)


if __name__ == '__main__':
    main()
