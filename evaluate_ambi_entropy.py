"""Paired entropy interventions on an inherited actor and one frozen fitted critic.

Workers own complete source episodes, including all arms and solver repetitions.
They never initialize W&B. Publication validates the complete expected panel.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_ambi_calibration import (
    FrozenCallbacks, _accumulate_timing, _bank_protocol, _branch_metrics,
    _cache_reference, _model_branches, _pair_indices, _science_identity,
    _synchronize, collect_root_bank, paired_noise,
)
from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _jsonable,
    _make_env, _outer_state_digest,
)
from utils.ambi_benchmark import atomic_json, canonical_hash, read_json, solver_seed
from utils.ambi_research import resolve_preset
from utils.checkpoint_context import load_checkpoint_context


def load_study(path):
    study = read_json(path)
    if study.get('schema_version') != 1 or study.get('kind') != 'prior_entropy_intervention':
        raise ValueError('Unsupported entropy study.')
    for name in ('episode_seeds', 'decisions', 'actor_updates'):
        values = study[name]
        if (not values or len(set(values)) != len(values)
                or any(type(x) is not int or x < 0 for x in values)):
            raise ValueError(f'Invalid unique integer coverage: {name}.')
    if study['actor_updates'][0] != 0 or sorted(study['actor_updates']) != study['actor_updates']:
        raise ValueError('Actor update schedule must start at actual initialization and increase.')
    for name in ('rollouts', 'batch_size', 'critic_updates', 'solver_repetitions',
                 'rollout_repetitions', 'tail_steps', 'model_probe_rollouts', 'bootstrap_resamples'):
        if type(study[name]) is not int or study[name] < 1:
            raise ValueError(f'{name} must be positive.')
    if study.get('horizon') != 1 or study.get('max_steps') != 500:
        raise ValueError('This experiment requires H1 and the original 500-decision cutoff.')
    if max(study['decisions']) >= 500 or study['tail_steps'] + 1 < 500 - min(study['decisions']):
        raise ValueError('Continuation must reach the original cutoff for every root.')
    if study.get('prefix_action_rules') != ['sampled', 'mean']:
        raise ValueError('Both sampled and mean prefixes are required.')
    identities = [c['id'] for c in study['checkpoints']]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate checkpoint identity.')
    return study


def checkpoint_matrix(study, source, context):
    """Change only the explicit inner protocol; preserve outer/checkpoint semantics."""
    saved = context.trial_run_params['alg_params']
    actor_reduction = saved.get('outer_q_actor_reduction', 'min_pair')
    params = dict(
        inner_operator='sac', inner_rounds=1, inner_rollouts_per_round=study['rollouts'],
        inner_rollout_horizon=1, inner_batch_size=study['batch_size'],
        inner_replay_capacity=study['rollouts'], inner_replay_sampling='with_replacement',
        inner_critic_updates_per_round=study['critic_updates'],
        inner_actor_updates_per_round=max(study['actor_updates']),
        inner_temperature_updates_per_round=0, inner_updates_per_round=None,
        inner_iterations=None, inner_rollouts=None, inner_horizon=None,
        inner_updates_per_iteration=None, inner_model_step_budget=None,
        inner_critic_updates_per_action=None, inner_actor_updates_per_action=None,
        inner_temperature_updates_per_action=None, inner_steps_per_update=None,
        inner_actor_adaptation='clone', inner_critic_adaptation='clone',
        inner_actor_initialization='prior', inner_critic_initialization='prior',
        inner_critic_target_initialization='online', inner_actor_initial_std=None,
        inner_temperature_mode='fixed', inner_temperature_initialization='fixed',
        inner_temperature=0.0, inner_target_entropy='inherit_outer',
        inner_actor_entropy_mode='squashed', inner_bootstrap_source='inner_target',
        inner_finite_horizon=True, inner_update_timing='round',
        inner_behavior_action='policy_sample', inner_execution_action='mean',
        inner_actor_writeback_coef=0.0, inner_critic_writeback_coef=0.0,
        inner_log_std_mapping=None, inner_log_std_min=None, inner_log_std_max=None,
        inner_q_actor_reduction=actor_reduction,
        inner_q_target_reduction=saved.get('outer_q_target_reduction', 'min_pair'),
        inner_actor_loss_scale_update='per_action', inner_sac_critic_target='reward_only',
        inner_actor_lr=0.0003, inner_critic_lr=0.0003,
        inner_actor_adam_eps=1e-5, inner_adam_eps=1e-8,
        inner_actor_grad_clip_norm=20.0, inner_critic_grad_clip_norm=20.0,
        inner_critic_dropout_enabled=True, inner_critic_loss_coef=1.0,
        inner_critic_target_tau=0.01, inner_critic_target_update_interval=1,
        inner_outer_policy_kl_coef=0.0, inner_outer_action_l2_coef=0.0,
        inner_explorer_mode='none', inner_outer_replay_fraction=0.0,
        inner_diagnostic_rollouts=0,
    )
    for component in ('actor', 'critic', 'temperature', 'replay', 'actor_optimizer',
                      'critic_optimizer', 'temperature_optimizer'):
        params[f'inner_{component}_scope'] = 'action'
    return dict(schema_version=1, base_alg_config='checkpoint', source_run=source['source_run'],
                shared_alg_params=params, evaluation=dict(seeds=study['episode_seeds'], max_steps=500),
                comparisons={'entropy': {'reference': 'shared_fit', 'variants': {
                    'shared_fit': {'description': 'One shared dataset and fitted critic; actor-only entropy arms.',
                                   'alg_params': {}}}}})


def metric_semantics(metrics, *, reward_only):
    """Never label a soft-Q hybrid score as predicted environmental reward."""
    if reward_only:
        return metrics
    renamed = {}
    names = {
        'model_return': 'hybrid_model_score', 'model_bootstrap': 'hybrid_model_bootstrap',
        'real_endpoint_q': 'soft_endpoint_q', 'real_bootstrap': 'soft_bootstrap',
        'real_bootstrapped_return': 'hybrid_real_prefix_score',
        'bootstrap_prediction_error': 'soft_bootstrap_minus_measured_reward_tail',
        'total_prediction_error': 'hybrid_score_minus_real_reward_return',
        'model_dynamics_prediction_error': 'hybrid_model_minus_real_prefix_score',
        'model_gain_vs_prior': 'hybrid_model_gain_vs_prior',
        'real_bootstrapped_gain_vs_prior': 'hybrid_real_prefix_gain_vs_prior',
    }
    for key, value in metrics.items():
        if key.startswith('prior_'):
            name = 'prior_' + names.get(key[6:], key[6:])
        else:
            name = names.get(key, key)
        renamed[name] = value
    return renamed


def expected_rows(study, source, episode_seeds, arms):
    return [dict(source_run=source['source_run'], checkpoint_step=source['step'],
                 checkpoint_sha256=source['sha256'], prefix_action_rule=rule,
                 episode_seed=seed, root_id=f'seed-{seed}-decision-{decision}',
                 solver_repetition=solver, rollout_repetition=rollout, arm=arm,
                 actor_updates=updates)
            for seed in episode_seeds for decision in study['decisions']
            for solver in range(study['solver_repetitions']) for arm in arms
            for updates in study['actor_updates'] for rule in study['prefix_action_rules']
            for rollout in range(study['rollout_repetitions'])]


def state_digest(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        data = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((data.dtype, tuple(data.shape))).encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def critic_probe(model, critic, root_z, policy, bounds, noise, indices):
    z = root_z.expand(len(noise), -1)
    action, _ = model.agent.model.pi(z, policy=policy, noise=torch.as_tensor(noise, device=z.device), **bounds)
    modes = [(m, m.training) for m in critic.modules()]
    try:
        critic.eval()
        qs = getattr(critic, '_forward_eager', critic)
        q = model.agent.model.Q(z, action, qs=qs, reduction=model.cfg.inner_q_actor_reduction,
                               pair_indices=indices, trusted_pair_indices=True)
        return q.reshape(-1).cpu().numpy()
    finally:
        for module, mode in modes:
            module.training = mode


def run(study_path, checkpoint, checkpoint_index, episode_seed, output_dir, *, smoke=False, device='cuda'):
    from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
    from utils.ambi_entropy_probe import entropy_arms, run_root_entropy_probe
    from utils.ambi_entropy_reporting import aggregate_entropy_rows
    from utils.ambi_real_calibration import SimulatorSnapshot, enable_continuing_calibration, evaluate_real_branches

    started = time.perf_counter()
    original_study = load_study(study_path)
    study = copy.deepcopy(original_study)
    if type(checkpoint_index) is not int or not 0 <= checkpoint_index < len(study['checkpoints']):
        raise ValueError('Checkpoint index is outside the declared study.')
    source = study['checkpoints'][checkpoint_index]
    if episode_seed not in study['episode_seeds']:
        raise ValueError('Episode seed is outside the declared study.')
    if smoke:
        study.update(decisions=[400], solver_repetitions=1, rollout_repetitions=1,
                     tail_steps=101, model_probe_rollouts=8)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Results are immutable; choose a fresh output directory.')
    context = load_checkpoint_context(checkpoint)
    if (context.metadata['checkpoint']['step'] != source['step']
            or _file_sha256(checkpoint) != source['sha256']):
        raise ValueError('Checkpoint differs from pinned source.')
    if source.get('metadata_sha256') and _file_sha256(context.source) != source['metadata_sha256']:
        raise ValueError('Checkpoint sidecar differs from pinned source.')
    matrix = checkpoint_matrix(study, source, context)
    resolved = resolve_preset(study_path, 'entropy/shared_fit', matrix, checkpoint_context=context)
    resolved['algorithm_config']['alg_params']['mppi_terminal_q_reduction'] = (
        context.trial_run_params['alg_params'].get('outer_q_actor_reduction', 'min_pair'))
    # Eager kernels avoid recompilation for this small diagnostic; equations/optimizers unchanged.
    resolved['algorithm_config']['alg_params'].update(compile=False, compile_strict=False)
    output.mkdir(parents=True)
    atomic_json(output / 'study.json', study)
    atomic_json(output / 'matrix.json', matrix)
    atomic_json(output / 'checkpoint.metadata.json', context.metadata)
    env = model = None
    branch_envs, rows, fits = [], [], []
    error = None
    timing = dict(optimization_seconds=0., model_probe_seconds=0., simulator_seconds=0.,
                  serialization_seconds=0., publication_seconds=0., frozen_policy_seconds=0.,
                  q_seconds=0., real_calibration_seconds=0.)
    try:
        env = _make_env(resolved)
        model, _ = _initialize_frozen_model(resolved, env, checkpoint, study['controller_seed'], device=device)
        digest = _outer_state_digest(model)
        reward_only = model.cfg.outer_critic_target == 'reward_only'
        arms, arm_metadata = entropy_arms(model, include_gaussian=source.get('gaussian_control', False))
        arm_names = [arm['name'] for arm in arms]
        atomic_json(output / 'arms.json', {'arms': arms, 'metadata': arm_metadata})
        fit_engine = InnerImprovementEngine(model.agent)
        science = canonical_hash({'base': _science_identity(), 'evaluator': _file_sha256(__file__),
                                 'probe': _file_sha256(Path(__file__).parent / 'utils/ambi_entropy_probe.py')})
        protocol = _bank_protocol(resolved, [episode_seed], study['controller_seed'],
                                  dict(max_steps=500, decisions=study['decisions']), science)
        bank, collected = collect_root_bank(env, model, checkpoint_sha256=source['sha256'], protocol=protocol)
        timing.update(collected)
        atomic_json(output / 'root-bank.json', bank)
        for _ in range(study['rollout_repetitions']):
            branch = _make_env(resolved)
            branch_envs.append(branch)
            branch.reset(seed=episode_seed)
            enable_continuing_calibration(branch)
        discount = float(model.agent.discount)
        for root in bank['roots']:
            snapshot = SimulatorSnapshot.from_dict(root['snapshot'])
            indices = _pair_indices(model, solver_seed(study['controller_seed'], 'entropy-q-pair', root['root_id']))
            prior = FrozenCallbacks(model, pair_indices=indices)
            references, noises, measure_cache = {}, {}, {}
            for rule in study['prefix_action_rules']:
                prefix, tail, noise_seed = paired_noise(study['controller_seed'], root['root_id'], horizon=1,
                    tail_steps=study['tail_steps'], rollouts=study['rollout_repetitions'],
                    action_dim=model.cfg.action_dim, prefix_action_rule=rule)
                noises[rule] = prefix, tail, noise_seed

            def measure(policy, bounds, rule):
                prefix, tail, noise_seed = noises[rule]
                key = canonical_hash({'state': state_digest(policy), 'bounds': bounds, 'rule': rule})
                if key in measure_cache:
                    timing['actor_measurement_cache_hits'] = timing.get('actor_measurement_cache_hits', 0) + 1
                    return measure_cache[key]
                before = time.perf_counter()
                predicted = _model_branches(model, root['observation'], policy, bounds, prefix, tail, indices)
                _synchronize(model)
                model_seconds = time.perf_counter() - before
                actor = FrozenCallbacks(model, policy, bounds, indices)
                real = evaluate_real_branches(branch_envs, snapshot, actor.actor, prior.actor, prior.q,
                    prefix, tail, discount=discount, horizon=1,
                    original_remaining_steps=500-root['decision_index'], reward_bound=2.0)
                if any(not r['mc_complete'] or not r['episode_cutoff_complete'] for r in real['rows']):
                    raise RuntimeError('Incomplete continuing branch.')
                measured = dict(model_rows=predicted, real=real, model_seconds=model_seconds,
                    model_work=dict(transition_rows=len(predicted), policy_rows=2*len(predicted), q_rows=len(predicted)))
                _accumulate_timing(timing, measured)
                measure_cache[key] = measured
                return measured

            prior_bounds = dict(log_std_mapping=model.agent.model._log_std_mapping,
                                log_std_min=model.agent.model._log_std_min_value,
                                log_std_max=model.agent.model._log_std_max_value)
            for rule in study['prefix_action_rules']:
                references[rule] = measure(model.agent.model._pi, prior_bounds, rule)
            for solver in range(study['solver_repetitions']):
                solve_rows, callbacks = [], []
                fit_seed = solver_seed(study['controller_seed'], 'entropy-fit', root['root_id'], solver)
                actor_seed = solver_seed(study['controller_seed'], 'entropy-actor', root['root_id'], solver)
                probe_seed = solver_seed(study['controller_seed'], 'entropy-probe', root['root_id'])

                def on_snapshot(metadata, actor_snapshot, critic, root_z):
                    policy = actor_snapshot.make_policy(model.agent.device)
                    bounds = actor_snapshot.policy_bounds
                    before = time.perf_counter()
                    extra = {}
                    for rule in study['prefix_action_rules']:
                        probe_prefix, probe_tail, _ = paired_noise(probe_seed, root['root_id'], horizon=1,
                            tail_steps=1, rollouts=study['model_probe_rollouts'], action_dim=model.cfg.action_dim,
                            prefix_action_rule=rule)
                        predicted_probe = _model_branches(model, root['observation'], policy, bounds,
                                                         probe_prefix, probe_tail, indices)
                        fitted_q = critic_probe(model, critic, root_z, policy, bounds, probe_prefix[0], indices)
                        direct = np.asarray([p['model_return'] for p in predicted_probe])
                        extra[rule] = dict(fitted_critic_q_mean=float(fitted_q.mean()),
                            fitted_critic_minus_direct_model_mean=float((fitted_q-direct).mean()),
                            fitted_critic_direct_model_rmse=float(np.sqrt(np.mean((fitted_q-direct)**2))),
                            independent_model_score_mean=float(direct.mean()))
                        count = study['model_probe_rollouts']
                        for name, amount in [('model_transition_rows', count),
                                             ('model_policy_rows', 3*count),
                                             ('model_q_rows', 2*count)]:
                            timing[name] = timing.get(name, 0) + amount
                    _synchronize(model)
                    timing['model_probe_seconds'] += time.perf_counter() - before
                    for rule in study['prefix_action_rules']:
                        measured = measure(policy, bounds, rule)
                        reference = references[rule]
                        for rollout, (pred, real) in enumerate(zip(measured['model_rows'], measured['real']['rows'])):
                            metrics = _branch_metrics(pred, real, reference['model_rows'][rollout], reference['real']['rows'][rollout])
                            metrics.update(extra[rule])
                            metrics.update({k: float(v) for k, v in metadata.get('metrics', {}).items()
                                            if isinstance(v, (float, int)) and not isinstance(v, bool)})
                            metrics = metric_semantics(metrics, reward_only=reward_only)
                            row = dict(source_run=source['source_run'], checkpoint_step=source['step'],
                                checkpoint_sha256=source['sha256'], prefix_action_rule=rule,
                                episode_seed=episode_seed, root_id=root['root_id'], decision_index=root['decision_index'],
                                solver_repetition=solver, rollout_repetition=rollout,
                                arm=metadata['arm'], actor_updates=actor_snapshot.actor_updates,
                                entropy_mode=metadata['mode'], entropy_alpha=metadata['alpha'],
                                alias_of=metadata.get('alias_of'),
                                critic_updates=study['critic_updates'], metrics=metrics,
                                actor_sha256=actor_snapshot.sha256, actor_state_sha256=state_digest(policy),
                                fit_seed=fit_seed, actor_seed=actor_seed, probe_seed=probe_seed,
                                policy_noise_seed=noises[rule][2], endpoint_action=real['endpoint_action'],
                                outer_critic_target=model.cfg.outer_critic_target,
                                score_semantics='reward_return' if reward_only else 'reward_prefix_plus_soft_Q_hybrid',
                                mc_complete=real['mc_complete'], terminated=real['terminated'], truncated=real['truncated'])
                            solve_rows.append(row)
                    callbacks.append(copy.deepcopy(metadata))
                    del policy

                before = time.perf_counter()
                result = run_root_entropy_probe(model, root['observation'], arms=arms,
                    fit_seed=fit_seed, actor_seed=actor_seed, probe_seed=probe_seed,
                    snapshot_updates=tuple(study['actor_updates']), probe_rollouts=study['model_probe_rollouts'],
                    on_snapshot=on_snapshot, fit_engine=fit_engine)
                elapsed = time.perf_counter() - before
                # Helper provides time excluding synchronous diagnostic callbacks.
                timing['optimization_seconds'] += result.get('optimization_seconds', 0.)
                timing['model_probe_seconds'] += result.get('timing', {}).get('model_probe_seconds', 0.)
                timing['serialization_seconds'] += result.get('timing', {}).get('serialization_seconds', 0.)
                for name, amount in [('optimization_model_transitions', result['collection_transitions']),
                                     ('critic_optimizer_updates', result['critic_updates']),
                                     ('actor_optimizer_updates', result['actor_optimizer_updates'])]:
                    timing[name] = timing.get(name, 0) + amount
                diagnostic_points = result['optimizer_arms'] * len(study['actor_updates'])
                for name in ('diagnostic_policy_rows', 'diagnostic_q_rows'):
                    timing[name] = timing.get(name, 0) + diagnostic_points * study['model_probe_rollouts']
                result['inclusive_wall_seconds'] = elapsed
                fits.append(dict(root_id=root['root_id'], solver_repetition=solver, result=_jsonable(result)))
                serialized = time.perf_counter()
                atomic_json(output / 'solves' / f"{root['root_id']}-solver-{solver}.json",
                            dict(rows=solve_rows, snapshots=callbacks, fit=fits[-1]))
                rows.extend(solve_rows)
                atomic_json(output / 'status.json', dict(status='running', completed_rows=len(rows),
                            completed_solves=len(fits), total_seconds=time.perf_counter()-started), overwrite=True)
                timing['serialization_seconds'] += time.perf_counter() - serialized
            del measure_cache
        if digest != _outer_state_digest(model):
            raise RuntimeError('Frozen outer state changed.')
        expected = expected_rows(study, source, [episode_seed], arm_names)
        validation = aggregate_entropy_rows(rows, expected_rows=expected, final=True,
            bootstrap_resamples=study['bootstrap_resamples'], bootstrap_seed=study['bootstrap_seed'])
        serialized = time.perf_counter()
        with gzip.open(output / 'measurements.jsonl.gz', 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        timing['serialization_seconds'] += time.perf_counter() - serialized
        timing['total_elapsed_seconds'] = time.perf_counter() - started
        record = dict(schema_version=1, complete=True, smoke=smoke, source=source,
            episode_seed=episode_seed, study_sha256=canonical_hash(original_study), science=science,
            arm_names=arm_names, arms=arms, arm_metadata=arm_metadata, rows=len(rows),
            outer_state_unchanged=True, outer_state_sha256=digest, timing=timing,
            resolved_config=_jsonable(vars(model.cfg)), discount=discount,
            semantics='Raw simulator rewards; old soft-Q scores remain explicit hybrid diagnostics.',
            timing_scope='Explicit phase timers; total ends before results/seal/final status I/O and cleanup.',
            fit_results=fits)
        atomic_json(output / 'results.json', record)
        files = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'status.json')
        atomic_json(output / 'checksums.json', {str(p.relative_to(output)): _file_sha256(p) for p in files})
        atomic_json(output / 'status.json', dict(status='complete', completed_rows=len(rows),
                    elapsed_through_seal_seconds=time.perf_counter()-started), overwrite=True)
        return record
    except BaseException as exc:
        error = exc
        atomic_json(output / 'status.json', dict(status='failed', completed_rows=len(rows),
                    error=f'{type(exc).__name__}: {exc}', timing=timing), overwrite=True)
        raise
    finally:
        _close_resources(model, env, *branch_envs, primary_error=error)


def validate_worker(directory, *, study, source, episode_seed):
    directory = Path(directory)
    checksums = read_json(directory / 'checksums.json')
    required = {'results.json', 'measurements.jsonl.gz', 'root-bank.json', 'study.json',
                'matrix.json', 'checkpoint.metadata.json', 'arms.json'}
    if not required <= checksums.keys():
        raise ValueError('Worker seal is missing required files.')
    for name, digest in checksums.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or _file_sha256(path) != digest:
            raise ValueError(f'Worker checksum mismatch: {name}')
    record = read_json(directory / 'results.json')
    expected_arms = ['off', 'prior_recipe', 'squashed_matched']
    if source.get('gaussian_control'):
        expected_arms.append('gaussian_control')
    if (not record.get('complete') or record.get('smoke') or not record.get('outer_state_unchanged')
            or record.get('study_sha256') != canonical_hash(study)
            or record.get('source') != source or record.get('episode_seed') != episode_seed
            or record.get('arm_names') != expected_arms):
        raise ValueError('Worker source, protocol, coverage or frozen-state validation differs.')
    with gzip.open(directory / 'measurements.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != record['rows'] or any(not r.get('mc_complete') or r.get('truncated') for r in rows):
        raise ValueError('Incomplete worker measurement rows.')
    return record, rows


def publish_campaign(campaign):
    """One explicit attempt, one locked CPU publisher; never infer completion."""
    import fcntl
    import subprocess
    from utils.ambi_entropy_reporting import (
        aggregate_entropy_rows, write_entropy_report, start_entropy_wandb, publish_entropy_wandb,
    )
    campaign = Path(campaign).resolve()
    spec = read_json(campaign)
    study = load_study(spec['study'])
    if spec['study_sha256'] != canonical_hash(study):
        raise ValueError('Campaign study identity changed.')
    expected_cells = {(i, seed) for i in range(len(study['checkpoints'])) for seed in study['episode_seeds']}
    actual_cells = [(t['checkpoint_index'], t['episode_seed']) for t in spec['tasks']]
    if len(set(actual_cells)) != len(actual_cells) or set(actual_cells) != expected_cells:
        raise ValueError('Campaign does not declare the exact requested checkpoint/episode panel.')
    output = campaign.parent / 'analysis'
    output.mkdir(exist_ok=True)
    lock = (campaign.parent / 'publisher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'publication-complete.json').exists():
        raise FileExistsError('This attempt has already been published.')
    began = time.perf_counter()
    receipt = campaign.parent / 'publication-started.json'
    receipt_identity = {'run_id': spec['wandb_run_id'], 'campaign': str(campaign)}
    if receipt.exists() and read_json(receipt) != receipt_identity:
        raise ValueError('Publisher resume receipt identifies a different attempt.')
    run = start_entropy_wandb(run_id=spec['wandb_run_id'], config={'study': study,
        'attempt': spec['attempt'], 'scope': 'shared_prior_roots', 'commit': spec['commit'],
        'comparison': 'entropy_form_at_checkpoint_matched_gaussian_coefficient'},
        entity=spec['entity'], project=spec['project'], name=spec['name'],
        resume='must' if receipt.exists() else 'never')
    atomic_json(receipt, receipt_identity, overwrite=True)
    try:
        while True:
            complete, failed = 0, []
            for task in spec['tasks']:
                status_path = Path(task['output_dir']) / 'status.json'
                status = read_json(status_path) if status_path.exists() else {}
                complete += status.get('status') == 'complete'
                if status.get('status') == 'failed':
                    failed.append({'task': task, 'error': status.get('error')})
            coverage = dict(completed_tasks=complete, expected_tasks=len(spec['tasks']), failed_tasks=failed)
            atomic_json(output / 'progress.json', coverage, overwrite=True)
            run.summary.update({'progress/completed_tasks': complete,
                                'progress/expected_tasks': len(spec['tasks']),
                                'progress/failed_tasks': len(failed)})
            if complete == len(spec['tasks']):
                break
            active = subprocess.check_output(['squeue', '-h', '-j', str(spec['compute_job_id'])], text=True).strip()
            if not active:
                raise RuntimeError(f'Worker array ended with incomplete coverage: {complete}/{len(spec["tasks"])}.')
            time.sleep(15)
        compute_wait_seconds = time.perf_counter()-began
        publication_started = time.perf_counter()
        rows, expected, records, files = [], [], [], []
        for task in spec['tasks']:
            source = study['checkpoints'][task['checkpoint_index']]
            record, measurements = validate_worker(task['output_dir'], study=study,
                                                  source=source, episode_seed=task['episode_seed'])
            records.append(record)
            rows.extend(measurements)
            expected.extend(expected_rows(study, source, [task['episode_seed']], record['arm_names']))
            files.append(Path(task['output_dir']))
        if len({r['science'] for r in records}) != 1:
            raise ValueError('Worker scientific source fingerprints differ.')
        current_science = canonical_hash({'base': _science_identity(), 'evaluator': _file_sha256(__file__),
            'probe': _file_sha256(Path(__file__).parent / 'utils/ambi_entropy_probe.py')})
        if records[0]['science'] != current_science:
            raise ValueError('Worker scientific fingerprint differs from the publishing checkout.')
        validation_seconds = time.perf_counter()-publication_started
        timings = {}
        for record in records:
            for name, value in record['timing'].items():
                if isinstance(value, (int, float)):
                    timings[name] = timings.get(name, 0) + value
        aggregation_started = time.perf_counter()
        report = aggregate_entropy_rows(rows, expected_rows=expected, final=True,
            bootstrap_resamples=study['bootstrap_resamples'], bootstrap_seed=study['bootstrap_seed'],
            required_metrics=('real_mc_return', 'real_mc_gain_vs_prior'),
            metadata={'study': study, 'campaign': spec, 'worker_records': records,
                'summed_worker_timing': timings, 'timing_note': 'Worker seconds are summed work, not campaign wall time.',
                'scientific_limits': ['One training seed per selected backbone; comparisons are within checkpoint.',
                    'Old SAC outer Q is soft: its prefix-plus-Q diagnostics are hybrid scores, not predicted reward returns.',
                    'Fixed data and critic test actor optimization; no recollection benefit is measured.']})
        aggregation_seconds = time.perf_counter()-aggregation_started
        serialization_started = time.perf_counter()
        report_files = write_entropy_report(report, output)
        report_serialization_seconds = time.perf_counter()-serialization_started
        publish_entropy_wandb(run, report, report_files)
        import wandb
        artifact = wandb.Artifact(f'entropy-worker-bundles-{run.id}', type='entropy-worker-bundles')
        artifact.add_file(str(campaign), name='campaign.json')
        for task, directory in zip(spec['tasks'], files):
            source = study['checkpoints'][task['checkpoint_index']]
            artifact.add_dir(str(directory), name=f"{source['id']}/seed-{task['episode_seed']}")
        run.log_artifact(artifact)
        for key, value in timings.items():
            run.summary[f'work/{key}'] = value
        run.summary['publication/elapsed_including_compute_wait_seconds'] = time.perf_counter()-began
        run.summary.update({'publication/compute_wait_seconds': compute_wait_seconds,
            'publication/validation_seconds': validation_seconds,
            'publication/aggregation_seconds': aggregation_seconds,
            'publication/report_serialization_seconds': report_serialization_seconds,
            'publication/seconds_before_upload_drain': time.perf_counter()-publication_started})
        run.finish()
        atomic_json(output / 'publication-complete.json', {'run_id': spec['wandb_run_id'],
                    'rows': len(rows), 'workers': len(records), 'files': report_files,
                    'publication_seconds_including_upload_drain': time.perf_counter()-publication_started})
    except BaseException as exc:
        run.summary['publication/error'] = f'{type(exc).__name__}: {exc}'
        run.summary['progress/complete'] = 0
        run.finish(exit_code=1)
        atomic_json(output / 'publication-failed.json', {'error': f'{type(exc).__name__}: {exc}'}, overwrite=True)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    worker = commands.add_parser('run')
    worker.add_argument('--study', dest='study_path', required=True, type=Path)
    worker.add_argument('--checkpoint', required=True, type=Path)
    worker.add_argument('--checkpoint-index', required=True, type=int)
    worker.add_argument('--episode-seed', required=True, type=int)
    worker.add_argument('--output-dir', required=True, type=Path)
    worker.add_argument('--device', default='cuda')
    worker.add_argument('--smoke', action='store_true')
    publisher = commands.add_parser('publish')
    publisher.add_argument('--campaign', required=True, type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    if command == 'run':
        result = run(**args)
        print(json.dumps({'complete': result['complete'], 'rows': result['rows'], 'timing': result['timing']}))
    else:
        publish_campaign(**args)


if __name__ == '__main__':
    main()
