"""Publication-free, pinned 24-setting warm-start evaluation on Oscar.

One L40S worker owns all five paired episodes for one setting. Preparation and
status inspection are CPU-only. Complete artifacts are immutable; interrupted
workers require a new campaign directory instead of silently overwriting data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_RUN = 'rwgao_b-brown-university/ambi/aux6428346x0'
CHECKPOINT_STEP = 575000
CHECKPOINT_SHA = '0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042'
INITIAL_ALPHA = .004603903274983168
SEEDS = [101, 102, 103, 104, 105]
SMOKE_SEEDS = [101, 102]
STEPS, SMOKE_STEPS = 500, 7
PRIOR_MATRIX = ROOT / 'configs/research/ambi_transfer_prior_reference_575k.json'
MATRICES = [ROOT / 'configs/research' / name for name in (
    'ambi_critic_transfer_sweep_575k.json', 'ambi_critic_transfer_hold_h_sweep_575k.json',
    'ambi_actor_transfer_sweep_575k.json', 'ambi_actor_transfer_hold_h_sweep_575k.json')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_new(path, value):
    """Atomic new-file publication; never replace another worker's output."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def source_commit():
    status = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    require(not status.strip(), 'Campaign execution requires a clean source checkout.')
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def cells(matrix_paths=MATRICES):
    from utils.ambi_research import load_preset_matrix
    panel = []
    for path in matrix_paths:
        path = Path(path).resolve()
        matrix = load_preset_matrix(path)
        require(matrix['base_alg_config'] == 'checkpoint' and matrix['source_run'] == SOURCE_RUN,
                'Sweep must inherit the selected 575K backbone.')
        require(matrix['checkpoint_contract'] == {'step': CHECKPOINT_STEP, 'sha256': CHECKPOINT_SHA},
                'Sweep checkpoint contract changed.')
        evaluation = matrix['evaluation']
        for key, expected in dict(seeds=SEEDS, controller_seed=55, max_steps=STEPS,
                                  togo_return_rollouts=32, transfer_diagnostics=True).items():
            require(evaluation.get(key) == expected, f'Unexpected evaluation.{key}.')
        for selector in evaluation['default_presets']:
            group, variant = selector.split('/')
            params = {**matrix['shared_alg_params'],
                      **matrix['comparisons'][group]['variants'][variant]['alg_params']}
            mode = ('actor_only' if params['inner_actor_scope'] == 'episode' else
                    'critic_only' if params['inner_critic_scope'] == 'episode' else 'fresh')
            require((params['inner_actor_scope'], params['inner_critic_scope']) == {
                'fresh': ('action', 'action'), 'actor_only': ('episode', 'action'),
                'critic_only': ('action', 'episode')}[mode], 'Only one component may persist.')
            source = params['inner_critic_source']
            require(source in ('sac', 'aux_return'), 'Unknown critic source.')
            kind = 'soft' if source == 'sac' else 'return'
            expected = {
                'inner_rollout_horizon': 3, 'inner_critic_updates_per_round': 16,
                'inner_actor_updates_per_round': 4, 'inner_rollouts_per_round': 128,
                'inner_batch_size': 256, 'inner_replay_capacity': 3072,
                'inner_first_action_rounds': None, 'inner_actor_source': 'sac',
                'inner_horizon_actor_source': 'sac', 'inner_horizon_critic_source': source,
                'inner_sac_critic_target': 'entropy_augmented' if source == 'sac' else 'reward_only',
                'inner_terminal_entropy': 'outer' if source == 'sac' else 'none',
                'inner_actor_adaptation': 'clone', 'inner_critic_adaptation': 'clone',
                'inner_critic_target_initialization': 'online', 'inner_rebase_persistent': False,
                'inner_actor_writeback_coef': 0., 'inner_critic_writeback_coef': 0.,
                'inner_eval_execution_action': 'mean', 'inner_temperature_mode': 'auto',
                'inner_temperature_initialization': 'inherit_outer', 'inner_target_entropy': 'inherit_outer',
                'inner_entropy_enabled': True, 'compile': True, 'compile_strict': True, 'wandb': False,
                **{f'inner_{c}_scope': 'action' for c in ('temperature', 'replay',
                     'actor_optimizer', 'critic_optimizer', 'temperature_optimizer')},
            }
            for key, value in expected.items():
                require(params.get(key) == value, f'{selector}: unexpected {key}.')
            rounds, interval = params['inner_rounds'], params['inner_solve_interval']
            require(rounds in (1, 8) and interval in (1, 3), 'Sweep must use J1/J8 and cadence1/3.')
            protocol = ('actor-transfer-hold-h-v1' if interval == 3 else 'actor-transfer-v2') if mode == 'actor_only' else (
                'critic-transfer-hold-h-v1' if interval == 3 else 'critic-transfer-v1')
            require(matrix['study_protocol'] == protocol, 'Incorrect transfer protocol.')
            panel.append(dict(name=f'{kind}_{mode}_h3_j{rounds}_i{interval}', selector=selector,
                              matrix=str(path), matrix_sha256=digest(path), study_protocol=protocol,
                              critic_kind=kind, transfer_mode=mode, H=3, J=rounds, solve_interval=interval,
                              requested_alg_params=params))
    keys = [(c['critic_kind'], c['transfer_mode'], c['J'], c['solve_interval']) for c in panel]
    expected_keys = set(itertools.product(('soft', 'return'), ('fresh', 'actor_only', 'critic_only'), (1, 8), (1, 3)))
    require(len(keys) == 24 and set(keys) == expected_keys, 'Sweep must contain exactly 24 distinct settings.')
    order = {'fresh': 0, 'actor_only': 1, 'critic_only': 2}
    # Largest nominal workloads first; Slurm controls live concurrency.
    return sorted(panel, key=lambda c: (-c['J'] * math.ceil(STEPS/c['solve_interval']),
                                        c['critic_kind'], order[c['transfer_mode']]))


def matching_config(actual, expected):
    actual, expected = deepcopy(actual), deepcopy(expected)
    actual.pop('device', None); expected.pop('device', None)
    require(actual == expected, 'Executed config differs from pinned resolved config: ' + str({
        key: (expected.get(key), actual.get(key)) for key in actual.keys() | expected.keys()
        if actual.get(key) != expected.get(key)}))


def prepare(args):
    from slurm.ambi_closed_loop_checkpoint_sweep import resolve_config
    from utils.eval_series_data import planner_identity, scientific_identity
    commit = source_commit()
    panel = cells(args.matrix or MATRICES)
    inventory = read(args.inventory)
    require(inventory['source_run'] == SOURCE_RUN, 'Wrong checkpoint inventory source.')
    rows = [row for row in inventory['checkpoints'] if row['step'] == CHECKPOINT_STEP]
    require(len(rows) == 1, 'Inventory must identify one 575K checkpoint.')
    row = rows[0]; checkpoint = Path(row['path']).resolve()
    metadata = Path(str(checkpoint) + '.metadata.json')
    require(Path(row['metadata_path']).resolve() == metadata, 'Checkpoint sidecar must be adjacent.')
    require(row['sha256'] == digest(checkpoint) == CHECKPOINT_SHA, 'Checkpoint SHA256 mismatch.')
    require(row['metadata_sha256'] == digest(metadata), 'Checkpoint sidecar SHA256 mismatch.')
    require(read(metadata)['checkpoint']['step'] == CHECKPOINT_STEP, 'Checkpoint sidecar step mismatch.')
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require('model' in state and 'aux_return_state' in state and 'log_ent_coef' in state,
            'Checkpoint lacks required actor/critic/temperature state.')
    initial_alpha = float(state['log_ent_coef'].exp().clamp_min(1e-8).item())
    require(math.isclose(initial_alpha, INITIAL_ALPHA, rel_tol=1e-6), 'Checkpoint alpha mismatch.')
    del state
    for index, cell in enumerate(panel):
        config = resolve_config(cell['matrix'], checkpoint, cell['selector'])
        require(config['sac_actor_loss_scale_mode'] == config['aux_return_sac_actor_loss_scale_mode'] == 'none',
                'Checkpoint Q scale semantics changed.')
        require(config['target_entropy'] == -10.5, 'Checkpoint entropy target changed.')
        cell.update(index=index, expected_config=config,
                    planner_identity=planner_identity(config, {}, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean'))
    smoke_indices = [cell['index'] for cell in panel if cell['J'] == 8]
    from utils.ambi_research import load_preset_matrix
    prior_matrix = Path(args.prior_matrix).resolve()
    prior = load_preset_matrix(prior_matrix)
    require(prior['source_run'] == SOURCE_RUN and prior['checkpoint_contract'] == {
        'step': CHECKPOINT_STEP, 'sha256': CHECKPOINT_SHA}, 'Prior checkpoint pin changed.')
    require(prior['evaluation']['default_presets'] == ['reference/prior'] and
            prior['evaluation']['seeds'] == SEEDS and prior['evaluation']['max_steps'] == STEPS and
            prior['evaluation']['controller_seed'] == 55, 'Prior reference protocol changed.')
    prior_config = resolve_config(prior_matrix, checkpoint, 'reference/prior')
    require(prior_config['inner_operator'] == 'none' and prior_config['inner_eval_execution_action'] == 'mean',
            'Prior reference must execute the frozen prior mean.')
    panel.append(dict(index=24, name='prior_reference', selector='reference/prior',
        matrix=str(prior_matrix), matrix_sha256=digest(prior_matrix), transfer_mode='prior',
        J=0, H=3, solve_interval=1, expected_config=prior_config,
        planner_identity=planner_identity(prior_config, {}, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean')))
    campaign = dict(schema_version=1, kind='ambi-warm-start-sweep-v1', source_commit=commit,
                    source_run=SOURCE_RUN, checkpoint=str(checkpoint), checkpoint_step=CHECKPOINT_STEP,
                    checkpoint_sha256=CHECKPOINT_SHA, metadata_sha256=digest(metadata),
                    inventory=str(Path(args.inventory).resolve()), inventory_sha256=digest(args.inventory),
                    source_dir=str(ROOT), initial_alpha=initial_alpha, cells=panel,
                    seeds=SEEDS, controller_seed=55, max_steps=STEPS,
                    smoke_indices=smoke_indices, smoke_seeds=SMOKE_SEEDS, smoke_steps=SMOKE_STEPS,
                    cuda_gate_index=smoke_indices[0], production_indices=list(range(25)),
                    science=scientific_identity('AMBITDMPC2/AMBITDMPC2', 'sac', commit),
                    publication='local_bundles_only_no_online_publication')
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=False)
    write_new(root / 'campaign.json', campaign)
    print(json.dumps(dict(root=str(root), settings=24, prior_references=1, smoke_indices=smoke_indices,
                          production_indices=campaign['production_indices'], source_commit=commit)))
    return campaign


def validate_trace(bundle, cell, *, seeds, steps, initial_alpha=INITIAL_ALPHA):
    manifest = read(Path(bundle) / 'manifest.json'); run, = manifest['runs']
    counts = defaultdict(Counter); rows = 0
    decision_order = []
    previous_key = None
    interval, rounds = cell['solve_interval'], cell['J']
    for name in run['trace_files']:
        target = (Path(bundle) / name).resolve()
        require(target.is_relative_to(Path(bundle).resolve()), 'Trace path escapes bundle.')
        with gzip.open(target, 'rt') as handle:
            for line in handle:
                row = json.loads(line); rows += 1
                key = row['episode_id'], row['decision_index']
                phase, values = row['phase'], row.get('metrics', {})
                if key != previous_key:
                    decision_order.append(key); previous_key = key
                require(row['event_index'] == sum(counts[key].values()), 'Incorrect trace event sequence.')
                if cell['transfer_mode'] == 'prior':
                    require(phase in ('initial', 'decision'), 'Prior evaluation must not perform inner work.')
                    require(not row.get('nonfinite'), 'Nonfinite prior trace.')
                    counts[key][phase] += 1
                    continue
                require(not row.get('nonfinite') and all(isinstance(v, (float, int)) and math.isfinite(v)
                        for v in values.values()), 'Nonfinite trace metrics.')
                decision = key[1]; solved = decision % interval == 0
                require(solved or phase == 'decision', 'Held decision contains solve/probe work.')
                counts[key][phase] += 1
                if phase == 'initial':
                    require(row['replay_size'] == 0, 'Imagined replay leaked across solves.')
                    require(math.isclose(values['alpha'], initial_alpha, rel_tol=1e-6), 'Alpha did not reset.')
                    for component in ('actor', 'critic', 'temperature'):
                        require(values[f'{component}_optimizer_steps_initial'] == 0, 'Optimizer did not reset.')
                    require(values['inner_actor_lifetime_updates_initial'] == (
                        4 * rounds * (decision // interval) if cell['transfer_mode'] == 'actor_only' else 0),
                        'Actor lifetime count disagrees with transfer mode.')
                if phase in ('initial', 'decision'):
                    metrics = {k.removeprefix('decision/'): v for k, v in values.items()}
                    require(metrics['inner_actor_transferred'] == (solved and decision > 0 and cell['transfer_mode'] == 'actor_only'),
                            'Actor transfer flag mismatch.')
                    if cell['transfer_mode'] == 'critic_only':
                        require(metrics['inner_critic_transferred'] == (solved and decision > 0), 'Critic transfer flag mismatch.')
                        require(metrics['inner_critic_target_reinitialized'] == solved, 'Critic target reset flag mismatch.')
                        require(metrics['inner_critic_updates_initial'] == (
                            16 * rounds * (decision // interval) if solved else 0), 'Critic cumulative count mismatch.')
                    if phase == 'decision':
                        dose = rounds if solved else 0
                        expected = dict(inner_rounds=dose, inner_first_action_rounds_applied=0,
                            inner_critic_optimizer_steps=16*dose, inner_actor_optimizer_steps=4*dose,
                            inner_temperature_optimizer_steps=4*dose, inner_model_steps=128*3*dose,
                            inner_compile_fallback=0)
                        if interval > 1:
                            expected.update(inner_solve_performed=int(solved), inner_policy_held=int(not solved),
                                inner_solve_index=decision//interval, inner_action_age=decision%interval,
                                inner_episode_decision_index=decision, inner_solve_interval=interval)
                        for field, value in expected.items():
                            require(metrics[field] == value, f'Incorrect decision workload: {field}.')
    expected_keys = {(f'seed-{seed}', d) for seed in seeds for d in range(steps)}
    require(set(counts) == expected_keys, 'Missing or unexpected episode decisions.')
    require(decision_order == [(f'seed-{seed}', d) for seed in seeds for d in range(steps)],
            'Incorrect episode decision sequence.')
    if cell['transfer_mode'] == 'prior':
        require(all(c == {'initial': 1, 'decision': 1} for c in counts.values()), 'Incorrect prior decision counts.')
        return dict(trace_rows_checked=rows, decisions=len(expected_keys), solves=0,
                    held_decisions=0, total_rounds=0)
    for (_, decision), count in counts.items():
        dose = rounds if decision % interval == 0 else 0
        for phase, amount in dict(initial=int(bool(dose)), collection=dose, update=20*dose,
                                  decision=1, probe=dose+3 if dose else 0,
                                  transfer_probe=dose+3 if dose else 0).items():
            require(count[phase] == amount, f'Unexpected {phase} trace count at decision {decision}.')
    solves = len(seeds) * math.ceil(steps / interval)
    return dict(trace_rows_checked=rows, decisions=len(expected_keys), solves=solves,
                held_decisions=len(expected_keys)-solves, total_rounds=solves*rounds)


def validate_completed(bundle, cell, campaign, *, smoke=False):
    from utils.ambi_benchmark import solver_seed
    from utils.eval_series_data import planner_identity, scientific_identity
    seeds, steps = (SMOKE_SEEDS, SMOKE_STEPS) if smoke else (SEEDS, STEPS)
    manifest = read(Path(bundle) / 'manifest.json')
    require(manifest['status'] == 'complete' and not manifest['code']['dirty'], 'Bundle incomplete or dirty.')
    require(manifest['code']['commit'] == campaign['source_commit'], 'Bundle source commit changed.')
    require(manifest['checkpoint']['sha256'] == CHECKPOINT_SHA and
            manifest['checkpoint']['source_run'] == SOURCE_RUN, 'Bundle checkpoint changed.')
    require(manifest['checkpoint']['metadata'] == read(campaign['checkpoint'] + '.metadata.json'),
            'Bundle checkpoint metadata changed.')
    require(scientific_identity('AMBITDMPC2/AMBITDMPC2', 'sac', manifest['code']['commit']) == campaign['science'],
            'Scientific implementation identity changed.')
    run, = manifest['runs']; result = run['result']
    require(run['selector'] == result['selector'] == cell['selector'], 'Wrong evaluated selector.')
    if cell['transfer_mode'] != 'prior':
        require(run['study_protocol'] == result['study_protocol'] == cell['study_protocol'], 'Wrong evaluated protocol.')
    matching_config(run['resolved_config'], cell['expected_config'])
    matching_config(result['resolved_config'], cell['expected_config'])
    require(planner_identity(run['resolved_config'], result, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean') ==
            cell['planner_identity'], 'Executed planner identity changed.')
    require(result['resolved_device'].startswith('cuda'), 'Production controller did not run on CUDA.')
    require(result['outer_state_unchanged'] and result['outer_updates_before'] == result['outer_updates_after'],
            'Outer state changed.')
    require(not result['nonfinite_model_metrics'] and not result['nonfinite_trace_metrics'], 'Nonfinite evaluator result.')
    require(result['action_rule'] == 'tanh_mean' and result['deterministic_execution'], 'Execution rule changed.')
    require(result['environment_seeds'] == seeds and result['controller_seed'] == 55, 'Episode seed panel changed.')
    require([e['seed'] for e in run['episodes']] == seeds and len(run['trace_files']) == len(seeds), 'Incomplete episode panel.')
    for episode in run['episodes']:
        require(episode['length'] == steps and math.isfinite(episode['return']), 'Incomplete/nonfinite episode.')
        require(episode['solver_seed'] == solver_seed(55, 'episode', episode['seed']), 'Controller RNG seed changed.')
        if cell['transfer_mode'] != 'prior':
            require(episode['solve_count'] == math.ceil(steps / cell['solve_interval']), 'Wrong episode solve count.')
            require(episode['held_decision_count'] == steps-math.ceil(steps / cell['solve_interval']), 'Wrong episode hold count.')
        if not smoke:
            require(not episode['truncated_by_evaluator'], 'Production episode ended before environment completion.')
    return manifest


def receipt(root, campaign, index, *, smoke=False, verify=False):
    cell = campaign['cells'][index]
    directory = Path(root) / ('smoke' if smoke else 'settings') / cell['name']
    path = directory / 'worker-completion.json'
    require(path.is_file(), f'Missing completed {"smoke" if smoke else "production"} setting {index}.')
    result = read(path)
    require(result['status'] == 'complete' and result['index'] == index and result['smoke'] == smoke and
            result['cell'] == cell['name'] and result['campaign_sha256'] == digest(Path(root) / 'campaign.json'),
            'Completion receipt does not match campaign.')
    require('L40S' in result['gpu'], 'Completion hardware differs.')
    if verify:
        require(digest(directory / 'bundle/manifest.json') == result['manifest_sha256'], 'Completed manifest changed.')
        for name, sha in result['trace_sha256'].items():
            require(digest(directory / 'bundle' / name) == sha, 'Completed trace changed.')
        if result.get('cuda_gate'):
            for name, sha in result['cuda_gate']['reports'].items():
                require(digest(directory / name) == sha, 'Completed CUDA gate report changed.')
            require(digest(directory / 'cuda-lifecycle-gate.log') == result['cuda_gate']['log_sha256'],
                    'Completed CUDA gate log changed.')
    return result


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    root = Path(args.root).resolve(); campaign = read(root / 'campaign.json')
    require(source_commit() == campaign['source_commit'], 'Worker source commit mismatch.')
    require(args.index in (campaign['smoke_indices'] if args.smoke else campaign['production_indices']), 'Invalid worker index.')
    cell = campaign['cells'][args.index]
    require(digest(cell['matrix']) == cell['matrix_sha256'], 'Worker matrix changed.')
    require(digest(campaign['checkpoint']) == CHECKPOINT_SHA, 'Worker checkpoint changed.')
    require(digest(campaign['checkpoint']+'.metadata.json') == campaign['metadata_sha256'], 'Worker sidecar changed.')
    require(digest(campaign['inventory']) == campaign['inventory_sha256'], 'Worker inventory changed.')
    directory = root / ('smoke' if args.smoke else 'settings') / cell['name']
    if (directory / 'worker-completion.json').exists():
        result = receipt(root, campaign, args.index, smoke=args.smoke, verify=True)
        print('ALREADY COMPLETE ' + cell['name']); return result
    if not args.smoke:
        for index in campaign['smoke_indices']:
            receipt(root, campaign, index, smoke=True, verify=True)
        gate = receipt(root, campaign, campaign['cuda_gate_index'], smoke=True)
        require(gate.get('cuda_gate', {}).get('status') == 'passed', 'CUDA lifecycle gate has not passed.')
    require(torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name(0), 'Workers require an NVIDIA L40S.')
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / 'worker-start.json', dict(index=args.index, host=socket.gethostname(),
        pid=os.getpid(), started_at=time.time(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID')))
    try:
        gate = None
        if args.smoke and args.index == campaign['cuda_gate_index']:
            gate_directory = directory / 'cuda-lifecycle-gate'
            gate_directory.mkdir(exist_ok=False)
            environment = {**os.environ, 'AMBI_RUN_CRITIC_TRANSFER_CUDA_GATE': '1',
                'AMBI_CRITIC_TRANSFER_GATE_OUTPUT_ROOT': str(gate_directory),
                'EXPECTED_ACTION_MODES_SHA': campaign['source_commit']}
            with (directory / 'cuda-lifecycle-gate.log').open('x') as output:
                subprocess.run([sys.executable, '-m', 'pytest', '-q', '-s',
                    'tests/test_ambi_critic_transfer_cuda_gate.py'], env=environment,
                    stdout=output, stderr=subprocess.STDOUT, check=True, close_fds=False)
            reports = [gate_directory / f'critic-{kind}-interval{interval}.json'
                       for kind in ('soft', 'return') for interval in (1, 3)]
            for report in reports:
                value = read(report)
                require(value['passed'] and value['compile_strict'] and not value['compile_fallback'] and
                        value['source_commit'] == campaign['source_commit'], 'CUDA lifecycle report failed.')
            gate = dict(status='passed', reports={str(p.relative_to(directory)): digest(p) for p in reports},
                        log_sha256=digest(directory / 'cuda-lifecycle-gate.log'))
        bundle = directory / 'bundle'
        evaluate_matrix(cell['matrix'], campaign['checkpoint'], selectors=[cell['selector']],
            seeds=SMOKE_SEEDS if args.smoke else SEEDS, controller_seed=55,
            max_steps=SMOKE_STEPS if args.smoke else STEPS, device='cuda', bundle_dir=bundle,
            checkpoint_inventory=campaign['inventory'])
        manifest = validate_completed(bundle, cell, campaign, smoke=args.smoke)
        trace_summary = validate_trace(bundle, cell, seeds=SMOKE_SEEDS if args.smoke else SEEDS,
            steps=SMOKE_STEPS if args.smoke else STEPS, initial_alpha=campaign['initial_alpha'])
        write_new(directory / 'trace-validation.json', trace_summary)
        seal_episode_bundle(bundle)
        result = dict(status='complete', index=args.index, cell=cell['name'], smoke=args.smoke,
            campaign_sha256=digest(root / 'campaign.json'), study_protocol=cell.get('study_protocol'),
            checkpoint_sha256=CHECKPOINT_SHA, metadata_sha256=campaign['metadata_sha256'],
            manifest_sha256=digest(bundle / 'manifest.json'), gpu=torch.cuda.get_device_name(0),
            trace_sha256={name: digest(bundle / name) for name in manifest['runs'][0]['trace_files']},
            episodes=manifest['runs'][0]['episodes'], **trace_summary,
            **({'cuda_gate': gate} if gate else {}))
        write_new(directory / 'worker-completion.json', result)
        print('COMPLETE ' + cell['name'], flush=True); return result
    except BaseException as error:
        write_new(directory / 'worker-failure.json', dict(status='failed', index=args.index,
            cell=cell['name'], error_type=type(error).__name__, error=str(error), time=time.time()))
        raise


def status(args):
    root = Path(args.root).resolve(); campaign = read(root / 'campaign.json')
    rows, completed = [], {}
    for cell in campaign['cells']:
        directory = root / 'settings' / cell['name']
        entry = dict(index=cell['index'], name=cell['name'])
        if (directory / 'worker-completion.json').is_file():
            record = receipt(root, campaign, cell['index'], verify=args.verify)
            completed[cell['name']] = record
            entry.update(status='complete', return_mean=sum(e['return'] for e in record['episodes'])/len(SEEDS))
        elif (directory / 'worker-failure.json').is_file():
            entry.update(status='failed', error=read(directory / 'worker-failure.json')['error'])
        else:
            entry['status'] = 'started_or_interrupted' if directory.exists() else 'pending'
        rows.append(entry)
    gains = []
    for cell in campaign['cells']:
        if cell['transfer_mode'] == 'prior':
            continue
        fresh = f"{cell['critic_kind']}_fresh_h3_j{cell['J']}_i{cell['solve_interval']}"
        if cell['transfer_mode'] != 'fresh' and cell['name'] in completed and fresh in completed:
            baseline = {e['seed']: e['return'] for e in completed[fresh]['episodes']}
            differences = [e['return']-baseline[e['seed']] for e in completed[cell['name']]['episodes']]
            gains.append(dict(name=cell['name'], fresh_reference=fresh, paired_gain_mean=sum(differences)/len(differences),
                              paired_gains=dict(zip(SEEDS, differences))))
    prior_gains = []
    if 'prior_reference' in completed:
        prior_returns = {e['seed']: e['return'] for e in completed['prior_reference']['episodes']}
        for name, record in completed.items():
            if name == 'prior_reference':
                continue
            differences = [e['return']-prior_returns[e['seed']] for e in record['episodes']]
            prior_gains.append(dict(name=name, paired_gain_mean=sum(differences)/len(differences),
                                    paired_gains=dict(zip(SEEDS, differences))))
    smoke = {str(i): (root / 'smoke' / campaign['cells'][i]['name'] / 'worker-completion.json').is_file()
             for i in campaign['smoke_indices']}
    result = dict(source_commit=campaign['source_commit'], counts=dict(Counter(r['status'] for r in rows)),
                  settings=rows, paired_vs_fresh=gains, paired_vs_prior=prior_gains, smoke_complete=smoke)
    print(json.dumps(result, indent=2)); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--inventory', type=Path, required=True)
    prep.add_argument('--matrix', type=Path, action='append')
    prep.add_argument('--prior-matrix', type=Path, default=PRIOR_MATRIX)
    run = commands.add_parser('worker')
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--index', type=int, required=True)
    run.add_argument('--smoke', action='store_true')
    inspect = commands.add_parser('status')
    inspect.add_argument('--root', type=Path, required=True)
    inspect.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    {'prepare': prepare, 'worker': worker, 'status': status}[args.command](args)


if __name__ == '__main__':
    main()
