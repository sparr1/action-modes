"""Actor-only warm starts with matched cold controls at the frozen 575k checkpoint.

GPU workers only evaluate/seal artifacts. CPU publication owns W&B and preserves
immutable prior references. Every solve, including the first, uses the selected J.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_aux_hj_sweep import ENTITY, PROJECT, SEEDS, digest, read, write, publish_performance
from slurm.ambi_closed_loop_checkpoint_sweep import (
    checkpoint_state_proof, load_prior, resolve_config, verify_reference,
)
from slurm.ambi_closed_loop_critics import CHECKPOINT_SHA, CHECKPOINT_STEP, INITIAL_ALPHA, SOURCE_RUN, source_commit
from slurm.ambi_closed_loop_publish import gpu_jobs_active, indexed_episodes, moments
from slurm.ambi_closed_loop_reward_retrace import historical_cell
from slurm.ambi_closed_loop_reward_retrace_publish import paired_comparison

MATRIX = ROOT / 'configs/research/ambi_actor_transfer_575k.json'
REFERENCES = ROOT / 'configs/research/ambi_actor_transfer_575k_refs.json'
GROUP = 'actor-transfer-v2-575k-20260925'
PROTOCOL = 'actor-transfer-v2'
HOLD_PROTOCOL = 'actor-transfer-hold-h-v1'
HORIZONS = (1, 2, 3)
ROUNDS = (1, 2, 4, 6, 8, 10)
MODES = ('cold', 'actor_warm')
FIRST_ROUNDS = None
OTHER_SCOPES = ('critic', 'temperature', 'replay', 'actor_optimizer', 'critic_optimizer', 'temperature_optimizer')


def requested_params(horizon, rounds, mode, *, hold_h=False):
    assert horizon in HORIZONS and rounds in ROUNDS and mode in MODES
    return {**historical_cell(horizon, rounds)['requested_alg_params'],
            'inner_first_action_rounds': FIRST_ROUNDS,
            'inner_actor_scope': 'episode' if mode == 'actor_warm' else 'action',
            'inner_replay_capacity': max(3072, 128*horizon*rounds),
            'inner_eval_execution_action': 'mean', 'inner_sac_return_estimator': 'one_step',
            'inner_retrace_lambda': 1., 'inner_retrace_batch_trajectories': None,
            **({'inner_solve_interval': horizon} if hold_h else {})}


def cells(matrix_path=MATRIX):
    from utils.ambi_research import load_preset_matrix
    matrix = load_preset_matrix(matrix_path)
    identities = [(h, j, mode) for h in HORIZONS for j in ROUNDS for mode in MODES]
    selectors = [f'sweep/{mode}_h{h}_j{j}_c16' for h, j, mode in identities]
    assert matrix['study_protocol'] in (PROTOCOL, HOLD_PROTOCOL)
    hold_h = matrix['study_protocol'] == HOLD_PROTOCOL
    assert matrix['source_run'] == SOURCE_RUN and matrix['base_alg_config'] == 'checkpoint'
    assert matrix['checkpoint_steps'] == [CHECKPOINT_STEP]
    assert matrix['evaluation'] == dict(controller_seed=55, seeds=SEEDS, max_steps=500,
        togo_return_rollouts=32, actor_transfer_diagnostics=True, default_presets=selectors)
    comparison = matrix['comparisons']['sweep']
    assert comparison['reference'] == 'prior'
    assert set(comparison['variants']) == {'prior', *(s.split('/')[1] for s in selectors)}
    assert comparison['variants']['prior']['alg_params']['inner_operator'] == 'none'
    assert comparison['variants']['prior']['alg_params']['inner_first_action_rounds'] is None
    panel = []
    for (h, j, mode), selector in zip(identities, selectors):
        name = selector.split('/')[1]
        params = {**matrix['shared_alg_params'], **comparison['variants'][name]['alg_params']}
        assert params == requested_params(h, j, mode, hold_h=hold_h), name
        assert all(params[f'inner_{component}_scope'] == 'action' for component in OTHER_SCOPES)
        panel.append(dict(name=name, selector=selector, actual_selector=selector,
            requested_alg_params=params, params={k: v for k, v in params.items() if v is not None},
            H=h, J=j, transfer_mode=mode, first_action_rounds=FIRST_ROUNDS,
            checkpoint_step=CHECKPOINT_STEP, training_decisions=CHECKPOINT_STEP,
            critic_kind='return_only', estimator='one_step', execution_mode='mean',
            alpha_mode='adaptive', reused=False,
            **(dict(solve_interval=h, solves_per_episode=(500+h-1)//h,
                    held_decisions_per_episode=500-(500+h-1)//h) if hold_h else {})))
    return panel


def effective_rounds(cell, decision):
    return cell['J'] if decision % cell.get('solve_interval', 1) == 0 else 0


def solve_count(cell, steps=500):
    interval = cell.get('solve_interval', 1)
    return (steps+interval-1)//interval


def rounds_policy(campaign):
    return ('J rounds at solves t=0,H,2H,...; fixed feedback actor between solves'
            if campaign['study_protocol'] == HOLD_PROTOCOL else 'J at every decision including the first')


def check_config(actual, expected):
    a, b = deepcopy(actual), deepcopy(expected)
    a.pop('device', None); b.pop('device', None)
    assert a == b, {k: (b.get(k), a.get(k)) for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


def check_j10_equivalence(cell, source):
    """The explicit first J10 override is redundant only for J10 cells."""
    assert cell['J'] == source['J'] == 10, 'Only J10 can reuse the first-J10 protocol'
    for key in ('H', 'name', 'selector', 'transfer_mode', 'checkpoint_step'):
        assert cell[key] == source[key], key
    assert not source.get('reused'), 'Reuse must point directly to an original evaluation'
    for field in ('requested_alg_params', 'expected_config'):
        wanted, original = deepcopy(cell[field]), deepcopy(source[field])
        assert wanted.pop('inner_first_action_rounds', None) is None
        assert original.pop('inner_first_action_rounds') == 10
        check_config(original, wanted)
    for key in ('checkpoint_sha256', 'metadata_sha256', 'initial_alpha'):
        assert cell[key] == source[key], key
    wanted, original = deepcopy(cell['identity']), deepcopy(source['identity'])
    assert original['planner']['settings'].pop('inner_first_action_rounds') == 10
    assert original['planner'].pop('semantics') == dict(evaluation_protocol='actor-transfer-v1', first_action_rounds=10)
    assert wanted == original, 'Scientific identity differs beyond the redundant first override'


def _reuse_source(cell, campaign):
    if cell.get('reuse_provenance', {}).get('kind') == 'hold_h_h1':
        return _reuse_h1_source(cell, campaign)
    from utils.eval_series_data import scientific_identity
    proof = cell['reuse_provenance']
    path = Path(proof['campaign_path'])
    assert digest(path) == proof['campaign_sha256'], 'Reuse source campaign changed'
    source_campaign = read(path)
    assert source_campaign['study_protocol'] == proof['study_protocol'] == 'actor-transfer-v1'
    assert source_campaign['source_commit'] == proof['source_commit']
    assert source_campaign['group'] == proof['group']
    assert source_campaign['first_action_rounds'] == 10
    current_science = scientific_identity('AMBITDMPC2/AMBITDMPC2', 'sac', campaign['source_commit'])
    assert proof['implementation_fingerprint'] == current_science == cell['identity']['science']
    for key in ('checkpoint_step', 'checkpoint_sha256', 'source_run'):
        assert source_campaign[key] == campaign[key], key
    assert source_campaign['prior_reference']['manifest_sha256'] == campaign['prior_reference']['manifest_sha256']
    source, = [row for row in source_campaign['cells'] if row['name'] == cell['name']]
    corrected = {**cell, 'identity': cell['corrected_identity']}
    check_j10_equivalence(corrected, source)
    assert cell['source_expected_config'] == source['expected_config']
    for key in ('directory', 'bundle', 'run_dir', 'performance_run_id', 'identity', 'checkpoint',
                'checkpoint_sha256', 'metadata_sha256', 'initial_alpha'):
        assert cell[key] == source[key], key
    return source_campaign, source


def validate_reused_j10_source(campaign, cell):
    """Read-only audit of complete original science, immutable publication and hashes."""
    from utils.eval_series import load_run
    from utils.eval_series_data import load_records
    assert cell.get('reused') and cell['J'] == 10
    _reuse_source(cell, campaign)
    directory, bundle = Path(cell['directory']), Path(cell['bundle'])
    receipt = read(directory / 'worker-completion.json')
    assert receipt['status'] == 'complete' and not receipt['smoke']
    assert receipt['study_protocol'] == 'actor-transfer-v1' and receipt['J'] == receipt['first_action_rounds'] == 10
    assert receipt['cell'] == cell['name'] and receipt['transfer_mode'] == cell['transfer_mode']
    assert receipt['H'] == cell['H'] and 'L40S' in receipt['gpu']
    assert receipt['checkpoint_sha256'] == CHECKPOINT_SHA and receipt['checkpoint_step'] == CHECKPOINT_STEP
    assert receipt['manifest_sha256'] == digest(bundle / 'manifest.json')
    assert receipt['diagnostics_sha256'] == digest(directory / 'transfer-diagnostics.json')
    manifest = validate_completed(bundle, cell, campaign)
    assert set(receipt['trace_sha256']) == set(manifest['runs'][0]['trace_files'])
    assert all(digest(bundle / name) == sha for name, sha in receipt['trace_sha256'].items())
    summary = summarize_trace(bundle, cell, seeds=SEEDS, steps=500)
    assert summary == read(directory / 'transfer-diagnostics.json'), 'Reused diagnostics disagree with raw full-episode traces'
    assert receipt['trace_rows_checked'] == summary['trace_rows_checked']
    record, = load_records(bundle, inventory_path=campaign['inventory'])
    assert record['identity'] == cell['identity'] == load_run(cell['run_dir'])['identity']
    assert not record['provenance']['missing_artifact_files']
    publication = read(directory / 'publication-completion.json')
    assert publication['status'] == 'complete' and publication['cell'] == cell['name']
    assert publication['performance']['run_id'] == cell['performance_run_id']
    return manifest


def reuse_j10_cells(panel, source_path, campaign):
    """Pin all six completed J10 cells before allocating any new publication IDs."""
    path = Path(source_path).resolve()
    if path.is_dir(): path = path / 'campaign.json'
    source_campaign = read(path)
    assert source_campaign['study_protocol'] == 'actor-transfer-v1'
    assert source_campaign['group'] != campaign['group']
    original = {cell['name']: cell for cell in source_campaign['cells']}
    assert len(original) == len(source_campaign['cells'])
    reused = []
    for cell in panel:
        if cell['J'] != 10: continue
        source = original[cell['name']]
        check_j10_equivalence(cell, source)
        corrected_identity = deepcopy(cell['identity'])
        cell.update({key: deepcopy(source[key]) for key in ('directory', 'bundle', 'run_dir', 'performance_run_id',
            'identity', 'checkpoint', 'checkpoint_sha256', 'metadata_sha256', 'initial_alpha')})
        cell.update(reused=True, source_expected_config=deepcopy(source['expected_config']), corrected_identity=corrected_identity,
            reuse_provenance=dict(campaign_path=str(path), campaign_sha256=digest(path),
                study_protocol=source_campaign['study_protocol'], group=source_campaign['group'],
                source_commit=source_campaign['source_commit'], implementation_fingerprint=corrected_identity['science'],
                reason='J10 at every decision in both protocols; only redundant first-override configuration differs'))
        validate_reused_j10_source(campaign, cell)
        reused.append(cell['name'])
    assert len(reused) == 6
    return reused


def check_h1_equivalence(cell, source):
    """H1 has no held decisions; retain original source identity after an audit."""
    assert cell['H'] == source['H'] == cell['solve_interval'] == 1
    for key in ('J', 'name', 'selector', 'transfer_mode', 'checkpoint_step',
                'checkpoint_sha256', 'metadata_sha256', 'initial_alpha'):
        assert cell[key] == source[key], key
    for field in ('requested_alg_params', 'expected_config'):
        wanted, original = deepcopy(cell[field]), deepcopy(source[field])
        assert wanted.pop('inner_solve_interval', 1) == original.pop('inner_solve_interval', 1) == 1
        assert wanted.get('inner_first_action_rounds') is None
        assert original.get('inner_first_action_rounds') is None
        check_config(original, wanted)
    wanted = deepcopy(cell['identity'])
    original = deepcopy(source.get('corrected_identity', source['identity']))
    wanted.pop('science'); original.pop('science')
    assert wanted == original, 'H1 science configuration differs beyond interval1/default'


def validate_h1_compatibility(path, baseline_science, candidate_science):
    proof = read(path)
    assert proof['schema_version'] == 1 and proof['kind'] == 'actor-transfer-h1-cadence-compatibility'
    assert proof['status'] == 'verified' and proof['differences'] == []
    assert proof['baseline_science'] == baseline_science and proof['candidate_science'] == candidate_science
    assert proof['solve_interval'] == 1 and proof['rounds'] == list(ROUNDS) and proof['modes'] == list(MODES)
    assert proof['episodes_per_case'] >= 2 and proof['decisions_per_episode'] >= 3
    assert set(proof['compared']) >= {'actions', 'actor', 'critic', 'optimizer', 'scalar_state', 'rng'}
    return proof


def _reuse_h1_source(cell, campaign):
    from utils.eval_series_data import scientific_identity
    assert campaign['study_protocol'] == HOLD_PROTOCOL and cell['H'] == cell['solve_interval'] == 1
    pin = cell['reuse_provenance']; path = Path(pin['campaign_path'])
    assert digest(path) == pin['campaign_sha256']
    source_campaign = read(path)
    assert source_campaign['study_protocol'] == pin['study_protocol'] == PROTOCOL
    assert source_campaign['source_commit'] == pin['source_commit']
    assert source_campaign['first_action_rounds'] is None
    for key in ('checkpoint_step', 'checkpoint_sha256', 'source_run'):
        assert campaign[key] == source_campaign[key], key
    assert source_campaign['prior_reference']['manifest_sha256'] == campaign['prior_reference']['manifest_sha256']
    source, = [c for c in source_campaign['cells'] if c['name'] == cell['name']]
    check_h1_equivalence({**cell, 'identity': cell['corrected_identity']}, source)
    assert cell['source_expected_config'] == source['expected_config']
    candidate_science = scientific_identity('AMBITDMPC2/AMBITDMPC2', 'sac', campaign['source_commit'])
    assert candidate_science == cell['corrected_identity']['science']
    assert digest(pin['compatibility_proof_path']) == pin['compatibility_proof_sha256']
    validate_h1_compatibility(pin['compatibility_proof_path'], source['identity']['science'], candidate_science)
    for key in ('directory', 'bundle', 'run_dir', 'performance_run_id', 'identity', 'checkpoint',
                'checkpoint_sha256', 'metadata_sha256', 'initial_alpha'):
        assert cell[key] == source[key], key
    return source_campaign, source


def validate_reused_h1_source(campaign, cell):
    from slurm.ambi_actor_transfer_publish import load_completed
    source_campaign, source = _reuse_h1_source(cell, campaign)
    load_completed(source_campaign, source)  # Includes original J10 inheritance when present.
    directory = Path(source['directory'])
    receipt = read(directory / 'worker-completion.json')
    assert 'L40S' in receipt['gpu'] and receipt['H'] == 1
    assert receipt['checkpoint_sha256'] == CHECKPOINT_SHA and receipt['checkpoint_step'] == CHECKPOINT_STEP
    summary = summarize_trace(source['bundle'], source, seeds=SEEDS, steps=500)
    assert summary == read(directory / 'transfer-diagnostics.json')
    assert summary['trace_rows_checked'] == receipt['trace_rows_checked']
    publication = read(directory / 'publication-completion.json')
    assert publication['status'] == 'complete' and publication['performance']['run_id'] == cell['performance_run_id']
    return read(Path(source['bundle']) / 'manifest.json')


def validate_reused_source(campaign, cell):
    if cell.get('reuse_provenance', {}).get('kind') == 'hold_h_h1':
        return validate_reused_h1_source(campaign, cell)
    return validate_reused_j10_source(campaign, cell)


def reuse_h1_cells(panel, source_path, proof_path, campaign):
    assert campaign['study_protocol'] == HOLD_PROTOCOL
    assert proof_path is not None, 'H1 reuse requires an explicit cross-revision compatibility audit'
    path = Path(source_path).resolve()
    if path.is_dir(): path = path / 'campaign.json'
    proof_path = Path(proof_path).resolve()
    original_campaign = read(path)
    assert original_campaign['study_protocol'] == PROTOCOL and original_campaign['group'] != campaign['group']
    originals = {c['name']: c for c in original_campaign['cells']}
    assert len(originals) == len(original_campaign['cells'])
    reused = []
    for cell in panel:
        if cell['H'] != 1: continue
        source = originals[cell['name']]
        check_h1_equivalence(cell, source)
        corrected_identity = deepcopy(cell['identity'])
        cell.update({key: deepcopy(source[key]) for key in ('directory', 'bundle', 'run_dir', 'performance_run_id',
            'identity', 'checkpoint', 'checkpoint_sha256', 'metadata_sha256', 'initial_alpha')})
        cell.update(reused=True, source_expected_config=deepcopy(source['expected_config']), corrected_identity=corrected_identity,
            reuse_provenance=dict(kind='hold_h_h1', campaign_path=str(path), campaign_sha256=digest(path),
                study_protocol=PROTOCOL, group=original_campaign['group'], source_commit=original_campaign['source_commit'],
                compatibility_proof_path=str(proof_path), compatibility_proof_sha256=digest(proof_path),
                reason='H1 solves at every decision; explicit action/state/RNG audit covers interval1 compatibility'))
        validate_reused_h1_source(campaign, cell)
        reused.append(cell['name'])
    assert len(reused) == 12
    return reused


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    from utils.eval_series_data import planner_identity
    panel, commit, pins = cells(args.matrix), source_commit(), read(args.references)
    study_protocol = read(args.matrix)['study_protocol']
    assert not getattr(args, 'reuse_j10_from', None) or study_protocol == PROTOCOL
    inventory = read(args.inventory)
    assert inventory['source_run'] == pins['source_run'] == SOURCE_RUN
    assert (pins['checkpoint_step'], pins['checkpoint_sha256']) == (CHECKPOINT_STEP, CHECKPOINT_SHA)
    row, = [r for r in inventory['checkpoints'] if r['step'] == CHECKPOINT_STEP]
    prior = load_prior(pins['prior_reference'], args.inventory)
    assert row['sha256'] == prior['checkpoint_sha256'] == CHECKPOINT_SHA
    assert row['metadata_sha256'] == prior['metadata_sha256']
    assert digest(row['path']) == CHECKPOINT_SHA and digest(row['metadata_path']) == row['metadata_sha256']
    assert Path(row['metadata_path']) == Path(row['path'] + '.metadata.json')
    assert read(row['metadata_path']) == read(Path(prior['bundle']) / 'manifest.json')['checkpoint']['metadata']
    proof = checkpoint_state_proof(row['path'], prior)
    assert math.isclose(proof['initial_alpha'], INITIAL_ALPHA, rel_tol=1e-6)
    args.root.mkdir(parents=True, exist_ok=False)
    specs = evaluate_matrix(args.matrix, row['path'], seeds=SEEDS, controller_seed=55, max_steps=500,
        bundle_dir=args.root / 'unused', checkpoint_inventory=args.inventory, reference_bundle=prior['bundle'],
        eval_series_spec_dir=args.root / 'specs')
    assert specs['mode'] == 'evaluation_series_specifications'
    assert set(specs['specs']) == {c['selector'] for c in panel}
    specifications = {}
    for cell in panel:
        config = resolve_config(args.matrix, row['path'], cell['selector'])
        spec = read(specs['specs'][cell['selector']])
        assert spec['identity']['backbone'] == prior['identity']['backbone'] == SOURCE_RUN
        assert spec['identity']['protocol'] == prior['identity']['protocol']
        assert spec['identity']['planner'] == planner_identity(config, {}, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean')
        specifications[cell['selector']] = spec
        directory = args.root / cell['name']; directory.mkdir()
        cell.update(directory=str(directory), bundle=str(directory / 'bundle'), checkpoint=row['path'],
            checkpoint_sha256=row['sha256'], metadata_sha256=row['metadata_sha256'],
            initial_alpha=proof['initial_alpha'], expected_config=config, identity=spec['identity'])
    reuse_context = dict(group=args.group, study_protocol=study_protocol, source_commit=commit, checkpoint_step=CHECKPOINT_STEP,
        checkpoint_sha256=CHECKPOINT_SHA, source_run=SOURCE_RUN, prior_reference=prior,
        inventory=str(args.inventory.resolve()))
    reused = reuse_j10_cells(panel, args.reuse_j10_from, reuse_context) if args.reuse_j10_from else []
    if getattr(args, 'reuse_h1_from', None):
        assert not args.reuse_j10_from
        reused = reuse_h1_cells(panel, args.reuse_h1_from, args.h1_compatibility_proof, reuse_context)
    # Resolve and validate the entire grid and references before allocating publication IDs.
    for cell in panel:
        if cell['reused']: continue
        registry = create_run(args.registry, specifications[cell['selector']], args.group + '-' + cell['name'],
                              PROJECT, ENTITY, 'oscar-rgao48')
        cell.update(run_dir=registry['run_dir'], performance_run_id=registry['run_id'])
    campaign = dict(schema_version=1, study_protocol=study_protocol, group=args.group, label=args.label,
        source_commit=commit, source_dir=str(ROOT), source_run=SOURCE_RUN,
        matrix=str(args.matrix.resolve()), inventory=str(args.inventory.resolve()),
        checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA, checkpoint_state_proof=proof,
        initial_alpha=proof['initial_alpha'], target_entropy=-10.5, H=list(HORIZONS), J=list(ROUNDS),
        first_action_rounds=FIRST_ROUNDS, modes=list(MODES), cells=panel, prior_reference=prior,
        smoke_steps=7 if study_protocol == HOLD_PROTOCOL else 3,
        production_indices=[i for i, cell in enumerate(panel) if not cell['reused']],
        smoke_indices=([16, 17, 32, 33, 34, 35] if study_protocol == HOLD_PROTOCOL and reused
                       else [0, 1, 16, 17, 32, 33]), reused_cells=reused,
        publisher_workers=3, overview_run_id=uuid.uuid4().hex,
        prior_compatibility_note='Immutable prior-mean episodes and explicitly validated equivalent cells retain their original performance identities. Consult the campaign study protocol for solve cadence.',
        timing_note='Identical L40S hardware. Controller timing is prediction wall time minus measured CUDA probe durations; it retains host trace overhead and is not untraced latency. Report first/steady latency, probe duration, and instrumented prediction time separately.')
    write(args.root / 'campaign.json', campaign)
    print(f'Prepared {len(panel)-len(reused)} new settings and {len(reused)} reused equivalent settings; overview {campaign["overview_run_id"]}', flush=True)
    return campaign


def summarize_trace(bundle, cell, *, seeds=SEEDS, steps=500):
    """Validate actual uniform-budget work and keep compact per-stage diagnostics."""
    assert not cell.get('reused') or cell['J'] == 10
    interval = cell.get('solve_interval', 1)
    manifest = read(Path(bundle) / 'manifest.json'); run, = manifest['runs']
    counts = defaultdict(Counter)
    # Aggregate decisions within each seed first, then weight seeds equally.
    stage_seed_values = defaultdict(list)
    decision_rows = []
    trace_rows = 0
    for name in run['trace_files']:
        with gzip.open(Path(bundle) / name, 'rt') as handle:
            for line in handle:
                event = json.loads(line); trace_rows += 1
                ep, decision, phase = event['episode_id'], event['decision_index'], event['phase']
                assert ep in {f'seed-{seed}' for seed in seeds} and 0 <= decision < steps
                assert not event.get('nonfinite'), event.get('nonfinite')
                key = (ep, decision); counts[key][phase] += 1
                budget = effective_rounds(cell, decision)
                if not budget: assert phase == 'decision', 'Held decisions must not contain solve/probe events'
                values = event.get('metrics', {})
                assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values())
                if phase == 'initial':
                    assert event['replay_size'] == 0
                    assert values['inner_rounds'] == budget
                    assert values['inner_actor_transferred'] == float(cell['transfer_mode'] == 'actor_warm' and decision > 0)
                    lifetime = 4*(decision//interval)*cell['J'] if cell['transfer_mode'] == 'actor_warm' and decision > 0 else 0
                    assert values['inner_actor_lifetime_updates_initial'] == lifetime
                    assert math.isclose(values['alpha'], cell.get('initial_alpha', INITIAL_ALPHA), rel_tol=1e-6)
                    for component in ('actor', 'critic', 'temperature'):
                        assert values[f'{component}_optimizer_steps_initial'] == 0
                elif phase == 'collection':
                    assert 1 <= event['round_index'] <= budget
                    assert event['replay_size'] == event['round_index'] * 128 * cell['H']
                elif phase == 'update':
                    c, a, r = event['critic_updates'], event['actor_updates'], event['round_index']
                    assert bool(event.get('updated_critic')) != bool(event.get('updated_actor'))
                    for component in ('critic', 'actor', 'temperature'):
                        if event.get('updated_' + component): counts[key]['updates_' + component] += 1
                    assert c == counts[key]['updates_critic'] and a == counts[key]['updates_actor']
                    if event.get('updated_critic'):
                        assert a == (r-1)*4 and (r-1)*16 < c <= r*16
                    else:
                        assert c == r*16 and (r-1)*4 < a <= r*4
                    assert bool(event.get('updated_temperature')) == bool(event.get('updated_actor'))
                elif phase in ('transfer_probe', 'probe'):
                    stage = event.get('stage', 'post_round')
                    for metric, value in values.items():
                        stage_seed_values[(phase, stage, event['round_index'], 'first' if decision == 0 else 'steady', ep, metric)].append(value)
                elif phase == 'decision':
                    decision_rows.append(dict(episode_id=ep, decision=decision, metrics=values))
                    # Engine flags establish the live lifecycle and dose, not just configuration.
                    lookup = {k.removeprefix('decision/'): v for k, v in values.items()}
                    assert lookup['inner_rounds'] == budget
                    assert lookup['inner_first_action_rounds_applied'] == float(bool(cell.get('reused')) and decision == 0)
                    if budget:
                        assert lookup['inner_actor_transferred'] == float(cell['transfer_mode'] == 'actor_warm' and decision > 0)
                    assert lookup['inner_critic_optimizer_steps'] == 16*budget
                    assert lookup['inner_actor_optimizer_steps'] == lookup['inner_temperature_optimizer_steps'] == 4*budget
                    assert lookup['inner_model_steps'] == 128*cell['H']*budget
                    assert lookup['inner_compile_fallback'] == 0
                    if 'solve_interval' in cell:
                        assert lookup['inner_solve_performed'] == float(bool(budget))
                        assert lookup['inner_policy_held'] == float(not budget)
                        assert lookup['inner_solve_index'] == decision//interval
                        assert lookup['inner_action_age'] == decision%interval
                        assert lookup['inner_episode_decision_index'] == decision
                        assert lookup['inner_solve_interval'] == interval
                        if not budget:
                            assert not any(k.startswith('inner_togo_') for k in lookup)
    expected = {(f'seed-{seed}', d) for seed in seeds for d in range(steps)}
    assert set(counts) == expected
    for (_, d), count in counts.items():
        budget = effective_rounds(cell, d)
        for phase, number in dict(initial=int(bool(budget)), collection=budget, update=20*budget, decision=1,
                updates_critic=16*budget, updates_actor=4*budget, updates_temperature=4*budget,
                probe=budget+3 if budget else 0, transfer_probe=budget+3 if budget else 0).items():
            assert count[phase] == number, (phase, count, d)
    grouped = defaultdict(list)
    for (phase, stage, round_index, decision_group, ep, metric), values in stage_seed_values.items():
        grouped[(phase, stage, round_index, decision_group, metric)].append(sum(values)/len(values))
    stage_rows = [dict(phase=k[0], stage=k[1], round_index=k[2], decision_group=k[3], metric=k[4],
                       **moments(values)) for k, values in sorted(grouped.items())]
    return dict(trace_rows_checked=trace_rows, decisions=len(expected), stage_rows=stage_rows,
                decision_rows=decision_rows, total_rounds=len(seeds)*solve_count(cell, steps)*cell['J'],
                **(dict(solve_decisions=len(seeds)*solve_count(cell, steps),
                        held_decisions=len(seeds)*(steps-solve_count(cell, steps)), solve_interval=interval)
                   if 'solve_interval' in cell else {}),
                aggregation='Average roots within each episode, then weight paired environment seeds equally; first decision separated from steady decisions.')


def validate_completed(bundle, cell, campaign, *, smoke=False):
    from utils.ambi_benchmark import episode_protocol, solver_seed
    if cell.get('reused'):
        assert not smoke
        source_campaign, source = _reuse_source(cell, campaign)
        return validate_completed(bundle, source, source_campaign)
    seeds, steps = ([101, 102], campaign.get('smoke_steps', 3)) if smoke else (SEEDS, 500)
    manifest = read(Path(bundle) / 'manifest.json')
    assert manifest['status'] == 'complete' and manifest['code']['dirty'] is False
    assert manifest['code']['commit'] == campaign['source_commit']
    assert manifest['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    assert manifest['checkpoint']['metadata']['checkpoint']['step'] == CHECKPOINT_STEP
    prior = campaign['prior_reference']; verify_reference(prior)
    assert manifest['code']['runtime'] == prior['runtime']
    assert episode_protocol(manifest['protocol']) == episode_protocol({**prior['protocol'], 'max_steps': steps})
    run, = manifest['runs']; result = run['result']; cfg = run['resolved_config']
    if campaign['study_protocol'] == HOLD_PROTOCOL:
        assert run['study_protocol'] == result['study_protocol'] == HOLD_PROTOCOL
    assert run['selector'] == result['selector'] == cell['selector']
    check_config(cfg, cell['expected_config'])
    assert cfg['compile'] and cfg['compile_strict'] and result['resolved_device'].startswith('cuda')
    assert result['outer_state_unchanged'] and result['outer_updates_before'] == result['outer_updates_after']
    assert not result['nonfinite_model_metrics'] and not result['nonfinite_trace_metrics']
    assert result['action_rule'] == 'tanh_mean' and result['deterministic_execution']
    assert result['environment_seeds'] == seeds and result['controller_seed'] == 55
    for key, stats in result['model_metrics'].items():
        assert all(math.isfinite(v) for v in stats.values()), key
        if key.endswith('_fallback'): assert stats['min'] == stats['mean'] == stats['max'] == 0
    for stat in ('mean', 'min', 'max'):
        assert math.isclose(result['model_metrics']['inner_alpha_initial'][stat], cell['initial_alpha'], rel_tol=1e-6)
        assert result['model_metrics']['inner_eval_execution_sampled'][stat] == 0
    assert [e['seed'] for e in run['episodes']] == seeds
    assert len(run['trace_files']) == len(seeds)
    assert not manifest.get('reference') if smoke else manifest['reference']['manifest_sha256'] == prior['manifest_sha256']
    baseline = indexed_episodes(prior['episodes'])
    expected_rows = {(f'seed-{seed}', d, stage, r) for seed in seeds for d in range(steps) if effective_rounds(cell, d)
        for stage, r in [('initial', 0), ('before_first_actor_block', 1), ('after_first_actor_block', 1),
                         *[('post_round', r) for r in range(1, effective_rounds(cell, d)+1)]]}
    assert {(r['episode_id'], r['decision_index'], r['stage'], r['round_index'])
            for r in run['togo_probe_rows']} == expected_rows
    assert len(run['togo_probe_rows']) == len(expected_rows)
    for row in run['togo_probe_rows']:
        assert row['critic_updates'] == row['round_index']*16
        assert row['actor_updates'] == (0 if row['stage'] == 'before_first_actor_block' else row['round_index']*4)
    for episode in run['episodes']:
        assert episode['length'] == steps and episode['solver_seed'] == solver_seed(55, 'episode', episode['seed'])
        assert math.isfinite(episode['return'])
        if campaign['study_protocol'] == HOLD_PROTOCOL:
            assert episode['solve_count'] == solve_count(cell, steps)
            assert episode['held_decision_count'] == steps-solve_count(cell, steps)
        if not smoke:
            assert not episode['truncated_by_evaluator']
            assert math.isclose(episode['paired_return_delta'], episode['return']-baseline[episode['seed'], episode['solver_seed']]['return'], abs_tol=1e-9)
    return manifest


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    campaign = read(args.root / 'campaign.json'); assert source_commit() == campaign['source_commit']
    assert args.index in (campaign['smoke_indices'] if args.smoke else campaign['production_indices'])
    cell = campaign['cells'][args.index]
    assert torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name(0)
    assert digest(cell['checkpoint']) == CHECKPOINT_SHA
    assert digest(cell['checkpoint'] + '.metadata.json') == cell['metadata_sha256']
    verify_reference(campaign['prior_reference'], traces=True)
    directory = args.root / 'smoke' / cell['name'] if args.smoke else Path(cell['directory'])
    directory.mkdir(parents=True, exist_ok=not args.smoke); bundle = directory / 'bundle'
    evaluate_matrix(campaign['matrix'], cell['checkpoint'], selectors=[cell['selector']],
        seeds=[101, 102] if args.smoke else SEEDS, controller_seed=55,
        max_steps=campaign.get('smoke_steps', 3) if args.smoke else 500,
        device='cuda', bundle_dir=bundle, checkpoint_inventory=campaign['inventory'],
        reference_bundle=None if args.smoke else campaign['prior_reference']['bundle'])
    manifest = validate_completed(bundle, cell, campaign, smoke=args.smoke)
    summary = summarize_trace(bundle, cell, seeds=[101, 102] if args.smoke else SEEDS,
                              steps=campaign.get('smoke_steps', 3) if args.smoke else 500)
    write(directory / 'transfer-diagnostics.json', summary)
    seal_episode_bundle(bundle)
    receipt = dict(status='complete', cell=cell['name'], selector=cell['selector'], bundle=str(bundle),
        smoke=args.smoke, study_protocol=campaign['study_protocol'], checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA,
        H=cell['H'], J=cell['J'], first_action_rounds=FIRST_ROUNDS, transfer_mode=cell['transfer_mode'],
        manifest_sha256=digest(bundle / 'manifest.json'),
        trace_sha256={name: digest(bundle / name) for name in manifest['runs'][0]['trace_files']},
        diagnostics_sha256=digest(directory / 'transfer-diagnostics.json'),
        trace_rows_checked=summary['trace_rows_checked'], gpu=torch.cuda.get_device_name(0))
    if campaign['study_protocol'] == HOLD_PROTOCOL:
        receipt.update(solve_interval=cell['solve_interval'], solve_decisions=summary['solve_decisions'],
                       held_decisions=summary['held_decisions'], total_rounds=summary['total_rounds'])
    write(directory / 'validation.json', receipt); write(directory / 'worker-completion.json', receipt)
    print('COMPLETE ' + cell['name'], flush=True)
    return receipt


def main(*, default_matrix=MATRIX, default_group=GROUP,
         default_label='575k actor-only transfer v2 | H1/2/3 J1/2/4/6/8/10 at every decision'):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('root', 'inventory', 'registry'): prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--references', type=Path, default=REFERENCES)
    prep.add_argument('--matrix', type=Path, default=default_matrix)
    prep.add_argument('--reuse-j10-from', type=Path,
        help='Original actor-transfer-v1 campaign directory; all six J10 cells must be complete and published')
    prep.add_argument('--reuse-h1-from', type=Path,
        help='Corrected actor-transfer-v2 campaign; reuse its 12 H1 cells after an explicit compatibility audit')
    prep.add_argument('--h1-compatibility-proof', type=Path)
    prep.add_argument('--group', default=default_group)
    prep.add_argument('--label', default=default_label)
    run = sub.add_parser('worker'); run.add_argument('--root', type=Path, required=True)
    run.add_argument('--index', type=int, required=True); run.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); {'prepare': prepare, 'worker': worker}[args.command](args)


if __name__ == '__main__': main()
