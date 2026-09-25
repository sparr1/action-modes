"""ERE f=.25, H1/2/3 and J1/2/4/6/8/10 on the frozen 575k shared backbone.

J1 reuses its verified uniform result because a single collection round cannot
be narrowed by ERE. Historical reference bundles and publication IDs stay intact.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slurm.ambi_aux_hj_sweep import ENTITY, PROJECT, SEEDS, digest, read, training_summary, write
from slurm.ambi_closed_loop_checkpoint_sweep import (
    checkpoint_state_proof, check_episodes, load_prior, resolve_config, verify_reference,
    validate_completed as validate_checkpoint,
)
from slurm.ambi_closed_loop_critics import CHECKPOINT_SHA, CHECKPOINT_STEP, INITIAL_ALPHA, SOURCE_RUN, source_commit
from slurm.ambi_closed_loop_publish import indexed_episodes
from slurm.ambi_closed_loop_reward_retrace import (
    historical_cell, load_reference as load_historical, normalized_config as normalize_historical,
)

MATRIX = ROOT / 'configs/research/ambi_closed_loop_ere_f025_575k.json'
REFERENCES = ROOT / 'configs/research/ambi_closed_loop_ere_f025_575k_refs.json'
GROUP = 'closed-loop-ere-f025-575k-20260925'
HORIZONS = (1, 2, 3)
ROUNDS = (1, 2, 4, 6, 8, 10)
ERE_DEFAULTS = dict(inner_replay_strategy='uniform', inner_ere_final_fraction=.25,
                    inner_ere_min_rounds=1, inner_ere_actor=True)
ERE_SETTINGS = {**ERE_DEFAULTS, 'inner_replay_strategy': 'ere'}


def requested_params(horizon, rounds):
    return {**historical_cell(horizon, rounds)['requested_alg_params'],
            'inner_eval_execution_action': 'mean', 'inner_sac_return_estimator': 'one_step',
            'inner_retrace_lambda': 1.0, 'inner_retrace_batch_trajectories': None, **ERE_SETTINGS}


def cells(matrix_path=MATRIX):
    from utils.ambi_research import load_preset_matrix
    matrix = load_preset_matrix(matrix_path)
    identities = [(h, j) for h in HORIZONS for j in ROUNDS]
    selectors = [f'sweep/ere_f025_h{h}_j{j}_c16' for h, j in identities]
    assert matrix['source_run'] == SOURCE_RUN and matrix['base_alg_config'] == 'checkpoint'
    assert matrix['checkpoint_steps'] == [CHECKPOINT_STEP]
    assert matrix['evaluation'] == dict(controller_seed=55, seeds=SEEDS, max_steps=500,
                                       togo_return_rollouts=32, default_presets=selectors)
    comparison = matrix['comparisons']['sweep']
    assert comparison['reference'] == 'prior'
    assert set(comparison['variants']) == {'prior', *(s.split('/')[1] for s in selectors)}
    assert comparison['variants']['prior']['alg_params']['inner_operator'] == 'none'
    assert comparison['variants']['prior']['alg_params']['inner_replay_strategy'] == 'uniform'
    panel = []
    for (h, j), selector in zip(identities, selectors):
        name = selector.split('/')[1]
        params = {**matrix['shared_alg_params'], **comparison['variants'][name]['alg_params']}
        assert params == requested_params(h, j), name
        assert params['inner_replay_capacity'] == max(3072, 384*j)
        panel.append(dict(name=name, selector=selector, actual_selector=selector,
            requested_alg_params=deepcopy(params), params={k: v for k, v in params.items() if v is not None},
            H=h, J=j, checkpoint_step=CHECKPOINT_STEP, training_decisions=CHECKPOINT_STEP,
            critic_kind='return_only', estimator='one_step', execution_mode='mean',
            alpha_mode='adaptive', replay_strategy='ere', ere_final_fraction=.25, reused=j == 1))
    return panel


def normalized_config(config):
    return {**ERE_DEFAULTS, **normalize_historical(config)}


def matching_uniform_config(actual, reference):
    """Permit only the declared uniform-to-ERE change and newly explicit defaults."""
    before, after = normalized_config(reference), normalized_config(actual)
    assert before['inner_replay_strategy'] == 'uniform'
    before.update(ERE_SETTINGS)
    before.pop('device', None); after.pop('device', None)
    assert before == after, {k: (before.get(k), after.get(k))
                             for k in before.keys() | after.keys() if before.get(k) != after.get(k)}


def load_uniform(pin, inventory):
    """Require immutable complete historical episodes and their original publication."""
    from utils.eval_series import load_run
    assert pin['execution'] == 'mean' and pin['estimator'] == 'one_step'
    assert (pin['checkpoint_step'], pin['checkpoint_sha256']) == (CHECKPOINT_STEP, CHECKPOINT_SHA)
    reference = load_historical(pin, inventory)
    check_episodes(reference['episodes'])
    assert normalized_config(reference['resolved_config'])['inner_replay_strategy'] == 'uniform'
    manifest = verify_reference(reference, traces=True)
    reference['reference_manifest_sha256'] = manifest['reference']['manifest_sha256']
    assert reference['protocol']['action_rule'] == 'tanh_mean'
    alpha = manifest['runs'][0]['result']['model_metrics']['inner_alpha_initial']
    assert all(math.isclose(alpha[s], INITIAL_ALPHA, rel_tol=1e-7) for s in ('mean', 'min', 'max'))
    reference['initial_alpha'] = alpha['mean']
    publication = read(Path(reference['bundle']).parent / 'publication-completion.json')
    assert publication['training_run_id'] == pin['training_run_id']
    registry = load_run(pin['run_dir'])
    assert registry['run_id'] == pin['performance_run_id'] and registry['identity'] == reference['identity']
    entry = read(Path(pin['run_dir']) / 'publication.json')['records'][reference['record_id']]
    assert entry['status'] == 'published'
    assert entry['checkpoint_step'] == CHECKPOINT_STEP and entry['checkpoint_sha256'] == CHECKPOINT_SHA
    assert entry['artifact_sha256']['manifest.json'] == pin['manifest_sha256']
    assert all(entry['artifact_sha256'][name] == sha for name, sha in reference['trace_sha256'].items())
    reference['publication_entry'] = entry
    return reference


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_benchmark import episode_protocol
    from utils.eval_series import create_run
    from utils.eval_series_data import planner_identity
    panel, commit, pins = cells(args.matrix), source_commit(), read(args.references)
    inventory = read(args.inventory)
    assert inventory['source_run'] == pins['source_run'] == SOURCE_RUN
    assert pins['checkpoint_step'] == CHECKPOINT_STEP and pins['checkpoint_sha256'] == CHECKPOINT_SHA
    row, = [r for r in inventory['checkpoints'] if r['step'] == CHECKPOINT_STEP]
    prior = load_prior(pins['prior_reference'], args.inventory)
    assert row['sha256'] == prior['checkpoint_sha256'] == CHECKPOINT_SHA
    assert row['metadata_sha256'] == prior['metadata_sha256']
    assert digest(row['path']) == row['sha256'] and digest(row['metadata_path']) == row['metadata_sha256']
    assert Path(row['metadata_path']) == Path(row['path'] + '.metadata.json')
    assert read(row['metadata_path']) == read(Path(prior['bundle']) / 'manifest.json')['checkpoint']['metadata']
    proof = checkpoint_state_proof(row['path'], prior)
    assert math.isclose(proof['initial_alpha'], INITIAL_ALPHA, rel_tol=1e-6)
    refs = {(p['H'], p['J']): load_uniform(p, args.inventory) for p in pins['uniform_references']}
    assert len(refs) == len(pins['uniform_references']) == 18
    assert set(refs) == {(h, j) for h in HORIZONS for j in ROUNDS}
    for reference in refs.values():
        assert reference['runtime'] == prior['runtime']
        assert episode_protocol(reference['protocol']) == episode_protocol(prior['protocol'])
        assert reference['reference_manifest_sha256'] == prior['manifest_sha256']
        assert indexed_episodes(reference['episodes']).keys() == indexed_episodes(prior['episodes']).keys()
        baseline = indexed_episodes(prior['episodes'])
        for ep in reference['episodes']:
            assert math.isclose(ep['paired_return_delta'], ep['return'] - baseline[(ep['seed'], ep['solver_seed'])]['return'], abs_tol=1e-9)
    args.root.mkdir(parents=True, exist_ok=False)
    result = evaluate_matrix(args.matrix, row['path'], seeds=SEEDS, controller_seed=55, max_steps=500,
        bundle_dir=args.root / 'unused', checkpoint_inventory=args.inventory, reference_bundle=prior['bundle'],
        eval_series_spec_dir=args.root / 'specs')
    assert result['mode'] == 'evaluation_series_specifications'
    assert set(result['specs']) == {c['selector'] for c in panel}
    specifications = {}
    for cell in panel:
        directory = args.root / cell['name']; directory.mkdir()
        uniform = refs[(cell['H'], cell['J'])]
        config = resolve_config(args.matrix, row['path'], cell['selector'])
        matching_uniform_config(config, uniform['resolved_config'])
        spec = read(result['specs'][cell['selector']])
        specifications[cell['selector']] = spec
        assert spec['identity']['backbone'] == prior['identity']['backbone'] == SOURCE_RUN
        assert spec['identity']['protocol'] == prior['identity']['protocol'] == uniform['identity']['protocol']
        assert spec['identity']['planner'] == planner_identity(config, {}, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean')
        old_planner = deepcopy(uniform['identity']['planner'])
        new_planner = deepcopy(spec['identity']['planner'])
        matching_uniform_config(new_planner.pop('settings'), old_planner.pop('settings'))
        assert new_planner == old_planner
        cell.update(directory=str(directory), checkpoint=row['path'], checkpoint_sha256=row['sha256'],
            metadata_sha256=row['metadata_sha256'], prior_reference=deepcopy(prior), uniform_reference=deepcopy(uniform),
            initial_alpha=proof['initial_alpha'], checkpoint_state_proof=deepcopy(proof), expected_config=config,
            identity=spec['identity'])
        if cell['reused']:
            # Preserve the original scientific identity; this is an equivalence reference, not a new measurement.
            cell.update(bundle=uniform['bundle'], reuse_reference=deepcopy(uniform), requested_identity=spec['identity'],
                identity=uniform['identity'], performance_run_id=uniform['performance_run_id'],
                training_run_id=uniform['training_run_id'], run_dir=uniform['run_dir'])
    # Validate the complete grid before creating any new publication identity.
    for cell in panel:
        if not cell['reused']:
            registry = create_run(args.registry, specifications[cell['selector']],
                args.group + '-' + cell['name'], PROJECT, ENTITY, 'oscar-rgao48')
            cell.update(bundle=str(Path(cell['directory']) / 'bundle'), run_dir=registry['run_dir'],
                        performance_run_id=registry['run_id'], training_run_id=uuid.uuid4().hex)
    campaign = dict(schema_version=1, group=args.group, label=args.label, matrix=str(args.matrix.resolve()),
        inventory=str(args.inventory.resolve()), source_commit=commit, source_dir=str(ROOT), source_run=SOURCE_RUN,
        checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA, checkpoint_steps=[CHECKPOINT_STEP]*18,
        target_entropy=-10.5, initial_alpha=proof['initial_alpha'], H=list(HORIZONS), J=list(ROUNDS), rounds=list(ROUNDS),
        critic_kind='return_only', execution_mode='mean', estimator='one_step', alpha_mode='adaptive',
        replay_strategy='ere', ere_final_fraction=.25, ere_min_rounds=1, ere_actor=True, ere_annealing=False,
        cells=panel, prior_reference=prior, uniform_references=list(refs.values()),
        production_indices=[i for i, c in enumerate(panel) if not c['reused']], smoke_indices=[1, 11, 17],
        publisher_workers=3, overview_run_id=uuid.uuid4().hex,
        prior_compatibility_note='Hash-pinned frozen-prior mean episodes retain their original identity.',
        reused_compatibility_note='J1 ERE windows always contain its sole round; exact uniform sampling and RNG parity are tested. Each J1 reference retains its historical identity and is not republished.')
    write(args.root / 'campaign.json', campaign)
    print(f'Prepared 18 settings, 15 new evaluations; overview {campaign["overview_run_id"]}', flush=True)
    return campaign


def validate_update(event, cell):
    from RL.tdmpc2_core.common.ere import round_windows
    r = event['round_index']
    for component, slots in (('critic', 16), ('actor', 4)):
        if not event.get('updated_' + component):
            continue
        k = event[component + '_updates'] - (r - 1)*slots - 1
        assert 0 <= k < slots
        window = round_windows(r, slots, .25, 1)[k]
        metrics, prefix = event['metrics'], component + '_replay_'
        assert metrics[prefix + 'window_rounds'] == window
        assert metrics[prefix + 'window_transitions'] == window*128*cell['H']
        assert math.isclose(metrics[prefix + 'window_fraction'], window/r, rel_tol=1e-6)
        assert math.isclose(metrics[prefix + 'window_round_fraction'], window/r, rel_tol=1e-6)
        assert 0 <= metrics[prefix + 'round_age_min'] <= metrics[prefix + 'round_age_mean'] <= metrics[prefix + 'round_age_max'] <= window - 1
        assert 0 <= metrics[prefix + 'newest_round_fraction'] <= 1
        assert 0 < metrics[prefix + 'batch_unique_fraction'] <= 1


def validate_decision(event, cell):
    for component, slots in (('critic', 16), ('actor', 4)):
        counts = [event['metrics'][f'decision/inner_{component}_replay_round_{r}_sample_count']
                  for r in range(1, cell['J'] + 1)]
        assert all(c >= 0 and float(c).is_integer() for c in counts)
        assert sum(counts) == slots*cell['J']*256


def validate_completed(bundle, cell, campaign, *, smoke=False):
    manifest = validate_checkpoint(bundle, cell, campaign, smoke=smoke)
    reference = cell['uniform_reference']; verify_reference(reference)
    matching_uniform_config(manifest['runs'][0]['resolved_config'], reference['resolved_config'])
    updates = decisions = 0
    for name in manifest['runs'][0]['trace_files']:
        with gzip.open(Path(bundle) / name, 'rt') as handle:
            for line in handle:
                event = json.loads(line)
                if event['phase'] == 'update':
                    validate_update(event, cell); updates += 1
                elif event['phase'] == 'decision':
                    validate_decision(event, cell); decisions += 1
    assert decisions == (3 if smoke else 2500)
    assert updates == decisions*cell['J']*20
    return manifest


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    campaign = read(args.root / 'campaign.json'); assert source_commit() == campaign['source_commit']
    assert args.index in (campaign['smoke_indices'] if args.smoke else campaign['production_indices'])
    cell = campaign['cells'][args.index]
    assert not cell['reused'] and torch.cuda.is_available()
    assert digest(cell['checkpoint']) == CHECKPOINT_SHA
    assert digest(cell['checkpoint'] + '.metadata.json') == cell['metadata_sha256']
    verify_reference(cell['prior_reference'], traces=True)
    verify_reference(cell['uniform_reference'], traces=True)
    directory = args.root / 'smoke' / cell['name'] if args.smoke else Path(cell['directory'])
    directory.mkdir(parents=True, exist_ok=not args.smoke); bundle = directory / 'bundle'
    evaluate_matrix(campaign['matrix'], cell['checkpoint'], selectors=[cell['selector']],
        seeds=[101] if args.smoke else SEEDS, controller_seed=55, max_steps=3 if args.smoke else 500,
        device='cuda', bundle_dir=bundle, checkpoint_inventory=campaign['inventory'],
        reference_bundle=None if args.smoke else cell['prior_reference']['bundle'])
    manifest = validate_completed(bundle, cell, campaign, smoke=args.smoke)
    summary = training_summary(bundle, cell, expected_steps=3 if args.smoke else 500)
    seal_episode_bundle(bundle)
    receipt = dict(status='complete', cell=cell['name'], selector=cell['selector'], bundle=str(bundle),
        reused=False, smoke=args.smoke, execution='mean', estimator='one_step', alpha_mode='adaptive',
        replay_strategy='ere', ere_final_fraction=.25, ere_min_rounds=1, ere_actor=True,
        checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA, H=cell['H'], J=cell['J'],
        manifest_sha256=digest(bundle / 'manifest.json'),
        trace_sha256={name: digest(bundle / name) for name in manifest['runs'][0]['trace_files']},
        trace_rows_checked=summary['trace_rows_checked'], gpu=torch.cuda.get_device_name(0))
    write(directory / 'validation.json', receipt); write(directory / 'worker-completion.json', receipt)
    print('COMPLETE ' + cell['name'], flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('root', 'inventory', 'registry'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--references', type=Path, default=REFERENCES)
    prep.add_argument('--matrix', type=Path, default=MATRIX)
    prep.add_argument('--group', default=GROUP)
    prep.add_argument('--label', default='575k shared −10.5 | ERE f=.25 | H1/2/3 J1–10 | return critics')
    run = sub.add_parser('worker')
    run.add_argument('--root', type=Path, required=True); run.add_argument('--index', type=int, required=True)
    run.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); {'prepare': prepare, 'worker': worker}[args.command](args)


if __name__ == '__main__':
    main()
