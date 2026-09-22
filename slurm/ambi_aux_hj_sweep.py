"""Configured H/J campaigns; complete panels and streaming trace publication."""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
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
MATRIX = ROOT / 'configs/research/ambi_aux_hj_sweep_625k.json'
SEEDS = [101, 102, 103, 104, 105]
CHECKPOINT_SHA = 'b91650597b585805e5e2c28fffb721e5aa3737b2a032360304e86aa81ff02ca3'
ENTITY = 'rwgao_b-brown-university'
PROJECT = 'ambi-inner-bench'
GROUP = 'aux625k-h123-j124-20260917'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from utils.ambi_benchmark import atomic_json
    atomic_json(Path(path), value, overwrite=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cells(matrix=None):
    matrix = read(matrix or MATRIX)
    result = []
    for selector in matrix['evaluation']['default_presets']:
        name = selector.split('/')[1]
        params = matrix['comparisons']['sweep']['variants'][name]['alg_params']
        result.append(dict(name=name, selector=selector, params=params,
                           H=params['inner_rollout_horizon'], J=params['inner_rounds']))
    return result


def actor_updates(cell):
    return int(cell['params'].get('inner_actor_updates_per_round', 4))


def critic_updates(cell):
    return int(cell['params'].get('inner_critic_updates_per_round', 32))


def validate(path, cell, *, seeds=SEEDS, steps=500, paired=True, checkpoint_sha=CHECKPOINT_SHA,
             checkpoint_step=625000):
    """Validate semantics, replay retention, exact work and trace coverage."""
    manifest = read(Path(path) / 'manifest.json')
    assert manifest['status'] == 'complete'
    assert manifest['checkpoint']['sha256'] == checkpoint_sha
    assert manifest['checkpoint']['metadata']['checkpoint']['step'] == checkpoint_step
    run, = manifest['runs']
    cfg, result = run['resolved_config'], run['result']
    assert run['status'] == 'complete'
    for key, value in cell['params'].items():
        assert cfg.get(key, 'none' if key == 'inner_terminal_entropy' else None) == value, (key, value)
    assert cfg['inner_batch_size'] == 256 and cfg['inner_rollouts_per_round'] == 128
    assert cfg['inner_critic_updates_per_round'] == critic_updates(cell) and cfg['inner_actor_updates_per_round'] == actor_updates(cell)
    assert cfg['inner_replay_capacity'] == cell['params'].get('inner_replay_capacity', 2048)
    assert cfg['inner_replay_sampling'] == 'with_replacement'
    round_only = cell['params'].get('inner_replay_reset_each_round', False)
    assert cfg.get('inner_replay_reset_each_round', False) == round_only
    assert cfg['inner_bootstrap_source'] == 'inner_target' and cfg['inner_finite_horizon']
    for component in ('actor', 'critic', 'temperature', 'replay', 'actor_optimizer', 'critic_optimizer', 'temperature_optimizer'):
        assert cfg[f'inner_{component}_scope'] == 'action'
    assert cfg['inner_actor_source'] == cfg['inner_horizon_actor_source'] == 'sac'
    assert cfg['inner_execution_action'] == 'mean' and cfg['inner_log_std_mapping'] == 'direct_clamp'
    assert cfg['inner_actor_initialization'] == cfg['inner_critic_initialization'] == 'prior'
    assert cfg['inner_critic_target_initialization'] == 'online'
    assert cfg['inner_temperature_initialization'] == cfg['inner_target_entropy'] == 'inherit_outer'
    assert cfg['inner_actor_lr'] == cfg['inner_critic_lr'] == 3e-4
    assert cfg['inner_critic_target_tau'] == cell['params'].get('inner_critic_target_tau', .01)
    assert cfg['inner_critic_target_update_interval'] == 1
    assert cfg['inner_critic_dropout_enabled'] and cfg['inner_outer_replay_fraction'] == 0
    assert result['outer_state_unchanged'] and result['outer_updates_before'] == result['outer_updates_after']
    assert not result['nonfinite_model_metrics'] and not result['nonfinite_trace_metrics']
    assert result['environment_seeds'] == seeds and result['controller_seed'] == 55
    j, h = cell['J'], cell['H']
    expected = dict(inner_model_steps=128*h*j, inner_buffer_size=128*h*(1 if round_only else j),
                    inner_critic_optimizer_steps=critic_updates(cell)*j, inner_actor_optimizer_steps=actor_updates(cell)*j,
                    inner_temperature_optimizer_steps=actor_updates(cell)*j if cfg['inner_entropy_enabled'] else 0,
                    inner_compile_fallback=0)
    for key, value in expected.items():
        for statistic in ('mean', 'min', 'max'):
            assert result['model_metrics'][key][statistic] == value, (key, statistic)
    initial = result['model_metrics']['inner_alpha_initial']
    for stat in ('mean', 'min', 'max'):
        assert (initial[stat] > 0) if cfg['inner_entropy_enabled'] else (initial[stat] == 0)
        if checkpoint_sha == CHECKPOINT_SHA and cfg['inner_entropy_enabled']:
            assert math.isclose(initial[stat], .004345251712948084, rel_tol=1e-6)
    probe = run['togo_return_probe']
    assert probe['rollouts'] == 32 and probe['horizon'] == h
    assert not probe['entropy_bonus'] and probe['cadence'] == 'initial_and_after_each_round'
    assert [e['seed'] for e in run['episodes']] == seeds
    assert len(run['trace_files']) == len(seeds)
    for ep in run['episodes']:
        assert ep['length'] == steps
        if paired:
            assert not ep['truncated_by_evaluator'] and 'paired_return_delta' in ep
        assert [(p['round_index'], p['critic_updates'], p['actor_updates'])
                for p in ep['togo_round_summaries']] == [(r, r*critic_updates(cell), r*actor_updates(cell)) for r in range(j+1)]
    return manifest


def matched_baseline(spec, record, *, setting, before, after, baseline_science=None):
    """Require a complete paired baseline differing only in one named setting."""
    for key in ('backbone', 'protocol'):
        assert spec['identity'][key] == record['identity'][key], key
    assert record['identity']['science'] == (baseline_science or spec['identity']['science'])
    expected = deepcopy(record['identity']['planner'])
    assert expected['settings'].get(setting, False) == before
    expected['settings'][setting] = after
    assert spec['identity']['planner'] == expected
    assert record['checkpoint']['step'] == 625000
    assert record['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert record['metrics']['eval/frozen_state_unchanged']
    assert sorted(e['seed'] for e in record['episodes']) == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in record['episodes'])
    return dict(record_id=record['record_id'],
                episodes=[{k:e[k] for k in ('seed','solver_seed','return')} for e in record['episodes']])


def polyak_baseline(spec, record):
    return dict(matched_baseline(spec, record, setting='inner_critic_target_tau', before=.01, after=.1), tau=.01)


def round_replay_baseline(spec, record, *, baseline_science=None):
    result = matched_baseline(spec, record, setting='inner_replay_reset_each_round',
                              before=False, after=True, baseline_science=baseline_science)
    return dict(result, kind='round_replay', replay_reset_each_round=False,
                source_comparison=dict(baseline=record['identity']['science'],
                                       candidate=spec['identity']['science'],
                                       change='Opt-in round replay reset; default-off parity tested.'))


def comparison_prefix(baseline):
    if baseline.get('kind') == 'critic_budget':
        return 'comparison/c32'
    if baseline.get('kind') == 'round_budget':
        return 'comparison/j4'
    if baseline.get('kind') == 'interleaved':
        return 'comparison/phased'
    if baseline.get('kind') == 'actor_budget':
        return 'comparison/a4'
    return 'comparison/all_round_replay' if baseline.get('kind') == 'round_replay' else 'comparison/tau001'


def polyak_comparison(episodes, baseline):
    """Paired episode differences, retaining the legacy public helper name."""
    import numpy as np
    old = {(e['seed'], e['solver_seed']):e['return'] for e in baseline['episodes']}
    new = {(e['seed'], e['solver_seed']):e['return'] for e in episodes}
    assert len(old) == len(new) == 5 and old.keys() == new.keys()
    rows = [dict(seed=k[0], solver_seed=k[1], baseline_return=old[k],
                 return_value=new[k], paired_gain=new[k]-old[k]) for k in sorted(old)]
    delta = np.array([r['paired_gain'] for r in rows])
    assert np.isfinite(delta).all()
    draws = np.random.default_rng(20260912).integers(0,5,size=(2000,5))
    low, high = np.percentile(delta[draws].mean(axis=1), [2.5,97.5])
    prefix = comparison_prefix(baseline)
    return dict(rows=rows, bootstrap_seed=20260912, bootstrap_resamples=2000,
                metrics={prefix+'_gain_mean':float(delta.mean()),
                         prefix+'_gain_sample_std':float(delta.std(ddof=1)),
                         prefix+'_gain_ci95_low':float(low),
                         prefix+'_gain_ci95_high':float(high),
                         prefix+'_paired_episodes':5,
                         prefix+'_return_mean':float(np.mean(list(old.values())))})


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    from utils.eval_series_data import load_records
    from utils.ambi_benchmark import episode_protocol
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    assert digest(args.checkpoint) == CHECKPOINT_SHA
    matrix, group = args.matrix, args.group
    evaluate_matrix(matrix, args.checkpoint, seeds=SEEDS, controller_seed=55, max_steps=500,
                    bundle_dir=root/'unused', checkpoint_inventory=args.inventory,
                    reference_bundle=args.reference, eval_series_spec_dir=root/'specs')
    prior_manifest = read(args.reference/'manifest.json')
    prior, = load_records(args.reference, inventory_path=args.inventory)
    assert prior['checkpoint']['sha256'] == CHECKPOINT_SHA and prior['checkpoint']['step'] == 625000
    assert prior['identity']['planner'] == {'type': 'prior', 'action_rule': 'tanh_mean'}
    assert prior['metrics']['eval/frozen_state_unchanged'] and [e['seed'] for e in prior['episodes']] == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in prior['episodes'])
    reuse = {'return_return_alpha_h1_j1': args.reuse_alpha, 'return_return_zero_h1_j1': args.reuse_zero}
    baseline_cells = ({c['name']:c for c in read(args.baseline_campaign/'campaign.json')['cells']}
                      if args.baseline_campaign else {})
    if getattr(args, 'baseline_extension_campaign', None):
        assert args.baseline_kind == 'critic_budget'
        for cell in read(args.baseline_extension_campaign/'campaign.json')['cells']:
            name = cell['name'].removesuffix('_jscale')
            assert name not in baseline_cells
            baseline_cells[name] = cell
    panel = []
    for cell in cells(matrix):
        directory = root/cell['name']
        directory.mkdir()
        spec = read(root/'specs'/f"{cell['selector'].replace('/', '__')}.json")
        assert spec['identity']['backbone'] == prior['identity']['backbone']
        assert spec['identity']['protocol'] == prior['identity']['protocol']
        source = reuse.get(cell['name'])
        if source:
            manifest = validate(source, cell)
            assert episode_protocol(manifest['protocol']) == episode_protocol(prior_manifest['protocol'])
            record, = load_records(source, inventory_path=args.inventory)
            assert record['identity']['backbone'] == spec['identity']['backbone']
            assert record['identity']['protocol'] == spec['identity']['protocol']
            assert record['identity']['planner'] == spec['identity']['planner'], cell['name']
            spec = record  # Retain original scientific source identity for reused measurements.
            bundle, actual_selector = str(source), manifest['runs'][0]['selector']
        else:
            bundle, actual_selector = str(directory/'bundle'), cell['selector']
        if baseline_cells:
            round_replay = args.baseline_kind == 'round_replay'
            interleaved = args.baseline_kind == 'interleaved'
            suffix = '_interleaved' if interleaved else '_roundreplay' if round_replay else '_tau010'
            baseline_name = cell['name'].removesuffix(suffix)
            if args.baseline_kind == 'round_budget':
                baseline_name = cell['name'].removesuffix('_jscale').replace(f"_j{cell['J']}", '_j4')
            elif args.baseline_kind == 'critic_budget':
                baseline_name = cell['name'].rsplit('_c', 1)[0]
            previous = baseline_cells[baseline_name]
            record, = load_records(previous['bundle'], inventory_path=args.inventory)
            if args.baseline_kind == 'critic_budget':
                from slurm.ambi_aux_critic_budget import historical_critic_baseline
                validate(previous['bundle'], previous)
                cell['baseline'] = historical_critic_baseline(spec, record, critic_updates(cell))
            elif args.baseline_kind == 'round_budget':
                from slurm.ambi_aux_round_budget import historical_baseline
                validate(previous['bundle'], previous)
                cell['baseline'] = historical_baseline(spec, record, cell['J'])
            elif interleaved:
                from slurm.ambi_aux_interleaved import phased_baseline
                cell['baseline'] = phased_baseline(spec, record)
                old_receipt = read(Path(previous['directory'])/'worker-completion.json')
                assert old_receipt['status'] == 'complete'
                assert digest(Path(previous['bundle'])/'manifest.json') == old_receipt['manifest_sha256']
                assert all(digest(Path(previous['bundle'])/n) == sha for n,sha in old_receipt['trace_sha256'].items())
                validate(previous['bundle'], previous)
                cell['baseline']['execution'] = old_receipt['execution']
            elif round_replay:
                from utils.eval_series_data import scientific_identity
                # Explicit historical source pin: the new opt-in reset changes
                # implementation identity. Never relax compatibility globally.
                baseline_science = scientific_identity('AMBITDMPC2/AMBITDMPC2', None,
                    '06d7077b32ec06d24e9467f6b8d38bb08133fd1f')
                cell['baseline'] = round_replay_baseline(spec, record, baseline_science=baseline_science)
            else:
                cell['baseline'] = polyak_baseline(spec, record)
            cell['baseline'].update(performance_run_id=previous['performance_run_id'],
                                    bundle=previous['bundle'],
                                    manifest_sha256=digest(Path(previous['bundle'])/'manifest.json'))
        registry = create_run(args.registry, spec, group+'-'+cell['name']+('-reused' if source else ''),
                              PROJECT, ENTITY, 'oscar-rgao48')
        cell.update(bundle=bundle, actual_selector=actual_selector, reused=bool(source),
                    run_dir=registry['run_dir'], performance_run_id=registry['run_id'],
                    training_run_id=uuid.uuid4().hex, directory=str(directory))
        if source:
            write(directory/'worker-completion.json', dict(status='complete', reused=True,
                  original_code=manifest['code'], manifest_sha256=digest(source/'manifest.json'),
                  trace_sha256={n:digest(source/n) for n in manifest['runs'][0]['trace_files']},
                  bundle=bundle, selector=actual_selector))
        panel.append(cell)
    campaign = dict(schema_version=1, group=group, label=args.label, matrix=str(matrix.resolve()), checkpoint=str(args.checkpoint),
                    checkpoint_sha256=CHECKPOINT_SHA, inventory=str(args.inventory), reference=str(args.reference),
                    prior_manifest_sha256=digest(args.reference/'manifest.json'),
                    prior_source_science=prior['identity']['science'],
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    overview_run_id=uuid.uuid4().hex, cells=panel)
    execution = read(matrix).get('execution')
    if execution:
        if execution.get('interleaved_comparison'):
            from slurm.ambi_aux_interleaved import prepare_campaign
            campaign['baseline_campaign'] = str(args.baseline_campaign)
        else:
            from slurm.ambi_aux_actor_budget import prepare_campaign
        prepare_campaign(campaign, root, execution)
    if read(matrix).get('round_budget_sweep'):
        assert args.baseline_kind == 'round_budget' and args.baseline_campaign
        assert len(panel) == 12 and not execution
        campaign['publisher_workers'] = 4
        campaign['baseline_campaign'] = str(args.baseline_campaign)
        campaign['extension_of_original_hj_table'] = True
    if read(matrix).get('critic_budget_sweep'):
        assert args.baseline_kind == 'critic_budget' and args.baseline_campaign
        assert args.baseline_extension_campaign and not execution
        campaign['publisher_workers'] = 4
        campaign['baseline_campaign'] = str(args.baseline_campaign)
        campaign['baseline_extension_campaign'] = str(args.baseline_extension_campaign)
        campaign['critic_budget_sweep'] = True
    write(root/'campaign.json', campaign)
    print(json.dumps(dict(root=str(root), conditions=len(panel), reused=sum(c['reused'] for c in panel),
                         overview_run_id=campaign['overview_run_id'])), flush=True)


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    campaign = read(args.root/'campaign.json')
    from slurm.ambi_aux_actor_budget import configure_execution, runtime_receipt
    configure_execution(campaign)
    assert torch.cuda.is_available()
    if args.smoke:
        # Both boundaries, both fitting rules and alpha-off, at the largest H/J.
        chosen = ([c for c in campaign['cells'] if actor_updates(c) in (4,16)]
                  if any(campaign.get('execution',{}).get(k) for k in ('actor_budget_comparison','interleaved_comparison')) else
                  [c for c in campaign['cells'] if c['H'] == max(x['H'] for x in campaign['cells'])
                   and c['J'] == max(x['J'] for x in campaign['cells'])])
    else:
        chosen = [campaign['cells'][args.index]]
    for cell in chosen:
        if getattr(args, 'phased_control', False):
            assert args.smoke and campaign['execution']['interleaved_comparison']
            cell = deepcopy(cell)
            cell['name'] = cell['name'].removesuffix('_interleaved')
            cell['selector'] = cell['selector'].removesuffix('_interleaved')
            cell['params'].pop('inner_component_update_order')
        assert args.smoke or not cell['reused']
        smoke_root = args.root/'smoke'
        if getattr(args,'replica','default') != 'default':
            smoke_root /= args.replica
        directory = smoke_root/cell['name'] if args.smoke else Path(cell['directory'])
        directory.mkdir(parents=True, exist_ok=args.smoke is False)
        bundle = directory/'bundle'
        evaluate_matrix(campaign['matrix'], campaign['checkpoint'], selectors=[cell['selector']],
                        seeds=[101] if args.smoke else SEEDS, controller_seed=55,
                        max_steps=3 if args.smoke else 500, device='cuda', bundle_dir=bundle,
                        checkpoint_inventory=campaign['inventory'],
                        reference_bundle=None if args.smoke else campaign['reference'])
        manifest = validate(bundle, cell, seeds=[101] if args.smoke else SEEDS,
                            steps=3 if args.smoke else 500, paired=not args.smoke)
        assert manifest['runs'][0]['resolved_config']['compile_strict']
        assert manifest['runs'][0]['result']['resolved_device'].startswith('cuda')
        training_summary(bundle, cell, expected_steps=3 if args.smoke else 500)
        execution = runtime_receipt(campaign)
        if execution:
            write(bundle/'execution.json', execution)
        seal_episode_bundle(bundle)
        write(directory/'worker-completion.json',dict(status='complete', cell=cell['name'],
              manifest_sha256=digest(bundle/'manifest.json'), gpu=torch.cuda.get_device_name(0),
              trace_sha256={n:digest(bundle/n) for n in manifest['runs'][0]['trace_files']},
              selector=cell['selector'], bundle=str(bundle), reused=False, execution=execution))
        print('COMPLETE '+cell['name'], flush=True)


class Moments:
    def __init__(self):
        self.n = 0; self.total = 0.; self.low = math.inf; self.high = -math.inf

    def add(self, value):
        assert isinstance(value, (float,int)) and math.isfinite(value)
        self.n += 1; self.total += value; self.low = min(self.low,value); self.high = max(self.high,value)

    def summary(self):
        return dict(mean=self.total/self.n, min=self.low, max=self.high, count=self.n)


def training_summary(bundle, cell, *, expected_steps=500):
    """Stream all raw rows; retain every update position and per-seed decision."""
    manifest = read(Path(bundle)/'manifest.json')
    run, = manifest['runs']
    curves = defaultdict(lambda: defaultdict(Moments))
    decisions = defaultdict(lambda: defaultdict(Moments))
    counts = defaultdict(lambda: defaultdict(int))
    for name in run['trace_files']:
        with gzip.open(Path(bundle)/name,'rt') as f:
            for line in f:
                e = json.loads(line)
                ep, decision = e['episode_id'], e['decision_index']
                key = (ep, decision)
                assert 0 <= decision < expected_steps
                assert not e.get('nonfinite'), e.get('nonfinite')
                phase = e['phase']
                if phase == 'initial':
                    assert e['replay_size'] == 0
                    counts[key]['initial'] += 1
                if phase == 'collection':
                    retained = 1 if cell['params'].get('inner_replay_reset_each_round',False) else e['round_index']
                    assert e['replay_size'] == retained*128*cell['H']
                    counts[key]['collection'] += 1
                if phase == 'update':
                    for component in ('critic','actor','temperature'):
                        if e.get('updated_'+component):
                            counts[key][component] += 1
                    if cell['params'].get('inner_component_update_order') == 'interleaved':
                        # Validate the chronology, not just final optimizer totals.
                        c, a = counts[key]['critic'], counts[key]['actor']
                        interval = critic_updates(cell) // actor_updates(cell)
                        assert bool(e.get('updated_critic')) != bool(e.get('updated_actor'))
                        assert e['critic_updates'] == c and e['actor_updates'] == a
                        assert c == a*interval if e.get('updated_actor') else a*interval < c <= (a+1)*interval
                        assert bool(e.get('updated_temperature')) == bool(e.get('updated_actor'))
                    elif cell['params'].get('inner_component_update_order') == 'critic_first':
                        c, a, r = counts[key]['critic'], counts[key]['actor'], e['round_index']
                        assert bool(e.get('updated_critic')) != bool(e.get('updated_actor'))
                        assert e['critic_updates'] == c and e['actor_updates'] == a
                        if e.get('updated_critic'):
                            assert a == (r-1)*actor_updates(cell) and (r-1)*critic_updates(cell) < c <= r*critic_updates(cell)
                        else:
                            assert c == r*critic_updates(cell) and (r-1)*actor_updates(cell) < a <= r*actor_updates(cell)
                        assert bool(e.get('updated_temperature')) == bool(e.get('updated_actor') and cell['params']['inner_entropy_enabled'])
                    for metric,value in e['metrics'].items():
                        assert value is not None
                        if metric.startswith(('critic_', 'q_', 'td_error')):
                            axis, index = 'critic_update', e['critic_updates']
                        elif metric.startswith(('temperature_', 'alpha_')):
                            if not e.get('updated_actor'): continue
                            axis, index = 'actor_update', e['actor_updates']
                        else:
                            axis, index = 'actor_update', e['actor_updates']
                        curves[(axis,index)][metric].add(value)
                        decisions[(ep,decision)][metric].add(value)
                elif phase == 'decision':
                    for metric,value in e['metrics'].items():
                        assert value is not None
                        decisions[(ep,decision)][metric].add(value)
                    counts[key]['decision'] += 1
    expected_n = len(run['episodes'])*expected_steps
    assert len(counts) == expected_n
    for count in counts.values():
        assert dict(count) == dict(initial=1, collection=cell['J'], critic=critic_updates(cell)*cell['J'],
                                  actor=actor_updates(cell)*cell['J'],
                                  **({'temperature':actor_updates(cell)*cell['J']} if cell['params']['inner_entropy_enabled'] else {}), decision=1), count
    required = {'critic_loss','critic_grad_norm','td_error_abs_mean','q_target_mean','actor_loss',
                'actor_grad_norm','actor_entropy','alpha_used'}
    assert required <= {k for d in curves.values() for k in d}, required
    packed = []
    for (axis,index),metrics in sorted(curves.items()):
        packed.append(dict(axis=axis,index=index,metrics={k:v.summary() for k,v in metrics.items()}))
    # Each seed/decision is retained; cross-seed summaries weight seeds equally.
    by_decision = defaultdict(lambda: defaultdict(Moments))
    per_seed = []
    for (ep,decision),metrics in sorted(decisions.items()):
        means = {k:v.total/v.n for k,v in metrics.items()}
        per_seed.append(dict(episode_id=ep,decision=decision,metrics=means))
        for k,v in means.items(): by_decision[decision][k].add(v)
    return dict(update_curves=packed, per_seed_decisions=per_seed,
                decision_curves=[dict(decision=i,metrics={k:v.summary() for k,v in ms.items()})
                                 for i,ms in sorted(by_decision.items())],
                trace_rows_checked=sum(sum(v.values()) for v in counts.values()),
                metric_catalog=manifest['metric_catalog'])


def publish_performance(run_dir):
    """Retry only through the publisher's identity/hash/SDK-slot reconciliation."""
    from utils.eval_series import publish_run, PublicationUncertainError
    for attempt in range(3):
        try:
            return publish_run(run_dir, owner='oscar-rgao48', acknowledgement_timeout=120)
        except PublicationUncertainError:
            if attempt == 2:
                raise
            time.sleep(15)


def publish_cell(args):
    from utils.ambi_benchmark import stage_completed_bundle
    from utils.eval_series import load_run
    from utils.eval_series_data import load_records
    from utils.ambi_diagnostic_series import record_from_model_bundle, write_diagnostic_bundle, diagnostic_history
    import wandb
    campaign = read(args.root/'campaign.json'); cell = campaign['cells'][args.index]
    directory, bundle = Path(cell['directory']), Path(cell['bundle'])
    if (directory/'publication-completion.json').exists(): return
    receipt = read(directory/'worker-completion.json')
    assert digest(bundle/'manifest.json') == receipt['manifest_sha256']
    assert all(digest(bundle/n) == sha for n,sha in receipt['trace_sha256'].items())
    checkpoint_step = campaign.get('checkpoint_step', 625000)
    manifest = validate(bundle,cell, checkpoint_step=checkpoint_step,
                        checkpoint_sha=campaign.get('checkpoint_sha256', CHECKPOINT_SHA))
    record, = load_records(bundle,inventory_path=campaign['inventory'])
    assert record['identity'] == load_run(cell['run_dir'])['identity']
    assert record['metrics']['eval/paired_episodes'] == 5
    if campaign.get('execution',{}).get('actor_budget_comparison'):
        from slurm.ambi_aux_actor_budget import fresh_actor_baseline
        cell['baseline'] = fresh_actor_baseline(campaign, cell, record)
    if campaign.get('execution',{}).get('interleaved_comparison'):
        from slurm.ambi_aux_interleaved import validate_phased_comparison
        validate_phased_comparison(campaign, cell, record, receipt)
    comparison = polyak_comparison(record['episodes'],cell['baseline']) if 'baseline' in cell else None
    staged = stage_completed_bundle(bundle,{cell['actual_selector']:cell['run_dir']},inventory_path=campaign['inventory'])
    assert staged[cell['actual_selector']]['status'] == 'queued'
    performance = publish_performance(cell['run_dir'])
    summary = training_summary(bundle,cell)
    write(directory/'training-summary.json',summary)
    comparison_file = ('replay-comparison.json' if cell.get('baseline',{}).get('kind') == 'round_replay'
                       else 'critic-budget-comparison.json' if cell.get('baseline',{}).get('kind') == 'critic_budget'
                       else 'round-budget-comparison.json' if cell.get('baseline',{}).get('kind') == 'round_budget'
                       else 'interleaved-comparison.json' if cell.get('baseline',{}).get('kind') == 'interleaved'
                       else 'actor-budget-comparison.json' if cell.get('baseline',{}).get('kind') == 'actor_budget'
                       else 'polyak-comparison.json')
    if comparison:
        write(directory/comparison_file, comparison)
    diagnostic = record_from_model_bundle(bundle,cell['actual_selector'],campaign['group']+'-'+cell['name'],
                                         bootstrap_resamples=2000,bootstrap_seed=20260912)
    assert diagnostic['status'] == 'complete' and len(diagnostic['rows']) == 2500*(cell['J']+1)
    write_diagnostic_bundle(directory/'model-series',diagnostic)
    journal = directory/'training-publication.json'
    if journal.exists(): raise RuntimeError('Training publication uncertain; inspect remote run before retry')
    write(journal,dict(status='uncertain',run_id=cell['training_run_id']))
    run = wandb.init(entity=ENTITY,project=PROJECT,id=cell['training_run_id'],resume='never',
                     name='Inner training | '+cell['name']+f' | {checkpoint_step//1000}k',group=campaign['group'],
                     job_type='inner-training-diagnostics',tags=['closed-loop','H-J-sweep',cell['name'].rsplit('_h',1)[0]],
                     config=dict(H=cell['H'],J=cell['J'],N=128,B=256,C=critic_updates(cell),A=actor_updates(cell),checkpoint_step=checkpoint_step,
                                 checkpoint_sha256=campaign.get('checkpoint_sha256', CHECKPOINT_SHA),
                                 campaign_group=campaign['group'],
                                 source_run=campaign.get('source_run'),
                                 critic_kind=cell.get('critic_kind'),
                                 overview_url=(f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{campaign["overview_run_id"]}'
                                               if campaign.get('overview_run_id') else None),
                                 execution=receipt.get('execution'),
                                 setting=cell['name'],resolved_config=manifest['runs'][0]['resolved_config'],
                                 source_code=manifest['code'],reused=cell['reused'],
                                 inner_critic_target_tau=manifest['runs'][0]['resolved_config']['inner_critic_target_tau'],
                                 inner_replay_capacity=manifest['runs'][0]['resolved_config']['inner_replay_capacity'],
                                 inner_replay_reset_each_round=manifest['runs'][0]['resolved_config'].get('inner_replay_reset_each_round',False),
                                 baseline_performance_run_id=cell.get('baseline',{}).get('performance_run_id'),
                                 performance_run_id=cell['performance_run_id'],
                                 aggregation='Update curves average all decision roots; decision curves weight five seeds equally.',
                                 probe_objective='Reward plus terminal Q; excludes explicit entropy.'),mode='online')
    try:
        for axis in ('critic_update','actor_update','decision'):
            run.define_metric('axis/'+axis)
            prefix = {'critic_update':'critic','actor_update':'actor','decision':'episode'}[axis]
            run.define_metric(prefix+'/*',step_metric='axis/'+axis)
        run.define_metric('seed/*',step_metric='axis/decision')
        run.define_metric('diagnostic/actor_updates')
        run.define_metric('diagnostic/*',step_metric='diagnostic/actor_updates')
        for row in summary['update_curves']:
            prefix = 'critic' if row['axis']=='critic_update' else 'actor'
            run.log({'axis/'+row['axis']:row['index'],
                     **{f'{prefix}/{k}/{s}':v for k,stats in row['metrics'].items() for s,v in stats.items()}})
        per_seed = defaultdict(dict)
        for row in summary['per_seed_decisions']:
            per_seed[row['decision']].update({f"seed/{row['episode_id']}/{k}":v for k,v in row['metrics'].items()})
        for row in summary['decision_curves']:
            run.log({'axis/decision':row['decision'],**per_seed[row['decision']],
                     **{f'episode/{k}/{s}':v for k,stats in row['metrics'].items() for s,v in stats.items()}})
        for row in diagnostic_history(diagnostic): run.log(row)
        if comparison: run.log(comparison['metrics'])
        artifact = wandb.Artifact('inner-training-'+cell['training_run_id'],type='inner-training-traces',
                                  metadata=dict(manifest_sha256=receipt['manifest_sha256'],reused=cell['reused']))
        for name in ['manifest.json',*manifest['runs'][0]['trace_files']]: artifact.add_file(str(bundle/name),name='bundle/'+name)
        artifact.add_file(str(directory/'training-summary.json'),name='training-summary.json')
        if (bundle/'execution.json').exists():
            artifact.add_file(str(bundle/'execution.json'),name='execution.json')
        if comparison: artifact.add_file(str(directory/comparison_file),name=comparison_file)
        for name in ('manifest.json','paired-rows.jsonl.gz','report.html'):
            artifact.add_file(str(directory/'model-series'/name),name='model-series/'+name)
        run.log_artifact(artifact)
        run.summary.update({**record['metrics'],**(comparison['metrics'] if comparison else {}),
                            'status':'complete','reused':cell['reused'],
                            'diagnostic/paired_rows':len(diagnostic['rows']),
                            'training/decisions':2500,'training/critic_updates':2500*critic_updates(cell)*cell['J'],
                            'training/actor_updates':2500*actor_updates(cell)*cell['J'],
                            'performance_url':f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{cell["performance_run_id"]}'})
        run.finish()
    except BaseException:
        run.finish(exit_code=1); raise
    write(journal,dict(status='complete',run_id=cell['training_run_id']))
    write(directory/'publication-completion.json',dict(status='complete',cell=cell['name'],reused=cell['reused'],
          performance=performance,training_run_id=cell['training_run_id'],
          metrics={**record['metrics'],**(comparison['metrics'] if comparison else {})}))


def watch(args):
    """One CPU owner with a bounded number of complete-panel subprocesses."""
    import wandb
    campaign = read(args.root/'campaign.json')
    total = len(campaign['cells'])
    publishers = int(campaign.get('publisher_workers', 2))
    assert 1 <= publishers <= 4
    marker = args.root/'watcher-started.json'
    if marker.exists(): raise RuntimeError('Watcher already started; inspect its state before recovery')
    write(marker,dict(pid=os.getpid(),started=time.time()))
    run = wandb.init(entity=ENTITY,project=PROJECT,id=campaign['overview_run_id'],resume='never',
                     name=campaign.get('label','625k H/J sweep')+f' | {total} settings',
                     group=campaign['group'],job_type='campaign-overview',
                     config=dict(H=sorted({c['H'] for c in campaign['cells']}),
                                 J=sorted({c['J'] for c in campaign['cells']}),N=128,B=256,
                                 C=sorted({critic_updates(c) for c in campaign['cells']}),
                                 A=sorted({actor_updates(c) for c in campaign['cells']}),seeds=SEEDS,
                                 execution=campaign.get('execution'),
                                 target_taus=sorted({c['params'].get('inner_critic_target_tau',.01) for c in campaign['cells']}),
                                 checkpoint_sha256=CHECKPOINT_SHA,source_commit=campaign['source_commit']),mode='online')
    attempted = set(); futures = {}; failures = {}
    def launch(index):
        cell = campaign['cells'][index]
        with (Path(cell['directory'])/'publisher.log').open('w') as log:
            return subprocess.run([sys.executable,__file__,'publish','--root',str(args.root),'--index',str(index)],
                                  stdout=log,stderr=subprocess.STDOUT).returncode
    def status_rows():
        rows=[]
        for index,cell in enumerate(campaign['cells']):
            d=Path(cell['directory']); completed=d/'publication-completion.json'
            state='published' if completed.exists() else 'publishing' if index in futures else 'failed' if index in failures else 'ready' if (d/'worker-completion.json').exists() else 'queued/running'
            result=read(completed) if completed.exists() else {}
            rows.append([cell['name'],cell['H'],cell['J'],state,cell['reused'],
                         result.get('metrics',{}).get('eval/return_mean'),result.get('metrics',{}).get('eval/paired_gain_mean'),
                         f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{cell["training_run_id"]}',
                         f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{cell["performance_run_id"]}',
                         result.get('metrics',{}).get(comparison_prefix(
                             {'kind':'actor_budget'} if campaign.get('execution',{}).get('actor_budget_comparison') else
                             cell.get('baseline',{}))+'_gain_mean')])
        return rows
    terminal_since=None; previous=None
    try:
        with ThreadPoolExecutor(max_workers=publishers) as pool:
            while True:
                for index,f in list(futures.items()):
                    if f.done():
                        rc=f.result()
                        if rc: failures[index]=rc
                        del futures[index]
                for index,cell in enumerate(campaign['cells']):
                    if len(futures)>=publishers: break
                    d=Path(cell['directory'])
                    baseline_ready = (not cell.get('actor_baseline_name') or
                                      (args.root/cell['actor_baseline_name']/'worker-completion.json').exists())
                    if index not in attempted and baseline_ready and (d/'worker-completion.json').exists() and not (d/'publication-completion.json').exists():
                        attempted.add(index); futures[index]=pool.submit(launch,index)
                rows=status_rows(); stamp=[r[3] for r in rows]
                if stamp!=previous:
                    table=wandb.Table(columns=['setting','H','J','status','reused','return','paired_gain','training_url','performance_url','paired_gain_vs_baseline'],data=rows)
                    done=sum(r[3]=='published' for r in rows)
                    run.log({'campaign/published':done,'campaign/failed_publications':len(failures),'campaign/settings':table})
                    write(args.root/'progress.json',dict(published=done,total=total,rows=rows,failures=failures))
                    print(f'Published {done}/{total}; publication failures {len(failures)}',flush=True)
                    previous=stamp
                if all(r[3]=='published' for r in rows): break
                submission=args.root/'submission.json'
                if submission.exists():
                    job_ids=read(submission)['gpu_job_ids']
                    active=subprocess.check_output(['squeue','--noheader','--jobs',','.join(job_ids),'-o','%i'],text=True).strip()
                    if not active and not futures:
                        terminal_since=terminal_since or time.time()
                        if time.time()-terminal_since>90: break
                    else: terminal_since=None
                time.sleep(15)
        complete=all(r[3]=='published' for r in status_rows())
        run.summary['status']='complete' if complete else 'incomplete'
        write(args.root/'campaign-completion.json',dict(status='complete' if complete else 'incomplete',rows=status_rows(),failures=failures))
        run.finish(exit_code=0 if complete else 1)
    except BaseException:
        run.finish(exit_code=1); raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','worker','publish','watch'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--matrix',type=Path,default=MATRIX)
    p.add_argument('--group',default=GROUP)
    p.add_argument('--label',default='625k H/J sweep')
    p.add_argument('--baseline-campaign',type=Path)
    p.add_argument('--baseline-extension-campaign',type=Path)
    p.add_argument('--baseline-kind',choices=['polyak','round_replay','interleaved','round_budget','critic_budget'],default='polyak')
    for name in ('checkpoint','inventory','reference','registry','reuse-alpha','reuse-zero'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--index',type=int)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--replica',default='default')
    p.add_argument('--phased-control',action='store_true')
    args=p.parse_args()
    {'prepare':prepare,'worker':worker,'publish':publish_cell,'watch':watch}[args.mode](args)


if __name__=='__main__': main()
