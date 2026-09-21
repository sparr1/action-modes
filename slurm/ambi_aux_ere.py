"""Whole-round ERE screen with nine immutable, previously published controls."""
from copy import deepcopy
from pathlib import Path
import json
import math
import subprocess
import uuid

REDUCED_SOURCE = 'fe87ae07b2a7ed751cd6865a3f60b2eae88e6abb'
C32_SOURCE = '28964eef209a2aa6deb73b898549ca405f38f293'


def control_key(cell):
    from slurm.ambi_aux_hj_sweep import critic_updates
    return f"h{cell['H']}_c{critic_updates(cell)}"


def match_identity(candidate, baseline, fraction, *, baseline_science=None, ere_actor=True):
    for key in ('backbone', 'protocol'):
        assert candidate['identity'][key] == baseline['identity'][key], key
    assert baseline['identity']['science'] == (baseline_science or candidate['identity']['science'])
    expected = deepcopy(baseline['identity']['planner'])
    s = expected['settings']
    assert fraction in (.25, .5)
    assert s['inner_rollout_horizon'] in (1, 2, 3)
    assert s['inner_critic_updates_per_round'] in (8, 16, 32)
    for key, value in dict(inner_rounds=8, inner_rollouts_per_round=128,
                           inner_batch_size=256, inner_actor_updates_per_round=4,
                           inner_replay_capacity=3072, inner_critic_target_tau=.01,
                           inner_actor_lr=3e-4, inner_critic_lr=3e-4,
                           inner_sac_critic_target='entropy_augmented', inner_terminal_entropy='outer',
                           inner_entropy_enabled=True, inner_temperature_mode='auto').items():
        assert s[key] == value, key
    assert s.get('inner_replay_strategy', 'uniform') == 'uniform'
    assert s.get('inner_update_timing', 'round') == 'round'
    assert s.get('inner_component_update_order', 'critic_first') == 'critic_first'
    assert s.get('inner_horizon_conditioning', 'none') == 'none'
    assert not s.get('inner_replay_reset_each_round', False)
    assert s.get('inner_steps_per_update') is None
    assert s.get('inner_critic_source', 'sac') == s.get('inner_horizon_critic_source', 'sac') == 'sac'
    s.update(inner_replay_strategy='ere', inner_ere_final_fraction=fraction, inner_ere_min_rounds=1)
    if not ere_actor:
        s['inner_ere_actor'] = False
    assert candidate['identity']['planner'] == expected


def baseline_result(candidate, baseline, cell):
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS, critic_updates
    from utils.eval_series_data import scientific_identity
    source = C32_SOURCE if critic_updates(cell) == 32 else REDUCED_SOURCE
    match_identity(candidate, baseline, cell['params']['inner_ere_final_fraction'],
                   ere_actor=cell['params'].get('inner_ere_actor', True),
                   baseline_science=scientific_identity('AMBITDMPC2/AMBITDMPC2', None, source))
    assert baseline['checkpoint']['sha256'] == CHECKPOINT_SHA and baseline['checkpoint']['step'] == 625000
    assert baseline['metrics']['eval/frozen_state_unchanged']
    assert sorted(e['seed'] for e in baseline['episodes']) == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in baseline['episodes'])
    return dict(kind='ere', record_id=baseline['record_id'], source_commit=source,
                comparison_note='Only replay sampling changes from uniform to whole-round ERE. '
                                'Matched H/J/C/A/N/B, checkpoint, seeds, learning rates, alpha, tau and objectives. '
                                'Historical mixed-GPU controls; exploratory five-seed comparison.',
                source_comparison=dict(baseline=baseline['identity']['science'], candidate=candidate['identity']['science']),
                episodes=[{k: e[k] for k in ('seed', 'solver_seed', 'return')} for e in baseline['episodes']])


def prepare_campaign(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    from utils.eval_series_data import load_records
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS, PROJECT, ENTITY, cells, digest, read, write
    from slurm.ambi_aux_horizon_campaign import checked_control
    from slurm.ambi_aux_rollout_batch import reference_cell
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    assert digest(args.checkpoint) == CHECKPOINT_SHA
    controls = {}
    for path, source, counts in ((args.baseline_campaign, REDUCED_SOURCE, (8, 16)),
                                 (args.baseline_extension_campaign, C32_SOURCE, (32,))):
        previous = read(path/'campaign.json')
        assert previous['source_commit'] == source
        assert read(path/'campaign-completion.json')['status'] == 'complete'
        for h in (1, 2, 3):
            for c in counts:
                name = f'soft_soft_h{h}_j8_jscale' if c == 32 else f'soft_soft_h{h}_j8_c{c}'
                control, = [cell for cell in previous['cells'] if cell['name'] == name]
                controls[f'h{h}_c{c}'] = control
    assert len(controls) == 9
    references = {key: checked_control(cell, args.inventory, traces=True) for key, cell in controls.items()}
    prior, = load_records(args.reference, inventory_path=args.inventory)
    assert prior['checkpoint']['sha256'] == CHECKPOINT_SHA and prior['checkpoint']['step'] == 625000
    assert prior['identity']['planner'] == {'type': 'prior', 'action_rule': 'tanh_mean'}
    assert prior['metrics']['eval/frozen_state_unchanged'] and sorted(e['seed'] for e in prior['episodes']) == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in prior['episodes'])
    evaluate_matrix(args.matrix, args.checkpoint, seeds=SEEDS, controller_seed=55, max_steps=500,
                    bundle_dir=root/'unused', checkpoint_inventory=args.inventory, reference_bundle=args.reference,
                    eval_series_spec_dir=root/'specs')
    panel = []
    for cell in cells(args.matrix):
        directory = root/cell['name']; directory.mkdir()
        spec = read(root/'specs'/f"{cell['selector'].replace('/', '__')}.json")
        assert spec['identity']['backbone'] == prior['identity']['backbone']
        assert spec['identity']['protocol'] == prior['identity']['protocol']
        record, receipt = references[control_key(cell)]
        control = controls[control_key(cell)]
        baseline = baseline_result(spec, record, cell)
        baseline.update(bundle=control['bundle'], manifest_sha256=receipt['manifest_sha256'],
                        performance_run_id=control['performance_run_id'])
        registry = create_run(args.registry, spec, args.group+'-'+cell['name'], PROJECT, ENTITY, 'oscar-rgao48')
        cell.update(directory=str(directory), bundle=str(directory/'bundle'), actual_selector=cell['selector'],
                    reused=False, run_dir=registry['run_dir'], performance_run_id=registry['run_id'],
                    training_run_id=uuid.uuid4().hex, baseline=baseline)
        panel.append(cell)
    assert len(panel) == 18
    assert {(c['H'], c['params']['inner_critic_updates_per_round'], c['params']['inner_ere_final_fraction'])
            for c in panel} == {(h, c, f) for h in (1, 2, 3) for c in (8, 16, 32) for f in (.25, .5)}
    for key, control in controls.items():
        directory = root/f'reused_uniform_{key}'; directory.mkdir()
        record, receipt = references[key]
        panel.append(reference_cell(control, directory, None, record, receipt, kind='ere'))
    campaign = dict(schema_version=1, group=args.group, label=args.label, matrix=str(args.matrix.resolve()),
                    checkpoint=str(args.checkpoint), checkpoint_sha256=CHECKPOINT_SHA, inventory=str(args.inventory),
                    reference=str(args.reference), prior_manifest_sha256=digest(args.reference/'manifest.json'),
                    prior_source_science=prior['identity']['science'],
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    overview_run_id=uuid.uuid4().hex, cells=panel, ere_sweep=True, publisher_workers=4,
                    baseline_campaign=str(args.baseline_campaign), baseline_extension_campaign=str(args.baseline_extension_campaign),
                    ere_controls=controls, production_indices=list(range(18)), new_episodes=90, reused_episodes=45)
    write(root/'campaign.json', campaign)
    print(json.dumps(dict(root=str(root), conditions=27, new_settings=18, reused=9, new_episodes=90,
                          overview_run_id=campaign['overview_run_id'])), flush=True)


def publication_baseline(campaign, cell, record):
    if campaign.get('ere_actor_ablation'):
        from slurm.ambi_aux_ere_actor import publication_baseline as actor_baseline
        return actor_baseline(campaign, cell, record)
    from slurm.ambi_aux_horizon_campaign import checked_control
    control = campaign['ere_controls'][control_key(cell)]
    baseline, receipt = checked_control(control, campaign['inventory'], traces=True)
    result = baseline_result(record, baseline, cell)
    assert result['record_id'] == cell['baseline']['record_id']
    assert receipt['manifest_sha256'] == cell['baseline']['manifest_sha256']
    result.update(bundle=control['bundle'], manifest_sha256=receipt['manifest_sha256'],
                  performance_run_id=control['performance_run_id'])
    return result


def validate_update(event, cell):
    """Require actual ERE telemetry at every update, including GPU smoke runs."""
    from RL.tdmpc2_core.common.ere import round_windows
    from slurm.ambi_aux_hj_sweep import critic_updates, actor_updates, rollouts
    r = event['round_index']
    for component, slots in (('critic', critic_updates(cell)), ('actor', actor_updates(cell))):
        if not event.get('updated_'+component):
            continue
        k = event[component+'_updates'] - (r-1)*slots - 1
        assert 0 <= k < slots
        w = round_windows(r, slots, cell['params'].get('inner_ere_final_fraction', .25),
                         cell['params'].get('inner_ere_min_rounds', 1))[k]
        if component == 'actor' and not cell['params'].get('inner_ere_actor', True):
            w = r
        metrics = event['metrics']
        prefix = component+'_replay_'
        assert metrics[prefix+'window_rounds'] == w
        assert metrics[prefix+'window_transitions'] == w*rollouts(cell)*cell['H']
        assert math.isclose(metrics[prefix+'window_fraction'], w/r, rel_tol=1e-6)
        assert math.isclose(metrics[prefix+'window_round_fraction'], w/r, rel_tol=1e-6)
        assert 0 <= metrics[prefix+'round_age_min'] <= metrics[prefix+'round_age_mean'] <= metrics[prefix+'round_age_max'] <= w-1
        assert 0 <= metrics[prefix+'newest_round_fraction'] <= 1
        assert 0 < metrics[prefix+'batch_unique_fraction'] <= 1


def validate_decision(event, cell):
    from slurm.ambi_aux_hj_sweep import critic_updates, actor_updates, batch_size
    for component, slots in (('critic', critic_updates(cell)), ('actor', actor_updates(cell))):
        counts = [event['metrics'][f'decision/inner_{component}_replay_round_{r}_sample_count']
                  for r in range(1, cell['J']+1)]
        assert all(c >= 0 and float(c).is_integer() for c in counts)
        assert sum(counts) == slots*cell['J']*batch_size(cell)
