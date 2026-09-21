"""One critic-ERE/uniform-actor condition with two immutable reused controls."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import uuid

ERE_SOURCE = 'b3a236f45eb20163eb6e22d8faaaa11f320753ea'


def match_ere_control(candidate, baseline):
    from utils.eval_series_data import scientific_identity
    assert baseline['identity']['science'] == scientific_identity('AMBITDMPC2/AMBITDMPC2', None, ERE_SOURCE)
    for key in ('backbone', 'protocol'):
        assert candidate['identity'][key] == baseline['identity'][key], key
    expected = deepcopy(baseline['identity']['planner'])
    settings = expected['settings']
    assert settings['inner_replay_strategy'] == 'ere'
    assert settings['inner_ere_final_fraction'] == .25
    assert settings.get('inner_ere_actor', True)
    settings['inner_ere_actor'] = False
    assert candidate['identity']['planner'] == expected


def checked_baselines(candidate, cell, controls, inventory):
    from slurm.ambi_aux_ere import baseline_result
    from slurm.ambi_aux_horizon_campaign import checked_control
    records = {k: checked_control(c, inventory, traces=True) for k, c in controls.items()}
    uniform, receipt = records['uniform']
    baseline = baseline_result(candidate, uniform, cell)
    ere, ere_receipt = records['ere_both']
    match_ere_control(candidate, ere)
    assert ere['checkpoint'] == uniform['checkpoint']
    assert sorted((e['seed'], e['solver_seed']) for e in ere['episodes']) == sorted(
        (e['seed'], e['solver_seed']) for e in uniform['episodes'])
    baseline.update(bundle=controls['uniform']['bundle'], manifest_sha256=receipt['manifest_sha256'],
                    performance_run_id=controls['uniform']['performance_run_id'],
                    ere_both=dict(kind='ere_both', episodes=ere['episodes'], record_id=ere['record_id'],
                                  manifest_sha256=ere_receipt['manifest_sha256'],
                                  performance_run_id=controls['ere_both']['performance_run_id']))
    return baseline, records


def prepare_campaign(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    from utils.eval_series_data import load_records
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS, PROJECT, ENTITY, cells, digest, read, write
    from slurm.ambi_aux_ere import REDUCED_SOURCE
    root = args.root
    root.mkdir(parents=True, exist_ok=False)
    assert digest(args.checkpoint) == CHECKPOINT_SHA
    controls = {}
    for key, path, source, name in (
        ('uniform', args.baseline_campaign, REDUCED_SOURCE, 'soft_soft_h3_j8_c16'),
        ('ere_both', args.baseline_extension_campaign, ERE_SOURCE, 'soft_soft_h3_j8_c16_ere025'),
    ):
        previous = read(path/'campaign.json')
        assert previous['source_commit'] == source
        assert read(path/'campaign-completion.json')['status'] == 'complete'
        controls[key], = [c for c in previous['cells'] if c['name'] == name]
    prior, = load_records(args.reference, inventory_path=args.inventory)
    assert prior['checkpoint']['sha256'] == CHECKPOINT_SHA and prior['checkpoint']['step'] == 625000
    assert prior['identity']['planner'] == {'type': 'prior', 'action_rule': 'tanh_mean'}
    assert prior['metrics']['eval/frozen_state_unchanged'] and sorted(e['seed'] for e in prior['episodes']) == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in prior['episodes'])
    evaluate_matrix(args.matrix, args.checkpoint, seeds=SEEDS, controller_seed=55, max_steps=500,
                    bundle_dir=root/'unused', checkpoint_inventory=args.inventory, reference_bundle=args.reference,
                    eval_series_spec_dir=root/'specs')
    cell, = cells(args.matrix)
    assert (cell['H'], cell['J'], cell['params']['inner_critic_updates_per_round']) == (3, 8, 16)
    assert cell['params']['inner_ere_actor'] is False
    spec = read(root/'specs'/f"{cell['selector'].replace('/', '__')}.json")
    assert spec['identity']['backbone'] == prior['identity']['backbone']
    assert spec['identity']['protocol'] == prior['identity']['protocol']
    baseline, records = checked_baselines(spec, cell, controls, args.inventory)
    directory = root/cell['name']; directory.mkdir()
    registry = create_run(args.registry, spec, args.group+'-'+cell['name'], PROJECT, ENTITY, 'oscar-rgao48')
    cell.update(directory=str(directory), bundle=str(directory/'bundle'), actual_selector=cell['selector'],
                reused=False, run_dir=registry['run_dir'], performance_run_id=registry['run_id'],
                training_run_id=uuid.uuid4().hex, baseline=baseline)
    panel = [cell]
    for key, control in controls.items():
        directory = root/('reused_'+key); directory.mkdir()
        reused = deepcopy(control)
        reused.update(name=control['name']+'_reused', directory=str(directory), reused=True)
        reused.pop('baseline', None)
        publication = read(Path(control['directory'])/'publication-completion.json')
        assert publication['status'] == 'complete'
        assert publication['training_run_id'] == control['training_run_id']
        write(directory/'worker-completion.json', {**records[key][1], 'reused': True, 'bundle': control['bundle']})
        write(directory/'publication-completion.json', {**publication, 'reused': True, 'cell': reused['name']})
        panel.append(reused)
    campaign = dict(schema_version=1, group=args.group, label=args.label, matrix=str(args.matrix.resolve()),
                    checkpoint=str(args.checkpoint), checkpoint_sha256=CHECKPOINT_SHA, inventory=str(args.inventory),
                    reference=str(args.reference), prior_manifest_sha256=digest(args.reference/'manifest.json'),
                    prior_source_science=prior['identity']['science'],
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    overview_run_id=uuid.uuid4().hex, cells=panel, ere_sweep=True, ere_actor_ablation=True,
                    actor_ablation_controls=controls, publisher_workers=1, production_indices=[0],
                    new_episodes=5, reused_episodes=10)
    write(root/'campaign.json', campaign)
    print(json.dumps(dict(root=str(root), conditions=3, new_settings=1, reused=2, new_episodes=5,
                          overview_run_id=campaign['overview_run_id'])), flush=True)


def publication_baseline(campaign, cell, record):
    result, _ = checked_baselines(record, cell, campaign['actor_ablation_controls'], campaign['inventory'])
    for key in ('record_id', 'manifest_sha256'):
        assert result[key] == cell['baseline'][key]
        assert result['ere_both'][key] == cell['baseline']['ere_both'][key]
    return result
