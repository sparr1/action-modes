"""Six new N/B panels with two immutable, already published C16 references."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import uuid

BASELINE_SOURCE = 'fe87ae07b2a7ed751cd6865a3f60b2eae88e6abb'


def match_identity(candidate, baseline, n, b, *, baseline_science=None):
    for key in ('backbone', 'protocol'):
        assert candidate['identity'][key] == baseline['identity'][key], key
    assert baseline['identity']['science'] == (baseline_science or candidate['identity']['science'])
    expected = deepcopy(baseline['identity']['planner'])
    s = expected['settings']
    assert n in (32, 128) and b in (64, 256)
    assert s['inner_rollouts_per_round'] == 128 and s['inner_batch_size'] == 256
    assert s['inner_rounds'] == 8 and s['inner_rollout_horizon'] in (1, 3)
    assert s['inner_critic_updates_per_round'] == 16 and s['inner_actor_updates_per_round'] == 4
    assert s['inner_critic_target_tau'] == .01 and s['inner_replay_capacity'] == 3072
    assert s['inner_sac_critic_target'] == 'entropy_augmented' and s['inner_terminal_entropy'] == 'outer'
    assert s.get('inner_critic_source', 'sac') == s.get('inner_horizon_critic_source', 'sac') == 'sac'
    assert s['inner_entropy_enabled'] and s['inner_temperature_mode'] == 'auto'
    assert s.get('inner_horizon_conditioning', 'none') == 'none'
    assert s.get('inner_update_timing', 'round') == 'round'
    assert s.get('inner_component_update_order', 'critic_first') == 'critic_first'
    assert not s.get('inner_replay_reset_each_round', False)
    assert s['inner_model_step_budget'] == 128*s['inner_rollout_horizon']*s['inner_rounds']
    s.update(inner_rollouts_per_round=n, inner_batch_size=b,
             inner_model_step_budget=n*s['inner_rollout_horizon']*s['inner_rounds'])
    assert candidate['identity']['planner'] == expected


def baseline_result(candidate, baseline, cell):
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS, rollouts, batch_size
    from utils.eval_series_data import scientific_identity
    match_identity(candidate, baseline, rollouts(cell), batch_size(cell),
                   baseline_science=scientific_identity('AMBITDMPC2/AMBITDMPC2', None, BASELINE_SOURCE))
    assert baseline['checkpoint']['sha256'] == CHECKPOINT_SHA and baseline['checkpoint']['step'] == 625000
    assert baseline['metrics']['eval/frozen_state_unchanged']
    assert sorted(e['seed'] for e in baseline['episodes']) == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in baseline['episodes'])
    return dict(kind='rollout_batch', record_id=baseline['record_id'], source_commit=BASELINE_SOURCE,
                comparison_note='Only N and shared actor/critic batch size change. Historical N128/B256 C16 reference; '
                                'same checkpoint, seeds, H/J, objectives, learning rates, tau, replay and round timing. '
                                'Horizon conditioning remains off. Original compiled mixed-GPU execution is exploratory.',
                source_comparison=dict(baseline=baseline['identity']['science'], candidate=candidate['identity']['science']),
                episodes=[{k:e[k] for k in ('seed','solver_seed','return')} for e in baseline['episodes']])


def reference_cell(previous, directory, manifest, record, receipt, *, kind='rollout_batch'):
    """Reference the existing bundle and W&B IDs; never allocate publication IDs."""
    from slurm.ambi_aux_hj_sweep import read, write, polyak_comparison
    cell = deepcopy(previous)
    cell.update(name=previous['name']+('_n128_b256_reused' if kind=='rollout_batch' else '_round_reused'), directory=str(directory), reused=True)
    cell.pop('baseline', None)
    publication = read(Path(previous['directory'])/'publication-completion.json')
    assert publication['status'] == 'complete' and publication['training_run_id'] == previous['training_run_id']
    baseline = dict(kind=kind, episodes=record['episodes'])
    publication.update(cell=cell['name'], reused=True,
                       metrics={**record['metrics'], **polyak_comparison(record['episodes'], baseline)['metrics']})
    write(directory/'worker-completion.json', {**receipt, 'reused':True, 'bundle':previous['bundle']})
    write(directory/'publication-completion.json', publication)
    return cell


def prepare_campaign(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    from utils.eval_series_data import load_records
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS, PROJECT, ENTITY, cells, digest, read, write
    from slurm.ambi_aux_horizon_campaign import checked_control
    root=args.root; root.mkdir(parents=True, exist_ok=False)
    assert digest(args.checkpoint) == CHECKPOINT_SHA
    old=read(args.baseline_campaign/'campaign.json')
    assert old['source_commit'] == BASELINE_SOURCE
    assert read(args.baseline_campaign/'campaign-completion.json')['status'] == 'complete'
    controls={c['H']:c for c in old['cells'] if c['name'] in ('soft_soft_h1_j8_c16','soft_soft_h3_j8_c16')}
    assert set(controls) == {1,3}
    references={h:checked_control(c,args.inventory,traces=True) for h,c in controls.items()}
    prior,=load_records(args.reference,inventory_path=args.inventory)
    assert prior['checkpoint']['sha256'] == CHECKPOINT_SHA and prior['checkpoint']['step'] == 625000
    assert prior['identity']['planner'] == {'type':'prior','action_rule':'tanh_mean'}
    assert prior['metrics']['eval/frozen_state_unchanged'] and sorted(e['seed'] for e in prior['episodes']) == SEEDS
    assert all(e['length']==500 and not e['truncated_by_evaluator'] for e in prior['episodes'])
    evaluate_matrix(args.matrix,args.checkpoint,seeds=SEEDS,controller_seed=55,max_steps=500,
                    bundle_dir=root/'unused',checkpoint_inventory=args.inventory,reference_bundle=args.reference,
                    eval_series_spec_dir=root/'specs')
    panel=[]
    for cell in cells(args.matrix):
        directory=root/cell['name']; directory.mkdir()
        spec=read(root/'specs'/f"{cell['selector'].replace('/','__')}.json")
        assert spec['identity']['backbone']==prior['identity']['backbone']
        assert spec['identity']['protocol']==prior['identity']['protocol']
        record,receipt=references[cell['H']];control=controls[cell['H']]
        baseline=baseline_result(spec,record,cell)
        baseline.update(bundle=control['bundle'],manifest_sha256=receipt['manifest_sha256'],
                        performance_run_id=control['performance_run_id'])
        registry=create_run(args.registry,spec,args.group+'-'+cell['name'],PROJECT,ENTITY,'oscar-rgao48')
        cell.update(directory=str(directory),bundle=str(directory/'bundle'),actual_selector=cell['selector'],
                    reused=False,run_dir=registry['run_dir'],performance_run_id=registry['run_id'],
                    training_run_id=uuid.uuid4().hex,baseline=baseline)
        panel.append(cell)
    assert len(panel)==6
    for h in (1,3):
        directory=root/f'reused_h{h}_n128_b256'; directory.mkdir()
        record,receipt=references[h]
        panel.append(reference_cell(controls[h],directory,read(Path(controls[h]['bundle'])/'manifest.json'),record,receipt))
    campaign=dict(schema_version=1,group=args.group,label=args.label,matrix=str(args.matrix.resolve()),
                  checkpoint=str(args.checkpoint),checkpoint_sha256=CHECKPOINT_SHA,inventory=str(args.inventory),
                  reference=str(args.reference),prior_manifest_sha256=digest(args.reference/'manifest.json'),
                  prior_source_science=prior['identity']['science'],
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  overview_run_id=uuid.uuid4().hex,cells=panel,rollout_batch_sweep=True,publisher_workers=4,
                  baseline_campaign=str(args.baseline_campaign),rollout_batch_controls=controls,
                  production_indices=list(range(6)),new_episodes=30,reused_episodes=10)
    write(root/'campaign.json',campaign)
    print(json.dumps(dict(root=str(root),conditions=8,new_settings=6,reused=2,new_episodes=30,
                          overview_run_id=campaign['overview_run_id'])),flush=True)


def publication_baseline(campaign, cell, record):
    from slurm.ambi_aux_horizon_campaign import checked_control
    control=campaign['rollout_batch_controls'][str(cell['H'])]
    baseline,receipt=checked_control(control,campaign['inventory'],traces=True)
    result=baseline_result(record,baseline,cell)
    assert result['record_id']==cell['baseline']['record_id']
    assert receipt['manifest_sha256']==cell['baseline']['manifest_sha256']
    result.update(bundle=control['bundle'],manifest_sha256=receipt['manifest_sha256'],
                  performance_run_id=control['performance_run_id'])
    return result
