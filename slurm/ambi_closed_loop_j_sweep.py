"""H1/H2/H3 round-budget sweeps at the frozen target -10.5/shared 650k checkpoint."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_aux_hj_sweep import ENTITY, PROJECT, SEEDS, digest, read, validate, write
from slurm.ambi_closed_loop_checkpoint_sweep import (
    check_episodes, checkpoint_state_proof, enrich_reference, load_prior, matching_config,
    resolve_config, validate_completed, verify_reference, worker,
)
from slurm.ambi_closed_loop_critics import SOURCE_RUN, source_commit
from slurm.ambi_closed_loop_publish import indexed_episodes
from slurm.ambi_closed_loop_reward_retrace import historical_cell
from slurm.ambi_closed_loop_sampled import verify_receipt

MATRIX = ROOT/'configs/research/ambi_closed_loop_h3_j_sweep_650k.json'
REFERENCES = ROOT/'configs/research/ambi_closed_loop_h3_j_sweep_650k_refs.json'
GROUP = 'closed-loop-h3-j-sweep-650k-20260924'
ROUNDS = (1, 2, 4, 6, 8, 10, 12, 14)
CHECKPOINT_STEP = 650000
CHECKPOINT_SHA = '021e7b4fb323f7181f7b25c0e0cf7c0f92e9097ecd60d9474284a47e50d6609e'
INITIAL_ALPHA = .0059470199048519135


def matrix_for(horizon):
    assert type(horizon) is int and horizon in (1,2,3)
    return ROOT/f'configs/research/ambi_closed_loop_h{horizon}_j_sweep_650k.json'


def requested_params(rounds, horizon=3):
    assert type(horizon) is int and horizon in (1,2,3)
    return {**historical_cell(horizon,rounds)['requested_alg_params'],
            'inner_eval_execution_action':'mean', 'inner_sac_return_estimator':'one_step',
            'inner_retrace_lambda':1.0, 'inner_retrace_batch_trajectories':None}


def cells(matrix_path=None, horizon=3):
    from utils.ambi_research import load_preset_matrix
    matrix_path = matrix_path or matrix_for(horizon)
    assert type(horizon) is int and horizon in (1,2,3)
    matrix = load_preset_matrix(matrix_path)
    selectors = [f'sweep/return_return_alpha_h{horizon}_j{j}_c16' for j in ROUNDS]
    assert matrix['source_run'] == SOURCE_RUN and matrix['base_alg_config'] == 'checkpoint'
    assert matrix['checkpoint_steps'] == [CHECKPOINT_STEP]
    assert matrix['evaluation'] == dict(controller_seed=55,seeds=SEEDS,max_steps=500,
                                       togo_return_rollouts=32,default_presets=selectors)
    comparison = matrix['comparisons']['sweep']
    assert comparison['reference'] == 'prior'
    assert set(comparison['variants']) == {'prior',*(s.split('/')[1] for s in selectors)}
    assert comparison['variants']['prior']['alg_params']['inner_operator'] == 'none'
    panel = []
    for j,selector in zip(ROUNDS,selectors):
        name = selector.split('/')[1]
        params = {**matrix['shared_alg_params'],**comparison['variants'][name]['alg_params']}
        assert params == requested_params(j,horizon),name
        assert params['inner_replay_capacity'] == max(3072,384*j)
        panel.append(dict(name=name,checkpoint_step=CHECKPOINT_STEP,training_decisions=CHECKPOINT_STEP,
            selector=selector,actual_selector=selector,requested_alg_params=deepcopy(params),
            params={k:v for k,v in params.items() if v is not None},H=horizon,J=j,critic_kind='return_only',
            estimator='one_step',execution_mode='mean',alpha_mode='adaptive',reused=horizon==3 and j==10))
    return panel


def verify_publication(pin, reference):
    """Check this record in its shared curve; cumulative counts need not be one."""
    from utils.eval_series import load_run
    bundle = Path(pin['bundle']); publication = read(bundle.parent/'publication-completion.json')
    assert publication['status'] == 'complete' and publication['performance']['published'] >= 1
    assert publication['performance']['run_id'] == pin['performance_run_id']
    assert publication['training_run_id'] == pin['training_run_id']
    run_dir = Path(pin['run_dir']); registry = load_run(run_dir)
    assert registry['run_id'] == pin['performance_run_id']
    assert registry['identity'] == reference['identity']
    entry = read(run_dir/'publication.json')['records'][reference['record_id']]
    assert entry['status'] == 'published'
    assert entry['checkpoint_step'] == CHECKPOINT_STEP and entry['checkpoint_sha256'] == CHECKPOINT_SHA
    stored = read(run_dir/'records'/(reference['record_id']+'.json'))
    assert stored['record_id'] == reference['record_id'] and stored['identity'] == reference['identity']
    assert stored['checkpoint']['step'] == CHECKPOINT_STEP and stored['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert indexed_episodes(stored['episodes']) == indexed_episodes(reference['episodes'])
    assert entry['artifact_sha256']['manifest.json'] == pin['manifest_sha256']
    for name,sha in reference.get('trace_sha256',{}).items():
        assert entry['artifact_sha256'][name] == sha
    if 'record_id' in pin:
        assert reference['record_id'] == pin['record_id']
    if 'publication_entry' in pin:
        assert entry == pin['publication_entry']
    return entry


def load_reused(pin, inventory):
    assert pin['checkpoint_step'] == CHECKPOINT_STEP and pin['checkpoint_sha256'] == CHECKPOINT_SHA
    assert pin['performance_run_id'] == '4aa700149cec4161886b916c4b135c07'
    assert pin['training_run_id'] == 'b4bfda4e5a2d497bb5fe7e6d74fadafc'
    reference = enrich_reference(pin,inventory); bundle = Path(pin['bundle'])
    receipt = read(bundle.parent/'worker-completion.json')
    assert receipt['manifest_sha256'] == pin['manifest_sha256']
    manifest = verify_receipt(bundle,receipt)
    validate(bundle,cells()[5],checkpoint_sha=CHECKPOINT_SHA,checkpoint_step=CHECKPOINT_STEP)
    reference['publication_entry'] = verify_publication(pin,reference)
    alpha = manifest['runs'][0]['result']['model_metrics']['inner_alpha_initial']
    assert all(math.isclose(alpha[s],INITIAL_ALPHA,rel_tol=1e-7) for s in ('mean','min','max'))
    reference.update(initial_alpha=alpha['mean'],reference_manifest_sha256=manifest['reference']['manifest_sha256'])
    return reference


def load_mppi_references(pin, inventory, prior):
    """Optional existing comparator points, checked without new evaluations."""
    from utils.ambi_benchmark import episode_protocol
    from utils.eval_series_data import load_records
    if not pin:
        return {}
    assert (pin['checkpoint_step'],pin['checkpoint_sha256']) == (CHECKPOINT_STEP,CHECKPOINT_SHA)
    bundle = Path(pin['bundle']); manifest = read(bundle/'manifest.json')
    assert digest(bundle/'manifest.json') == pin['manifest_sha256']
    assert manifest['status'] == 'complete' and manifest['code']['dirty'] is False
    assert manifest['code']['commit'] == pin['source_commit']
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    assert manifest['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert manifest['checkpoint']['metadata']['checkpoint']['step'] == CHECKPOINT_STEP
    assert manifest['protocol']['action_rule'] == 'mppi_proposal_mean'
    assert episode_protocol({**manifest['protocol'],'action_rule':'tanh_mean'}) == episode_protocol(prior['protocol'])
    assert manifest['code']['runtime'] == prior['runtime']
    result = {}
    for selector,kind in [('bootstrap/soft_q','soft'),('bootstrap/return_q','return_only')]:
        run, = [r for r in manifest['runs'] if r['selector'] == selector]
        record, = [r for r in load_records(bundle,inventory_path=inventory) if r['selector'] == selector]
        assert record['metrics']['eval/frozen_state_unchanged']
        assert not record['provenance']['missing_artifact_files']
        assert run['resolved_config']['inner_operator'] == 'mppi'
        assert run['resolved_config']['inner_horizon_critic_source'] == ('sac' if kind == 'soft' else 'aux_return')
        check_episodes(run['episodes'])
        result[kind] = dict(pin,selector=selector,episodes=run['episodes'],identity=record['identity'],
            record_id=record['record_id'],resolved_config=run['resolved_config'],protocol=manifest['protocol'],
            runtime=manifest['code']['runtime'],trace_sha256={name:digest(bundle/name) for name in run['trace_files']})
    return result


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    from utils.eval_series_data import planner_identity
    horizon = getattr(args,'horizon',3)
    args.matrix = args.matrix or matrix_for(horizon)
    args.group = args.group or f'closed-loop-h{horizon}-j-sweep-650k-20260924'
    args.label = args.label or f'650k H{horizon} return critics | J sweep'
    panel = cells(args.matrix,horizon); commit = source_commit(); pins = read(args.references)
    inventory = read(args.inventory)
    assert inventory['source_run'] == pins['source_run'] == SOURCE_RUN
    assert pins['checkpoint_step'] == CHECKPOINT_STEP
    row, = [r for r in inventory['checkpoints'] if r['step'] == CHECKPOINT_STEP]
    prior = load_prior(pins['prior_reference'],args.inventory)
    reused = load_reused(pins['reused_evaluation'],args.inventory) if horizon == 3 else None
    assert row['sha256'] == prior['checkpoint_sha256'] == CHECKPOINT_SHA
    assert row['metadata_sha256'] == prior['metadata_sha256']
    assert digest(row['path']) == row['sha256'] and digest(row['metadata_path']) == row['metadata_sha256']
    assert Path(row['metadata_path']) == Path(row['path']+'.metadata.json')
    assert read(row['metadata_path']) == read(Path(prior['bundle'])/'manifest.json')['checkpoint']['metadata']
    proof = checkpoint_state_proof(row['path'],prior)
    assert math.isclose(proof['initial_alpha'],INITIAL_ALPHA,rel_tol=1e-6)
    mppi = load_mppi_references(pins.get('mppi_reference'),args.inventory,prior)
    args.root.mkdir(parents=True,exist_ok=False)
    for cell in panel:
        directory = args.root/cell['name']; directory.mkdir()
        cell.update(checkpoint=row['path'],checkpoint_sha256=row['sha256'],metadata_sha256=row['metadata_sha256'],
            prior_reference=deepcopy(prior),initial_alpha=proof['initial_alpha'],checkpoint_state_proof=deepcopy(proof),
            expected_config=resolve_config(args.matrix,row['path'],cell['selector']))
        result = evaluate_matrix(args.matrix,row['path'],selectors=[cell['selector']],seeds=SEEDS,
            controller_seed=55,max_steps=500,bundle_dir=directory/'unused',checkpoint_inventory=args.inventory,
            reference_bundle=prior['bundle'],eval_series_spec_dir=directory/'specs')
        assert result['mode'] == 'evaluation_series_specifications' and set(result['specs']) == {cell['selector']}
        spec = read(result['specs'][cell['selector']])
        assert spec['identity']['backbone'] == prior['identity']['backbone'] == SOURCE_RUN
        assert spec['identity']['protocol'] == prior['identity']['protocol']
        assert spec['identity']['planner'] == planner_identity(cell['expected_config'],{},'AMBITDMPC2/AMBITDMPC2','tanh_mean')
        cell.update(directory=str(directory),identity=spec['identity'])
        if cell['reused']:
            matching_config(reused['resolved_config'],cell['expected_config'])
            assert reused['identity']['planner'] == spec['identity']['planner']
            assert reused['identity']['protocol'] == spec['identity']['protocol']
            assert reused['checkpoint_sha256'] == cell['checkpoint_sha256']
            assert reused['reference_manifest_sha256'] == prior['manifest_sha256']
            assert math.isclose(reused['initial_alpha'],cell['initial_alpha'],rel_tol=1e-6)
            baseline = indexed_episodes(prior['episodes'])
            for episode in reused['episodes']:
                key = (episode['seed'],episode['solver_seed'])
                assert math.isclose(episode['paired_return_delta'],episode['return']-baseline[key]['return'],abs_tol=1e-9)
            cell.update(bundle=reused['bundle'],reuse_reference=reused,
                performance_run_id=reused['performance_run_id'],training_run_id=reused['training_run_id'],
                run_dir=reused['run_dir'])
        else:
            registry = create_run(args.registry,spec,args.group+'-'+cell['name'],PROJECT,ENTITY,'oscar-rgao48')
            cell.update(bundle=str(directory/'bundle'),run_dir=registry['run_dir'],
                performance_run_id=registry['run_id'],training_run_id=uuid.uuid4().hex)
    campaign = dict(schema_version=1,group=args.group,label=args.label,matrix=str(args.matrix.resolve()),
        inventory=str(args.inventory.resolve()),source_commit=commit,source_dir=str(ROOT),source_run=SOURCE_RUN,
        checkpoint_step=CHECKPOINT_STEP,checkpoint_sha256=CHECKPOINT_SHA,checkpoint_steps=[CHECKPOINT_STEP]*len(panel),
        target_entropy=-10.5,H=horizon,rounds=list(ROUNDS),cells=panel,mppi_references=mppi,
        production_indices=[i for i,c in enumerate(panel) if not c['reused']],smoke_indices=[7],
        publisher_workers=3,overview_run_id=uuid.uuid4().hex,
        prior_compatibility_note='Prior mean inference and full-episode pairing are unchanged; historical references retain original scientific identities.',
        reused_compatibility_note=('Exact H3/J10 planner and full-episode protocol reused at 650k; historical source, shared performance curve and training ID retained.'
            if horizon == 3 else 'No refinement episodes are reused; prior and MPPI references retain their original identities.'))
    write(args.root/'campaign.json',campaign)
    print(f'Prepared H{horizon}, 8 round budgets, {len(campaign["production_indices"])} new evaluations; overview {campaign["overview_run_id"]}',flush=True)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    for name in ('root','inventory','registry'):prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--references',type=Path,default=REFERENCES); prep.add_argument('--matrix',type=Path)
    prep.add_argument('--horizon',type=int,choices=(1,2,3),default=3)
    prep.add_argument('--group'); prep.add_argument('--label')
    run = sub.add_parser('worker'); run.add_argument('--root',type=Path,required=True)
    run.add_argument('--index',type=int,required=True); run.add_argument('--smoke',action='store_true')
    args = parser.parse_args(); {'prepare':prepare,'worker':worker}[args.command](args)


if __name__ == '__main__':main()
