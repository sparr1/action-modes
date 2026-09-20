"""Only N/B change; existing panels retain their original result/publication IDs."""
from copy import deepcopy
import json
import pytest
from slurm.ambi_aux_hj_sweep import MATRIX,cells,read,polyak_comparison,rollouts,batch_size
from slurm.ambi_aux_rollout_batch import match_identity,baseline_result,reference_cell,BASELINE_SOURCE
from tests.test_aux_round_budget import identity

PATH=MATRIX.with_name('ambi_aux_rollout_batch_625k.json')
OLD=MATRIX.with_name('ambi_aux_soft_critic_budget_625k.json')
def anchor(cell):
    return next(c for c in cells(OLD) if c['name']==f"soft_soft_h{cell['H']}_j8_c16")

def test_six_new_cells_no_reference_reruns():
    panel=cells(PATH)
    assert len(panel)==6
    assert {(c['H'],rollouts(c),batch_size(c)) for c in panel}=={
        (h,n,b) for h in (1,3) for n,b in [(32,256),(128,64),(32,64)]}
    assert read(PATH)['rollout_batch_sweep'] and not read(PATH).get('execution')
    assert read(PATH)['shared_alg_params']==read(OLD)['shared_alg_params']

@pytest.mark.parametrize('cell',cells(PATH),ids=lambda c:c['name'])
def test_only_requested_data_and_minibatch_sizes_change(cell):
    old=anchor(cell)
    assert cell['params']=={**old['params'],'inner_rollouts_per_round':rollouts(cell),'inner_batch_size':batch_size(cell)}
    match_identity(identity(cell,PATH),identity(old,OLD),rollouts(cell),batch_size(cell))
    assert rollouts(cell)*cell['H']*cell['J']<=3072

@pytest.mark.parametrize('key,value',[
    ('inner_actor_updates_per_round',8),('inner_critic_updates_per_round',32),
    ('inner_actor_lr',1e-4),('inner_critic_target_tau',.02),('inner_replay_capacity',384),
    ('inner_replay_reset_each_round',True),('inner_update_timing','step'),
    ('inner_horizon_conditioning','one_hot'),('inner_terminal_entropy','none'),
    ('inner_rounds',4),('inner_model_step_budget',1)])
def test_identity_rejects_other_changes(key,value):
    cell=cells(PATH)[0];old=identity(anchor(cell),OLD);new=identity(cell,PATH)
    new['identity']['planner']['settings'][key]=value
    with pytest.raises(AssertionError):match_identity(new,old,rollouts(cell),batch_size(cell))

def record(cell):
    from utils.eval_series_data import scientific_identity
    r=identity(anchor(cell),OLD)
    r['identity']['science']=scientific_identity('AMBITDMPC2/AMBITDMPC2',None,BASELINE_SOURCE)
    r.update(record_id='original',metrics={'eval/frozen_state_unchanged':True},
             episodes=[dict(seed=s,solver_seed=55,length=500,truncated_by_evaluator=False,
                            **{'return':float(s)}) for s in range(101,106)])
    return r

def test_historical_pairing_and_seeds():
    cell=cells(PATH)[0];old=record(cell);spec=identity(cell,PATH);spec.pop('checkpoint')
    baseline=baseline_result(spec,old,cell)
    new=[{**e,'return':e['return']+2} for e in old['episodes']]
    assert polyak_comparison(new,baseline)['metrics']['comparison/n128_b256_gain_mean']==2
    new[0]['solver_seed']=56
    with pytest.raises(AssertionError):polyak_comparison(new,baseline)
    old['identity']['science']={'wrong':True}
    with pytest.raises(AssertionError):baseline_result(spec,old,cell)

def test_reference_reuses_existing_ids_and_bundles(tmp_path):
    source=tmp_path/'source';source.mkdir();target=tmp_path/'reference';target.mkdir()
    previous={**anchor(cells(PATH)[0]),'directory':str(source),'bundle':'immutable-bundle',
              'performance_run_id':'existing-performance','training_run_id':'existing-training',
              'run_dir':'existing-registry','baseline':{'kind':'critic_budget'}}
    (source/'publication-completion.json').write_text(json.dumps(dict(status='complete',
        training_run_id='existing-training',performance={'status':'published'},metrics={})))
    cell=reference_cell(previous,target,{},record(cells(PATH)[0]),{'status':'complete','manifest_sha256':'sha'})
    assert cell['reused'] and 'baseline' not in cell
    for key in ('performance_run_id','training_run_id','bundle','run_dir'):assert cell[key]==previous[key]
    publication=read(target/'publication-completion.json')
    assert publication['reused'] and publication['metrics']['comparison/n128_b256_gain_mean']==0
