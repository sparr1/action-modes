"""Only timing changes; complete matched historical controls retain their IDs."""
from copy import deepcopy
import json
import pytest
from slurm.ambi_aux_hj_sweep import MATRIX,cells,read,rollouts,polyak_comparison
from slurm.ambi_aux_step_timing import match_identity,baseline_result,BASELINE_SOURCES
from slurm.ambi_aux_rollout_batch import reference_cell
from tests.test_aux_round_budget import identity

PATH=MATRIX.with_name('ambi_aux_step_timing_625k.json')
NB=MATRIX.with_name('ambi_aux_rollout_batch_625k.json')

def reference(cell):
    c=deepcopy(cell);c['params'].pop('inner_update_timing')
    return c

def record(cell):
    from utils.eval_series_data import scientific_identity
    r=identity(reference(cell),PATH)
    r['identity']['science']=scientific_identity('AMBITDMPC2/AMBITDMPC2',None,BASELINE_SOURCES[rollouts(cell)])
    r.update(record_id='original',metrics={'eval/frozen_state_unchanged':True},
             episodes=[dict(seed=s,solver_seed=55,length=500,truncated_by_evaluator=False,
                            **{'return':float(s)}) for s in range(101,106)])
    return r


def test_two_settings_reuse_controls_and_prior():
    panel=cells(PATH)
    assert len(panel)==2 and {rollouts(c) for c in panel}=={32,128}
    assert read(PATH)['shared_alg_params']==read(NB)['shared_alg_params']
    assert read(PATH)['step_timing_sweep'] and not read(PATH).get('execution')
    for cell in panel:
        old=next(c for c in cells(NB) if c['name']=='soft_soft_h3_j8_c16_n32_b256')
        assert cell['params']=={**old['params'],'inner_rollouts_per_round':rollouts(cell),'inner_update_timing':'step'}
        match_identity(identity(cell,PATH),identity(reference(cell),PATH),rollouts(cell))


@pytest.mark.parametrize('cell',cells(PATH),ids=lambda c:c['name'])
def test_historical_sources_and_pairing(cell):
    old=record(cell);spec=identity(cell,PATH);spec.pop('checkpoint')
    baseline=baseline_result(spec,old,cell)
    episodes=[{**e,'return':e['return']+3} for e in old['episodes']]
    assert polyak_comparison(episodes,baseline)['metrics']['comparison/round_timing_gain_mean']==3
    episodes[0]['solver_seed']=56
    with pytest.raises(AssertionError):polyak_comparison(episodes,baseline)
    old['identity']['science']={'wrong_source':True}
    with pytest.raises(AssertionError):baseline_result(spec,old,cell)


@pytest.mark.parametrize('key,value',[
    ('inner_actor_updates_per_round',8),('inner_critic_updates_per_round',48),
    ('inner_actor_lr',1e-4),('inner_critic_target_tau',.02),('inner_replay_capacity',768),
    ('inner_replay_reset_each_round',True),('inner_update_timing','round'),
    ('inner_horizon_conditioning','one_hot'),('inner_terminal_entropy','none'),
    ('inner_rounds',4),('inner_batch_size',64),('inner_steps_per_update',1)])
def test_rejects_any_change_beyond_timing(key,value):
    cell=cells(PATH)[0];old=identity(reference(cell),PATH);new=identity(cell,PATH)
    new['identity']['planner']['settings'][key]=value
    with pytest.raises(AssertionError):match_identity(new,old,rollouts(cell))


def test_reused_control_has_no_new_publication_identity(tmp_path):
    source=tmp_path/'old';source.mkdir();target=tmp_path/'reference';target.mkdir()
    c={**reference(cells(PATH)[0]),'directory':str(source),'bundle':'original-bundle',
       'performance_run_id':'original-performance','training_run_id':'original-training',
       'run_dir':'original-registry','baseline':{'kind':'rollout_batch'}}
    (source/'publication-completion.json').write_text(json.dumps(dict(status='complete',training_run_id='original-training',metrics={})))
    result=reference_cell(c,target,None,record(cells(PATH)[0]),{'status':'complete'},kind='step_timing')
    for key in ('bundle','performance_run_id','training_run_id','run_dir'):assert result[key]==c[key]
    assert result['reused'] and 'baseline' not in result
    assert read(target/'publication-completion.json')['metrics']['comparison/round_timing_gain_mean']==0
