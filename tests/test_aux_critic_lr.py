"""Only critic LR changes; matching C8/C16 controls retain publication IDs."""
from copy import deepcopy
import json
import pytest
from slurm.ambi_aux_hj_sweep import MATRIX, cells, read, critic_updates, polyak_comparison
from slurm.ambi_aux_critic_lr import match_identity, baseline_result, BASELINE_SOURCE
from slurm.ambi_aux_rollout_batch import reference_cell
from tests.test_aux_round_budget import identity

PATH = MATRIX.with_name('ambi_aux_critic_lr_625k.json')
OLD = MATRIX.with_name('ambi_aux_soft_critic_budget_625k.json')


def reference(cell):
    return next(c for c in cells(OLD) if c['name'] == f'soft_soft_h3_j8_c{critic_updates(cell)}')


def record(cell):
    from utils.eval_series_data import scientific_identity
    r = identity(reference(cell), OLD)
    r['identity']['science'] = scientific_identity('AMBITDMPC2/AMBITDMPC2', None, BASELINE_SOURCE)
    r.update(record_id='original', metrics={'eval/frozen_state_unchanged':True},
             episodes=[dict(seed=s, solver_seed=55, length=500, truncated_by_evaluator=False,
                            **{'return':float(s)}) for s in range(101,106)])
    return r


def test_exact_four_settings_and_unchanged_non_lr_settings():
    panel = cells(PATH)
    assert len(panel) == 4 and len({c['name'] for c in panel}) == 4
    assert {(critic_updates(c), c['params']['inner_critic_lr']) for c in panel} == {
        (c, lr) for c in (8,16) for lr in (6e-4,1e-3)}
    assert read(PATH)['shared_alg_params'] == read(OLD)['shared_alg_params']
    assert read(PATH)['critic_lr_sweep'] and not read(PATH).get('execution')
    for c in panel:
        old = reference(c)
        assert c['params'] == {**old['params'], 'inner_critic_lr':c['params']['inner_critic_lr'],
                              'inner_rollouts_per_round':128, 'inner_batch_size':256}
        match_identity(identity(c,PATH), identity(old,OLD), critic_updates(c), c['params']['inner_critic_lr'])


@pytest.mark.parametrize('cell', cells(PATH), ids=lambda c:c['name'])
def test_historical_sources_and_paired_lr_comparison(cell):
    old = record(cell); spec = identity(cell,PATH); spec.pop('checkpoint')
    baseline = baseline_result(spec,old,cell)
    assert baseline['kind'] == 'critic_lr'
    episodes = [{**e,'return':e['return']+3} for e in old['episodes']]
    assert polyak_comparison(episodes,baseline)['metrics']['comparison/critic_lr_3e4_gain_mean'] == 3
    episodes[0]['solver_seed'] = 56
    with pytest.raises(AssertionError): polyak_comparison(episodes,baseline)
    old['identity']['science'] = {'wrong_source':True}
    with pytest.raises(AssertionError): baseline_result(spec,old,cell)


@pytest.mark.parametrize('key,value', [
    ('inner_actor_updates_per_round',8), ('inner_critic_updates_per_round',32),
    ('inner_actor_lr',6e-4), ('inner_temperature_lr',1e-3), ('inner_critic_target_tau',.0199),
    ('inner_replay_capacity',768), ('inner_replay_reset_each_round',True),
    ('inner_update_timing','step'), ('inner_horizon_conditioning','one_hot'),
    ('inner_terminal_entropy','none'), ('inner_rounds',4), ('inner_batch_size',64),
    ('inner_critic_lr',3e-4), ('inner_rollouts_per_round',32)])
def test_rejects_unrequested_changes(key,value):
    c = cells(PATH)[0]; old = identity(reference(c),OLD); new = identity(c,PATH)
    new['identity']['planner']['settings'][key] = value
    with pytest.raises(AssertionError):
        match_identity(new,old,critic_updates(c),c['params']['inner_critic_lr'])


@pytest.mark.parametrize('count',[8,16])
def test_reused_control_keeps_all_publication_ids(tmp_path,count):
    source = tmp_path/'old'; source.mkdir(); target = tmp_path/'reference'; target.mkdir()
    cell = next(c for c in cells(PATH) if critic_updates(c) == count)
    old = {**reference(cell), 'directory':str(source), 'bundle':'original-bundle',
           'performance_run_id':'original-performance', 'training_run_id':'original-training',
           'run_dir':'original-registry', 'baseline':{'kind':'critic_budget'}}
    (source/'publication-completion.json').write_text(json.dumps(dict(
        status='complete',training_run_id='original-training',metrics={})))
    reused = reference_cell(old,target,None,record(cell),{'status':'complete'},kind='critic_lr')
    for key in ('bundle','performance_run_id','training_run_id','run_dir'):
        assert reused[key] == old[key]
    assert reused['reused'] and 'baseline' not in reused
    assert reused['name'].endswith('_lr3e4_reused')
    assert read(target/'publication-completion.json')['metrics']['comparison/critic_lr_3e4_gain_mean'] == 0
