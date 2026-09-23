"""The overnight reward-only screen retains provenance, pairing and partial results."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_reward_retrace as campaign
from slurm import ambi_closed_loop_reward_retrace_publish as publication
from utils.ambi_benchmark import solver_seed


def episodes(offset=0):
    return [dict(seed=s, solver_seed=solver_seed(55, 'episode', s), length=500,
                 truncated_by_evaluator=False, **{'return': float(s - 100 + offset)})
            for s in campaign.SEEDS]


def state_fixture():
    panel = campaign.cells()
    refs = []
    for h in (1, 2, 3):
        for j in (1, 2, 4, 6, 8, 10):
            for execution in ('mean', 'policy_sample') if j <= 4 else ('mean',):
                refs.append(dict(H=h, J=j, execution=execution, estimator='one_step',
                    performance_run_id=f'{h}-{j}-{execution}', manifest_sha256='pinned',
                    episodes=episodes(j + (2 if execution == 'policy_sample' else 0))))
    for c in panel:
        c.update(performance_run_id=c['name']+'-performance', training_run_id=c['name']+'-training',
                 mean_reference=next(r for r in refs if (r['H'],r['J'],r['execution']) == (c['H'],c['J'],'mean')))
    return dict(cells=panel, references=refs, checkpoint_step=575000, checkpoint_sha256=campaign.CHECKPOINT_SHA,
        source_run=campaign.SOURCE_RUN, source_commit='tested', initial_alpha=campaign.INITIAL_ALPHA,
        target_entropy=-10.5, H=[1,2,3], J=[1,2,4,6,8,10], estimator=['one_step','retrace'],
        group='test', overview_run_id='overview', label='Test')


def test_scope_retains_all_old_results_and_covers_six_smoke_boundaries():
    panel = campaign.cells()
    assert len(panel) == 27
    assert [(c['H'],c['estimator'],c['J']) for c in panel] == campaign.identities()
    assert [(c['H'],c['estimator']) for c in panel[:6]] == [
        (h,e) for h in (1,2,3) for e in ('one_step','retrace')]
    assert all(c['J'] == 10 for c in panel[:6])
    assert sum(c['estimator'] == 'one_step' for c in panel) == 9
    assert all(c['J'] in (6,8,10) for c in panel if c['estimator'] == 'one_step')
    for c in panel:
        p = c['params']
        assert p['inner_critic_source'] == p['inner_horizon_critic_source'] == 'aux_return'
        assert p['inner_sac_critic_target'] == 'reward_only' and p['inner_terminal_entropy'] == 'none'
        assert p['inner_eval_execution_action'] == 'policy_sample'
        assert p['inner_replay_capacity'] == (3840 if c['J'] == 10 else 3072)
        assert p['inner_replay_capacity'] >= 128*c['H']*c['J']
        assert p['inner_actor_updates_per_round'] == 4 and p['inner_critic_updates_per_round'] == 16


@pytest.mark.parametrize('index', range(27))
def test_every_resolved_cell_has_expected_work_and_trajectory_batch(index):
    from tests.test_ambi_root_local_sac import _build_cfg
    cell = campaign.cells()[index]
    cfg = _build_cfg(**cell['params'], aux_return_mode='sac', log_std_mapping='direct_clamp',
                     target_entropy=-10.5, sac_actor_loss_scale_mode='none')
    assert cfg.inner_model_step_budget == 128*cell['H']*cell['J']
    assert cfg.inner_critic_updates_per_action == 16*cell['J']
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 4*cell['J']
    if cell['estimator'] == 'retrace':
        assert cfg.inner_retrace_lambda == .9
        assert cfg.inner_retrace_batch_trajectories == {1:256,2:128,3:86}[cell['H']]
        assert cfg.inner_retrace_batch_trajectories*cell['H'] in (256,258)
    else:
        assert cfg.inner_retrace_batch_trajectories is None


@pytest.mark.parametrize('mutation', ['missing','duplicate','soft','lambda','capacity','execution'])
def test_matrix_rejects_scientific_or_scope_drift(tmp_path, mutation):
    matrix = campaign.read(campaign.MATRIX)
    selectors = matrix['evaluation']['default_presets']
    params = matrix['comparisons']['sweep']['variants'][selectors[1].split('/')[1]]['alg_params']
    if mutation == 'missing': selectors.pop()
    elif mutation == 'duplicate': selectors[0] = selectors[1]
    else:
        key,value = {'soft':('inner_critic_source','sac'), 'lambda':('inner_retrace_lambda',1.),
                     'capacity':('inner_replay_capacity',3072),
                     'execution':('inner_eval_execution_action','mean')}[mutation]
        params[key] = value
    path = tmp_path/'matrix.json'; path.write_text(json.dumps(matrix))
    with pytest.raises(AssertionError): campaign.cells(path)


@pytest.mark.parametrize('estimator', ['one_step','retrace'])
def test_old_config_defaults_normalize_but_other_changes_are_rejected(estimator):
    cell = next(c for c in campaign.cells() if c['H'] == 3 and c['estimator'] == estimator)
    old = campaign.historical_cell(3,10)['params']
    actual = {**old,campaign.EXECUTION_KEY:'policy_sample',**campaign.estimator_settings(estimator,3)}
    campaign.matching_config(actual, old, cell)
    campaign.matching_config(actual, {**old,**campaign.RETRACE_DEFAULTS}, cell)
    for key,value in [('inner_actor_lr',.01),('inner_replay_capacity',3072),('inner_retrace_lambda',.5)]:
        with pytest.raises(AssertionError): campaign.matching_config({**actual,key:value},old,cell)
    with pytest.raises(AssertionError):
        campaign.matching_config(actual,{**old,'inner_retrace_lambda':.9},cell)


def test_reference_inventory_is_exact_complete_and_has_no_duplicate_identity():
    refs = campaign.read(campaign.REFERENCES)['references']
    keys = [campaign.reference_key(r) for r in refs]
    assert len(set(keys)) == len(keys) == 27
    assert set(keys) == ({(h,j,'mean') for h in (1,2,3) for j in (1,2,4,6,8,10)}
                         | {(h,j,'policy_sample') for h in (1,2,3) for j in (1,2,4)})
    assert len({r['manifest_sha256'] for r in refs}) == 27
    assert len({r['performance_run_id'] for r in refs}) == 27
    assert all(r['estimator'] == 'one_step' for r in refs)


def test_estimator_pairing_waits_for_matching_sampled_baseline_and_never_imputes():
    state = state_fixture()
    retrace, = [c for c in state['cells'] if c['H']==3 and c['J']==10 and c['estimator']=='retrace']
    one_step, = [c for c in state['cells'] if c['H']==3 and c['J']==10 and c['estimator']=='one_step']
    pending = publication.aggregate_results(state,{})
    assert len(pending['points']) == 27 and pending['new_evaluated'] == 0
    assert pending['estimator_comparisons'] == []
    assert 'comparison/retrace_minus_one_step_vs_J' not in publication.chart_payloads(pending)
    done = {retrace['name']:episodes(20)}
    early = publication.aggregate_results(state,done)
    point = next(p for p in early['points'] if p['setting']==retrace['name'])
    assert point['mean_difference']['mean'] == 10
    assert 'estimator_metrics' not in point and not early['estimator_comparisons']
    first_ids = {identity for identity,_ in publication.numeric_rows(early)}
    done[one_step['name']] = list(reversed(episodes(12)))
    late = publication.aggregate_results(state,done)
    pair = late['estimator_comparisons']
    assert len(pair)==5 and all(r['retrace_minus_one_step']==8 for r in pair)
    new_ids = {identity for identity,_ in publication.numeric_rows(late)}-first_ids
    assert retrace['name']+'/retrace_minus_one_step' in new_ids
    assert len(late['points'])==29
    with pytest.raises(ValueError,match='Unexpected completed'):
        publication.aggregate_results(state,{'bad':episodes()})


def test_paired_estimator_comparison_rejects_solver_seed_mismatch():
    changed = episodes(1); changed[0]['solver_seed'] += 1
    with pytest.raises(ValueError,match='solver seeds differ'):
        publication.estimator_comparison(changed,episodes())


def test_existing_publications_still_populate_all_new_points_and_pairs(tmp_path, monkeypatch):
    state = state_fixture()
    for cell in state['cells']:
        directory = tmp_path/cell['name']; directory.mkdir()
        (directory/'worker-completion.json').write_text('{}')
        cell['directory'] = str(directory)
    (tmp_path/'campaign.json').write_text(json.dumps(state))
    logged = []
    run = SimpleNamespace(summary={},log=lambda row:logged.append(deepcopy(row)),
                          define_metric=lambda *a,**k:None,finish=lambda **k:None)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=lambda **k:run))
    monkeypatch.setattr(publication,'verify_reference',lambda *a,**k:None)
    monkeypatch.setattr(publication,'publication_complete',lambda c:True)
    monkeypatch.setattr(publication,'load_completed',lambda state,c:(
        episodes(c['J']+(4 if c['estimator']=='retrace' else 2)),{'execution/sample_flag_mean':1}))
    monkeypatch.setattr(publication,'overview_log',lambda *a:{})
    publication.watch(SimpleNamespace(root=tmp_path))
    aggregate = campaign.read(tmp_path/'comparison-results.json')
    assert len(aggregate['points'])==54 and len(aggregate['estimator_comparisons'])==90
    assert run.summary['status']=='complete' and run.summary['evaluated']==27
    assert sum(any('/retrace_minus_one_step/mean' in key for key in row) for row in logged)==18


def test_points_table_has_scalar_axes_provenance_and_missing_estimator_deltas():
    aggregate = publication.aggregate_results(state_fixture(),{})
    wb = SimpleNamespace(Table=lambda **kwargs:kwargs, plot=SimpleNamespace(line_series=lambda **kwargs:kwargs))
    output = publication.overview_log(wb,aggregate,[])
    table = output['comparison/points']; rows = [dict(zip(table['columns'],r)) for r in table['data']]
    assert len(rows)==27 and all(r['H'] in (1,2,3) and r['return_episodes']==5 for r in rows)
    assert all(r['performance_run_id'] and r['retrace_minus_one_step_mean'] is None for r in rows)


def test_prepare_creates_only_new_identities_and_keeps_dynamic_estimator_references(tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint
    from utils import eval_series
    panel = campaign.cells()
    pins = campaign.read(campaign.REFERENCES)['references']
    refs = {}
    for pin in pins:
        cfg = campaign.historical_cell(pin['H'],pin['J'])['params']
        if pin['execution']=='policy_sample': cfg[campaign.EXECUTION_KEY]='policy_sample'
        protocol = dict(action_rule=campaign.ACTION_RULE if pin['execution']=='policy_sample' else 'tanh_mean',
                        max_steps=500,controller_seed=55,seed_scheme='sha256-v1',environment={'id':'test'})
        refs[campaign.reference_key(pin)] = {**pin,'resolved_config':cfg,'episodes':episodes(),
            'protocol':protocol,'identity':dict(backbone=campaign.SOURCE_RUN,protocol=protocol,
             planner=dict(type='inner_sac',action_rule=protocol['action_rule'],settings=deepcopy(cfg)))}
    monkeypatch.setattr(campaign,'source_commit',lambda:'tested')
    monkeypatch.setattr(campaign,'digest',lambda p:campaign.CHECKPOINT_SHA)
    monkeypatch.setattr(campaign,'load_reference',lambda pin,inventory:deepcopy(refs[campaign.reference_key(pin)]))
    def evaluate(*args,**kwargs):
        specs = kwargs['eval_series_spec_dir']; specs.mkdir()
        assert kwargs['reference_bundle'] is None and kwargs['seeds']==campaign.SEEDS
        for cell in panel:
            identity = deepcopy(refs[(cell['H'],cell['J'],'mean')]['identity'])
            identity['protocol']['action_rule'] = campaign.ACTION_RULE
            identity['planner']['action_rule'] = campaign.ACTION_RULE
            identity['planner']['settings'].update({campaign.EXECUTION_KEY:'policy_sample',
                **campaign.estimator_settings(cell['estimator'],cell['H'])})
            campaign.write(specs/(cell['selector'].replace('/','__')+'.json'),{'identity':identity})
    monkeypatch.setattr(evaluate_ambi_checkpoint,'evaluate_matrix',evaluate)
    created=[]
    def create(*args):
        created.append(args[2]); return {'run_id':f'run{len(created)}','run_dir':str(tmp_path/f'run{len(created)}')}
    monkeypatch.setattr(eval_series,'create_run',create)
    state = campaign.prepare(SimpleNamespace(root=tmp_path/'new',matrix=campaign.MATRIX,
        references=campaign.REFERENCES,checkpoint=tmp_path/'checkpoint',inventory=tmp_path/'inventory',
        registry=tmp_path/'registry',group='test',label='Test'))
    assert len(created)==len(set(created))==27 and len(state['references'])==27
    assert state['smoke_indices']==list(range(6)) and state['prior_reference'] is None
    for cell in state['cells']:
        if cell['estimator']=='retrace':
            assert bool(cell['one_step_reference']) == (cell['J']<=4)
            assert bool(cell['one_step_cell']) == (cell['J']>=6)
