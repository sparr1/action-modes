"""Critic dose must retain immutable references, complete pairing and metrics."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from slurm import publish_ambi_critic_dose as pub
from utils import wandb_critic_dose_layout as layout


def campaign(seeds=(101, 102)):
    cfg = dict(protocol=pub.PROTOCOL, campaign_id='a' * 64, arms=list(pub.NEW_ARMS),
        seeds=list(seeds), controller_seed=55, max_steps=4, smoke=True,
        horizon=1, rounds=6, mc_rollouts=32, validation_rollouts=32, diagnostic_decisions=[0, 3],
        actor_updates_per_round=4, rollouts_per_round=128, batch_size=256,
        reference_snapshot_sha256=pub.REFERENCE_SNAPSHOT_SHA256, reference_campaign_id=pub.REFERENCE_CAMPAIGN_ID)
    cfg['task_list'] = [dict(task_id=f'{arm}-seed-{seed}', arm=arm, env_seed=seed, controller_seed=55,
                           selector=arm.rsplit('_c', 1)[0], critic_updates_per_round=int(arm.rsplit('_c', 1)[1]))
                        for arm in pub.NEW_ARMS for seed in seeds]
    return cfg


def diagnostic(decision=0):
    q = list(np.arange(770, dtype=float)); model = list(reversed(q))
    indices = dict(actor_mean=0, prior_mean=1, learned_q=769, model_score=0)
    def stats(value): return dict(mean=float(value), se=0., draws=[float(value)] * 32)
    return dict(decision=decision, bank_count=770, replay_count=768, mc_rollouts=32,
        selection_seed=1, validation_seed=2, selection_q_scores=q, selection_model_means=model,
        choices={name: dict(index=i, normalized_action=[0., .5], learned_q=q[i],
            selection_model=stats(model[i]), validation_model=stats(model[i]),
            validation_gain_vs_prior=stats(model[i] - model[1]),
            validation_gain_vs_actor=stats(model[i] - model[0])) for name, i in indices.items()})


def record(task=None, cfg=None):
    cfg = campaign() if cfg is None else cfg
    task = cfg['task_list'][0] if task is None else task
    n = cfg['max_steps']; times = [1.] * n; selection = [.1] * n
    work = dict(actor_updates=24, critic_updates=6 * task['critic_updates_per_round'], model_steps=768)
    return dict(schema_version=1, protocol=pub.PROTOCOL, campaign_id=cfg['campaign_id'], **task,
        complete=True, episode_return=float(task['env_seed'] * 10 + task['critic_updates_per_round']),
        episode_length=n, terminated=False, truncated=True, controller_times_s=times, selection_times_s=selection,
        controller_time_mean_s=1., controller_time_p95_s=1., selection_time_mean_s=.1, selection_time_p95_s=.1,
        diagnostics=[diagnostic(d) for d in cfg['diagnostic_decisions']],
        expected_work_per_decision=work, realized_work_totals={k: v*n for k,v in work.items()},
        runtime=dict(execution='eager', gpu='NVIDIA RTX A5000'),
        checks={k: True for k in ('outer_state_unchanged','full_episode','finite_returns','action_bounds',
                                 'complete_candidate_banks','independent_selection_validation','critic_update_dose_verified','diagnostic_coverage')})


def reference(cfg):
    episodes=[]
    for old, arm in pub.REF_ARMS.items():
        for seed in cfg['seeds']:
            episodes.append(dict(**pub.arm_info(arm), task_id=f'{arm}-seed-{seed}', env_seed=seed,
                controller_seed=55, state='complete', episode_length=500,
                episode_return=float(seed*10+16), reference_reused=False,
                controller_time_mean_s=1., controller_time_p95_s=1., selection_time_mean_s=.1, selection_time_p95_s=.1))
    return dict(snapshot_sha256=pub.REFERENCE_SNAPSHOT_SHA256, episodes=episodes, summary=[], diagnostics=[], result_fingerprints=[])


def write_result(root, data):
    path=root/'tasks'/data['task_id']/'result.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


def test_exact_grid_accepts_reordering_but_not_duplicate_or_wrong_dose():
    cfg=campaign();cfg['task_list'].reverse();assert len(pub.validate_campaign(cfg))==8
    cfg['task_list'][0]['critic_updates_per_round']=16
    with pytest.raises(pub.DoseDataError,match='task'):pub.validate_campaign(cfg)


def test_actual_runner_task_order_is_accepted():
    from evaluate_ambi_critic_dose import load_config, task_list
    cfg=load_config(Path('configs/research/ambi_800k_critic_dose.json'))
    cfg.update(campaign_id='a'*64,task_list=task_list(cfg))
    assert pub.validate_campaign(cfg)==cfg['task_list']
    assert cfg['task_list'][0]['arm']=='learned_q_c64'
    cfg['batch_size']=128
    with pytest.raises(pub.DoseDataError,match='A4/N128/B256'):pub.validate_campaign(cfg)
    cfg=campaign();cfg['task_list'][1]=cfg['task_list'][0]
    with pytest.raises(pub.DoseDataError,match='task'):pub.validate_campaign(cfg)


def test_complete_seed_coverage_paired_c16_gain_and_dose_metadata(tmp_path):
    cfg=campaign();ref=reference(cfg)
    first=record(cfg['task_list'][0],cfg);write_result(tmp_path,first)
    partial=pub.collect(tmp_path,cfg,ref)
    assert partial['complete']==1 and partial['pending']==7
    assert not any(r['source']=='new' for r in partial['summary'])
    for task in cfg['task_list']:write_result(tmp_path,record(task,cfg))
    final=pub.collect(tmp_path,cfg,ref,workers_active=False)
    assert final['complete']==8 and final['state']=='complete' and len(final['episodes'])==18
    pair=next(r for r in final['pairs'] if r['arm']=='actor_mean_c32' and r['baseline']=='matching_c16')
    assert pair['mean']==pair['ci_low']==pair['ci_high']==16 and pair['win_rate']==1
    assert pair['n']==2 and pair['critic_updates']==32 and pair['method']=='Actor mean'
    assert {r['metric'] for r in final['summary'] if r['arm']=='actor_mean_c32'} >= {'bank_spearman_model','bank_model_regret_of_qpick','heldout_q_gain_vs_actor'}


def test_normalization_preserves_true_identity_and_computes_ranking():
    cfg=campaign();data=record();before=deepcopy(data)
    episode,timing,diagnostics=pub.normalize_result(data,cfg['task_list'][0],cfg)
    assert data==before and episode['arm']=='actor_mean_c32' and episode['critic_updates']==32
    rho=[r['value'] for r in diagnostics if r['metric']=='bank_spearman_model']
    assert np.allclose(rho,[-1,-1])
    regret=[r['value'] for r in diagnostics if r['metric']=='bank_model_regret_of_qpick']
    assert regret==[769.,769.]
    assert np.allclose(pub.average_ranks([2,1,1,3]),[2,.5,.5,3])


@pytest.mark.parametrize('mutation,match',[
    (lambda r:r['checks'].pop('action_bounds'),'integrity'),
    (lambda r:r['realized_work_totals'].update(critic_updates=1),'dose'),
    (lambda r:r['runtime'].update(gpu='Other GPU'),'A5000'),
    (lambda r:r['diagnostics'].pop(),'diagnostic decisions'),
    (lambda r:r['diagnostics'][0].update(bank_count=769),'bank'),
    (lambda r:r['diagnostics'][0].update(validation_seed=1),'distinct'),
    (lambda r:r['diagnostics'][0]['selection_q_scores'].pop(),'scores'),
    (lambda r:r['diagnostics'][0]['choices']['learned_q'].update(index=0),'index'),
    (lambda r:r['diagnostics'][0]['choices']['learned_q']['validation_model']['draws'].pop(),'32 draws'),
    (lambda r:r['diagnostics'][0]['choices']['learned_q']['validation_model'].update(mean=900.),'differs'),
])
def test_missing_or_inconsistent_science_rejected(mutation,match):
    cfg=campaign();data=record();mutation(data)
    with pytest.raises(pub.DoseDataError,match=match):pub.normalize_result(data,cfg['task_list'][0],cfg)


def test_corrupt_reference_snapshot_rejected_before_raw_access(tmp_path):
    p=tmp_path/'snapshot.json';p.write_text(json.dumps(dict(snapshot_sha256=pub.REFERENCE_SNAPSHOT_SHA256)))
    with pytest.raises(pub.DoseDataError,match='content/hash'):pub.load_references(p,tmp_path/'missing')


def test_reference_loader_requires_exact_recomputation_and_runtime(tmp_path,monkeypatch):
    cfg=campaign();ref=reference(cfg)
    raw=record();raw['diagnostics']=[]
    old_episode=dict(ref['episodes'][0]);old_episode.update(arm='actor_mean',task_id='actor_mean-seed-101')
    snapshot=dict(campaign_id=pub.REFERENCE_CAMPAIGN_ID,complete=100,state='complete',episodes=[old_episode],summary=[],diagnostics=[],result_fingerprints=[])
    sha=pub._hash(snapshot);snapshot['snapshot_sha256']=sha
    monkeypatch.setattr(pub,'REFERENCE_SNAPSHOT_SHA256',sha)
    p=tmp_path/'snapshot.json';p.write_text(json.dumps(snapshot));(tmp_path/'campaign.json').write_text('{}')
    raw['task_id']='actor_mean-seed-101';write_result(tmp_path,raw)
    monkeypatch.setattr(pub.previous,'collect',lambda *a,**k:deepcopy(snapshot))
    loaded=pub.load_references(p,tmp_path)
    assert loaded['episodes'][0]['arm']=='actor_mean_c16'
    monkeypatch.setattr(pub.previous,'collect',lambda *a,**k:{})
    with pytest.raises(pub.DoseDataError,match='reproduce'):pub.load_references(p,tmp_path)


def test_diagnostic_intervals_cluster_roots_within_seed(tmp_path):
    cfg=campaign();ref=reference(cfg)
    for task in cfg['task_list']:write_result(tmp_path,record(task,cfg))
    snapshot=pub.collect(tmp_path,cfg,ref)
    row=next(r for r in snapshot['summary'] if r['arm']=='actor_mean_c32' and r['metric']=='bank_spearman_model')
    assert row['n']==2 and row['mean']==pytest.approx(-1) # Four roots are two seeds, not four independent samples.


def test_constant_critic_scores_preserve_undefined_rank_and_coverage(tmp_path):
    cfg=campaign();ref=reference(cfg)
    for task in cfg['task_list']:
        data=record(task,cfg)
        if task['env_seed']==101:
            for diagnostic_row in data['diagnostics']:
                diagnostic_row['selection_q_scores']=[0.]*770
                diagnostic_row['choices']['learned_q']=deepcopy(diagnostic_row['choices']['actor_mean'])
                for choice in diagnostic_row['choices'].values():choice['learned_q']=0.
        write_result(tmp_path,data)
    snapshot=pub.collect(tmp_path,cfg,ref)
    row=next(r for r in snapshot['summary'] if r['arm']=='actor_mean_c32' and r['metric']=='bank_spearman_model')
    coverage=next(r for r in snapshot['summary'] if r['arm']=='actor_mean_c32' and r['metric']=='bank_spearman_defined_root_fraction')
    assert row['n']==1 and coverage['n']==2 and coverage['mean']==.5
    assert any(r['bank_spearman_model'] is None for r in snapshot['diagnostic_table'] if r['candidate']=='bank')


def test_actual_sdk_serializes_all_final_diagnostic_rows_without_global_limits_change():
    import wandb
    old_limit=wandb.Table.MAX_ROWS
    long=[]
    arms=[a for a in layout.ARMS if a not in ('prior','mppi_h3')]
    for arm in arms:
        for seed in range(101,121):
            for decision in (0,25,100,250,400,499):
                base=dict(**pub.arm_info(arm),env_seed=seed,controller_seed=55,decision=decision)
                for candidate in ('actor_mean','learned_q','model_score','prior_mean'):
                    for metric in pub.DIAGNOSTIC_METRICS[:5]:
                        long.append(dict(**base,candidate=candidate,metric=metric,value=1.))
                for metric in pub.DIAGNOSTIC_METRICS[-3:]:
                    long.append(dict(**base,candidate='bank',metric=metric,value=1.))
    assert len(long)==19320
    compact=pub.compact_diagnostics(long)
    assert len(compact)==4200
    table=wandb.Table(columns=list(pub.DIAGNOSTIC_COLUMNS),data=[[r.get(c) for c in pub.DIAGNOSTIC_COLUMNS] for r in compact])
    serialized=table._to_table_json()
    assert len(serialized['data'])==4200 and serialized['data']==table.data
    assert wandb.Table.MAX_ROWS==old_limit


def test_acknowledgement_supports_sdk_subdict_and_checks_counts(tmp_path):
    class SubDict:
        def __init__(self,value):self.value=value
        def get(self,key,default=None):return self.value.get(key,default)
    snapshot=pub.collect(tmp_path,campaign(),reference(campaign()))
    values={'critic_dose/snapshot_sha256':snapshot['snapshot_sha256']}
    values.update({key:SubDict(dict(nrows=len(snapshot[name]),ncols=len(cols))) for key,(name,cols) in pub.TABLES.items()})
    api=SimpleNamespace(run=lambda _:SimpleNamespace(summary=SubDict(values)))
    state=dict(entity='e',project='p',run_id='r')
    assert pub.acknowledged(api,state,snapshot)
    values[pub.SUMMARY_KEY].value['nrows']+=1
    assert not pub.acknowledged(api,state,snapshot)


def template():
    return dict(section=dict(runSets=[{'keep':'filters'}],panelBankConfig=dict(sections=[],panelPlacementOverrides={})))


class Service:
    def __init__(self):
        self.views=[dict(id='personal',name=layout.DEFAULT_VIEW_NAME,type='project-view',spec=json.dumps(template()),displayName='Personal')]
        self.chart=None;self.writes=0
    def execute_graphql(self,query,variables):
        if 'CreateCriticDoseChart' in query:
            self.chart=dict(id=variables['name'],type=variables['type'],spec=variables['spec']);return {'createCustomChart':{'chart':deepcopy(self.chart)}}
        if 'CriticDoseChart' in query:return {'customChart':deepcopy(self.chart)}
        if 'mutation ' in query:
            self.writes+=1;self.views.append(dict(id='new',name=variables['name'],type=variables['type'],spec=variables['spec'],displayName=variables['displayName']))
            return {'upsertView':{'view':{'id':'new'}}}
        return {'project':{'allViews':{'edges':[{'node':deepcopy(v)} for v in self.views]}}}


def test_saved_view_is_distinct_and_colors_match_methods_not_doses(tmp_path):
    service=Service();before=deepcopy(service.views);api=SimpleNamespace(_service_api=service)
    kw=dict(entity='e',project='ambi-inner-bench',publication_id='abc123',receipt_dir=tmp_path)
    result=layout.ensure_saved_view(api,**kw)
    assert result['url'].endswith('?nw=criticdoseabc123') and service.views[:1]==before
    assert not layout.ensure_saved_view(api,**kw)['changed'] and service.writes==1
    definition=layout.chart_definition()
    assert definition['encoding']['color']['scale']['domain']==list(layout.METHODS)
    assert len(set(layout.COLORS))==5
    assert definition['layer'][0]['encoding']['x']['axis']['values']==[16,32,64]
    panels=[p for s in layout.sections('e') for p in s['panels']]
    assert len([p for p in panels if p['viewType']=='Vega2'])==10
    spec=json.loads(service.views[-1]['spec']);spec['section']['runSets'][0]['name']='User change';service.views[-1]['spec']=json.dumps(spec)
    with pytest.raises(layout.DoseLayoutConflict,match='preserving user edits'):layout.ensure_saved_view(api,**kw)
