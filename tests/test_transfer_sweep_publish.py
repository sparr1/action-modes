"""Read-only evaluation ingestion and resumable publication contracts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_sweep_publish as publication
from slurm import ambi_transfer_sweep_campaign as campaign_helpers


def campaign():
    panel = campaign_helpers.cells()
    for index, cell in enumerate(panel):
        cell['index'] = index
    panel.append(dict(index=24, name='prior_reference', transfer_mode='prior', J=0, solve_interval=1))
    return dict(cells=panel, source_commit='e'*40, checkpoint_sha256='a'*64,
                checkpoint_step=575000, source_run=campaign_helpers.SOURCE_RUN)


def episodes(offset):
    return [dict(seed=seed, solver_seed=seed+1234, return_=offset+index, length=500,
                 control_seconds=5., terminated=False, truncated=True)
            for index, seed in enumerate(publication.SEEDS)]


def completed(offset):
    rows = episodes(offset)
    for row in rows:
        row['return'] = row.pop('return_')
    return {'episodes': rows, 'record': {'identity': {'fixture': True}, 'record_id': str(offset)}}


class FakeWandb:
    def __init__(self):
        self.plot = SimpleNamespace(line_series=lambda **kwargs: kwargs)
        self.logs=[]; self.summary={}
    def Table(self, **kwargs):
        return kwargs
    def init(self, **kwargs):
        self.id=kwargs['id']; self.init_args=kwargs
        return self
    def log(self, payload):
        self.logs.append(payload)
    def finish(self, **kwargs):
        self.finished=kwargs


def test_pending_overview_has_all25_rows_and8_curves(tmp_path):
    value=campaign(); state={'cells':{}}
    progress=publication.progress_rows(tmp_path,value,state,{}, {})
    snapshot=publication.aggregate(value,progress,{})
    payload=publication.overview_payload(FakeWandb(),snapshot)
    assert snapshot['completed']==0 and snapshot['total']==25
    assert len(payload['transfer_sweep/settings']['data'])==25
    assert len(payload['transfer_sweep/results']['data'])==25
    assert all(row['state']=='pending' and row['return_mean'] is None for row in snapshot['results'])
    assert len([k for k in payload if '_return_vs_' in k])==8
    assert payload['transfer_sweep/episodes']['data']==[]
    assert payload['transfer_sweep/paired_effects']['data']==[]


def test_progress_counts_only_complete_episodes_without_publishing_partial_results(tmp_path):
    value=campaign(); cell=value['cells'][0]
    directory=tmp_path/'settings'/cell['name']/'bundle'; directory.mkdir(parents=True)
    (directory/'manifest.json').write_text(json.dumps({'runs':[{'episodes':[{'length':500},{'length':217}]}]}))
    progress=publication.progress_rows(tmp_path,value,{'cells':{}},{},{})
    assert progress[0]['state']=='running' and progress[0]['completed_episodes']==1
    snapshot=publication.aggregate(value,progress,{})
    assert snapshot['results'][0]['return_mean'] is None
    assert snapshot['episodes']==[]


def test_aggregate_pairs_complete_panels_by_both_seeds_and_retains_prior(tmp_path):
    value=campaign(); fresh='soft_fresh_h3_j8_i1'; actor='soft_actor_only_h3_j8_i1'; critic='soft_critic_only_h3_j8_i1'
    ready={fresh:completed(20), actor:completed(30), critic:completed(40), 'prior_reference':completed(10)}
    ready[actor]['episodes'].reverse()
    progress=publication.progress_rows(tmp_path,value,{'cells':{}},ready,{})
    snapshot=publication.aggregate(value,progress,ready)
    actor_result=next(row for row in snapshot['results'] if row['setting']==actor)
    assert actor_result['return_mean']==32 and actor_result['paired_vs_fresh_mean']==10
    assert actor_result['paired_vs_prior_mean']==20 and actor_result['controller_seconds_per_decision']==.01
    assert len(snapshot['episodes'])==20 and len(snapshot['paired_effects'])==5
    assert all(row['paired_episodes']==5 for row in snapshot['paired_effects'])
    ready[actor]['episodes'][0]['solver_seed']+=1
    with pytest.raises(ValueError,match='five-seed'):
        publication.aggregate(value,progress,ready)


def test_publication_state_reuses_ids_and_rejects_binding_changes(tmp_path):
    root=tmp_path/'campaign'; root.mkdir(); output=tmp_path/'publication'; output.mkdir()
    value=campaign(); (root/'campaign.json').write_text(json.dumps(value))
    first=publication.publication_state(root,output,value,'p'*40)
    assert publication.publication_state(root,output,value,'p'*40)==first
    with pytest.raises(ValueError,match='ownership'):
        publication.publication_state(root,output,value,'q'*40)
    (root/'campaign.json').write_text(json.dumps({**value,'changed':True}))
    with pytest.raises(ValueError,match='ownership'):
        publication.publication_state(root,output,value,'p'*40)


def test_publisher_lock_prevents_simultaneous_owners(tmp_path):
    with publication.publisher_lock(tmp_path):
        with pytest.raises(RuntimeError,match='Another publisher'):
            with publication.publisher_lock(tmp_path):
                pass


def test_cell_publication_only_writes_registry_and_reuses_mapping(tmp_path,monkeypatch):
    from utils import eval_series as series
    root=tmp_path/'publication'; root.mkdir(); frozen=tmp_path/'bundle'; frozen.mkdir()
    (frozen/'manifest.json').write_text('immutable'); before=publication.digest(frozen/'manifest.json')
    value=completed(20); state={'cells':{}}; cell={'name':'soft_critic_only_h3_j8_i1'}
    calls=[]; registry={'identity':value['record']['identity'],'run_id':'fixed','run_dir':str(root/'registry/cell/fixed')}
    monkeypatch.setattr(series,'create_run',lambda *a,**k:(calls.append('create') or registry))
    monkeypatch.setattr(series,'load_run',lambda *a:registry)
    monkeypatch.setattr(series,'stage_record',lambda *a:(calls.append('stage') or {'status':'staged'}))
    monkeypatch.setattr(series,'publish_run',lambda *a,**k:pytest.fail('Performance publishing must not initialize W&B in the overview process'))
    def child(command, **kwargs):
        calls.append('child')
        assert command[1:] == [str(publication.ROOT/'eval_series.py'), 'publish', registry['run_dir'], '--owner', 'oscar-rgao48']
        assert kwargs['check'] and kwargs['close_fds'] is False
        directory=Path(registry['run_dir']); directory.mkdir(parents=True,exist_ok=True)
        (directory/'publication.json').write_text(json.dumps({'records':{value['record']['record_id']:{'status':'published'}}}))
    monkeypatch.setattr(publication.subprocess,'run',child)
    publication.publish_cell(root,state,cell,value)
    publication.publish_cell(root,state,cell,value)
    assert calls==['create','stage','child','stage','child']
    assert state['cells'][cell['name']]['run_id']=='fixed'
    assert publication.digest(frozen/'manifest.json')==before
    assert list(frozen.iterdir())==[frozen/'manifest.json']


def test_watch_logs_progress_and_layout_before_completed_bundle_ingestion(tmp_path,monkeypatch):
    import sys
    value=campaign(); root=tmp_path/'campaign'; root.mkdir(); output=tmp_path/'publication'
    (root/'campaign.json').write_text(json.dumps(value))
    fake=FakeWandb(); calls=[]
    monkeypatch.setitem(sys.modules,'wandb',fake)
    monkeypatch.setattr(publication,'publisher_commit',lambda:'p'*40)
    monkeypatch.setattr(publication,'gpu_jobs_active',lambda ids:True)
    def layout(wandb,run,directory):
        assert len(fake.logs[0]['transfer_sweep/settings']['data'])==25
        calls.append('layout')
        return {'url':'https://wandb.ai/entity/project/runs/overview?nw=personal','status':'verified'}
    monkeypatch.setattr(publication,'install_layout',layout)
    args=SimpleNamespace(root=root,publication_root=output,gpu_job_id=['6966411'],once=True,
                         poll_seconds=.1,terminal_grace=0)
    publication.watch(args)
    assert calls==['layout']
    assert fake.init_args['mode']=='online' and fake.init_args['resume']=='allow'
    assert fake.summary['status']=='running'
    assert publication.read(output/'publisher-status.json')['overview_url'].endswith('?nw=personal')


def test_watch_refuses_publication_inside_campaign(tmp_path,monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'wandb',FakeWandb())
    args=SimpleNamespace(root=tmp_path,publication_root=tmp_path/'publication')
    with pytest.raises(ValueError,match='separate'):
        publication.watch(args)


def test_layout_failure_remains_visible_and_fails_watcher(tmp_path,monkeypatch):
    import sys
    root=tmp_path/'campaign'; root.mkdir(); output=tmp_path/'publication'
    (root/'campaign.json').write_text(json.dumps(campaign()))
    fake=FakeWandb(); monkeypatch.setitem(sys.modules,'wandb',fake)
    monkeypatch.setattr(publication,'publisher_commit',lambda:'p'*40)
    def fail(*args): raise RuntimeError('layout verification failed')
    monkeypatch.setattr(publication,'install_layout',fail)
    args=SimpleNamespace(root=root,publication_root=output,gpu_job_id=['6966411'],once=True,
                         poll_seconds=.1,terminal_grace=0)
    with pytest.raises(RuntimeError,match='layout verification'):
        publication.watch(args)
    assert publication.read(output/'publisher-failure.json')['error']=='layout verification failed'
    assert fake.summary['publisher/status']=='failed'
    assert fake.finished['exit_code']==1


def test_actual_install_layout_import_and_helper_contract(tmp_path,monkeypatch):
    from utils import wandb_transfer_sweep_layout as layout
    calls=[]
    api=object()
    wandb=SimpleNamespace(Api=lambda **kwargs:api)
    run=SimpleNamespace(id='overview',summary={})
    expected=dict(status='verified',url='https://wandb.ai/entity/project/runs/overview?nw=personal',
                  workspace_url='https://wandb.ai/entity/project/workspace?nw=personal')
    def install(actual_api, **kwargs):
        assert actual_api is api
        assert kwargs['entity']==publication.ENTITY and kwargs['project']==publication.PROJECT
        assert kwargs['run_id']=='overview' and kwargs['receipt_dir']==tmp_path/'results-layout'
        calls.append(kwargs)
        return expected
    monkeypatch.setattr(layout,'ensure_transfer_sweep_results_layout',install)
    assert publication.install_layout(wandb,run,tmp_path)==expected
    assert len(calls)==1 and run.summary['results_layout/schema_verified'] is True
    assert run.summary['results_layout/url']==expected['url']


def test_child_publication_requires_remote_acknowledgement(tmp_path,monkeypatch):
    from utils import eval_series as series
    root=tmp_path/'publication'; root.mkdir()
    run_dir=root/'registry'; run_dir.mkdir()
    value=completed(20); name='setting'; state={'cells':{name:dict(run_dir=str(run_dir),run_id='id',record_id='20')}}
    monkeypatch.setattr(series,'load_run',lambda *a:{'identity':value['record']['identity']})
    monkeypatch.setattr(series,'stage_record',lambda *a:None)
    def child(*args,**kwargs):
        (run_dir/'publication.json').write_text(json.dumps({'records':{'20':{'status':'queued'}}}))
    monkeypatch.setattr(publication.subprocess,'run',child)
    with pytest.raises(ValueError,match='did not acknowledge'):
        publication.publish_cell(root,state,{'name':name},value)
    assert state['cells'][name].get('status')!='published'
