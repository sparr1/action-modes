"""Visible results are a publication contract, separate from stored metric data."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from utils import wandb_results_layout as layout


def sample_spec():
    return {'vizExpanded':False,'section':{'runSets':[{'filters':{'preserve':True}}],
        'settings':{'preserve':'value'},'workspaceSettings':{'shouldAutoGeneratePanels':False},
        'panelBankConfig':{'state':1,'settings':{'showEmptySections':False},
            'panelPlacementOverrides':{'existing':{'sectionId':'legacy'}},
            'sections':[{'__id__':'legacy','name':'User results','isOpen':False,'panels':[{'custom':1}]},
                        {'__id__':'system','name':'System','isOpen':True,'panels':[]}]}}}


def view(spec=None, name=layout.DEFAULT_VIEW_NAME, id='personal', kind='project-view'):
    return dict(id=id,name=name,type=kind,displayName='Existing personal project workspace',
                spec=json.dumps(spec or sample_spec()),entityName='entity',projectName='project',
                projectId=42,parentId=None,userId=7)


class Service:
    def __init__(self, *, timeout=False, apply=True, concurrent=False):
        self.views=[view(),view(name=layout.DEFAULT_VIEW_NAME,id='legacy-run',kind='run-view'),
                    view(name='nw-other-v',id='other',kind='project-view')]
        self.timeout=timeout;self.apply=apply;self.concurrent=concurrent
        self.reads=0;self.writes=0;self.mutations=[]
    def execute_graphql(self, query, variables):
        if 'mutation ' in query:
            self.writes+=1;self.mutations.append(variables)
            if self.apply:self.views[0]['spec']=variables['spec']
            if self.timeout:raise TimeoutError('simulated response loss')
            return {'upsertView':{'view':{'id':'personal','name':layout.DEFAULT_VIEW_NAME},'inserted':False}}
        self.reads+=1
        if self.concurrent and self.reads==2:self.views[0]['displayName']='Changed concurrently'
        assert 'project(' in query and 'viewType:"project-view"' in query and 'viewer' not in query
        assert variables=={'entityName':'entity','name':'project'}
        return {'project':{'allViews':{'edges':[{'node':deepcopy(v)} for v in self.views]}}}


def install(tmp_path, service):
    return layout.ensure_actor_transfer_results_layout(SimpleNamespace(_service_api=service),
        entity='entity',project='project',receipt_dir=tmp_path)


def test_patch_is_idempotent_and_preserves_unrelated_settings_and_filters():
    original=sample_spec();before=deepcopy(original)
    patched=layout.patch_actor_transfer_spec(original)
    assert original==before
    assert layout.patch_actor_transfer_spec(patched)==patched
    assert layout._without_owned(patched)==original
    assert patched['section']['runSets']==before['section']['runSets']
    assert patched['section']['settings']==before['section']['settings']
    assert len(patched['section']['panelBankConfig']['sections'])==5


def test_visible_panel_schema_and_exact_publisher_keys():
    sections=layout.actor_transfer_sections()
    assert all(s['isOpen'] and not s['isPanelsAuto'] for s in sections)
    panels=[p for s in sections for p in s['panels']]
    assert all(p['isAuto'] is False for p in panels)
    charts=[p for p in panels if p['viewType']=='Vega2']
    assert len(charts)==6
    actual=[]
    for panel in charts:
        cfg=panel['config'];assert cfg['panelDefId']=='wandb/lineseries/v0'
        assert cfg['fieldSettings']=={'step':'step','lineKey':'lineKey','lineVal':'lineVal'}
        actual.append(cfg['userQuery']['queryFields'][0]['fields'][0]['args'][0]['value'])
    assert actual==[key+'_table' for key in layout.CHART_KEYS]
    assert {p['config']['mediaKeys'][0] for p in panels if p['viewType']=='Media Browser'}==set(layout.TABLE_KEYS)
    assert len({p['__id__'] for p in panels})==len(panels)
    intro=next(p['config']['value'] for p in panels if p['viewType']=='Markdown Panel')
    assert 'pending' in intro and 'completed' in intro and '575k' not in intro


def test_install_reads_back_preserves_other_views_and_is_noop_on_repeat(tmp_path):
    service=Service();before=deepcopy(service.views)
    receipt=install(tmp_path,service)
    assert receipt['status']=='verified' and receipt['changed'] and service.writes==1
    assert service.views[1:]==before[1:]
    assert not {'entityName','projectName','projectId','parentId','userId'} & service.mutations[0].keys()
    assert service.mutations[0]['id']=='personal' and service.mutations[0]['type']=='project-view'
    assert layout._without_owned(json.loads(service.views[0]['spec']))==json.loads(before[0]['spec'])
    second=install(tmp_path,service)
    assert second['status']=='verified' and second['changed'] is False and service.writes==1
    assert json.loads((tmp_path/'results-layout-receipt.json').read_text())==second


def test_concurrent_edit_aborts_before_write(tmp_path):
    service=Service(concurrent=True)
    with pytest.raises(layout.ResultsLayoutError,match='changed during preparation'):install(tmp_path,service)
    assert service.writes==0


def test_uncertain_applied_write_is_reconciled_without_duplicate_mutation(tmp_path):
    service=Service(timeout=True)
    receipt=install(tmp_path,service)
    assert receipt['status']=='verified' and receipt['uncertain_response_reconciled']
    assert install(tmp_path,service)['changed'] is False
    assert service.writes==1


def test_unapplied_write_keeps_failure_receipt_and_safe_retry(tmp_path):
    service=Service(timeout=True,apply=False)
    with pytest.raises(layout.ResultsLayoutError,match='could not be verified'):install(tmp_path,service)
    assert json.loads((tmp_path/'results-layout-receipt.json').read_text())['status']=='uncertain'
    service.timeout=False;service.apply=True
    assert install(tmp_path,service)['status']=='verified'
    ids=[s['__id__'] for s in json.loads(service.views[0]['spec'])['section']['panelBankConfig']['sections']]
    assert len(ids)==len(set(ids))


def test_missing_or_nonpersonal_view_never_creates_view(tmp_path):
    service=Service();service.views[0]['name']='nw-different-user-w'
    with pytest.raises(layout.ResultsLayoutError,match='Expected one'):install(tmp_path,service)
    assert service.writes==0


def test_publisher_layout_failure_is_visible_and_does_not_abort_results(tmp_path,monkeypatch):
    from slurm.ambi_actor_transfer_publish import install_results_layout
    def fail(*args,**kwargs):raise layout.ResultsLayoutError('simulated transport issue')
    monkeypatch.setattr(layout,'ensure_actor_transfer_results_layout',fail)
    run=SimpleNamespace(summary={})
    receipt=install_results_layout(SimpleNamespace(Api=lambda **kw:object()),run,{},tmp_path)
    assert receipt['status']=='failed'
    assert run.summary['results_layout/status']=='failed'
    assert run.summary['results_layout/schema_verified'] is False
    assert (tmp_path/'results-layout-failure.json').exists()


def test_locked_sdk_legacy_graphql_transport_without_workspaces_dependency():
    calls=[]
    class Client:
        def execute(self,document,variable_values):
            calls.append((document,variable_values));return {'ok':True}
    assert layout._execute(SimpleNamespace(client=Client()),'query Results { viewer { id } }',{'x':1})=={'ok':True}
    assert len(calls)==1 and calls[0][1]=={'x':1}


def test_publisher_results_url_selects_exact_overview_in_verified_personal_layout(tmp_path,monkeypatch):
    from slurm.ambi_actor_transfer_publish import install_results_layout, ENTITY, PROJECT
    workspace_url=f'https://wandb.ai/{ENTITY}/{PROJECT}/workspace?nw=nwuserrwgao_b'
    expected=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/specific-overview?nw=nwuserrwgao_b'
    def installed(*args,**kwargs):
        assert kwargs['run_id']=='specific-overview'
        return dict(status='verified',url=expected,workspace_url=workspace_url,view_name=layout.DEFAULT_VIEW_NAME)
    monkeypatch.setattr(layout,'ensure_actor_transfer_results_layout',installed)
    run=SimpleNamespace(summary={})
    receipt=install_results_layout(SimpleNamespace(Api=lambda **kw:object()),run,
        {'overview_run_id':'specific-overview'},tmp_path)
    expected=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/specific-overview?nw=nwuserrwgao_b'
    assert receipt['url']==run.summary['results_layout/url']==expected
    assert receipt['workspace_url']==run.summary['results_layout/workspace_url']==workspace_url
    assert json.loads((tmp_path/'results-layout'/'campaign-results-layout.json').read_text())==receipt
