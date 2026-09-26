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
        assert 'first J10' not in cfg['stringSettings']['xname']
        assert 'subsequent' not in cfg['stringSettings']['title']
        actual.append(cfg['userQuery']['queryFields'][0]['fields'][0]['args'][0]['value'])
    assert actual==[key+'_table' for key in layout.CHART_KEYS]
    assert {p['config']['mediaKeys'][0] for p in panels if p['viewType']=='Media Browser'}==set(layout.TABLE_KEYS)
    assert len({p['__id__'] for p in panels})==len(panels)
    intro=next(p['config']['value'] for p in panels if p['viewType']=='Markdown Panel')
    assert 'pending' in intro and 'completed' in intro and '575k' not in intro
    assert 'corrected study uses J at every decision' in intro
    assert 'original study used first J10' in intro


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


def ui_normalized_spec():
    """Omissions observed after the live W&B UI resaved the results workspace."""
    spec = layout.patch_actor_transfer_spec(sample_spec())
    overview, curves, tables = layout._bank(spec)['sections'][:3]
    for section in (overview, curves, tables):
        section.pop('type')
        for panel in section['panels']:
            panel.pop('layout')
    overview['flowConfig'] = {'columnsPerPage':1}
    curves.pop('flowConfig')
    tables['flowConfig'] = {'columnsPerPage':2, 'rowsPerPage':1}
    return spec


def test_installed_accepts_observed_omitted_defaults_without_rewriting_view(tmp_path):
    spec = ui_normalized_spec(); before = deepcopy(spec)
    assert layout._installed(spec) and spec == before
    service = Service(); service.views[0]['spec'] = json.dumps(spec)
    before_views = deepcopy(service.views)
    receipt = install(tmp_path, service)
    assert receipt['status'] == 'verified' and receipt['changed'] is False
    assert service.writes == 0 and service.views == before_views


@pytest.mark.parametrize('corruption', [
    'section_id', 'panel_id', 'panel_missing', 'panel_order', 'section_hidden', 'automatic_panels',
    'panel_automatic', 'query', 'title', 'media_key', 'section_type', 'flow_value', 'flow_null',
    'flow_extra', 'layout_value', 'layout_null', 'layout_partial',
])
def test_installed_normalization_still_rejects_changed_values_and_content(corruption):
    spec = ui_normalized_spec()
    overview, curves, tables = layout._bank(spec)['sections'][:3]
    chart = curves['panels'][0]
    if corruption == 'section_id': curves['__id__'] = 'wrong-section'
    elif corruption == 'panel_id': chart['__id__'] = 'wrong-panel'
    elif corruption == 'panel_missing': curves['panels'].pop()
    elif corruption == 'panel_order': curves['panels'].reverse()
    elif corruption == 'section_hidden': curves['isOpen'] = False
    elif corruption == 'automatic_panels': curves['isPanelsAuto'] = True
    elif corruption == 'panel_automatic': chart['isAuto'] = True
    elif corruption == 'query': chart['config']['userQuery']['queryFields'][0]['fields'][0]['args'][0]['value'] = 'wrong_table'
    elif corruption == 'title': chart['config']['stringSettings']['title'] = 'Stale results'
    elif corruption == 'media_key': tables['panels'][0]['config']['mediaKeys'] = ['wrong_metric']
    elif corruption == 'section_type': curves['type'] = 'grid'
    elif corruption == 'flow_value': overview['flowConfig']['columnsPerPage'] = 99
    elif corruption == 'flow_null': curves['flowConfig'] = None
    elif corruption == 'flow_extra': overview['flowConfig']['unknown'] = True
    elif corruption == 'layout_value': chart['layout'] = {'x':0, 'y':0, 'w':1, 'h':6}
    elif corruption == 'layout_null': chart['layout'] = None
    elif corruption == 'layout_partial': chart['layout'] = {'w':8, 'h':6}
    assert not layout._installed(spec)


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


def test_uniform_overview_preserves_reused_identity_and_pending_nulls(monkeypatch):
    from slurm import ambi_actor_transfer_publish as publication
    panel = [dict(name='cold_h1_j10_c16', H=1, J=10, transfer_mode='cold',
                  performance_run_id='original-id', reused=True,
                  reuse_provenance={'source_protocol':'actor-transfer-v1'}),
             dict(name='actor_warm_h1_j1_c16', H=1, J=1, transfer_mode='actor_warm',
                  performance_run_id='new-id')]
    monkeypatch.setattr(publication, 'validate_scope', lambda campaign: panel)
    result = publication.aggregate_results({'study_protocol':'uniform-J'}, {
        panel[0]['name']:dict(episodes=[{'return':v} for v in (10,20,30,40,50)],
            metrics={'runtime/control_seconds':125., 'runtime/control_seconds_per_decision':.05},
            diagnostics={'stage_rows':[]})})
    inherited, pending = result['points']
    assert inherited['return_mean'] == 30 and inherited['first_action_rounds'] == 10
    assert inherited['performance_url'].endswith('/runs/original-id')
    assert inherited['reuse_provenance']['source_protocol'] == 'actor-transfer-v1'
    assert inherited['reused'] is True
    assert pending['first_action_rounds'] == 1 and pending['return_mean'] is None
    assert pending['reused'] is False and result['paired_comparisons'] == []
    assert result['rounds_policy'] == 'J at every decision including the first'


def test_uniform_scope_accepts_only_pinned_six_j10_reuses(tmp_path, monkeypatch):
    from slurm import ambi_actor_transfer_campaign as campaign
    from slurm import ambi_actor_transfer_publish as publication
    from utils import eval_series_data
    science = {'version':1, 'source_fingerprint':'same-controller-source'}
    monkeypatch.setattr(eval_series_data, 'scientific_identity', lambda *args:science)
    panel = campaign.cells()
    for index, cell in enumerate(panel):
        cell.update(expected_config=deepcopy(cell['requested_alg_params']),
            performance_run_id=f'new-{index}',
            identity={'science':science, 'planner':{'settings':deepcopy(cell['params'])}, 'controller_seed':55},
            directory=f'/original/{cell["name"]}', bundle=f'/original/{cell["name"]}/bundle',
            run_dir=f'/registry/{cell["name"]}', checkpoint='/checkpoints/575000.pt',
            checkpoint_sha256=campaign.CHECKPOINT_SHA, metadata_sha256='metadata-hash', initial_alpha=.01)
    source = dict(matrix=str(campaign.MATRIX), study_protocol='actor-transfer-v1',
        group='original-group', source_commit='original-commit', source_run=campaign.SOURCE_RUN,
        checkpoint_step=campaign.CHECKPOINT_STEP, checkpoint_sha256=campaign.CHECKPOINT_SHA,
        first_action_rounds=10, prior_reference={'manifest_sha256':'prior-hash'}, cells=deepcopy(panel))
    for cell in source['cells']:
        cell['first_action_rounds'] = 10
        cell['requested_alg_params']['inner_first_action_rounds'] = 10
        cell['expected_config']['inner_first_action_rounds'] = 10
        cell['performance_run_id'] = 'original-' + cell['name']
        cell['identity']['planner']['settings']['inner_first_action_rounds'] = 10
        cell['identity']['planner']['semantics'] = dict(evaluation_protocol='actor-transfer-v1', first_action_rounds=10)
    path = tmp_path/'original-campaign.json'
    path.write_text(json.dumps(source))
    reused = []
    for cell, original in zip(panel, source['cells']):
        if cell['J'] != 10: continue
        cell.update(reused=True, corrected_identity=deepcopy(cell['identity']), performance_run_id=original['performance_run_id'],
            identity=deepcopy(original['identity']), source_expected_config=deepcopy(original['expected_config']),
            reuse_provenance=dict(campaign_path=str(path), campaign_sha256=campaign.digest(path),
                study_protocol='actor-transfer-v1', group=source['group'], source_commit=source['source_commit'],
                implementation_fingerprint=science))
        reused.append(cell['name'])
    corrected = {**source, 'study_protocol':campaign.PROTOCOL, 'group':'corrected-group',
        'source_commit':'corrected-commit', 'first_action_rounds':None, 'cells':panel,
        'overview_run_id':'corrected-overview', 'reused_cells':reused}
    assert len(publication.validate_scope(corrected)) == 36 and len(reused) == 6
    point = next(c for c in panel if c['reused'])
    assert point['expected_config']['inner_first_action_rounds'] is None
    assert point['identity']['planner']['settings']['inner_first_action_rounds'] == 10
    for key, value in [('identity', point['corrected_identity']), ('performance_run_id', 'duplicate-new-run'),
                       ('source_expected_config', point['expected_config'])]:
        corrupt = deepcopy(corrected)
        next(c for c in corrupt['cells'] if c['reused'])[key] = value
        with pytest.raises(AssertionError): publication.validate_scope(corrupt)
    corrupt = deepcopy(corrected)
    corrupt['cells'][0].update(reused=True, reuse_provenance=deepcopy(point['reuse_provenance']))
    with pytest.raises(AssertionError, match='Only the six J10'):
        publication.validate_scope(corrupt)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(AssertionError, match='Reuse source campaign changed'):
        publication.validate_scope(corrected)


def test_uniform_overview_chart_labels_and_compute_order():
    from slurm.ambi_actor_transfer_publish import overview_payload
    fake = SimpleNamespace(Table=lambda **kw:kw, plot=SimpleNamespace(line_series=lambda **kw:kw))
    points = [dict(H=1, J=j, transfer_mode='cold', return_mean=ret,
                   control_seconds_per_decision=seconds, first_action_rounds=j,
                   reused=(j==10), performance_url='original' if j==10 else 'new')
              for j,ret,seconds in ((1,20.,.2),(10,30.,.1))]
    aggregate = dict(points=points, paired_comparisons=[], diagnostics=[], completed=2, total=36)
    payload = overview_payload(fake, aggregate)
    rounds = payload['comparison/h1_return_vs_rounds']
    assert rounds['xname'] == 'J rounds / decision'
    assert 'every decision' in rounds['title'] and 'first J10' not in rounds['title']
    assert rounds['keys'] == ['Cold (reset actor)', 'Warm (retain actor)']
    compute = payload['comparison/h1_return_vs_compute']
    assert compute['xs'][0] == [.1,.2] and compute['ys'][0] == [30.,20.]
    assert 'reused' in payload['comparison/returns_and_compute']['columns']
    assert 'comparison/h2_return_vs_rounds' not in payload


def test_reused_completed_publication_never_stages_or_uploads(tmp_path, monkeypatch):
    from slurm import ambi_actor_transfer_publish as publication
    from utils import ambi_benchmark, eval_series
    cell = dict(name='cold_h1_j10_c16', reused=True, directory=str(tmp_path/'old-cell'),
                run_dir='original-registry', identity={'original':True}, performance_run_id='original')
    campaign = {'cells':[cell]}
    (tmp_path/'campaign.json').write_text(json.dumps(campaign))
    directory = tmp_path/'old-cell';directory.mkdir()
    receipt = {'status':'complete','performance':{'run_id':'original'}}
    (directory/'publication-completion.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(publication,'validate_scope',lambda campaign:[cell])
    monkeypatch.setattr(publication,'load_completed',lambda *args:{'metrics':{}})
    monkeypatch.setattr(eval_series,'load_run',lambda directory:{'identity':cell['identity']})
    monkeypatch.setattr(ambi_benchmark,'stage_completed_bundle',lambda *args,**kwargs:pytest.fail('Restaged reused data'))
    monkeypatch.setattr(publication,'publish_performance',lambda *args:pytest.fail('Republished reused data'))
    assert publication.publish_cell(SimpleNamespace(root=tmp_path,index=0)) == receipt
    (directory/'publication-completion.json').unlink()
    with pytest.raises(RuntimeError,match='never republish'):
        publication.publish_cell(SimpleNamespace(root=tmp_path,index=0))
