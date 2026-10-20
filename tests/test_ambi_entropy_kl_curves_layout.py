"""Native line layout remains connected and verified across repeated installs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from slurm import ambi_entropy_kl_curves_layout as layout


def catalog():
    return dict(entity='entity', project='project',
        workspace_url='https://wandb.ai/entity/project?nw=3w61ds5ze92',
        curves=[dict(run_id='run'+str(i), name=name, color=color, mode=mode, run_dir='/runs/'+mode)
                for i, (mode, name, color) in enumerate([
                    ('prior', 'MPPI-trained | sampled prior', '#6B7280'),
                    ('entropy', 'MPPI-trained | entropy only | H3 J8', '#0F766E'),
                    ('kl', 'MPPI-trained | KL only | H3 J8', '#DB2777'),
                    ('both', 'MPPI-trained | entropy + KL | H3 J8', '#2563EB')])])


class Object(SimpleNamespace):
    pass


class Filter:
    def __init__(self, name, values):
        self.name, self.values = name, values
    def to_model(self):
        return Object(model_dump=lambda **kwargs: dict(op='IN', key=dict(name=self.name), value=self.values))


class Metric:
    def __init__(self, name): self.name = name
    def isin(self, values): return Filter(self.name, values)


class LinePlot(Object):
    pass


def backend(value, mutate=None, expanded_colors=False):
    state = {'saved': {}, 'saves': 0}
    initial = Object(entity=value['entity'], project=value['project'], url=value['workspace_url'],
        sections=[Object(name='User notes', panels=[Object(text='Keep me')]),
                  Object(name=layout.REPLACED_PREFIX+'Return and improvement', panels=[]),
                  Object(name='Unrelated diagnostics', panels=[])],
        settings=Object(), runset_settings=Object())
    class Workspace:
        @classmethod
        def from_url(cls, url):
            work = deepcopy(state['saved'].get(url, initial))
            work.url = url
            def save():
                state['saves'] += 1
                state['saved'][url] = deepcopy(work)
            work.save = save
            if url in state['saved']:
                if expanded_colors:
                    for section in work.sections:
                        for panel in section.panels:
                            if isinstance(panel, LinePlot):
                                panel.line_colors = {key: dict(color=layout._color(color), transparentColor='#00000019')
                                                     for key, color in panel.line_colors.items()}
                if mutate: mutate(work)
            return work
    ws = Object(Workspace=Workspace, Section=Object,
                RunsetSettings=lambda **kwargs: Object(query='', **kwargs), RunSettings=Object)
    wr = Object(LinePlot=LinePlot, MarkdownPanel=Object, ScalarChart=Object, Layout=Object)
    return (ws, wr, Object(Metric=Metric)), state


def plots(work):
    return [p for section in work.sections for p in section.panels if isinstance(p, LinePlot)]


@pytest.mark.parametrize('expanded', [False, True])
def test_replaces_old_sections_preserves_unrelated_and_is_idempotent(tmp_path, expanded):
    value = catalog()
    sdk, state = backend(value, expanded_colors=expanded)
    receipt = tmp_path/'layout.json'
    first = layout.install_catalog(value, receipt, backend=sdk)
    second = layout.install_catalog(value, receipt, backend=sdk)
    assert first['url'] == second['url'] == value['workspace_url']
    assert second['api_verified'] and state['saves'] == 2
    work = state['saved'][value['workspace_url']]
    assert len(work.sections) == 5
    assert work.name == layout.WORKSPACE_NAME
    assert work.sections[0].panels[0].text == 'Keep me'
    assert work.sections[-1].name == 'Unrelated diagnostics'
    assert len(plots(work)) == len(layout.METRICS)
    for panel in plots(work):
        assert panel.x == layout.series.X_AXIS and panel.aggregate is False
        assert panel.smoothing_type == 'none'
        assert set(panel.line_marks.values()) == {'solid'}
        assert set(panel.line_titles.values()) == {row['name'] for row in value['curves']}
    assert work.settings.tooltip_number_of_runs == 'all_runs'
    assert layout._filter_model(work.runset_settings.filters[0])['value'] == [r['run_id'] for r in value['curves']]


@pytest.mark.parametrize('failure', ['marks', 'colors', 'axis', 'filters', 'grouping', 'aggregate',
                                     'legend', 'visibility', 'metrics', 'sections'])
def test_server_dropped_settings_fail_and_leave_unverified_receipt(tmp_path, failure):
    value = catalog()
    def corrupt(work):
        panel = plots(work)[0]
        if failure == 'marks': panel.line_marks = {key:'points' for key in panel.line_marks}
        elif failure == 'colors': panel.line_colors = {key:dict(color='#FFFFFF') for key in panel.line_colors}
        elif failure == 'axis': panel.x = 'Step'
        elif failure == 'filters': work.runset_settings.filters = [Metric('ID').isin(['wrong'])]
        elif failure == 'grouping': work.runset_settings.groupby = ['mode']
        elif failure == 'aggregate': panel.aggregate = True
        elif failure == 'legend': panel.line_titles = {}
        elif failure == 'visibility': work.runset_settings.run_settings['run0'].disabled = True
        elif failure == 'metrics': panel.y = ['missing/metric']
        elif failure == 'sections': work.sections.pop(1)
    sdk, _ = backend(value, mutate=corrupt)
    receipt = tmp_path/'layout.json'
    with pytest.raises(layout.LayoutError):
        layout.install_catalog(value, receipt, backend=sdk)
    saved = layout.series._read(receipt)
    assert saved['api_verified'] is False and saved['error_type'] == 'LayoutError'


def test_all_requested_views_are_verified(tmp_path):
    value = catalog()
    value['other_workspace_urls'] = ['https://wandb.ai/entity/project?nw=legacy']
    sdk, state = backend(value)
    result = layout.install_catalog(value, tmp_path/'layout.json', backend=sdk)
    assert len(result['views']) == state['saves'] == 2
    assert {v['view_id'] for v in result['views']} == {'3w61ds5ze92','legacy'}


@pytest.mark.parametrize('invalid', ['duplicate_run', 'duplicate_color', 'checkpoint_name', 'wrong_project', 'missing_view'])
def test_catalog_rejects_ambiguous_or_wrong_target_identity(invalid):
    value = catalog()
    if invalid == 'duplicate_run': value['curves'][1]['run_id'] = value['curves'][0]['run_id']
    if invalid == 'duplicate_color': value['curves'][1]['color'] = value['curves'][0]['color'].lower()
    if invalid == 'checkpoint_name': value['curves'][0]['name'] += ' 500k'
    if invalid == 'wrong_project': value['workspace_url'] = 'https://wandb.ai/entity/another?nw=id'
    if invalid == 'missing_view': value['workspace_url'] = 'https://wandb.ai/entity/project'
    with pytest.raises(layout.LayoutError):layout.validate_catalog(value)


def test_pinned_sdk_serialization_and_server_color_readback(tmp_path, monkeypatch):
    ws = pytest.importorskip('wandb_workspaces.workspaces')
    wr = pytest.importorskip('wandb_workspaces.reports.v2')
    if (layout.importlib.metadata.version('wandb') != '0.28.1'
            or layout.importlib.metadata.version('wandb-workspaces') != '0.4.5'):
        pytest.skip('Needs the isolated pinned layout runtime')
    value = catalog()
    initial = ws.Workspace(entity='entity', project='project', sections=[
        ws.Section(name='User notes', panels=[]),
        ws.Section(name=layout.REPLACED_PREFIX+'old', panels=[])])
    state = {'work': initial, 'saved': False}
    monkeypatch.setattr(ws.Workspace, 'url', property(lambda self:value['workspace_url']))
    def save(self):
        state['work'] = deepcopy(self)
        state['saved'] = True
    monkeypatch.setattr(ws.Workspace, 'save', save)
    def read_back(cls, url):
        work = deepcopy(state['work'])
        if state['saved']:
            for section in work.sections:
                for panel in section.panels:
                    if isinstance(panel, wr.LinePlot):
                        panel.line_colors = {key: {'color':layout._color(color), 'transparentColor':'#FFFFFF19'}
                                             for key,color in panel.line_colors.items()}
        return work
    monkeypatch.setattr(ws.Workspace, 'from_url', classmethod(read_back))
    result = layout.install_catalog(value, tmp_path/'layout.json')
    assert result['api_verified'] and result['panel_count'] == layout.PANEL_COUNT
    # Exercise conversion to the service model, including native ID filters.
    model = state['work']._to_model()
    wire = model.model_dump_json()
    assert 'solid' in wire and layout.series.X_AXIS in wire
    layout._verify_view_spec(model.model_dump(by_alias=True)['spec'], value)


def remote_spec(value):
    return dict(sections=[dict(name=layout.PREFIX+'Results', panels=[
        dict(viewType='Run History Line Plot', config=dict(xAxis=layout.series.X_AXIS,
            metrics=[metric], aggregate=False,
            overrideMarks={r['run_id']+':'+metric:'solid' for r in value['curves']},
            overrideSeriesTitles={r['run_id']+':'+metric:r['name'] for r in value['curves']},
            overrideColors={r['run_id']+':'+metric:dict(color=r['color'],transparentColor='#FFFFFF19')
                            for r in value['curves']})) for _,metric in layout.METRICS])],
        progress=['progress/checkpoints_complete','progress/failed_cells'])


def receipt_for(value, path):
    urls = layout.validate_catalog(value)
    saved = dict(api_verified=True, catalog_sha256=layout.series._digest(value), curves=value['curves'],
        panel_count=layout.PANEL_COUNT, url=urls[0], views=[dict(url=url,
            view_id=layout._workspace_id(url,value['entity'],value['project']),
            panel_count=layout.PANEL_COUNT, api_verified=True) for url in urls])
    layout.series._atomic_json(path,saved)
    return saved


def api_for(spec, legacy=False):
    calls=[]
    response={'project':{'allViews':{'edges':[{'node':{'id':'node','spec':spec}}]}}}
    def execute(query, **kwargs):
        calls.append((query,kwargs))
        return response
    api=Object(client=Object(execute=execute)) if legacy else Object(_service_api=Object(execute_graphql=execute))
    return Object(Api=lambda **kwargs:api),calls,response


@pytest.mark.parametrize('legacy', [False,True])
def test_cpu_verifier_supports_both_api_generations_without_workspace_sdk(tmp_path,monkeypatch,legacy):
    value=catalog();value['other_workspace_urls']=['https://wandb.ai/entity/project?nw=second']
    path=tmp_path/'receipt.json';saved=receipt_for(value,path)
    api,calls,_=api_for(layout.json.dumps(remote_spec(value)),legacy)
    def imports(name):
        assert name in ('wandb_gql','wandb.vendor.gql')
        if name=='wandb_gql':raise ImportError('old runtime')
        return Object(gql=lambda query:query)
    monkeypatch.setattr(layout.importlib,'import_module',imports)
    assert layout.verify_installed_catalog(value,path,api)==saved
    assert len(calls)==2
    var_key='variable_values' if legacy else 'variables'
    assert {call[1][var_key]['viewName'] for call in calls}=={'nw-3w61ds5ze92-v','nw-second-v'}


@pytest.mark.parametrize('missing', ['run','color','layout','metric','solid','axis'])
def test_cpu_verifier_rejects_missing_live_curve_contract(tmp_path,missing):
    value=catalog();path=tmp_path/'receipt.json';receipt_for(value,path)
    spec=remote_spec(value)
    if missing=='run':spec=layout.json.loads(layout.json.dumps(spec).replace('run0','wrongrun'))
    elif missing=='color':spec=layout.json.loads(layout.json.dumps(spec).replace('#6B7280','#FFFFFF'))
    elif missing=='layout':spec['sections'][0]['name']='Other layout'
    elif missing=='metric':spec['sections'][0]['panels'].pop()
    elif missing=='solid':spec['sections'][0]['panels'][0]['config']['overrideMarks']['run0:eval/return_mean']='points'
    elif missing=='axis':spec['sections'][0]['panels'][0]['config']['xAxis']='Step'
    api,_,_=api_for(spec)
    with pytest.raises(layout.LayoutError):layout.verify_installed_catalog(value,path,api)


@pytest.mark.parametrize('field,value', [('api_verified',False),('catalog_sha256','bad'),('curves',[]),('views',[])])
def test_cpu_verifier_rejects_unverified_or_wrong_receipt_before_api(tmp_path,field,value):
    data=catalog();path=tmp_path/'receipt.json';saved=receipt_for(data,path)
    saved[field]=value;layout.series._atomic_json(path,saved)
    api=Object(Api=lambda **kwargs:pytest.fail('Invalid receipt must fail before API access'))
    with pytest.raises(layout.LayoutError):layout.verify_installed_catalog(data,path,api)


def test_cpu_verifier_rejects_deleted_remote_view(tmp_path):
    value=catalog();path=tmp_path/'receipt.json';receipt_for(value,path)
    api,_,response=api_for(remote_spec(value));response['project']['allViews']['edges']=[]
    with pytest.raises(layout.LayoutError,match='missing remotely'):
        layout.verify_installed_catalog(value,path,api)
