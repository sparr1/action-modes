"""Owned W&B view for 800k Bernoulli critic transfer over inner rounds J."""
from copy import deepcopy
import json
from pathlib import Path
import re

from utils.wandb_results_layout import (
    DEFAULT_VIEW_NAME, ResultsLayoutError, _bank, _execute, _hash, _panel,
    _selected, _spec, _views, _write,
)

VERSION = 'transfer-j10-v1'
SUMMARY_KEY = 'transfer_j10/summary'
PAIRS_KEY = 'transfer_j10/pairs'
EPISODES_KEY = 'transfer_j10/episodes'
PROGRESS_KEY = 'transfer_j10/progress'
DIAGNOSTICS_KEY = 'transfer_j10/diagnostics'
METHODS = ('Bernoulli 50% critic', 'Frozen prior')
COLORS = ('#e69f00', '#333333')
_QUERY = 'query TransferJ10Chart($id:ID!){customChart(id:$id){id name type spec}}'
_CREATE_CHART = '''mutation CreateTransferJ10Chart($entity:String!,$name:String!,
 $displayName:String!,$type:String!,$access:String!,$spec:JSONString!){
 createCustomChart(input:{entity:$entity,name:$name,displayName:$displayName,
 type:$type,access:$access,spec:$spec}){chart{id name type spec}}}'''
_CREATE_VIEW = '''mutation CreateTransferJ10View($entityName:String,$projectName:String,
 $type:String,$name:String,$displayName:String,$spec:String){
 upsertView(input:{entityName:$entityName,projectName:$projectName,type:$type,
 name:$name,displayName:$displayName,spec:$spec,createdUsing:WANDB_SDK}){
 view{id name} inserted}}'''


class J10LayoutConflict(ResultsLayoutError):
    """A user-edited owned layout must not be overwritten."""


def chart_definition():
    colors = {'field': '${field:method}', 'type': 'nominal', 'title': 'Controller',
              'scale': {'domain': list(METHODS), 'range': list(COLORS)},
              'legend': {'orient': 'bottom', 'columns': 2, 'labelLimit': 250}}
    x = {'field': '${field:J}', 'type': 'quantitative',
         'title': 'Inner SAC rounds J (C16/A4/N128/B256)',
         'scale': {'domain': [3.5, 10.5], 'zero': False}, 'axis': {'values': [4, 6, 8, 10]}}
    y = {'field': '${field:mean}', 'type': 'quantitative', 'title': '${string:yname}', 'scale': {'zero': False}}
    return {
        '$schema': 'https://vega.github.io/schema/vega-lite/v5.json',
        'data': {'name': 'wandb'}, 'width': 'container', 'height': 300,
        'title': {'text': '${string:title}', 'subtitle': '${string:subtitle}'},
        'transform': [
            {'filter': "datum['${field:state}'] === 'complete' && datum['${field:metric}'] === '${string:metric}' && datum['${field:baseline}'] === '${string:baseline}'"},
            {'filter': "isNumber(datum['${field:mean}']) && isFinite(datum['${field:mean}'])"},
        ],
        'encoding': {'color': colors,
            'tooltip': [{'field': '${field:' + k + '}', 'type': 'nominal' if k in ('label', 'method', 'metric', 'baseline') else 'quantitative'}
                        for k in ('label', 'method', 'metric', 'baseline', 'J', 'n', 'mean', 'lower', 'upper', 'sd')]},
        'layer': [
            {'transform': [{'filter': "isNumber(datum['${field:J}'])"}],
             'mark': {'type': 'line', 'point': {'filled': True, 'size': 85}},
             'encoding': {'x': x, 'y': y}},
            {'transform': [{'filter': "isNumber(datum['${field:J}'])"}],
             'mark': {'type': 'rule', 'strokeWidth': 2}, 'encoding': {'x': x,
                 'y': {'field': '${field:lower}', 'type': 'quantitative', 'title': '${string:yname}', 'scale': {'zero': False}},
                 'y2': {'field': '${field:upper}'}}},
            {'transform': [{'filter': "!isNumber(datum['${field:J}'])"}],
             'mark': {'type': 'rule', 'strokeDash': [5, 4]}, 'encoding': {'y': y}},
        ],
        'config': {'view': {'stroke': None}, 'axis': {'gridColor': '#e8edf2'}},
    }


def chart_id(entity):
    return entity + '/transfer_j10_' + _hash(chart_definition())[:16]


def ensure_chart(api, entity):
    ident, expected = chart_id(entity), chart_definition()
    chart = _execute(api, _QUERY, {'id': ident}).get('customChart')
    error = None
    if chart is None:
        try:
            _execute(api, _CREATE_CHART, dict(entity=entity, name=ident.split('/', 1)[1],
                displayName='800k Bernoulli critic transfer over J', type='vega2', access='PRIVATE',
                spec=json.dumps(expected, separators=(',', ':'))))
        except Exception as exc:
            error = exc
        chart = _execute(api, _QUERY, {'id': ident}).get('customChart')
    if not chart:
        raise ResultsLayoutError('Transfer-J10 chart is not visible; retry its content-addressed identity.') from error
    if chart.get('type') != 'vega2' or _spec(chart) != expected:
        raise J10LayoutConflict('Transfer-J10 chart could not be verified; no existing chart overwritten.') from error
    return ident


def runset(publication_id):
    return dict(id='rs//Subsection 1', name='800k Bernoulli critic transfer over J', enabled=True,
        runFeed=dict(version=2, columnVisible={}, columnPinned={}, columnWidths={}, columnOrder=[], pageSize=50, onlyShowSelected=False),
        search={'query': ''}, searchHistory=[], grouping=[],
        filters={'filterFormat': 'filterV2', 'filters': [
            {'key': {'section': 'config', 'name': 'transfer_j10_publication'},
             'op': '=', 'value': publication_id, 'disabled': False}]},
        sort={'keys': []}, selections={'root': 1, 'bounds': [], 'tree': []}, expandedRowAddresses=[])


def sections(entity):
    intro = ('### 800k: Bernoulli 50% critic transfer over J\n\n'
        '**H1, C16/A4/N128/B256, compiled L40S execution and persistent episode RNG stay fixed.** '
        'J4/J6/J8 and the frozen prior are immutable published references; J10 is the sole new cell. '
        'Each complete point uses the same five environment seeds 101–105, controller seed 55, and 500 decisions. '
        'Error bars show sample standard deviation across episodes, not confidence intervals. '
        'J10 remains explicitly pending until all five episodes pass receipt and trace validation. '
        'Controller time includes initialization and transfer, excludes diagnostics; steady time excludes decisions 0–9. '
        'Historical prior timing is unavailable. Paired gains use identical environment seeds. '
        'Diagnostics are sampled model-based measurements, not simulator ground truth. Raw traces remain on Oscar.')
    groups = [('progress', '800k transfer over J | progress', [
        _panel('intro', 'Markdown Panel', {'value': intro}, width=24, height=6),
        _panel('progress', 'Media Browser', {'chartTitle': 'J10: five-seed progress and validation', 'mediaKeys': [PROGRESS_KEY]}, width=24, height=7)], 1)]
    plots = []
    specs = (
        ('return', SUMMARY_KEY, 'episode_return', 'none', 'Full-episode return versus J', 'Undiscounted reward'),
        ('prior', PAIRS_KEY, 'episode_return', 'prior', 'Paired improvement over frozen prior', 'Return difference'),
        ('j8', PAIRS_KEY, 'episode_return', 'J8', 'Paired change relative to J8', 'Return difference'),
        ('variance', SUMMARY_KEY, 'episode_return_sd', 'none', 'Episode-return variability', 'Sample standard deviation'),
        ('time', SUMMARY_KEY, 'controller_time_s', 'none', 'Mean controller time including initialization', 'Seconds / decision'),
        ('steady-time', SUMMARY_KEY, 'steady_controller_time_s', 'none', 'Steady controller time: decisions 10–499', 'Seconds / decision'),
        ('q-error', SUMMARY_KEY, 'final_critic_rmse', 'none', 'Sampled final critic RMSE', 'Model-target RMSE'),
        ('regret', SUMMARY_KEY, 'final_critic_selection_regret', 'none', 'Sampled final critic selection regret', 'Model-return difference'),
    )
    for identifier, table, metric, baseline, title, yname in specs:
        plots.append(_panel(identifier, 'Vega2', {
            'transform': {'name': 'tableWithLeafColNames'},
            'userQuery': {'queryFields': [{'name': 'runSets', 'args': [{'name': 'runSets', 'value': '${runSets}'}, {'name': 'limit', 'value': 500.}],
                'fields': [{'name': 'summaryTable', 'args': [{'name': 'tableKey', 'value': table}], 'fields': []}, {'name': 'id', 'value': []}, {'name': 'name', 'value': []}]}]},
            'panelDefId': chart_id(entity),
            'fieldSettings': {k: k for k in ('label', 'method', 'J', 'metric', 'baseline', 'state', 'n', 'mean', 'lower', 'upper', 'sd')},
            'stringSettings': dict(title=title, subtitle='Five paired seeds; bars = episode sample SD', metric=metric, baseline=baseline, yname=yname),
        }, width=12, height=9))
    groups.append(('curves', '800k transfer over J | return, variability and time', plots, 2))
    groups.append(('data', '800k transfer over J | compact verified data', [
        _panel(name, 'Media Browser', {'chartTitle': title, 'mediaKeys': [key]}, width=24, height=8)
        for name, title, key in (
            ('summary', 'Complete five-seed summaries', SUMMARY_KEY),
            ('episodes', 'Per-seed returns and timing', EPISODES_KEY),
            ('pairs', 'Paired gains, sample SD and seed win rates', PAIRS_KEY),
            ('diagnostics', 'Sampled diagnostic summaries by seed', DIAGNOSTICS_KEY))], 1))
    result = []
    for suffix, title, panels, columns in groups:
        for i, panel in enumerate(panels):
            panel['__id__'] = f'ambi-{VERSION}-{suffix}-{i}'
        result.append(dict(__id__=f'ambi-{VERSION}-{suffix}', name=title, isOpen=True, type='flow',
            flowConfig=dict(snapToColumns=True, columnsPerPage=columns, rowsPerPage=3,
                            gutterWidth=16, boxWidth=560, boxHeight=430),
            sorted=0, pinned=True, isPanelsAuto=False, panels=panels))
    return result


def saved_spec(template, entity, publication_id):
    result = deepcopy(template)
    result['section'].update(runSets=[runset(publication_id)], openRunSet=0,
                             workspaceSettings={'shouldAutoGeneratePanels': False})
    bank = _bank(result)
    bank['sections'], bank['panelPlacementOverrides'] = sections(entity), {}
    return result


def installed(spec, entity, publication_id):
    wanted, actual = sections(entity), deepcopy(_bank(spec)['sections'])
    if len(actual) != len(wanted) or spec['section'].get('runSets') != [runset(publication_id)]:
        return False
    for section, target in zip(actual, wanted):
        section.setdefault('type', target['type'])
        for k, v in target['flowConfig'].items():
            section.setdefault('flowConfig', {}).setdefault(k, v)
        if len(section.get('panels', [])) != len(target['panels']):
            return False
        for panel, expected in zip(section['panels'], target['panels']):
            panel.setdefault('layout', expected['layout'])
    return actual == wanted and not _bank(spec).get('panelPlacementOverrides')


def ensure_saved_view(api, *, entity, project, publication_id, receipt_dir):
    if not re.fullmatch(r'[A-Za-z0-9]+', publication_id):
        raise ValueError('Publication ID must be alphanumeric.')
    root, name = Path(receipt_dir), 'nw-transferj10' + publication_id + '-v'
    receipt = dict(layout_version=VERSION, view_name=name, browser_verified=False,
                   url=f'https://wandb.ai/{entity}/{project}?nw={name[3:-2]}')
    try:
        receipt['chart_id'] = ensure_chart(api, entity)
        views = _views(api, entity, project)
        matches = [v for v in views if v['name'] == name]
        if matches:
            if len(matches) != 1 or not installed(_spec(matches[0]), entity, publication_id):
                raise J10LayoutConflict('Existing transfer-J10 view differs; preserving user edits.')
            receipt.update(status='verified', changed=False, view_id=matches[0]['id'])
            _write(root / 'results-layout-receipt.json', receipt)
            return receipt
        proposed = saved_spec(_spec(_selected(views, DEFAULT_VIEW_NAME)), entity, publication_id)
        _write(root / 'views-before.json', views); _write(root / 'view-proposed.json', proposed)
        if _views(api, entity, project) != views:
            raise ResultsLayoutError('Views changed during preparation; retry from fresh state.')
        error = None
        try:
            _execute(api, _CREATE_VIEW, dict(entityName=entity, projectName=project, type='project-view',
                name=name, displayName='800k Bernoulli critic transfer over J', spec=json.dumps(proposed, separators=(',', ':'))))
        except Exception as exc:
            error = type(exc).__name__
        after = _views(api, entity, project)
        _write(root / 'views-after.json', after)
        matches = [v for v in after if v['name'] == name and v['type'] == 'project-view']
        remaining = {v['id']: v for v in after}
        if len(matches) != 1 or _spec(matches[0]) != proposed or any(remaining.get(v['id']) != v for v in views):
            raise ResultsLayoutError('Transfer-J10 view readback or existing-view preservation failed.')
        receipt.update(status='verified', changed=True, view_id=matches[0]['id'],
                       preserved_views=len(views), uncertain_response_reconciled=error is not None)
        _write(root / 'results-layout-receipt.json', receipt)
        return receipt
    except Exception as exc:
        _write(root / 'results-layout-receipt.json', dict(receipt, status='failed', error=f'{type(exc).__name__}: {exc}'))
        raise
