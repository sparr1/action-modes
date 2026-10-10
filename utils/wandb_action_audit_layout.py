"""A separate, content-addressed W&B view for the 800k matched-action audit."""
from copy import deepcopy
import json
from pathlib import Path
import re

from utils.wandb_results_layout import (
    DEFAULT_VIEW_NAME, ResultsLayoutError, _bank, _execute, _hash, _panel,
    _selected, _spec, _views, _write,
)

VERSION = 'action-audit-v1'
TABLE_KEY = 'action_audit/measurements'
PROGRESS_KEY = 'action_audit/progress'
CRITICS_KEY = 'action_audit/critics'
CHECKS_KEY = 'action_audit/checks'
CANDIDATES = ('prior_mean', 'sac_j4_mean', 'sac_j6_mean', 'sac_j8_mean',
              'mppi_h1_mean', 'mppi_h3_mean', 'broad_best_h1', 'broad_best_h3',
              'replay_best_h1', 'replay_best_h3')
COLORS = ('#111111', '#0072b2', '#56b4e9', '#332288', '#e69f00', '#d55e00',
          '#009e73', '#117733', '#cc79a7', '#882255')
_QUERY = 'query ActionAuditChart($id:ID!){customChart(id:$id){id name type spec}}'
_CREATE_CHART = '''mutation CreateActionAuditChart($entity:String!,$name:String!,
 $displayName:String!,$type:String!,$access:String!,$spec:JSONString!){
 createCustomChart(input:{entity:$entity,name:$name,displayName:$displayName,
 type:$type,access:$access,spec:$spec}){chart{id name type spec}}}'''
_CREATE_VIEW = '''mutation CreateActionAuditView($entityName:String,$projectName:String,
 $type:String,$name:String,$displayName:String,$spec:String){
 upsertView(input:{entityName:$entityName,projectName:$projectName,type:$type,
 name:$name,displayName:$displayName,spec:$spec,createdUsing:WANDB_SDK}){
 view{id name} inserted}}'''


class AuditLayoutConflict(ResultsLayoutError):
    """Existing content differs from the exact owned layout; never overwrite it."""


def chart_definition():
    return {
        '$schema': 'https://vega.github.io/schema/vega-lite/v5.json',
        'data': {'name': 'wandb'}, 'width': 'container', 'height': 310,
        'title': {'text': '${string:title}', 'subtitle': '${string:subtitle}'},
        'transform': [
            {'filter': "datum['${field:state}'] === 'complete' && datum['${field:horizon}'] === ${string:horizon}"},
            {'filter': "isNumber(datum['${field:x}']) && isFinite(datum['${field:x}']) && isNumber(datum['${field:y}']) && isFinite(datum['${field:y}'])"},
        ],
        'mark': {'type': 'point', 'filled': True, 'size': 65, 'opacity': .85},
        'encoding': {
            'x': {'field': '${field:x}', 'type': 'quantitative', 'title': '${string:xname}', 'scale': {'zero': False}},
            'y': {'field': '${field:y}', 'type': 'quantitative', 'title': '${string:yname}', 'scale': {'zero': False}},
            'color': {'field': '${field:candidate}', 'type': 'nominal', 'title': 'Candidate',
                      'scale': {'domain': list(CANDIDATES), 'range': list(COLORS)},
                      'legend': {'orient': 'bottom', 'columns': 2, 'labelLimit': 240, 'symbolOpacity': 1}},
            'shape': {'field': '${field:history}', 'type': 'nominal', 'title': 'Root history'},
            'opacity': {'condition': {'param': 'selected_candidates', 'value': .85}, 'value': .12},
            'tooltip': [{'field': '${field:' + k + '}', 'type': 'nominal' if k in ('candidate', 'history') else 'quantitative'}
                        for k in ('candidate', 'history', 'seed', 'decision', 'horizon', 'x', 'y')],
        },
        'params': [{'name': 'selected_candidates', 'select': {'type': 'point', 'fields': ['${field:candidate}']}, 'bind': 'legend'}],
        'config': {'view': {'stroke': None}, 'axis': {'gridColor': '#e8edf2'}},
    }


def chart_id(entity):
    return entity + '/action_audit_' + _hash(chart_definition())[:16]


def ensure_chart(api, entity):
    ident, expected = chart_id(entity), chart_definition()
    chart = _execute(api, _QUERY, {'id': ident}).get('customChart')
    error = None
    if chart is None:
        try:
            _execute(api, _CREATE_CHART, dict(entity=entity, name=ident.split('/', 1)[1],
                displayName='800k matched-action audit', type='vega2', access='PRIVATE',
                spec=json.dumps(expected, separators=(',', ':'))))
        except Exception as exc:
            error = exc
        chart = _execute(api, _QUERY, {'id': ident}).get('customChart')
    actual = _spec(chart) if chart else None
    if not chart:
        raise ResultsLayoutError('New action-audit chart is not visible yet; retry its content-addressed identity.') from error
    if chart.get('type') != 'vega2' or actual != expected:
        raise AuditLayoutConflict('Action-audit chart could not be verified; no existing chart overwritten.') from error
    return ident


def runset(publication_id):
    return dict(id='rs//Subsection 1', name='800k action audit', enabled=True,
        runFeed=dict(version=2, columnVisible={}, columnPinned={}, columnWidths={}, columnOrder=[], pageSize=50, onlyShowSelected=False),
        search={'query': ''}, searchHistory=[], grouping=[],
        filters={'filterFormat': 'filterV2', 'filters': [
            {'key': {'section': 'config', 'name': 'action_audit_publication'},
             'op': '=', 'value': publication_id, 'disabled': False}]},
        sort={'keys': []}, selections={'root': 1, 'bounds': [], 'tree': []}, expandedRowAddresses=[])


def sections(entity):
    intro = ('### 800k matched-action audit\n\n'
        '**Progress and integrity checks are explicit; pending or failed measurements never become zeros.** '
        'Each point is one saved root, candidate and audit horizon. Shapes distinguish the controller that visited the root; colors identify the action candidate. '
        'Model/real comparisons use fresh noise independent of selection, shared within each model/real pair. A separate held-out 32-draw model bank is retained in the measurements table. '
        'Real-prefix value uses simulator rewards followed by the frozen terminal estimate. The real-tail comparison uses a finite 500-step continuation, not an infinite-horizon ground truth. '
        'Gains subtract the prior action at the same root and horizon. Root histories differ; this is not a full-episode policy comparison. '
        'Integrity pass/fail refers to execution checks, not whether a candidate improves reward. '
        'Critic bank metrics describe this constructed candidate mixture, including replay actions; they are not population estimates over all states or actions. '
        'Raw banks, simulator branches and detailed records stay in the campaign directory on Oscar; W&B receives compact summaries only.')
    groups = [('progress', 'Action audit | progress and checks', [
        _panel('intro', 'Markdown Panel', {'value': intro}, width=24, height=6),
        _panel('progress', 'Media Browser', {'chartTitle': 'Every requested root: pending, complete or failed', 'mediaKeys': [PROGRESS_KEY]}, width=24),
        _panel('checks', 'Media Browser', {'chartTitle': 'Execution integrity checks: pass, fail or pending', 'mediaKeys': [CHECKS_KEY]}, width=24)], 1)]
    plots = []
    specs = (
        ('prefix', 'Paired model gain versus real-prefix value gain', 'model_gain', 'prefix_gain', 'Paired model gain', 'Real-prefix value gain'),
        ('tail', 'Paired model gain versus finite real-tail gain', 'model_gain', 'tail_gain', 'Paired model gain', 'Finite real-tail gain'),
        ('bias', 'Model-prefix bias versus terminal bias', 'model_prefix_bias', 'terminal_bias', 'Model-prefix bias', 'Terminal bias'),
    )
    for h in (1, 3):
        for key, title, x, y, xtitle, ytitle in specs:
            plots.append(_panel(f'{key}-h{h}', 'Vega2', {
                'transform': {'name': 'tableWithLeafColNames'},
                'userQuery': {'queryFields': [{'name': 'runSets', 'args': [{'name': 'runSets', 'value': '${runSets}'}, {'name': 'limit', 'value': 500.}],
                    'fields': [{'name': 'summaryTable', 'args': [{'name': 'tableKey', 'value': TABLE_KEY}], 'fields': []}, {'name': 'id', 'value': []}, {'name': 'name', 'value': []}]}]},
                'panelDefId': chart_id(entity),
                'fieldSettings': dict(x=x, y=y, candidate='candidate', history='history', seed='seed', decision='decision', horizon='horizon', state='state'),
                'stringSettings': dict(title=f'H{h}: {title}', subtitle='Independent of selection; model and real share continuation noise', horizon=str(h), xname=xtitle, yname=ytitle),
            }, width=12, height=9))
    groups.append(('scores', 'Action audit | model and real simulator comparisons', plots, 2))
    groups.append(('values', 'Action audit | exact compact measurements', [
        _panel('measurements', 'Media Browser', {'chartTitle': 'Paired candidate estimates', 'mediaKeys': [TABLE_KEY]}, width=24, height=9),
        _panel('critics', 'Media Browser', {'chartTitle': 'Critic calibration and ranking: full selection bank versus held-out selected actions', 'mediaKeys': [CRITICS_KEY]}, width=24, height=9)], 1))
    result = []
    for suffix, title, panels, columns in groups:
        for i, p in enumerate(panels):
            p['__id__'] = f'ambi-{VERSION}-{suffix}-{i}'
        result.append(dict(__id__=f'ambi-{VERSION}-{suffix}', name=title, isOpen=True, type='flow',
            flowConfig=dict(snapToColumns=True, columnsPerPage=columns, rowsPerPage=3 if suffix != 'values' else 2,
                            gutterWidth=16, boxWidth=560, boxHeight=430),
            sorted=0, pinned=True, isPanelsAuto=False, panels=panels))
    return result


def saved_spec(template, entity, publication_id):
    result = deepcopy(template)
    result['section'].update(runSets=[runset(publication_id)], openRunSet=0,
                             workspaceSettings={'shouldAutoGeneratePanels': False})
    bank = _bank(result)
    bank['sections'] = sections(entity)
    bank['panelPlacementOverrides'] = {}
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
    root = Path(receipt_dir)
    name = 'nw-actionaudit' + publication_id + '-v'
    receipt = dict(layout_version=VERSION, view_name=name, browser_verified=False,
                   url=f'https://wandb.ai/{entity}/{project}?nw={name[3:-2]}')
    try:
        receipt['chart_id'] = ensure_chart(api, entity)
        views = _views(api, entity, project)
        matches = [v for v in views if v['name'] == name]
        if matches:
            if len(matches) != 1 or not installed(_spec(matches[0]), entity, publication_id):
                raise AuditLayoutConflict('Existing audit view differs; preserving user edits.')
            receipt.update(status='verified', changed=False, view_id=matches[0]['id'])
            _write(root / 'results-layout-receipt.json', receipt)
            return receipt
        proposed = saved_spec(_spec(_selected(views, DEFAULT_VIEW_NAME)), entity, publication_id)
        _write(root / 'views-before.json', views)
        _write(root / 'view-proposed.json', proposed)
        if _views(api, entity, project) != views:
            raise ResultsLayoutError('Views changed during preparation; retry from fresh state.')
        error = None
        try:
            _execute(api, _CREATE_VIEW, dict(entityName=entity, projectName=project, type='project-view',
                name=name, displayName='800k matched-action audit', spec=json.dumps(proposed, separators=(',', ':'))))
        except Exception as exc:
            error = type(exc).__name__
        after = _views(api, entity, project)
        _write(root / 'views-after.json', after)
        matches = [v for v in after if v['name'] == name and v['type'] == 'project-view']
        remaining = {v['id']: v for v in after}
        if len(matches) != 1 or _spec(matches[0]) != proposed or any(remaining.get(v['id']) != v for v in views):
            raise ResultsLayoutError('Audit view readback or existing-view preservation failed.')
        receipt.update(status='verified', changed=True, view_id=matches[0]['id'],
                       preserved_views=len(views), uncertain_response_reconciled=error is not None)
        _write(root / 'results-layout-receipt.json', receipt)
        return receipt
    except Exception as exc:
        _write(root / 'results-layout-receipt.json', dict(receipt, status='failed', error=f'{type(exc).__name__}: {exc}'))
        raise
