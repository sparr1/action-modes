"""Owned W&B view for full-episode critic-bypass outcomes, timing and progress."""
from copy import deepcopy
import json
from pathlib import Path
import re

from utils.wandb_results_layout import (
    DEFAULT_VIEW_NAME, ResultsLayoutError, _bank, _execute, _hash, _panel,
    _selected, _spec, _views, _write,
)

VERSION = 'critic-bypass-v1'
SUMMARY_KEY = 'critic_bypass/summary'
PAIRS_KEY = 'critic_bypass/pairs'
EPISODES_KEY = 'critic_bypass/episodes'
PROGRESS_KEY = 'critic_bypass/progress'
DIAGNOSTICS_KEY = 'critic_bypass/diagnostics'
ARMS = ('actor_mean', 'learned_q', 'model_score', 'prior', 'mppi_h3')
LABELS = ('SAC actor mean', 'SAC learned-Q selection', 'SAC direct-model selection',
          'Frozen prior', 'MPPI H3')
COLORS = ('#0072b2', '#cc79a7', '#009e73', '#333333', '#d55e00')
_QUERY = 'query CriticBypassChart($id:ID!){customChart(id:$id){id name type spec}}'
_CREATE_CHART = '''mutation CreateCriticBypassChart($entity:String!,$name:String!,
 $displayName:String!,$type:String!,$access:String!,$spec:JSONString!){
 createCustomChart(input:{entity:$entity,name:$name,displayName:$displayName,
 type:$type,access:$access,spec:$spec}){chart{id name type spec}}}'''
_CREATE_VIEW = '''mutation CreateCriticBypassView($entityName:String,$projectName:String,
 $type:String,$name:String,$displayName:String,$spec:String){
 upsertView(input:{entityName:$entityName,projectName:$projectName,type:$type,
 name:$name,displayName:$displayName,spec:$spec,createdUsing:WANDB_SDK}){
 view{id name} inserted}}'''


class BypassLayoutConflict(ResultsLayoutError):
    """A user-edited owned layout must not be overwritten."""


def chart_definition():
    return {
        '$schema': 'https://vega.github.io/schema/vega-lite/v5.json',
        'data': {'name': 'wandb'}, 'width': 'container', 'height': 300,
        'title': {'text': '${string:title}', 'subtitle': '${string:subtitle}'},
        'transform': [
            {'filter': "datum['${field:state}'] === 'complete' && datum['${field:metric}'] === '${string:metric}' && datum['${field:baseline}'] === '${string:baseline}'"},
            {'filter': "isNumber(datum['${field:mean}']) && isFinite(datum['${field:mean}'])"},
        ],
        'encoding': {
            'x': {'field': '${field:label}', 'type': 'nominal', 'title': None,
                  'sort': list(LABELS), 'axis': {'labelAngle': -20, 'labelLimit': 200}},
            'color': {'field': '${field:label}', 'type': 'nominal', 'title': 'Controller',
                      'scale': {'domain': list(LABELS), 'range': list(COLORS)},
                      'legend': {'orient': 'bottom', 'columns': 2, 'labelLimit': 250}},
            'tooltip': [{'field': '${field:' + k + '}', 'type': 'nominal' if k in ('label', 'metric', 'baseline') else 'quantitative'}
                        for k in ('label', 'metric', 'baseline', 'n', 'mean', 'ci_low', 'ci_high', 'sd')],
        },
        'layer': [
            {'mark': {'type': 'rule', 'strokeWidth': 2}, 'encoding': {
                'y': {'field': '${field:ci_low}', 'type': 'quantitative', 'title': '${string:yname}', 'scale': {'zero': False}},
                'y2': {'field': '${field:ci_high}'}}},
            {'mark': {'type': 'point', 'filled': True, 'size': 95}, 'encoding': {
                'y': {'field': '${field:mean}', 'type': 'quantitative', 'title': '${string:yname}', 'scale': {'zero': False}}}},
        ],
        'config': {'view': {'stroke': None}, 'axis': {'gridColor': '#e8edf2'}},
    }


def chart_id(entity):
    return entity + '/critic_bypass_' + _hash(chart_definition())[:16]


def ensure_chart(api, entity):
    ident, expected = chart_id(entity), chart_definition()
    chart = _execute(api, _QUERY, {'id': ident}).get('customChart')
    error = None
    if chart is None:
        try:
            _execute(api, _CREATE_CHART, dict(entity=entity, name=ident.split('/', 1)[1],
                displayName='800k critic bypass', type='vega2', access='PRIVATE',
                spec=json.dumps(expected, separators=(',', ':'))))
        except Exception as exc:
            error = exc
        chart = _execute(api, _QUERY, {'id': ident}).get('customChart')
    if not chart:
        raise ResultsLayoutError('Critic-bypass chart is not visible; retry its content-addressed identity.') from error
    if chart.get('type') != 'vega2' or _spec(chart) != expected:
        raise BypassLayoutConflict('Critic-bypass chart could not be verified; no existing chart overwritten.') from error
    return ident


def runset(publication_id):
    return dict(id='rs//Subsection 1', name='800k critic bypass', enabled=True,
        runFeed=dict(version=2, columnVisible={}, columnPinned={}, columnWidths={}, columnOrder=[], pageSize=50, onlyShowSelected=False),
        search={'query': ''}, searchHistory=[], grouping=[],
        filters={'filterFormat': 'filterV2', 'filters': [
            {'key': {'section': 'config', 'name': 'critic_bypass_publication'},
             'op': '=', 'value': publication_id, 'disabled': False}]},
        sort={'keys': []}, selections={'root': 1, 'bounds': [], 'tree': []}, expandedRowAddresses=[])


def sections(entity):
    intro = ('### 800k full-episode critic bypass\n\n'
        '**Progress is incremental. Outcome summaries appear only when every requested seed for an arm is complete and verified.** '
        'SAC H1/J6 starts fresh at every decision. The three SAC arms differ only in the executed action: actor mean, learned-Q replay selection, or direct-model replay selection. '
        'The common bank contains all 768 imagined replay actions plus the prior and adapted means. '
        'Direct-model scores use reward plus discounted frozen terminal Q; they are not simulator ground truth. '
        'Primary return is undiscounted full-episode environment reward over up to 500 decisions. '
        'Intervals use 2,000 paired environment-seed bootstrap resamples; 20 seeds with controller seed 55 are exploratory, not independent training runs. '
        'Timing includes action selection and excludes sampled diagnostic work; p95 is pooled across all timed decisions, without an interval. '
        'Verified historical prior episodes contribute return only: timing excludes these references, and each timing row reports its own seed count n. '
        'Counterfactual held-out diagnostics compare the three choices at the same visited state; independently visited trajectories must not be treated as matched states. '
        'Raw per-decision records remain on Oscar; only compact tables are uploaded.')
    groups = [('progress', 'Critic bypass | progress and integrity', [
        _panel('intro', 'Markdown Panel', {'value': intro}, width=24, height=6),
        _panel('progress', 'Media Browser', {'chartTitle': 'Every arm and seed: pending, complete or failed', 'mediaKeys': [PROGRESS_KEY]}, width=24, height=8)], 1)]
    plots = []
    for identifier, table, metric, baseline, title, yname, subtitle in (
        ('return', SUMMARY_KEY, 'episode_return', 'none', 'Full-episode return', 'Undiscounted reward', 'Complete arms only; mean and 95% seed bootstrap interval'),
        ('gain-prior', PAIRS_KEY, 'episode_return', 'prior', 'Paired improvement over frozen prior', 'Return difference', 'Same environment/controller seeds; 95% paired bootstrap interval'),
        ('gain-actor', PAIRS_KEY, 'episode_return', 'actor_mean', 'Paired improvement over actor mean', 'Return difference', 'Same environment/controller seeds; 95% paired bootstrap interval'),
        ('variance', SUMMARY_KEY, 'episode_return_sd', 'none', 'Episode-return variability', 'Sample standard deviation', 'Across environment seeds; no interval'),
        ('control-mean', SUMMARY_KEY, 'controller_time_mean_s', 'none', 'Mean controller time including selection', 'Seconds / decision', 'Equal seed weighting; 95% seed bootstrap interval'),
        ('control-p95', SUMMARY_KEY, 'controller_time_p95_s', 'none', 'p95 controller time including selection', 'Seconds / decision', 'Pooled executed decisions; no interval'),
        ('selection-mean', SUMMARY_KEY, 'selection_time_mean_s', 'none', 'Mean selection overhead', 'Seconds / decision', 'Equal seed weighting; diagnostic work excluded'),
        ('selection-p95', SUMMARY_KEY, 'selection_time_p95_s', 'none', 'p95 selection overhead', 'Seconds / decision', 'Pooled executed decisions; no interval'),
    ):
        plots.append(_panel(identifier, 'Vega2', {
            'transform': {'name': 'tableWithLeafColNames'},
            'userQuery': {'queryFields': [{'name': 'runSets', 'args': [{'name': 'runSets', 'value': '${runSets}'}, {'name': 'limit', 'value': 500.}],
                'fields': [{'name': 'summaryTable', 'args': [{'name': 'tableKey', 'value': table}], 'fields': []}, {'name': 'id', 'value': []}, {'name': 'name', 'value': []}]}]},
            'panelDefId': chart_id(entity),
            'fieldSettings': {k: k for k in ('arm', 'label', 'metric', 'baseline', 'state', 'n', 'mean', 'ci_low', 'ci_high', 'sd')},
            'stringSettings': dict(title=title, subtitle=subtitle, metric=metric, baseline=baseline, yname=yname),
        }, width=12, height=9))
    groups.append(('outcomes', 'Critic bypass | full episodes and controller time', plots, 2))
    groups.append(('data', 'Critic bypass | compact verified data', [
        _panel('episodes', 'Media Browser', {'chartTitle': 'Completed episodes (partial coverage is not a final arm summary)', 'mediaKeys': [EPISODES_KEY]}, width=24, height=8),
        _panel('pairs', 'Media Browser', {'chartTitle': 'Complete-seed paired comparisons including variance and win rate', 'mediaKeys': [PAIRS_KEY]}, width=24, height=8),
        _panel('diagnostics', 'Media Browser', {'chartTitle': 'Same-state diagnostics; independent held-out model scores', 'mediaKeys': [DIAGNOSTICS_KEY]}, width=24, height=8)], 1))
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
    root, name = Path(receipt_dir), 'nw-criticbypass' + publication_id + '-v'
    receipt = dict(layout_version=VERSION, view_name=name, browser_verified=False,
                   url=f'https://wandb.ai/{entity}/{project}?nw={name[3:-2]}')
    try:
        receipt['chart_id'] = ensure_chart(api, entity)
        views = _views(api, entity, project)
        matches = [v for v in views if v['name'] == name]
        if matches:
            if len(matches) != 1 or not installed(_spec(matches[0]), entity, publication_id):
                raise BypassLayoutConflict('Existing critic-bypass view differs; preserving user edits.')
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
                name=name, displayName='800k critic bypass', spec=json.dumps(proposed, separators=(',', ':'))))
        except Exception as exc:
            error = type(exc).__name__
        after = _views(api, entity, project)
        _write(root / 'views-after.json', after)
        matches = [v for v in after if v['name'] == name and v['type'] == 'project-view']
        remaining = {v['id']: v for v in after}
        if len(matches) != 1 or _spec(matches[0]) != proposed or any(remaining.get(v['id']) != v for v in views):
            raise ResultsLayoutError('Critic-bypass view readback or existing-view preservation failed.')
        receipt.update(status='verified', changed=True, view_id=matches[0]['id'],
                       preserved_views=len(views), uncertain_response_reconciled=error is not None)
        _write(root / 'results-layout-receipt.json', receipt)
        return receipt
    except Exception as exc:
        _write(root / 'results-layout-receipt.json', dict(receipt, status='failed', error=f'{type(exc).__name__}: {exc}'))
        raise
