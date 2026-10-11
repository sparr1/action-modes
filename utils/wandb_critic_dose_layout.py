"""Owned W&B view for full-episode critic-dose outcomes, timing and progress."""
from copy import deepcopy
import json
from pathlib import Path
import re

from utils.wandb_results_layout import (
    DEFAULT_VIEW_NAME, ResultsLayoutError, _bank, _execute, _hash, _panel,
    _selected, _spec, _views, _write,
)

VERSION = 'critic-dose-v1'
SUMMARY_KEY = 'critic_dose/summary'
PAIRS_KEY = 'critic_dose/pairs'
EPISODES_KEY = 'critic_dose/episodes'
PROGRESS_KEY = 'critic_dose/progress'
DIAGNOSTICS_KEY = 'critic_dose/diagnostics'
ARMS = ('actor_mean_c16', 'learned_q_c16', 'model_score_c16', 'prior', 'mppi_h3',
        'actor_mean_c32', 'learned_q_c32', 'actor_mean_c64', 'learned_q_c64')
METHODS = ('Actor mean', 'Learned-Q selection', 'Direct-model selection', 'Frozen prior', 'MPPI H3')
LABELS = tuple(arm.replace('_', ' ') for arm in ARMS)
COLORS = ('#0072b2', '#cc79a7', '#009e73', '#333333', '#d55e00')
_QUERY = 'query CriticDoseChart($id:ID!){customChart(id:$id){id name type spec}}'
_CREATE_CHART = '''mutation CreateCriticDoseChart($entity:String!,$name:String!,
 $displayName:String!,$type:String!,$access:String!,$spec:JSONString!){
 createCustomChart(input:{entity:$entity,name:$name,displayName:$displayName,
 type:$type,access:$access,spec:$spec}){chart{id name type spec}}}'''
_CREATE_VIEW = '''mutation CreateCriticDoseView($entityName:String,$projectName:String,
 $type:String,$name:String,$displayName:String,$spec:String){
 upsertView(input:{entityName:$entityName,projectName:$projectName,type:$type,
 name:$name,displayName:$displayName,spec:$spec,createdUsing:WANDB_SDK}){
 view{id name} inserted}}'''


class DoseLayoutConflict(ResultsLayoutError):
    """A user-edited owned layout must not be overwritten."""


def chart_definition():
    colors = {'field': '${field:method}', 'type': 'nominal', 'title': 'Executed action',
              'scale': {'domain': list(METHODS), 'range': list(COLORS)},
              'legend': {'orient': 'bottom', 'columns': 2, 'labelLimit': 250}}
    x = {'field': '${field:critic_updates}', 'type': 'quantitative',
         'title': 'Critic updates per round C (J = 6)',
         'scale': {'domain': [12, 68], 'zero': False}, 'axis': {'values': [16, 32, 64]}}
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
                        for k in ('label', 'method', 'metric', 'baseline', 'critic_updates', 'n', 'mean', 'ci_low', 'ci_high', 'sd')]},
        'layer': [
            {'transform': [{'filter': "isNumber(datum['${field:critic_updates}'])"}],
             'mark': {'type': 'line', 'point': {'filled': True, 'size': 85}},
             'encoding': {'x': x, 'y': y}},
            {'transform': [{'filter': "isNumber(datum['${field:critic_updates}'])"}],
             'mark': {'type': 'rule', 'strokeWidth': 2}, 'encoding': {'x': x,
                 'y': {'field': '${field:ci_low}', 'type': 'quantitative', 'title': '${string:yname}', 'scale': {'zero': False}},
                 'y2': {'field': '${field:ci_high}'}}},
            {'transform': [{'filter': "!isNumber(datum['${field:critic_updates}'])"}],
             'mark': {'type': 'rule', 'strokeDash': [5, 4]}, 'encoding': {'y': y}},
        ],
        'config': {'view': {'stroke': None}, 'axis': {'gridColor': '#e8edf2'}},
    }


def chart_id(entity):
    return entity + '/critic_dose_' + _hash(chart_definition())[:16]


def ensure_chart(api, entity):
    ident, expected = chart_id(entity), chart_definition()
    chart = _execute(api, _QUERY, {'id': ident}).get('customChart')
    error = None
    if chart is None:
        try:
            _execute(api, _CREATE_CHART, dict(entity=entity, name=ident.split('/', 1)[1],
                displayName='800k critic update dose', type='vega2', access='PRIVATE',
                spec=json.dumps(expected, separators=(',', ':'))))
        except Exception as exc:
            error = exc
        chart = _execute(api, _QUERY, {'id': ident}).get('customChart')
    if not chart:
        raise ResultsLayoutError('Critic-bypass chart is not visible; retry its content-addressed identity.') from error
    if chart.get('type') != 'vega2' or _spec(chart) != expected:
        raise DoseLayoutConflict('Critic-bypass chart could not be verified; no existing chart overwritten.') from error
    return ident


def runset(publication_id):
    return dict(id='rs//Subsection 1', name='800k critic update dose', enabled=True,
        runFeed=dict(version=2, columnVisible={}, columnPinned={}, columnWidths={}, columnOrder=[], pageSize=50, onlyShowSelected=False),
        search={'query': ''}, searchHistory=[], grouping=[],
        filters={'filterFormat': 'filterV2', 'filters': [
            {'key': {'section': 'config', 'name': 'critic_dose_publication'},
             'op': '=', 'value': publication_id, 'disabled': False}]},
        sort={'keys': []}, selections={'root': 1, 'bounds': [], 'tree': []}, expandedRowAddresses=[])


def sections(entity):
    intro = ('### 800k critic-update dose: C16 → C32 → C64\n\n'
        '**Only critic updates per round change; J6, H1, A4, N128, B256 and fresh initialization stay fixed.** '
        'Color denotes executed-action method. The x axis is critic updates per round (96/192/384 total critic updates per decision). '
        'C16 results and the prior/MPPI references are immutable verified reuse; only C32/C64 actor-mean and learned-Q episodes are new. '
        'Outcome summaries require all 20 environment seeds per arm, controller seed 55. Pending and failed tasks remain explicit. '
        'Return is undiscounted full-episode reward; bars are exploratory 95% environment-seed percentile bootstrap intervals. '
        'Controller timings use eager A5000 execution, include necessary selection, and exclude sampled diagnostic work. '
        'P95 is pooled over timed decisions; five historical prior episodes contribute return but not timing. '
        'Diagnostic roots compare counterfactual choices on the same bank; different controller trajectories visit different states. '
        'Bank ranking uses all 770 candidate actions, including repeated replay entries, and the selection model draws. Held-out gains use independent draws. '
        'Diagnostics average requested roots within each environment seed before uncertainty across seeds. '
        'Spearman is undefined for constant-score banks: its n counts seeds with defined ranks, and the scalar summaries retain the fraction of defined roots. '
        'Shared-model ranking and return predictions do not constitute simulator truth. Only compact scalar tables are uploaded.')
    groups = [('progress', 'Critic dose | progress and integrity', [
        _panel('intro', 'Markdown Panel', {'value': intro}, width=24, height=7),
        _panel('progress', 'Media Browser', {'chartTitle': 'New C32/C64 tasks: pending, complete or failed', 'mediaKeys': [PROGRESS_KEY]}, width=24, height=8)], 1)]
    plots = []
    specs = (
        ('return', SUMMARY_KEY, 'episode_return', 'none', 'Full-episode return', 'Undiscounted reward'),
        ('gain-prior', PAIRS_KEY, 'episode_return', 'prior', 'Paired improvement over frozen prior', 'Return difference'),
        ('gain-c16', PAIRS_KEY, 'episode_return', 'matching_c16', 'Paired improvement over matching C16 method', 'Return difference'),
        ('variance', SUMMARY_KEY, 'episode_return_sd', 'none', 'Episode-return variability', 'Sample standard deviation'),
        ('control-mean', SUMMARY_KEY, 'controller_time_mean_s', 'none', 'Mean controller time including selection', 'Seconds / decision'),
        ('control-p95', SUMMARY_KEY, 'controller_time_p95_s', 'none', 'p95 controller time including selection', 'Seconds / decision'),
        ('selection', SUMMARY_KEY, 'selection_time_mean_s', 'none', 'Mean selection overhead', 'Seconds / decision'),
        ('ranking', SUMMARY_KEY, 'bank_spearman_model', 'none', 'Critic ranking versus the model candidate bank', 'Within-bank Spearman correlation'),
        ('heldout', SUMMARY_KEY, 'heldout_q_gain_vs_actor', 'none', 'Held-out model gain: learned-Q choice minus actor mean', 'Discounted model-return difference'),
        ('regret', SUMMARY_KEY, 'bank_model_regret_of_qpick', 'none', 'Model-bank regret of the learned-Q choice', 'Best model score minus chosen score'),
    )
    for identifier, table, metric, baseline, title, yname in specs:
        plots.append(_panel(identifier, 'Vega2', {
            'transform': {'name': 'tableWithLeafColNames'},
            'userQuery': {'queryFields': [{'name': 'runSets', 'args': [{'name': 'runSets', 'value': '${runSets}'}, {'name': 'limit', 'value': 500.}],
                'fields': [{'name': 'summaryTable', 'args': [{'name': 'tableKey', 'value': table}], 'fields': []}, {'name': 'id', 'value': []}, {'name': 'name', 'value': []}]}]},
            'panelDefId': chart_id(entity),
            'fieldSettings': {k: k for k in ('arm', 'label', 'method', 'critic_updates', 'metric', 'baseline', 'state', 'n', 'mean', 'ci_low', 'ci_high', 'sd')},
            'stringSettings': dict(title=title, subtitle='20 complete paired seeds per arm; fixed colors denote action selection', metric=metric, baseline=baseline, yname=yname),
        }, width=12, height=9))
    groups.append(('outcomes', 'Critic dose | performance, runtime and diagnostics', plots, 2))
    groups.append(('data', 'Critic dose | compact verified data', [
        _panel('summary', 'Media Browser', {'chartTitle': 'Complete-arm scalar summaries and diagnostic coverage', 'mediaKeys': [SUMMARY_KEY]}, width=24, height=8),
        _panel('episodes', 'Media Browser', {'chartTitle': 'Completed episodes, including verified C16 references', 'mediaKeys': [EPISODES_KEY]}, width=24, height=8),
        _panel('pairs', 'Media Browser', {'chartTitle': 'Paired comparisons, uncertainty and seed win rates', 'mediaKeys': [PAIRS_KEY]}, width=24, height=8),
        _panel('diagnostics', 'Media Browser', {'chartTitle': 'Same-state diagnostics and independent held-out scores', 'mediaKeys': [DIAGNOSTICS_KEY]}, width=24, height=8)], 1))
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
    root, name = Path(receipt_dir), 'nw-criticdose' + publication_id + '-v'
    receipt = dict(layout_version=VERSION, view_name=name, browser_verified=False,
                   url=f'https://wandb.ai/{entity}/{project}?nw={name[3:-2]}')
    try:
        receipt['chart_id'] = ensure_chart(api, entity)
        views = _views(api, entity, project)
        matches = [v for v in views if v['name'] == name]
        if matches:
            if len(matches) != 1 or not installed(_spec(matches[0]), entity, publication_id):
                raise DoseLayoutConflict('Existing critic-dose view differs; preserving user edits.')
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
                name=name, displayName='800k critic update dose', spec=json.dumps(proposed, separators=(',', ':'))))
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
