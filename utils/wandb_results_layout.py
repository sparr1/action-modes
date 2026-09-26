"""Install explicit results panels in an existing personal W&B project view.

Logging metrics/custom charts does not display them when automatic panels are
disabled. This module uses only the existing W&B SDK transport; panel dictionaries
match wandb-workspaces 0.4.5 serialization. It never edits runs or their history.
The caller must be authorized to update the explicitly selected project
workspace. The current run UI also reads this project view, verified by a
native panel edit; the legacy viewer-owned run-view is not used here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path

LAYOUT_VERSION = 'actor-transfer-results-v1'
DEFAULT_VIEW_NAME = 'nw-nwuserrwgao_b-w'
OWNED_SECTION_IDS = tuple(f'ambi-{LAYOUT_VERSION}-{name}' for name in ('overview', 'curves', 'tables'))
TABLE_KEYS = ('campaign/settings', 'comparison/returns_and_compute', 'comparison/warm_minus_cold')
CHART_KEYS = tuple(f'comparison/h{h}_return_vs_{axis}' for axis in ('rounds', 'compute') for h in (1, 2, 3))
_QUERY = '''query ResultsViews($entityName:String,$name:String){
  project(name:$name,entityName:$entityName){allViews(viewType:"project-view"){
    edges{node{id name type displayName spec entityName projectName projectId parentId userId}}}}
}'''
_MUTATION = '''mutation InstallResultsPanels($id:ID,$type:String,$name:String,$displayName:String,$spec:String){
  upsertView(input:{id:$id,type:$type,name:$name,displayName:$displayName,spec:$spec,createdUsing:WANDB_SDK}){
    view{id name} inserted
  }
}'''


class ResultsLayoutError(RuntimeError):
    """A presentation failure; evaluation and immutable publication may continue."""


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def _execute(api, document, variables):
    """Use the same current/legacy transport selection as wandb-workspaces."""
    service = getattr(api, '__dict__', {}).get('_service_api')
    if service is not None and hasattr(service, 'execute_graphql'):
        return service.execute_graphql(document, variables=dict(variables))
    # Importing wandb installs its vendored wandb_gql module on older SDKs.
    import wandb  # noqa: F401
    gql = importlib.import_module('wandb_gql').gql
    return api.client.execute(gql(document), variable_values=dict(variables))


def _views(api, entity=None, project=None):
    # A native panel edit from the current single-run UI updates this project's
    # project-view. The similarly named legacy viewer run-view is unrelated.
    if not entity or not project:
        raise ResultsLayoutError('An explicit entity and project are required.')
    response = _execute(api, _QUERY, {'entityName': entity, 'name': project})
    connection = (response.get('project') or {}).get('allViews')
    if connection is None:
        raise ResultsLayoutError('The requested project did not return workspace views.')
    return [edge['node'] for edge in connection['edges']]


def _selected(views, name):
    matches = [v for v in views if v['name'] == name and v['type'] == 'project-view']
    if len(matches) != 1:
        raise ResultsLayoutError('Expected one existing project-view with the explicit requested personal name.')
    if not name.startswith('nw-nwuser') or not name.endswith('-w'):
        raise ResultsLayoutError('Results panels may be installed only into an explicitly named personal project view.')
    return matches[0]


def _spec(view):
    return json.loads(view['spec']) if isinstance(view['spec'], str) else deepcopy(view['spec'])


def _panel(identifier, kind, config, *, width=8, height=6):
    return {'__id__': f'ambi-{LAYOUT_VERSION}-{identifier}',
            'layout': {'x': 0, 'y': 0, 'w': width, 'h': height},
            'viewType': kind, 'config': config, 'isAuto': False}


def _chart(key, title, xname):
    return _panel(key.replace('/', '-'), 'Vega2', {
        'transform': {'name': 'tableWithLeafColNames'},
        'userQuery': {'queryFields': [{
            'name': 'runSets', 'args': [{'name': 'runSets', 'value': '${runSets}'}, {'name': 'limit', 'value': 500.0}],
            'fields': [{'name': 'summaryTable', 'args': [{'name': 'tableKey', 'value': key + '_table'}], 'fields': []},
                       {'name': 'id', 'value': []}, {'name': 'name', 'value': []}]}]},
        'panelDefId': 'wandb/lineseries/v0',
        'fieldSettings': {'step': 'step', 'lineKey': 'lineKey', 'lineVal': 'lineVal'},
        'stringSettings': {'title': title, 'xname': xname}})


def actor_transfer_sections():
    """Static panels work before results arrive and as the same run is updated."""
    intro = ('### Evaluation results\n\n'
        'Results panels display the summaries published by the selected run. '
        'The settings table distinguishes pending evaluations from completed results; '
        '**missing chart points are unavailable, never zero measurements.**')
    overview = [_panel('explanation', 'Markdown Panel', {'value': intro}, width=24, height=4),
                _panel('progress', 'Media Browser', {'chartTitle': 'Evaluation settings and publication progress',
                                                    'mediaKeys': ['campaign/settings']}, width=24)]
    charts = [_chart(f'comparison/h{h}_return_vs_{axis}',
                     f'H{h}: full-episode return versus ' + ('subsequent J' if axis == 'rounds' else 'controller time'),
                     'Subsequent J (first J10)' if axis == 'rounds' else 'Controller seconds / decision')
              for axis in ('rounds', 'compute') for h in (1, 2, 3)]
    tables = [_panel('measurements', 'Media Browser', {'chartTitle': 'Episode returns and timing',
                 'mediaKeys': ['comparison/returns_and_compute']}, width=12),
              _panel('paired-effect', 'Media Browser', {'chartTitle': 'Warm minus matched cold, paired uncertainty',
                 'mediaKeys': ['comparison/warm_minus_cold']}, width=12)]
    result = []
    for identifier, name, panels, columns, rows in zip(OWNED_SECTION_IDS,
            ('Results | evaluation progress', 'Results | actor-transfer return and compute', 'Results | actor-transfer measurements'),
            (overview, charts, tables), (1, 3, 2), (2, 2, 1)):
        result.append({'__id__': identifier, 'name': name, 'isOpen': True, 'type': 'flow',
            'flowConfig': {'snapToColumns': True, 'columnsPerPage': columns, 'rowsPerPage': rows,
                           'gutterWidth': 16, 'boxWidth': 460, 'boxHeight': 300},
            'sorted': 0, 'pinned': True, 'isPanelsAuto': False, 'panels': panels})
    return result


def _bank(spec):
    # The installer uses project-view's nested form. Top-level support is useful
    # for inspecting legacy run-view receipts without changing routing.
    bank = spec.get('panelBankConfig')
    if bank is None:
        bank = (spec.get('section') or {}).get('panelBankConfig')
    if not isinstance(bank, dict) or not isinstance(bank.get('sections'), list):
        raise ResultsLayoutError('Unrecognized workspace structure; no write attempted.')
    return bank


def patch_actor_transfer_spec(spec):
    """Replace only our stable section IDs; preserve every unrelated field."""
    proposed = deepcopy(spec)
    bank = _bank(proposed)
    remaining = [section for section in bank['sections'] if section.get('__id__') not in OWNED_SECTION_IDS]
    bank['sections'] = actor_transfer_sections() + remaining
    return proposed


def _without_owned(spec):
    result = deepcopy(spec); bank = _bank(result)
    bank['sections'] = [s for s in bank['sections'] if s.get('__id__') not in OWNED_SECTION_IDS]
    return result


def _installed(spec):
    return [s for s in _bank(spec)['sections'] if s.get('__id__') in OWNED_SECTION_IDS] == actor_transfer_sections()


def ensure_actor_transfer_results_layout(api, *, entity, project, receipt_dir,
                                         view_name=DEFAULT_VIEW_NAME, run_id=None):
    """Idempotently patch and verify the project's personal workspace, with receipts.

    Two fresh reads detect changes before mutation. This helper does not perform
    an atomic compare-and-swap: a concurrent edit after the second read can be overwritten
    without readback detecting it. Keep this remaining race window explicit.
    An uncertain response is reconciled by reading the same view, never creating
    another view or duplicate sections. No run history or filters are changed.
    """
    root = Path(receipt_dir); root.mkdir(parents=True, exist_ok=True)
    views = _views(api, entity, project); view = _selected(views, view_name)
    before = _spec(view); proposed = patch_actor_transfer_spec(before)
    fingerprint = _hash(proposed)
    assert _without_owned(before) == _without_owned(proposed)
    workspace_url = f'https://wandb.ai/{entity}/{project}/workspace?nw={view_name[3:-2]}'
    url = f'https://wandb.ai/{entity}/{project}/runs/{run_id}?nw={view_name[3:-2]}' if run_id else None
    receipt = dict(schema_version=1, layout_version=LAYOUT_VERSION, entity=entity, project=project,
        view_id=view['id'], view_name=view_name, view_type='project-view',
        layout_scope='selected personal project workspace and its run pages',
        url=url, workspace_url=workspace_url, run_id=run_id, owned_section_ids=list(OWNED_SECTION_IDS),
        expected_chart_keys=list(CHART_KEYS), expected_table_keys=list(TABLE_KEYS),
        before_sha256=_hash(before), proposed_sha256=fingerprint,
        verification='saved workspace schema read back; browser rendering must be checked separately')
    _write(root / ('before-' + _hash(before)[:16] + '.json'), views)
    _write(root / ('proposed-' + fingerprint[:16] + '.json'), proposed)
    if _installed(before):
        receipt.update(status='verified', changed=False, after_sha256=_hash(before))
        _write(root / 'results-layout-receipt.json', receipt)
        return receipt
    current_views = _views(api, entity, project); current = _selected(current_views, view_name)
    if current != view:
        raise ResultsLayoutError('Personal workspace changed during preparation; retry from its fresh state.')
    _write(root / 'results-layout-intent.json', {**receipt, 'status': 'prepared'})
    mutation_error = None
    try:
        _execute(api, _MUTATION, {'id': view['id'], 'type': view['type'], 'name': view_name, 'displayName': view['displayName'],
            'spec': json.dumps(proposed, separators=(',', ':'))})
    except Exception as exc:
        mutation_error = type(exc).__name__
    try:
        after_views = _views(api, entity, project)
        after_view = _selected(after_views, view_name); after = _spec(after_view)
        _write(root / ('after-' + _hash(after)[:16] + '.json'), after_views)
        old_others = {v['id']: v for v in views if v['id'] != view['id']}
        new_others = {v['id']: v for v in after_views if v['id'] != view['id']}
        if (after != proposed or {k:v for k,v in after_view.items() if k != 'spec'} !=
                {k:v for k,v in view.items() if k != 'spec'} or
                any(new_others.get(key) != value for key, value in old_others.items())):
            raise ResultsLayoutError('Workspace readback differs; preserve the receipt and inspect concurrent edits before retrying.')
        receipt.update(status='verified', changed=True, after_sha256=_hash(after),
                       uncertain_response_reconciled=mutation_error is not None,
                       preserved_existing_views=len(old_others))
        _write(root / 'results-layout-receipt.json', receipt)
        return receipt
    except Exception as exc:
        _write(root / 'results-layout-receipt.json', {**receipt, 'status': 'uncertain',
            'mutation_error_type': mutation_error, 'verification_error_type': type(exc).__name__})
        raise ResultsLayoutError('Results-layout write could not be verified; inspect the saved before/after receipts. Evaluation data are unaffected.') from exc
