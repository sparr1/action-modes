"""Visible, idempotent W&B panels for the 575K mechanism discovery campaign."""
from copy import deepcopy
import errno
import json
from pathlib import Path
import re
import time
from utils.ambi_benchmark import atomic_json
from utils.wandb_results_layout import (_views, _selected, _spec, _bank, _hash, _write,
    _execute, _MUTATION, _panel, _chart, ResultsLayoutError, DEFAULT_VIEW_NAME)
LAYOUT_VERSION = 'transfer-discovery-v1'
OWNED_SECTION_IDS = tuple('ambi-' + LAYOUT_VERSION + '-' + s for s in ('progress', 'curves', 'results'))
TABLE_KEYS = ('discovery/settings', 'discovery/results', 'discovery/episodes')
CHART_KEYS = tuple(f'discovery/h{h}_return_vs_{axis}' for axis in ('j', 'compute') for h in (1,2,3))
_CREATE_VIEW = '''mutation CreateDiscoveryView($entityName:String,$projectName:String,
  $type:String,$name:String,$displayName:String,$spec:String){
  upsertView(input:{entityName:$entityName,projectName:$projectName,type:$type,
    name:$name,displayName:$displayName,spec:$spec,createdUsing:WANDB_SDK}){
    view{id name} inserted
  }
}'''

def discovery_sections():
    intro = ('### 575K transfer mechanism discovery · H1/2/3 · J1/2/4/6/8/10\n\n'
        '**252 configurations, three paired development seeds (101–103), 500 decisions per episode.** '
        'Nine actor/critic retention combinations plus behavior transfer, recent replay with fresh or '
        'joint weights, full learner-state carry, and joint carry with prior anchoring. '
        'Return-only inner critic and frozen return tail; C16/A4/N128/B256; solve every decision. '
        '**Pending values are null, never zero. Partial episodes are progress only.** '
        'Curves include only complete three-seed panels. These are exploratory results, not confirmation. '
        'Timing includes controller bookkeeping and first-solve compilation; first and later solves '
        'are separated in the artifact. Historical controls are not substituted. '
        'Paired gains compare the same H/J fresh controller; uncertainty is episode-level sample SD.')
    panels = [
        [_panel('intro','Markdown Panel',{'value':intro},width=24,height=5),
         _panel('settings','Media Browser',{'chartTitle':'All settings and live progress','mediaKeys':[TABLE_KEYS[0]]},width=24,height=9)],
        [_chart(key, f'H{h}: return versus ' + ('J' if axis=='j' else 'controller time'),
                'J rounds per solve' if axis=='j' else 'Controller seconds per decision')
         for axis in ('j','compute') for h in (1,2,3) for key in [f'discovery/h{h}_return_vs_{axis}']],
        [_panel('results','Media Browser',{'chartTitle':'Complete returns, paired gains and timing','mediaKeys':[TABLE_KEYS[1]]},width=24,height=9),
         _panel('episodes','Media Browser',{'chartTitle':'Completed per-seed outcomes','mediaKeys':[TABLE_KEYS[2]]},width=24,height=8)]
    ]
    sections=[]
    for identifier,name,items,columns,rows in zip(OWNED_SECTION_IDS,
            ('Transfer discovery | progress','Transfer discovery | return and compute','Transfer discovery | measurements'),
            panels,(1,3,1),(2,2,2)):
        for i,p in enumerate(items):
            p['__id__']=identifier+'-panel-'+str(i)
        sections.append(dict(__id__=identifier,name=name,isOpen=True,type='flow',
            flowConfig=dict(snapToColumns=True,columnsPerPage=columns,rowsPerPage=rows,
                            gutterWidth=16,boxWidth=460,boxHeight=320),
            sorted=0,pinned=True,isPanelsAuto=False,panels=items))
    return sections

def patch_discovery_spec(spec):
    result=deepcopy(spec); bank=_bank(result)
    bank['sections']=discovery_sections()+[s for s in bank['sections'] if s.get('__id__') not in OWNED_SECTION_IDS]
    return result

def _without_owned(spec):
    result=deepcopy(spec); bank=_bank(result)
    bank['sections']=[s for s in bank['sections'] if s.get('__id__') not in OWNED_SECTION_IDS]
    return result

def _installed(spec):
    """Accept observed UI omission of layout defaults, never changed values/content.

    W&B's UI drops section ``type``/flow defaults and whole panel ``layout``
    mappings when resaving this flow workspace. Restore only absent defaults
    in a copy for comparison; IDs, queries, configs, visibility, order, and any
    explicitly saved layout values still have to match exactly.
    """
    actual = deepcopy([s for s in _bank(spec)['sections'] if s.get('__id__') in OWNED_SECTION_IDS])
    expected = discovery_sections()
    if len(actual) != len(expected):
        return False
    for section, wanted in zip(actual, expected):
        section.setdefault('type', wanted['type'])
        flow = section.setdefault('flowConfig', {})
        if not isinstance(flow, dict):
            return False
        for key, value in wanted['flowConfig'].items():
            flow.setdefault(key, value)
        panels = section.get('panels')
        if not isinstance(panels, list) or len(panels) != len(wanted['panels']):
            return False
        for panel, wanted_panel in zip(panels, wanted['panels']):
            if not isinstance(panel, dict):
                return False
            panel.setdefault('layout', wanted_panel['layout'])
    return actual == expected


def ensure_discovery_results_layout(api, *, entity, project, receipt_dir,
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
    before = _spec(view); proposed = patch_discovery_spec(before)
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


def _campaign_runset(run_id):
    return dict(id='rs//Subsection 1', name='575K transfer discovery', enabled=True,
        runFeed=dict(version=2, columnVisible={}, columnPinned={}, columnWidths={},
                     columnOrder=[], pageSize=10, onlyShowSelected=False),
        search={'query':''}, searchHistory=[], grouping=[],
        filters={'filterFormat':'filterV2', 'filters':[
            {'key':{'section':'config','name':'publication_id'}, 'op':'=',
             'value':run_id, 'disabled':False}]},
        sort={'keys':[{'key':{'section':'run','name':'createdAt'},'ascending':False}]},
        selections={'root':1,'bounds':[],'tree':[]}, expandedRowAddresses=[])


def _saved_spec(template, run_id):
    result = deepcopy(template)
    section = result['section']
    section['runSets'] = [_campaign_runset(run_id)]
    section['openRunSet'] = 0
    section['workspaceSettings'] = {'shouldAutoGeneratePanels':False}
    bank = _bank(result)
    bank['sections'] = discovery_sections()
    bank['panelPlacementOverrides'] = {}
    return result


def _saved_installed(spec, run_id):
    return (_installed(spec)
        and len(_bank(spec)['sections']) == len(OWNED_SECTION_IDS)
        and spec['section'].get('runSets') == [_campaign_runset(run_id)]
        and not _bank(spec).get('panelPlacementOverrides'))


def _saved_receipt_write(path, value):
    # Retry only local scratch I/O, never a potentially successful API mutation.
    delays = (1, 2, 4, 8)
    for attempt in range(len(delays) + 1):
        try:
            return atomic_json(path, value, overwrite=True)
        except OSError as exc:
            if exc.errno != errno.ESTALE or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def ensure_discovery_saved_view(api, *, entity, project, receipt_dir, run_id):
    """Create one campaign-filtered saved view, never update existing views.

    The deterministic name reconciles an uncertain create response on retry.
    A name collision or user-edited saved view fails visibly instead of being
    overwritten. Readback verifies the selected run, panels and preservation of
    every prior view; browser rendering requires a separate authenticated check.
    """
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ResultsLayoutError('A valid explicit publication run ID is required.')
    root = Path(receipt_dir)
    name = 'nw-transfer575-' + run_id + '-v'
    display_name = '575K transfer discovery · H1/2/3 · J sweep'
    url = f'https://wandb.ai/{entity}/{project}/workspace?nw={name[3:-2]}'
    receipt = dict(schema_version=1, layout_version=LAYOUT_VERSION, entity=entity,
        project=project, view_name=name, view_type='project-view', run_id=run_id,
        layout_scope='dedicated campaign-filtered saved project workspace',
        url=url, workspace_url=url,
        run_url=f'https://wandb.ai/{entity}/{project}/runs/{run_id}?nw={name[3:-2]}',
        owned_section_ids=list(OWNED_SECTION_IDS), expected_chart_keys=list(CHART_KEYS),
        expected_table_keys=list(TABLE_KEYS),
        verification='saved workspace schema read back; browser rendering must be checked separately')
    views = _views(api, entity, project)
    matches = [v for v in views if v['name'] == name]
    if matches:
        if (len(matches) != 1 or matches[0]['type'] != 'project-view'
                or not _saved_installed(_spec(matches[0]), run_id)):
            raise ResultsLayoutError('Existing discovery saved view differs; preserve it and inspect before retrying.')
        receipt.update(status='verified', changed=False, view_id=matches[0]['id'],
                       after_sha256=_hash(_spec(matches[0])))
        _saved_receipt_write(root/'results-layout-receipt.json', receipt)
        return receipt
    proposed = _saved_spec(_spec(_selected(views, DEFAULT_VIEW_NAME)), run_id)
    _saved_receipt_write(root/('before-' + _hash(views)[:16] + '.json'), views)
    _saved_receipt_write(root/('proposed-' + _hash(proposed)[:16] + '.json'), proposed)
    current = _views(api, entity, project)
    if {v['id']:v for v in current} != {v['id']:v for v in views}:
        raise ResultsLayoutError('Project views changed during preparation; retry from fresh state.')
    _saved_receipt_write(root/'results-layout-intent.json', {**receipt, 'status':'prepared'})
    mutation_error = None
    try:
        _execute(api, _CREATE_VIEW, dict(entityName=entity, projectName=project,
            type='project-view', name=name, displayName=display_name,
            spec=json.dumps(proposed, separators=(',', ':'))))
    except Exception as exc:
        mutation_error = type(exc).__name__
    try:
        after_views = _views(api, entity, project)
        _saved_receipt_write(root/('after-' + _hash(after_views)[:16] + '.json'), after_views)
        matches = [v for v in after_views if v['name'] == name and v['type'] == 'project-view']
        remaining = {v['id']:v for v in after_views}
        if (len(matches) != 1 or _spec(matches[0]) != proposed
                or any(remaining.get(v['id']) != v for v in views)):
            raise ResultsLayoutError('Saved view or existing views differ from expected readback.')
        receipt.update(status='verified', changed=True, view_id=matches[0]['id'],
            after_sha256=_hash(_spec(matches[0])), preserved_existing_views=len(views),
            uncertain_response_reconciled=mutation_error is not None)
        _saved_receipt_write(root/'results-layout-receipt.json', receipt)
        return receipt
    except Exception as exc:
        _saved_receipt_write(root/'results-layout-receipt.json', {**receipt, 'status':'uncertain',
            'mutation_error_type':mutation_error, 'verification_error_type':type(exc).__name__})
        raise ResultsLayoutError('Saved discovery view could not be verified; inspect receipts before retrying. Evaluation data are unaffected.') from exc
