"""Visible, idempotent W&B panels for the 575K mechanism discovery campaign."""
from copy import deepcopy
import errno
import hashlib
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
DIAGNOSTIC_CHARTS = (
    ('initial_critic_rmse', 'Initial critic RMSE against fixed model target'),
    ('final_critic_rmse', 'Final critic RMSE against fixed model target'),
    ('initial_policy_kl', 'Initial actor KL to pretrained prior'),
    ('final_model_first_action_gain_vs_prior', 'Final first-action model gain; prior continuation'),
    ('final_action_gradient_gain', 'Final critic-gradient model improvement'),
    ('initial_actor_feature_effective_rank_fraction', 'Initial actor effective feature rank / available rank'),
    ('critic_initial_stationary_heldout_improvement', 'Transferred critic fixed-target heldout improvement'),
    ('critic_prior_stationary_heldout_improvement', 'Prior critic fixed-target heldout improvement'),
    ('diagnostic_seconds_per_decision', 'Diagnostic seconds per episode decision; separate from control'),
)
_CHART_QUERY = 'query TransferChart($id:ID!){customChart(id:$id){id name type spec}}'
_CREATE_CHART = '''mutation CreateTransferChart($entity:String!,$name:String!,
 $displayName:String!,$type:String!,$access:String!,$spec:JSONString!){
 createCustomChart(input:{entity:$entity,name:$name,displayName:$displayName,
 type:$type,access:$access,spec:$spec}){chart{id name type spec}}
}'''


def campaign_chart_definition(campaign, component=None):
    """Map color to the mechanism, not to the single publication run."""
    publication = campaign['publication']
    styles = [row for row in publication['arm_styles'] if component is None
              or publication.get('arm_components', {}).get(row[0]) == component]
    if campaign.get('historical_reference'):
        styles.append(['rho_a0_c0', 'Fresh prior reset (historical)', '#000000'])
    if (len({row[0] for row in styles}) != len(styles)
            or len({row[2] for row in styles}) != len(styles)):
        raise ResultsLayoutError('Campaign arm styles must have unique identities and colors.')
    labels = {arm: label for arm, label, _ in styles}
    definition = {
        '$schema':'https://vega.github.io/schema/vega-lite/v5.json',
        'data':{'name':'wandb'}, 'width':'container', 'height':280,
        'autosize':{'type':'fit','contains':'padding'},
        'title':{'text':'${string:title}', 'anchor':'start'},
        'transform':[
            {'filter':"isNumber(datum['${field:step}']) && isFinite(datum['${field:step}']) && isNumber(datum['${field:lineVal}']) && isFinite(datum['${field:lineVal}'])"},
            {'calculate':json.dumps(labels, separators=(',', ':')) + "[datum['${field:lineKey}']]", 'as':'Mechanism'},
        ],
        'mark':{'type':'line', 'point':True, 'strokeWidth':2.5},
        'encoding':{
            'x':{'field':'${field:step}', 'type':'quantitative', 'title':'${string:xname}', 'scale':{'zero':False}},
            'y':{'field':'${field:lineVal}', 'type':'quantitative', 'title':'${string:yname}', 'scale':{'zero':False}},
            'color':{'field':'Mechanism', 'type':'nominal', 'title':None,
                'scale':{'domain':list(labels.values()), 'range':[row[2] for row in styles]},
                'legend':{'orient':'bottom','columns':1,'labelLimit':240}},
            'detail':{'field':'${field:lineKey}', 'type':'nominal'},
            'order':{'field':'${field:step}', 'type':'quantitative'},
            'tooltip':[{'field':'Mechanism','type':'nominal'},
                {'field':'${field:step}','type':'quantitative','title':'${string:xname}'},
                {'field':'${field:lineVal}','type':'quantitative','title':'${string:yname}'}],
        },
        'config':{'view':{'stroke':None}, 'axis':{'gridColor':'#e8edf2'}, 'legend':{'labelFontSize':11}},
    }

    if publication.get('probability_sweep'):
        probabilities = {arm: f'{100 * probability:g}%' for arm, probability
                         in publication['arm_probabilities'].items()}
        probabilities['rho_a0_c0'] = 'Fresh'
        definition['transform'].append({'calculate':json.dumps(probabilities, separators=(',', ':'))
            + "[datum['${field:lineKey}']]", 'as':'Retention'})
        definition['encoding']['strokeDash'] = {'field':'Retention', 'type':'nominal',
            'scale':{'domain':['25%', '50%', '75%', 'Fresh'], 'range':[[2,3], [], [8,3], []]}, 'legend':None}
    return definition


def campaign_chart_id(entity, campaign, component=None):
    if campaign['publication'].get('probability_sweep') and component is None:
        return {component: campaign_chart_id(entity, campaign, component) for component in ('actor', 'critic', 'joint')}
    return entity + '/bernoulli_transfer_' + _hash(campaign_chart_definition(campaign, component))[:16]


def campaign_curve_groups(campaign):
    publication = campaign.get('publication', {})
    fresh = ['rho_a0_c0'] if campaign.get('historical_reference') else []
    if publication.get('probability_sweep'):
        return [(component+'/', component.capitalize()+' | ',
                 [arm for arm, _, _ in publication['arm_styles']
                  if publication['arm_components'][arm] == component] + fresh)
                for component in ('actor', 'critic', 'joint')]
    return [('', '', list(dict.fromkeys([c['arm'] for c in campaign['cells']] + fresh)))]


def expected_campaign_chart_keys(campaign):
    keys = []
    for prefix, _, _ in campaign_curve_groups(campaign):
        keys.extend(f'discovery/{prefix}h{h}_return_vs_{axis}' for axis in ('j', 'compute') for h in campaign['H'])
        if campaign.get('diagnostics', {}).get('enabled'):
            keys.extend(f'discovery/{prefix}h{h}_{metric}_vs_j' for metric, _ in DIAGNOSTIC_CHARTS for h in campaign['H'])
    return keys


def ensure_campaign_chart(api, *, entity, campaign, component=None):
    """Create an immutable content-addressed chart, reconcile, and verify it."""
    if campaign['publication'].get('probability_sweep') and component is None:
        return {component: ensure_campaign_chart(api, entity=entity, campaign=campaign, component=component)
                for component in ('actor', 'critic', 'joint')}
    identifier = campaign_chart_id(entity, campaign, component)
    expected = campaign_chart_definition(campaign, component)
    chart = _execute(api, _CHART_QUERY, {'id':identifier}).get('customChart')
    mutation_error = None
    if chart is None:
        try:
            _execute(api, _CREATE_CHART, dict(entity=entity, name=identifier.split('/', 1)[1],
                displayName='Bernoulli transfer mechanisms and diagnostics', type='vega2', access='PRIVATE',
                spec=json.dumps(expected, separators=(',', ':'))))
        except Exception as exc:
            mutation_error = exc
        chart = _execute(api, _CHART_QUERY, {'id':identifier}).get('customChart')
    actual = json.loads(chart['spec']) if chart and isinstance(chart['spec'], str) else chart.get('spec') if chart else None
    if not chart or chart.get('type') != 'vega2' or actual != expected:
        raise ResultsLayoutError('Campaign mechanism-color chart registration could not be verified: ' + identifier) from mutation_error
    return identifier


def _campaign_curve(key, title, xname, yname, chart_id):
    panel = _chart(key, title, xname)
    panel['config']['panelDefId'] = chart_id
    panel['config']['stringSettings']['yname'] = yname
    panel['layout']['h'] = 8
    return panel


def campaign_sections(campaign, chart_id):
    diagnostics = bool(campaign.get('diagnostics', {}).get('enabled'))
    reference = campaign.get('historical_reference')
    sweep = campaign['publication'].get('probability_sweep', False)
    count = len(campaign['cells'])
    reference_rounds = {record['cell']['J'] for item in campaign.get('historical_references', [])
                        for record in item.get('records', [])}
    new_rounds = set(campaign['J'])
    round_label = 'J=' + ','.join(map(str, campaign['J']))
    if reference_rounds - new_rounds:
        round_label = ('new ' + round_label + '; plotted J='
                       + ','.join(map(str, sorted(reference_rounds | new_rounds))))
    scope = 'H=' + ','.join(map(str, campaign['H'])) + '; ' + round_label
    intro = ('### ' + campaign['publication']['view_title'] + '\n\n'
        f'**{count} new configurations, three paired development seeds (101–103), 500 decisions per episode.** '
        + scope + '; C16/A4/N128/B256; solve every decision. '
        'Blue: actor-only; orange: critic-only; green: joint Bernoulli copying. ')
    if sweep:
        intro += ('**25% and 75% are new evaluations; 50% reuses the completed screen.** '
            'Separate actor, critic and joint panels compare at most four curves each. '
            'Dotted/light: 25%; solid/medium: 50%; dashed/dark: 75%; black: fresh. '
            'p is the independent probability of retaining each adapted scalar parameter. ')
    else:
        intro += 'p=0.5 retains each adapted scalar parameter with probability one half. '
    intro += ('Other parameters restore their frozen prior. Adam, replay and temperature reset; '
        'target critic copies the resulting online critic. '
        '**Pending values are null, never zero. Partial episodes are progress only.** '
        'Curves require complete three-seed panels. Return uncertainty in the table is episode sample SD. '
        'Diagnostic probes are isolated from controller learning and their measured time is reported separately. '
        'Diagnostic means average sampled-root measurements within each episode, then average episodes; model probes are not environment returns. '
        'This is an exploratory screen on one checkpoint, not confirmation. ')
    if reference:
        intro += ('**Black is a reused historical fresh-prior baseline**, pinned to source `' + reference['source_commit'][:12] + '`. '
            'Paired gains use the same environment and solver seeds; trajectories visit different states. '
            'Fresh controls have no new diagnostic measurements. Reused panels are excluded from new-run completion counts. '
            'Controller time includes first-solve compilation and transfer bookkeeping. ')
    else:
        intro += 'Historical fresh controls are not present in this view; paired-versus-fresh values remain unavailable. '
    for item in campaign.get('historical_references', []):
        if item.get('kind') == 'bernoulli':
            intro += ('**Reused 50% results and diagnostics** are pinned to source `' + item['source_commit'][:12]
                + '` with matching diagnostic settings. Tables label provenance explicitly. ')
            if sweep:
                intro += 'Diagnostic sections are initially collapsed to keep the return comparison quick to read. '
    blocks = [
        ('progress', 'Bernoulli transfer | protocol and live progress', 1, [
            _panel('intro','Markdown Panel',{'value':intro},width=24,height=9 if sweep else 7),
            _panel('settings','Media Browser',{'chartTitle':f'{count} new settings and live progress','mediaKeys':['discovery/settings']},width=24,height=9)]),
    ]
    groups = campaign_curve_groups(campaign)
    for prefix, label, _ in groups:
        identifier = chart_id[prefix.rstrip('/')] if isinstance(chart_id, dict) else chart_id
        suffix = '-' + prefix.rstrip('/') if prefix else ''
        blocks.append(('curves'+suffix, 'Bernoulli transfer | '+label+'environment return and controller time', 3, [
            _campaign_curve(f'discovery/{prefix}h{h}_return_vs_{axis}', f'{label}H{h}: return versus ' + ('J' if axis == 'j' else 'controller time'),
                'J rounds per solve' if axis == 'j' else 'Controller seconds per decision', 'Mean episode return', identifier)
            for axis in ('j','compute') for h in campaign['H']]))
    blocks.append(('results', 'Bernoulli transfer | complete results and provenance', 1, [
        _panel('results','Media Browser',{'chartTitle':'Complete returns, paired fresh gains, control and diagnostic timing','mediaKeys':['discovery/results']},width=24,height=10),
        _panel('episodes','Media Browser',{'chartTitle':'Per-seed results; historical rows explicitly labeled','mediaKeys':['discovery/episodes']},width=24,height=8)]))
    if diagnostics:
        for prefix, label, _ in groups:
            identifier = chart_id[prefix.rstrip('/')] if isinstance(chart_id, dict) else chart_id
            suffix = '-' + prefix.rstrip('/') if prefix else ''
            blocks.append(('diagnostics'+suffix, 'Bernoulli transfer | '+label+'sampled-root diagnostics', 3, [
                _campaign_curve(f'discovery/{prefix}h{h}_{metric}_vs_j', f'{label}H{h}: {title}', 'J rounds per solve', title, identifier)
                for metric, title in DIAGNOSTIC_CHARTS for h in campaign['H']]))
        blocks.append(('diagnostic-table', 'Bernoulli transfer | complete diagnostic measurements', 1, [
            _panel('diagnostic-table','Media Browser',{'chartTitle':'All diagnostic means, episode SDs, coverage and provenance','mediaKeys':['discovery/diagnostics']},width=24,height=12)]))
    sections = []
    for suffix, title, columns, panels in blocks:
        identifier = 'ambi-bernoulli-transfer-v1-' + suffix
        for index, panel in enumerate(panels):
            panel['__id__'] = identifier + '-panel-' + str(index)
        sections.append(dict(__id__=identifier, name=title, isOpen=not (sweep and suffix.startswith('diagnostic')), type='flow',
            flowConfig=dict(snapToColumns=True, columnsPerPage=columns, rowsPerPage=2,
                gutterWidth=16, boxWidth=460, boxHeight=430 if columns == 3 else 320),
            sorted=0, pinned=True, isPanelsAuto=False, panels=panels))
    return sections


_CREATE_VIEW = '''mutation CreateDiscoveryView($entityName:String,$projectName:String,
  $type:String,$name:String,$displayName:String,$spec:String){
  upsertView(input:{entityName:$entityName,projectName:$projectName,type:$type,
    name:$name,displayName:$displayName,spec:$spec,createdUsing:WANDB_SDK}){
    view{id name} inserted
  }
}'''

def discovery_sections(campaign=None, chart_id=None):
    if campaign and campaign.get('publication', {}).get('arm_styles'):
        return campaign_sections(campaign, chart_id or campaign_chart_id('rwgao_b-brown-university', campaign))
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

def _installed(spec, campaign=None, chart_id=None):
    """Accept observed UI omission of layout defaults, never changed values/content.

    W&B's UI drops section ``type``/flow defaults and whole panel ``layout``
    mappings when resaving this flow workspace. Restore only absent defaults
    in a copy for comparison; IDs, queries, configs, visibility, order, and any
    explicitly saved layout values still have to match exactly.
    """
    expected = discovery_sections(campaign, chart_id)
    owned = {s['__id__'] for s in expected}
    actual = deepcopy([s for s in _bank(spec)['sections'] if s.get('__id__') in owned])
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


def _campaign_runset(run_id, campaign=None):
    return dict(id='rs//Subsection 1', name=(campaign or {}).get('publication', {}).get('view_title', '575K transfer discovery'), enabled=True,
        runFeed=dict(version=2, columnVisible={}, columnPinned={}, columnWidths={},
                     columnOrder=[], pageSize=10, onlyShowSelected=False),
        search={'query':''}, searchHistory=[], grouping=[],
        filters={'filterFormat':'filterV2', 'filters':[
            {'key':{'section':'config','name':'publication_id'}, 'op':'=',
             'value':run_id, 'disabled':False}]},
        sort={'keys':[{'key':{'section':'run','name':'createdAt'},'ascending':False}]},
        selections={'root':1,'bounds':[],'tree':[]}, expandedRowAddresses=[])


def _saved_spec(template, run_id, campaign=None, chart_id=None):
    result = deepcopy(template)
    section = result['section']
    section['runSets'] = [_campaign_runset(run_id, campaign)]
    section['openRunSet'] = 0
    section['workspaceSettings'] = {'shouldAutoGeneratePanels':False}
    bank = _bank(result)
    bank['sections'] = discovery_sections(campaign, chart_id)
    bank['panelPlacementOverrides'] = {}
    return result


def _saved_installed(spec, run_id, campaign=None, chart_id=None):
    return (_installed(spec, campaign, chart_id)
        and len(_bank(spec)['sections']) == len(discovery_sections(campaign, chart_id))
        and spec['section'].get('runSets') == [_campaign_runset(run_id, campaign)]
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


def ensure_discovery_saved_view(api, *, entity, project, receipt_dir, run_id, campaign=None):
    """Create one campaign-filtered saved view, never update existing views.

    The deterministic name reconciles an uncertain create response on retry.
    A name collision or user-edited saved view fails visibly instead of being
    overwritten. Readback verifies the selected run, panels and preservation of
    every prior view; browser rendering requires a separate authenticated check.
    """
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ResultsLayoutError('A valid explicit publication run ID is required.')
    root = Path(receipt_dir)
    # Internal hyphens in the saved-view slug stall the authenticated W&B UI
    # before panels render. Keep the native slug alphanumeric, preserving the
    # full publication ID in the run filter and receipts.
    suffix = (run_id if re.fullmatch(r'[A-Za-z0-9]+', run_id)
              else 'h' + hashlib.sha256(run_id.encode('utf-8')).hexdigest())
    publication = (campaign or {}).get('publication', {})
    prefix = publication.get('slug_prefix', 'transfer575')
    if not re.fullmatch(r'[A-Za-z0-9]+', prefix):
        raise ResultsLayoutError('Campaign view prefix must be alphanumeric.')
    chart_id = ensure_campaign_chart(api, entity=entity, campaign=campaign) if publication.get('arm_styles') else None
    name = 'nw-' + prefix + suffix + '-v'
    display_name = publication.get('view_title', '575K transfer discovery · H1/2/3 · J sweep')
    url = f'https://wandb.ai/{entity}/{project}?nw={name[3:-2]}'
    receipt = dict(schema_version=1, layout_version=LAYOUT_VERSION, entity=entity,
        project=project, view_name=name, view_type='project-view', run_id=run_id,
        layout_scope='dedicated campaign-filtered saved project workspace',
        url=url, workspace_url=url,
        run_url=f'https://wandb.ai/{entity}/{project}/runs/{run_id}?nw={name[3:-2]}',
        owned_section_ids=[s['__id__'] for s in discovery_sections(campaign, chart_id)],
        expected_chart_keys=expected_campaign_chart_keys(campaign) if campaign else list(CHART_KEYS), custom_chart_id=chart_id,
        expected_table_keys=list(TABLE_KEYS) + (['discovery/diagnostics'] if (campaign or {}).get('diagnostics', {}).get('enabled') else []),
        verification='saved workspace schema read back; browser rendering must be checked separately')
    views = _views(api, entity, project)
    matches = [v for v in views if v['name'] == name]
    if matches:
        if (len(matches) != 1 or matches[0]['type'] != 'project-view'
                or not _saved_installed(_spec(matches[0]), run_id, campaign, chart_id)):
            raise ResultsLayoutError('Existing discovery saved view differs; preserve it and inspect before retrying.')
        receipt.update(status='verified', changed=False, view_id=matches[0]['id'],
                       after_sha256=_hash(_spec(matches[0])))
        _saved_receipt_write(root/'results-layout-receipt.json', receipt)
        return receipt
    proposed = _saved_spec(_spec(_selected(views, DEFAULT_VIEW_NAME)), run_id, campaign, chart_id)
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
