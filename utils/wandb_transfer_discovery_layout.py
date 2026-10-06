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


def spectral_arm_metadata(campaign, arm):
    """Describe the declared transfer operator without inferring measured rank."""
    specification = campaign.get('arms', {}).get(arm)
    if specification is None:
        if arm == 'rho_a0_c0':
            specification = {'actor_rho': 0., 'critic_rho': 0., 'parameter_scope': 'matrices'}
        else:
            raise ResultsLayoutError('Spectral publication requires the prepared arms mapping: ' + arm)
    active = []
    for component in ('actor', 'critic'):
        spectral = specification.get(component + '_spectral')
        if spectral is not None:
            method = spectral['method']
            if method not in ('svd', 'activation', 'gradient'):
                raise ResultsLayoutError('Unknown spectral method: ' + method)
            active.append(dict(component=component, method=method, requested_rank=spectral['rank'],
                strength=spectral['strength'], norm_matched=spectral['norm_matched']))
        elif specification.get(component + '_bernoulli_p', 0.) > 0:
            active.append(dict(component=component, method='bernoulli', requested_rank=None,
                strength=specification[component + '_bernoulli_p'], norm_matched=False))
        elif specification.get(component + '_rho', 0.) > 0:
            active.append(dict(component=component, method='blend' if specification[component + '_rho'] < 1 else 'carry',
                requested_rank=None, strength=specification[component + '_rho'], norm_matched=False))
    if not active:
        return dict(component='fresh', method='fresh', requested_rank=None, strength=0.,
            norm_matched=False, parameter_scope=specification.get('parameter_scope', 'matrices'),
            label='Fresh prior reset', color='#000000')
    component = active[0]['component'] if len(active) == 1 else 'joint'
    names = dict(svd='SVD', activation='Activation weighted', gradient='Gradient ranked',
                 bernoulli='Bernoulli', blend='Dense blend', carry='Dense carry')
    def describe(item):
        if item['requested_rank'] is not None:
            mode = 'Dense norm match to ' if item['norm_matched'] else ''
            return f"{mode}{names[item['method']]} r{item['requested_rank']} s{item['strength']:g}"
        return f"{names[item['method']]} {100 * item['strength']:g}%"
    shared = {key: active[0][key] if all(item[key] == active[0][key] for item in active) else None
              for key in ('method', 'requested_rank', 'strength', 'norm_matched')}
    label = describe(active[0]) if all(describe(item) == describe(active[0]) for item in active) else ' / '.join(
        item['component'] + ': ' + describe(item) for item in active)
    colors = {'svd': ('#0072b2', '#56b4e9'), 'activation': ('#d55e00', '#e69f00'),
              'gradient': ('#009e73', '#7fcdbb'), 'bernoulli': ('#cc79a7', '#cc79a7'),
              'blend': ('#777777', '#777777'), 'carry': ('#333333', '#333333')}
    method = shared['method'] or 'mixed'
    color = colors.get(method, ('#8c564b', '#b4948f'))[int(bool(shared['norm_matched']))]
    return dict(component=component, **{**shared, 'method': method},
        parameter_scope=specification.get('parameter_scope', 'matrices'), label=label, color=color)


def _spectral_groups(campaign):
    arms = list(dict.fromkeys(c['arm'] for c in campaign['cells']))
    if campaign.get('historical_reference') and 'rho_a0_c0' not in arms:
        arms.append('rho_a0_c0')
    metadata = {arm: spectral_arm_metadata(campaign, arm) for arm in arms}
    fresh = [arm for arm in arms if metadata[arm]['component'] == 'fresh']
    return [(component+'/', component.capitalize()+' | ',
             [arm for arm in arms if metadata[arm]['component'] == component] + fresh)
            for component in ('actor', 'critic', 'joint')
            if any(value['component'] == component for value in metadata.values())]


def campaign_diagnostic_charts(campaign, prefix=''):
    if campaign.get('family') != 'spectral_transfer':
        return DIAGNOSTIC_CHARTS if campaign.get('diagnostics', {}).get('enabled') else ()
    component = prefix.rstrip('/')
    components = ('actor', 'critic') if component == 'joint' else (component,)
    charts = []
    for name in components:
        for stage, title in (('prior', 'Prior'), ('initial', 'Initial'), ('final', 'Post-J')):
            charts.append((f'spectral_{stage}_{name}_loss', f'{name.capitalize()} {title.lower()} heldout loss; fixed-model proxy'))
        for suffix, title in (
            ('initial_loss_gain_vs_prior', 'Initial heldout loss reduction versus prior; fixed-model proxy'),
            ('post_j_loss_gain_vs_prior', 'Post-J heldout loss reduction versus prior; fixed-model proxy'),
            ('transferred_energy_ratio', 'Transfer / donor squared norm; may exceed one'),
            ('initial_squared_norm', 'Transferred matrix-delta squared norm'),
            ('mean_rank_90', 'Donor rank capturing 90% spectral energy; layer mean'),
            ('mean_effective_rank', 'Donor effective spectral rank; layer mean'),
            ('mean_positive_benefit_energy_fraction', 'Donor energy with positive first-order benefit; fixed-model proxy'),
            ('initial_first_order_benefit', 'Transferred first-order benefit; fixed-model proxy')):
            charts.append((f'spectral_{name}_{suffix}', name.capitalize()+': '+title))
    charts.extend((('spectral_probe_seconds_per_decision', 'Scoring probe seconds per decision; included in controller'),
                   ('spectral_filter_seconds_per_decision', 'Spectral filter seconds per decision; included in controller')))
    return tuple(charts)


def campaign_chart_definition(campaign, component=None):
    """Map color to the mechanism, not to the single publication run."""
    publication = campaign.get('publication', {})
    spectral = campaign.get('family') == 'spectral_transfer'
    if spectral:
        selected = [arm for prefix, _, arms in _spectral_groups(campaign)
                    if component is None or prefix == component+'/' for arm in arms]
        styles = [[arm, spectral_arm_metadata(campaign, arm)['label'], spectral_arm_metadata(campaign, arm)['color']]
                  for arm in dict.fromkeys(selected)]
    else:
        styles = [row for row in publication['arm_styles'] if component is None
                  or publication.get('arm_components', {}).get(row[0]) == component]
    if campaign.get('historical_reference') and not (spectral and any(row[0] == 'rho_a0_c0' for row in styles)):
        styles.append(['rho_a0_c0', 'Fresh prior reset (historical)', '#000000'])
    if (len({row[0] for row in styles}) != len(styles)
            or (not spectral and len({row[2] for row in styles}) != len(styles))):
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

    if spectral:
        forms = {arm: ('Dense norm-matched control' if spectral_arm_metadata(campaign, arm)['norm_matched']
                      else 'Filtered transfer' if spectral_arm_metadata(campaign, arm)['method'] in ('svd','activation','gradient')
                      else 'Weight-copy control') for arm, _, _ in styles}
        definition['transform'].append({'calculate':json.dumps(forms, separators=(',', ':'))
            + "[datum['${field:lineKey}']]", 'as':'Transfer form'})
        definition['encoding']['strokeDash'] = {'field':'Transfer form', 'type':'nominal',
            'scale':{'domain':['Filtered transfer','Dense norm-matched control','Weight-copy control'],
                     'range':[[],[8,3],[2,3]]}, 'legend':None}
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
    if campaign.get('family') == 'spectral_transfer' and component is None:
        return {prefix.rstrip('/'): campaign_chart_id(entity, campaign, prefix.rstrip('/'))
                for prefix, _, _ in _spectral_groups(campaign)}
    if campaign.get('publication', {}).get('probability_sweep') and component is None:
        return {component: campaign_chart_id(entity, campaign, component) for component in ('actor', 'critic', 'joint')}
    family = 'spectral_transfer_' if campaign.get('family') == 'spectral_transfer' else 'bernoulli_transfer_'
    return entity + '/' + family + _hash(campaign_chart_definition(campaign, component))[:16]


def campaign_curve_groups(campaign):
    if campaign.get('family') == 'spectral_transfer':
        return _spectral_groups(campaign)
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
        keys.extend(f'discovery/{prefix}h{h}_{metric}_vs_j' for metric, _ in campaign_diagnostic_charts(campaign, prefix) for h in campaign['H'])
    return keys


def ensure_campaign_chart(api, *, entity, campaign, component=None):
    """Create an immutable content-addressed chart, reconcile, and verify it."""
    if campaign.get('family') == 'spectral_transfer' and component is None:
        return {prefix.rstrip('/'): ensure_campaign_chart(api, entity=entity, campaign=campaign, component=prefix.rstrip('/'))
                for prefix, _, _ in _spectral_groups(campaign)}
    if campaign.get('publication', {}).get('probability_sweep') and component is None:
        return {component: ensure_campaign_chart(api, entity=entity, campaign=campaign, component=component)
                for component in ('actor', 'critic', 'joint')}
    identifier = campaign_chart_id(entity, campaign, component)
    expected = campaign_chart_definition(campaign, component)
    chart = _execute(api, _CHART_QUERY, {'id':identifier}).get('customChart')
    mutation_error = None
    if chart is None:
        try:
            _execute(api, _CREATE_CHART, dict(entity=entity, name=identifier.split('/', 1)[1],
                displayName=('Spectral transfer methods and fixed-model proxies' if campaign.get('family') == 'spectral_transfer'
                             else 'Bernoulli transfer mechanisms and diagnostics'), type='vega2', access='PRIVATE',
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
        fresh_rounds = {record['cell']['J'] for item in campaign.get('historical_references', [reference])
                        if item.get('kind', 'fresh') == 'fresh' for record in item.get('records', [])}
        missing_fresh_rounds = new_rounds - fresh_rounds
        if missing_fresh_rounds:
            intro += ('**No fresh-prior controls are available at J='
                + ','.join(map(str, sorted(missing_fresh_rounds)))
                + '; paired gains at these rounds remain unavailable.** '
                'The black curve stops at the available historical controls. ')
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


def spectral_sections(campaign, chart_id):
    """Install pending result panels and explicit fixed-model proxy labels."""
    publication = campaign.get('publication', {})
    count = len(campaign['cells'])
    seed_text = ', '.join(map(str, campaign.get('seeds', []))) or 'the configured paired seeds'
    intro = ('### ' + publication.get('view_title', 'Spectral transfer') + '\n\n'
        f'**{count} configurations; seeds {seed_text}; up to {campaign.get("max_steps", 500)} real decisions per episode.** '
        'Panels separate actor, critic and joint transfer, with a separate plot for each horizon. '
        'Only weight matrices transfer; vector parameters restore their frozen prior. '
        'SVD is blue, activation-weighted selection orange, gradient-ranked selection green; '
        'lighter dashed curves are the corresponding **dense norm-matched controls**, not rank-constrained transfers. '
        'Bernoulli is purple, dense carry/blend gray, and fresh reset black. Labels state requested rank and strength. '
        '**Scoring-probe and spectral-filter time are controller work**, included in return-versus-controller-time curves. '
        'Their additional timing panels show components of that total and must not be added to it again. '
        'Heldout diagnostic time is reported separately. '
        '**Heldout losses, gains and first-order benefits are fixed-model proxies, not environment returns.** '
        'Actor loss is negative frozen-prior Q at the candidate mean action; critic loss is decoded-Q squared error '
        'against fixed prior-continuation model labels. Lower loss and positive loss reduction are favorable. '
        'Spectral energy concentration describes donor matrices; the transfer/donor squared-norm ratio can exceed one. '
        'Requested rank, donor effective rank and actual transfer norm describe different quantities. '
        '**Pending values are null, never zero. Partial episodes are progress only.** '
        'Curves require complete seed panels; tables retain episode uncertainty, coverage and provenance. '
        'Donor geometry excludes donorless first decisions; heldout objectives retain valid first-decision measurements. '
        'This screen is exploratory; model-proxy changes alone do not establish a causal explanation for return gains.')
    blocks = [('progress', 'Spectral transfer | protocol and live progress', 1, True, [
        _panel('intro','Markdown Panel',{'value':intro},width=24,height=10),
        _panel('settings','Media Browser',{'chartTitle':f'{count} settings and live progress','mediaKeys':['discovery/settings']},width=24,height=9)])]
    for prefix, label, _ in campaign_curve_groups(campaign):
        component = prefix.rstrip('/'); identifier = chart_id[component]
        blocks.append(('curves-'+component, 'Spectral transfer | '+label+'return and controller time', 3, True, [
            _campaign_curve(f'discovery/{prefix}h{h}_return_vs_{axis}', f'{label}H{h}: return versus '+('J' if axis=='j' else 'controller time'),
                'J rounds per solve' if axis=='j' else 'Controller seconds per decision; includes scoring and filtering',
                'Mean episode return', identifier)
            for axis in ('j','compute') for h in campaign['H']]))
        charts = campaign_diagnostic_charts(campaign, prefix)
        for suffix, title, visible, selected in (
            ('overhead', 'controller overhead', True, [(k,t) for k,t in charts if '_seconds_per_decision' in k]),
            ('proxies', 'heldout losses and transfer geometry; fixed-model proxies', False,
             [(k,t) for k,t in charts if '_seconds_per_decision' not in k])):
            blocks.append((suffix+'-'+component, 'Spectral transfer | '+label+title, 3, visible, [
                _campaign_curve(f'discovery/{prefix}h{h}_{metric}_vs_j', f'{label}H{h}: {description}',
                    'J rounds per solve', description, identifier)
                for metric, description in selected for h in campaign['H']]))
    blocks.append(('results', 'Spectral transfer | complete measurements and provenance', 1, True, [
        _panel('results','Media Browser',{'chartTitle':'Complete return, controller time, method and requested rank','mediaKeys':['discovery/results']},width=24,height=10),
        _panel('episodes','Media Browser',{'chartTitle':'Per-seed outcomes; scoring and filtering are included controller work','mediaKeys':['discovery/episodes']},width=24,height=8),
        _panel('diagnostics','Media Browser',{'chartTitle':'All numeric diagnostic summaries, episode SDs and per-metric root counts; fixed-model proxies','mediaKeys':['discovery/diagnostics']},width=24,height=12)]))
    sections = []
    for suffix, title, columns, visible, panels in blocks:
        identifier = 'ambi-spectral-transfer-v1-' + suffix
        for index, panel in enumerate(panels):
            panel['__id__'] = identifier + '-panel-' + str(index)
        sections.append(dict(__id__=identifier, name=title, isOpen=visible, type='flow',
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
    if campaign and campaign.get('family') == 'spectral_transfer':
        return spectral_sections(campaign, chart_id or campaign_chart_id('rwgao_b-brown-university', campaign))
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
    spectral = (campaign or {}).get('family') == 'spectral_transfer'
    prefix = publication.get('slug_prefix', 'spectral575' if spectral else 'transfer575')
    if not re.fullmatch(r'[A-Za-z0-9]+', prefix):
        raise ResultsLayoutError('Campaign view prefix must be alphanumeric.')
    chart_id = ensure_campaign_chart(api, entity=entity, campaign=campaign) if spectral or publication.get('arm_styles') else None
    name = 'nw-' + prefix + suffix + '-v'
    display_name = publication.get('view_title', 'Spectral transfer' if spectral else '575K transfer discovery · H1/2/3 · J sweep')
    url = f'https://wandb.ai/{entity}/{project}?nw={name[3:-2]}'
    receipt = dict(schema_version=1, layout_version=LAYOUT_VERSION, entity=entity,
        project=project, view_name=name, view_type='project-view', run_id=run_id,
        layout_scope='dedicated campaign-filtered saved project workspace',
        url=url, workspace_url=url,
        run_url=f'https://wandb.ai/{entity}/{project}/runs/{run_id}?nw={name[3:-2]}',
        owned_section_ids=[s['__id__'] for s in discovery_sections(campaign, chart_id)],
        expected_chart_keys=expected_campaign_chart_keys(campaign) if campaign else list(CHART_KEYS), custom_chart_id=chart_id,
        expected_table_keys=list(TABLE_KEYS) + (['discovery/diagnostics'] if spectral or (campaign or {}).get('diagnostics', {}).get('enabled') else []),
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
