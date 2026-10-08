"""Isolated, idempotent checkpoint comparison view with explicit curve colors."""
from copy import deepcopy
import json
from pathlib import Path
import re

from utils.wandb_results_layout import (_views,_selected,_spec,_bank,_hash,_write,
    _execute,_panel,_MUTATION,ResultsLayoutError,DEFAULT_VIEW_NAME)
from utils.wandb_transfer_discovery_layout import _CREATE_VIEW
from utils.transfer_checkpoint_publication import settings

VERSION='transfer-checkpoint-curves-v3'
LEGACY_VERSION='transfer-checkpoint-curves-v2'
TABLE_KEY='transfer_curves/points'
PROGRESS_KEY='transfer_curves/progress'
_QUERY='query TransferCurveChart($id:ID!){customChart(id:$id){id name type spec}}'
_CREATE='''mutation CreateTransferCurveChart($entity:String!,$name:String!,$displayName:String!,
$type:String!,$access:String!,$spec:JSONString!){createCustomChart(input:{entity:$entity,
name:$name,displayName:$displayName,type:$type,access:$access,spec:$spec}){chart{id name type spec}}}'''


def chart_definition(campaign, *, legacy=False):
    styles=settings(campaign)
    definition = {'$schema':'https://vega.github.io/schema/vega-lite/v5.json',
        'data':{'name':'wandb'},'width':'container','height':360,
        'autosize':{'type':'fit','contains':'padding'},
        'title':{'text':'${string:title}','anchor':'start','subtitle':'${string:subtitle}'},
        'transform':[{'filter':"isValid(datum['${field:value}']) && isNumber(datum['${field:value}']) && isFinite(datum['${field:value}'])"}],
        'encoding':{'x':{'field':'${field:step}','type':'quantitative','title':'Checkpoint training decisions','scale':{'zero':False}},
            'y':{'field':'${field:value}','type':'quantitative','title':'${string:ytitle}','scale':{'zero':True}},
            'color':{'field':'${field:label}','type':'nominal','title':None,
                'scale':{'domain':[row['label'] for row in styles],'range':[row['color'] for row in styles]},
                'legend':{'orient':'bottom','columns':2,'labelLimit':270,'symbolStrokeWidth':3}},
            'strokeDash':{'field':'${field:setting}','type':'nominal','legend':None,
                'scale':{'domain':[row['setting_id'] for row in styles],
                         'range':[[6,3] if row.get('role')=='fresh' else [1,0] for row in styles]}},
            'detail':[{'field':'${field:setting}','type':'nominal'},{'field':'${field:segment}','type':'nominal'}],
            'order':{'field':'${field:step}','type':'quantitative'},
            'tooltip':[{'field':'${field:label}','type':'nominal','title':'Setting'},
                {'field':'${field:step}','type':'quantitative','title':'Checkpoint','format':',d'},
                {'field':'${field:value}','type':'quantitative','title':'${string:statistic}','format':'.4~g'},
                {'field':'${field:sd}','type':'quantitative','title':'Episode sample SD','format':'.4~g'}]},
        'layer':[
            {'mark':{'type':'area','opacity':.1},'encoding':{
                'y':{'field':'${field:lower}','type':'quantitative'},'y2':{'field':'${field:upper}'}}},
            {'mark':{'type':'rule','opacity':.3,'strokeWidth':1},'encoding':{
                'y':{'field':'${field:lower}','type':'quantitative'},'y2':{'field':'${field:upper}'}}},
            {'mark':{'type':'line','strokeWidth':2.4}},
            {'mark':{'type':'point','filled':True,'size':48}},
            {'transform':[{'filter':"datum['${field:setting}'] === 'prior'"}],
             'mark':{'type':'line','strokeWidth':3.5},'encoding':{'color':{'value':'#000000'}}}],
        'config':{'view':{'stroke':None},'axis':{'gridColor':'#e8edf2'},
                  'legend':{'labelFontSize':11,'rowPadding':4}}}
    if not legacy:
        definition['params']=[{'name':'selected_curves','select':{'type':'point','fields':['${field:label}']},'bind':'legend'}]
        definition['encoding']['opacity']={'condition':{'param':'selected_curves','value':1},'value':.12}
    return definition


def chart_id(campaign,entity, *, legacy=False):
    return entity+'/transfer_checkpoint_'+_hash(chart_definition(campaign,legacy=legacy))[:16]


def ensure_chart(api,campaign,entity):
    """Register a content-addressed private definition once; reconcile uncertainty."""
    ident=chart_id(campaign,entity); expected=chart_definition(campaign)
    chart=_execute(api,_QUERY,{'id':ident}).get('customChart')
    error=None
    if chart is None:
        try:
            _execute(api,_CREATE,dict(entity=entity,name=ident.split('/',1)[1],
                displayName='Transfer checkpoint curves · mean and episode SD',type='vega2',
                access='PRIVATE',spec=json.dumps(expected,separators=(',',':'))))
        except Exception as exc:
            error=exc
        chart=_execute(api,_QUERY,{'id':ident}).get('customChart')
    actual=(json.loads(chart['spec']) if isinstance(chart.get('spec'),str) else chart.get('spec')) if chart else None
    if not chart or chart.get('type')!='vega2' or actual!=expected:
        raise ResultsLayoutError('Required transfer checkpoint chart could not be verified: '+ident) from error
    return ident


def runset(publication_id, *, legacy=False):
    return dict(id='rs//Subsection 1',name='Transfer checkpoint comparison',enabled=True,
        runFeed=dict(version=2,columnVisible={},columnPinned={},columnWidths={},columnOrder=[],pageSize=50,onlyShowSelected=False),
        search={'query':''},searchHistory=[],grouping=[],
        filters={'filterFormat':'filterV2','filters':[{'key':{'section':'config','name':'transfer_curve_campaign' if legacy else 'transfer_curve_overview'},
            'op':'=','value':publication_id,'disabled':False}]},
        sort={'keys':[{'key':{'section':'config','name':'transfer_curve_order'},'ascending':True}]},
        selections={'root':1,'bounds':[],'tree':[]},expandedRowAddresses=[])


def sections(campaign,entity, *, legacy=False):
    styles=settings(campaign)
    intro=('### Transfer across the checkpoint bank\n\n'
        f"**{len(campaign['checkpoints'])} checkpoints · {sum(row.get('role')=='transfer' for row in styles)} transfer settings, "
        f"{sum(row.get('role')=='fresh' for row in styles)} fresh SAC controls and the matched base actor.** "
        'Five paired environment seeds (101–105), controller seed 55, full 500-decision mean-action episodes. '
        'The x-axis is training decisions of the frozen checkpoint. Bands are episode sample SD, not confidence intervals. '
        'Gains subtract the same checkpoint\'s prior return separately for each seed before computing mean/SD. '
        'Fresh-control gains additionally subtract fresh SAC at the same checkpoint, H and J, paired by seed; they stay pending until both settings finish. '
        'Full controller time includes initialization and transfer bookkeeping; steady time excludes decisions 0–9. Both exclude sampled diagnostics. Historical prior timing is unavailable. '
        '**Pending/failed points remain visible in progress; missing measurements are null and never plotted as zero.** '
        'One run per setting; use the run list to hide/show curves. All colors are fixed across panels. '
        'This is a frozen-checkpoint screen, not a measurement of online training speedup.')
    if not legacy:
        intro=intro.replace('One run per setting; use the run list to hide/show curves. All colors are fixed across panels. ',
            'The live overview displays six curves with fixed colors; click legend entries to emphasize selected curves. ')
    panels=[_panel('curve-intro','Markdown Panel',{'value':intro},width=24,height=5),
        _panel('curve-progress','Media Browser',{'chartTitle':'Checkpoint progress, including pending and failed evaluations',
            'mediaKeys':[PROGRESS_KEY]},width=24,height=9)]
    curves=[]
    for key,title,ytitle in (('return','Full-episode return','Episode return'),
                            ('gain','Paired improvement over base actor','Paired return gain'),
                            ('fresh_gain','Paired improvement over matched fresh SAC','Paired return gain'),
                            ('return_std','Return variability across five seeds','Episode sample SD'),
                            ('return_min','Lowest observed seed return','Lowest episode return'),
                            ('control','Full controller time, including initialization','Seconds per real decision'),
                            ('late','Steady controller time, decisions 10–499','Seconds per real decision')):
        scalar = key in ('return_std','return_min')
        value = key if scalar else key+'_seconds' if key in ('late','control') else key+'_mean'
        curves.append(_panel('curve-'+key,'Vega2',{
            'transform':{'name':'tableWithLeafColNames'},
            'userQuery':{'queryFields':[{'name':'runSets','args':[{'name':'runSets','value':'${runSets}'},{'name':'limit','value':500.}],
                'fields':[{'name':'summaryTable','args':[{'name':'tableKey','value':TABLE_KEY}],'fields':[]},
                          {'name':'id','value':[]},{'name':'name','value':[]}]}]},
            'panelDefId':chart_id(campaign,entity,legacy=legacy),
            'fieldSettings':dict(step='step',setting='setting',label='label',
                segment='fresh_segment' if key=='fresh_gain' else 'segment',value=value,
                sd='interval_not_applicable' if scalar else key+'_std',
                lower='interval_not_applicable' if scalar else key+'_lower',
                upper='interval_not_applicable' if scalar else key+'_upper'),
            'stringSettings':dict(title=title,ytitle=ytitle,statistic=ytitle if scalar else 'Mean',
                subtitle=('Five paired episodes; base actor in black; fresh controls dashed' if scalar else
                          'Mean ± episode sample SD; five paired seeds; base actor in black'))},width=12,height=9))
    tables=[_panel('curve-values','Media Browser',{'chartTitle':'Exact completed means, episode SDs, gains and timing',
        'mediaKeys':[TABLE_KEY]},width=24,height=9)]
    result=[]
    version=LEGACY_VERSION if legacy else VERSION
    for suffix,name,items,columns in [('progress','Transfer curves | progress',panels,1),
            ('curves','Transfer curves | performance, variance and computation',curves,2),
            ('values','Transfer curves | measurements',tables,1)]:
        for index,panel in enumerate(items): panel['__id__']=f'ambi-{version}-{suffix}-{index}'
        result.append(dict(__id__=f'ambi-{version}-{suffix}',name=name,isOpen=True,type='flow',
            flowConfig=dict(snapToColumns=True,columnsPerPage=columns,rowsPerPage=4 if suffix=='curves' else 2 if suffix=='progress' else 1,
                gutterWidth=16,boxWidth=540,boxHeight=500),sorted=0,pinned=True,isPanelsAuto=False,panels=items))
    return result


def saved_spec(template,campaign,entity,publication_id, *, legacy=False):
    result=deepcopy(template)
    result['section'].update(runSets=[runset(publication_id,legacy=legacy)],openRunSet=0,
        workspaceSettings={'shouldAutoGeneratePanels':False})
    bank=_bank(result);bank['sections']=sections(campaign,entity,legacy=legacy);bank['panelPlacementOverrides']={}
    return result


def _installed(spec,campaign,entity,publication_id, *, legacy=False):
    expected=sections(campaign,entity,legacy=legacy); actual=deepcopy(_bank(spec)['sections'])
    if len(actual)!=len(expected) or spec['section'].get('runSets')!=[runset(publication_id,legacy=legacy)]: return False
    for section,wanted in zip(actual,expected):
        section.setdefault('type',wanted['type'])
        for key,value in wanted['flowConfig'].items(): section.setdefault('flowConfig',{}).setdefault(key,value)
        if len(section.get('panels',[]))!=len(wanted['panels']):return False
        for panel,target in zip(section['panels'],wanted['panels']):panel.setdefault('layout',target['layout'])
    return actual==expected and not _bank(spec).get('panelPlacementOverrides')


def ensure_saved_view(api,*,campaign,entity,project,publication_id,receipt_dir):
    """Create a view or upgrade our exact v2; preserve unknown edits and other views."""
    if not re.fullmatch(r'[A-Za-z0-9]+',publication_id):
        raise ResultsLayoutError('Publication ID must be alphanumeric for the W&B saved-view URL.')
    root=Path(receipt_dir);root.mkdir(parents=True,exist_ok=True)
    name='nw-transfercurves'+publication_id+'-v'
    url=f'https://wandb.ai/{entity}/{project}?nw={name[3:-2]}'
    receipt=dict(layout_version=VERSION,view_name=name,url=url,browser_verified=False)
    try:
        receipt['chart_id']=ensure_chart(api,campaign,entity)
        views=_views(api,entity,project)
        matches=[view for view in views if view['name']==name]
        if matches:
            if len(matches)!=1 or matches[0]['type']!='project-view':
                raise ResultsLayoutError('Existing comparison view differs; preserving user edits.')
            before=matches[0]
            if _installed(_spec(before),campaign,entity,publication_id):
                receipt.update(status='verified',changed=False,view_id=before['id'])
                _write(root/'results-layout-receipt.json',receipt);return receipt
            if not _installed(_spec(before),campaign,entity,publication_id,legacy=True):
                raise ResultsLayoutError('Existing comparison view differs; preserving user edits.')
            proposed=saved_spec(_spec(before),campaign,entity,publication_id)
            receipt['upgraded_from']=LEGACY_VERSION
        else:
            before=None
            proposed=saved_spec(_spec(_selected(views,DEFAULT_VIEW_NAME)),campaign,entity,publication_id)
        _write(root/'views-before.json',views);_write(root/'view-proposed.json',proposed)
        if _views(api,entity,project)!=views:raise ResultsLayoutError('Views changed during preparation; retry from fresh state.')
        _write(root/'results-layout-intent.json',receipt)
        mutation_error=None
        try:
            if before is None:
                _execute(api,_CREATE_VIEW,dict(entityName=entity,projectName=project,type='project-view',
                    name=name,displayName='Transfer mechanisms across checkpoints',spec=json.dumps(proposed,separators=(',',':'))))
            else:
                _execute(api,_MUTATION,dict(id=before['id'],type=before['type'],name=before['name'],
                    displayName=before['displayName'],spec=json.dumps(proposed,separators=(',',':'))))
        except Exception as exc:mutation_error=type(exc).__name__
        after=_views(api,entity,project);_write(root/'views-after.json',after)
        matches=[view for view in after if view['name']==name and view['type']=='project-view']
        remaining={view['id']:view for view in after}
        untouched=[view for view in views if before is None or view['id']!=before['id']]
        if (len(matches)!=1 or _spec(matches[0])!=proposed
                or any(remaining.get(view['id'])!=view for view in untouched)
                or (before is not None and {key:value for key,value in matches[0].items() if key!='spec'} !=
                    {key:value for key,value in before.items() if key!='spec'})):
            raise ResultsLayoutError('Workspace readback or existing-view preservation failed.')
        receipt.update(status='verified',changed=True,view_id=matches[0]['id'],preserved_views=len(untouched),
            uncertain_response_reconciled=mutation_error is not None)
        _write(root/'results-layout-receipt.json',receipt);return receipt
    except Exception as exc:
        _write(root/'results-layout-receipt.json',dict(receipt,status='failed',error=f'{type(exc).__name__}: {exc}'))
        raise
