"""Isolated, idempotent checkpoint comparison view with explicit curve colors."""
from copy import deepcopy
import json
from pathlib import Path
import re

from utils.wandb_results_layout import (_views,_selected,_spec,_bank,_hash,_write,
    _execute,_panel,ResultsLayoutError,DEFAULT_VIEW_NAME)
from utils.wandb_transfer_discovery_layout import _CREATE_VIEW
from utils.transfer_checkpoint_publication import settings

VERSION='transfer-checkpoint-curves-v1'
TABLE_KEY='transfer_curves/points'
PROGRESS_KEY='transfer_curves/progress'
_QUERY='query TransferCurveChart($id:ID!){customChart(id:$id){id name type spec}}'
_CREATE='''mutation CreateTransferCurveChart($entity:String!,$name:String!,$displayName:String!,
$type:String!,$access:String!,$spec:JSONString!){createCustomChart(input:{entity:$entity,
name:$name,displayName:$displayName,type:$type,access:$access,spec:$spec}){chart{id name type spec}}}'''


def chart_definition(campaign):
    styles=settings(campaign)
    return {'$schema':'https://vega.github.io/schema/vega-lite/v5.json',
        'data':{'name':'wandb'},'width':'container','height':360,
        'autosize':{'type':'fit','contains':'padding'},
        'title':{'text':'${string:title}','anchor':'start','subtitle':'Mean ± sample SD across three paired episodes; prior in black'},
        'transform':[{'filter':"isValid(datum['${field:value}']) && isNumber(datum['${field:value}']) && isFinite(datum['${field:value}'])"}],
        'encoding':{'x':{'field':'${field:step}','type':'quantitative','title':'Checkpoint training decisions','scale':{'zero':False}},
            'y':{'field':'${field:value}','type':'quantitative','title':'${string:ytitle}','scale':{'zero':True}},
            'color':{'field':'${field:label}','type':'nominal','title':None,
                'scale':{'domain':[row['label'] for row in styles],'range':[row['color'] for row in styles]},
                'legend':{'orient':'bottom','columns':2,'labelLimit':270,'symbolStrokeWidth':3}},
            'detail':[{'field':'${field:setting}','type':'nominal'},{'field':'${field:segment}','type':'nominal'}],
            'order':{'field':'${field:step}','type':'quantitative'},
            'tooltip':[{'field':'${field:label}','type':'nominal','title':'Setting'},
                {'field':'${field:step}','type':'quantitative','title':'Checkpoint','format':',d'},
                {'field':'${field:value}','type':'quantitative','title':'Mean','format':'.4~g'},
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


def chart_id(campaign,entity):
    return entity+'/transfer_checkpoint_'+_hash(chart_definition(campaign))[:16]


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


def runset(publication_id):
    return dict(id='rs//Subsection 1',name='Transfer checkpoint comparison',enabled=True,
        runFeed=dict(version=2,columnVisible={},columnPinned={},columnWidths={},columnOrder=[],pageSize=50,onlyShowSelected=False),
        search={'query':''},searchHistory=[],grouping=[],
        filters={'filterFormat':'filterV2','filters':[{'key':{'section':'config','name':'transfer_curve_campaign'},
            'op':'=','value':publication_id,'disabled':False}]},
        sort={'keys':[{'key':{'section':'config','name':'transfer_curve_order'},'ascending':True}]},
        selections={'root':1,'bounds':[],'tree':[]},expandedRowAddresses=[])


def sections(campaign,entity):
    styles=settings(campaign)
    intro=('### Transfer across the checkpoint bank\n\n'
        f"**{len(campaign['checkpoints'])} checkpoints · {len(styles)-1} transfer settings and the matched prior.** "
        'Three paired environment seeds (101–103), controller seed 55, full 500-decision mean-action episodes. '
        'The x-axis is training decisions of the frozen checkpoint. Bands are episode sample SD, not confidence intervals. '
        'Gains subtract the same checkpoint\'s prior return separately for each seed before computing mean/SD. '
        'Late controller time includes transfer bookkeeping and excludes decisions 0–9. Historical prior late timing is unavailable. '
        '**Pending/failed points remain visible in progress; missing measurements are null and never plotted as zero.** '
        'One run per setting; use the run list to hide/show curves. All colors are fixed across panels. '
        'This is a frozen-checkpoint screen, not a measurement of online training speedup.')
    panels=[_panel('curve-intro','Markdown Panel',{'value':intro},width=24,height=5),
        _panel('curve-progress','Media Browser',{'chartTitle':'Checkpoint progress, including pending and failed evaluations',
            'mediaKeys':[PROGRESS_KEY]},width=24,height=9)]
    curves=[]
    for key,title,ytitle in (('return','Full-episode return','Episode return'),
                            ('gain','Paired gain over the checkpoint prior','Paired return gain'),
                            ('late','Late controller time','Seconds per real decision')):
        value='late_seconds' if key=='late' else key+'_mean'
        curves.append(_panel('curve-'+key,'Vega2',{
            'transform':{'name':'tableWithLeafColNames'},
            'userQuery':{'queryFields':[{'name':'runSets','args':[{'name':'runSets','value':'${runSets}'},{'name':'limit','value':500.}],
                'fields':[{'name':'summaryTable','args':[{'name':'tableKey','value':TABLE_KEY}],'fields':[]},
                          {'name':'id','value':[]},{'name':'name','value':[]}]}]},
            'panelDefId':chart_id(campaign,entity),
            'fieldSettings':dict(step='step',setting='setting',label='label',segment='segment',value=value,
                sd=key+'_std',lower=key+'_lower',upper=key+'_upper'),
            'stringSettings':dict(title=title,ytitle=ytitle)},width=8,height=9))
    tables=[_panel('curve-values','Media Browser',{'chartTitle':'Exact completed means, episode SDs, gains and timing',
        'mediaKeys':[TABLE_KEY]},width=24,height=9)]
    result=[]
    for suffix,name,items,columns in [('progress','Transfer curves | progress',panels,1),
            ('curves','Transfer curves | return, gain and computation',curves,3),
            ('values','Transfer curves | measurements',tables,1)]:
        for index,panel in enumerate(items): panel['__id__']=f'ambi-{VERSION}-{suffix}-{index}'
        result.append(dict(__id__=f'ambi-{VERSION}-{suffix}',name=name,isOpen=True,type='flow',
            flowConfig=dict(snapToColumns=True,columnsPerPage=columns,rowsPerPage=2 if suffix=='progress' else 1,
                gutterWidth=16,boxWidth=540,boxHeight=500),sorted=0,pinned=True,isPanelsAuto=False,panels=items))
    return result


def saved_spec(template,campaign,entity,publication_id):
    result=deepcopy(template)
    result['section'].update(runSets=[runset(publication_id)],openRunSet=0,
        workspaceSettings={'shouldAutoGeneratePanels':False})
    bank=_bank(result);bank['sections']=sections(campaign,entity);bank['panelPlacementOverrides']={}
    return result


def _installed(spec,campaign,entity,publication_id):
    expected=sections(campaign,entity); actual=deepcopy(_bank(spec)['sections'])
    if len(actual)!=len(expected) or spec['section'].get('runSets')!=[runset(publication_id)]: return False
    for section,wanted in zip(actual,expected):
        section.setdefault('type',wanted['type'])
        for key,value in wanted['flowConfig'].items(): section.setdefault('flowConfig',{}).setdefault(key,value)
        if len(section.get('panels',[]))!=len(wanted['panels']):return False
        for panel,target in zip(section['panels'],wanted['panels']):panel.setdefault('layout',target['layout'])
    return actual==expected and not _bank(spec).get('panelPlacementOverrides')


def ensure_saved_view(api,*,campaign,entity,project,publication_id,receipt_dir):
    """Never overwrite existing views; mismatching edits fail with a saved receipt."""
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
            if len(matches)!=1 or matches[0]['type']!='project-view' or not _installed(_spec(matches[0]),campaign,entity,publication_id):
                raise ResultsLayoutError('Existing comparison view differs; preserving user edits.')
            receipt.update(status='verified',changed=False,view_id=matches[0]['id'])
            _write(root/'results-layout-receipt.json',receipt);return receipt
        proposed=saved_spec(_spec(_selected(views,DEFAULT_VIEW_NAME)),campaign,entity,publication_id)
        _write(root/'views-before.json',views);_write(root/'view-proposed.json',proposed)
        if _views(api,entity,project)!=views:raise ResultsLayoutError('Views changed during preparation; retry from fresh state.')
        _write(root/'results-layout-intent.json',receipt)
        mutation_error=None
        try:
            _execute(api,_CREATE_VIEW,dict(entityName=entity,projectName=project,type='project-view',
                name=name,displayName='Transfer mechanisms across checkpoints',spec=json.dumps(proposed,separators=(',',':'))))
        except Exception as exc:mutation_error=type(exc).__name__
        after=_views(api,entity,project);_write(root/'views-after.json',after)
        matches=[view for view in after if view['name']==name and view['type']=='project-view']
        remaining={view['id']:view for view in after}
        if len(matches)!=1 or _spec(matches[0])!=proposed or any(remaining.get(view['id'])!=view for view in views):
            raise ResultsLayoutError('Workspace readback or existing-view preservation failed.')
        receipt.update(status='verified',changed=True,view_id=matches[0]['id'],preserved_views=len(views),
            uncertain_response_reconciled=mutation_error is not None)
        _write(root/'results-layout-receipt.json',receipt);return receipt
    except Exception as exc:
        _write(root/'results-layout-receipt.json',dict(receipt,status='failed',error=f'{type(exc).__name__}: {exc}'))
        raise
