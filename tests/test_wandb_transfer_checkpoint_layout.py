"""A visible comparison view is created separately and never silently overwritten."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from utils import wandb_transfer_checkpoint_layout as layout
from utils import transfer_checkpoint_publication as adapter


def campaign():
    return dict(checkpoints=[dict(step=25000),dict(step=50000)],candidates=[dict(setting_id=f's{i}',
        role='transfer' if i<4 else 'fresh',label=f'Setting {i}') for i in range(1,6)])


def comparisons():
    return [dict(setting_id='mppi_h3_return_q',label='MPPI H3 · Return Q',color='#cc79a7',role='comparison'),
            dict(setting_id='mppi_h3_soft_q',label='MPPI H3 · Soft Q',color='#56b4e9',role='comparison')]


def spec():
    return dict(section=dict(runSets=[{'preserve':'filters'}],settings={'preserve':'yes'},
        panelBankConfig=dict(sections=[dict(__id__='old',panels=[])],panelPlacementOverrides={'old':1})))


class Service:
    def __init__(self,*,timeout=False,apply=True):
        self.views=[dict(id='personal',name=layout.DEFAULT_VIEW_NAME,type='project-view',spec=json.dumps(spec()),displayName='Personal'),
                    dict(id='other',name='nw-other-v',type='project-view',spec=json.dumps(spec()),displayName='Other')]
        self.chart=None;self.chart_writes=0;self.view_writes=0;self.timeout=timeout;self.apply=apply
    def execute_graphql(self,query,variables):
        if 'CreateTransferCurveChart' in query:
            self.chart_writes+=1
            self.chart=dict(id=variables['entity']+'/'+variables['name'],name=variables['name'],type=variables['type'],spec=variables['spec'])
            return {'createCustomChart':{'chart':deepcopy(self.chart)}}
        if 'TransferCurveChart' in query:return {'customChart':deepcopy(self.chart)}
        if 'mutation ' in query:
            self.view_writes+=1
            if self.apply:
                if variables.get('id'):
                    row=next(row for row in self.views if row['id']==variables['id'])
                    row['spec']=variables['spec']
                else:
                    self.views.append(dict(id='new',name=variables['name'],type=variables['type'],spec=variables['spec'],displayName=variables['displayName']))
            if self.timeout:raise TimeoutError('response lost')
            return {'upsertView':{'view':{'id':'new'}}}
        return {'project':{'allViews':{'edges':[{'node':deepcopy(row)} for row in self.views]}}}


def install(tmp_path,service):
    return layout.ensure_saved_view(SimpleNamespace(_service_api=service),campaign=campaign(),entity='entity',
        project='ambi-inner-bench',publication_id='abc123',receipt_dir=tmp_path)


def test_layout_has_distinct_fixed_colors_bands_and_pending_progress():
    data=campaign();definition=layout.chart_definition(data)
    assert definition['encoding']['color']['scale']['range']==['#000000',*adapter.COLORS[:5]]
    assert len(set(definition['encoding']['color']['scale']['range']))==6
    assert definition['encoding']['strokeDash']['scale']['range']==[[1,0]]*4+[[6,3]]*2
    assert 'params' not in definition
    assert 'opacity' not in definition['encoding']
    assert 'selected_curves' not in json.dumps(definition)
    assert any(layer['mark']['type']=='area' for layer in definition['layer'])
    assert any(row['field']=='${field:segment}' for row in definition['encoding']['detail'])
    sections=layout.sections(data,'entity');panels=[panel for section in sections for panel in section['panels']]
    charts=[panel for panel in panels if panel['viewType']=='Vega2']
    assert len(charts)==7
    assert [row['config']['fieldSettings']['value'] for row in charts]==['return_mean','gain_mean','fresh_gain_mean','return_std','return_min','control_seconds','late_seconds']
    assert {p['config']['mediaKeys'][0] for p in panels if p['viewType']=='Media Browser'}=={layout.TABLE_KEY,layout.PROGRESS_KEY}
    assert 'Pending' in panels[0]['config']['value'] and 'sample SD' in panels[0]['config']['value']
    assert 'Five paired' in panels[0]['config']['value'] and '3 transfer settings' in panels[0]['config']['value']
    assert 'use the run list' not in panels[0]['config']['value']
    assert 'click legend' not in panels[0]['config']['value']
    assert charts[2]['config']['fieldSettings']['segment']=='fresh_segment'
    assert all(charts[i]['config']['fieldSettings']['lower']=='interval_not_applicable' for i in [3,4])
    assert all(not panel['isAuto'] for panel in panels)
    assert all(panel['config']['userQuery']['queryFields'][0]['fields'][0]['args'][0]['value']==layout.TABLE_KEY for panel in charts)


def test_separate_view_preserves_every_existing_view_and_is_idempotent(tmp_path):
    service=Service();original=deepcopy(service.views)
    result=install(tmp_path,service)
    assert result['status']=='verified' and result['browser_verified'] is False
    assert result['url'].endswith('?nw=transfercurvesabc123')
    assert service.views[:2]==original
    second=install(tmp_path,service)
    assert not second['changed'] and service.chart_writes==service.view_writes==1


def test_lost_mutation_response_is_reconciled_without_duplicate_view(tmp_path):
    service=Service(timeout=True)
    result=install(tmp_path,service)
    assert result['uncertain_response_reconciled'] is True and service.view_writes==1
    install(tmp_path,service)
    assert service.view_writes==1


def test_failed_install_is_surfaced_and_receipt_saved(tmp_path):
    service=Service(timeout=True,apply=False)
    with pytest.raises(layout.ResultsLayoutError,match='readback'):install(tmp_path,service)
    receipt=json.loads((tmp_path/'results-layout-receipt.json').read_text())
    assert receipt['status']=='failed' and receipt['browser_verified'] is False
    assert len(service.views)==2


def test_existing_user_edit_is_preserved_and_reported(tmp_path):
    service=Service();install(tmp_path,service)
    data=json.loads(service.views[-1]['spec']);data['section']['runSets'][0]['name']='My changes'
    service.views[-1]['spec']=json.dumps(data);before=deepcopy(service.views)
    with pytest.raises(layout.ResultsLayoutError,match='preserving user edits'):install(tmp_path,service)
    assert service.views==before and service.view_writes==1


def test_chart_definition_mismatch_never_overwrites_existing_chart(tmp_path):
    service=Service();install(tmp_path,service);service.chart['spec']='{}'
    with pytest.raises(layout.ResultsLayoutError,match='chart could not be verified'):install(tmp_path,service)
    assert service.chart_writes==1


def test_content_addressed_chart_changes_with_candidate_palette():
    data=campaign();first=layout.chart_id(data,'entity')
    data['candidates']=data['candidates'][:1]
    assert layout.chart_id(data,'entity')!=first


@pytest.mark.parametrize('timeout',[False,True])
@pytest.mark.parametrize('version',[layout.LEGACY_VERSION,layout.INTERACTIVE_VERSION,layout.PREVIOUS_VERSION])
def test_exact_owned_previous_view_is_upgraded_in_place_without_touching_other_views(tmp_path,timeout,version):
    service=Service(timeout=timeout)
    old=layout.saved_spec(spec(),campaign(),'entity','abc123',version=version)
    service.views.append(dict(id='old-owned',name='nw-transfercurvesabc123-v',type='project-view',
        displayName='Transfer mechanisms across checkpoints',spec=json.dumps(old)))
    others=deepcopy(service.views[:2])
    result=install(tmp_path,service)
    assert result['changed'] and result['upgraded_from']==version
    assert result['view_id']=='old-owned' and len(service.views)==3 and service.views[:2]==others
    new=json.loads(service.views[-1]['spec'])
    assert layout._installed(new,campaign(),'entity','abc123')
    assert new['section']['runSets'][0]['filters']['filters'][0]['key']['name']=='transfer_curve_overview'
    assert new['section']['runSets'][0]['filters']['filters'][0]['value']=='abc123'
    assert not install(tmp_path,service)['changed'] and service.view_writes==1


@pytest.mark.parametrize('version',[layout.LEGACY_VERSION,layout.INTERACTIVE_VERSION,layout.PREVIOUS_VERSION])
def test_modified_previous_view_is_not_upgraded(tmp_path,version):
    service=Service()
    old=layout.saved_spec(spec(),campaign(),'entity','abc123',version=version)
    old['section']['panelBankConfig']['sections'][0]['panels'][0]['config']['value']+=' User annotation'
    service.views.append(dict(id='old-owned',name='nw-transfercurvesabc123-v',type='project-view',
        displayName='Transfer mechanisms across checkpoints',spec=json.dumps(old)))
    before=deepcopy(service.views)
    with pytest.raises(layout.ResultsLayoutError,match='preserving user edits'):install(tmp_path,service)
    assert service.views==before and service.view_writes==0


def test_v3_recognition_preserves_exact_broken_chart_for_safe_upgrade():
    data=campaign()
    previous=layout.chart_definition(data,version=layout.INTERACTIVE_VERSION)
    assert previous['params'][0]['bind']=='legend'
    assert previous['encoding']['opacity']['condition']['param']=='selected_curves'
    assert layout.chart_id(data,'entity',version=layout.INTERACTIVE_VERSION)!=layout.chart_id(data,'entity')
    assert layout.chart_definition(data,legacy=True)==layout.chart_definition(data)


@pytest.mark.parametrize(('version','digest'),[
    ('transfer-checkpoint-curves-v2','ac470be0a207522b02bfbc512cfe08e85e1120d775833a2b804bf11750e1680c'),
    ('transfer-checkpoint-curves-v3','702a368d86f5d8a7dd79d428a3ee3455e194428208a24156cfd14039cbe12d2e'),
    ('transfer-checkpoint-curves-v4','11bfa3c3dac47ef914fef7c8f2fc996136d2bfb313cb59a6c958f58809059589'),
])
def test_legacy_recognition_is_byte_exact_even_with_new_comparison_styles(version,digest):
    # Pins captured from deployed 69160834 before extending this layout.
    data=dict(campaign(),comparison_styles=comparisons())
    assert layout._hash(layout.saved_spec(spec(),data,'entity','abc123',version=version))==digest


def test_eight_curve_view_selects_both_overviews_and_preserves_all_seven_panels():
    data=dict(campaign(),comparison_styles=comparisons())
    definition=layout.chart_definition(data)
    scale=definition['encoding']['color']['scale']
    assert scale['range']==['#000000',*adapter.COLORS[:5],'#cc79a7','#56b4e9']
    assert len(set(scale['domain']))==len(set(scale['range']))==8
    view=layout.saved_spec(spec(),data,'entity','abc123')
    selected=view['section']['runSets'][0]
    assert selected['selections']=={'root':1,'bounds':[],'tree':[]}
    assert selected['filters']['filters']==[dict(key={'section':'config','name':'transfer_curve_overview'},
        op='=',value='abc123',disabled=False)]
    panels=[panel for section in view['section']['panelBankConfig']['sections'] for panel in section['panels']]
    charts=[panel for panel in panels if panel['viewType']=='Vega2']
    assert len(charts)==7
    for chart in charts:
        config=chart['config']
        assert config['transform']['name']=='tableWithLeafColNames'
        query=config['userQuery']['queryFields'][0]
        assert query['name']=='runSets' and query['args'][0]['value']=='${runSets}'
        assert query['fields'][0]['args'][0]['value']==layout.TABLE_KEY
    intro=panels[0]['config']['value']
    assert '8 curves' in intro and 'two existing H3 MPPI comparisons' in intro
    assert 'actor mean' in intro and 'proposal mean' in intro and 'mean-action episodes' not in intro
    assert 'no matched fresh SAC control' in intro and 'mixed GPUs' in intro
    assert 'matching steady-time measurements' in intro


def test_exact_six_curve_v4_upgrades_to_eight_curve_v5_with_same_url(tmp_path):
    service=Service();data=dict(campaign(),comparison_styles=comparisons())
    old=layout.saved_spec(spec(),campaign(),'entity','abc123',version=layout.PREVIOUS_VERSION)
    service.views.append(dict(id='old-owned',name='nw-transfercurvesabc123-v',type='project-view',
        displayName='My comparison title',spec=json.dumps(old)))
    others=deepcopy(service.views[:2])
    kwargs=dict(campaign=data,entity='entity',project='ambi-inner-bench',publication_id='abc123',receipt_dir=tmp_path)
    result=layout.ensure_saved_view(SimpleNamespace(_service_api=service),**kwargs)
    assert result['view_id']=='old-owned' and result['url'].endswith('?nw=transfercurvesabc123')
    assert result['upgraded_from']=='transfer-checkpoint-curves-v4'
    assert service.views[:2]==others and service.views[-1]['displayName']=='My comparison title'
    assert layout._installed(json.loads(service.views[-1]['spec']),data,'entity','abc123')
    assert not layout.ensure_saved_view(SimpleNamespace(_service_api=service),**kwargs)['changed']
    assert service.view_writes==1


@pytest.mark.parametrize('field,value',[
    ('setting_id','prior'),('label','Setting 1'),('color','#000000'),('role','transfer'),('color','red')])
def test_invalid_or_ambiguous_comparison_style_is_rejected(field,value):
    external=comparisons();external[0][field]=value
    with pytest.raises(layout.ResultsLayoutError):
        layout.chart_definition(dict(campaign(),comparison_styles=external))


def extensions():
    return [dict(setting_id=key,label='J6 '+key,role='fresh' if key.endswith('_fresh') else 'transfer',color=color)
            for key,color in adapter.J6_COLORS.items()]


def test_host_extension_preserves_exact_v5_and_eleven_curve_style_contract(tmp_path):
    original=dict(campaign(),comparison_styles=comparisons())
    data=dict(original,extension_styles=extensions())
    assert layout.saved_spec(spec(),data,'entity','abc123',version=layout.VERSION)==layout.saved_spec(spec(),original,'entity','abc123')
    # Captured from deployed e1a7e94a before adding hosted extensions.
    assert layout._hash(layout.saved_spec(spec(),data,'entity','abc123',version=layout.VERSION))=='e5aab93f95dbd914b65790f6904b055a4148d773961b301b8c2139af7c1b9540'
    definition=layout.chart_definition(data)
    assert definition['encoding']['color']['scale']['range']==['#000000',*adapter.COLORS[:5],'#cc79a7','#56b4e9',*adapter.J6_COLORS.values()]
    assert definition['encoding']['strokeDash']['scale']['range'][-3:]==[[6,3],[1,0],[1,0]]
    assert len(set(definition['encoding']['color']['scale']['range']))==11
    service=Service();old=layout.saved_spec(spec(),original,'entity','abc123')
    service.views.append(dict(id='owned',name='nw-transfercurvesabc123-v',type='project-view',
        displayName='My saved comparison',spec=json.dumps(old)))
    before=deepcopy(service.views[:2]);kwargs=dict(campaign=data,entity='entity',project='ambi-inner-bench',
        publication_id='abc123',receipt_dir=tmp_path)
    result=layout.ensure_saved_view(SimpleNamespace(_service_api=service),**kwargs)
    assert result['upgraded_from']==layout.VERSION and result['layout_version']==layout.HOST_VERSION
    assert result['view_id']=='owned' and result['url'].endswith('?nw=transfercurvesabc123')
    assert service.views[:2]==before and service.views[-1]['displayName']=='My saved comparison'
    panels=[p for s in json.loads(service.views[-1]['spec'])['section']['panelBankConfig']['sections'] for p in s['panels']]
    assert len([p for p in panels if p['viewType']=='Vega2'])==7
    assert '11 curves' in panels[0]['config']['value'] and '60 checkpoints from 525k through 2M' in panels[0]['config']['value']
    assert not layout.ensure_saved_view(SimpleNamespace(_service_api=service),**kwargs)['changed']
    assert service.view_writes==1


def test_host_extension_rejects_unknown_v5_user_edits(tmp_path):
    data=dict(campaign(),comparison_styles=comparisons(),extension_styles=extensions())
    service=Service();old=layout.saved_spec(spec(),data,'entity','abc123',version=layout.VERSION)
    old['section']['panelBankConfig']['sections'][1]['panels'][0]['config']['stringSettings']['title']='My title'
    service.views.append(dict(id='owned',name='nw-transfercurvesabc123-v',type='project-view',displayName='Title',spec=json.dumps(old)))
    before=deepcopy(service.views)
    with pytest.raises(layout.ResultsLayoutError,match='preserving user edits'):
        layout.ensure_saved_view(SimpleNamespace(_service_api=service),campaign=data,entity='entity',
            project='ambi-inner-bench',publication_id='abc123',receipt_dir=tmp_path)
    assert service.views==before and service.view_writes==0


def followups():
    return [dict(setting_id=key,label='J8 '+key,role='fresh' if key.endswith('_fresh') else 'transfer',color=color)
            for key,color in adapter.J8_COLORS.items()]


@pytest.mark.parametrize('previous_version',[layout.HOST_VERSION,layout.FOLLOWUP_VERSION])
def test_j8_view_preserves_colors_and_upgrades_same_view_to_fourteen_curves(tmp_path,previous_version):
    original=dict(campaign(),comparison_styles=comparisons(),extension_styles=extensions())
    data=dict(original,followup_styles=followups())
    old=layout.saved_spec(spec(),data,'entity','abc123',version=previous_version)
    # Captured from deployed dc72b61 before adding J8.
    if previous_version==layout.HOST_VERSION:
        assert old==layout.saved_spec(spec(),original,'entity','abc123')
        assert layout._hash(old)=='abc45cde5522c0629d82e5600c3c8c4537161b865c19dd4e2d240e1c5e1062be'
    original_scale=layout.chart_definition(original)['encoding']['color']['scale']
    definition=layout.chart_definition(data);scale=definition['encoding']['color']['scale']
    assert scale['range']==original_scale['range']+list(adapter.J8_COLORS.values())
    assert len(set(scale['domain']))==len(set(scale['range']))==14
    assert definition['encoding']['strokeDash']['scale']['range'][-3:]==[[6,3],[1,0],[1,0]]
    service=Service();service.views.append(dict(id='owned',name='nw-transfercurvesabc123-v',type='project-view',
        displayName='My saved comparison',spec=json.dumps(old)))
    others=deepcopy(service.views[:2]);kwargs=dict(campaign=data,entity='entity',project='ambi-inner-bench',
        publication_id='abc123',receipt_dir=tmp_path)
    result=layout.ensure_saved_view(SimpleNamespace(_service_api=service),**kwargs)
    assert result['upgraded_from']==previous_version and result['layout_version']==layout.LEGEND_VERSION
    assert result['view_id']=='owned' and result['url'].endswith('?nw=transfercurvesabc123')
    assert service.views[:2]==others and service.views[-1]['displayName']=='My saved comparison'
    installed=json.loads(service.views[-1]['spec']);panels=[p for s in installed['section']['panelBankConfig']['sections'] for p in s['panels']]
    charts=[p for p in panels if p['viewType']=='Vega2']
    assert len(charts)==7 and '14 curves' in panels[0]['config']['value']
    assert 'own matched fresh J8 control' in panels[0]['config']['value']
    assert installed['section']['runSets'][0]['selections']=={'root':1,'bounds':[],'tree':[]}
    assert all(p['config']['panelDefId']==layout.chart_id(data,'entity') for p in charts)
    assert not layout.ensure_saved_view(SimpleNamespace(_service_api=service),**kwargs)['changed']
    assert service.view_writes==1


def test_j8_preserves_user_modified_v6_view(tmp_path):
    data=dict(campaign(),comparison_styles=comparisons(),extension_styles=extensions(),followup_styles=followups())
    service=Service();old=layout.saved_spec(spec(),data,'entity','abc123',version=layout.HOST_VERSION)
    old['section']['panelBankConfig']['sections'][1]['panels'][0]['config']['stringSettings']['title']='Custom title'
    service.views.append(dict(id='owned',name='nw-transfercurvesabc123-v',type='project-view',displayName='Custom',spec=json.dumps(old)))
    before=deepcopy(service.views)
    with pytest.raises(layout.ResultsLayoutError,match='preserving user edits'):
        layout.ensure_saved_view(SimpleNamespace(_service_api=service),campaign=data,entity='entity',
            project='ambi-inner-bench',publication_id='abc123',receipt_dir=tmp_path)
    assert service.views==before and service.view_writes==0
