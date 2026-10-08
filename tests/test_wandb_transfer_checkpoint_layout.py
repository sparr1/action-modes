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
@pytest.mark.parametrize('version',[layout.LEGACY_VERSION,layout.PREVIOUS_VERSION])
def test_exact_owned_v2_or_v3_is_upgraded_in_place_without_touching_other_views(tmp_path,timeout,version):
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


@pytest.mark.parametrize('version',[layout.LEGACY_VERSION,layout.PREVIOUS_VERSION])
def test_modified_v2_or_v3_is_not_upgraded(tmp_path,version):
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
    previous=layout.chart_definition(data,version=layout.PREVIOUS_VERSION)
    assert previous['params'][0]['bind']=='legend'
    assert previous['encoding']['opacity']['condition']['param']=='selected_curves'
    assert layout.chart_id(data,'entity',version=layout.PREVIOUS_VERSION)!=layout.chart_id(data,'entity')
    assert layout.chart_definition(data,legacy=True)==layout.chart_definition(data)
