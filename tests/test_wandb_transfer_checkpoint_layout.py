"""A visible comparison view is created separately and never silently overwritten."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from utils import wandb_transfer_checkpoint_layout as layout
from utils import transfer_checkpoint_publication as adapter


def campaign():
    return dict(checkpoints=[dict(step=25000),dict(step=50000)],candidates=[dict(setting_id=f's{i}',label=f'H{i} J4 · actor 100%, critic 50%') for i in range(1,4)])


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
            if self.apply:self.views.append(dict(id='new',name=variables['name'],type=variables['type'],spec=variables['spec'],displayName=variables['displayName']))
            if self.timeout:raise TimeoutError('response lost')
            return {'upsertView':{'view':{'id':'new'}}}
        return {'project':{'allViews':{'edges':[{'node':deepcopy(row)} for row in self.views]}}}


def install(tmp_path,service):
    return layout.ensure_saved_view(SimpleNamespace(_service_api=service),campaign=campaign(),entity='entity',
        project='ambi-inner-bench',publication_id='abc123',receipt_dir=tmp_path)


def test_layout_has_distinct_fixed_colors_bands_and_pending_progress():
    data=campaign();definition=layout.chart_definition(data)
    assert definition['encoding']['color']['scale']['range']==['#000000',*adapter.COLORS[:3]]
    assert len(set(definition['encoding']['color']['scale']['range']))==4
    assert any(layer['mark']['type']=='area' for layer in definition['layer'])
    assert any(row['field']=='${field:segment}' for row in definition['encoding']['detail'])
    sections=layout.sections(data,'entity');panels=[panel for section in sections for panel in section['panels']]
    charts=[panel for panel in panels if panel['viewType']=='Vega2']
    assert len(charts)==3
    assert [row['config']['fieldSettings']['value'] for row in charts]==['return_mean','gain_mean','late_seconds']
    assert {p['config']['mediaKeys'][0] for p in panels if p['viewType']=='Media Browser'}=={layout.TABLE_KEY,layout.PROGRESS_KEY}
    assert 'Pending' in panels[0]['config']['value'] and 'sample SD' in panels[0]['config']['value']
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
