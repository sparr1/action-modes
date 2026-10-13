"""Seven-curve horizon follow-up reuses immutable H2/H1 references."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_checkpoint_publish as publish
from utils import transfer_checkpoint_publication as adapter
from utils import transfer_checkpoint_references as refs
from utils import wandb_transfer_checkpoint_layout as layout
from utils.eval_series import SeriesError, _record_fingerprint, stage_record
from test_transfer_selected_references import selected
from test_transfer_j10_publication import candidate, record, save
from test_wandb_transfer_checkpoint_layout import Service


@pytest.fixture
def horizon(selected, tmp_path):
    old,args=selected;state=publish.prepare(args)
    entry=state['runs'][refs.SETTING];directory=Path(entry['run_dir']);journal={'records':{}}
    setting=adapter.settings(old)[1]
    for cp in old['checkpoints']:
        value=record(dict(old,checkpoints=[cp]),setting,8)
        value['provenance']['resolved_config']['algorithm_config']['alg_params']['inner_rollout_horizon']=2
        value['metrics'].update({'eval/return_min':min(e['return'] for e in value['episodes']),
            'eval/paired_gain_mean':10.,'eval/paired_gain_sample_std':1.,
            'runtime/control_seconds_per_decision':1.,'runtime/control_sample_std':.1,
            'runtime/late_control_seconds_per_decision':.9,'runtime/late_control_sample_std':.1})
        rid=value['record_id'];save(directory/'records'/(rid+'.json'),value)
        journal['records'][rid]=dict(status='published',checkpoint_step=cp['step'],
            artifact_sha256={'source.json':'e'*64},record_sha256=_record_fingerprint(value,{'source.json':'e'*64}))
    save(directory/'publication.json',journal)
    snap=publish.collect(args.root,old,state,jobs_active=False)
    state['layout']=dict(status='verified',layout_version=layout.SELECTED_VERSION)
    state['overview']=dict(snapshot_sha256=publish.fingerprint(dict(snapshot=snap,layout=state['layout'])))
    save(args.publication_root/'publication.json',state);save(args.publication_root/'snapshot.json',snap)
    save(args.publication_root/'publisher-complete.json',dict(status='complete',completed=60,total=60))
    newroot=tmp_path/'horizon'/'campaign';campaign=deepcopy(old)
    h3=dict(candidate(8),setting_id=refs.HORIZON_TRANSFER,H=3,label='H3 J8 critic50%')
    fresh=dict(candidate(8),setting_id=refs.HORIZON_FRESH,H=2,arm='fresh',role='fresh',label='H2 J8 fresh',arm_definition={})
    campaign.update(selection_id=refs.HORIZON_SELECTION,candidates=[h3,fresh],cells=[])
    for ci,cp in enumerate(campaign['checkpoints']):
        for s in campaign['candidates']:
            i=len(campaign['cells'])
            campaign['cells'].append(dict(s,index=i,checkpoint_index=ci,step=cp['step'],name=f'{cp["step"]}-{s["setting_id"]}',
                result_dir=str(newroot/'settings'/str(cp['step'])/s['setting_id'])))
    save(newroot/'campaign.json',campaign)
    newargs=SimpleNamespace(**vars(args));newargs.root=newroot;newargs.publication_root=newroot.parent/'publication'
    newargs.comparison_host_publication_root=args.publication_root
    newargs.comparison_extension_publication_root=None;newargs.comparison_mppi_data=None
    return campaign,newargs


def test_horizon_reuses_five_references_and_allocates_only_two_new_curves(horizon):
    campaign,args=horizon;state=publish.prepare(args)
    assert set(state['runs'])=={'prior',refs.HORIZON_TRANSFER,refs.HORIZON_FRESH}
    assert state['runs']['prior']['read_only']
    assert state['publication_id']==state['comparison_publication_id']
    snapshot=publish.collect(args.root,campaign,state,jobs_active=True)
    assert snapshot['completed']==0 and snapshot['total']==120 and not snapshot['failures']
    assert len(snapshot['curves'])==7 and sum(len(c['points']) for c in snapshot['curves'].values())==420
    assert all(c['completed']==60 for k,c in snapshot['curves'].items() if k not in (refs.HORIZON_TRANSFER,refs.HORIZON_FRESH))
    assert snapshot['curves'][refs.SETTING]['points'][0]['fresh_comparison_state']=='pending'
    assert snapshot['curves'][refs.HORIZON_TRANSFER]['points'][0]['fresh_comparison_state']=='not_applicable'
    assert snapshot['curves'][refs.REFERENCE]['points'][0]['fresh_comparison_state']=='not_applicable'
    assert publish.prepare(args)['runs']==state['runs']
    assert not (args.publication_root/'registry'/refs.SETTING).exists()


def test_h2_fresh_gain_pairs_by_seed_only_when_fresh_finishes(horizon):
    campaign,args=horizon;state=publish.prepare(args)
    inherited=json.loads((args.publication_root/'comparison-host.json').read_text())
    donor=json.loads(Path(inherited['paired_reference_records']['525000']).read_text())
    fresh=deepcopy(donor);fresh['record_id']='f'*64
    fresh['identity']=adapter.identity(campaign,next(s for s in campaign['candidates'] if s['setting_id']==refs.HORIZON_FRESH))
    deltas=[3.,-2.,6.,-7.,10.]
    for e,d in zip(fresh['episodes'],deltas):e['return']-=d
    fresh['metrics']['eval/return_mean'],fresh['metrics']['eval/return_sample_std']=adapter.moments([e['return'] for e in fresh['episodes']])
    source=args.root/'fresh-source.json';save(source,{'verified_test':True})
    fresh['artifact_files']={'source.json':str(source)}
    stage_record(state['runs'][refs.HORIZON_FRESH]['run_dir'],fresh)
    snapshot=publish.collect(args.root,campaign,state,jobs_active=True)
    assert snapshot['completed']==1 and snapshot['total']==120
    gain=snapshot['curves'][refs.SETTING]['points'][0]
    assert (gain['fresh_gain_mean'],gain['fresh_gain_std'])==pytest.approx(adapter.moments(deltas))
    assert gain['fresh_comparison_state']=='complete' and gain['fresh_setting']==refs.HORIZON_FRESH
    assert snapshot['curves'][refs.SETTING]['points'][1]['fresh_gain_mean'] is None
    assert snapshot['curves'][refs.HORIZON_FRESH]['points'][0]['fresh_gain_mean']==0.
    assert json.loads(Path(inherited['paired_reference_records']['525000']).read_text())==donor


def test_horizon_rejects_incomplete_or_unpublished_h2_host(horizon):
    campaign,args=horizon;root=args.comparison_host_publication_root
    path=root/'publisher-complete.json';value=json.loads(path.read_text());value['completed']=59;save(path,value)
    with pytest.raises(SeriesError,match='complete without failures'):publish.prepare(args)


@pytest.mark.parametrize('verified_snapshot',[False,True])
def test_horizon_accepts_hover_host_only_with_matching_final_snapshot(horizon,verified_snapshot):
    _,args=horizon;root=args.comparison_host_publication_root
    path=root/'publication.json';state=json.loads(path.read_text())
    snapshot=json.loads((root/'snapshot.json').read_text())
    state['layout']['layout_version']=layout.HOVER_SELECTED_VERSION
    if verified_snapshot:
        state['overview']['snapshot_sha256']=publish.fingerprint(dict(snapshot=snapshot,layout=state['layout']))
    save(path,state);before=path.read_bytes()
    if verified_snapshot:
        prepared=publish.prepare(args)
        assert prepared['prepared'] and prepared['runs']['prior']['read_only']
    else:
        with pytest.raises(SeriesError,match='verified final snapshot'):publish.prepare(args)
    assert path.read_bytes()==before


def test_horizon_rejects_unknown_host_layout_even_with_matching_snapshot(horizon):
    _,args=horizon;root=args.comparison_host_publication_root
    path=root/'publication.json';state=json.loads(path.read_text())
    snapshot=json.loads((root/'snapshot.json').read_text())
    state['layout']['layout_version']='unknown-layout'
    state['overview']['snapshot_sha256']=publish.fingerprint(dict(snapshot=snapshot,layout=state['layout']))
    save(path,state)
    with pytest.raises(SeriesError,match='verified final snapshot'):publish.prepare(args)


def test_horizon_rejects_wrong_horizon_in_reference_config(horizon):
    campaign,args=horizon;root=args.comparison_host_publication_root
    state=json.loads((root/'publication.json').read_text());directory=Path(state['runs'][refs.SETTING]['run_dir'])
    path=next((directory/'records').glob('*.json'));value=json.loads(path.read_text())
    value['provenance']['resolved_config']['algorithm_config']['alg_params']['inner_rollout_horizon']=1;save(path,value)
    journal=json.loads((directory/'publication.json').read_text());rid=value['record_id']
    journal['records'][rid]['record_sha256']=_record_fingerprint(value,journal['records'][rid]['artifact_sha256']);save(directory/'publication.json',journal)
    with pytest.raises(SeriesError,match='C/A/N/B or compile protocol'):publish.prepare(args)


def test_horizon_pins_final_snapshot_and_inherited_reference_files(horizon):
    campaign,args=horizon;state=publish.prepare(args)
    path=args.comparison_host_publication_root/'snapshot.json';data=json.loads(path.read_text())
    data['curves'][refs.SETTING]['points'][0]['return_mean']+=1;save(path,data)
    with pytest.raises(SeriesError,match='Pinned selected reference changed'):
        publish.collect(args.root,campaign,state,jobs_active=True)
    with pytest.raises(SeriesError,match='verified final snapshot'):publish.prepare(args)


def test_horizon_layout_has_seven_true_colors_paired_gain_and_preserves_views(horizon,tmp_path):
    campaign,args=horizon;state=publish.prepare(args)
    data=publish.comparison_layout_campaign(campaign,state,args.publication_root)
    styles=layout.display_settings(data)
    assert len(styles)==len({s['color'] for s in styles})==7
    chart=layout.chart_definition(data)
    assert chart['encoding']['color']['legend']['symbolOpacity']==1
    assert chart['encoding']['color']['legend']['labelColor']
    panels=[p for section in layout.sections(data,'entity') for p in section['panels']]
    assert len([p for p in panels if p['viewType']=='Vega2'])==7
    assert '120 new evaluations' in panels[0]['config']['value']
    gain=[p for p in panels if p['viewType']=='Vega2' and p['config']['fieldSettings']['value']=='fresh_gain_mean']
    assert len(gain)==1 and gain[0]['config']['stringSettings']['title']=='H2 Bernoulli transfer improvement over H2 fresh SAC'
    service=Service();before=deepcopy(service.views);api=SimpleNamespace(_service_api=service)
    kw=dict(campaign=data,entity='entity',project='ambi-inner-bench',publication_id=state['publication_id'],receipt_dir=tmp_path/'view')
    assert layout.ensure_saved_view(api,**kw)['status']=='verified'
    assert not layout.ensure_saved_view(api,**kw)['changed'] and service.views[:2]==before
    spec=json.loads(service.views[-1]['spec']);spec['section']['panelBankConfig']['sections'][0]['name']='User edit'
    service.views[-1]['spec']=json.dumps(spec)
    with pytest.raises(layout.ResultsLayoutError,match='preserving user edits'):layout.ensure_saved_view(api,**kw)


def test_horizon_requires_its_single_explicit_h2_host(horizon):
    _,args=horizon;args.comparison_mppi_data='/unrelated.json'
    with pytest.raises(SeriesError,match='omit extension/MPPI paths'):publish.prepare(args)
