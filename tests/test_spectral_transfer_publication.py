"""Spectral publication keeps proxy metrics, controller overhead and pending data explicit."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_discovery_publish as publisher
from tests.test_bernoulli_probability_publication import MultiChartService
from utils import wandb_transfer_discovery_layout as layout


def spectral(method='svd', *, norm_matched=False, rank=16, strength=.5):
    return dict(method=method, rank=rank, strength=strength, norm_matched=norm_matched)


def campaign():
    arms = {'fresh': dict(actor_rho=0., critic_rho=0., parameter_scope='matrices')}
    for method in ('svd','activation','gradient'):
        for matched in (False,True):
            arms[f'actor_{method}_{int(matched)}'] = dict(actor_spectral=spectral(method,norm_matched=matched),
                critic_rho=0., parameter_scope='matrices')
    arms.update(actor_bernoulli=dict(actor_bernoulli_p=.5, critic_rho=0., parameter_scope='matrices'),
        actor_blend=dict(actor_rho=.5, critic_rho=0., parameter_scope='matrices'),
        actor_carry=dict(actor_rho=1., critic_rho=0., parameter_scope='matrices'),
        critic_svd=dict(actor_rho=0., critic_spectral=spectral(), parameter_scope='matrices'),
        joint_gradient=dict(actor_spectral=spectral('gradient'), critic_spectral=spectral('gradient'), parameter_scope='matrices'))
    cells = [dict(index=i,name=f'h{h}_j{j}_{arm}',H=h,J=j,arm=arm)
             for i,(h,j,arm) in enumerate((h,j,arm) for h in (1,2) for j in (1,4) for arm in arms)]
    return dict(family='spectral_transfer', arms=arms, cells=cells, H=[1,2], J=[1,4], seeds=[101,102,103],
        max_steps=500, diagnostics={'enabled':False}, publication={'view_title':'Spectral transfer test'})


def fake_wandb():
    return SimpleNamespace(Table=lambda **kwargs:kwargs, plot=SimpleNamespace(line_series=lambda **kwargs:kwargs))


def episodes(*, counts=True):
    values=[]
    for index, seed in enumerate((101,102,103)):
        summary=dict(spectral_initial_actor_loss=float(index), spectral_final_actor_loss=float(index)-.25,
            spectral_actor_transferred_energy_ratio=1.2+index*.2,
            spectral_actor_custom_new_metric=float(index)*2,
            spectral_actor_ignored_bool=True, spectral_actor_ignored_nan=float('nan'),
            spectral_actor_ignored_text='unavailable')
        coverage={key:5 if 'energy_ratio' in key else 6 for key in summary} if counts else {}
        values.append(dict(seed=seed,solver_seed=seed,**{'return':100.+index},length=500,control_seconds=10.,
            spectral_probe_seconds=2.,spectral_filter_seconds=1.,diagnostic_seconds=.5,diagnostic_samples=6,
            diagnostics=dict(enabled=True,summary=summary,summary_counts=coverage)))
    return values


def test_spectral_styles_separate_methods_dense_norm_controls_and_components():
    value=campaign()
    groups=layout.campaign_curve_groups(value)
    assert [prefix for prefix,_,_ in groups] == ['actor/','critic/','joint/']
    assert all(arms[-1]=='fresh' for _,_,arms in groups)
    metadata={arm:layout.spectral_arm_metadata(value,arm) for arm in value['arms']}
    assert len({metadata[f'actor_{method}_{int(matched)}']['color'] for method in ('svd','activation','gradient')
                for matched in (False,True)}) == 6
    assert metadata['actor_svd_1']['label'].startswith('Dense norm match to SVD')
    assert metadata['actor_svd_1']['requested_rank']==16 and metadata['actor_svd_1']['norm_matched']
    assert metadata['actor_bernoulli']['method']=='bernoulli'
    assert metadata['actor_blend']['method']=='blend' and metadata['actor_carry']['method']=='carry'
    assert metadata['critic_svd']['component']=='critic' and metadata['joint_gradient']['component']=='joint'
    spec=layout.campaign_chart_definition(value,'actor')
    assert spec['encoding']['color']['scale']['range'][-1]=='#000000'
    assert spec['encoding']['strokeDash']['scale']['range']==[[],[8,3],[2,3]]


def test_spectral_layout_is_visible_before_any_result_and_idempotent(tmp_path):
    value=campaign(); service=MultiChartService(); before=deepcopy(service.views)
    options=dict(entity='entity',project='project',receipt_dir=tmp_path,run_id='spectral123',campaign=value)
    receipt=layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service),**options)
    assert receipt['url'].endswith('?nw=spectral575spectral123')
    assert service.views[:len(before)]==before and service.chart_writes==3 and service.writes==1
    assert all('/spectral_transfer_' in identifier for identifier in receipt['custom_chart_id'].values())
    spec=json.loads(service.views[-1]['spec']); sections=layout._bank(spec)['sections']
    assert all(s['__id__'].startswith('ambi-spectral-transfer-v1-') for s in sections)
    intro=sections[0]['panels'][0]['config']['value']
    for phrase in ('fixed-model proxies','not environment returns','dense norm-matched controls','included in',
                   'can exceed one','Pending values are null'):
        assert phrase in intro
    snapshot=dict(settings=[],results=[],episodes=[],diagnostics=[],completed=0,total=len(value['cells']),failed=0)
    payload=publisher.payload(fake_wandb(),snapshot,value)
    assert 'discovery/diagnostics' in payload
    assert all(key in payload for key in receipt['expected_chart_keys'])
    assert all(not any(payload[key]['ys']) for key in receipt['expected_chart_keys'])
    assert 'discovery/actor/h1_spectral_initial_actor_loss_vs_j' in payload
    assert 'discovery/actor/h1_spectral_actor_initial_loss_vs_j' not in payload
    assert 'discovery/joint/h1_spectral_final_critic_loss_vs_j' in payload
    assert layout._saved_installed(spec,'spectral123',value,receipt['custom_chart_id'])
    assert not layout.ensure_discovery_saved_view(SimpleNamespace(_service_api=service),**options)['changed']
    assert service.writes==1 and service.chart_writes==3


def test_spectral_summary_aggregates_unknown_numeric_metrics_and_per_metric_coverage():
    value=campaign(); cell=publisher._cell_metadata(value['cells'][1],value)
    row,diagnostics=publisher._measurement_rows(cell,episodes(),'current campaign','complete',spectral=True)
    assert row['controller_seconds_per_decision']==.02  # Includes probe/filter; never add them twice.
    assert row['spectral_probe_seconds']==6. and row['spectral_filter_seconds']==3.
    assert row['spectral_probe_seconds_per_decision']==.004
    assert row['spectral_filter_seconds_per_decision']==.002
    measurements={r['metric']:r for r in diagnostics}
    assert measurements['spectral_initial_actor_loss']['mean']==1.
    assert measurements['spectral_initial_actor_loss']['episode_std']==1.
    assert measurements['spectral_initial_actor_loss']['samples']==18
    assert measurements['spectral_actor_transferred_energy_ratio']['samples']==15
    assert measurements['spectral_actor_transferred_energy_ratio']['mean']==pytest.approx(1.4)
    assert measurements['spectral_actor_custom_new_metric']['mean']==2.
    assert measurements['spectral_actor_custom_new_metric']['sample_count_basis']=='per-metric contributing roots'
    assert not any('ignored' in metric for metric in measurements)
    row2,measurements2=publisher._measurement_rows(cell,episodes(counts=False),'current campaign','complete',spectral=True)
    assert all(measurement['samples'] is None for measurement in measurements2)
    assert all(measurement['sample_count_basis']=='metric root coverage unavailable' for measurement in measurements2)
    columns=publisher.table_columns(value)
    assert set(publisher.SPECTRAL_TIMERS)<=set(columns['episodes'])
    assert {'method','requested_rank','norm_matched','parameter_scope'}<=set(columns['results'])


def test_spectral_payload_uses_heldout_metrics_and_controller_timer_fields():
    value=campaign(); cell=publisher._cell_metadata(value['cells'][1],value)
    row,diagnostics=publisher._measurement_rows(cell,episodes(),'current campaign','complete',spectral=True)
    snapshot=dict(settings=[],results=[row],episodes=[],diagnostics=diagnostics,completed=1,total=len(value['cells']),failed=0)
    payload=publisher.payload(fake_wandb(),snapshot,value)
    initial=payload['discovery/actor/h1_spectral_initial_actor_loss_vs_j']
    arm_index=initial['keys'].index(cell['arm'])
    assert initial['ys'][arm_index]==[1.]
    assert payload['discovery/actor/h1_spectral_probe_seconds_per_decision_vs_j']['ys'][arm_index]==[.004]
    assert payload['discovery/actor/h1_spectral_filter_seconds_per_decision_vs_j']['ys'][arm_index]==[.002]
    assert 'fixed-model proxy' in initial['title']


def test_pending_spectral_results_do_not_invent_timer_or_metric_zeroes():
    row,diagnostics=publisher._measurement_rows({'arm':'actor_svd_0'},None,'current campaign','pending',spectral=True)
    assert row['return_mean'] is None and row['spectral_probe_seconds'] is None
    assert row['spectral_filter_seconds_per_decision'] is None and diagnostics==[]


@pytest.mark.parametrize('bad',[float('inf'),float('nan'),-1.,True,'slow'])
def test_invalid_controller_timer_fails_publication(bad):
    data=episodes(); data[0]['spectral_filter_seconds']=bad
    with pytest.raises(ValueError,match='Invalid spectral controller timer'):
        publisher._measurement_rows({'arm':'actor_svd_0'},data,'current campaign','complete',spectral=True)


def test_spectral_fresh_control_with_custom_name_supports_paired_gains(tmp_path):
    value=campaign(); value['cells']=value['cells'][:2]
    fresh=episodes(); transferred=deepcopy(fresh)
    for episode in transferred:
        episode['return']+=5
    completed={value['cells'][0]['name']:fresh,value['cells'][1]['name']:transferred}
    snapshot=publisher.snapshot(tmp_path,value,True,completed)
    assert snapshot['results'][0]['paired_vs_fresh_mean']==0.
    assert snapshot['results'][1]['paired_vs_fresh_mean']==5.


def test_spectral_family_overrides_legacy_palette_without_mutating_configuration():
    value=campaign(); expected=layout.campaign_chart_definition(value,'actor')
    value['publication']['arm_styles']=[[arm,'Wrong label','#ff0000'] for arm in value['arms']]
    before=deepcopy(value)
    assert layout.campaign_chart_definition(value,'actor')==expected
    assert value==before
