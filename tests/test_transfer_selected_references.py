"""H2 publication reuses references without duplicating scientific runs/artifacts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_checkpoint_publish as publish
from utils import transfer_checkpoint_publication as adapter
from utils import transfer_checkpoint_references as refs
from utils import wandb_transfer_checkpoint_layout as layout
from utils.eval_series import SeriesError, _record_fingerprint
from utils.transfer_mppi_comparison import STYLES
from test_transfer_j10_publication import campaign_at, candidate, record, save
from test_wandb_transfer_checkpoint_layout import Service


@pytest.fixture
def selected(tmp_path):
    newroot=tmp_path/'new'/'campaign';campaign=campaign_at(newroot)
    newsetting=dict(candidate(8),setting_id=refs.SETTING,H=2,label='H2 J8 critic50%')
    campaign.update(selection_id=refs.SELECTION,candidates=[newsetting],checkpoints=[],cells=[])
    for i,step in enumerate(range(525000,2000001,25000)):
        campaign['checkpoints'].append(dict(step=step,checkpoint_sha256=f'{step:064x}',metadata_sha256='d'*64,
            prior_manifest_code={'commit':'a'*40},prior_pin={'step':step}))
        campaign['cells'].append(dict(newsetting,index=i,checkpoint_index=i,step=step,name=str(step),
            result_dir=str(newroot/'settings'/str(step))))
    save(newroot/'campaign.json',campaign)
    roots=[]
    for j,total in ((4,400),(8,180)):
        root=tmp_path/f'old{j}'/'publication';old=deepcopy(campaign)
        old.update(selection_id='old',candidates=[candidate(j)])
        oldpath=root.parent/'campaign'/'campaign.json';save(oldpath,old)
        runs={}
        # Only the original prior and selected H1 transfer curve are needed.
        for setting,key in ((adapter.settings(old)[0],'prior'),) if j==4 else ((candidate(8),8),):
            directory=root/'registry'/setting['setting_id']/'abc123';ident=adapter.identity(old,setting)
            registry=dict(schema_version=1,run_dir=str(directory),run_id='abc123',identity=ident,
                identity_sha256=publish.fingerprint(ident),entity='entity',project='ambi-inner-bench')
            save(directory/'run.json',registry);journal={'records':{}}
            for cp in campaign['checkpoints']:
                pointcampaign=dict(old,checkpoints=[cp]);value=record(pointcampaign,setting,key)
                value['metrics'].update({'eval/return_min':min(e['return'] for e in value['episodes']),
                    'eval/paired_gain_mean':0.,'eval/paired_gain_sample_std':0.,
                    'runtime/control_seconds_per_decision':9. if key=='prior' else .8,
                    'runtime/control_sample_std':None if key=='prior' else 0.,
                    'runtime/late_control_seconds_per_decision':None if key=='prior' else .7,
                    'runtime/late_control_sample_std':None if key=='prior' else 0.})
                rid=value['record_id'];save(directory/'records'/(rid+'.json'),value)
                journal['records'][rid]=dict(status='published',checkpoint_step=cp['step'],
                    artifact_sha256={'source.json':'e'*64},record_sha256=_record_fingerprint(value,{'source.json':'e'*64}))
            save(directory/'publication.json',journal)
            runs[setting['setting_id']]=dict(run_dir=str(directory),run_id='abc123')
        state=dict(prepared=True,entity='entity',project='ambi-inner-bench',campaign_root=str(oldpath.parent),
            campaign_sha256=adapter.digest(oldpath),publication_id=f'original{j}',runs=runs)
        save(root/'publication.json',state);save(root/'publisher-complete.json',dict(status='complete',completed=total,total=total));roots.append(root)
    data=dict(curves=deepcopy(STYLES),sources={},prior={'records':{}},points=[],progress=[])
    for style,key in zip(STYLES,('return_q','soft_q')):
        data['sources'][key]=dict(run_id=key,records={})
        for cp in campaign['checkpoints']:
            step=cp['step'];pin=dict(checkpoint_sha256=cp['checkpoint_sha256'])
            data['sources'][key]['records'][str(step)]=pin;data['prior']['records'][str(step)]=pin
            row={c:None for c in adapter.POINT_COLUMNS}
            row.update(step=step,setting=style['setting_id'],label=style['label'],state='complete',episodes=5,
                segment=0,fresh_segment=0,return_mean=300.,return_std=2.,gain_mean=1.,gain_std=.5,
                fresh_comparison_state='not_applicable')
            data['points'].append(row);data['progress'].append(dict(step=step,setting=style['setting_id'],state='complete'))
    datapath=tmp_path/'mppi-data.json';save(datapath,data)
    save(roots[0]/'mppi-overlay.json',dict(status='published',publication_id='original4',
        campaign_sha256=adapter.digest(roots[0].parent/'campaign'/'campaign.json'),content_sha256=publish.fingerprint(data),
        curves=deepcopy(STYLES),source_runs={key:key for key in ('return_q','soft_q')}))
    args=SimpleNamespace(root=newroot,publication_root=newroot.parent/'publication',entity='entity',project='ambi-inner-bench',
        owner='tester',attempt_label='H2 sweep',comparison_host_publication_root=roots[0],
        comparison_extension_publication_root=roots[1],comparison_mppi_data=datapath)
    return campaign,args


def test_prepare_and_collect_reuse_only_published_references(selected):
    campaign,args=selected
    state=publish.prepare(args)
    assert state['runs']['prior']['read_only'] is True
    assert set(state['runs'])=={'prior',refs.SETTING}
    assert not (args.publication_root/'registry'/'prior').exists()
    assert state['comparison_publication_id']==state['publication_id']!='original4'
    snap=publish.collect(args.root,campaign,state,jobs_active=True)
    assert snap['completed']==0 and snap['total']==60 and len(snap['curves'])==5
    assert snap['curves'][refs.SETTING]['completed']==0
    assert len(snap['curves'][refs.SETTING]['points'])==60
    assert all(p['return_mean'] is None and p['fresh_comparison_state']=='not_applicable'
        for p in snap['curves'][refs.SETTING]['points'])
    assert all(curve['completed']==60 for k,curve in snap['curves'].items() if k!=refs.SETTING)
    assert all(p['control_seconds'] is None and p['late_seconds'] is None for p in snap['curves']['prior']['points'])
    second=publish.prepare(args)
    assert second['runs']==state['runs'] and second['publication_id']==state['publication_id']


def test_selected_reference_pins_cannot_silently_change(selected):
    campaign,args=selected;state=publish.prepare(args)
    data=json.loads(args.comparison_mppi_data.read_text());data['points'][0]['return_mean']+=1;save(args.comparison_mppi_data,data)
    with pytest.raises(SeriesError,match='Pinned selected reference changed'):
        publish.collect(args.root,campaign,state,jobs_active=True)
    with pytest.raises(SeriesError,match='MPPI compact data differs'):publish.prepare(args)


def test_reference_requires_complete_seed_and_immutable_fingerprint(selected):
    campaign,args=selected
    state=json.loads((args.comparison_extension_publication_root/'publication.json').read_text())
    directory=Path(state['runs'][refs.REFERENCE]['run_dir']);path=next((directory/'records').iterdir())
    value=json.loads(path.read_text());value['episodes'].pop();save(path,value)
    with pytest.raises(SeriesError,match='fingerprint'):publish.prepare(args)


def test_missing_completed_checkpoint_rejected(selected):
    campaign,args=selected
    state=json.loads((args.comparison_extension_publication_root/'publication.json').read_text())
    path=Path(state['runs'][refs.REFERENCE]['run_dir'])/'publication.json'
    journal=json.loads(path.read_text());journal['records'].pop(next(iter(journal['records'])));save(path,journal)
    with pytest.raises(SeriesError,match='all requested checkpoints'):publish.prepare(args)


def test_selected_layout_has_five_true_colors_no_fresh_gain_and_preserves_views(selected,tmp_path):
    campaign,args=selected;state=publish.prepare(args)
    data=publish.comparison_layout_campaign(campaign,state,args.publication_root)
    styles=layout.display_settings(data)
    assert len(styles)==5 and len({s['color'] for s in styles})==5
    chart=layout.chart_definition(data)
    assert chart['encoding']['color']['legend']['symbolOpacity']==1
    assert 'labelColor' in chart['encoding']['color']['legend']
    panels=[p for section in layout.sections(data,'entity') for p in section['panels']]
    assert len([p for p in panels if p['viewType']=='Vega2'])==6
    assert 'No matched H2 fresh control' in panels[0]['config']['value']
    service=Service();before=deepcopy(service.views);api=SimpleNamespace(_service_api=service)
    first=layout.ensure_saved_view(api,campaign=data,entity='entity',project='ambi-inner-bench',
        publication_id=state['publication_id'],receipt_dir=tmp_path/'view')
    second=layout.ensure_saved_view(api,campaign=data,entity='entity',project='ambi-inner-bench',
        publication_id=state['publication_id'],receipt_dir=tmp_path/'view')
    assert first['status']=='verified' and not second['changed'] and service.views[:2]==before
    spec=json.loads(service.views[-1]['spec']);spec['section']['panelBankConfig']['sections'][0]['name']='User edit'
    service.views[-1]['spec']=json.dumps(spec)
    with pytest.raises(layout.ResultsLayoutError,match='preserving user edits'):
        layout.ensure_saved_view(api,campaign=data,entity='entity',project='ambi-inner-bench',
            publication_id=state['publication_id'],receipt_dir=tmp_path/'view')


def test_compact_artifact_preserves_trace_hashes_and_diagnostics(selected,tmp_path):
    campaign,args=selected;r=record(campaign,campaign['candidates'][0],8)
    r['artifact_files']={name:str(tmp_path/name) for name in ('manifest.json','results.json','worker-receipt.json')}
    hashes={}
    for seed in range(101,106):
        name=f'decisions-seed-{seed}.jsonl';path=tmp_path/name;path.write_text('{"metrics":1}\n')
        r['artifact_files'][name]=str(path);hashes[name]=adapter.digest(path)
    r['provenance']['worker_receipt']['file_sha256']=hashes
    value=refs.compact_scientific_record(r)
    assert len(value['artifact_files'])==3
    assert value['episodes']==r['episodes'] and len(value['provenance']['raw_trace_references'])==5
    assert value['provenance']['raw_trace_references']['decisions-seed-101.jsonl']['sha256']==hashes['decisions-seed-101.jsonl']
    (tmp_path/'decisions-seed-101.jsonl').write_text('changed')
    with pytest.raises(SeriesError,match='hash mismatch'):refs.compact_scientific_record(r)


def test_absence_of_matched_fresh_control_is_explicit_only():
    campaign=campaign_at(Path('/tmp/test'));setting=candidate(8)
    with pytest.raises(SeriesError,match='Missing matched fresh'):adapter.fresh_setting(campaign,setting)
    campaign.update(selection_id=refs.SELECTION);setting.update(setting_id=refs.SETTING,H=2)
    assert adapter.fresh_setting(campaign,setting) is None


def test_completed_h2_cell_stages_only_compact_artifact_and_counts_new_work(selected,tmp_path,monkeypatch):
    from slurm import ambi_transfer_curve_campaign as worker
    campaign,args=selected;state=publish.prepare(args);cell=campaign['cells'][0]
    directory=Path(cell['result_dir']);directory.mkdir(parents=True)
    value=record(campaign,campaign['candidates'][0],8)
    value['metrics'].update({'eval/return_min':100.,'eval/paired_gain_mean':24.,'eval/paired_gain_sample_std':1.})
    value['artifact_files']={name:str(directory/name) for name in ('manifest.json','results.json','worker-receipt.json')}
    hashes={}
    for name,path in value['artifact_files'].items():save(Path(path),{'test':name})
    for seed in range(101,106):
        name=f'decisions-seed-{seed}.jsonl';path=directory/name;path.write_text('{"tested_trace":true}\n')
        value['artifact_files'][name]=str(path);hashes[name]=adapter.digest(path)
    value['provenance']['worker_receipt']['file_sha256']=hashes
    save(args.root/'receipts'/'0.json',{'complete':True})
    monkeypatch.setattr(worker,'validate_receipt',lambda *a:dict(historical_reuse=False))
    monkeypatch.setattr(worker,'validate_result',lambda *a,**kw:({},{}))
    monkeypatch.setattr(publish,'normalize_transfer',lambda *a,**kw:deepcopy(value))
    snapshot=publish.collect(args.root,campaign,state,jobs_active=True)
    assert snapshot['completed']==1 and snapshot['total']==60
    assert snapshot['curves'][refs.SETTING]['points'][0]['state']=='complete'
    assert not snapshot['failures']
    records=publish.records_by_step(state['runs'][refs.SETTING]['run_dir'])
    staged=records[525000]
    assert set(staged['artifact_files'])=={'manifest.json','results.json','worker-receipt.json'}
    assert len(staged['provenance']['raw_trace_references'])==5
    assert publish.collect(args.root,campaign,state,jobs_active=True)==snapshot


def test_cpu_prepare_wrapper_forwards_pinned_mppi_data_intact(tmp_path):
    import os
    import subprocess
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    git=bin_dir/'git';git.write_text('#!/bin/sh\nif [ "$1" = rev-parse ]; then echo abc123; fi\n');git.chmod(0o755)
    python=bin_dir/'capture-python';python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');python.chmod(0o755)
    script=Path(__file__).resolve().parents[1]/'slurm/run_ambi_transfer_curves_cpu_oscar.sbatch'
    env=dict(os.environ,PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],SLURM_SUBMIT_DIR=str(script.parents[1]),
        CAMPAIGN_ROOT=str(tmp_path/'campaign'),EXPECTED_ACTION_MODES_SHA='abc123',CAMPAIGN_MODE='publication_prepare',
        PYTHON_BIN=str(python),SLURM_JOB_ID='test',SLURM_RESTART_COUNT='0',PUBLICATION_ROOT=str(tmp_path/'publication'),
        ATTEMPT_LABEL='H2 test',COMPARISON_HOST_PUBLICATION_ROOT='/host data',
        COMPARISON_EXTENSION_PUBLICATION_ROOT='/H1 J8 data',COMPARISON_MPPI_DATA='/reference data/mppi overlay.json')
    args=subprocess.check_output(['bash',str(script)],env=env,text=True).splitlines()
    assert args[args.index('--comparison-mppi-data')+1]=='/reference data/mppi overlay.json'
    assert args[args.index('--comparison-host-publication-root')+1]=='/host data'
    assert args[args.index('--comparison-extension-publication-root')+1]=='/H1 J8 data'


@pytest.fixture
def publish_wrapper(tmp_path):
    import os
    import subprocess
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    git=bin_dir/'git';git.write_text('#!/bin/sh\nif [ "$1" = rev-parse ]; then echo abc123; fi\n');git.chmod(0o755)
    python=bin_dir/'capture-python';python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');python.chmod(0o755)
    script=Path(__file__).resolve().parents[1]/'slurm/run_ambi_transfer_curves_cpu_oscar.sbatch'
    env={key:value for key,value in os.environ.items() if key not in {'GPU_JOB_ID','GPU_JOB_IDS','PUBLISH_ONCE'}}
    env.update(PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],SLURM_SUBMIT_DIR=str(script.parents[1]),
        CAMPAIGN_ROOT=str(tmp_path/'campaign'),EXPECTED_ACTION_MODES_SHA='abc123',CAMPAIGN_MODE='publish',
        PYTHON_BIN=str(python),SLURM_JOB_ID='test',SLURM_RESTART_COUNT='0',PUBLICATION_ROOT=str(tmp_path/'publication'))
    return lambda extra:subprocess.run(['bash',str(script)],env=dict(env,**extra),text=True,capture_output=True)


@pytest.mark.parametrize('extra,expected',[
    ({'GPU_JOB_ID':'7330868'},['--gpu-job-id','7330868']),
    ({'GPU_JOB_IDS':'7330868,7330999'},['--gpu-job-id','7330868','--gpu-job-id','7330999']),
    ({'GPU_JOB_IDS':'7330868,7330999','GPU_JOB_ID':'1'},['--gpu-job-id','7330868','--gpu-job-id','7330999']),
    ({'GPU_JOB_IDS':'','GPU_JOB_ID':'7330868'},['--gpu-job-id','7330868']),
    ({'PUBLISH_ONCE':'1'},['--once']),
])
def test_cpu_publish_wrapper_forwards_all_gpu_arrays(publish_wrapper,extra,expected):
    result=publish_wrapper(extra)
    assert result.returncode==0,result.stderr
    args=result.stdout.splitlines()
    assert args[:2]==['slurm/ambi_transfer_checkpoint_publish.py','watch']
    assert args[6:]==expected


@pytest.mark.parametrize('extra',[
    {},{'GPU_JOB_ID':'abc'},{'GPU_JOB_IDS':'7330868,abc'},
    {'GPU_JOB_IDS':',7330868'},{'GPU_JOB_IDS':'7330868,'},
    {'GPU_JOB_IDS':'7330868,,7330999'},{'GPU_JOB_IDS':'7330868, 7330999'},
    {'GPU_JOB_IDS':'7330868\n7330999'},{'GPU_JOB_IDS':'7330868;echo unsafe'},
    {'GPU_JOB_IDS':'invalid','GPU_JOB_ID':'7330868'},
])
def test_cpu_publish_wrapper_rejects_invalid_gpu_ids_before_python(publish_wrapper,extra):
    result=publish_wrapper(extra)
    assert result.returncode==2
    assert 'must contain numeric job IDs separated by commas' in result.stderr
    assert not result.stdout
