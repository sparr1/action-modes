#!/usr/bin/env python3
"""Publish validated transfer checkpoint curves using durable eval-series runs.

``prepare`` is local-only and allocates exactly one new run per setting plus a
prior run unless an explicit comparison host reuses its published prior read-only.
``watch`` serially publishes completed scientific records, retains per-run summary
tables, and logs combined display tables on a separate overview
run. It never inserts non-result rows into immutable scientific curve history.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.ambi_benchmark import atomic_json
from utils.eval_series import create_run, load_run, stage_record, Publisher
from utils.transfer_checkpoint_publication import (read as read_json,digest,require,identity,settings,
    normalize_prior,normalize_transfer,table_rows,fresh_setting,POINT_COLUMNS,PROGRESS_COLUMNS,PRIOR_ID)
from utils.wandb_transfer_checkpoint_layout import (ensure_saved_view,display_settings,TABLE_KEY,PROGRESS_KEY)
from slurm.ambi_transfer_sweep_publish import publisher_lock
from slurm.ambi_closed_loop_publish import gpu_jobs_active


_READ_RETRY_DELAYS=(0.25,1.,4.)
_TRANSIENT_READ_ERRNOS=frozenset((errno.ESTALE,errno.EAGAIN,errno.ETIMEDOUT))


def read(path):
    """Reopen a JSON file after a transient shared-filesystem read failure.

    Four attempts allow a stale handle to expire without retrying corruption,
    missing files, permissions, scientific validation, or any external write.
    """
    for attempt in range(len(_READ_RETRY_DELAYS)+1):
        try:
            return read_json(path)
        except OSError as exc:
            if exc.errno not in _TRANSIENT_READ_ERRNOS or attempt==len(_READ_RETRY_DELAYS):
                raise
            delay=_READ_RETRY_DELAYS[attempt]
            print(json.dumps(dict(event='transient_json_read_retry',path=str(path),errno=exc.errno,
                next_attempt=attempt+2,delay_seconds=delay)),file=sys.stderr,flush=True)
            time.sleep(delay)


def write(path,value):
    atomic_json(Path(path),value,overwrite=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def comparison_host_receipt(campaign, campaign_path, host_root, entity, project):
    """Pin the completed original comparison without allocating/copying its runs."""
    from utils.eval_series import validate_record, _record_fingerprint
    from utils.transfer_checkpoint_publication import J6_COLORS
    host_root=Path(host_root).resolve(); path=host_root/'publication.json'; host=read(path)
    host_campaign_path=Path(host['campaign_root'])/'campaign.json'; original=read(host_campaign_path)
    require(host.get('prepared') is True and host['campaign_sha256']==digest(host_campaign_path),
            'Comparison host campaign binding differs.')
    require(host['entity']==entity and host['project']==project,'Comparison host W&B project differs.')
    require(re.fullmatch(r'[A-Za-z0-9]+',host['publication_id']) is not None,'Invalid comparison host ID.')
    require(host.get('layout',{}).get('status')=='verified'
            and host['layout']['layout_version']=='transfer-checkpoint-curves-v5',
            'Comparison host must be the verified original v5 view.')
    require([cp['step'] for cp in original['checkpoints']]==list(range(25000,2000001,25000)),
            'Comparison host must contain the original 80 checkpoints.')
    require([cp['step'] for cp in campaign['checkpoints']]==list(range(525000,2000001,25000)),
            'Hosted H1 J6 requires exactly the 60 checkpoints after 500k.')
    require(len(campaign['candidates'])==3 and {row['setting_id'] for row in campaign['candidates']}==set(J6_COLORS)
            and all(row['H']==1 and row['J']==6 for row in campaign['candidates']),
            'Comparison host supports only the three H1 J6 settings.')
    require(all(campaign[key]==original[key] for key in ('protocol','source_run','seeds','controller_seed','max_steps')),
            'Comparison host backbone or episode protocol differs.')
    require(all(campaign.get(key)==original.get(key) for key in ('scientific_source','diagnostics','gpu_hardware')),
            'Comparison host science, diagnostics or timing hardware differs.')
    layout_campaign=comparison_layout_campaign(original,host,host_root)
    require(len(display_settings(layout_campaign))==8,'Comparison host requires the original six curves and two published MPPI curves.')
    completion=read(host_root/'publisher-complete.json')
    require(completion.get('status')=='complete' and completion.get('completed')==completion.get('total')==400,
            'Comparison host publication must be complete.')
    prior_entry=deepcopy(host['runs'][PRIOR_ID]); directory=Path(prior_entry['run_dir'])
    registry=load_run(directory); journal=read(directory/'publication.json')
    require(registry['run_id']==prior_entry['run_id'] and registry['identity']==identity(original,settings(original)[0]),
            'Comparison host prior identity differs.')
    require(len(journal['records'])==80 and all(row['status']=='published' for row in journal['records'].values()),
            'Comparison host prior must contain 80 published records.')
    indexed={row['checkpoint_step']:(rid,row) for rid,row in journal['records'].items()}
    require(len(indexed)==80,'Repeated host prior checkpoint.')
    original_checkpoints={row['step']:row for row in original['checkpoints']}; pins={}
    for checkpoint in campaign['checkpoints']:
        step=checkpoint['step']; old=original_checkpoints[step]
        require(all(checkpoint[key]==old[key] for key in
                    ('checkpoint_sha256','metadata_sha256','prior_manifest_code','prior_pin')),
                'Comparison host checkpoint or historical prior pin differs.')
        rid,entry=indexed[step]; record_path=directory/'records'/(rid+'.json')
        record=validate_record(read(record_path),registry['identity'])
        require(record['record_id']==rid and record['checkpoint']==dict(step=step,sha256=checkpoint['checkpoint_sha256'])
                and _record_fingerprint(record,entry['artifact_sha256'])==entry['record_sha256'],
                'Published host prior record fingerprint differs.')
        normalized=normalize_prior(campaign,checkpoint,checkpoint['prior_pin'])
        require(record['episodes']==normalized['episodes'] and record['metrics']==normalized['metrics']
                and record['provenance']==normalized['provenance'],
                'Published prior episodes or provenance differ from the pinned reference.')
        alias='record-'+hashlib.sha256(rid.encode()).hexdigest()[:20]
        pins[str(step)]=dict(record_id=rid,record_sha256=entry['record_sha256'],
            record_file_sha256=digest(record_path),checkpoint=deepcopy(record['checkpoint']),
            artifact=f"{entity}/{project}/eval-{registry['run_id']}-{step}:{alias}")
    styles=[{key:row[key] for key in ('setting_id','label','color','role')} for row in settings(campaign)[1:]]
    result=dict(format_version=1,kind='h1-j6-after500k-comparison-host',campaign_sha256=digest(campaign_path),
        host_publication_root=str(host_root),host_publication_id=host['publication_id'],
        host_publication_sha256=digest(path),host_campaign_path=str(host_campaign_path),
        host_campaign_sha256=digest(host_campaign_path),mppi_overlay_sha256=digest(host_root/'mppi-overlay.json'),
        host_overview_run_id=host['overview']['run_id'],prior_run=dict(run_dir=str(directory),run_id=registry['run_id'],
            identity_sha256=registry['identity_sha256'],journal_sha256=digest(directory/'publication.json')),
        prior_records=pins,extension_styles=styles)
    display_settings(dict(layout_campaign,extension_styles=styles))
    return result


def validate_comparison_host(campaign,state,output):
    """Read-only resume guard; never normalize/stage or rewrite a host record."""
    expected=state.get('comparison_host_sha256')
    if not expected:return None
    path=Path(output)/'comparison-host.json'
    require(digest(path)==expected,'Comparison host receipt changed.')
    host=read(path)
    require(host.get('format_version')==1 and host.get('kind')=='h1-j6-after500k-comparison-host'
            and host['campaign_sha256']==state['campaign_sha256']
            and host['host_publication_id']==state.get('comparison_publication_id'),'Comparison host receipt binding differs.')
    root=Path(host['host_publication_root'])
    require(digest(root/'publication.json')==host['host_publication_sha256']
            and digest(host['host_campaign_path'])==host['host_campaign_sha256']
            and digest(root/'mppi-overlay.json')==host['mppi_overlay_sha256'],
            'Pinned comparison host changed.')
    require(set(host['prior_records'])=={str(cp['step']) for cp in campaign['checkpoints']},
            'Comparison host checkpoint selection changed.')
    entry=host['prior_run']; directory=Path(entry['run_dir']); registry=load_run(directory)
    require(registry['run_id']==entry['run_id'] and registry['identity_sha256']==entry['identity_sha256']
            and digest(directory/'publication.json')==entry['journal_sha256'],
            'Published prior registry or journal changed.')
    require(state['runs'][PRIOR_ID]['run_dir']==entry['run_dir']
            and state['runs'][PRIOR_ID]['run_id']==entry['run_id']
            and state['runs'][PRIOR_ID].get('read_only') is True,'Hosted prior must remain read-only.')
    for pin in host['prior_records'].values():
        require(digest(directory/'records'/(pin['record_id']+'.json'))==pin['record_file_sha256'],
                'Pinned published prior record changed.')
    return host


def owned_settings(campaign,state):
    return [row for row in settings(campaign) if not state['runs'][row['setting_id']].get('read_only')]


def comparison_publication_id(state):
    return state.get('comparison_publication_id',state['publication_id'])


def prepare(args):
    """Allocate once, recover an interrupted allocation, and stage pinned priors."""
    root=Path(args.root).resolve(); output=Path(args.publication_root).resolve()
    campaign=read(root/'campaign.json')
    binding=dict(campaign_root=str(root),campaign_sha256=digest(root/'campaign.json'),
        entity=args.entity,project=args.project,owner=args.owner,attempt_label=args.attempt_label)
    require(args.project=='ambi-inner-bench','Transfer curves belong in ambi-inner-bench.')
    requested_host=getattr(args,'comparison_host_publication_root',None)
    host=(comparison_host_receipt(campaign,root/'campaign.json',requested_host,args.entity,args.project)
          if requested_host else None)
    binding['comparison_host_sha256']=None
    with publisher_lock(output):
        host_path=output/'comparison-host.json'
        if host:
            if host_path.exists():
                require(read(host_path)==host,'Comparison host receipt differs.')
            else:write(host_path,host)
            # Receipt hash is its exact on-disk bytes, not its JSON encoding.
            binding['comparison_host_sha256']=digest(host_path)
        path=output/'publication.json'
        if path.exists():
            state=read(path)
            require(all(state.get(key)==value for key,value in binding.items()),'Publication binding differs.')
        else:
            state=dict(schema_version=1,publication_id=uuid.uuid4().hex,**binding,runs={})
            write(path,state)
        if host:
            state['comparison_publication_id']=host['host_publication_id']
            inherited=host['prior_run']
            state['runs'][PRIOR_ID]=dict(run_dir=inherited['run_dir'],run_id=inherited['run_id'],read_only=True,
                label=settings(campaign)[0]['label'],color='#000000',order=0)
            write(path,state)
        for order,setting in enumerate(settings(campaign)):
            if host and setting['setting_id']==PRIOR_ID:continue
            key=setting['setting_id']; template=dict(identity=identity(campaign,setting),label=setting['label'])
            registry_root=output/'registry'/key
            if key not in state['runs']:
                candidates=list(registry_root.glob('*/run.json'))
                require(len(candidates)<=1,'Ambiguous interrupted registry allocation.')
                registry=(load_run(candidates[0].parent) if candidates else
                    create_run(registry_root,template,args.attempt_label,args.project,args.entity,args.owner))
                require(registry['identity']==template['identity'],'Recovered registry identity differs.')
                state['runs'][key]=dict(run_dir=registry['run_dir'],run_id=registry['run_id'],
                    label=setting['label'],color=setting['color'],order=order)
                write(path,state)
            registry=load_run(state['runs'][key]['run_dir'])
            require(registry['identity']==template['identity'],'Registry identity differs.')
        for checkpoint in ([] if host else campaign['checkpoints']):
            record=normalize_prior(campaign,checkpoint,checkpoint['prior_pin'])
            stage_record(state['runs'][PRIOR_ID]['run_dir'],record)
        state['prepared']=True
        write(path,state)
    return state


def records_by_step(run_dir):
    journal=read(Path(run_dir)/'publication.json')
    records={}
    for rid,entry in journal['records'].items():
        record=read(Path(run_dir)/'records'/(rid+'.json'))
        require(record['checkpoint']['step']==entry['checkpoint_step'],'Registry record checkpoint differs.')
        require(entry['checkpoint_step'] not in records,'Repeated checkpoint in curve registry.')
        records[entry['checkpoint_step']]=record
    return records


def pack_decision_traces(record, cache_root):
    """Losslessly pack new artifact traces, leaving worker files/records untouched.

    gzip headers contain no filename and mtime=0. An existing cache entry is
    accepted only after a complete roundtrip/hash check; unknown contents are
    never overwritten. Atomic hard-link publication also handles a raced writer.
    """
    result=deepcopy(record)
    require(re.fullmatch(r'[0-9a-f]{64}',record['record_id']) is not None,'Invalid trace-cache record identity.')
    expected={f"decisions-seed-{episode['seed']}.jsonl" for episode in record['episodes']}
    selected={name:path for name,path in record['artifact_files'].items()
              if re.fullmatch(r'decisions-seed-\d+\.jsonl',name)}
    require(set(selected)==expected,'Complete decision traces are required before compression.')
    directory=Path(cache_root).resolve()/record['record_id']
    directory.mkdir(parents=True,exist_ok=True)
    packed={}
    for name,path in sorted(selected.items()):
        source=Path(path);raw_sha=digest(source);raw_bytes=source.stat().st_size
        target=directory/(name+'.gz')
        def verify(candidate):
            require(not candidate.is_symlink(),'Trace cache must not be a symlink.')
            with candidate.open('rb') as stream:
                header=stream.read(10)
            require(len(header)==10 and header[:3]==b'\x1f\x8b\x08' and header[3]==0
                    and header[4:8]==b'\0'*4,'Trace cache has a non-deterministic gzip header.')
            decoded=hashlib.sha256();count=0
            with gzip.open(candidate,'rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):
                    decoded.update(chunk);count+=len(chunk)
            require(count==raw_bytes and decoded.hexdigest()==raw_sha,'Trace cache roundtrip differs from raw worker output.')
        if not target.exists():
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(dir=directory,prefix='.packing-',delete=False) as handle:
                    temporary=Path(handle.name)
                    with source.open('rb') as stream,gzip.GzipFile(filename='',mode='wb',fileobj=handle,
                                                                  compresslevel=6,mtime=0) as zipped:
                        for chunk in iter(lambda:stream.read(1024*1024),b''):zipped.write(chunk)
                    handle.flush();os.fsync(handle.fileno())
                verify(temporary)
                try:os.link(temporary,target)
                except FileExistsError:pass
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)
        verify(target)
        require(source.stat().st_size==raw_bytes and digest(source)==raw_sha,'Raw decision trace changed during compression.')
        result['artifact_files'].pop(name)
        result['artifact_files'][name+'.gz']=str(target)
        packed[name+'.gz']=dict(original_filename=name,raw_sha256=raw_sha,raw_bytes=raw_bytes,
            encoding='gzip',gzip_mtime=0,gzip_filename='',compression_level=6,
            compressed_sha256=digest(target),compressed_bytes=target.stat().st_size,
            roundtrip_verified=True)
    result['provenance']['decision_trace_storage']=dict(format_version=1,lossless=True,files=packed)
    return result


def collect(root,campaign,state,*,jobs_active):
    """Only receipt-verified complete seed panels can become data points."""
    from slurm.ambi_transfer_curve_campaign import validate_receipt,validate_result,receipt_path
    root=Path(root)
    output=Path(state['runs'][campaign['candidates'][0]['setting_id']]['run_dir']).parents[2]
    host=validate_comparison_host(campaign,state,output)
    selected_steps={row['step'] for row in campaign['checkpoints']}
    records={key:{step:record for step,record in records_by_step(row['run_dir']).items() if step in selected_steps}
             for key,row in state['runs'].items()}
    progress={key:{} for key in state['runs']}
    failures={}
    for cell in campaign['cells']:
        key,step=cell['setting_id'],cell['step']; directory=Path(cell['result_dir'])
        if step in records[key]:
            continue
        marker=receipt_path(root,cell)
        if marker.exists():
            try:
                receipt=validate_receipt(root,campaign,cell)
                result,manifest=validate_result(directory,campaign,cell,allow_historical=receipt['historical_reuse'])
                record=normalize_transfer(campaign,cell,receipt,result,manifest,records[PRIOR_ID][step],receipt_path=marker,
                    prior_reference=host['prior_records'][str(step)] if host else None)
                # Registry is publication_root/registry/setting/run_id. Existing
                # accepted records were skipped above, preserving their hashes.
                cache_root=Path(state['runs'][key]['run_dir']).parents[2]/'packed-traces'
                record=pack_decision_traces(record,cache_root)
                stage_record(state['runs'][key]['run_dir'],record)
                records[key][step]=record
                continue
            except Exception as exc:
                failures[cell['name']]=f'{type(exc).__name__}: {exc}'
                progress[key][step]=dict(state='validation_failed',error=failures[cell['name']])
                continue
        failure=root/'failures'/f"{cell['index']}.json"
        status=read(directory/'progress.json') if (directory/'progress.json').exists() else {}
        error=read(failure) if failure.exists() else status.get('error')
        if error:
            failures[cell['name']]=error if isinstance(error,str) else json.dumps(error,sort_keys=True)
        status_name=('evaluation_failed' if error else
                     'awaiting_receipt' if status.get('status')=='complete' else
                     status.get('status','pending') if jobs_active else 'incomplete')
        progress[key][step]=dict(state=status_name,completed_episodes=status.get('completed_episodes',0),
            seed=status.get('seed'),decision=status.get('decision'),error=failures.get(cell['name']))
    curves={}
    for setting in owned_settings(campaign,state):
        key=setting['setting_id']
        matched=fresh_setting(campaign,setting)
        points,states=table_rows(campaign,setting,records[key],progress[key],
                                 fresh_records=records.get(matched,{}))
        curves[key]=dict(points=points,progress=states,completed=len(records[key]),total=len(campaign['checkpoints']))
    return dict(curves=curves,failures=failures,
        completed=sum(len(value) for key,value in records.items() if key!=PRIOR_ID),total=len(campaign['cells']),
        jobs_active=jobs_active)


def summary_payload(wandb,curve):
    return {TABLE_KEY:wandb.Table(columns=POINT_COLUMNS,data=[[row.get(key) for key in POINT_COLUMNS] for row in curve['points']]),
        PROGRESS_KEY:wandb.Table(columns=PROGRESS_COLUMNS,data=[[row.get(key) for key in PROGRESS_COLUMNS] for row in curve['progress']]),
        'transfer_curves/completed_checkpoints':curve['completed'],
        'transfer_curves/total_checkpoints':curve['total'],
        'transfer_curves/status':'complete' if curve['completed']==curve['total'] else 'pending'}


def publish_snapshot(campaign,state,snapshot,output,wandb):
    """Summary tables update in place; only eval-series owns the history stream."""
    output=Path(output)
    for setting in owned_settings(campaign,state):
        key=setting['setting_id']; entry=state['runs'][key]; curve=snapshot['curves'][key]
        stamp=fingerprint(curve)
        journal=read(Path(entry['run_dir'])/'publication.json')
        pending=any(row['status']!='published' for row in journal['records'].values())
        if entry.get('snapshot_sha256')==stamp and not pending:
            continue
        with Publisher(entry['run_dir'],owner=state['owner'],wandb_module=wandb) as publisher:
            publisher.run.name=setting['label']
            publisher.run.config.update(dict(curve_label=setting['label'],transfer_curve_campaign=state['publication_id'],
                transfer_curve_setting=key,transfer_curve_order=entry['order'],transfer_curve_color=setting['color']),allow_val_change=True)
            # WBValue summary serialization binds tables without creating a
            # history row, which is essential for immutable checkpoint journals.
            publisher.run.summary.update(summary_payload(wandb,curve))
            if state.get('layout'):
                publisher.run.summary.update({'results_layout/status':state['layout']['status'],
                    'results_layout/url':state['layout']['url'],'results_layout/schema_verified':True})
            publisher.publish_pending()
        entry['snapshot_sha256']=stamp
        entry['published_checkpoints']=sum(row['status']=='published' for row in read(Path(entry['run_dir'])/'publication.json')['records'].values())
        write(output/'publication.json',state)


def overview_payload(wandb, snapshot):
    """One publication-only stream registers tables without changing result history."""
    points=[row for curve in snapshot['curves'].values() for row in curve['points']]
    progress=[row for curve in snapshot['curves'].values() for row in curve['progress']]
    return {TABLE_KEY:wandb.Table(columns=POINT_COLUMNS,data=[[row.get(key) for key in POINT_COLUMNS] for row in points]),
        PROGRESS_KEY:wandb.Table(columns=PROGRESS_COLUMNS,data=[[row.get(key) for key in PROGRESS_COLUMNS] for row in progress]),
        'campaign/completed':snapshot['completed'],'campaign/total':snapshot['total'],
        'campaign/failed':len(snapshot['failures'])}


def publish_overview(campaign,state,snapshot,output,wandb):
    """W&B hides summary-only tables; explicitly log them on a separate run.

    Scientific checkpoint runs retain their immutable result-only histories.
    Allocation is persisted before network access so interrupted retries resume
    the same overview. Logging a repeated snapshot after an uncertain response
    is harmless here: this run is presentation state, never scientific records.
    """
    output=Path(output)
    if 'overview' not in state:
        state['overview']=dict(run_id=uuid.uuid4().hex)
        write(output/'publication.json',state)
    entry=state['overview']
    stamp=fingerprint(dict(snapshot=snapshot,layout=state.get('layout')))
    if entry.get('snapshot_sha256')==stamp:
        return
    run=wandb.init(entity=state['entity'],project=state['project'],id=entry['run_id'],resume='allow',
        mode='online',name='J6 after 500k · live comparison' if state.get('comparison_host_sha256') else 'Transfer backbone · live comparison',
        job_type='transfer-checkpoint-overview',
        group='transfer-backbone-'+state['publication_id'][:8],
        tags=['transfer-backbone','publication-only','five-seed','full-episode'],
        config=dict(transfer_curve_overview=comparison_publication_id(state),source_campaign_sha256=state['campaign_sha256'],
            scientific_run_ids={key:entry['run_id'] for key,entry in state['runs'].items()},
            source_commit=campaign['source_commit'],seeds=campaign['seeds'],
            max_steps=campaign['max_steps'],source_run=campaign['source_run']),
        dir=str(output),reinit=True)
    try:
        run.log(overview_payload(wandb,snapshot))
        run.summary.update({'status':'complete' if snapshot['completed']==snapshot['total'] else 'running',
            'completed_settings':snapshot['completed'],'total_settings':snapshot['total'],
            'publication_only':True})
        if state.get('layout'):
            run.summary.update({'results_layout/status':state['layout']['status'],
                'results_layout/url':state['layout']['url'],'results_layout/schema_verified':True})
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    entry['snapshot_sha256']=stamp
    write(output/'publication.json',state)


def comparison_layout_campaign(campaign,state,output):
    """Bind a separately published MPPI overview to presentation only.

    The receipt is deliberately separate from publication.json, which the live
    publisher rewrites from its in-memory state. No new scientific run, record,
    or campaign cell is introduced by this display overlay.
    """
    if state.get('comparison_host_sha256'):
        host=validate_comparison_host(campaign,state,output)
        root=Path(host['host_publication_root']); original=read(host['host_campaign_path'])
        result=comparison_layout_campaign(original,read(root/'publication.json'),root)
        result['extension_styles']=deepcopy(host['extension_styles'])
        display_settings(result)
        return result
    path=Path(output)/'mppi-overlay.json'
    if not path.exists():
        return campaign
    overlay=read(path)
    require(type(overlay.get('format_version')) is int and overlay['format_version']==1,
            'Unsupported MPPI overlay receipt format.')
    require(overlay.get('publication_id')==state['publication_id']
            and overlay.get('campaign_sha256')==state['campaign_sha256'],
            'MPPI overlay belongs to a different publication or campaign.')
    require(overlay.get('status')=='published','MPPI overview has not been published.')
    run_id=overlay.get('run_id')
    require(isinstance(run_id,str) and re.fullmatch(r'[A-Za-z0-9]+',run_id) is not None,
            'Invalid MPPI overview run ID.')
    reserved={row['run_id'] for row in state['runs'].values()}|{state.get('overview',{}).get('run_id')}
    require(run_id not in reserved,'MPPI overview must use a separate presentation run.')
    curves=overlay.get('curves')
    require(isinstance(curves,list) and len(curves)==2,'Expected two MPPI overlay curves.')
    result=dict(campaign,comparison_styles=deepcopy(curves))
    display_settings(result)
    return result


def watch(args):
    root=Path(args.root).resolve(); output=Path(args.publication_root).resolve()
    with publisher_lock(output):
        state=read(output/'publication.json'); campaign=read(root/'campaign.json')
        require(state.get('prepared') is True and state['campaign_root']==str(root)
                and state['campaign_sha256']==digest(root/'campaign.json'),'Prepare the exact campaign first.')
        import wandb
        # Local preparation publishes nothing. First online pass initializes all
        # setting runs, including empty pending curves, then installs the view.
        ended=None;layout_checked=False
        while True:
            snapshot=None;phase='scheduler_status'
            try:
                active=True if args.once and not args.gpu_job_id else gpu_jobs_active(args.gpu_job_id)
                phase='collection'
                snapshot=collect(root,campaign,state,jobs_active=active)
                phase='snapshot_write'
                write(output/'snapshot.json',snapshot)
                phase='publication_or_layout'
                publish_snapshot(campaign,state,snapshot,output,wandb)
                publish_overview(campaign,state,snapshot,output,wandb)
                if not layout_checked:
                    previous_layout=state.get('layout')
                    layout_campaign=comparison_layout_campaign(campaign,state,output)
                    state['layout']=ensure_saved_view(wandb.Api(timeout=30),campaign=layout_campaign,
                        entity=state['entity'],project=state['project'],publication_id=comparison_publication_id(state),
                        receipt_dir=output/'results-layout')
                    layout_checked=True
                    # Reflect a newly installed view in every run, while a
                    # resumed publisher rechecks the view without extra writes.
                    if previous_layout != state['layout']:
                        for entry in state['runs'].values():
                            if not entry.get('read_only'):entry.pop('snapshot_sha256',None)
                    write(output/'publication.json',state)
                    publish_snapshot(campaign,state,snapshot,output,wandb)
                    publish_overview(campaign,state,snapshot,output,wandb)
            except Exception as exc:
                failure=dict(error=f'{type(exc).__name__}: {exc}',phase=phase,
                    completed=snapshot['completed'] if snapshot is not None else None,total=len(campaign['cells']))
                try:
                    write(output/'publisher-failure.json',failure)
                except OSError as receipt_error:
                    # A shared-filesystem outage can also prevent its receipt.
                    # Preserve the original exception and expose both in Slurm stderr.
                    print(json.dumps(dict(event='publisher_failure_receipt_write_failed',failure=failure,
                        receipt_error=f'{type(receipt_error).__name__}: {receipt_error}')),file=sys.stderr,flush=True)
                raise
            print(json.dumps(dict(completed=snapshot['completed'],total=snapshot['total'],
                failures=snapshot['failures'],url=state['layout']['url'])),flush=True)
            if snapshot['completed']==snapshot['total']:
                write(output/'publisher-complete.json',dict(status='complete',completed=snapshot['completed'],
                    total=snapshot['total'],url=state['layout']['url'],browser_verified=False))
                return snapshot
            if args.once:return snapshot
            if not active:
                ended=ended or time.monotonic()
                if time.monotonic()-ended>=args.drain_seconds:
                    error=dict(error='GPU jobs ended with incomplete or invalid checkpoint cells.',**snapshot)
                    write(output/'publisher-failure.json',error)
                    raise RuntimeError(error['error'])
            else:ended=None
            time.sleep(args.poll_seconds)


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prepare_parser=sub.add_parser('prepare',help='Local-only registry allocation and pinned prior validation.')
    watch_parser=sub.add_parser('watch',help='Publish pending/results and verify the isolated comparison view.')
    for child in (prepare_parser,watch_parser):
        child.add_argument('--root',type=Path,required=True)
        child.add_argument('--publication-root',type=Path,required=True)
    prepare_parser.add_argument('--comparison-host-publication-root',type=Path,
        help='Reuse the completed original comparison and its prior read-only for the H1 J6 extension.')
    prepare_parser.add_argument('--attempt-label',required=True)
    prepare_parser.add_argument('--owner',default='oscar-rgao48')
    prepare_parser.add_argument('--entity',default='rwgao_b-brown-university')
    prepare_parser.add_argument('--project',default='ambi-inner-bench')
    watch_parser.add_argument('--gpu-job-id',action='append',default=[])
    watch_parser.add_argument('--poll-seconds',type=float,default=60)
    watch_parser.add_argument('--drain-seconds',type=float,default=180)
    watch_parser.add_argument('--once',action='store_true')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.command=='prepare':
        value=prepare(args)
        print(json.dumps(dict(publication_id=value['publication_id'],runs=value['runs'])),flush=True)
    else:
        require(args.once or bool(args.gpu_job_id),'watch requires --gpu-job-id (or --once for a single refresh).')
        require(0<args.poll_seconds<=60 and args.drain_seconds>=0,'Invalid polling/drain interval.')
        watch(args)
