#!/usr/bin/env python3
"""Publish validated transfer checkpoint curves using durable eval-series runs.

``prepare`` is local-only and allocates exactly one new run per setting plus a
prior run. ``watch`` serially publishes completed records and summary-only
progress tables; it never inserts non-result rows into immutable curve history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.ambi_benchmark import atomic_json
from utils.eval_series import create_run, load_run, stage_record, Publisher
from utils.transfer_checkpoint_publication import (read,digest,require,identity,settings,
    normalize_prior,normalize_transfer,table_rows,fresh_setting,POINT_COLUMNS,PROGRESS_COLUMNS,PRIOR_ID)
from utils.wandb_transfer_checkpoint_layout import (ensure_saved_view,TABLE_KEY,PROGRESS_KEY)
from slurm.ambi_transfer_sweep_publish import publisher_lock
from slurm.ambi_closed_loop_publish import gpu_jobs_active


def write(path,value):
    atomic_json(Path(path),value,overwrite=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def prepare(args):
    """Allocate once, recover an interrupted allocation, and stage pinned priors."""
    root=Path(args.root).resolve(); output=Path(args.publication_root).resolve()
    campaign=read(root/'campaign.json')
    binding=dict(campaign_root=str(root),campaign_sha256=digest(root/'campaign.json'),
        entity=args.entity,project=args.project,owner=args.owner,attempt_label=args.attempt_label)
    require(args.project=='ambi-inner-bench','Transfer curves belong in ambi-inner-bench.')
    with publisher_lock(output):
        path=output/'publication.json'
        if path.exists():
            state=read(path)
            require(all(state.get(key)==value for key,value in binding.items()),'Publication binding differs.')
        else:
            state=dict(schema_version=1,publication_id=uuid.uuid4().hex,**binding,runs={})
            write(path,state)
        for order,setting in enumerate(settings(campaign)):
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
        for checkpoint in campaign['checkpoints']:
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


def collect(root,campaign,state,*,jobs_active):
    """Only receipt-verified complete seed panels can become data points."""
    from slurm.ambi_transfer_curve_campaign import validate_receipt,validate_result,receipt_path
    root=Path(root)
    records={key:records_by_step(row['run_dir']) for key,row in state['runs'].items()}
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
                record=normalize_transfer(campaign,cell,receipt,result,manifest,records[PRIOR_ID][step],receipt_path=marker)
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
    for setting in settings(campaign):
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
    for setting in settings(campaign):
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
            active=True if args.once and not args.gpu_job_id else gpu_jobs_active(args.gpu_job_id)
            snapshot=collect(root,campaign,state,jobs_active=active)
            write(output/'snapshot.json',snapshot)
            try:
                publish_snapshot(campaign,state,snapshot,output,wandb)
                if not layout_checked:
                    previous_layout=state.get('layout')
                    state['layout']=ensure_saved_view(wandb.Api(timeout=30),campaign=campaign,
                        entity=state['entity'],project=state['project'],publication_id=state['publication_id'],
                        receipt_dir=output/'results-layout')
                    layout_checked=True
                    # Reflect a newly installed view in every run, while a
                    # resumed publisher rechecks the view without extra writes.
                    if previous_layout is None:
                        for entry in state['runs'].values():entry.pop('snapshot_sha256',None)
                    write(output/'publication.json',state)
                    publish_snapshot(campaign,state,snapshot,output,wandb)
            except Exception as exc:
                write(output/'publisher-failure.json',dict(error=f'{type(exc).__name__}: {exc}',
                    phase='publication_or_layout',completed=snapshot['completed'],total=snapshot['total']))
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
