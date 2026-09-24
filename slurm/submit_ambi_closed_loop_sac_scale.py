"""Journaled Oscar dispatch and smoke-based timing for the 575k SAC scale sweep.

Run on an Oscar login node from the clean, Git-synchronized checkout. Each
phase refuses an existing intent journal, including an uncertain submission.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_aux_hj_sweep import digest, read, write

SCRIPT = ROOT / 'slurm/run_ambi_closed_loop_sac_scale_oscar.sbatch'
EXCLUDE = 'gpu2708,gpu3003,gpu3105'


def command(args):
    return subprocess.check_output(args, text=True).strip()


def source_check(expected):
    actual = command(['git', 'rev-parse', 'HEAD'])
    if actual != expected or command(['git', 'status', '--porcelain', '--untracked-files=normal']):
        raise RuntimeError('Dispatch requires the exact expected commit and a clean checkout')
    return actual


def submit(journal, label, options, exports):
    """Persist intent before sbatch; an ambiguous outcome is never retried blindly."""
    data = read(journal)
    if label in data['jobs']:
        raise RuntimeError(f'Submission already attempted: {label}')
    export = ','.join(['ALL', *(f'{k}={v}' for k, v in exports.items())])
    if any(',' in str(v) or '\n' in str(v) for v in exports.values()):
        raise ValueError('Slurm export values cannot contain commas or newlines')
    argv = ['sbatch', '--parsable', *options, '--export='+export, str(SCRIPT)]
    data['jobs'][label] = dict(status='intent', argv=argv, timestamp=time.time())
    write(journal, data)
    result = subprocess.run(argv, text=True, capture_output=True)
    job = result.stdout.strip().split(';')[0]
    data['jobs'][label].update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode or not job.isdigit():
        data['jobs'][label]['status'] = 'uncertain_or_failed'
        write(journal, data)
        raise RuntimeError(f'sbatch did not confirm {label}; inspect {journal}')
    data['jobs'][label].update(status='submitted', job_id=job)
    write(journal, data)
    print(f'{label}: {job}', flush=True)
    return job


def estimate(campaign):
    """Use full-budget smoke measurements; queue delay is deliberately separate."""
    from slurm.ambi_closed_loop_sac_scale_publish import validate_scope
    validate_scope(campaign)
    expected_smoke = [i for i,c in enumerate(campaign['cells']) if c['H'] in (1,3) and c['J'] == 10]
    if sorted(campaign['smoke_indices']) != expected_smoke or len(expected_smoke) != 4:
        raise ValueError('Timing requires all four H1/H3, J10, P1/P5 smoke cells')
    if campaign['smoke_steps'] != 8:
        raise ValueError('Expected the eight-decision timing protocol')
    measured = {}
    for index in campaign['smoke_indices']:
        cell = campaign['cells'][index]
        path = Path(cell['directory']).parent / 'smoke' / cell['name'] / 'worker-completion.json'
        receipt = read(path)
        assert receipt['status'] == 'complete' and receipt['smoke']
        assert receipt['cell'] == cell['name'] and receipt['selector'] == cell['selector']
        assert all(receipt[k] == cell[k] for k in ('H','J','N','B','G','P','T'))
        assert receipt['checkpoint_step'] == cell['checkpoint_step'] == 575000
        assert receipt['checkpoint_sha256'] == campaign['checkpoint_sha256']
        bundle = Path(receipt['bundle'])
        assert bundle == path.parent/'bundle'
        assert digest(bundle/'manifest.json') == receipt['manifest_sha256']
        assert all(digest(bundle/name) == sha for name,sha in receipt['trace_sha256'].items())
        manifest = read(bundle/'manifest.json')
        assert manifest['code']['commit'] == campaign['source_commit'] and manifest['code']['dirty'] is False
        assert manifest['runs'][0]['selector'] == cell['selector']
        timing = receipt['timing']
        assert timing['decisions'] == campaign['smoke_steps']
        for key in ('control_seconds','probe_seconds','serialization_seconds','initialization_seconds',
                    'warmup_including_compile_seconds'):
            value = timing[key]
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid measured timing: {key}')
        if timing['control_seconds'] <= 0:
            raise ValueError('Timing smoke contains no measured control work')
        measured[(cell['H'], cell['P'])] = dict(cell=cell, timing=timing)
    predictions = []
    for index,cell in enumerate(campaign['cells']):
        candidates = [v for (h,p),v in measured.items() if p == cell['P'] and (h == cell['H'] or cell['H'] == 2)]
        if not candidates:
            raise ValueError('No matching full-budget smoke for this setting')
        # H2 conservatively uses the slower H1/H3 observation for its schedule.
        predictions_for_cell = []
        for sample in candidates:
            t = sample['timing']; count = t['decisions']
            per_decision = (t['control_seconds'] + t['probe_seconds'] + t['serialization_seconds']) / count
            startup = t['initialization_seconds'] + t['warmup_including_compile_seconds']
            seconds = startup + per_decision * 2500 * cell['J'] / sample['cell']['J']
            predictions_for_cell.append(seconds)
        seconds = max(predictions_for_cell)
        predictions.append(dict(index=index, setting=cell['name'], H=cell['H'], J=cell['J'], P=cell['P'],
            estimated_seconds=seconds, time_limit_minutes=max(20, math.ceil((seconds*1.35+300)/60))))
    slots = [(0.,i) for i in range(12)]; heapq.heapify(slots)
    for row in sorted(predictions,key=lambda r:r['estimated_seconds'],reverse=True):
        elapsed,slot = heapq.heappop(slots)
        heapq.heappush(slots,(elapsed+row['estimated_seconds'],slot))
    return dict(settings=predictions, gpu_hours=sum(r['estimated_seconds'] for r in predictions)/3600,
        ideal_12_gpu_elapsed_hours=max(x[0] for x in slots)/3600,
        queue_delay_included=False, timing_basis='Eight full-budget measured decisions after compilation; J-scaled, H2 uses slower H1/H3.',
        limitations='Early-episode smoke timing; later-state variation, publication, resource contention and queue delays remain uncertain.')


def dispatch(args):
    import os
    os.chdir(ROOT)
    source_check(args.sha)
    parent = args.root.parent; parent.mkdir(parents=True,exist_ok=True)
    logs = parent/'slurm'; logs.mkdir(exist_ok=True)
    journal = parent/f'dispatch-{args.phase}.json'
    if journal.exists():
        raise RuntimeError(f'Existing dispatch journal requires inspection: {journal}')
    base = dict(CAMPAIGN_ROOT=str(args.root),EXPECTED_ACTION_MODES_SHA=args.sha)
    common = ['--output='+str(logs/'%x-%A_%a.out'), '--error='+str(logs/'%x-%A_%a.err')]
    campaign = None
    if args.phase != 'prepare':
        campaign = read(args.root/'campaign.json')
        assert campaign['source_commit'] == args.sha
    prediction = estimate(campaign) if args.phase == 'production' else None
    with journal.open('x') as stream:
        json.dump(dict(phase=args.phase,source_commit=args.sha,campaign_root=str(args.root),jobs={}),stream)
    if args.phase == 'prepare':
        submit(journal,'prepare',[*common,'--partition=batch','--cpus-per-task=4','--mem=16G','--time=00:20:00'],
               dict(base,EVAL_MODE='prepare',CHECKPOINT_INVENTORY=str(args.inventory)))
    elif args.phase == 'smoke':
        indices = ','.join(str(i) for i in campaign['smoke_indices'])
        submit(journal,'smoke',[*common,'--partition=gpu-debug','--gres=gpu:l40s:1',
            '--cpus-per-task=4','--mem=32G','--time=00:20:00','--array='+indices+'%2','--exclude='+EXCLUDE],
            dict(base,EVAL_MODE='worker',EVAL_SMOKE='1'))
    else:
        write(args.root/'timing-estimate.json',prediction)
        job_ids = []
        # Longer cells enter the queue first; shorter limits allow backfill.
        groups = {}
        for row in prediction['settings']:
            groups.setdefault((row['J'],row['P']),[]).append(row)
        for (j,p),rows in sorted(groups.items(),key=lambda item:(-item[0][0],item[0][1])):
            minutes = max(row['time_limit_minutes'] for row in rows)
            indices = ','.join(str(row['index']) for row in rows)
            job_ids.append(submit(journal,f'production-j{j}-p{p}',[*common,
                '--job-name='+f'sac575-j{j}-p{p}', '--partition=gpu','--qos=pri-gpu+',
                '--gres=gpu:l40s:1','--cpus-per-task=4','--mem=32G',f'--time={minutes}',
                '--array='+indices,'--exclude='+EXCLUDE],dict(base,EVAL_MODE='worker',EVAL_SMOKE='0')))
        write(args.root/'submission.json',dict(gpu_job_ids=job_ids,source_commit=args.sha,
            settings=len(prediction['settings']),maximum_useful_gpus=12,created=time.time()))
        watch_hours = max(12,math.ceil(prediction['ideal_12_gpu_elapsed_hours']*2+6))
        submit(journal,'publisher',[*common,'--job-name=sac575-publish','--partition=batch',
            '--cpus-per-task=8','--mem=64G',f'--time={watch_hours*60}'],dict(base,EVAL_MODE='watch'))
        print(json.dumps({k:v for k,v in prediction.items() if k != 'settings'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','smoke','production'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--sha',required=True)
    parser.add_argument('--inventory',type=Path,default=Path('/oscar/scratch/rgao48/ambi/aux-return-mppi/20260921-other-backbones/target10p5_shared/checkpoint-manifest.json'))
    dispatch(parser.parse_args())
