"""Gate and launch Hydra Q10 jobs with the shared training recipe."""

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slurm.ambi_transfer_training import read
from slurm.benchmark_ambi_training_cpu import physical_core_cpus
from slurm.mrsq_q10_suite import (
    HYDRA_SEED56_TASKS, HYDRA_SEED57_TASKS, MAX_SECONDS_PER_DECISION, check_speed,
    manifest_path, record_handoff, task_path, train, validate_recipe,
)
from slurm.posttanh_dog_trot_resume import benchmark_worker


CAMPAIGNS = {
    56: 'mrsq-q10-dogs-s56-hydra-20261124',
    57: 'mrsq-q10-seed57-hydra-20261125',
}
TASKS = {56: HYDRA_SEED56_TASKS, 57: HYDRA_SEED57_TASKS}
L40_MAX_SECONDS_PER_DECISION = 0.46


def check_hydra_speed(report, task, *, seed):
    gpu = report['runtime']['gpu']
    limits = {'NVIDIA L40S': MAX_SECONDS_PER_DECISION}
    if seed == 57:
        limits['NVIDIA L40'] = L40_MAX_SECONDS_PER_DECISION
    if gpu not in limits:
        raise RuntimeError(f'{gpu} is unsupported for Hydra Q10 seed {seed}')
    check_speed(report, task, seed=seed, gpu=gpu, limit=limits[gpu])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('node-check', 'train', 'record-handoff'))
    parser.add_argument('task', choices=sorted(set(HYDRA_SEED56_TASKS + HYDRA_SEED57_TASKS)))
    parser.add_argument('--seed', type=int, choices=CAMPAIGNS, default=56)
    parser.add_argument('--resume-mode', choices=('new', 'required'))
    parser.add_argument('--drain-seconds', type=int)
    args = parser.parse_args()
    if args.task not in TASKS[args.seed]:
        parser.error(f'{args.task} is unsupported for Hydra Q10 seed {args.seed}')
    validate_recipe(args.task, args.seed)
    campaign = Path('/data/public/rgao48/ambi') / CAMPAIGNS[args.seed]
    root = task_path(args.task, campaign, seed=args.seed)
    allocation = f"{os.environ['SLURM_JOB_ID']}.{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    gate = root / 'node-checks' / f'{allocation}.json'

    if args.mode == 'node-check':
        assert os.environ['WANDB_MODE'] == 'disabled'
        cpus = physical_core_cpus(os.sched_getaffinity(0))
        assert cpus and os.environ['SLURM_CPUS_PER_TASK'] == '1'
        gate.parent.mkdir(parents=True, exist_ok=True)
        benchmark_worker(SimpleNamespace(index=0, native_reference=False, worker_cpus=str(cpus[0]), output=gate),
                         manifest_path=manifest_path(args.task, args.seed))
        report = read(gate)
        check_hydra_speed(report, args.task, seed=args.seed)
        print('Hydra Q10 GPU gate passed', args.task, report['runtime']['gpu'],
              report['measured']['seconds_per_decision'], flush=True)
    elif args.mode == 'record-handoff':
        record_handoff(args.task, root, seed=args.seed)
    else:
        assert args.resume_mode and args.drain_seconds and args.drain_seconds > 0
        segment = os.environ['AMBI_SEGMENT_ID']
        assert segment in (allocation + '.verify', allocation + '.train')
        args.output = root / 'launches' / segment
        report = read(gate)
        check_hydra_speed(report, args.task, seed=args.seed)
        train(args, root, expected_runtime=report['runtime'])


if __name__ == '__main__':
    main()
