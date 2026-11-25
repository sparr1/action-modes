"""Gate and launch the Hydra Dog Q10 jobs with the shared training recipe."""

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slurm.ambi_transfer_training import read
from slurm.benchmark_ambi_training_cpu import physical_core_cpus
from slurm.mrsq_q10_suite import check_speed, manifest_path, record_handoff, train, validate_recipe
from slurm.posttanh_dog_trot_resume import benchmark_worker


CAMPAIGN = Path('/data/public/rgao48/ambi/mrsq-q10-dogs-s56-hydra-20261124')
TASKS = ('dog-stand', 'dog-trot', 'dog-walk')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('node-check', 'train', 'record-handoff'))
    parser.add_argument('task', choices=TASKS)
    parser.add_argument('--resume-mode', choices=('new', 'required'))
    parser.add_argument('--drain-seconds', type=int)
    args = parser.parse_args()
    args.seed = 56
    validate_recipe(args.task, args.seed)
    root = CAMPAIGN / f'{args.task}-s56'
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
        check_speed(report, args.task, seed=args.seed)
        print('Hydra Q10 L40S gate passed', args.task, report['measured']['seconds_per_decision'], flush=True)
    elif args.mode == 'record-handoff':
        record_handoff(args.task, root, seed=args.seed)
    else:
        assert args.resume_mode and args.drain_seconds and args.drain_seconds > 0
        segment = os.environ['AMBI_SEGMENT_ID']
        assert segment in (allocation + '.verify', allocation + '.train')
        args.output = root / 'launches' / segment
        report = read(gate)
        check_speed(report, args.task, seed=args.seed)
        train(args, root, expected_runtime=report['runtime'])


if __name__ == '__main__':
    main()
