"""Pinned Oscar execution of the 575K transfer discovery grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.ambi_benchmark import atomic_json, solver_seed

CHECKPOINT_SHA = '0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042'
METADATA_SHA = '8acc74b7ad4993050a5cc0d3c4c0860441fceb48f36a79540f5e1943e6b3e1d2'
SEEDS = [101, 102, 103]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    atomic_json(Path(path), value, overwrite=False)


def source():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError('Campaign requires a clean tested checkout.')
    return dict(source_commit=git('rev-parse', 'HEAD'), source_tree=git('rev-parse', 'HEAD^{tree}'),
                source_dir=str(ROOT))


def prepare(args):
    checkpoint = args.checkpoint.resolve()
    metadata = Path(str(checkpoint) + '.metadata.json')
    if digest(checkpoint) != CHECKPOINT_SHA or digest(metadata) != METADATA_SHA:
        raise ValueError('The checkpoint or sidecar differs from the audited 575K source.')
    cells = json.loads(subprocess.check_output([sys.executable, str(ROOT/'evaluate_ambi_transfer_campaign.py'),
                                               '--list-cells'], text=True))
    if len(cells) != 252 or {c['H'] for c in cells} != {1, 2, 3} or {c['J'] for c in cells} != {1, 2, 4, 6, 8, 10}:
        raise ValueError('Discovery grid differs from the authorized H/J grid.')
    if sorted(c['index'] for c in cells) != list(range(252)) or len({c['name'] for c in cells}) != 252:
        raise ValueError('Cell identities are not unique and contiguous.')
    # Every mechanism exercises two handoffs. Full-state carry also checks the
    # shorter horizons; high-J fresh controls exercise capacity/update limits.
    smoke = [c['index'] for c in cells if (c['J'] == 2 and
        (c['H'] == 3 or c['arm'] == 'full_state_replay25')) or
        (c['J'] == 10 and c['arm'] == 'rho_a0_c0')]
    args.root.mkdir(parents=True, exist_ok=False)
    campaign = dict(schema_version=1, protocol='inner-sac-transfer-discovery-v1', **source(),
        checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA, metadata_sha256=METADATA_SHA,
        checkpoint_step=575000, seeds=SEEDS, controller_seed=55, max_steps=500,
        H=[1, 2, 3], J=[1, 2, 4, 6, 8, 10], cells=cells, smoke_indices=smoke,
        smoke_seeds=[101, 102], smoke_steps=3, gpu_hardware='L40S',
        historical_reuse='none; matched controls rerun under this implementation',
        selection='exploratory development screen; three paired episodes per cell')
    write(args.root/'campaign.json', campaign)
    print(json.dumps({k: v for k, v in campaign.items() if k != 'cells'}), flush=True)


def validate_result(directory, campaign, cell, *, smoke=False):
    result = read(directory/'results.json')
    episodes = result['episodes']
    expected_seeds = campaign['smoke_seeds'] if smoke else campaign['seeds']
    if len(episodes) != len(expected_seeds) or sorted(e['seed'] for e in episodes) != expected_seeds:
        raise ValueError('Completed result has a different episode panel.')
    limit = campaign['smoke_steps'] if smoke else campaign['max_steps']
    if any(not math.isfinite(e['return']) or not 0 < e['length'] <= limit for e in episodes):
        raise ValueError('Nonfinite return or invalid episode length.')
    if any(e['solver_seed'] != solver_seed(campaign['controller_seed'], 'episode', e['seed']) for e in episodes):
        raise ValueError('Episode solver seed differs from the paired protocol.')
    if any(not math.isfinite(e['control_seconds']) or e['control_seconds'] < 0 for e in episodes):
        raise ValueError('Invalid controller timing.')
    manifest = read(directory/'manifest.json')
    expected = dict(cell_id=cell['name'], arm=cell['arm'], horizon=cell['H'], rounds=cell['J'],
                    checkpoint_sha256=campaign['checkpoint_sha256'], seeds=expected_seeds, smoke=smoke)
    if any(result.get(k) != v or manifest.get(k) != v for k,v in expected.items()):
        raise ValueError('Result or manifest differs from assigned scientific cell.')
    if not result.get('complete') or not result.get('frozen_outer_verified'):
        raise ValueError('Result lacks completion/frozen-state verification.')
    if manifest['source']['git_head'] != campaign['source_commit'] or result['source'] != manifest['source']:
        raise ValueError('Result scientific source differs from campaign.')
    if manifest['max_steps'] != limit or manifest['controller_seed'] != campaign['controller_seed']:
        raise ValueError('Result evaluation protocol differs.')
    if any(e['length'] != limit and not e.get('terminated') and not e.get('truncated') for e in episodes):
        raise ValueError('Incomplete episode was recorded as complete.')
    return result, manifest


def worker(args):
    campaign = read(args.root/'campaign.json')
    actual = source()
    if any(campaign[k] != actual[k] for k in actual):
        raise ValueError('Worker source differs from campaign pin.')
    if not 0 <= args.index < len(campaign['cells']):
        raise ValueError('Worker index outside campaign.')
    cell = campaign['cells'][args.index]
    if cell['index'] != args.index:
        raise ValueError('Cell index mismatch.')
    if args.smoke and args.index not in campaign['smoke_indices']:
        raise ValueError('Unexpected smoke cell.')
    if not args.smoke:
        for index in campaign['smoke_indices']:
            prior = campaign['cells'][index]
            receipt = read(args.root/'smoke'/prior['name']/'worker-completion.json')
            if receipt['source_commit'] != campaign['source_commit'] or receipt['campaign_sha256'] != digest(args.root/'campaign.json'):
                raise ValueError('Production requires all smoke receipts at this commit.')
    directory = args.root/('smoke' if args.smoke else 'settings')/cell['name']
    directory.parent.mkdir(exist_ok=True)
    if directory.exists():
        raise FileExistsError(f'Refusing to overwrite {directory}')
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True).strip()
    if 'L40S' not in hardware:
        raise ValueError('Timing campaign requires an L40S allocation.')
    command = [sys.executable, str(ROOT/'evaluate_ambi_transfer_campaign.py'),
               '--checkpoint', campaign['checkpoint'], '--horizon', str(cell['H']), '--rounds', str(cell['J']),
               '--arm', cell['arm'], '--output-dir', str(directory), '--device', 'cuda',
               '--seeds', *map(str, campaign['smoke_seeds'] if args.smoke else campaign['seeds']),
               '--max-steps', str(campaign['smoke_steps'] if args.smoke else campaign['max_steps'])]
    if args.smoke:
        command.append('--smoke')
    start = time.time()
    try:
        subprocess.run(command, cwd=str(ROOT), check=True)
        result, manifest = validate_result(directory, campaign, cell, smoke=args.smoke)
        write(directory/'worker-completion.json', dict(status='complete', source_commit=campaign['source_commit'],
            campaign_sha256=digest(args.root/'campaign.json'), cell=cell, smoke=args.smoke,
            result_sha256=digest(directory/'results.json'), manifest_sha256=digest(directory/'manifest.json'),
            hardware=hardware, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            elapsed_seconds=time.time()-start, episodes=len(result['episodes'])))
    except BaseException as error:
        directory.mkdir(exist_ok=True)
        write(directory/'worker-failure.json', dict(error_type=type(error).__name__, error=str(error),
            source_commit=campaign['source_commit'], time=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--checkpoint', type=Path, required=True)
    work = sub.add_parser('worker'); work.add_argument('--root', type=Path, required=True)
    work.add_argument('--index', type=int, required=True); work.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else worker)(args)


if __name__ == '__main__':
    main()
