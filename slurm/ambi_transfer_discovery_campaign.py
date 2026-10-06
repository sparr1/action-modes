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


def reference_manifest(root, horizons, rounds, *, kind='fresh', expected_matrix=None):
    """Pin completed references without rerunning or rewriting their evidence."""
    if root is None:
        return None
    root = root.resolve()
    historical = read(root/'campaign.json')
    if (historical['checkpoint_sha256'] != CHECKPOINT_SHA or historical['seeds'] != SEEDS
            or historical['controller_seed'] != 55 or historical['max_steps'] != 500):
        raise ValueError('Historical controls differ from the paired checkpoint protocol.')
    wanted_arms = ['rho_a0_c0']
    if kind == 'bernoulli':
        wanted_arms = ['bernoulli_a05_c0', 'bernoulli_a0_c05', 'bernoulli_a05_c05']
        if (historical.get('campaign_kind') != 'bernoulli-weight-transfer-v1'
                or not historical.get('matrix_sha256')
                or digest(historical['matrix_path']) != historical['matrix_sha256']):
            raise ValueError('Historical Bernoulli matrix identity differs.')
        old_matrix = read(historical['matrix_path'])
        if bernoulli_probabilities(old_matrix) != [0.5]:
            raise ValueError('Historical Bernoulli reference must be the completed p=0.5 screen.')
        paired_keys = ('base_matrix', 'base_preset', 'critic_updates', 'actor_updates',
                       'rollouts', 'batch_size', 'diagnostics')
        if expected_matrix is None or any(old_matrix[k] != expected_matrix[k] for k in paired_keys):
            raise ValueError('Historical Bernoulli evaluation or diagnostic protocol differs.')
    elif kind != 'fresh':
        raise ValueError('Unknown historical reference kind.')
    cells = [c for c in historical['cells'] if c['arm'] in wanted_arms
             and c['H'] in horizons and c['J'] in rounds]
    expected = {(h, j, arm) for h in horizons for j in rounds for arm in wanted_arms}
    if len(cells) != len(expected) or {(c['H'], c['J'], c['arm']) for c in cells} != expected:
        raise ValueError('Historical reference panel is incomplete.')
    records = []
    for cell in cells:
        directory = root/'settings'/cell['name']
        result, manifest = validate_result(directory, historical, cell)
        receipt = read(directory/'worker-completion.json')
        hashes = {name: digest(directory/name) for name in ('results.json','manifest.json','worker-completion.json')}
        if (receipt['status'] != 'complete' or receipt['cell'] != cell or receipt['smoke']
                or receipt['source_commit'] != historical['source_commit']
                or receipt['campaign_sha256'] != digest(root/'campaign.json')
                or receipt['result_sha256'] != hashes['results.json']
                or receipt['manifest_sha256'] != hashes['manifest.json']):
            raise ValueError('Historical reference receipt binding differs.')
        records.append(dict(cell=cell, hashes=hashes))
    return dict(root=str(root), campaign_sha256=digest(root/'campaign.json'), kind=kind,
        probability=0.5 if kind == 'bernoulli' else None,
        source_commit=historical['source_commit'], records=records,
        provenance=f'Historical {kind} controls, original source and timing retained; paired environment/solver seeds, not identical visited states.')


def bernoulli_probabilities(matrix):
    """Recognize only the two explicitly versioned probability screens."""
    probabilities = matrix.get('probabilities', [0.5])
    if probabilities not in ([0.5], [0.25, 0.75]):
        raise ValueError('Bernoulli probabilities differ from the authorized screens.')
    expected = {}
    for probability in probabilities:
        token = str(probability).replace('.', '')
        expected.update({
            f'bernoulli_a{token}_c0': dict(actor_bernoulli_p=probability, critic_rho=0.0),
            f'bernoulli_a0_c{token}': dict(actor_rho=0.0, critic_bernoulli_p=probability),
            f'bernoulli_a{token}_c{token}': dict(actor_bernoulli_p=probability, critic_bernoulli_p=probability),
        })
    actual = {name: {k: v for k, v in arm.items() if k != 'description'}
              for name, arm in matrix['arms'].items()}
    if list(actual) != list(expected) or actual != expected:
        raise ValueError('Bernoulli arms differ from the authorized probability grid.')
    return probabilities


def prepare(args):
    checkpoint = args.checkpoint.resolve()
    metadata = Path(str(checkpoint) + '.metadata.json')
    if digest(checkpoint) != CHECKPOINT_SHA or digest(metadata) != METADATA_SHA:
        raise ValueError('The checkpoint or sidecar differs from the audited 575K source.')
    cells = json.loads(subprocess.check_output([sys.executable, str(ROOT/'evaluate_ambi_transfer_campaign.py'),
                                               '--campaign', str(args.matrix.resolve()), '--list-cells'], text=True))
    matrix_path = args.matrix.resolve()
    matrix = read(matrix_path)
    horizons, rounds, arms = matrix['horizons'], matrix['rounds'], list(matrix['arms'])
    expected_count = len(horizons) * len(rounds) * len(arms)
    if (matrix['checkpoint_contract'] != dict(step=575000, sha256=CHECKPOINT_SHA)
            or matrix['seeds'] != SEEDS or matrix['controller_seed'] != 55
            or matrix['max_steps'] != 500 or horizons != [1, 2, 3]):
        raise ValueError('Campaign differs from the audited checkpoint and paired protocol.')
    if (len(cells) != expected_count or {c['H'] for c in cells} != set(horizons)
            or {c['J'] for c in cells} != set(rounds) or {c['arm'] for c in cells} != set(arms)):
        raise ValueError('Evaluator cells differ from the versioned matrix.')
    if sorted(c['index'] for c in cells) != list(range(expected_count)) or len({c['name'] for c in cells}) != expected_count:
        raise ValueError('Cell identities are not unique and contiguous.')
    bernoulli = matrix.get('campaign_kind') == 'bernoulli-weight-transfer-v1'
    reference_rounds = dict(fresh=rounds, bernoulli=rounds)
    if bernoulli:
        probabilities = bernoulli_probabilities(matrix)
        j8_extension = probabilities == [0.5] and rounds == [8]
        if rounds != [1, 2, 4, 6] and not j8_extension:
            raise ValueError('Bernoulli screen differs from the authorized H/J grid.')
        if j8_extension:
            reference_rounds = dict(fresh=[1, 2, 4, 6, 8], bernoulli=[1, 2, 4, 6])
            if matrix.get('reference_rounds') != reference_rounds:
                raise ValueError('J8 extension requires the audited lower-J and fresh-J8 reference panel.')
            smoke = [c['index'] for c in cells if c['H'] == 3]
        elif 'reference_rounds' in matrix:
            raise ValueError('Reference-round overrides are only authorized for the p=0.5 J8 extension.')
        elif probabilities == [0.5]:
            # Every mechanism, all horizons, two transfer boundaries and max J.
            smoke = [c['index'] for c in cells if
                (c['J'] == 2 and (c['H'] == 3 or c['arm'] == 'bernoulli_a05_c05'))
                or (c['H'] == 3 and c['J'] == 6 and c['arm'] == 'bernoulli_a05_c05')]
        else:
            # Six H3/J2 mechanisms plus both probabilities at max J, across H1/H2.
            smoke = [c['index'] for c in cells if (c['H'] == 3 and c['J'] == 2)
                or (c['H'] == 1 and c['J'] == 6 and c['arm'] == 'bernoulli_a025_c025')
                or (c['H'] == 2 and c['J'] == 6 and c['arm'] == 'bernoulli_a075_c075')]
    else:
        if expected_count != 252 or rounds != [1, 2, 4, 6, 8, 10]:
            raise ValueError('Discovery grid differs from the authorized H/J grid.')
        smoke = [c['index'] for c in cells if (c['J'] == 2 and
            (c['H'] == 3 or c['arm'] == 'full_state_replay25')) or
            (c['J'] == 10 and c['arm'] == 'rho_a0_c0')]
    reference = reference_manifest(args.reference_root, horizons, reference_rounds['fresh'])
    bernoulli_reference = reference_manifest(getattr(args, 'bernoulli_reference_root', None),
        horizons, reference_rounds['bernoulli'], kind='bernoulli', expected_matrix=matrix)
    args.root.mkdir(parents=True, exist_ok=False)
    campaign = dict(schema_version=1, protocol='inner-sac-transfer-discovery-v1', **source(),
        checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA, metadata_sha256=METADATA_SHA,
        checkpoint_step=575000, seeds=SEEDS, controller_seed=55, max_steps=500,
        H=horizons, J=rounds, cells=cells, smoke_indices=smoke,
        matrix_path=str(matrix_path), matrix_sha256=digest(matrix_path),
        campaign_kind=matrix.get('campaign_kind', 'transfer-discovery-v1'),
        diagnostics=matrix.get('diagnostics', {}), publication=matrix.get('publication', {}),
        historical_reference=reference,
        historical_references=[r for r in (reference, bernoulli_reference) if r is not None],
        reference_rounds=reference_rounds,
        smoke_seeds=[101, 102], smoke_steps=3, gpu_hardware='L40S',
        historical_reuse=('No new controls; historical discovery references are separate and retain their original provenance.'
            if bernoulli else 'none; matched controls rerun under this implementation'),
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
    if campaign.get('diagnostics', {}).get('enabled'):
        from utils.transfer_campaign_diagnostics import verify_episode_diagnostics
        for episode in episodes:
            verify_episode_diagnostics(episode, campaign['diagnostics'], smoke=smoke)
            if smoke and episode.get('diagnostic_isolation_verified') is not True:
                raise ValueError('Diagnostic smoke lacks exact controller/RNG isolation verification.')
    if campaign.get('matrix_sha256') and manifest.get('campaign_sha256') != campaign['matrix_sha256']:
        raise ValueError('Result campaign matrix differs from the prepared input.')
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
    if campaign.get('matrix_sha256') and digest(campaign['matrix_path']) != campaign['matrix_sha256']:
        raise ValueError('Versioned campaign matrix changed after preparation.')
    command = [sys.executable, str(ROOT/'evaluate_ambi_transfer_campaign.py'),
               '--checkpoint', campaign['checkpoint'], '--horizon', str(cell['H']), '--rounds', str(cell['J']),
               '--arm', cell['arm'], '--output-dir', str(directory), '--device', 'cuda',
               '--seeds', *map(str, campaign['smoke_seeds'] if args.smoke else campaign['seeds']),
               '--max-steps', str(campaign['smoke_steps'] if args.smoke else campaign['max_steps'])]
    if campaign.get('matrix_path'):
        command.extend(['--campaign', campaign['matrix_path']])
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
    prep.add_argument('--reference-root', type=Path, help='Read-only historical discovery root for matched fresh controls.')
    prep.add_argument('--bernoulli-reference-root', type=Path, help='Read-only completed p=0.5 Bernoulli campaign.')
    prep.add_argument('--matrix', type=Path, default=ROOT/'configs/research/ambi_transfer_discovery_575k.json')
    work = sub.add_parser('worker'); work.add_argument('--root', type=Path, required=True)
    work.add_argument('--index', type=int, required=True); work.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else worker)(args)


if __name__ == '__main__':
    main()
