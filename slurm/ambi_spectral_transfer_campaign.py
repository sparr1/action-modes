"""Generate and pin a spectral-transfer campaign for Oscar.

Generation writes only a reviewed configuration, never a job. Preparation
requires the audited checkpoint and a clean, tested source checkout. Worker
execution and publication use the established transfer campaign infrastructure.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slurm import ambi_transfer_discovery_campaign as common
from utils.spectral_transfer import validate_spectral_spec
from utils.transfer_campaign import PROTOCOL, load_campaign, validate_arm

CHECKPOINT_SHA, METADATA_SHA, SEEDS = common.CHECKPOINT_SHA, common.METADATA_SHA, common.SEEDS
DEFAULT_MATRIX = ROOT / 'configs/research/ambi_spectral_transfer_575k.json'
COMPONENTS = {'actor': ('actor',), 'critic': ('critic',), 'joint': ('actor', 'critic')}
METHODS = ('svd', 'activation', 'gradient')
PALETTE = ('#000000', '#0072b2', '#d55e00', '#009e73', '#cc79a7', '#e69f00', '#56b4e9',
           '#332288', '#aa4499', '#44aa99', '#882255', '#999933', '#661100', '#6699cc', '#117733', '#aa7744')


def _unique(values, name, check):
    values = list(values)
    if not values or len(values) != len(set(values)) or any(not check(value) for value in values):
        raise ValueError(f'{name} must be nonempty, unique and valid.')
    return values


def _grid(*, methods, ranks, strengths, rounds, components):
    return dict(
        methods=_unique(methods, 'methods', lambda v: v in METHODS),
        ranks=_unique(ranks, 'ranks', lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0),
        strengths=_unique(strengths, 'strengths', lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1),
        rounds=_unique(rounds, 'rounds', lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0),
        components=_unique(components, 'components', lambda v: v in COMPONENTS))


def _strength_token(value):
    return format(Decimal(str(value)).normalize(), 'f').replace('.', 'p')


def _arms(grid):
    arms = {'fresh': dict(parameter_scope='matrices', actor_rho=0.0, critic_rho=0.0,
                         description='Fresh actor and critic from the frozen prior at every decision.')}
    for component in grid['components']:
        for label, field, value in (('full', 'rho', 1.0), ('blend05', 'rho', .5), ('bernoulli05', 'bernoulli_p', .5)):
            arms[f'matrix_{label}_{component}'] = dict(parameter_scope='matrices',
                **{f'{name}_{field}': value for name in COMPONENTS[component]},
                description=f'{component}: matrix-only {label}; biases, normalization and buffers reset.')
    for method in grid['methods']:
        for rank in grid['ranks']:
            for strength in grid['strengths']:
                for component in grid['components']:
                    name = f'spectral_{method}_r{rank}_s{_strength_token(strength)}_{component}'
                    for norm_matched in (False, True):
                        spec = validate_spectral_spec(dict(method=method, rank=rank,
                            strength=strength, norm_matched=norm_matched))
                        arms[name + ('_norm' if norm_matched else '')] = dict(parameter_scope='matrices',
                            **{f'{part}_spectral': deepcopy(spec) for part in COMPONENTS[component]},
                            description=(f'{component}: {method} candidate rank {rank}, strength {strength}; '
                                + ('dense donor delta with the same per-layer transferred norm.' if norm_matched
                                   else 'carry selected matrix delta; continue dense SAC training.')))
    return arms


def generate_matrix(*, methods=('svd',), ranks=(32,), strengths=(1.0,), rounds=(1, 2, 4, 6),
                    components=('actor', 'critic', 'joint')):
    """Construct a complete paired grid including per-candidate norm controls."""
    grid = _grid(methods=methods, ranks=ranks, strengths=strengths, rounds=rounds, components=components)
    arms = _arms(grid)
    return dict(schema_version=1, protocol=PROTOCOL, family='spectral_transfer',
        campaign_kind='spectral-transfer-v1',
        description='Spectral matrix-transfer screen on the frozen 575k checkpoint. '
            'Same-implementation fresh, full carry, dense blend, Bernoulli and per-layer norm-matched controls. '
            'All arms reset nonmatrix parameters, replay, optimizer and temperature state.',
        base_matrix='ambi_critic_transfer_575k.json', base_preset='return_return/fresh',
        checkpoint_contract=dict(step=575000, sha256=CHECKPOINT_SHA),
        horizons=[1, 2, 3], rounds=grid['rounds'], seeds=list(SEEDS), controller_seed=55,
        max_steps=500, critic_updates=16, actor_updates=4, rollouts=128, batch_size=256,
        spectral_grid={key: value for key, value in grid.items() if key != 'rounds'}, arms=arms,
        diagnostics=dict(enabled=True, decisions=[0, 1, 25, 100, 250, 499],
            stationary_decisions=[25, 250], mc_rollouts=8, action_count=8, state_count=32, fit_steps=4),
        spectral_diagnostics=dict(enabled=True, decisions=[0, 1, 25, 100, 250, 499],
            ranks=sorted(set([1, 2, 4, 8, 16, 32, 64, 128] + grid['ranks'])), state_count=32, action_count=8, mc_rollouts=8),
        spectral_probe=dict(state_count=32, action_count=8, mc_rollouts=8),
        publication=dict(title=('575K truncated SVD matrix transfer' if grid['methods'] == ['svd']
                                else '575K spectral matrix transfer'),
            view_title='575K spectral transfer · paired controls',
            slug_prefix='spectral575', group_prefix='spectral-transfer-575k',
            tags=['575k', 'spectral-transfer', 'matrix-only', 'full-episode', 'development-screen'],
            arm_styles=[[name, name.replace('_', ' '), PALETTE[index % len(PALETTE)]] for index, name in enumerate(arms)]))


def validate_matrix(matrix):
    """Reject protocol drift and missing/confounded candidate controls."""
    expected = dict(schema_version=1, protocol=PROTOCOL, family='spectral_transfer',
        campaign_kind='spectral-transfer-v1', base_matrix='ambi_critic_transfer_575k.json',
        base_preset='return_return/fresh', checkpoint_contract=dict(step=575000, sha256=CHECKPOINT_SHA),
        horizons=[1, 2, 3], seeds=SEEDS, controller_seed=55, max_steps=500,
        critic_updates=16, actor_updates=4, rollouts=128, batch_size=256)
    if any(matrix.get(key) != value for key, value in expected.items()):
        raise ValueError('Spectral campaign differs from the audited checkpoint and paired protocol.')
    if any(key in matrix for key in ('historical_reference', 'historical_references', 'reference_rounds')):
        raise ValueError('Spectral controls must run under the same implementation; historical references are unsupported.')
    raw_grid = matrix.get('spectral_grid', {})
    if set(raw_grid) != {'methods', 'ranks', 'strengths', 'components'}:
        raise ValueError('Spectral grid requires methods, ranks, strengths and components.')
    grid = _grid(**raw_grid, rounds=matrix.get('rounds', []))
    wanted = _arms(grid)
    actual = matrix.get('arms', {})
    strip = lambda arm: {key: value for key, value in arm.items() if key != 'description'}
    if set(actual) != set(wanted) or any(strip(actual[name]) != strip(wanted[name]) for name in wanted):
        raise ValueError('Spectral arms must include all matched controls and per-candidate norm-matched controls.')
    for arm in actual.values():
        validate_arm(arm)
    from utils.spectral_campaign import spectral_settings, validate_probe_settings
    from utils.transfer_campaign_diagnostics import diagnostic_settings
    if not diagnostic_settings(matrix.get('diagnostics')) or not spectral_settings(matrix.get('spectral_diagnostics')):
        raise ValueError('Spectral launch requires enabled basic and spectral diagnostics.')
    if (not {0, 1} <= set(matrix['spectral_diagnostics'].get('decisions', []))
            or not set(grid['ranks']) <= set(matrix['spectral_diagnostics'].get('ranks', []))):
        raise ValueError('Spectral diagnostics must cover the first handoff and every candidate rank.')
    if 'spectral_probe' not in matrix:
        raise ValueError('Spectral launch requires explicit selection probe settings.')
    validate_probe_settings(matrix.get('spectral_probe'))
    return matrix


def prepare(args):
    """Create a new immutable campaign only after every input is validated."""
    matrix_path, checkpoint = args.matrix.resolve(), args.checkpoint.resolve()
    matrix = validate_matrix(load_campaign(matrix_path))
    base_matrix_path = Path(matrix['base_matrix_path'])
    if common.read(base_matrix_path).get('checkpoint_contract') != matrix['checkpoint_contract']:
        raise ValueError('Base matrix checkpoint contract differs from the spectral campaign.')
    metadata = Path(str(checkpoint) + '.metadata.json')
    if common.digest(checkpoint) != CHECKPOINT_SHA or common.digest(metadata) != METADATA_SHA:
        raise ValueError('The checkpoint or sidecar differs from the audited 575K source.')
    pinned_source = common.source()
    expected_sha = getattr(args, 'expected_source', None)
    if expected_sha is not None and pinned_source['source_commit'] != expected_sha:
        raise ValueError('Clean source differs from the requested tested commit.')
    if any(getattr(args, key, None) is not None for key in
           ('reference_root', 'bernoulli_reference_root', 'bernoulli_j8_reference_root', 'bernoulli_j10_reference_root')):
        raise ValueError('Spectral campaigns do not accept historical reference roots.')
    listed = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'evaluate_ambi_transfer_campaign.py'),
        '--campaign', str(matrix_path), '--list-cells'], text=True))
    expected_cells = {(h, j, arm) for h in matrix['horizons'] for j in matrix['rounds'] for arm in matrix['arms']}
    expected_names = {f'h{h}_j{j}_{arm}' for h, j, arm in expected_cells}
    if (len(listed) != len(expected_cells) or {(cell['H'], cell['J'], cell['arm']) for cell in listed} != expected_cells
            or {cell['name'] for cell in listed} != expected_names
            or [cell['index'] for cell in listed] != list(range(len(expected_cells)))
            or any(cell['name'] != f"h{cell['H']}_j{cell['J']}_{cell['arm']}" for cell in listed)):
        raise ValueError('Evaluator cells are not the unique contiguous configured grid.')
    max_h, max_j = max(matrix['horizons']), max(matrix['rounds'])
    smoke = [cell['index'] for cell in listed if (cell['H'] == max_h and cell['J'] == max_j)
             or (cell['arm'] == 'fresh' and cell['H'] == min(matrix['horizons']) and cell['J'] == min(matrix['rounds']))]
    campaign = dict(schema_version=1, protocol=PROTOCOL, **pinned_source,
        checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA, metadata_sha256=METADATA_SHA,
        metadata_path=str(metadata),
        checkpoint_step=575000, seeds=list(SEEDS), controller_seed=55, max_steps=500,
        H=matrix['horizons'], J=matrix['rounds'], cells=listed, arms=deepcopy(matrix['arms']),
        family='spectral_transfer', campaign_kind='spectral-transfer-v1',
        matrix_path=str(matrix_path), matrix_sha256=common.digest(matrix_path),
        base_matrix_path=str(base_matrix_path), base_matrix_sha256=common.digest(base_matrix_path),
        smoke_indices=smoke, smoke_seeds=[101, 102], smoke_steps=3, gpu_hardware='L40S',
        diagnostics=deepcopy(matrix['diagnostics']), spectral_diagnostics=deepcopy(matrix['spectral_diagnostics']),
        spectral_probe=deepcopy(matrix['spectral_probe']), spectral_grid=deepcopy(matrix['spectral_grid']),
        publication=deepcopy(matrix['publication']), historical_reference=None, historical_references=[],
        historical_reuse='none; all controls rerun with matrix-only transfer under this implementation',
        selection='exploratory development screen; three paired full episodes per cell',
        smoke_contract='Every arm at max H/J and fresh at min H/J; two seeds, three decisions, exact diagnostic/RNG isolation.')
    from utils.spectral_campaign import spectral_settings, validate_probe_settings
    from utils.transfer_campaign_diagnostics import diagnostic_settings
    campaign.update(diagnostics=diagnostic_settings(matrix['diagnostics']),
        spectral_diagnostics=spectral_settings(matrix['spectral_diagnostics']),
        spectral_probe=validate_probe_settings(matrix['spectral_probe']))
    args.root.mkdir(parents=True, exist_ok=False)
    common.write(args.root / 'campaign.json', campaign)
    print(json.dumps({key: value for key, value in campaign.items() if key not in {'cells', 'arms'}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    generate = sub.add_parser('generate', help='Write a configuration only; does not prepare or submit jobs.')
    generate.add_argument('--output', required=True, type=Path)
    generate.add_argument('--methods', choices=METHODS, nargs='+', default=['svd'])
    generate.add_argument('--ranks', type=int, nargs='+', default=[32])
    generate.add_argument('--strengths', type=float, nargs='+', default=[1.0])
    generate.add_argument('--rounds', type=int, nargs='+', default=[1, 2, 4, 6])
    generate.add_argument('--components', choices=tuple(COMPONENTS), nargs='+', default=list(COMPONENTS))
    prep = sub.add_parser('prepare')
    prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--checkpoint', type=Path, required=True)
    prep.add_argument('--matrix', type=Path, default=DEFAULT_MATRIX)
    prep.add_argument('--expected-source', help='Require this exact tested Git commit in addition to a clean checkout.')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    else:
        matrix = generate_matrix(**{key: getattr(args, key) for key in ('methods', 'ranks', 'strengths', 'rounds', 'components')})
        validate_matrix(matrix)
        common.write(args.output, matrix)
        print(json.dumps(dict(output=str(args.output.resolve()), arms=len(matrix['arms']),
            cells=len(matrix['horizons']) * len(matrix['rounds']) * len(matrix['arms']), submitted=False)))


if __name__ == '__main__':
    main()
