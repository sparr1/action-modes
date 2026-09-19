"""Strict historical and fresh control pairing for horizon-conditioned SAC."""
from copy import deepcopy
from pathlib import Path

BASELINE_SOURCE = 'fe87ae07b2a7ed751cd6865a3f60b2eae88e6abb'


def match_identity(candidate, baseline, *, baseline_science=None):
    for key in ('backbone', 'protocol'):
        assert candidate['identity'][key] == baseline['identity'][key], key
    assert baseline['identity']['science'] == (baseline_science or candidate['identity']['science'])
    expected = deepcopy(baseline['identity']['planner'])
    settings = expected['settings']
    assert settings.get('inner_horizon_conditioning', 'none') == 'none'
    assert settings['inner_rollout_horizon'] in (2, 3)
    assert settings['inner_rounds'] in (1, 2, 4, 8)
    assert settings['inner_critic_updates_per_round'] == 16
    assert settings['inner_actor_updates_per_round'] == 4
    assert settings['inner_terminal_entropy'] == 'outer'
    assert settings['inner_sac_critic_target'] == 'entropy_augmented'
    settings.update(inner_horizon_conditioning='one_hot',
                    horizon_conditioning_horizon=settings['inner_rollout_horizon'])
    assert candidate['identity']['planner'] == expected


def baseline_result(candidate, baseline, *, historical=False):
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS
    from utils.eval_series_data import scientific_identity
    science = scientific_identity('AMBITDMPC2/AMBITDMPC2', None, BASELINE_SOURCE) if historical else None
    match_identity(candidate, baseline, baseline_science=science)
    assert baseline['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert baseline['checkpoint']['step'] == 625000
    assert baseline['metrics']['eval/frozen_state_unchanged']
    assert sorted(e['seed'] for e in baseline['episodes']) == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in baseline['episodes'])
    return dict(kind='horizon_conditioning', record_id=baseline['record_id'], historical=historical,
                source_comparison=dict(baseline=baseline['identity']['science'], candidate=candidate['identity']['science']),
                comparison_note='Matched C16 unconditioned control; same checkpoint, protocol, H/J and objectives. '
                                'Original compiled mixed-GPU execution; exploratory five-seed comparison. '
                                'Historical controls lack per-horizon minibatch diagnostics.',
                episodes=[{k: e[k] for k in ('seed', 'solver_seed', 'return')} for e in baseline['episodes']])


def checked_control(cell, inventory, *, traces=False):
    from slurm.ambi_aux_hj_sweep import read, digest, validate
    from utils.eval_series_data import load_records
    bundle = Path(cell['bundle'])
    receipt = read(Path(cell['directory'])/'worker-completion.json')
    assert receipt['status'] == 'complete'
    assert digest(bundle/'manifest.json') == receipt['manifest_sha256']
    if traces:
        assert all(digest(bundle/name) == sha for name, sha in receipt['trace_sha256'].items())
    validate(bundle, cell)
    record, = load_records(bundle, inventory_path=inventory)
    return record, receipt


def prepare_campaign(campaign, root, baseline_root):
    from slurm.ambi_aux_hj_sweep import read
    assert baseline_root is not None and len(campaign['cells']) == 10
    historical = read(baseline_root/'campaign.json')
    assert historical['source_commit'] == BASELINE_SOURCE
    previous = {c['name']: c for c in historical['cells']}
    current = {c['name']: c for c in campaign['cells']}
    references = {}
    for cell in campaign['cells']:
        if cell['params']['inner_horizon_conditioning'] == 'none':
            assert cell['J'] == 1
            continue
        candidate = read(root/'specs'/f"sweep__{cell['name']}.json")
        if cell['J'] == 1:
            name = cell['name'].replace('_one_hot', '_none')
            assert name in current
            reference = read(root/'specs'/f'sweep__{name}.json')
            match_identity(candidate, reference)
            cell['horizon_baseline_name'] = name
        else:
            name = cell['name'].removesuffix('_one_hot')
            control = previous[name]
            record, receipt = checked_control(control, campaign['inventory'])
            result = baseline_result(candidate, record, historical=True)
            result.update(bundle=control['bundle'], manifest_sha256=receipt['manifest_sha256'],
                          performance_run_id=control['performance_run_id'])
            cell['baseline'] = result
            references[cell['name']] = control
    assert len(references) == 6
    campaign.update(horizon_conditioning_sweep=True, publisher_workers=4,
                    baseline_campaign=str(baseline_root), horizon_historical_controls=references)


def publication_baseline(campaign, cell, record):
    historical = 'horizon_baseline_name' not in cell
    control = (campaign['horizon_historical_controls'][cell['name']] if historical else
               next(c for c in campaign['cells'] if c['name'] == cell['horizon_baseline_name']))
    baseline, receipt = checked_control(control, campaign['inventory'], traces=True)
    result = baseline_result(record, baseline, historical=historical)
    if historical:
        assert result['record_id'] == cell['baseline']['record_id']
        assert receipt['manifest_sha256'] == cell['baseline']['manifest_sha256']
    result.update(bundle=control['bundle'], manifest_sha256=receipt['manifest_sha256'],
                  performance_run_id=control['performance_run_id'])
    return result
