"""Strict historical J4 pairing for the original H/J table's J6/J8 extension."""
from copy import deepcopy

ORIGINAL_SOURCE = 'ea4892b48d46cfb61ecce11077505aba15bf7564'


def match_round_identity(candidate, baseline, rounds, *, baseline_science=None):
    """Allow J, its resolved totals, and capacity needed to retain all data."""
    for key in ('backbone', 'protocol'):
        assert candidate['identity'][key] == baseline['identity'][key], key
    assert baseline['identity']['science'] == (baseline_science or candidate['identity']['science'])
    expected = deepcopy(baseline['identity']['planner'])
    settings = expected['settings']
    assert settings['inner_rounds'] == 4 and rounds in (6, 8)
    assert settings['inner_critic_updates_per_round'] == 32
    assert settings['inner_actor_updates_per_round'] == 4
    assert settings['inner_rollouts_per_round'] == 128
    assert settings['inner_replay_capacity'] == 2048
    assert settings.get('inner_component_update_order', 'critic_first') == 'critic_first'
    assert not settings.get('inner_replay_reset_each_round', False)
    h = settings['inner_rollout_horizon']
    assert 128*h*4 <= 2048 and 128*h*rounds <= 3072
    for key, per_round in (
        ('inner_model_step_budget', 128*h),
        ('inner_critic_updates_per_action', 32),
        ('inner_actor_updates_per_action', 4),
        ('inner_temperature_updates_per_action', 4),
    ):
        assert settings[key] == 4*per_round, key
        settings[key] = rounds*per_round
    settings['inner_rounds'] = rounds
    settings['inner_replay_capacity'] = 3072
    assert candidate['identity']['planner'] == expected


def historical_baseline(candidate, baseline, rounds):
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS
    from utils.eval_series_data import scientific_identity
    old_science = scientific_identity('AMBITDMPC2/AMBITDMPC2', None, ORIGINAL_SOURCE)
    match_round_identity(candidate, baseline, rounds, baseline_science=old_science)
    assert candidate['checkpoint']['sha256'] == baseline['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert candidate['checkpoint']['step'] == baseline['checkpoint']['step'] == 625000
    assert baseline['metrics']['eval/frozen_state_unchanged']
    assert sorted(e['seed'] for e in baseline['episodes']) == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in baseline['episodes'])
    return dict(kind='round_budget', record_id=baseline['record_id'],
                source_commit=ORIGINAL_SOURCE,
                comparison_note='Original compiled H/J protocol; historical mixed-GPU results. '
                                'Replay capacity grows from 2048 to 3072 to preserve full retention.',
                episodes=[{k: e[k] for k in ('seed', 'solver_seed', 'return')} for e in baseline['episodes']])
