"""Strict pairing of reduced-C soft/soft solves with the historical C32 table."""
from copy import deepcopy

ORIGINAL_SOURCE = 'ea4892b48d46cfb61ecce11077505aba15bf7564'
EXTENSION_SOURCE = '28964eef209a2aa6deb73b898549ca405f38f293'


def match_critic_identity(candidate, baseline, count, *, baseline_science=None):
    """Change only C, its resolved total, and capacity with no replay eviction."""
    for key in ('backbone', 'protocol'):
        assert candidate['identity'][key] == baseline['identity'][key], key
    assert baseline['identity']['science'] == (baseline_science or candidate['identity']['science'])
    expected = deepcopy(baseline['identity']['planner'])
    settings = expected['settings']
    assert count in (8, 16)
    assert settings['inner_critic_updates_per_round'] == 32
    assert settings['inner_actor_updates_per_round'] == 4
    assert settings['inner_rollouts_per_round'] == 128
    assert settings['inner_critic_target_tau'] == .01
    # Canonical identities omit the default SAC source spellings.
    assert settings.get('inner_critic_source', 'sac') == settings.get('inner_horizon_critic_source', 'sac') == 'sac'
    assert settings['inner_sac_critic_target'] == 'entropy_augmented'
    assert settings['inner_terminal_entropy'] == 'outer'
    assert settings['inner_entropy_enabled'] and settings['inner_temperature_mode'] == 'auto'
    assert settings.get('inner_component_update_order', 'critic_first') == 'critic_first'
    assert not settings.get('inner_replay_reset_each_round', False)
    h, j = settings['inner_rollout_horizon'], settings['inner_rounds']
    assert h in (1, 2, 3) and j in (1, 2, 4, 6, 8)
    assert settings['inner_replay_capacity'] in (2048, 3072)
    assert 128*h*j <= settings['inner_replay_capacity'] and 128*h*j <= 3072
    assert settings['inner_critic_updates_per_action'] == 32*j
    settings['inner_critic_updates_per_round'] = count
    settings['inner_critic_updates_per_action'] = count*j
    settings['inner_replay_capacity'] = 3072
    assert candidate['identity']['planner'] == expected


def historical_critic_baseline(candidate, baseline, count):
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS
    from utils.eval_series_data import scientific_identity
    rounds = baseline['identity']['planner']['settings']['inner_rounds']
    source = EXTENSION_SOURCE if rounds in (6, 8) else ORIGINAL_SOURCE
    science = scientific_identity('AMBITDMPC2/AMBITDMPC2', None, source)
    match_critic_identity(candidate, baseline, count, baseline_science=science)
    assert baseline['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert baseline['checkpoint']['step'] == 625000
    assert baseline['metrics']['eval/frozen_state_unchanged']
    assert sorted(e['seed'] for e in baseline['episodes']) == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in baseline['episodes'])
    return dict(kind='critic_budget', record_id=baseline['record_id'], source_commit=source,
                comparison_note='Original compiled soft/soft protocol; C32 historical reference. '
                                'Only critic count and its resolved total change; tau remains .01. '
                                'Capacity3072 retains all data. Mixed-device results are exploratory.',
                episodes=[{k:e[k] for k in ('seed','solver_seed','return')} for e in baseline['episodes']])
