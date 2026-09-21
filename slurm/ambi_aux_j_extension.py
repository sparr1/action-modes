"""Strict historical J8 pairing for soft/soft J10/J12/J16 extensions."""
from copy import deepcopy

SOURCES = {16: 'fe87ae07b2a7ed751cd6865a3f60b2eae88e6abb',
           32: '28964eef209a2aa6deb73b898549ca405f38f293'}


def match_j_identity(candidate, baseline, rounds, *, baseline_science=None):
    for key in ('backbone', 'protocol'):
        assert candidate['identity'][key] == baseline['identity'][key], key
    assert baseline['identity']['science'] == (baseline_science or candidate['identity']['science'])
    expected = deepcopy(baseline['identity']['planner'])
    settings = expected['settings']
    h, c = settings['inner_rollout_horizon'], settings['inner_critic_updates_per_round']
    assert h in (1, 2, 3) and c in SOURCES and (c == 16 or h == 3)
    assert settings['inner_rounds'] == 8 and rounds in (10, 12, 16)
    assert rounds != 10 or c == 16
    assert settings['inner_rollouts_per_round'] == 128 and settings['inner_actor_updates_per_round'] == 4
    assert settings['inner_replay_capacity'] == 3072
    assert settings.get('inner_replay_strategy', 'uniform') == 'uniform'
    assert not settings.get('inner_replay_reset_each_round', False)
    assert settings.get('inner_component_update_order', 'critic_first') == 'critic_first'
    assert settings.get('inner_update_timing', 'round') == 'round'
    assert settings.get('inner_critic_source', 'sac') == settings.get('inner_horizon_critic_source', 'sac') == 'sac'
    assert settings['inner_sac_critic_target'] == 'entropy_augmented'
    assert settings['inner_terminal_entropy'] == 'outer'
    assert settings['inner_entropy_enabled'] and settings['inner_temperature_mode'] == 'auto'
    assert 128*h*8 <= 3072 and 128*h*rounds <= 6144
    for key, per_round in (
        ('inner_model_step_budget', 128*h), ('inner_critic_updates_per_action', c),
        ('inner_actor_updates_per_action', 4), ('inner_temperature_updates_per_action', 4),
    ):
        assert settings[key] == 8*per_round, key
        settings[key] = rounds*per_round
    settings.update(inner_rounds=rounds, inner_replay_capacity=6144)
    assert candidate['identity']['planner'] == expected


def historical_j8_baseline(candidate, baseline, rounds):
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS
    from utils.eval_series_data import scientific_identity
    source = SOURCES[baseline['identity']['planner']['settings']['inner_critic_updates_per_round']]
    match_j_identity(candidate, baseline, rounds,
                     baseline_science=scientific_identity('AMBITDMPC2/AMBITDMPC2', None, source))
    assert baseline['checkpoint']['sha256'] == CHECKPOINT_SHA and baseline['checkpoint']['step'] == 625000
    assert baseline['metrics']['eval/frozen_state_unchanged']
    assert sorted(e['seed'] for e in baseline['episodes']) == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in baseline['episodes'])
    return dict(kind='j_extension', record_id=baseline['record_id'], source_commit=source,
                comparison_note='Matched historical J8 at the same H and C. Only J, its derived totals, '
                                'and non-evicting replay capacity change. Original compiled mixed-GPU protocol.',
                episodes=[{k:e[k] for k in ('seed','solver_seed','return')} for e in baseline['episodes']])
