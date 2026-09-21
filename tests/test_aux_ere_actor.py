"""Exact single-condition scope, controls and paired publication."""
from copy import deepcopy
import pytest
from slurm.ambi_aux_hj_sweep import cells, MATRIX, polyak_comparison
from slurm.ambi_aux_ere import baseline_result, validate_update
from slurm.ambi_aux_ere_actor import match_ere_control, ERE_SOURCE
from tests.test_aux_ere import record
from tests.test_aux_round_budget import identity
from utils.eval_series_data import scientific_identity

PATH = MATRIX.with_name('ambi_aux_ere_actor_625k.json')


def test_one_new_condition_matches_both_existing_controls():
    cell, = cells(PATH)
    assert (cell['H'], cell['J'], cell['params']['inner_critic_updates_per_round']) == (3,8,16)
    assert cell['params']['inner_ere_actor'] is False
    candidate = identity(cell, PATH)
    uniform = record(cell)
    baseline_result(candidate, uniform, cell)
    both = deepcopy(candidate)
    both['identity']['planner']['settings'].pop('inner_ere_actor')
    both['identity']['science'] = scientific_identity('AMBITDMPC2/AMBITDMPC2', None, ERE_SOURCE)
    match_ere_control(candidate, both)
    wrong = deepcopy(candidate); wrong['identity']['planner']['settings']['inner_actor_lr'] = .001
    with pytest.raises(AssertionError): match_ere_control(wrong, both)
    episodes = [{**e, 'return': e['return']+2} for e in uniform['episodes']]
    comparison = polyak_comparison(episodes, dict(kind='ere_both', episodes=uniform['episodes']))
    assert comparison['metrics']['comparison/ere_both_gain_mean'] == 2


def test_actor_telemetry_requires_full_replay():
    cell, = cells(PATH)
    event = dict(round_index=8, actor_updates=32, updated_actor=True, metrics={
        'actor_replay_window_rounds': 8, 'actor_replay_window_transitions': 3072,
        'actor_replay_window_fraction': 1., 'actor_replay_window_round_fraction': 1.,
        'actor_replay_round_age_min': 0, 'actor_replay_round_age_mean': 3.5,
        'actor_replay_round_age_max': 7, 'actor_replay_newest_round_fraction': .125,
        'actor_replay_batch_unique_fraction': .95})
    validate_update(event, cell)
    event['metrics']['actor_replay_window_rounds'] = 2
    with pytest.raises(AssertionError): validate_update(event, cell)
