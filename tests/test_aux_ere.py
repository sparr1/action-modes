"""ERE campaign grid, exact historical reuse, pairing and telemetry guards."""
from copy import deepcopy
import json

import pytest

from slurm.ambi_aux_hj_sweep import MATRIX, cells, read, critic_updates, polyak_comparison
from slurm.ambi_aux_ere import match_identity, baseline_result, REDUCED_SOURCE, C32_SOURCE, validate_update, validate_decision
from slurm.ambi_aux_rollout_batch import reference_cell
from tests.test_aux_round_budget import identity

PATH = MATRIX.with_name('ambi_aux_ere_625k.json')
OLD = MATRIX.with_name('ambi_aux_soft_critic_budget_625k.json')
EXTENSION = MATRIX.with_name('ambi_aux_j68_625k.json')


def reference(cell):
    c, h = critic_updates(cell), cell['H']
    path = EXTENSION if c == 32 else OLD
    name = f'soft_soft_h{h}_j8_jscale' if c == 32 else f'soft_soft_h{h}_j8_c{c}'
    return next(x for x in cells(path) if x['name'] == name), path


def record(cell):
    from utils.eval_series_data import scientific_identity
    old, path = reference(cell)
    r = identity(old, path)
    source = C32_SOURCE if critic_updates(cell) == 32 else REDUCED_SOURCE
    r['identity']['science'] = scientific_identity('AMBITDMPC2/AMBITDMPC2', None, source)
    r.update(record_id='original', metrics={'eval/frozen_state_unchanged': True},
             episodes=[dict(seed=s, solver_seed=55, length=500, truncated_by_evaluator=False,
                            **{'return': float(s)}) for s in range(101, 106)])
    return r


def test_exact_new_grid_has_no_uniform_or_prior_work():
    panel = cells(PATH)
    assert len(panel) == len({c['name'] for c in panel}) == 18
    assert {(c['H'], critic_updates(c), c['params']['inner_ere_final_fraction']) for c in panel} == {
        (h, c, f) for h in (1, 2, 3) for c in (8, 16, 32) for f in (.25, .5)}
    matrix = read(PATH)
    assert matrix['ere_sweep'] and not matrix.get('execution')
    assert matrix['shared_alg_params'] == {**read(OLD)['shared_alg_params'], 'inner_horizon_diagnostics': True}
    assert all(c['J'] == 8 and c['params']['inner_replay_strategy'] == 'ere' for c in panel)


@pytest.mark.parametrize('cell', cells(PATH), ids=lambda c: c['name'])
def test_every_cell_matches_a_pinned_uniform_control(cell):
    old = record(cell)
    spec = identity(cell, PATH); spec.pop('checkpoint')
    baseline = baseline_result(spec, old, cell)
    assert baseline['kind'] == 'ere'
    episodes = [{**e, 'return': e['return']+3} for e in reversed(old['episodes'])]
    comparison = polyak_comparison(episodes, baseline)
    assert comparison['metrics']['comparison/uniform_gain_mean'] == 3
    episodes[0]['solver_seed'] = 56
    with pytest.raises(AssertionError): polyak_comparison(episodes, baseline)
    old['identity']['science'] = {'wrong_source': True}
    with pytest.raises(AssertionError): baseline_result(spec, old, cell)


@pytest.mark.parametrize('key,value', [
    ('inner_actor_updates_per_round', 8), ('inner_critic_updates_per_round', 16),
    ('inner_actor_lr', 6e-4), ('inner_critic_lr', 6e-4), ('inner_critic_target_tau', .1),
    ('inner_rollout_horizon', 2), ('inner_rounds', 4), ('inner_batch_size', 64),
    ('inner_rollouts_per_round', 32), ('inner_replay_reset_each_round', True),
    ('inner_update_timing', 'step'), ('inner_terminal_entropy', 'none'),
    ('inner_replay_strategy', 'uniform'), ('inner_ere_min_rounds', 2),
    ('inner_ere_final_fraction', .5),
])
def test_rejects_unrequested_changes(key, value):
    cell = cells(PATH)[0]
    old, path = reference(cell)
    baseline = identity(old, path); candidate = identity(cell, PATH)
    candidate['identity']['planner']['settings'][key] = value
    with pytest.raises(AssertionError): match_identity(candidate, baseline, .25)


@pytest.mark.parametrize('count', [8, 16, 32])
def test_reused_control_keeps_original_artifacts_and_run_ids(tmp_path, count):
    source = tmp_path/'old'; source.mkdir()
    target = tmp_path/'reference'; target.mkdir()
    cell = next(c for c in cells(PATH) if critic_updates(c) == count)
    previous, _ = reference(cell)
    previous.update(directory=str(source), bundle='existing-bundle', performance_run_id='existing-performance',
                    training_run_id='existing-training', run_dir='existing-registry', baseline={'kind': 'older-test'})
    (source/'publication-completion.json').write_text(json.dumps(dict(
        status='complete', training_run_id='existing-training', metrics={})))
    reused = reference_cell(previous, target, None, record(cell), {'status': 'complete'}, kind='ere')
    assert reused['reused'] and 'baseline' not in reused
    assert reused['name'].endswith('_uniform_reused')
    for key in ('bundle', 'performance_run_id', 'training_run_id', 'run_dir'):
        assert reused[key] == previous[key]
    assert read(target/'publication-completion.json')['metrics']['comparison/uniform_gain_mean'] == 0


def test_telemetry_guards_require_the_actual_window_and_all_draw_counts():
    cell = cells(PATH)[0]  # H3/J8/C32, f=.25.
    event = dict(round_index=8, critic_updates=256, updated_critic=True, metrics={
        'critic_replay_window_rounds': 2, 'critic_replay_window_transitions': 768,
        'critic_replay_window_fraction': .25, 'critic_replay_window_round_fraction': .25,
        'critic_replay_round_age_min': 0, 'critic_replay_round_age_mean': .5,
        'critic_replay_round_age_max': 1, 'critic_replay_newest_round_fraction': .5,
        'critic_replay_batch_unique_fraction': .85})
    validate_update(event, cell)
    wrong = deepcopy(event); wrong['metrics']['critic_replay_window_rounds'] = 8
    with pytest.raises(AssertionError): validate_update(wrong, cell)
    decision = dict(metrics={f'decision/inner_{component}_replay_round_{r}_sample_count': slots*256
                            for component, slots in [('critic', 32), ('actor', 4)] for r in range(1, 9)})
    validate_decision(decision, cell)
    decision['metrics']['decision/inner_critic_replay_round_1_sample_count'] -= 1
    with pytest.raises(AssertionError): validate_decision(decision, cell)
