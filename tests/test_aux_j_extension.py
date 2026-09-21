"""J12/J16 scope, paired J8 identities, and full replay retention."""
from copy import deepcopy

import pytest
import torch

from slurm.ambi_aux_hj_sweep import MATRIX, cells, read, polyak_comparison
from slurm.ambi_aux_j_extension import SOURCES, match_j_identity, historical_j8_baseline
from tests.test_aux_round_budget import identity
from tests.test_ambi_root_local_sac import _tiny_component_model

PATH = MATRIX.with_name('ambi_aux_soft_j1216_625k.json')


def baseline(cell):
    c = cell['params']['inner_critic_updates_per_round']
    path = MATRIX.with_name('ambi_aux_soft_critic_budget_625k.json' if c == 16 else 'ambi_aux_j68_625k.json')
    return next(x for x in cells(path) if x['H'] == cell['H'] and x['J'] == 8
                and x['name'].startswith('soft_soft_')
                and x['params'].get('inner_critic_updates_per_round', 32) == c), path


def test_exact_new_grid_and_prior_is_reference_only():
    from utils.ambi_research import load_preset_matrix
    matrix = load_preset_matrix(PATH)
    panel = cells(PATH)
    assert len(panel) == 8
    assert {(x['H'], x['J'], x['params']['inner_critic_updates_per_round']) for x in panel} == (
        {(h,j,16) for h in (1,2,3) for j in (12,16)} | {(3,j,32) for j in (12,16)})
    assert matrix['comparisons']['sweep']['variants']['prior']['alg_params']['inner_operator'] == 'none'
    assert not matrix.get('execution')
    assert matrix['evaluation']['seeds'] == [101,102,103,104,105]
    assert matrix['evaluation']['controller_seed'] == 55 and matrix['evaluation']['max_steps'] == 500


@pytest.mark.parametrize('cell', cells(PATH), ids=lambda x:x['name'])
def test_only_j_totals_and_nonevicting_capacity_change(cell):
    old, path = baseline(cell)
    before = {**read(path)['shared_alg_params'], **old['params']}
    after = {**read(PATH)['shared_alg_params'], **cell['params']}
    before.update(inner_rounds=cell['J'], inner_replay_capacity=6144,
                  inner_component_update_order='critic_first',
                  inner_critic_updates_per_round=cell['params']['inner_critic_updates_per_round'])
    assert before == after
    match_j_identity(identity(cell, PATH), identity(old, path), cell['J'])


@pytest.mark.parametrize('cell', cells(PATH), ids=lambda x:x['name'])
def test_historical_pairing_and_publication_metric(cell):
    from utils.eval_series_data import scientific_identity
    old, path = baseline(cell)
    record = identity(old, path)
    record['identity']['science'] = scientific_identity('AMBITDMPC2/AMBITDMPC2', None,
        SOURCES[cell['params']['inner_critic_updates_per_round']])
    record.update(record_id='historical-j8', metrics={'eval/frozen_state_unchanged':True},
                  episodes=[dict(seed=s,solver_seed=55,length=500,truncated_by_evaluator=False,
                                 **{'return':float(s)}) for s in range(101,106)])
    candidate = identity(cell, PATH)
    reference = historical_j8_baseline(candidate, record, cell['J'])
    episodes = [{**e,'return':e['return']+3} for e in reversed(record['episodes'])]
    assert polyak_comparison(episodes,reference)['metrics']['comparison/j8_gain_mean'] == 3
    for key, value in [('inner_critic_target_tau',.1), ('inner_actor_lr',.001),
                       ('inner_replay_strategy','ere'), ('inner_model_step_budget',1),
                       ('inner_component_update_order','interleaved')]:
        wrong = deepcopy(candidate); wrong['identity']['planner']['settings'][key] = value
        with pytest.raises(AssertionError): historical_j8_baseline(wrong,record,cell['J'])
    episodes[0]['solver_seed'] = 56
    with pytest.raises(AssertionError): polyak_comparison(episodes,reference)
    record['identity']['science'] = {'unknown': True}
    with pytest.raises(AssertionError): historical_j8_baseline(candidate,record,cell['J'])


def test_h3_j16_keeps_all_rounds_and_exact_update_budgets(monkeypatch):
    holder = _tiny_component_model(inner_rounds=16, inner_rollouts_per_round=128,
        inner_rollout_horizon=3, inner_replay_capacity=6144, inner_batch_size=256,
        inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
        inner_finite_horizon=True, inner_component_update_order='critic_first')
    try:
        engine = holder.agent.inner_engine
        sample = engine._sample_batch; last = []
        def checked(indices=None):
            batch = sample(indices); replay = engine.state.replay
            assert replay.size == replay.next_sample_id
            if replay.size == 6144:
                torch.testing.assert_close(replay.sample_id[:6144].sort().values, torch.arange(6144))
                last.append(batch['sample_ids'].clone())
            return batch
        monkeypatch.setattr(engine, '_sample_batch', checked)
        engine.reset_for_evaluation(3818519826)
        holder.agent.act(torch.tensor([.7,.3,-.2]),t0=True,eval_mode=True)
        assert len(last) == 20 and any(bool((ids < 384).any()) for ids in last)
        metrics = holder.agent.last_inner_metrics
        assert metrics['inner_model_steps'] == metrics['inner_buffer_size'] == 6144
        assert metrics['inner_critic_optimizer_steps'] == 256
        assert metrics['inner_actor_optimizer_steps'] == metrics['inner_temperature_optimizer_steps'] == 64
    finally:
        holder.close()
