"""Matched eight-condition launch, new-head lifecycle, and gate contract."""
from copy import deepcopy
from itertools import product
import json
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_sweep_campaign as campaign
from tests.test_transfer_sweep_campaign import synthetic_trace, trace_bundle
from tests.test_critic_transfer_presets import resolved, source_context
from tests.test_ambi_root_local_sac import _build_cfg


def test_hidden_sweep_exactly_matches_previous_critic_cells_except_head():
    old = {(c['critic_kind'], c['J'], c['solve_interval']): c
           for c in campaign.cells() if c['transfer_mode'] == 'critic_only'}
    new = campaign.cells(campaign_mode='critic-hidden')
    assert len(new) == len({c['name'] for c in new}) == 8
    assert {(c['critic_kind'], c['J'], c['solve_interval']) for c in new} == set(old)
    assert [(c['J'], c['solve_interval']) for c in new] == [(8, 1)]*2 + [(8, 3)]*2 + [(1, 1)]*2 + [(1, 3)]*2
    for cell in new:
        assert cell['transfer_mode'] == 'critic_hidden'
        expected = old[cell['critic_kind'], cell['J'], cell['solve_interval']]['requested_alg_params']
        assert cell['requested_alg_params'] == {**expected, 'inner_critic_transfer_head': 'random'}
        assert cell['study_protocol'] == ('critic-hidden-transfer-v1' if cell['solve_interval'] == 1
                                         else 'critic-hidden-transfer-hold-h-v1')


@pytest.mark.parametrize('interval,kind,rounds', product((1, 3), ('soft_soft', 'return_return'), (1, 8)))
def test_hidden_sweep_resolves_full_checkpoint_workload(source_context, interval, kind, rounds):
    filename = f"ambi_critic_hidden_transfer_{'hold_h_' if interval == 3 else ''}sweep_575k.json"
    params = resolved(filename, f'{kind}_j{rounds}/critic_hidden_warm', source_context)
    cfg = _build_cfg(**params)
    assert cfg.inner_critic_transfer_head == 'random'
    assert cfg.inner_critic_scope == 'episode' and cfg.inner_actor_scope == 'action'
    assert cfg.inner_rounds == rounds and cfg.inner_solve_interval == interval
    assert cfg.inner_critic_target_initialization == 'online'
    assert cfg.inner_replay_capacity == 3072 and cfg.compile and cfg.compile_strict


def test_plan_is_read_only_and_has_40_new_episodes(capsys):
    value = campaign.plan(SimpleNamespace(matrix=None, campaign_mode='critic-hidden'))
    assert value['new_settings'] == 8 and value['new_episodes'] == 40
    assert value['new_prior_references'] == 0
    assert value['transfer_metadata'] == campaign.HIDDEN_TRANSFER_METADATA
    assert json.loads(capsys.readouterr().out) == value


@pytest.mark.parametrize('field,value', [('inner_actor_lr', .1), ('inner_critic_transfer_head', 'retain'),
                                        ('inner_critic_scope', 'action')])
def test_changed_hidden_setting_rejected(tmp_path, field, value):
    paths = []
    for index, original in enumerate(campaign.HIDDEN_MATRICES):
        matrix = campaign.read(original)
        if index == 0:
            matrix['shared_alg_params'][field] = value
            for group in matrix['comparisons'].values():
                for variant in group['variants'].values():
                    variant['alg_params'].pop(field, None)
        path = tmp_path / original.name
        path.write_text(json.dumps(matrix)); paths.append(path)
    with pytest.raises(ValueError):
        campaign.cells(paths, campaign_mode='critic-hidden')


@pytest.mark.parametrize('decision,phase', [(0, 'initial'), (0, 'decision'), (1, 'decision'),
                                           (3, 'initial'), (3, 'decision'), (6, 'decision')])
def test_head_reset_trace_rejects_wrong_first_later_and_held_flags(tmp_path, decision, phase):
    cell = dict(transfer_mode='critic_hidden', solve_interval=3, J=1)
    rows = list(synthetic_trace(cell))
    row = next(r for r in rows if r['decision_index'] == decision and r['phase'] == phase)
    key = ('decision/' if phase == 'decision' else '') + 'inner_critic_head_reinitialized'
    row['metrics'][key] = 1 - row['metrics'][key]
    bundle = trace_bundle(tmp_path, rows)
    with pytest.raises(ValueError, match='head reset flag'):
        campaign.validate_trace(bundle, cell, seeds=[101, 102], steps=7)


def gate_records():
    check = dict(passed=True, compile_strict=True, compile_fallback=False, source_commit='a'*40,
                 allocation_reuse=True, exact_lifecycle_boundaries=True, optimizer_references_and_reset=True)
    return [dict(passed=True, critic=critic, solve_interval=interval,
                 inner_critic_transfer_head='random', weight_initializer='xavier_uniform',
                 bias_initializer='zeros', reset_timing='every_solve_including_first',
                 deterministic_numerical_parity=deepcopy(check), compiled_stochastic_lifecycle=deepcopy(check))
            for critic, interval in product(('soft', 'return'), (1, 3))]


def write_gate_log(tmp_path, records):
    path = tmp_path/'gate.log'
    path.write_text('\n'.join('.CRITIC_HIDDEN_TRANSFER_CUDA_GATE_REPORT '+json.dumps(row) for row in records))
    return path


def test_gate_seals_four_successful_cases_with_commit(tmp_path):
    records = gate_records(); log = write_gate_log(tmp_path, records)
    reports = campaign.hidden_cuda_reports(log, tmp_path, 'a'*40)
    assert len(reports) == 4
    for path in reports:
        report = campaign.read(path)
        assert report['source_commit'] == 'a'*40 and report['compile_strict']


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'wrong_head', 'fallback', 'wrong_commit'])
def test_gate_rejects_incomplete_or_wrong_reports(tmp_path, failure):
    records = gate_records()
    if failure == 'missing': records.pop()
    if failure == 'duplicate': records[-1] = deepcopy(records[0])
    if failure == 'wrong_head': records[0]['reset_timing'] = 'subsequent_solves'
    if failure == 'fallback': records[0]['compiled_stochastic_lifecycle']['compile_fallback'] = True
    if failure == 'wrong_commit': records[0]['deterministic_numerical_parity']['source_commit'] = 'b'*40
    with pytest.raises(ValueError):
        campaign.hidden_cuda_reports(write_gate_log(tmp_path, records), tmp_path, 'a'*40)
