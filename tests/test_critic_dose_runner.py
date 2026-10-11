"""Critic-dose protocol guards, paired streams, and GPU smoke failure paths."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import evaluate_ambi_critic_dose as runner
from evaluate_ambi_critic_bypass import fingerprint
from utils.ambi_benchmark import solver_seed

CONFIG = Path(__file__).resolve().parents[1] / 'configs/research/ambi_800k_critic_dose.json'


def test_only_eighty_new_tasks_and_full_budget_smoke():
    full, smoke = runner.load_config(CONFIG), runner.load_config(CONFIG, smoke=True)
    tasks = runner.task_list(full)
    assert len(tasks) == len({t['task_id'] for t in tasks}) == 80
    assert {t['critic_updates_per_round'] for t in tasks} == {32, 64}
    assert {t['selector'] for t in tasks} == {'actor_mean', 'learned_q'}
    for arm in runner.ARMS:
        assert [t['env_seed'] for t in tasks if t['arm'] == arm] == list(range(101, 121))
    assert len(runner.task_list(smoke)) == 4
    assert smoke['max_steps'] == 4 and smoke['diagnostic_decisions'] == [0, 3]
    for key in ('horizon', 'rounds', 'rollouts_per_round', 'actor_updates_per_round',
                'batch_size', 'critic_updates_per_round', 'mc_rollouts', 'validation_rollouts',
                'reference_campaign_id', 'reference_snapshot_sha256'):
        assert smoke[key] == full[key]
    assert fingerprint(full) != fingerprint(smoke)


@pytest.mark.parametrize('key,value', [('rounds', 8), ('critic_updates_per_round', [16, 32]),
    ('actor_updates_per_round', 8), ('batch_size', 128), ('max_steps', 100),
    ('seeds', list(range(101, 106))), ('diagnostic_decisions', [0, 499]),
    ('reference_snapshot_sha256', ''), ('reference_campaign_id', 'bad')])
def test_contract_rejects_silent_science_or_reference_change(tmp_path, key, value):
    cfg = json.loads(CONFIG.read_text()); cfg[key] = value
    path = tmp_path / 'config.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): runner.load_config(path)


def test_dose_resolution_changes_only_critic_updates_and_does_not_mutate_base():
    base = dict(algorithm_config=dict(alg_params=dict(inner_rollouts_per_round=128,
        inner_critic_updates_per_round=16, inner_actor_updates_per_round=4, inner_batch_size=256,
        inner_replay_capacity=3072)))
    before = deepcopy(base)
    c32, c64 = runner.resolved_dose(base, 32), runner.resolved_dose(base, 64)
    left, right = c32['algorithm_config']['alg_params'], c64['algorithm_config']['alg_params']
    assert {key for key in left if left[key] != right[key]} == {'inner_critic_updates_per_round'}
    assert (left['inner_critic_updates_per_round'], right['inner_critic_updates_per_round']) == (32, 64)
    assert left['inner_rollout_horizon'] == 1 and left['inner_rounds'] == 6
    assert left['inner_actor_updates_per_round'] == 4 and left['inner_batch_size'] == 256
    assert base == before
    for bad in (16, True, 128):
        with pytest.raises(ValueError): runner.resolved_dose(base, bad)


def test_root_streams_match_c16_and_have_no_dose_or_selector_dependency():
    expected_episode = solver_seed(55, 'episode', 101)
    expected_root = solver_seed(expected_episode, 'root', 25)
    expected = {f'{s}_seed': solver_seed(expected_root, s) for s in ('solve', 'selection', 'validation')}
    for task in runner.task_list(runner.load_config(CONFIG))[:4]:
        assert runner.root_seeds(task['controller_seed'], task['env_seed'], 25) == expected
    assert len(set(expected.values())) == 3
    assert runner.root_seeds(55, 102, 25) != expected
    assert runner.root_seeds(55, 101, 26) != expected


@pytest.mark.parametrize('failure', [None, 'action', 'work', 'model', 'q'])
def test_smoke_validation_checks_execution_work_and_both_audit_scores(monkeypatch, failure):
    work = dict(actor_updates=24, critic_updates=192, model_steps=768)
    choices = {name: dict(normalized_action=[i / 4.], learned_q=float(i),
        validation_model=dict(draws=[float(i)] * 32))
        for i, name in enumerate(('actor_mean', 'learned_q', 'model_score', 'prior_mean'))}
    output = dict(action=np.array([.1]), work=work, diagnostics=dict(choices=choices))
    repeated = deepcopy(output)
    if failure == 'action': repeated['action'][0] += 1
    if failure == 'work': repeated['work']['critic_updates'] = 96
    def solve(*args, **kwargs):
        assert kwargs['critic_updates_per_round'] == 32 and kwargs['shadow'] is False
        assert kwargs['arm'] == 'learned_q'
        return repeated
    def audit(*args, **kwargs):
        assert kwargs['horizons'] == (1,) and kwargs['mc_rollouts'] == 32
        rows = [dict(label=label, q=dict(final=value['learned_q']),
            model=dict(h1=dict(draws=value['validation_model']['draws'][:])))
            for label, value in choices.items()]
        if failure == 'model': rows[0]['model']['h1']['draws'][0] += 1
        if failure == 'q': rows[0]['q']['final'] += 1
        return dict(actions=rows)
    monkeypatch.setattr(runner, 'run_bypass_decision', solve)
    monkeypatch.setattr(runner, 'score_action_bank', audit)
    reference = SimpleNamespace(engine=SimpleNamespace(_action_pool=SimpleNamespace(critic=object())))
    task = dict(selector='learned_q', critic_updates_per_round=32)
    if failure is None:
        assert all(runner.smoke_validate(reference, object(), [0.], task, output,
            dict(validation_seed=123)).values())
    else:
        with pytest.raises(RuntimeError):
            runner.smoke_validate(reference, object(), [0.], task, output, dict(validation_seed=123))
