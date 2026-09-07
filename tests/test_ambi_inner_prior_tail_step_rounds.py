"""Matched prior-tail experiment changes targets while retaining update budgets."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_benchmark_evaluation import checkpoint_matrix, events
from tests.test_ambi_inner_benchmark_launcher import launch_env
from tests.test_ambi_inner_step_rounds import _step_env
from tests.test_checkpoint_research_configs import checkpoint_context, _build_cfg
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'configs/research/ambi_humanoid_inner_prior_tail_step_rounds.json'
BASE = ROOT / 'configs/research/ambi_humanoid_inner_step_rounds.json'
LAUNCHER = ROOT / 'slurm/run_ambi_inner_prior_tail_step_rounds_oscar.sbatch'


def test_prior_tail_changes_only_boundary_bootstrap(checkpoint_context):
    matrix, base = load_preset_matrix(MATRIX), load_preset_matrix(BASE)
    assert normalize_selectors(matrix) == [f'prior_tail_rounds/j{j}' for j in (1, 3, 6)]
    assert matrix['source_run'] == base['source_run']
    new_eval, old_eval = deepcopy(matrix['evaluation']), deepcopy(base['evaluation'])
    new_eval.pop('default_presets')
    old_eval.pop('default_presets')
    assert new_eval == old_eval
    for j in (1, 3, 6):
        new = resolve_preset(MATRIX, f'prior_tail_rounds/j{j}', matrix,
                             checkpoint_context=checkpoint_context)['algorithm_config']
        old = resolve_preset(BASE, f'step_rounds/j{j}', base,
                             checkpoint_context=checkpoint_context)['algorithm_config']
        comparable = deepcopy(new)
        assert comparable['alg_params'].pop('inner_finite_horizon') is True
        old['alg_params'].pop('inner_finite_horizon')
        assert comparable == old
        cfg = _build_cfg(new)
        assert cfg.inner_finite_horizon and cfg.inner_bootstrap_source == 'inner_target'
        assert cfg.inner_update_timing == 'step' and cfg.inner_steps_per_update == 512
        assert cfg.inner_model_step_budget == 1536 * j
        assert cfg.inner_critic_updates_per_action == cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 3 * j


@pytest.mark.parametrize('rounds', [1, 3, 6])
def test_prior_tail_frozen_evaluator_preserves_step_order(checkpoint_matrix, tmp_path, rounds):
    checkpoint, matrix_path = checkpoint_matrix
    matrix = json.loads(matrix_path.read_text())
    matrix['shared_alg_params'].update(inner_rounds=rounds,
        inner_critic_updates_per_round=None, inner_actor_updates_per_round=None,
        inner_updates_per_round=None, inner_steps_per_update=2, inner_update_timing='step',
        inner_replay_capacity=24, inner_finite_horizon=True)
    matrix_path.write_text(json.dumps(matrix))
    bundle = tmp_path / 'tail-bundle'
    result = evaluator.evaluate_matrix(matrix_path, checkpoint, selectors=['budget/sac'], bundle_dir=bundle)
    run = result['results'][0]
    assert run['outer_state_unchanged']
    assert run['resolved_config']['inner_finite_horizon'] is True
    assert run['model_metrics']['inner_finite_horizon']['mean'] == 1
    assert run['model_metrics']['inner_critic_optimizer_steps']['mean'] == 2 * rounds
    rows = [row for row in events(bundle) if row['phase'] != 'decision']
    assert len(rows) == 6 * (1 + 4 * rounds)
    for index in range(0, len(rows), 1 + 4 * rounds):
        assert [r['phase'] for r in rows[index:index + 1 + 4 * rounds]] == ['initial'] + ['collection', 'update'] * (2 * rounds)


@pytest.mark.parametrize('index', [1, 6, 10])
def test_production_reuses_all_priors_and_selects_only_prior_tail(launch_env, index):
    env = _step_env(launch_env)
    prior = Path(env['AMBI_BENCHMARK_REFERENCE_ROOT']) / 'step_50000/prior'
    prior.mkdir(parents=True)
    (prior / 'manifest.json').write_text('{}')
    env['SLURM_ARRAY_TASK_ID'] = str(index)
    subprocess.run(['bash', str(LAUNCHER)], env=env, check=True, capture_output=True, text=True)
    calls = [json.loads(line) for line in Path(env['TEST_CALLS']).read_text().splitlines()]
    assert len(calls) == 2
    evaluate = calls[0]
    assert evaluate[evaluate.index('--matrix') + 1] == str(MATRIX.relative_to(ROOT))
    assert [evaluate[i + 1] for i, arg in enumerate(evaluate) if arg == '--preset'] == [f'prior_tail_rounds/j{j}' for j in (1, 3, 6)]
    assert '--reference-bundle' in evaluate and '--eval-run-map' in evaluate
    assert not any('named_run/prior' in call or '--wandb' in call for call in calls)


def test_missing_50k_prior_fails_before_any_evaluation(launch_env):
    env = _step_env(launch_env)
    env['SLURM_ARRAY_TASK_ID'] = '1'
    result = subprocess.run(['bash', str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'paired prior reference missing' in result.stderr
    assert not Path(env['TEST_CALLS']).exists()
