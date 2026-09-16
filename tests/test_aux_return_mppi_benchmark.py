"""Real auxiliary checkpoints preserve routes, pairing, budgets and frozen state."""
import copy
import json
from pathlib import Path

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_root_local_sac import _tiny_model, _tiny_params
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.checkpoint_context import load_checkpoint_context
from utils.eval_series_data import planner_identity
from utils.eval_series import _planner_display_label

MATRIX = Path(__file__).resolve().parents[1] / 'configs/research/ambi_aux_return_mppi.json'


def test_paired_auxiliary_mppi_checkpoint_bundle(tmp_path):
    options = dict(aux_return_mode='sac', inner_operator='none',
                   inner_rounds=0, inner_rollouts_per_round=0, inner_updates_per_round=0)
    model = _tiny_model(**options)
    checkpoint = tmp_path / 'model_25000'
    model.agent.save(checkpoint)
    model.env.close()
    metadata = dict(schema_version=1, checkpoint=dict(kind='periodic', step=25000, episode=50, best_score=None, best_window=100),
                    trial_run_params=dict(alg='AMBITDMPC2/AMBITDMPC2', env='Pendulum-v1',
                        seed=55, device='cpu', total_steps=2000000, alg_params=_tiny_params(**options)),
                    experiment_params=dict(env_params=dict(max_episode_steps=3)))
    Path(str(checkpoint) + '.metadata.json').write_text(json.dumps(metadata))
    matrix = load_preset_matrix(MATRIX)
    assert matrix['evaluation']['seeds'] == [101, 102, 103, 104, 105]
    context = load_checkpoint_context(checkpoint)
    configs = [resolve_preset(MATRIX, 'bootstrap/' + arm, checkpoint_context=context)
               for arm in ('soft_q', 'return_q')]
    left, right = [copy.deepcopy(c['algorithm_config']) for c in configs]
    assert left['alg_params'].pop('inner_horizon_critic_source') == 'sac'
    assert right['alg_params'].pop('inner_horizon_critic_source') == 'aux_return'
    # Preset names are descriptive; scientific settings differ in one critic route.
    left.pop('name', None); right.pop('name', None)
    assert left == right
    results = []
    for index, seeds in enumerate(([101, 102], [102, 101])):
        result = evaluator.evaluate_matrix(MATRIX, checkpoint, seeds=seeds, max_steps=3,
                                          device='cpu', bundle_dir=tmp_path / f'bundle{index}')
        arms = result['results']
        assert [r['value_routing']['inner_horizon_critic_source'] for r in arms] == ['sac', 'aux_return']
        identities = []
        for arm in arms:
            assert arm['outer_state_unchanged']
            cfg = arm['resolved_config']
            assert cfg['inner_mppi_num_samples'] == 512
            assert cfg['inner_mppi_iterations'] == 8
            assert cfg['mppi_terminal_q_reduction'] == 'mean_pair'
            assert arm['model_metrics']['inner_model_steps']['mean'] == 12336
            assert arm['model_metrics']['inner_actor_optimizer_steps']['mean'] == 0
            identity = planner_identity(cfg, arm, 'AMBITDMPC2/AMBITDMPC2', 'mppi_proposal_mean')
            identities.append(identity)
        assert identities[0] != identities[1]
        assert 'soft Q' in _planner_display_label(identities[0])
        assert 'return-only Q' in _planner_display_label(identities[1])
        results.append([{e['seed']: (e['return'], e['solver_seed']) for e in a['episodes']} for a in arms])
    assert results[0] == results[1]
