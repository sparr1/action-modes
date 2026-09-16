"""J1 adapts the requested critics and publishes seed-paired prior gains."""
import copy
import json
from pathlib import Path

import pytest

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_root_local_sac import _tiny_model, _tiny_params
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.checkpoint_context import load_checkpoint_context
from utils.eval_series import _planner_display_label
from utils.eval_series_data import _metrics, planner_identity

MATRIX = Path(__file__).resolve().parents[1] / 'configs/research/ambi_aux_return_j1.json'


def test_j1_paired_bundle_uses_selected_critic_and_preserves_outer(tmp_path):
    options = dict(aux_return_mode='sac', log_std_mapping='direct_clamp',
                   sac_actor_loss_scale_mode='none', inner_operator='none',
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
    assert matrix['evaluation']['togo_return_rollouts'] == 0
    context = load_checkpoint_context(checkpoint)
    configs = [resolve_preset(MATRIX, 'critic/' + arm, checkpoint_context=context)
               for arm in ('soft_q', 'return_q')]
    left, right = [copy.deepcopy(c['algorithm_config']) for c in configs]
    for key in ('inner_critic_source', 'inner_horizon_critic_source'):
        assert left['alg_params'].pop(key) == 'sac'
        assert right['alg_params'].pop(key) == 'aux_return'
    left.pop('name', None); right.pop('name', None)
    assert left == right
    # Only runtime compilation is disabled for the small CPU integration test.
    matrix['shared_alg_params']['compile'] = False
    local_matrix = tmp_path / 'matrix.json'
    local_matrix.write_text(json.dumps(matrix))
    prior = evaluator.evaluate_matrix(local_matrix, checkpoint, selectors=['critic/prior'],
                                     seeds=[101, 102], max_steps=3, device='cpu',
                                     bundle_dir=tmp_path / 'prior')
    reference = {e['seed']: e['return'] for e in prior['results'][0]['episodes']}
    result = evaluator.evaluate_matrix(local_matrix, checkpoint, seeds=[102, 101], max_steps=3,
                                       device='cpu', bundle_dir=tmp_path / 'j1',
                                       reference_bundle=tmp_path / 'prior')
    manifest = json.loads((tmp_path / 'j1/manifest.json').read_text())
    assert len(result['results']) == 2
    for arm, source, bundle_run in zip(result['results'], ('sac', 'aux_return'), manifest['runs']):
        assert arm['outer_state_unchanged'] and not arm['nonfinite_model_metrics']
        cfg = arm['resolved_config']
        assert cfg['inner_rounds'] == cfg['inner_rollout_horizon'] == 1
        assert cfg['inner_rollouts_per_round'] == 128 and cfg['inner_batch_size'] == 256
        assert cfg['inner_actor_source'] == cfg['inner_horizon_actor_source'] == 'sac'
        assert cfg['inner_critic_source'] == cfg['inner_horizon_critic_source'] == source
        assert cfg['inner_log_std_mapping'] == 'direct_clamp'
        assert cfg['inner_sac_critic_target'] == 'reward_only' and cfg['inner_finite_horizon']
        assert cfg['sac_actor_loss_scale_mode'] == 'none'
        assert cfg.get('aux_return_sac_actor_loss_scale_mode') in (None, 'none')
        for metric, expected in [('inner_model_steps', 128), ('inner_critic_optimizer_steps', 32),
                                 ('inner_actor_optimizer_steps', 4), ('inner_temperature_optimizer_steps', 0),
                                 ('inner_alpha', 0)]:
            assert arm['model_metrics'][metric]['mean'] == expected
        episodes = bundle_run['episodes']
        assert all(e['length'] == 3 and not e['truncated_by_evaluator'] for e in episodes)
        gains = [e['return'] - reference[e['seed']] for e in episodes]
        assert [e['paired_return_delta'] for e in episodes] == gains
        assert _metrics(episodes)['eval/paired_gain_mean'] == pytest.approx(sum(gains) / 2)
        label = _planner_display_label(planner_identity(cfg, arm, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean'))
        assert ('return-only Q init+tail' if source == 'aux_return' else 'SAC Q init+tail') in label
