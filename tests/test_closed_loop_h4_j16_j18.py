"""The longer H4 screen changes only rounds and retained replay capacity."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from slurm import ambi_closed_loop_critics as campaign
from tests.test_ambi_eval_execution import _capture_execution
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_component_model
from tests.test_closed_loop_critics_575k import matrix_path, prior_fixture


@pytest.mark.parametrize('rounds', [16, 18])
def test_single_h4_setting_changes_only_rounds_and_replay_from_completed_h4(rounds):
    from utils.ambi_research import load_preset_matrix
    from utils.eval_series_data import planner_identity
    matrix = load_preset_matrix(matrix_path(4, rounds=rounds))
    cell, = campaign.cells(matrix_path(4, rounds=rounds))
    original = load_preset_matrix(campaign.MATRIX.with_name('ambi_closed_loop_alpha0_lambda1_h4_575k.json'))
    previous = {**original['shared_alg_params'], **original['comparisons']['sweep']['variants'][
        'h4_mean_h4_j14_c16_one_step']['alg_params']}
    assert cell['requested_alg_params'] == {**previous, 'inner_rounds': rounds,
                                           'inner_replay_capacity': 512 * rounds}
    assert matrix['evaluation'] == {**original['evaluation'], 'default_presets': [cell['selector']]}
    assert set(matrix['comparisons']['sweep']['variants']) == {'prior', cell['name']}
    assert matrix['comparisons']['sweep']['reference'] == 'prior'
    assert cell['critic_kind'] == 'return_only'
    assert campaign.campaign_horizon({'cells': [cell], 'H': 4}) == 4
    cfg = _build_cfg(**cell['params'], aux_return_mode='sac', log_std_mapping='direct_clamp',
                     target_entropy=-10.5, sac_actor_loss_scale_mode='none', train_unroll_horizon=3)
    assert cfg.inner_model_step_budget == cfg.inner_replay_capacity == 512 * rounds
    assert cfg.inner_critic_updates_per_action == 16 * rounds
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 4 * rounds
    planner = planner_identity(vars(cfg), {}, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean')
    assert planner['type'] == 'sac'
    assert planner['settings']['inner_rollout_horizon'] == 4
    assert planner['settings']['inner_critic_source'] == planner['settings']['inner_horizon_critic_source'] == 'aux_return'


@pytest.mark.parametrize('change', ['prior', 'duplicate', 'both_rounds', 'soft', 'sampled', 'retrace',
                                   'alpha_zero', 'old_rounds', 'capacity', 'missing_reference'])
def test_unrequested_h4_scope_or_recipe_is_rejected(tmp_path, change):
    from utils.ambi_research import PresetMatrixError
    matrix = campaign.read(matrix_path(4, rounds=16))
    selector, = matrix['evaluation']['default_presets']
    variants = matrix['comparisons']['sweep']['variants']
    variant = variants[selector.split('/')[1]]
    if change == 'prior':
        matrix['evaluation']['default_presets'].append('sweep/prior')
    elif change == 'duplicate':
        matrix['evaluation']['default_presets'].append(selector)
    elif change == 'both_rounds':
        other = campaign.read(matrix_path(4, rounds=18))
        variants.update(other['comparisons']['sweep']['variants'])
        matrix['evaluation']['default_presets'] += other['evaluation']['default_presets']
    elif change == 'soft':
        variants.pop(selector.split('/')[1])
        variants['soft_soft_h4_j16_c16'] = variant
        variant['alg_params'].update(inner_critic_source='sac', inner_horizon_critic_source='sac',
                                     inner_sac_critic_target='entropy_augmented', inner_terminal_entropy='outer')
        matrix['evaluation']['default_presets'] = ['sweep/soft_soft_h4_j16_c16']
    elif change == 'sampled': variant['alg_params']['inner_eval_execution_action'] = 'policy_sample'
    elif change == 'retrace': variant['alg_params']['inner_sac_return_estimator'] = 'retrace'
    elif change == 'alpha_zero': variant['alg_params']['inner_entropy_enabled'] = False
    elif change == 'old_rounds': variant['alg_params']['inner_rounds'] = 14
    elif change == 'capacity': variant['alg_params']['inner_replay_capacity'] = 7168
    elif change == 'missing_reference': matrix['comparisons']['sweep'].pop('reference')
    path = tmp_path / 'matrix.json'; path.write_text(json.dumps(matrix))
    with pytest.raises((AssertionError, PresetMatrixError)):
        campaign.cells(path)


@pytest.mark.parametrize('rounds', [16, 18])
def test_real_prepare_writes_one_canonical_spec_and_registry_without_evaluating(tmp_path, monkeypatch, rounds):
    """Keep real schema, resolution, reference protocol and registry validation."""
    import evaluate_ambi_checkpoint as evaluator
    from utils import ambi_benchmark, eval_series_data
    from utils.ambi_research import load_preset_matrix, resolve_preset
    from utils.checkpoint_context import load_checkpoint_context
    from utils.eval_series import load_run
    checkpoint = tmp_path / 'checkpoint.pt'; checkpoint.write_bytes(b'metadata-only checkpoint fixture')
    actual_sha = campaign.digest(checkpoint)
    base = dict(aux_return_mode='sac', aux_return_detach_representation=False, target_entropy=-10.5,
                log_std_mapping='direct_clamp', sac_actor_loss_scale_mode='none', train_unroll_horizon=3)
    trial = dict(alg='AMBITDMPC2/AMBITDMPC2', env='DMControl-v0', seed=55, total_steps=1000000,
                 alg_params=base, resolved_runtime={'observation': dict(mode='state', shape=[67],
                                                                       action_dim=21, episode_length=500)})
    metadata = dict(schema_version=1, trial_run_params=trial,
                    experiment_params={'env_params': {'task': 'humanoid-walk', 'obs': 'state'}},
                    checkpoint=dict(kind='periodic', step=575000, episode=1150, best_score=None, best_window=1))
    Path(str(checkpoint) + '.metadata.json').write_text(json.dumps(metadata))
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps(dict(source_run=campaign.SOURCE_RUN,
                                        checkpoints=[dict(step=575000, sha256=actual_sha)])))
    matrix_file = matrix_path(4, rounds=rounds)
    matrix = load_preset_matrix(matrix_file)
    context = load_checkpoint_context(checkpoint)
    prior_resolved = resolve_preset(matrix_file, 'sweep/prior', matrix=matrix, checkpoint_context=context)
    protocol = ambi_benchmark.protocol_for(prior_resolved, 55, 500)
    checkpoint_info = dict(metadata=metadata, path=str(checkpoint), sha256=actual_sha, source_run=campaign.SOURCE_RUN)
    monkeypatch.setattr(campaign, 'CHECKPOINT_SHA', actual_sha)
    manifest, record = prior_fixture()
    manifest.update(schema_version=ambi_benchmark.SCHEMA_VERSION, protocol=protocol)
    manifest['runs'][0].update(status='complete', config=prior_resolved['algorithm_config'])
    record['identity'] = eval_series_data.identity_for_ambi_checkpoint(
        checkpoint_info, prior_resolved, protocol, campaign.SEEDS,
        dict(commit=campaign.PRIOR_SOURCE_COMMIT, dirty=False), path=checkpoint, inventory_path=inventory)
    reference = tmp_path / 'reference'; reference.mkdir()
    (reference / 'manifest.json').write_text(json.dumps(manifest))
    # Only the external historical-record ingestion and clean-source guard are fixtures.
    monkeypatch.setattr(eval_series_data, 'load_records', lambda *a, **k: [record])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=campaign.ROOT, text=True).strip()
    monkeypatch.setattr(campaign, 'source_commit', lambda: commit)
    monkeypatch.setattr(ambi_benchmark, 'code_identity', lambda: dict(commit=commit, dirty=False))
    monkeypatch.setattr(evaluator, '_make_env', lambda *a, **k: pytest.fail('prepare constructed an environment'))
    monkeypatch.setattr(evaluator, 'evaluate_preset', lambda *a, **k: pytest.fail('prepare evaluated an episode'))
    args = SimpleNamespace(root=tmp_path / 'campaign', matrix=matrix_file, checkpoint=checkpoint,
                           inventory=inventory, reference=reference, registry=tmp_path / 'registry',
                           group=f'h4-j{rounds}', label=f'H4 J{rounds}')
    prepared = campaign.prepare(args)
    cell, = prepared['cells']
    spec_path, = (args.root / 'specs').glob('*.json')
    spec = campaign.read(spec_path)
    registered, = args.registry.iterdir()
    assert load_run(registered)['identity'] == spec['identity']
    assert spec['selector'] == cell['selector'] and cell['run_dir'] == str(registered)
    assert spec['identity']['protocol'] == record['identity']['protocol']
    assert spec['identity']['backbone'] == campaign.SOURCE_RUN
    assert spec['identity']['planner']['settings']['inner_rounds'] == rounds
    assert spec['identity']['planner']['settings']['inner_replay_capacity'] == 512 * rounds
    assert prepared['H'] == 4 and prepared['prior_manifest_sha256'] == campaign.digest(reference / 'manifest.json')
    assert not (args.root / 'unused').exists()


def test_full_j18_solve_retains_all_imagined_data_and_executes_mean(monkeypatch):
    cell, = campaign.cells(matrix_path(4, rounds=18))
    params = {**cell['params'], 'compile': False, 'compile_strict': False}
    holder = _tiny_component_model(**params, aux_return_mode='sac', train_unroll_horizon=3)
    try:
        agent, engine = holder.agent, holder.agent.inner_engine
        captured = _capture_execution(monkeypatch, engine)
        outer = deepcopy(agent.checkpoint_state())
        global_rng = torch.random.get_rng_state().clone()
        engine.reset_for_evaluation(919)
        action = agent.act(torch.zeros(3), t0=True, eval_mode=True)
        torch.testing.assert_close(action, captured[-1]['mean'], rtol=0, atol=0)
        metrics = agent.last_inner_metrics
        assert engine._critic_base is engine._horizon_critic is engine.model._aux_return_Qs
        assert metrics['inner_model_steps'] == metrics['inner_buffer_size'] == 9216
        assert metrics['inner_critic_optimizer_steps'] == 288
        assert metrics['inner_actor_optimizer_steps'] == metrics['inner_temperature_optimizer_steps'] == 72
        assert metrics['inner_alpha_initial'] > 0 and metrics['inner_alpha_final'] > 0
        assert metrics['inner_eval_execution_sampled'] == metrics['inner_eval_execution_mean_action_l2'] == 0
        _assert_tree_equal(agent.checkpoint_state(), outer)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        holder.close()
