"""The C32 sweep increases critic dose without changing the paired protocol."""
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


MATRIX = campaign.MATRIX.with_name('ambi_closed_loop_critics_h3_c32_j_sweep_575k.json')
ROUNDS = [1, 2, 4, 6, 8, 10, 12, 14]


def test_c32_sweep_changes_only_critic_update_count_at_each_historical_j():
    from utils.ambi_research import load_preset_matrix
    from utils.eval_series_data import planner_identity
    matrix = load_preset_matrix(MATRIX)
    panel = campaign.cells(MATRIX)
    assert [cell['J'] for cell in panel] == ROUNDS
    assert set(matrix['comparisons']['sweep']['variants']) == {'prior', *(c['name'] for c in panel)}
    assert matrix['evaluation'] == {**campaign.read(campaign.MATRIX)['evaluation'],
                                    'default_presets': [c['selector'] for c in panel]}
    assert campaign.campaign_horizon({'cells': panel, 'H': 3}) == 3
    for cell in panel:
        rounds = cell['J']
        old = campaign.cells(matrix_path(3, rounds=None if rounds <= 4 else rounds))
        baseline, = [c for c in old if c['J'] == rounds and c['critic_kind'] == 'return_only']
        assert cell['requested_alg_params'] == {**baseline['requested_alg_params'],
                                               'inner_critic_updates_per_round': 32}
        cfg = _build_cfg(**cell['params'], aux_return_mode='sac', log_std_mapping='direct_clamp',
                         target_entropy=-10.5, sac_actor_loss_scale_mode='none', train_unroll_horizon=3)
        assert cfg.inner_model_step_budget == 384 * rounds
        assert cfg.inner_replay_capacity == max(3072, 384 * rounds)
        assert cfg.inner_critic_updates_per_action == 32 * rounds
        assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 4 * rounds
        planner = planner_identity(vars(cfg), {}, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean')
        assert planner['settings']['inner_critic_updates_per_round'] == 32
        assert planner['settings']['inner_critic_source'] == planner['settings']['inner_horizon_critic_source'] == 'aux_return'


@pytest.mark.parametrize('change', ['partial', 'duplicate', 'mixed_c', 'soft', 'sampled', 'retrace',
                                   'alpha_zero', 'capacity', 'actor_updates', 'learning_rate'])
def test_unrequested_scope_or_recipe_is_rejected(tmp_path, change):
    matrix = campaign.read(MATRIX)
    selectors = matrix['evaluation']['default_presets']
    variant = matrix['comparisons']['sweep']['variants'][selectors[-1].split('/')[1]]['alg_params']
    if change == 'partial': selectors.pop()
    elif change == 'duplicate': selectors[-1] = selectors[-2]
    elif change == 'mixed_c': variant['inner_critic_updates_per_round'] = 16
    elif change == 'soft': variant.update(inner_critic_source='sac', inner_horizon_critic_source='sac',
                                         inner_sac_critic_target='entropy_augmented', inner_terminal_entropy='outer')
    elif change == 'sampled': variant['inner_eval_execution_action'] = 'policy_sample'
    elif change == 'retrace': variant['inner_sac_return_estimator'] = 'retrace'
    elif change == 'alpha_zero': variant['inner_entropy_enabled'] = False
    elif change == 'capacity': variant['inner_replay_capacity'] = 3840
    elif change == 'actor_updates': variant['inner_actor_updates_per_round'] = 8
    elif change == 'learning_rate': variant['inner_critic_lr'] = .0006
    path = tmp_path / 'matrix.json'; path.write_text(json.dumps(matrix))
    with pytest.raises(AssertionError):
        campaign.cells(path)


def test_probe_coordinates_require_c32_update_counts():
    cell = campaign.cells(MATRIX)[-1]
    rows = [dict(episode_id='seed-101', decision_index=decision, round_index=r,
                 critic_updates=32*r, actor_updates=4*r,
                 metrics=dict(probe_model_steps=96*(2 if r == 0 else 1),
                              probe_q_evaluations=32*(2 if r == 0 else 1)))
            for decision in range(3) for r in range(15)]
    campaign.validate_probe_rows({'togo_probe_rows': rows}, cell, seeds=[101], steps=3)
    wrong = deepcopy(rows); wrong[-1]['critic_updates'] = 16 * 14
    with pytest.raises(AssertionError):
        campaign.validate_probe_rows({'togo_probe_rows': wrong}, cell, seeds=[101], steps=3)


def test_real_prepare_writes_eight_canonical_specs_and_distinct_registries(tmp_path, monkeypatch):
    """Keep real configuration resolution and canonical identity/registry creation."""
    import evaluate_ambi_checkpoint as evaluator
    from slurm import ambi_closed_loop_publish as publisher
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
    matrix = load_preset_matrix(MATRIX)
    context = load_checkpoint_context(checkpoint)
    prior_resolved = resolve_preset(MATRIX, 'sweep/prior', matrix=matrix, checkpoint_context=context)
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
    monkeypatch.setattr(eval_series_data, 'load_records', lambda *a, **k: [record])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=campaign.ROOT, text=True).strip()
    monkeypatch.setattr(campaign, 'source_commit', lambda: commit)
    monkeypatch.setattr(ambi_benchmark, 'code_identity', lambda: dict(commit=commit, dirty=False))
    monkeypatch.setattr(evaluator, '_make_env', lambda *a, **k: pytest.fail('prepare constructed an environment'))
    monkeypatch.setattr(evaluator, 'evaluate_preset', lambda *a, **k: pytest.fail('prepare evaluated an episode'))
    checked = []
    def check_references(settings):
        assert not (tmp_path / 'campaign').exists()
        assert set(settings['comparison_references']) == set(map(str, ROUNDS))
        assert settings['checkpoint_sha256'] == actual_sha
        checked.append(settings)
    monkeypatch.setattr(publisher, 'load_comparison_references', check_references)
    args = SimpleNamespace(root=tmp_path / 'campaign', matrix=MATRIX, checkpoint=checkpoint,
                           inventory=inventory, reference=reference, registry=tmp_path / 'registry',
                           group='h3-c32', label='H3 C32')
    with monkeypatch.context() as blocked:
        def reject_reference(settings):
            raise ValueError('Pinned C16 manifest changed')
        blocked.setattr(publisher, 'load_comparison_references', reject_reference)
        with pytest.raises(ValueError, match='Pinned C16 manifest changed'):
            campaign.prepare(args)
        assert not args.root.exists() and not args.registry.exists()
    prepared = campaign.prepare(args)
    assert len(checked) == 1
    assert prepared['comparison_references'] == checked[0]['comparison_references']
    specs = [campaign.read(path) for path in (args.root / 'specs').glob('*.json')]
    registered = [load_run(path) for path in args.registry.iterdir()]
    assert len(specs) == len(registered) == 8
    assert len({c['performance_run_id'] for c in prepared['cells']}) == 8
    assert len({c['training_run_id'] for c in prepared['cells']}) == 8
    assert set(c['performance_run_id'] for c in prepared['cells']).isdisjoint(
        c['training_run_id'] for c in prepared['cells'])
    for spec in specs:
        assert any(run['identity'] == spec['identity'] for run in registered)
        assert spec['identity']['protocol'] == record['identity']['protocol']
        assert spec['identity']['planner']['settings']['inner_critic_updates_per_round'] == 32
    assert prepared['publisher_workers'] == 3
    assert not (args.root / 'unused').exists()


def test_full_j14_c32_solve_keeps_actor_dose_and_executes_mean(monkeypatch):
    cell = campaign.cells(MATRIX)[-1]
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
        assert metrics['inner_model_steps'] == metrics['inner_buffer_size'] == 5376
        assert metrics['inner_critic_optimizer_steps'] == 448
        assert metrics['inner_actor_optimizer_steps'] == metrics['inner_temperature_optimizer_steps'] == 56
        assert metrics['inner_alpha_initial'] > 0 and metrics['inner_alpha_final'] > 0
        assert metrics['inner_eval_execution_sampled'] == metrics['inner_eval_execution_mean_action_l2'] == 0
        _assert_tree_equal(agent.checkpoint_state(), outer)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        holder.close()
