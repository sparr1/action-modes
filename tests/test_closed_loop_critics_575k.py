"""The selected checkpoint, objectives, pairing and worker ownership are fixed."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import slurm.ambi_closed_loop_critics as campaign
from tests.test_ambi_root_local_sac import _build_cfg
from utils.ambi_benchmark import solver_seed


def prior_fixture():
    episodes = [dict(seed=seed, solver_seed=solver_seed(55, 'episode', seed),
                     length=500, truncated_by_evaluator=False, **{'return': float(seed)})
                for seed in campaign.SEEDS]
    protocol = dict(environment=dict(id='DMControl-v0', params=dict(task='humanoid-walk', obs='state')),
                    observation='state', action_rule='tanh_mean', controller_seed=55,
                    max_steps=500, seed_scheme='sha256-v1')
    checkpoint = dict(sha256=campaign.CHECKPOINT_SHA, source_run=campaign.SOURCE_RUN,
                      metadata=dict(checkpoint=dict(step=575000)))
    config = dict(target_entropy=-10.5, aux_return_detach_representation=False,
                  aux_return_mode='sac', inner_operator='none', log_std_mapping='direct_clamp',
                  sac_actor_loss_scale_mode='none', aux_return_sac_actor_loss_scale_mode='none')
    manifest = dict(status='complete', checkpoint=checkpoint, protocol=protocol,
                    code=dict(commit=campaign.PRIOR_SOURCE_COMMIT, dirty=False, runtime={'python': '3.10'}),
                    runs=[dict(resolved_config=config, episodes=episodes,
                               result=dict(model_metrics=dict(inner_alpha_initial={
                                   key: campaign.INITIAL_ALPHA for key in ('mean', 'min', 'max')})))])
    record = dict(checkpoint=dict(step=575000, sha256=campaign.CHECKPOINT_SHA),
                  identity=dict(backbone=campaign.SOURCE_RUN, protocol=protocol,
                                planner={'type': 'prior', 'action_rule': 'tanh_mean'}, science={'old': True}),
                  metrics={'eval/frozen_state_unchanged': True}, episodes=deepcopy(episodes))
    return manifest, record


def matrix_path(horizon, *, rounds=None):
    if rounds is not None:
        return campaign.MATRIX.with_name(f'ambi_closed_loop_critics_h{horizon}_j{rounds}_575k.json')
    return campaign.MATRIX if horizon == 3 else campaign.MATRIX.with_name(f'ambi_closed_loop_critics_h{horizon}_575k.json')


@pytest.mark.parametrize('horizon', [1, 2, 3])
def test_six_cells_keep_accepted_recipe_and_objectives(horizon):
    panel = campaign.cells(matrix_path(horizon))
    assert [(c['J'], c['critic_kind']) for c in panel] == [
        (j, arm) for j in (4, 2, 1) for arm in ('soft', 'return_only')]
    old = campaign.read(campaign.MATRIX.with_name('ambi_aux_hj_sweep_625k.json'))
    for cell in panel:
        arm = 'soft_soft' if cell['critic_kind'] == 'soft' else 'return_return_alpha'
        previous = {**old['shared_alg_params'],
                    **old['comparisons']['sweep']['variants'][f'{arm}_h{horizon}_j{cell["J"]}']['alg_params']}
        previous.update(inner_replay_capacity=3072, inner_critic_updates_per_round=16,
                        inner_actor_updates_per_round=4, inner_component_update_order='critic_first',
                        inner_replay_reset_each_round=False)
        assert cell['requested_alg_params'] == previous
        cfg = _build_cfg(**cell['params'], aux_return_mode='sac', log_std_mapping='direct_clamp',
                         target_entropy=-10.5, sac_actor_loss_scale_mode='none')
        assert cfg.inner_model_step_budget == 128 * horizon * cell['J']
        assert cfg.inner_critic_updates_per_action == 16 * cell['J']
        assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 4 * cell['J']
        assert cfg.inner_model_step_budget <= cfg.inner_replay_capacity
        assert cfg.inner_target_entropy == cfg.inner_temperature_initialization == 'inherit_outer'
        assert cfg.inner_actor_source == cfg.inner_horizon_actor_source == 'sac'


@pytest.mark.parametrize('horizon', [1, 2, 3])
@pytest.mark.parametrize('rounds', [6, 8])
def test_extension_selects_only_two_new_arms_and_changes_only_rounds(horizon, rounds):
    old = campaign.cells(matrix_path(horizon))
    new = campaign.cells(matrix_path(horizon, rounds=rounds))
    assert [(cell['J'], cell['critic_kind']) for cell in new] == [(rounds, 'soft'), (rounds, 'return_only')]
    assert not {cell['selector'] for cell in old}.intersection(cell['selector'] for cell in new)
    assert len(campaign.cells()) == 6
    for before, after in zip(old[:2], new):
        assert after['requested_alg_params'] == {**before['requested_alg_params'], 'inner_rounds': rounds}
        cfg = _build_cfg(**after['params'], aux_return_mode='sac', log_std_mapping='direct_clamp',
                         target_entropy=-10.5, sac_actor_loss_scale_mode='none')
        assert cfg.inner_model_step_budget == 128 * horizon * rounds
        assert cfg.inner_model_step_budget <= cfg.inner_replay_capacity == 3072
        assert (cfg.inner_model_step_budget == cfg.inner_replay_capacity) == (horizon == 3 and rounds == 8)
        assert cfg.inner_critic_updates_per_action == 16 * rounds
        assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 4 * rounds
    matrix = campaign.read(matrix_path(horizon, rounds=rounds))
    assert set(matrix['comparisons']['sweep']['variants']) == {'prior', *(cell['name'] for cell in new)}
    assert matrix['evaluation'] == {**campaign.read(matrix_path(horizon))['evaluation'],
                                    'default_presets': [cell['selector'] for cell in new]}


@pytest.mark.parametrize('rounds', [6, 8])
@pytest.mark.parametrize('change', ['one_arm', 'duplicate', 'append_old', 'j4_only', 'mixed_extension'])
def test_unrequested_partial_or_mixed_round_grids_are_rejected(tmp_path, change, rounds):
    matrix = campaign.read(matrix_path(3, rounds=rounds))
    selectors = matrix['evaluation']['default_presets']
    old = campaign.read(matrix_path(3))
    if change == 'one_arm':
        selectors.pop()
    elif change == 'duplicate':
        selectors[1] = selectors[0]
    elif change == 'mixed_extension':
        other = campaign.read(matrix_path(3, rounds=8 if rounds == 6 else 6))
        matrix['comparisons']['sweep']['variants'].update(other['comparisons']['sweep']['variants'])
        selectors[1] = other['evaluation']['default_presets'][1]
    else:
        matrix['comparisons']['sweep']['variants'].update(old['comparisons']['sweep']['variants'])
        if change == 'append_old':
            selectors.append(old['evaluation']['default_presets'][0])
        else:
            matrix['evaluation']['default_presets'] = old['evaluation']['default_presets'][:2]
    path = tmp_path / 'matrix.json'; path.write_text(json.dumps(matrix))
    with pytest.raises(AssertionError, match='original screen or both J6/J8'):
        campaign.cells(path)


@pytest.mark.parametrize('horizon', [1, 2])
def test_shorter_horizon_changes_only_horizon_and_descriptive_names(horizon):
    old = campaign.read(matrix_path(3))
    new = campaign.read(matrix_path(horizon))
    assert new['evaluation'] == {**old['evaluation'], 'default_presets': [
        selector.replace('_h3_', f'_h{horizon}_') for selector in old['evaluation']['default_presets']]}
    assert new['shared_alg_params'] == {**old['shared_alg_params'], 'inner_rollout_horizon': horizon}
    assert new['source_run'] == old['source_run']
    assert campaign.cells()[0]['H'] == 3  # Existing callers retain the H3 default.
    for before, after in zip(campaign.cells(matrix_path(3)), campaign.cells(matrix_path(horizon))):
        assert after['params'] == {**before['params'], 'inner_rollout_horizon': horizon}
        assert after['name'] == before['name'].replace('_h3_', f'_h{horizon}_')


def test_mixed_horizon_matrix_and_campaign_are_rejected(tmp_path):
    matrix = campaign.read(matrix_path(2))
    old_selector = matrix['evaluation']['default_presets'][-1]
    new_selector = old_selector.replace('_h2_', '_h3_')
    variants = matrix['comparisons']['sweep']['variants']
    changed = variants.pop(old_selector.split('/')[1])
    changed['alg_params']['inner_rollout_horizon'] = 3
    variants[new_selector.split('/')[1]] = changed
    matrix['evaluation']['default_presets'][-1] = new_selector
    path = tmp_path / 'mixed.json'; path.write_text(json.dumps(matrix))
    with pytest.raises(AssertionError, match='common horizon'):
        campaign.cells(path)
    panel = campaign.cells(matrix_path(2))
    assert campaign.campaign_horizon({'cells': panel, 'H': 2}) == 2
    with pytest.raises(AssertionError):
        campaign.campaign_horizon({'cells': panel, 'H': 3})
    panel[-1]['H'] = 3
    with pytest.raises(AssertionError, match='common horizon'):
        campaign.campaign_horizon({'cells': panel})


@pytest.mark.parametrize('horizon', [1, 2, 3])
@pytest.mark.parametrize('rounds', [None, 6, 8])
def test_probe_work_uses_selected_horizon_and_complete_coordinates(horizon, rounds):
    cell = campaign.cells(matrix_path(horizon, rounds=rounds))[0]
    rows = [dict(episode_id='seed-101', decision_index=decision, round_index=r,
                 critic_updates=16*r, actor_updates=4*r,
                 metrics=dict(probe_model_steps=32*horizon*(2 if r == 0 else 1),
                              probe_q_evaluations=32*(2 if r == 0 else 1), togo_return_mean=1.0))
            for decision in range(3) for r in range(cell['J'] + 1)]
    campaign.validate_probe_rows({'togo_probe_rows': rows}, cell, seeds=[101], steps=3)
    wrong = deepcopy(rows)
    wrong[0]['metrics']['probe_model_steps'] = 32*(3 if horizon == 2 else 2)*2
    with pytest.raises(AssertionError):
        campaign.validate_probe_rows({'togo_probe_rows': wrong}, cell, seeds=[101], steps=3)
    with pytest.raises(AssertionError):
        campaign.validate_probe_rows({'togo_probe_rows': rows[:-1]}, cell, seeds=[101], steps=3)
    wrong = deepcopy(rows); wrong[-1] = deepcopy(wrong[0])
    with pytest.raises(AssertionError):
        campaign.validate_probe_rows({'togo_probe_rows': wrong}, cell, seeds=[101], steps=3)


def test_matching_prior_is_accepted():
    campaign.check_prior(*prior_fixture())


@pytest.mark.parametrize('change', ['source', 'step', 'hash', 'seed', 'solver', 'length', 'alpha', 'entropy', 'detached', 'commit'])
def test_mismatched_prior_is_rejected(change):
    manifest, record = prior_fixture()
    if change == 'source': record['identity']['backbone'] = 'other/project/run'
    elif change == 'step': manifest['checkpoint']['metadata']['checkpoint']['step'] = 625000
    elif change == 'hash': manifest['checkpoint']['sha256'] = '0' * 64
    elif change == 'seed': record['episodes'][0]['seed'] = 999
    elif change == 'solver': record['episodes'][0]['solver_seed'] += 1
    elif change == 'length': record['episodes'][0]['length'] = 3
    elif change == 'alpha': manifest['runs'][0]['result']['model_metrics']['inner_alpha_initial']['mean'] *= 2
    elif change == 'entropy': manifest['runs'][0]['resolved_config']['target_entropy'] = -21
    elif change == 'detached': manifest['runs'][0]['resolved_config']['aux_return_detach_representation'] = True
    elif change == 'commit': manifest['code']['commit'] = '0' * 40
    with pytest.raises(AssertionError):
        campaign.check_prior(manifest, record)


@pytest.mark.parametrize('horizon', [1, 2, 3])
@pytest.mark.parametrize('rounds', [None, 6, 8])
def test_prepare_allocates_only_selected_new_planners_and_reuses_prior(tmp_path, monkeypatch, horizon, rounds):
    import evaluate_ambi_checkpoint
    import utils.eval_series
    import utils.eval_series_data
    prior, record = prior_fixture()
    reference = tmp_path / 'reference'; reference.mkdir()
    (reference / 'manifest.json').write_text(json.dumps(prior))
    checkpoint = tmp_path / 'checkpoint'; checkpoint.write_text('weight')
    args = SimpleNamespace(root=tmp_path / 'campaign', checkpoint=checkpoint,
                           reference=reference, inventory=tmp_path / 'inventory.json',
                           registry=tmp_path / 'registry', matrix=matrix_path(horizon, rounds=rounds),
                           group='test-group', label='Test')
    real_digest = campaign.digest
    monkeypatch.setattr(campaign, 'digest', lambda path: campaign.CHECKPOINT_SHA
                        if Path(path) == checkpoint else real_digest(path))
    monkeypatch.setattr(campaign, 'source_commit', lambda: '1' * 40)
    monkeypatch.setattr(utils.eval_series_data, 'load_records', lambda *a, **k: [record])
    evaluation_calls, registry_calls = [], []
    def specifications(*a, **kwargs):
        evaluation_calls.append(kwargs)
        root = kwargs['eval_series_spec_dir']; root.mkdir()
        for cell in campaign.cells(args.matrix):
            spec = dict(identity={**record['identity'], 'planner': {'type': 'sac', 'name': cell['name']}},
                        selector=cell['selector'])
            (root / (cell['selector'].replace('/', '__') + '.json')).write_text(json.dumps(spec))
    def new_run(*args):
        registry_calls.append(args)
        return dict(run_dir=str(tmp_path / f'run{len(registry_calls)}'), run_id=str(len(registry_calls)))
    monkeypatch.setattr(evaluate_ambi_checkpoint, 'evaluate_matrix', specifications)
    monkeypatch.setattr(utils.eval_series, 'create_run', new_run)
    result = campaign.prepare(args)
    assert len(evaluation_calls) == 1 and len(registry_calls) == (6 if rounds is None else 2)
    assert evaluation_calls[0]['reference_bundle'] == reference
    assert evaluation_calls[0]['eval_series_spec_dir'] == args.root / 'specs'
    assert result['source_run'] == campaign.SOURCE_RUN and result['checkpoint_step'] == 575000
    assert result['H'] == horizon
    assert result['prior_manifest_sha256'] == real_digest(reference / 'manifest.json')
    assert result['prior_source_science'] == {'old': True}
    assert all(not cell['reused'] and cell['actual_selector'] == cell['selector'] for cell in result['cells'])
    assert not (args.root / 'unused').exists()  # Specifications do not construct an evaluator bundle.


@pytest.mark.parametrize('smoke,index', [(True, 0), (True, 1), (False, 0), (False, 'last')])
@pytest.mark.parametrize('horizon', [1, 2, 3])
@pytest.mark.parametrize('rounds', [None, 6, 8])
def test_worker_owns_one_complete_setting(tmp_path, monkeypatch, smoke, index, horizon, rounds):
    import torch
    import evaluate_ambi_checkpoint
    import utils.ambi_seed_shards
    panel = campaign.cells(matrix_path(horizon, rounds=rounds))
    index = len(panel) - 1 if index == 'last' else index
    for cell in panel:
        cell['directory'] = str(tmp_path / cell['name'])
        cell['bundle'] = str(Path(cell['directory']) / 'bundle')
    settings = dict(source_commit='1' * 40, checkpoint_step=575000, checkpoint_sha256=campaign.CHECKPOINT_SHA,
                    source_run=campaign.SOURCE_RUN, cells=panel, matrix=str(matrix_path(horizon, rounds=rounds)),
                    H=horizon,
                    checkpoint='checkpoint', inventory='inventory', reference='existing-prior')
    (tmp_path / 'campaign.json').write_text(json.dumps(settings))
    monkeypatch.setattr(campaign, 'source_commit', lambda: '1' * 40)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'Test GPU')
    calls = []
    def evaluate(*a, **kwargs):
        calls.append(kwargs)
        directory = kwargs['bundle_dir']; directory.mkdir()
        (directory / 'manifest.json').write_text('{}')
        (directory / 'trace.jsonl.gz').write_bytes(b'trace')
    monkeypatch.setattr(evaluate_ambi_checkpoint, 'evaluate_matrix', evaluate)
    monkeypatch.setattr(campaign, 'validate_completed', lambda *a, **k: {'runs': [{'trace_files': ['trace.jsonl.gz']}]})
    monkeypatch.setattr(campaign, 'training_summary', lambda *a, **k: {'trace_rows_checked': 99})
    sealed = []
    monkeypatch.setattr(utils.ambi_seed_shards, 'seal_episode_bundle', lambda path: sealed.append(path))
    receipt = campaign.worker(SimpleNamespace(root=tmp_path, index=index, smoke=smoke))
    assert len(calls) == len(sealed) == 1
    assert calls[0]['selectors'] == [panel[index]['selector']]
    assert calls[0]['seeds'] == ([101] if smoke else campaign.SEEDS)
    assert calls[0]['max_steps'] == (3 if smoke else 500)
    assert calls[0]['reference_bundle'] == (None if smoke else 'existing-prior')
    assert receipt['status'] == 'complete' and receipt['checkpoint_step'] == 575000
    assert receipt['H'] == horizon
    assert receipt['trace_sha256']['trace.jsonl.gz']
    assert (Path(receipt['bundle']).parent / 'validation.json').is_file()
    assert (Path(receipt['bundle']).parent / 'worker-completion.json').is_file()
