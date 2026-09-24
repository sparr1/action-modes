"""The lambda-one value-sampling campaign has 48 fresh, paired-seed settings."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from slurm import ambi_closed_loop_retrace_values as campaign


def test_full_scope_and_largest_combined_smokes():
    panel = campaign.cells()
    assert len(panel) == len({c['selector'] for c in panel}) == 48
    assert [campaign.cell_key(c) for c in panel] == campaign.identities()
    assert set(campaign.identities()) == {
        (h, j, ki, kb) for h in (2, 3) for j in (1, 2, 4, 6, 8, 10)
        for ki, kb in ((1, 1), (4, 1), (1, 4), (4, 4))}
    assert [campaign.cell_key(panel[i]) for i in campaign.SMOKE_INDICES] == [(2, 10, 4, 4), (3, 10, 4, 4)]
    assert sum(c['Ki'] == c['Kb'] == 1 for c in panel) == 12
    assert len(panel) * len(campaign.SEEDS) == 240
    for cell in panel:
        p = cell['params']
        assert p['inner_critic_source'] == p['inner_horizon_critic_source'] == 'aux_return'
        assert p['inner_sac_critic_target'] == 'reward_only' and p['inner_terminal_entropy'] == 'none'
        assert p['inner_eval_execution_action'] == 'policy_sample'
        assert p['inner_retrace_lambda'] == 1.0
        assert p['inner_replay_capacity'] == (3840 if cell['J'] == 10 else 3072)
        assert p['inner_replay_capacity'] >= 128 * cell['H'] * cell['J']


@pytest.mark.parametrize('index', range(48))
def test_all_cells_resolve_requested_work_without_changing_update_dose(index):
    from tests.test_ambi_root_local_sac import _build_cfg
    cell = campaign.cells()[index]
    cfg = _build_cfg(**cell['params'], aux_return_mode='sac', log_std_mapping='direct_clamp',
                     target_entropy=-10.5, sac_actor_loss_scale_mode='none')
    assert cfg.inner_model_step_budget == 128 * cell['H'] * cell['J']
    assert cfg.inner_critic_updates_per_action == 16 * cell['J']
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 4 * cell['J']
    assert cfg.inner_retrace_batch_trajectories == {2: 128, 3: 86}[cell['H']]
    assert cfg.inner_retrace_value_samples == cell['Ki']
    assert cfg.inner_retrace_boundary_value_samples == cell['Kb']
    assert cfg.inner_retrace_lambda == 1.0
    assert cfg.inner_entropy_enabled and cfg.inner_temperature_mode == 'auto'
    assert cfg.inner_temperature_initialization == cfg.inner_target_entropy == 'inherit_outer'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'soft', 'lambda', 'capacity',
                                     'execution', 'inner_samples', 'boundary_samples', 'seeds'])
def test_matrix_rejects_scope_or_scientific_drift(tmp_path, mutation):
    matrix = campaign.read(campaign.MATRIX)
    selectors = matrix['evaluation']['default_presets']
    params = matrix['comparisons']['sweep']['variants'][selectors[1].split('/')[1]]['alg_params']
    if mutation == 'missing':
        selectors.pop()
    elif mutation == 'duplicate':
        selectors[0] = selectors[1]
    elif mutation == 'seeds':
        matrix['evaluation']['seeds'][-1] = 106
    else:
        key, value = {'soft': ('inner_critic_source', 'sac'), 'lambda': ('inner_retrace_lambda', .9),
            'capacity': ('inner_replay_capacity', 3072), 'execution': ('inner_eval_execution_action', 'mean'),
            'inner_samples': ('inner_retrace_value_samples', 1),
            'boundary_samples': ('inner_retrace_boundary_value_samples', 4)}[mutation]
        params[key] = value
    path = tmp_path / 'matrix.json'
    path.write_text(json.dumps(matrix))
    with pytest.raises(AssertionError):
        campaign.cells(path)


def test_prepare_stages_48_distinct_fresh_identities_without_reference_bundles(tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint
    from utils import eval_series
    panel = campaign.cells()
    monkeypatch.setattr(campaign, 'source_commit', lambda: 'tested')
    monkeypatch.setattr(campaign, 'digest', lambda path: campaign.CHECKPOINT_SHA)
    def evaluate(*args, **kwargs):
        assert kwargs['reference_bundle'] is None
        assert kwargs['seeds'] == campaign.SEEDS and kwargs['controller_seed'] == 55
        assert kwargs['max_steps'] == 500
        specs = kwargs['eval_series_spec_dir']
        specs.mkdir()
        for cell in panel:
            identity = dict(backbone=campaign.SOURCE_RUN, protocol=dict(action_rule=campaign.ACTION_RULE),
                planner=dict(type='inner_sac', action_rule=campaign.ACTION_RULE, settings=deepcopy(cell['params'])))
            campaign.write(specs / (cell['selector'].replace('/', '__') + '.json'), {'identity': identity})
    monkeypatch.setattr(evaluate_ambi_checkpoint, 'evaluate_matrix', evaluate)
    created = []
    def create_run(registry, spec, name, *args):
        created.append((spec, name))
        return dict(run_dir=str(registry / name), run_id='performance-' + str(len(created)))
    monkeypatch.setattr(eval_series, 'create_run', create_run)
    args = SimpleNamespace(root=tmp_path / 'campaign', matrix=campaign.MATRIX,
        checkpoint=tmp_path / 'checkpoint.pt', inventory=tmp_path / 'inventory.json',
        registry=tmp_path / 'registry', group='new-lambda1', label='Fresh value averages')
    state = campaign.prepare(args)
    assert len(created) == len(state['cells']) == 48
    assert state['references'] == [] and state['prior_reference'] is None
    assert state['retrace_lambda'] == 1.0 and state['smoke_indices'] == [3, 7]
    assert len({c['performance_run_id'] for c in state['cells']}) == 48
    assert len({c['training_run_id'] for c in state['cells']}) == 48
    assert all(c['reused'] is False and 'mean_reference' not in c for c in state['cells'])
    assert state['source_commit'] == 'tested' and state['matrix_sha256'] == campaign.CHECKPOINT_SHA


def worker_fixture(tmp_path, monkeypatch):
    import torch
    import evaluate_ambi_checkpoint
    from utils import ambi_seed_shards
    panel = campaign.cells()
    for cell in panel:
        directory = tmp_path / cell['name']
        directory.mkdir()
        cell.update(directory=str(directory), bundle=str(directory / 'bundle'))
    state = dict(source_commit='tested', matrix=str(campaign.MATRIX), matrix_sha256='digest',
        checkpoint=str(tmp_path / 'checkpoint.pt'), inventory=str(tmp_path / 'inventory.json'),
        checkpoint_sha256=campaign.CHECKPOINT_SHA, checkpoint_step=campaign.CHECKPOINT_STEP,
        source_run=campaign.SOURCE_RUN, retrace_lambda=1.0, cells=panel, smoke_indices=[3, 7])
    campaign.write(tmp_path / 'campaign.json', state)
    monkeypatch.setattr(campaign, 'source_commit', lambda: 'tested')
    monkeypatch.setattr(campaign, 'digest', lambda path: 'digest')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'mock GPU')
    monkeypatch.setattr(campaign, 'validate_completed', lambda *a, **kw: {'runs': [{'trace_files': ['trace.gz']}]})
    monkeypatch.setattr(campaign, 'training_summary', lambda *a, **kw: {'trace_rows_checked': 123})
    monkeypatch.setattr(ambi_seed_shards, 'seal_episode_bundle', lambda path: None)
    return state, evaluate_ambi_checkpoint


@pytest.mark.parametrize('index,smoke', [(0, False), (3, True), (7, True)])
def test_worker_preserves_complete_seed_ownership_and_emits_hashed_receipt(tmp_path, monkeypatch, index, smoke):
    state, evaluator = worker_fixture(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(evaluator, 'evaluate_matrix', lambda *a, **kw: calls.append(kw))
    receipt = campaign.worker(SimpleNamespace(root=tmp_path, index=index, smoke=smoke))
    assert len(calls) == 1
    call = calls[0]
    assert call['selectors'] == [state['cells'][index]['selector']]
    assert call['seeds'] == ([101] if smoke else campaign.SEEDS)
    assert call['max_steps'] == (3 if smoke else 500) and call['controller_seed'] == 55
    assert call['device'] == 'cuda' and call['reference_bundle'] is None
    assert receipt['execution'] == 'policy_sample' and receipt['retrace_lambda'] == 1.0
    assert (receipt['Ki'], receipt['Kb']) == ((4, 4) if smoke else (1, 1))
    assert receipt['manifest_sha256'] == 'digest' and receipt['trace_sha256'] == {'trace.gz': 'digest'}
    assert receipt['status'] == 'complete' and receipt['reused'] is False


def test_worker_reports_failure_without_publishing_completion(tmp_path, monkeypatch):
    state, evaluator = worker_fixture(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError('test evaluator failure')
    monkeypatch.setattr(evaluator, 'evaluate_matrix', fail)
    with pytest.raises(RuntimeError, match='test evaluator failure'):
        campaign.worker(SimpleNamespace(root=tmp_path, index=0, smoke=False))
    directory = tmp_path / state['cells'][0]['name']
    assert campaign.read(directory / 'worker-failure.json')['status'] == 'failed'
    assert not (directory / 'worker-completion.json').exists()


def test_smoke_rejects_unapproved_small_value_count_cell(tmp_path, monkeypatch):
    _, evaluator = worker_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluator, 'evaluate_matrix', lambda *a, **kw: pytest.fail('must not evaluate'))
    with pytest.raises(AssertionError):
        campaign.worker(SimpleNamespace(root=tmp_path, index=0, smoke=True))


def test_oscar_wrapper_uses_new_modules_and_needs_no_historical_references():
    source = (campaign.ROOT / 'slurm/run_ambi_closed_loop_retrace_values_oscar.sbatch').read_text()
    assert 'slurm/ambi_closed_loop_retrace_values.py' in source
    assert 'slurm/ambi_closed_loop_retrace_values_publish.py' in source
    assert 'EVAL_REFERENCES_PATH' not in source and '--references' not in source
    assert 'EXPECTED_ACTION_MODES_SHA' in source and 'git status --porcelain' in source
    assert 'TORCHINDUCTOR_COMPILE_THREADS=1' in source
