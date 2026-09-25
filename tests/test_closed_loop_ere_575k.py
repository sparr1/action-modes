"""Pin ERE scope, historical reuse, explicit registry identities, and replay telemetry."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_ere as campaign
from utils.ambi_benchmark import solver_seed


def episodes(offset=0):
    return [dict(seed=s, solver_seed=solver_seed(55, 'episode', s), length=500,
                 truncated_by_evaluator=False, paired_return_delta=float(offset),
                 **{'return': float(s + offset)}) for s in campaign.SEEDS]


def test_complete_scope_and_exact_uniform_j1_reuse():
    panel = campaign.cells()
    assert [(c['H'], c['J']) for c in panel] == [(h, j) for h in (1, 2, 3) for j in (1, 2, 4, 6, 8, 10)]
    assert len(panel) == len({c['name'] for c in panel}) == 18
    assert [i for i, c in enumerate(panel) if c['reused']] == [0, 6, 12]
    assert sum(not c['reused'] for c in panel)*len(campaign.SEEDS) == 75
    for cell in panel:
        p = cell['params']
        assert {k: p[k] for k in campaign.ERE_SETTINGS} == campaign.ERE_SETTINGS
        assert p['inner_critic_source'] == p['inner_horizon_critic_source'] == 'aux_return'
        assert p['inner_sac_critic_target'] == 'reward_only' and p['inner_terminal_entropy'] == 'none'
        assert p['inner_eval_execution_action'] == p['inner_execution_action'] == 'mean'
        assert p['inner_sac_return_estimator'] == 'one_step'
        assert p['inner_replay_capacity'] == max(3072, 384*cell['J'])
        assert p['inner_replay_capacity'] >= 128*cell['H']*cell['J']
        assert p['inner_temperature_initialization'] == p['inner_target_entropy'] == 'inherit_outer'
        assert p['inner_temperature_mode'] == 'auto'


@pytest.mark.parametrize('cell', campaign.cells(), ids=lambda c: c['name'])
def test_resolved_update_budget_and_replay_scope(cell):
    from tests.test_ambi_root_local_sac import _build_cfg
    cfg = _build_cfg(**cell['params'], aux_return_mode='sac', log_std_mapping='direct_clamp',
                     target_entropy=-10.5, sac_actor_loss_scale_mode='none')
    assert cfg.inner_model_step_budget == 128*cell['H']*cell['J']
    assert cfg.inner_critic_updates_per_action == 16*cell['J']
    assert cfg.inner_actor_updates_per_action == cfg.inner_temperature_updates_per_action == 4*cell['J']
    assert cfg.inner_replay_strategy == 'ere' and cfg.inner_ere_final_fraction == .25
    assert cfg.inner_ere_min_rounds == 1 and cfg.inner_ere_actor
    assert cfg.inner_actor_scope == cfg.inner_critic_scope == cfg.inner_replay_scope == 'action'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'fraction', 'actor', 'min_rounds', 'capacity', 'critic', 'execution', 'retrace'])
def test_rejects_scope_or_scientific_drift(tmp_path, mutation):
    matrix = campaign.read(campaign.MATRIX)
    selectors = matrix['evaluation']['default_presets']
    p = matrix['comparisons']['sweep']['variants'][selectors[-1].split('/')[1]]['alg_params']
    if mutation == 'missing': selectors.pop()
    elif mutation == 'duplicate': selectors[0] = selectors[1]
    else:
        key, value = {'fraction': ('inner_ere_final_fraction', .5), 'actor': ('inner_ere_actor', False),
            'min_rounds': ('inner_ere_min_rounds', 2), 'capacity': ('inner_replay_capacity', 3072),
            'critic': ('inner_critic_source', 'sac'), 'execution': ('inner_eval_execution_action', 'policy_sample'),
            'retrace': ('inner_sac_return_estimator', 'retrace')}[mutation]
        p[key] = value
    path = tmp_path / 'matrix.json'; campaign.write(path, matrix)
    with pytest.raises(AssertionError): campaign.cells(path)


def test_old_defaults_normalize_without_allowing_unrelated_differences():
    old = campaign.historical_cell(3, 10)['params']
    actual = {**old, 'inner_eval_execution_action': 'mean', 'inner_sac_return_estimator': 'one_step',
              'inner_retrace_lambda': 1., 'inner_retrace_batch_trajectories': None, **campaign.ERE_SETTINGS}
    campaign.matching_uniform_config(actual, old)
    campaign.matching_uniform_config({**actual, 'device': 'cpu'}, {**old, 'device': 'cuda'})
    for key, value in [('inner_actor_lr', .01), ('inner_replay_capacity', 3072),
                       ('inner_ere_final_fraction', .5), ('inner_ere_actor', False),
                       ('inner_temperature_initialization', 'fixed')]:
        with pytest.raises(AssertionError): campaign.matching_uniform_config({**actual, key: value}, old)
    with pytest.raises(AssertionError):
        campaign.matching_uniform_config(actual, {**old, 'inner_replay_strategy': 'ere'})


def test_uniform_inventory_is_exact_and_preserves_all_historical_run_ids():
    pins = campaign.read(campaign.REFERENCES)
    refs = pins['uniform_references']
    assert pins['checkpoint_step'] == 575000 and pins['checkpoint_sha256'] == campaign.CHECKPOINT_SHA
    assert pins['prior_reference']['initial_alpha'] == campaign.INITIAL_ALPHA
    assert len(refs) == 18 and {(r['H'], r['J']) for r in refs} == {(c['H'], c['J']) for c in campaign.cells()}
    assert len({r['manifest_sha256'] for r in refs}) == len({r['performance_run_id'] for r in refs}) == 18
    historical = campaign.read(campaign.ROOT / 'configs/research/ambi_closed_loop_reward_retrace_refs_575k.json')['references']
    for ref in refs:
        old, = [r for r in historical if (r['H'], r['J'], r['execution']) == (ref['H'], ref['J'], 'mean')]
        assert all(ref[k] == value for k, value in old.items())
        assert Path(ref['run_dir']).name == ref['performance_run_id']
        assert ref['checkpoint_sha256'] == campaign.CHECKPOINT_SHA


@pytest.mark.parametrize('horizon', [1, 2, 3])
@pytest.mark.parametrize('component,slots', [('critic', 16), ('actor', 4)])
def test_every_update_requires_exact_round_based_window(horizon, component, slots):
    from RL.tdmpc2_core.common.ere import round_windows
    cell = next(c for c in campaign.cells() if c['H'] == horizon and c['J'] == 10)
    for r in range(1, 11):
        for k, window in enumerate(round_windows(r, slots, .25, 1)):
            prefix = component + '_replay_'
            event = dict(round_index=r, **{component + '_updates': (r - 1)*slots + k + 1,
                'updated_' + component: True}, metrics={prefix + key: value for key, value in dict(
                window_rounds=window, window_transitions=window*128*horizon,
                window_fraction=window/r, window_round_fraction=window/r,
                round_age_min=0, round_age_mean=(window - 1)/2, round_age_max=window - 1,
                newest_round_fraction=1/window, batch_unique_fraction=.5).items()})
            campaign.validate_update(event, cell)
            broken = deepcopy(event); broken['metrics'][prefix + 'window_rounds'] += 1
            with pytest.raises(AssertionError): campaign.validate_update(broken, cell)


def test_decision_requires_complete_actor_and_critic_draw_totals():
    cell = campaign.cells()[-1]
    event = dict(metrics={f'decision/inner_{component}_replay_round_{r}_sample_count': slots*256
                         for component, slots in [('critic', 16), ('actor', 4)] for r in range(1, 11)})
    campaign.validate_decision(event, cell)
    event['metrics']['decision/inner_actor_replay_round_10_sample_count'] -= 1
    with pytest.raises(AssertionError): campaign.validate_decision(event, cell)


@pytest.mark.parametrize('late_config_drift', [False, True])
def test_prepare_creates_fifteen_new_identities_and_never_republishes_j1(tmp_path, monkeypatch, late_config_drift):
    import evaluate_ambi_checkpoint
    from utils import eval_series, eval_series_data
    pins = campaign.read(campaign.REFERENCES)
    checkpoint = tmp_path / 'checkpoint.pt'; metadata_path = Path(str(checkpoint) + '.metadata.json')
    metadata = {'checkpoint': {'step': 575000}}
    campaign.write(metadata_path, metadata)
    prior_bundle = tmp_path / 'prior'; prior_bundle.mkdir()
    campaign.write(prior_bundle / 'manifest.json', {'checkpoint': {'metadata': metadata}})
    protocol = dict(action_rule='tanh_mean', max_steps=500, controller_seed=55,
                    seed_scheme='sha256-v1', environment={'id': 'test'})
    prior = {**pins['prior_reference'], 'bundle': str(prior_bundle), 'runtime': {'python': 'locked'},
        'protocol': protocol, 'episodes': episodes(),
        'identity': dict(backbone=campaign.SOURCE_RUN, protocol=protocol, planner={'type': 'prior'})}
    refs = {}
    def planner(cfg, *args): return dict(type='inner_sac', action_rule='tanh_mean', settings=deepcopy(cfg))
    for pin in pins['uniform_references']:
        cfg = campaign.historical_cell(pin['H'], pin['J'])['params']
        refs[(pin['H'], pin['J'])] = {**pin, 'runtime': prior['runtime'], 'protocol': protocol,
            'resolved_config': cfg, 'episodes': episodes(pin['J']),
            'reference_manifest_sha256': prior['manifest_sha256'],
            'identity': dict(backbone=campaign.SOURCE_RUN, protocol=protocol, planner=planner(cfg))}
    row = dict(step=575000, path=str(checkpoint), metadata_path=str(metadata_path),
               sha256=campaign.CHECKPOINT_SHA, metadata_sha256=prior['metadata_sha256'])
    inventory_path = tmp_path / 'inventory.json'
    campaign.write(inventory_path, dict(source_run=campaign.SOURCE_RUN, checkpoints=[row]))
    monkeypatch.setattr(campaign, 'source_commit', lambda: 'tested')
    monkeypatch.setattr(campaign, 'digest', lambda p: row['metadata_sha256'] if str(p).endswith('.metadata.json') else row['sha256'])
    monkeypatch.setattr(campaign, 'load_prior', lambda *a: deepcopy(prior))
    monkeypatch.setattr(campaign, 'load_uniform', lambda pin, inv: deepcopy(refs[(pin['H'], pin['J'])]))
    monkeypatch.setattr(campaign, 'checkpoint_state_proof', lambda *a: dict(initial_alpha=campaign.INITIAL_ALPHA))
    configs = {c['selector']: c['params'] for c in campaign.cells()}
    if late_config_drift:
        configs[campaign.cells()[-1]['selector']]['inner_actor_lr'] = .01
    monkeypatch.setattr(campaign, 'resolve_config', lambda path, cp, sel: deepcopy(configs[sel]))
    monkeypatch.setattr(eval_series_data, 'planner_identity', planner)
    def evaluate(*args, **kwargs):
        specs = kwargs['eval_series_spec_dir']; specs.mkdir()
        assert kwargs['reference_bundle'] == prior['bundle'] and kwargs['seeds'] == campaign.SEEDS
        result = {}
        for cell in campaign.cells():
            path = specs / (cell['name'] + '.json')
            campaign.write(path, dict(identity=dict(backbone=campaign.SOURCE_RUN, protocol=protocol,
                                                    planner=planner(cell['params']))))
            result[cell['selector']] = str(path)
        return dict(mode='evaluation_series_specifications', specs=result)
    monkeypatch.setattr(evaluate_ambi_checkpoint, 'evaluate_matrix', evaluate)
    created = []
    def create(*args):
        created.append(args[2]); return dict(run_id=f'new-{len(created)}', run_dir=str(tmp_path / f'new-{len(created)}'))
    monkeypatch.setattr(eval_series, 'create_run', create)
    args = SimpleNamespace(root=tmp_path / 'campaign', inventory=inventory_path,
        matrix=campaign.MATRIX, references=campaign.REFERENCES, registry=tmp_path / 'registry', group='test', label='Test')
    if late_config_drift:
        with pytest.raises(AssertionError): campaign.prepare(args)
        assert not created
        return
    state = campaign.prepare(args)
    assert len(created) == len(set(created)) == 15
    assert len(state['production_indices']) == 15 and state['smoke_indices'] == [1, 11, 17]
    assert len(state['uniform_references']) == 18 and not state['ere_annealing']
    for cell in state['cells']:
        old = refs[(cell['H'], cell['J'])]
        assert cell['uniform_reference'] == old
        if cell['reused']:
            assert cell['identity'] == old['identity'] and cell['requested_identity'] != old['identity']
            for key in ('bundle', 'run_dir', 'performance_run_id', 'training_run_id'):
                assert cell[key] == old[key]
        else:
            assert cell['performance_run_id'].startswith('new-') and cell['bundle'].startswith(str(tmp_path / 'campaign'))


def test_prepare_uses_real_checkpoint_metadata_specs_and_registry(tmp_path, monkeypatch):
    """Exercise actual config/identity resolution for all 18 cells without live evaluation."""
    import subprocess
    import torch
    import evaluate_ambi_checkpoint as evaluator
    from utils import ambi_benchmark
    from utils.eval_series import load_run
    from utils.eval_series_data import identity_for_ambi_checkpoint, resolved_checkpoint_config, planner_identity
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=campaign.ROOT, text=True).strip()
    monkeypatch.setattr(ambi_benchmark, 'code_identity', lambda: dict(commit=commit, dirty=False))
    monkeypatch.setattr(campaign, 'source_commit', lambda: commit)
    monkeypatch.setattr(evaluator, '_make_env', lambda *a, **k: pytest.fail('preparation constructed environment'))
    monkeypatch.setattr(evaluator, 'evaluate_preset', lambda *a, **k: pytest.fail('preparation evaluated episode'))
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(dict(model={}, aux_return_state={}, log_ent_coef=torch.tensor(campaign.INITIAL_ALPHA).log()), checkpoint)
    params = dict(aux_return_mode='sac', aux_return_detach_representation=False, target_entropy=-10.5,
        log_std_mapping='direct_clamp', sac_actor_loss_scale_mode='none', aux_return_sac_actor_loss_scale_mode='none',
        train_unroll_horizon=3)
    trial = dict(alg='AMBITDMPC2/AMBITDMPC2', env='DMControl-v0', seed=55, total_steps=1000000,
        alg_params=params, resolved_runtime={'observation': dict(mode='state', shape=[67], action_dim=21, episode_length=500)})
    metadata = dict(schema_version=1, trial_run_params=trial,
        experiment_params={'env_params': {'task': 'humanoid-walk', 'obs': 'state'}},
        checkpoint=dict(kind='periodic', step=575000, episode=1150, best_score=None, best_window=1))
    sidecar = Path(str(checkpoint) + '.metadata.json'); campaign.write(sidecar, metadata)
    row = dict(step=575000, path=str(checkpoint), metadata_path=str(sidecar), sha256=campaign.digest(checkpoint),
               metadata_sha256=campaign.digest(sidecar))
    monkeypatch.setattr(campaign, 'CHECKPOINT_SHA', row['sha256'])
    inventory = tmp_path / 'inventory.json'
    campaign.write(inventory, dict(source_run=campaign.SOURCE_RUN, checkpoints=[row]))
    cp = dict(path=row['path'], metadata=metadata, sha256=row['sha256'], source_run=campaign.SOURCE_RUN)
    prior_resolved = dict(algorithm_config={**trial, 'alg_params': {**params, 'inner_operator': 'none'}},
                         environment=dict(id='DMControl-v0', params=metadata['experiment_params']['env_params']))
    protocol = ambi_benchmark.protocol_for(prior_resolved, 55, 500)
    identity = identity_for_ambi_checkpoint(cp, prior_resolved, protocol, campaign.SEEDS,
        dict(commit=commit, dirty=False), path=row['path'], inventory_path=inventory)
    directory = tmp_path / 'prior'; directory.mkdir()
    manifest = dict(schema_version=1, status='complete', checkpoint=cp, protocol=protocol,
        runs=[dict(status='complete', config={'alg_params': {'inner_operator': 'none'}}, episodes=episodes())])
    campaign.write(directory / 'manifest.json', manifest)
    prior = dict(checkpoint_step=575000, checkpoint_sha256=row['sha256'], metadata_sha256=row['metadata_sha256'],
        bundle=str(directory), manifest_sha256=campaign.digest(directory / 'manifest.json'), source_commit=commit,
        initial_alpha=campaign.INITIAL_ALPHA, episodes=episodes(), identity=identity, protocol=protocol,
        runtime={'locked': True}, resolved_config=resolved_checkpoint_config(cp, prior_resolved))
    references = []
    for cell in campaign.cells():
        h, j = cell['H'], cell['J']
        suffix = '' if h == 3 else f'_h{h}'
        if j >= 6: suffix = f'_h{h}_j{j}'
        path = campaign.ROOT / f'configs/research/ambi_closed_loop_critics{suffix}_575k.json'
        selector = campaign.historical_cell(h, j)['selector']
        old = campaign.resolve_config(path, checkpoint, selector)
        # Simulate saved manifests written before the opt-in ERE and execution fields existed.
        for key in (*campaign.ERE_DEFAULTS, 'inner_eval_execution_action', 'inner_sac_return_estimator',
                    'inner_retrace_lambda', 'inner_retrace_batch_trajectories'):
            old.pop(key, None)
        references.append({**deepcopy(prior), 'H': h, 'J': j, 'execution': 'mean', 'estimator': 'one_step',
            'resolved_config': old, 'reference_manifest_sha256': prior['manifest_sha256'],
            'performance_run_id': f'old-{h}-{j}', 'training_run_id': f'training-{h}-{j}', 'run_dir': f'/original/{h}/{j}',
            'identity': {**identity, 'planner': planner_identity(old, {}, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean')}})
    pins = tmp_path / 'pins.json'
    campaign.write(pins, dict(source_run=campaign.SOURCE_RUN, checkpoint_step=575000,
        checkpoint_sha256=row['sha256'], prior_reference=prior, uniform_references=references))
    monkeypatch.setattr(campaign, 'load_prior', lambda pin, inv: deepcopy(pin))
    monkeypatch.setattr(campaign, 'load_uniform', lambda pin, inv: deepcopy(pin))
    args = SimpleNamespace(root=tmp_path / 'campaign', inventory=inventory, references=pins,
                           matrix=campaign.MATRIX, registry=tmp_path / 'registry', group='test', label='Test')
    result = campaign.prepare(args)
    assert len(list(args.registry.iterdir())) == 15
    assert not (args.root / 'unused').exists()
    for cell in result['cells']:
        spec = campaign.read(args.root / 'specs' / (cell['selector'].replace('/', '__') + '.json'))
        assert spec['identity']['planner']['settings']['inner_replay_strategy'] == 'ere'
        assert cell['expected_config']['inner_rounds'] == cell['J']
        assert cell['expected_config']['inner_rollout_horizon'] == cell['H']
        if cell['reused']:
            assert cell['identity'] == cell['uniform_reference']['identity']
            assert cell['requested_identity'] == spec['identity']
        else:
            assert load_run(cell['run_dir'])['identity'] == spec['identity']
