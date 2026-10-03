"""Pinned workloads, immutable ownership, smoke gates, and trace validation."""
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_transfer_sweep_campaign as campaign


def test_selected_panel_is_exactly_24_and_heavy_first():
    cells = campaign.cells()
    assert len(cells) == len({c['name'] for c in cells}) == 24
    assert [(c['J'], c['solve_interval']) for c in cells] == [(8, 1)]*6 + [(8, 3)]*6 + [(1, 1)]*6 + [(1, 3)]*6
    assert {c['transfer_mode'] for c in cells} == {'fresh', 'actor_only', 'critic_only'}
    assert {c['critic_kind'] for c in cells} == {'soft', 'return'}
    assert sum(c['J'] == 8 for c in cells) == 12


def test_atomic_output_never_overwrites(tmp_path):
    output = tmp_path / 'receipt.json'
    campaign.write_new(output, {'first': True})
    with pytest.raises(FileExistsError):
        campaign.write_new(output, {'first': False})
    assert campaign.read(output) == {'first': True}
    assert list(tmp_path.iterdir()) == [output]


def synthetic_trace(cell, *, steps=7, seeds=(101, 102)):
    for seed in seeds:
        for d in range(steps):
            solved = d % cell['solve_interval'] == 0
            dose = cell['J'] if solved else 0
            actor = solved and d > 0 and cell['transfer_mode'] == 'actor_only'
            metrics = dict(inner_rounds=dose, inner_actor_transferred=int(actor),
                inner_first_action_rounds_applied=0, inner_critic_optimizer_steps=16*dose,
                inner_actor_optimizer_steps=4*dose, inner_temperature_optimizer_steps=4*dose,
                inner_model_steps=128*3*dose, inner_compile_fallback=0,
                inner_solve_performed=int(solved), inner_policy_held=int(not solved),
                inner_solve_index=d//cell['solve_interval'], inner_action_age=d%cell['solve_interval'],
                inner_episode_decision_index=d, inner_solve_interval=cell['solve_interval'])
            if cell['transfer_mode'] in ('critic_only', 'critic_hidden'):
                metrics.update(inner_critic_transferred=int(solved and d > 0),
                    inner_critic_target_reinitialized=int(solved),
                    inner_critic_updates_initial=16*cell['J']*(d//cell['solve_interval']) if solved else 0)
            if cell['transfer_mode'] == 'critic_hidden':
                metrics['inner_critic_head_reinitialized'] = int(solved)
            common = dict(episode_id=f'seed-{seed}', decision_index=d, nonfinite={})
            if solved:
                initial = dict(metrics, alpha=campaign.INITIAL_ALPHA,
                    inner_actor_lifetime_updates_initial=4*cell['J']*(d//cell['solve_interval']) if actor else 0,
                    actor_optimizer_steps_initial=0, critic_optimizer_steps_initial=0,
                    temperature_optimizer_steps_initial=0)
                yield dict(common, phase='initial', replay_size=0, metrics=initial)
                for phase, count in [('collection', dose), ('update', 20*dose),
                                     ('probe', dose+3), ('transfer_probe', dose+3)]:
                    for _ in range(count):
                        yield dict(common, phase=phase, metrics={})
            yield dict(common, phase='decision', metrics={f'decision/{k}': v for k, v in metrics.items()})


def trace_bundle(tmp_path, rows):
    bundle = tmp_path / 'bundle'; bundle.mkdir()
    with gzip.open(bundle / 'events.jsonl.gz', 'wt') as stream:
        counters = {}
        for row in rows:
            key = row['episode_id'], row['decision_index']
            row.setdefault('event_index', counters.get(key, 0))
            counters[key] = counters.get(key, 0) + 1
            stream.write(json.dumps(row)+'\n')
    (bundle / 'manifest.json').write_text(json.dumps({'runs': [{'trace_files': ['events.jsonl.gz']}]}))
    return bundle


@pytest.mark.parametrize('mode', ['fresh', 'actor_only', 'critic_only', 'critic_hidden'])
@pytest.mark.parametrize('interval', [1, 3])
@pytest.mark.parametrize('rounds', [1, 8])
def test_trace_checks_work_and_episode_counters(tmp_path, mode, interval, rounds):
    cell = dict(transfer_mode=mode, solve_interval=interval, J=rounds)
    bundle = trace_bundle(tmp_path, synthetic_trace(cell))
    summary = campaign.validate_trace(bundle, cell, seeds=[101, 102], steps=7)
    assert summary['decisions'] == 14
    assert summary['solves'] == 2*math.ceil(7/interval)
    assert summary['total_rounds'] == summary['solves']*rounds


@pytest.mark.parametrize('metric', ['inner_critic_transferred', 'inner_critic_target_reinitialized',
                                    'inner_critic_updates_initial', 'inner_compile_fallback'])
def test_trace_rejects_misreported_critic_lifecycle(tmp_path, metric):
    cell = dict(transfer_mode='critic_only', solve_interval=3, J=8)
    rows = list(synthetic_trace(cell))
    row = next(r for r in rows if r['phase'] == 'decision' and r['decision_index'] == 3)
    row['metrics']['decision/'+metric] += 1
    bundle = trace_bundle(tmp_path, rows)
    with pytest.raises(ValueError):
        campaign.validate_trace(bundle, cell, seeds=[101, 102], steps=7)


@pytest.mark.parametrize('mode', ['full', 'critic-hidden'])
def test_prepare_pins_jobs_and_refuses_existing_root(tmp_path, monkeypatch, mode):
    import torch
    from slurm import ambi_closed_loop_checkpoint_sweep as helpers
    from utils import eval_series_data as identities
    checkpoint = tmp_path / '575k.pt'; checkpoint.write_bytes(b'fixture')
    sidecar = Path(str(checkpoint)+'.metadata.json')
    sidecar.write_text(json.dumps({'checkpoint': {'step': 575000}}))
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps(dict(source_run=campaign.SOURCE_RUN, checkpoints=[dict(
        step=575000, path=str(checkpoint), sha256=campaign.CHECKPOINT_SHA,
        metadata_path=str(sidecar), metadata_sha256=campaign.digest(sidecar))])))
    digest = campaign.digest
    monkeypatch.setattr(campaign, 'digest', lambda p: campaign.CHECKPOINT_SHA if Path(p)==checkpoint else digest(p))
    monkeypatch.setattr(campaign, 'source_commit', lambda: 'a'*40)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: dict(model={}, aux_return_state={},
                       log_ent_coef=torch.tensor(campaign.INITIAL_ALPHA).log()))
    def config(path, checkpoint, selector):
        matrix = campaign.read(path); group, name = selector.split('/')
        return {**matrix['shared_alg_params'], **matrix['comparisons'][group]['variants'][name]['alg_params'],
                'sac_actor_loss_scale_mode': 'none', 'aux_return_sac_actor_loss_scale_mode': 'none',
                'target_entropy': -10.5}
    monkeypatch.setattr(helpers, 'resolve_config', config)
    monkeypatch.setattr(identities, 'scientific_identity', lambda *a: {'fixture': True})
    args = SimpleNamespace(root=tmp_path/'campaign', inventory=inventory, matrix=None,
                           prior_matrix=campaign.PRIOR_MATRIX, campaign_mode=mode)
    value = campaign.prepare(args)
    hidden = mode == 'critic-hidden'
    assert len(value['cells']) == (8 if hidden else 25)
    assert value['production_indices'] == list(range(8 if hidden else 25))
    assert value['smoke_indices'] == list(range(8 if hidden else 12))
    assert value['cuda_gate_index'] == 0
    if hidden:
        assert value['schema_version'] == 2
        assert value['kind'] == 'ambi-critic-hidden-transfer-sweep-v1'
        campaign.validate_hidden_campaign(value)
        wrong = deepcopy(value)
        wrong['cells'][0]['requested_alg_params']['inner_critic_transfer_head'] = 'retain'
        with pytest.raises(ValueError, match='setting identity'):
            campaign.validate_hidden_campaign(wrong)
    else:
        assert value['schema_version'] == 1
        assert value['cells'][24]['transfer_mode'] == 'prior'
        assert value['cells'][24]['selector'] == 'reference/prior'
    with pytest.raises(FileExistsError):
        campaign.prepare(args)


def test_receipt_rejects_modified_smoke_gate_output(tmp_path):
    root = tmp_path
    value = dict(cells=[dict(name='setting')]); campaign.write_new(root/'campaign.json', value)
    directory = root/'smoke/setting'; (directory/'bundle').mkdir(parents=True)
    (directory/'bundle/manifest.json').write_text('{}')
    (directory/'gate.json').write_text('{}'); (directory/'cuda-lifecycle-gate.log').write_text('passed')
    receipt = dict(status='complete', index=0, smoke=True, cell='setting',
        campaign_sha256=campaign.digest(root/'campaign.json'), gpu='NVIDIA L40S',
        manifest_sha256=campaign.digest(directory/'bundle/manifest.json'), trace_sha256={},
        cuda_gate=dict(status='passed', reports={'gate.json': campaign.digest(directory/'gate.json')},
            log_sha256=campaign.digest(directory/'cuda-lifecycle-gate.log')))
    campaign.write_new(directory/'worker-completion.json', receipt)
    campaign.receipt(root, value, 0, smoke=True, verify=True)
    (directory/'gate.json').write_text('changed')
    with pytest.raises(ValueError, match='CUDA gate report changed'):
        campaign.receipt(root, value, 0, smoke=True, verify=True)


@pytest.mark.parametrize('mode', ['fresh', 'actor_only', 'critic_only', 'critic_hidden', 'prior'])
def test_actual_evaluator_bundle_matches_campaign_validation(tmp_path, monkeypatch, mode):
    prior = mode == 'prior'
    import evaluate_ambi_checkpoint as evaluator
    from tests.test_ambi_root_local_sac import _tiny_component_model, _tiny_params
    from utils import eval_series_data as identities
    options = dict(aux_return_mode='sac', inner_actor_scope='episode' if mode == 'actor_only' else 'action',
        inner_critic_scope='episode' if mode in ('critic_only', 'critic_hidden') else 'action',
        inner_critic_transfer_head='random' if mode == 'critic_hidden' else 'retain',
        inner_critic_source='aux_return', inner_horizon_critic_source='aux_return',
        inner_sac_critic_target='reward_only', inner_rebase_persistent=False,
        inner_rounds=1, inner_first_action_rounds=None, inner_rollouts_per_round=128,
        inner_rollout_horizon=3, inner_replay_capacity=3072, inner_batch_size=256,
        inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
        inner_actor_initialization='prior', inner_critic_initialization='prior',
        inner_critic_target_initialization='online', inner_solve_interval=3)
    # Keep fixture weights tiny; the workload and trace counts match production.
    model = _tiny_component_model(**options)
    checkpoint = tmp_path/'tiny.pt'; model.agent.save(checkpoint)
    model.close()
    params = _tiny_params(**options); params.pop('inner_updates_per_round')
    config = dict(alg='AMBITDMPC2/AMBITDMPC2', env='Pendulum-v1', seed=13, device='cpu', total_steps=10, alg_params=params)
    metadata = dict(schema_version=1, trial_run_params=config, experiment_params={'env_params': {'max_episode_steps': 7}},
        checkpoint=dict(kind='periodic', step=10, episode=2, best_score=None, best_window=1))
    sidecar = Path(str(checkpoint)+'.metadata.json'); sidecar.write_text(json.dumps(metadata))
    override = (dict(inner_operator='none', inner_rounds=0, inner_rollouts_per_round=0,
                    inner_critic_updates_per_round=None, inner_actor_updates_per_round=None,
                    inner_updates_per_round=0, inner_critic_scope='action', inner_solve_interval=1,
                    inner_temperature_mode='inherit_outer') if prior else {})
    evaluation = dict(seeds=[101, 102], controller_seed=55, max_steps=7, default_presets=['test/arm'])
    if not prior:
        evaluation.update(transfer_diagnostics=True, togo_return_rollouts=2)
    matrix = dict(schema_version=1, base_alg_config='checkpoint', source_run=campaign.SOURCE_RUN,
        shared_alg_params=override, evaluation=evaluation,
        comparisons={'test': {'reference': 'arm', 'variants': {'arm': {'alg_params': {}}}}})
    if not prior:
        matrix['study_protocol'] = 'actor-transfer-hold-h-v1' if mode == 'actor_only' else 'critic-transfer-hold-h-v1'
    if mode == 'critic_hidden':
        matrix['study_protocol'] = 'critic-hidden-transfer-hold-h-v1'
    path=tmp_path/'matrix.json'; path.write_text(json.dumps(matrix)); bundle=tmp_path/'bundle'
    evaluator.evaluate_matrix(path, checkpoint, bundle_dir=bundle)
    manifest=campaign.read(bundle/'manifest.json'); run=manifest['runs'][0]
    # CPU fixture validates all recorder contracts; only device/source proof is mocked.
    manifest['code'].update(commit='a'*40, dirty=False)
    manifest['checkpoint']['sha256']=campaign.CHECKPOINT_SHA
    run['result']['resolved_device']='cuda:0'
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(identities,'scientific_identity',lambda *a: {'fixture':True})
    record=dict(source_commit='a'*40, checkpoint=str(checkpoint), metadata_sha256=campaign.digest(sidecar), science={'fixture':True})
    if mode == 'critic_hidden':
        record['transfer_metadata'] = campaign.HIDDEN_TRANSFER_METADATA
    cell=dict(selector='test/arm', expected_config=run['resolved_config'], transfer_mode=mode,
        J=0 if prior else 1, H=3, solve_interval=1 if prior else 3,
        study_protocol=matrix.get('study_protocol'),
        planner_identity=identities.planner_identity(run['resolved_config'],run['result'],'AMBITDMPC2/AMBITDMPC2','tanh_mean'))
    campaign.validate_completed(bundle,cell,record,smoke=True)
    initial_alpha=run['result']['model_metrics']['inner_alpha_initial']['mean'] if not prior else 1
    summary=campaign.validate_trace(bundle,cell,seeds=[101,102],steps=7,initial_alpha=initial_alpha)
    assert summary['decisions']==14
    assert summary['solves']==(0 if prior else 6)


def test_trace_rejects_out_of_order_events(tmp_path):
    cell = dict(transfer_mode='critic_only', solve_interval=3, J=1)
    rows = list(synthetic_trace(cell))
    bundle = trace_bundle(tmp_path, rows)
    with gzip.open(bundle/'events.jsonl.gz', 'rt') as handle:
        emitted = [json.loads(line) for line in handle]
    emitted[1], emitted[2] = emitted[2], emitted[1]
    with gzip.open(bundle/'events.jsonl.gz', 'wt') as handle:
        handle.writelines(json.dumps(row)+'\n' for row in emitted)
    with pytest.raises(ValueError, match='event sequence'):
        campaign.validate_trace(bundle, cell, seeds=[101, 102], steps=7)
