"""Deterministic execution, controls and fresh A4 pairing for the actor screen."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare_campaign(campaign, root, execution):
    assert execution == dict(mode='deterministic-v1', actor_budget_comparison=True,
                             gpu_family='L40S', fresh_prior=True)
    campaign['execution'] = execution
    campaign['preparation_reference'] = campaign['reference']
    campaign['preparation_prior_manifest_sha256'] = campaign.pop('prior_manifest_sha256')
    campaign['preparation_prior_source_science'] = campaign.pop('prior_source_science')
    campaign['reference'] = str(root/'deterministic-prior')
    names = {c['name'] for c in campaign['cells']}
    for cell in campaign['cells']:
        baseline = cell['name'].rsplit('_a',1)[0]+'_a4'
        assert baseline in names
        cell['actor_baseline_name'] = baseline
        candidate = json.loads((root/'specs'/f"sweep__{cell['name']}.json").read_text())
        reference = json.loads((root/'specs'/f'sweep__{baseline}.json').read_text())
        for key in ('backbone','protocol','science'):
            assert candidate['identity'][key] == reference['identity'][key]
        expected = deepcopy(reference['identity']['planner'])
        a = cell['params']['inner_actor_updates_per_round']
        expected['settings']['inner_actor_updates_per_round'] = a
        for key in ('inner_actor_updates_per_action','inner_temperature_updates_per_action'):
            expected['settings'][key] = a*cell['J']
        assert candidate['identity']['planner'] == expected, cell['name']


def configure_execution(campaign):
    if not campaign.get('execution'):
        return
    import torch
    assert campaign['execution']['mode'] == 'deterministic-v1'
    assert not torch.cuda.is_initialized(), 'Set reproducibility before CUDA initialization'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def runtime_receipt(campaign):
    if not campaign.get('execution'):
        return None
    import torch
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    gpu = torch.cuda.get_device_name(0)
    assert 'L40S' in gpu, gpu
    cache = Path(os.environ['TORCHINDUCTOR_CACHE_DIR'])
    artifacts = {str(p.relative_to(cache)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(cache.rglob('*')) if p.is_file() and p.suffix in ('.py','.json')}
    return dict(mode='deterministic-v1',python=platform.python_version(),torch=torch.__version__,
                cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),hostname=platform.node(),gpu=gpu,
                device_provenance=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version',
                                                          '--format=csv,noheader'],text=True).strip(),
                deterministic=True,deterministic_warn_only=False,cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                environment={k:os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE',
                   'TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','OMP_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS')},
                compiler_artifacts=artifacts)


def execution_identity(receipt):
    return {k:receipt[k] for k in ('mode','python','torch','cuda','cudnn','gpu','deterministic',
             'deterministic_warn_only','cudnn_benchmark','matmul_precision','allow_tf32','cudnn_allow_tf32')}


def actor_budget_baseline(candidate, baseline, a):
    """Allow only actor count and its resolved actor/temperature totals to differ."""
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS
    for key in ('backbone','protocol','science'):
        assert candidate['identity'][key] == baseline['identity'][key], key
    assert candidate['checkpoint']['sha256'] == baseline['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert candidate['checkpoint']['step'] == baseline['checkpoint']['step'] == 625000
    expected = deepcopy(baseline['identity']['planner'])
    settings = expected['settings']; j = settings['inner_rounds']
    assert settings['inner_actor_updates_per_round'] == 4
    assert settings['inner_actor_updates_per_action'] == settings['inner_temperature_updates_per_action'] == 4*j
    settings['inner_actor_updates_per_round'] = a
    settings['inner_actor_updates_per_action'] = settings['inner_temperature_updates_per_action'] = a*j
    assert candidate['identity']['planner'] == expected
    assert baseline['metrics']['eval/frozen_state_unchanged']
    assert sorted(e['seed'] for e in baseline['episodes']) == SEEDS
    assert all(e['length']==500 and not e['truncated_by_evaluator'] for e in baseline['episodes'])
    return dict(kind='actor_budget',record_id=baseline['record_id'],
                episodes=[{k:e[k] for k in ('seed','solver_seed','return')} for e in baseline['episodes']])


def fresh_actor_baseline(campaign, cell, record):
    from slurm.ambi_aux_hj_sweep import actor_updates, read, digest, validate
    from utils.eval_series_data import load_records
    baseline = next(c for c in campaign['cells'] if c['name']==cell['actor_baseline_name'])
    receipt = read(Path(baseline['directory'])/'worker-completion.json')
    assert receipt['status']=='complete'
    assert digest(Path(baseline['bundle'])/'manifest.json')==receipt['manifest_sha256']
    validate(baseline['bundle'], baseline)
    current = read(Path(cell['directory'])/'worker-completion.json')
    assert execution_identity(current['execution']) == execution_identity(receipt['execution'])
    previous, = load_records(baseline['bundle'], inventory_path=campaign['inventory'])
    result = actor_budget_baseline(record, previous, actor_updates(cell))
    result.update(performance_run_id=baseline['performance_run_id'],bundle=baseline['bundle'],
                  manifest_sha256=receipt['manifest_sha256'])
    return result


def without_timing(value):
    if isinstance(value,dict):
        return {k:without_timing(v) for k,v in value.items() if 'seconds' not in k and k!='timestamp'}
    if isinstance(value,list):
        return [without_timing(v) for v in value]
    return value


def control_payload(bundle):
    from slurm.ambi_aux_hj_sweep import read
    manifest=read(bundle/'manifest.json'); run,=manifest['runs']
    traces=[]
    for name in run['trace_files']:
        with gzip.open(bundle/name,'rt') as f:
            traces.extend(without_timing(json.loads(line)) for line in f)
    # Selector and directories remain identical across replicas. Timing is the
    # only discarded field; every logged training, replay and probe value is checked.
    return without_timing(dict(episodes=run['episodes'],metrics=run['result']['model_metrics'],traces=traces))


def gate(root):
    from slurm.ambi_aux_hj_sweep import read, write, actor_updates, SEEDS, CHECKPOINT_SHA
    campaign=read(root/'campaign.json')
    checks=[]
    for cell in campaign['cells']:
        if actor_updates(cell) not in (4,16):
            continue
        paths=[root/'smoke'/rep/cell['name']/'bundle' for rep in ('a','b')]
        payloads=[control_payload(p) for p in paths]
        if payloads[0]!=payloads[1]:
            write(root/'control-failure.json',dict(cell=cell['name'],status='different'))
            raise AssertionError('Independent deterministic controls differ: '+cell['name'])
        receipts=[read(p/'execution.json') for p in paths]
        assert execution_identity(receipts[0])==execution_identity(receipts[1])
        assert receipts[0]['environment']['TORCHINDUCTOR_CACHE_DIR']!=receipts[1]['environment']['TORCHINDUCTOR_CACHE_DIR']
        checks.append(dict(cell=cell['name'],exact=True,
                           payload_sha256=hashlib.sha256(json.dumps(payloads[0],sort_keys=True).encode()).hexdigest()))
    assert len(checks)==8
    configure_execution(campaign)
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    bundle=Path(campaign['reference'])
    evaluate_matrix(campaign['matrix'],campaign['checkpoint'],selectors=['sweep/prior'],seeds=SEEDS,
                    controller_seed=55,max_steps=500,device='cuda',bundle_dir=bundle,
                    checkpoint_inventory=campaign['inventory'])
    prior=read(bundle/'manifest.json'); run,=prior['runs']
    assert prior['status']==run['status']=='complete' and prior['checkpoint']['sha256']==CHECKPOINT_SHA
    assert run['result']['outer_state_unchanged']
    assert [e['seed'] for e in run['episodes']]==SEEDS
    assert all(e['length']==500 and not e['truncated_by_evaluator'] for e in run['episodes'])
    write(bundle/'execution.json',runtime_receipt(campaign))
    seal_episode_bundle(bundle)
    write(root/'control-completion.json',dict(status='complete',checks=checks,
          scope='One seed, three real decisions for both arms, H2/H3, A4/A16. No universal determinism claim.',
          prior_manifest_sha256=hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest()))
    print('PASS: eight exact independent controls and a fresh deterministic five-episode prior',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    gate(parser.parse_args().root)
