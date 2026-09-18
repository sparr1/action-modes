"""Strict historical pairing and independent controls for update interleaving."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PHASED_COMMIT = 'd02e78fe5fe8a29b6898dcbe8e93ac09a547613e'


def phased_baseline(candidate, baseline, *, baseline_science=None):
    from slurm.ambi_aux_hj_sweep import matched_baseline
    from utils.eval_series_data import scientific_identity
    if baseline_science is None:
        baseline_science = scientific_identity('AMBITDMPC2/AMBITDMPC2', None, PHASED_COMMIT)
    old = deepcopy(baseline)
    assert old['identity']['planner']['settings'].pop('inner_component_update_order', 'critic_first') == 'critic_first'
    result = matched_baseline(candidate, old, setting='inner_component_update_order',
                              before=False, after='interleaved', baseline_science=baseline_science)
    return dict(result, kind='interleaved',
                source_comparison=dict(baseline=baseline['identity']['science'],
                    candidate=candidate['identity']['science'], baseline_commit=PHASED_COMMIT,
                    change='Only component update order; default-path exact parity required by GPU gate.'))


def prepare_campaign(campaign, root, execution):
    from slurm.ambi_aux_hj_sweep import read, digest
    from slurm.ambi_aux_actor_budget import execution_identity
    assert execution == dict(mode='deterministic-v1', interleaved_comparison=True,
                             gpu_family='L40S', fresh_prior=False)
    old = read(Path(campaign['baseline_campaign'])/'campaign.json')
    assert old['source_commit'] == PHASED_COMMIT
    assert read(Path(campaign['baseline_campaign'])/'control-completion.json')['status'] == 'complete'
    assert Path(campaign['reference']) == Path(old['reference'])
    prior_receipt = read(Path(campaign['reference'])/'execution.json')
    assert all(c['baseline']['kind']=='interleaved' for c in campaign['cells'])
    for cell in campaign['cells']:
        assert execution_identity(cell['baseline']['execution']) == execution_identity(prior_receipt)
    assert digest(Path(campaign['reference'])/'manifest.json') == campaign['prior_manifest_sha256']
    campaign['execution'] = execution


def validate_phased_comparison(campaign, cell, record, receipt):
    from slurm.ambi_aux_hj_sweep import read, digest
    from slurm.ambi_aux_actor_budget import execution_identity
    from utils.eval_series_data import load_records
    baseline = cell['baseline']; bundle = Path(baseline['bundle'])
    assert digest(bundle/'manifest.json') == baseline['manifest_sha256']
    previous, = load_records(bundle, inventory_path=campaign['inventory'])
    paired = phased_baseline(record, previous)
    for key in ('record_id', 'episodes', 'source_comparison'):
        assert paired[key] == baseline[key]
    assert execution_identity(receipt['execution']) == execution_identity(baseline['execution'])
    assert digest(Path(campaign['reference'])/'manifest.json') == campaign['prior_manifest_sha256']
    controls = read(Path(cell['directory']).parent/'control-completion.json')
    assert controls['status']=='complete' and controls['source_commit']==campaign['source_commit']
    assert len(controls['checks']) == 16


def gate(root):
    from slurm.ambi_aux_hj_sweep import read, write, digest, actor_updates
    from slurm.ambi_aux_actor_budget import control_payload, execution_identity
    campaign = read(root/'campaign.json')
    old_root = Path(campaign['baseline_campaign'])
    checks = []
    for cell in campaign['cells']:
        if actor_updates(cell) not in (4,16):
            continue
        old_name = cell['name'].removesuffix('_interleaved')
        for kind, paths in (
            ('independent_interleaved', [root/'smoke'/rep/cell['name']/'bundle' for rep in ('a','b')]),
            ('default_path_parity', [old_root/'smoke/a'/old_name/'bundle', root/'smoke/phased'/old_name/'bundle']),
        ):
            payloads = [control_payload(p) for p in paths]
            if payloads[0] != payloads[1]:
                write(root/'control-failure.json', dict(cell=cell['name'],kind=kind,status='different'))
                raise AssertionError(f'{kind} differs: {cell["name"]}')
            receipts = [read(p/'execution.json') for p in paths]
            assert execution_identity(receipts[0]) == execution_identity(receipts[1])
            assert execution_identity(receipts[1]) == execution_identity(cell['baseline']['execution'])
            assert receipts[0]['environment']['TORCHINDUCTOR_CACHE_DIR'] != receipts[1]['environment']['TORCHINDUCTOR_CACHE_DIR']
            checks.append(dict(cell=cell['name'],kind=kind,exact=True,
                payload_sha256=hashlib.sha256(json.dumps(payloads[0],sort_keys=True).encode()).hexdigest(),
                manifests=[dict(bundle=str(p),sha256=digest(p/'manifest.json')) for p in paths]))
    assert len(checks)==16
    assert digest(Path(campaign['reference'])/'manifest.json') == campaign['prior_manifest_sha256']
    write(root/'control-completion.json', dict(status='complete',source_commit=campaign['source_commit'],
        checks=checks,prior_manifest_sha256=campaign['prior_manifest_sha256']))
    print('CONTROL GATE PASSED: 8 independent repeats and 8 historical default-path parity checks',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    gate(parser.parse_args().root)
