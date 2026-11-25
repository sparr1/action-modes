import hashlib
import json

import pytest

from slurm import queue_mrsq_q10_suite as suite
from slurm import release_mrsq_q5_suite as resources
from slurm.mrsq_q10_workspace import CAMPAIGN, ENTITY, NAME, PROJECT, WorkspaceError, discover_rows, install, markdown


def test_hold_validation_checks_all_cpu_fields_and_node_exclusion():
    plan = {'cpus': 2, 'wall_hours': 102, 'memory_gib': 14}
    fields = {'JobState': 'PENDING', 'Reason': 'JobHeldUser', 'Partition': 'gpu',
              'QOS': 'pri-gpu+', 'Dependency': 'afterok:123', 'ExcNodeList': 'gpu2708',
              'TresPerNode': 'gres/gpu:l40s:1', 'NumCPUs': '2', 'CPUs/Task': '2',
              'MinCPUsNode': '2', 'TresPerTask': 'cpu:2', 'MinMemoryNode': '14336M',
              'TimeLimit': '4-06:00:00'}
    suite.verify_held(fields, plan, '123')
    for key in ('MinCPUsNode', 'TresPerTask', 'ExcNodeList'):
        with pytest.raises(AssertionError):
            suite.verify_held({**fields, key: '1'}, plan, '123')


def test_scratch_reservation_counts_staging_and_portable_checkpoints(monkeypatch, tmp_path):
    campaign = tmp_path / 'campaign'
    monkeypatch.setattr(suite, 'CAMPAIGN', campaign)
    plan = {'full_generation_bytes': 2**30, 'checkpoint_bytes': 2**25}
    prior = campaign / 'dog-trot' / 'release.json'
    prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps({'plan': plan}))
    lineage = suite.lineage_path('dog-trot', campaign)
    assert lineage == campaign / 'dog-trot' / 'lineage'
    generation = lineage / 'generations' / 'step-500' / 'replay.pt'
    generation.parent.mkdir(parents=True)
    with generation.open('wb') as file:
        file.truncate(2**30)
    weight = lineage.parent / 'models' / 'run' / 'models' / 'model:dog_trot_25000'
    weight.parent.mkdir(parents=True)
    with weight.open('wb') as file:
        file.truncate(2**25)
    weight.with_suffix('.metadata.json').write_bytes(b'{}')
    assert suite.booked_scratch_gib('humanoid-run', plan) == 7.5 - 1/32
    (lineage / 'DONE').touch()
    assert suite.booked_scratch_gib('humanoid-run', plan) == 4.25


def test_quota_uses_scratch_row(monkeypatch):
    output = 'rgao48 /oscar/home 95.57 95.57 100 125 0 0 0 OK None\n' \
             'rgao48 /oscar/scratch 407.68 79.63 512 10240 0 0 0 OK None\n'
    monkeypatch.setattr(suite, 'command', lambda *args: output)
    assert suite.scratch_free_gib() == pytest.approx(104.32)


def test_reservation_bound_uses_live_concurrency_and_excludes_replaced_job(monkeypatch):
    qos = 'pri-gpu+|cpu=800,gres/gpu=240|cpu=4,gres/gpu=2,mem=64G|'
    def command(args):
        if args[0] == 'sacctmgr':
            return qos
        # Job 10 is running. Job 11 is pending and retains its full request.
        active = '01:15:00' if '%L' in args[-1] else '04:00:00'
        return (f'10|pri-gpu+|2|{active}|gres/gpu:l40s:1\n'
                '11|pri-gpu+|1|01:50:00|gres/gpu:l40s:1\n'
                '12|pri-gpu+|1|02:15:00|gres/gpu:l40s:1')

    monkeypatch.setattr(resources, 'command', command)
    plan = {'cpus': 1, 'wall_hours': 2}
    result = resources.reservation_check('12', plan, concurrent=True)
    assert result['remaining_reservation_upper_bound_minutes'] == {'cpu': 380, 'gres/gpu': 230}
    assert result['allocation_limits'] == {'cpu': 4, 'gres/gpu': 2}
    with pytest.raises(AssertionError):
        resources.reservation_check('12', plan)


@pytest.mark.parametrize('task', suite.SEED56_TASKS)
def test_seed56_reuses_only_matching_seed55_recipe_and_measurements(monkeypatch, tmp_path, task):
    monkeypatch.setattr(suite, 'CAMPAIGN', tmp_path)
    old_root = suite.task_path(task, tmp_path)
    new_root = suite.task_path(task, tmp_path, seed=56)
    gate = old_root / 'benchmark'
    gate.mkdir(parents=True)
    new_root.mkdir()
    parity = b'<testsuite><testcase/><testcase/></testsuite>'
    (gate / 'parity.xml').write_bytes(parity)
    reference = {'passed': True, 'task': task, 'cpus': 1, 'cpu_replay': True, 'wandb': False,
                 'binding': {**suite.binding(suite.manifest_path(task)), 'commit': 'old'},
                 'runtime': {'gpu': 'NVIDIA L40S'}, 'gate_job': '10',
                 'parity_sha256': hashlib.sha256(parity).hexdigest()}
    (gate / 'plan.json').write_text(json.dumps(reference))
    (old_root / 'release.json').write_text(json.dumps({'allocation_verified': True, 'plan': reference,
                                                       'gate_job': '10', 'training_job': '11'}))
    (old_root / 'scheduling.json').write_text(json.dumps({'gate': '10', 'commit': 'old', 'tree': 'old-tree'}))

    def git(*args):
        if args == ('git', 'rev-parse', 'old^{tree}'):
            return 'old-tree'
        assert args[:5] == ('git', 'diff', '--quiet', 'old', 'HEAD')

    monkeypatch.setattr(suite, 'command', git)
    assert suite.lineage_path(task, tmp_path, seed=56).parent == new_root != old_root
    plan = suite.reuse_timing_plan(task, new_root)
    assert plan['binding'] == suite.binding(suite.manifest_path(task, 56))
    assert plan['sizing_reference'] == {'seed': 55, 'binding': reference['binding'], 'training_job': '11'}
    assert suite.reuse_timing_plan(task, new_root) == plan
    assert json.loads((gate / 'plan.json').read_text()) == reference


def test_workspace_matches_old_seedless_receipts_and_keeps_seed56_independent():
    class API:
        def runs(self, *args, **kwargs):
            return []

    rows = discover_rows(API(), [{'task': 'dog-run', 'run_id': 'old'},
                                {'task': 'humanoid-walk', 'run_id': 'old-walk'}])
    assert len(rows) == len({row['group'] for row in rows}) == 13
    dog = {row['seed']: row for row in rows if row['task'] == 'dog-run'}
    assert dog[55]['run_id'] == 'old'
    assert dog[56]['run_id'] is None
    assert '| dog-run | 56 | pending |' in markdown(rows)
    assert '| humanoid-stand | 56 | pending |' in markdown(rows)
    walk = {row['seed']: row for row in rows if row['task'] == 'humanoid-walk'}
    assert walk[55]['run_id'] == 'old-walk'
    assert walk[56]['run_id'] is None
    assert '| humanoid-walk | 56 | pending |' in markdown(rows)
    hydra = {row['task']: row for row in rows if row['cluster'] == 'Hydra'}
    assert set(hydra) == {'dog-stand', 'dog-trot', 'dog-walk'}
    assert all(
        row['seed'] == 56 and row['group'] == f'mrsq-q10-dogs-s56-hydra-20261124-{task}'
        for task, row in hydra.items()
    )
    assert 'Oscar + Hydra' in markdown(rows)
    assert '| dog-stand | 56 | pending | pending | Hydra |' in markdown(rows)


def test_workspace_keeps_existing_receipt_when_explicit_view_differs(tmp_path):
    receipt = tmp_path / 'workspace.json'
    url = f'https://wandb.ai/{ENTITY}/{PROJECT}?nw=wcqgbdphdpb'
    previous = dict(entity=ENTITY, project=PROJECT, campaign=CAMPAIGN, name=NAME, url=url)
    receipt.write_text(json.dumps(previous))
    with pytest.raises(WorkspaceError, match='different view than requested'):
        install(receipt, existing_url=f'https://wandb.ai/{ENTITY}/{PROJECT}?nw=other')
    assert json.loads(receipt.read_text()) == previous
