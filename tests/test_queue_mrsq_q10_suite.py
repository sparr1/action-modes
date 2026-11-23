import json

import pytest

from slurm import queue_mrsq_q10_suite as suite


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
