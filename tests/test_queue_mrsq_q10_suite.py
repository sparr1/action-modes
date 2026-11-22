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


def test_storage_reservation_counts_staging_and_existing_lineages(monkeypatch, tmp_path):
    campaign, durable = tmp_path / 'campaign', tmp_path / 'durable'
    monkeypatch.setattr(suite, 'CAMPAIGN', campaign)
    monkeypatch.setattr(suite, 'DURABLE', durable)
    plan = {'full_generation_bytes': 2**30}
    prior = campaign / 'dog-trot' / 'release.json'
    prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps({'plan': plan}))
    generation = durable / 'dog-trot' / 'generations' / 'step-500' / 'replay.pt'
    generation.parent.mkdir(parents=True)
    with generation.open('wb') as file:
        file.truncate(2**30)
    assert suite.booked_home_gib('humanoid-run', plan) == 5
    (durable / 'dog-trot' / 'DONE').touch()
    assert suite.booked_home_gib('humanoid-run', plan) == 3
