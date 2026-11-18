"""Guard against inconsistent Slurm CPU fields and incorrect release budgets."""
from copy import deepcopy
import pytest
from slurm.release_mrsq_q5_suite import allocation_updates, verify_allocation, wall_minutes, reservation_check


PLAN = {'cpus': 1, 'memory_gib': 10, 'wall_hours': 58}
FIELDS = {'NumCPUs': '1', 'CPUs/Task': '1', 'MinCPUsNode': '1',
          'TresPerTask': 'cpu=1', 'MinMemoryNode': '10G', 'TimeLimit': '2-10:00:00'}


def test_verified_allocation_accepts_slurm_day_format():
    verify_allocation(FIELDS, PLAN)
    assert wall_minutes('35:00') == 35


def test_memory_update_uses_slurm_integer_mib_units():
    updates = allocation_updates(42, {'cpus': 2, 'memory_gib': 12, 'wall_hours': 60})
    assert 'MinMemoryNode=12288' in updates
    assert all(value in updates for value in ('NumCPUs=2', 'CPUsPerTask=2', 'TresPerTask=cpu:2', 'MinCPUsNode=2'))


@pytest.mark.parametrize('key,value', [('NumCPUs', '2'), ('CPUs/Task', '2'),
    ('MinCPUsNode', '2'), ('TresPerTask', 'cpu=2'), ('MinMemoryNode', '8G'), ('TimeLimit', '2-00:00:00')])
def test_inconsistent_allocation_is_rejected(key, value):
    actual = deepcopy(FIELDS)
    actual[key] = value
    with pytest.raises(AssertionError):
        verify_allocation(actual, PLAN)


def test_reservations_replace_own_request_and_count_expanded_arrays(monkeypatch):
    from slurm import release_mrsq_q5_suite as module
    def fake(args, **kwargs):
        if args[0] == 'sacctmgr':
            return 'pri-gpu+|cpu=414720,gres/gpu=69120|'
        assert '-r' in args
        return '\n'.join(['42|pri-gpu+|2|4-00:00:00|gres/gpu:l40s:1',
            '100_1|pri-gpu+|1|35:00|gres/gpu:1', '100_2|pri-gpu+|1|35:00|gres/gpu:1',
            '101|gpu-debug|1|20:00|gres/gpu:l40s:1'])
    monkeypatch.setattr(module, 'command', fake)
    result = reservation_check(42, PLAN)['full_reservation_upper_bound_minutes']
    assert result == {'cpu': 58 * 60 + 70, 'gres/gpu': 58 * 60 + 70}
