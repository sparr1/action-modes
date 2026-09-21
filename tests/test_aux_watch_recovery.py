"""A scheduler timeout must not fail publication or duplicate completed work."""
from types import SimpleNamespace
import fcntl
import subprocess
import sys

import pytest
from slurm import ambi_aux_hj_sweep as campaign


@pytest.mark.parametrize('failure', [subprocess.CalledProcessError(1, ['squeue']),
                                    subprocess.TimeoutExpired(['squeue'], 20), OSError('unavailable')])
def test_scheduler_failure_is_unknown_not_empty(monkeypatch, failure):
    def run(*args, **kwargs):
        assert kwargs['timeout'] == 20 and kwargs['check']
        raise failure
    monkeypatch.setattr(campaign.subprocess, 'run', run)
    assert campaign.active_gpu_jobs(['123']) is None


@pytest.mark.parametrize('output,expected', [('123_0\n', '123_0'), ('', '')])
def test_successful_scheduler_query_distinguishes_running_from_empty(monkeypatch, output, expected):
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=output))
    assert campaign.active_gpu_jobs(['123']) == expected


def test_resume_keeps_run_and_completed_publications_during_scheduler_failures(tmp_path, monkeypatch):
    cells = []
    for name in ('reused', 'new'):
        directory = tmp_path/name; directory.mkdir()
        cells.append(dict(name=name, H=3, J=8, params={}, directory=str(directory),
                          reused=name=='reused', training_run_id=name+'-training', performance_run_id=name+'-performance'))
    campaign.write(tmp_path/'campaign.json', dict(cells=cells, group='test', overview_run_id='same-run', source_commit='old-source'))
    campaign.write(tmp_path/'submission.json', dict(gpu_job_ids=['123']))
    campaign.write(tmp_path/'watcher-started.json', dict(pid=111, started=1))
    campaign.write(tmp_path/'reused'/'publication-completion.json', dict(metrics={'eval/return_mean': 3.}))
    original_marker = (tmp_path/'watcher-started.json').read_bytes()
    original_publication = (tmp_path/'reused'/'publication-completion.json').read_bytes()
    calls = []; finished = []; summary = {}
    run = SimpleNamespace(log=lambda row: calls.append(row), summary=summary, finish=lambda **k: finished.append(k))
    def init(**kwargs):
        assert kwargs['id'] == 'same-run' and kwargs['resume'] == 'must'
        return run
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init, Table=lambda **k:k))
    active = iter(['', None, '', None, '123'])
    monkeypatch.setattr(campaign, 'active_gpu_jobs', lambda ids: next(active))
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 5:
            campaign.write(tmp_path/'new'/'publication-completion.json', dict(metrics={'eval/return_mean': 4.}))
    monkeypatch.setattr(campaign, 'time', SimpleNamespace(time=lambda:1000+100*len(sleeps), time_ns=lambda:12345, sleep=sleep))
    campaign.watch(SimpleNamespace(root=tmp_path, resume_watch=True))
    assert len(sleeps) == 5 and finished == [{'exit_code': 0}]
    assert summary['status'] == 'complete'
    assert (tmp_path/'watcher-started.json').read_bytes() == original_marker
    assert (tmp_path/'reused'/'publication-completion.json').read_bytes() == original_publication
    assert (tmp_path/'watcher-resumed-12345.json').exists()
    assert campaign.read(tmp_path/'campaign-completion.json')['status'] == 'complete'
    assert [row['campaign/published'] for row in calls] == [1, 2]


def test_second_watcher_cannot_acquire_publication_lock(tmp_path):
    with (tmp_path/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='publication lock'):
            campaign.watch(SimpleNamespace(root=tmp_path, resume_watch=True))
