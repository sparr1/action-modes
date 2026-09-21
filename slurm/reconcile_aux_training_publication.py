"""Finish an interrupted training publication after auditing its uploaded prefix.

Run on an allocated CPU node, after the previous publisher has stopped. This
does not generate episodes, replace artifacts, or repeat published history.
"""
import argparse
import base64
import fcntl
import hashlib
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def missing_history(expected, actual):
    """Accept only an exact, contiguous committed prefix of intended rows."""
    rows = {}
    for row in actual:
        step = row['_step']
        assert isinstance(step, int) and 0 <= step < len(expected)
        assert step not in rows, 'Duplicate remote history step; inspect before repair.'
        for key, value in expected[step].items():
            observed = row.get(key)
            assert isinstance(observed, (int, float)) and math.isclose(
                observed, value, rel_tol=1e-6, abs_tol=1e-8), (step, key)
        rows[step] = row
    assert sorted(rows) == list(range(len(rows))), 'Non-prefix history; inspect before repair.'
    return expected[len(rows):]


def recover(root, index):
    import wandb
    from slurm.ambi_aux_hj_sweep import read, write, digest, log_training_curves, publish_performance, PROJECT, ENTITY
    from utils.ambi_diagnostic_series import diagnostic_history
    from utils.eval_series_data import load_records
    campaign = read(root/'campaign.json'); cell = campaign['cells'][index]
    directory = Path(cell['directory']); bundle = Path(cell['bundle'])
    assert not (directory/'publication-completion.json').exists()
    journal = read(directory/'training-publication.json')
    assert journal == dict(status='uncertain', run_id=cell['training_run_id'])
    receipt = read(directory/'worker-completion.json')
    assert digest(bundle/'manifest.json') == receipt['manifest_sha256']
    assert all(digest(bundle/n) == sha for n, sha in receipt['trace_sha256'].items())
    record, = load_records(bundle, inventory_path=campaign['inventory'])
    assert record['metrics']['eval/frozen_state_unchanged'] and record['metrics']['eval/paired_episodes'] == 5
    summary = read(directory/'training-summary.json')
    diagnostic = read(directory/'model-series/manifest.json')
    comparison = read(directory/'ere-comparison.json')
    expected = []
    class Collector:
        def define_metric(self, *args, **kwargs): pass
        def log(self, row): expected.append(row)
    log_training_curves(Collector(), summary)
    expected.extend(diagnostic_history(diagnostic))
    expected.append(comparison['metrics'])
    api = wandb.Api(timeout=60)
    remote = api.run(f'{ENTITY}/{PROJECT}/{cell["training_run_id"]}')
    assert remote.state in ('crashed', 'failed', 'finished'), remote.state
    artifacts = [a for a in remote.logged_artifacts() if a.type == 'inner-training-traces']
    artifact, = artifacts
    assert artifact.state == 'COMMITTED'
    assert artifact.metadata['manifest_sha256'] == receipt['manifest_sha256']
    files = artifact.manifest.entries
    wanted = {'bundle/manifest.json', 'training-summary.json', 'ere-comparison.json',
              'model-series/manifest.json', 'model-series/paired-rows.jsonl.gz', 'model-series/report.html'}
    wanted.update('bundle/'+n for n in receipt['trace_sha256'])
    assert set(files) == wanted
    for name, entry in files.items():
        local = directory/name
        h = hashlib.md5()
        with local.open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024), b''): h.update(chunk)
        assert base64.b64encode(h.digest()).decode() == entry.digest, name
    missing = missing_history(expected, remote.scan_history(page_size=50))
    start = len(expected)-len(missing)
    write(directory/'training-recovery-audit.json', dict(
        run_id=cell['training_run_id'], artifact=artifact.name, expected_rows=len(expected),
        committed_prefix_rows=start, append_rows=len(missing), manifest_sha256=receipt['manifest_sha256']))
    performance = publish_performance(cell['run_dir'])
    run = wandb.init(entity=ENTITY, project=PROJECT, id=cell['training_run_id'], resume='must', mode='online')
    try:
        for step, row in enumerate(missing, start): run.log(row, step=step)
        metrics = {**record['metrics'], **comparison['metrics']}
        run.summary.update({**metrics, 'status':'complete', 'reused':False,
            'diagnostic/paired_rows':len(diagnostic['expected']),
            'training/decisions':record['metrics']['work/environment_decisions'],
            'training/critic_updates':record['metrics']['work/critic_updates'],
            'training/actor_updates':record['metrics']['work/actor_updates'],
            'performance_url':f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{cell["performance_run_id"]}'})
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    write(directory/'training-publication-before-recovery.json', journal)
    write(directory/'training-publication.json', dict(status='complete',run_id=cell['training_run_id']))
    write(directory/'publication-completion.json', dict(status='complete',cell=cell['name'],reused=False,
        performance=performance,training_run_id=cell['training_run_id'],metrics=metrics))
    print(f'Recovered {cell["training_run_id"]}: retained {start} rows, appended {len(missing)}.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Use an allocated CPU job.'
    with (args.root/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recover(args.root, args.index)
