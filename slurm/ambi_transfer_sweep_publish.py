"""Resume-safe CPU publication for an existing 575K transfer sweep.

Evaluation checkouts, campaign records, and sealed GPU bundles are read-only.
Publisher ownership, run IDs, journals, and layout receipts live separately.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_sweep_campaign import digest, read, receipt, validate_completed, SEEDS
from slurm.ambi_closed_loop_publish import gpu_jobs_active
from utils.ambi_benchmark import atomic_json

ENTITY = 'rwgao_b-brown-university'
PROJECT = 'ambi'
MODES = ('fresh', 'actor_only', 'critic_only')
HIDDEN_MODES = (*MODES, 'critic_hidden')
HISTORICAL_COMMIT = '7694cfca76875735b421de13ee6a5d829518d4ff'
TABLE_COLUMNS = {
    'settings': ['index', 'setting', 'critic', 'transfer', 'J', 'solve_interval', 'state',
                 'completed_episodes', 'expected_episodes', 'performance_url', 'error'],
    'results': ['setting', 'critic', 'transfer', 'J', 'solve_interval', 'state', 'return_mean',
                'return_std', 'controller_seconds_per_decision', 'paired_vs_fresh_mean',
                'paired_vs_prior_mean', 'performance_url'],
    'paired_effects': ['setting', 'reference', 'comparison', 'paired_episodes', 'gain_mean', 'gain_std'],
    'episodes': ['setting', 'seed', 'solver_seed', 'return', 'length', 'control_seconds'],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def url(run_id):
    return f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}'


def _write(path, value):
    atomic_json(Path(path), value, overwrite=True)


@contextmanager
def publisher_lock(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.publisher.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another publisher owns this publication directory.') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def publisher_commit():
    require(not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip(),
            'Launch the publisher from its own clean checkout.')
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def publication_state(campaign_root, publication_root, campaign, commit, *, reference_binding=None):
    path = Path(publication_root) / 'publication.json'
    binding = dict(campaign_root=str(Path(campaign_root).resolve()),
        campaign_sha256=digest(Path(campaign_root) / 'campaign.json'),
        evaluation_commit=campaign['source_commit'], publisher_commit=commit)
    if reference_binding is not None:
        binding['comparison_reference'] = reference_binding
    if path.exists():
        state = read(path)
        require(all(state.get(k) == v for k, v in binding.items()),
                'Publication ownership differs from campaign or pinned publisher commit.')
    else:
        state = dict(schema_version=1, **binding, overview_run_id=uuid.uuid4().hex, cells={})
        _write(path, state)
    return state


def load_completed(campaign_root, campaign, cell):
    from utils.eval_series_data import load_records
    root = Path(campaign_root)
    receipt(root, campaign, cell['index'], verify=True)
    bundle = root / 'settings' / cell['name'] / 'bundle'
    manifest = validate_completed(bundle, cell, campaign)
    record, = load_records(bundle, inventory_path=campaign['inventory'])
    require(record['identity']['science'] == campaign['science'], 'Result scientific implementation differs.')
    require(record['identity']['planner'] == cell['planner_identity'], 'Result planner identity differs.')
    require(record['identity']['backbone'] == campaign['source_run'], 'Result backbone differs.')
    require(not record['provenance']['missing_artifact_files'], 'Completed result has missing trace artifacts.')
    return dict(record=record, episodes=manifest['runs'][0]['episodes'])


def publish_cell(publication_root, state, cell, completed):
    """Use existing immutable publication journals without writing GPU bundles."""
    from utils.eval_series import create_run, load_run, stage_record
    root = Path(publication_root)
    key = cell['name']; record = completed['record']
    registry_root = root / 'registry' / key
    if key not in state['cells']:
        # Recover a registry allocated before a crash but not yet entered in the map.
        candidates = list(registry_root.glob('*/run.json'))
        require(len(candidates) <= 1, 'Ambiguous publisher registry recovery.')
        registry = (load_run(candidates[0].parent) if candidates else
                    create_run(registry_root, record, 'transfer-sweep-575k-' + key,
                               PROJECT, ENTITY, 'oscar-rgao48'))
        require(registry['identity'] == record['identity'], 'Registry recovery identity differs.')
        state['cells'][key] = dict(run_dir=registry['run_dir'], run_id=registry['run_id'],
                                   status='allocated', record_id=record['record_id'])
        _write(root / 'publication.json', state)
    entry = state['cells'][key]
    require(entry['record_id'] == record['record_id'], 'Completed result changed after publication allocation.')
    registry = load_run(entry['run_dir'])
    require(registry['identity'] == record['identity'], 'Performance registry identity differs.')
    stage_record(entry['run_dir'], record)
    # W&B's registry publisher uses reinit=True. Keep it in a separate process
    # so publishing a performance run cannot finish this process's live overview.
    logs = root / 'performance-logs'; logs.mkdir(exist_ok=True)
    with (logs / (key + '.log')).open('a') as output:
        subprocess.run([sys.executable, str(ROOT / 'eval_series.py'), 'publish',
                        entry['run_dir'], '--owner', 'oscar-rgao48'],
                       stdout=output, stderr=subprocess.STDOUT, check=True, close_fds=False)
    journal = read(Path(entry['run_dir']) / 'publication.json')
    require(journal['records'].get(record['record_id'], {}).get('status') == 'published',
            'Child publisher did not acknowledge the staged performance record.')
    published = dict(run_id=entry['run_id'], accepted=len(journal['records']),
                     published=sum(item['status'] == 'published' for item in journal['records'].values()))
    entry.update(status='published', publication=published)
    _write(root / 'publication.json', state)
    return entry


def progress_rows(campaign_root, campaign, state, completed, failures, *, jobs_active=True):
    rows = []
    for cell in campaign['cells']:
        directory = Path(campaign_root) / 'settings' / cell['name']
        entry = state['cells'].get(cell['name'], {})
        error = failures.get(cell['name'])
        if cell['name'] in completed:
            count = len(completed[cell['name']]['episodes'])
            status = 'published' if entry.get('status') == 'published' else 'publication_failed' if error else 'evaluated'
        elif (directory / 'worker-failure.json').exists():
            error = read(directory / 'worker-failure.json').get('error', 'Evaluation failed.')
            count, status = 0, 'evaluation_failed'
        else:
            manifest_path = directory / 'bundle/manifest.json'
            manifest = read(manifest_path) if manifest_path.is_file() else {}
            runs = manifest.get('runs', [])
            count = sum(ep.get('length') == 500 for ep in (runs[0].get('episodes', []) if runs else []))
            status = ('running' if jobs_active else 'incomplete') if directory.exists() else ('pending' if jobs_active else 'not_started')
        rows.append(dict(index=cell['index'], setting=cell['name'], critic=cell.get('critic_kind', 'prior'),
            transfer=cell['transfer_mode'], J=cell['J'], solve_interval=cell['solve_interval'], state=status,
            completed_episodes=count, expected_episodes=len(SEEDS),
            performance_url=url(entry['run_id']) if entry else None, error=error))
    return rows


def paired_effect(candidate, reference, name, reference_name, comparison):
    key = lambda e: (e['seed'], e['solver_seed'])
    left, right = {key(e): e['return'] for e in candidate}, {key(e): e['return'] for e in reference}
    require(len(left) == len(right) == len(SEEDS) and set(left) == set(right), 'Pairing requires both complete five-seed panels.')
    require({seed for seed, _ in left} == set(SEEDS), 'Pairing seed panel differs.')
    differences = [left[k] - right[k] for k in sorted(left)]
    return dict(setting=name, reference=reference_name, comparison=comparison,
        paired_episodes=len(differences), gain_mean=statistics.mean(differences), gain_std=statistics.stdev(differences))


def aggregate(campaign, progress, completed):
    statuses = {row['setting']: row for row in progress}
    results, pairs, episodes = [], [], []
    for cell in campaign['cells']:
        name = cell['name']; status = statuses[name]
        row = dict(setting=name, critic=cell.get('critic_kind', 'prior'), transfer=cell['transfer_mode'],
            J=cell['J'], solve_interval=cell['solve_interval'], state=status['state'],
            return_mean=None, return_std=None, controller_seconds_per_decision=None,
            paired_vs_fresh_mean=None, paired_vs_prior_mean=None, performance_url=status['performance_url'])
        if name in completed:
            data = completed[name]['episodes']; returns = [ep['return'] for ep in data]
            require(len(returns) == len(SEEDS), 'Only completed five-episode panels may become result points.')
            row.update(return_mean=statistics.mean(returns), return_std=statistics.stdev(returns),
                controller_seconds_per_decision=sum(ep['control_seconds'] for ep in data)/sum(ep['length'] for ep in data))
            episodes.extend(dict(setting=name, **{k: ep[k] for k in TABLE_COLUMNS['episodes'][1:]}) for ep in data)
            if cell['transfer_mode'] != 'prior':
                fresh = f"{cell['critic_kind']}_fresh_h3_j{cell['J']}_i{cell['solve_interval']}"
                for reference, comparison in ((fresh, 'vs_fresh'), ('prior_reference', 'vs_prior')):
                    if reference in completed and reference != name:
                        effect = paired_effect(data, completed[reference]['episodes'], name, reference, comparison)
                        pairs.append(effect); row['paired_' + comparison + '_mean'] = effect['gain_mean']
            if cell['transfer_mode'] == 'critic_hidden':
                for mode, comparison in (('critic_only', 'vs_full_critic'), ('actor_only', 'vs_actor')):
                    reference = f"{cell['critic_kind']}_{mode}_h3_j{cell['J']}_i{cell['solve_interval']}"
                    if reference in completed:
                        effect = paired_effect(data, completed[reference]['episodes'], name, reference, comparison)
                        pairs.append(effect); row['paired_' + comparison + '_mean'] = effect['gain_mean']
        if 'origin' in status:
            row.update({key: status[key] for key in ('origin', 'evaluation_commit')})
        results.append(row)
    return dict(settings=progress, results=results, paired_effects=pairs, episodes=episodes,
        completed=len(completed), total=len(campaign['cells']),
        published=sum(row['state'] == 'published' for row in progress))


def overview_payload(wandb, snapshot, *, hidden_comparison=False):
    namespace = 'critic_hidden_sweep' if hidden_comparison else 'transfer_sweep'
    table_columns = {name: list(columns) for name, columns in TABLE_COLUMNS.items()}
    if hidden_comparison:
        for name in ('settings', 'results'):
            table_columns[name] += ['origin', 'evaluation_commit']
        table_columns['results'] += ['paired_vs_full_critic_mean', 'paired_vs_actor_mean']
    payload = {f'{namespace}/{name}': wandb.Table(columns=columns,
        data=[[row.get(k) for k in columns] for row in snapshot[name]])
        for name, columns in table_columns.items()}
    payload.update({'campaign/completed': snapshot['completed'], 'campaign/total': snapshot['total'],
                    'campaign/published': snapshot['published']})
    for critic in ('soft', 'return'):
        for interval in (1, 3):
            groups = [[row for row in snapshot['results'] if row['critic'] == critic and
                       row['solve_interval'] == interval and row['transfer'] == mode and row['return_mean'] is not None]
                      for mode in (HIDDEN_MODES if hidden_comparison else MODES)]
            for axis, field in (('j', 'J'), ('compute', 'controller_seconds_per_decision')):
                series = [sorted(group, key=lambda r:r[field]) for group in groups]
                # Register the chart even while empty; explicit panels show pending context.
                payload[f'{namespace}/{critic}_i{interval}_return_vs_{axis}'] = wandb.plot.line_series(
                    xs=[[row[field] for row in group] for group in series],
                    ys=[[row['return_mean'] for row in group] for group in series],
                    keys=(['Fresh (historical)', 'Actor-only (historical)', 'Full critic (historical)',
                           'Hidden layers / random head (new)'] if hidden_comparison else
                          ['Fresh', 'Actor-only transfer', 'Critic-only transfer']),
                    title=f'{critic} critic, solve every {interval} decision(s): return versus {axis}',
                    xname='J rounds per solve' if axis == 'j' else 'Controller seconds per real decision')
    return payload


def install_layout(wandb, run, publication_root, *, hidden_comparison=False):
    from utils.wandb_transfer_sweep_layout import DEFAULT_VIEW_NAME, ensure_transfer_sweep_results_layout
    receipt = ensure_transfer_sweep_results_layout(wandb.Api(timeout=30), entity=ENTITY, project=PROJECT,
        receipt_dir=Path(publication_root)/'results-layout',
        view_name=os.environ.get('WANDB_RESULTS_VIEW_NAME', DEFAULT_VIEW_NAME), run_id=run.id,
        **({'hidden_comparison': True} if hidden_comparison else {}))
    run.summary.update({'results_layout/status': receipt['status'], 'results_layout/url': receipt['url'],
                        'results_layout/workspace_url': receipt['workspace_url'], 'results_layout/schema_verified': True})
    return receipt


def load_references(args, campaign):
    """Read historical bundles under their own identities; never publish/rekey them."""
    from utils.eval_series import load_run
    root = Path(args.reference_root).resolve()
    publication = Path(args.reference_publication_root).resolve()
    audit_path = Path(args.reference_audit).resolve()
    reference = read(root / 'campaign.json')
    audit = read(audit_path)
    require(audit.get('scope') == 'comparison_only_no_identity_reuse', 'Reference audit scope differs.')
    require(reference['source_commit'] == audit.get('reference_commit') == HISTORICAL_COMMIT,
            'Reference must be the explicitly audited historical evaluation revision.')
    require(digest(root / 'campaign.json') == audit.get('reference_campaign_sha256'),
            'Reference audit campaign fingerprint differs.')
    require(audit.get('cpu_default_path_proof', {}).get('status') == 'passed',
            'Reference audit default-path proof has not passed.')
    require(audit.get('new_source_commit') == campaign['source_commit'],
            'Reference audit candidate commit differs from the new campaign.')
    candidate_hashes = audit.get('candidate_source_sha256', {})
    require(isinstance(candidate_hashes, dict) and 'RL/tdmpc2_core/inner_improvement.py' in candidate_hashes,
            'Reference audit lacks candidate scientific source fingerprints.')
    for path, expected in candidate_hashes.items():
        require(not Path(path).is_absolute() and '..' not in Path(path).parts,
                'Reference audit source path escapes repository.')
        source = subprocess.check_output(['git', '-C', str(ROOT), 'show', f"{campaign['source_commit']}:{path}"])
        require(hashlib.sha256(source).hexdigest() == expected, f'Audited candidate source changed: {path}.')
    require(len(reference['cells']) == 25 and reference['cells'][24]['transfer_mode'] == 'prior',
            'Expected the complete original 25-cell campaign.')
    for key in ('source_run', 'checkpoint_sha256', 'metadata_sha256', 'checkpoint_step',
                'seeds', 'controller_seed', 'max_steps'):
        require(reference[key] == campaign[key], f'Reference comparison {key} differs.')
    published = read(publication / 'publication.json')
    require(published['campaign_root'] == str(root) and
            published['campaign_sha256'] == digest(root / 'campaign.json') and
            published['evaluation_commit'] == reference['source_commit'], 'Reference publisher binding differs.')
    complete = {}
    for cell in reference['cells']:
        name = cell['name']
        value = load_completed(root, reference, cell)
        entry = published['cells'].get(name, {})
        require(entry.get('status') == 'published' and entry.get('record_id') == value['record']['record_id'],
                f'Reference {name} is not published under its original record identity.')
        registry_path = Path(entry['run_dir']).resolve()
        require(registry_path.is_relative_to(publication), 'Reference registry escapes publication root.')
        registry = load_run(registry_path)
        require(registry['identity'] == value['record']['identity'] and registry['run_id'] == entry['run_id'],
                'Reference immutable registry identity differs.')
        journal = read(registry_path / 'publication.json')
        require(journal['records'].get(entry['record_id'], {}).get('status') == 'published',
                'Reference registry lacks publication acknowledgement.')
        complete[name] = value
    binding = dict(campaign_root=str(root), campaign_sha256=digest(root / 'campaign.json'),
        publication_root=str(publication), publication_sha256=digest(publication / 'publication.json'),
        evaluation_commit=reference['source_commit'], science=reference['science'],
        audit_path=str(audit_path), audit_sha256=digest(audit_path), audit=audit,
        policy='comparison_only_no_identity_reuse', overview_url=url(published['overview_run_id']))
    return dict(campaign=reference, completed=complete, state=published, binding=binding)


def comparison_snapshot(campaign_root, campaign, state, completed, failures, *, jobs_active=True, references=None):
    progress = progress_rows(campaign_root, campaign, state, completed, failures, jobs_active=jobs_active)
    if references is None:
        return aggregate(campaign, progress, completed)
    for row in progress:
        row.update(origin='new_evaluation', evaluation_commit=campaign['source_commit'])
    historical = progress_rows(references['binding']['campaign_root'], references['campaign'],
        references['state'], references['completed'], {}, jobs_active=False)
    for index, row in enumerate(historical, start=len(progress)):
        row['index'] = index
        row.update(origin='historical_reference', evaluation_commit=references['campaign']['source_commit'])
    all_cells = campaign['cells'] + references['campaign']['cells']
    require(len({cell['name'] for cell in all_cells}) == len(all_cells), 'Reference and candidate names overlap.')
    snapshot = aggregate({'cells': all_cells}, progress + historical, {**completed, **references['completed']})
    snapshot.update(new_completed=len(completed), new_total=len(campaign['cells']),
        new_published=sum(row['state'] == 'published' for row in progress),
        historical_completed=len(references['completed']), reference_provenance=references['binding'])
    return snapshot


def watch(args):
    import wandb
    campaign_root, output = Path(args.root).resolve(), Path(args.publication_root).resolve()
    require(not output.is_relative_to(campaign_root), 'Publication directory must be separate from the immutable campaign.')
    campaign = read(campaign_root/'campaign.json')
    hidden = campaign.get('campaign_mode') == 'critic-hidden'
    if hidden:
        require(len(campaign['cells']) == 8 and all(c['transfer_mode'] == 'critic_hidden' for c in campaign['cells']),
                'Expected eight hidden-transfer settings.')
        require(all(getattr(args, key, None) for key in ('reference_root', 'reference_publication_root', 'reference_audit')),
                'Hidden comparison requires explicit reference campaign, publication and audit.')
        require(not output.is_relative_to(Path(args.reference_root).resolve()) and
                not output.is_relative_to(Path(args.reference_publication_root).resolve()),
                'New publisher state must be separate from historical references.')
        references = load_references(args, campaign)
    else:
        require(not any(getattr(args, key, None) for key in ('reference_root', 'reference_publication_root', 'reference_audit')),
                'Historical references are supported only for the explicit hidden comparison.')
        require(len(campaign['cells']) == 25 and campaign['cells'][24]['transfer_mode'] == 'prior', 'Expected 24 settings plus prior.')
        references = None
    total = len(campaign['cells']) + (len(references['campaign']['cells']) if references else 0)
    new_total = len(campaign['cells'])
    with publisher_lock(output):
        state = publication_state(campaign_root, output, campaign, publisher_commit(),
            **({'reference_binding': references['binding']} if references else {}))
        run = wandb.init(entity=ENTITY, project=PROJECT, id=state['overview_run_id'], resume='allow',
            name=('575K hidden critic transfer | random head vs full critic and fresh | H3 J1/J8' if hidden else
                  '575K warm starts | fresh vs actor-only vs critic-only | H3 J1/J8'),
            group='transfer-sweep-575k-' + state['overview_run_id'][:8], job_type='transfer-sweep-overview',
            tags=['575k', 'warm-start', 'critic-transfer', 'soft-and-return'] +
                 (['hidden-critic-transfer', 'historical-comparison'] if hidden else []), mode='online',
            config=dict(evaluation_commit=campaign['source_commit'], publisher_commit=state['publisher_commit'],
                checkpoint_sha256=campaign['checkpoint_sha256'], checkpoint_step=campaign['checkpoint_step'],
                source_run=campaign['source_run'], campaign_sha256=state['campaign_sha256'],
                H=3, J=[1, 8], solve_intervals=[1, 3], modes=list(HIDDEN_MODES if hidden else MODES), critic_kinds=['soft', 'return'],
                seeds=SEEDS, controller_seed=55, max_steps=500, C=16, A=4, N=128, B=256,
                **({'comparison_reference': references['binding'], 'new_settings': new_total,
                    'historical_settings': len(references['campaign']['cells'])} if references else {})))
        completed, failures, attempts = {}, {}, {}
        previous, terminal_since, layout = None, None, None
        try:
            # Historical evidence is already verified; show every row before new result uploads.
            initial = comparison_snapshot(campaign_root, campaign, state, {}, {}, references=references)
            run.log(overview_payload(wandb, initial, hidden_comparison=hidden))
            run.summary.update({'status': 'running', 'completed_settings': initial['completed'], 'total_settings': total})
            layout = install_layout(wandb, run, output, **({'hidden_comparison': True} if hidden else {}))
            print('LIVE OVERVIEW ' + layout['url'], flush=True)
            while True:
                active = gpu_jobs_active(args.gpu_job_id)
                for cell in campaign['cells']:
                    name = cell['name']; directory = campaign_root/'settings'/name
                    if name not in completed and (directory/'worker-completion.json').is_file():
                        completed[name] = load_completed(campaign_root, campaign, cell)
                snapshot = comparison_snapshot(campaign_root, campaign, state, completed, failures,
                    jobs_active=active, references=references)
                stamp = json.dumps(snapshot, sort_keys=True)
                if stamp != previous:
                    _write(output/'snapshot.json', snapshot)
                    run.log(overview_payload(wandb, snapshot, hidden_comparison=hidden))
                    run.summary.update({'completed_settings': snapshot['completed'], 'published_settings': snapshot['published'],
                                        'total_settings': total, 'new_completed_settings': len(completed), 'status': 'running' if active else 'publishing'})
                    previous = stamp
                    print(f'Evaluated {snapshot["completed"]}/{total}; published {snapshot["published"]}/{total}', flush=True)
                # One per loop bounds CPU and publication overhead; journals make retries safe.
                pending = [cell for cell in campaign['cells'] if cell['name'] in completed and
                    state['cells'].get(cell['name'], {}).get('status') != 'published' and attempts.get(cell['name'], 0) < 3]
                if pending:
                    cell = pending[0]; name = cell['name']; attempts[name] = attempts.get(name, 0)+1
                    try:
                        publish_cell(output, state, cell, completed[name]); failures.pop(name, None)
                    except Exception as error:
                        failures[name] = f'{type(error).__name__}: {error}'
                        _write(output/'publication-errors.json', failures)
                        print('PUBLICATION RETRY ' + name + ': ' + failures[name], flush=True)
                    continue
                if len(completed) == new_total and snapshot['published'] == total:
                    break
                if args.once:
                    break
                if not active:
                    terminal_since = terminal_since or time.time()
                    if time.time()-terminal_since >= args.terminal_grace:
                        break
                else:
                    terminal_since = None
                time.sleep(args.poll_seconds)
            complete = len(completed) == new_total and all(state['cells'].get(c['name'], {}).get('status') == 'published' for c in campaign['cells'])
            status = 'complete' if complete else 'running' if args.once and active else 'incomplete'
            run.summary.update({'status': status, 'completed_settings': len(completed) +
                                (len(references['completed']) if references else 0), 'total_settings': total,
                                'new_completed_settings': len(completed)})
            _write(output/'publisher-status.json', dict(status=status, overview_url=layout['url'],
                overview_run_id=run.id, completed=len(completed), failures=failures, results_layout=layout))
            run.finish(exit_code=0 if complete or args.once else 1)
            return state
        except BaseException as error:
            _write(output/'publisher-failure.json', dict(error_type=type(error).__name__, error=str(error),
                overview_url=url(state['overview_run_id']), time=time.time()))
            try:
                run.summary.update({'publisher/status': 'failed', 'publisher/error': str(error)})
                run.finish(exit_code=1)
            finally:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--publication-root', type=Path, required=True)
    parser.add_argument('--gpu-job-id', action='append', required=True)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--reference-publication-root', type=Path)
    parser.add_argument('--reference-audit', type=Path)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--terminal-grace', type=float, default=120)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    require(args.poll_seconds > 0 and args.terminal_grace >= 0, 'Invalid polling interval.')
    watch(args)


if __name__ == '__main__':
    main()
