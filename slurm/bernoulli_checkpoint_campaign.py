"""Target selected checkpoint failures with existing Bernoulli critic transfer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import eval_series as series
from slurm.graph_checkpoint_campaign import git, now

MATRIX = ROOT / 'configs/research/ambi_checkpoint_bernoulli50.json'
SELECTOR = 'controller/bernoulli50'
OWNER = 'oscar-rgao48'


def load(path):
    campaign = series._read(path / 'campaign.json')
    if git('status', '--porcelain') or git('rev-parse', 'HEAD') != campaign['source_commit']:
        raise ValueError('Campaign requires its exact clean source commit')
    if series._file_digest(MATRIX) != campaign['matrix_sha256']:
        raise ValueError('Matrix changed')
    return campaign


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    if git('status', '--porcelain'):
        raise ValueError('Commit source before preparing')
    args.campaign.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (reference_root, step) in enumerate(args.reference):
        previous = series._read(Path(reference_root) / 'campaign.json')
        row = dict(next(r for r in previous['checkpoints'] if r['step'] == int(step)))
        reference = Path(reference_root) / 'production' / str(step) / 'bundle/manifest.json'
        for path, digest in (('path', 'sha256'), ('metadata_path', 'metadata_sha256')):
            if series._file_digest(row[path]) != row[digest]:
                raise ValueError('Checkpoint inventory hash mismatch')
        row.update(source_run=previous['source_run'],
                   label=('Outer frozen' if 'a5edf86' in previous['source_run'] else 'Outer adapt') + ' · critic Bernoulli 50%',
                   reference=str(reference), reference_sha256=series._file_digest(reference),
                   baseline_runs=previous['runs'])
        cell = args.campaign / str(index)
        cell.mkdir()
        inventory = cell / 'inventory.json'
        series._atomic_json(inventory, {'source_run': row['source_run'], 'checkpoints': [row]})
        specs = evaluate_matrix(MATRIX, row['path'], source_run=row['source_run'],
                                checkpoint_inventory=inventory, reference_bundle=reference,
                                eval_series_spec_dir=cell / 'specs')
        registry = series.create_run(args.registry, series._read(specs['specs'][SELECTOR]),
                                     args.campaign.name, 'ambi-inner-bench', 'rwgao_b-brown-university', OWNER)
        row.update(run_id=registry['run_id'], run_dir=registry['run_dir'], inventory=str(inventory))
        series._atomic_json(cell / 'run-map.json', {SELECTOR: row['run_dir']})
        rows.append(row)
    campaign = dict(source_commit=git('rev-parse', 'HEAD'), source_tree=git('rev-parse', 'HEAD^{tree}'),
                    matrix_sha256=series._file_digest(MATRIX), created_at=now(), rows=rows,
                    seeds=[101, 102, 103, 104, 105], controller_seed=55, max_steps=500)
    series._atomic_json(args.campaign / 'campaign.json', campaign)
    print(json.dumps(campaign), flush=True)


def validate_result(payload, seeds, steps):
    if len(payload['results']) != 1:
        raise ValueError('Expected exactly one Bernoulli result')
    result = payload['results'][0]
    if (result['selector'] != SELECTOR or not result['outer_state_unchanged']
            or result['outer_updates_before'] != result['outer_updates_after']
            or result['environment_seeds'] != seeds):
        raise ValueError('Frozen state or seed coverage failed')
    for episode in result['episodes']:
        metrics = episode['model_metrics']
        expected = {'inner_critic_bernoulli_p': .5, 'inner_critic_bernoulli_applied': (steps-1)/steps,
                    'inner_critic_optimizer_steps': 128., 'inner_actor_optimizer_steps': 32.,
                    'inner_model_steps': 3072., 'inner_compile_fallback': 0.}
        if episode['length'] != steps:
            raise ValueError('Incomplete episode')
        for key, value in expected.items():
            if abs(metrics.get(key, float('inf')) - value) > 1e-8:
                raise ValueError(f'Unexpected {key}: {metrics.get(key)} vs {value}')
    return result


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    if os.environ.get('WANDB_MODE') != 'disabled' or not torch.cuda.is_available():
        raise ValueError('Workers require CUDA and disabled W&B')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    campaign = load(args.campaign)
    row = campaign['rows'][args.index]
    for path, digest in (('path', 'sha256'), ('metadata_path', 'metadata_sha256'), ('reference', 'reference_sha256')):
        if series._file_digest(row[path]) != row[digest]:
            raise ValueError('Frozen input changed')
    cell = args.campaign / str(args.index)
    output = cell / ('smoke' if args.smoke else 'production')
    output.mkdir(exist_ok=False)
    seeds, steps = ([101, 102], 8) if args.smoke else (campaign['seeds'], campaign['max_steps'])
    started = time.perf_counter()
    payload = evaluate_matrix(MATRIX, row['path'], seeds=seeds, max_steps=steps, device='cuda',
                              bundle_dir=output / 'bundle', source_run=row['source_run'],
                              checkpoint_inventory=row['inventory'],
                              reference_bundle=None if args.smoke else row['reference'],
                              eval_run_map=None if args.smoke else cell / 'run-map.json', stage_results=False)
    result = validate_result(payload, seeds, steps)
    series._atomic_json(output / 'results.json', payload)
    receipt = dict(passed=True, seconds=time.perf_counter()-started, gpu=torch.cuda.get_device_name(),
                   source_commit=campaign['source_commit'], step=row['step'], seeds=seeds,
                   max_steps=steps, result=result)
    series._atomic_json(output / 'receipt.json', receipt)
    if not args.smoke:
        series.stage_result(row['run_dir'], output / 'bundle/manifest.json', selector=SELECTOR,
                            source_run=row['source_run'], inventory_path=row['inventory'])
    print(json.dumps(receipt), flush=True)


def layout(args):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    campaign = load(args.campaign)
    with series._lock(args.campaign / '.layout.lock'):
        rows = campaign['rows']
        progress = []
        ids, labels = [], {}
        for index, row in enumerate(rows):
            result_path = args.campaign / str(index) / 'production/receipt.json'
            receipt = series._read(result_path) if result_path.exists() else None
            result = f"{receipt['result']['return']['mean']:.3f}" if receipt else 'Pending'
            progress.append(f"| {row['label']} | {row['step']:,} | {result} |")
            ids.append(row['run_id'])
            labels[row['run_id']] = row['label']
            for baseline in row['baseline_runs']:
                ids.append(baseline['id'])
                labels[baseline['id']] = row['label'].split(' · ')[0] + ' · ' + baseline['label']
        markdown = ('**Critic-only Bernoulli 50% at the failed checkpoints.** Five paired full episodes, '
                    'seeds 101–105, controller seed 55, 500 mean-action decisions; checkpoint H3/J8/K1, '
                    'N128/C16/A4/B256 and entropy/Q-scale settings inherited. Each critic scalar retains '
                    'its previous adapted value with probability 0.5; actor/optimizer/replay/temperature reset. '
                    'First decision resets the critic too. Outer checkpoint stays frozen.\n\n'
                    'New evaluation uses compiled native execution; historical refinement uses isolated CUDA graph RNG. '
                    'The seed streams and controller timing therefore differ. Existing prior references are reused.\n\n'
                    '| Condition | Training decisions | New return mean |\n|---|---:|---:|\n' + '\n'.join(progress))
        panels = [wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=12))]
        for title, metric in [('Episode return · mean', 'eval/return_mean'),
                              ('Improvement over paired prior', 'eval/paired_gain_mean'),
                              ('Episode return · sample SD', 'eval/return_sample_std')]:
            panels.append(wr.LinePlot(title=title, x=series.X_AXIS, y=[metric], aggregate=False,
                                     smoothing_type='none', max_runs_to_show=len(ids),
                                     legend_template='${run:displayName}',
                                     line_titles={rid+':'+metric:label for rid,label in labels.items()},
                                     layout=wr.Layout(w=12,h=8)))
        section = ws.Section(name='Bernoulli critic / Results and progress', is_open=True, panels=panels)
        path = args.campaign / 'workspace.json'
        if path.exists():
            work = ws.Workspace.from_url(series._read(path)['url'])
            work.sections = [section if s.name == section.name else s for s in work.sections]
        else:
            work = ws.Workspace(entity='rwgao_b-brown-university', project='ambi-inner-bench',
                                name='Failed checkpoints · Bernoulli critic 50%', auto_generate_panels=False,
                                sections=[section], runset_settings=ws.RunsetSettings(
                                    filters=[expr.Metric('ID').isin(ids)], groupby=[]))
        work.save()
        verified = ws.Workspace.from_url(work.url)
        if not any(s.name == section.name and len(s.panels) == 4 for s in verified.sections):
            raise RuntimeError('Result panel setup failed')
        series._atomic_json(path, {'url':work.url,'api_verified':True,'updated_at':now()})
        print(json.dumps({'workspace_url':work.url}), flush=True)


def publish(args):
    campaign = load(args.campaign)
    row = campaign['rows'][args.index]
    command = [args.workspace_python, __file__, 'layout', '--campaign', str(args.campaign)]
    subprocess.run(command, check=True)
    with series.Publisher(row['run_dir'], owner=OWNER) as publisher:
        publisher.run.name = row['label'] + f" · {row['step']//1000}k dip"
        publisher.run.config.update({'curve_label':row['label']}, allow_val_change=True)
        while True:
            status = publisher.publish_pending()
            publisher.run.summary.update({'evaluation/checkpoints_expected':1,
                                          'evaluation/checkpoints_accepted':status['accepted'],
                                          'evaluation/status':'complete' if status['accepted']==1 else 'pending'})
            if status['accepted'] == 1:
                break
            active = subprocess.check_output(['squeue','-h','-j',args.compute_job,'-o','%i'],text=True).strip()
            if not active:
                raise RuntimeError('Compute ended without a complete validated result')
            time.sleep(15)
    subprocess.run(command, check=True)
    series._atomic_json(args.campaign / f'published-{args.index}.json', {'run_id':row['run_id'],'published':1})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','worker','layout','publish'])
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--reference', nargs=2, action='append')
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--compute-job')
    parser.add_argument('--workspace-python', default=sys.executable)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == '__main__':
    main()
