"""Install and verify the named Q10 training workspace without creating runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_alpha_carry_workspace import ENTITY, PROJECT, WorkspaceError, _filter_wire, _view_id, _write
from slurm.mrsq_q10_suite import HYDRA_SEED56_TASKS, HYDRA_SEED57_TASKS, SEED56_TASKS

NAME = 'AMBI Q10 actor mean / target min — Oscar training'
PREFIX = 'AMBI Q10 training / '
CAMPAIGN = 'mrsq-q10-suite-20261122'
HYDRA_CAMPAIGNS = {
    56: 'mrsq-q10-dogs-s56-hydra-20261124',
    57: 'mrsq-q10-seed57-hydra-20261125',
}
TASKS = ('humanoid-walk', 'humanoid-run', 'dog-trot', 'dog-run', 'humanoid-stand', 'dog-walk')
CELLS = (
    tuple((task, 55, 'Oscar') for task in TASKS)
    + tuple((task, 56, 'Oscar') for task in SEED56_TASKS)
    + tuple((task, 56, 'Hydra') for task in HYDRA_SEED56_TASKS)
    + tuple((task, 57, 'Hydra') for task in HYDRA_SEED57_TASKS)
)
STEP = 'env_step'
METRICS = {
    'Learning': {
        'Sampled training episode return': 'episode/return',
        'Training return, rolling 20 episodes': 'episode/return_window20_mean',
    },
    'Training progress': {
        'Training decisions completed': 'time/total_timesteps',
        'Decisions per second': 'time/fps',
        'Whole-inner CUDA graph use': 'train/inner_cuda_graph_used',
        'Inner solve seconds per action': 'time/inner_seconds_per_action',
    },
}


def discover_rows(api, previous_rows):
    previous_ids = {(row['task'], row.get('seed', 55)): row['run_id'] for row in previous_rows}
    rows = []
    for task, seed, cluster in CELLS:
        name = f"ambi_{task.replace('-', '_')}_mrsq_q10_seed{seed}"
        path = ROOT / 'configs/dmcontrol/algs' / f'{name}.json'
        params = json.loads(path.read_text())['alg_params']
        if cluster == 'Hydra':
            group = f'{HYDRA_CAMPAIGNS[seed]}-{task}'
        else:
            group = f'{CAMPAIGN}-{task}' + ('' if seed == 55 else f'-s{seed}')
        if (params['wandb_group'], params['wandb_entity'], params['wandb_project']) != (group, ENTITY, PROJECT):
            raise WorkspaceError(f'Unexpected W&B destination for {task}')
        runs = list(api.runs(f'{ENTITY}/{PROJECT}', filters={'group': group}, per_page=2))
        if len(runs) > 1 or any((run.group, run.name) != (group, params['wandb_run_name']) for run in runs):
            raise WorkspaceError(f'Unexpected or duplicate W&B training run for {task}')
        run = runs[0] if runs else None
        known_id = previous_ids.get((task, seed))
        if known_id and run and known_id != run.id:
            raise WorkspaceError(f'W&B training identity changed for {task}')
        rows.append(dict(task=task, seed=seed, cluster=cluster, group=group, name=params['wandb_run_name'],
                         run_id=run.id if run else known_id, state=run.state if run else 'pending',
                         decisions=run.summary.get(STEP) if run else None))
    return rows


def markdown(rows):
    status = '\n'.join(f"| {row['task']} | {row['seed']} | {row['state']} | "
                       f"{row['decisions'] if row['decisions'] is not None else 'pending'} | {row['cluster']} |"
                       for row in rows)
    return f"""**Q10 · actor mean / target min · Oscar + Hydra**

Each task trains for 1,000,000 decisions. The actors average all ten critics.
The targets and terminal bootstrap use their minimum. All 25k model checkpoints
are retained. Full training state is saved separately for exact resume.
Dog Walk seed 55 is a fresh attempt with its own training history.

The return curves show sampled training episodes. Preliminary GPU tests keep
W&B disabled. Panels populate as production uploads durable training metrics.

| Task | Seed | State at refresh | Uploaded decisions | Cluster |
| --- | --- | --- | --- | --- |
{status}
"""


def sections(rows, ws, wr):
    result = [ws.Section(name=PREFIX + 'Protocol and status', is_open=True,
                         panels=[wr.MarkdownPanel(markdown=markdown(rows),
                                                  layout=wr.Layout(w=24, h=len(CELLS) + 1))])]
    for section, metrics in METRICS.items():
        result.append(ws.Section(name=PREFIX + section, is_open=True, panels=[
            wr.LinePlot(title=title, x=STEP, y=[metric], title_x='Training decisions',
                        aggregate=False, smoothing_type='none', legend_template='${run:displayName}',
                        max_runs_to_show=len(CELLS), layout=wr.Layout(w=12, h=8))
            for title, metric in metrics.items()]))
    return result


def verify(work, rows, url, wanted, wr, expr):
    if ((work.entity, work.project, work.name) != (ENTITY, PROJECT, NAME)
            or _view_id(work.url) != _view_id(url)):
        raise WorkspaceError('Saved Q10 workspace identity differs from the receipt')
    settings = work.runset_settings
    if settings.groupby or settings.query or _filter_wire(settings.filters, expr) != _filter_wire(wanted, expr):
        raise WorkspaceError('Saved Q10 workspace lost its exact training-group filter')
    owned = [section for section in work.sections if section.name.startswith(PREFIX)]
    expected = {title: metric for metrics in METRICS.values() for title, metric in metrics.items()}
    if len(owned) != len(METRICS) + 1 or not all(section.is_open for section in owned):
        raise WorkspaceError('Saved Q10 workspace is missing visible sections')
    panels = [panel for section in owned for panel in section.panels]
    status = [panel.markdown for panel in panels if isinstance(panel, wr.MarkdownPanel)]
    if status != [markdown(rows)]:
        raise WorkspaceError('Saved Q10 workspace is missing its pending/progress rows')
    curves = [panel for panel in panels if isinstance(panel, wr.LinePlot)]
    if len(curves) != len(expected) or set(expected) != {panel.title for panel in curves}:
        raise WorkspaceError('Saved Q10 workspace is missing expected training charts')
    for panel in curves:
        if (getattr(panel.x, 'name', panel.x) != STEP
                or [getattr(value, 'name', value) for value in panel.y] != [expected[panel.title]]
                or panel.aggregate or panel.smoothing_type != 'none'
                or panel.max_runs_to_show < len(CELLS)):
            raise WorkspaceError(f'Saved Q10 chart has incorrect series or axes: {panel.title}')


def install(output, *, existing_url=None, api=None, backend=None):
    previous = json.loads(output.read_text()) if output.exists() else {}
    identity = dict(entity=ENTITY, project=PROJECT, campaign=CAMPAIGN, name=NAME)
    if previous and any(previous.get(key) != value for key, value in identity.items()):
        raise WorkspaceError('Existing receipt belongs to another campaign')
    url = previous.get('url')
    if existing_url:
        if url and _view_id(url) != _view_id(existing_url):
            raise WorkspaceError('Existing receipt points to a different view than requested')
        url = existing_url
    if url:
        _view_id(url)
    if backend is None:
        import wandb
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        from wandb_workspaces import expr
        api = api or wandb.Api(timeout=30)
    else:
        ws, wr, expr = backend
    rows = discover_rows(api, previous.get('rows', []))
    plan = sections(rows, ws, wr)
    wanted = expr.Metric('Group').isin([row['group'] for row in rows])
    receipt = dict(**identity, url=url, rows=rows, api_verified=False,
                   updated_at=datetime.now(timezone.utc).isoformat(),
                   panel_count=sum(len(section.panels) for section in plan))
    if url:
        work = ws.Workspace.from_url(url)
        if ((work.entity, work.project, work.name) != (ENTITY, PROJECT, NAME)
                or _view_id(work.url) != _view_id(url)):
            raise WorkspaceError('Existing receipt points to an unrelated workspace')
        work.sections = [section for section in work.sections if not section.name.startswith(PREFIX)] + plan
    else:
        work = ws.Workspace(entity=ENTITY, project=PROJECT, name=NAME, sections=plan,
                            auto_generate_panels=False)
    work.runset_settings = ws.RunsetSettings(filters=[wanted], groupby=[],
                                            pinned_columns=['run:displayName', 'summary:env_step'])
    work.settings = ws.WorkspaceSettings(x_axis=STEP, max_runs=len(CELLS),
                                        tooltip_number_of_runs='all_runs', tooltip_color_run_names=True)
    _write(output, receipt)
    work.save()
    receipt.update(url=work.url, view_id=_view_id(work.url))
    _write(output, receipt)  # Preserve the URL if independent verification fails.
    verify(ws.Workspace.from_url(work.url), rows, work.url, wanted, wr, expr)
    receipt['api_verified'] = True
    _write(output, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Unique campaign workspace receipt')
    parser.add_argument('--existing-url', help='Update this existing view even if the receipt is missing')
    args = parser.parse_args()
    print(json.dumps(install(args.output, existing_url=args.existing_url), indent=2))
