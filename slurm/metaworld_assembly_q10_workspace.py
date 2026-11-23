"""Install and verify one MetaWorld Q10 workspace without initializing runs."""

import argparse
from datetime import datetime, timezone
import fcntl
from itertools import islice
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_alpha_carry_workspace import ENTITY, PROJECT, WorkspaceError, _filter_wire, _view_id, _write

TASKS = ('assembly', 'hand-insert')
STEP = 'env_step'
CURVES = (
    ('Final-step evaluation success rate', 'eval/success_rate', 'Success fraction'),
    ('Evaluation episode return', 'eval/episode_reward', 'Episode return'),
    ('Sampled training episode return', 'episode/return', 'Episode return'),
)
REDUCTIONS = dict(num_q=10, q_pair_size=10, outer_q_actor_reduction='mean_all',
                  inner_q_actor_reduction='mean_all', outer_q_target_reduction='min_all',
                  inner_q_target_reduction='min_all', mppi_terminal_q_reduction='min_all')


def task_title(task):
    return task.replace('-', ' ').title()


def campaign_identity(task):
    if task not in TASKS:
        raise WorkspaceError(f'Unsupported MetaWorld Q10 task: {task}')
    return dict(entity=ENTITY, project=PROJECT,
                name=f'Meta-World {task_title(task)} · Q10 actor mean / target min',
                group=f'metaworld-{task}-mrsq-q10-s55')


def discover(api, previous_id, lineage, task='assembly'):
    group = campaign_identity(task)['group']
    name = f'Meta-World {task_title(task)} | Q10 actor mean / target min | seed55'
    matches = list(islice(api.runs(f'{ENTITY}/{PROJECT}', filters={'group': group}, per_page=2), 2))
    if len(matches) > 1:
        raise WorkspaceError(f'Duplicate runs in the {task} Q10 group')
    if not matches:
        return dict(run_id=previous_id, state='awaiting first upload', decisions=None)
    run = matches[0]
    if (run.group != group or run.name != name or not re.fullmatch('[0-9a-f]{32}', run.id)
            or (previous_id and previous_id != run.id)):
        raise WorkspaceError(f'{task} Q10 group contains an unexpected run identity')
    if not lineage.is_file():
        raise WorkspaceError(f'{task} Q10 registration has no native lineage metadata yet')
    metadata = json.loads(lineage.read_text())['metadata']
    if metadata['initial_wandb_run_id'] != run.id or metadata['total_steps'] != 1_000_000:
        raise WorkspaceError(f'The remote run differs from the native {task} Q10 lineage')
    config = run.config
    params, training = config.get('alg_params', {}), config.get('run_params', {})
    selected = training.get('resolved_runtime', {}).get('observation', {})
    expected = dict(**REDUCTIONS, eval_freq=25_000, eval_episodes=10, compile=True,
                    compile_strict=True, inner_cuda_graph=True, critic_geometry_diagnostics=True)
    if (any(params.get(key) != value for key, value in expected.items())
            or training.get('seed') != 55 or training.get('total_steps') != 1_000_000
            or training.get('env') != 'MetaWorld-v0' or selected.get('task') != task
            or selected.get('episode_length') != 100 or selected.get('action_repeat') != 2):
        raise WorkspaceError(f'{task} Q10 online publication config differs from the pinned protocol')
    return dict(run_id=run.id, state=run.state, decisions=run.summary.get(STEP))


def progress(row, task):
    title = task_title(task)
    decisions = 'pending' if row['decisions'] is None else f"{int(row['decisions']):,}"
    return f"""**Meta-World {title} · Q10 actor mean / target min · seed 55**

The run trains for 1,000,000 decisions. Both actors average all ten critics.
Targets and terminal values use the minimum over all ten critics.

The online evaluation runs ten consecutive 100-decision episodes with mean
actions at step zero and every 25,000 training decisions. Success is measured
at the final step of each episode. The sampled training return has its own chart.

| Task | W&B state at refresh | Uploaded decisions |
| --- | --- | ---: |
| {title} | {row['state']} | {decisions} |

Charts update as training uploads metrics. Unobserved points remain pending.
"""


def panels(row, wr, task):
    result = [wr.MarkdownPanel(markdown=progress(row, task), layout=wr.Layout(w=24, h=11))]
    result.extend(wr.LinePlot(title=title, x=STEP, y=[metric], title_x='Training decisions',
                             title_y=ylabel, aggregate=False, smoothing_type='none',
                             legend_template='${run:displayName}', max_runs_to_show=1,
                             layout=wr.Layout(w=12, h=8)) for title, metric, ylabel in CURVES)
    return result


def verify(work, row, expected_filter, wr, expr, task):
    if (work.entity, work.project, work.name) != (ENTITY, PROJECT, campaign_identity(task)['name']):
        raise WorkspaceError(f'Saved view does not belong to this {task} Q10 campaign')
    settings = work.runset_settings
    if settings.groupby or settings.query or _filter_wire(settings.filters, expr) != _filter_wire(expected_filter, expr):
        raise WorkspaceError(f'Saved view lost its exact {task} Q10 run filter')
    section_name = f'{task_title(task)} Q10 evaluation and training'
    selected = [section for section in work.sections if section.name == section_name]
    if len(selected) != 1 or not selected[0].is_open or len(selected[0].panels) != 4:
        raise WorkspaceError(f'Saved view is missing visible {task} panels')
    status, *plots = selected[0].panels
    if not isinstance(status, wr.MarkdownPanel) or status.markdown != progress(row, task):
        raise WorkspaceError(f'Saved {task} progress panel differs from the requested content')
    for panel, (title, metric, ylabel) in zip(plots, CURVES):
        if (not isinstance(panel, wr.LinePlot) or panel.title != title
                or getattr(panel.x, 'name', panel.x) != STEP
                or [getattr(value, 'name', value) for value in panel.y] != [metric]
                or panel.title_y != ylabel or panel.aggregate or panel.smoothing_type != 'none'
                or panel.max_runs_to_show != 1):
            raise WorkspaceError(f'Saved {task} chart differs from the requested metric: {title}')


def install(output, *, api=None, backend=None, lineage=None, task='assembly'):
    if backend is None:
        import wandb
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        from wandb_workspaces import expr
        api = api or wandb.Api(timeout=30)
    else:
        ws, wr, expr = backend
    campaign_identity(task)
    if lineage is None:
        campaign = Path(f'/oscar/scratch/rgao48/ambi-metaworld-{task}-mrsq-q10-20261123')
        lineage = campaign / f'{task}/ambi_state_mrsq_q10/seed_55/lineage/LINEAGE.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _install(output, api, ws, wr, expr, lineage, task)


def _install(output, api, ws, wr, expr, lineage, task):
    identity = campaign_identity(task)
    previous = json.loads(output.read_text()) if output.exists() else {}
    if previous and any(previous.get(key) != value for key, value in identity.items()):
        raise WorkspaceError('Existing receipt belongs to another campaign')
    url = previous.get('url')
    if not url and previous.get('creation_attempted'):
        raise WorkspaceError('Previous view creation outcome is uncertain; verify its URL before retrying')
    row = discover(api, previous.get('run', {}).get('run_id'), lineage, task)
    expected_filter = expr.Metric('ID') == row['run_id'] if row['run_id'] else expr.Metric('Group') == identity['group']
    if url:
        _view_id(url)
        work = ws.Workspace.from_url(url)
        if (work.entity, work.project, work.name) != (ENTITY, PROJECT, identity['name']):
            raise WorkspaceError('Existing receipt points to an unrelated view')
    else:
        work = ws.Workspace(entity=ENTITY, project=PROJECT, name=identity['name'], auto_generate_panels=False)
    section_name = f'{task_title(task)} Q10 evaluation and training'
    work.sections = [ws.Section(name=section_name, is_open=True, panels=panels(row, wr, task)),
                     *[section for section in work.sections if section.name != section_name]]
    work.runset_settings = ws.RunsetSettings(filters=[expected_filter], groupby=[],
        pinned_columns=['run:displayName', 'summary:env_step', 'summary:eval/success_rate'])
    work.settings = ws.WorkspaceSettings(x_axis=STEP, max_runs=1, tooltip_number_of_runs='all_runs')
    receipt = dict(**identity, url=url, run=row, api_verified=False, creation_attempted=True,
                   updated_at=datetime.now(timezone.utc).isoformat(), panel_count=4)
    _write(output, receipt)
    try:
        work.save()
        receipt.update(url=work.url, view_id=_view_id(work.url))
        _write(output, receipt)
        live = ws.Workspace.from_url(work.url)
        if _view_id(live.url) != receipt['view_id']:
            raise WorkspaceError('Saved workspace URL differs from the receipt')
        verify(live, row, expected_filter, wr, expr, task)
    except Exception as exc:
        receipt['failure_type'] = type(exc).__name__
        _write(output, receipt)
        raise
    receipt['api_verified'] = True
    _write(output, receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Campaign workspace receipt')
    parser.add_argument('--task', choices=TASKS, default='assembly')
    args = parser.parse_args()
    print(json.dumps(install(args.output, task=args.task), indent=2))
