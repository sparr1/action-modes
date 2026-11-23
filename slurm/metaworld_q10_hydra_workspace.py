"""Install and verify the named Hydra Q10 MetaWorld training workspace."""

import argparse
from datetime import datetime, timezone
import fcntl
from itertools import islice
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_alpha_carry_workspace import ENTITY, PROJECT, WorkspaceError, _filter_wire, _view_id, _write
from slurm.metaworld_assembly_q10_workspace import CURVES, STEP

CAMPAIGN = Path('/data/public/rgao48/ambi/metaworld-q10-push-stick-pull-20261123')
VIEW_NAME = 'Meta-World Push and Stick Pull · Q10 actor mean / target min'
SECTION = 'Hydra Q10 evaluation and training'
TASKS = {'push': 'Push', 'stick-pull': 'Stick Pull'}


def discover(api, previous):
    known_ids = {row['task']: row['run_id'] for row in previous}
    rows = []
    for task, title in TASKS.items():
        group = f'metaworld-{task}-mrsq-q10-s55'
        name = f'Meta-World {title} | Q10 actor mean / target min | seed55'
        runs = list(islice(api.runs(f'{ENTITY}/{PROJECT}', filters={'group': group}, per_page=2), 2))
        if len(runs) > 1 or any(run.name != name or run.group != group for run in runs):
            raise WorkspaceError(f'Unexpected or duplicate W&B run in {group}')
        run = runs[0] if runs else None
        known = known_ids.get(task)
        if known and run and run.id != known:
            raise WorkspaceError(f'W&B run identity changed for {task}')
        rows.append(dict(task=task, title=title, name=name, group=group,
                         run_id=run.id if run else known, state=run.state if run else 'awaiting first upload',
                         decisions=run.summary.get(STEP) if run else None))
    return rows


def progress(rows):
    status = '\n'.join(f"| {row['title']} | {row['state']} | "
                       f"{row['decisions'] if row['decisions'] is not None else 'pending'} |"
                       for row in rows)
    return f"""**Meta-World Q10 · actor mean / target min · seed 55 · Hydra**

Push and Stick Pull each train for 1,000,000 decisions with ten critics.
The two actors average the ensemble. Both TD targets and terminal values use its minimum.

Online evaluation uses ten 100-decision episodes with mean actions at step zero
and every 25,000 training decisions. Success is the fraction successful at the
final episode step. Sampled training return is charted separately.

| Task | W&B state at refresh | Uploaded decisions |
| --- | --- | ---: |
{status}

Charts populate as training uploads metrics.
"""


def panels(rows, wr):
    return [wr.MarkdownPanel(markdown=progress(rows), layout=wr.Layout(w=24, h=11)),
            *(wr.LinePlot(title=title, x=STEP, y=[metric], title_x='Training decisions',
                          title_y=ylabel, aggregate=False, smoothing_type='none',
                          legend_template='${run:displayName}', max_runs_to_show=len(TASKS),
                          layout=wr.Layout(w=12, h=8)) for title, metric, ylabel in CURVES)]


def verify(work, rows, url, wanted, wr, expr):
    if ((work.entity, work.project, work.name) != (ENTITY, PROJECT, VIEW_NAME)
            or _view_id(work.url) != _view_id(url)):
        raise WorkspaceError('Saved view identity does not match the Hydra Q10 campaign')
    settings = work.runset_settings
    if settings.groupby or settings.query or _filter_wire(settings.filters, expr) != _filter_wire(wanted, expr):
        raise WorkspaceError('Saved view lost its exact two Hydra Q10 training groups')
    owned = [section for section in work.sections if section.name == SECTION]
    if len(owned) != 1 or not owned[0].is_open or len(owned[0].panels) != 4:
        raise WorkspaceError('Saved view is missing the visible Hydra Q10 panels')
    status, *plots = owned[0].panels
    if not isinstance(status, wr.MarkdownPanel) or status.markdown != progress(rows):
        raise WorkspaceError('Saved Hydra Q10 progress panel failed readback')
    for panel, (title, metric, ylabel) in zip(plots, CURVES):
        if (not isinstance(panel, wr.LinePlot) or panel.title != title
                or getattr(panel.x, 'name', panel.x) != STEP
                or [getattr(value, 'name', value) for value in panel.y] != [metric]
                or panel.title_y != ylabel or panel.aggregate or panel.smoothing_type != 'none'
                or panel.max_runs_to_show != len(TASKS)):
            raise WorkspaceError(f'Saved Hydra Q10 chart differs from the requested metric: {title}')


def install(output, *, api=None):
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identity = dict(entity=ENTITY, project=PROJECT, name=VIEW_NAME, campaign=CAMPAIGN.name)
        previous = json.loads(output.read_text()) if output.exists() else {}
        if previous and any(previous.get(key) != value for key, value in identity.items()):
            raise WorkspaceError('Workspace receipt belongs to another campaign')
        url = previous.get('url')
        if not url and previous.get('creation_attempted'):
            raise WorkspaceError('Previous view creation outcome is uncertain; check its URL before retrying')
        rows = discover(api or wandb.Api(timeout=30), previous.get('rows', []))
        wanted = expr.Metric('Group').isin([row['group'] for row in rows])
        if url:
            _view_id(url)
            work = ws.Workspace.from_url(url)
            if (work.entity, work.project, work.name) != (ENTITY, PROJECT, VIEW_NAME):
                raise WorkspaceError('Existing receipt points to an unrelated view')
        else:
            work = ws.Workspace(entity=ENTITY, project=PROJECT, name=VIEW_NAME, auto_generate_panels=False)
        work.sections = [ws.Section(name=SECTION, is_open=True, panels=panels(rows, wr)),
                         *[section for section in work.sections if section.name != SECTION]]
        work.runset_settings = ws.RunsetSettings(filters=[wanted], groupby=[],
            pinned_columns=['run:displayName', 'summary:env_step', 'summary:eval/success_rate'])
        work.settings = ws.WorkspaceSettings(x_axis=STEP, max_runs=len(TASKS),
                                            tooltip_number_of_runs='all_runs')
        receipt = dict(**identity, url=url, rows=rows, api_verified=False, creation_attempted=True,
                       updated_at=datetime.now(timezone.utc).isoformat(), panel_count=4)
        _write(output, receipt)
        try:
            work.save()
            receipt.update(url=work.url, view_id=_view_id(work.url))
            _write(output, receipt)
            verify(ws.Workspace.from_url(work.url), rows, work.url, wanted, wr, expr)
        except Exception as exc:
            receipt['failure_type'] = type(exc).__name__
            _write(output, receipt)
            raise
        receipt['api_verified'] = True
        _write(output, receipt)
        return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=CAMPAIGN / 'workspace.json')
    args = parser.parse_args()
    print(json.dumps(install(args.output), indent=2))
