"""Idempotent result and progress panels for native TD-MPC2 Humanoid Run."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import wandb
STATE_PATH = Path(sys.argv[1]).resolve()
HERE = STATE_PATH.parent
import fcntl
_lock = (HERE / 'workspace.lock').open('a')
fcntl.flock(_lock, fcntl.LOCK_EX)
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr
from wandb_workspaces import expr

state = json.loads(STATE_PATH.read_text())
api = wandb.Api()
specs = state['runs'] + state.get('controls', [])
entity, project = 'rwgao_b-brown-university', 'ambi'
ids = [r['id'] for r in specs]
existing = {r.id:r for r in api.runs(entity+'/'+project, filters={'name':{'$in':ids}}, per_page=8)}
rows=[]
for spec in specs:
    r=existing.get(spec['id'])
    summary={} if r is None else dict(r.summary)
    pending_state = ("submitted / awaiting training start" if spec.get('job') else
                     f"pending GPU validation / gate {spec.get('gate_job', 'not submitted')}")
    rows.append({**spec,'state':pending_state if r is None else r.state,
                 'decisions':summary.get('env_step','pending'), 'job':spec.get('job','pending submission')})
metrics={'Returns and progress': [('Training episode return','episode/return'),('Decisions per second','time/fps')],
         'Native learner': [('Total model loss','train/total_loss'),('Policy loss','train/pi_loss'),('Value loss','train/value_loss'),('Reward loss','train/reward_loss')]}
timestamp=datetime.now(timezone.utc).isoformat()
status='\n'.join(f"| {r['label'].replace('|','/')} | {r['job']} | {r['state']} | {r['decisions']} |" for r in rows)
if not state.get('protocol'):
    raise ValueError('A scientific protocol and pending/progress description are required.')
if state.get('protocol'):
    md = state['protocol'] + f"\n\nSource `{state['commit']}`. Outputs `{state['output']}`.\n\n| Arm | Job | State at {timestamp} | Uploaded decisions |\n|---|---|---|---|\n{status}\n\nThis table is a timestamped snapshot; charts update with training. Missing metrics remain pending."

sections=[ws.Section(name='TD-MPC2 / Protocol and progress',is_open=True,
                    panels=[wr.MarkdownPanel(markdown=md,layout=wr.Layout(w=24,h=15))])]
for name,group in metrics.items():
    panels=[wr.LinePlot(title=title,x='env_step',y=[metric],title_x='Agent decisions',aggregate=False,
                       smoothing_type='none',legend_template='${run:displayName}',max_runs_to_show=8,
                       layout=wr.Layout(w=12,h=8),
                       line_colors={s['id']+':'+metric:s['color'] for s in specs},
                       line_titles={s['id']+':'+metric:s['label'] for s in specs}) for title,metric in group]
    sections.append(ws.Section(name='TD-MPC2 / '+name,is_open=True,panels=panels,
                               layout_settings=ws.SectionLayoutSettings(columns=2,rows=2)))
receipt_path=HERE/'workspace.json'
if receipt_path.exists() or state.get('workspace_url'):
    url = json.loads(receipt_path.read_text())['url'] if receipt_path.exists() else state['workspace_url']
    work=ws.Workspace.from_url(url)
    replacements={s.name:s for s in sections}
    work.sections=[replacements.pop(s.name,s) for s in work.sections]
    work.sections.extend(replacements.values())
    work.runset_settings.filters=[expr.Metric('ID').isin(ids)]
    work.runset_settings.run_settings={s['id']:ws.RunSettings(color=s['color'],disabled=False) for s in specs}
else:
    work=ws.Workspace(entity=entity,project=project,name=state.get('workspace_name', 'TD-MPC2 — Humanoid Run'),
        auto_generate_panels=False,sections=sections,
        settings=ws.WorkspaceSettings(x_axis='env_step',max_runs=8,tooltip_number_of_runs='all_runs',tooltip_color_run_names=True),
        runset_settings=ws.RunsetSettings(filters=[expr.Metric('ID').isin(ids)],groupby=[],
            run_settings={s['id']:ws.RunSettings(color=s['color'],disabled=False) for s in specs},
            pinned_columns=['run:displayName','summary:env_step','summary:episode/return']))
work.save()
receipt={'url':work.url,'run_ids':ids,'updated_at':timestamp,'api_verified':False,'runs':rows}
temporary = HERE / f'workspace.{os.getpid()}.tmp'
temporary.write_text(json.dumps(receipt,indent=2)+'\n')
os.replace(temporary, receipt_path)
readback=ws.Workspace.from_url(work.url)
assert (readback.entity,readback.project)==(entity,project)
assert readback.settings.max_runs==8 and readback.settings.x_axis=='env_step'
assert readback.runset_settings.filters == f'Metric("ID") in {ids!r}'
owned={s.name:s for s in readback.sections if s.name.startswith('TD-MPC2 / ')}
expected_panels=1+sum(len(group) for group in metrics.values())
assert len(owned)==1+len(metrics) and sum(len(s.panels) for s in owned.values())==expected_panels
for title,items in metrics.items():
    for panel,(_,metric) in zip(owned['TD-MPC2 / '+title].panels,items):
        assert [getattr(v,'name',v) for v in panel.y]==[metric]
receipt.update(api_verified=True,panel_count=expected_panels)
temporary = HERE / f'workspace.{os.getpid()}.tmp'
temporary.write_text(json.dumps(receipt,indent=2)+'\n')
os.replace(temporary, receipt_path)
print(json.dumps(receipt,indent=2))
