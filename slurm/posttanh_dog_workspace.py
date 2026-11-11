import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import wandb
STATE_PATH = Path(sys.argv[1]).resolve()
HERE = STATE_PATH.parent
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr
from wandb_workspaces import expr

state = json.loads(STATE_PATH.read_text())
api = wandb.Api()
specs = state['runs'] + state.get('controls', [])
entity, project = 'rwgao_b-brown-university', 'ambi'
ids = [r['id'] for r in specs]
max_runs = max(8, len(specs))
existing = {r.id:r for r in api.runs(entity+'/'+project, filters={'name':{'$in':ids}}, per_page=8)}
rows=[]
for spec in specs:
    r=existing.get(spec['id'])
    summary={} if r is None else dict(r.summary)
    rows.append({**spec,'state':spec.get('phase', 'pending GPU validation') if r is None else r.state,
                 'decisions':summary.get('env_step','pending'), 'job':spec.get('training_job','not submitted'), 'gate_job':spec.get('gate_job','not submitted')})
metrics={
    'Returns and progress':[
        ('Training episode return','episode/return'),
        ('Rolling 20-episode mean','episode/return_window20_mean'),
        ('Rolling mean drawdown','episode/return_window20_drawdown'),
        ('Decisions per second','time/fps')],
    'Policy entropy and saturation':[
        ('Joint Gaussian entropy before tanh','train/actor_pre_tanh_entropy'),
        ('True joint post-tanh entropy','train/actor_post_tanh_entropy'),
        ('Sampled actions at |a| >= 0.99','train/actor_action_abs_ge_0p99_fraction'),
        ('Deterministic mean actions at |a| >= 0.99','train/actor_mean_action_abs_ge_0p99_fraction')],
    'Executed actions':[
        ('Executed saturation at |a| >= 0.99','train/executed_action_abs_ge_0p99_fraction'),
        ('Within-episode action change RMS','train/executed_action_delta_rms'),
        ('Executed coordinate sign flips','train/executed_action_sign_flip_fraction'),
        ('Opposite-bound flips','train/executed_action_bound_flip_fraction')],
    'Q normalization and learning':[
        ('Actor Q normalization scale','train/actor_loss_scale'),
        ('Instantaneous Q percentile range','train/actor_q_range_instant'),
        ('Entropy coefficient in joint return units','train/actor_entropy_coefficient_joint_units'),
        ('Critic fitting loss','train/critic_loss')],
    'Coverage and numerical health':[
        ('Observed return-window episodes','episode/return_window20_count'),
        ('Actor mean vs minimum ensemble Q','train/actor_q_mean_all_minus_min_all'),
        ('Actor gradient norm','train/actor_grad_norm'),
        ('Compilation fallback (must stay zero)','train/compile_fallback')]
}
metrics['Inner normalization']=[
    ('Inner / outer scale at solve start','train/inner_actor_loss_scale_initial_over_outer'),
    ('Inner / outer scale at solve end','train/inner_actor_loss_scale_final_over_outer'),
    ('Inner instantaneous Q range','train/inner_actor_q_range_instant'),
    ('Inner entropy coefficient in joint return units','train/inner_actor_entropy_coefficient_joint_units')]
metrics['Inner policy']=[
    ('Inner pre-tanh entropy','train/inner_actor_pre_tanh_entropy'),
    ('Inner post-tanh entropy','train/inner_actor_post_tanh_entropy'),
    ('Inner sampled action saturation','train/inner_actor_action_abs_ge_0p99_fraction'),
    ('Inner actor gradient norm','train/inner_actor_grad_norm')]
metrics['CUDA graph execution']=[
    ('Whole-inner CUDA graph used','train/inner_cuda_graph_used'),
    ('Inner solve seconds per decision','time/inner_seconds_per_action'),
    ('Inner compilation fallback','train/inner_compile_fallback'),
    ('Outer compilation fallback','train/compile_fallback')]
for term, label in [('mpr', 'MPR geometry'), ('action_smoothness', 'Action geometry')]:
    metrics[label]=[
        ('Unweighted consistency loss',f'train/critic_geometry_primary_{term}_loss'),
        ('Relative gradient change',f'train/critic_geometry_primary_{term}_relative_gradient_change'),
        ('Gradient direction agreement',f'train/critic_geometry_primary_{term}_cosine_similarity'),
        ('Valid direction fraction',f'train/critic_geometry_primary_{term}_cosine_valid_fraction')]
metrics['Critic gradient magnitude']=[
    ('Action gradient RMS','train/critic_geometry_primary_action_gradient_rms'),
    ('Nonzero gradient validity','train/critic_geometry_primary_gradient_scale_valid')]
timestamp=datetime.now(timezone.utc).isoformat()
status='\n'.join(f"| {r['label'].replace('|','/')} | {r['gate_job']} | {r['job']} | {r['state']} | {r['decisions']} |" for r in rows)
md=f'''**H3/J8/K1 entropy and Q-normalization comparison — Oscar L40S.**

Six Humanoid Walk runs: pre/post-tanh entropy crossed with outer-frozen,
outer-reset/adapt, and run-carry/adapt inner normalization.
Each uses seed55 and **1,000,000 decisions**, N128/C16/A4/B256, a reward-only
five-head critic, and finite-horizon prior bootstrapping.
Entropy mode matches inner/outer: fixed native coefficient0.0001, or
joint post-tanh coefficient0.0021=21×0.0001. No temperature tuning.
Q scale is P95–P5, floor1, EMA0.01 per actor update. Carry survives episodes.

All **40 checkpoints per arm** are retained every25k decisions. Compilation,
vectorized critics, GPU replay and TF32 are enabled. Whole-inner CUDA graphs
execute each supported solve, with sparse native diagnostic solves retained.
Every arm passes GPU parity and end-to-end throughput validation before launch;
CPU and walltime requests come from those measurements. Graph-private RNG changes
the random trajectory relative to uncaptured execution; objectives and update dose
are unchanged. Critic geometry is observed every 1,000 outer updates on up to64
replay transitions, using observation/action noise0.01/0.02; no penalties are added. Diagnostic gates never create W&B runs.
The two prior-only controls run on Hydra L40 hardware; runtime is not a matched
hardware comparison. Training-return curves use agent decisions and no smoothing.

Source `{state['commit']}`. Checkpoint root `{state['output']}`.

| Task | GPU validation job | Training job | State at {timestamp} | Uploaded decisions |
|---|---|---|---|---|
{status}

This table is a timestamped snapshot. Charts update with training; missing
inner metrics on prior-only controls are intentional. Pending arms have no
fabricated data. Return drawdowns require 20 complete episodes.
'''
if state.get('protocol'):
    md = f"""{state['protocol']}

Source `{state['commit']}`. Checkpoint root `{state['output']}`.

| Task | GPU validation job | Training job | State at {timestamp} | Uploaded decisions |
|---|---|---|---|---|
{status}

The table is a timestamped snapshot; charts update with training. Startup
validation has W&B disabled. Missing results remain explicitly pending.
"""
sections=[ws.Section(name='Dog post-tanh / Protocol and progress',is_open=True,
                    panels=[wr.MarkdownPanel(markdown=md,layout=wr.Layout(w=24,h=15))])]
for name,group in metrics.items():
    panels=[wr.LinePlot(title=title,x='env_step',y=[metric],title_x='Agent decisions',aggregate=False,
                       smoothing_type='none',legend_template='${run:displayName}',max_runs_to_show=max_runs,
                       layout=wr.Layout(w=12,h=8),
                       line_colors={s['id']+':'+metric:s['color'] for s in specs},
                       line_titles={s['id']+':'+metric:s['label'] for s in specs}) for title,metric in group]
    sections.append(ws.Section(name='Dog post-tanh / '+name,is_open=True,panels=panels,
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
    work=ws.Workspace(entity=entity,project=project,name=state.get('workspace_name','Dog Run, Walk, Trot - post-tanh outer adapt'),
        auto_generate_panels=False,sections=sections,
        settings=ws.WorkspaceSettings(x_axis='env_step',max_runs=max_runs,tooltip_number_of_runs='all_runs',tooltip_color_run_names=True),
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
assert readback.settings.max_runs==max_runs and readback.settings.x_axis=='env_step'
assert readback.runset_settings.filters == f'Metric("ID") in {ids!r}'
owned={s.name:s for s in readback.sections if s.name.startswith('Dog post-tanh / ')}
expected_panels=1+sum(len(group) for group in metrics.values())
assert len(owned)==1+len(metrics) and sum(len(s.panels) for s in owned.values())==expected_panels
for title,items in metrics.items():
    for panel,(_,metric) in zip(owned['Dog post-tanh / '+title].panels,items):
        assert [getattr(v,'name',v) for v in panel.y]==[metric]
receipt.update(api_verified=True,panel_count=expected_panels)
temporary = HERE / f'workspace.{os.getpid()}.tmp'
temporary.write_text(json.dumps(receipt,indent=2)+'\n')
os.replace(temporary, receipt_path)
print(json.dumps(receipt,indent=2))
