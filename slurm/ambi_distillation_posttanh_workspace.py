"""Idempotent W&B panels for the authorized post-tanh distillation comparison."""
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
existing = {r.id:r for r in api.runs(entity+'/'+project, filters={'name':{'$in':ids}}, per_page=8)}
rows=[]
for spec in specs:
    r=existing.get(spec['id'])
    summary={} if r is None else dict(r.summary)
    pending_state = ("submitted / awaiting training start" if spec.get('job') else
                     f"pending GPU validation / gate {spec.get('gate_job', 'not submitted')}")
    rows.append({**spec,'state':pending_state if r is None else r.state,
                 'decisions':summary.get('env_step','pending'), 'job':spec.get('job','pending submission')})
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
metrics['Teacher coverage and freshness']=[
    ('Valid teacher fraction','train/distillation_valid_fraction'),
    ('Teacher age mean (outer updates)','train/distillation_teacher_age_mean'),
    ('Teacher age maximum','train/distillation_teacher_age_max'),
    ('Skipped prior updates','train/distillation_actor_skipped')]
metrics['Distillation objective']=[
    ('Reverse KL per action dimension','train/distillation_kl'),
    ('Normalized reverse KL','train/distillation_normalized_kl'),
    ('Independent KL scale','train/distillation_scale'),
    ('Actual actor updates','train/distillation_actor_updates')]
metrics['Reanalysis work']=[
    ('Refresh seconds','train/distillation_reanalyze_seconds'),
    ('Refreshed roots','train/distillation_reanalyze_roots'),
    ('Imagined model steps','train/distillation_reanalyze_model_steps'),
    ('CUDA graph reanalysis','train/distillation_reanalyze_cuda_graph')]
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
status='\n'.join(f"| {r['label'].replace('|','/')} | {r['job']} | {r['state']} | {r['decisions']} |" for r in rows)
settings = state['settings']
md=f'''**Post-tanh outer-adapt K1: lazy prior distillation — Oscar L40S.**

One fresh Humanoid Walk run, seed 55, matched to the original post-tanh K1
control: H3/J8/N128/C16/A4/B256, reward-only five-head critic, finite-horizon
prior bootstrapping, fixed joint squashed-entropy coefficient 0.0021, and
fresh action-local inner adaptation. Every collection solve initializes its
private Q scale from the outer scale and updates it per actor update.

The new prior minimizes normalized reverse KL to cached adapted Gaussian
teachers plus the existing entropy term. Q learning and its scale are separate.
Refresh every **{settings['reanalyze_interval']} outer updates**, using up to
**{settings['reanalyze_batch_size']} sequences**. Root width
{settings['reanalyze_parallel_roots']}, workers {settings['reanalyze_batched_workers']},
first-layer fusion {settings['reanalyze_batched_first_layer_fusion']}.
Collection and teacher CUDA graphs retain the complete inner update budget.

Target **{settings['total_steps']:,} decisions** with portable checkpoints
retained every 25k. This deliberately shorter run fits the user's 48-hour
completion budget; compare returns at matched decision counts. A longer
refresh interval reduces teacher freshness; its learning-performance tradeoff
has not been established for this task. Geometry diagnostics remain enabled
at their original cadence. Gate/benchmark executions have no W&B runs.

Source `{state['commit']}`. Outputs `{state['output']}`.
Estimated production duration: {state.get('estimated_hours', 'pending gate')} hours.

| Arm | Job | State at {timestamp} | Uploaded decisions |
|---|---|---|---|
{status}

This table is a timestamped snapshot; charts update with training. Pending
training has no fabricated data. The existing Q-maximization control does not
have distillation metrics. No smoothing is applied.
'''

sections=[ws.Section(name='Lazy distillation / Protocol and progress',is_open=True,
                    panels=[wr.MarkdownPanel(markdown=md,layout=wr.Layout(w=24,h=15))])]
for name,group in metrics.items():
    panels=[wr.LinePlot(title=title,x='env_step',y=[metric],title_x='Agent decisions',aggregate=False,
                       smoothing_type='none',legend_template='${run:displayName}',max_runs_to_show=8,
                       layout=wr.Layout(w=12,h=8),
                       line_colors={s['id']+':'+metric:s['color'] for s in specs},
                       line_titles={s['id']+':'+metric:s['label'] for s in specs}) for title,metric in group]
    sections.append(ws.Section(name='Lazy distillation / '+name,is_open=True,panels=panels,
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
    work=ws.Workspace(entity=entity,project=project,name='Post-tanh K1 - lazy prior distillation - Oscar',
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
owned={s.name:s for s in readback.sections if s.name.startswith('Lazy distillation / ')}
expected_panels=1+sum(len(group) for group in metrics.values())
assert len(owned)==1+len(metrics) and sum(len(s.panels) for s in owned.values())==expected_panels
for title,items in metrics.items():
    for panel,(_,metric) in zip(owned['Lazy distillation / '+title].panels,items):
        assert [getattr(v,'name',v) for v in panel.y]==[metric]
receipt.update(api_verified=True,panel_count=expected_panels)
temporary = HERE / f'workspace.{os.getpid()}.tmp'
temporary.write_text(json.dumps(receipt,indent=2)+'\n')
os.replace(temporary, receipt_path)
print(json.dumps(receipt,indent=2))
