"""Install and verify the dedicated upstream BOOM reference panels."""
import json
from pathlib import Path
import sys
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr
from wandb_workspaces import expr

root = Path(sys.argv[1])
state = json.loads((root / "submission.json").read_text())
entropy_ablation = state.get("entropy") == "ambi-posttanh"
workspace_name = ("BOOM: released versus AMBI post-tanh entropy" if entropy_ablation
                  else "Released BOOM and TD-M(PC)2 Humanoid Walk - threshold references")
run_id = state["run_id"]
references = state.get("references", [{"method": "BOOM", "run_id": run_id, "root": str(root)}])
run_ids = [r["run_id"] for r in references]
api = wandb.Api()
try:
    run = api.run(f"rwgao_b-brown-university/ambi/{run_id}")
    status = f"{run.state}; uploaded decision {run.summary.get('env_step', 'pending')}"
except wandb.errors.CommError as e:
    if "Could not find run" not in str(e) and "not found" not in str(e).lower():
        raise
    status = "Pending startup; the charts will populate with training."
groups = {
    "Returns": [("Planner training return", "train/episode_reward"),
                ("Planner evaluation return", "eval/episode_reward"),
                ("Prior-only evaluation return", "eval/episode_reward_pi")],
    "Threshold": [("S: mean, minimum, maximum", "diagnostic/pi_scale", "diagnostic/pi_scale_min", "diagnostic/pi_scale_max"),
                  ("Fraction of updates with S > 2", "diagnostic/gate_active"),
                  ("First crossing", "event/crossing_scale")],
    "Actor update": [("RL and weighted alignment objectives", "diagnostic/pi_q_loss", "diagnostic/weighted_alignment_loss"),
                     ("Alignment spikes", "diagnostic/weighted_alignment_loss_max"),
                     ("Actor gradient mean and maximum", "diagnostic/actor_grad_norm", "diagnostic/actor_grad_norm_max"),
                     ("Fraction clipped", "diagnostic/actor_clip_active")],
    "Policy": [("Joint post-tanh entropy (upstream floored Jacobian)", "diagnostic/post_tanh_entropy_floored"),
               ("Gaussian entropy", "diagnostic/pre_tanh_entropy"),
               ("Action saturation", "diagnostic/action_saturation", "diagnostic/mean_saturation"),
               ("Actor standard deviation", "diagnostic/actor_std_mean", "diagnostic/actor_std_min")],
    "Learning and progress": [("Model and critic fitting", "diagnostic/consistency_loss", "diagnostic/value_loss"),
                              ("TD-M(PC)^2 Monte Carlo versus predicted value", "eval/mc_value", "eval/q_value"),
                              ("Completed updates", "progress/updates"),
                              ("Update duration", "diagnostic/update_seconds")],
}
if entropy_ablation:
    groups["Policy"].extend([
        ("Exact post-tanh entropy (ablation)", "diagnostic/post_tanh_entropy_exact"),
        ("Alignment standard deviation at floor (ablation)", "diagnostic/alignment_std_floor_fraction"),
    ])
reference_status = []
for ref in references:
    sub = json.loads((Path(ref["root"]) / "submission.json").read_text())
    reference_status.append(f"| {ref['method']} | {sub.get('job', 'pending validation')} | {sub['commit']} |")
table = "\n".join(reference_status)
md = f"""**Released BOOM and TD-M(PC)² — DMC Humanoid Walk — threshold references.**

{status} Slurm job `{state.get('job', 'pending submission')}`.

One seed (55) each and original S > 2 gates. BOOM uses released lambda=0.42;
TD-M(PC)² uses beta=1. Original entropy, Gaussian penalties, replay, planners,
and training equations. Budgets are 400,000 decisions each
(two raw control steps per decision). State observations, summed rewards,
500-decision episodes. Both capture the MPPI planner iterations and model forward/backward pass with CUDA graphs. BOOM also retains
native module compilation and TF32. Both vectorize critic heads and preserve
their native training equations and precision settings.
Original evaluations every 50k: BOOM ten planner/prior episodes; TD one of each
plus 100 Monte Carlo prior episodes for value estimation. Videos are disabled.

| Reference | Slurm job | Observer commit |
|---|---|---|
{table}

Every update is retained locally; displayed diagnostics summarize 100 updates
with minima and maxima. Checkpoints immediately before/after the first active
actor update and every 25k decisions support follow-up inspection. The entropy
diagnostic removes the extra upstream Gaussian factor, but retains the
upstream Jacobian floor; the learning objective is unchanged.

BOOM upstream `b3b2f09c03533c15b804ced9abc6767e73ff8e1e`.
TD-M(PC)² upstream `d1c2632c36effd2f7b661bfe5f822a3db8054d40`.
Observer `{state['commit']}`. Outputs: `{root}`.
"""
if entropy_ablation:
    md = f"""**BOOM entropy ablation — DMC Humanoid Walk — seed 55.**

{status} New Slurm job `{state.get('job', 'pending validation')}`.

The new run changes only actor entropy to AMBI's corrected joint Gaussian
log density and exact stable tanh Jacobian, with alpha=0.0021. The existing
released BOOM control uses its original entropy helper and alpha=0.0001.
This preserves Gaussian entropy pressure while changing the tanh correction.
The ablation is not an unmodified official BOOM run.

Both retain BOOM's native alignment Gaussian helper (including its extra
dimension factor), detached standard deviation with floor 0.05, Q weighting,
lambda=0.42, strict S>2 gate, MPPI, replay, optimizers, and clip threshold 20.
Both use 400,000 decisions, action repeat 2, H3, state observations, seed 55,
model/planner CUDA graphs, TF32, checkpoints every 25k, and ten planner/prior
evaluation episodes every 50k decisions. The control is reused, not rerun.

| Run | Slurm job | Source commit |
|---|---|---|
{table}

Every update is retained locally. Charts show 100-update means and extrema.
The floored post-tanh entropy diagnostic is common to both runs. Exact
post-tanh entropy and the fraction of standard deviations below the alignment
floor are additional diagnostics for the ablation. Pre/post first-gate
checkpoints are retained. BOOM upstream:
`b3b2f09c03533c15b804ced9abc6767e73ff8e1e`.
Outputs: `{root}`.
"""
sections = [ws.Section(name="BOOM reference / Protocol and progress", is_open=True,
                       panels=[wr.MarkdownPanel(markdown=md, layout=wr.Layout(w=24, h=12))])]
for name, metrics in groups.items():
    sections.append(ws.Section(name="BOOM reference / " + name, is_open=True,
        panels=[wr.LinePlot(title=row[0], x="env_step", y=list(row[1:]), aggregate=False,
                            smoothing_type="none", layout=wr.Layout(w=12, h=8)) for row in metrics],
        layout_settings=ws.SectionLayoutSettings(columns=2, rows=2)))
receipt_path = root / "workspace.json"
if receipt_path.exists():
    work = ws.Workspace.from_url(json.loads(receipt_path.read_text())["url"])
    replacement = {s.name: s for s in sections}
    work.sections = [replacement.pop(s.name, s) for s in work.sections] + list(replacement.values())
    work.name = workspace_name
    work.settings.max_runs = len(run_ids)
    work.runset_settings.filters = [expr.Metric("ID").isin(run_ids)]
else:
    work = ws.Workspace(entity="rwgao_b-brown-university", project="ambi",
        name=workspace_name, auto_generate_panels=False,
        sections=sections, settings=ws.WorkspaceSettings(x_axis="env_step", max_runs=len(run_ids)),
        runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID").isin(run_ids)]))
work.save()
check = ws.Workspace.from_url(work.url)
assert check.runset_settings.filters == f'Metric("ID") in {run_ids!r}'
count = sum(len(s.panels) for s in check.sections if s.name.startswith("BOOM reference / "))
assert count == 1 + sum(len(rows) for rows in groups.values())
receipt_path.write_text(json.dumps({"url": work.url, "run_id": run_id,
                                    "panel_count": count, "api_verified": True}, indent=2))
print(receipt_path.read_text())

