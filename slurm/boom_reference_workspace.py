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
run_id = state["run_id"]
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
                              ("Completed updates", "progress/updates"),
                              ("Update duration", "diagnostic/update_seconds")],
}
md = f"""**Released BOOM — DMC Humanoid Walk — threshold reference.**

{status} Slurm job `{state.get('job', 'pending submission')}`.

One seed (55), original S > 2 gate and released lambda=0.42; original entropy,
Gaussian penalty, replay, planner, and training equations. Original budget:
2,001,000 decisions (two raw control steps per decision). State observations,
reward summation, 500-decision episodes. Native compilation, vectorized critic
ensemble and TF32 are enabled. Upstream default evaluation: ten planner and
ten prior-only episodes every 50,000 decisions. This is a diagnostic seed.

Every update is retained locally; displayed diagnostics summarize 100 updates
with minima and maxima. Checkpoints immediately before/after the first active
actor update and every 25k decisions support follow-up inspection. The entropy
diagnostic removes the extra upstream Gaussian factor, but retains the
upstream Jacobian floor; the learning objective is unchanged.

Upstream `b3b2f09c03533c15b804ced9abc6767e73ff8e1e`.
Observer `{state['commit']}`. Outputs: `{root}`.
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
else:
    work = ws.Workspace(entity="rwgao_b-brown-university", project="ambi",
        name="Released BOOM Humanoid Walk - S threshold reference", auto_generate_panels=False,
        sections=sections, settings=ws.WorkspaceSettings(x_axis="env_step", max_runs=1),
        runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID").isin([run_id])]))
work.save()
check = ws.Workspace.from_url(work.url)
assert check.runset_settings.filters == f'Metric("ID") in {[run_id]!r}'
count = sum(len(s.panels) for s in check.sections if s.name.startswith("BOOM reference / "))
assert count == 1 + sum(len(rows) for rows in groups.values())
receipt_path.write_text(json.dumps({"url": work.url, "run_id": run_id,
                                    "panel_count": count, "api_verified": True}, indent=2))
print(receipt_path.read_text())

