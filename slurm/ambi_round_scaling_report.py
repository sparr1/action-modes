"""CPU-only, complete-panel round-scaling report and live W&B progress.

This does not evaluate policies. It trusts the maintained seed merger's trace
validation, verifies its sealed manifests/receipts, and independently validates
the episode panel and realized work. Partial cells never produce science curves.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import time

SCHEMA = "ambi-round-scaling-comparison-v1"
WORK = {"actor_updates": "inner_actor_optimizer_steps",
        "critic_updates": "inner_critic_optimizer_steps",
        "model_transitions": "inner_optimization_model_steps"}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value, message):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), message)
    return float(value)


def validate_campaign(campaign):
    require(bool(campaign.get("attempt_label", "").strip()), "An explicit attempt is required")
    cells = campaign.get("cells", [])
    require(len(cells) == 9, "Exactly nine J/controller cells are required")
    require(len({c["cell_id"] for c in cells}) == 9, "Duplicate cell IDs")
    controllers = sorted({c["controller_seed"] for c in cells})
    require(len(controllers) == 3, "Exactly three controller seeds are required")
    require({(c["rounds"], c["controller_seed"]) for c in cells}
            == {(j, s) for j in (1, 2, 4) for s in controllers}, "Incomplete J/controller grid")
    seeds = cells[0]["seeds"]
    require(seeds and len(seeds) == len(set(seeds))
            and all(type(s) is int for s in seeds), "Invalid environment seeds")
    for cell in cells:
        require(cell["seeds"] == seeds, "Cells must use identical environment seeds")
        require(Path(cell["bundle"]).is_absolute(), "Bundle paths must be absolute")
    return seeds, controllers


def sealed_manifest(directory):
    directory = Path(directory)
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    seal = read(directory / "seed-shard-checksums.json")
    require(seal.get("kind") == "ambi_episode_seed_shard"
            and seal.get("sha256") == digest({k: v for k, v in seal.items() if k != "sha256"}),
            "Invalid episode checksum seal")
    require(seal["files"].get("manifest.json") == hashlib.sha256(raw).hexdigest(),
            "Sealed manifest checksum mismatch")
    receipt_path = directory / "seed-shard-merge.json"
    receipt = read(receipt_path)
    require(seal["files"].get(receipt_path.name)
            == hashlib.sha256(receipt_path.read_bytes()).hexdigest(), "Merge receipt checksum mismatch")
    require(manifest.get("seed_shard_merge") == receipt
            and receipt.get("kind") == "ambi_episode_seed_merge", "Missing validated seed merge")
    require(manifest.get("status") == "complete", "Episode bundle is incomplete")
    return manifest, seal


def episodes(run, seeds, max_steps):
    result = run.get("result", {})
    require(run.get("status") == "complete" and run.get("kind") == "episodes", "Incomplete episode run")
    require(result.get("outer_state_unchanged") is True
            and type(result.get("outer_updates_before")) is int
            and result["outer_updates_before"] == result.get("outer_updates_after"), "Outer state changed")
    require(result.get("environment_seeds") == seeds, "Wrong environment seed panel")
    rows = result.get("episodes", [])
    require([r.get("seed") for r in rows] == seeds, "Missing or duplicated result episodes")
    declared = {r["seed"]: r for r in run.get("episodes", [])}
    require(len(declared) == len(seeds) and set(declared) == set(seeds), "Missing manifest episodes")
    for row in rows:
        finite(row["return"], "Nonfinite return")
        require(type(row.get("length")) is int and 0 < row["length"] <= max_steps
                and (row.get("terminated") is True or row.get("truncated") is True), "Incomplete episode")
        require(all(row.get(k) == declared[row["seed"]].get(k)
                    for k in ("return", "length", "solver_seed", "terminated", "truncated")),
                "Manifest/result episode mismatch")
        require(not row.get("nonfinite_model_metrics"), "Nonfinite model work")
    return rows


def load_cell(cell):
    manifest, seal = sealed_manifest(cell["bundle"])
    seeds = cell["seeds"]
    require(manifest["seed_shard_merge"]["expected_seeds"] == seeds, "Merge has wrong seeds")
    runs = [r for r in manifest["runs"] if r["selector"] == cell["selector"]]
    require(len(runs) == 1, "Missing or duplicate selected run")
    run = runs[0]
    result = run["result"]
    require(result.get("controller_seed") == cell["controller_seed"]
            and manifest["protocol"].get("controller_seed") == cell["controller_seed"], "Wrong controller seed")
    require(manifest["protocol"].get("action_rule") == "tanh_mean", "Expected mean-action control")
    params = result["alg_params"]
    expected = {"inner_rounds": cell["rounds"], "inner_rollout_horizon": 1,
                "inner_rollouts_per_round": 128, "inner_actor_updates_per_round": 4,
                "inner_critic_updates_per_round": 32, "inner_batch_size": 256,
                "inner_temperature": 0.0, "inner_actor_initialization": "prior",
                "inner_critic_initialization": "prior"}
    require(all(params.get(k) == v for k, v in expected.items()), "Wrong inner learner schedule")
    require(run.get("config_hash") == digest(run["config"]), "Run configuration checksum mismatch")
    rows = episodes(run, seeds, manifest["protocol"]["max_steps"])
    output = []
    for row in rows:
        metrics = row["model_metrics"]
        record = {"cell_id": cell["cell_id"], "rounds": cell["rounds"],
                  "controller_seed": cell["controller_seed"], "seed": row["seed"],
                  "solver_seed": row["solver_seed"], "return": row["return"], "length": row["length"],
                  "reused": bool(cell.get("reused")), "control_seconds": finite(row["control_seconds"], "Invalid control time"),
                  "probe_seconds": finite(row.get("togo_probe_seconds"), "Missing model-probe time"),
                  "probe_model_transitions": finite(row.get("togo_probe_model_steps"), "Missing probe work")}
        require(record["control_seconds"] >= 0 and record["probe_seconds"] >= 0, "Negative runtime")
        for name, key in WORK.items():
            value = finite(metrics.get(key), f"Missing realized {name}")
            require(value == cell["rounds"] * {"actor_updates": 4, "critic_updates": 32,
                                               "model_transitions": 128}[name], f"Unexpected realized {name}")
            record[name + "_per_decision"] = value
            record[name + "_per_episode"] = value * row["length"]
        record["control_seconds_per_decision"] = record["control_seconds"] / row["length"]
        output.append(record)
    prior_runs = [r for r in manifest["runs"] if r["selector"] == "initialization/prior"]
    require(len(prior_runs) == 1, "Comparison requires the verified inline prior run")
    prior = episodes(prior_runs[0], seeds, manifest["protocol"]["max_steps"])
    protocol = {k: v for k, v in manifest["protocol"].items() if k != "controller_seed"}
    science = {k: v for k, v in params.items() if k != "inner_rounds"}
    return {"cell": cell, "rows": output, "prior": prior,
            "compatibility": {"checkpoint": manifest["checkpoint"], "protocol": protocol,
                              "alg_params_except_rounds": science, "runtime": manifest["code"]["runtime"]},
            "identity": {"cell_id": cell["cell_id"], "bundle": cell["bundle"], "reused": bool(cell.get("reused")),
                         "manifest_sha256": seal["files"]["manifest.json"], "seal_sha256": seal["sha256"],
                         "checkpoint": manifest["checkpoint"], "code": manifest["code"],
                         "config_hash": run["config_hash"], "model_series": cell.get("model_series"),
                         "eval_run_dir": cell.get("eval_run_dir")}}


def summarize(values, resamples=2000):
    values = list(values)
    require(values, "Empty summary")
    rng = random.Random(20260912)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples))
    def quantile(p):
        pos = p * (len(means) - 1)
        low = int(pos)
        return means[low] + (means[min(low + 1, len(means) - 1)] - means[low]) * (pos - low)
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.,
            "ci95_low": quantile(.025), "ci95_high": quantile(.975), "episode_count": len(values)}


def build_report(campaign, loaded, resamples=2000):
    seeds, controllers = validate_campaign(campaign)
    require(set(loaded) == {c["cell_id"] for c in campaign["cells"]}, "No science from incomplete panels")
    values = [loaded[c["cell_id"]] for c in campaign["cells"]]
    require(all(v["compatibility"] == values[0]["compatibility"] for v in values), "Scientific configuration mismatch")
    priors = [{r["seed"]: r["return"] for r in v["prior"]} for v in values]
    require(all(p == priors[0] for p in priors), "Paired frozen-prior returns differ")
    rows = [copy.deepcopy(r) for value in values for r in value["rows"]]
    index = {(r["rounds"], r["controller_seed"], r["seed"]): r for r in rows}
    for row in rows:
        row["prior_return"] = priors[0][row["seed"]]
        row["gain_vs_prior"] = row["return"] - row["prior_return"]
        row["gain_vs_j1"] = row["return"] - index[1, row["controller_seed"], row["seed"]]["return"]
    by_episode, summaries = [], []
    numeric = ["return", "gain_vs_prior", "gain_vs_j1", "control_seconds", "control_seconds_per_decision",
               "probe_seconds", "probe_model_transitions", *[n + suffix for n in WORK
                   for suffix in ("_per_decision", "_per_episode")]]
    for j in (1, 2, 4):
        per_episode = []
        for seed in seeds:
            group = [index[j, c, seed] for c in controllers]
            # index points at the same dicts as rows, including the paired gains above.
            average = {"rounds": j, "seed": seed, "controller_repetitions": len(group),
                       **{k: statistics.fmean(r[k] for r in group) for k in numeric}}
            per_episode.append(average)
        by_episode.extend(per_episode)
        summaries.append({"rounds": j, "controller_repetitions": len(controllers),
                          "metrics": {k: summarize((r[k] for r in per_episode), resamples) for k in numeric},
                          "controller_mean_returns": {str(c): statistics.fmean(index[j, c, s]["return"] for s in seeds)
                                                      for c in controllers},
                          "timing_by_origin": {origin: summarize([statistics.fmean(r["control_seconds_per_decision"] for r in rows
                            if r["rounds"] == j and r["reused"] == reused and r["seed"] == seed) for seed in seeds], resamples)
                            for origin, reused in (("new", False), ("historical_reused", True))
                            if any(r["rounds"] == j and r["reused"] == reused for r in rows)}})
    return {"schema": SCHEMA, "status": "complete", "attempt_label": campaign["attempt_label"],
            "campaign_sha256": digest(campaign), "seeds": seeds, "controller_seeds": controllers,
            "aggregation": "Average controller repetitions within environment seed, then weight environment seeds equally",
            "interval": {"method": "paired environment-episode percentile bootstrap", "resamples": resamples,
                         "confidence": .95, "seed": 20260912},
            "timing_semantics": "Measured control time excludes model probes and warmup; historical reused timing remains identified",
            "validation": "Complete merger-sealed manifests and receipts, full paired episode panels and actual work; merger owns trace validation",
            "prior": {"summary": summarize(priors[0].values(), resamples),
                      "episodes": [{k: r[k] for k in ("seed", "solver_seed", "return", "length", "control_seconds")}
                                   for r in values[0]["prior"]]},
            "scientific_identity": values[0]["compatibility"],
            "identities": [v["identity"] for v in values], "summaries": summaries,
            "episode_averages": by_episode, "paired_rows": rows}


def render_html(report):
    data = json.dumps(report, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return """<!doctype html><meta charset="utf-8"><title>J1 / J2 / J4 control</title>
<style>body{font:16px system-ui;margin:32px auto;max-width:1150px;color:#162336;background:#f6f8fb}h1{font-size:28px}section{background:white;padding:24px;border:1px solid #dce1eb;border-radius:12px;margin:18px 0}svg{width:100%;height:350px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:right;border-bottom:1px solid #e5e8ee}th:first-child,td:first-child{text-align:left}small{color:#52627a}select{font:inherit;padding:8px}pre{white-space:pre-wrap;font-size:12px}</style>
<h1>Repeated mean-action control: J1 / J2 / J4</h1><p id="state"></p>
<section><p>Return against <select id="axis"><option value="actor_updates_per_decision">actor updates / decision</option><option value="critic_updates_per_decision">critic updates / decision</option><option value="model_transitions_per_decision">optimization model transitions / decision</option><option value="control_seconds_per_decision">measured control seconds / decision</option></select></p><svg id="chart" viewBox="0 0 1000 350"></svg><small>Points average controller repetitions within each environment seed, then episodes equally. Error bars are 95% episode-cluster bootstrap intervals. Prior reference is horizontal. Timing includes one explicitly identified historical J4 repetition; timing by origin is retained below.</small></section>
<section><h2>Paired complete-panel results</h2><div id="summary"></div></section>
<section><h2>Controller repetitions and timing provenance</h2><div id="detail"></div></section>
<section><h2>Raw paired returns</h2><div id="raw"></div></section>
<script id="report-data" type="application/json">""" + data + """</script><script>
const D=JSON.parse(document.getElementById('report-data').textContent),fmt=x=>Number(x).toFixed(3), S=D.summaries||[];
document.getElementById('state').textContent=`${D.attempt_label}: ${D.status}. ${D.seeds?.length||0} environment seeds, ${D.controller_seeds?.length||0} controller repetitions.`;
const range=m=>`${fmt(m.mean)} [${fmt(m.ci95_low)}, ${fmt(m.ci95_high)}]`;
const table=(heads,rows)=>'<table><tr>'+heads.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</table>';
if(D.status==='complete'){
document.getElementById('summary').innerHTML=table(['Rounds','Return [95% CI]','Gain vs prior','Gain vs J1','Episode SD'],S.map(r=>['J'+r.rounds,range(r.metrics.return),range(r.metrics.gain_vs_prior),range(r.metrics.gain_vs_j1),fmt(r.metrics.return.std)]));
document.getElementById('detail').innerHTML=table(['Rounds','Controller means','New seconds / decision','Historical seconds / decision'],S.map(r=>['J'+r.rounds,Object.entries(r.controller_mean_returns).map(([s,v])=>s+': '+fmt(v)).join('; '),r.timing_by_origin.new?fmt(r.timing_by_origin.new.mean):'—',r.timing_by_origin.historical_reused?fmt(r.timing_by_origin.historical_reused.mean):'—']));
document.getElementById('raw').innerHTML=table(['Rounds','Controller','Environment seed','Return','Prior gain','J1 gain','Reused'],D.paired_rows.map(r=>['J'+r.rounds,r.controller_seed,r.seed,fmt(r.return),fmt(r.gain_vs_prior),fmt(r.gain_vs_j1),r.reused]));
function draw(){let key=document.getElementById('axis').value, points=S.map(r=>({x:r.metrics[key].mean,y:r.metrics.return.mean,lo:r.metrics.return.ci95_low,hi:r.metrics.return.ci95_high,j:r.rounds}));let xmin=0,xmax=Math.max(...points.map(p=>p.x))*1.12,ymin=Math.min(D.prior.summary.mean,...points.map(p=>p.lo))-10,ymax=Math.max(D.prior.summary.mean,...points.map(p=>p.hi))+10,X=x=>65+880*(x-xmin)/(xmax-xmin),Y=y=>290-250*(y-ymin)/(ymax-ymin);let a=[];for(let i=0;i<5;i++){let y=ymin+i*(ymax-ymin)/4;a.push(`<line x1="65" x2="945" y1="${Y(y)}" y2="${Y(y)}" stroke="#e4e8ef"/><text x="8" y="${Y(y)+5}" font-size="12">${y.toFixed(1)}</text>`)}let py=Y(D.prior.summary.mean);a.push(`<line x1="65" x2="945" y1="${py}" y2="${py}" stroke="#8a96a8" stroke-dasharray="5 4"/><text x="800" y="${py-7}" font-size="12">Frozen prior</text>`);a.push(`<polyline points="${points.map(p=>X(p.x)+','+Y(p.y)).join(' ')}" fill="none" stroke="#3c64ca" stroke-width="2"/>`);for(const p of points)a.push(`<line x1="${X(p.x)}" x2="${X(p.x)}" y1="${Y(p.lo)}" y2="${Y(p.hi)}" stroke="#3c64ca"/><circle cx="${X(p.x)}" cy="${Y(p.y)}" r="5" fill="#3c64ca"/><text x="${X(p.x)+8}" y="${Y(p.y)-8}">J${p.j}</text><text x="${X(p.x)-20}" y="320" font-size="12">${p.x.toFixed(key.includes('seconds')?3:0)}</text>`);document.getElementById('chart').innerHTML=a.join('')};document.getElementById('axis').onchange=draw;draw();
}else{document.getElementById('summary').textContent='Scientific curves are withheld until every cell is complete.';document.getElementById('detail').textContent=JSON.stringify(D.missing||[],null,2)}
</script>"""


def job_state(job_ids, run=subprocess.run):
    if not job_ids:
        return {"finished": False, "error": "No scheduler jobs supplied"}
    command = ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T"]
    queued = run(command, check=True, text=True, capture_output=True, timeout=30).stdout.strip()
    if queued:
        return {"finished": False, "queue": queued.splitlines()}
    output = run(["sacct", "-n", "-X", "-P", "-j", ",".join(job_ids), "--format=JobIDRaw,State,ExitCode"],
                 check=True, text=True, capture_output=True, timeout=30).stdout
    rows = [r.split("|") for r in output.splitlines() if r.strip()]
    rows = [r for r in rows if len(r) >= 3 and any(r[0] == j or r[0].startswith(j + "_") for j in job_ids)]
    seen = {j for j in job_ids if any(r[0] == j or r[0].startswith(j + "_") for r in rows)}
    return {"finished": seen == set(job_ids) and bool(rows)
            and all(r[1].split()[0].rstrip("+") in TERMINAL for r in rows), "accounting": rows,
            "unknown_jobs": sorted(set(job_ids) - seen)}


def inspect(campaign, loaded):
    missing = []
    for cell in campaign["cells"]:
        if cell["cell_id"] in loaded:
            continue
        try:
            loaded[cell["cell_id"]] = load_cell(cell)
        except (OSError, ValueError, KeyError, TypeError) as error:
            missing.append({"cell_id": cell["cell_id"], "reason": str(error)})
    receipts = campaign.get("worker_receipts", [])
    complete_workers = 0
    for path in receipts:
        try:
            complete_workers += read(path).get("status") == "complete"
        except (OSError, ValueError, TypeError):
            pass
    return {"complete_cells": len(loaded), "expected_cells": len(campaign["cells"]),
            "complete_workers": complete_workers, "expected_workers": len(receipts), "missing": missing}


def publish_science(run, wandb, report, output):
    fingerprint = digest(report)
    if run.summary.get("comparison/report_sha256") == fingerprint:
        return
    for axis in (*[n + "_per_decision" for n in WORK], "control_seconds_per_decision"):
        run.define_metric("compute/" + axis)
        for metric in ("return", "gain_vs_prior", "gain_vs_j1"):
            run.define_metric(f"comparison_vs_{axis}/{metric}/*", step_metric="compute/" + axis)
    for row in report["summaries"]:
        metrics = {"compute/rounds": row["rounds"]}
        for axis in (*[n + "_per_decision" for n in WORK], "control_seconds_per_decision"):
            metrics["compute/" + axis] = row["metrics"][axis]["mean"]
            for name in ("return", "gain_vs_prior", "gain_vs_j1"):
                for statistic in ("mean", "std", "ci95_low", "ci95_high"):
                    metrics[f"comparison_vs_{axis}/{name}/{statistic}"] = row["metrics"][name][statistic]
        run.log(metrics)
    run.log({"comparison/report": wandb.Html(str(output / "report.html"), inject=False)})
    artifact = wandb.Artifact("round-scaling-" + digest(report["attempt_label"])[:16], type="round-scaling-comparison",
                              metadata={"schema": SCHEMA, "status": "complete", "report_sha256": fingerprint})
    for name in ("report.json", "report.html", "campaign.json"):
        artifact.add_file(str(output / name), name=name)
    run.log_artifact(artifact)
    run.summary["comparison/report_sha256"] = fingerprint
    run.summary["comparison/prior_return"] = report["prior"]["summary"]["mean"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--owner", default="oscar-rgao48")
    parser.add_argument("--compute-jobs", nargs="*", default=[])
    parser.add_argument("--merge-jobs", nargs="*", default=[])
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--mode", choices=("online", "disabled"), default="disabled")
    parser.add_argument("--output")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--max-watch-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    campaign = read(args.campaign)
    seeds, controllers = validate_campaign(campaign)
    require(args.poll_seconds > 0 and args.max_watch_seconds > 0, "Invalid watch limits")
    require(not args.watch or args.compute_jobs or args.merge_jobs, "Watching requires scheduler job IDs")
    output = Path(args.output) if args.output else Path(args.campaign).parent / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "campaign.json").exists():
        require(read(output / "campaign.json") == campaign, "Output belongs to a different campaign")
    write(output / "campaign.json", campaign)
    run = wandb = None
    loaded, started, monitoring_errors = {}, time.monotonic(), 0
    progress = {"schema": SCHEMA, "attempt_label": campaign["attempt_label"], "status": "starting",
                "seeds": seeds, "controller_seeds": controllers}
    old_handlers = {}
    def interrupted(signum, frame):
        raise RuntimeError(f"Publisher received signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, interrupted)
    try:
        if args.mode == "online":
            require(args.wandb_run_id, "Reserve an explicit W&B run ID")
            require(args.owner == "oscar-rgao48", "Oscar owns this publisher")
            import wandb
            run = wandb.init(entity=campaign.get("wandb_entity", "rwgao_b-brown-university"),
                             project=campaign.get("wandb_project", "ambi-inner-bench"),
                             id=args.wandb_run_id, resume="allow", mode="online",
                             name="J1/J2/J4 repeated mean control | " + campaign["attempt_label"],
                             job_type="round-scaling-comparison", tags=["round-scaling", "live-progress", "complete-panel"],
                             config={"comparison_schema": SCHEMA, "attempt_label": campaign["attempt_label"],
                                     "campaign_sha256": digest(campaign), "owner": args.owner,
                                     "science_publication": "Only after all nine cells complete"})
            run.define_metric("progress/elapsed_seconds")
            run.define_metric("progress/*", step_metric="progress/elapsed_seconds")
            run.summary["comparison/status"] = "running"
            run.log({"progress/elapsed_seconds": 0., "progress/complete_cells": 0,
                     "progress/expected_cells": 9, "progress/complete_workers": 0,
                     "progress/expected_workers": len(campaign.get("worker_receipts", []))})
        while True:
            progress.update(inspect(campaign, loaded))
            progress["elapsed_seconds"] = time.monotonic() - started
            progress["status"] = "running"
            if run:
                run.log({"progress/" + key: progress[key] for key in
                         ("elapsed_seconds", "complete_cells", "expected_cells", "complete_workers", "expected_workers")})
            if not progress["missing"]:
                report = build_report(campaign, loaded)
                write(output / "report.json", report)
                (output / "report.html").write_text(render_html(report))
                if run:
                    publish_science(run, wandb, report, output)
                progress["status"] = "complete"
                progress["stop_reason"] = "All nine complete panels published"
                break
            if not args.watch:
                progress["status"] = "incomplete"
                progress["stop_reason"] = "Snapshot inspection requested without watching"
                break
            try:
                progress["scheduler"] = job_state(args.compute_jobs + args.merge_jobs)
                monitoring_errors = 0
            except (OSError, subprocess.SubprocessError) as error:
                monitoring_errors += 1
                progress["scheduler"] = {"finished": False, "error": str(error), "consecutive_errors": monitoring_errors}
            if progress["scheduler"]["finished"] or monitoring_errors >= 3 or progress["elapsed_seconds"] >= args.max_watch_seconds:
                progress["status"] = "incomplete"
                progress["stop_reason"] = ("All compute and merge jobs terminated" if progress["scheduler"]["finished"]
                    else "Scheduler inspection failed three times" if monitoring_errors >= 3 else "Watch deadline exceeded")
                break
            write(output / "progress.json", progress)
            time.sleep(min(args.poll_seconds, 60))
        if progress["status"] != "complete":
            write(output / "report.json", progress)
            (output / "report.html").write_text(render_html(progress))
        write(output / "progress.json", progress)
        if run:
            run.summary["comparison/status"] = progress["status"]
            run.summary["comparison/missing_cells"] = progress["missing"]
            run.summary["comparison/stop_reason"] = progress.get("stop_reason")
        print(json.dumps({"status": progress["status"], "complete_cells": progress["complete_cells"],
                          "output": str(output), "missing": progress["missing"]}))
        return 0 if progress["status"] == "complete" else 2
    except BaseException as error:
        progress.update(status="failed", error=str(error) or type(error).__name__)
        write(output / "progress.json", progress)
        write(output / "failure.json", progress)
        if run:
            run.summary["comparison/status"] = "failed"
            run.summary["comparison/error"] = str(error)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if run:
            run.finish(exit_code=0 if progress["status"] == "complete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
