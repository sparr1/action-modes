"""Strict complete-panel critic-update sweep, with streamed training traces.

This CPU-only reporter never runs a policy. It publishes one comparison after
21 production merge receipts and their sealed episode/model-series exports
validate. C32 is an exact historical reference; C0 still performs actor learning.
"""
from __future__ import annotations

import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from slurm.ambi_round_scaling_report import (WORK, digest, episodes, finite, job_state,
                                            read, require, sealed_manifest, summarize, write)
from slurm.ambi_entropy_episode_report import source_identity, validate_model_series, _summaries

SCHEMA = "ambi-critic-steps-comparison-v1"
CRITIC_UPDATES = [0, 1, 4, 8, 16, 32, 64]
SEEDS = list(range(101, 121))
CONTROLLERS = [55, 56, 57]
CHECKPOINT = "909e5c1d125aecc952e0544d802b4b946ae5089909c7a55885f272c32ed3e12f"
AXES = ["critic_updates_per_decision", "optimizer_updates_per_decision", "control_seconds_per_decision"]
TRAINING_METRICS = ("critic_loss", "td_error_abs_mean", "critic_grad_norm", "q_mean",
                    "q_abs_mean", "q_target_mean", "q_target_clip_fraction")


def validate_campaign(campaign):
    require(bool(campaign.get("attempt_label", "").strip()), "Explicit attempt required")
    require(campaign.get("critic_updates") == CRITIC_UPDATES and campaign.get("rounds") == 1,
            "Wrong critic-count sweep")
    require(campaign.get("seeds") == SEEDS and campaign.get("controller_seeds") == CONTROLLERS,
            "Expected twenty paired seeds and three controller seeds")
    cells = campaign.get("cells", [])
    require(len(cells) == 21 and len({c["cell_id"] for c in cells}) == 21, "Expected 21 unique cells")
    require({(c["critic_updates"], c["controller_seed"]) for c in cells}
            == {(n, s) for n in CRITIC_UPDATES for s in CONTROLLERS}, "Incomplete critic/controller grid")
    for c in cells:
        require(c["seeds"] == SEEDS and c["rounds"] == 1 and Path(c["bundle"]).is_absolute(),
                "Wrong seeds, rounds or relative bundle")
        require(bool(c.get("reused")) == (c["critic_updates"] == 32), "Only all three historical C32 cells are reused")
        if c["reused"]:
            require(all(isinstance(c.get(k), str) and len(c[k]) == 64 for k in
                        ("bundle_manifest_sha256", "bundle_seal_sha256")), "Historical bundle hashes required")
    return SEEDS, CONTROLLERS


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def training_trace_summaries(cell, run, seal, episode_rows):
    """Stream one gzip at a time, retaining only per-episode/step accumulators.

    Bitmasks validate every optimizer coordinate without retaining millions of
    update rows. Original gzip traces remain in the checksum-sealed source;
    their paths/hashes plus all per-episode step summaries enter the report.
    """
    root = Path(cell["bundle"]).resolve()
    count = cell["critic_updates"]
    expected = {(f"seed-{r['seed']}", decision) for r in episode_rows for decision in range(r["length"])}
    coordinates, accumulated, sources = {}, {}, []
    paths = run.get("trace_files", [])
    require(paths and len(paths) == len(set(paths)), "Missing or duplicate training trace files")
    for relative in paths:
        path = root / relative
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Unsafe training trace path")
        checksum = file_digest(path)
        require(seal["files"].get(relative) == checksum, "Training trace checksum mismatch")
        sources.append({"path": str(path), "relative_path": relative, "sha256": checksum,
                        "compressed_bytes": path.stat().st_size})
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("phase") != "update":
                    continue
                key = (row["episode_id"], row["decision_index"])
                require(key in expected and row.get("round_index") == 1, "Unexpected optimizer trace root")
                require(row.get("measurement") == "pre_update_minibatch" and not row.get("nonfinite"),
                        "Wrong training measurement phase or nonfinite trace")
                critic, actor = row.get("updated_critic"), row.get("updated_actor")
                require(type(critic) is bool and type(actor) is bool and critic != actor
                        and row.get("updated_temperature") is False, "Expected critic-first or actor-only update")
                masks = coordinates.setdefault(key, [0, 0])
                if critic:
                    step = row["critic_updates"]
                    require(type(step) is int and 1 <= step <= count and row["actor_updates"] == 0,
                            "Wrong critic update counter")
                    require(masks[0] == (1 << (step - 1)) - 1 and masks[1] == 0,
                            "Duplicate or out-of-order critic update")
                    masks[0] |= 1 << (step - 1)
                    values = accumulated.setdefault((key[0], step), {})
                    for name in TRAINING_METRICS:
                        value = finite(row["metrics"].get(name), "Missing/nonfinite critic training metric")
                        stat = values.setdefault(name, {"count": 0, "mean": 0., "m2": 0., "min": value, "max": value})
                        stat["count"] += 1
                        delta = value - stat["mean"]
                        stat["mean"] += delta / stat["count"]
                        stat["m2"] += delta * (value - stat["mean"])
                        stat["min"], stat["max"] = min(stat["min"], value), max(stat["max"], value)
                else:
                    step = row["actor_updates"]
                    require(type(step) is int and 1 <= step <= 4 and row["critic_updates"] == count,
                            "Wrong actor update counter")
                    require(masks[0] == (1 << count) - 1 and masks[1] == (1 << (step - 1)) - 1,
                            "Duplicate or out-of-order actor update")
                    masks[1] |= 1 << (step - 1)
    require(set(coordinates) == expected and all(masks == [(1 << count) - 1, 15] for masks in coordinates.values()),
            "Incomplete critic/actor trace coverage")
    summaries = {}
    for episode in episode_rows:
        episode_id = f"seed-{episode['seed']}"
        points = []
        for step in range(1, count + 1):
            raw = accumulated[episode_id, step]
            require(all(s["count"] == episode["length"] for s in raw.values()), "Incomplete critic metric coverage")
            points.append({"critic_updates": step, "critic_update_index": step, "measurement": "pre_update_minibatch", "metrics": {name: {
                "count": s["count"], "mean": s["mean"], "std": math.sqrt(max(0., s["m2"]) / s["count"]),
                "min": s["min"], "max": s["max"]} for name, s in raw.items()}})
        summaries[episode["seed"]] = points
    return summaries, sources


def load_cell(cell):
    manifest, seal = sealed_manifest(cell["bundle"])
    if cell.get("reused"):
        require(seal["files"]["manifest.json"] == cell["bundle_manifest_sha256"]
                and file_digest(Path(cell["bundle"]) / "seed-shard-checksums.json") == cell["bundle_seal_sha256"],
                "Historical bundle identity mismatch")
    require(manifest["seed_shard_merge"]["expected_seeds"] == cell["seeds"], "Merge has wrong seeds")
    checkpoint, protocol, code = manifest["checkpoint"], manifest["protocol"], manifest["code"]
    require(checkpoint["sha256"] == CHECKPOINT and checkpoint.get("source_run", "").endswith("/mey3rxj8")
            and checkpoint["metadata"]["checkpoint"]["step"] == 200000, "Wrong backbone checkpoint")
    require(protocol.get("controller_seed") == cell["controller_seed"]
            and protocol.get("action_rule") == "tanh_mean" and protocol.get("max_steps") == 500,
            "Wrong controller or real execution protocol")
    runs = [r for r in manifest["runs"] if r["selector"] == cell["selector"]]
    require(len(runs) == 1, "Missing or duplicate selected run")
    run, count = runs[0], cell["critic_updates"]
    result, config = run["result"], run["config"]
    require(result.get("controller_seed") == cell["controller_seed"], "Wrong result controller seed")
    require(run.get("config_hash") == digest(config), "Run configuration checksum mismatch")
    params = result["alg_params"]
    expected = {"inner_rounds": 1, "inner_rollout_horizon": 1, "inner_rollouts_per_round": 128,
                "inner_actor_updates_per_round": 4, "inner_critic_updates_per_round": count,
                "inner_batch_size": 256, "inner_temperature": 0.0, "inner_actor_entropy_mode": "tdmpc2_scaled",
                "inner_actor_initialization": "prior", "inner_critic_initialization": "prior",
                "inner_temperature_mode": "fixed", "inner_temperature_initialization": "fixed",
                "inner_execution_action": "mean", "inner_behavior_action": "policy_sample",
                "inner_finite_horizon": True, "inner_sac_critic_target": "reward_only"}
    require(all(params.get(k) == v and config["alg_params"].get(k) == v for k, v in expected.items()),
            "Wrong critic-count learner schedule")
    require(not result.get("nonfinite_trace_metrics") and not result.get("nonfinite_model_metrics"),
            "Nonfinite diagnostic metrics")
    probe = result.get("togo_return_probe")
    require(probe and probe.get("rollouts") == 32 and probe.get("horizon") == 1
            and probe.get("tail_actor") == "outer" and probe.get("tail_critic") == "outer_online"
            and probe.get("tail_q_reduction") == "mean_pair" and probe.get("entropy_bonus") is False,
            "Wrong model probe protocol")
    episode_rows = episodes(run, cell["seeds"], 500)
    training, trace_sources = training_trace_summaries(cell, run, seal, episode_rows)
    output = []
    for row in episode_rows:
        metrics = {k: finite(v, "Nonfinite episode diagnostic") for k, v in row["model_metrics"].items()}
        record = {"cell_id": cell["cell_id"], "critic_updates": count, "rounds": 1,
                  "controller_seed": cell["controller_seed"], "seed": row["seed"],
                  "solver_seed": row["solver_seed"], "return": row["return"], "length": row["length"],
                  "terminated": row["terminated"], "truncated": row["truncated"],
                  "reused": bool(cell.get("reused")), "model_metrics": metrics,
                  "control_seconds": finite(row["control_seconds"], "Invalid control time"),
                  "probe_seconds": finite(row.get("togo_probe_seconds"), "Missing model probe time"),
                  "probe_model_transitions": finite(row.get("togo_probe_model_steps"), "Missing probe work"),
                  "critic_training_steps": training[row["seed"]]}
        require(record["control_seconds"] >= 0 and record["probe_seconds"] >= 0, "Negative runtime")
        for name, key in WORK.items():
            value = finite(metrics.get(key), f"Missing realized {name}")
            require(value == {"actor_updates": 4, "critic_updates": count, "model_transitions": 128}[name],
                    f"Unexpected realized {name}")
            record[name + "_per_decision"], record[name + "_per_episode"] = value, value * row["length"]
        record["optimizer_updates_per_decision"] = count + 4
        record["optimizer_updates_per_episode"] = (count + 4) * row["length"]
        record["control_seconds_per_decision"] = record["control_seconds"] / row["length"]
        points = row.get("togo_round_summaries", [])
        require([r["round_index"] for r in points] == [0, 1], "Incomplete round probes")
        for point in points:
            r = point["round_index"]
            require(point["actor_updates"] == 4 * r and point["critic_updates"] == count * r,
                    "Wrong probe update counts")
            require(bool(point["metrics"]), "Missing round diagnostics")
            for key, summary in point["metrics"].items():
                require(summary.get("count") == row["length"], "Incomplete probe decision coverage")
                for k, v in summary.items():
                    finite(v, f"Nonfinite round metric {key}/{k}")
        record["togo_round_summaries"] = points
        output.append(record)
    priors = [r for r in manifest["runs"] if r["selector"] == "initialization/prior"]
    require(len(priors) == 1, "Missing verified inline prior run")
    prior = episodes(priors[0], cell["seeds"], 500)
    model_series_sha256 = validate_model_series(cell, manifest, run)
    science = {k: v for k, v in params.items() if k != "inner_critic_updates_per_round"}
    compatibility = {"checkpoint": checkpoint, "protocol": {k: v for k, v in protocol.items() if k != "controller_seed"},
                     "alg_params_except_critic_updates": science, "runtime": code["runtime"], "probe": probe,
                     "scientific_source": source_identity(config["alg"], code["commit"], code["dirty"], code.get("source_sha256"))}
    return {"cell": cell, "rows": output, "prior": prior, "compatibility": compatibility,
            "identity": {"cell_id": cell["cell_id"], "critic_updates": count, "rounds": 1,
                         "controller_seed": cell["controller_seed"], "bundle": cell["bundle"],
                         "reused": bool(cell.get("reused")), "manifest_sha256": seal["files"]["manifest.json"],
                         "seal_sha256": file_digest(Path(cell["bundle"]) / "seed-shard-checksums.json"),
                         "seal_content_sha256": seal["sha256"], "config_hash": run["config_hash"], "code": code,
                         "model_series": cell.get("model_series"), "model_series_record_sha256": model_series_sha256,
                         "training_trace_sources": trace_sources}}


def build_report(campaign, loaded, resamples=2000):
    seeds, controllers = validate_campaign(campaign)
    require(set(loaded) == {c["cell_id"] for c in campaign["cells"]}, "No science from incomplete panels")
    values = [loaded[c["cell_id"]] for c in campaign["cells"]]
    require(all(v["compatibility"] == values[0]["compatibility"] for v in values), "Scientific configuration mismatch")
    priors = [{r["seed"]: (r["return"], r["length"], r["terminated"], r["truncated"]) for r in v["prior"]} for v in values]
    require(all(p == priors[0] for p in priors), "Paired frozen-prior results differ")
    rows = [copy.deepcopy(r) for v in values for r in v["rows"]]
    index = {(r["critic_updates"], r["controller_seed"], r["seed"]): r for r in rows}
    require(len(rows) == 420 and len(index) == 420, "Incomplete or duplicate episode grid")
    numeric = ["return", "gain_vs_prior", "gain_vs_c0", "gain_vs_c32", "length", "control_seconds",
               "control_seconds_per_decision", "probe_seconds", "probe_model_transitions",
               *[n + suffix for n in (*WORK, "optimizer_updates") for suffix in ("_per_decision", "_per_episode")]]
    for row in rows:
        count, c, seed = row["critic_updates"], row["controller_seed"], row["seed"]
        row["prior_return"] = priors[0][seed][0]
        row["gain_vs_prior"] = row["return"] - row["prior_return"]
        for target in CRITIC_UPDATES:
            require(row["solver_seed"] == index[target, c, seed]["solver_seed"], "Unpaired controller RNG seeds")
        row["gain_vs_c0"] = row["return"] - index[0, c, seed]["return"]
        row["gain_vs_c32"] = row["return"] - index[32, c, seed]["return"]
    by_episode, summaries, probes, training, contrasts, availability = [], [], [], [], [], {}
    for count in CRITIC_UPDATES:
        group = [r for r in rows if r["critic_updates"] == count]
        keys = sorted(group[0]["model_metrics"])
        require(all(set(r["model_metrics"]) == set(keys) for r in group), "Diagnostic availability differs within C")
        availability[str(count)] = keys
        averaged = []
        for seed in seeds:
            selected = [index[count, c, seed] for c in controllers]
            averaged.append({"critic_updates": count, "seed": seed,
                             **{k: statistics.fmean(r[k] for r in selected) for k in numeric},
                             "model_metrics": {k: statistics.fmean(r["model_metrics"][k] for r in selected) for k in keys}})
        by_episode.extend(averaged)
        summaries.append({"critic_updates": count, "reused": count == 32,
                          "metrics": _summaries(averaged, numeric, resamples),
                          "model_metrics": _summaries([r["model_metrics"] for r in averaged], keys, resamples),
                          "controller_mean_returns": {str(c): statistics.fmean(index[count, c, s]["return"] for s in seeds) for c in controllers},
                          "raw_episode_return_std": statistics.stdev(r["return"] for r in group)})
        for boundary in (0, 1):
            episode_metrics = []
            for seed in seeds:
                selected = [index[count, c, seed]["togo_round_summaries"][boundary]["metrics"] for c in controllers]
                probe_keys = set(selected[0])
                require(all(set(p) == probe_keys for p in selected), "Probe metric availability differs")
                episode_metrics.append({k: statistics.fmean(p[k]["mean"] for p in selected) for k in probe_keys})
            require(all(set(p) == set(episode_metrics[0]) for p in episode_metrics), "Probe metrics differ across episodes")
            probes.append({"configured_critic_updates": count, "round_index": boundary, "actor_updates": boundary * 4,
                           "critic_updates": boundary * count,
                           "metrics": _summaries(episode_metrics, sorted(episode_metrics[0]), resamples)})
        for step in range(1, count + 1):
            episode_metrics = [{k: statistics.fmean(index[count, c, seed]["critic_training_steps"][step - 1]["metrics"][k]["mean"]
                                for c in controllers) for k in TRAINING_METRICS} for seed in seeds]
            training.append({"configured_critic_updates": count, "critic_updates": step, "critic_update_index": step,
                             "measurement": "pre_update_minibatch", "metrics": _summaries(episode_metrics, TRAINING_METRICS, resamples)})
        for reference in (0, 32):
            differences = [statistics.fmean(index[count, c, s]["return"] - index[reference, c, s]["return"] for c in controllers) for s in seeds]
            contrasts.append({"target_critic_updates": count, "reference_critic_updates": reference,
                              "summary": summarize(differences, resamples), "episode_differences": dict(zip(map(str, seeds), differences))})
    return {"schema": SCHEMA, "status": "complete", "attempt_label": campaign["attempt_label"],
            "campaign_sha256": digest(campaign), "critic_updates": CRITIC_UPDATES, "rounds": 1,
            "seeds": seeds, "controller_seeds": controllers,
            "aggregation": "Average decisions within episode for diagnostics; average controller repetitions within environment seed; weight twenty seeds equally",
            "interval": {"method": "paired environment-episode percentile bootstrap", "resamples": resamples, "confidence": .95, "seed": 20260912},
            "return_units": "Undiscounted environment rewards in closed-loop mean-action episodes of at most 500 decisions",
            "diagnostic_scope": "Controller-specific states. Different C settings may visit different states and collect different subsequent replay. Training curves use fresh pre-update sampled minibatches and sampled bootstrap targets, not a fixed held-out evaluation set. Critic loss is two-hot distributional cross-entropy; decoded TD error is in reward/Q units.",
            "timing_semantics": "Recorded control time already excludes model probe time and warmup; no further subtraction. C32 timing is historical and is not paired under identical machine load.",
            "constant_work": {"actor_updates_per_decision": 4, "optimization_model_transitions_per_decision": 128,
                              "rollout_horizon": 1, "minibatch_size": 256, "rounds": 1},
            "c0_semantics": "Inherited critic receives no updates; actor still receives four updates from 128 imagined transitions. Critic training metrics are unavailable, never zero-filled.",
            "training_trace_storage": "Full gzip traces retained in their sealed source bundles; all per-episode/per-step summaries and trace path/checksum provenance are portable in this report",
            "prior": {"summary": summarize([priors[0][s][0] for s in seeds], resamples), "episodes": values[0]["prior"]},
            "scientific_identity": values[0]["compatibility"], "identities": [v["identity"] for v in values],
            "model_metric_availability": availability, "summaries": summaries, "contrasts": contrasts,
            "model_probes": probes, "critic_training": training, "episode_averages": by_episode, "paired_rows": rows}


def render_html(report):
    compact = {k: v for k, v in report.items() if k not in ("paired_rows", "episode_averages", "identities", "scientific_identity")}
    raw = base64.b64encode(gzip.compress(json.dumps(report, separators=(",", ":"), allow_nan=False).encode(), mtime=0)).decode()
    data = json.dumps(compact, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return """<!doctype html><meta charset="utf-8"><title>Critic updates and episode control</title>
<style>body{font:15px system-ui;max-width:1180px;margin:28px auto;color:#18273c;background:#f6f8fa}section{background:white;padding:20px;margin:15px 0;border:1px solid #ddd;border-radius:10px}select,button{font:inherit;padding:5px;max-width:100%}svg{width:100%;height:360px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:right;border-bottom:1px solid #eee}td:first-child,th:first-child{text-align:left}small{color:#53647b}</style>
<h1>Critic updates during repeated mean-action control</h1><p id="status"></p>
<section><b>Full-episode return and diagnostics</b><p><select id="metric"></select> against <select id="axis"></select></p><svg id="chart" viewBox="0 0 1050 360"></svg><small id="scope"></small><p id="constant"></p></section>
<section><b>Critic training trajectory</b><p><select id="trainingmetric"></select> against critic update index (measurement before update)</p><svg id="training" viewBox="0 0 1050 360"></svg><small>Each loss/TD-error measurement uses the minibatch before that update, not a post-update held-out error. Each curve uses the states visited by its own controller. C0 has no critic-training curve.</small></section>
<section><b>Model return before and after adaptation</b><p><select id="probemetric"></select> against configured critic updates</p><svg id="probes" viewBox="0 0 1050 360"></svg><small>Initial: zero actor and critic updates. Final: four actor updates and C critic updates. Sampled-policy probes on controller-specific states, including C0 actor learning.</small></section>
<section><b>Paired complete-panel results</b><div id="summary"></div></section>
<section><b>Paired comparisons</b><div id="contrasts"></div><small>95% percentile intervals use 2,000 paired episode-cluster resamples without multiplicity adjustment.</small></section>
<section><button id="download">Download complete report JSON (420 paired episodes, per-critic-step summaries and source trace checksums)</button><p id="timing"></p><p id="storage"></p></section>
<script id="report-data" type="application/json">""" + data + """</script><script id="raw-gzip" type="application/octet-stream">""" + raw + """</script><script>
const D=JSON.parse(document.getElementById('report-data').textContent),S=D.summaries||[],fmt=x=>Number(x).toFixed(3),range=m=>`${fmt(m.mean)} [${fmt(m.ci95_low)}, ${fmt(m.ci95_high)}]`;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const table=(h,r)=>'<table><tr>'+h.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>'+r.map(a=>'<tr>'+a.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</table>';
const options=(id,items)=>{document.getElementById(id).innerHTML=items.map(([v,l])=>`<option value="${esc(v)}">${esc(l)}</option>`).join('')};
document.getElementById('status').textContent=`${D.attempt_label}: ${D.status}. ${D.complete_cells??21}/21 cells. Three controller repetitions × twenty environment episodes per C; C32 reused exactly.`;
function draw(id,groups,zero=false,prior=null){let P=groups.flatMap(g=>g.points);if(!P.length){document.getElementById(id).innerHTML='';return}let xmax=Math.max(...P.map(p=>p.x),.001)*1.08,ymin=Math.min(...P.map(p=>p.m.ci95_low),...(zero?[0]:[]),...(prior===null?[]:[prior])),ymax=Math.max(...P.map(p=>p.m.ci95_high),...(zero?[0]:[]),...(prior===null?[]:[prior])),pad=Math.max((ymax-ymin)*.12,.001);ymin-=pad;ymax+=pad;let X=x=>95+850*x/xmax,Y=y=>290-215*(y-ymin)/(ymax-ymin),z=[];for(let i=0;i<5;i++){let y=ymin+i*(ymax-ymin)/4;z.push(`<line x1="95" x2="970" y1="${Y(y)}" y2="${Y(y)}" stroke="#e2e6ed"/><text x="3" y="${Y(y)+4}" font-size="11">${y.toFixed(3)}</text>`);let x=i*xmax/4;z.push(`<text x="${X(x)-10}" y="330" font-size="11">${x.toFixed(xmax<1?3:1)}</text>`)}if(prior!==null)z.push(`<line x1="95" x2="970" y1="${Y(prior)}" y2="${Y(prior)}" stroke="#999" stroke-dasharray="4 3"/>`);groups.forEach((g,i)=>{let color=['#206dcd','#d37818','#298a62','#9759a3','#d54a59','#64748b','#13a2ac'][i%7];z.push(`<text x="${105+(i%4)*235}" y="${18+Math.floor(i/4)*20}" fill="${color}" font-size="13">${esc(g.label)}</text><polyline points="${g.points.map(p=>X(p.x)+','+Y(p.m.mean)).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`);g.points.forEach(p=>z.push(`<line x1="${X(p.x)}" x2="${X(p.x)}" y1="${Y(p.m.ci95_low)}" y2="${Y(p.m.ci95_high)}" stroke="${color}"/><circle cx="${X(p.x)}" cy="${Y(p.m.mean)}" r="4" fill="${color}"><title>${esc(g.label+' '+(p.label||''))}: ${range(p.m)}</title></circle>`))});document.getElementById(id).innerHTML=z.join('')}
if(D.status==='complete'){
options('metric',[...Object.keys(S[0].metrics).map(k=>['metrics/'+k,k]),...Array.from(new Set(S.flatMap(s=>Object.keys(s.model_metrics)))).sort().map(k=>['model_metrics/'+k,'Episode diagnostic: '+k])]);options('axis',[['critic_updates_per_decision','critic updates / decision'],['optimizer_updates_per_decision','total optimizer updates / decision'],['control_seconds_per_decision','recorded control seconds / decision']]);options('trainingmetric',Object.keys(D.critic_training[0].metrics).map(k=>[k,k]));options('probemetric',Object.keys(D.model_probes[0].metrics).map(k=>[k,k]));document.getElementById('trainingmetric').value='td_error_abs_mean';document.getElementById('probemetric').value='togo_return_gain_vs_outer';
function main(){let [family,key]=document.getElementById('metric').value.split('/'),axis=document.getElementById('axis').value;draw('chart',[{label:'C0 / 1 / 4 / 8 / 16 / 32 (historical) / 64',points:S.filter(s=>s[family][key]).map(s=>({x:s.metrics[axis].mean,m:s[family][key],label:'C'+s.critic_updates}))}],key.includes('gain'),key==='return'?D.prior.summary.mean:null)};
function train(){let key=document.getElementById('trainingmetric').value;draw('training',D.critic_updates.filter(c=>c>0).map(c=>({label:'C'+c+(c===32?' (historical)':''),points:D.critic_training.filter(s=>s.configured_critic_updates===c).map(s=>({x:s.critic_updates,m:s.metrics[key]}))})))};
function probes(){let key=document.getElementById('probemetric').value;draw('probes',[0,1].map(r=>({label:r===0?'Initial (A0, C0)':'After adaptation (A4, C)',points:D.model_probes.filter(s=>s.round_index===r).map(s=>({x:s.configured_critic_updates,m:s.metrics[key]}))})),key.includes('gain'))};
document.getElementById('metric').onchange=main;document.getElementById('axis').onchange=main;document.getElementById('trainingmetric').onchange=train;document.getElementById('probemetric').onchange=probes;main();train();probes();
document.getElementById('summary').innerHTML=table(['C','Return [95% CI]','Gain vs prior','Gain vs C0','Gain vs C32','Episode-mean SD','60-return SD'],S.map(s=>[s.critic_updates+(s.reused?' (reused)':''),range(s.metrics.return),range(s.metrics.gain_vs_prior),range(s.metrics.gain_vs_c0),range(s.metrics.gain_vs_c32),fmt(s.metrics.return.std),fmt(s.raw_episode_return_std)]));document.getElementById('contrasts').innerHTML=table(['Contrast','Difference [95% CI]'],D.contrasts.map(c=>[`C${c.target_critic_updates} − C${c.reference_critic_updates}`,range(c.summary)]));
document.getElementById('scope').textContent=D.return_units+'. '+D.diagnostic_scope;document.getElementById('constant').textContent='Fixed: 128 model transitions, 4 actor updates, H1, J1, B256 per decision. '+D.c0_semantics;document.getElementById('timing').textContent=D.timing_semantics;document.getElementById('storage').textContent=D.training_trace_storage;
}else document.getElementById('summary').textContent='No scientific curves until all cells validate. Missing: '+JSON.stringify(D.missing||[]);
document.getElementById('download').onclick=async()=>{let bytes=Uint8Array.from(atob(document.getElementById('raw-gzip').textContent),c=>c.charCodeAt(0)),stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')),blob=await new Response(stream).blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='critic-steps-report.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script>"""


def inspect(campaign, loaded, campaign_file_sha256):
    missing = []
    for cell in campaign["cells"]:
        if cell["cell_id"] in loaded:
            continue
        try:
            receipt_path = Path(cell.get("merge_receipt") or Path(campaign["output_root"]) / "production" / cell["cell_id"] / "merge-completion.json")
            receipt = read(receipt_path)
            expected = {"status": "complete", "cell_id": cell["cell_id"], "critic_updates": cell["critic_updates"],
                        "reused": cell["reused"], "seeds": cell["seeds"], "campaign_sha256": campaign_file_sha256}
            require(all(receipt.get(k) == v for k, v in expected.items()) and not receipt.get("smoke", False),
                    "Merge completion receipt differs from production campaign")
            loaded[cell["cell_id"]] = load_cell(cell)
        except (OSError, ValueError, KeyError, TypeError) as error:
            missing.append({"cell_id": cell["cell_id"], "reason": str(error)})
    receipts, complete = campaign.get("worker_receipts", []), 0
    for path in receipts:
        try:
            complete += read(path).get("status") == "complete"
        except (OSError, ValueError, TypeError):
            pass
    return {"complete_cells": len(loaded), "expected_cells": 21, "complete_workers": complete,
            "expected_workers": len(receipts), "missing": missing}


def publish_science(run, wandb, report, output):
    fingerprint = digest(report)
    if run.summary.get("comparison/report_sha256") == fingerprint:
        return
    for axis in AXES:
        run.define_metric("compute/" + axis)
        run.define_metric(f"episodes/vs_{axis}/*", step_metric="compute/" + axis)
    run.define_metric("episode_diagnostics/*", step_metric="compute/critic_updates_per_decision")
    for boundary in (0, 1):
        run.define_metric(f"probes/boundary_{boundary}/configured_critic_updates")
        run.define_metric(f"model_probes/boundary_{boundary}/*", step_metric=f"probes/boundary_{boundary}/configured_critic_updates")
    for count in CRITIC_UPDATES:
        if count:
            run.define_metric(f"training/c{count}/critic_update_index")
            run.define_metric(f"critic_training/c{count}/*", step_metric=f"training/c{count}/critic_update_index")
    for row in report["summaries"]:
        values = {"compute/" + axis: row["metrics"][axis]["mean"] for axis in AXES}
        for axis in AXES:
            for name in ("return", "gain_vs_prior", "gain_vs_c0", "gain_vs_c32"):
                for stat in ("mean", "std", "ci95_low", "ci95_high"):
                    values[f"episodes/vs_{axis}/{name}/{stat}"] = row["metrics"][name][stat]
        for name, summary in row["model_metrics"].items():
            for stat in ("mean", "ci95_low", "ci95_high"):
                values[f"episode_diagnostics/{name}/{stat}"] = summary[stat]
        run.log(values)
    for row in report["model_probes"]:
        boundary = row["round_index"]
        run.log({f"probes/boundary_{boundary}/configured_critic_updates": row["configured_critic_updates"],
                 **{f"model_probes/boundary_{boundary}/{name}/{stat}": summary[stat]
                    for name, summary in row["metrics"].items() for stat in ("mean", "ci95_low", "ci95_high")}})
    for row in report["critic_training"]:
        count = row["configured_critic_updates"]
        run.log({f"training/c{count}/critic_update_index": row["critic_updates"],
                 **{f"critic_training/c{count}/{name}/{stat}": summary[stat]
                    for name, summary in row["metrics"].items() for stat in ("mean", "ci95_low", "ci95_high")}})
    run.log({"comparison/report": wandb.Html(str(output / "report.html"), inject=False)})
    artifact = wandb.Artifact("critic-steps-" + digest(report["attempt_label"])[:16], type="critic-steps-comparison",
                              metadata={"schema": SCHEMA, "report_sha256": fingerprint})
    for name in ("report.json", "report.html", "campaign.json"):
        artifact.add_file(str(output / name), name=name)
    for identity in report["identities"]:
        if identity.get("model_series"):
            require(Path(identity["model_series"]).is_dir(), "Missing complete model-series artifact")
            artifact.add_dir(identity["model_series"], name="model-series/" + identity["cell_id"])
    run.log_artifact(artifact)
    run.summary.update({"comparison/report_sha256": fingerprint, "comparison/prior_return": report["prior"]["summary"]["mean"],
                        "comparison/c0_training_metrics": "unavailable: zero critic updates", "comparison/training_measurement": "pre_update_minibatch"})


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
    campaign_file_sha256 = hashlib.sha256(Path(args.campaign).read_bytes()).hexdigest()
    seeds, controllers = validate_campaign(campaign)
    require(args.poll_seconds > 0 and args.max_watch_seconds > 0, "Invalid watch limits")
    require(not args.watch or args.compute_jobs or args.merge_jobs, "Watching requires scheduler job IDs")
    output = Path(args.output) if args.output else Path(args.campaign).parent / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "campaign.json").exists():
        require(read(output / "campaign.json") == campaign, "Output belongs to a different campaign")
    write(output / "campaign.json", campaign)
    run = wandb = None
    loaded, started, errors = {}, time.monotonic(), 0
    progress = {"schema": SCHEMA, "attempt_label": campaign["attempt_label"], "status": "starting",
                "seeds": seeds, "controller_seeds": controllers}
    old_handlers = {}
    def interrupted(signum, frame):
        raise RuntimeError(f"Publisher received signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, interrupted)
    try:
        if args.mode == "online":
            require(args.wandb_run_id and args.owner == "oscar-rgao48", "Explicit new W&B ID and Oscar owner required")
            identity = {"run_id": args.wandb_run_id, "campaign_sha256": digest(campaign), "owner": args.owner}
            receipt_path = output / "wandb-owner.json"
            resume = "never"
            if receipt_path.exists():
                require(read(receipt_path) == identity, "Publisher receipt identifies a different run or campaign")
                resume = "must"
            import wandb
            run = wandb.init(entity=campaign.get("wandb_entity", "rwgao_b-brown-university"),
                             project=campaign.get("wandb_project", "ambi-inner-bench"),
                             id=args.wandb_run_id, resume=resume, mode="online",
                             name="Critic updates with fixed actor/data | " + campaign["attempt_label"],
                             job_type="critic-steps-comparison", tags=["critic-steps", "full-episodes", "complete-panel"],
                             config={"comparison_schema": SCHEMA, "attempt_label": campaign["attempt_label"],
                                     "campaign_sha256": digest(campaign), "owner": args.owner, "critic_updates": CRITIC_UPDATES,
                                     "checkpoint_sha256": CHECKPOINT, "source_run": "rwgao_b-brown-university/ambi/mey3rxj8",
                                     "checkpoint_step": 200000, "seeds": seeds, "controller_seeds": controllers,
                                     "protocol": "Full500 mean decisions, J1 H1 N128 B256 A4; C=0/1/4/8/16/32/64; prior weights, alpha0 and saved Q scale",
                                     "science_publication": "Only after all 21 cells validate"})
            write(receipt_path, identity)
            run.define_metric("progress/elapsed_seconds")
            run.define_metric("progress/*", step_metric="progress/elapsed_seconds")
            run.summary["comparison/status"] = "running"
            run.log({"progress/elapsed_seconds": 0., "progress/complete_cells": 0,
                     "progress/expected_cells": 21, "progress/complete_workers": 0,
                     "progress/expected_workers": len(campaign.get("worker_receipts", []))})
        while True:
            progress.update(inspect(campaign, loaded, campaign_file_sha256))
            progress.update(elapsed_seconds=time.monotonic() - started, status="running")
            if run:
                run.log({"progress/" + k: progress[k] for k in ("elapsed_seconds", "complete_cells", "expected_cells", "complete_workers", "expected_workers")})
            if not progress["missing"]:
                report = build_report(campaign, loaded)
                write(output / "report.json", report)
                (output / "report.html").write_text(render_html(report))
                if run:
                    publish_science(run, wandb, report, output)
                progress.update(status="complete", stop_reason="All 21 complete cells published")
                break
            if not args.watch:
                progress.update(status="incomplete", stop_reason="Snapshot without watching")
                break
            try:
                progress["scheduler"] = job_state(args.compute_jobs + args.merge_jobs)
                errors = 0
            except (OSError, subprocess.SubprocessError) as error:
                errors += 1
                progress["scheduler"] = {"finished": False, "error": str(error), "consecutive_errors": errors}
            if progress["scheduler"]["finished"] or errors >= 3 or progress["elapsed_seconds"] >= args.max_watch_seconds:
                progress.update(status="incomplete", stop_reason="All jobs terminated" if progress["scheduler"]["finished"]
                                else "Scheduler inspection failed three times" if errors >= 3 else "Watch deadline exceeded")
                break
            write(output / "progress.json", progress)
            time.sleep(min(args.poll_seconds, 60))
        if progress["status"] != "complete":
            write(output / "report.json", progress)
            (output / "report.html").write_text(render_html(progress))
        progress["elapsed_seconds"] = time.monotonic() - started
        write(output / "progress.json", progress)
        if run:
            run.summary.update({"comparison/status": progress["status"], "comparison/missing_cells": progress["missing"],
                                "comparison/stop_reason": progress.get("stop_reason")})
        print(json.dumps({"status": progress["status"], "complete_cells": progress["complete_cells"], "output": str(output), "missing": progress["missing"]}))
        return 0 if progress["status"] == "complete" else 2
    except BaseException as error:
        progress.update(status="failed", error=str(error) or type(error).__name__)
        write(output / "failure.json", progress)
        write(output / "progress.json", progress)
        if run:
            run.summary.update({"comparison/status": "failed", "comparison/error": str(error)})
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if run:
            run.finish(exit_code=0 if progress["status"] == "complete" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
