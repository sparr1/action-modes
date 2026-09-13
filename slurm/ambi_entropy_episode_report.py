"""Complete-panel entropy comparison for repeated, mean-action episode control.

This CPU publisher does no evaluation. It requires all 27 merger-sealed cells,
including exact historical off seals, before publishing any scientific curves.
Controller repetitions are averaged within environment episode before inference.
"""
from __future__ import annotations

import argparse
import base64
import copy
from functools import lru_cache
import gzip
import hashlib
import subprocess
import json
from pathlib import Path
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from slurm.ambi_round_scaling_report import (WORK, digest, episodes, finite, job_state,
                                            read, require, sealed_manifest, summarize, write)

SCHEMA = "ambi-entropy-episode-comparison-v1"
ARMS = {"off": {"mode": "tdmpc2_scaled", "alpha": 0.0},
        "native": {"mode": "tdmpc2_scaled", "alpha": .0001},
        "squashed": {"mode": "squashed", "alpha": .0021}}
SEEDS = list(range(101, 121))
CONTROLLERS = [55, 56, 57]
ROUNDS = (1, 2, 4)
CHECKPOINT = "909e5c1d125aecc952e0544d802b4b946ae5089909c7a55885f272c32ed3e12f"
AXES = [n + "_per_decision" for n in WORK] + ["control_seconds_per_decision"]


def validate_campaign(campaign):
    require(bool(campaign.get("attempt_label", "").strip()), "Explicit attempt required")
    require(campaign.get("arms") == ARMS, "Wrong entropy arm protocol")
    require(campaign.get("seeds") == SEEDS and campaign.get("controller_seeds") == CONTROLLERS,
            "Expected twenty paired seeds and three controller seeds")
    cells = campaign.get("cells", [])
    require(len(cells) == 27 and len({c["cell_id"] for c in cells}) == 27, "Expected 27 unique cells")
    require({(c["arm"], c["rounds"], c["controller_seed"]) for c in cells}
            == {(a, j, c) for a in ARMS for j in ROUNDS for c in CONTROLLERS}, "Incomplete entropy/J/controller grid")
    for c in cells:
        require(c["seeds"] == SEEDS and Path(c["bundle"]).is_absolute(), "Wrong seeds or relative bundle")
        require(bool(c.get("reused")) == (c["arm"] == "off"), "Only all nine historical off cells are reused")
        if c["reused"]:
            require(all(isinstance(c.get(k), str) and len(c[k]) == 64 for k in
                        ("bundle_manifest_sha256", "bundle_seal_sha256")), "Historical bundle hashes required")
    return SEEDS, CONTROLLERS


@lru_cache(maxsize=32)
def source_identity(algorithm, commit, dirty, source_sha256):
    from utils.eval_series_data import scientific_identity
    require(dirty is False, "Only clean committed scientific sources are accepted")
    return scientific_identity(algorithm, None, commit, dirty, source_sha256)


def validate_model_series(cell, manifest, run):
    """Bind the portable per-decision export to the selected sealed episode run."""
    if not cell.get("model_series"):
        return None
    from utils.ambi_diagnostic_series import read_diagnostic_bundle, _coord
    record = read_diagnostic_bundle(cell["model_series"])
    checkpoint = dict(manifest["checkpoint"])
    checkpoint.setdefault("step", checkpoint["metadata"]["checkpoint"]["step"])
    expected = {"checkpoint": checkpoint, "setting": run["config"],
                "protocol": {**manifest["protocol"], "togo_return_probe": run["togo_return_probe"]},
                "code": manifest["code"], "scope": "controller_episode"}
    require(record.get("status") == "complete" and record["identity"] == expected,
            "Model-series source identity mismatch")
    require(record["rows"] == sorted(run.get("togo_probe_rows", []), key=_coord),
            "Model-series paired rows differ from sealed episode bundle")
    return record["record_sha256"]


def load_cell(cell):
    manifest, seal = sealed_manifest(cell["bundle"])
    if cell.get("reused"):
        require(seal["files"]["manifest.json"] == cell["bundle_manifest_sha256"]
                and hashlib.sha256((Path(cell["bundle"]) / "seed-shard-checksums.json").read_bytes()).hexdigest() == cell["bundle_seal_sha256"], "Historical bundle identity mismatch")
    require(manifest["seed_shard_merge"]["expected_seeds"] == cell["seeds"], "Merge has wrong seeds")
    checkpoint, protocol, code = manifest["checkpoint"], manifest["protocol"], manifest["code"]
    require(checkpoint["sha256"] == CHECKPOINT and checkpoint.get("source_run", "").endswith("/mey3rxj8")
            and checkpoint["metadata"]["checkpoint"]["step"] == 200000, "Wrong verified backbone checkpoint")
    require(protocol.get("controller_seed") == cell["controller_seed"]
            and protocol.get("action_rule") == "tanh_mean" and protocol.get("max_steps") == 500,
            "Wrong controller or real execution protocol")
    runs = [r for r in manifest["runs"] if r["selector"] == cell["selector"]]
    require(len(runs) == 1, "Missing or duplicate selected run")
    run, j = runs[0], cell["rounds"]
    result, config = run["result"], run["config"]
    require(result.get("controller_seed") == cell["controller_seed"], "Wrong result controller seed")
    require(run.get("config_hash") == digest(config), "Run configuration checksum mismatch")
    params = result["alg_params"]
    expected = {"inner_rounds": j, "inner_rollout_horizon": 1, "inner_rollouts_per_round": 128,
                "inner_actor_updates_per_round": 4, "inner_critic_updates_per_round": 32,
                "inner_batch_size": 256, "inner_temperature": ARMS[cell["arm"]]["alpha"],
                "inner_actor_entropy_mode": ARMS[cell["arm"]]["mode"],
                "inner_actor_initialization": "prior", "inner_critic_initialization": "prior",
                "inner_temperature_mode": "fixed", "inner_temperature_initialization": "fixed",
                "inner_execution_action": "mean", "inner_behavior_action": "policy_sample",
                "inner_finite_horizon": True, "inner_sac_critic_target": "reward_only"}
    require(all(params.get(k) == v and config["alg_params"].get(k) == v for k, v in expected.items()),
            "Wrong entropy learner schedule")
    require(not result.get("nonfinite_trace_metrics") and not result.get("nonfinite_model_metrics"),
            "Nonfinite diagnostic metrics")
    probe = result.get("togo_return_probe")
    require(probe and probe.get("rollouts") == 32 and probe.get("horizon") == 1
            and probe.get("tail_actor") == "outer" and probe.get("tail_critic") == "outer_online"
            and probe.get("tail_q_reduction") == "mean_pair" and probe.get("entropy_bonus") is False,
            "Wrong model probe protocol")
    output = []
    for row in episodes(run, cell["seeds"], 500):
        metrics = {k: finite(v, "Nonfinite episode diagnostic") for k, v in row["model_metrics"].items()}
        record = {"cell_id": cell["cell_id"], "arm": cell["arm"], "rounds": j,
                  "controller_seed": cell["controller_seed"], "seed": row["seed"],
                  "solver_seed": row["solver_seed"], "return": row["return"], "length": row["length"],
                  "terminated": row["terminated"], "truncated": row["truncated"],
                  "reused": bool(cell.get("reused")), "model_metrics": metrics,
                  "control_seconds": finite(row["control_seconds"], "Invalid control time"),
                  "probe_seconds": finite(row.get("togo_probe_seconds"), "Missing model probe time"),
                  "probe_model_transitions": finite(row.get("togo_probe_model_steps"), "Missing probe work")}
        require(record["control_seconds"] >= 0 and record["probe_seconds"] >= 0, "Negative runtime")
        for name, key in WORK.items():
            value = finite(metrics.get(key), f"Missing realized {name}")
            require(value == j * {"actor_updates": 4, "critic_updates": 32, "model_transitions": 128}[name],
                    f"Unexpected realized {name}")
            record[name + "_per_decision"], record[name + "_per_episode"] = value, value * row["length"]
        record["control_seconds_per_decision"] = record["control_seconds"] / row["length"]
        rounds = row.get("togo_round_summaries", [])
        require([r["round_index"] for r in rounds] == list(range(j + 1)), "Incomplete round probes")
        for point in rounds:
            r = point["round_index"]
            require(point["actor_updates"] == 4 * r and point["critic_updates"] == 32 * r,
                    "Wrong probe update counts")
            require(bool(point["metrics"]), "Missing round diagnostics")
            for key, summary in point["metrics"].items():
                require(summary.get("count") == row["length"], "Incomplete probe decision coverage")
                for k, v in summary.items():
                    finite(v, f"Nonfinite round metric {key}/{k}")
        record["togo_round_summaries"] = rounds
        output.append(record)
    priors = [r for r in manifest["runs"] if r["selector"] == "initialization/prior"]
    require(len(priors) == 1, "Missing verified inline prior run")
    prior = episodes(priors[0], cell["seeds"], 500)
    model_series_sha256 = validate_model_series(cell, manifest, run)
    science = {k: v for k, v in params.items()
               if k not in ("inner_rounds", "inner_temperature", "inner_actor_entropy_mode")}
    compatibility = {"checkpoint": checkpoint, "protocol": {k: v for k, v in protocol.items() if k != "controller_seed"},
                     "alg_params_except_entropy_and_rounds": science, "runtime": code["runtime"], "probe": probe,
                     "scientific_source": source_identity(config["alg"], code["commit"], code["dirty"], code.get("source_sha256"))}
    return {"cell": cell, "rows": output, "prior": prior, "compatibility": compatibility,
            "identity": {"cell_id": cell["cell_id"], "arm": cell["arm"], "rounds": j,
                         "controller_seed": cell["controller_seed"], "bundle": cell["bundle"],
                         "reused": bool(cell.get("reused")), "manifest_sha256": seal["files"]["manifest.json"],
                         "seal_sha256": seal["sha256"], "config_hash": run["config_hash"], "code": code,
                         "model_series": cell.get("model_series"), "model_series_record_sha256": model_series_sha256,
                         "eval_run_dir": cell.get("eval_run_dir")}}


def _summaries(rows, keys, resamples):
    return {k: summarize([r[k] for r in rows], resamples) for k in keys}


def build_report(campaign, loaded, resamples=2000):
    seeds, controllers = validate_campaign(campaign)
    require(set(loaded) == {c["cell_id"] for c in campaign["cells"]}, "No science from incomplete panels")
    values = [loaded[c["cell_id"]] for c in campaign["cells"]]
    require(all(v["compatibility"] == values[0]["compatibility"] for v in values), "Scientific configuration mismatch")
    priors = [{r["seed"]: (r["return"], r["length"], r["terminated"], r["truncated"]) for r in v["prior"]} for v in values]
    require(all(p == priors[0] for p in priors), "Paired frozen-prior results differ")
    rows = [copy.deepcopy(r) for v in values for r in v["rows"]]
    index = {(r["arm"], r["rounds"], r["controller_seed"], r["seed"]): r for r in rows}
    require(len(rows) == 540 and len(index) == 540, "Incomplete or duplicate episode grid")
    model_keys_by_arm = {}
    for a in ARMS:
        groups = [r for r in rows if r["arm"] == a]
        keys = set(groups[0]["model_metrics"])
        require(all(set(r["model_metrics"]) == keys for r in groups), "Diagnostic availability differs within arm")
        model_keys_by_arm[a] = sorted(keys)
    numeric = ["return", "gain_vs_prior", "gain_vs_off", "gain_vs_native", "gain_vs_j1", "gain_vs_j2", "length",
               "control_seconds", "control_seconds_per_decision", "probe_seconds", "probe_model_transitions",
               *[n + suffix for n in WORK for suffix in ("_per_decision", "_per_episode")]]
    for row in rows:
        a, j, c, s = (row[k] for k in ("arm", "rounds", "controller_seed", "seed"))
        row["prior_return"] = priors[0][s][0]
        row["gain_vs_prior"] = row["return"] - row["prior_return"]
        for target in ARMS:
            require(row["solver_seed"] == index[target, j, c, s]["solver_seed"], "Unpaired controller RNG seeds")
        for name, key in (("gain_vs_off", ("off", j, c, s)), ("gain_vs_native", ("native", j, c, s)),
                          ("gain_vs_j1", (a, 1, c, s)), ("gain_vs_j2", (a, 2, c, s))):
            row[name] = row["return"] - index[key]["return"]
    by_episode, summaries, dynamics = [], [], []
    for a in ARMS:
        model_keys = model_keys_by_arm[a]
        for j in ROUNDS:
            averaged = []
            for seed in seeds:
                group = [index[a, j, c, seed] for c in controllers]
                averaged.append({"arm": a, "rounds": j, "seed": seed,
                                 **{k: statistics.fmean(r[k] for r in group) for k in numeric},
                                 "model_metrics": {k: statistics.fmean(r["model_metrics"][k] for r in group) for k in model_keys}})
            by_episode.extend(averaged)
            summaries.append({"arm": a, "rounds": j, "entropy": ARMS[a], "reused": a == "off",
                              "metrics": _summaries(averaged, numeric, resamples),
                              "model_metrics": _summaries([r["model_metrics"] for r in averaged], model_keys, resamples),
                              "controller_mean_returns": {str(c): statistics.fmean(index[a, j, c, s]["return"] for s in seeds) for c in controllers},
                              "raw_episode_return_std": statistics.stdev(index[a, j, c, s]["return"] for c in controllers for s in seeds)})
            for r in range(j + 1):
                episode_metrics = []
                for seed in seeds:
                    points = [index[a, j, c, seed]["togo_round_summaries"][r]["metrics"] for c in controllers]
                    keys = set(points[0])
                    require(all(set(p) == keys for p in points), "Round diagnostic availability differs")
                    episode_metrics.append({k: statistics.fmean(p[k]["mean"] for p in points) for k in keys})
                require(all(set(p) == set(episode_metrics[0]) for p in episode_metrics), "Round metrics differ across episodes")
                dynamics.append({"arm": a, "rounds": j, "round_index": r, "actor_updates": r * 4,
                                 "critic_updates": r * 32, "metrics": _summaries(episode_metrics, sorted(episode_metrics[0]), resamples)})
    contrasts = []
    for j in ROUNDS:
        for target, reference in (("native", "off"), ("squashed", "off"), ("squashed", "native")):
            vals = [statistics.fmean(index[target, j, c, s]["return"] - index[reference, j, c, s]["return"] for c in controllers) for s in seeds]
            contrasts.append({"kind": "entropy_within_round", "target": target, "reference": reference,
                              "rounds": j, "summary": summarize(vals, resamples), "episode_differences": dict(zip(map(str, seeds), vals))})
    for a in ARMS:
        for target, reference in ((2, 1), (4, 1), (4, 2)):
            vals = [statistics.fmean(index[a, target, c, s]["return"] - index[a, reference, c, s]["return"] for c in controllers) for s in seeds]
            contrasts.append({"kind": "round_within_entropy", "arm": a, "target": target, "reference": reference,
                              "summary": summarize(vals, resamples), "episode_differences": dict(zip(map(str, seeds), vals))})
    return {"schema": SCHEMA, "status": "complete", "attempt_label": campaign["attempt_label"],
            "campaign_sha256": digest(campaign), "arms": ARMS, "seeds": seeds, "controller_seeds": controllers,
            "aggregation": "Average controller repetitions within environment seed, then episodes equally; diagnostics first average decisions within episode",
            "interval": {"method": "paired environment-episode percentile bootstrap", "resamples": resamples, "confidence": .95, "seed": 20260912},
            "return_units": "Undiscounted environment reward summed over a closed-loop episode of at most 500 decisions",
            "diagnostic_scope": "Controller-specific episode states; different arms need not visit identical states. Model probes are sampled-policy reward-plus-outer-Q returns without entropy bonus.",
            "timing_semantics": "Recorded control seconds exclude measured model probe time and warmup; no further subtraction. All off timing is historical, including its historical J4 repetition; do not interpret timings as paired measurements under identical machine load.",
            "prior": {"summary": summarize([priors[0][s][0] for s in seeds], resamples), "episodes": values[0]["prior"]},
            "scientific_identity": values[0]["compatibility"], "identities": [v["identity"] for v in values],
            "model_metric_availability": model_keys_by_arm,
            "summaries": summaries, "contrasts": contrasts, "round_dynamics": dynamics,
            "episode_averages": by_episode, "paired_rows": rows}


def render_html(report):
    # Full paired episodes and diagnostics are portable, without adding thousands
    # of DOM rows or reparsing them on every chart selection.
    compact = {k: v for k, v in report.items() if k not in ("paired_rows", "episode_averages", "identities", "scientific_identity")}
    encoded = base64.b64encode(gzip.compress(json.dumps(report, separators=(",", ":"), allow_nan=False).encode(), mtime=0)).decode()
    data = json.dumps(compact, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return """<!doctype html><meta charset="utf-8"><title>Entropy and repeated episode control</title>
<style>body{font:15px system-ui;max-width:1180px;margin:28px auto;color:#18273c;background:#f6f8fa}section{background:white;padding:20px;margin:15px 0;border:1px solid #ddd;border-radius:10px}select,button{font:inherit;padding:5px;max-width:100%}svg{width:100%;height:340px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:right;border-bottom:1px solid #eee}td:first-child,th:first-child{text-align:left}small{color:#53647b}pre{white-space:pre-wrap}</style>
<h1>Entropy during repeated mean-action control</h1><p id="status"></p>
<section><b>Full-episode return and diagnostics</b><p><select id="metric"></select> against <select id="axis"></select></p><svg id="chart" viewBox="0 0 1050 340"></svg><small id="scope"></small></section>
<section><b>Inner model probe dynamics</b><p><select id="roundmetric"></select> for <select id="rounds"><option>1</option><option>2</option><option selected>4</option></select> rounds</p><svg id="dynamics" viewBox="0 0 1050 340"></svg><small>Actor update axis. Each point first averages visited decisions, then controller repetitions, then the twenty source episodes. These are controller-specific states, not shared roots.</small></section>
<section><b>Paired complete-panel results</b><div id="summary"></div></section>
<section><b>Paired entropy and round contrasts</b><div id="contrasts"></div><small>95% percentile intervals use 2,000 paired episode-cluster resamples, without multiplicity adjustment.</small></section>
<section><button id="download">Download complete report JSON (includes all 540 paired episodes and per-round diagnostics)</button><p id="timing"></p></section>
<script id="report-data" type="application/json">""" + data + """</script><script id="raw-gzip" type="application/octet-stream">""" + encoded + """</script><script>
const D=JSON.parse(document.getElementById('report-data').textContent),S=D.summaries||[],fmt=x=>Number(x).toFixed(3),range=m=>`${fmt(m.mean)} [${fmt(m.ci95_low)}, ${fmt(m.ci95_high)}]`;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const table=(h,r)=>'<table><tr>'+h.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>'+r.map(a=>'<tr>'+a.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</table>';
document.getElementById('status').textContent=`${D.attempt_label}: ${D.status}. ${D.complete_cells??27}/27 cells. Three controller repetitions × twenty environment episodes per arm and round count.`;
const options=(id,items)=>{document.getElementById(id).innerHTML=items.map(([v,l])=>`<option value="${esc(v)}">${esc(l)}</option>`).join('')};
function draw(id,groups,zero=false,prior=null){let P=groups.flatMap(g=>g.points);if(!P.length)return;let xmax=Math.max(...P.map(p=>p.x),.001)*1.12,ymin=Math.min(...P.map(p=>p.m.ci95_low),...(zero?[0]:[]),...(prior===null?[]:[prior])),ymax=Math.max(...P.map(p=>p.m.ci95_high),...(zero?[0]:[]),...(prior===null?[]:[prior])),pad=Math.max((ymax-ymin)*.12,.001);ymin-=pad;ymax+=pad;let X=x=>90+840*x/xmax,Y=y=>275-230*(y-ymin)/(ymax-ymin),z=[];for(let i=0;i<5;i++){let y=ymin+i*(ymax-ymin)/4;z.push(`<line x1="90" x2="970" y1="${Y(y)}" y2="${Y(y)}" stroke="#e2e6ed"/><text x="3" y="${Y(y)+4}" font-size="11">${y.toFixed(3)}</text>`);let x=i*xmax/4;z.push(`<text x="${X(x)-10}" y="306" font-size="11">${x.toFixed(xmax<1?3:1)}</text>`)}if(prior!==null)z.push(`<line x1="90" x2="970" y1="${Y(prior)}" y2="${Y(prior)}" stroke="#999" stroke-dasharray="4 3"/>`);groups.forEach((g,i)=>{let color=['#64748b','#d37818','#206dcd'][i%3];z.push(`<text x="${110+i*260}" y="18" fill="${color}" font-size="13">${esc(g.label)}</text><polyline points="${g.points.map(p=>X(p.x)+','+Y(p.m.mean)).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`);g.points.forEach(p=>z.push(`<line x1="${X(p.x)}" x2="${X(p.x)}" y1="${Y(p.m.ci95_low)}" y2="${Y(p.m.ci95_high)}" stroke="${color}"/><circle cx="${X(p.x)}" cy="${Y(p.m.mean)}" r="4" fill="${color}"><title>${esc(g.label)}: ${range(p.m)}</title></circle>`))});document.getElementById(id).innerHTML=z.join('')}
if(D.status==='complete'){
options('metric',[...Object.keys(S[0].metrics).map(k=>['metrics/'+k,k]),...Array.from(new Set(S.flatMap(s=>Object.keys(s.model_metrics)))).sort().map(k=>['model_metrics/'+k,'Episode diagnostic: '+k])]);options('axis',[['actor_updates_per_decision','actor updates / decision'],['critic_updates_per_decision','critic updates / decision'],['model_transitions_per_decision','model transitions / decision'],['control_seconds_per_decision','recorded control seconds / decision']]);options('roundmetric',Object.keys(D.round_dynamics[0].metrics).map(k=>[k,k]));document.getElementById('roundmetric').value='togo_return_gain_vs_outer';
function main(){let [family,key]=document.getElementById('metric').value.split('/'),axis=document.getElementById('axis').value;draw('chart',Object.keys(D.arms).map(a=>({label:a+(a==='off'?' (historical)':''),points:S.filter(s=>s.arm===a&&s[family][key]).map(s=>({x:s.metrics[axis].mean,m:s[family][key]}))})),key.includes('gain'),key==='return'?D.prior.summary.mean:null)};
function dyn(){let j=Number(document.getElementById('rounds').value),key=document.getElementById('roundmetric').value;draw('dynamics',Object.keys(D.arms).map(a=>({label:a,points:D.round_dynamics.filter(s=>s.arm===a&&s.rounds===j).map(s=>({x:s.actor_updates,m:s.metrics[key]}))})),key.includes('gain'))};
document.getElementById('metric').onchange=main;document.getElementById('axis').onchange=main;document.getElementById('rounds').onchange=dyn;document.getElementById('roundmetric').onchange=dyn;main();dyn();
document.getElementById('summary').innerHTML=table(['Arm / rounds','Return [95% CI]','Gain vs prior','Gain vs off','Episode-mean SD','60-return SD'],S.map(s=>[s.arm+' J'+s.rounds,range(s.metrics.return),range(s.metrics.gain_vs_prior),range(s.metrics.gain_vs_off),fmt(s.metrics.return.std),fmt(s.raw_episode_return_std)]));document.getElementById('contrasts').innerHTML=table(['Contrast','Difference [95% CI]'],D.contrasts.map(c=>[c.kind==='entropy_within_round'?`${c.target} − ${c.reference}, J${c.rounds}`:`${c.arm}: J${c.target} − J${c.reference}`,range(c.summary)]));
document.getElementById('scope').textContent=D.return_units+'. '+D.diagnostic_scope;document.getElementById('timing').textContent=D.timing_semantics;
}else document.getElementById('summary').textContent='No scientific curves until all cells validate. Missing: '+JSON.stringify(D.missing||[]);
document.getElementById('download').onclick=async()=>{let bytes=Uint8Array.from(atob(document.getElementById('raw-gzip').textContent),c=>c.charCodeAt(0)),stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')),blob=await new Response(stream).blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='entropy-episode-report.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script>"""


def inspect(campaign, loaded, campaign_file_sha256):
    missing = []
    for cell in campaign["cells"]:
        if cell["cell_id"] in loaded:
            continue
        try:
            receipt_path = Path(cell.get("merge_receipt") or Path(campaign["output_root"]) / "production" / cell["cell_id"] / "merge-completion.json")
            receipt = read(receipt_path)
            expected = {"status": "complete", "cell_id": cell["cell_id"], "arm": cell["arm"],
                        "reused": cell["reused"], "seeds": cell["seeds"], "campaign_sha256": campaign_file_sha256}
            require(all(receipt.get(k) == v for k, v in expected.items()) and not receipt.get("smoke", False),
                    "Merge completion receipt differs from production campaign")
            loaded[cell["cell_id"]] = load_cell(cell)
        except (OSError, ValueError, KeyError, TypeError) as error:
            missing.append({"cell_id": cell["cell_id"], "reason": str(error)})
    receipts = campaign.get("worker_receipts", [])
    complete = 0
    for path in receipts:
        try:
            complete += read(path).get("status") == "complete"
        except (OSError, ValueError, TypeError):
            pass
    return {"complete_cells": len(loaded), "expected_cells": 27, "complete_workers": complete,
            "expected_workers": len(receipts), "missing": missing}


def publish_science(run, wandb, report, output):
    fingerprint = digest(report)
    if run.summary.get("comparison/report_sha256") == fingerprint:
        return
    for a in ARMS:
        for axis in AXES:
            x = f"compute/{a}/{axis}"
            run.define_metric(x)
            run.define_metric(f"episodes/{a}/vs_{axis}/*", step_metric=x)
        for j in ROUNDS:
            x = f"inner/{a}/j{j}/actor_updates"
            run.define_metric(x)
            run.define_metric(f"model_probes/{a}/j{j}/*", step_metric=x)
    for row in report["summaries"]:
        a = row["arm"]
        values = {}
        for axis in AXES:
            values[f"compute/{a}/{axis}"] = row["metrics"][axis]["mean"]
            for name in ("return", "gain_vs_prior", "gain_vs_off", "gain_vs_native", "gain_vs_j1", "gain_vs_j2"):
                for stat in ("mean", "std", "ci95_low", "ci95_high"):
                    values[f"episodes/{a}/vs_{axis}/{name}/{stat}"] = row["metrics"][name][stat]
        x = f"compute/{a}/actor_updates_per_decision"
        run.define_metric(f"episode_diagnostics/{a}/*", step_metric=x)
        for name, summary in row["model_metrics"].items():
            for stat in ("mean", "ci95_low", "ci95_high"):
                values[f"episode_diagnostics/{a}/{name}/{stat}"] = summary[stat]
        run.log(values)
    for row in report["round_dynamics"]:
        stem = f"{row['arm']}/j{row['rounds']}"
        run.log({f"inner/{stem}/actor_updates": row["actor_updates"],
                 **{f"model_probes/{stem}/{name}/{stat}": summary[stat]
                    for name, summary in row["metrics"].items() for stat in ("mean", "ci95_low", "ci95_high")}})
    run.log({"comparison/report": wandb.Html(str(output / "report.html"), inject=False)})
    artifact = wandb.Artifact("entropy-episodes-" + digest(report["attempt_label"])[:16],
                              type="entropy-episode-comparison", metadata={"schema": SCHEMA, "report_sha256": fingerprint})
    for name in ("report.json", "report.html", "campaign.json"):
        artifact.add_file(str(output / name), name=name)
    # Complete per-decision probes remain in the native model-series export;
    # upload them in this one comparison artifact, without creating extra runs.
    for identity in report["identities"]:
        directory = identity.get("model_series")
        if directory:
            require(Path(directory).is_dir(), "Missing complete model-series artifact")
            artifact.add_dir(str(directory), name="model-series/" + identity["cell_id"])
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
                             name="Entropy × J1/J2/J4 full episodes | " + campaign["attempt_label"],
                             job_type="entropy-episode-comparison", tags=["entropy", "full-episodes", "complete-panel"],
                             config={"comparison_schema": SCHEMA, "attempt_label": campaign["attempt_label"],
                                     "campaign_sha256": digest(campaign), "owner": args.owner, "arms": ARMS,
                                     "checkpoint_sha256": CHECKPOINT, "source_run": "rwgao_b-brown-university/ambi/mey3rxj8",
                                     "checkpoint_step": 200000, "seeds": seeds, "controller_seeds": controllers,
                                     "protocol": "Full500 mean decisions, H1 N128 B256 C32 A4 per round; prior weights, fixed alpha and saved Q scale",
                                     "science_publication": "Only after all 27 cells validate"})
            write(receipt_path, identity)
            run.define_metric("progress/elapsed_seconds")
            run.define_metric("progress/*", step_metric="progress/elapsed_seconds")
            run.summary["comparison/status"] = "running"
            run.log({"progress/elapsed_seconds": 0., "progress/complete_cells": 0,
                     "progress/expected_cells": 27, "progress/complete_workers": 0,
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
                progress.update(status="complete", stop_reason="All 27 complete cells published")
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
