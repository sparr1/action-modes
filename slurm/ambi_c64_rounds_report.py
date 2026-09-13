"""Complete C64 round-scaling comparison, with C32 and historical references.

This CPU-only reporter never runs a policy. It publishes one comparison after
12 production merge receipts and their sealed episode/model-series exports
validate. C32 at all rounds and C64 J1 are exact historical references.
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

SCHEMA = "ambi-c64-rounds-comparison-v1"
CRITIC_UPDATES = [32, 64]
ROUNDS = [1, 2]
SEEDS = list(range(101, 121))
CONTROLLERS = [55, 56, 57]
CHECKPOINT = "909e5c1d125aecc952e0544d802b4b946ae5089909c7a55885f272c32ed3e12f"
AXES = ["rounds", "critic_updates_per_decision", "actor_updates_per_decision",
        "model_transitions_per_decision", "control_seconds_per_decision"]
TRAINING_METRICS = ("critic_loss", "td_error_abs_mean", "critic_grad_norm", "q_mean",
                    "q_abs_mean", "q_target_mean", "q_target_clip_fraction")


def validate_campaign(campaign):
    require(bool(campaign.get("attempt_label", "").strip()), "Explicit attempt required")
    require(campaign.get("critic_updates") == CRITIC_UPDATES and campaign.get("rounds") == ROUNDS,
            "Wrong C64 round-scaling sweep")
    require(campaign.get("seeds") == SEEDS and campaign.get("controller_seeds") == CONTROLLERS,
            "Expected twenty paired seeds and three controller seeds")
    cells = campaign.get("cells", [])
    require(len(cells) == 12 and len({c["cell_id"] for c in cells}) == 12, "Expected 12 unique cells")
    require({(c["critic_updates"], c["rounds"], c["controller_seed"]) for c in cells}
            == {(n, j, s) for n in CRITIC_UPDATES for j in ROUNDS for s in CONTROLLERS}, "Incomplete critic/controller grid")
    for c in cells:
        require(c["seeds"] == SEEDS and c["rounds"] in ROUNDS and Path(c["bundle"]).is_absolute(),
                "Wrong seeds, rounds or relative bundle")
        require(bool(c.get("reused")) == (c["critic_updates"] == 32 or c["rounds"] == 1), "Only C32 and C64 J1 cells are reused")
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
    count, rounds = cell["critic_updates"], cell["rounds"]
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
                round_index = row.get("round_index")
                require(key in expected and type(round_index) is int and 1 <= round_index <= rounds,
                        "Unexpected optimizer trace root/round")
                require(row.get("measurement") == "pre_update_minibatch" and not row.get("nonfinite"),
                        "Wrong training measurement phase or nonfinite trace")
                critic, actor = row.get("updated_critic"), row.get("updated_actor")
                require(type(critic) is bool and type(actor) is bool and critic != actor
                        and row.get("updated_temperature") is False, "Expected critic-first or actor-only update")
                masks = coordinates.setdefault(key, [0, 0])
                if critic:
                    step = row["critic_updates"]
                    require(type(step) is int and count * (round_index - 1) < step <= count * round_index
                            and row["actor_updates"] == 4 * (round_index - 1),
                            "Wrong critic update counter")
                    require(masks[0] == (1 << (step - 1)) - 1 and masks[1] == (1 << (4 * (round_index - 1))) - 1,
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
                    require(type(step) is int and 4 * (round_index - 1) < step <= 4 * round_index
                            and row["critic_updates"] == count * round_index,
                            "Wrong actor update counter")
                    require(masks[0] == (1 << (count * round_index)) - 1 and masks[1] == (1 << (step - 1)) - 1,
                            "Duplicate or out-of-order actor update")
                    masks[1] |= 1 << (step - 1)
    require(set(coordinates) == expected and all(masks == [(1 << (count * rounds)) - 1, (1 << (4 * rounds)) - 1] for masks in coordinates.values()),
            "Incomplete critic/actor trace coverage")
    summaries = {}
    for episode in episode_rows:
        episode_id = f"seed-{episode['seed']}"
        points = []
        for step in range(1, count * rounds + 1):
            raw = accumulated[episode_id, step]
            require(all(s["count"] == episode["length"] for s in raw.values()), "Incomplete critic metric coverage")
            points.append({"round_index": (step - 1) // count + 1,
                           "critic_update_in_round": (step - 1) % count + 1,
                           "actor_updates": 4 * ((step - 1) // count),
                           "critic_updates": step, "critic_update_index": step, "measurement": "pre_update_minibatch", "metrics": {name: {
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
    run, count, rounds = runs[0], cell["critic_updates"], cell["rounds"]
    result, config = run["result"], run["config"]
    require(result.get("controller_seed") == cell["controller_seed"], "Wrong result controller seed")
    require(run.get("config_hash") == digest(config), "Run configuration checksum mismatch")
    params = result["alg_params"]
    expected = {"inner_rounds": rounds, "inner_rollout_horizon": 1, "inner_rollouts_per_round": 128,
                "inner_actor_updates_per_round": 4, "inner_critic_updates_per_round": count,
                "inner_batch_size": 256, "inner_temperature": 0.0, "inner_actor_entropy_mode": "tdmpc2_scaled",
                "inner_actor_initialization": "prior", "inner_critic_initialization": "prior",
                "inner_temperature_mode": "fixed", "inner_temperature_initialization": "fixed",
                "inner_execution_action": "mean", "inner_behavior_action": "policy_sample",
                "inner_finite_horizon": True, "inner_sac_critic_target": "reward_only"}
    require(all(params.get(k) == v and config["alg_params"].get(k) == v for k, v in expected.items()),
            "Wrong C/J learner schedule")
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
        record = {"cell_id": cell["cell_id"], "critic_updates": count, "rounds": rounds,
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
            require(value == rounds * {"actor_updates": 4, "critic_updates": count, "model_transitions": 128}[name],
                    f"Unexpected realized {name}")
            record[name + "_per_decision"], record[name + "_per_episode"] = value, value * row["length"]
        record["optimizer_updates_per_decision"] = (count + 4) * rounds
        record["optimizer_updates_per_episode"] = (count + 4) * rounds * row["length"]
        record["control_seconds_per_decision"] = record["control_seconds"] / row["length"]
        points = row.get("togo_round_summaries", [])
        require([r["round_index"] for r in points] == list(range(rounds + 1)), "Incomplete round probes")
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
    science = {k: v for k, v in params.items() if k not in ("inner_critic_updates_per_round", "inner_rounds")}
    compatibility = {"checkpoint": checkpoint, "protocol": {k: v for k, v in protocol.items() if k != "controller_seed"},
                     "alg_params_except_critic_updates_and_rounds": science, "runtime": code["runtime"], "probe": probe,
                     "scientific_source": source_identity(config["alg"], code["commit"], code["dirty"], code.get("source_sha256"))}
    return {"cell": cell, "rows": output, "prior": prior, "compatibility": compatibility,
            "identity": {"cell_id": cell["cell_id"], "critic_updates": count, "rounds": rounds,
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
    index = {(r["critic_updates"], r["rounds"], r["controller_seed"], r["seed"]): r for r in rows}
    require(len(rows) == 240 and len(index) == 240, "Incomplete or duplicate episode grid")
    numeric = ["return", "gain_vs_prior", "gain_vs_j1", "gain_vs_j2", "gain_vs_c32", "rounds", "length",
               "control_seconds", "control_seconds_per_decision", "probe_seconds", "probe_model_transitions",
               *[n + suffix for n in (*WORK, "optimizer_updates") for suffix in ("_per_decision", "_per_episode")]]
    for row in rows:
        count, rounds, c, seed = (row[k] for k in ("critic_updates", "rounds", "controller_seed", "seed"))
        row["prior_return"] = priors[0][seed][0]
        row["gain_vs_prior"] = row["return"] - row["prior_return"]
        for target_c in CRITIC_UPDATES:
            for target_j in ROUNDS:
                require(row["solver_seed"] == index[target_c, target_j, c, seed]["solver_seed"], "Unpaired controller RNG seeds")
        row["gain_vs_j1"] = row["return"] - index[count, 1, c, seed]["return"]
        row["gain_vs_j2"] = row["return"] - index[count, 2, c, seed]["return"]
        row["gain_vs_c32"] = row["return"] - index[32, rounds, c, seed]["return"]
    by_episode, summaries, probes, training, contrasts, availability = [], [], [], [], [], {}
    for count in CRITIC_UPDATES:
        for rounds in ROUNDS:
            group = [r for r in rows if r["critic_updates"] == count and r["rounds"] == rounds]
            keys = sorted(group[0]["model_metrics"])
            require(all(set(r["model_metrics"]) == set(keys) for r in group), "Diagnostic availability differs within C/J")
            availability[f"c{count}-j{rounds}"] = keys
            averaged = []
            for seed in seeds:
                selected = [index[count, rounds, c, seed] for c in controllers]
                averaged.append({"critic_updates": count, "rounds": rounds, "seed": seed,
                                 **{k: statistics.fmean(r[k] for r in selected) for k in numeric},
                                 "model_metrics": {k: statistics.fmean(r["model_metrics"][k] for r in selected) for k in keys}})
            by_episode.extend(averaged)
            summaries.append({"critic_updates": count, "rounds": rounds, "reused": count == 32 or rounds == 1,
                              "metrics": _summaries(averaged, numeric, resamples),
                              "model_metrics": _summaries([r["model_metrics"] for r in averaged], keys, resamples),
                              "controller_mean_returns": {str(c): statistics.fmean(index[count, rounds, c, s]["return"] for s in seeds) for c in controllers},
                              "raw_episode_return_std": statistics.stdev(r["return"] for r in group)})
            for boundary in range(rounds + 1):
                episode_metrics = []
                for seed in seeds:
                    selected = [index[count, rounds, c, seed]["togo_round_summaries"][boundary]["metrics"] for c in controllers]
                    probe_keys = set(selected[0])
                    require(all(set(p) == probe_keys for p in selected), "Probe metric availability differs")
                    episode_metrics.append({k: statistics.fmean(p[k]["mean"] for p in selected) for k in probe_keys})
                require(all(set(p) == set(episode_metrics[0]) for p in episode_metrics), "Probe metrics differ across episodes")
                probes.append({"configured_critic_updates": count, "configured_rounds": rounds,
                               "round_index": boundary, "actor_updates": boundary * 4, "critic_updates": boundary * count,
                               "metrics": _summaries(episode_metrics, sorted(episode_metrics[0]), resamples)})
            for step in range(1, count * rounds + 1):
                episode_metrics = [{k: statistics.fmean(index[count, rounds, c, seed]["critic_training_steps"][step - 1]["metrics"][k]["mean"]
                                    for c in controllers) for k in TRAINING_METRICS} for seed in seeds]
                training.append({"configured_critic_updates": count, "configured_rounds": rounds,
                                 "round_index": (step - 1) // count + 1, "critic_update_in_round": (step - 1) % count + 1,
                                 "critic_updates": step, "critic_update_index": step, "actor_updates": 4 * ((step - 1) // count),
                                 "measurement": "pre_update_minibatch", "metrics": _summaries(episode_metrics, TRAINING_METRICS, resamples)})
    def contrast(kind, label, components):
        differences = [statistics.fmean(sum(sign * index[c, j, controller, s]["return"] for c, j, sign in components)
                                        for controller in controllers) for s in seeds]
        role = "primary" if label in ("C64: J2 minus J1", "J2: C64 minus C32") else "exploratory" if kind == "exploratory_round_scaling_interaction" else "reference"
        contrasts.append({"kind": kind, "role": role, "label": label, "components": components,
                          "summary": summarize(differences, resamples), "episode_differences": dict(zip(map(str, seeds), differences))})
    for count in CRITIC_UPDATES:
        for target, reference in ((2, 1),):
            contrast("rounds_within_critic_count", f"C{count}: J{target} minus J{reference}",
                     [(count, target, 1), (count, reference, -1)])
    for rounds in ROUNDS:
        contrast("critic_count_at_matched_rounds", f"J{rounds}: C64 minus C32", [(64, rounds, 1), (32, rounds, -1)])
    for target, reference in ((2, 1),):
        contrast("exploratory_round_scaling_interaction", f"C64 minus C32: J{target} minus J{reference}",
                 [(64, target, 1), (64, reference, -1), (32, target, -1), (32, reference, 1)])
    return {"schema": SCHEMA, "status": "complete", "attempt_label": campaign["attempt_label"],
            "campaign_sha256": digest(campaign), "critic_updates": CRITIC_UPDATES, "rounds": ROUNDS,
            "seeds": seeds, "controller_seeds": controllers,
            "aggregation": "Average decisions within episode for diagnostics; average controller repetitions within environment seed; weight twenty seeds equally",
            "interval": {"method": "paired environment-episode percentile bootstrap", "resamples": resamples, "confidence": .95, "seed": 20260912},
            "return_units": "Undiscounted environment rewards in closed-loop mean-action episodes of at most 500 decisions",
            "diagnostic_scope": "Controller-specific states. Different C/J settings may visit different states and collect different subsequent replay. Training curves use fresh pre-update sampled minibatches and sampled bootstrap targets, not a fixed held-out evaluation set. Critic loss is two-hot cross-entropy; decoded TD error is in reward/Q units. Each round collects 128 additional transitions, then performs C critic updates and four actor updates.",
            "timing_semantics": "Recorded control time already excludes model probe time and warmup; no further subtraction. All C32 and C64 J1 timing is historical, not paired under identical machine load.",
            "protocol": {"rollouts_per_round": 128, "actor_updates_per_round": 4, "rollout_horizon": 1,
                         "minibatch_size": 256, "inner_alpha": 0., "actor_initialization": "prior", "critic_initialization": "prior"},
            "training_trace_storage": "Full gzip traces retained in sealed source bundles; per-episode/per-round/per-step summaries and trace paths/checksums are portable in this report",
            "prior": {"summary": summarize([priors[0][s][0] for s in seeds], resamples), "episodes": values[0]["prior"]},
            "scientific_identity": values[0]["compatibility"], "identities": [v["identity"] for v in values],
            "model_metric_availability": availability, "summaries": summaries, "contrasts": contrasts,
            "model_probes": probes, "critic_training": training, "episode_averages": by_episode, "paired_rows": rows}


def render_html(report):
    compact = {k: v for k, v in report.items() if k not in ("paired_rows", "episode_averages", "identities", "scientific_identity")}
    raw = base64.b64encode(gzip.compress(json.dumps(report, separators=(",", ":"), allow_nan=False).encode(), mtime=0)).decode()
    data = json.dumps(compact, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return """<!doctype html><meta charset="utf-8"><title>C64 rounds and episode control</title>
<style>body{font:15px system-ui;max-width:1180px;margin:28px auto;color:#18273c;background:#f6f8fa}section{background:white;padding:20px;margin:15px 0;border:1px solid #ddd;border-radius:10px}select,button{font:inherit;padding:5px;max-width:100%}svg{width:100%;height:360px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:right;border-bottom:1px solid #eee}td:first-child,th:first-child{text-align:left}small{color:#53647b}</style>
<h1>C64 round scaling with C32 references</h1><p id="status"></p>
<section><b>Full-episode return and diagnostics</b><p><select id="metric"></select> against <select id="axis"></select></p><svg id="chart" viewBox="0 0 1050 360"></svg><small id="scope"></small><p id="constant"></p></section>
<section><b>Critic training trajectory</b><p><select id="trainingmetric"></select> against cumulative critic update index (measurement before update), for <select id="trainrounds"><option>1</option><option selected>2</option></select> rounds</p><svg id="training" viewBox="0 0 1050 360"></svg><small>Each loss/TD-error measurement uses the minibatch before that update, not a post-update held-out error. Each curve uses its own controller states. Every round collects new data and performs actor updates between critic-training phases. Downloaded points retain round and within-round indices.</small></section>
<section><b>Model return before and after adaptation</b><p><select id="probemetric"></select> against actor updates, for <select id="proberounds"><option>1</option><option selected>2</option></select> rounds</p><svg id="probes" viewBox="0 0 1050 360"></svg><small>Initial and every completed round: actor updates 0,4,…,4J; critic updates 0,C,…,CJ. Sampled-policy probes use controller-specific states.</small></section>
<section><b>Paired complete-panel results</b><div id="summary"></div></section>
<section><b>Paired comparisons</b><div id="contrasts"></div><small>95% percentile intervals use 2,000 paired episode-cluster resamples without multiplicity adjustment.</small></section>
<section><button id="download">Download complete report JSON (240 paired episodes, per-round critic-step summaries and source trace checksums)</button><p id="timing"></p><p id="storage"></p></section>
<script id="report-data" type="application/json">""" + data + """</script><script id="raw-gzip" type="application/octet-stream">""" + raw + """</script><script>
const D=JSON.parse(document.getElementById('report-data').textContent),S=D.summaries||[],fmt=x=>Number(x).toFixed(3),range=m=>`${fmt(m.mean)} [${fmt(m.ci95_low)}, ${fmt(m.ci95_high)}]`;
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const table=(h,r)=>'<table><tr>'+h.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>'+r.map(a=>'<tr>'+a.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</table>';
const options=(id,items)=>{document.getElementById(id).innerHTML=items.map(([v,l])=>`<option value="${esc(v)}">${esc(l)}</option>`).join('')};
document.getElementById('status').textContent=`${D.attempt_label}: ${D.status}. ${D.complete_cells??12}/12 cells. Three controller repetitions × twenty environment episodes per C/J. All C32 and C64 J1 are reused exactly.`;
function draw(id,groups,zero=false,prior=null){let P=groups.flatMap(g=>g.points);if(!P.length){document.getElementById(id).innerHTML='';return}let xmax=Math.max(...P.map(p=>p.x),.001)*1.08,ymin=Math.min(...P.map(p=>p.m.ci95_low),...(zero?[0]:[]),...(prior===null?[]:[prior])),ymax=Math.max(...P.map(p=>p.m.ci95_high),...(zero?[0]:[]),...(prior===null?[]:[prior])),pad=Math.max((ymax-ymin)*.12,.001);ymin-=pad;ymax+=pad;let X=x=>95+850*x/xmax,Y=y=>290-215*(y-ymin)/(ymax-ymin),z=[];for(let i=0;i<5;i++){let y=ymin+i*(ymax-ymin)/4;z.push(`<line x1="95" x2="970" y1="${Y(y)}" y2="${Y(y)}" stroke="#e2e6ed"/><text x="3" y="${Y(y)+4}" font-size="11">${y.toFixed(3)}</text>`);let x=i*xmax/4;z.push(`<text x="${X(x)-10}" y="330" font-size="11">${x.toFixed(xmax<1?3:1)}</text>`)}if(prior!==null)z.push(`<line x1="95" x2="970" y1="${Y(prior)}" y2="${Y(prior)}" stroke="#999" stroke-dasharray="4 3"/>`);groups.forEach((g,i)=>{let color=['#206dcd','#d37818','#298a62','#9759a3','#d54a59','#64748b','#13a2ac'][i%7];z.push(`<text x="${105+(i%4)*235}" y="${18+Math.floor(i/4)*20}" fill="${color}" font-size="13">${esc(g.label)}</text><polyline points="${g.points.map(p=>X(p.x)+','+Y(p.m.mean)).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`);g.points.forEach(p=>z.push(`<line x1="${X(p.x)}" x2="${X(p.x)}" y1="${Y(p.m.ci95_low)}" y2="${Y(p.m.ci95_high)}" stroke="${color}"/><circle cx="${X(p.x)}" cy="${Y(p.m.mean)}" r="4" fill="${color}"><title>${esc(g.label+' '+(p.label||''))}: ${range(p.m)}</title></circle>`))});document.getElementById(id).innerHTML=z.join('')}
if(D.status==='complete'){
options('metric',[...Object.keys(S[0].metrics).map(k=>['metrics/'+k,k]),...Array.from(new Set(S.flatMap(s=>Object.keys(s.model_metrics)))).sort().map(k=>['model_metrics/'+k,'Episode diagnostic: '+k])]);options('axis',[['rounds','rounds'],['critic_updates_per_decision','total critic updates / decision'],['actor_updates_per_decision','actor updates / decision'],['model_transitions_per_decision','model transitions / decision'],['control_seconds_per_decision','recorded control seconds / decision']]);options('trainingmetric',Object.keys(D.critic_training[0].metrics).map(k=>[k,k]));options('probemetric',Object.keys(D.model_probes[0].metrics).map(k=>[k,k]));document.getElementById('trainingmetric').value='td_error_abs_mean';document.getElementById('probemetric').value='togo_return_gain_vs_outer';
function main(){let [family,key]=document.getElementById('metric').value.split('/'),axis=document.getElementById('axis').value;draw('chart',D.critic_updates.map(c=>({label:'C'+c+' per round'+(c===32?' (historical)':''),points:S.filter(s=>s.critic_updates===c&&s[family][key]).map(s=>({x:s.metrics[axis].mean,m:s[family][key],label:'J'+s.rounds+(s.reused?' historical':'' )}))})),key.includes('gain'),key==='return'?D.prior.summary.mean:null)};
function train(){let key=document.getElementById('trainingmetric').value,j=Number(document.getElementById('trainrounds').value);draw('training',D.critic_updates.map(c=>({label:'C'+c+' J'+j,points:D.critic_training.filter(s=>s.configured_critic_updates===c&&s.configured_rounds===j).map(s=>({x:s.critic_updates,m:s.metrics[key],label:'round '+s.round_index+', update '+s.critic_update_in_round}))})))};
function probes(){let key=document.getElementById('probemetric').value,j=Number(document.getElementById('proberounds').value);draw('probes',D.critic_updates.map(c=>({label:'C'+c+' J'+j,points:D.model_probes.filter(s=>s.configured_critic_updates===c&&s.configured_rounds===j).map(s=>({x:s.actor_updates,m:s.metrics[key],label:'round '+s.round_index+', critic '+s.critic_updates}))})),key.includes('gain'))};
document.getElementById('metric').onchange=main;document.getElementById('axis').onchange=main;document.getElementById('trainingmetric').onchange=train;document.getElementById('trainrounds').onchange=train;document.getElementById('probemetric').onchange=probes;document.getElementById('proberounds').onchange=probes;main();train();probes();
document.getElementById('summary').innerHTML=table(['C / J','Return [95% CI]','Gain vs prior','Gain vs J1','Gain vs J2','Gain vs C32','Episode-mean SD','60-return SD'],S.map(s=>['C'+s.critic_updates+' J'+s.rounds+(s.reused?' (reused)':''),range(s.metrics.return),range(s.metrics.gain_vs_prior),range(s.metrics.gain_vs_j1),range(s.metrics.gain_vs_j2),range(s.metrics.gain_vs_c32),fmt(s.metrics.return.std),fmt(s.raw_episode_return_std)]));document.getElementById('contrasts').innerHTML=table(['Contrast','Difference [95% CI]'],D.contrasts.map(c=>[c.label,range(c.summary)]));
document.getElementById('scope').textContent=D.return_units+'. '+D.diagnostic_scope;document.getElementById('constant').textContent='Per round: 128 new model transitions, C critic updates, 4 actor updates. H1 and B256 fixed. Interaction comparisons are exploratory.';document.getElementById('timing').textContent=D.timing_semantics;document.getElementById('storage').textContent=D.training_trace_storage;
}else document.getElementById('summary').textContent='No scientific curves until all cells validate. Missing: '+JSON.stringify(D.missing||[]);
document.getElementById('download').onclick=async()=>{let bytes=Uint8Array.from(atob(document.getElementById('raw-gzip').textContent),c=>c.charCodeAt(0)),stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')),blob=await new Response(stream).blob(),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='c64-rounds-report.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script>"""


def inspect(campaign, loaded, campaign_file_sha256):
    missing = []
    for cell in campaign["cells"]:
        if cell["cell_id"] in loaded:
            continue
        try:
            receipt_path = Path(cell.get("merge_receipt") or Path(campaign["output_root"]) / "production" / cell["cell_id"] / "merge-completion.json")
            receipt = read(receipt_path)
            expected = {"status": "complete", "cell_id": cell["cell_id"], "critic_updates": cell["critic_updates"], "rounds": cell["rounds"],
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
    return {"complete_cells": len(loaded), "expected_cells": 12, "complete_workers": complete,
            "expected_workers": len(receipts), "missing": missing}


def publish_science(run, wandb, report, output):
    fingerprint = digest(report)
    if run.summary.get("comparison/report_sha256") == fingerprint:
        return
    for count in CRITIC_UPDATES:
        for axis in AXES:
            run.define_metric(f"compute/c{count}/{axis}")
            run.define_metric(f"episodes/c{count}/vs_{axis}/*", step_metric=f"compute/c{count}/{axis}")
        run.define_metric(f"episode_diagnostics/c{count}/*", step_metric=f"compute/c{count}/rounds")
        for rounds in ROUNDS:
            stem = f"c{count}/j{rounds}"
            run.define_metric(f"probes/{stem}/actor_updates")
            run.define_metric(f"model_probes/{stem}/*", step_metric=f"probes/{stem}/actor_updates")
            run.define_metric(f"training/{stem}/critic_update_index")
            run.define_metric(f"critic_training/{stem}/*", step_metric=f"training/{stem}/critic_update_index")
    for row in report["summaries"]:
        count = row["critic_updates"]
        values = {f"compute/c{count}/" + axis: row["metrics"][axis]["mean"] for axis in AXES}
        for axis in AXES:
            for name in ("return", "gain_vs_prior", "gain_vs_j1", "gain_vs_j2", "gain_vs_c32"):
                for stat in ("mean", "std", "ci95_low", "ci95_high"):
                    values[f"episodes/c{count}/vs_{axis}/{name}/{stat}"] = row["metrics"][name][stat]
        for name, summary in row["model_metrics"].items():
            for stat in ("mean", "ci95_low", "ci95_high"):
                values[f"episode_diagnostics/c{count}/{name}/{stat}"] = summary[stat]
        run.log(values)
    for row in report["model_probes"]:
        stem = f"c{row['configured_critic_updates']}/j{row['configured_rounds']}"
        run.log({f"probes/{stem}/actor_updates": row["actor_updates"],
                 f"probes/{stem}/critic_updates": row["critic_updates"], f"probes/{stem}/round_index": row["round_index"],
                 **{f"model_probes/{stem}/{name}/{stat}": summary[stat]
                    for name, summary in row["metrics"].items() for stat in ("mean", "ci95_low", "ci95_high")}})
    for row in report["critic_training"]:
        stem = f"c{row['configured_critic_updates']}/j{row['configured_rounds']}"
        run.log({f"training/{stem}/critic_update_index": row["critic_updates"],
                 f"training/{stem}/round_index": row["round_index"], f"training/{stem}/critic_update_in_round": row["critic_update_in_round"],
                 **{f"critic_training/{stem}/{name}/{stat}": summary[stat]
                    for name, summary in row["metrics"].items() for stat in ("mean", "ci95_low", "ci95_high")}})
    for contrast in report["contrasts"]:
        label = contrast["label"].replace(":", "").replace(" ", "_").lower()
        for stat in ("mean", "std", "ci95_low", "ci95_high"):
            run.summary[f"contrasts/{label}/{stat}"] = contrast["summary"][stat]
        run.summary[f"contrasts/{label}/role"] = contrast["role"]
    run.log({"comparison/report": wandb.Html(str(output / "report.html"), inject=False)})
    artifact = wandb.Artifact("c64-rounds-" + digest(report["attempt_label"])[:16], type="c64-rounds-comparison",
                              metadata={"schema": SCHEMA, "report_sha256": fingerprint})
    for name in ("report.json", "report.html", "campaign.json"):
        artifact.add_file(str(output / name), name=name)
    for identity in report["identities"]:
        if identity.get("model_series"):
            require(Path(identity["model_series"]).is_dir(), "Missing complete model-series artifact")
            artifact.add_dir(identity["model_series"], name="model-series/" + identity["cell_id"])
    run.log_artifact(artifact)
    run.summary.update({"comparison/report_sha256": fingerprint, "comparison/prior_return": report["prior"]["summary"]["mean"],
                        "comparison/training_measurement": "pre_update_minibatch", "comparison/historical_rows": 180})


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
                             name="C64 J1/J2 with C32 references | " + campaign["attempt_label"],
                             job_type="c64-rounds-comparison", tags=["c64-rounds", "full-episodes", "complete-panel"],
                             config={"comparison_schema": SCHEMA, "attempt_label": campaign["attempt_label"],
                                     "campaign_sha256": digest(campaign), "owner": args.owner, "critic_updates": CRITIC_UPDATES, "rounds": ROUNDS,
                                     "checkpoint_sha256": CHECKPOINT, "source_run": "rwgao_b-brown-university/ambi/mey3rxj8",
                                     "checkpoint_step": 200000, "seeds": seeds, "controller_seeds": controllers,
                                     "protocol": "Full500 mean decisions; J1/J2, H1 N128 B256 A4 per round, C32/C64 per round; prior weights, alpha0 and saved Q scale",
                                     "science_publication": "Only after all 12 cells validate"})
            write(receipt_path, identity)
            run.define_metric("progress/elapsed_seconds")
            run.define_metric("progress/*", step_metric="progress/elapsed_seconds")
            run.summary["comparison/status"] = "running"
            run.log({"progress/elapsed_seconds": 0., "progress/complete_cells": 0,
                     "progress/expected_cells": 12, "progress/complete_workers": 0,
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
                progress.update(status="complete", stop_reason="All 12 complete cells published")
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
