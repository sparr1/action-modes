"""Paired, round-axis diagnostics kept separate from checkpoint evaluation curves.

This module imports neither the learner nor W&B. Publication is explicit and
accepts an injected SDK for offline testing. Rows remain intact in the compressed
bundle and the portable HTML; every reduction uses the declared sampling units.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
from html import escape
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import time


SCHEMA_VERSION = 1
SCOPES = {"common_prior_roots", "controller_episode", "common_prior_roots_model"}
COORDINATES = ("episode_id", "root_id", "solver_repeat", "rollout_repeat", "round_index")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _id(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        raise ValueError(f"{name} must be a nonempty string or integer")
    return value


def _coord(row):
    for key in COORDINATES[:2]:
        _id(row.get(key), key)
    for key in COORDINATES[2:]:
        _integer(row.get(key), key)
    return _json([row[key] for key in COORDINATES])


def _coordinates(expected):
    if isinstance(expected, list):
        result = [{key: row[key] for key in COORDINATES} for row in expected]
    elif isinstance(expected, dict):
        roots = expected.get("roots", [])
        solvers = _integer(expected.get("solver_repeats"), "solver_repeats", 1)
        rollouts = _integer(expected.get("rollout_repeats"), "rollout_repeats", 1)
        rounds = expected.get("rounds", [])
        if not rounds or len(set(rounds)) != len(rounds):
            raise ValueError("Expected rounds must be nonempty and unique")
        result = [{"episode_id": root["episode_id"], "root_id": root["root_id"],
                   "solver_repeat": solver, "rollout_repeat": rollout, "round_index": round_index}
                  for root in roots for solver in range(solvers)
                  for rollout in range(rollouts) for round_index in rounds]
    else:
        raise ValueError("expected must be a coordinate list or compact sampling plan")
    if not result:
        raise ValueError("Expected sampling panel must not be empty")
    keys = [_coord(row) for row in result]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate expected coordinate")
    return result


def _sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def _mean(values):
    return statistics.fmean(values) if values else None


def _quantile(values, p):
    index = (len(values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _metric_summary(rows, name, resamples, seed):
    # A source episode has equal weight regardless of how many roots it yielded.
    solvers = defaultdict(list)
    for row in rows:
        value = row["metrics"].get(name)
        if value is not None:
            solvers[_json([row["episode_id"], row["root_id"], row["solver_repeat"]])].append(value)
    roots = defaultdict(list)
    for key, values in solvers.items():
        episode, root, _ = json.loads(key)
        roots[_json([episode, root])].append(statistics.fmean(values))
    episodes = defaultdict(list)
    for key, values in roots.items():
        episode, _ = json.loads(key)
        episodes[_json(episode)].append(statistics.fmean(values))
    episode_values = [statistics.fmean(episodes[key]) for key in sorted(episodes)]
    rng = random.Random(seed)
    samples = sorted(statistics.fmean(rng.choices(episode_values, k=len(episode_values)))
                     for _ in range(resamples)) if len(episode_values) > 1 else []
    return {"mean": _mean(episode_values), "episode_std": _sd(episode_values),
            "ci95_low": _quantile(samples, .025) if samples else None,
            "ci95_high": _quantile(samples, .975) if samples else None,
            "episodes": len(episodes), "roots": len(roots), "solvers": len(solvers),
            "rollouts": sum(map(len, solvers.values())),
            "within_solver_rollout_std_mean": _mean([v for x in solvers.values() if (v := _sd(x)) is not None]),
            "within_root_solver_std_mean": _mean([v for x in roots.values() if (v := _sd(x)) is not None]),
            "within_episode_root_std_mean": _mean([v for x in episodes.values() if (v := _sd(x)) is not None]),
            "episode_means": [{"episode_id": json.loads(key), "mean": statistics.fmean(episodes[key])}
                              for key in sorted(episodes)]}


def build_diagnostic_record(identity, rows, expected, *, attempt_label=None, timing=None,
                            bootstrap_resamples=2000, bootstrap_seed=0):
    """Validate a panel and compute hierarchy-aware per-round summaries.

    Missing coordinates or null/missing metrics produce an explicit incomplete
    record, which may be inspected locally but cannot be published to W&B.
    Unexpected/duplicate coordinates and mixed counter schedules are errors.
    """
    identity = json.loads(_json(identity))
    if identity.get("scope") not in SCOPES:
        raise ValueError(f"Diagnostic scope must be one of {sorted(SCOPES)}")
    checkpoint = identity.get("checkpoint", {})
    if not re.fullmatch(r"[a-f0-9]{64}", str(checkpoint.get("sha256", ""))):
        raise ValueError("Diagnostic checkpoint SHA256 is required")
    _integer(checkpoint.get("step", checkpoint.get("training_decisions")), "checkpoint step")
    for key in ("setting", "protocol", "code"):
        if not isinstance(identity.get(key), dict) or not identity[key]:
            raise ValueError(f"Diagnostic identity requires nonempty {key}")
    attempt_label = attempt_label if attempt_label is not None else identity.pop("attempt", None)
    if not isinstance(attempt_label, str) or not attempt_label.strip():
        raise ValueError("An explicit nonempty attempt_label is required")
    _integer(bootstrap_resamples, "bootstrap_resamples", 1)
    _integer(bootstrap_seed, "bootstrap_seed")
    coordinates = _coordinates(expected)
    expected_by_key = {_coord(row): row for row in coordinates}
    rows = json.loads(_json(list(rows)))
    seen, schedules, metric_names = set(), {}, set()
    if isinstance(expected, dict):
        metric_names.update(expected.get("metrics", []))
    for row in rows:
        key = _coord(row)
        if key in seen:
            raise ValueError("Duplicate diagnostic coordinate: " + key)
        if key not in expected_by_key:
            raise ValueError("Unexpected diagnostic coordinate: " + key)
        seen.add(key)
        for counter in ("actor_updates", "critic_updates"):
            _integer(row.get(counter), counter)
        if "decision_index" in row:
            _integer(row["decision_index"], "decision_index")
        counts = (row["actor_updates"], row["critic_updates"])
        previous = schedules.setdefault(row["round_index"], counts)
        if previous != counts:
            raise ValueError("Inconsistent update counts within round")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Rows require a nonempty metrics mapping")
        for name, value in metrics.items():
            if not isinstance(name, str) or not name or value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("Metrics must contain finite numeric values or null")
        metric_names.update(metrics)
    ordered = sorted(schedules)
    if any(any(b < a for a, b in zip(schedules[left], schedules[right]))
           for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Update counts decrease across rounds")
    missing = [expected_by_key[key] for key in sorted(set(expected_by_key) - seen)]
    missing_metrics = [{**{key: row[key] for key in COORDINATES}, "metrics": absent}
                       for row in rows
                       if (absent := sorted(name for name in metric_names if row["metrics"].get(name) is None))]
    summaries = []
    for round_index in sorted({row["round_index"] for row in coordinates}):
        group = [row for row in rows if row["round_index"] == round_index]
        counters = schedules.get(round_index, (None, None))
        summaries.append({"round_index": round_index, "actor_updates": counters[0], "critic_updates": counters[1],
                          "metrics": {name: _metric_summary(group, name, bootstrap_resamples, bootstrap_seed)
                                      for name in sorted(metric_names)}})
    record = {"schema_version": SCHEMA_VERSION, "kind": "ambi_return_diagnostics",
              "identity": identity, "attempt_label": attempt_label,
              "series_id": _digest({"identity": identity, "attempt_label": attempt_label})[:32],
              "status": "incomplete" if missing or missing_metrics else "complete",
              "expected": coordinates, "missing_coordinates": missing, "missing_metrics": missing_metrics,
              "rows": sorted(rows, key=_coord), "summaries": summaries, "timing": timing or {},
              "aggregation": {"order": ["rollout_repeat", "solver_repeat", "root_id", "episode_id"],
                              "episode_weighting": "equal", "interval": "episode_cluster_percentile_bootstrap",
                              "confidence": .95, "bootstrap_resamples": bootstrap_resamples,
                              "bootstrap_seed": bootstrap_seed,
                              "single_episode_interval": "unavailable"}}
    record["record_sha256"] = _digest(record)
    return record


def _validate_record(record):
    if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != "ambi_return_diagnostics":
        raise ValueError("Unsupported diagnostic record")
    if record.get("record_sha256") != _digest({k: v for k, v in record.items() if k != "record_sha256"}):
        raise ValueError("Diagnostic record checksum mismatch")
    return record


def _safe_artifact(name):
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or str(path) in (".", ""):
        raise ValueError("Artifact names must be safe relative paths")
    return path


def write_diagnostic_bundle(directory, record, *, artifact_files=None, elapsed_before_bundle=None):
    """Write a new immutable result directory; never replace an existing result.

    ``elapsed_before_bundle`` enables finalization timing in the saved record;
    the caller's record is unchanged. Timings include serialization and report
    generation, stopping before their final metadata is persisted. The separate
    completion receipt includes that final persistence work as well.
    """
    start = time.perf_counter()
    _validate_record(record)
    if elapsed_before_bundle is not None:
        if (isinstance(elapsed_before_bundle, bool) or not isinstance(elapsed_before_bundle, (int, float))
                or not math.isfinite(elapsed_before_bundle) or elapsed_before_bundle < 0):
            raise ValueError("elapsed_before_bundle must be finite and nonnegative")
        record = json.loads(_json(record))
    sources = {str(_safe_artifact(name)): Path(path).resolve() for name, path in (artifact_files or {}).items()}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {key: value for key, value in record.items() if key != "rows"}
    with gzip.open(directory / "paired-rows.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in record["rows"]:
            handle.write(_json(row) + "\n")
    manifest["rows_file"] = "paired-rows.jsonl.gz"
    manifest["rows_sha256"] = _file_digest(directory / manifest["rows_file"])
    manifest["artifact_files"] = {}
    for name, source in sources.items():
        target = directory / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest["artifact_files"]["artifacts/" + name] = _file_digest(target)
    (directory / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    serialization_seconds = time.perf_counter() - start
    report_start = time.perf_counter()
    render_diagnostic_html(record, directory / "report.html")
    if elapsed_before_bundle is not None:
        report_generation_seconds = time.perf_counter() - report_start
        record["timing"].update(calibration_seconds_before_bundle=float(elapsed_before_bundle),
                                bundle_serialization_seconds=serialization_seconds,
                                report_generation_seconds=report_generation_seconds,
                                total_elapsed_seconds=elapsed_before_bundle + time.perf_counter() - start)
        record["record_sha256"] = _digest({key: value for key, value in record.items() if key != "record_sha256"})
        manifest.update({key: value for key, value in record.items() if key != "rows"})
        (directory / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        # These files were just created by this call and have not been returned
        # or published yet. Keep the final HTML payload and manifest identical.
        (directory / "report.html").write_text(render_diagnostic_html(record), encoding="utf-8")
        completion = {"record_sha256": record["record_sha256"],
                      "bundle_total_seconds": time.perf_counter() - start,
                      "total_elapsed_seconds": elapsed_before_bundle + time.perf_counter() - start,
                      "includes_final_metadata_persistence": True}
        (directory / "completion.json").write_text(_json(completion) + "\n", encoding="utf-8")
    return directory


def read_diagnostic_bundle(directory):
    """Read and hash-verify all paired measurements and referenced artifacts."""
    directory = Path(directory).expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    row_path = directory / _safe_artifact(manifest["rows_file"])
    if not row_path.resolve().is_relative_to(directory) or row_path.is_symlink() or _file_digest(row_path) != manifest["rows_sha256"]:
        raise ValueError("Diagnostic rows checksum mismatch")
    for name, digest in manifest.get("artifact_files", {}).items():
        path = directory / _safe_artifact(name)
        if not path.resolve().is_relative_to(directory) or _file_digest(path) != digest:
            raise ValueError("Diagnostic artifact checksum mismatch")
    record = {k: v for k, v in manifest.items() if k not in ("rows_file", "rows_sha256", "artifact_files")}
    with gzip.open(row_path, "rt", encoding="utf-8") as handle:
        record["rows"] = [json.loads(line) for line in handle if line.strip()]
    _validate_record(record)
    completion_path = directory / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("record_sha256") != record["record_sha256"]:
            raise ValueError("Diagnostic completion receipt does not match the record")
    report_path = directory / "report.html"
    if report_path.exists() and extract_diagnostic_html_data(report_path.read_text(encoding="utf-8")) != record:
        raise ValueError("Diagnostic HTML does not match the paired measurements")
    return record


_HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>body{font:15px system-ui,sans-serif;margin:32px auto;padding:0 24px;max-width:1180px;color:#172b3a;background:#f6f8fa}h1{font-size:26px}section{background:white;padding:20px;margin:20px 0;border:1px solid #d8e0e5;border-radius:10px}label{margin-right:16px}select,button{padding:6px}svg{width:100%;height:330px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{text-align:left;border-bottom:1px solid #eee;padding:6px}pre{overflow:auto;max-height:360px}#table{overflow:auto;max-height:400px}.muted{color:#516574}</style>
<h1>__TITLE__</h1><p id="identity"></p><p id="status"></p>
<section><label>Metric <select id="metric"></select></label><label>Axis <select id="axis"><option value="actor_updates">Actor updates</option><option value="critic_updates">Critic updates</option><option value="round_index">Round</option></select></label><svg id="plot" viewBox="0 0 1100 330" role="img" aria-label="Diagnostic means and episode-cluster confidence intervals"></svg><p class="muted">Equal weight per source episode; repetitions, solvers, and roots are averaged in that order. Bars show 95% episode-cluster bootstrap intervals. A single episode has no interval.</p><div id="summary"></div></section>
<section><h2>Complete paired measurements</h2><label>Root <select id="root"><option value="">All roots</option></select></label><button id="download">Download complete JSON</button><p id="count"></p><div id="table"></div></section><section><h2>Protocol, timing and completeness</h2><pre id="metadata"></pre></section>
<script id="diagnostic-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('diagnostic-data').textContent),$=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
function option(parent,v,t=v){const e=document.createElement('option');e.value=v;e.textContent=t;parent.append(e)}
const metrics=Array.from(new Set(data.rows.flatMap(r=>Object.keys(r.metrics)))).sort();metrics.forEach(m=>option($('metric'),m));
Array.from(new Set(data.rows.map(r=>JSON.stringify([r.episode_id,r.root_id])))).sort().forEach(r=>option($('root'),r));
$('identity').textContent=`${data.identity.scope} · checkpoint ${data.identity.checkpoint.step??data.identity.checkpoint.training_decisions} · attempt ${data.attempt_label}`;
$('status').textContent=`${data.status.toUpperCase()} · ${data.rows.length} of ${data.expected.length} expected paired measurements`;
$('metadata').textContent=JSON.stringify({identity:data.identity,aggregation:data.aggregation,timing:data.timing,missing_coordinates:data.missing_coordinates,missing_metrics:data.missing_metrics},null,2);
function table(parent,heads,rows){parent.replaceChildren();const t=document.createElement('table');const h=t.insertRow();heads.forEach(x=>{const c=document.createElement('th');c.textContent=x;h.append(c)});rows.forEach(r=>{const tr=t.insertRow();r.forEach(x=>{tr.insertCell().textContent=x===null||x===undefined?'—':String(x)})});parent.append(t)}
function svg(tag,attrs){const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);$('plot').append(e);return e}
function update(){const m=$('metric').value,a=$('axis').value,points=data.summaries.filter(r=>r[a]!==null&&r.metrics[m]?.mean!==null),values=points.flatMap(r=>[r.metrics[m].mean,r.metrics[m].ci95_low,r.metrics[m].ci95_high]).filter(x=>x!==null);$('plot').replaceChildren();if(values.length){let lo=Math.min(...values),hi=Math.max(...values);const pad=Math.max((hi-lo)*.12,.01);lo-=pad;hi+=pad;const xmax=Math.max(1,...points.map(r=>r[a])),x=v=>70+v/xmax*980,y=v=>285-(v-lo)/(hi-lo)*245;for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;svg('line',{x1:70,x2:1050,y1:y(v),y2:y(v),stroke:'#e0e6eb'});svg('text',{x:3,y:y(v)+4,fill:'#516574','font-size':12}).textContent=v.toPrecision(4)}points.forEach(r=>{const s=r.metrics[m],px=x(r[a]);if(s.ci95_low!==null)svg('line',{x1:px,x2:px,y1:y(s.ci95_low),y2:y(s.ci95_high),stroke:'#247c91','stroke-width':3});svg('circle',{cx:px,cy:y(s.mean),r:5,fill:'#125a75'});svg('text',{x:px,y:312,'text-anchor':'middle','font-size':12}).textContent=r[a]})}table($('summary'),['Round','Actor updates','Mean','Episode SD','95% low','95% high','Episodes'],points.map(r=>[r.round_index,r.actor_updates,...['mean','episode_std','ci95_low','ci95_high','episodes'].map(k=>r.metrics[m][k])]));const rows=data.rows.filter(r=>!$('root').value||JSON.stringify([r.episode_id,r.root_id])===$('root').value);$('count').textContent=`${rows.length} rows; the complete dataset is embedded and downloadable.`;table($('table'),['Episode','Root','Solver','Rollout','Round','Actor updates',m],rows.map(r=>[r.episode_id,r.root_id,r.solver_repeat,r.rollout_repeat,r.round_index,r.actor_updates,r.metrics[m]]))}
['metric','axis','root'].forEach(id=>$(id).addEventListener('change',update));$('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='return-diagnostics.json';a.click();URL.revokeObjectURL(url)});update();
</script></html>'''


def render_diagnostic_html(record, output=None, *, title="AMBI to-go return calibration"):
    """Return portable HTML, optionally writing a new report without replacement."""
    if isinstance(record, (str, Path)):
        record = read_diagnostic_bundle(record)
    _validate_record(record)
    encoded = _json(record).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = re.sub(r"__(TITLE|DATA)__", lambda m: {"TITLE": escape(title), "DATA": encoded}[m.group(1)], _HTML)
    if output is not None:
        with Path(output).open("x", encoding="utf-8") as handle:
            handle.write(html)
    return html


def extract_diagnostic_html_data(html):
    match = re.search(r'<script id="diagnostic-data" type="application/json">(.*?)</script>', html, re.S)
    if match is None:
        raise ValueError("No diagnostic payload in HTML")
    return _validate_record(json.loads(match.group(1)))


def diagnostic_history(record):
    """Produce exact W&B rows on round/update axes, never checkpoint axes."""
    _validate_record(record)
    if record["status"] != "complete":
        raise ValueError("Incomplete diagnostic panels cannot be published")
    history = []
    for summary in record["summaries"]:
        row = {"diagnostic/round": summary["round_index"], "diagnostic/actor_updates": summary["actor_updates"],
               "diagnostic/critic_updates": summary["critic_updates"]}
        for name, stats in summary["metrics"].items():
            for statistic, value in stats.items():
                if statistic != "episode_means" and value is not None:
                    row[f"diagnostic/{name}/{statistic}"] = value
        history.append(row)
    return history


@contextmanager
def _publication_lock(directory):
    with (directory / ".publication.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another publisher owns this diagnostic bundle") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def publish_diagnostic_bundle(directory, *, entity, project="ambi-inner-bench", mode="offline", wandb_module=None):
    """Publish an explicitly selected attempt. Uncertain writes require review.

    Completed publications are idempotent. A journal is created before SDK
    initialization; failures retain an uncertain receipt and are never blindly
    resumed, avoiding duplicated round history. This function makes no changes
    to existing checkpoint-axis evaluation runs or project workspaces.
    """
    directory = Path(directory).resolve()
    record = read_diagnostic_bundle(directory)
    history = diagnostic_history(record)
    completed_elapsed = record["timing"].get("total_elapsed_seconds", 0.)
    if (directory / "completion.json").is_file():
        completed_elapsed = json.loads((directory / "completion.json").read_text())["total_elapsed_seconds"]
    if mode not in ("offline", "online", "disabled"):
        raise ValueError("W&B mode must be offline, online, or disabled")
    if not isinstance(entity, str) or not entity or not project:
        raise ValueError("Explicit W&B entity and project are required")
    target = {"entity": entity, "project": project, "mode": mode,
              "series_id": record["series_id"], "record_sha256": record["record_sha256"]}
    with _publication_lock(directory):
        journal = directory / "publication.json"
        if journal.exists():
            previous = json.loads(journal.read_text())
            if previous.get("target") != target:
                raise ValueError("Publication target differs from existing receipt")
            if previous.get("status") == "complete":
                return previous
            raise ValueError("Previous publication is uncertain; inspect W&B before any retry")
        if wandb_module is None:
            import wandb as wandb_module
        start = time.perf_counter()
        receipt = {"target": target, "status": "uncertain"}
        journal.write_text(_json(receipt) + "\n")
        run = None
        try:
            run = wandb_module.init(entity=entity, project=project, id=record["series_id"],
                                    name=f"To-go {record['identity']['scope']} {record['attempt_label']}",
                                    mode=mode, resume="never", job_type="return-diagnostics",
                                    config={"diagnostic_schema": SCHEMA_VERSION, "diagnostic_identity": record["identity"],
                                            "attempt_label": record["attempt_label"], "aggregation": record["aggregation"]},
                                    tags=["return-diagnostics", record["identity"]["scope"]])
            for axis in ("round", "actor_updates", "critic_updates"):
                run.define_metric("diagnostic/" + axis)
            run.define_metric("diagnostic/*", step_metric="diagnostic/actor_updates")
            for row in history:
                run.log(row)
            artifact = wandb_module.Artifact("return-diagnostics-" + record["series_id"], type="return-diagnostics",
                                            metadata={"record_sha256": record["record_sha256"], "scope": record["identity"]["scope"]})
            manifest = json.loads((directory / "manifest.json").read_text())
            files = ["manifest.json", "paired-rows.jsonl.gz", "report.html", *manifest.get("artifact_files", {})]
            if (directory / "completion.json").is_file():
                files.append("completion.json")
            for name in files:
                artifact.add_file(str(directory / name), name=name)
            run.log_artifact(artifact)
            publication_before_finish = time.perf_counter() - start
            run.summary.update({"diagnostic/status": "complete", "diagnostic/paired_rows": len(record["rows"]),
                                "diagnostic/record_sha256": record["record_sha256"],
                                **{"runtime/" + key: value for key, value in record["timing"].items()
                                   if key != "publication_seconds"},
                                "runtime/publication_seconds_before_finish": publication_before_finish,
                                "runtime/total_seconds_before_publication_finish":
                                    completed_elapsed + publication_before_finish})
            run.finish()
            receipt.update(status="complete", publication_seconds=time.perf_counter() - start,
                           total_elapsed_seconds=completed_elapsed + time.perf_counter() - start,
                           wandb_path=f"{entity}/{project}/{record['series_id']}")
            journal.write_text(_json(receipt) + "\n")
            return receipt
        except BaseException:
            if run is not None:
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass
            raise


def record_from_model_bundle(bundle_path, selector, attempt_label, *, bootstrap_resamples=2000, bootstrap_seed=0):
    """Adapt retained per-round model rows without altering checkpoint curves."""
    bundle_path = Path(bundle_path)
    manifest = json.loads((bundle_path / "manifest.json").read_text())
    matches = [run for run in manifest.get("runs", []) if selector in (run.get("selector"), run.get("id"))]
    if len(matches) != 1:
        raise ValueError("Select exactly one run from the model bundle")
    run = matches[0]
    probe = run.get("togo_return_probe")
    rows = run.get("togo_probe_rows", [])
    if not probe or not rows:
        raise ValueError("Selected run has no retained to-go probe rows")
    if run.get("roots") and run.get("episodes"):
        raise ValueError("Export episode and shared-root diagnostics as separate scopes")
    rounds = list(range(int(run.get("config", {}).get("alg_params", run.get("config", {})).get("inner_rounds", 0)) + 1))
    if len(rounds) == 1 and any(row["round_index"] > 0 for row in rows):
        raise ValueError("Saved configuration lacks inner_rounds required for completeness validation")
    expected = []
    if run.get("roots"):
        for root in run["roots"]:
            # Legacy observation banks may lack episode provenance; never invent
            # independent episodes and confidence intervals from their roots.
            if "episode_id" not in root:
                raise ValueError("Shared-root diagnostic requires source episode_id provenance")
            expected.extend({"episode_id": root["episode_id"], "root_id": root["root_id"],
                             "solver_repeat": root.get("repeat", 0), "rollout_repeat": 0, "round_index": j}
                            for j in rounds)
        scope = "common_prior_roots_model"
    else:
        for episode in run.get("episodes", []):
            for decision in range(episode["length"]):
                expected.extend({"episode_id": episode["episode_id"],
                                 "root_id": f"seed-{episode['seed']}-decision-{decision}",
                                 "solver_repeat": 0, "rollout_repeat": 0, "round_index": j} for j in rounds)
        scope = "controller_episode"
    checkpoint = dict(manifest["checkpoint"])
    checkpoint.setdefault("step", checkpoint.get("training_decisions", checkpoint.get("train_steps",
                          checkpoint.get("metadata", {}).get("checkpoint", {}).get("step"))))
    identity = {"checkpoint": checkpoint, "setting": run.get("config", {}),
                "protocol": {**manifest["protocol"], "togo_return_probe": probe},
                "code": manifest.get("code", {}), "scope": scope}
    record = build_diagnostic_record(identity, rows, expected, attempt_label=attempt_label,
                                     timing=run.get("timing", {}), bootstrap_resamples=bootstrap_resamples,
                                     bootstrap_seed=bootstrap_seed)
    source_status = run.get("status", manifest.get("status"))
    if source_status not in ("complete", "completed", "finished", "success"):
        record["status"] = "incomplete"
        record["source_status"] = source_status
        record["record_sha256"] = _digest({key: value for key, value in record.items() if key != "record_sha256"})
    return record
