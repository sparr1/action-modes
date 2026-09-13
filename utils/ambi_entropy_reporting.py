"""Paired, episode-clustered reports for frozen-prior entropy diagnostics.

This module does not run a learner or simulator. Publication is explicit and
W&B is imported only when an online run is requested. Expected row identities
come from the preflighted campaign, never from the measurements themselves.
"""
from __future__ import annotations

from collections import defaultdict
import base64
import copy
import gzip
import html
import json
import math
from pathlib import Path
import statistics
from urllib.parse import quote


PANEL_FIELDS = (
    "source_run", "checkpoint_step", "checkpoint_sha256", "prefix_action_rule",
)
PAIR_FIELDS = ("episode_seed", "root_id", "solver_repetition", "rollout_repetition")
ROW_FIELDS = PANEL_FIELDS + PAIR_FIELDS + ("arm", "actor_updates")
SCHEMA_VERSION = 1
DEFAULT_CONTRASTS = (
    ("prior_recipe", "off"), ("squashed_matched", "off"),
    ("squashed_matched", "prior_recipe"), ("gaussian_control", "off"),
    ("gaussian_control", "squashed_matched"),
)


def _identity(row, fields=ROW_FIELDS):
    missing = [key for key in fields if key not in row]
    if missing:
        raise ValueError(f"Missing row identity fields: {missing}")
    result = tuple(row[key] for key in fields)
    try:
        hash(result)
    except TypeError as exc:
        raise ValueError("Row identities must be scalar and hashable") from exc
    for key in ("checkpoint_step", "episode_seed", "solver_repetition",
                "rollout_repetition", "actor_updates"):
        if key in fields:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
    if row.get("prefix_action_rule") not in ("mean", "sampled"):
        raise ValueError("prefix_action_rule must be mean or sampled")
    for key in ("source_run", "checkpoint_sha256", "root_id", "arm"):
        if key in fields and (not isinstance(row[key], str) or not row[key]):
            raise ValueError(f"{key} must be a nonempty string")
    return result


def _index(rows, *, measurements):
    index = {}
    for row in rows:
        key = _identity(row)
        if key in index:
            raise ValueError(f"Duplicate row identity: {dict(zip(ROW_FIELDS, key))}")
        if measurements:
            metrics = row.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError("Every measurement needs a nonempty metrics mapping")
            for name, value in metrics.items():
                if (not isinstance(name, str) or not name
                        or isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value)):
                    raise ValueError(f"Metric {name!r} must be a finite number")
        index[key] = row
    return index


def _episode_means(rows, metric):
    """Rollouts -> solvers -> roots -> equally weighted source episodes."""
    solvers = defaultdict(list)
    for row in rows:
        solvers[(row["episode_seed"], row["root_id"], row["solver_repetition"])].append(
            row["metrics"][metric])
    roots = defaultdict(list)
    for (seed, root, _), values in solvers.items():
        roots[(seed, root)].append(statistics.fmean(values))
    episodes = defaultdict(list)
    for (seed, _), values in roots.items():
        episodes[seed].append(statistics.fmean(values))
    return {str(seed): statistics.fmean(values) for seed, values in sorted(episodes.items())}


def _summaries(rows, *, resamples, seed, draws_cache):
    import numpy as np

    names = sorted(set().union(*(row["metrics"] for row in rows)))
    missing = {name: sum(name not in row["metrics"] for row in rows) for name in names}
    missing = {name: count for name, count in missing.items() if count}
    if missing:
        raise ValueError(f"Incomplete metric coverage within an actor-dose point: {missing}")
    summaries = {}
    for name in names:
        episodes = _episode_means(rows, name)
        values = np.asarray(list(episodes.values()), dtype=np.float64)
        n = len(values)
        ci = None
        if n > 1:
            cache_key = tuple(episodes)
            if cache_key not in draws_cache:
                draws_cache[cache_key] = np.random.default_rng(seed).integers(
                    n, size=(resamples, n))
            boot_means = values[draws_cache[cache_key]].mean(axis=1)
            ci = np.quantile(boot_means, [0.025, 0.975]).tolist()
        summaries[name] = {
            "mean": float(values.mean()),
            "episode_sd": float(values.std(ddof=1)) if n > 1 else None,
            "ci95": ci, "n_episodes": n, "episode_means": episodes,
        }
    return summaries


def _paired_rows(left, right):
    a = {_identity(row, PAIR_FIELDS): row for row in left}
    b = {_identity(row, PAIR_FIELDS): row for row in right}
    if a.keys() != b.keys():
        raise ValueError("Paired comparisons require identical episode/root/solver/rollout identities")
    names_a = set().union(*(r["metrics"] for r in left))
    names_b = set().union(*(r["metrics"] for r in right))
    # Arm-specific diagnostics are retained in their own series. Only metrics
    # measured on both sides define a paired difference.
    names = names_a & names_b
    if not names:
        raise ValueError("Paired comparisons have no shared metrics")
    out = []
    for key in sorted(a):
        row = {field: a[key][field] for field in ROW_FIELDS}
        row["metrics"] = {name: a[key]["metrics"][name] - b[key]["metrics"][name]
                          for name in sorted(names)}
        out.append(row)
    return out


def aggregate_entropy_rows(rows, *, expected_rows, final=True,
                           bootstrap_resamples=2000, bootstrap_seed=20260913,
                           required_metrics=(), contrasts=DEFAULT_CONTRASTS,
                           metadata=None):
    """Validate a declared panel, then compute within-checkpoint paired curves.

    ``expected_rows`` contains the complete preflighted ROW_FIELDS identities.
    Extra measurement/provenance fields remain in ``raw_rows``. Nonfinal calls
    deliberately produce coverage only, even when all rows happen to exist.
    """
    if (isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int)
            or bootstrap_resamples < 1):
        raise ValueError("bootstrap_resamples must be a positive integer")
    rows = copy.deepcopy(list(rows))
    expected_rows = list(expected_rows)
    expected = _index(expected_rows, measurements=False)
    observed = _index(rows, measurements=True)
    if not expected:
        raise ValueError("Expected coverage must be declared and nonempty")
    unexpected = observed.keys() - expected.keys()
    if unexpected:
        raise ValueError(f"Unexpected measurement identities: {len(unexpected)}")
    missing = expected.keys() - observed.keys()
    required_missing = [
        {**dict(zip(ROW_FIELDS, key)), "missing_metrics": sorted(set(required_metrics) - row["metrics"].keys())}
        for key, row in observed.items() if set(required_metrics) - row["metrics"].keys()
    ]
    if required_missing:
        raise ValueError(f"Missing required metrics: {required_missing[:3]}")
    coverage_by_panel = {}
    for key in sorted(expected):
        panel = key[:len(PANEL_FIELDS)]
        if panel not in coverage_by_panel:
            coverage_by_panel[panel] = {**dict(zip(PANEL_FIELDS, panel)), "expected": 0, "observed": 0}
        coverage_by_panel[panel]["expected"] += 1
        coverage_by_panel[panel]["observed"] += key in observed
    coverage = {
        "expected_rows": len(expected), "observed_rows": len(observed),
        "complete": not missing, "missing_rows": [dict(zip(ROW_FIELDS, key)) for key in sorted(missing)],
        "panels": list(coverage_by_panel.values()),
    }
    if final and missing:
        raise ValueError(f"Incomplete final panel: {len(observed)}/{len(expected)} rows; "
                         f"first missing={coverage['missing_rows'][:1]}")
    report = {
        "schema_version": SCHEMA_VERSION, "status": "complete" if final else "partial",
        "metadata": copy.deepcopy(metadata or {}), "coverage": coverage,
        "aggregation": "rollouts within solver, solvers within root, roots within source episode, episodes equally",
        "uncertainty": {"method": "paired source-episode cluster percentile bootstrap",
                        "resamples": bootstrap_resamples, "seed": bootstrap_seed,
                        "confidence": 0.95, "pointwise": True, "across_bank_pooling": False},
        "series": [], "contrasts": [], "initialization_deltas": [], "raw_rows": rows,
    }
    if not final:
        return report
    groups = defaultdict(list)
    for row in rows:
        groups[_identity(row, PANEL_FIELDS) + (row["arm"], row["actor_updates"])].append(row)
    cache = {}
    def summarize(group):
        return _summaries(group, resamples=bootstrap_resamples, seed=bootstrap_seed, draws_cache=cache)
    for key, group in sorted(groups.items()):
        panel, arm, updates = key[:-2], key[-2], key[-1]
        identity = {**dict(zip(PANEL_FIELDS, panel)), "arm": arm, "actor_updates": updates}
        report["series"].append({**identity, "metrics": summarize(group), "n_rows": len(group)})
        baseline = groups.get(panel + (arm, 0))
        if baseline is None:
            raise ValueError(f"Missing actual A0 initialization for {identity}")
        report["initialization_deltas"].append({
            **identity, "baseline_actor_updates": 0,
            "metrics": summarize(_paired_rows(group, baseline)),
        })
    for panel in sorted({key[:-2] for key in groups}):
        arms = {key[-2] for key in groups if key[:-2] == panel}
        updates = sorted({key[-1] for key in groups if key[:-2] == panel})
        for left, right in contrasts:
            if left not in arms or right not in arms:
                continue
            for update in updates:
                left_rows, right_rows = groups.get(panel + (left, update)), groups.get(panel + (right, update))
                if left_rows is None or right_rows is None:
                    raise ValueError(f"Unpaired actor dose {update}: {left} versus {right}")
                report["contrasts"].append({
                    **dict(zip(PANEL_FIELDS, panel)), "arm": left, "reference_arm": right,
                    "actor_updates": update, "metrics": summarize(_paired_rows(left_rows, right_rows)),
                })
    return report


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=False)


_REPORT_JS = r"""
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('entropy-chart-data').content.textContent);
  const esc = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const colors = ['#2463eb','#d54a23','#16866a','#9951bb','#967019','#c52b79'];
  const family = document.getElementById('family'), metric = document.getElementById('metric');
  const priority = ['real_mc_gain_vs_prior','model_gain_vs_prior','hybrid_model_gain_vs_prior',
                    'actor_action_exact_saturation_fraction','q_gradient_norm','entropy_gradient_norm',
                    'mean_action_displacement','fittedcritic_direct_model_rmse'];
  function plot(points, name) {
    const groups = new Map(), values = [0];
    for (const p of points) {
      const stat = p.metrics[name]; if (!stat) continue;
      const label = p.arm + (p.reference_arm ? ' − ' + p.reference_arm : '');
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push([p.actor_updates, stat]); values.push(...(stat.ci95 || [stat.mean]));
    }
    if (!groups.size) return '<p>Metric is unavailable in this panel.</p>';
    const xs = [...groups.values()].flatMap(g => g.map(p => p[0]));
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    let low = Math.min(...values), high = Math.max(...values);
    const margin = (high-low)*.08 || Math.max(1,Math.abs(high)*.08); low-=margin; high+=margin;
    const w=660,h=270,pad=48, x=v=>pad+(v-xmin)/Math.max(xmax-xmin,1)*(w-2*pad),
          y=v=>h-pad-(v-low)/(high-low)*(h-2*pad);
    let svg=`<svg role="img" aria-label="${esc(name)} versus actor updates" viewBox="0 0 ${w} ${h}"><path d="M ${pad} ${pad} V ${h-pad} H ${w-pad}" fill="none" stroke="#6b7280"/><path d="M ${pad} ${y(0)} H ${w-pad}" stroke="#d1d5db" stroke-dasharray="4 3"/>`;
    for (const v of [low,(low+high)/2,high]) svg+=`<text x="${pad-5}" y="${y(v)+4}" text-anchor="end" font-size="10">${esc(v.toPrecision(3))}</text>`;
    for (const v of [...new Set(xs)].sort((a,b)=>a-b)) svg+=`<text x="${x(v)}" y="${h-pad+17}" text-anchor="middle" font-size="11">${v}</text>`;
    const legends=[];
    [...groups.entries()].sort((a,b)=>a[0].localeCompare(b[0])).forEach(([label,group],i)=>{
      const color=colors[i%colors.length]; group.sort((a,b)=>a[0]-b[0]);
      svg+=`<polyline points="${group.map(([v,s])=>x(v)+','+y(s.mean)).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;
      for (const [v,s] of group) {
        if(s.ci95) svg+=`<path d="M ${x(v)} ${y(s.ci95[0])} V ${y(s.ci95[1])}" stroke="${color}" opacity=".55"/>`;
        const title=`${label}; A=${v}; mean=${s.mean.toPrecision(6)}; 95% CI=${JSON.stringify(s.ci95)}; episodes=${s.n_episodes}`;
        svg+=`<circle cx="${x(v)}" cy="${y(s.mean)}" r="3" fill="${color}"><title>${esc(title)}</title></circle>`;
      }
      legends.push(`<span style="color:${color}">${esc(label)}</span>`);
    });
    svg+=`<text x="${w/2}" y="${h-7}" text-anchor="middle" font-size="12">Actor updates</text></svg>`;
    return '<div class="legend">'+legends.join(' · ')+'</div>'+svg;
  }
  function render() {
    const panels = new Map();
    for (const point of data[family.value]) {
      if (!point.metrics[metric.value]) continue;
      const key=JSON.stringify([point.source_run,point.checkpoint_step,point.checkpoint_sha256,point.prefix_action_rule]);
      if(!panels.has(key))panels.set(key,[]);panels.get(key).push(point);
    }
    document.getElementById('charts').innerHTML=[...panels.entries()].sort().map(([key,points])=>{
      const [source,step,sha,rule]=JSON.parse(key);
      return `<section class="chart"><h2>${esc(source)} · ${step.toLocaleString()} · ${esc(rule)}</h2><p>${esc(metric.value)}</p>${plot(points,metric.value)}<small>Checkpoint SHA ${esc(sha.slice(0,12))}</small></section>`;
    }).join('') || '<p>No paired measurements for this metric.</p>';
  }
  function options() {
    const old=metric.value,names=[...new Set(data[family.value].flatMap(p=>Object.keys(p.metrics)))].sort();
    metric.replaceChildren(...names.map(name=>{const o=document.createElement('option');o.value=name;o.textContent=name;return o;}));
    metric.value=names.includes(old)?old:(priority.find(name=>names.includes(name))||names[0]||'');render();
  }
  if(data.status==='complete') {family.addEventListener('change',options);metric.addEventListener('change',render);options();}
  document.getElementById('download').addEventListener('click',async()=>{
    const status=document.getElementById('download-status');status.textContent='Decompressing…';
    try {
      const encoded=document.getElementById('entropy-report-data').content.textContent.trim();
      const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
      const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
      const blob=await new Response(stream).blob(),url=URL.createObjectURL(blob),a=document.createElement('a');
      a.href=url;a.download='entropy-paired-report.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status.textContent='Downloaded.';
    } catch(error) {status.textContent='Download unavailable in this browser. Use the report.json artifact.';}
  });
})();
"""


def write_entropy_report(report, output_dir):
    """Write full raw data and standalone HTML with lazy metric selection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path, rows_path, html_path = (output_dir / name for name in ("report.json", "paired-rows.jsonl", "report.html"))
    report_path.write_text(_json(report) + "\n", encoding="utf-8")
    rows_path.write_text("".join(_json(row)+"\n" for row in report["raw_rows"]), encoding="utf-8")
    cov = report["coverage"]
    body = [f'<h1>Frozen-prior entropy diagnostics</h1><p class="status">{html.escape(report["status"].upper())}: '
            f'{cov["observed_rows"]:,} / {cov["expected_rows"]:,} paired measurements</p>',
            '<p>Each source checkpoint and prefix action rule is shown separately. Rollouts are averaged within solver, '
            'solvers within root, roots within source episode, and episodes equally. Bars show pointwise 95% '
            'episode-cluster bootstrap intervals; they do not adjust for multiple comparisons.</p>',
            '<p>Predicted Q-based totals inherit the saved critic’s reward or soft-value semantics. '
            'Measured continuation returns contain no critic bootstrap at their cutoff. '
            'Saturation and gradient diagnostics describe their recorded sample population.</p>']
    if report["status"] != "complete":
        body.append('<p>INCOMPLETE: scientific curves are withheld until the declared panel is complete.</p>')
        body.append('<details><summary>Missing identities</summary><pre>' + html.escape(_json(cov["missing_rows"])) + '</pre></details>')
    else:
        body.append('<p class="controls"><label>Comparison <select id="family">'
                    '<option value="series">Measurements</option><option value="initialization_deltas">Paired change from actual A0</option>'
                    '<option value="contrasts">Paired entropy-arm differences</option></select></label> '
                    '<label>Metric <select id="metric"></select></label></p><main id="charts" class="charts"></main>'
                    '<noscript>Enable JavaScript for interactive charts; complete measurements are also in report.json.</noscript>')
    body.append('<details><summary>Provenance and coverage</summary><pre>' + html.escape(_json({"metadata": report["metadata"], "coverage": cov})) + '</pre></details>')
    # Full rows are compressed and only inflated on explicit download. The
    # small chart payload excludes raw measurements, avoiding tens of megabytes
    # of JSON parsing and hundreds of eagerly rendered metric grids on open.
    payload = base64.b64encode(gzip.compress(_json(report).encode("utf-8"), mtime=0)).decode("ascii")
    body.append('<p><button id="download">Download complete paired report JSON</button> <span id="download-status"></span></p>')
    body.append('<template id="entropy-report-data" data-encoding="gzip-base64">' + payload + '</template>')
    chart_payload = {key: report[key] for key in ("status", "series", "contrasts", "initialization_deltas")}
    body.append('<template id="entropy-chart-data">' + html.escape(_json(chart_payload)) + '</template>')
    body.append('<script>' + _REPORT_JS + '</script>')
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' \
        '<title>Frozen-prior entropy diagnostics</title><style>body{font:15px system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#17212f;background:#f8fafc}' \
        'h2{margin-top:40px;font-size:20px}.status{font-weight:700}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:16px}' \
        '.chart{background:white;border:1px solid #dbe2ea;border-radius:8px;padding:14px;min-width:0}summary{font-weight:600;overflow-wrap:anywhere}' \
        'svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}.legend{font-size:12px;margin-top:12px}' \
        'select,button{font:inherit;padding:7px;max-width:100%}.controls{display:flex;gap:16px;flex-wrap:wrap}.chart h2{margin:0;font-size:16px}</style><body>' + ''.join(body) + '</body></html>'
    html_path.write_text(document, encoding="utf-8")
    return {"report": str(report_path), "rows": str(rows_path), "html": str(html_path)}


def start_entropy_wandb(*, run_id, config, project="ambi-inner-bench", entity=None,
                       name=None, mode="online", resume="never", wandb_module=None):
    """Create the explicitly identified comparison run; immediately log progress.

    Pass resume='must' only when reconnecting the same campaign publisher.
    Disabled mode imports no W&B module and returns None.
    """
    if mode == "disabled":
        return None
    if mode != "online" or resume not in ("never", "must") or not run_id:
        raise ValueError("Explicit run_id, online/disabled mode and never/must resume are required")
    if wandb_module is None:
        import wandb as wandb_module
    run = wandb_module.init(project=project, entity=entity, id=run_id, name=name,
                            config={**copy.deepcopy(config), "diagnostic_kind": "frozen-prior-entropy",
                                    "entropy_reporting_schema": SCHEMA_VERSION},
                            mode=mode, resume=resume)
    run.define_metric("progress/observed_rows")
    run.define_metric("progress/*", step_metric="progress/observed_rows")
    run.log({"progress/observed_rows": 0, "progress/complete": 0})
    return run


def _segment(value):
    # Percent-encoding is reversible; replacing punctuation with underscores
    # silently collides for metric names such as q/gradient and q_gradient.
    return quote(str(value), safe="._-")


def publish_entropy_wandb(run, report, files=None, *, wandb_module=None):
    """Publish coverage, and publish scientific curves only for a complete panel.

    Caller owns run.finish(), including failure/incomplete exit status. Full
    report JSON, paired rows and standalone HTML are uploaded in one artifact.
    """
    if run is None:
        return
    coverage = report["coverage"]
    run.log({"progress/observed_rows": coverage["observed_rows"],
             "progress/expected_rows": coverage["expected_rows"],
             "progress/missing_rows": len(coverage["missing_rows"]),
             "progress/complete": int(report["status"] == "complete")})
    run.summary["entropy_status"] = report["status"]
    run.summary["entropy_coverage"] = coverage
    if report["status"] == "complete":
        defined = set()
        def define_once(name, **kwargs):
            if name not in defined:
                run.define_metric(name, **kwargs)
                defined.add(name)
        for family in ("series", "initialization_deltas", "contrasts"):
            for point in report[family]:
                panel = f"{_segment(point['source_run'])}/{point['checkpoint_step']}_{point['checkpoint_sha256'][:12]}/{point['prefix_action_rule']}"
                arm = _segment(point["arm"])
                if "reference_arm" in point:
                    arm += "_minus_" + _segment(point["reference_arm"])
                prefix = f"entropy/{panel}/{family}/{arm}"
                axis = prefix + "/actor_updates"
                define_once(axis)
                payload = {axis: point["actor_updates"]}
                for metric, stat in point["metrics"].items():
                    key = prefix + "/" + _segment(metric)
                    for suffix in ("mean", "ci95_low", "ci95_high", "episode_sd", "n_episodes"):
                        define_once(key + "/" + suffix, step_metric=axis)
                    payload.update({key+"/mean": stat["mean"], key+"/n_episodes": stat["n_episodes"]})
                    if stat["ci95"] is not None:
                        payload.update({key+"/ci95_low": stat["ci95"][0], key+"/ci95_high": stat["ci95"][1]})
                    if stat["episode_sd"] is not None:
                        payload[key+"/episode_sd"] = stat["episode_sd"]
                run.log(payload)
    if files:
        if wandb_module is None:
            import wandb as wandb_module
        artifact = wandb_module.Artifact(f"entropy-diagnostics-{run.id}", type="entropy-diagnostics",
                                         metadata={"status": report["status"], "schema_version": SCHEMA_VERSION})
        for name in ("report", "rows", "html"):
            artifact.add_file(files[name], name=Path(files[name]).name)
        run.log_artifact(artifact)
        run.summary["entropy_report_html"] = wandb_module.Html(Path(files["html"]).read_text(encoding="utf-8"), inject=False)
