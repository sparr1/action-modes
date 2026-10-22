"""Frozen MPPI-500k alpha preservation: three actual methods, episode 3 primary.

All dispersion is descriptive across five initialized histories with recurring
balanced seed orders. Reused prior/fresh-alpha results are not new trials.
"""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import eval_series as series
from slurm.ambi_alpha_carry_chain_publish import (
    PublicationError, _finite, _mean_sd, now, stream_seeds, validate_chain_summary,
    reference_chain, reduce_decisions, hardware_provenance, history_rows, publish_entry,
    AXIS, EPISODE_AXIS, DECISION_METRICS, ENTITY, PROJECT,
)
CHECKPOINTS = {"mppi_aux_500k": (500000, "MPPI-trained")}
PREFIX = "Alpha preservation MPPI500k / "
METHODS = {
    "prior": ("Sampled prior · reused", "#9CA3AF"),
    "reset": ("Fresh alpha each decision · reused", "#4B5563"),
    "chain_carry": ("Carry alpha across three episodes", "#2563EB"),
}
EPISODE_METRICS = [
    ("Episode return · episode 3 is primary", "return_mean"),
    ("Return sample SD across five histories", "return_sample_std"),
    ("Gain over sampled prior · positive means improvement", "prior_gain_mean"),
    ("Paired gain over prior sample SD", "prior_gain_sample_std"),
    ("Paired gain over fresh alpha", "reset_gain_mean"),
    ("Paired gain over fresh alpha sample SD", "reset_gain_sample_std"),
    ("Retained fresh-alpha improvement · ratio of mean gains", "retained_gain_ratio"),
    ("Fraction of paired returns above fresh alpha", "reset_win_fraction"),
    ("Worst paired gain over fresh alpha", "reset_worst_gain"),
    ("Controller seconds per decision · consult hardware", "control_seconds_per_decision"),
]



def reduce_episodes(summary, references):
    """Compute each episode across five streams, and aggregate within stream first."""
    episodes = {(episode["stream_index"], episode["episode_index"]): episode for episode in summary["episodes"]}
    if len(episodes) != 15 or len(summary["episodes"]) != 15:
        raise PublicationError("Episode statistics require five streams of three episodes")
    refs = {name: {episode["seed"]: episode for episode in reference["episodes"]}
            for name, reference in references.items()}
    if set(refs) != {"prior", "reset"}:
        raise PublicationError("Exactly the pinned prior and fresh-alpha references are required")
    output, per_stream = [], {stream: [] for stream in range(5)}
    for index in range(3):
        selected = [episodes[stream, index] for stream in range(5)]
        point = {EPISODE_AXIS: index + 1, "streams": 5}
        for stream, episode in enumerate(selected):
            if episode["seed"] != stream_seeds(stream)[index] or episode.get("length") != 500:
                raise PublicationError("Episode/reference pairing changed")
            for reference in refs.values():
                paired = reference[episode["seed"]]
                if paired.get("solver_seed") != episode.get("solver_seed"):
                    raise PublicationError("Reference controller seeds do not match the chain episode")
            per_stream[stream].append(episode["return"])
        point["return_mean"], point["return_sample_std"] = _mean_sd([ep["return"] for ep in selected])
        for name, reference in refs.items():
            gains = [episode["return"] - reference[episode["seed"]]["return"] for episode in selected]
            point[name + "_gain_mean"], point[name + "_gain_sample_std"] = _mean_sd(gains)
        gains = [episode["return"] - refs["reset"][episode["seed"]]["return"] for episode in selected]
        point.update(reset_wins=sum(gain > 0 for gain in gains), reset_losses=sum(gain < 0 for gain in gains),
                     reset_ties=sum(gain == 0 for gain in gains), reset_win_fraction=sum(gain > 0 for gain in gains)/5,
                     reset_worst_gain=min(gains), reset_best_gain=max(gains))
        baseline_gain = statistics.mean(refs["reset"][episode["seed"]]["return"] -
                                        refs["prior"][episode["seed"]]["return"] for episode in selected)
        point["fresh_prior_gain_mean"] = baseline_gain
        if baseline_gain > 0:
            point["retained_gain_ratio"] = point["prior_gain_mean"] / baseline_gain
        times = [ep.get("control_seconds") for ep in selected]
        if all(_finite(value) for value in times):
            point["control_seconds_per_decision"] = statistics.mean(times) / 500
        output.append(point)
    # Explicitly collapse the three dependent episodes before reporting an
    # overall distribution. These are five values, never fifteen.
    stream_means = [statistics.mean(per_stream[stream]) for stream in range(5)]
    mean, sd = _mean_sd(stream_means)
    overall = dict(streams=5, episodes_per_stream=3, stream_mean_return=mean,
                   stream_mean_return_sample_std=sd, statistical_unit="initialized temperature stream")
    primary = output[2]
    overall.update(primary_episode=3, **{"primary_" + key: value for key, value in primary.items() if key != EPISODE_AXIS})
    overall["retained_gain_definition"] = "ratio of mean gains over the same prior; omitted unless fresh-alpha mean gain is positive"
    overall["interpretation"] = "descriptive paired results; no prespecified acceptance or noninferiority threshold"
    return output, overall


def build_payload(checkpoint_id, method, summary, references, provenance):
    episode_points, overall = reduce_episodes(summary, references)
    return dict(checkpoint_id=checkpoint_id, method=method, provenance=provenance,
                decision_points=reduce_decisions(summary) if summary.get("rows") else [],
                episode_points=episode_points, overall=overall)


def load_campaign(path):
    from slurm import ambi_alpha_preservation as driver
    campaign = series._read(path)
    driver.validate_campaign(campaign)
    return campaign


def display_rows(campaign):
    rows = []
    for checkpoint in CHECKPOINTS:
        for method, (label, color) in METHODS.items():
            identity = dict(kind="alpha-preservation-publication-v1", campaign_sha256=series._digest(campaign),
                checkpoint_id=checkpoint, method=method, protocol="run-temperature-chain-v1",
                reference_reuse=method != "chain_carry", histories=5, episodes_per_history=3, primary_episode=3,
                seed_orders=[stream_seeds(stream) for stream in range(5)],
                aggregation="descriptive five-history statistics at each episode position")
            rows.append(dict(run_id=series._digest(identity)[:32], checkpoint_id=checkpoint, method=method,
                name=f"{CHECKPOINTS[checkpoint][1]} {CHECKPOINTS[checkpoint][0]//1000}k | {label}",
                color=color, identity=identity))
    return rows


def collect(campaign, *, require_complete=False):
    """The driver verifies all source receipts before any data enters publication."""
    from slurm import ambi_alpha_preservation as driver
    snapshot = driver.publication_snapshot(campaign)
    entries = []
    for display in display_rows(campaign):
        checkpoint = snapshot["checkpoints"][display["checkpoint_id"]]
        references = checkpoint["reference_summaries"]
        streams = checkpoint["streams"]
        complete = len(streams)
        if complete > 5:
            raise PublicationError("Too many completed temperature histories")
        failed = sum(row["state"] in {"failed", "validation_failed"} for row in checkpoint.get("status", []))
        if not 0 <= failed <= 5 - complete:
            raise PublicationError("Invalid history completion counts")
        pending = 5 - complete - failed
        if require_complete:
            failed += pending
            pending = 0
        method = display["method"]
        status = "complete" if complete == 5 else "failed" if failed else "pending"
        provenance = dict(checkpoint=campaign.get("checkpoints", {}).get(display["checkpoint_id"], {}),
            source_commit=campaign["source_commit"], science_id=checkpoint["science_id"],
            source_receipts=checkpoint.get("status", []),
            references={name: value.get("provenance", {}) for name, value in references.items()},
            publication_policy="compact aggregates only; complete per-history summaries and full traces retained on Oscar")
        summary = None
        if method == "chain_carry" and complete == 5:
            summary = {"rows": [row for stream in streams for row in stream["rows"]],
                       "episodes": [episode for stream in streams for episode in stream["episodes"]]}
            validate_chain_summary(summary)
            provenance["production_hardware"] = hardware_provenance(campaign, display["checkpoint_id"])
        elif method != "chain_carry":
            summary = reference_chain(references[method], decision_rows=method != "prior")
            status = "complete"
            # New-history progress changes must not invalidate an immutable
            # historical reference display after it has been published.
            provenance.pop("source_receipts", None)
        payload = build_payload(display["checkpoint_id"], method, summary, references, provenance) if summary else None
        entries.append({**display, "status": status, "payload": payload,
            "progress": {"complete": complete, "pending": pending, "failed": failed, "expected": 5}
                        if method == "chain_carry" else {}})
    return entries


def install_workspace(campaign, path, *, backend=None):
    from slurm.ambi_entropy_kl_curves_layout import _color, _filter_model
    if backend is None:
        for package, version in (("wandb", "0.28.1"), ("wandb-workspaces", "0.4.5")):
            if importlib.metadata.version(package) != version:
                raise PublicationError(f"Layout requires isolated {package}=={version}")
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        from wandb_workspaces import expr
    else:
        ws, wr, expr = backend
    rows, path = display_rows(campaign), Path(path)
    old = series._read(path) if path.exists() else {}
    if old and old.get("campaign_sha256") != series._digest(campaign):
        raise PublicationError("Workspace belongs to another chain campaign")
    markdown = """**MPPI-trained 500k: preserving policy improvement with carried temperature. Episode 3 is primary.**

Three actual methods: gray sampled prior, dark-gray fresh alpha each decision
(both reused by environment/controller seed), and blue alpha+Adam carried across
three episodes. H3/J8, sampled actions, temperature LR 0.0003, no KL, frozen outer
checkpoint. Actor, critic and replay are fresh at every real decision. Only the
blue temperature state persists across episodes; environment/controller RNG reset.

Five initialized histories use balanced cyclic seed orders, three 500-decision
episodes each. Mean and sample SD describe five histories at each episode position;
recurring seeds do not create 15 independent trials or confidence intervals.
Primary outcome is episode-3 paired return change versus fresh alpha, alongside
gain over the prior, wins/losses, and worst paired change. No acceptable-loss or
noninferiority threshold is assumed. The retention ratio divides mean gain over
prior by the fresh-alpha mean gain over prior, and is omitted unless that
denominator is positive. The signed gain is never clipped to appear positive.

Global decisions 501 and 1001 begin episodes 2 and 3. Episode reward resets there;
chain reward does not. Entropy is the actor-update batch statistic, not final-root
executed-action entropy. Lines are unsmoothed with true method colors and native
hover. New outcome curves appear only after all five histories validate; pending
and failure counts remain explicit. Compact aggregates only, no W&B artifacts;
full traces remain on Oscar. Raw timing can differ by GPU; inspect provenance.
"""
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True, panels=[
        wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=12)),
        *[wr.ScalarChart(title=title, metric="progress/histories_" + key, layout=wr.Layout(w=8, h=5))
          for title, key in (("Validated histories per checkpoint", "complete"),
                             ("Pending histories per checkpoint", "pending"), ("Failed histories per checkpoint", "failed"))]])]
    for checkpoint in CHECKPOINTS:
        chosen = [row for row in rows if row["checkpoint_id"] == checkpoint]
        def plot(title, metric, axis, prefix):
            key = prefix + "/" + checkpoint + "/" + metric
            plotted = chosen if prefix == "episode" else [row for row in chosen if row["method"] != "prior"]
            return wr.LinePlot(title=title, x=axis, y=[key], aggregate=False, smoothing_type="none",
                legend_template="${run:name}", max_runs_to_show=8, layout=wr.Layout(w=12, h=8),
                title_x="Episode position" if prefix == "episode" else "Global decision (new episodes at 501 and 1001)",
                line_colors={row["run_id"] + ":" + key: row["color"] for row in plotted},
                line_titles={row["run_id"] + ":" + key: row["name"] for row in plotted},
                line_marks={row["run_id"] + ":" + key: "solid" for row in plotted})
        sections.append(ws.Section(name=PREFIX + checkpoint + " episode outcomes", is_open=True,
            panels=[plot(title, metric, EPISODE_AXIS, "episode") for title, metric in EPISODE_METRICS]))
        sections.append(ws.Section(name=PREFIX + checkpoint + " temperature dynamics", is_open=True,
            panels=[plot(title, metric, AXIS, "decision") for title, metric in DECISION_METRICS]))
    if old.get("url"):
        work = ws.Workspace.from_url(old["url"])
        if work.entity != ENTITY or work.project != PROJECT:
            raise PublicationError("Workspace target changed")
        work.sections = [section for section in work.sections if not section.name.startswith(PREFIX)] + sections
    else:
        work = ws.Workspace(entity=ENTITY, project=PROJECT, name="MPPI500k alpha preservation · episode 3 primary", sections=sections,
                            auto_generate_panels=False)
    wanted = expr.Metric("ID").isin([row["run_id"] for row in rows])
    work.settings = ws.WorkspaceSettings(x_axis=AXIS, max_runs=8, tooltip_number_of_runs="all_runs",
        tooltip_color_run_names=True, remove_legends_from_panels=False)
    work.runset_settings = ws.RunsetSettings(filters=[wanted], groupby=[],
        run_settings={row["run_id"]: ws.RunSettings(color=row["color"], disabled=False) for row in rows})
    work.save()
    receipt = dict(campaign_sha256=series._digest(campaign), url=work.url, api_verified=False, runs=rows)
    series._atomic_json(path, receipt)
    saved = ws.Workspace.from_url(work.url)
    owned = [section for section in saved.sections if section.name.startswith(PREFIX)]
    panels = [panel for section in owned for panel in section.panels]
    if (saved.entity != ENTITY or saved.project != PROJECT or saved.url != work.url
            or len(owned) != 3 or len(panels) != 4 + len(EPISODE_METRICS) + len(DECISION_METRICS)
            or saved.runset_settings.groupby or _filter_model(saved.runset_settings.filters, expr) != _filter_model(wanted)):
        raise PublicationError("Saved chain workspace lost its exact target, panels or run filter")
    for row in rows:
        settings = saved.runset_settings.run_settings[row["run_id"]]
        if settings.disabled or _color(settings.color) != row["color"]:
            raise PublicationError("Saved chain run visibility or color changed")
    actual_metrics = set()
    for panel in (panel for panel in panels if isinstance(panel, wr.LinePlot)):
        metric = getattr(panel.y[0], "name", panel.y[0])
        axis = EPISODE_AXIS if metric.startswith("episode/") else AXIS
        if (panel.aggregate or panel.smoothing_type != "none" or panel.legend_template != "${run:name}"
                or getattr(panel.x, "name", panel.x) != axis or metric in actual_metrics):
            raise PublicationError("Saved chain panel lost its independent connected curves")
        actual_metrics.add(metric)
        checkpoint = metric.split("/")[1]
        for row in rows:
            if row["checkpoint_id"] != checkpoint or (axis == AXIS and row["method"] == "prior"):
                continue
            key = row["run_id"] + ":" + metric
            if (panel.line_titles.get(key) != row["name"] or _color(panel.line_colors.get(key)) != row["color"]
                    or panel.line_marks.get(key) != "solid"):
                raise PublicationError("Saved chain curve lost a name, color or solid line")
    expected_metrics = {prefix + "/" + checkpoint + "/" + metric for checkpoint in CHECKPOINTS
                        for prefix, definitions in (("episode", EPISODE_METRICS), ("decision", DECISION_METRICS))
                        for _, metric in definitions}
    if actual_metrics != expected_metrics:
        raise PublicationError("Saved chain workspace metric set changed")
    receipt.update(api_verified=True, panel_count=len(panels), updated_at=now())
    series._atomic_json(path, receipt)
    return receipt


def verify_workspace(campaign, path, wandb_module):
    from urllib.parse import parse_qs, urlparse
    path = Path(path)
    if not path.exists():
        raise PublicationError("Install and verify the dedicated chain workspace before publication")
    receipt = series._read(path)
    if not receipt.get("api_verified") or receipt.get("campaign_sha256") != series._digest(campaign):
        raise PublicationError("Dedicated chain workspace receipt changed")
    parsed = urlparse(receipt["url"])
    if parsed.hostname != "wandb.ai" or parsed.path.rstrip("/") != f"/{ENTITY}/{PROJECT}":
        raise PublicationError("Workspace URL is outside the intended project")
    view = parse_qs(parsed.query).get("nw", [None])[0]
    if not view:
        raise PublicationError("Named chain workspace ID is missing")
    query = 'query View($entityName:String,$name:String,$viewName:String){project(name:$name,entityName:$entityName){allViews(viewType:"project-view",viewName:$viewName){edges{node{id spec}}}}}'
    variables = dict(entityName=ENTITY, name=PROJECT, viewName="nw-" + view + "-v")
    api = wandb_module.Api(timeout=30)
    if hasattr(api, "_service_api"):
        response = api._service_api.execute_graphql(query, variables=variables)
    else:
        try:
            gql = importlib.import_module("wandb_gql").gql
        except ImportError:
            gql = importlib.import_module("wandb.vendor.gql").gql
        response = api.client.execute(gql(query), variable_values=variables)
    edges = (response.get("project") or {}).get("allViews", {}).get("edges", [])
    wire = json.dumps(edges[0]["node"]["spec"]) if len(edges) == 1 else ""
    tokens = [AXIS, EPISODE_AXIS, "progress/histories_pending", *[row["run_id"] for row in display_rows(campaign)],
        *[prefix + "/" + checkpoint + "/" + metric for checkpoint in CHECKPOINTS
          for prefix, definitions in (("episode", EPISODE_METRICS), ("decision", DECISION_METRICS)) for _, metric in definitions]]
    if any(token not in wire for token in tokens):
        raise PublicationError("Saved remote chain workspace lacks required panels or runs")
    return receipt


def render_report(campaign, entries, output):
    value = dict(campaign_id=campaign.get("campaign_id"), source_commit=campaign["source_commit"], entries=entries,
                 decision_metrics=DECISION_METRICS, episode_metrics=EPISODE_METRICS)
    encoded = json.dumps(value, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    Path(output).write_text(_HTML.replace("__DATA__", encoded))
    return value


def publish_once(campaign, root, *, wandb_module=None, require_complete=False, workspace_verifier=verify_workspace):
    if wandb_module is None:
        import wandb as wandb_module
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with series._lock(root / ".publisher.lock", blocking=False):
        workspace = workspace_verifier(campaign, root / "workspace.json", wandb_module)
        entries = collect(campaign, require_complete=require_complete)
        render_report(campaign, entries, root / "report.html")
        for entry in entries:
            publish_entry(entry, root, wandb_module)
        progress = {entry["checkpoint_id"]: {"status": entry["status"], **entry["progress"]}
                    for entry in entries if entry["method"] == "chain_carry"}
        result = dict(updated_at=now(), workspace_url=workspace["url"], checkpoints=progress,
                      all_complete=all(row["status"] == "complete" for row in progress.values()))
        series._atomic_json(root / "status.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("layout", "publish", "report"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    campaign = load_campaign(args.campaign)
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "layout":
            result = install_workspace(campaign, args.root / "workspace.json")
        elif args.command == "report":
            entries = collect(campaign)
            render_report(campaign, entries, args.root / "report.html")
            result = {"report": str(args.root / "report.html")}
        else:
            result = publish_once(campaign, args.root, require_complete=args.require_complete)
            if args.require_complete and not result["all_complete"]:
                raise PublicationError("Missing final histories were marked failed; publication remains incomplete")
    except Exception as error:
        message = str(error) if isinstance(error, PublicationError) else "Inspect preservation receipts and publication journals"
        series._atomic_json(args.root / "publication-failure.json", dict(error_type=type(error).__name__, message=message, at=now()))
        raise SystemExit(f"Preservation publication failed ({type(error).__name__}): {message}") from None
    print(series._json(result))


_HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MPPI500k alpha preservation</title><style>body{font:15px system-ui;margin:30px auto;max-width:1250px;padding:0 24px;color:#183044;background:#f6f8fb}section{background:white;padding:20px;margin:18px 0;border:1px solid #dde4ec;border-radius:10px}select{padding:7px}svg{width:100%;height:360px}button{background:white;border:1px solid #ddd;padding:8px;margin:4px;cursor:pointer}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:7px;border-bottom:1px solid #eee}.muted{color:#607386}#tooltip{min-height:27px}</style><h1>MPPI500k alpha preservation: episode 3 primary</h1><p>Frozen H3/J8 · sampled actions · five initialized temperature histories with balanced cyclic seed orders. Episode 3 is the primary comparison. Each history contains three 500-decision episodes. Only alpha and Adam persist across episode resets.</p><p class="muted">Mean and sample SD describe five histories at each episode position. Seeds recur across positions; these are not 15 independent trials. Reused prior and fresh-alpha references are matched by environment/controller seed and are not new evaluations. Retention is a ratio of mean gains and is omitted unless fresh-alpha gain is positive. Paired differences and sample SD are descriptive, with no assumed acceptable-loss threshold.</p><section><label>Checkpoint <select id="checkpoint"></select></label> <label>Axis <select id="axis"><option value="episode">Episode position</option><option value="decision">Global environment decision</option></select></label> <label>Metric <select id="metric"></select></label><div id="legend"></div><svg id="plot" viewBox="0 0 1180 360"></svg><div id="tooltip"></div><p class="muted">Solid lines connect measured means without smoothing. Dashed vertical lines mark episode boundaries. Hover a curve to highlight it; click legend labels to hide/show. Episode reward accumulation resets; chain reward does not.</p></section><section><h2>Progress</h2><div id="progress"></div></section><section><h2>Episode results</h2><div id="results" style="overflow:auto"></div></section><section><h2>Provenance</h2><pre id="provenance" style="overflow:auto;max-height:300px"></pre></section><script id="chain-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('chain-data').textContent),$=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg',hidden=new Set();function option(p,v,t){const e=document.createElement('option');e.value=v;e.textContent=t;p.append(e)}Array.from(new Set(data.entries.map(e=>e.checkpoint_id))).forEach(c=>option($('checkpoint'),c,c));function metricOptions(){$('metric').replaceChildren();data[$('axis').value+'_metrics'].forEach(([t,k])=>option($('metric'),k,t));draw()}function add(tag,attrs){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));$('plot').append(e);return e}function draw(){const axis=$('axis').value,m=$('metric').value,ak=axis==='decision'?'chain/global_decision':'chain/episode',entries=data.entries.filter(e=>e.checkpoint_id===$('checkpoint').value);$('plot').replaceChildren();$('legend').replaceChildren();$('tooltip').textContent='';entries.forEach(e=>{const b=document.createElement('button');b.textContent=e.name.split(' | ')[1]+' · '+e.status;b.style.color=e.color;b.style.opacity=hidden.has(e.run_id)?.4:1;b.onclick=()=>{hidden.has(e.run_id)?hidden.delete(e.run_id):hidden.add(e.run_id);draw()};$('legend').append(b)});const available=entries.filter(e=>e.payload&&!hidden.has(e.run_id)),points=e=>e.payload[axis+'_points'].filter(p=>Number.isFinite(p[m])),vals=available.flatMap(e=>points(e).map(p=>p[m]));if(!vals.length){add('text',{x:80,y:160,fill:'#607386'}).textContent='Pending complete five-history results';return}let lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max((hi-lo)*.1,.01);lo-=pad;hi+=pad;const xmax=axis==='decision'?1500:3,x=d=>70+(d-1)/(xmax-1)*1060,y=v=>305-(v-lo)/(hi-lo)*265;for(let i=0;i<=5;i++){const v=lo+(hi-lo)*i/5;add('line',{x1:70,x2:1130,y1:y(v),y2:y(v),stroke:'#e5eaf0'});add('text',{x:2,y:y(v)+4,'font-size':12}).textContent=v.toPrecision(4)}(axis==='decision'?[1,250,500,750,1000,1250,1500]:[1,2,3]).forEach(d=>add('text',{x:x(d),y:330,'text-anchor':'middle','font-size':12}).textContent=d);if(axis==='decision')[500.5,1000.5].forEach(d=>{add('line',{x1:x(d),x2:x(d),y1:30,y2:305,stroke:'#8b98a5','stroke-dasharray':'5 5'});add('text',{x:x(d)+5,y:23,'font-size':11}).textContent=d<1000?'Episode 2':'Episode 3'});available.forEach(e=>{const ps=points(e);if(!ps.length)return;const line=add('polyline',{points:ps.map(p=>`${x(p[ak])},${y(p[m])}`).join(' '),fill:'none',stroke:e.color,'stroke-width':2.6});line.onmouseenter=()=>{$('plot').querySelectorAll('polyline').forEach(p=>p.style.opacity=p===line?1:.18);line.setAttribute('stroke-width',4)};line.onmouseleave=()=>{$('plot').querySelectorAll('polyline').forEach(p=>p.style.opacity=1);line.setAttribute('stroke-width',2.6)};line.onmousemove=event=>{const p=$('plot').createSVGPoint();p.x=event.clientX;p.y=event.clientY;const q=p.matrixTransform($('plot').getScreenCTM().inverse()),d=Math.max(1,Math.min(xmax,Math.round(1+(q.x-70)/1060*(xmax-1)))),nearest=ps.reduce((a,b)=>Math.abs(a[ak]-d)<Math.abs(b[ak]-d)?a:b),sdk=m.endsWith('_mean')?m.slice(0,-5)+'_sample_std':m+'_sample_std';$('tooltip').textContent=`${e.name} · ${axis} ${nearest[ak]} · ${nearest[m].toPrecision(5)}${Number.isFinite(nearest[sdk])?' ± '+nearest[sdk].toPrecision(4)+' sample SD':''}`}})}function table(parent,heads,rows){const t=document.createElement('table'),h=t.insertRow();heads.forEach(v=>{const e=document.createElement('th');e.textContent=v;h.append(e)});rows.forEach(row=>{const tr=t.insertRow();row.forEach(v=>tr.insertCell().textContent=Number.isFinite(v)?v.toFixed(4):String(v??'—'))});parent.append(t)}table($('progress'),['Checkpoint','Status','Complete histories','Pending','Failed'],data.entries.filter(e=>e.method==='chain_carry').map(e=>[e.checkpoint_id,e.status,e.progress.complete,e.progress.pending,e.progress.failed]));table($('results'),['Checkpoint','Method','Episode (3 primary)','Return mean','Return SD','Gain vs fresh alpha','Paired gain SD','Gain vs prior','Retained gain','Wins / losses','Worst paired gain'],data.entries.flatMap(e=>(e.payload?.episode_points??[]).map(p=>[e.checkpoint_id,e.name.split(' | ')[1],p['chain/episode'],p.return_mean,p.return_sample_std,p.reset_gain_mean,p.reset_gain_sample_std,p.prior_gain_mean,p.retained_gain_ratio,p.reset_wins+' / '+p.reset_losses,p.reset_worst_gain])));$('provenance').textContent=JSON.stringify({campaign:data.campaign_id,source_commit:data.source_commit,entries:data.entries.map(e=>({...e,payload:e.payload?{overall:e.payload.overall,provenance:e.payload.provenance}:null}))},null,2);$('axis').onchange=metricOptions;$('checkpoint').onchange=draw;$('metric').onchange=draw;metricOptions();</script></html>'''


if __name__ == "__main__":
    main()
