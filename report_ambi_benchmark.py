#!/usr/bin/env python3
"""Build an offline, self-contained inspector for frozen AMBI benchmarks.

Inputs are local version-1 result bundles. This command does not import the
training stack, contact W&B, smooth traces, or interpolate optimizer schedules.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import tempfile
from pathlib import Path


SCHEMA_VERSION = 1
_ASSET_DIR = Path(__file__).resolve().parent / "utils"
_COUNTERS = ("event_index", "round_index", "critic_updates", "actor_updates", "temperature_updates")
_METRIC_SEMANTICS = ("definition", "unit", "sampling_phase", "preferred_axis")


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant {value}; use null with a nonfinite status.")


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _identity(value, name):
    if not isinstance(value, (str, int)) or isinstance(value, bool) or value == "":
        raise ValueError(f"{name} must be a nonempty string or integer.")
    return value


def _finite(value, name, *, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data.")
    return value


def _trace_path(directory, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("trace_files entries must be relative paths.")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Trace path escapes its bundle: {relative!r}.")
    resolved = (directory / candidate).resolve()
    if not resolved.is_relative_to(directory.resolve()):
        raise ValueError(f"Trace path escapes its bundle: {relative!r}.")
    return resolved


def _root_key(root, protocol):
    bank_id = root.get("root_bank_id", protocol.get("root_bank_id"))
    if bank_id is None:
        raise ValueError("Shared-root records require an explicit root_bank_id.")
    root_id = _identity(root.get("root_id"), "root_id")
    repeat = _integer(root.get("repeat"), "repeat")
    solver_seed = _integer(root.get("solver_seed"), "solver_seed")
    return _canonical([_identity(bank_id, "root_bank_id"), root_id, repeat, solver_seed])


def _root_probe_signature(root):
    # Probe settings can be absent when no policy-quality probe was requested.
    # Runtime and total model-step cost are measurements, not comparison rules.
    # Cost can differ with the number of rounds even when each probe is matched.
    return {key: root[key] for key in ("probe_seed", "probe_rollouts", "probe_horizon")
            if key in root}


def _load_run(directory, manifest, run, run_key):
    run_id = _identity(run.get("id"), "run.id")
    episodes = run.get("episodes", [])
    roots = run.get("roots", [])
    if not isinstance(episodes, list) or not isinstance(roots, list):
        raise ValueError(f"Run {run_id}: episodes and roots must be arrays.")
    episode_map, root_map = {}, {}
    for episode in episodes:
        episode_id = _identity(episode.get("episode_id"), "episode_id")
        key = _canonical(episode_id)
        if key in episode_map:
            raise ValueError(f"Run {run_id}: duplicate episode_id {episode_id!r}.")
        _integer(episode.get("seed"), "episode seed")
        _integer(episode.get("length"), "episode length")
        _finite(episode.get("return"), "episode return", optional=True)
        episode_map[key] = episode
    for root in roots:
        _root_key(root, manifest["protocol"])
        key = _canonical([root["root_id"], root["repeat"]])
        if key in root_map:
            raise ValueError(f"Run {run_id}: duplicate root_id/repeat {key}.")
        root_map[key] = root

    solves, seen_events, seen_paths = {}, set(), set()
    for relative in run.get("trace_files", []):
        path = _trace_path(directory, relative)
        if path in seen_paths:
            raise ValueError(f"Run {run_id}: duplicate trace file {relative!r}.")
        seen_paths.add(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line, parse_constant=_reject_constant)
                where = f"{relative}:{line_index}"
                if row.get("run_id") != run_id:
                    raise ValueError(f"{where}: trace run_id does not match its manifest run.")
                decision = _integer(row.get("decision_index"), f"{where}: decision_index")
                is_root = row.get("root_id") is not None
                if is_root:
                    if row.get("episode_id") is not None:
                        raise ValueError(f"{where}: a trace cannot be both an episode and a shared root.")
                    root_key = _canonical([row["root_id"], row.get("repeat")])
                    if root_key not in root_map:
                        raise ValueError(f"{where}: trace references an unknown root/repeat.")
                    root = root_map[root_key]
                    for field in ("solver_seed", "root_bank_id"):
                        if field in row and row[field] != root.get(field, manifest["protocol"].get(field)):
                            raise ValueError(f"{where}: trace {field} conflicts with its root record.")
                    selection_key = _root_key(root, manifest["protocol"])
                    solve_key = _canonical(["bank", selection_key, decision])
                    info = {**root, "mode": "bank", "selection_key": selection_key}
                    info["root_bank_id"] = root.get("root_bank_id", manifest["protocol"].get("root_bank_id"))
                else:
                    episode_key = _canonical(row.get("episode_id"))
                    if episode_key not in episode_map:
                        raise ValueError(f"{where}: trace references an unknown episode_id.")
                    episode = episode_map[episode_key]
                    complete = episode.get("status", run.get("status", manifest.get("status"))) in ("complete", "completed", "finished", "success")
                    if decision > episode["length"] or (complete and decision >= episode["length"]):
                        raise ValueError(f"{where}: decision_index exceeds the recorded episode length.")
                    selection_key = _canonical([episode["episode_id"], episode["seed"]])
                    solve_key = _canonical(["episodes", selection_key, decision])
                    info = {"mode": "episodes", "selection_key": selection_key,
                            "episode_id": episode["episode_id"], "seed": episode["seed"]}
                event = _integer(row.get("event_index"), f"{where}: event_index")
                if (solve_key, event) in seen_events:
                    raise ValueError(f"{where}: duplicate event_index within a solve.")
                seen_events.add((solve_key, event))
                phase = row.get("phase")
                if not isinstance(phase, str) or not phase:
                    raise ValueError(f"{where}: phase must be a nonempty string.")
                for counter in _COUNTERS[1:]:
                    if row.get(counter) is not None:
                        _integer(row[counter], f"{where}: {counter}", minimum=-1 if counter == "round_index" else 0)
                metrics = row.get("metrics", {})
                nonfinite = row.get("nonfinite", {})
                if not isinstance(metrics, dict) or not isinstance(nonfinite, dict):
                    raise ValueError(f"{where}: metrics/nonfinite must be objects.")
                for name, value in metrics.items():
                    if name not in manifest["metric_catalog"]:
                        raise ValueError(f"{where}: metric {name!r} is absent from metric_catalog.")
                    if value is not None:
                        _finite(value, f"{where}: {name}")
                    if name in nonfinite and (value is not None or nonfinite[name] not in ("nan", "inf", "-inf")):
                        raise ValueError(f"{where}: invalid nonfinite status for {name!r}.")
                if set(nonfinite) - set(metrics):
                    raise ValueError(f"{where}: nonfinite status requires a null metric entry.")
                if solve_key not in solves:
                    solves[solve_key] = {**info, "decision_index": decision, "_rows": []}
                solves[solve_key]["_rows"].append(row)

    root_decisions = {}
    for solve in solves.values():
        if solve["mode"] == "bank":
            previous = root_decisions.setdefault(solve["selection_key"], solve["decision_index"])
            if previous != solve["decision_index"]:
                raise ValueError(f"Run {run_id}: one shared root/repeat must describe exactly one solve.")
    traces = []
    for solve_key, solve in sorted(solves.items()):
        rows = sorted(solve.pop("_rows"), key=lambda item: item["event_index"])
        for counter in _COUNTERS[2:]:
            present = [row[counter] for row in rows if row.get(counter) is not None]
            if any(later < earlier for earlier, later in zip(present, present[1:])):
                raise ValueError(f"Run {run_id}: decreasing {counter} in solve {solve_key}.")
        names = sorted({name for row in rows for name in row.get("metrics", {})})
        solve.update({key: [row.get(key) for row in rows] for key in (*_COUNTERS, "phase")})
        solve["metrics"] = {name: [row.get("metrics", {}).get(name) for row in rows] for name in names}
        solve["nonfinite"] = {name: [row.get("nonfinite", {}).get(name) for row in rows] for name in names
                              if any(name in row.get("nonfinite", {}) for row in rows)}
        traces.append(solve)
    return {"key": run_key, "id": run_id, "evaluation_id": manifest["evaluation_id"],
            "label": str(run.get("selector", run_id)), "selector": run.get("selector"),
            "config": run.get("config", {}), "config_hash": run.get("config_hash"),
            "kind": run.get("kind"), "status": run.get("status", manifest.get("status")),
            "wandb_path": run.get("wandb_path"), "episodes": episodes, "roots": roots, "traces": traces}


def load_bundles(bundle_paths):
    """Validate compatible local bundles and preserve their exact scalar traces."""
    if not bundle_paths:
        raise ValueError("At least one result bundle is required.")
    result = {"schema_version": SCHEMA_VERSION, "metric_catalog": {}, "runs": [], "sources": []}
    reference = None
    seen_run_keys, probe_signatures = set(), {}
    for supplied in bundle_paths:
        path = Path(supplied).expanduser().resolve()
        directory = path.parent if path.name == "manifest.json" else path
        manifest = _read_json(directory / "manifest.json")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"{directory}: unsupported schema_version; expected {SCHEMA_VERSION}.")
        evaluation_id = _identity(manifest.get("evaluation_id"), "evaluation_id")
        checkpoint = manifest.get("checkpoint", {})
        if not isinstance(checkpoint.get("sha256"), str) or not checkpoint["sha256"]:
            raise ValueError(f"{directory}: checkpoint.sha256 is required.")
        protocol = manifest.get("protocol")
        if not isinstance(protocol, dict):
            raise ValueError(f"{directory}: protocol must be an object.")
        for field in ("environment", "action_rule", "max_steps", "controller_seed", "seed_scheme"):
            if field not in protocol:
                raise ValueError(f"{directory}: protocol.{field} is required.")
        signature = {"checkpoint": checkpoint["sha256"],
                     "protocol": {key: value for key, value in protocol.items() if key != "root_bank_id"}}
        if reference is None:
            reference = _canonical(signature)
            result.update({"checkpoint": checkpoint, "protocol": signature["protocol"]})
        elif _canonical(signature) != reference:
            raise ValueError(f"{directory}: incompatible checkpoint or evaluation protocol.")
        catalog = manifest.get("metric_catalog")
        if not isinstance(catalog, dict):
            raise ValueError(f"{directory}: metric_catalog must be an object.")
        for name, semantic in catalog.items():
            if not isinstance(semantic, dict) or any(key not in semantic for key in _METRIC_SEMANTICS):
                raise ValueError(f"Metric {name!r} requires {', '.join(_METRIC_SEMANTICS)}.")
            if name in result["metric_catalog"] and _canonical(semantic) != _canonical(result["metric_catalog"][name]):
                raise ValueError(f"Metric {name!r} has incompatible semantics across bundles.")
            result["metric_catalog"][name] = semantic
        result["sources"].append({"evaluation_id": evaluation_id, "code": manifest.get("code", {}),
                                  "status": manifest.get("status"), "directory": str(directory)})
        for run in manifest.get("runs", []):
            run_key = _canonical([evaluation_id, run.get("id")])
            if run_key in seen_run_keys:
                raise ValueError(f"Duplicate evaluation/run identity: {run_key}.")
            seen_run_keys.add(run_key)
            loaded = _load_run(directory, manifest, run, run_key)
            for root in run.get("roots", []):
                key = _root_key(root, protocol)
                probe = _canonical(_root_probe_signature(root))
                if key in probe_signatures and probe_signatures[key] != probe:
                    raise ValueError(f"Shared root {key}: incompatible probe protocol.")
                probe_signatures[key] = probe
            result["runs"].append(loaded)
    if not result["runs"]:
        raise ValueError("Result bundles contain no runs.")
    return result


def render_html(data, *, title="AMBI inner optimization benchmark"):
    """Render a portable report without executable user-provided strings."""
    # Escaping '<' prevents embedded labels from closing the JSON script tag.
    encoded = json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    from html import escape
    template = (_ASSET_DIR / "ambi_benchmark_report.html").read_text(encoding="utf-8")
    script = (_ASSET_DIR / "ambi_benchmark_report.js").read_text(encoding="utf-8")
    replacements = {"TITLE": escape(title), "DATA": encoded, "SCRIPT": script}
    return re.sub(r"__REPORT_(TITLE|DATA|SCRIPT)__", lambda match: replacements[match.group(1)], template)


def write_report(data, output, *, overwrite=False, title="AMBI inner optimization benchmark"):
    """Atomically publish HTML, preserving an existing report by default."""
    output = Path(output).expanduser()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {output}. Pass --overwrite to replace it.")
    html = render_html(data, title=title)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(html)
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", required=True, help="Result directory or manifest.json; repeat to compare.")
    parser.add_argument("--output", type=Path, required=True, help="Portable HTML output file.")
    parser.add_argument("--metric", action="append", help="Include only named metrics; repeat to select several.")
    parser.add_argument("--title", default="AMBI inner optimization benchmark")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = load_bundles(args.bundle)
        if args.metric:
            unknown = set(args.metric) - set(data["metric_catalog"])
            if unknown:
                raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}.")
            data["metric_catalog"] = {key: data["metric_catalog"][key] for key in args.metric}
            for run in data["runs"]:
                for trace in run["traces"]:
                    for field in ("metrics", "nonfinite"):
                        trace[field] = {key: values for key, values in trace[field].items() if key in data["metric_catalog"]}
        output = write_report(data, args.output, overwrite=args.overwrite, title=args.title)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"html": str(output.resolve()), "runs": len(data["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
