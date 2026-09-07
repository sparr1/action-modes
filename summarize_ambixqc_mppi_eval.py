"""Validate and summarize a manifest-selected frozen XQC prior/MPPI campaign.

Reads portable bundles, never checkpoint tensors or environments. All inputs are
validated before writing output or optionally publishing a single W&B run.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import re
import statistics
from pathlib import Path

from report_ambi_benchmark import load_bundles
from run_ambixqc_mppi_evaluation import MATRIX, SOURCE_RUN, file_sha256
from utils.ambi_benchmark import (
    MPPI_ACTION_RULE, atomic_json, atomic_write, canonical_hash, controller_type,
    episode_protocol, run_controller_type, solver_seed, validate_evaluation_controller,
)


def _read(path):
    def reject(value):
        raise ValueError(f"{path}: nonfinite JSON value {value}.")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data.")
    return value


def _sha(value, length, name):
    if not isinstance(value, str) or re.fullmatch(rf"[a-f0-9]{{{length}}}", value) is None:
        raise ValueError(f"{name} must be a full lowercase SHA digest.")
    return value


def _stats(values):
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values)}


def _validate_run(run, traces, controller, *, seeds, max_steps, controller_seed):
    result = run.get("result", {})
    if (run.get("status") != "complete" or result.get("outer_state_unchanged") is not True
            or run.get("outer_state_unchanged", True) is not True
            or result.get("outer_updates_before") != result.get("outer_updates_after")
            or type(result.get("outer_updates_before")) is not int
            or result.get("outer_updates_before", -1) < 0):
        raise ValueError("Every run must be complete with verified unchanged full outer state.")
    if result.get("controller") != controller or result.get("environment_seeds") != list(seeds):
        raise ValueError("Run result controller/seeds do not match the campaign.")
    if result.get("controller_seed") != controller_seed or result.get("seed_scheme") != "sha256-v1":
        raise ValueError("Run result solver-seed protocol does not match the campaign.")
    action = MPPI_ACTION_RULE if controller == "mppi" else "tanh_mean"
    if run.get("action_rule") != action or result.get("action_rule") != action:
        raise ValueError("Run action rule does not match the executed controller.")
    if result.get("deterministic_execution") is not (controller == "prior"):
        raise ValueError("Run execution determinism does not match the controller.")
    if run.get("nonfinite_trace_metrics") or result.get("nonfinite_model_metrics"):
        raise ValueError("Campaign aggregation rejects nonfinite diagnostic data.")
    if run.get("config_hash") != canonical_hash(run.get("config", {})):
        raise ValueError("Run configuration hash does not match its contents.")
    if run.get("kind") != "episodes" or run.get("roots") or run.get("diagnostic_capabilities") != {
        "decision_metrics": True, "optimizer_traces": False, "shared_observation_probes": False,
    }:
        raise ValueError("Campaign requires decision-only episode diagnostics.")
    episodes = run.get("episodes", [])
    if [episode.get("seed") for episode in episodes] != list(seeds):
        raise ValueError("Episode seeds are missing, duplicated, or out of order.")
    by_seed = {seed: [] for seed in seeds}
    for trace in traces:
        if trace["mode"] != "episodes" or trace["seed"] not in by_seed or trace["nonfinite"]:
            raise ValueError("Unexpected or nonfinite diagnostic trace.")
        if trace["phase"] != ["decision"] or any(trace[key] != [0] for key in (
            "critic_updates", "actor_updates", "temperature_updates",
        )):
            raise ValueError("Prior and MPPI must record decisions with zero optimizer updates.")
        if any(value is None for values in trace["metrics"].values() for value in values):
            raise ValueError("Campaign decision diagnostics cannot contain null measurements.")
        by_seed[trace["seed"]].append(trace)
    settings = run.get("evaluation_controller", {}).get("settings", {})
    # The actor proposes H actions but needs only H-1 transitions to construct
    # those actions. Candidate scoring evaluates all H transitions each round.
    expected_work = (settings["horizon"] * settings["num_samples"] * settings["effective_iterations"]
                     + (settings["horizon"] - 1) * settings["num_pi_trajs"]) if controller == "mppi" else 0
    for episode in episodes:
        seed = episode["seed"]
        if episode.get("status") != "complete" or episode.get("length") != max_steps:
            raise ValueError("Campaign episode did not complete the requested decision count.")
        if episode.get("solver_seed") != solver_seed(controller_seed, "episode", seed):
            raise ValueError("Episode solver seed does not match sha256-v1 derivation.")
        if episode.get("nonfinite_model_metrics"):
            raise ValueError("Campaign episode contains nonfinite model diagnostics.")
        value = _finite(episode.get("return"), "Episode return")
        elapsed = _finite(episode.get("control_seconds"), "Episode control time")
        if elapsed < 0:
            raise ValueError("Episode control time cannot be negative.")
        decisions = sorted(by_seed[seed], key=lambda trace: trace["decision_index"])
        if [trace["decision_index"] for trace in decisions] != list(range(max_steps)):
            raise ValueError("Missing or duplicated real-decision diagnostics.")
        rewards, timings = [], []
        for trace in decisions:
            metrics = trace["metrics"]
            if metrics.get("decision/inner_model_steps") != [expected_work]:
                raise ValueError("Actual model-step count does not match the controller search budget.")
            if controller == "mppi" and metrics.get("decision/inner_mppi_iterations") != [settings["effective_iterations"]]:
                raise ValueError("Actual MPPI iteration count does not match the resolved search budget.")
            reward = _finite(metrics.get("decision/reward", [None])[0], "Decision reward")
            timing = _finite(metrics.get("decision/control_seconds", [None])[0], "Decision control time")
            if timing < 0:
                raise ValueError("Decision control time cannot be negative.")
            rewards.append(reward)
            timings.append(timing)
        if not math.isclose(math.fsum(rewards), value, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("Episode return does not match its real-decision rewards.")
        if not math.isclose(math.fsum(timings), elapsed, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("Episode control time does not match its decision timings.")
    return episodes


def summarize(manifest_path, results_root, *, expected_source_sha, seeds=(101, 102, 103, 104, 105),
              max_steps=500, controller_seed=12345, allow_partial=False):
    """Return strict JSON-ready statistics only after validating every present bundle."""
    _sha(expected_source_sha, 40, "Expected evaluation commit")
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be distinct valid NumPy seed integers.")
    if type(max_steps) is not int or max_steps < 1 or type(controller_seed) is not int or not 0 <= controller_seed < 2**32:
        raise ValueError("Invalid decision budget or controller seed.")
    manifest_path, results_root = Path(manifest_path).resolve(), Path(results_root).resolve()
    campaign = _read(manifest_path)
    entries = campaign.get("checkpoints", [])
    if campaign.get("source_run") != SOURCE_RUN or not entries:
        raise ValueError("Expected an explicit checkpoint manifest for the seed-55 prior-only source run.")
    steps = [entry.get("step") for entry in entries]
    if any(type(step) is not int or step <= 0 for step in steps) or steps != sorted(set(steps)):
        raise ValueError("Checkpoint manifest steps must be positive, unique, and ordered.")
    for entry in entries:
        _sha(entry.get("sha256"), 64, "Checkpoint hash")
        _sha(entry.get("metadata_sha256"), 64, "Checkpoint metadata hash")
        if not isinstance(entry.get("path"), str) or not Path(entry["path"]).is_absolute():
            raise ValueError("Checkpoint manifest paths must be absolute.")
    matrix = _read(MATRIX)
    expected_settings = matrix["comparisons"]["controller"]["variants"]["mppi"]["evaluation_controller"]["params"]
    expected_settings = {**expected_settings, "effective_iterations": 8}
    signature, common_protocol, shared_code = None, None, None
    rows, missing = [], []
    for entry in entries:
        step = entry["step"]
        destination = results_root / "production" / f"step_{step}"
        bundle_path = destination / "bundle"
        if not destination.exists():
            missing.append(step)
            continue
        manifest = _read(bundle_path / "manifest.json")
        provenance = _read(destination / "provenance.json")
        if (provenance.get("checkpoint") != entry or provenance.get("source_run") != SOURCE_RUN
                or provenance.get("checkpoint_manifest_sha256") != file_sha256(manifest_path)
                or provenance.get("matrix_sha256") != file_sha256(MATRIX)
                or provenance.get("mode") != "production" or provenance.get("seeds") != seeds
                or provenance.get("max_steps") != max_steps or provenance.get("controller_seed") != controller_seed):
            raise ValueError(f"Checkpoint {step}: launch provenance does not match the campaign.")
        checkpoint, code = manifest.get("checkpoint", {}), manifest.get("code", {})
        if (checkpoint.get("sha256") != entry["sha256"]
                or checkpoint.get("metadata_sha256") != entry["metadata_sha256"]
                or checkpoint.get("metadata", {}).get("checkpoint", {}).get("step") != step):
            raise ValueError(f"Checkpoint {step}: bundle checkpoint identity mismatch.")
        if manifest.get("status") != "complete" or code.get("commit") != expected_source_sha or code.get("dirty") is not False:
            raise ValueError(f"Checkpoint {step}: incomplete bundle or foreign/dirty evaluation source.")
        _sha(code.get("source_sha256"), 64, "Evaluation source fingerprint")
        if not isinstance(code.get("runtime"), dict) or not code["runtime"]:
            raise ValueError("Evaluation runtime provenance is required.")
        protocol = episode_protocol(manifest.get("protocol", {}))
        environment = protocol.get("environment", {})
        if (environment.get("id") != "DMControl-v0"
                or environment.get("params", {}).get("task") != "humanoid-walk"
                or environment.get("params", {}).get("obs") != "state"
                or protocol.get("observation") != "state" or protocol.get("action_rule") != "tanh_mean"
                or protocol.get("seed_scheme") != "sha256-v1" or protocol.get("max_steps") != max_steps
                or protocol.get("controller_seed") != controller_seed or "root_bank_id" in manifest["protocol"]):
            raise ValueError(f"Checkpoint {step}: evaluation protocol differs from this campaign.")
        runs = manifest.get("runs", [])
        if len(runs) != 2 or {run_controller_type(run) for run in runs} != {"prior", "mppi"}:
            raise ValueError("Each checkpoint must have exactly one prior run and one MPPI run.")
        runs = {run_controller_type(run): run for run in runs}
        controller = runs["mppi"].get("evaluation_controller")
        validate_evaluation_controller(runs["mppi"]["config"], controller)
        if controller["settings"] != expected_settings:
            raise ValueError("Resolved MPPI settings differ from the native Humanoid campaign.")
        if runs["mppi"].get("controller_hash") != canonical_hash(controller):
            raise ValueError("MPPI controller hash does not match its recorded protocol/settings.")
        adapter_protocol = controller["protocol"]
        expected_semantics = {
            "algorithm": "tdmpc2_mppi_over_frozen_xqc", "terminal_value_semantics": "learned_soft_q_tail_without_entropy_correction",
            "reward_units": "raw_environment_reward", "batchnorm_mode": "running", "rng": "private_device_generator",
            "warm_start": "shift_previous_mean_within_episode_reset_before_episode",
            "discount": 0.99, "value_finite_guard": "torch.nan_to_num(nan=0)",
            "upstream_tdmpc2_commit": "8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe",
        }
        if any(adapter_protocol.get(key) != value for key, value in expected_semantics.items()):
            raise ValueError("MPPI semantic protocol does not match native frozen XQC evaluation.")
        prior_config, mppi_config = copy.deepcopy(runs["prior"]["config"]), copy.deepcopy(runs["mppi"]["config"])
        prior_config.pop("evaluation_controller", None)
        mppi_config.pop("evaluation_controller", None)
        if prior_config != mppi_config or controller_type(prior_config) != "prior":
            raise ValueError("Prior and MPPI must share the identical frozen base configuration.")
        semantic_protocol = {key: value for key, value in adapter_protocol.items() if key != "reward_scale"}
        current_signature = canonical_hash({"protocol": protocol, "code": code, "settings": controller["settings"],
                                            "controller_protocol": semantic_protocol})
        if signature is None:
            signature, common_protocol, shared_code = current_signature, protocol, code
        elif current_signature != signature:
            raise ValueError("Campaign checkpoints have inconsistent source, runtime, or controller protocol.")
        # The detailed reader validates identities, gzip shards, metrics, and
        # duplicate rows. Different checkpoint hashes remain separate reports.
        data = load_bundles([bundle_path])
        traces = {run["controller_type"]: run["traces"] for run in data["runs"]}
        episodes = {kind: _validate_run(run, traces[kind], kind, seeds=seeds, max_steps=max_steps,
                                       controller_seed=controller_seed) for kind, run in runs.items()}
        deltas = []
        for prior, mppi in zip(episodes["prior"], episodes["mppi"]):
            delta = mppi["return"] - prior["return"]
            if not math.isclose(_finite(mppi.get("paired_return_delta"), "Paired return delta"), delta,
                                rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError("Paired return delta does not match the reference episode.")
            deltas.append(delta)
        row = {"checkpoint_step": step, "checkpoint_sha256": entry["sha256"],
               "metadata_sha256": entry["metadata_sha256"], "episode_count": len(seeds),
               "bundle": str(bundle_path), "bundle_manifest_sha256": file_sha256(bundle_path / "manifest.json"),
               "mppi_reward_scale": adapter_protocol["reward_scale"], "mppi_settings": controller["settings"],
               "paired": {"delta_mean": _stats(deltas)["mean"], "delta_std": _stats(deltas)["std"],
                          "deltas": deltas}, "outer_state_unchanged": True}
        for kind, values in episodes.items():
            returns = _stats([value["return"] for value in values])
            timings = _stats([value["control_seconds"] for value in values])
            row[kind] = {"return_mean": returns["mean"], "return_std": returns["std"],
                         "control_seconds_mean": timings["mean"], "control_seconds_std": timings["std"],
                         "control_seconds_per_decision": timings["mean"] / max_steps,
                         "returns": [value["return"] for value in values]}
        rows.append(row)
    if missing and not allow_partial:
        raise ValueError(f"Missing checkpoint outputs: {missing}. Use --allow-partial to summarize completed checkpoints only.")
    if not rows:
        raise ValueError("No completed checkpoint bundles are available.")
    return {"schema_version": 1, "status": "partial" if missing else "complete", "source_run": SOURCE_RUN,
            "checkpoint_manifest": str(manifest_path), "checkpoint_manifest_sha256": file_sha256(manifest_path),
            "evaluation_source_sha": expected_source_sha, "code": shared_code, "protocol": common_protocol,
            "seeds": seeds, "expected_steps": steps, "missing_steps": missing, "rows": rows,
            "statistics": "Paired seeds; population standard deviations (ddof=0); returns use raw environment rewards."}


def render_campaign_html(summary):
    """Portable campaign curves; detailed decision traces stay in each step report."""
    rows = summary["rows"]
    def plot(series, title):
        points = [(row["checkpoint_step"], row[group][metric]) for row in rows for group, metric, _ in series]
        xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
        ymin, ymax = min(y for _, y in points), max(y for _, y in points)
        pad = max((ymax - ymin) * .08, 1.)
        ymin, ymax = ymin - pad, ymax + pad
        xcoord = lambda x: 65 + 810 * (x - xmin) / max(xmax - xmin, 1)
        ycoord = lambda y: 260 - 220 * (y - ymin) / (ymax - ymin)
        curves = []
        for group, metric, color in series:
            path = " ".join(f"{xcoord(row['checkpoint_step']):.2f},{ycoord(row[group][metric]):.2f}" for row in rows)
            curves.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{path}"/>')
            for row in rows:
                value, step = row[group][metric], row["checkpoint_step"]
                curves.append(f'<circle cx="{xcoord(step):.2f}" cy="{ycoord(value):.2f}" r="4" fill="{color}"><title>{group} at {step:,}: {value:.4f}</title></circle>')
        labels = ''.join(f'<text x="8" y="{ycoord(value):.2f}">{value:.1f}</text>' for value in (ymin, (ymin + ymax)/2, ymax))
        return f'<h2>{title}</h2><svg viewBox="0 0 900 305" role="img" aria-label="{title}"><path d="M65 35V260H880" fill="none" stroke="#9ba8bc"/>{labels}{"".join(curves)}<text x="65" y="282">{xmin:,}</text><text x="795" y="282">{xmax:,}</text><text x="370" y="302">Training decisions at checkpoint</text></svg>'
    body = []
    for row in rows:
        fmt = lambda kind, metric: f'{row[kind][metric + "_mean"]:.3f} ± {row[kind][metric + "_std"]:.3f}'
        body.append(f'<tr><td>{row["checkpoint_step"]:,}</td><td>{fmt("prior", "return")}</td><td>{fmt("mppi", "return")}</td><td>{fmt("paired", "delta")}</td><td>{row["prior"]["control_seconds_per_decision"]:.6f}</td><td>{row["mppi"]["control_seconds_per_decision"]:.6f}</td></tr>')
    return ('<!doctype html><html><head><meta charset="utf-8"><title>Frozen XQC prior versus MPPI</title>'
            '<style>body{font:16px system-ui;margin:40px auto;max-width:1080px;padding:0 24px;color:#17243a;background:#f6f8fc}h1{font-size:30px}h2{font-size:21px;margin-top:30px}p{line-height:1.6}svg{width:100%;background:white;border:1px solid #d9dfeb;border-radius:8px}svg text{font:12px system-ui;fill:#526176}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:right;padding:12px;border-bottom:1px solid #dde3ed}th:first-child,td:first-child{text-align:left}code{overflow-wrap:anywhere}</style></head><body>'
            f'<h1>Frozen XQC prior versus MPPI</h1><p>Status: <b>{html.escape(summary["status"])}</b>; {len(rows)} of {len(summary["expected_steps"])} checkpoints. '
            f'Paired environment seeds: {html.escape(str(summary["seeds"]))}. {html.escape(summary["statistics"])}</p>'
            '<p><span style="color:#2563eb">Blue: deterministic actor prior.</span> <span style="color:#dc6724">Orange: MPPI weighted-elite action.</span> '
            'MPPI H3/N512/E64/pi24/J8 uses frozen online twin soft Q converted with the real reward scale. Both controllers make zero optimizer updates.</p>'
            + plot([("prior", "return_mean", "#2563eb"), ("mppi", "return_mean", "#dc6724")], "Mean episode return")
            + plot([("paired", "delta_mean", "#198566")], "Mean paired gain: MPPI minus prior")
            + '<h2>Checkpoint comparisons</h2><table><thead><tr><th>Checkpoint</th><th>Prior return ± SD</th><th>MPPI return ± SD</th><th>Paired gain ± SD</th><th>Prior seconds/decision</th><th>MPPI seconds/decision</th></tr></thead><tbody>'
            + ''.join(body) + '</tbody></table>'
            + f'<p>Missing steps: {html.escape(str(summary["missing_steps"]))}. Verified unchanged full outer state for every included controller.</p>'
            + f'<p>Evaluation commit: <code>{html.escape(summary["evaluation_source_sha"])}</code><br>Checkpoint manifest SHA-256: <code>{summary["checkpoint_manifest_sha256"]}</code></p></body></html>')


def publish_campaign(summary, *, project=None, entity=None, mode="online", eval_run_map=None, inventory_path=None):
    """Stage validated raw results for the shared CPU publisher; never create a run."""
    from utils.ambi_benchmark import resolve_eval_run_map, stage_completed_bundle
    assigned = resolve_eval_run_map(['controller/prior', 'controller/mppi'], run_map=eval_run_map, wandb=True)
    return {str(row["checkpoint_step"]): stage_completed_bundle(
        row["bundle"], assigned, source_run=summary["source_run"], inventory_path=inventory_path,
    ) for row in summary["rows"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Omit absent checkpoint directories; reject any present incomplete/invalid output.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--eval-run-map", type=Path, help="Explicit selector-to-run-directory mapping for staging completed raw results.")
    parser.add_argument("--wandb-project", default="ambi-inner-bench")
    parser.add_argument("--wandb-entity", default="rwgao_b-brown-university")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    outputs = [path.resolve() for path in (args.output, args.html) if path is not None]
    if len(set(outputs)) != len(outputs) or any(path == root or root in path.parents for path in outputs):
        raise ValueError("Distinct campaign outputs must be outside the source checkout.")
    for path in outputs:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}; use --overwrite.")
    summary = summarize(args.manifest, args.results_root, expected_source_sha=args.expected_source_sha,
                        allow_partial=args.allow_partial)
    atomic_json(args.output, summary, overwrite=args.overwrite)
    if args.html:
        atomic_write(args.html, render_campaign_html(summary).encode(), overwrite=args.overwrite)
    if args.wandb or args.eval_run_map:
        summary["publication"] = publish_campaign(summary, eval_run_map=args.eval_run_map, inventory_path=args.manifest)
        atomic_json(args.output, summary, overwrite=True)
    print(json.dumps({"output": str(args.output), "status": summary["status"], "checkpoints": len(summary["rows"]),
                      "wandb_url": summary.get("wandb_url")}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
