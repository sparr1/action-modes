"""Validate native inner-XQC checkpoint results against immutable prior references."""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
from pathlib import Path

from report_ambi_benchmark import load_bundles
from run_ambixqc_inner_evaluation import campaign_profile
from run_ambixqc_mppi_evaluation import SOURCE_RUN, STEPS, file_sha256
from summarize_ambixqc_mppi_eval import _finite, _read, _sha, _stats, _validate_run as _validate_prior
from utils.ambi_benchmark import atomic_json, atomic_write, canonical_hash, episode_protocol, reference_returns, run_controller_type, solver_seed

MATRIX = Path(__file__).resolve().parent / "configs/research/ambixqc_humanoid_inner_j6_benchmark.json"
REFERENCE_SHA = "b5e70ba2c57cba36137305899e3270ef1b8e7f95"
BUDGET = {"inner_operator": "xqc", "inner_rounds": 6, "inner_rollouts_per_round": 512,
          "inner_rollout_horizon": 3, "inner_updates_per_round": 3, "inner_batch_size": 512,
          "inner_replay_capacity": 9216, "inner_replay_sampling": "with_replacement"}
COUNTS = {"critic": 18, "actor": 6, "temperature": 6}
CAPABILITIES = {"decision_metrics": True, "optimizer_traces": False, "shared_observation_probes": False}
NUMERICAL_SETTINGS = {"device_type": "cuda", "deterministic_algorithms": True, "deterministic_warn_only": False,
                      "cudnn_deterministic": True, "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8"}


def _numerical_settings(provenance, paired, entry, *, matrix=None):
    expected_hash = canonical_hash(NUMERICAL_SETTINGS)
    if any(canonical_hash(record.get("numerical_settings")) != expected_hash for record in (provenance, paired)):
        raise ValueError("Production numerical settings must record strict deterministic CUDA, deterministic cuDNN without benchmarking, and CuBLAS :4096:8 in both provenance.json and paired.json.")
    if paired.get("checkpoint_sha256") != entry["sha256"] or paired.get("matrix_sha256") != file_sha256(matrix or MATRIX):
        raise ValueError("Numerical settings belong to a different checkpoint/matrix evaluation payload.")


def _identity(manifest, entry, source):
    checkpoint, code = manifest.get("checkpoint", {}), manifest.get("code", {})
    if (manifest.get("status") != "complete" or code.get("commit") != source or code.get("dirty") is not False
            or checkpoint.get("sha256") != entry["sha256"] or checkpoint.get("metadata_sha256") != entry["metadata_sha256"]
            or checkpoint.get("metadata", {}).get("checkpoint", {}).get("step") != entry["step"]):
        raise ValueError("Incomplete bundle or mismatched checkpoint/metadata/source identity.")
    _sha(code.get("source_sha256"), 64, "Evaluation source fingerprint")
    if not isinstance(code.get("runtime"), dict) or not code["runtime"]:
        raise ValueError("Evaluation runtime provenance is required.")


def _budget(variant):
    profile = campaign_profile(variant)
    return {**BUDGET,
            **({"inner_terminal_bootstrap": "outer"} if profile["terminal_bootstrap"] == "outer" else {}),
            **({"inner_update_timing": "step", "inner_policy_delay": 1} if profile["update_timing"] == "step" else {})}


def _signature(value):
    result = copy.deepcopy(value or {})
    result.setdefault("inner_terminal_bootstrap", "inner")
    result.setdefault("inner_update_timing", "round")
    result.setdefault("inner_policy_delay", result.get("policy_delay", 3))
    return result


def _configuration(run, prior, metadata, controller_seed, *, variant="inner"):
    result, reference = run["result"], prior["result"]
    saved = metadata["trial_run_params"]
    if result.get("saved_algorithm_config") != saved or reference.get("saved_algorithm_config") != saved:
        raise ValueError("Saved algorithm configuration differs from checkpoint metadata.")
    expected = copy.deepcopy(prior["config"])
    budget = _budget(variant)
    expected["alg_params"].update(budget)
    if run["config"] != expected or expected["alg_params"].get("xqc_policy_delay") != 3:
        raise ValueError("Inner configuration must inherit the prior and change only the J6/N512/H3/G3/B512 budget, terminal bootstrap, update timing, and inner policy delay.")
    evaluated = copy.deepcopy(result.get("evaluated_algorithm_config", {}))
    expected["seed"] = controller_seed
    expected["alg_params"]["wandb"] = False
    for config in (expected, evaluated):
        config.pop("device", None)
        config.get("alg_params", {}).pop("device", None)
    if evaluated != expected:
        raise ValueError("Evaluated algorithm settings differ from the recorded configuration.")
    resolved = result.get("resolved_config", {})
    required = {**budget, "inner_reward_normalization": "frozen_real_scale",
                "inner_actor_lr": 5e-5, "inner_critic_lr": 5e-5, "xqc_policy_delay": 3}
    if any(resolved.get(key) != value for key, value in required.items()):
        raise ValueError("Resolved XQC defaults/budget must retain frozen normalization and inherited learning rates.")
    profile = campaign_profile(variant)
    terminal, timing = profile["terminal_bootstrap"], profile["update_timing"]
    if resolved.get("inner_terminal_bootstrap", "inner") != terminal:
        raise ValueError("Resolved terminal bootstrap differs from the selected variant.")
    if resolved.get("inner_update_timing", "round") != timing:
        raise ValueError("Resolved update timing differs from the selected variant.")
    delay = 1 if timing == "step" else 3
    if resolved.get("inner_policy_delay", resolved.get("xqc_policy_delay")) != delay:
        raise ValueError("Resolved inner policy delay differs from the selected variant.")
    provenance, prior_provenance = result.get("checkpoint_evaluation_provenance", {}), reference.get("checkpoint_evaluation_provenance", {})
    if (provenance.get("frozen_evaluation") is not True or prior_provenance.get("frozen_evaluation") is not True
            or _signature(provenance.get("saved_semantic_signature")) != _signature(prior_provenance.get("saved_semantic_signature"))):
        raise ValueError("Frozen checkpoint semantic provenance is missing or mismatched.")
    saved_signature = _signature(provenance.get("saved_semantic_signature"))
    if (saved_signature.get("collection_operator") != "none" or saved_signature.get("algorithm") != "AMBIXQC"
            or saved_signature.get("inner_lifecycle") != "fresh_per_action"
            or saved_signature.get("reward_normalization") != "real_discounted_return_only"
            or saved_signature["inner_terminal_bootstrap"] != "inner"
            or saved_signature["inner_update_timing"] != "round"
            or saved_signature["inner_policy_delay"] != 3
            or _signature(prior_provenance.get("evaluated_semantic_signature")) != saved_signature):
        raise ValueError("Reference must retain the unchanged prior-only checkpoint semantics.")
    expected_signature = copy.deepcopy(saved_signature)
    expected_signature["collection_operator"] = "xqc"
    expected_signature["inner_terminal_bootstrap"] = terminal
    expected_signature["inner_update_timing"] = timing
    expected_signature["inner_policy_delay"] = delay
    expected_signature["inner_schedule"].update(rounds=6, rollouts=512, horizon=3, updates=3, batch_size=512, replay_capacity=9216)
    if _signature(provenance.get("evaluated_semantic_signature")) != expected_signature:
        raise ValueError("Inner evaluation changed unsupported outer or normalization semantics.")


def _validate_xqc(run, traces, *, seeds, max_steps, controller_seed, variant="inner"):
    profile = campaign_profile(variant)
    counts = profile["optimizer_steps"]
    result = run.get("result", {})
    if (run.get("status") != "complete" or result.get("outer_state_unchanged") is not True
            or run.get("outer_state_unchanged") is not True
            or type(result.get("outer_updates_before")) is not int or result["outer_updates_before"] < 0
            or result["outer_updates_before"] != result.get("outer_updates_after")):
        raise ValueError("Inner run must complete with verified unchanged full outer state.")
    if (result.get("controller") != "xqc" or result.get("deterministic_execution") is not True
            or run.get("action_rule") != "tanh_mean" or result.get("action_rule") != "tanh_mean"
            or result.get("controller_seed") != controller_seed or result.get("seed_scheme") != "sha256-v1"
            or result.get("environment_seeds") != list(seeds)):
        raise ValueError("Inner execution/controller/seed protocol mismatch.")
    if (run.get("kind") != "episodes" or run.get("roots") or run.get("diagnostic_capabilities") != CAPABILITIES
            or run.get("config_hash") != canonical_hash(run.get("config", {}))
            or run.get("nonfinite_trace_metrics") or result.get("nonfinite_model_metrics")):
        raise ValueError("Invalid inner configuration hash or decision-only diagnostics.")
    episodes = run.get("episodes", [])
    if [episode.get("seed") for episode in episodes] != list(seeds):
        raise ValueError("Inner episode seeds are missing, duplicated, or out of order.")
    by_seed = {seed: [] for seed in seeds}
    for trace in traces:
        if (trace["mode"] != "episodes" or trace["seed"] not in by_seed or trace["phase"] != ["decision"]
                or trace["nonfinite"] or any(value is None for values in trace["metrics"].values() for value in values)):
            raise ValueError("Invalid, null, or nonfinite inner decision diagnostics.")
        if any(trace[f"{component}_updates"] != [count] for component, count in counts.items()):
            raise ValueError(f"Each inner decision must perform C{counts['critic']}/A{counts['actor']}/T{counts['temperature']} accepted optimizer updates.")
        metrics = trace["metrics"]
        if metrics.get("decision/inner_policy_delay", [3]) != [profile["inner_policy_delay"]]:
            raise ValueError("Measured inner policy delay differs from the selected variant.")
        required = {"inner_model_steps": 9216, "inner_reward_normalizer_imagined_updates": 0,
                    **{f"inner_{component}_optimizer_steps": count for component, count in counts.items()}}
        if any(metrics.get(f"decision/{key}") != [value] for key, value in required.items()):
            raise ValueError("Measured inner work or frozen reward normalization differs from the campaign.")
        step_counts = {"inner_update_timing_step": 1, "inner_updates_per_rollout_step": 1,
                       "inner_collection_steps": 18}
        if profile["update_timing"] == "step":
            if any(metrics.get(f"decision/{key}") != [value] for key, value in step_counts.items()):
                raise ValueError("Step-update decision measurements must record 18 collection steps and one update slot per step.")
        elif any(metrics.get(f"decision/{key}", [0]) != [0] for key in step_counts):
            raise ValueError("Round-update variant cannot report step-update work.")
        if profile["terminal_bootstrap"] == "outer":
            outer_counts = {"inner_terminal_bootstrap_outer": 1, "inner_outer_terminal_boundary_rows": 3072,
                            "inner_outer_terminal_policy_evaluations": 9216, "inner_outer_terminal_q_evaluations": 9216}
            sampled = _finite(metrics.get("decision/inner_outer_terminal_bootstrap_rows", [None])[0], "Outer-terminal sampled rows")
            if (any(metrics.get(f"decision/{key}") != [value] for key, value in outer_counts.items())
                    or not 0 <= sampled <= 9216 or int(sampled) != sampled):
                raise ValueError("Outer-terminal decision measurements differ from the selected variant.")
        elif any(metrics.get(f"decision/{key}", [0]) != [0] for key in (
            "inner_terminal_bootstrap_outer", "inner_outer_terminal_boundary_rows", "inner_outer_terminal_bootstrap_rows",
            "inner_outer_terminal_policy_evaluations", "inner_outer_terminal_q_evaluations",
        )):
            raise ValueError("Inner variant cannot report outer-terminal bootstrap work.")
        by_seed[trace["seed"]].append(trace)
    for episode in episodes:
        if (episode.get("status") != "complete" or episode.get("length") != max_steps
                or episode.get("solver_seed") != solver_seed(controller_seed, "episode", episode["seed"])
                or episode.get("nonfinite_model_metrics")
                or episode.get("actual_optimizer_steps") != {key: value * max_steps for key, value in counts.items()}):
            raise ValueError("Incomplete inner episode or mismatched seeds/optimizer totals.")
        decisions = sorted(by_seed[episode["seed"]], key=lambda trace: trace["decision_index"])
        if [trace["decision_index"] for trace in decisions] != list(range(max_steps)):
            raise ValueError("Missing or duplicated real-decision diagnostics.")
        for field, metric in (("return", "reward"), ("control_seconds", "control_seconds")):
            values = [_finite(trace["metrics"].get(f"decision/{metric}", [None])[0], metric) for trace in decisions]
            measured = _finite(episode.get(field), field)
            if field == "control_seconds" and (measured < 0 or any(value < 0 for value in values)):
                raise ValueError("Control times cannot be negative.")
            if not math.isclose(math.fsum(values), measured, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"Episode {field} differs from its real-decision measurements.")
    if (run.get("actual_optimizer_steps_scope") != "completed_episodes"
            or run.get("actual_optimizer_steps") != {key: value * len(seeds) * max_steps for key, value in counts.items()}):
        raise ValueError("Completed-episode optimizer totals do not match measured decisions.")
    return episodes


def summarize(manifest_path, results_root, *, expected_source_sha, reference_root=None,
              seeds=(101, 102, 103, 104, 105), max_steps=500, controller_seed=12345, allow_partial=False, variant="inner"):
    profile = campaign_profile(variant)
    _sha(expected_source_sha, 40, "Expected evaluation commit")
    manifest_path, results_root = Path(manifest_path).resolve(), Path(results_root).resolve()
    inventory = _read(manifest_path)
    entries = inventory.get("checkpoints", [])
    if inventory.get("source_run") != SOURCE_RUN or [entry.get("step") for entry in entries] != list(STEPS):
        raise ValueError("Expected all 30 ordered checkpoints from the seed-55 prior-only source run.")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be distinct valid NumPy seeds.")
    if type(max_steps) is not int or max_steps <= 0 or type(controller_seed) is not int or not 0 <= controller_seed < 2**32:
        raise ValueError("Invalid decision budget or controller seed.")
    rows, missing, signature, protocol, sources = [], [], None, None, None
    for entry in entries:
        for key in ("sha256", "metadata_sha256", "reference_manifest_sha256"):
            _sha(entry.get(key), 64, key)
        if any(not isinstance(entry.get(key), str) or not Path(entry[key]).is_absolute() for key in ("path", "reference_bundle")):
            raise ValueError("Checkpoint/reference inventory paths must be absolute.")
        step = entry["step"]
        destination = results_root / "production" / f"step_{step}"
        if not destination.exists():
            missing.append(step)
            continue
        reference_path = Path(entry["reference_bundle"]) if reference_root is None else Path(reference_root) / "production" / f"step_{step}" / "bundle"
        candidate_path = destination / "bundle"
        if file_sha256(reference_path / "manifest.json") != entry["reference_manifest_sha256"]:
            raise ValueError("Prior reference manifest differs from the immutable inventory.")
        reference, candidate, provenance = _read(reference_path / "manifest.json"), _read(candidate_path / "manifest.json"), _read(destination / "provenance.json")
        paired = _read(destination / "paired.json")
        if any(record.get("variant", "inner") != variant for record in (provenance, paired)):
            raise ValueError("Recorded campaign variant differs from the selected variant.")
        _numerical_settings(provenance, paired, entry, matrix=profile["matrix"])
        expected_provenance = {"source_run": SOURCE_RUN, "checkpoint": entry, "checkpoint_manifest_sha256": file_sha256(manifest_path),
                               "matrix_sha256": file_sha256(profile["matrix"]), "mode": "production", "seeds": list(seeds),
                               "max_steps": max_steps, "controller_seed": controller_seed,
                               "reference_bundle": entry["reference_bundle"], "reference_manifest_sha256": entry["reference_manifest_sha256"]}
        if any(provenance.get(key) != value for key, value in expected_provenance.items()):
            raise ValueError("Launch provenance differs from the checkpoint/reference/protocol inventory.")
        _identity(reference, entry, REFERENCE_SHA)
        _identity(candidate, entry, expected_source_sha)
        if candidate.get("reference", {}).get("manifest_sha256") != entry["reference_manifest_sha256"]:
            raise ValueError("Candidate paired against a different reference manifest.")
        current_protocol = episode_protocol(candidate.get("protocol", {}))
        expected_protocol = {"environment": {"id": "DMControl-v0", "params": {"task": "humanoid-walk", "obs": "state", "render_mode": None}},
                             "env_wrappers": [], "env_wrapper": None, "observation": "state", "action_rule": "tanh_mean",
                             "max_steps": max_steps, "controller_seed": controller_seed, "seed_scheme": "sha256-v1"}
        if current_protocol != expected_protocol or "root_bank_id" in candidate.get("protocol", {}):
            raise ValueError("Candidate protocol differs from the paired Humanoid campaign.")
        prior_values = reference_returns(reference_path, entry["sha256"], current_protocol)
        priors = [run for run in reference["runs"] if run_controller_type(run) == "prior"]
        if len(priors) != 1 or len(candidate.get("runs", [])) != 1 or run_controller_type(candidate["runs"][0]) != "xqc":
            raise ValueError("Expected exactly one reference prior and one new native XQC run.")
        prior, run = priors[0], candidate["runs"][0]
        if candidate["checkpoint"]["metadata"] != reference["checkpoint"]["metadata"]:
            raise ValueError("Checkpoint metadata contents differ across reference and candidate.")
        _configuration(run, prior, candidate["checkpoint"]["metadata"], controller_seed, variant=variant)
        current_sources = {"reference": reference["code"], "inner_xqc": candidate["code"]}
        current_signature = canonical_hash({"protocol": current_protocol, "sources": current_sources, "config": run["config"]})
        if signature is not None and current_signature != signature:
            raise ValueError("Campaign checkpoints have inconsistent source/runtime/settings provenance.")
        signature, protocol, sources = current_signature, current_protocol, current_sources
        data = load_bundles([reference_path, candidate_path])
        loaded_prior = next(item for item in data["runs"] if item["controller_type"] == "prior")
        loaded_inner = next(item for item in data["runs"] if item["controller_type"] == "xqc"
                            and item["evaluation_id"] == candidate["evaluation_id"])
        options = {"seeds": seeds, "max_steps": max_steps, "controller_seed": controller_seed}
        episodes = {"prior": _validate_prior(prior, loaded_prior["traces"], "prior", **options),
                    "inner_xqc": _validate_xqc(run, loaded_inner["traces"], variant=variant, **options)}
        deltas = [episode["return"] - prior_values[episode["seed"]] for episode in episodes["inner_xqc"]]
        if any(not math.isclose(_finite(episode.get("paired_return_delta"), "Paired delta"), delta, rel_tol=1e-9, abs_tol=1e-8)
               for episode, delta in zip(episodes["inner_xqc"], deltas)):
            raise ValueError("Paired return delta differs from the reference episode.")
        row = {"checkpoint_step": step, "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"],
               "bundle": str(candidate_path), "bundle_manifest_sha256": file_sha256(candidate_path / "manifest.json"),
               "paired_json_sha256": file_sha256(destination / "paired.json"),
               "reference_bundle": str(reference_path), "reference_manifest_sha256": entry["reference_manifest_sha256"],
               "episode_count": len(seeds), "outer_state_unchanged": True, "actual_optimizer_steps": run["actual_optimizer_steps"],
               "paired": {"delta_mean": _stats(deltas)["mean"], "delta_std": _stats(deltas)["std"], "deltas": deltas}}
        for kind, values in episodes.items():
            returns, times = _stats([item["return"] for item in values]), _stats([item["control_seconds"] for item in values])
            row[kind] = {"return_mean": returns["mean"], "return_std": returns["std"], "returns": [item["return"] for item in values],
                         "control_seconds_mean": times["mean"], "control_seconds_std": times["std"], "control_seconds_per_decision": times["mean"] / max_steps}
        rows.append(row)
    if missing and not allow_partial:
        raise ValueError(f"Missing checkpoint outputs: {missing}; --allow-partial permits absent directories only.")
    if not rows:
        raise ValueError("No completed checkpoint results.")
    return {"schema_version": 1, "status": "partial" if missing else "complete", "source_run": SOURCE_RUN,
            "checkpoint_manifest_sha256": file_sha256(manifest_path), "evaluation_source_sha": expected_source_sha, "sources": sources,
            "protocol": protocol, "seeds": list(seeds), "expected_steps": list(STEPS), "missing_steps": missing,
            "variant": variant, "inner_settings": _budget(variant), "optimizer_steps_per_decision": profile["optimizer_steps"], "model_steps_per_decision": 9216,
            "numerical_settings": copy.deepcopy(NUMERICAL_SETTINGS), "numerical_settings_scope": "new_inner_xqc_evaluations",
            "statistics": "Raw environment returns; five paired seeds; population SD (ddof=0). XQC values use normalized reward units.", "rows": rows}


def render_html(summary):
    profile = campaign_profile(summary.get("variant", "inner"))
    outer, step = profile["terminal_bootstrap"] == "outer", profile["update_timing"] == "step"
    label = "XQC with outer terminal bootstrap" if outer else "inner XQC"
    if step:
        label += " and step updates"
    title = f"Frozen XQC: prior versus {label}"
    terminal_description = ("Only the final imagined transition bootstraps with the frozen outer actor and online critic using running BatchNorm statistics; the entropy weight remains the adapting inner temperature. Earlier transitions retain the inner actor and target critic."
                            if outer else "Imagined transitions bootstrap with the adapting inner actor and inner target critic.")
    schedule_description = ("After each parallel imagined timestep, append its 512 transitions and run one critic, actor, and temperature update before advancing the current latent states. G3 is the total per round, distributed across H3; replay accumulates within the action. The inner policy delay is 1; the frozen outer learner retains delay 3."
                            if step else "After each complete H3 imagined rollout round, run G3 XQC update slots on accumulated action-local replay.")
    counts = profile["optimizer_steps"]
    dose = f"C{counts['critic']}/A{counts['actor']}/T{counts['temperature']}"
    rows = summary["rows"]
    def chart(series, title, zero=False):
        xs = [row["checkpoint_step"] for row in rows]
        ys = [row[group][key] for row in rows for group, key, _ in series] + ([0] if zero else [])
        lo, hi = min(ys), max(ys)
        pad = max((hi - lo) * .1, 1)
        lo, hi = lo - pad, hi + pad
        x = lambda value: 65 + 810 * (value - min(xs)) / max(max(xs) - min(xs), 1)
        y = lambda value: 260 - 220 * (value - lo) / (hi - lo)
        elements = [f'<path d="M65 {y(0):.2f}H880" stroke="#777" stroke-dasharray="5 4"/>' if zero else '']
        for group, key, color in series:
            points = ' '.join(f'{x(row["checkpoint_step"]):.2f},{y(row[group][key]):.2f}' for row in rows)
            elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
            elements.extend(f'<circle cx="{x(row["checkpoint_step"]):.2f}" cy="{y(row[group][key]):.2f}" r="3" fill="{color}"><title>{group}, {row["checkpoint_step"]}: {row[group][key]:.3f}</title></circle>' for row in rows)
        elements.extend(f'<text x="5" y="{y(value):.2f}">{value:.1f}</text>' for value in (lo, (lo+hi)/2, hi))
        return f'<h2>{title}</h2><svg viewBox="0 0 900 305" role="img" aria-label="{title}"><path d="M65 35V260H880" fill="none" stroke="#aaa"/>{"".join(elements)}<text x="65" y="285">{min(xs):,}</text><text x="790" y="285">{max(xs):,}</text><text x="335" y="303">Training decisions at checkpoint</text></svg>'
    body = []
    for row in rows:
        values = [f'{row[group][key+"_mean"]:.3f} ± {row[group][key+"_std"]:.3f}' for group,key in (("prior","return"),("inner_xqc","return"),("paired","delta"))]
        body.append(f'<tr><td>{row["checkpoint_step"]:,}</td>'+''.join(f'<td>{value}</td>' for value in values)+f'<td>{row["prior"]["control_seconds_per_decision"]:.6f}</td><td>{row["inner_xqc"]["control_seconds_per_decision"]:.6f}</td></tr>')
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>body{{font:16px system-ui;color:#18263b;background:#f7f9fc;max-width:1100px;margin:40px auto;padding:0 24px}}p{{line-height:1.6}}svg{{background:white;width:100%;border:1px solid #dbe1ea}}svg text{{font:12px system-ui}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{text-align:right;padding:12px;border-bottom:1px solid #ddd}}td:first-child,th:first-child{{text-align:left}}code{{overflow-wrap:anywhere}}</style></head><body>'
            f'<h1>{title}</h1><p>{html.escape(summary["status"].capitalize())}: {len(rows)}/{len(summary["expected_steps"])} checkpoints. Seeds {html.escape(str(summary["seeds"]))}. {html.escape(summary["statistics"])}</p>'
            f'<p><span style="color:#2563a6">Blue: persistent prior.</span> <span style="color:#c46722">Orange: {label}.</span> Both execute the actor mean. J6/N512/H3/G3/B512, inner policy delay {1 if step else 3}; each decision performs {dose} and 9,216 model steps. Full outer state and real reward normalization remain frozen.</p><p>{schedule_description}</p><p>{terminal_description}</p>'
            + chart([("prior","return_mean","#2563a6"),("inner_xqc","return_mean","#c46722")],"Mean raw episode return")
            + chart([("paired","delta_mean","#21836f")],f"Mean paired gain: {label} minus prior",True)
            + f'<h2>Checkpoint comparisons</h2><table><tr><th>Checkpoint</th><th>Prior return ± SD</th><th>{label[0].upper()+label[1:]} return ± SD</th><th>Paired gain ± SD</th><th>Prior s/decision</th><th>Inner s/decision</th></tr>'+''.join(body)+'</table>'
            + f'<p>Missing steps: {html.escape(str(summary["missing_steps"]))}. Reference commit: <code>{REFERENCE_SHA}</code>. New evaluation commit: <code>{html.escape(summary["evaluation_source_sha"])}</code>.</p></body></html>')


def publish(summary, *, project=None, entity=None, mode="online", eval_run_map=None, inventory_path=None):
    """Stage validated raw results for the shared CPU publisher; never create a run."""
    from utils.ambi_benchmark import resolve_eval_run_map, stage_completed_bundle
    assigned = resolve_eval_run_map(['controller/xqc'], run_map=eval_run_map, wandb=True)
    return {str(row["checkpoint_step"]): stage_completed_bundle(
        row["bundle"], assigned, source_run=summary["source_run"], inventory_path=inventory_path,
    ) for row in summary["rows"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "results-root", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--variant", choices=("inner", "outer_terminal", "outer_terminal_step"), default="inner")
    parser.add_argument("--reference-root", type=Path, help="Relocate downloaded references under ROOT/production/step_N/bundle; hashes stay fixed.")
    parser.add_argument("--html", type=Path)
    for name in ("allow-partial", "overwrite", "wandb"):
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--eval-run-map", type=Path, help="Explicit selector-to-run-directory mapping for staging completed raw results.")
    parser.add_argument("--wandb-project", default="ambi-inner-bench")
    parser.add_argument("--wandb-entity", default="rwgao_b-brown-university")
    parser.add_argument("--wandb-mode", choices=("online","offline"), default="online")
    args = parser.parse_args(argv)
    outputs = [path.resolve() for path in (args.output,args.html) if path is not None]
    root = Path(__file__).resolve().parent
    if len(set(outputs)) != len(outputs) or any(path == root or root in path.parents for path in outputs):
        raise ValueError("Distinct outputs must be outside the source checkout.")
    for path in outputs:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}; use --overwrite.")
    summary = summarize(args.manifest,args.results_root,expected_source_sha=args.expected_source_sha,reference_root=args.reference_root,allow_partial=args.allow_partial,variant=args.variant)
    atomic_json(args.output,summary,overwrite=args.overwrite)
    if args.html:
        atomic_write(args.html,render_html(summary).encode(),overwrite=args.overwrite)
    if args.wandb or args.eval_run_map:
        summary["publication"] = publish(summary, eval_run_map=args.eval_run_map, inventory_path=args.manifest)
        atomic_json(args.output,summary,overwrite=True)
    print(json.dumps({"output":str(args.output),"status":summary["status"],"checkpoints":len(summary["rows"]),"wandb_url":summary.get("wandb_url")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
