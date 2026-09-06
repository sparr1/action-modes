"""Evaluate native inner XQC against immutable prior-reference episode bundles."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import json
import math
import os
from pathlib import Path

from run_ambixqc_mppi_evaluation import SOURCE_RUN, file_sha256, select_checkpoint

ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "configs/research/ambixqc_humanoid_inner_j6_benchmark.json"
CONTROLLER_SEED = 12345
INNER_SETTINGS = {
    "inner_operator": "xqc", "inner_rounds": 6, "inner_rollouts_per_round": 512,
    "inner_rollout_horizon": 3, "inner_updates_per_round": 3, "inner_batch_size": 512,
    "inner_replay_capacity": 9216, "inner_reward_normalization": "frozen_real_scale",
    "inner_actor_lr": 5e-5, "inner_critic_lr": 5e-5, "xqc_policy_delay": 3,
}
OPTIMIZER_STEPS = {"critic": 18, "actor": 6, "temperature": 6}
MODEL_STEPS = 9216
OUTER_TERMINAL_METRICS = {
    "inner_terminal_bootstrap_outer": 1,
    "inner_outer_terminal_boundary_rows": 3072,
    "inner_outer_terminal_policy_evaluations": 9216,
    "inner_outer_terminal_q_evaluations": 9216,
}


def campaign_profile(variant="inner"):
    """Return the matrix and acceptance settings for a named XQC campaign."""
    if variant not in {"inner", "outer_terminal"}:
        raise ValueError("Variant must be inner or outer_terminal.")
    settings = dict(INNER_SETTINGS)
    matrix, label = MATRIX, "inner XQC J6/N512/H3/G3"
    if variant == "outer_terminal":
        settings["inner_terminal_bootstrap"] = "outer"
        matrix = ROOT / "configs/research/ambixqc_humanoid_outer_terminal_j6_benchmark.json"
        label = "outer-terminal XQC J6/N512/H3/G3"
    return {"variant": variant, "matrix": matrix, "inner_settings": settings, "label": label}


@contextmanager
def deterministic_evaluation(device="cpu"):
    """Scope deterministic numerical kernels to this evaluation campaign."""
    import torch

    device_type = torch.device(device).type
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device_type == "cuda" and workspace not in {":4096:8", ":16:8"}:
        raise ValueError(
            "CUDA evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 "
            "set before starting Python/CUDA; the runner will not configure it late."
        )
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    settings = {
        "device_type": device_type,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": workspace if device_type == "cuda" else None,
    }
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        yield settings
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cudnn.benchmark = previous[3]


def _saved_setting(params, key):
    # The actual source bank omits this optional key; AMBIXQC has always
    # defaulted it to frozen_real_scale. Explicit alternatives still fail.
    if key == "inner_reward_normalization":
        return params.get(key, "frozen_real_scale")
    return params.get(key)


def _read_json(path):
    from utils.ambi_benchmark import read_json
    return read_json(path)


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_frozen_run(run):
    result = run.get("result", {})
    before, after = result.get("outer_updates_before"), result.get("outer_updates_after")
    if (run.get("status") != "complete" or result.get("outer_state_unchanged") is not True
            or run.get("outer_state_unchanged", True) is not True
            or type(before) is not int or before < 0 or before != after):
        raise ValueError("Run must be complete with unchanged frozen outer state and update counters.")
    if result.get("nonfinite_model_metrics") or result.get("nonfinite_trace_metrics"):
        raise ValueError("Run contains nonfinite model or decision diagnostics.")


def _check_episodes(run, *, seeds, max_steps):
    episodes = run.get("episodes", [])
    if [episode.get("seed") for episode in episodes] != list(seeds):
        raise ValueError("Episode seeds differ from the requested protocol.")
    from utils.ambi_benchmark import solver_seed
    for episode in episodes:
        if (episode.get("status", "complete") != "complete"
                or type(episode.get("length")) is not int or episode["length"] != max_steps
                or not _finite_number(episode.get("return"))
                or episode.get("solver_seed") != solver_seed(CONTROLLER_SEED, "episode", episode["seed"])):
            raise ValueError("Episodes must have finite returns and complete matching lengths/solver seeds.")
    result = run["result"]
    if (result.get("environment_seeds") != list(seeds)
            or result.get("controller_seed") != CONTROLLER_SEED
            or result.get("seed_scheme") != "sha256-v1"):
        raise ValueError("Run result has an incompatible episode seed protocol.")
    return episodes


def validate_reference_bundle(path, *, checkpoint, expected_manifest_sha256, seeds, max_steps, protocol):
    """Validate a recorded prior before creating output or any environment."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Prior reference must be an absolute bundle or manifest path.")
    manifest_path = path if path.name == "manifest.json" else path / "manifest.json"
    if (not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64
            or file_sha256(manifest_path) != expected_manifest_sha256):
        raise ValueError("Prior reference manifest hash differs from the checkpoint inventory.")
    manifest = _read_json(manifest_path)
    from utils.ambi_benchmark import episode_protocol, run_controller_type
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "complete"
            or manifest.get("checkpoint", {}).get("sha256") != checkpoint["sha256"]
            or manifest.get("checkpoint", {}).get("metadata_sha256") != checkpoint["metadata_sha256"]):
        raise ValueError("Prior reference must be complete and match checkpoint and metadata hashes.")
    if episode_protocol(manifest.get("protocol", {})) != episode_protocol(protocol):
        raise ValueError("Prior reference environment/action/seed protocol differs from the requested evaluation.")
    priors = [run for run in manifest.get("runs", []) if run_controller_type(run) == "prior"]
    if len(priors) != 1:
        raise ValueError("Prior reference must contain exactly one prior controller.")
    prior = priors[0]
    if (prior.get("config", {}).get("alg") != "AMBIXQC/AMBIXQC"
            or prior.get("action_rule", "tanh_mean") != "tanh_mean"):
        raise ValueError("Reference must execute the AMBI-XQC persistent actor mean.")
    _check_frozen_run(prior)
    episodes = _check_episodes(prior, seeds=seeds, max_steps=max_steps)
    return {"manifest": manifest, "prior_run": prior,
            "returns": {episode["seed"]: episode["return"] for episode in episodes},
            "path": str(manifest_path.parent.resolve()), "manifest_sha256": expected_manifest_sha256}


def validate_bundle(bundle_path, *, checkpoint, reference, protocol, seeds, max_steps, variant="inner"):
    """Check the full native XQC dose and seed-paired outcomes after evaluation."""
    profile = campaign_profile(variant)
    bundle_path = Path(bundle_path)
    manifest = _read_json(bundle_path / "manifest.json")
    from utils.ambi_benchmark import episode_protocol, run_controller_type
    if (manifest.get("status") != "complete" or len(manifest.get("runs", [])) != 1
            or manifest.get("checkpoint", {}).get("sha256") != checkpoint["sha256"]
            or manifest.get("checkpoint", {}).get("metadata_sha256") != checkpoint["metadata_sha256"]):
        raise ValueError("Expected one complete candidate bundle matching the immutable checkpoint.")
    if (episode_protocol(manifest.get("protocol", {})) != episode_protocol(protocol)
            or manifest.get("reference", {}).get("manifest_sha256") != reference["manifest_sha256"]):
        raise ValueError("Candidate bundle reference or evaluation protocol differs from preflight.")
    run = manifest["runs"][0]
    if (run.get("selector") != "controller/xqc" or run_controller_type(run) != "xqc"
            or run.get("config", {}).get("alg") != "AMBIXQC/AMBIXQC"
            or run.get("action_rule") != "tanh_mean"):
        raise ValueError("Expected only the native inner-XQC controller with policy-mean real actions.")
    _check_frozen_run(run)
    episodes = _check_episodes(run, seeds=seeds, max_steps=max_steps)
    cfg = run["result"].get("resolved_config", {})
    if (any(cfg.get(key) != value for key, value in profile["inner_settings"].items())
            or cfg.get("inner_terminal_bootstrap", "inner") != ("outer" if variant == "outer_terminal" else "inner")):
        raise ValueError("Resolved inner-XQC settings differ from the authorized J6/N512/H3/G3/B512 dose.")
    seen = set()
    traces = run.get("trace_files", [])
    if len(set(traces)) != len(traces):
        raise ValueError("Candidate bundle contains duplicate trace files.")
    for relative in traces:
        path = (bundle_path / relative).resolve()
        if not path.is_relative_to(bundle_path.resolve()) or path == bundle_path.resolve():
            raise ValueError("Candidate trace path escapes its bundle.")
        with gzip.open(path, "rt") as stream:
            for line in stream:
                event = json.loads(line)
                identity = (event.get("episode_id"), event.get("decision_index"))
                if (event.get("phase") != "decision" or identity in seen
                        or type(event.get("decision_index")) is not int
                        or event.get("event_index") != 0 or event.get("nonfinite")):
                    raise ValueError("Candidate decision traces are nonfinite, duplicated, or not decision-only.")
                seen.add(identity)
                metrics = event.get("metrics", {})
                if (metrics.get("decision/inner_model_steps") != MODEL_STEPS
                        or any(event.get(f"{component}_updates") != value
                               or metrics.get(f"decision/inner_{component}_optimizer_steps") != value
                               for component, value in OPTIMIZER_STEPS.items())):
                    raise ValueError("Actual inner-XQC work must be 9216 model steps and C18/A6/T6 per decision.")
                if not all(_finite_number(value) for value in metrics.values()):
                    raise ValueError("Candidate decision metrics must be finite.")
                sampled = metrics.get("decision/inner_outer_terminal_bootstrap_rows",
                                      None if variant == "outer_terminal" else 0)
                if variant == "outer_terminal":
                    if (any(metrics.get(f"decision/{name}") != value
                            for name, value in OUTER_TERMINAL_METRICS.items())
                            or not _finite_number(sampled) or int(sampled) != sampled
                            or not 0 <= sampled <= MODEL_STEPS):
                        raise ValueError("Outer-terminal diagnostics must prove boundary rows and frozen outer policy/Q work.")
                elif sampled != 0 or any(metrics.get(f"decision/{name}", 0) != 0
                                         for name in OUTER_TERMINAL_METRICS):
                    raise ValueError("Native inner bootstrap cannot report outer-terminal work.")
    expected = {(f"seed-{seed}", decision) for seed in seeds for decision in range(max_steps)}
    if seen != expected:
        raise ValueError("Candidate per-decision diagnostics are missing, duplicated, or use unexpected episode identities.")
    gains = []
    for episode in episodes:
        gain = episode["return"] - reference["returns"][episode["seed"]]
        if not _finite_number(episode.get("paired_return_delta")) or abs(episode["paired_return_delta"] - gain) > 1e-9:
            raise ValueError("Paired return delta does not match the immutable prior reference.")
        gains.append(gain)
    return {"outer_state_unchanged": True, "decision_counts": {"controller/xqc": len(seen)},
            "optimizer_updates_per_decision": dict(OPTIMIZER_STEPS),
            "inner_model_steps_per_decision": MODEL_STEPS,
            "paired_return_delta_mean": sum(gains) / len(gains)}


def run(manifest_path, index, result_root, *, mode="production", device="cuda", wandb=False,
        smoke_reference_bundle=None, smoke_reference_manifest_sha256=None, variant="inner"):
    profile = campaign_profile(variant)
    matrix = profile["matrix"]
    if mode not in {"smoke", "production"}:
        raise ValueError("Mode must be smoke or production.")
    if wandb and mode == "smoke":
        raise ValueError("Smoke evaluation must not publish to W&B.")
    if mode == "smoke" and (not smoke_reference_bundle or not smoke_reference_manifest_sha256):
        raise ValueError("Smoke requires an explicit matching smoke reference bundle and manifest hash.")
    if mode == "production" and (smoke_reference_bundle is not None or smoke_reference_manifest_sha256 is not None):
        raise ValueError("Smoke reference overrides cannot be used for production evaluation.")
    row = select_checkpoint(manifest_path, index)
    from utils.checkpoint_context import load_checkpoint_context
    from utils.ambi_research import resolve_preset
    from utils.ambi_benchmark import atomic_json, protocol_for
    context = load_checkpoint_context(row["path"])
    saved = context.trial_run_params["alg_params"]
    frozen_defaults = {key: INNER_SETTINGS[key] for key in
                       ("inner_reward_normalization", "inner_actor_lr", "inner_critic_lr", "xqc_policy_delay")}
    if any(_saved_setting(saved, key) != value for key, value in frozen_defaults.items()):
        raise ValueError("Source checkpoint must retain frozen_real_scale, 5e-5 inner learning rates, and XQC policy delay 3.")
    resolved = resolve_preset(matrix, "controller/xqc", checkpoint_context=context)
    if any(_saved_setting(resolved["algorithm_config"]["alg_params"], key) != value
           for key, value in profile["inner_settings"].items()):
        raise ValueError("Research matrix differs from the authorized native inner-XQC settings.")
    if resolved["algorithm_config"]["alg_params"].get("inner_terminal_bootstrap", "inner") != (
            "outer" if variant == "outer_terminal" else "inner"):
        raise ValueError("Research matrix terminal bootstrap differs from the selected campaign variant.")
    seeds, max_steps = (list(range(101, 106)), 500) if mode == "production" else ([101, 102], 3)
    reference_path = row.get("reference_bundle") if mode == "production" else smoke_reference_bundle
    reference_sha = row.get("reference_manifest_sha256") if mode == "production" else smoke_reference_manifest_sha256
    if not reference_path or not reference_sha:
        raise ValueError("Checkpoint inventory is missing its immutable prior-reference bundle/hash.")
    protocol = protocol_for(resolved, CONTROLLER_SEED, max_steps)
    reference = validate_reference_bundle(reference_path, checkpoint=row, expected_manifest_sha256=reference_sha,
                                          seeds=seeds, max_steps=max_steps, protocol=protocol)
    destination = Path(result_root).resolve() / f"step_{row['step']}"
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Evaluation results must be outside the source checkout.")
    with deterministic_evaluation(device=device) as numerical_settings:
        destination.mkdir(parents=True, exist_ok=False)
        atomic_json(destination / "provenance.json", {
            "source_run": SOURCE_RUN, "checkpoint": row,
            "checkpoint_manifest_sha256": file_sha256(manifest_path),
            "matrix_sha256": file_sha256(matrix), "mode": mode, "variant": variant,
            "seeds": seeds, "max_steps": max_steps, "controller_seed": CONTROLLER_SEED,
            "reference_bundle": reference["path"], "reference_manifest_sha256": reference["manifest_sha256"],
            "numerical_settings": numerical_settings,
        })
        from evaluate_ambi_checkpoint import evaluate_matrix
        payload = evaluate_matrix(
            matrix, row["path"], selectors=["controller/xqc"], seeds=seeds,
            controller_seed=CONTROLLER_SEED, max_steps=max_steps, device=device,
            bundle_dir=destination / "bundle", reference_bundle=reference["path"],
            wandb_options={"project": "ambi-inner-bench", "entity": "rwgao_b-brown-university",
                           "mode": "online"} if wandb else None,
        )
    payload["numerical_settings"] = numerical_settings
    payload["variant"] = variant
    # Preserve evaluated results if validation or HTML generation later fails.
    atomic_json(destination / "paired.json", payload)
    if payload.get("checkpoint_sha256") != row["sha256"]:
        raise ValueError("Evaluated checkpoint hash differs from preflight.")
    if file_sha256(Path(reference["path"]) / "manifest.json") != reference["manifest_sha256"]:
        raise ValueError("Prior reference changed during evaluation.")
    validation = validate_bundle(destination / "bundle", checkpoint=row, reference=reference,
                                 protocol=protocol, seeds=seeds, max_steps=max_steps, variant=variant)
    atomic_json(destination / "validation.json", {"step": row["step"], "mode": mode,
                                                  "variant": variant, **validation})
    from report_ambi_benchmark import load_bundles, write_report
    report = load_bundles([reference["path"], destination / "bundle"])
    report["runs"] = [item for item in report["runs"] if item["controller_type"] in {"prior", "xqc"}]
    names = {name for item in report["runs"] for trace in item["traces"] for name in trace["metrics"]}
    report["metric_catalog"] = {key: value for key, value in report["metric_catalog"].items() if key in names}
    write_report(report, destination / "comparison.html",
                 title=f"AMBI-XQC prior versus {profile['label']} at {row['step']:,} decisions")
    print(json.dumps({"step": row["step"], "variant": variant,
                      "output": str(destination), **validation}, sort_keys=True))
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), default="production")
    parser.add_argument("--variant", choices=("inner", "outer_terminal"), default="inner")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--smoke-reference-bundle", type=Path)
    parser.add_argument("--smoke-reference-manifest-sha256")
    args = parser.parse_args(argv)
    run(args.manifest, args.index, args.result_root, mode=args.mode, device=args.device, wandb=args.wandb,
        variant=args.variant,
        smoke_reference_bundle=args.smoke_reference_bundle,
        smoke_reference_manifest_sha256=args.smoke_reference_manifest_sha256)


if __name__ == "__main__":
    main()
