"""Evaluate AMBI inner operators from a frozen outer-model checkpoint.

This runner never calls ``learn`` or ``agent.update``.  Every preset is loaded
into a fresh AMBI instance, receives the same controller seed and environment
seeds, and is checked after evaluation to ensure the outer model, outer
optimizers, outer entropy state, and update counters are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from RL.tdmpc2_core import MODEL_SIZE
from utils.cleanup import add_cleanup_notes, raise_cleanup_errors
from utils.ambi_research import (
    PresetMatrixError,
    load_preset_matrix,
    materialize_presets,
    normalize_selectors,
    resolve_preset,
)


DEFAULT_MATRIX = (
    Path(__file__).resolve().parent
    / "configs"
    / "research"
    / "ambi_inner_decoupling.json"
)
_MAX_NUMPY_SEED = 2**32 - 1


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate AMBI inner-loop presets without updating the outer model."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX,
        help="Frozen-checkpoint matrix (default: configs/research/ambi_inner_decoupling.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Frozen AMBI checkpoint. Not required for --list-presets or materialization only.",
    )
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        help="Preset selector such as inner_operator/sac. Repeat to compare several.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        dest="comparisons",
        help="Evaluate/materialize every variant in a comparison. Repeat as needed.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all selectors and their descriptions.",
    )
    parser.add_argument(
        "--materialize-dir",
        type=Path,
        help="Write selected presets as ordinary configs/algs-style JSON files.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Explicit environment seed per evaluation episode.",
    )
    parser.add_argument(
        "--controller-seed",
        type=int,
        help="Override the common inner-controller RNG seed from the matrix.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional per-episode safety cap; defaults to the matrix evaluation setting.",
    )
    parser.add_argument(
        "--device",
        help="Override the base algorithm device, for example cpu, cuda, or auto.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write strict JSON results here instead of printing them to stdout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing --output file.",
    )
    parser.add_argument(
        "--allow-nonfinite-metrics",
        action="store_true",
        help="Record rather than fail on NaN/Inf model diagnostics.",
    )
    parser.add_argument("--metadata", type=Path, help="Checkpoint sidecar override for checkpoint-based matrices.")
    parser.add_argument("--bundle-dir", type=Path, help="Create a new portable benchmark bundle with full inner traces.")
    parser.add_argument("--save-root-bank", type=Path, help="Save shared observations from a prior-only evaluation (requires --bundle-dir).")
    parser.add_argument("--root-bank", type=Path, help="Evaluate the same saved observations with extra policy-quality probes.")
    parser.add_argument("--bank-only", action="store_true", help="Run shared-observation solves without full episodes.")
    parser.add_argument("--bank-repetitions", type=int, help="Solver repetitions per shared observation (matrix default: 3).")
    parser.add_argument("--reference-bundle", type=Path, help="Completed prior-only bundle for matched episode return deltas.")
    parser.add_argument("--wandb", action="store_true", help="Publish benchmark summaries and artifacts (requires --bundle-dir).")
    parser.add_argument("--wandb-project", default="ambi-inner-bench")
    parser.add_argument("--wandb-entity", default="rwgao_b-brown-university")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    return parser


def _list_presets(matrix, selectors=None):
    selected = None if selectors is None else set(selectors)
    for comparison_name, comparison in matrix["comparisons"].items():
        variants = [
            (variant_name, variant)
            for variant_name, variant in comparison["variants"].items()
            if selected is None
            or f"{comparison_name}/{variant_name}" in selected
        ]
        if not variants:
            continue
        print(f"{comparison_name} (reference: {comparison['reference']})")
        if comparison.get("description"):
            print(f"  {comparison['description']}")
        for variant_name, variant in variants:
            marker = "*" if variant_name == comparison["reference"] else " "
            print(
                f"  {marker} {comparison_name}/{variant_name}: "
                f"{variant.get('description', '')}"
            )


def _make_env(resolved):
    import domains  # noqa: F401  # Register project environments lazily.

    run_config = resolved["algorithm_config"]
    environment = resolved["environment"]
    env = gym.make(environment["id"], **copy.deepcopy(environment.get("params", {})))
    wrappers = list(run_config.get("env_wrappers", []))
    if "env_wrapper" in run_config:
        wrappers.append(run_config["env_wrapper"])
    try:
        for wrapper in wrappers:
            if not isinstance(wrapper, dict) or "name" not in wrapper:
                raise ValueError(f"Invalid environment wrapper configuration: {wrapper!r}")
            from utils.core import setup_wrapper

            env = setup_wrapper(env, wrapper["name"], wrapper.get("wrapper_params", {}))
        return env
    except BaseException as exc:
        _close_resources(env, primary_error=exc)
        raise


def _seed_spaces(env, seed):
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


def _close_resources(*resources, primary_error=None):
    """Attempt every close and preserve the operation's primary failure."""

    cleanup_errors = []
    for resource in resources:
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as exc:
            cleanup_errors.append(exc)
    if not cleanup_errors:
        return
    if primary_error is not None:
        add_cleanup_notes(primary_error, cleanup_errors)
        return
    raise_cleanup_errors(cleanup_errors)


def _preflight_output(path, *, overwrite):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file():
            raise IsADirectoryError(f"Evaluation output is not a file: {path}")
        if not overwrite:
            raise FileExistsError(
                f"Evaluation output already exists: {path}. Pass --overwrite to replace it."
            )


def _write_output_atomic(path, serialized, *, overwrite):
    """Publish one result file atomically, without clobbering by default."""

    path = Path(path)
    _preflight_output(path, overwrite=overwrite)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Evaluation output already exists: {path}. "
                    "Pass --overwrite to replace it."
                ) from exc
            temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _digest_update(digest, value):
    """Hash nested PyTorch state without retaining a second model-sized copy."""
    if torch.is_tensor(value):
        tensor = value.detach().contiguous().cpu()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        if tensor.numel():
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            _digest_update(digest, key)
            _digest_update(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("utf-8"))
        for item in value:
            _digest_update(digest, item)
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(repr(value).encode("utf-8"))


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _outer_state_digest(model):
    agent = model.agent
    state = {
        "model": agent.model.state_dict(),
        "world_optimizer": agent.optim.state_dict(),
        "actor_optimizer": agent.pi_optim.state_dict(),
        "outer_alpha": agent.alpha.detach(),
        "entropy_optimizer": (
            None if agent.ent_coef_optim is None else agent.ent_coef_optim.state_dict()
        ),
        "num_updates": int(agent.num_updates),
        "outer_version": int(getattr(agent, "outer_version", agent.num_updates)),
    }
    digest = hashlib.sha256()
    _digest_update(digest, state)
    return digest.hexdigest()


def _as_numeric_float(value):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return "ignored", None
        value = value.detach().float().mean().cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "ignored", None
    return ("finite", value) if math.isfinite(value) else ("nonfinite", value)


def _numeric_metrics(metrics):
    result = {}
    nonfinite = {}
    for key, value in (metrics or {}).items():
        status, value = _as_numeric_float(value)
        if status == "finite":
            result[str(key)] = value
        elif status == "nonfinite":
            nonfinite[str(key)] = repr(value)
    return result, nonfinite


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "sum": 0.0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "sum": float(array.sum()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _aggregate_metrics(metric_values):
    return {
        key: _summary(values)
        for key, values in sorted(metric_values.items())
        if values
    }


def _attach_paired_return_deltas(results):
    """Attach seed-paired return deltas whenever a reference was selected."""
    grouped = {}
    for result in results:
        grouped.setdefault(result["comparison"], []).append(result)
    for group in grouped.values():
        reference_name = group[0]["reference_variant"]
        reference = next(
            (result for result in group if result["variant"] == reference_name),
            None,
        )
        if reference is None:
            continue
        reference_returns = {
            episode["seed"]: episode["return"] for episode in reference["episodes"]
        }
        for result in group:
            candidate_returns = {
                episode["seed"]: episode["return"] for episode in result["episodes"]
            }
            common_seeds = sorted(set(reference_returns) & set(candidate_returns))
            result["paired_return_delta_vs_reference"] = _summary(
                [
                    candidate_returns[seed] - reference_returns[seed]
                    for seed in common_seeds
                ]
            )


def _critic_architecture_key(resolved):
    params = resolved["algorithm_config"]["alg_params"]
    representation = str(params.get("q_representation", "distributional")).lower()
    num_q = params.get("num_q")
    if num_q is None:
        if representation == "scalar":
            # AMBI deliberately keeps scalar SAC as a twin-Q ablation,
            # independent of the TD-MPC2 model-size ensemble.
            num_q = 2
        else:
            model_size = params.get("model_size", 5)
            model_size = 5 if model_size is None else int(model_size)
            try:
                num_q = MODEL_SIZE[model_size]["num_q"]
            except KeyError as exc:
                raise ValueError(
                    f"Cannot resolve critic architecture for model_size={model_size}; "
                    f"expected one of {list(MODEL_SIZE)}."
                ) from exc
    num_q = int(num_q)
    if representation == "scalar":
        return representation, num_q, 1, None, None
    q_num_bins = params.get("q_num_bins")
    q_vmin = params.get("q_vmin")
    q_vmax = params.get("q_vmax")
    return (
        representation,
        num_q,
        int(params.get("num_bins", 101) if q_num_bins is None else q_num_bins),
        float(params.get("vmin", -10) if q_vmin is None else q_vmin),
        float(params.get("vmax", 10) if q_vmax is None else q_vmax),
    )


def _observation_architecture_key(resolved):
    """Resolve the single-task observation contract before model creation."""
    alg_params = resolved["algorithm_config"].get("alg_params", {})
    env_params = resolved.get("environment", {}).get("params", {})
    env_mode = env_params.get("obs", env_params.get("observation_type"))
    alg_mode = alg_params.get("obs")
    if env_mode is not None:
        env_mode = str(env_mode).lower()
    if alg_mode is not None:
        alg_mode = str(alg_mode).lower()
    if env_mode is not None and alg_mode is not None and env_mode != alg_mode:
        raise ValueError(
            f"Preset {resolved['selector']!r} has conflicting observation modes: "
            f"environment={env_mode!r}, algorithm={alg_mode!r}."
        )
    mode = env_mode or alg_mode or "state"
    if mode not in {"state", "rgb"}:
        raise ValueError(
            f"Preset {resolved['selector']!r} has unsupported observation mode "
            f"{mode!r}; expected 'state' or 'rgb'."
        )
    if mode == "rgb":
        return mode, (9, 64, 64), "uint8"
    return mode, None, "float32"


def _validate_frozen_selection(matrix, resolved_presets):
    for resolved in resolved_presets:
        comparison = matrix["comparisons"][resolved["comparison"]]
        if not comparison.get("frozen_evaluation", True):
            reason = comparison.get(
                "frozen_evaluation_reason",
                "This comparison is materialization/train-only.",
            )
            raise ValueError(
                f"Preset {resolved['selector']!r} cannot be used in frozen evaluation: "
                f"{reason}"
            )
    architectures = {
        _critic_architecture_key(resolved) for resolved in resolved_presets
    }
    if len(architectures) > 1:
        raise ValueError(
            "A single checkpoint cannot evaluate presets with different critic "
            "architectures. Evaluate exactly one Q-representation preset per invocation "
            "with its matching checkpoint."
        )
    observations = {
        _observation_architecture_key(resolved) for resolved in resolved_presets
    }
    if len(observations) > 1:
        raise ValueError(
            "A single checkpoint cannot evaluate presets with different observation "
            "contracts. Evaluate state and RGB presets separately with their matching "
            "checkpoints."
        )


def _initialize_frozen_model(resolved, env, checkpoint, controller_seed, device=None):
    run_config = copy.deepcopy(resolved["algorithm_config"])
    if run_config.get("alg_params", {}).get("inner_outer_replay_fraction", 0.0) > 0:
        raise ValueError(
            "Frozen checkpoint evaluation has no real replay: "
            "inner_outer_replay_fraction must be 0. Evaluate replay mixing with "
            "a training agent whose outer replay contains usable transitions."
        )
    run_config["env"] = resolved["environment"]["id"]
    run_config["seed"] = int(controller_seed)
    if device is not None:
        run_config["device"] = device
        run_config.setdefault("alg_params", {})["device"] = device
    # W&B is initialized only by learn(), which this utility never calls.  Set
    # the flag false as an additional guard for future wrappers.
    run_config.setdefault("alg_params", {})["wandb"] = False
    algorithm_path = run_config.get("alg", "")
    if "/" not in algorithm_path or algorithm_path.startswith("baselines/"):
        raise ValueError(
            "Frozen AMBI evaluation requires a project algorithm path such as "
            "'AMBITDMPC2/AMBITDMPC2'."
        )
    module_name, class_name = algorithm_path.rsplit("/", 1)
    module = importlib.import_module(f"RL.{module_name.replace('/', '.')}")
    algorithm_class = getattr(module, class_name)
    model = algorithm_class(
        class_name,
        env,
        run_config["alg_params"],
        run_config,
        {"frozen_checkpoint_evaluation": True},
    )
    try:
        model.load(str(checkpoint))
        model.agent.model.eval()
    except BaseException as exc:
        if isinstance(exc, Exception):
            error = RuntimeError(
                f"Preset {resolved['selector']!r} could not load checkpoint {checkpoint}: {exc}. "
                "Q-representation comparisons require a separately trained checkpoint with the "
                "matching critic architecture."
            )
        else:
            error = exc
        _close_resources(model, primary_error=error)
        if error is exc:
            raise
        raise error from exc
    return model, run_config


def evaluate_preset(
    resolved,
    checkpoint,
    seeds,
    *,
    controller_seed,
    max_steps=None,
    device=None,
    allow_nonfinite_metrics=False,
    bundle=None,
    bundle_run=None,
    root_bank=None,
    bank_only=False,
    bank_repetitions=3,
    captured_roots=None,
    root_decisions=(0, 100, 200, 300, 400),
    probe_rollouts=8,
    probe_horizon=3,
):
    """Evaluate one resolved preset and verify outer-state immutability."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= int(seed) <= _MAX_NUMPY_SEED
        for seed in seeds
    ):
        raise ValueError("Evaluation seeds must be valid NumPy seed integers.")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must not contain duplicates.")
    if isinstance(controller_seed, bool) or not 0 <= int(controller_seed) <= _MAX_NUMPY_SEED:
        raise ValueError("controller_seed must be a valid NumPy seed integer.")
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("max_steps must be positive when provided.")

    env = _make_env(resolved)
    model = None
    primary_error = None
    pending_events = []
    phase_id = "initialization"
    active_episode = False
    from utils.ambi_benchmark import capture_root, solver_seed
    try:
        started = time.perf_counter()
        model, run_config = _initialize_frozen_model(
            resolved, env, checkpoint, controller_seed, device=device
        )
        digest_before = _outer_state_digest(model)
        updates_before = int(model.agent.num_updates)
        metric_values = {}
        nonfinite_metric_counts = {}
        episodes = []
        bank_metric_values = {}

        if bundle is not None:
            from RL.tdmpc2_core.inner_trace import InnerActionTrace
            bundle_run["initialization_seconds"] = time.perf_counter() - started
            bundle_run["resolved_config"] = _jsonable(vars(model.cfg))
            if root_bank is not None:
                for root in root_bank["roots"]:
                    if tuple(root["shape"]) != tuple(env.observation_space.shape):
                        raise ValueError("Bank observation shape does not match checkpoint environment.")
            # One unscored action pays lazy compilation before the measured
            # solves. Reset RNG afterwards; warmup never enters return statistics.
            warm_observation = (np.asarray(root_bank["roots"][0]["observation"], dtype=np.float32)
                                if root_bank is not None else env.reset(seed=int(seeds[0]))[0])
            started = time.perf_counter()
            model.predict(warm_observation, deterministic=True, episode_start=True)
            bundle_run["warmup_including_compile_seconds"] = time.perf_counter() - started

        if root_bank is not None:
            for root in root_bank["roots"]:
                for repeat in range(bank_repetitions):
                    root_seed = solver_seed(controller_seed, "root", root["root_id"], repeat)
                    model.agent.inner_engine.reset_for_evaluation(root_seed, reuse_action_pool=True)
                    trace = InnerActionTrace(probes=True,
                        probe_seed=solver_seed(controller_seed, "probe", root["root_id"], repeat),
                        probe_rollouts=probe_rollouts, probe_horizon=probe_horizon)
                    phase_id = f"{root['root_id']}-repeat-{repeat}"
                    started = time.perf_counter()
                    model.predict(np.asarray(root["observation"], dtype=np.float32),
                                  deterministic=True, episode_start=True, trace=trace)
                    elapsed = time.perf_counter() - started
                    pending_events = [{"root_id": root["root_id"], "repeat": repeat,
                                       "decision_index": root["decision_index"], **event}
                                      for event in trace.events]
                    finite, nonfinite = _numeric_metrics(model.agent.last_inner_metrics)
                    probes = [event["metrics"] for event in trace.events if event["phase"] == "probe"]
                    diagnostic_seconds = sum(item.get("probe_seconds", 0.0) for item in probes)
                    diagnostic_model_steps = sum(item.get("probe_model_steps", 0.0) for item in probes)
                    bank_values = {**finite,
                        **{f"final_probe/{key}": value for key, value in (probes[-1] if probes else {}).items()},
                        "prediction_seconds": elapsed, "probe_seconds": diagnostic_seconds,
                        "probe_model_steps": diagnostic_model_steps,
                        "control_seconds": max(0.0, elapsed - diagnostic_seconds)}
                    for key, value in bank_values.items():
                        if math.isfinite(value):
                            bank_metric_values.setdefault(key, []).append(value)
                    root_result = {"root_id": root["root_id"], "repeat": repeat,
                                   "solver_seed": root_seed, "root_bank_id": root_bank["id"],
                                   "probe_seed": trace.probe_seed, "probe_rollouts": probe_rollouts,
                                   "probe_horizon": probe_horizon, "prediction_seconds": elapsed,
                                   "probe_seconds": diagnostic_seconds, "probe_model_steps": diagnostic_model_steps,
                                   "control_seconds": max(0.0, elapsed - diagnostic_seconds),
                                   "model_metrics": finite, "nonfinite_model_metrics": nonfinite}
                    bundle_run["roots"].append(root_result)
                    bundle.write_trace(bundle_run, phase_id, pending_events)
                    pending_events = []
                    for key in nonfinite:
                        nonfinite_metric_counts[key] = nonfinite_metric_counts.get(key, 0) + 1

        for seed in ([] if bank_only else seeds):
            seed = int(seed)
            episode_seed = solver_seed(controller_seed, "episode", seed)
            engine = getattr(model.agent, "inner_engine", None)
            if engine is not None:
                engine.reset_for_evaluation(episode_seed, reuse_action_pool=bundle is not None)
            _seed_spaces(env, seed)
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            episode_return = 0.0
            episode_steps = 0
            episode_metric_values = {}
            episode_nonfinite_counts = {}
            truncated_by_evaluator = False
            control_seconds = 0.0
            phase_id = f"seed-{seed}"
            active_episode = True

            while not (terminated or truncated):
                if captured_roots is not None and episode_steps in root_decisions:
                    captured_roots.append(capture_root(observation, seed, episode_steps, episode_return))
                trace = InnerActionTrace() if bundle is not None else None
                predict_options = {"trace": trace} if trace is not None else {}
                started = time.perf_counter()
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                    episode_start=(episode_steps == 0),
                    **predict_options,
                )
                action_seconds = time.perf_counter() - started
                control_seconds += action_seconds
                if trace is not None:
                    pending_events.extend({"episode_id": phase_id, "decision_index": episode_steps, **event}
                                          for event in trace.events)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_steps += 1

                finite_metrics, nonfinite_metrics = _numeric_metrics(
                    getattr(model.agent, "last_inner_metrics", {})
                )
                if trace is not None:
                    tail = trace.events[-1] if trace.events else {}
                    pending_events.append({
                        "episode_id": phase_id, "decision_index": episode_steps - 1,
                        "event_index": len(trace.events), "phase": "decision",
                        **{key: tail.get(key, 0) for key in
                           ("round_index", "critic_updates", "actor_updates", "temperature_updates")},
                        "metrics": {**{f"decision/{key}": value for key, value in finite_metrics.items()},
                                    **{f"decision/{key}": None for key in nonfinite_metrics},
                                    "decision/reward": float(reward),
                                    "decision/control_seconds": action_seconds},
                        "nonfinite": {f"decision/{key}": value for key, value in nonfinite_metrics.items()},
                    })
                for key, value in finite_metrics.items():
                    metric_values.setdefault(key, []).append(value)
                    episode_metric_values.setdefault(key, []).append(value)
                for key in nonfinite_metrics:
                    nonfinite_metric_counts[key] = nonfinite_metric_counts.get(key, 0) + 1
                    episode_nonfinite_counts[key] = episode_nonfinite_counts.get(key, 0) + 1

                if (
                    max_steps is not None
                    and episode_steps >= int(max_steps)
                    and not (terminated or truncated)
                ):
                    truncated_by_evaluator = True
                    truncated = True

            if not math.isfinite(episode_return):
                raise RuntimeError(
                    f"Non-finite return for preset {resolved['selector']} at seed {seed}."
                )
            episodes.append(
                {
                    "seed": seed,
                    "solver_seed": episode_seed,
                    "return": episode_return,
                    "length": episode_steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "truncated_by_evaluator": truncated_by_evaluator,
                    "control_seconds": control_seconds,
                    "model_metrics": {
                        key: float(np.mean(values))
                        for key, values in sorted(episode_metric_values.items())
                    },
                    "nonfinite_model_metrics": dict(
                        sorted(episode_nonfinite_counts.items())
                    ),
                }
            )
            if bundle is not None:
                bundle.episode(bundle_run, episodes[-1], pending_events)
                pending_events = []
            active_episode = False

        digest_after = _outer_state_digest(model)
        updates_after = int(model.agent.num_updates)
        if digest_after != digest_before or updates_after != updates_before:
            raise RuntimeError(
                f"Frozen evaluation invariant failed for {resolved['selector']}: "
                "outer state changed during action selection."
            )
        trace_nonfinite = bundle_run.get("nonfinite_trace_metrics", {}) if bundle_run is not None else {}
        if (nonfinite_metric_counts or trace_nonfinite) and not allow_nonfinite_metrics:
            raise RuntimeError(
                f"Non-finite model metrics for {resolved['selector']}: "
                f"{dict(sorted(nonfinite_metric_counts.items()))}; trace: {trace_nonfinite}. Use "
                "--allow-nonfinite-metrics only for diagnostic collection."
            )

        returns = [episode["return"] for episode in episodes]
        lengths = [episode["length"] for episode in episodes]
        return {
            "selector": resolved["selector"],
            "comparison": resolved["comparison"],
            "variant": resolved["variant"],
            "reference_variant": resolved["reference"],
            "description": resolved["description"],
            "critic_spec": copy.deepcopy(model.agent.model.critic_signature),
            "controller_seed": int(controller_seed),
            "environment_seeds": [int(seed) for seed in seeds],
            "seed_scheme": "sha256-v1",
            "outer_updates_before": updates_before,
            "outer_updates_after": updates_after,
            "outer_state_unchanged": True,
            "resolved_config": _jsonable(vars(model.cfg)),
            "resolved_device": str(model.agent.device),
            "return": _summary(returns),
            "episode_length": _summary(lengths),
            "episodes": episodes,
            "model_metrics": _aggregate_metrics(metric_values),
            "bank_metrics": _aggregate_metrics(bank_metric_values),
            "bank_solve_count": len(bundle_run["roots"]) if bundle_run is not None else 0,
            "model_metric_availability": sorted(metric_values),
            "nonfinite_model_metrics": dict(sorted(nonfinite_metric_counts.items())),
            "alg_params": run_config["alg_params"],
            "nonfinite_trace_metrics": trace_nonfinite,
        }
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if primary_error is not None and bundle is not None and active_episode:
            if not any(item["episode_id"] == phase_id for item in bundle_run["episodes"]):
                bundle_run["episodes"].append({
                    "episode_id": phase_id, "seed": seed, "solver_seed": episode_seed,
                    "return": episode_return if math.isfinite(episode_return) else None,
                    "length": episode_steps, "status": "failed",
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "capped": truncated_by_evaluator, "control_seconds": control_seconds,
                    "inner_metrics_mean": {key: float(np.mean(values))
                                           for key, values in episode_metric_values.items()},
                })
        if bundle is not None and pending_events:
            try:
                bundle.write_trace(bundle_run, f"{phase_id}-partial", pending_events)
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
                    _close_resources(model, env, primary_error=exc)
                    raise
                add_cleanup_notes(primary_error, [exc])
        _close_resources(model, env, primary_error=primary_error)


def evaluate_matrix(
    matrix_path,
    checkpoint,
    selectors=None,
    comparisons=None,
    *,
    seeds=None,
    controller_seed=None,
    max_steps=None,
    device=None,
    allow_nonfinite_metrics=False,
    metadata_path=None,
    bundle_dir=None,
    save_root_bank=None,
    root_bank_path=None,
    bank_only=False,
    bank_repetitions=None,
    reference_bundle=None,
    wandb_options=None,
):
    """Evaluate selected presets from a matrix with paired seeds."""
    evaluation_started = time.perf_counter()
    matrix_path = Path(matrix_path).resolve()
    matrix = load_preset_matrix(matrix_path)
    selectors = normalize_selectors(matrix, selectors, comparisons)
    evaluation = matrix.get("evaluation", {})
    if bank_repetitions is None:
        bank_repetitions = evaluation.get("bank_repetitions", 3)
    root_decisions = evaluation.get("bank_decisions", [0, 100, 200, 300, 400])
    probe_rollouts = evaluation.get("diagnostic_rollouts", 8)
    probe_horizon = evaluation.get("diagnostic_horizon", 3)
    if (not isinstance(root_decisions, list) or not root_decisions
            or any(isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in root_decisions)
            or len(set(root_decisions)) != len(root_decisions)):
        raise ValueError("bank_decisions must be a nonempty list of distinct nonnegative integers.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (probe_rollouts, probe_horizon)):
        raise ValueError("Bank diagnostic_rollouts and diagnostic_horizon must be positive integers.")
    seeds = list(evaluation.get("seeds", [])) if seeds is None else list(seeds)
    if not seeds:
        raise PresetMatrixError("No evaluation seeds were supplied by CLI or matrix.")
    if controller_seed is None:
        controller_seed = int(evaluation.get("controller_seed", 0))
    else:
        if isinstance(controller_seed, bool):
            raise ValueError("controller_seed must be a valid NumPy seed integer.")
        controller_seed = int(controller_seed)
    if max_steps is None:
        max_steps = evaluation.get("max_steps")

    from utils.ambi_benchmark import (
        BenchmarkBundle, atomic_json, load_bank, make_bank, protocol_for, reference_returns,
    )
    context = None
    if matrix["base_alg_config"] == "checkpoint":
        from utils.checkpoint_context import load_checkpoint_context
        context = load_checkpoint_context(checkpoint, metadata_path=metadata_path)
    elif metadata_path is not None:
        raise ValueError("--metadata requires a checkpoint-based preset matrix.")
    resolved_presets = [resolve_preset(matrix_path, selector, matrix=matrix, checkpoint_context=context)
                        for selector in selectors]
    _validate_frozen_selection(matrix, resolved_presets)
    if not isinstance(bank_repetitions, int) or isinstance(bank_repetitions, bool) or bank_repetitions < 1:
        raise ValueError("bank_repetitions must be a positive integer.")
    if (save_root_bank or root_bank_path or bank_only or reference_bundle or wandb_options) and bundle_dir is None:
        raise ValueError("Banks, references, and W&B require --bundle-dir.")
    if bank_only and not root_bank_path:
        raise ValueError("--bank-only requires --root-bank.")
    if save_root_bank and (root_bank_path or bank_only):
        raise ValueError("Capture a prior bank separately from evaluating a saved bank.")
    if save_root_bank and (len(resolved_presets) != 1 or
            resolved_presets[0]["algorithm_config"]["alg_params"].get("inner_operator") != "none"):
        raise ValueError("Saving a bank requires exactly one prior-only preset.")
    if save_root_bank and Path(save_root_bank).exists():
        raise FileExistsError(f"Observation bank already exists: {save_root_bank}")
    if bundle_dir is not None and Path(bundle_dir).exists():
        raise FileExistsError(f"Benchmark bundle already exists: {bundle_dir}. Choose a new directory.")
    if bundle_dir is not None:
        # A selected prior supplies reference outcomes to subsequent runs and
        # their artifacts. Never add an unselected baseline or SAC workload.
        resolved_presets.sort(key=lambda item: item["algorithm_config"]["alg_params"].get("inner_operator") != "none")
        for resolved in resolved_presets:
            params = resolved["algorithm_config"]["alg_params"]
            if _observation_architecture_key(resolved)[0] != "state":
                raise ValueError("Benchmark bundles currently support state observations only.")
            if params.get("inner_operator", "sac") not in {"none", "sac"}:
                raise ValueError("Benchmark bundles currently support prior-only and SAC presets.")
            scopes = [key for key in params if key.startswith("inner_") and key.endswith("_scope")
                      and key != "inner_mppi_warm_start_scope"]
            if any(params[key] != "action" for key in scopes):
                raise ValueError("Benchmark presets require fresh action-local inner state.")
            if any(params.get(key, 0) for key in ("inner_actor_writeback_coef", "inner_critic_writeback_coef")):
                raise ValueError("Benchmark presets must disable prior writeback.")
            if int(params.get("inner_diagnostic_rollouts", 0)) != 0:
                raise ValueError("Use bank probes for extra imagined evaluations; set inner_diagnostic_rollouts=0.")
    checkpoint_sha256 = _file_sha256(checkpoint)
    protocols = [protocol_for(resolved, controller_seed, max_steps) for resolved in resolved_presets]
    if bundle_dir is not None and any(protocol != protocols[0] for protocol in protocols):
        raise ValueError("A benchmark bundle must use one common environment/action protocol.")
    protocol = protocols[0]
    root_bank = load_bank(root_bank_path, checkpoint_sha256, protocol) if root_bank_path else None
    reference = reference_returns(reference_bundle, checkpoint_sha256, protocol) if reference_bundle else None
    if reference is not None and not bank_only and any(int(seed) not in reference for seed in seeds):
        raise ValueError("Prior reference is missing requested episode seeds.")
    if root_bank is not None:
        protocol["root_bank_id"] = root_bank["id"]
    bundle = BenchmarkBundle(bundle_dir, checkpoint={
        "path": str(Path(checkpoint).resolve()), "sha256": checkpoint_sha256,
        "source_run": matrix.get("source_run"),
        "source_run_verified": False,
        "metadata": None if context is None else context.metadata,
    }, protocol=protocol, wandb=wandb_options, reference=reference) if bundle_dir is not None else None
    if bundle is not None:
        bundle.started = evaluation_started
    results = []
    captured_roots = [] if save_root_bank else None
    primary_error = None
    try:
        if root_bank is not None:
            atomic_json(bundle.path / "root_bank.json", root_bank)
        if reference_bundle is not None:
            reference_path = Path(reference_bundle)
            if reference_path.is_dir():
                reference_path /= "manifest.json"
            bundle.manifest["reference"] = {"path": str(reference_path.resolve()),
                                            "manifest_sha256": _file_sha256(reference_path)}
        for resolved in resolved_presets:
            kind = "bank" if bank_only else "both" if root_bank is not None else "episodes"
            bundle_run = bundle.start_run(resolved, kind) if bundle is not None else None
            try:
                result = evaluate_preset(
                    resolved, checkpoint, seeds,
                    controller_seed=controller_seed, max_steps=max_steps, device=device,
                    allow_nonfinite_metrics=allow_nonfinite_metrics,
                    bundle=bundle, bundle_run=bundle_run, root_bank=root_bank,
                    bank_only=bank_only, bank_repetitions=bank_repetitions,
                    captured_roots=captured_roots, root_decisions=root_decisions,
                    probe_rollouts=probe_rollouts, probe_horizon=probe_horizon,
                )
                results.append(result)
                if bundle is not None and resolved["algorithm_config"]["alg_params"].get("inner_operator") == "none":
                    bundle.reference = {episode["seed"]: episode["return"] for episode in result["episodes"]}
                if save_root_bank:
                    bank = make_bank(checkpoint_sha256, protocol, captured_roots, complete=True)
                    atomic_json(save_root_bank, bank)
                    if Path(save_root_bank).resolve() != (bundle.path / "root_bank.json").resolve():
                        atomic_json(bundle.path / "root_bank.json", bank)
                    bundle.manifest["protocol"]["root_bank_id"] = bank["id"]
                if bundle is not None:
                    bundle.finish_run(bundle_run, result=result)
            except BaseException as exc:
                if bundle is not None:
                    try:
                        bundle.finish_run(bundle_run, error=exc)
                    except BaseException as cleanup:
                        add_cleanup_notes(exc, [cleanup])
                raise
    except BaseException as exc:
        primary_error = exc
        if captured_roots and bundle is not None and not (bundle.path / "root_bank.json").exists():
            try:
                atomic_json(bundle.path / "root_bank.json",
                            make_bank(checkpoint_sha256, protocol, captured_roots, complete=False))
            except BaseException as cleanup:
                add_cleanup_notes(exc, [cleanup])
        raise
    finally:
        if bundle is not None:
            try:
                bundle.finish(error=primary_error)
            except BaseException as exc:
                if primary_error is None:
                    raise
                add_cleanup_notes(primary_error, [exc])
    _attach_paired_return_deltas(results)
    metric_sets = [set(result["model_metric_availability"]) for result in results]
    return {
        "schema_version": 1,
        "matrix": str(matrix_path),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "matrix_sha256": _file_sha256(matrix_path),
        "frozen_outer_learning": True,
        "deterministic_execution": True,
        "environment": copy.deepcopy(resolved_presets[0]["environment"]),
        "common_model_metrics": sorted(set.intersection(*metric_sets)) if metric_sets else [],
        "available_model_metrics": sorted(set.union(*metric_sets)) if metric_sets else [],
        "results": results,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        matrix = load_preset_matrix(args.matrix)
        selectors = normalize_selectors(matrix, args.presets, args.comparisons)
        if args.list_presets:
            _list_presets(
                matrix,
                selectors if args.presets or args.comparisons else None,
            )
        if args.materialize_dir is not None:
            context = None
            if matrix["base_alg_config"] == "checkpoint":
                if args.checkpoint is None:
                    parser.error("Checkpoint-based materialization requires --checkpoint and its sidecar.")
                from utils.checkpoint_context import load_checkpoint_context
                context = load_checkpoint_context(args.checkpoint, metadata_path=args.metadata)
            written = materialize_presets(
                args.matrix,
                args.materialize_dir,
                selectors=selectors,
                checkpoint_context=context,
            )
            for path in written:
                print(f"materialized {path}")
            if matrix["base_alg_config"] == "checkpoint":
                # --checkpoint supplies the base configuration here; it is not
                # an implicit request to execute the materialized workloads.
                return 0
        if args.checkpoint is None:
            if args.list_presets or args.materialize_dir is not None:
                return 0
            parser.error("--checkpoint is required for evaluation.")
        if args.overwrite and args.output is None:
            parser.error("--overwrite requires --output.")
        if args.output is not None:
            _preflight_output(args.output, overwrite=args.overwrite)

        payload = evaluate_matrix(
            args.matrix,
            args.checkpoint,
            selectors=selectors,
            seeds=args.seeds,
            controller_seed=args.controller_seed,
            max_steps=args.max_steps,
            device=args.device,
            allow_nonfinite_metrics=args.allow_nonfinite_metrics,
            metadata_path=args.metadata,
            bundle_dir=args.bundle_dir,
            save_root_bank=args.save_root_bank,
            root_bank_path=args.root_bank,
            bank_only=args.bank_only,
            bank_repetitions=args.bank_repetitions,
            reference_bundle=args.reference_bundle,
            wandb_options={"project": args.wandb_project, "entity": args.wandb_entity,
                           "mode": args.wandb_mode} if args.wandb else None,
        )
        serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            print(serialized, end="")
        else:
            _write_output_atomic(
                args.output,
                serialized,
                overwrite=args.overwrite,
            )
            print(f"wrote {args.output}")
        return 0
    except (PresetMatrixError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
