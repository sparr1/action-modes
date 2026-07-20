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
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from RL.tdmpc2_core import MODEL_SIZE
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
    / "ambi"
    / "legacy"
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
        help="Legacy frozen-checkpoint matrix (default: configs/ambi/legacy/ambi_inner_decoupling.json).",
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
        "--allow-nonfinite-metrics",
        action="store_true",
        help="Record rather than fail on NaN/Inf model diagnostics.",
    )
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
    except Exception:
        env.close()
        raise


def _seed_spaces(env, seed):
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


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


def _initialize_frozen_model(resolved, env, checkpoint, controller_seed, device=None):
    run_config = copy.deepcopy(resolved["algorithm_config"])
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
    except Exception as exc:
        raise RuntimeError(
            f"Preset {resolved['selector']!r} could not load checkpoint {checkpoint}: {exc}. "
            "Q-representation comparisons require a separately trained checkpoint with the "
            "matching critic architecture."
        ) from exc
    model.agent.model.eval()
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
    try:
        model, run_config = _initialize_frozen_model(
            resolved, env, checkpoint, controller_seed, device=device
        )
        digest_before = _outer_state_digest(model)
        updates_before = int(model.agent.num_updates)
    except Exception:
        env.close()
        raise
    metric_values = {}
    nonfinite_metric_counts = {}
    episodes = []

    try:
        for seed in seeds:
            seed = int(seed)
            _seed_spaces(env, seed)
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            episode_return = 0.0
            episode_steps = 0
            episode_metric_values = {}
            episode_nonfinite_counts = {}
            truncated_by_evaluator = False

            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                    episode_start=(episode_steps == 0),
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_steps += 1

                finite_metrics, nonfinite_metrics = _numeric_metrics(
                    getattr(model.agent, "last_inner_metrics", {})
                )
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
                    "return": episode_return,
                    "length": episode_steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "truncated_by_evaluator": truncated_by_evaluator,
                    "model_metrics": {
                        key: float(np.mean(values))
                        for key, values in sorted(episode_metric_values.items())
                    },
                    "nonfinite_model_metrics": dict(
                        sorted(episode_nonfinite_counts.items())
                    ),
                }
            )
    finally:
        env.close()

    digest_after = _outer_state_digest(model)
    updates_after = int(model.agent.num_updates)
    if digest_after != digest_before or updates_after != updates_before:
        raise RuntimeError(
            f"Frozen evaluation invariant failed for {resolved['selector']}: "
            "outer state changed during action selection."
        )
    if nonfinite_metric_counts and not allow_nonfinite_metrics:
        raise RuntimeError(
            f"Non-finite model metrics for {resolved['selector']}: "
            f"{dict(sorted(nonfinite_metric_counts.items()))}. Use "
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
        "outer_updates_before": updates_before,
        "outer_updates_after": updates_after,
        "outer_state_unchanged": True,
        "resolved_config": _jsonable(vars(model.cfg)),
        "resolved_device": str(model.agent.device),
        "return": _summary(returns),
        "episode_length": _summary(lengths),
        "episodes": episodes,
        "model_metrics": _aggregate_metrics(metric_values),
        "model_metric_availability": sorted(metric_values),
        "nonfinite_model_metrics": dict(sorted(nonfinite_metric_counts.items())),
        "alg_params": run_config["alg_params"],
    }


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
):
    """Evaluate selected presets from a matrix with paired seeds."""
    matrix_path = Path(matrix_path).resolve()
    matrix = load_preset_matrix(matrix_path)
    selectors = normalize_selectors(matrix, selectors, comparisons)
    evaluation = matrix.get("evaluation", {})
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

    resolved_presets = [
        resolve_preset(matrix_path, selector, matrix=matrix) for selector in selectors
    ]
    _validate_frozen_selection(matrix, resolved_presets)
    results = []
    for resolved in resolved_presets:
        results.append(
            evaluate_preset(
                resolved,
                checkpoint,
                seeds,
                controller_seed=controller_seed,
                max_steps=max_steps,
                device=device,
                allow_nonfinite_metrics=allow_nonfinite_metrics,
            )
        )
    _attach_paired_return_deltas(results)
    metric_sets = [set(result["model_metric_availability"]) for result in results]
    return {
        "schema_version": 1,
        "matrix": str(matrix_path),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "matrix_sha256": _file_sha256(matrix_path),
        "frozen_outer_learning": True,
        "deterministic_execution": True,
        "environment": copy.deepcopy(matrix["environment"]),
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
            written = materialize_presets(
                args.matrix,
                args.materialize_dir,
                selectors=selectors,
            )
            for path in written:
                print(f"materialized {path}")
        if args.checkpoint is None:
            if args.list_presets or args.materialize_dir is not None:
                return 0
            parser.error("--checkpoint is required for evaluation.")

        payload = evaluate_matrix(
            args.matrix,
            args.checkpoint,
            selectors=selectors,
            seeds=args.seeds,
            controller_seed=args.controller_seed,
            max_steps=args.max_steps,
            device=args.device,
            allow_nonfinite_metrics=args.allow_nonfinite_metrics,
        )
        serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            print(serialized, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
            print(f"wrote {args.output}")
        return 0
    except (PresetMatrixError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
