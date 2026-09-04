"""Same-state Monte Carlo evaluation of one frozen TD-MPC2 MPPI action.

This evaluator measures the real-environment return effect of replacing the
policy-prior action at a state visited by native MPPI with an MPPI action, then
using the deterministic frozen policy prior for every subsequent decision.
States are reconstructed by exactly replaying the native-MPPI action prefix.

An existing paired-evaluator JSON can provide the behavior trajectory.  That
path is intentionally one-draw-per-anchor: the JSON records the realized MPPI
action but not TD-MPC2's pre-plan warm-start tensor, so additional native
warm-started draws cannot be reconstructed from it.  Without ``--behavior-json``
the evaluator generates behavior and four independent MPPI draws per anchor by
default, all from the same saved native planner warm start.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import domains  # noqa: F401  # Register environments before build_env.
from evaluate_tdmpc2_mppi_checkpoint import (
    TDMPC2MPPIEvaluationError,
    _bootstrap_mean_interval,
    _capture_global_rng,
    _controller_mpc_flag,
    _file_sha256,
    _module_digest,
    _namespaced_seed,
    _numeric_metrics,
    _predicted_action_gain,
    _preflight_output,
    _restore_global_rng,
    _sample_std,
    _set_controller,
    _write_json,
)
from render_checkpoint import (
    RenderCheckpointError,
    _backend_for,
    _close_resources,
    _initialize_model,
    _prepare_run_params,
    _saved_seed,
    _seed_controller,
    _seed_spaces,
    _validate_rollout_options,
    resolve_checkpoint_path,
    resolve_render_context,
)
from utils.cleanup import add_cleanup_notes, raise_cleanup_errors
from utils.core import build_env


SCHEMA_VERSION = 1
_PAIRED_BEHAVIOR_SCHEMA_VERSION = 1
_TDMPC2_BACKEND = "tdmpc2"
_MAX_SEED = 2**32 - 1


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def _nonnegative_seed(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected an integer seed") from exc
    if not 0 <= result <= _MAX_SEED:
        raise argparse.ArgumentTypeError("seed must be between 0 and 2^32-1")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the same-state real-return gain of one native TD-MPC2 "
            "MPPI action followed by the frozen deterministic policy prior."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="TD-MPC2 checkpoint to load.")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON path.")
    parser.add_argument(
        "--behavior-json",
        type=Path,
        help=(
            "Paired evaluator JSON supplying validated native-MPPI behavior. "
            "Its recorded action is the sole MPPI draw at each anchor."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=_positive_int,
        default=12,
        help="Number of reset-seed behavior episodes (default: 12).",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_seed,
        help="First environment reset seed (default: saved trial seed).",
    )
    parser.add_argument(
        "--controller-seed",
        type=_nonnegative_seed,
        default=12345,
        help="Base seed for planner, continuation, anchors, and bootstrap streams.",
    )
    parser.add_argument(
        "--block-size",
        type=_positive_int,
        default=25,
        help="Decisions per stratified trajectory block (default: 25).",
    )
    parser.add_argument(
        "--action-draws",
        type=_positive_int,
        default=4,
        help=(
            "Fresh warm-started MPPI draws per anchor when behavior is generated "
            "locally (default: 4; ignored with --behavior-json)."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=_positive_int,
        default=20000,
        help="Whole-episode cluster-bootstrap draws (default: 20000).",
    )
    parser.add_argument(
        "--device", default="auto", help="Inference device override (default: auto)."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Explicit checkpoint metadata sidecar, overriding discovery.",
    )
    parser.add_argument(
        "--trial-settings",
        type=Path,
        help="Explicit alg_settings.json; requires --experiment-settings.",
    )
    parser.add_argument(
        "--experiment-settings",
        type=Path,
        help="Explicit settings.json; requires --trial-settings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output JSON.",
    )
    return parser


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TDMPC2MPPIEvaluationError(f"{label} must be numeric, not boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TDMPC2MPPIEvaluationError(f"{label} must be numeric.") from exc
    if not math.isfinite(result):
        raise TDMPC2MPPIEvaluationError(f"{label} must be finite.")
    return result


def _finite_action(value: Any, label: str, *, action_dim: int) -> np.ndarray:
    try:
        action = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TDMPC2MPPIEvaluationError(f"{label} must be a numeric vector.") from exc
    if action.size != action_dim or not bool(np.isfinite(action).all()):
        raise TDMPC2MPPIEvaluationError(
            f"{label} must contain exactly {action_dim} finite values."
        )
    return action


def _validated_predicted_action_gain(
    value: Any,
    label: str,
    *,
    action_dim: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TDMPC2MPPIEvaluationError(f"{label} must be an object.")
    required_scalars = (
        "target_q_mppi_mean_all",
        "target_q_policy_prior_mean_all",
        "target_q_mppi_minus_policy_prior",
        "policy_prior_to_mppi_action_l2",
        "diagnostic_seconds",
    )
    normalized = {
        key: _finite_float(value.get(key), f"{label}.{key}")
        for key in required_scalars
    }
    if normalized["diagnostic_seconds"] < 0.0:
        raise TDMPC2MPPIEvaluationError(
            f"{label}.diagnostic_seconds must be nonnegative."
        )
    prior_action = _finite_action(
        value.get("policy_prior_action_at_mppi_state"),
        f"{label}.policy_prior_action_at_mppi_state",
        action_dim=action_dim,
    )
    normalized["policy_prior_action_at_mppi_state"] = [
        float(component) for component in prior_action
    ]
    return normalized


def _compare_predicted_action_gains(
    recomputed: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    action_dim: int,
) -> dict[str, Any]:
    recomputed_normalized = _validated_predicted_action_gain(
        recomputed, "recomputed predicted_action_gain", action_dim=action_dim
    )
    source_normalized = _validated_predicted_action_gain(
        source, "source predicted_action_gain", action_dim=action_dim
    )
    fields = (
        "target_q_mppi_mean_all",
        "target_q_policy_prior_mean_all",
        "target_q_mppi_minus_policy_prior",
        "policy_prior_to_mppi_action_l2",
    )
    absolute_differences = {
        key: abs(recomputed_normalized[key] - source_normalized[key]) for key in fields
    }
    prior_action_exact = np.array_equal(
        np.asarray(
            recomputed_normalized["policy_prior_action_at_mppi_state"],
            dtype=np.float64,
        ),
        np.asarray(
            source_normalized["policy_prior_action_at_mppi_state"],
            dtype=np.float64,
        ),
    )
    return {
        "value_fields_exact": all(value == 0.0 for value in absolute_differences.values()),
        "policy_prior_action_exact": bool(prior_action_exact),
        "all_scientific_fields_exact": bool(
            prior_action_exact
            and all(value == 0.0 for value in absolute_differences.values())
        ),
        "absolute_differences": absolute_differences,
        "timing_field_intentionally_ignored": "diagnostic_seconds",
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TDMPC2MPPIEvaluationError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TDMPC2MPPIEvaluationError(f"{label} must contain a JSON object.")
    return value


def _anchor_steps(
    episode_length: int,
    block_size: int,
    *,
    seed_base: int,
    environment_seed: int,
) -> list[dict[str, int]]:
    """Choose one reproducible hash-offset anchor in every trajectory block."""

    anchors: list[dict[str, int]] = []
    for block, start in enumerate(range(0, episode_length, block_size)):
        stop = min(start + block_size, episode_length)
        offset_seed = _namespaced_seed(
            seed_base,
            f"same-state-anchor-block-{block}",
            environment_seed,
        )
        step = start + (offset_seed % (stop - start))
        anchors.append(
            {
                "block": block,
                "block_start": start,
                "block_stop_exclusive": stop,
                "step": step,
                "offset_seed": offset_seed,
            }
        )
    return anchors


def _observation_sha256(observation: Any) -> str:
    array = np.asarray(observation)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array).reshape(-1).view(np.uint8).tobytes())
    return digest.hexdigest()


def _assert_observation_exact(actual: Any, expected: Any, label: str) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if (
        actual_array.dtype != expected_array.dtype
        or actual_array.shape != expected_array.shape
        or not np.array_equal(actual_array, expected_array)
    ):
        raise TDMPC2MPPIEvaluationError(
            f"{label} did not reproduce the exact anchor observation."
        )


def _validate_behavior_json(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    algorithm: str,
    environment: str,
    first_seed: int,
    episodes: int,
    controller_seed: int,
    episode_length: int,
    action_dim: int,
) -> list[dict[str, Any]]:
    """Validate and normalize paired-evaluator native-MPPI trajectories."""

    if payload.get("schema_version") != _PAIRED_BEHAVIOR_SCHEMA_VERSION:
        raise TDMPC2MPPIEvaluationError(
            "Behavior JSON must use paired-evaluator schema_version 1."
        )
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise TDMPC2MPPIEvaluationError(
            "Behavior JSON checkpoint SHA-256 does not match the loaded checkpoint."
        )
    if payload.get("algorithm") != algorithm or payload.get("environment") != environment:
        raise TDMPC2MPPIEvaluationError(
            "Behavior JSON algorithm/environment does not match checkpoint metadata."
        )

    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TDMPC2MPPIEvaluationError("Behavior JSON protocol must be an object.")
    controllers = protocol.get("controllers")
    if controllers != ["policy_prior_mean", "native_mppi"]:
        raise TDMPC2MPPIEvaluationError(
            "Behavior JSON must be the paired prior/native-MPPI protocol."
        )
    if protocol.get("max_steps") is not None:
        raise TDMPC2MPPIEvaluationError(
            "Behavior JSON must contain uncapped full episodes (max_steps=null)."
        )
    if protocol.get("controller_seed_base") != int(controller_seed):
        raise TDMPC2MPPIEvaluationError(
            "Behavior JSON controller_seed_base does not match --controller-seed."
        )

    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) < episodes:
        raise TDMPC2MPPIEvaluationError(
            f"Behavior JSON must contain at least {episodes} episodes."
        )
    all_seeds: list[int] = []
    for index, raw_episode in enumerate(raw_episodes):
        if not isinstance(raw_episode, Mapping):
            raise TDMPC2MPPIEvaluationError(
                f"Behavior episode {index + 1} must be an object."
            )
        seed_value = raw_episode.get("environment_seed")
        if isinstance(seed_value, bool) or not isinstance(seed_value, Integral):
            raise TDMPC2MPPIEvaluationError(
                f"Behavior episode {index + 1} has an invalid environment seed."
            )
        all_seeds.append(int(seed_value))
    if all_seeds != list(range(all_seeds[0], all_seeds[0] + len(all_seeds))):
        raise TDMPC2MPPIEvaluationError(
            "Behavior JSON environment seeds must be unique and consecutive."
        )
    if protocol.get("environment_seed_first") != all_seeds[0] or protocol.get(
        "environment_seed_last"
    ) != all_seeds[-1]:
        raise TDMPC2MPPIEvaluationError(
            "Behavior JSON protocol seed range disagrees with its episode records."
        )

    selected = raw_episodes[:episodes]
    expected_seeds = list(range(first_seed, first_seed + episodes))
    if all_seeds[:episodes] != expected_seeds:
        raise TDMPC2MPPIEvaluationError(
            "Behavior JSON first selected environment seeds do not match --seed/--episodes."
        )

    normalized: list[dict[str, Any]] = []
    for index, raw_episode in enumerate(selected):
        label = f"behavior episode {index + 1}"
        if raw_episode.get("episode") != index + 1:
            raise TDMPC2MPPIEvaluationError(f"{label} has a nonsequential episode index.")
        arm = raw_episode.get("native_mppi")
        if not isinstance(arm, Mapping) or arm.get("controller") != "native_mppi":
            raise TDMPC2MPPIEvaluationError(f"{label} lacks a native_mppi arm.")
        environment_seed = expected_seeds[index]
        expected_controller_seed = _namespaced_seed(
            controller_seed, "native_mppi", environment_seed
        )
        if arm.get("controller_seed") != expected_controller_seed:
            raise TDMPC2MPPIEvaluationError(
                f"{label} native-MPPI controller seed is inconsistent with the protocol."
            )
        if arm.get("length") != episode_length:
            raise TDMPC2MPPIEvaluationError(
                f"{label} must have the full {episode_length}-decision length."
            )
        if arm.get("capped") is not False or arm.get("terminated") is not False:
            raise TDMPC2MPPIEvaluationError(
                f"{label} must be uncapped and non-terminated."
            )
        if arm.get("truncated") is not True:
            raise TDMPC2MPPIEvaluationError(
                f"{label} must end by the fixed-horizon truncation."
            )
        steps = arm.get("steps")
        if not isinstance(steps, list) or len(steps) != episode_length:
            raise TDMPC2MPPIEvaluationError(
                f"{label} must contain all {episode_length} native-MPPI steps."
            )

        cumulative = 0.0
        normalized_steps: list[dict[str, Any]] = []
        for step_index, raw_step in enumerate(steps):
            step_label = f"{label} step {step_index}"
            if not isinstance(raw_step, Mapping) or raw_step.get("step") != step_index:
                raise TDMPC2MPPIEvaluationError(
                    f"{step_label} is missing or has the wrong step index."
                )
            action = _finite_action(
                raw_step.get("action"), f"{step_label} action", action_dim=action_dim
            )
            reward = _finite_float(raw_step.get("reward"), f"{step_label} reward")
            cumulative = _finite_float(cumulative + reward, f"{step_label} cumulative")
            recorded_cumulative = _finite_float(
                raw_step.get("cumulative_return"), f"{step_label} cumulative_return"
            )
            if recorded_cumulative != cumulative:
                raise TDMPC2MPPIEvaluationError(
                    f"{step_label} cumulative_return is inconsistent with rewards."
                )
            expected_truncated = step_index == episode_length - 1
            if raw_step.get("terminated") is not False or raw_step.get(
                "truncated"
            ) is not expected_truncated:
                raise TDMPC2MPPIEvaluationError(
                    f"{step_label} has invalid fixed-horizon termination flags."
                )
            planner = raw_step.get("planner")
            if not isinstance(planner, Mapping):
                raise TDMPC2MPPIEvaluationError(
                    f"{step_label} lacks native-MPPI planner diagnostics."
                )
            normalized_steps.append(
                {
                    "step": step_index,
                    "action": action,
                    "reward": reward,
                    "cumulative_return": cumulative,
                    "terminated": False,
                    "truncated": expected_truncated,
                    "planner": _numeric_metrics(planner),
                    "source_predicted_action_gain": _validated_predicted_action_gain(
                        raw_step.get("predicted_action_gain"),
                        f"{step_label} predicted_action_gain",
                        action_dim=action_dim,
                    ),
                }
            )
        recorded_return = _finite_float(arm.get("return"), f"{label} return")
        if recorded_return != cumulative:
            raise TDMPC2MPPIEvaluationError(
                f"{label} return is inconsistent with its reward trace."
            )
        normalized.append(
            {
                "episode": index + 1,
                "environment_seed": environment_seed,
                "controller_seed": expected_controller_seed,
                "return": cumulative,
                "steps": normalized_steps,
            }
        )
    return normalized


def _capture_operational_state(model: Any) -> dict[str, Any]:
    agent = model.agent
    previous_mean = getattr(agent, "_prev_mean", None)
    if previous_mean is not None and not torch.is_tensor(previous_mean):
        raise TDMPC2MPPIEvaluationError("TD-MPC2 agent._prev_mean must be a tensor.")
    return {
        "agent_prev_mean": (
            None if previous_mean is None else previous_mean.detach().clone()
        ),
        "last_plan_metrics": copy.deepcopy(getattr(agent, "last_plan_metrics", {})),
        "resume_boundary_prepared": copy.deepcopy(
            getattr(agent, "_resume_boundary_prepared", None)
        ),
        "predict_t0": copy.deepcopy(getattr(model, "_predict_t0", None)),
        "agent_mpc": copy.deepcopy(getattr(agent.cfg, "mpc", None)),
        "model_mpc": copy.deepcopy(getattr(getattr(model, "cfg", None), "mpc", None)),
        "module_training": [
            (module, bool(module.training)) for module in agent.model.modules()
        ],
    }


def _restore_operational_state(model: Any, state: Mapping[str, Any]) -> None:
    agent = model.agent
    previous_mean = state["agent_prev_mean"]
    if previous_mean is not None:
        with torch.no_grad():
            agent._prev_mean.copy_(previous_mean)
    agent.last_plan_metrics = copy.deepcopy(state["last_plan_metrics"])
    if hasattr(agent, "_resume_boundary_prepared"):
        agent._resume_boundary_prepared = copy.deepcopy(
            state["resume_boundary_prepared"]
        )
    if hasattr(model, "_predict_t0"):
        model._predict_t0 = copy.deepcopy(state["predict_t0"])
    agent.cfg.mpc = copy.deepcopy(state["agent_mpc"])
    model_cfg = getattr(model, "cfg", None)
    if (
        model_cfg is not None
        and model_cfg is not agent.cfg
        and hasattr(model_cfg, "mpc")
    ):
        model_cfg.mpc = copy.deepcopy(state["model_mpc"])
    for module, training in state["module_training"]:
        module.training = training


def _operational_state_matches(model: Any, state: Mapping[str, Any]) -> bool:
    current = _capture_operational_state(model)
    before_mean = state["agent_prev_mean"]
    after_mean = current["agent_prev_mean"]
    mean_equal = (before_mean is None and after_mean is None) or (
        before_mean is not None
        and after_mean is not None
        and torch.equal(before_mean, after_mean)
    )
    return bool(
        mean_equal
        and current["last_plan_metrics"] == state["last_plan_metrics"]
        and current["resume_boundary_prepared"] == state["resume_boundary_prepared"]
        and current["predict_t0"] == state["predict_t0"]
        and current["agent_mpc"] == state["agent_mpc"]
        and current["model_mpc"] == state["model_mpc"]
        and [training for _, training in current["module_training"]]
        == [training for _, training in state["module_training"]]
    )


def _rng_state_matches(
    first: tuple[Any, tuple[Any, ...], torch.Tensor, Any],
    second: tuple[Any, tuple[Any, ...], torch.Tensor, Any],
) -> bool:
    if first[0] != second[0] or not np.array_equal(first[1][1], second[1][1]):
        return False
    if first[1][0] != second[1][0] or first[1][2:] != second[1][2:]:
        return False
    if not torch.equal(first[2], second[2]):
        return False
    first_cuda, second_cuda = first[3], second[3]
    # CUDA initialization is irreversible in-process. If no CUDA streams
    # existed at entry, only the Python/NumPy/CPU Torch streams can be restored;
    # newly created CUDA streams are deliberately outside the comparison.
    if second_cuda is None:
        return True
    if first_cuda is None:
        return False
    return len(first_cuda) == len(second_cuda) and all(
        torch.equal(left, right) for left, right in zip(first_cuda, second_cuda)
    )


def _capture_plan_state(model: Any) -> dict[str, Any]:
    return {
        "operational": _capture_operational_state(model),
        "rng": _capture_global_rng(),
    }


def _restore_plan_state(model: Any, state: Mapping[str, Any]) -> None:
    _restore_operational_state(model, state["operational"])
    _restore_global_rng(state["rng"])


def _predict_action(
    model: Any,
    observation: Any,
    *,
    episode_start: bool,
    label: str,
    action_dim: int,
) -> np.ndarray:
    prediction = model.predict(
        observation,
        deterministic=True,
        episode_start=episode_start,
    )
    action = prediction[0] if isinstance(prediction, tuple) else prediction
    return _finite_action(action, label, action_dim=action_dim)


def _generate_behavior_episode(
    model: Any,
    env: Any,
    *,
    episode: int,
    environment_seed: int,
    controller_seed_base: int,
    anchors: Sequence[Mapping[str, int]],
    action_draws: int,
    episode_length: int,
    action_dim: int,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    _seed_spaces(env, environment_seed)
    observation, _ = env.reset(seed=environment_seed)
    behavior_controller_seed = _namespaced_seed(
        controller_seed_base, "native_mppi", environment_seed
    )
    _seed_controller(behavior_controller_seed)
    _set_controller(model, "native_mppi")
    model.reset()
    anchor_by_step = {int(row["step"]): row for row in anchors}
    draws_by_step: dict[int, list[dict[str, Any]]] = {}
    steps: list[dict[str, Any]] = []
    cumulative = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        step = len(steps)
        if step >= episode_length:
            raise TDMPC2MPPIEvaluationError(
                "Generated native-MPPI behavior exceeded checkpoint episode_length."
            )
        current_observation = observation
        if step in anchor_by_step:
            before = _capture_plan_state(model)
            candidates: list[dict[str, Any]] = []
            selected_after: dict[str, Any] | None = None
            for draw in range(action_draws):
                _restore_plan_state(model, before)
                if draw == 0:
                    planner_seed = None
                else:
                    planner_seed = _namespaced_seed(
                        controller_seed_base,
                        f"same-state-mppi-draw-{step}-{draw}",
                        environment_seed,
                    )
                    _seed_controller(planner_seed)
                action = _predict_action(
                    model,
                    current_observation,
                    episode_start=(step == 0),
                    label="generated native-MPPI action",
                    action_dim=action_dim,
                )
                candidate = {
                    "draw": draw + 1,
                    "planner_seed": planner_seed,
                    "behavior_episode_stream_seed": (
                        behavior_controller_seed if draw == 0 else None
                    ),
                    "action": action,
                    "planner": _numeric_metrics(model.agent.last_plan_metrics),
                    "predicted_action_gain": _predicted_action_gain(
                        model, current_observation, action
                    ),
                }
                candidates.append(candidate)
                if draw == 0:
                    selected_after = _capture_plan_state(model)
            assert selected_after is not None
            _restore_plan_state(model, selected_after)
            draws_by_step[step] = candidates
            action = candidates[0]["action"]
            planner = candidates[0]["planner"]
        else:
            action = _predict_action(
                model,
                current_observation,
                episode_start=(step == 0),
                label="generated native-MPPI behavior action",
                action_dim=action_dim,
            )
            planner = _numeric_metrics(model.agent.last_plan_metrics)

        observation, reward, terminated, truncated, _ = env.step(action)
        reward = _finite_float(reward, "generated behavior reward")
        cumulative = _finite_float(cumulative + reward, "generated behavior return")
        steps.append(
            {
                "step": step,
                "action": action,
                "reward": reward,
                "cumulative_return": cumulative,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "planner": planner,
            }
        )

    if (
        len(steps) != episode_length
        or bool(terminated)
        or not bool(truncated)
        or any(bool(row["terminated"]) for row in steps)
        or any(bool(row["truncated"]) for row in steps[:-1])
    ):
        raise TDMPC2MPPIEvaluationError(
            "Generated behavior must be a full non-terminated fixed-horizon episode."
        )
    return (
        {
            "episode": episode,
            "environment_seed": environment_seed,
            "controller_seed": behavior_controller_seed,
            "return": cumulative,
            "steps": steps,
        },
        draws_by_step,
    )


def _reconstruct_anchor(
    env: Any,
    *,
    environment_seed: int,
    behavior_steps: Sequence[Mapping[str, Any]],
    anchor_step: int,
) -> tuple[Any, dict[str, Any]]:
    """Reset and replay a recorded prefix, checking every observable exactly."""

    _seed_spaces(env, environment_seed)
    observation, _ = env.reset(seed=environment_seed)
    prefix_return = 0.0
    for step in range(anchor_step):
        expected = behavior_steps[step]
        observation, reward, terminated, truncated, _ = env.step(expected["action"])
        reward = _finite_float(reward, f"replayed prefix reward at step {step}")
        if reward != float(expected["reward"]):
            raise TDMPC2MPPIEvaluationError(
                f"Exact prefix replay reward mismatch at environment seed "
                f"{environment_seed}, step {step}."
            )
        if bool(terminated) != bool(expected["terminated"]) or bool(truncated) != bool(
            expected["truncated"]
        ):
            raise TDMPC2MPPIEvaluationError(
                f"Exact prefix replay termination mismatch at environment seed "
                f"{environment_seed}, step {step}."
            )
        prefix_return = _finite_float(prefix_return + reward, "replayed prefix return")
    return observation, {
        "actions_replayed": anchor_step,
        "rewards_checked_exactly": anchor_step,
        "termination_flags_checked_exactly": anchor_step,
        "prefix_return": prefix_return,
    }


def _run_prior_continuation_branch(
    model: Any,
    env: Any,
    observation: Any,
    *,
    first_action: np.ndarray | None,
    anchor_step: int,
    episode_length: int,
    action_dim: int,
    discount: float,
    controller_seed: int,
) -> dict[str, Any]:
    """Run one anchor action and the prior mean through fixed-horizon truncation."""

    _set_controller(model, "policy_prior_mean")
    _seed_controller(controller_seed)
    model.reset()
    undiscounted = 0.0
    discounted = 0.0
    discount_power = 1.0
    suffix_steps = 0
    terminated = False
    truncated = False
    initial_action: np.ndarray | None = None
    first_reward: float | None = None
    first_terminated: bool | None = None
    first_truncated: bool | None = None
    current_observation = observation
    while not (terminated or truncated):
        if suffix_steps == 0 and first_action is not None:
            action = first_action
        else:
            action = _predict_action(
                model,
                current_observation,
                episode_start=(
                    suffix_steps == 0
                    or (suffix_steps == 1 and first_action is not None)
                ),
                label="deterministic policy-prior continuation action",
                action_dim=action_dim,
            )
        if suffix_steps == 0:
            initial_action = action.copy()
        current_observation, reward, terminated, truncated, _ = env.step(action)
        reward = _finite_float(reward, "counterfactual branch reward")
        if suffix_steps == 0:
            first_reward = reward
            first_terminated = bool(terminated)
            first_truncated = bool(truncated)
        undiscounted = _finite_float(
            undiscounted + reward, "counterfactual undiscounted return"
        )
        discounted = _finite_float(
            discounted + discount_power * reward,
            "counterfactual discounted return",
        )
        discount_power *= discount
        suffix_steps += 1
        if anchor_step + suffix_steps > episode_length:
            raise TDMPC2MPPIEvaluationError(
                "Counterfactual branch exceeded checkpoint episode_length."
            )

    if (
        anchor_step + suffix_steps != episode_length
        or bool(terminated)
        or not bool(truncated)
    ):
        raise TDMPC2MPPIEvaluationError(
            "Counterfactual branch did not reach the expected non-terminated "
            "fixed-horizon truncation."
        )
    assert initial_action is not None
    assert first_reward is not None
    assert first_terminated is not None
    assert first_truncated is not None
    return {
        "initial_action": [float(value) for value in initial_action],
        "initial_reward": first_reward,
        "initial_terminated": first_terminated,
        "initial_truncated": first_truncated,
        "undiscounted_return_from_anchor": undiscounted,
        "discounted_return_from_anchor": discounted,
        "suffix_length": suffix_steps,
        "terminated": False,
        "truncated": True,
    }


def _trajectory_digest(steps: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in steps:
        digest.update(np.asarray(row["action"], dtype=np.float64).tobytes())
        digest.update(np.asarray([row["reward"]], dtype=np.float64).tobytes())
        digest.update(bytes((bool(row["terminated"]), bool(row["truncated"]))))
    return digest.hexdigest()


def _metric_summary(
    values: Sequence[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "sample_std": _sample_std(values),
        "min": min(values),
        "max": max(values),
        "positive_fraction": float(
            np.mean(np.asarray(values, dtype=np.float64) > 0.0)
        ),
        "conditional_episode_cluster_bootstrap_95_interval": _bootstrap_mean_interval(
            values,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }


def _summaries(
    episodes: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    controller_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metric_keys = (
        "undiscounted_mc_gain",
        "discounted_mc_gain",
        "predicted_target_q_gain",
        "target_q_minus_discounted_mc_gain",
    )
    episode_values = {
        key: [float(row["episode_anchor_mean"][key]) for row in episodes]
        for key in metric_keys
    }
    summary: dict[str, Any] = {
        "episodes": len(episodes),
        "anchors_per_episode": len(episodes[0]["anchors"]),
        "anchor_action_draws": len(episodes[0]["anchors"][0]["mppi_action_draws"]),
        "aggregation": (
            "mean draws within anchor, mean anchors within reset-seed episode, "
            "then summarize/bootstrap episode clusters"
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "behavior_episode",
    }
    for key, values in episode_values.items():
        summary[key] = _metric_summary(
            values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=_namespaced_seed(
                controller_seed, f"same-state-summary-{key}", 0
            ),
        )
    summary["mppi_action_positive_gain_episode_fraction"] = float(
        np.mean(np.asarray(episode_values["undiscounted_mc_gain"]) > 0.0)
    )

    blocks = len(episodes[0]["anchors"])
    block_summary: list[dict[str, Any]] = []
    common_bootstrap_seed = _namespaced_seed(
        controller_seed, "same-state-block-episode-bootstrap", 0
    )
    generator = np.random.default_rng(common_bootstrap_seed)
    bootstrap_indices = generator.integers(
        0, len(episodes), size=(bootstrap_samples, len(episodes))
    )
    for block in range(blocks):
        rows = [episode["anchors"][block] for episode in episodes]
        output: dict[str, Any] = {
            "block": block,
            "block_start": int(rows[0]["block_start"]),
            "block_stop_exclusive": int(rows[0]["block_stop_exclusive"]),
            "anchor_step_mean": float(statistics.fmean(row["step"] for row in rows)),
            "anchor_step_min": min(int(row["step"]) for row in rows),
            "anchor_step_max": max(int(row["step"]) for row in rows),
            "episodes": len(rows),
        }
        for key in metric_keys:
            values = np.asarray(
                [float(row["draw_mean"][key]) for row in rows], dtype=np.float64
            )
            bootstrap_means = values[bootstrap_indices].mean(axis=1)
            lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
            output[f"{key}_mean"] = float(values.mean())
            output[f"{key}_median"] = float(np.median(values))
            output[f"{key}_sample_std"] = _sample_std(values.tolist())
            output[f"{key}_positive_fraction"] = float(np.mean(values > 0.0))
            output[
                f"{key}_conditional_pointwise_episode_cluster_bootstrap_95_interval"
            ] = [float(lower), float(upper)]
        block_summary.append(output)
    summary["block_pointwise_bootstrap_seed"] = common_bootstrap_seed
    return summary, block_summary


def evaluate_tdmpc2_mppi_action_mc(
    checkpoint: Path,
    *,
    output: Path,
    behavior_json: Path | None = None,
    episodes: int = 12,
    seed: int | None = None,
    controller_seed: int = 12345,
    block_size: int = 25,
    action_draws: int = 4,
    bootstrap_samples: int = 20000,
    device: str = "auto",
    metadata_path: Path | None = None,
    trial_settings: Path | None = None,
    experiment_settings: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run same-state MPPI-action versus prior-action Monte Carlo branches."""

    checkpoint = resolve_checkpoint_path(checkpoint)
    output = Path(output).expanduser().resolve()
    _preflight_output(output, overwrite=overwrite)
    behavior_path = (
        None if behavior_json is None else Path(behavior_json).expanduser().resolve()
    )
    if behavior_path == output:
        raise TDMPC2MPPIEvaluationError(
            "--behavior-json and --output must be different files."
        )
    context = resolve_render_context(
        checkpoint,
        metadata_path=metadata_path,
        trial_settings=trial_settings,
        experiment_settings=experiment_settings,
    )
    backend = _backend_for(context.trial_run_params["alg"])
    if backend != _TDMPC2_BACKEND:
        raise TDMPC2MPPIEvaluationError(
            "This evaluator requires a native TD-MPC2 checkpoint."
        )
    saved_alg_params = context.trial_run_params.get("alg_params", {})
    saved_env_params = context.experiment_params.get("env_params", {})
    algorithm_obs = saved_alg_params.get("obs")
    environment_obs = (
        saved_env_params.get("obs") if isinstance(saved_env_params, Mapping) else None
    )
    if (
        algorithm_obs is not None
        and environment_obs is not None
        and str(algorithm_obs).lower() != str(environment_obs).lower()
    ):
        raise TDMPC2MPPIEvaluationError(
            "Saved algorithm and environment observation modes disagree."
        )
    observation_mode = algorithm_obs if algorithm_obs is not None else environment_obs
    if str(observation_mode or "state").lower() != "state":
        raise TDMPC2MPPIEvaluationError(
            "Prefix-replay same-state evaluation supports state observations only."
        )

    first_seed = seed if seed is not None else _saved_seed(context.trial_run_params)
    _validate_rollout_options(episodes=episodes, seed=first_seed, max_steps=None)
    for value, label in (
        (block_size, "block_size"),
        (action_draws, "action_draws"),
        (bootstrap_samples, "bootstrap_samples"),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise TDMPC2MPPIEvaluationError(f"{label} must be a positive integer.")
    if (
        isinstance(controller_seed, bool)
        or not isinstance(controller_seed, Integral)
        or not 0 <= int(controller_seed) <= _MAX_SEED
    ):
        raise TDMPC2MPPIEvaluationError(
            "controller_seed must be an integer between 0 and 2^32-1."
        )

    checkpoint_sha256 = _file_sha256(checkpoint)
    behavior_payload = (
        None
        if behavior_path is None
        else _load_json_object(behavior_path, "behavior JSON")
    )
    behavior_sha256 = None if behavior_path is None else _file_sha256(behavior_path)

    run_params, experiment_params = _prepare_run_params(
        context,
        backend=backend,
        device=device,
        controller_seed=int(controller_seed),
    )
    alg_params = run_params.setdefault("alg_params", {})
    alg_params["wandb"] = False
    alg_params["eval_freq"] = None
    alg_params["eval_csv_path"] = None
    alg_params["buffer_size"] = 1

    model = None
    model_env = None
    behavior_env = None
    branch_env = None
    primary_error: BaseException | None = None
    original_operational: dict[str, Any] | None = None
    entry_rng_state = _capture_global_rng()
    try:
        model_env = build_env(run_params, experiment_params, render_mode=None)
        if behavior_payload is None:
            behavior_env = build_env(run_params, experiment_params, render_mode=None)
        branch_env = build_env(run_params, experiment_params, render_mode=None)
        model = _initialize_model(
            checkpoint, run_params, experiment_params, model_env, backend
        )
        _controller_mpc_flag(model)
        original_operational = _capture_operational_state(model)
        model.agent.model.eval()
        model_digest_before = _module_digest(model.agent.model)
        updates_before = int(getattr(model.agent, "num_updates", 0))

        episode_length = int(model.agent.cfg.episode_length)
        action_dim = int(model.agent.cfg.action_dim)
        discount = _finite_float(model.agent.discount, "checkpoint discount")
        if episode_length <= 0 or action_dim <= 0 or not 0.0 <= discount <= 1.0:
            raise TDMPC2MPPIEvaluationError(
                "Checkpoint episode length, action dimension, or discount is invalid."
            )
        if int(block_size) > episode_length:
            raise TDMPC2MPPIEvaluationError(
                "block_size cannot exceed checkpoint episode_length."
            )

        if behavior_payload is None:
            behavior_episodes: list[dict[str, Any]] = []
        else:
            behavior_episodes = _validate_behavior_json(
                behavior_payload,
                checkpoint_sha256=checkpoint_sha256,
                algorithm=context.trial_run_params["alg"],
                environment=context.trial_run_params["env"],
                first_seed=int(first_seed),
                episodes=int(episodes),
                controller_seed=int(controller_seed),
                episode_length=episode_length,
                action_dim=action_dim,
            )

        records: list[dict[str, Any]] = []
        for episode_index in range(int(episodes)):
            environment_seed = int(first_seed) + episode_index
            anchors = _anchor_steps(
                episode_length,
                int(block_size),
                seed_base=int(controller_seed),
                environment_seed=environment_seed,
            )
            if behavior_payload is None:
                assert behavior_env is not None
                behavior, generated_draws = _generate_behavior_episode(
                    model,
                    behavior_env,
                    episode=episode_index + 1,
                    environment_seed=environment_seed,
                    controller_seed_base=int(controller_seed),
                    anchors=anchors,
                    action_draws=int(action_draws),
                    episode_length=episode_length,
                    action_dim=action_dim,
                )
                behavior_episodes.append(behavior)
            else:
                behavior = behavior_episodes[episode_index]
                generated_draws = {}

            anchor_records: list[dict[str, Any]] = []
            for anchor in anchors:
                step = int(anchor["step"])
                behavior_step = behavior["steps"][step]
                baseline_observation, baseline_prefix = _reconstruct_anchor(
                    branch_env,
                    environment_seed=environment_seed,
                    behavior_steps=behavior["steps"],
                    anchor_step=step,
                )
                continuation_seed = _namespaced_seed(
                    int(controller_seed),
                    f"prior-continuation-block-{int(anchor['block'])}",
                    environment_seed,
                )
                baseline = _run_prior_continuation_branch(
                    model,
                    branch_env,
                    baseline_observation,
                    first_action=None,
                    anchor_step=step,
                    episode_length=episode_length,
                    action_dim=action_dim,
                    discount=discount,
                    controller_seed=continuation_seed,
                )

                if behavior_payload is None:
                    candidate_specs = generated_draws[step]
                    action_source = "fresh_native_mppi_same_saved_warm_start"
                else:
                    candidate_specs = [
                        {
                            "draw": 1,
                            "planner_seed": None,
                            "behavior_episode_stream_seed": behavior[
                                "controller_seed"
                            ],
                            "action": behavior_step["action"],
                            "planner": behavior_step["planner"],
                            "source_predicted_action_gain": behavior_step.get(
                                "source_predicted_action_gain"
                            ),
                        }
                    ]
                    action_source = "recorded_realized_native_mppi_action"

                draw_records: list[dict[str, Any]] = []
                for candidate in candidate_specs:
                    candidate_observation, candidate_prefix = _reconstruct_anchor(
                        branch_env,
                        environment_seed=environment_seed,
                        behavior_steps=behavior["steps"],
                        anchor_step=step,
                    )
                    _assert_observation_exact(
                        candidate_observation,
                        baseline_observation,
                        f"Seed {environment_seed}, anchor {step}",
                    )
                    action = _finite_action(
                        candidate["action"],
                        "MPPI candidate action",
                        action_dim=action_dim,
                    )
                    diagnostic = candidate.get("predicted_action_gain")
                    if diagnostic is None:
                        diagnostic = _predicted_action_gain(
                            model, candidate_observation, action
                        )
                    source_diagnostic = candidate.get("source_predicted_action_gain")
                    source_comparison = None
                    if source_diagnostic is not None:
                        saved_prior_action = np.asarray(
                            source_diagnostic[
                                "policy_prior_action_at_mppi_state"
                            ],
                            dtype=np.float64,
                        )
                        baseline_prior_action = np.asarray(
                            baseline["initial_action"], dtype=np.float64
                        )
                        if not np.array_equal(saved_prior_action, baseline_prior_action):
                            raise TDMPC2MPPIEvaluationError(
                                "Recomputed policy-prior baseline action does not "
                                "exactly match the source behavior diagnostic at "
                                f"seed {environment_seed}, step {step}."
                            )
                        source_comparison = _compare_predicted_action_gains(
                            diagnostic,
                            source_diagnostic,
                            action_dim=action_dim,
                        )
                    branch = _run_prior_continuation_branch(
                        model,
                        branch_env,
                        candidate_observation,
                        first_action=action,
                        anchor_step=step,
                        episode_length=episode_length,
                        action_dim=action_dim,
                        discount=discount,
                        controller_seed=continuation_seed,
                    )
                    if behavior_payload is not None and (
                        branch["initial_reward"] != float(behavior_step["reward"])
                        or branch["initial_terminated"]
                        != bool(behavior_step["terminated"])
                        or branch["initial_truncated"]
                        != bool(behavior_step["truncated"])
                    ):
                        raise TDMPC2MPPIEvaluationError(
                            "Recorded MPPI action did not exactly reproduce its "
                            "stored anchor reward/termination flags at environment "
                            f"seed {environment_seed}, step {step}."
                        )
                    undiscounted_gain = _finite_float(
                        branch["undiscounted_return_from_anchor"]
                        - baseline["undiscounted_return_from_anchor"],
                        "undiscounted MC gain",
                    )
                    discounted_gain = _finite_float(
                        branch["discounted_return_from_anchor"]
                        - baseline["discounted_return_from_anchor"],
                        "discounted MC gain",
                    )
                    predicted_gain = _finite_float(
                        diagnostic["target_q_mppi_minus_policy_prior"],
                        "predicted target-Q gain",
                    )
                    draw_records.append(
                        {
                            "draw": int(candidate["draw"]),
                            "planner_seed": (
                                None
                                if candidate["planner_seed"] is None
                                else int(candidate["planner_seed"])
                            ),
                            "behavior_episode_stream_seed": candidate.get(
                                "behavior_episode_stream_seed"
                            ),
                            "action_source": action_source,
                            "action": [float(value) for value in action],
                            "planner": copy.deepcopy(candidate.get("planner", {})),
                            "predicted_action_gain": copy.deepcopy(diagnostic),
                            "source_predicted_action_gain": copy.deepcopy(
                                source_diagnostic
                            ),
                            "source_predicted_action_gain_comparison": source_comparison,
                            "source_policy_prior_action_matches_recomputed_baseline_exactly": (
                                source_diagnostic is not None
                            ),
                            "prefix_replay": candidate_prefix,
                            "branch": branch,
                            "undiscounted_mc_gain": undiscounted_gain,
                            "discounted_mc_gain": discounted_gain,
                            "predicted_target_q_gain": predicted_gain,
                            "target_q_minus_discounted_mc_gain": _finite_float(
                                predicted_gain - discounted_gain,
                                "target-Q calibration residual",
                            ),
                        }
                    )

                metric_keys = (
                    "undiscounted_mc_gain",
                    "discounted_mc_gain",
                    "predicted_target_q_gain",
                    "target_q_minus_discounted_mc_gain",
                )
                draw_mean = {
                    key: float(statistics.fmean(row[key] for row in draw_records))
                    for key in metric_keys
                }
                anchor_records.append(
                    {
                        **{key: int(value) for key, value in anchor.items()},
                        "anchor_observation_sha256": _observation_sha256(
                            baseline_observation
                        ),
                        "behavior_action": [
                            float(value) for value in behavior_step["action"]
                        ],
                        "baseline_prefix_replay": baseline_prefix,
                        "policy_prior_baseline": baseline,
                        "mppi_action_draws": draw_records,
                        "draw_mean": draw_mean,
                    }
                )

            episode_anchor_mean = {
                key: float(
                    statistics.fmean(row["draw_mean"][key] for row in anchor_records)
                )
                for key in (
                    "undiscounted_mc_gain",
                    "discounted_mc_gain",
                    "predicted_target_q_gain",
                    "target_q_minus_discounted_mc_gain",
                )
            }
            records.append(
                {
                    "episode": episode_index + 1,
                    "environment_seed": environment_seed,
                    "behavior_episode_stream_seed": int(behavior["controller_seed"]),
                    "behavior_return": float(behavior["return"]),
                    "behavior_trajectory_sha256": _trajectory_digest(
                        behavior["steps"]
                    ),
                    "anchors": anchor_records,
                    "episode_anchor_mean": episode_anchor_mean,
                }
            )
            print(
                f"Episode {episode_index + 1}/{episodes} | seed={environment_seed} | "
                f"anchors={len(anchor_records)} | "
                f"undiscounted_gain={episode_anchor_mean['undiscounted_mc_gain']:.6g} | "
                f"discounted_gain={episode_anchor_mean['discounted_mc_gain']:.6g}",
                flush=True,
            )

        model_digest_after = _module_digest(model.agent.model)
        updates_after = int(getattr(model.agent, "num_updates", 0))
        if model_digest_after != model_digest_before or updates_after != updates_before:
            raise TDMPC2MPPIEvaluationError(
                "Frozen evaluation changed TD-MPC2 model or update state."
            )

        summary, block_summary = _summaries(
            records,
            bootstrap_samples=int(bootstrap_samples),
            controller_seed=int(controller_seed),
        )
        _restore_operational_state(model, original_operational)
        operational_restored = _operational_state_matches(model, original_operational)
        _restore_global_rng(entry_rng_state)
        restored_rng_state = _capture_global_rng()
        rng_restored = _rng_state_matches(restored_rng_state, entry_rng_state)
        if not operational_restored or not rng_restored:
            raise TDMPC2MPPIEvaluationError(
                "Evaluator failed to restore controller or global RNG state."
            )

        effective_draws = 1 if behavior_payload is not None else int(action_draws)
        configured_iterations = int(saved_alg_params.get("iterations", 6))
        effective_iterations = int(model.agent.cfg.iterations)
        planning_horizon = int(model.agent.cfg.outer_planning_horizon)
        num_samples = int(model.agent.cfg.num_samples)
        num_pi_trajs = int(model.agent.cfg.num_pi_trajs)
        planner_model_steps_per_action = (
            num_pi_trajs * max(0, planning_horizon - 1)
            + effective_iterations * num_samples * planning_horizon
        )
        total_anchors = sum(len(row["anchors"]) for row in records)
        counterfactual_branches = total_anchors * (1 + effective_draws)
        generated_planner_calls = (
            0
            if behavior_payload is not None
            else int(episodes) * episode_length
            + total_anchors * max(0, int(action_draws) - 1)
        )
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "configuration_source": str(context.source),
            "algorithm": context.trial_run_params["alg"],
            "environment": context.trial_run_params["env"],
            "behavior_source": {
                "mode": (
                    "validated_paired_evaluator_json"
                    if behavior_path is not None
                    else "generated_native_mppi"
                ),
                "path": None if behavior_path is None else str(behavior_path),
                "sha256": behavior_sha256,
                "schema_version": (
                    None
                    if behavior_payload is None
                    else behavior_payload["schema_version"]
                ),
                "declared_checkpoint": (
                    None
                    if behavior_payload is None
                    else behavior_payload.get("checkpoint")
                ),
                "declared_configuration_source": (
                    None
                    if behavior_payload is None
                    else behavior_payload.get("configuration_source")
                ),
                "declared_algorithm": (
                    context.trial_run_params["alg"]
                    if behavior_payload is None
                    else behavior_payload.get("algorithm")
                ),
                "declared_environment": (
                    context.trial_run_params["env"]
                    if behavior_payload is None
                    else behavior_payload.get("environment")
                ),
                "checkpoint_sha256": checkpoint_sha256,
                "environment_seed_first": int(first_seed),
                "environment_seed_last": int(first_seed) + int(episodes) - 1,
                "selected_episodes": int(episodes),
                "validation": {
                    "schema_and_protocol": True,
                    "checkpoint_sha256": True,
                    "environment_seed_bank": True,
                    "full_uncapped_nonterminated_episodes": True,
                    "finite_actions_and_rewards": True,
                    "exact_prefix_rewards_and_flags_replayed": True,
                    "exact_reconstructed_anchor_observations_across_branches": True,
                },
            },
            "protocol": {
                "estimand": (
                    "real-environment return of one native-MPPI action minus the "
                    "policy-prior-mean action at the same reconstructed behavior "
                    "state, with the deterministic frozen policy prior thereafter"
                ),
                "behavior_occupancy": "native_mppi",
                "continuation_policy": "deterministic_frozen_policy_prior_mean",
                "environment_seed_first": int(first_seed),
                "environment_seed_last": int(first_seed) + int(episodes) - 1,
                "controller_seed_base": int(controller_seed),
                "episode_length": episode_length,
                "block_size": int(block_size),
                "blocks_per_episode": len(records[0]["anchors"]),
                "anchor_selection": (
                    "one stable hash-seeded uniform offset within each block, "
                    "independently for each reset-seed episode"
                ),
                "requested_action_draws_per_anchor": int(action_draws),
                "effective_action_draws": effective_draws,
                "action_draw_breakdown": {
                    "recorded_behavior_actions": (
                        1 if behavior_payload is not None else 0
                    ),
                    "untouched_generated_behavior_stream_actions": (
                        0 if behavior_payload is not None else 1
                    ),
                    "fresh_namespaced_observational_actions": (
                        0
                        if behavior_payload is not None
                        else max(0, int(action_draws) - 1)
                    ),
                },
                "action_draw_semantics": (
                    "the one recorded realized native-MPPI action; extra draws are "
                    "not reconstructible because paired JSON lacks pre-plan _prev_mean"
                    if behavior_payload is not None
                    else "one realized native behavior-stream draw plus fixed "
                    "namespaced planner draws, all from the exact same saved "
                    "native-MPPI warm start at each anchor; observational draws "
                    "are removed before behavior continues"
                ),
                "discount": discount,
                "returns": [
                    "undiscounted full remaining episode",
                    "checkpoint-discounted full remaining episode",
                ],
                "prefix_reconstruction": (
                    "reset with the recorded environment seed and exactly replay all "
                    "recorded native-MPPI actions preceding the anchor"
                ),
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_unit": "behavior_episode",
                "block_interval_warning": (
                    "Block intervals use one common whole-episode cluster "
                    "resample matrix but are pointwise, not a simultaneous band."
                ),
                "conditionality_warning": (
                    "Intervals are conditional on one frozen training-seed checkpoint "
                    "and this finite native-MPPI behavior seed bank; they do not "
                    "measure uncertainty across independently trained checkpoints."
                ),
                "target_q_warning": (
                    "Target-Q action gain is a learned-model diagnostic and is "
                    "compared only with checkpoint-discounted Monte Carlo gain."
                ),
            },
            "planner": {
                "configured_iterations": configured_iterations,
                "effective_iterations": effective_iterations,
                "num_samples": num_samples,
                "num_elites": int(model.agent.cfg.num_elites),
                "num_pi_trajs": num_pi_trajs,
                "planning_horizon": planning_horizon,
                "model_transitions_per_action": planner_model_steps_per_action,
            },
            "compute_accounting": {
                "total_anchors": total_anchors,
                "counterfactual_branches": counterfactual_branches,
                "prefix_reconstructions": counterfactual_branches,
                "environment_decisions_in_prefix_plus_suffix_branches": (
                    counterfactual_branches * episode_length
                ),
                "environment_decisions_generating_behavior": (
                    0
                    if behavior_payload is not None
                    else int(episodes) * episode_length
                ),
                "native_mppi_planner_calls_in_this_evaluator": generated_planner_calls,
                "native_mppi_model_transitions_in_this_evaluator": (
                    generated_planner_calls * planner_model_steps_per_action
                ),
                "recorded_behavior_planner_calls_reused_without_rerun": (
                    int(episodes) * episode_length
                    if behavior_payload is not None
                    else 0
                ),
                "target_q_diagnostic_calls": total_anchors * effective_draws,
            },
            "frozen_state": {
                "model_digest_before": model_digest_before,
                "model_digest_after": model_digest_after,
                "num_updates_before": updates_before,
                "num_updates_after": updates_after,
                "model_and_updates_unchanged": True,
                "controller_operational_state_restored": operational_restored,
                "global_rng_streams_present_at_entry_restored": rng_restored,
                "cuda_rng_streams_present_at_entry": (
                    0 if entry_rng_state[3] is None else len(entry_rng_state[3])
                ),
                "cuda_initialized_during_evaluation": bool(
                    entry_rng_state[3] is None and restored_rng_state[3] is not None
                ),
                "cuda_rng_state_restored_exactly": (
                    None if entry_rng_state[3] is None else True
                ),
                "rng_restoration_scope": (
                    "Python, NumPy, Torch CPU, and every CUDA RNG stream that "
                    "already existed on evaluator entry; CUDA initialization "
                    "itself is irreversible"
                ),
                "unchanged_and_restored": True,
            },
            "summary": summary,
            "block_summary": block_summary,
            "episodes": records,
        }
        if context.metadata is not None:
            payload["checkpoint_metadata"] = copy.deepcopy(
                context.metadata.get("checkpoint", {})
            )
        resolved_runtime = context.trial_run_params.get("resolved_runtime")
        if isinstance(resolved_runtime, Mapping):
            payload["resolved_runtime"] = copy.deepcopy(dict(resolved_runtime))

        _write_json(output, payload, overwrite=overwrite)
        print(f"Wrote {output}", flush=True)
        interval = summary["undiscounted_mc_gain"][
            "conditional_episode_cluster_bootstrap_95_interval"
        ]
        print(
            "Same-state MPPI-action gain: "
            f"mean={summary['undiscounted_mc_gain']['mean']:.6g}, "
            f"conditional_episode_cluster_bootstrap95={interval}",
            flush=True,
        )
        return payload
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if model is not None and original_operational is not None:
            try:
                _restore_operational_state(model, original_operational)
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            _close_resources(model, branch_env, behavior_env, model_env)
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            _restore_global_rng(entry_rng_state)
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            if primary_error is not None:
                add_cleanup_notes(
                    primary_error,
                    cleanup_errors,
                    prefix="Additional same-state MPPI MC evaluator cleanup failure",
                )
            else:
                raise_cleanup_errors(cleanup_errors)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evaluate_tdmpc2_mppi_action_mc(
            args.checkpoint,
            output=args.output,
            behavior_json=args.behavior_json,
            episodes=args.episodes,
            seed=args.seed,
            controller_seed=args.controller_seed,
            block_size=args.block_size,
            action_draws=args.action_draws,
            bootstrap_samples=args.bootstrap_samples,
            device=args.device,
            metadata_path=args.metadata,
            trial_settings=args.trial_settings,
            experiment_settings=args.experiment_settings,
            overwrite=args.overwrite,
        )
    except (RenderCheckpointError, TDMPC2MPPIEvaluationError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
