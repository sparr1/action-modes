"""Paired Monte Carlo evaluation of TD-MPC2's policy prior and native MPPI.

The evaluator loads one frozen TD-MPC2 checkpoint, rebuilds two independent
environment stacks, and runs both controllers from the same bank of reset
seeds.  It records complete reward/cumulative-return traces so controller-level
gain can be inspected over an episode.  Aligned timestep differences are not
single-action counterfactuals: the two state trajectories generally diverge
after their first actions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import time
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import domains  # noqa: F401  # Register environments before build_env.
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
from RL.tdmpc2_core.common import math as tdmpc_math
from utils.cleanup import add_cleanup_notes, raise_cleanup_errors
from utils.core import build_env


SCHEMA_VERSION = 1
_TDMPC2_BACKEND = "tdmpc2"
_MAX_SEED = 2**32 - 1
_CONTROLLERS = ("policy_prior_mean", "native_mppi")


class TDMPC2MPPIEvaluationError(RuntimeError):
    """An actionable frozen TD-MPC2 comparison error."""


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
            "Compare a frozen TD-MPC2 policy-prior mean with native eval-mode "
            "MPPI using paired full-episode Monte Carlo rollouts."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="TD-MPC2 checkpoint to load.")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON path.")
    parser.add_argument(
        "--episodes",
        type=_positive_int,
        default=12,
        help="Number of paired full episodes (default: 12).",
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
        help="Base seed for isolated per-episode controller streams.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=_positive_int,
        default=20000,
        help="Whole-episode bootstrap draws for the paired 95%% interval.",
    )
    parser.add_argument(
        "--device", default="auto", help="Inference device override (default: auto)."
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        help="Optional common per-episode safety cap.",
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


def _namespaced_seed(base_seed: int, namespace: str, episode: int) -> int:
    payload = f"tdmpc2-mppi-eval:{int(base_seed)}:{namespace}:{int(episode)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        _MAX_SEED + 1
    )


def _capture_global_rng() -> tuple[Any, tuple[Any, ...], torch.Tensor, Any]:
    cuda_state = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_state = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return (
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.random.get_rng_state().clone(),
        cuda_state,
    )


def _restore_global_rng(
    state: tuple[Any, tuple[Any, ...], torch.Tensor, Any]
) -> None:
    python_state, numpy_state, torch_cpu_state, torch_cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_cpu_state)
    if torch_cuda_state is not None:
        torch.cuda.set_rng_state_all(list(torch_cuda_state))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        if not torch.is_tensor(value):
            raise TDMPC2MPPIEvaluationError(
                f"Model state {key!r} is not a tensor."
            )
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _numeric_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        return {}
    output: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or isinstance(value, (bool, np.bool_)):
            continue
        if not isinstance(value, (Real, np.integer, np.floating)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            output[key] = numeric
    return output


def _controller_mpc_flag(model: Any) -> bool:
    value = getattr(getattr(model, "agent", None), "cfg", None)
    value = getattr(value, "mpc", None)
    if not isinstance(value, (bool, np.bool_)):
        raise TDMPC2MPPIEvaluationError("TD-MPC2 agent.cfg.mpc must be boolean.")
    return bool(value)


def _set_controller(model: Any, controller: str) -> None:
    if controller not in _CONTROLLERS:
        raise TDMPC2MPPIEvaluationError(f"Unknown controller {controller!r}.")
    enabled = controller == "native_mppi"
    model.agent.cfg.mpc = enabled
    if getattr(model, "cfg", None) is not model.agent.cfg:
        model.cfg.mpc = enabled


@torch.no_grad()
def _predicted_action_gain(
    model: Any,
    observation: Any,
    mppi_environment_action: Any,
) -> dict[str, Any]:
    """Compare actions under the unchanged target-Q ensemble at one MPPI state."""

    agent = model.agent
    device = torch.device(agent.device)
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device() if device.index is None else int(device.index)
        ]

    started = time.perf_counter()
    # world_model.pi samples before returning its mean. Preserve that RNG draw
    # so this observational diagnostic cannot change the next MPPI action.
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        obs_t = model._obs_to_tensor(observation).to(
            device, non_blocking=True
        ).unsqueeze(0)
        latent = agent.model.encode(obs_t, None)
        _, policy_info = agent.model.pi(latent, None)
        prior_action = policy_info["mean"]
        mppi_action = torch.as_tensor(
            model._scale_action(mppi_environment_action),
            dtype=prior_action.dtype,
            device=device,
        ).reshape(1, -1)
        actions = torch.cat((mppi_action, prior_action), dim=0)
        latents = latent.expand(actions.shape[0], -1)
        logits = agent.model.Q(
            latents,
            actions,
            None,
            return_type="all",
            target=True,
        )
        q_values = tdmpc_math.two_hot_inv(logits, agent.cfg)
        q_mean_all = q_values.mean(dim=0).reshape(actions.shape[0], -1).mean(dim=1)
        mppi_q = float(q_mean_all[0].cpu())
        prior_q = float(q_mean_all[1].cpu())
        action_l2 = float(torch.linalg.vector_norm(mppi_action - prior_action).cpu())

    return {
        "target_q_mppi_mean_all": _finite_float(mppi_q, "target Q for MPPI action"),
        "target_q_policy_prior_mean_all": _finite_float(
            prior_q, "target Q for policy-prior action"
        ),
        "target_q_mppi_minus_policy_prior": _finite_float(
            mppi_q - prior_q, "predicted target-Q action gain"
        ),
        "policy_prior_to_mppi_action_l2": _finite_float(
            action_l2, "policy-prior-to-MPPI action distance"
        ),
        "policy_prior_action_at_mppi_state": [
            _finite_float(value, "policy-prior action")
            for value in prior_action[0].detach().cpu().tolist()
        ],
        "diagnostic_seconds": max(0.0, time.perf_counter() - started),
    }


def _reset_paired_environments(
    prior_env: Any,
    mppi_env: Any,
    *,
    seed: int,
) -> tuple[Any, Any]:
    _seed_spaces(prior_env, seed)
    _seed_spaces(mppi_env, seed)
    prior_observation, _ = prior_env.reset(seed=seed)
    mppi_observation, _ = mppi_env.reset(seed=seed)
    if not np.array_equal(np.asarray(prior_observation), np.asarray(mppi_observation)):
        raise TDMPC2MPPIEvaluationError(
            "Paired environment resets produced different initial observations "
            f"for seed {seed}."
        )
    return prior_observation, mppi_observation


def _run_arm(
    model: Any,
    env: Any,
    observation: Any,
    *,
    controller: str,
    controller_seed: int,
    max_steps: int | None,
) -> dict[str, Any]:
    _set_controller(model, controller)
    _seed_controller(controller_seed)
    model.reset()

    cumulative_return = 0.0
    steps: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    capped = False
    started = time.perf_counter()
    while not (terminated or truncated):
        current_observation = observation
        prediction = model.predict(
            current_observation,
            deterministic=True,
            episode_start=(len(steps) == 0),
        )
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size == 0 or not bool(np.isfinite(action_array).all()):
            raise TDMPC2MPPIEvaluationError(
                f"{controller} produced a non-finite or empty action."
            )

        record: dict[str, Any] = {
            "step": len(steps),
            "action": [float(value) for value in action_array],
        }
        if controller == "native_mppi":
            record["planner"] = _numeric_metrics(model.agent.last_plan_metrics)
            record["predicted_action_gain"] = _predicted_action_gain(
                model, current_observation, action
            )

        observation, reward, terminated, truncated, _ = env.step(action)
        reward = _finite_float(reward, f"{controller} environment reward")
        cumulative_return = _finite_float(
            cumulative_return + reward, f"{controller} cumulative return"
        )
        record["reward"] = reward
        record["cumulative_return"] = cumulative_return
        record["terminated"] = bool(terminated)
        record["truncated"] = bool(truncated)
        steps.append(record)

        if (
            max_steps is not None
            and len(steps) >= max_steps
            and not (terminated or truncated)
        ):
            capped = True
            break

    return {
        "controller": controller,
        "controller_seed": int(controller_seed),
        "return": cumulative_return,
        "length": len(steps),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "capped": bool(capped),
        "seconds": max(0.0, time.perf_counter() - started),
        "steps": steps,
    }


def _sample_std(values: Sequence[float]) -> float:
    return 0.0 if len(values) < 2 else float(statistics.stdev(values))


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not bool(np.isfinite(array).all()):
        raise TDMPC2MPPIEvaluationError(
            "Bootstrap values must be a non-empty finite vector."
        )
    if array.size == 1:
        value = float(array[0])
        return [value, value]
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(int(samples), array.size))
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return [float(lower), float(upper)]


def _summary(
    episodes: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    prior = [float(episode["policy_prior_mean"]["return"]) for episode in episodes]
    mppi = [float(episode["native_mppi"]["return"]) for episode in episodes]
    deltas = [float(episode["return_delta"]) for episode in episodes]
    return {
        "paired_episodes": len(episodes),
        "policy_prior_return_mean": float(statistics.fmean(prior)),
        "policy_prior_return_median": float(statistics.median(prior)),
        "policy_prior_return_sample_std": _sample_std(prior),
        "native_mppi_return_mean": float(statistics.fmean(mppi)),
        "native_mppi_return_median": float(statistics.median(mppi)),
        "native_mppi_return_sample_std": _sample_std(mppi),
        "paired_return_delta_mean": float(statistics.fmean(deltas)),
        "paired_return_delta_median": float(statistics.median(deltas)),
        "paired_return_delta_sample_std": _sample_std(deltas),
        "paired_return_delta_min": min(deltas),
        "paired_return_delta_max": max(deltas),
        "paired_mppi_win_fraction": float(
            np.mean(np.asarray(deltas, dtype=np.float64) > 0.0)
        ),
        "paired_return_delta_conditional_bootstrap_95_interval": _bootstrap_mean_interval(
            deltas,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_unit": "paired_episode",
    }


def _trajectory_summary(
    episodes: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    common_length = min(
        min(
            len(episode["policy_prior_mean"]["steps"]),
            len(episode["native_mppi"]["steps"]),
        )
        for episode in episodes
    )
    bootstrap_generator = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = bootstrap_generator.integers(
        0,
        len(episodes),
        size=(int(bootstrap_samples), len(episodes)),
    )
    output = []
    for step in range(common_length):
        rows = list(episodes)
        prior_rewards = [
            float(row["policy_prior_mean"]["steps"][step]["reward"])
            for row in rows
        ]
        mppi_rewards = [
            float(row["native_mppi"]["steps"][step]["reward"])
            for row in rows
        ]
        prior_cumulative = [
            float(row["policy_prior_mean"]["steps"][step]["cumulative_return"])
            for row in rows
        ]
        mppi_cumulative = [
            float(row["native_mppi"]["steps"][step]["cumulative_return"])
            for row in rows
        ]
        reward_deltas = [mppi - prior for prior, mppi in zip(prior_rewards, mppi_rewards)]
        cumulative_deltas = [
            mppi - prior for prior, mppi in zip(prior_cumulative, mppi_cumulative)
        ]
        bootstrap_means = np.asarray(cumulative_deltas, dtype=np.float64)[
            bootstrap_indices
        ].mean(axis=1)
        pointwise_lower, pointwise_upper = np.quantile(
            bootstrap_means, (0.025, 0.975)
        )
        predicted_gains = [
            float(
                row["native_mppi"]["steps"][step]["predicted_action_gain"][
                    "target_q_mppi_minus_policy_prior"
                ]
            )
            for row in rows
        ]
        action_distances = [
            float(
                row["native_mppi"]["steps"][step]["predicted_action_gain"][
                    "policy_prior_to_mppi_action_l2"
                ]
            )
            for row in rows
        ]
        output.append(
            {
                "step": step,
                "paired_episodes": len(rows),
                "policy_prior_reward_mean": float(statistics.fmean(prior_rewards)),
                "native_mppi_reward_mean": float(statistics.fmean(mppi_rewards)),
                "trajectory_reward_difference_mean": float(
                    statistics.fmean(reward_deltas)
                ),
                "policy_prior_cumulative_return_mean": float(
                    statistics.fmean(prior_cumulative)
                ),
                "native_mppi_cumulative_return_mean": float(
                    statistics.fmean(mppi_cumulative)
                ),
                "trajectory_cumulative_return_difference_mean": float(
                    statistics.fmean(cumulative_deltas)
                ),
                "trajectory_cumulative_return_difference_sample_std": _sample_std(
                    cumulative_deltas
                ),
                "trajectory_cumulative_return_difference_conditional_pointwise_bootstrap_95_interval": [
                    float(pointwise_lower),
                    float(pointwise_upper),
                ],
                "predicted_target_q_action_gain_mean": float(
                    statistics.fmean(predicted_gains)
                ),
                "predicted_target_q_action_gain_sample_std": _sample_std(
                    predicted_gains
                ),
                "policy_prior_to_mppi_action_l2_mean": float(
                    statistics.fmean(action_distances)
                ),
            }
        )
    return output


def _preflight_output(path: Path, *, overwrite: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TDMPC2MPPIEvaluationError(
            f"Could not create output directory {path.parent}: {exc}"
        ) from exc
    if path.exists() and not overwrite:
        raise TDMPC2MPPIEvaluationError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
    if path.exists() and not path.is_file():
        raise TDMPC2MPPIEvaluationError(f"Output target is not a file: {path}.")


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    _preflight_output(path, overwrite=overwrite)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        temporary = None
    except FileExistsError as exc:
        raise TDMPC2MPPIEvaluationError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        ) from exc
    except OSError as exc:
        raise TDMPC2MPPIEvaluationError(f"Could not write {path}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def evaluate_tdmpc2_mppi_checkpoint(
    checkpoint: Path,
    *,
    output: Path,
    episodes: int = 12,
    seed: int | None = None,
    controller_seed: int = 12345,
    bootstrap_samples: int = 20000,
    device: str = "auto",
    max_steps: int | None = None,
    metadata_path: Path | None = None,
    trial_settings: Path | None = None,
    experiment_settings: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run a frozen, seed-paired policy-prior-versus-MPPI comparison."""

    checkpoint = resolve_checkpoint_path(checkpoint)
    output = Path(output).expanduser().resolve()
    _preflight_output(output, overwrite=overwrite)
    context = resolve_render_context(
        checkpoint,
        metadata_path=metadata_path,
        trial_settings=trial_settings,
        experiment_settings=experiment_settings,
    )
    backend = _backend_for(context.trial_run_params["alg"])
    if backend != _TDMPC2_BACKEND:
        raise TDMPC2MPPIEvaluationError(
            "This evaluator requires a native TD-MPC2 checkpoint; received "
            f"{context.trial_run_params['alg']!r}."
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
            "Saved algorithm and environment observation modes disagree: "
            f"alg_params.obs={algorithm_obs!r}, env_params.obs={environment_obs!r}."
        )
    observation_mode = algorithm_obs if algorithm_obs is not None else environment_obs
    if observation_mode is not None and not isinstance(observation_mode, str):
        raise TDMPC2MPPIEvaluationError(
            "Saved observation mode must be a string when provided."
        )
    if str(observation_mode or "state").lower() != "state":
        raise TDMPC2MPPIEvaluationError(
            "Paired TD-MPC2 MPPI evaluation currently supports state observations only."
        )

    first_seed = seed if seed is not None else _saved_seed(context.trial_run_params)
    _validate_rollout_options(
        episodes=episodes,
        seed=first_seed,
        max_steps=max_steps,
    )
    if (
        isinstance(controller_seed, bool)
        or not isinstance(controller_seed, Integral)
        or not 0 <= int(controller_seed) <= _MAX_SEED
    ):
        raise TDMPC2MPPIEvaluationError(
            "controller_seed must be an integer between 0 and 2^32-1."
        )
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, Integral)
        or int(bootstrap_samples) <= 0
    ):
        raise TDMPC2MPPIEvaluationError("bootstrap_samples must be positive.")

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
    prior_env = None
    mppi_env = None
    primary_error: BaseException | None = None
    original_mpc: bool | None = None
    entry_rng_state = _capture_global_rng()
    try:
        model_env = build_env(run_params, experiment_params, render_mode=None)
        prior_env = build_env(run_params, experiment_params, render_mode=None)
        mppi_env = build_env(run_params, experiment_params, render_mode=None)
        if prior_env is mppi_env:
            raise TDMPC2MPPIEvaluationError(
                "Paired controller evaluation requires independent environments."
            )
        model = _initialize_model(
            checkpoint,
            run_params,
            experiment_params,
            model_env,
            backend,
        )
        model.agent.model.eval()
        original_mpc = _controller_mpc_flag(model)
        model_digest_before = _module_digest(model.agent.model)
        updates_before = int(getattr(model.agent, "num_updates", 0))

        records: list[dict[str, Any]] = []
        for episode in range(int(episodes)):
            environment_seed = int(first_seed) + episode
            prior_observation, mppi_observation = _reset_paired_environments(
                prior_env,
                mppi_env,
                seed=environment_seed,
            )
            prior_controller_seed = _namespaced_seed(
                int(controller_seed), "policy_prior_mean", environment_seed
            )
            mppi_controller_seed = _namespaced_seed(
                int(controller_seed), "native_mppi", environment_seed
            )
            prior_result = _run_arm(
                model,
                prior_env,
                prior_observation,
                controller="policy_prior_mean",
                controller_seed=prior_controller_seed,
                max_steps=max_steps,
            )
            mppi_result = _run_arm(
                model,
                mppi_env,
                mppi_observation,
                controller="native_mppi",
                controller_seed=mppi_controller_seed,
                max_steps=max_steps,
            )
            record = {
                "episode": episode + 1,
                "environment_seed": environment_seed,
                "policy_prior_mean": prior_result,
                "native_mppi": mppi_result,
                "return_delta": _finite_float(
                    mppi_result["return"] - prior_result["return"],
                    "paired episode return delta",
                ),
            }
            records.append(record)
            print(
                f"Episode {episode + 1}/{episodes} | seed={environment_seed} | "
                f"prior={prior_result['return']:.6g} | "
                f"mppi={mppi_result['return']:.6g} | "
                f"delta={record['return_delta']:.6g}",
                flush=True,
            )

        model_digest_after = _module_digest(model.agent.model)
        updates_after = int(getattr(model.agent, "num_updates", 0))
        if model_digest_after != model_digest_before or updates_after != updates_before:
            raise TDMPC2MPPIEvaluationError(
                "Frozen evaluation changed TD-MPC2 model or update state."
            )

        configured_iterations = int(saved_alg_params.get("iterations", 6))
        effective_iterations = int(model.agent.cfg.iterations)
        planning_horizon = int(model.agent.cfg.outer_planning_horizon)
        num_samples = int(model.agent.cfg.num_samples)
        num_pi_trajs = int(model.agent.cfg.num_pi_trajs)
        planner_model_steps_per_action = (
            num_pi_trajs * max(0, planning_horizon - 1)
            + effective_iterations * num_samples * planning_horizon
        )
        bootstrap_seed = _namespaced_seed(
            int(controller_seed), "paired_episode_bootstrap", 0
        )
        trajectory_bootstrap_seed = _namespaced_seed(
            int(controller_seed), "trajectory_episode_bootstrap", 0
        )
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "configuration_source": str(context.source),
            "algorithm": context.trial_run_params["alg"],
            "environment": context.trial_run_params["env"],
            "protocol": {
                "estimand": (
                    "paired full-episode undiscounted return of native eval-mode "
                    "MPPI minus deterministic policy-prior mean"
                ),
                "controllers": list(_CONTROLLERS),
                "environment_seed_first": int(first_seed),
                "environment_seed_last": int(first_seed) + int(episodes) - 1,
                "controller_seed_base": int(controller_seed),
                "planner_rng": (
                    "independent fixed namespaced stream per environment seed"
                ),
                "trajectory_warning": (
                    "Aligned timestep reward and cumulative-return differences are "
                    "controller-trajectory comparisons, not causal gains of individual "
                    "actions, because states diverge after the first action."
                ),
                "predicted_action_gain_warning": (
                    "Target-Q action gain is a learned-model diagnostic, not real "
                    "Monte Carlo improvement."
                ),
                "max_steps": max_steps,
                "bootstrap_samples": int(bootstrap_samples),
                "paired_return_bootstrap_seed": bootstrap_seed,
                "trajectory_pointwise_bootstrap_seed": trajectory_bootstrap_seed,
                "trajectory_band": (
                    "conditional pointwise 95% interval from common whole-episode "
                    "cluster resamples; not a simultaneous band"
                ),
                "conditionality_warning": (
                    "Bootstrap intervals are conditional on this one frozen "
                    "training-seed checkpoint. They combine environment-reset and "
                    "one-draw-per-reset MPPI planner variability and do not measure "
                    "uncertainty across training seeds."
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
            "frozen_state": {
                "model_digest_before": model_digest_before,
                "model_digest_after": model_digest_after,
                "num_updates_before": updates_before,
                "num_updates_after": updates_after,
                "unchanged": True,
            },
            "summary": _summary(
                records,
                bootstrap_samples=int(bootstrap_samples),
                bootstrap_seed=bootstrap_seed,
            ),
            "trajectory_summary": _trajectory_summary(
                records,
                bootstrap_samples=int(bootstrap_samples),
                bootstrap_seed=trajectory_bootstrap_seed,
            ),
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
        print(
            "Paired MPPI-minus-prior return: "
            f"mean={payload['summary']['paired_return_delta_mean']:.6g}, "
            "conditional_bootstrap95="
            f"{payload['summary']['paired_return_delta_conditional_bootstrap_95_interval']}",
            flush=True,
        )
        return payload
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if model is not None and original_mpc is not None:
            try:
                _set_controller(
                    model,
                    "native_mppi" if original_mpc else "policy_prior_mean",
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            _close_resources(model, mppi_env, prior_env, model_env)
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
                    prefix="Additional TD-MPC2 MC evaluator cleanup failure",
                )
            else:
                raise_cleanup_errors(cleanup_errors)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evaluate_tdmpc2_mppi_checkpoint(
            args.checkpoint,
            output=args.output,
            episodes=args.episodes,
            seed=args.seed,
            controller_seed=args.controller_seed,
            bootstrap_samples=args.bootstrap_samples,
            device=args.device,
            max_steps=args.max_steps,
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
