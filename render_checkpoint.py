"""Evaluate a saved model checkpoint in a window, MP4 files, or JSON.

Checkpoint sidecars written by the training runner are the preferred source of
the resolved algorithm and environment configuration.  Older run directories
remain usable through their ``settings.json`` and ``alg_settings.json`` files.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import domains  # noqa: F401  # Register project environments before build_env/gym.make.
from utils.checkpoint_context import (
    CheckpointContext as RenderContext,
    CheckpointContextError as RenderCheckpointError,
    load_checkpoint_context,
    load_json_object as _load_json_object,
    validate_context_params as _validate_context_params,
)
from utils.cleanup import add_cleanup_notes, raise_cleanup_errors
from utils.core import build_env, initialize_alg


_MAX_NUMPY_SEED = 2**32 - 1
_NATIVE_SAC = "SAC/SAC"
_TDMPC2 = "TDMPC2/TDMPC2Baseline"
_AMBI_TDMPC2 = "AMBITDMPC2/AMBITDMPC2"
_SB3_REPLAY_ALGORITHMS = {"DDPG", "DQN", "SAC", "TD3"}


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    seed: int
    episode_return: float
    length: int
    capped: bool
    video_path: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display or record complete rollouts from an AMBI checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Checkpoint file to load.")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--display",
        action="store_true",
        help="Open the environment's live human-rendering window.",
    )
    output.add_argument(
        "--video-dir",
        type=Path,
        help="Stream one complete MP4 into this directory per episode.",
    )
    output.add_argument(
        "--results-json",
        type=Path,
        help="Run without rendering and atomically write rollout results as JSON.",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes (default: 1)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="First rollout seed (default: the saved trial seed).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device override (default: auto).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic inference.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Optional per-episode safety cap.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Explicit checkpoint metadata sidecar, overriding discovery.",
    )
    parser.add_argument(
        "--trial-settings",
        type=Path,
        help="Explicit resolved alg_settings.json; requires --experiment-settings.",
    )
    parser.add_argument(
        "--experiment-settings",
        type=Path,
        help="Explicit settings.json; requires --trial-settings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace calculated video or JSON outputs that already exist.",
    )
    return parser


def _is_int(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _find_experiment_root(checkpoint: Path) -> Path | None:
    for directory in (checkpoint.parent, *checkpoint.parent.parents):
        if (directory / "settings.json").is_file():
            return directory
    return None


def _legacy_trial_settings(experiment_root: Path, checkpoint: Path) -> Path:
    checkpoint_name = checkpoint.name
    candidates: list[tuple[int, Path]] = []
    try:
        children = list(experiment_root.iterdir())
    except OSError as exc:
        raise RenderCheckpointError(
            f"Could not inspect legacy run directory {experiment_root}: {exc}"
        ) from exc

    for child in children:
        settings = child / "alg_settings.json"
        if not child.is_dir() or not settings.is_file():
            continue
        prefix = f"model:{child.name}"
        if checkpoint_name == prefix or checkpoint_name.startswith(
            (prefix + "_", prefix + ".")
        ):
            candidates.append((len(prefix), settings))

    if not candidates:
        raise RenderCheckpointError(
            "Found legacy settings.json but could not match the checkpoint name to a "
            f"trial alg_settings.json under {experiment_root}. Supply --metadata or both "
            "--trial-settings and --experiment-settings."
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise RenderCheckpointError(
            f"Checkpoint {checkpoint.name} ambiguously matches multiple legacy trials. "
            "Supply explicit settings files."
        )
    return candidates[0][1]


def resolve_render_context(
    checkpoint: Path,
    *,
    metadata_path: Path | None = None,
    trial_settings: Path | None = None,
    experiment_settings: Path | None = None,
) -> RenderContext:
    """Resolve complete run/environment settings in strict precedence order."""
    checkpoint = Path(checkpoint)
    has_trial = trial_settings is not None
    has_experiment = experiment_settings is not None
    if has_trial != has_experiment:
        raise RenderCheckpointError(
            "--trial-settings and --experiment-settings must be supplied together."
        )
    if metadata_path is not None and has_trial:
        raise RenderCheckpointError(
            "Use either --metadata or the paired settings overrides, not both."
        )

    if metadata_path is not None:
        return load_checkpoint_context(checkpoint, metadata_path)

    if has_trial:
        trial_path = Path(trial_settings).expanduser().resolve()
        experiment_path = Path(experiment_settings).expanduser().resolve()
        run_params = _load_json_object(trial_path, "trial settings")
        experiment_params = _load_json_object(
            experiment_path, "experiment settings"
        )
        _validate_context_params(run_params, experiment_params, trial_path)
        return RenderContext(
            trial_run_params=copy.deepcopy(run_params),
            experiment_params=copy.deepcopy(experiment_params),
            source=trial_path,
        )

    adjacent = Path(str(checkpoint) + ".metadata.json")
    if adjacent.exists():
        # A present sidecar is authoritative.  Never hide stale/corrupt metadata
        # by silently falling back to a legacy directory.
        return load_checkpoint_context(checkpoint)

    experiment_root = _find_experiment_root(checkpoint)
    if experiment_root is None:
        raise RenderCheckpointError(
            f"No adjacent metadata found for {checkpoint}, and no legacy settings.json "
            "was found in its parent tree. Supply --metadata or both settings overrides."
        )
    experiment_path = experiment_root / "settings.json"
    trial_path = _legacy_trial_settings(experiment_root, checkpoint)
    run_params = _load_json_object(trial_path, "trial settings")
    experiment_params = _load_json_object(experiment_path, "experiment settings")
    _validate_context_params(run_params, experiment_params, trial_path)
    return RenderContext(
        trial_run_params=copy.deepcopy(run_params),
        experiment_params=copy.deepcopy(experiment_params),
        source=trial_path,
    )


def resolve_checkpoint_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.suffix:
        alternatives = [
            candidate
            for candidate in (Path(str(path) + ".zip"), Path(str(path) + ".pt"))
            if candidate.is_file()
        ]
        if len(alternatives) == 1:
            return alternatives[0].resolve()
        if len(alternatives) > 1:
            raise RenderCheckpointError(
                f"Checkpoint path {path} is ambiguous; both .zip and .pt files exist."
            )
    raise RenderCheckpointError(f"Checkpoint does not exist: {path}")


def _backend_for(algorithm: str) -> str:
    if algorithm.startswith("baselines/") and algorithm.count("/") == 1:
        if algorithm.rsplit("/", 1)[1]:
            return "sb3"
    if algorithm == _NATIVE_SAC:
        return "native_sac"
    if algorithm == _TDMPC2:
        return "tdmpc2"
    if algorithm == _AMBI_TDMPC2:
        return "ambi_tdmpc2"
    raise RenderCheckpointError(
        f"Algorithm {algorithm!r} is not supported by this renderer. Supported "
        "checkpoints are SB3 baselines ('baselines/...'), native SAC ('SAC/SAC'), "
        "TD-MPC2 ('TDMPC2/TDMPC2Baseline'), and AMBI-TD-MPC2 "
        "('AMBITDMPC2/AMBITDMPC2'). Legacy AMBI, PAMDP, modes, Random, and "
        "Stationary models require a backend-specific evaluator."
    )


def _prepare_run_params(
    context: RenderContext,
    *,
    backend: str,
    device: str,
    controller_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_params = copy.deepcopy(context.trial_run_params)
    experiment_params = copy.deepcopy(context.experiment_params)
    alg_params = run_params.setdefault("alg_params", {})
    if not isinstance(alg_params, dict):
        raise RenderCheckpointError("alg_params must be an object.")

    run_params["device"] = str(device)
    run_params["seed"] = int(controller_seed)
    alg_params["device"] = str(device)
    alg_params["seed"] = int(controller_seed)
    alg_params["wandb"] = False
    uses_replay = backend == "native_sac" or (
        backend == "sb3"
        and run_params["alg"].rsplit("/", 1)[-1] in _SB3_REPLAY_ALGORITHMS
    )
    if uses_replay:
        # Off-policy constructors allocate replay in __init__. Rendering never
        # trains or restores replay, and capacity is not checkpoint model
        # compatibility, so avoid needless GB-scale allocations.
        alg_params["buffer_size"] = 1
    if backend in {"tdmpc2", "ambi_tdmpc2"}:
        alg_params["compile"] = False
        alg_params["compile_strict"] = False
    if backend == "ambi_tdmpc2":
        alg_params["inner_diagnostic_rollouts"] = 0
    if "wandb" in experiment_params:
        experiment_params["wandb"] = False
    return run_params, experiment_params


def _saved_seed(run_params: Mapping[str, Any]) -> int | None:
    value = run_params.get("seed")
    if value is None and isinstance(run_params.get("alg_params"), Mapping):
        value = run_params["alg_params"].get("seed")
    if value is None:
        return None
    if not _is_int(value):
        raise RenderCheckpointError(f"Saved trial seed is not an integer: {value!r}.")
    return int(value)


def _validate_rollout_options(
    *, episodes: int, seed: int | None, max_steps: int | None
) -> None:
    if not _is_int(episodes) or int(episodes) <= 0:
        raise RenderCheckpointError("episodes must be a positive integer.")
    if seed is None or not _is_int(seed) or not 0 <= int(seed) <= _MAX_NUMPY_SEED:
        raise RenderCheckpointError(
            "No valid saved trial seed is available; provide --seed between 0 and 2^32-1."
        )
    if int(seed) + int(episodes) - 1 > _MAX_NUMPY_SEED:
        raise RenderCheckpointError("Incremented episode seeds exceed 2^32-1.")
    if max_steps is not None and (
        not _is_int(max_steps) or int(max_steps) <= 0
    ):
        raise RenderCheckpointError("max_steps must be positive when provided.")


def _seed_spaces(env: Any, seed: int) -> None:
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


def _seed_controller(seed: int) -> None:
    """Seed controller-side sampling before constructing or loading a model."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _initialize_model(
    checkpoint: Path,
    run_params: dict[str, Any],
    experiment_params: dict[str, Any],
    model_env: Any,
    backend: str,
) -> Any:
    algorithm = run_params["alg"]
    model = None
    try:
        model, _, _ = initialize_alg(
            algorithm,
            run_params.get("alg_params", {}),
            model_env,
            full_run_params=run_params,
            experiment_params=experiment_params,
        )
        if backend == "sb3":
            # Baseline.load() does not forward either the environment or the
            # requested device to SB3's classmethod.  Load explicitly here so
            # --device=cpu cannot accidentally restore a CUDA model and the
            # model retains its non-rendering environment.
            load_options = {
                "env": model_env,
                "device": run_params["device"],
            }
            if algorithm.rsplit("/", 1)[-1] in _SB3_REPLAY_ALGORITHMS:
                # SB3 restores saved constructor attributes before _setup_model;
                # replace capacity there too, or loading would allocate the
                # training-sized replay buffer a second time.
                load_options["custom_objects"] = {"buffer_size": 1}
            loaded = model.get_model().load(str(checkpoint), **load_options)
            model.model = loaded
        else:
            model.load(str(checkpoint))

        if backend == "sb3":
            sb3_model = model.get_model()
            policy = getattr(sb3_model, "policy", None)
            if policy is not None and hasattr(policy, "set_training_mode"):
                policy.set_training_mode(False)
        elif backend == "native_sac":
            actor = getattr(getattr(model, "agent", None), "actor", None)
            if actor is not None and hasattr(actor, "eval"):
                actor.eval()
        else:
            world_model = getattr(getattr(model, "agent", None), "model", None)
            if world_model is not None and hasattr(world_model, "eval"):
                world_model.eval()
    except BaseException as exc:
        if isinstance(exc, Exception):
            error = RenderCheckpointError(
                f"Could not initialize {algorithm!r} and load {checkpoint}: {exc}"
            )
        else:
            error = exc
        _close_resources(model, primary_error=error)
        if error is exc:
            raise
        raise error from exc
    return model


def _predict_action(
    model: Any,
    observation: Any,
    *,
    backend: str,
    deterministic: bool,
    episode_start: bool,
) -> Any:
    if backend == "sb3":
        prediction = model.get_model().predict(
            observation, deterministic=deterministic
        )
    elif backend == "native_sac":
        prediction = model.predict(observation, deterministic=deterministic)
    elif backend == "tdmpc2":
        prediction = model.predict(
            observation,
            deterministic=deterministic,
            episode_start=episode_start,
        )
    else:
        prediction = model.predict(
            observation,
            deterministic=deterministic,
            episode_start=episode_start,
            collect_diagnostics=False,
        )
    if isinstance(prediction, tuple):
        return prediction[0]
    return prediction


def _render_fps(env: Any) -> float:
    metadata = getattr(env, "metadata", None)
    fps = metadata.get("render_fps") if isinstance(metadata, Mapping) else None
    if isinstance(fps, bool) or not isinstance(fps, Real) or float(fps) <= 0:
        raise RenderCheckpointError(
            "Video rendering requires a positive environment metadata['render_fps']."
        )
    return float(fps)


def _safe_checkpoint_stem(checkpoint: Path) -> str:
    stem = checkpoint.stem or checkpoint.name
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe or "checkpoint"


def video_target_path(
    video_dir: Path,
    checkpoint: Path,
    *,
    episode: int,
    seed: int,
) -> Path:
    return Path(video_dir) / (
        f"{_safe_checkpoint_stem(checkpoint)}_episode-{episode:03d}_seed-{seed}.mp4"
    )


def _preflight_video_outputs(
    video_dir: Path,
    checkpoint: Path,
    *,
    episodes: int,
    first_seed: int,
    overwrite: bool,
) -> None:
    try:
        video_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RenderCheckpointError(
            f"Could not create video output directory {video_dir}: {exc}"
        ) from exc
    if overwrite:
        return
    for index in range(episodes):
        target = video_target_path(
            video_dir,
            checkpoint,
            episode=index + 1,
            seed=first_seed + index,
        )
        if target.exists():
            raise RenderCheckpointError(
                f"Video already exists: {target}. Use --overwrite to replace it."
            )


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RenderCheckpointError(
            "Video output requires the project's pinned opencv-python dependency."
        ) from exc
    return cv2


class _AtomicVideoWriter:
    def __init__(self, target: Path, *, fps: float, overwrite: bool, cv2_module: Any):
        self.target = Path(target)
        self.fps = float(fps)
        self.overwrite = bool(overwrite)
        self.cv2 = cv2_module
        self._writer = None
        self._temporary: Path | None = None
        self._size: tuple[int, int] | None = None
        self._finished = False

    @staticmethod
    def _rgb_frame(frame: Any) -> np.ndarray:
        if frame is None:
            raise RenderCheckpointError(
                "Environment render() returned no RGB frame in rgb_array mode."
            )
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise RenderCheckpointError(
                f"Expected an HxWx3 RGB frame, got shape {array.shape}."
            )
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def write(self, frame: Any) -> None:
        rgb = self._rgb_frame(frame)
        height, width = rgb.shape[:2]
        size = (int(width), int(height))
        if self._writer is None:
            if self.target.exists() and not self.overwrite:
                raise RenderCheckpointError(
                    f"Video already exists: {self.target}. Use --overwrite to replace it."
                )
            fd, temporary = tempfile.mkstemp(
                prefix=f".{self.target.stem}.",
                suffix=".tmp.mp4",
                dir=self.target.parent,
            )
            os.close(fd)
            self._temporary = Path(temporary)
            fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = self.cv2.VideoWriter(
                str(self._temporary), fourcc, self.fps, size
            )
            if not self._writer.isOpened():
                self.abort()
                raise RenderCheckpointError(
                    f"OpenCV could not open a video writer for {self.target}."
                )
            self._size = size
        elif size != self._size:
            raise RenderCheckpointError(
                f"Environment frame size changed from {self._size} to {size}."
            )

        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def finish(self) -> Path:
        if self._finished:
            return self.target
        if self._writer is None or self._temporary is None:
            raise RenderCheckpointError("Cannot finish a video with no rendered frames.")
        self._writer.release()
        self._writer = None
        if not self._temporary.is_file() or self._temporary.stat().st_size == 0:
            self.abort()
            raise RenderCheckpointError(
                f"OpenCV did not produce a valid temporary MP4 for {self.target}."
            )
        try:
            if self.overwrite:
                os.replace(self._temporary, self.target)
            else:
                # Linking is an atomic no-replace publication on the same
                # filesystem, closing the race between collision checks and
                # final publication.
                os.link(self._temporary, self.target)
                self._temporary.unlink()
        except FileExistsError as exc:
            self.abort()
            raise RenderCheckpointError(
                f"Video already exists: {self.target}. Use --overwrite to replace it."
            ) from exc
        except OSError as exc:
            self.abort()
            raise RenderCheckpointError(
                f"Could not atomically publish video {self.target}: {exc}"
            ) from exc
        self._temporary = None
        self._finished = True
        return self.target

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._temporary is not None:
            try:
                self._temporary.unlink()
            except OSError:
                pass
            self._temporary = None


def _print_episode(result: EpisodeResult) -> None:
    parts = [
        f"Episode {result.episode}",
        f"return={result.episode_return:.6g}",
        f"length={result.length}",
        f"seed={result.seed}",
    ]
    if result.capped:
        parts.append("capped=yes")
    if result.video_path is not None:
        parts.append(f"video={result.video_path}")
    print(" | ".join(parts))


def _rollout(
    model: Any,
    env: Any,
    *,
    checkpoint: Path,
    backend: str,
    episodes: int,
    first_seed: int,
    deterministic: bool,
    max_steps: int | None,
    video_dir: Path | None,
    overwrite: bool,
) -> list[EpisodeResult]:
    video_targets: list[Path | None]
    cv2_module = None
    fps = None
    if video_dir is None:
        video_targets = [None] * episodes
    else:
        _preflight_video_outputs(
            video_dir,
            checkpoint,
            episodes=episodes,
            first_seed=first_seed,
            overwrite=overwrite,
        )
        video_targets = [
            video_target_path(
                video_dir,
                checkpoint,
                episode=index + 1,
                seed=first_seed + index,
            )
            for index in range(episodes)
        ]
        fps = _render_fps(env)
        cv2_module = _import_cv2()

    results = []
    for index, target in enumerate(video_targets):
        episode_seed = first_seed + index
        _seed_controller(episode_seed)
        _seed_spaces(env, episode_seed)
        observation, _ = env.reset(seed=episode_seed)
        writer = None
        if target is not None:
            writer = _AtomicVideoWriter(
                target,
                fps=fps,
                overwrite=overwrite,
                cv2_module=cv2_module,
            )
        episode_return = 0.0
        episode_length = 0
        terminated = False
        truncated = False
        capped = False
        try:
            if writer is not None:
                writer.write(env.render())
            while not (terminated or truncated):
                action = _predict_action(
                    model,
                    observation,
                    backend=backend,
                    deterministic=deterministic,
                    episode_start=(episode_length == 0),
                )
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                if writer is not None:
                    writer.write(env.render())
                if (
                    max_steps is not None
                    and episode_length >= max_steps
                    and not (terminated or truncated)
                ):
                    capped = True
                    break
            video_path = writer.finish() if writer is not None else None
        except BaseException:
            if writer is not None:
                writer.abort()
            raise

        result = EpisodeResult(
            episode=index + 1,
            seed=episode_seed,
            episode_return=episode_return,
            length=episode_length,
            capped=capped,
            video_path=video_path,
        )
        results.append(result)
        _print_episode(result)
    return results


def _close_resources(
    *resources: Any, primary_error: BaseException | None = None
) -> None:
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


def _preflight_results_output(path: Path, *, overwrite: bool) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RenderCheckpointError(
            f"Could not create results output directory {path.parent}: {exc}"
        ) from exc
    if path.exists() and not overwrite:
        raise RenderCheckpointError(
            f"Results JSON already exists: {path}. Pass --overwrite to replace it."
        )
    if path.exists() and not path.is_file():
        raise RenderCheckpointError(f"Results JSON target is not a file: {path}.")


def _results_payload(
    *,
    checkpoint: Path,
    context: RenderContext,
    backend: str,
    deterministic: bool,
    max_steps: int | None,
    results: Sequence[EpisodeResult],
) -> dict[str, Any]:
    returns = [float(result.episode_return) for result in results]
    lengths = [int(result.length) for result in results]
    mean_return = sum(returns) / len(returns)
    mean_length = sum(lengths) / len(lengths)
    return_std = math.sqrt(
        sum((value - mean_return) ** 2 for value in returns) / len(returns)
    )
    payload = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "configuration_source": str(context.source),
        "algorithm": context.trial_run_params["alg"],
        "environment": context.trial_run_params["env"],
        "backend": backend,
        "deterministic": bool(deterministic),
        "max_steps": max_steps,
        "seeds": [int(result.seed) for result in results],
        "summary": {
            "episodes": len(results),
            "return_mean": mean_return,
            "return_std": return_std,
            "return_min": min(returns),
            "return_max": max(returns),
            "length_mean": mean_length,
            "length_min": min(lengths),
            "length_max": max(lengths),
            "capped_episodes": sum(bool(result.capped) for result in results),
        },
        "episodes": [
            {
                "episode": int(result.episode),
                "seed": int(result.seed),
                "return": float(result.episode_return),
                "length": int(result.length),
                "capped": bool(result.capped),
            }
            for result in results
        ],
    }
    resolved_runtime = context.trial_run_params.get("resolved_runtime")
    if isinstance(resolved_runtime, Mapping):
        payload["resolved_runtime"] = copy.deepcopy(dict(resolved_runtime))
    if context.metadata is not None:
        payload["checkpoint_metadata"] = copy.deepcopy(
            context.metadata.get("checkpoint", {})
        )
    return payload


def _write_results_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    _preflight_results_output(path, overwrite=overwrite)
    serialized = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
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
                raise RenderCheckpointError(
                    f"Results JSON already exists: {path}. "
                    "Pass --overwrite to replace it."
                ) from exc
            temporary_path.unlink()
        temporary_path = None
    except RenderCheckpointError:
        raise
    except OSError as exc:
        raise RenderCheckpointError(
            f"Could not atomically publish results JSON {path}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def render_checkpoint(
    checkpoint: Path,
    *,
    display: bool = False,
    video_dir: Path | None = None,
    results_json: Path | None = None,
    episodes: int = 1,
    seed: int | None = None,
    device: str = "auto",
    stochastic: bool = False,
    max_steps: int | None = None,
    metadata_path: Path | None = None,
    trial_settings: Path | None = None,
    experiment_settings: Path | None = None,
    overwrite: bool = False,
) -> list[EpisodeResult]:
    """Load one checkpoint and run complete deterministic or sampled rollouts."""
    output_modes = int(bool(display)) + int(video_dir is not None) + int(
        results_json is not None
    )
    if output_modes != 1:
        raise RenderCheckpointError(
            "Choose exactly one output mode: --display, --video-dir, or --results-json."
        )
    checkpoint = resolve_checkpoint_path(checkpoint)
    context = resolve_render_context(
        checkpoint,
        metadata_path=metadata_path,
        trial_settings=trial_settings,
        experiment_settings=experiment_settings,
    )
    backend = _backend_for(context.trial_run_params["alg"])
    first_seed = seed if seed is not None else _saved_seed(context.trial_run_params)
    _validate_rollout_options(
        episodes=episodes, seed=first_seed, max_steps=max_steps
    )
    run_params, experiment_params = _prepare_run_params(
        context,
        backend=backend,
        device=device,
        controller_seed=first_seed,
    )
    resolved_video_dir = (
        None if video_dir is None else Path(video_dir).expanduser().resolve()
    )
    resolved_results_json = (
        None if results_json is None else Path(results_json).expanduser().resolve()
    )
    if resolved_video_dir is not None:
        _preflight_video_outputs(
            resolved_video_dir,
            checkpoint,
            episodes=int(episodes),
            first_seed=int(first_seed),
            overwrite=overwrite,
        )
    if resolved_results_json is not None:
        _preflight_results_output(resolved_results_json, overwrite=overwrite)

    # Seed model/controller construction as well as environment resets.  Each
    # backend also consumes the overridden seed in its own constructor.
    _seed_controller(first_seed)

    model = None
    model_env = None
    rollout_env = None
    primary_error = None
    try:
        model_env = build_env(
            run_params, experiment_params, render_mode=None
        )
        rollout_env = build_env(
            run_params,
            experiment_params,
            render_mode=(
                "human" if display else "rgb_array" if resolved_video_dir else None
            ),
        )
        model = _initialize_model(
            checkpoint,
            run_params,
            experiment_params,
            model_env,
            backend,
        )
        results = _rollout(
            model,
            rollout_env,
            checkpoint=checkpoint,
            backend=backend,
            episodes=int(episodes),
            first_seed=int(first_seed),
            deterministic=not stochastic,
            max_steps=max_steps,
            video_dir=resolved_video_dir,
            overwrite=overwrite,
        )
        if resolved_results_json is not None:
            _write_results_json(
                resolved_results_json,
                _results_payload(
                    checkpoint=checkpoint,
                    context=context,
                    backend=backend,
                    deterministic=not stochastic,
                    max_steps=max_steps,
                    results=results,
                ),
                overwrite=overwrite,
            )
            print(f"Wrote {resolved_results_json}")
        return results
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _close_resources(
            model,
            rollout_env,
            model_env,
            primary_error=primary_error,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        render_checkpoint(
            args.checkpoint,
            display=args.display,
            video_dir=args.video_dir,
            results_json=args.results_json,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            stochastic=args.stochastic,
            max_steps=args.max_steps,
            metadata_path=args.metadata,
            trial_settings=args.trial_settings,
            experiment_settings=args.experiment_settings,
            overwrite=args.overwrite,
        )
    except RenderCheckpointError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
