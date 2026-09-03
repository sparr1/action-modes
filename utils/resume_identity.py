"""Canonical scientific identity for strict training lineages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


LINEAGE_IDENTITY_SCHEMA_VERSION = 1
SUPPORTED_RESUME_ALGORITHMS = {
    "TDMPC2/TDMPC2Baseline",
    "AMBITDMPC2/AMBITDMPC2",
}
_DEPENDENCIES = (
    "torch",
    "torchrl",
    "tensordict",
    "gymnasium",
    "numpy",
    "wandb",
    "dm-control",
    "mujoco",
)

# These are the only run-configuration fields currently consumed as output
# destinations by the supported exact-resume learners. Keep this allowlist
# explicit: paths to environments, pretrained models, or other scientific
# inputs must remain fingerprinted.
_OPERATIONAL_ALGORITHM_FIELDS = frozenset({"eval_csv_path"})
_AMBI_ALGORITHM = "AMBITDMPC2/AMBITDMPC2"
_AMBI_RESOLVED_IDENTITY_DEFAULTS = {
    "inner_operator": "sac",
    "inner_q_objective": "legacy_continuing",
    "inner_critic_horizon_mode": "shared",
    "inner_return_estimator": "td0",
    "inner_return_steps": None,
    "inner_return_lambda": None,
    "inner_leaf_q_source": "outer_target",
    "inner_leaf_value_samples": 1,
    "inner_search_replay_retention": "action",
    "inner_offpolicy_mode": "none",
    "inner_search_bootstrap_critic": "target",
    "inner_target_update_event": "optimizer_step",
    "inner_depth_update_order": "mixed",
    "inner_vtrace_rho_clip": 1.0,
    "inner_vtrace_c_clip": 1.0,
    "inner_vtrace_pg_rho_clip": 1.0,
    "inner_vtrace_distill_updates": 64,
    "inner_vtrace_distill_action_samples": 4,
    "inner_bootstrap_source": "inner_target",
    "inner_actor_writeback_coef": 0.0,
    "inner_critic_writeback_coef": 0.0,
    "inner_explorer_mode": "none",
    "inner_prior_rollout_weight": 0.5,
    "inner_behavior_action": "policy_sample",
    "inner_behavior_std_scale": 1.0,
    "inner_log_std_mapping": None,
    "inner_log_std_min": None,
    "inner_log_std_max": None,
    "inner_mixture_target_estimator": "stratified",
    "inner_explorer_actor_updates_per_round": None,
    "inner_explorer_critic_updates_per_round": None,
    "inner_explorer_temperature_updates_per_round": None,
    "inner_param_noise_actor_count": None,
    "inner_param_noise_target_action_rms": 0.1,
    "inner_param_noise_sigma_init": 1e-3,
    "inner_param_noise_sigma_min": 1e-6,
    "inner_param_noise_sigma_max": 0.1,
    "inner_param_noise_calibration_directions": 8,
    "inner_param_noise_calibration_batch_size": 32,
    "inner_param_noise_calibration_max_probes": 8,
    "inner_execution_policy_source": "primary",
    "inner_execution_handoff_samples": 8,
    "eval_inner_comparison": False,
    "eval_inner_comparison_episodes": 5,
    "eval_inner_comparison_seed": 12345,
}


class ResumeConfigurationError(ValueError):
    """Raised when a requested run cannot satisfy the resume contract."""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(
        f"Scientific lineage value {type(value).__name__} is not JSON serializable."
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _run_git(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def source_identity(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Return exact commit, tracked diff, and non-ignored untracked state."""
    root = Path(repo_root).resolve()
    commit = _run_git(root, "rev-parse", "HEAD").decode().strip()
    status = _run_git(root, "status", "--porcelain", "--untracked-files=all")
    diff = _run_git(root, "diff", "--binary", "HEAD", "--")
    untracked = [
        item.decode("utf-8", errors="surrogateescape")
        for item in _run_git(
            root, "ls-files", "--others", "--exclude-standard", "-z"
        ).split(b"\0")
        if item
    ]
    untracked_hash = hashlib.sha256()
    for relative in sorted(untracked):
        path = root / relative
        encoded = relative.encode("utf-8", errors="surrogateescape")
        untracked_hash.update(len(encoded).to_bytes(8, "big"))
        untracked_hash.update(encoded)
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
        elif path.is_file():
            payload = path.read_bytes()
        else:
            raise ResumeConfigurationError(
                f"Untracked source path is not a regular file or symlink: {relative}"
            )
        untracked_hash.update(len(payload).to_bytes(8, "big"))
        untracked_hash.update(payload)
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_count": len(untracked),
        "untracked_sha256": untracked_hash.hexdigest(),
    }


def dependency_identity() -> dict[str, str | None]:
    versions = {
        "python": platform.python_version(),
    }
    for distribution in _DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def scientific_trial_parameters(
    trial_run_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one resolved trial onto its audited scientific fields.

    Segment-local output destinations are deliberately absent. This is an
    exact key allowlist rather than a suffix rule, so a scientific input path
    cannot accidentally disappear from the lineage identity.
    """

    projected = dict(trial_run_params)
    raw_algorithm = projected.get("alg_params")
    if raw_algorithm is not None:
        if not isinstance(raw_algorithm, Mapping):
            raise TypeError("trial_run_params.alg_params must be a mapping.")
        algorithm = dict(raw_algorithm)
        for field in _OPERATIONAL_ALGORITHM_FIELDS:
            algorithm.pop(field, None)
        if projected.get("alg") == _AMBI_ALGORITHM:
            # These controls are resolved by AMBI even when omitted. Preserve
            # each default's scalar kind so omitted and explicit defaults
            # describe the same scientific lineage.
            for field, default in _AMBI_RESOLVED_IDENTITY_DEFAULTS.items():
                value = algorithm.get(field, default)
                if isinstance(default, bool) and isinstance(
                    value, (bool, np.bool_)
                ):
                    value = bool(value)
                elif isinstance(default, int) and isinstance(
                    value, (int, np.integer)
                ) and not isinstance(value, (bool, np.bool_)):
                    value = int(value)
                elif isinstance(default, float) and isinstance(
                    value, (int, float, np.integer, np.floating)
                ) and not isinstance(value, (bool, np.bool_)):
                    value = float(value)
                    if value == 0.0:
                        value = 0.0
                algorithm[field] = value
            for field in (
                "inner_operator",
                "inner_q_objective",
                "inner_critic_horizon_mode",
                "inner_return_estimator",
                "inner_leaf_q_source",
                "inner_search_replay_retention",
                "inner_offpolicy_mode",
                "inner_search_bootstrap_critic",
                "inner_target_update_event",
                "inner_depth_update_order",
                "inner_bootstrap_source",
                "inner_explorer_mode",
                "inner_behavior_action",
                "inner_execution_policy_source",
                "inner_mixture_target_estimator",
            ):
                value = algorithm.get(field)
                if isinstance(value, str):
                    algorithm[field] = value.lower()
            if (
                algorithm.get("inner_q_objective") == "finite_horizon"
                or algorithm.get("inner_operator") == "vtrace"
            ):
                # The resolved finite-search config deliberately removes the
                # legacy continuing-task bootstrap selector.
                algorithm["inner_bootstrap_source"] = None
            inner_mapping = algorithm.get("inner_log_std_mapping")
            if inner_mapping is None:
                inner_mapping = algorithm.get(
                    "log_std_mapping", "direct_clamp"
                )
            if isinstance(inner_mapping, str):
                inner_mapping = inner_mapping.lower()
            algorithm["inner_log_std_mapping"] = inner_mapping
            if algorithm.get("inner_log_std_min") is None:
                algorithm["inner_log_std_min"] = algorithm.get(
                    "log_std_min", -20
                )
            if algorithm.get("inner_log_std_max") is None:
                algorithm["inner_log_std_max"] = algorithm.get(
                    "log_std_max", 2
                )
            for field in ("inner_log_std_min", "inner_log_std_max"):
                value = algorithm[field]
                if isinstance(
                    value, (int, float, np.integer, np.floating)
                ) and not isinstance(value, (bool, np.bool_)):
                    algorithm[field] = float(value)
        projected["alg_params"] = algorithm
    return _json_safe(projected)


def validate_resume_selection(
    *,
    algorithm: str,
    observation_mode: str | None,
    num_runs: int,
    save_trials: str | None,
    checkpoint_minutes: float,
    drain_after_seconds: float | None,
) -> None:
    if algorithm not in SUPPORTED_RESUME_ALGORITHMS:
        raise ResumeConfigurationError(
            f"Algorithm {algorithm!r} does not implement exact training resume."
        )
    if observation_mode is not None and str(observation_mode).lower() != "state":
        raise ResumeConfigurationError("Training resume v1 supports state observations only.")
    if num_runs != 1:
        raise ResumeConfigurationError(
            "Training resume requires --num-runs 1 and one explicit algorithm/trial cell."
        )
    if save_trials not in (None, "none"):
        raise ResumeConfigurationError(
            "Exact resume requires save_trials='none'; use the generation-local "
            "latest/best checkpoint strategy for portable model artifacts."
        )
    checkpoint_minutes = float(checkpoint_minutes)
    if not np.isfinite(checkpoint_minutes) or checkpoint_minutes <= 0:
        raise ResumeConfigurationError(
            "--resume-checkpoint-minutes must be finite and positive."
        )
    if drain_after_seconds is not None:
        drain_after_seconds = float(drain_after_seconds)
        if not np.isfinite(drain_after_seconds) or drain_after_seconds <= 0:
            raise ResumeConfigurationError(
                "--drain-after-seconds must be finite and positive."
            )


def lineage_identity(
    *,
    trial_run_params: Mapping[str, Any],
    experiment_params: Mapping[str, Any],
    repo_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build and hash the immutable scientific contract for one lineage."""
    scientific = {
        "trial_run_params": scientific_trial_parameters(trial_run_params),
        "experiment_params": _json_safe(dict(experiment_params)),
        "source": source_identity(repo_root),
        "dependencies": dependency_identity(),
    }
    digest = hashlib.sha256(canonical_json(scientific).encode()).hexdigest()
    return {
        "schema_version": LINEAGE_IDENTITY_SCHEMA_VERSION,
        "fingerprint": digest,
        "scientific": scientific,
    }


__all__ = [
    "ResumeConfigurationError",
    "SUPPORTED_RESUME_ALGORITHMS",
    "canonical_json",
    "dependency_identity",
    "lineage_identity",
    "scientific_trial_parameters",
    "source_identity",
    "validate_resume_selection",
]
