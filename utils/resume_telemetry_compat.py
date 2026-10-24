"""Explicit, audited compatibility for the W&B sparse-history repair.

This module never changes a lineage identity or a checkpoint. An operator may
approve one exact committed patch using both environment variables below. The
returned attestation records the original and execution sources separately.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from utils.resume_identity import (
    LINEAGE_IDENTITY_SCHEMA_VERSION,
    canonical_json,
    source_identity,
)


BASE_COMMIT_ENV = "AMBI_RESUME_TELEMETRY_BASE_COMMIT"
PATCH_SHA256_ENV = "AMBI_RESUME_TELEMETRY_PATCH_SHA256"
COMPATIBILITY_RULE = "wandb-sparse-history-null-padding-v1"
ALLOWED_CHANGED_PATHS = frozenset({
    "utils/wandb_resume.py",
    "utils/resume_training.py",
    "utils/resume_telemetry_compat.py",
    "slurm/ambi_reward_training.py",
    "README.md",
    "tests/test_wandb_resume.py",
    "tests/test_resume_telemetry_compat.py",
    "tests/test_resume_session.py",
    "tests/test_ambi_reward_training_campaign.py",
})
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class ResumeTelemetryCompatibilityError(ValueError):
    """An explicit telemetry compatibility proof is missing or invalid."""


def _opt_in() -> tuple[str, str] | None:
    base = os.environ.get(BASE_COMMIT_ENV)
    patch = os.environ.get(PATCH_SHA256_ENV)
    if base is None and patch is None:
        return None
    if base is None or patch is None:
        raise ResumeTelemetryCompatibilityError(
            f"Telemetry compatibility requires both {BASE_COMMIT_ENV} and {PATCH_SHA256_ENV}."
        )
    if _COMMIT_RE.fullmatch(base) is None:
        raise ResumeTelemetryCompatibilityError(
            f"{BASE_COMMIT_ENV} must be a full lowercase Git commit SHA."
        )
    if _SHA256_RE.fullmatch(patch) is None:
        raise ResumeTelemetryCompatibilityError(
            f"{PATCH_SHA256_ENV} must be a lowercase SHA-256 digest."
        )
    return base, patch


def _git(root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ("git", *args), cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ResumeTelemetryCompatibilityError(
            "Could not verify the committed telemetry compatibility patch."
        ) from exc
    return result.stdout


def _pristine_source(commit: str) -> dict[str, Any]:
    return {
        "commit": commit,
        "dirty": False,
        "tracked_diff_sha256": _EMPTY_SHA256,
        "untracked_count": 0,
        "untracked_sha256": _EMPTY_SHA256,
    }


def _same(left: Any, right: Any) -> bool:
    # Canonical JSON also distinguishes bools from numbers, unlike dict ==.
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError) as exc:
        raise ResumeTelemetryCompatibilityError(
            "Compatibility identities must contain finite JSON values."
        ) from exc


def telemetry_source_compatibility(
    repo_root: str | os.PathLike[str], expected_source: Mapping[str, Any],
    actual_source: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Verify an explicitly approved telemetry patch and return its provenance.

    Return ``None`` only when neither opt-in environment variable is present.
    Otherwise require the expected source to be the pristine approved base,
    the actual source to equal this clean checkout, and HEAD to descend from
    the base. The full ``git diff --binary --full-index BASE HEAD`` digest must match the
    operator's reviewed value. Only the exact allowlist above may change, and
    ``utils/wandb_resume.py`` must change. Invalid proofs raise
    :class:`ResumeTelemetryCompatibilityError`; no identity is rewritten.
    """
    selection = _opt_in()
    if selection is None:
        return None
    base, patch_sha256 = selection
    root = Path(repo_root).resolve()
    if not isinstance(expected_source, Mapping) or not _same(
        expected_source, _pristine_source(base)
    ):
        raise ResumeTelemetryCompatibilityError(
            "The lineage source is not the pristine approved base commit."
        )
    try:
        observed = source_identity(root)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise ResumeTelemetryCompatibilityError(
            "Could not inspect the execution source identity."
        ) from exc
    if not isinstance(actual_source, Mapping) or not _same(actual_source, observed):
        raise ResumeTelemetryCompatibilityError(
            "The supplied execution source does not match this checkout."
        )
    head = observed["commit"]
    if _COMMIT_RE.fullmatch(head) is None or not _same(observed, _pristine_source(head)):
        raise ResumeTelemetryCompatibilityError(
            "Telemetry compatibility requires a clean committed execution checkout."
        )
    if _git(root, "rev-parse", f"{base}^{{commit}}").decode().strip() != base:
        raise ResumeTelemetryCompatibilityError("The approved base is not a commit.")
    _git(root, "merge-base", "--is-ancestor", base, head)
    patch = _git(root, "diff", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", base, head, "--")
    if hashlib.sha256(patch).hexdigest() != patch_sha256:
        raise ResumeTelemetryCompatibilityError(
            "The committed telemetry patch does not match the reviewed SHA-256 digest."
        )
    changed = sorted(
        path.decode("utf-8") for path in
        _git(root, "diff", "--name-only", "--no-renames", "-z", base, head, "--").split(b"\0")
        if path
    )
    if "utils/wandb_resume.py" not in changed or not set(changed) <= ALLOWED_CHANGED_PATHS:
        raise ResumeTelemetryCompatibilityError(
            "The reviewed patch must change utils/wandb_resume.py and only approved telemetry paths."
        )
    return {
        "rule": COMPATIBILITY_RULE,
        "base_source": copy.deepcopy(dict(expected_source)),
        "execution_source": copy.deepcopy(observed),
        "patch_sha256": patch_sha256,
        "changed_paths": changed,
    }


def _validated_scientific(identity: Any) -> Mapping[str, Any]:
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"schema_version", "fingerprint", "scientific"}
        or type(identity["schema_version"]) is not int
        or identity["schema_version"] != LINEAGE_IDENTITY_SCHEMA_VERSION
        or not isinstance(identity["scientific"], Mapping)
        or "source" not in identity["scientific"]
        or not isinstance(identity["fingerprint"], str)
        or _SHA256_RE.fullmatch(identity["fingerprint"]) is None
    ):
        raise ResumeTelemetryCompatibilityError("Malformed scientific lineage identity.")
    scientific = identity["scientific"]
    try:
        digest = hashlib.sha256(canonical_json(scientific).encode()).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ResumeTelemetryCompatibilityError("Malformed scientific lineage identity.") from exc
    if digest != identity["fingerprint"]:
        raise ResumeTelemetryCompatibilityError("Scientific identity fingerprint is invalid.")
    return scientific


def compatible_scientific_identity(
    repo_root: str | os.PathLike[str], expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Attest an approved source-only difference between honest identities.

    Both schema versions and fingerprints are verified. All scientific fields
    other than ``source`` must be exactly equal, including configuration and
    dependency versions. Return ``None`` without opt-in; reject invalid opted-in
    comparisons. The original identity remains unchanged.
    """
    if _opt_in() is None:
        return None
    before, after = _validated_scientific(expected), _validated_scientific(actual)
    if not _same(
        {key: value for key, value in before.items() if key != "source"},
        {key: value for key, value in after.items() if key != "source"},
    ):
        raise ResumeTelemetryCompatibilityError(
            "Telemetry compatibility cannot change scientific configuration or dependencies."
        )
    proof = telemetry_source_compatibility(repo_root, before["source"], after["source"])
    assert proof is not None  # Both opt-in variables were validated above.
    return {
        **proof,
        "original_fingerprint": expected["fingerprint"],
        "execution_fingerprint": actual["fingerprint"],
    }


def compatible_gate_binding(
    repo_root: str | os.PathLike[str], expected_binding: Mapping[str, Any],
    actual_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Attest reuse of an original gate for only the reviewed telemetry repair.

    Every binding field except ``commit`` must match exactly, including the
    manifest/config/lock hashes. Require pristine source identities and verify
    the actual commit against HEAD using :func:`telemetry_source_compatibility`.
    This does not rewrite the gate receipt or attest a new benchmark execution.
    """
    if _opt_in() is None:
        return None
    for binding in (expected_binding, actual_binding):
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("commit"), str)
            or _COMMIT_RE.fullmatch(binding["commit"]) is None
        ):
            raise ResumeTelemetryCompatibilityError("Malformed gate source binding.")
    if not _same(
        {key: value for key, value in expected_binding.items() if key != "commit"},
        {key: value for key, value in actual_binding.items() if key != "commit"},
    ):
        raise ResumeTelemetryCompatibilityError(
            "Telemetry compatibility cannot change gate configuration or lock hashes."
        )
    return telemetry_source_compatibility(
        repo_root, _pristine_source(expected_binding["commit"]),
        _pristine_source(actual_binding["commit"]),
    )
