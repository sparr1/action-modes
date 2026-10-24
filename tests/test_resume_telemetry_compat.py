"""The sparse-history repair cannot relax unrelated resume identities."""

import copy
import hashlib
import subprocess

import pytest

from utils.resume_identity import canonical_json, source_identity
from utils.resume_telemetry_compat import (
    ALLOWED_CHANGED_PATHS,
    BASE_COMMIT_ENV,
    COMPATIBILITY_RULE,
    PATCH_SHA256_ENV,
    ResumeTelemetryCompatibilityError,
    compatible_gate_binding,
    compatible_scientific_identity,
    telemetry_source_compatibility,
)


def _git(root, *args):
    return subprocess.check_output(
        ["git", "-c", "user.name=Resume Test", "-c", "user.email=resume@example.test", *args],
        cwd=root, stderr=subprocess.PIPE,
    )


def _commit(root, message):
    _git(root, "add", "--all")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD").decode().strip()


def _approve(root, base, monkeypatch):
    patch = _git(root, "diff", "--binary", base, "HEAD")
    digest = hashlib.sha256(patch).hexdigest()
    monkeypatch.setenv(BASE_COMMIT_ENV, base)
    monkeypatch.setenv(PATCH_SHA256_ENV, digest)
    return digest


@pytest.fixture(autouse=True)
def no_inherited_approval(monkeypatch):
    monkeypatch.delenv(BASE_COMMIT_ENV, raising=False)
    monkeypatch.delenv(PATCH_SHA256_ENV, raising=False)


@pytest.fixture
def repair(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "utils").mkdir()
    (root / "utils/wandb_resume.py").write_text("original verifier\n")
    (root / "README.md").write_text("original documentation\n")
    base = _commit(root, "base")
    before = source_identity(root)
    (root / "utils/wandb_resume.py").write_text("reviewed sparse-null repair\n")
    head = _commit(root, "telemetry repair")
    after = source_identity(root)
    digest = _approve(root, base, monkeypatch)
    return root, base, head, before, after, digest


def _identity(source):
    scientific = {
        "source": copy.deepcopy(source),
        "trial_run_params": {"seed": 55, "total_steps": 1000000},
        "experiment_params": {"configs": ["reward-only"]},
        "dependencies": {"torch": "2.0", "wandb": "0.25.1"},
    }
    return _rehash({"schema_version": 1, "scientific": scientific})


def _rehash(identity):
    identity["fingerprint"] = hashlib.sha256(
        canonical_json(identity["scientific"]).encode()
    ).hexdigest()
    return identity


def _binding(commit):
    return {
        "commit": commit,
        "manifest": "configs/campaign.json",
        "manifest_sha256": "a" * 64,
        "config_sha256": {"reward-only": "b" * 64},
        "lock_sha256": "c" * 64,
    }


def test_without_opt_in_helpers_return_none(repair, monkeypatch):
    root, _, _, before, after, _ = repair
    monkeypatch.delenv(BASE_COMMIT_ENV)
    monkeypatch.delenv(PATCH_SHA256_ENV)
    assert telemetry_source_compatibility(root, before, after) is None
    assert compatible_scientific_identity(root, _identity(before), _identity(after)) is None
    assert compatible_gate_binding(root, _binding(before["commit"]), _binding(after["commit"])) is None


def test_approved_patch_attests_both_sources_without_mutation(repair):
    root, base, head, before, after, digest = repair
    before_copy, after_copy = copy.deepcopy(before), copy.deepcopy(after)
    proof = telemetry_source_compatibility(root, before, after)
    assert proof == {
        "rule": COMPATIBILITY_RULE,
        "base_source": before,
        "execution_source": after,
        "patch_sha256": digest,
        "changed_paths": ["utils/wandb_resume.py"],
    }
    assert proof["base_source"]["commit"] == base
    assert proof["execution_source"]["commit"] == head
    proof["base_source"]["commit"] = "changed returned copy"
    assert before == before_copy and after == after_copy


@pytest.mark.parametrize("missing", [BASE_COMMIT_ENV, PATCH_SHA256_ENV])
def test_partial_opt_in_is_rejected(repair, monkeypatch, missing):
    root, _, _, before, after, _ = repair
    monkeypatch.delenv(missing)
    with pytest.raises(ResumeTelemetryCompatibilityError, match="requires both"):
        telemetry_source_compatibility(root, before, after)


@pytest.mark.parametrize("variable,value", [
    (BASE_COMMIT_ENV, ""), (BASE_COMMIT_ENV, "shortsha"),
    (BASE_COMMIT_ENV, "A" * 40), (PATCH_SHA256_ENV, ""),
    (PATCH_SHA256_ENV, "a" * 63), (PATCH_SHA256_ENV, "A" * 64),
])
def test_invalid_opt_in_values_are_rejected(repair, monkeypatch, variable, value):
    root, _, _, before, after, _ = repair
    monkeypatch.setenv(variable, value)
    with pytest.raises(ResumeTelemetryCompatibilityError):
        telemetry_source_compatibility(root, before, after)


def test_wrong_patch_digest_is_rejected(repair, monkeypatch):
    root, _, _, before, after, _ = repair
    monkeypatch.setenv(PATCH_SHA256_ENV, "0" * 64)
    with pytest.raises(ResumeTelemetryCompatibilityError, match="reviewed SHA-256"):
        telemetry_source_compatibility(root, before, after)


@pytest.mark.parametrize("kind", ["tracked", "staged", "untracked"])
def test_dirty_execution_checkout_is_rejected(repair, kind):
    root, _, _, before, _, _ = repair
    path = root / ("untracked.txt" if kind == "untracked" else "README.md")
    path.write_text("unreviewed change\n")
    if kind == "staged":
        _git(root, "add", "README.md")
    with pytest.raises(ResumeTelemetryCompatibilityError, match="clean committed"):
        telemetry_source_compatibility(root, before, source_identity(root))


def test_supplied_source_must_match_checkout(repair):
    root, _, _, before, after, _ = repair
    after["commit"] = before["commit"]
    with pytest.raises(ResumeTelemetryCompatibilityError, match="does not match this checkout"):
        telemetry_source_compatibility(root, before, after)


@pytest.mark.parametrize("field,value", [
    ("dirty", True), ("dirty", 0), ("untracked_count", 1),
    ("tracked_diff_sha256", "0" * 64), ("untracked_sha256", "0" * 64),
    ("commit", "0" * 40),
])
def test_original_source_must_be_pristine_approved_base(repair, field, value):
    root, _, _, before, after, _ = repair
    before[field] = value
    with pytest.raises(ResumeTelemetryCompatibilityError, match="pristine approved base"):
        telemetry_source_compatibility(root, before, after)


@pytest.mark.parametrize("path", ["RL/model.py", "configs/campaign.json", "environments/dmcontrol/uv.lock"])
def test_even_reviewed_digest_cannot_admit_scientific_file_changes(repair, monkeypatch, path):
    root, base, _, before, _, _ = repair
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed scientific implementation\n")
    _commit(root, "forbidden scientific change")
    _approve(root, base, monkeypatch)
    with pytest.raises(ResumeTelemetryCompatibilityError, match="only approved telemetry paths"):
        telemetry_source_compatibility(root, before, source_identity(root))


def test_repair_must_change_wandb_resume(repair, monkeypatch):
    root, base, _, before, _, _ = repair
    # A descendant can undo the repair; a README-only final diff is insufficient.
    (root / "utils/wandb_resume.py").write_text("original verifier\n")
    (root / "README.md").write_text("documentation only\n")
    _commit(root, "undo repair")
    _approve(root, base, monkeypatch)
    with pytest.raises(ResumeTelemetryCompatibilityError, match="must change utils/wandb_resume.py"):
        telemetry_source_compatibility(root, before, source_identity(root))


def test_all_reviewed_support_paths_are_allowed(repair, monkeypatch):
    root, base, _, before, _, _ = repair
    for relative in ALLOWED_CHANGED_PATHS - {"utils/wandb_resume.py"}:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reviewed telemetry support\n")
    _commit(root, "reviewed support")
    _approve(root, base, monkeypatch)
    proof = telemetry_source_compatibility(root, before, source_identity(root))
    assert proof["changed_paths"] == sorted(ALLOWED_CHANGED_PATHS)


def test_non_descendant_execution_is_rejected(repair, monkeypatch):
    root, base, _, before, _, _ = repair
    _git(root, "checkout", "--orphan", "unrelated")
    (root / "utils/wandb_resume.py").write_text("unrelated tree\n")
    _commit(root, "unrelated root")
    _approve(root, base, monkeypatch)
    with pytest.raises(ResumeTelemetryCompatibilityError, match="committed telemetry"):
        telemetry_source_compatibility(root, before, source_identity(root))


def test_scientific_identity_accepts_only_honest_source_difference(repair):
    root, _, _, before, after, _ = repair
    expected, actual = _identity(before), _identity(after)
    snapshots = copy.deepcopy((expected, actual))
    proof = compatible_scientific_identity(root, expected, actual)
    assert proof["base_source"] == before and proof["execution_source"] == after
    assert proof["original_fingerprint"] == expected["fingerprint"]
    assert proof["execution_fingerprint"] == actual["fingerprint"]
    assert (expected, actual) == snapshots


@pytest.mark.parametrize("section,key,value", [
    ("trial_run_params", "seed", 56),
    ("trial_run_params", "total_steps", 2000000),
    ("trial_run_params", "seed", 55.0),
    ("experiment_params", "configs", ["different"]),
    ("dependencies", "wandb", "0.26.0"),
])
def test_scientific_config_and_dependencies_stay_exact(repair, section, key, value):
    root, _, _, before, after, _ = repair
    actual = _identity(after)
    actual["scientific"][section][key] = value
    _rehash(actual)
    with pytest.raises(ResumeTelemetryCompatibilityError, match="cannot change scientific"):
        compatible_scientific_identity(root, _identity(before), actual)


@pytest.mark.parametrize("which", ["expected", "actual"])
@pytest.mark.parametrize("fault", ["fingerprint", "schema", "bool_schema", "extra_field", "missing_source"])
def test_malformed_scientific_identities_are_rejected(repair, which, fault):
    root, _, _, before, after, _ = repair
    identities = {"expected": _identity(before), "actual": _identity(after)}
    target = identities[which]
    if fault == "fingerprint":
        target["fingerprint"] = "0" * 64
    elif fault == "schema":
        target["schema_version"] = 2
    elif fault == "bool_schema":
        target["schema_version"] = True
    elif fault == "extra_field":
        target["extra"] = 1
    else:
        target["scientific"].pop("source")
        _rehash(target)
    with pytest.raises(ResumeTelemetryCompatibilityError):
        compatible_scientific_identity(root, identities["expected"], identities["actual"])


def test_gate_binding_preserves_original_receipt_identity(repair):
    root, base, head, _, _, _ = repair
    expected, actual = _binding(base), _binding(head)
    snapshots = copy.deepcopy((expected, actual))
    proof = compatible_gate_binding(root, expected, actual)
    assert proof["base_source"]["commit"] == base
    assert proof["execution_source"]["commit"] == head
    assert (expected, actual) == snapshots


@pytest.mark.parametrize("field,value", [
    ("manifest", "configs/other.json"), ("manifest_sha256", "d" * 64),
    ("config_sha256", {"reward-only": "d" * 64}), ("lock_sha256", "d" * 64),
    ("extra", "unexpected"),
])
def test_gate_binding_cannot_change_other_fields(repair, field, value):
    root, base, head, _, _, _ = repair
    actual = _binding(head)
    actual[field] = value
    with pytest.raises(ResumeTelemetryCompatibilityError, match="cannot change gate"):
        compatible_gate_binding(root, _binding(base), actual)


def test_gate_execution_commit_must_be_head(repair):
    root, base, _, _, _, _ = repair
    with pytest.raises(ResumeTelemetryCompatibilityError, match="does not match this checkout"):
        compatible_gate_binding(root, _binding(base), _binding(base))
