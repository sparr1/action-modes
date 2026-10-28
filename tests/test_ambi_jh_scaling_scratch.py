"""The opt-in scratch campaign retains the exact-resume safety checks."""
from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from slurm import ambi_reward_training as campaign
from tests.test_ambi_return_jh_scaling_training import CASES, scaling_receipt

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "slurm/run_ambi_jh_scaling_scratch.sh"
APPROVED = "/oscar/scratch/rgao48/ambi/jh-scaling-training/20261028"


@pytest.mark.parametrize("path,spec", CASES)
def test_gate_dispatch_binds_scratch_lineage_to_exact_recipe(monkeypatch, tmp_path, path, spec):
    receipt, _ = scaling_receipt(spec)
    receipt["artifacts"] = {}
    gate = tmp_path / "gate.json"
    campaign.write_new(gate, receipt)
    monkeypatch.setattr(campaign, "binding", lambda _: receipt["binding"])
    monkeypatch.setattr(campaign, "runtime", lambda: receipt["runtime"])
    _, _, rounds, interval, horizon, _ = spec
    lineage = f"{APPROVED}/lineages/h{horizon}-j{rounds}-k{interval}-seed55"
    for key, value in {
        "AMBI_RETURN_STORAGE": "scratch-jh-scaling", "AMBI_DURABLE_ROOT": APPROVED,
        "AMBI_DURABLE_QUOTA_LABEL": "rgao48", "AMBI_DURABLE_QUOTA_PATH": "/oscar/scratch",
        "AMBI_LINEAGE_DIR": lineage, "SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_ID": "123",
        "SLURM_RESTART_COUNT": "0", "AMBI_DRAIN_AFTER_SECONDS": "3600", "AMBI_RESUME_MODE": "new",
    }.items():
        monkeypatch.setenv(key, value)
    for key in ("AMBI_RUN_CONFIG", "AMBI_ALG_DIR", "AMBI_PYTHON", "SLURM_SUBMIT_DIR", "AMBI_JH_LAUNCH_RECEIPT"):
        monkeypatch.setenv(key, "test-placeholder")
    calls = []
    monkeypatch.setattr(campaign.os, "execv", lambda *args: calls.append(args))
    args = Namespace(manifest=path, gate=gate, output=tmp_path / "output")
    campaign.launch(args)
    record = json.loads((args.output / "launch-123.0.json").read_text())
    assert record["lineage_dir"] == lineage and record["resume_mode"] == "new"
    assert record["storage"] == {"mode": "scratch-jh-scaling", "root": APPROVED, "quota_path": "/oscar/scratch"}
    assert calls == [("/bin/bash", ["/bin/bash", str(SCRIPT)])]
    assert os.environ["AMBI_JH_LAUNCH_RECEIPT"] == str(args.output / "launch-123.0.json")
    # A neighboring recipe cannot reuse this lineage, even with its own valid gate.
    monkeypatch.setenv("AMBI_LINEAGE_DIR", lineage + "-other")
    with pytest.raises(AssertionError, match="selected recipe"):
        campaign.launch(args)


def _executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


def _record(lineage, segment, *, done=False):
    generation = "generation-1"
    (lineage / "generations" / generation).mkdir(parents=True, exist_ok=True)
    (lineage / "LINEAGE.json").write_text(json.dumps({"schema_version": 1, "metadata": {"total_steps": 1000000}}))
    (lineage / "LATEST").write_text(generation + "\n")
    (lineage / "generations" / generation / "manifest.json").write_text(json.dumps({
        "generation_id": generation, "metadata": {"segment_id": segment, "global_step": 1000000 if done else 500000},
    }))
    record = {"schema_version": 1, "slurm_job_id": segment.split(".")[0], "segment_id": segment, "generation_id": generation}
    (lineage / "HANDOFF.json").write_text(json.dumps(record))
    if done:
        (lineage / "DONE").write_text(json.dumps({**record, "global_step": 1000000, "target_step": 1000000}))


def _environment(tmp_path, *, status=75, restart=0):
    source = tmp_path / "source"
    (source / "utils").mkdir(parents=True)
    (source / "utils/__init__.py").touch()
    shutil.copyfile(ROOT / "utils/oscar_resume_launcher.py", source / "utils/oscar_resume_launcher.py")
    (source / "main.py").write_text("raise AssertionError('scheduler must be mocked')\n")
    manifest = source / "configs/dmcontrol/experiments/ambi_return_only_sac_h3_j12_k3_training.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n")
    algs = source / "configs/dmcontrol/algs"
    algs.mkdir()
    (algs / "algorithm.json").write_text("{}\n")
    for command in (["git", "init", "-q"], ["git", "add", "main.py", "configs", "utils"],
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "test"]):
        subprocess.run(command, cwd=source, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    output = tmp_path / "scratch"
    output.mkdir()
    lineage = output / "lineages/h3-j12-k3-seed55"
    if restart:
        _record(lineage, f"123.{restart - 1}")
    result = tmp_path / "result"
    _record(result, f"123.{restart}", done=status == 0)
    gate = tmp_path / "gate.json"
    gate.write_text('{"artifacts": {}}\n')
    receipt = tmp_path / "launch.json"
    receipt.write_text(json.dumps({
        "binding": {"commit": commit}, "config": "ambi_full_h3_j12_k3_reward_only",
        "lineage_dir": str(lineage), "segment": f"123.{restart}", "resume_mode": "required" if restart else "new",
        "storage": {"mode": "scratch-jh-scaling", "root": str(output), "quota_path": "/oscar/scratch"},
        "gate_receipt": str(gate), "gate_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
    }))
    binary = tmp_path / "bin"
    binary.mkdir()
    _executable(binary / "checkquota", """#!/bin/sh
sleep "${FAKE_PREFLIGHT_SECONDS:-0}"
printf '%s\n' "rgao48 /oscar/home 1 TB 2 TB OK None" "rgao48 /oscar/scratch 1 TB 2 TB ${FAKE_QUOTA_STATE:-OK} None"
""")
    _executable(binary / "srun", """#!/bin/sh
printf '%s\n' "$@" > "$FAKE_SRUN_ARGS"
printf '%s\n' "$AMBI_SEGMENT_ID" "$WANDB__REQUIRE_CORE" "$WANDB_MODE" "${WANDB_RUN_ID:-unset}" "${WANDB_RESUME:-unset}" > "$FAKE_SRUN_ENV"
if [ -n "${FAKE_RESULT:-}" ]; then mkdir -p "$AMBI_LINEAGE_DIR"; cp -R "$FAKE_RESULT"/. "$AMBI_LINEAGE_DIR"/; fi
exit "$FAKE_STATUS"
""")
    _executable(binary / "scontrol", "#!/bin/sh\nprintf '%s\n' \"$*\" >> \"$FAKE_REQUEUE\"\n")
    # Only relocate the fixed campaign root for the sandbox; all checks execute.
    script = tmp_path / "launcher.sh"
    script.write_text(SCRIPT.read_text().replace(f'approved_root="{APPROVED}"', f'approved_root="{output}"'))
    env = os.environ.copy()
    for key in ("SLURM_ARRAY_TASK_ID", "AMBI_RESUME_GENERATION", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update({
        "PATH": f"{binary}{os.pathsep}{env['PATH']}", "SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "1",
        "SLURM_RESTART_COUNT": str(restart), "AMBI_PROJECT_DIR": str(source), "AMBI_EXPECTED_COMMIT": commit,
        "AMBI_PYTHON": sys.executable, "AMBI_JH_LAUNCH_RECEIPT": str(receipt), "AMBI_LINEAGE_DIR": str(lineage),
        "AMBI_DURABLE_ROOT": str(output), "AMBI_DURABLE_QUOTA_LABEL": "rgao48", "AMBI_DURABLE_QUOTA_PATH": "/oscar/scratch",
        "AMBI_RETURN_STORAGE": "scratch-jh-scaling", "AMBI_RUN_CONFIG": str(manifest), "AMBI_ALG_DIR": str(algs),
        "AMBI_DRAIN_AFTER_SECONDS": "3600", "AMBI_RESUME_MODE": "new", "FAKE_STATUS": str(status),
        "FAKE_RESULT": str(result), "FAKE_SRUN_ARGS": str(tmp_path / "srun-args"), "FAKE_SRUN_ENV": str(tmp_path / "srun-env"),
        "FAKE_REQUEUE": str(tmp_path / "requeue"), "WANDB_RUN_ID": "wrong-id", "WANDB_RESUME": "never",
    })
    return env, script


def _run(env, script):
    return subprocess.run(["bash", str(script)], env=env, cwd=ROOT, text=True, capture_output=True)


@pytest.mark.parametrize("status,restart", [(75, 0), (0, 0), (75, 1), (0, 1)])
def test_fresh_and_restarted_training_verify_completion_or_handoff(tmp_path, status, restart):
    env, script = _environment(tmp_path, status=status, restart=restart)
    result = _run(env, script)
    assert result.returncode == 0, result.stderr
    args = Path(env["FAKE_SRUN_ARGS"]).read_text().splitlines()
    assert args[args.index("--resume-mode") + 1] == ("required" if restart else "new")
    assert args[args.index("--lineage-dir") + 1] == env["AMBI_LINEAGE_DIR"]
    assert args[args.index("--resume-wandb-mode") + 1] == "online"
    assert Path(env["FAKE_SRUN_ENV"]).read_text().splitlines() == [f"123.{restart}", "true", "online", "unset", "unset"]
    assert Path(env["FAKE_REQUEUE"]).exists() == (status == 75)
    if status == 75:
        assert Path(env["FAKE_REQUEUE"]).read_text() == "requeue 123\n"


@pytest.mark.parametrize("condition", ["existing", "source", "quota", "root", "lineage", "manifest", "selection", "gate", "symlink", "missing-proof"])
def test_invalid_preflight_never_starts_or_requeues(tmp_path, condition):
    env, script = _environment(tmp_path)
    if condition == "existing":
        _record(Path(env["AMBI_LINEAGE_DIR"]), "old.0")
    elif condition == "source":
        (Path(env["AMBI_PROJECT_DIR"]) / "main.py").write_text("changed\n")
    elif condition == "quota":
        env["FAKE_QUOTA_STATE"] = "GRACE_EXPIRED"
    elif condition == "root":
        env["AMBI_DURABLE_ROOT"] += "-other"
    elif condition == "lineage":
        env["AMBI_LINEAGE_DIR"] += "-other"
    elif condition == "manifest":
        env["AMBI_RUN_CONFIG"] = env["AMBI_RUN_CONFIG"].replace("h3_j12_k3", "h4_j10_k4")
    elif condition == "selection":
        env.pop("AMBI_RETURN_STORAGE")
    elif condition == "gate":
        (tmp_path / "gate.json").write_text("changed")
    elif condition == "symlink":
        target = tmp_path / "other-lineages"
        target.mkdir()
        (Path(env["AMBI_DURABLE_ROOT"]) / "lineages").symlink_to(target)
    else:
        env.pop("AMBI_JH_LAUNCH_RECEIPT")
    result = _run(env, script)
    assert result.returncode != 0
    assert not Path(env["FAKE_SRUN_ARGS"]).exists()
    assert not Path(env["FAKE_REQUEUE"]).exists()


@pytest.mark.parametrize("status", [0, 75, 23])
def test_unverified_success_handoff_and_failure_never_requeue(tmp_path, status):
    env, script = _environment(tmp_path, status=status)
    env.pop("FAKE_RESULT")
    result = _run(env, script)
    assert result.returncode != 0
    assert not Path(env["FAKE_REQUEUE"]).exists()


def test_stale_restart_handoff_never_starts(tmp_path):
    env, script = _environment(tmp_path, restart=2)
    _record(Path(env["AMBI_LINEAGE_DIR"]), "123.0")
    result = _run(env, script)
    assert result.returncode != 0 and "stale segment" in result.stderr
    assert not Path(env["FAKE_SRUN_ARGS"]).exists()


def test_quota_preflight_consumes_deadline(tmp_path):
    env, script = _environment(tmp_path)
    env.update(AMBI_DRAIN_AFTER_SECONDS="1", FAKE_PREFLIGHT_SECONDS="2")
    result = _run(env, script)
    assert result.returncode != 0 and "expired during startup" in result.stderr
    assert not Path(env["FAKE_SRUN_ARGS"]).exists()


def test_fixed_production_root_and_syntax():
    assert f'approved_root="{APPROVED}"' in SCRIPT.read_text()
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
