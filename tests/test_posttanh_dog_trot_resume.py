"""Restart identity and fail-closed Slurm handoff orchestration."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from slurm.posttanh_dog_trot_resume import validate_recipe, record_handoff, check_speed

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_posttanh_dog_trot_resume_oscar.sbatch"


def test_restart_retains_seed56_recipe_and_all_model_checkpoints():
    _, manifest, run = validate_recipe()
    assert run["seed"] == 56 and run["total_steps"] == 1_000_000
    assert len(manifest["configs"]) == 1 and manifest["save_trials"] == "none"
    assert run["checkpoint_every"] == 25_000 and run["save_strat"] == "all"
    assert run["alg_params"]["critic_geometry_diagnostics"]


def _launcher(tmp_path, *, final_status=75, first_status=75, restart=0,
              handoff_failure=False, resume_mode="new", node_status=0):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    audit = tmp_path / "audit.jsonl"
    program = f'''#!{sys.executable}
import json,os,sys
from pathlib import Path
name=Path(sys.argv[0]).name
args=sys.argv[1:]
with open(os.environ["AUDIT"],"a") as f: f.write(json.dumps([name,args])+"\\n")
if name=="git":
 if args==["rev-parse","--show-toplevel"]: print(os.environ["SLURM_SUBMIT_DIR"])
 elif args==["rev-parse","HEAD"]: print(os.environ["AMBI_EXPECTED_COMMIT"])
elif name=="python":
 if "handoff" in args and os.environ["FAIL_HANDOFF"]=="1": sys.exit(9)
elif name=="srun":
 if "node-check" in args: sys.exit(int(os.environ["NODE_STATUS"]))
 mode=args[args.index("--resume-mode")+1]
 sys.exit(int(os.environ["FIRST_STATUS"] if mode=="new" else os.environ["FINAL_STATUS"]))
'''
    for name in ("git", "python", "srun", "scontrol", "mkdir", "checkquota"):
        path = bin_dir / name
        path.write_text(program)
        path.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ["PATH"],
               AUDIT=str(audit), SLURM_JOB_ID="12345", SLURM_RESTART_COUNT=str(restart),
               SLURM_SUBMIT_DIR=str(tmp_path), SLURM_CPUS_PER_TASK="1",
               AMBI_EXPECTED_COMMIT="a"*40, AMBI_PYTHON=str(bin_dir/"python"),
               AMBI_WORKSPACE_PYTHON=str(bin_dir/"python"),
               AMBI_RESUME_CAMPAIGN="/oscar/scratch/rgao48/ambi/test-resume",
               AMBI_LINEAGE_DIR="/oscar/home/rgao48/ambi-durable/lineages/test-resume",
               AMBI_TRAIN_WALLTIME_SECONDS="219600", AMBI_RESUME_MODE=resume_mode,
               NODE_STATUS=str(node_status),
               FIRST_STATUS=str(first_status), FINAL_STATUS=str(final_status),
               FAIL_HANDOFF=str(int(handoff_failure)))
    result = subprocess.run(["bash", str(LAUNCHER), "train"], cwd=tmp_path, env=env,
                            text=True, capture_output=True)
    calls = [json.loads(line) for line in audit.read_text().splitlines()]
    return result, calls


@pytest.mark.parametrize("final_status", [0, 75])
def test_real_bootstrap_is_followed_by_required_resume(tmp_path, final_status):
    result, calls = _launcher(tmp_path, final_status=final_status)
    assert result.returncode == 0, result.stderr
    segments = [args for name, args in calls if name == "srun" and "--resume-mode" in args]
    assert [a[a.index("--resume-mode")+1] for a in segments] == ["new", "required"]
    assert segments[0][segments[0].index("--drain-seconds")+1] == "600"
    assert all(a[a.index("--checkpoint-minutes")+1] == "15" for a in segments)
    assert [args for name,args in calls if name == "scontrol"] == (
        [["requeue", "12345"]] if final_status == 75 else [])


def test_failed_first_segment_never_starts_required_or_requeues(tmp_path):
    result, calls = _launcher(tmp_path, first_status=12)
    assert result.returncode != 0
    assert sum(name == "srun" and "--resume-mode" in args for name,args in calls) == 1
    assert not any(name == "scontrol" for name,_ in calls)


def test_failed_prior_handoff_prevents_requeued_training(tmp_path):
    result, calls = _launcher(tmp_path, restart=1, handoff_failure=True)
    assert result.returncode != 0
    assert not any(name in ("srun", "scontrol") for name,_ in calls)


def test_requeue_requires_latest_without_repeating_bootstrap(tmp_path):
    result, calls = _launcher(tmp_path, restart=1)
    assert result.returncode == 0, result.stderr
    segments = [args for name,args in calls if name == "srun" and "--resume-mode" in args]
    assert len(segments) == 1
    assert segments[0][segments[0].index("--resume-mode")+1] == "required"
    checks = [args for name,args in calls if name == "python" and "handoff" in args]
    assert checks[0][-1] == "12345.0.train"


def test_bootstrap_proof_rejects_untrained_checkpoint(tmp_path, monkeypatch):
    lineage = tmp_path / "lineage"
    generation = lineage / "generations" / "first"
    generation.mkdir(parents=True)
    (lineage / "LATEST").write_text("first\n")
    (generation / "manifest.json").write_text(json.dumps({"metadata": {
        "global_step": 2500, "num_updates": 0, "replay_rows": 2500}}))
    monkeypatch.setenv("AMBI_LINEAGE_DIR", str(lineage))
    from types import SimpleNamespace
    with pytest.raises(AssertionError):
        record_handoff(SimpleNamespace(output=tmp_path))
    assert not (tmp_path / "bootstrap-handoff.json").exists()


def test_new_job_can_resume_after_unplanned_failure(tmp_path):
    result, calls = _launcher(tmp_path, resume_mode="required", final_status=0)
    assert result.returncode == 0, result.stderr
    segments = [args for name, args in calls if name == "srun" and "--resume-mode" in args]
    assert len(segments) == 1
    assert segments[0][segments[0].index("--resume-mode") + 1] == "required"
    assert not any("record-handoff" in args for _, args in calls)


def test_slow_node_never_starts_training_or_requeues(tmp_path):
    result, calls = _launcher(tmp_path, node_status=3)
    assert result.returncode != 0
    assert not any("--resume-mode" in args for _, args in calls)
    assert not any(name == "scontrol" for name, _ in calls)


@pytest.mark.parametrize("seconds, accepted", [(.18, True), (.225, True), (.31, False), (float("nan"), False)])
def test_rejects_timed_out_node_throughput(seconds, accepted):
    report = {"passed": True, "cpu_replay": True, "gpu_replay": False,
              "wandb": False, "compile_fallback": False,
              "measured": {"seconds_per_decision": seconds}}
    if accepted:
        check_speed(report)
    else:
        with pytest.raises(AssertionError):
            check_speed(report)
