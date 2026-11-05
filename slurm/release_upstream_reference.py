"""Submit the scientific reference only after its GPU validation and sizing."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def resource_plan(rows, cfg, eval_seconds, method):
    seconds = {c: sum(x["seconds"] for x in rows if x["cpus"] == c) /
               sum(x["decisions"] for x in rows if x["cpus"] == c) for c in (1, 2)}
    cpus = 1 if seconds[1] <= 1.05 * seconds[2] else 2
    # BOOM's timing uses one instead of ten evaluation episodes. TD's timing
    # includes its complete native one-episode plus 100-trajectory MC evaluation.
    evaluation_multiplier = 10 if method == "boom" else 1
    evaluations = 1 + cfg["steps"] // cfg["eval_freq"]
    total_seconds = cfg["steps"] * seconds[cpus] + evaluations * eval_seconds * evaluation_multiplier
    return {"cpus": cpus, "seconds_per_decision": seconds,
            "eval_seconds": eval_seconds, "eval_multiplier": evaluation_multiplier,
            "margin": 1.25, "startup_hours": .5,
            "wall_hours": math.ceil(1.25 * total_seconds / 3600 + .5),
            "memory_gb": 8, "gpu": "l40s:1"}


def main():
    root = Path(os.environ["BOOM_REFERENCE_ROOT"])
    method = os.environ.get("BOOM_REFERENCE_METHOD", "boom")
    gate = os.environ["SLURM_JOB_ID"]
    state_path = root / "submission.json"
    state = json.loads(state_path.read_text())
    assert str(state["gate_job"]) == gate
    assert not str(state["job"]).isdigit(), "Scientific job already submitted"
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert sha == state["commit"] == os.environ["BOOM_OBSERVER_COMMIT"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    checks = json.loads((root / f"validate-{gate}/parity.json").read_text())["checks"]
    assert len(checks) == 2 and all(x["numerical_parity"] and x["rng_exact"] for x in checks)
    if os.environ.get("REFERENCE_ENTROPY", "released") == "ambi-posttanh":
        entropy = json.loads((root / f"validate-{gate}/entropy_ablation_parity.json").read_text())
        assert all(entropy[k] for k in ("actions_exact", "alignment_gradients_exact", "torch_rng_exact"))
        assert math.isclose(entropy["entropy_coefficient"], 0.0021)
    if os.environ.get("REFERENCE_ACCELERATION") == "model-planner-graph":
        model = json.loads((root / f"validate-{gate}/model_parity.json").read_text())
        assert len(model["checks"]) == 2 and all(x["numerical_parity"] and x["rng_exact"] for x in model["checks"])
    bench = root / f"benchmark-{gate}"
    assert (bench / "checkpoints/final.pt").exists(), "Timing run must complete"
    rows = json.loads((bench / "benchmark.json").read_text())
    assert len(rows) == 4 and all(x["seconds"] > 0 and x["decisions"] == 300 for x in rows)
    cfg = json.loads((bench / "resolved_config.json").read_text())
    cfg["steps"] = int(os.environ.get("REFERENCE_STEPS", "400000"))
    assert cfg["steps"] == state["requested_steps"]
    assert cfg.get("reference_entropy", "released") == os.environ.get("REFERENCE_ENTROPY", "released")
    planner = json.loads((root / f"validate-{gate}/planner_parity.json").read_text())
    assert len(planner["checks"]) == 3 and all(x["rng_exact"] for x in planner["checks"])
    events = [json.loads(line) for line in (bench / "events.jsonl").read_text().splitlines()]
    initial_eval = next(x["eval/total_time"] for x in events if x.get("eval/step") == 0)
    plan = resource_plan(rows, cfg, initial_eval, method)
    env = os.environ.copy()
    env["WANDB_RUN_ID"] = state["run_id"]
    # srun must receive the production allocation, not the sizing job's CPU count.
    for key in list(env):
        if key.startswith("SLURM_"): del env[key]
    cmd = ["sbatch", "--parsable", "--dependency=afterok:" + gate,
           "--job-name=" + ("boom-posttanh-s55" if os.environ.get("REFERENCE_ENTROPY") == "ambi-posttanh" else "boom-ref-s55" if method == "boom" else "pc2-ref-s55"),
           "--cpus-per-task=" + str(plan["cpus"]), "--mem=8G",
           "--time=" + str(plan["wall_hours"]) + ":00:00", "--exclude=gpu2708",
           "--output=" + str(root / "logs/train-%j.out"),
           "--error=" + str(root / "logs/train-%j.err"),
           "slurm/run_boom_reference_oscar.sbatch", "train"]
    job = subprocess.check_output(cmd, env=env, text=True).strip()
    state.update(job=job, plan=plan)
    state_path.write_text(json.dumps(state, indent=2))
    print("SCIENTIFIC_REFERENCE_SUBMITTED", job, json.dumps(plan), flush=True)
    subprocess.run([os.environ["BOOM_WORKSPACE_PYTHON"], "slurm/boom_reference_workspace.py",
                    os.environ.get("BOOM_WORKSPACE_ROOT", str(root))], check=True)


if __name__ == "__main__":
    main()
