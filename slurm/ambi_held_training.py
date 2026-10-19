"""Gate one H3/J8/K3 aux-return run, then use the durable resume launcher."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_training import binding, digest, read, recipe, resolve_manifest, runtime, write_new

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_held_training.json"
NAME = "ambi_full_h3_j8_k3_fresh"
CPU_COUNTS = (1, 2, 3)
CUDA_TEST = "tests/test_ambi_held_policy_cuda_gate.py::test_production_shape_held_policy_cuda_gate[aux-fresh-k3]"


def validate_campaign(manifest_path=MANIFEST):
    path = resolve_manifest(manifest_path)
    manifest = read(path)
    assert manifest["configs"] == [NAME] and manifest["trials"] == 1
    name, _, run = recipe(0, path)
    assert (run["seed"], run["total_steps"], run["checkpoint_every"], run["save_strat"]) == (55, 500000, 25000, "all")
    p = run["alg_params"]
    expected = {
        "inner_operator": "sac", "inner_rounds": 8, "inner_rollout_horizon": 3,
        "inner_solve_interval": 3, "inner_critic_bernoulli_p": 0.,
        "inner_rollouts_per_round": 128, "inner_batch_size": 256,
        "inner_critic_updates_per_round": 16, "inner_actor_updates_per_round": 4,
        "inner_replay_capacity": 3072, "inner_execution_action": "policy_sample",
        "aux_return_mode": "sac", "aux_return_joint_vmap": False,
        "inner_critic_source": "aux_return", "inner_horizon_critic_source": "aux_return",
        "inner_sac_critic_target": "reward_only", "outer_critic_target": "entropy_augmented",
        "inner_actor_source": "sac", "inner_horizon_actor_source": "sac",
        "inner_actor_initialization": "prior", "inner_critic_initialization": "prior",
        "compile": True, "compile_strict": True, "critic_ensemble_backend": "vmap",
        "wandb": True, "wandb_project": "ambi", "wandb_mode": "online",
        "seed_steps": 2500, "pretrain_steps": 2500, "utd": 1,
    }
    assert all(p.get(key) == value for key, value in expected.items()), "Production recipe changed"
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    return name, manifest, run


def validate_receipt(receipt, campaign_binding, current_runtime, cpus):
    assert receipt["passed"] is True and receipt["binding"] == campaign_binding, "Gate source/configuration mismatch"
    assert cpus in CPU_COUNTS, "Production CPU count was not measured"
    for key in ("python", "torch", "cuda", "executable", "gpu"):
        assert receipt["runtime"][key] == current_runtime[key], f"Gate used another {key}"
    gate = receipt["cuda_gate"]
    assert gate["passed"] is True and gate["binding"] == campaign_binding
    assert gate["runtime"] == receipt["runtime"]
    assert (gate["config"], gate["interval"], gate["bernoulli_probability"]) == (NAME, 3, 0.)
    assert gate["compile_strict"] is True and gate["fallbacks"] == 0
    assert gate["checkpoint_roundtrip"] is True and gate["boundary_training_state_roundtrip"] is True
    cases = receipt["cases"]
    assert len(cases) == len(CPU_COUNTS) and {row["physical_cores"] for row in cases} == set(CPU_COUNTS)
    for row in cases:
        assert row["passed"] is True and row["binding"] == campaign_binding and row["config"] == NAME
        assert row["runtime"] == receipt["runtime"] and row["compile_fallback"] is False
        assert row["replay_storage_device"] == "cpu", "Benchmark must match exact-resume CPU replay"
        assert row["compile_graph_counts"]["before"] == row["compile_graph_counts"]["after"]
        assert row["measured"]["decisions"] == row["measured"]["outer_updates"]
        assert row["measured"]["decisions"] >= 100
        assert math.isfinite(row["measured"]["seconds_per_decision"]) and row["measured"]["seconds_per_decision"] > 0


def smoke(args):
    assert os.environ.get("SLURM_JOB_ID"), "Run the gate inside a Slurm GPU job"
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) >= max(CPU_COUNTS)
    validate_campaign(args.manifest)
    output = args.output.resolve()
    assert output != ROOT and ROOT not in output.parents
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, WANDB_MODE="disabled", AMBI_RUN_HELD_POLICY_CUDA_GATE="1")
    commands = [
        ("cuda", [sys.executable, "-m", "pytest", "-q", "-s", CUDA_TEST,
                  "--basetemp", str(output / "cuda")]),
        ("cpu", [sys.executable, "slurm/benchmark_ambi_training_cpu.py",
                 "--manifest", str(resolve_manifest(args.manifest)), "--cell-index", "0",
                 "--cpu-counts", ",".join(map(str, CPU_COUNTS)),
                 "--warmup", str(args.warmup), "--decisions", str(args.decisions),
                 "--output", str(output / "cpu")]),
    ]
    for label, command in commands:
        with (output / f"{label}.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f"HELD_TRAINING_GATE_PROGRESS {label}", flush=True)
    cuda_path, cpu_path = output / "cuda/held-policy-cuda-gate.json", output / "cpu/results.json"
    cuda_gate, benchmark = read(cuda_path), read(cpu_path)
    assert benchmark["passed"] is True
    receipt = {"passed": True, "binding": binding(args.manifest), "runtime": runtime(),
               "cuda_gate": cuda_gate, "cases": benchmark["cases"],
               "artifacts": {str(path): digest(path) for path in (cuda_path, cpu_path)},
               "created_utc": datetime.now(timezone.utc).isoformat()}
    validate_receipt(receipt, receipt["binding"], receipt["runtime"], CPU_COUNTS[0])
    fastest = min(row["measured"]["seconds_per_decision"] for row in receipt["cases"])
    receipt["recommended_cpus_within_5pct_of_best"] = min(
        row["physical_cores"] for row in receipt["cases"]
        if row["measured"]["seconds_per_decision"] <= fastest / .95
    )
    write_new(output / "receipt.json", receipt)
    print("HELD_TRAINING_GATE_PASSED " + str(output / "receipt.json"), flush=True)


def launch(args):
    name, manifest, run = validate_campaign(args.manifest)
    campaign_binding, current_runtime = binding(args.manifest), runtime()
    receipt = read(args.gate)
    validate_receipt(receipt, campaign_binding, current_runtime, int(os.environ["SLURM_CPUS_PER_TASK"]))
    for path, checksum in receipt["artifacts"].items():
        assert digest(path) == checksum, "Gate artifact changed"
    assert os.environ["AMBI_DURABLE_ROOT"] == "/oscar/home/rgao48/ambi-durable"
    lineage = Path(os.environ["AMBI_LINEAGE_DIR"])
    assert lineage.is_absolute() and lineage != Path(os.environ["AMBI_DURABLE_ROOT"])
    lineage.resolve().relative_to(Path(os.environ["AMBI_DURABLE_ROOT"]).resolve())
    assert int(os.environ["AMBI_DRAIN_AFTER_SECONDS"]) > 0
    segment = os.environ["SLURM_JOB_ID"] + "." + os.environ.get("SLURM_RESTART_COUNT", "0")
    args.output.mkdir(parents=True, exist_ok=True)
    write_new(args.output / f"launch-{segment}.json", {
        "binding": campaign_binding, "config": name, "run_params": run,
        "gate_receipt": str(args.gate), "gate_sha256": digest(args.gate),
        "runtime": current_runtime, "lineage_dir": str(lineage), "segment": segment,
        "resume_mode": "required" if os.environ.get("SLURM_RESTART_COUNT", "0") != "0" else os.environ.get("AMBI_RESUME_MODE", "new"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    os.environ.update(AMBI_RUN_CONFIG=str(resolve_manifest(args.manifest)),
                      AMBI_ALG_DIR=str(ROOT / "configs/dmcontrol/algs"),
                      AMBI_PYTHON=sys.executable, SLURM_SUBMIT_DIR=str(ROOT))
    # The existing launcher owns lineage locking, quota preflight, safe draining,
    # LATEST/HANDOFF/DONE verification, W&B continuation and scheduler requeue.
    os.execv("/bin/bash", ["/bin/bash", str(ROOT / "run_ambi_oscar.sh")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--decisions", type=int, default=120)
    args = parser.parse_args()
    assert binding(args.manifest)["commit"] == os.environ["AMBI_EXPECTED_COMMIT"]
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip()
    if args.mode == "smoke":
        assert args.warmup >= 12 and args.warmup % 3 == 0
        assert args.decisions >= 100 and args.decisions % 3 == 0
        smoke(args)
    else:
        assert args.gate is not None
        launch(args)


if __name__ == "__main__":
    main()
