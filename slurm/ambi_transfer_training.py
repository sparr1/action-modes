"""Validate and launch the six full AMBI training cells on a Slurm GPU."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_transfer_training.json"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_manifest(manifest_path=MANIFEST):
    """Select a versioned campaign within this pinned source checkout."""
    path = Path(manifest_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=True)
    path.relative_to(ROOT.resolve())
    return path


def binding(manifest_path=MANIFEST):
    manifest_path = resolve_manifest(manifest_path)
    manifest = read(manifest_path)
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": digest(manifest_path),
            "config_sha256": {name: digest(ROOT / "configs/dmcontrol/algs" / (name + ".json"))
                              for name in manifest["configs"]},
            "lock_sha256": digest(ROOT / "environments/dmcontrol/uv.lock")}


def recipe(index, manifest_path=MANIFEST):
    manifest = read(resolve_manifest(manifest_path))
    name = manifest["configs"][index]
    config = read(ROOT / "configs/dmcontrol/algs" / (name + ".json"))
    run = {"name": name, **config, **manifest["overrides_alg"]}
    return name, manifest, run


def runtime():
    import torch
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "executable": sys.executable}


def write_new(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def smoke(output, manifest_path=MANIFEST):
    """Exercise real Humanoid training and checkpoint I/O at production shapes."""
    import gymnasium as gym
    import torch
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    assert torch.cuda.is_available(), "Smoke must run on an allocated GPU"
    manifest_path = resolve_manifest(manifest_path)
    campaign_binding = binding(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index in range(6):
        name, manifest, run = recipe(index, manifest_path)
        original = deepcopy(run)
        overrides = {"wandb": False, "seed_steps": 500, "pretrain_steps": 2, "buffer_size": 4096}
        params = {**run["alg_params"], **overrides}
        run = {**run, "total_steps": 512, "alg_params": params}
        torch._dynamo.reset()
        env = gym.make(run["env"], **manifest["env_params"])
        learner = agent = act = observed_act = None
        started = time.perf_counter()
        try:
            learner = AMBITDMPC2(name + "-gpu-smoke", env, params, run, manifest)
            cfg, agent = learner.cfg, learner.agent
            assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
            assert cfg.aux_return_mode == "sac" and not agent.aux_return.has_actor
            rounds = int(original["alg_params"]["inner_rounds"])
            assert rounds > 0 and cfg.inner_rounds == rounds
            assert (cfg.inner_rollouts_per_round, cfg.inner_batch_size) == (128, 256)
            assert (cfg.inner_critic_updates_per_round, cfg.inner_actor_updates_per_round) == (16, 4)
            expected_critic_steps, expected_actor_steps = rounds * 16, rounds * 4
            assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (
                expected_critic_steps, expected_actor_steps,
            )
            actions = []
            act = agent.act

            def observed_act(*args, **kwargs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = act(*args, **kwargs)
                torch.cuda.synchronize()
                metrics = agent.last_inner_metrics
                assert metrics["inner_compile_fallback"] == 0.
                assert metrics["inner_critic_optimizer_steps"] == expected_critic_steps
                assert metrics["inner_actor_optimizer_steps"] == expected_actor_steps
                assert torch.isfinite(result).all()
                actions.append({"seconds": time.perf_counter() - start,
                                "applied": float(metrics.get("inner_critic_bernoulli_applied", 0.)),
                                "outer_updates": agent.num_updates})
                return result

            agent.act = observed_act
            with tempfile.TemporaryDirectory(prefix=name + "-") as temporary:
                models = Path(temporary) / "models"
                learner.set_checkpointing(256, str(models), "smoke", save_strat="all")
                learner.learn(total_timesteps=512)
                learner.flush_checkpoints()
                assert learner._global_step == 512 and agent.num_updates >= 2
                assert learner._pretrained and len(actions) >= 10
                assert actions[0]["applied"] == 0.
                expected = float(cfg.inner_critic_bernoulli_p > 0)
                assert all(row["applied"] == expected for row in actions[1:])
                assert len({row["outer_updates"] for row in actions}) > 1
                assert learner._last_train_metrics["compile_fallback"] == 0.
                for step in (256, 512):
                    checkpoint = models / f"smoke_{step}"
                    assert checkpoint.is_file() and Path(str(checkpoint) + ".metadata.json").is_file()
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                agent.load(checkpoint)
                for key, value in agent.model.state_dict().items():
                    torch.testing.assert_close(value.cpu(), saved["model"][key], rtol=0, atol=0)
                case = {"name": name, "passed": True, "checkpoint_bytes": checkpoint.stat().st_size,
                        "checkpoint_roundtrip": True, "checkpoint_sidecars": True,
                        "real_decisions": 512, "optimizer_updates": agent.num_updates,
                        "inner_rounds": rounds,
                        "critic_updates_per_action": expected_critic_steps,
                        "actor_updates_per_action": expected_actor_steps,
                        "inner_actions": len(actions), "transfer_count": sum(row["applied"] for row in actions),
                        "warm_control_seconds": statistics.median(row["seconds"] for row in actions[-5:]),
                        "seconds": time.perf_counter() - started,
                        "compile_fallback": False,
                        "production_overrides": {**overrides, "total_steps": 512, "checkpoint_every": 256},
                        "production_config_sha256": campaign_binding["config_sha256"][name]}
                assert original["checkpoint_every"] == 25000
                write_new(output / (name + ".json"), case)
                cases.append(case)
                print("TRANSFER_TRAINING_GPU_CASE " + json.dumps(case), flush=True)
        finally:
            if learner is not None:
                learner.close()
            del agent, act, observed_act
            env.close()
            del learner
            gc.collect()
            torch.cuda.empty_cache()
    receipt = {"passed": True, "binding": campaign_binding, "runtime": runtime(), "cases": cases,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "estimated_checkpoint_bytes": 20 * sum(row["checkpoint_bytes"] for row in cases)}
    write_new(output / "receipt.json", receipt)
    print("TRANSFER_TRAINING_GPU_GATE_PASSED " + str(output / "receipt.json"), flush=True)


def launch(index, output, gate, manifest_path=MANIFEST):
    manifest_path = resolve_manifest(manifest_path)
    campaign_binding = binding(manifest_path)
    receipt = read(gate)
    assert receipt["passed"] and receipt["binding"] == campaign_binding, "GPU gate does not match this source/configuration"
    assert len(receipt["cases"]) == 6 and all(row["passed"] for row in receipt["cases"])
    current_runtime = runtime()
    for key in ("python", "torch", "cuda", "executable"):
        assert current_runtime[key] == receipt["runtime"][key], f"GPU gate used another {key}"
    name, manifest, run = recipe(index, manifest_path)
    if not os.environ.get("WANDB_API_KEY"):
        import netrc
        auth = netrc.netrc().authenticators("api.wandb.ai")
        assert auth and auth[2], "W&B credential unavailable"
    output.mkdir(parents=True, exist_ok=False)
    launch_data = {"binding": campaign_binding, "config": name, "run_params": run,
                   "gate_receipt": str(gate), "gate_sha256": digest(gate),
                   "runtime": current_runtime, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "slurm": {key: os.environ.get(key) for key in
                             ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST")},
                   "wandb_run_id": os.environ["WANDB_RUN_ID"]}
    write_new(output / "launch.json", launch_data)
    command = [sys.executable, "main.py", "--run", str(manifest_path), "--alg-dir", "configs/dmcontrol/algs",
               "--log-dir", str(output), "--alg-index", str(index), "--trial-index", "0", "--num-runs", "1"]
    print("TRANSFER_TRAINING_LAUNCH " + json.dumps(launch_data), flush=True)
    os.execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--index", type=int, choices=range(6))
    parser.add_argument("--manifest", type=Path, default=MANIFEST,
                        help="Campaign manifest within the pinned checkout (defaults to J6).")
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke(args.output, args.manifest)
    else:
        assert args.gate is not None and args.index is not None
        launch(args.index, args.output, args.gate, args.manifest)


if __name__ == "__main__":
    main()
