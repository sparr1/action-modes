"""Fail-closed receipts and launch metadata for the three-cell auxiliary SAC training-horizon study."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


if not __debug__:
    raise RuntimeError("Campaign validation requires Python assertions; unset PYTHONOPTIMIZE and do not use -O.")


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("configs/dmcontrol/experiments/ambi_aux_return_sac_horizons.json")
HORIZONS = (1, 5, 7)
CASES = tuple(f"ambi_aux_return_sac_clip_target10p5_shared_h{h}" for h in HORIZONS)
BASELINE = "ambi_aux_return_sac_clip_target10p5_shared"

FLAGS = {"aggregate", "outer_update", "sac_online", "sac_target", "aux_online", "aux_target"}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipes(root=ROOT):
    manifest = read_json(root / MANIFEST)
    assert tuple(manifest["configs"]) == CASES, "campaign cell inventory changed"
    assert manifest["trials"] == 1 and manifest["overrides_alg"]["seed"] == 55
    assert "alg_params" not in manifest["overrides_alg"], "partial algorithm override"
    files = {name: root / "configs/dmcontrol/algs" / (name + ".json") for name in CASES}
    baseline = read_json(root / "configs/dmcontrol/algs" / (BASELINE + ".json"))
    for horizon, path in zip(HORIZONS, files.values()):
        config = read_json(path)
        params = config["alg_params"]
        assert params["train_unroll_horizon"] == horizon
        assert params["wandb_group"] == "ambi-aux-return-sac-horizons-20261002"
        comparable = deepcopy(config)
        for key in ("train_unroll_horizon", "wandb_tags", "wandb_group"):
            comparable["alg_params"][key] = baseline["alg_params"][key]
        assert comparable == baseline, "unapproved scientific recipe change"
    return manifest, files


def binding(source_commit, root=ROOT):
    assert len(source_commit) == 40 and all(c in "0123456789abcdef" for c in source_commit)
    _, files = recipes(root)
    return {
        "source_commit": source_commit,
        "manifest_sha256": sha256(root / MANIFEST),
        "config_sha256": {name: sha256(path) for name, path in files.items()},
        "environment_lock_sha256": sha256(root / "environments/dmcontrol/uv.lock"),
    }


def validate_flags(status):
    assert isinstance(status, list) and status, "missing compile observations"
    for flags in status:
        assert set(flags) == FLAGS, "incomplete compile fallback inventory"
        assert all(value is False for value in flags.values()), "CUDA compile fallback"


def validate_case(case, name, expected):
    assert case["config"] == name and case["source_commit"] == expected["source_commit"]
    assert case["config_sha256"] == expected["config_sha256"][name], "gate tested different configuration"
    for key in ("passed", "compile_strict", "checkpoint_roundtrip", "exact_checkpoint_roundtrip",
                "next_update_reproducible", "gradient_routing", "no_return_actor", "no_inner_updates",
                "stochastic_sac_collection"):
        assert case[key] is True, f"gate requirement failed: {key}"
    assert case["cold_production_shape_updates"] == 2
    assert case["continuation_validation_executions"] == 2
    assert len(case["compile_status"]) == 4
    assert case["optimizer_updates"] >= 2 and case["real_decisions"] == 32
    assert case["replay_seed"] == 55 and len(case["replay_sha256"]) == 64
    assert case["checkpoint_size_bytes"] > 0
    assert case["production_overrides"] == {"wandb": False}
    assert case["production_warmup_and_pretraining_executed"] is False
    assert case["detach_representation"] is False
    assert case["train_unroll_horizon"] == HORIZONS[CASES.index(name)]
    norms = case["auxiliary_gradient_l1"]
    assert norms["critic"] > 0
    assert norms["encoder"] > 0
    # At H=1 the current-state auxiliary prediction has no dynamics path.
    assert norms["dynamics"] >= 0
    assert (norms["dynamics"] > 0) is (case["train_unroll_horizon"] > 1)
    validate_flags(case["compile_status"])


def validate_fresh(fresh, expected, name):
    assert fresh["passed"] is True and fresh["source_commit"] == expected["source_commit"]
    assert fresh["config"] == name and fresh["real_decisions"] == 512
    assert fresh["config_sha256"] == expected["config_sha256"][name]
    assert fresh["train_unroll_horizon"] == HORIZONS[CASES.index(name)]
    assert fresh["optimizer_updates"] >= 2 and fresh["checkpoint_size_bytes"] > 0
    assert fresh["checkpoint_roundtrip"] is True and fresh["checkpoint_sidecar"] is True
    assert fresh["production_overrides"] == {
        "wandb": False, "seed_steps": 500, "pretrain_steps": 2,
        "buffer_size": 4096, "total_steps": 512, "checkpoint_every": 256,
    }
    validate_flags(fresh["compile_status"])


def validate_receipt(receipt, source_commit, index, root=ROOT):
    expected = binding(source_commit, root)
    assert 0 <= index < len(CASES)
    assert receipt["schema"] == 1 and receipt["passed"] is True
    assert receipt["binding"] == expected, "receipt does not bind the current commit and all recipes"
    assert receipt["index"] == index and receipt["config"] == CASES[index]
    validate_case(receipt["case"], CASES[index], expected)
    validate_fresh(receipt["fresh_training"], expected, CASES[index])
    return receipt


def runtime():
    import torch

    assert torch.cuda.is_available(), "this campaign requires an allocated CUDA GPU"
    return {
        "python_executable": str(Path(sys.executable).absolute()), "python": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
    }


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def collect_receipt(gate_dir, source_commit, index):
    expected = binding(source_commit)
    receipt = {
        "schema": 1, "passed": True, "binding": expected,
        "index": index, "config": CASES[index],
        "case": read_json(gate_dir / (CASES[index] + ".json")),
        "fresh_training": read_json(gate_dir / (CASES[index] + "-fresh-training.json")),
        "runtime": runtime(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_receipt(receipt, source_commit, index)
    receipt["estimated_checkpoint_bank_bytes"] = 80 * receipt["case"]["checkpoint_size_bytes"]
    path = gate_dir / "receipt.json"
    write_new(path, receipt)
    print("AUX_RETURN_HORIZON_GATE_RECEIPT", path)
    print("Estimated 80-checkpoint bank bytes:", receipt["estimated_checkpoint_bank_bytes"])


def prepare_training(receipt_path, source_commit, index, run_root):
    receipt = validate_receipt(read_json(receipt_path), source_commit, index)
    current_runtime = runtime()
    for key in ("python_executable", "python", "torch", "cuda", "gpu", "compute_capability"):
        assert current_runtime[key] == receipt["runtime"][key], f"gate used another {key}"
    # Credentials are read in place, never copied to the metadata or output.
    if not os.environ.get("WANDB_API_KEY"):
        import netrc

        auth = netrc.netrc().authenticators("api.wandb.ai")
        assert auth and auth[2], "existing W&B credentials required"
    assert 0 <= index < len(CASES)
    name = CASES[index]
    manifest, files = recipes()
    config = read_json(files[name])
    params = config["alg_params"]
    assert params["compile"] and params["compile_strict"]
    assert params["critic_value_mode"] == "single" and params["aux_return_mode"] == "sac"
    assert params["inner_operator"] == "none" and params["inner_actor_source"] == "sac"
    run = {"name": name, **config, **manifest["overrides_alg"]}

    # Resolve through the actual environment and config builder, without allocating
    # a model, touching learner RNG, or starting training/W&B.
    sys.path.insert(0, str(ROOT))
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    env = gym.make(run["env"], **manifest["env_params"])
    try:
        algorithm = object.__new__(AMBITDMPC2)
        algorithm.env = env
        algorithm.run_params = run
        algorithm.custom_params = deepcopy(params)
        algorithm.experiment_params = manifest
        resolved = vars(algorithm._build_cfg({"device": run["device"], **params}))
    finally:
        env.close()
    wandb_id = os.environ["WANDB_RUN_ID"]
    wandb_path = f"{params['wandb_entity']}/{params['wandb_project']}/{wandb_id}"
    metadata = {
        "schema": 1, "binding": binding(source_commit), "config": name, "seed": 55,
        "campaign": os.environ["AMBI_AUX_CAMPAIGN"],
        "created_at": datetime.now(timezone.utc).isoformat(), "runtime": current_runtime,
        "gate_receipt": str(receipt_path), "gate_receipt_sha256": sha256(receipt_path),
        "slurm": {key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ACCOUNT",
            "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
            "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES")},
        "wandb": {"id": wandb_id, "path": wandb_path, "url": f"https://wandb.ai/{params['wandb_entity']}/{params['wandb_project']}/runs/{wandb_id}",
                  "name": f"AMBITDMPC2-{name}-seed55", "group": params["wandb_group"], "mode": "online"},
        "command": [sys.executable, "main.py", "--run", str(MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
                    "--log-dir", str(run_root), "--alg-index", str(index), "--trial-index", "0", "--num-runs", "1"],
        "run_params": run, "experiment_params": manifest, "resolved_config": resolved,
    }
    write_new(run_root / "launch.json", metadata)
    print("AUX_RETURN_HORIZON_LAUNCH", run_root / "launch.json")
    print("W&B:", metadata["wandb"]["url"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("receipt", "train-metadata"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--gate-dir", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if args.mode == "receipt":
        assert args.gate_dir is not None and args.index is not None
        collect_receipt(args.gate_dir, args.source_commit, args.index)
    else:
        assert args.receipt is not None and args.index is not None and args.run_root is not None
        prepare_training(args.receipt, args.source_commit, args.index, args.run_root)


if __name__ == "__main__":
    main()
