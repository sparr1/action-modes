"""Run one setting from the Humanoid Walk frozen-checkpoint ablation."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.checkpoint_context import load_checkpoint_context
from utils.evaluation_episodes import consecutive_protocol, validate_consecutive_episodes
from utils.eval_series import _atomic_json, _file_digest


MATRIX = ROOT / "configs/research/ambi_humanoid_walk_posttanh_1m_ablations.json"
GRAPH = "frozen_mean_cuda_graph_v1"
STEP = 1_000_000


def validate_checkpoint(checkpoint, checkpoint_sha256, metadata_sha256):
    context = load_checkpoint_context(checkpoint)
    if _file_digest(checkpoint) != checkpoint_sha256 or _file_digest(context.source) != metadata_sha256:
        raise ValueError("Checkpoint or sidecar SHA-256 differs from the verified source.")
    run = context.trial_run_params
    env = context.experiment_params["env_params"]
    parameters = run["alg_params"]
    if (context.metadata["checkpoint"]["step"] != STEP
            or run["name"] != "ambi_inner_entropy_posttanh_outer_adapt"
            or run["seed"] != 55
            or run["env"] != "DMControl-v0"
            or env.get("task") != "humanoid-walk" or env.get("obs", "state") != "state"
            or parameters.get("train_unroll_horizon") != 3):
        raise ValueError("Expected the seed-55 post-tanh sampled Humanoid Walk state checkpoint at 1M.")
    return context


def expected_workload(parameters):
    if parameters.get("inner_operator") == "none":
        return {
            "inner_rounds": 0, "inner_model_steps": 0, "inner_critic_optimizer_steps": 0,
            "inner_actor_optimizer_steps": 0, "inner_compile_fallback": 0,
        }
    rounds = parameters["inner_rounds"]
    return {
        "inner_rounds": rounds,
        "inner_model_steps": rounds * parameters["inner_rollout_horizon"] * parameters["inner_rollouts_per_round"],
        "inner_critic_optimizer_steps": rounds * parameters["inner_critic_updates_per_round"],
        "inner_actor_optimizer_steps": rounds * parameters["inner_actor_updates_per_round"],
        "inner_cuda_graph_used": 1,
        "inner_compile_fallback": 0,
    }


def validate_result(payload, resolved, checkpoint_sha256, sampling, max_steps):
    result, = payload["results"]
    parameters = resolved["algorithm_config"]["alg_params"]
    expects_graph = parameters["inner_operator"] == "sac"
    if (payload["checkpoint_sha256"] != checkpoint_sha256
            or payload["matrix_sha256"] != _file_digest(MATRIX)
            or not payload["deterministic_execution"]
            or result["selector"] != resolved["selector"]
            or result["execution_action"] != "mean"
            or not result["outer_state_unchanged"]
            or result["outer_updates_before"] != result["outer_updates_after"]):
        raise ValueError("Frozen ablation source, setting, execution, or outer state did not match.")
    if (result["controller_seed"] != resolved["evaluation"]["controller_seed"]
            or result["episode_sampling"] != sampling):
        raise ValueError("Episode stream differs from the requested consecutive protocol.")
    episodes = result["episodes"]
    validate_consecutive_episodes(episodes, sampling, result["controller_seed"])
    if any(episode["length"] != max_steps or not math.isfinite(episode["return"]) for episode in episodes):
        raise ValueError("Missing decisions or non-finite episode return.")
    decisions = max_steps * sampling["num_episodes"]
    if expects_graph:
        config = result["resolved_config"]
        if (result.get("execution_backend") != GRAPH or result.get("graph_replays") != decisions
                or config["critic_ensemble_backend"] != "vmap"
                or not config["compile"] or not config["compile_strict"]
                or config["inner_replay_capacity"] != parameters["inner_replay_capacity"]
                or config["inner_solve_interval"] != 1):
            raise ValueError("Expected one strict compiled vmap graph replay per decision with action-local replay.")
    elif result.get("graph_replays", 0) or result["resolved_config"]["inner_operator"] != "none":
        raise ValueError("J0 must execute only the frozen policy prior.")
    workload = expected_workload(parameters)
    for episode in episodes:
        metrics = episode["model_metrics"]
        for key, expected in workload.items():
            if metrics.get(key) != expected:
                raise ValueError(f"Unexpected workload in episode {episode['episode_index']} for {key}: "
                                 f"{metrics.get(key)!r}, expected {expected}.")


def main(argv=None):
    matrix = load_preset_matrix(MATRIX)
    settings = list(matrix["comparisons"]["ablation"]["variants"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-settings", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Verify checkpoint provenance without CUDA.")
    parser.add_argument("--setting", choices=settings)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--metadata-sha256")
    parser.add_argument("--gpu-type", help="Exact torch.cuda.get_device_name() shared by all settings.")
    parser.add_argument("--output", type=Path, help="A new, setting-owned output directory.")
    parser.add_argument("--smoke", action="store_true", help="Run two 8-decision episodes instead of the paper protocol.")
    args = parser.parse_args(argv)
    if args.list_settings:
        print("\n".join(settings))
        return
    if not all((args.checkpoint, args.checkpoint_sha256, args.metadata_sha256)):
        parser.error("Provide --checkpoint, --checkpoint-sha256 and --metadata-sha256.")
    context = validate_checkpoint(args.checkpoint, args.checkpoint_sha256, args.metadata_sha256)
    if args.preflight:
        for setting in settings:
            resolve_preset(MATRIX, f"ablation/{setting}", matrix=matrix, checkpoint_context=context)
        print(json.dumps({"checkpoint_step": STEP, "settings": settings, "source_run": matrix["source_run"]}))
        return
    if not all((args.setting, args.gpu_type, args.output)):
        parser.error("Provide --setting, --gpu-type and --output for evaluation.")
    if os.environ.get("WANDB_MODE") != "disabled":
        raise ValueError("Set WANDB_MODE=disabled for the ablation workers.")

    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix

    if not torch.cuda.is_available() or torch.cuda.get_device_name() != args.gpu_type:
        raise ValueError(f"Expected a CUDA device of type {args.gpu_type!r}.")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    selector = f"ablation/{args.setting}"
    resolved = resolve_preset(MATRIX, selector, matrix=matrix, checkpoint_context=context)
    evaluation = resolved["evaluation"]
    max_steps = 8 if args.smoke else evaluation["max_steps"]
    count = 2 if args.smoke else evaluation["num_episodes"]
    sampling = consecutive_protocol({**evaluation, "num_episodes": count})
    args.output.mkdir(parents=True, exist_ok=False)
    payload = evaluate_matrix(
        MATRIX, args.checkpoint, selectors=[selector], num_episodes=count, max_steps=max_steps,
        device="cuda", bundle_dir=args.output / "bundle", source_run=matrix["source_run"], stage_results=False,
    )
    validate_result(payload, resolved, args.checkpoint_sha256, sampling, max_steps)
    _atomic_json(args.output / "results.json", payload)
    receipt = {
        "setting": args.setting, "source_run": matrix["source_run"], "checkpoint_step": STEP,
        "checkpoint_sha256": args.checkpoint_sha256, "metadata_sha256": args.metadata_sha256,
        "matrix_sha256": payload["matrix_sha256"], "gpu": torch.cuda.get_device_name(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "smoke": args.smoke, "workload": expected_workload(resolved["algorithm_config"]["alg_params"]),
        "episodes": count, "decisions": max_steps * count, "return": payload["results"][0]["return"],
    }
    _atomic_json(args.output / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
