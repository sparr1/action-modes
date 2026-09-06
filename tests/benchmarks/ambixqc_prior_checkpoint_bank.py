#!/usr/bin/env python3
"""Guarded AMBI-XQC prior-bank training and its full-size GPU smoke.

Run from the repository root through the Oscar launcher. Smoke preserves the
production architecture and 2,500-decision warmup/pretrain, changing only the
run length, checkpoint cadence, and W&B publication.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STEM = "ambixqc_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare_inputs(mode, run_name):
    """Keep the scientific source fixed and return explicit operational copies."""
    if mode not in {"smoke", "production"}:
        raise ValueError("mode must be smoke or production")
    algorithm = read_json(ROOT / "configs/dmcontrol/algs" / f"{STEM}.json")
    manifest = read_json(ROOT / "configs/dmcontrol/experiments" / f"{STEM}.json")
    params = algorithm["alg_params"]
    expected = {
        "inner_operator": "none", "model_size": 5, "obs": "state",
        "seed_steps": 2500, "pretrain_steps": 2500, "eval_freq": None,
        "compile": False, "compile_strict": False, "mpc": False,
        "xqc_actor_net_arch": [256] * 4, "xqc_critic_net_arch": [512] * 4,
        "inner_rounds": 2, "inner_rollouts_per_round": 32,
        "inner_rollout_horizon": 3, "inner_updates_per_round": 4,
        "inner_batch_size": 64, "inner_replay_capacity": 192,
        "wandb_entity": "rwgao_b-brown-university", "wandb_project": "ambi",
        "xqc_optimizer_backend": "auto",
    }
    if any(params.get(key) != value for key, value in expected.items()):
        raise ValueError("The versioned no-inner scientific configuration changed.")
    if (
        algorithm["alg"] != "AMBIXQC/AMBIXQC"
        or algorithm["seed"] != 55 or algorithm["total_steps"] != 1_500_000
        or manifest["env_params"] != {
            "task": "humanoid-walk", "obs": "state", "render_mode": None,
        }
        or manifest["trials"] != 1 or manifest["configs"] != [STEM]
        or manifest["checkpoint_every"] != 25_000
        or manifest["save_strat"] != ["all"]
        or manifest["save_trials"] != "none"
    ):
        raise ValueError("The versioned no-inner experiment protocol changed.")
    params["wandb_run_name"] = run_name
    if mode == "smoke":
        algorithm["total_steps"] = 3000
        params["wandb"] = False
        params["wandb_mode"] = "disabled"
        manifest["checkpoint_every"] = 1000
    return algorithm, manifest


def assert_finite(value, location="checkpoint"):
    import torch

    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite tensor at {location}")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite scalar at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{location}[{index}]")


def validate_checkpoint_bank(root, *, total_steps, cadence):
    import torch

    records = []
    for sidecar in Path(root).rglob("*.metadata.json"):
        metadata = read_json(sidecar)
        checkpoint = Path(str(sidecar).removesuffix(".metadata.json"))
        step = metadata["checkpoint"]["step"]
        trial = metadata["trial_run_params"]
        if (not checkpoint.is_file() or trial["seed"] != 55
                or trial["total_steps"] != total_steps
                or trial["alg_params"]["inner_operator"] != "none"):
            raise ValueError(f"Invalid checkpoint metadata: {sidecar}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (state["checkpoint_version"] not in {2, 3}
                or state["semantic_signature"]["collection_operator"] != "none"
                or state["inner"]["action_index"] != 0
                or state["reward_normalizer"]["count"] != step):
            raise ValueError(f"Invalid no-inner checkpoint: {checkpoint}")
        assert_finite(state)
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        records.append({"path": str(checkpoint), "step": step,
                        "sha256": digest.hexdigest(), "bytes": checkpoint.stat().st_size,
                        "outer_updates": state["num_updates"]})
    records.sort(key=lambda record: record["step"])
    if [record["step"] for record in records] != list(range(cadence, total_steps + 1, cadence)):
        raise ValueError("Checkpoint bank has missing, duplicate, or unexpected steps.")
    return records


def frozen_smoke(checkpoint, algorithm, manifest):
    """Three real decisions per controller only; this is a loading canary."""
    import torch
    from RL.AMBIXQC import AMBIXQC
    from utils.core import build_env

    results = {}
    for operator in ("none", "xqc"):
        config = deepcopy(algorithm)
        config["alg_params"].update(inner_operator=operator, wandb=False, wandb_mode="disabled")
        env = build_env(config, manifest)
        model = None
        try:
            model = AMBIXQC("AMBIXQC", env, config["alg_params"], config, manifest)
            model.load(checkpoint, frozen_evaluation=True)
            model.reset_for_evaluation(101)
            before = model.agent.frozen_outer_state()
            observation, _ = env.reset(seed=101)
            rewards = []
            decision_update_counts = []
            for _ in range(3):
                action, _ = model.predict(observation, deterministic=True)
                actual_steps = tuple(model.agent.last_inner_metrics[key] for key in (
                    "inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                    "inner_temperature_optimizer_steps",
                ))
                expected_steps = (8, 3, 3) if operator == "xqc" else (0, 0, 0)
                if actual_steps != expected_steps:
                    raise ValueError(f"Frozen XQC solve has incorrect update counts: {actual_steps}")
                decision_update_counts.append(actual_steps)
                observation, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            after = model.agent.frozen_outer_state()

            def equal(left, right):
                if torch.is_tensor(left):
                    return torch.equal(left, right)
                if isinstance(left, dict):
                    return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
                if isinstance(left, (list, tuple)):
                    return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
                return left == right

            if not equal(before, after) or not all(math.isfinite(r) for r in rewards):
                raise ValueError("Frozen-checkpoint smoke mutated outer state or returned non-finite rewards.")
            results[operator] = {"decisions": len(rewards), "return": sum(rewards),
                                 "outer_unchanged": True,
                                 "decision_update_counts": decision_update_counts,
                                 "last_inner_metrics": model.agent.last_inner_metrics}
        finally:
            if model is not None:
                model.flush_checkpoints()
                model.agent.inner_engine.clear_all()
            env.close()
    return results


def run(mode, output, run_name):
    import torch
    import main as training_main
    from RL.AMBIXQC import AMBIXQC

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This full-size training entry point requires one allocated CUDA GPU.")
    algorithm, manifest = prepare_inputs(mode, run_name)
    output = Path(output).resolve()
    if ROOT == output or ROOT in output.parents:
        raise ValueError("Training output must be outside the source checkout.")
    output.mkdir(parents=False, exist_ok=False)
    alg_dir = output / "algs"
    alg_dir.mkdir()
    write_json(alg_dir / f"{STEM}.json", algorithm)
    write_json(output / "experiment.json", manifest)
    models = []
    counters = {"stochastic_prior_actions": 0}
    original_make = AMBIXQC._make_agent

    def forbidden(*args, **kwargs):
        raise AssertionError("No-inner training invoked adaptation or online evaluation.")

    started = time.perf_counter()
    with ExitStack() as guards:
        def make(wrapper, cfg):
            agent = original_make(wrapper, cfg)
            if (tuple(cfg.obs_shape["state"]) != (67,) or cfg.action_dim != 21
                    or cfg.latent_dim != 512 or cfg.inner_operator != "none"):
                raise ValueError("The GPU smoke must use the full Humanoid model.")
            models.append(wrapper)
            original_sample = agent.xqc_controller.sample_action

            def sample(*args, **kwargs):
                if kwargs.get("deterministic") is not False or kwargs.get("noise") is None:
                    raise AssertionError("Prior collection did not use explicit stochastic action noise.")
                counters["stochastic_prior_actions"] += 1
                return original_sample(*args, **kwargs)

            guards.enter_context(patch.object(agent.xqc_controller, "sample_action", sample))
            guards.enter_context(patch.object(agent.xqc_controller, "clone_for_inner", forbidden))
            for name in ("act", "_prepare_action", "_collect_round", "_update_slot"):
                guards.enter_context(patch.object(agent.inner_engine, name, forbidden))
            return agent

        guards.enter_context(patch.object(AMBIXQC, "_make_agent", make))
        guards.enter_context(patch.object(AMBIXQC, "_evaluate_policy", forbidden))
        guards.enter_context(patch.object(sys, "argv", [
            "main.py", "--run", str(output / "experiment.json"),
            "--alg-dir", str(alg_dir), "--log-dir", str(output / "training"),
            "--alg-index", "0", "--trial-index", "0", "--num-runs", "1",
        ]))
        training_main.main()
    if len(models) != 1:
        raise ValueError("The guarded launcher must create exactly one training model.")
    model = models[0]
    total = algorithm["total_steps"]
    workspace = model.agent.xqc_workspace
    optimizer_fused = {
        key: all(group.get("fused") is True for group in getattr(workspace, key).param_groups)
        for key in ("actor_optimizer", "critic_optimizer", "temperature_optimizer")
    }
    if (counters["stochastic_prior_actions"] != total - 2501
            or model.agent.num_updates != total - 1
            or workspace.actor_optimizer_steps != (total - 2) // 3 + 1
            or workspace.temperature_optimizer_steps != (total - 2) // 3 + 1
            or model.agent.inner_engine.action_index != 0
            or model.agent.reward_normalizer.count != total
            or model._wandb_inner_actions != 0 or model._wandb_inner_seconds != 0
            or not all(optimizer_fused.values())):
        raise ValueError("Training violated the no-inner warmup, update, or logging counters.")
    records = validate_checkpoint_bank(
        output / "training", total_steps=total, cadence=manifest["checkpoint_every"]
    )
    result = {"schema": "ambixqc-prior-bank-validation-v1", "mode": mode,
              "total_steps": total, "seed": 55, "outer_updates": model.agent.num_updates,
              **counters, "inner_actions": 0, "all_finite": True,
              "optimizer_fused": optimizer_fused,
              "training_seconds": time.perf_counter() - started,
              "checkpoints": records}
    if mode == "smoke":
        # Release training replay before constructing two frozen canaries.
        model.buffer = None
        result["frozen_evaluation"] = frozen_smoke(records[-1]["path"], algorithm, manifest)
    write_json(output / "validation.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "production"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    run(args.mode, args.output, args.run_name)


if __name__ == "__main__":
    main()
