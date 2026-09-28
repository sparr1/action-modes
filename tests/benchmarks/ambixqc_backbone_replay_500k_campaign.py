#!/usr/bin/env python3
"""Guarded six-cell 500k-decision Hydra campaign and full-size CUDA timing gate."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ambixqc_prior_checkpoint_bank as prior_bank

STEM = "ambixqc_humanoid_walk_backbone_replay_500k"
CELLS = tuple(f"{arm}_utd{utd}" for utd in (1, 2)
              for arm in ("baseline", "aux_shared", "aux_detached"))
TOTAL_STEPS = 500_000
REPLAY_CAPACITY = 1_000_000
SMOKE_STEPS = 4000
STEADY_START = 3000
STEADY_STOP = 3900
read_json = prior_bank.read_json
write_json = prior_bank.write_json


def split_cell(cell):
    if cell not in CELLS:
        raise ValueError("Unknown 500k backbone campaign cell.")
    arm, ratio = cell.rsplit("_utd", 1)
    return arm, int(ratio)


def protocol_digest(arm):
    name = f"{STEM}_{arm}"
    source = {
        "algorithm": read_json(ROOT / "configs/dmcontrol/algs" / f"{name}.json"),
        "experiment": read_json(ROOT / "configs/dmcontrol/experiments" / f"{name}.json"),
    }
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def validate_smoke_gate(path, arm, expected_sha):
    gate = read_json(path)
    if (gate.get("schema") != "ambixqc-backbone-replay-500k-smoke-gate-v1"
            or gate.get("passed") is not True or not expected_sha
            or gate.get("source_sha") != expected_sha or set(gate.get("arms", {})) != set(CELLS)):
        raise ValueError("Production requires a complete smoke gate from this exact commit.")
    for selected in CELLS:
        entry = gate["arms"][selected]
        validation_path = Path(entry["validation"])
        if (entry["protocol_sha256"] != protocol_digest(selected)
                or hashlib.sha256(validation_path.read_bytes()).hexdigest() != entry["validation_sha256"]):
            raise ValueError("The smoke gate configuration or validation artifact has changed.")
    return gate["arms"][arm]


def prepare_inputs(mode, arm, run_name):
    if mode not in {"smoke", "production"} or arm not in CELLS:
        raise ValueError("Unknown backbone replay campaign mode or arm.")
    name = f"{STEM}_{arm}"
    algorithm = read_json(ROOT / "configs/dmcontrol/algs" / f"{name}.json")
    manifest = read_json(ROOT / "configs/dmcontrol/experiments" / f"{name}.json")
    params = algorithm["alg_params"]
    variant, ratio = split_cell(arm)
    expected = {
        "inner_operator": "none", "model_size": 5, "obs": "state",
        "seed_steps": 2500, "pretrain_steps": 2500, "utd": 1, "xqc_utd": ratio,
        "replay_capacity": REPLAY_CAPACITY, "buffer_size": REPLAY_CAPACITY, "eval_freq": None,
        "compile": False, "compile_strict": False, "mpc": False,
        "xqc_actor_net_arch": [256] * 4, "xqc_critic_net_arch": [512] * 4,
        "xqc_optimizer_backend": "auto", "inner_reward_normalization": "frozen_real_scale",
        "inner_critic_source": "xqc", "inner_horizon_critic_source": "xqc",
        "inner_critic_target": "entropy_augmented",
        "aux_return_mode": "off" if variant == "baseline" else "xqc",
        "aux_return_detach_representation": variant != "aux_shared",
        "aux_return_critic_coef": .1,
        "wandb_entity": "rwgao_b-brown-university", "wandb_project": "ambi",
        "wandb_group": "ambixqc-humanoid-walk-backbone-replay-500k-utd",
    }
    if any(params.get(key) != value for key, value in expected.items()):
        raise ValueError("The versioned backbone scientific configuration changed.")
    if (algorithm["alg"] != "AMBIXQC/AMBIXQC" or algorithm["seed"] != 55
            or algorithm["total_steps"] != TOTAL_STEPS
            or manifest["env_params"] != {"task": "humanoid-walk", "obs": "state", "render_mode": None}
            or manifest["trials"] != 1 or manifest["configs"] != [name]
            or manifest["checkpoint_every"] != 25_000 or manifest["save_strat"] != ["all"]
            or manifest["save_trials"] != "none" or manifest["save_replay_buffer"] is not True):
        raise ValueError("The versioned backbone experiment protocol changed.")
    params["wandb_run_name"] = run_name
    if mode == "smoke":
        algorithm["total_steps"] = SMOKE_STEPS
        params.update(wandb=False, wandb_mode="disabled")
        manifest["checkpoint_every"] = 1000
    return algorithm, manifest


def validate_checkpoint_bank(root, *, arm, total_steps, cadence):
    import torch
    from utils.replay_archive import load_checkpoint_replay

    records = prior_bank.validate_checkpoint_bank(root, total_steps=total_steps, cadence=cadence)
    variant, ratio = split_cell(arm)
    archive_ids = set()
    chunk_names = set()
    for record in records:
        checkpoint = Path(record["path"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        semantics = state["semantic_signature"]
        signature = semantics["aux_return"]
        enabled = variant != "baseline"
        critic_updates = state["num_updates"] * ratio
        if (semantics["xqc_utd"] != ratio
                or semantics["lr_transition_steps"] != total_steps * ratio
                or state["xqc_workspace"]["update_step"] != critic_updates):
            raise ValueError("Checkpoint XQC ratio, schedule, or counters are invalid.")
        if (signature["mode"] != ("xqc" if enabled else "off")
                or ("aux_return" in state) != enabled):
            raise ValueError("Checkpoint auxiliary learner does not match its campaign arm.")
        if enabled and (signature["detach_representation"] != (variant == "aux_detached")
                        or state["aux_return"]["update_step"] != critic_updates):
            raise ValueError("Checkpoint auxiliary representation or update counters are invalid.")
        view = load_checkpoint_replay(checkpoint)
        metadata = view.metadata
        archive_ids.add(metadata["archive_id"])
        chunk_names.update(chunk["path"] for chunk in metadata["chunks"])
        # Humanoid's fixed 500-step episodes insert 501 rows, including the
        # initial observation. Checkpoint replay is clipped to resident rows.
        completed = record["step"] // 500
        expected_rows = min(completed * 501, REPLAY_CAPACITY)
        if (view.num_rows != expected_rows or metadata["total_episodes"] != completed
                or metadata["capacity"] != REPLAY_CAPACITY):
            raise ValueError("Checkpoint replay includes unfinished or missing episodes.")
        rng = torch.get_rng_state().clone()
        sample = view.sample_sequences(batch_size=4, horizon=5, seed=101)
        repeated = view.sample_sequences(batch_size=4, horizon=5, seed=101)
        if (sample[0].shape != (6, 4, 67) or sample[1].shape != (5, 4, 21)
                or sample[2].shape != (5, 4, 1) or sample[4] is not None
                or not torch.equal(rng, torch.get_rng_state())):
            raise ValueError("Checkpoint replay sampling has an invalid layout or changed global RNG.")
        for first, second in zip(sample[:4], repeated[:4]):
            if first.device.type != "cpu" or not torch.isfinite(first).all() or not torch.equal(first, second):
                raise ValueError("Checkpoint replay samples must be finite, seeded CPU tensors.")
        record.update(xqc_updates=critic_updates, replay_rows=view.num_rows, replay_transitions=view.num_transitions,
                      replay_episodes=view.num_episodes, replay_manifest=metadata,
                      replay_hashes_verified=True)
    if len(archive_ids) != 1:
        raise ValueError("One training arm must use one shared replay archive.")
    return records


def validate_live_replay(buffer, checkpoint):
    """Compare archived raw rows with the actual final training replay."""
    import torch
    from utils.replay_archive import load_checkpoint_replay

    view = load_checkpoint_replay(checkpoint)
    if view.num_rows != buffer.size:
        raise ValueError("Final archive size differs from training replay.")
    storage = buffer._buffer._storage._storage
    cursor = int(buffer._buffer._writer._cursor)
    offset = 0
    for fragment in view.fragments:
        fields = fragment["fields"]
        rows = len(fields["obs"])
        indices = torch.arange(offset, offset + rows)
        if buffer.size == buffer.capacity:
            indices = (indices + cursor) % buffer.capacity
        for key, actual in fields.items():
            torch.testing.assert_close(actual, storage[key][indices].detach().cpu(), rtol=0, atol=0, equal_nan=True)
        offset += rows
    return True


def timing_record(started, measurements):
    if set(measurements) != {STEADY_START, STEADY_STOP}:
        raise ValueError("The CUDA smoke did not complete its measured steady window.")
    warmup = measurements[STEADY_START] - started
    steady = measurements[STEADY_STOP] - measurements[STEADY_START]
    if not math.isfinite(steady) or steady <= 0 or warmup <= 0:
        raise ValueError("Invalid CUDA smoke timing.")
    per_decision = steady / (STEADY_STOP - STEADY_START)
    return {
        "warmup_and_startup_seconds": warmup,
        "warmup_end_step": STEADY_START, "steady_end_step": STEADY_STOP,
        "steady_decisions": STEADY_STOP - STEADY_START,
        "steady_seconds": steady, "steady_seconds_per_decision": per_decision,
        "estimated_500k_hours": (warmup + (TOTAL_STEPS - STEADY_START) * per_decision) / 3600,
        "estimate_excludes_periodic_checkpoint_cost": True,
        "measurement": "synchronized wall clock after pretraining; excludes checkpoint boundaries",
    }


def run(mode, arm, output, run_name, smoke_gate=None):
    import torch
    import main as training_main
    from RL.AMBIXQC import AMBIXQC

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This full-size campaign requires one allocated CUDA GPU.")
    algorithm, manifest = prepare_inputs(mode, arm, run_name)
    variant, ratio = split_cell(arm)
    if mode == "production":
        if smoke_gate is None:
            raise ValueError("Production requires --smoke-gate from the same commit.")
        entry = validate_smoke_gate(smoke_gate, arm, os.environ.get("EXPECTED_ACTION_MODES_SHA"))
        if shutil.disk_usage(Path(output).parent).free < entry["required_free_bytes"]:
            raise ValueError("Insufficient free storage for measured checkpoint and replay budget.")
    output = Path(output).resolve()
    if ROOT == output or ROOT in output.parents:
        raise ValueError("Training output must be outside the source checkout.")
    output.mkdir(parents=False, exist_ok=False)
    alg_dir = output / "algs"
    alg_dir.mkdir()
    write_json(alg_dir / f"{STEM}_{arm}.json", algorithm)
    write_json(output / "experiment.json", manifest)
    models, measurements = [], {}
    counters = {"stochastic_prior_actions": 0}
    original_make = AMBIXQC._make_agent

    def forbidden(*args, **kwargs):
        raise AssertionError("Backbone-only training invoked adaptation or online evaluation.")

    started = time.perf_counter()
    with ExitStack() as guards:
        def make(wrapper, cfg):
            agent = original_make(wrapper, cfg)
            if (tuple(cfg.obs_shape["state"]) != (67,) or cfg.action_dim != 21
                    or cfg.latent_dim != 512 or cfg.inner_operator != "none"
                    or cfg.xqc_utd != ratio or cfg.replay_capacity != REPLAY_CAPACITY
                    or cfg.xqc_lr_transition_steps != algorithm["total_steps"] * ratio):
                raise ValueError("The campaign must use the full Humanoid backbone.")
            models.append(wrapper)
            original_sample = agent.xqc_controller.sample_action

            def sample(*args, **kwargs):
                if kwargs.get("deterministic") is not False or kwargs.get("noise") is None:
                    raise AssertionError("Collection must use explicit stochastic prior noise.")
                counters["stochastic_prior_actions"] += 1
                return original_sample(*args, **kwargs)

            guards.enter_context(patch.object(agent.xqc_controller, "sample_action", sample))
            guards.enter_context(patch.object(agent.xqc_controller, "clone_for_inner", forbidden))
            for name in ("act", "_prepare_action", "_collect_round", "_update_slot"):
                guards.enter_context(patch.object(agent.inner_engine, name, forbidden))
            if mode == "smoke":
                original_checkpoint = wrapper._maybe_checkpoint

                def checkpoint():
                    result = original_checkpoint()
                    if wrapper._global_step in {STEADY_START, STEADY_STOP}:
                        wrapper.flush_checkpoints()
                        torch.cuda.synchronize()
                        measurements[wrapper._global_step] = time.perf_counter()
                    return result

                guards.enter_context(patch.object(wrapper, "_maybe_checkpoint", checkpoint))
            return agent

        guards.enter_context(patch.object(AMBIXQC, "_make_agent", make))
        guards.enter_context(patch.object(AMBIXQC, "_evaluate_policy", forbidden))
        guards.enter_context(patch.object(sys, "argv", [
            "main.py", "--run", str(output / "experiment.json"), "--alg-dir", str(alg_dir),
            "--log-dir", str(output / "training"), "--alg-index", "0", "--trial-index", "0", "--num-runs", "1",
        ]))
        training_main.main()
    training_seconds = time.perf_counter() - started
    if len(models) != 1:
        raise ValueError("The campaign must create exactly one training model per arm.")
    model, total = models[0], algorithm["total_steps"]
    workspace = model.agent.xqc_workspace
    expected_updates = (total - 1) * ratio
    expected_actor_updates = (expected_updates - 1) // 3 + 1
    fused = {name: all(group.get("fused") is True for group in getattr(workspace, name).param_groups)
             for name in ("actor_optimizer", "critic_optimizer", "temperature_optimizer")}
    if model.agent.aux_return is not None:
        fused["auxiliary_critic_optimizer"] = all(group.get("fused") is True
                                                 for group in model.agent.aux_return.critic_optimizer.param_groups)
        if model.agent.aux_return.update_step != expected_updates:
            raise ValueError("Auxiliary critic update dose differs from the primary critic.")
    if (counters["stochastic_prior_actions"] != total - 2501
            or model.agent.num_updates != total - 1
            or workspace.update_step != expected_updates
            or workspace.actor_optimizer_steps != expected_actor_updates
            or workspace.temperature_optimizer_steps != expected_actor_updates
            or model.buffer.capacity != REPLAY_CAPACITY
            or model.agent.inner_engine.action_index != 0
            or model.agent.reward_normalizer.count != total
            or model._wandb_inner_actions != 0 or model._wandb_inner_seconds != 0
            or not all(fused.values())):
        raise ValueError("Training violated the no-inner update, normalization, or logging counters.")
    records = validate_checkpoint_bank(
        output / "training", arm=arm, total_steps=total, cadence=manifest["checkpoint_every"]
    )
    replay_matches = validate_live_replay(model.buffer, records[-1]["path"])
    result = {
        "schema": "ambixqc-backbone-replay-500k-validation-v1", "mode": mode, "arm": arm,
        "source_sha": os.environ.get("EXPECTED_ACTION_MODES_SHA"),
        "total_steps": total, "seed": 55, "outer_updates": model.agent.num_updates,
        "xqc_utd": ratio, "xqc_updates": workspace.update_step,
        "actor_updates": workspace.actor_optimizer_steps,
        "auxiliary_updates": (model.agent.aux_return.update_step if model.agent.aux_return is not None else 0),
        "replay_capacity": model.buffer.capacity,
        "lr_transition_steps": model.agent.cfg.xqc_lr_transition_steps,
        **counters, "inner_actions": 0, "all_finite": True, "optimizer_fused": fused,
        "training_seconds": training_seconds, "checkpoints": records,
        "final_raw_replay_matches_training": replay_matches,
        "gpu": torch.cuda.get_device_name(0),
        "algorithm_sha256": hashlib.sha256(json.dumps(algorithm, sort_keys=True).encode()).hexdigest(),
        "protocol_sha256": protocol_digest(arm),
    }
    if mode == "smoke":
        result["timing"] = timing_record(started, measurements)
        result["storage"] = storage_estimate(records)
        model.buffer = None
        result["frozen_evaluation"] = prior_bank.frozen_smoke(records[-1]["path"], algorithm, manifest)
    write_json(output / "validation.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "checkpoints"}, sort_keys=True, allow_nan=False), flush=True)
    return result


def storage_estimate(records):
    """Conservative production budget derived from real full-size smoke files."""
    checkpoint_bytes = max(record["bytes"] for record in records)
    # Twenty all-strategy checkpoints; experience is archived once. Humanoid
    # rows have 67 obs + 21 actions + reward + flag + episode int64 (~368 B).
    replay_bytes = (TOTAL_STEPS // 500) * 501 * 384
    estimated_bytes = (TOTAL_STEPS // 25_000) * checkpoint_bytes + replay_bytes
    return {"largest_checkpoint_bytes": checkpoint_bytes,
            "estimated_production_bytes": estimated_bytes,
            "required_free_bytes": math.ceil(estimated_bytes * 1.5) + 2 * 1024 ** 3}


def summarize_smoke(output):
    output = Path(output).resolve()
    paths = {}
    for arm in CELLS:
        matches = sorted(output.glob(f"smoke-{arm}-seed55-job*-task*/run-{arm}/validation.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one completed smoke validation for {arm}; found {len(matches)}.")
        if not (matches[0].parent.parent / "PASS").is_file():
            raise ValueError(f"Smoke job has not passed for {arm}.")
        paths[arm] = matches[0]
    records = {arm: read_json(paths[arm]) for arm in CELLS}
    for arm, result in records.items():
        if (result["schema"] != "ambixqc-backbone-replay-500k-validation-v1" or result["mode"] != "smoke"
                or result["arm"] != arm or result["total_steps"] != SMOKE_STEPS
                or result["xqc_utd"] != split_cell(arm)[1]
                or result["xqc_updates"] != (SMOKE_STEPS - 1) * split_cell(arm)[1]
                or result["replay_capacity"] != REPLAY_CAPACITY
                or result["lr_transition_steps"] != SMOKE_STEPS * split_cell(arm)[1]
                or result["protocol_sha256"] != protocol_digest(arm)
                or not result["all_finite"] or not result["final_raw_replay_matches_training"]
                or not all(result["optimizer_fused"].values()) or len(result["checkpoints"]) != 4):
            raise ValueError(f"Incomplete smoke validation for {arm}.")
    if len({result["source_sha"] for result in records.values()}) != 1:
        raise ValueError("Smoke arms came from different commits.")
    baseline = records["baseline_utd1"]["timing"]["steady_seconds_per_decision"]
    summary = {
        "schema": "ambixqc-backbone-replay-500k-smoke-gate-v1", "passed": True,
        "source_sha": records["baseline_utd1"]["source_sha"],
        "required_campaign_free_bytes": sum(result["storage"]["required_free_bytes"] for result in records.values()),
        "arms": {arm: {**result["timing"], **result["storage"], "steady_ratio_to_baseline": result["timing"]["steady_seconds_per_decision"] / baseline,
                       "validation": str(paths[arm]),
                       "validation_sha256": hashlib.sha256(paths[arm].read_bytes()).hexdigest(),
                       "protocol_sha256": result["protocol_sha256"]}
                 for arm, result in records.items()},
    }
    write_json(output / "smoke-gate.json", summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("smoke", "production", "summarize"))
    parser.add_argument("--arm", choices=CELLS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--smoke-gate", type=Path)
    args = parser.parse_args()
    if args.mode == "summarize":
        summarize_smoke(args.output)
    else:
        if args.arm is None or args.run_name is None:
            parser.error("--arm and --run-name are required for training")
        run(args.mode, args.arm, args.output, args.run_name, smoke_gate=args.smoke_gate)


if __name__ == "__main__":
    main()
