"""Calibrate PAVE gradient scales on frozen AMBI checkpoints and real transitions.

This command never calls learn/update/optimizer.step, writes a checkpoint,
publishes to W&B, or submits a job. New environment collection is explicit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _jsonable,
    _make_env, _outer_state_digest, _write_output_atomic,
)
from utils.checkpoint_context import load_checkpoint_context
from utils.pave_probe_report import recommend_scales, render_report, summarize_batches


def _positive(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive integer.") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def _cap(value):
    return None if value.lower() in {"all", "null"} else _positive(value)


def _nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Expected a finite nonnegative number.")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="Repeat for early and mature checkpoints; each needs its saved sidecar.")
    parser.add_argument("--metadata", type=Path, action="append",
                        help="Optional sidecar overrides, one per checkpoint in the same order.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bank", type=Path, action="append",
                        help="Saved real-transition NPZ bank: one shared bank, or one per checkpoint.")
    source.add_argument("--collect-episodes", type=_positive,
                        help="Explicitly collect this many stochastic-prior episodes per checkpoint.")
    parser.add_argument("--max-steps", type=_positive, default=500,
                        help="Per-episode collection cap; a forced cutoff is recorded as truncation.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are never replaced.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto (default cpu).")
    parser.add_argument("--ordinary-execution", choices=("eager", "saved"), default="eager",
                        help="Use eager ordinary losses, or the checkpoint's saved compile setting; PAVE stays eager.")
    parser.add_argument("--critics", nargs="+", choices=("primary", "auxiliary"),
                        help="Default: measure all available critics, separately and together.")
    parser.add_argument("--batches", type=_positive, default=32)
    parser.add_argument("--warmup-batches", type=int, default=1,
                        help="Unreported frozen diagnostic passes before timing (default 1).")
    parser.add_argument("--pave-batch-size", type=_cap, default=64, help="Positive cap, or all/null.")
    parser.add_argument("--obs-noise-std", type=_nonnegative, default=.01)
    parser.add_argument("--curvature-margin", type=_nonnegative, default=1.)
    parser.add_argument("--scales", type=_nonnegative, nargs="+", default=[.01, .03, .1, .3, 1., 3., 10.])
    parser.add_argument("--seed", type=int, default=12345, help="Probe/collection seed; saved training seed is preserved for PAVE.")
    parser.add_argument("--td-norm-floor", type=_nonnegative, default=1e-10,
                        help="Ratios with smaller TD gradient norms are marked unreliable.")
    return parser


def _inputs(args):
    count = len(args.checkpoint)
    if args.metadata is not None and len(args.metadata) != count:
        raise ValueError("Supply one --metadata per --checkpoint, or omit all overrides.")
    if args.bank is not None and len(args.bank) not in {1, count}:
        raise ValueError("Supply one shared --bank, or one --bank per checkpoint.")
    if args.warmup_batches < 0:
        raise ValueError("warmup-batches must be nonnegative.")
    if not 0 <= args.seed <= 2**32 - 1 or args.seed + (args.collect_episodes or 1) - 1 > 2**32 - 1:
        raise ValueError("Probe and incremented episode seeds must lie in [0, 2^32-1].")
    if len(set(args.scales)) != len(args.scales) or not args.scales:
        raise ValueError("Candidate scales must be a nonempty list of unique values.")
    if args.critics is not None and len(set(args.critics)) != len(args.critics):
        raise ValueError("Critic selections must be unique.")
    records = []
    for index, path in enumerate(args.checkpoint):
        path = path.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Checkpoint does not exist: {path}")
        context = load_checkpoint_context(path, args.metadata[index] if args.metadata else None)
        config = copy.deepcopy(context.trial_run_params)
        if config["alg"] != "AMBITDMPC2/AMBITDMPC2":
            raise ValueError("PAVE probing requires an AMBITDMPC2 checkpoint.")
        params = config.setdefault("alg_params", {})
        if params.get("obs", "state") != "state" or params.get("critic_value_mode", "single") != "single":
            raise ValueError("PAVE probing supports state observations and single-value critics only.")
        if float(params.get("value_equivalence_loss_coef", 0.)) != 0:
            raise ValueError("PAVE probe v1 requires value_equivalence_loss_coef=0.")
        seed = config.get("seed", params.get("seed"))
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("A valid saved training seed is required in checkpoint metadata.")
        if args.ordinary_execution == "eager":
            params.update(compile=False, compile_strict=False)
        environment = {"id": config["env"], "params": copy.deepcopy(context.experiment_params.get("env_params", {}))}
        wrappers = list(config.get("env_wrappers", []))
        if "env_wrapper" in config:
            wrappers.append(config["env_wrapper"])
        bank_path = (args.bank[0 if len(args.bank) == 1 else index].expanduser().resolve()
                     if args.bank else None)
        if bank_path is not None and not bank_path.is_file():
            raise ValueError(f"Transition bank does not exist: {bank_path}")
        records.append(dict(path=path, context=context, seed=seed, bank_path=bank_path,
                            env_signature={**environment, "wrappers": wrappers},
                            resolved={"algorithm_config": config, "environment": environment,
                                      "selector": "frozen_pave_probe"}))
    return records


def _source_identity():
    root = Path(__file__).resolve().parent
    paths = [Path(__file__), root / "RL/AMBITDMPC2.py", root / "RL/TDMPC2.py",
             root / "evaluate_ambi_checkpoint.py", root / "utils/checkpoint_context.py",
             root / "utils/pave_identity.py", *sorted((root / "RL/tdmpc2_core").rglob("*.py")),
             *sorted((root / "utils").glob("pave_probe*.py")),
             *sorted((root / "domains").rglob("*.py"))]
    hashes = {str(path.relative_to(root)): _file_sha256(path) for path in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": hashes}


def _validate_bank(bank, record, model):
    if bank.provenance["env_signature"] != record["env_signature"]:
        raise ValueError("Transition bank environment/wrappers differ from checkpoint metadata.")
    if bank.observations.shape[1:] != tuple(model.cfg.obs_shape["state"]):
        raise ValueError("Transition bank observation dimensions differ from the checkpoint.")
    if bank.actions.shape[1] != model.cfg.action_dim:
        raise ValueError("Transition bank action dimensions differ from the checkpoint.")


def _initialize_probe_model(record, env, device):
    # The inner evaluator forbids replay mixing because it executes inner
    # adaptation without saved replay. A frozen outer-loss probe never enters
    # that path, so retain the saved inner configuration without that restriction.
    from RL.AMBITDMPC2 import AMBITDMPC2
    config = copy.deepcopy(record["resolved"]["algorithm_config"])
    config["device"] = device
    config.setdefault("alg_params", {}).update(device=device, wandb=False)
    model = AMBITDMPC2("AMBITDMPC2", env, config["alg_params"], config,
                      {"frozen_checkpoint_evaluation": True})
    try:
        model.load(str(record["path"]))
        model.agent.model.eval()
    except BaseException as exc:
        _close_resources(model, primary_error=exc)
        raise
    return model, config


def run_probe(args):
    from RL.tdmpc2_core.pave_probe import probe_batch
    from utils.pave_probe_data import collect_prior_bank, load_probe_bank, sample_probe_windows, save_probe_bank

    records = _inputs(args)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "ambi-pave-gradient-probe", "version": 1, "complete": False,
        "source_identity": _source_identity(),
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform(), "requested_device": args.device,
                    "ordinary_execution": args.ordinary_execution},
        "protocol": {"batches": args.batches, "warmup_batches": args.warmup_batches,
                     "pave_batch_size": args.pave_batch_size, "obs_noise_std": args.obs_noise_std,
                     "curvature_margin": args.curvature_margin, "scales": args.scales,
                     "seed": args.seed, "td_norm_floor": args.td_norm_floor,
                     "sampling": "uniform_with_replacement_over_valid_contiguous_horizon_windows",
                     "pave_sampling": "uniform_without_replacement_over_flattened_rows_when_capped",
                     "optimizer_steps": 0, "target_updates": 0},
        "checkpoints": [],
    }
    try:
        for index, record in enumerate(records):
            env = model = None
            error = None
            try:
                checkpoint_hash = _file_sha256(record["path"])
                env = _make_env(record["resolved"])
                model, run_config = _initialize_probe_model(record, env, args.device)
                before = _outer_state_digest(model)
                collected_seconds = None
                if record["bank_path"] is None:
                    started = time.perf_counter()
                    bank = collect_prior_bank(
                        model, env, seeds=list(range(args.seed, args.seed + args.collect_episodes)),
                        max_steps=args.max_steps, controller_seed=args.seed,
                        provenance={"source_kind": "stochastic_prior_proxy", "env_signature": record["env_signature"],
                                    "source_checkpoint_sha256": checkpoint_hash},
                    )
                    collected_seconds = time.perf_counter() - started
                    bank_path = output / f"checkpoint-{index}-bank.npz"
                    save_probe_bank(bank_path, bank)
                else:
                    bank_path = record["bank_path"]
                    bank = load_probe_bank(bank_path)
                _validate_bank(bank, record, model)
                critics = tuple(args.critics or (["primary", "auxiliary"] if model.agent.aux_return is not None else ["primary"]))
                if "auxiliary" in critics and model.agent.aux_return is None:
                    raise ValueError("Auxiliary probing requires a saved auxiliary return critic.")
                batch_results = []
                for batch_index in range(-args.warmup_batches, args.batches):
                    # Warmup uses a separate deterministic index range. Measured
                    # batches therefore never depend on the requested warmup count.
                    draw_index = batch_index if batch_index >= 0 else 1_000_000 - batch_index
                    draw_seed = (args.seed + 104729 * draw_index) % (2**32)
                    batch = sample_probe_windows(bank, horizon=model.cfg.train_unroll_horizon,
                                                 batch_size=model.cfg.batch_size, seed=draw_seed,
                                                 device=model.agent.device)
                    measured = probe_batch(
                        model.agent, *batch, seed=draw_seed,
                        update_number=int(model.agent.num_updates) + 1 + draw_index,
                        critics=critics, batch_size=args.pave_batch_size,
                        obs_noise_std=args.obs_noise_std, curvature_margin=args.curvature_margin,
                        scales=tuple(args.scales), td_norm_floor=args.td_norm_floor,
                    )
                    if batch_index >= 0:
                        measured["batch_index"] = batch_index
                        measured["window_sampling_seed"] = draw_seed
                        batch_results.append(measured)
                after = _outer_state_digest(model)
                if before != after:
                    raise RuntimeError("Frozen probe changed the checkpoint model/optimizer/temperature/counters.")
                checkpoint_result = {
                    "checkpoint": {"path": str(record["path"]), "sha256": checkpoint_hash,
                                   "metadata_path": str(record["context"].source),
                                   "metadata_sha256": _file_sha256(record["context"].source),
                                   "saved_checkpoint": record["context"].metadata["checkpoint"],
                                   "num_updates": int(model.agent.num_updates)},
                    "saved_config": record["context"].trial_run_params,
                    "resolved_run_config": run_config, "resolved_cfg": _jsonable(vars(model.cfg)),
                    "device": str(model.agent.device),
                    "bank": {"path": str(bank_path), "sha256": _file_sha256(bank_path),
                             "provenance": bank.provenance, "transition_count": len(bank.actions),
                             "collection_seconds": collected_seconds},
                    "frozen_state": {"before_sha256": before, "after_sha256": after, "unchanged": True},
                    "batches": batch_results, "summary": summarize_batches(batch_results),
                }
                result["checkpoints"].append(checkpoint_result)
                _write_output_atomic(output / f"checkpoint-{index}.json",
                                     json.dumps(checkpoint_result, indent=2, allow_nan=False) + "\n", overwrite=False)
            except BaseException as exc:
                error = exc
                raise
            finally:
                _close_resources(model, env, primary_error=error)
        result["recommendations"] = recommend_scales(result["checkpoints"])
        result["complete"] = True
        _write_output_atomic(output / "report.md", render_report(result), overwrite=False)
        _write_output_atomic(output / "results.json", json.dumps(result, indent=2, allow_nan=False) + "\n", overwrite=False)
        return result
    except BaseException as exc:
        _write_output_atomic(output / "failure.json", json.dumps({"complete": False, "error": str(exc),
                              "completed_checkpoints": len(result["checkpoints"])}, indent=2) + "\n", overwrite=False)
        raise


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_probe(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"PAVE probe failed: {exc}", file=sys.stderr)
        return 1
    print(f"PAVE probe saved to {args.output_dir.resolve() / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
