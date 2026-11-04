"""Bounded CUDA diagnostic for arithmetic changes in full-budget teachers.

Run through the scheduler with the locked interpreter::

    WANDB_MODE=disabled python -m tests.diagnose_ambi_teacher_precision \
        --output /scratch/teacher-precision.json

This reports differences without relaxing or replacing the correctness gates.
All variants use identical initial parameters, encoded roots, and root seeds.
Only one CUDA solver is live at a time; two reference snapshots remain on CPU.
"""

import argparse
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from unittest.mock import patch

os.environ["WANDB_MODE"] = "disabled"

import torch

from RL.tdmpc2_core import batched_teacher
from RL.tdmpc2_core.common.math import diagonal_gaussian_reverse_kl
from RL.tdmpc2_core.reanalyze import _preserve_global_rng
from tests.test_ambi_batched_reference_recipe import (
    _assert_outer_unchanged, _reference_agent, _snapshot_outer,
)
from tests.test_ambi_reanalyze import _assert_globals, _globals


_RTOL = 5e-4
_ATOL = 1e-5
_FEATURES = frozenset({"selective_q", "actor_cache", "fused_updates"})
_ROOT_SEEDS = [173, 174, 175, 176]


def _cpu_tensor(value):
    return value.detach().cpu().clone()


def _snapshot_core(core):
    groups = {
        name: {key: _cpu_tensor(value) for key, value in getattr(core, name).items()}
        for name in ("actor", "critic", "target", "prior_actor", "prior_critic")
    }
    groups["replay"] = {"packed_valid_rows": _cpu_tensor(core.replay[:, :core.size])}
    groups["scalars"] = {
        name: _cpu_tensor(getattr(core, name))
        for name in ("scale", "log_alpha", "alpha_fixed")
    }
    for name, optimizer in (("actor", core.actor_optim), ("critic", core.critic_optim),
                            ("temperature", core.temperature_optim)):
        if optimizer is None:
            continue
        groups[f"adam_{name}"] = {
            f"parameter_{index}.{key}": _cpu_tensor(value)
            for index, state in enumerate(optimizer.state.values())
            for key, value in state.items()
        }
    return groups


def _tensor_difference(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
    actual, expected = actual.double(), expected.double()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    invalid = int((~finite).sum())
    actual, expected = actual[finite], expected[finite]
    difference = (actual - expected).abs()
    allowance = _ATOL + _RTOL * expected.abs()
    count = difference.numel()
    return {
        "elements": count + invalid,
        "nonfinite_elements": invalid,
        "max_abs_error": difference.max().item() if count else None,
        "max_normalized_error": (difference / allowance).max().item() if count else None,
        "violating_elements": int((difference > allowance).sum()),
        "squared_error_sum": difference.square().sum().item(),
        "reference_squared_sum": expected.square().sum().item(),
    }


def _group_difference(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("State tensor names differ between arithmetic variants.")
    tensors = {key: _tensor_difference(value, expected[key]) for key, value in actual.items()}
    elements = sum(value["elements"] for value in tensors.values())
    squared_error = sum(value["squared_error_sum"] for value in tensors.values())
    reference_squared = sum(value["reference_squared_sum"] for value in tensors.values())
    worst = max(tensors, key=lambda key: tensors[key]["max_normalized_error"] or 0.) if tensors else None
    return {
        "elements": elements,
        "nonfinite_elements": sum(value["nonfinite_elements"] for value in tensors.values()),
        "max_abs_error": max((value["max_abs_error"] or 0. for value in tensors.values()), default=0.),
        "max_normalized_error": max((value["max_normalized_error"] or 0. for value in tensors.values()), default=0.),
        "violating_elements": sum(value["violating_elements"] for value in tensors.values()),
        "rms_abs_error": math.sqrt(squared_error / max(elements, 1)),
        "relative_l2_error": math.sqrt(squared_error / max(reference_squared, 1e-300)),
        "worst_normalized_tensor": worst,
        "tensors": tensors,
    }


def _compare(actual, expected):
    teachers = {
        key: _tensor_difference(actual[key], expected[key])
        for key in ("mean", "log_std")
    }
    actual_policy = actual["mean"].double(), actual["log_std"].double()
    expected_policy = expected["mean"].double(), expected["log_std"].double()
    kl = {}
    for direction, current, teacher in (
        ("actual_to_reference", actual_policy, expected_policy),
        ("reference_to_actual", expected_policy, actual_policy),
    ):
        value = diagonal_gaussian_reverse_kl(*current, *teacher) / 21
        valid = bool(torch.isfinite(value).all())
        kl[direction] = {
            "finite": valid,
            "max_kl_per_dimension": value.max().item() if valid else None,
            "mean_kl_per_dimension": value.mean().item() if valid else None,
            "per_root_kl_per_dimension": value.flatten().tolist() if valid else None,
        }
    return {
        "teachers": teachers,
        "gaussian_kl": kl,
        "work_counts_equal": torch.equal(actual["counts"], expected["counts"]),
        "work_counts": actual["counts"].tolist(),
        "state": {name: _group_difference(group, expected["state"][name])
                  for name, group in actual["state"].items()},
    }


def _comparison_summary(comparison):
    return {
        "teacher_max_abs_error": {key: value["max_abs_error"] for key, value in comparison["teachers"].items()},
        "teacher_max_normalized_error": {key: value["max_normalized_error"] for key, value in comparison["teachers"].items()},
        "max_kl_per_dimension": {key: value["max_kl_per_dimension"] for key, value in comparison["gaussian_kl"].items()},
        "state_max_normalized_error": {key: value["max_normalized_error"] for key, value in comparison["state"].items()},
    }


def _initial_inputs():
    # Match the high-precision gate's seed and initialization order. Encode only
    # once, so the high/highest comparison isolates teacher arithmetic rather
    # than including an encoder precision difference.
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(193)
    initial = _reference_agent(False)
    initial_state = {key: _cpu_tensor(value) for key, value in initial.state_dict().items()}
    discarded = _reference_agent(True)
    del discarded
    with torch.no_grad():
        observations = torch.randn(4, 67, device="cuda")
        roots = initial.model.encode(observations).detach().cpu().clone()
    return initial_state, roots


def _run_variant(initial_state, roots_cpu, precision, features):
    torch._dynamo.reset()
    torch.set_float32_matmul_precision(precision)
    agent = _reference_agent(bool(features))
    agent.load_state_dict(initial_state)
    roots = roots_cpu.to("cuda")
    before = _snapshot_outer(agent)
    solver = batched_teacher.BatchedTeacherSolver(agent)
    host_rng, cuda_rng = _globals(), torch.cuda.get_rng_state().clone()
    # The standalone diagnostic selects features without extending production
    # configuration or changing its checkpoint specification.
    factory = partial(batched_teacher.BatchedTeacherCore, optimization_features=features)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with patch.object(batched_teacher, "BatchedTeacherCore", factory):
        output = solver.solve(roots, _ROOT_SEEDS)
    elapsed = time.perf_counter() - started
    _assert_globals(host_rng)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    _assert_outer_unchanged(agent, before)
    counts = _cpu_tensor(output[2])
    torch.testing.assert_close(counts, torch.tensor([12288., 128., 512.]), rtol=0, atol=0)
    return {
        "mean": _cpu_tensor(output[0]), "log_std": _cpu_tensor(output[1]),
        "counts": counts, "state": _snapshot_core(solver.core),
        "metadata": {
            "precision": precision, "optimization_features": sorted(features),
            "setup_and_solve_seconds": elapsed, "metrics": output[3],
            "outer_state_and_rng_preserved": True,
        },
    }


def _save(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def diagnose(output):
    if not torch.cuda.is_available():
        raise RuntimeError("This full-budget diagnostic requires a scheduled CUDA GPU.")
    if output.exists():
        raise FileExistsError(f"Choose a fresh diagnostic output: {output}")
    previous_precision = torch.get_float32_matmul_precision()
    previous_threads = torch.get_num_threads()
    cache_size = torch._dynamo.config.cache_size_limit
    accumulated_size = torch._dynamo.config.accumulated_cache_size_limit
    if cache_size != 8:
        raise RuntimeError(f"Diagnostic requires production cache_size_limit=8; got {cache_size}.")
    result = {
        "status": "running", "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "host": platform.node(), "cuda": torch.version.cuda,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_tree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip(),
        "scope": "Numerical diagnostic only; setup+solve time is not a steady-state benchmark.",
        "rtol": _RTOL, "atol": _ATOL,
        "normalized_error_definition": "abs(actual-reference)/(atol+rtol*abs(reference)); >1 violates existing gate",
        "compile_cache_size_limit": cache_size,
        "compile_accumulated_cache_size_limit": accumulated_size,
        "root_seeds": _ROOT_SEEDS, "roots": 4,
        "wandb": False, "checkpoints": False,
        "variants": {}, "comparisons": {},
    }
    _save(result, output)
    try:
        torch.set_num_threads(1)
        with _preserve_global_rng():
            initial_state, roots = _initial_inputs()
            gc.collect()
            torch.cuda.empty_cache()
            references = {}
            variants = (
                ("reference_high", "high", frozenset()),
                ("reference_highest", "highest", frozenset()),
                ("selective_q_high", "high", frozenset({"selective_q"})),
                ("actor_cache_high", "high", frozenset({"actor_cache"})),
                ("fused_updates_high", "high", frozenset({"fused_updates"})),
                ("all_high", "high", _FEATURES),
                ("all_highest", "highest", _FEATURES),
            )
            for name, precision, features in variants:
                print(json.dumps({"starting_variant": name}), flush=True)
                current = _run_variant(initial_state, roots, precision, features)
                result["variants"][name] = current["metadata"]
                result["variants"][name]["work_counts"] = current["counts"].tolist()
                if name.startswith("reference_"):
                    references[name] = current
                if name != "reference_high":
                    result["comparisons"][f"{name}_vs_reference_high"] = _compare(current, references["reference_high"])
                if name == "reference_highest":
                    result["comparisons"]["reference_high_vs_reference_highest"] = _compare(
                        references["reference_high"], current,
                    )
                elif name in {"all_high", "all_highest"}:
                    result["comparisons"][f"{name}_vs_reference_highest"] = _compare(current, references["reference_highest"])
                _save(result, output)
                summaries = {key: _comparison_summary(value) for key, value in result["comparisons"].items()
                             if key.startswith(name + "_vs_")}
                print(json.dumps({"completed_variant": name, **current["metadata"],
                                  "comparisons": summaries}, allow_nan=False), flush=True)
                del current
                torch._dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()
        result["status"] = "complete"
        _save(result, output)
        return result
    except BaseException:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        _save(result, output)
        raise
    finally:
        torch._dynamo.reset()
        torch._dynamo.config.cache_size_limit = cache_size
        torch._dynamo.config.accumulated_cache_size_limit = accumulated_size
        torch.set_float32_matmul_precision(previous_precision)
        torch.set_num_threads(previous_threads)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    diagnose(arguments.output)
