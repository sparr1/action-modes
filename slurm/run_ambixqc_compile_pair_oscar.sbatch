#!/usr/bin/env bash
#SBATCH --job-name=ambixqc-compile-pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/ambixqc-compile-pair-%j.out
#SBATCH --error=/tmp/ambixqc-compile-pair-%j.err

set -Eeuo pipefail
umask 077

# Submit through submit_ambixqc_compile_pair_oscar.sh. All repetitions and both
# canaries run in one allocation so they see the same physical L40S, driver,
# checkout, locked environment, and artifact root.

readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed Action Modes commit SHA.}"
: "${AMBIXQC_BENCHMARK_RESULTS_ROOT:?Export an existing absolute durable result directory.}"
: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation.}"

fail() {
  echo "AMBI-XQC Oscar paired benchmark error: $*" >&2
  exit 2
}

[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail \
  "EXPECTED_ACTION_MODES_SHA must be a full lowercase 40-character SHA"
[[ "$AMBIXQC_BENCHMARK_RESULTS_ROOT" == /* ]] || fail \
  "AMBIXQC_BENCHMARK_RESULTS_ROOT must be absolute"
[[ -d "$AMBIXQC_BENCHMARK_RESULTS_ROOT" ]] || fail \
  "AMBIXQC_BENCHMARK_RESULTS_ROOT must already exist"
[[ -w "$AMBIXQC_BENCHMARK_RESULTS_ROOT" ]] || fail \
  "AMBIXQC_BENCHMARK_RESULTS_ROOT is not writable"
readonly PROJECT_DIR_REAL="$(cd "$PROJECT_DIR" && pwd -P)"
readonly RESULTS_ROOT_REAL="$(cd "$AMBIXQC_BENCHMARK_RESULTS_ROOT" && pwd -P)"
readonly EXPECTED_RESULTS_ROOT="/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-compile/$EXPECTED_ACTION_MODES_SHA"
readonly DEFAULT_PYTHON="$PROJECT_DIR_REAL/environments/dmcontrol/.venv/bin/python"
readonly PYTHON="${AMBIXQC_PYTHON:-$DEFAULT_PYTHON}"
readonly PAIR_ORDER="${AMBIXQC_PAIR_ORDER:-eager-first}"
readonly JOB_ROOT="$RESULTS_ROOT_REAL/$SLURM_JOB_ID"
readonly MANIFEST="$PROJECT_DIR_REAL/configs/dmcontrol/experiments/ambixqc_humanoid_walk_state_timing_pair.json"
readonly ALG_DIR="$PROJECT_DIR_REAL/configs/dmcontrol/algs"
readonly COMPUTE_BENCHMARK="$PROJECT_DIR_REAL/tests/benchmarks/ambixqc_compute_throughput.py"
readonly SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
readonly CACHE_ROOT="$SCRATCH_BASE/ambixqc-compile-pair-$SLURM_JOB_ID"

[[ "$PYTHON" == /* && -x "$PYTHON" ]] || fail \
  "AMBIXQC_PYTHON must be an executable at an absolute path: $PYTHON"
[[ "$PAIR_ORDER" == eager-first || "$PAIR_ORDER" == compiled-first ]] || fail \
  "AMBIXQC_PAIR_ORDER must be eager-first or compiled-first"
[[ -f "$MANIFEST" && -f "$COMPUTE_BENCHMARK" ]] || fail \
  "paired manifest or compute benchmark is missing"
case "$RESULTS_ROOT_REAL/" in
  "$PROJECT_DIR_REAL/"*) fail "durable results must be outside the Git checkout" ;;
esac
[[ "$RESULTS_ROOT_REAL" == "$EXPECTED_RESULTS_ROOT" ]] || fail \
  "results root must be exactly $EXPECTED_RESULTS_ROOT"
[[ -d "$RESULTS_ROOT_REAL" && ! -L "$RESULTS_ROOT_REAL" ]] || fail \
  "results root must be one existing non-symlinked SHA directory"
[[ ! -e "$JOB_ROOT" && ! -L "$JOB_ROOT" ]] || fail \
  "refusing to overwrite an existing job directory: $JOB_ROOT"
[[ ! -e "$CACHE_ROOT" && ! -L "$CACHE_ROOT" ]] || fail \
  "refusing to reuse node-local cache root: $CACHE_ROOT"

require_clean_sha() {
  local checkout="$1"
  local expected="$2"
  local actual
  actual="$(git -C "$checkout" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] || fail \
    "Action Modes commit mismatch: expected $expected, found $actual"
  if [[ -n "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)" ]]; then
    git -C "$checkout" status --short >&2
    fail "Action Modes checkout must be clean"
  fi
}

require_lock_hash() {
  local lockfile="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$lockfile" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail \
    "DMControl lock mismatch: expected $expected, found $actual"
}

for command in git sha256sum nvidia-smi srun; do
  command -v "$command" >/dev/null 2>&1 || fail "missing required command: $command"
done
require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"
require_lock_hash "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" "$ACTION_LOCK_SHA"

mkdir "$JOB_ROOT"
mkdir "$CACHE_ROOT"
exec > >(tee "$JOB_ROOT/job.log") 2>&1

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
unset TDMPC2_EVAL_CSV WANDB_RUN_ID WANDB_RESUME

echo "Job: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Pair order: $PAIR_ORDER"
echo "Action Modes: $PROJECT_DIR_REAL @ $EXPECTED_ACTION_MODES_SHA"
echo "Python: $PYTHON"
echo "Durable artifacts: $JOB_ROOT"
git -C "$PROJECT_DIR_REAL" show --no-patch --format=fuller HEAD
sha256sum "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader | tee "$JOB_ROOT/gpu.txt"

cd "$PROJECT_DIR_REAL"
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" -c \
  'import sys, torch; from importlib.metadata import version; assert torch.cuda.is_available(); assert torch.get_default_dtype() == torch.float32; assert not torch.is_autocast_enabled(); print(sys.version); print({name: version(name) for name in ("dm-control", "gymnasium", "mujoco", "numpy", "torch")}); print(torch.version.cuda, torch.cuda.get_device_name(0))' \
  | tee "$JOB_ROOT/runtime.txt"

configure_process_cache() {
  local label="$1"
  local process_root="$CACHE_ROOT/$label"
  [[ ! -e "$process_root" && ! -L "$process_root" ]] || fail \
    "refusing to reuse process cache: $process_root"
  mkdir "$process_root"
  mkdir "$process_root/xdg" "$process_root/torchinductor" \
    "$process_root/triton" "$process_root/cuda"
  export XDG_CACHE_HOME="$process_root/xdg"
  export TORCHINDUCTOR_CACHE_DIR="$process_root/torchinductor"
  export TRITON_CACHE_DIR="$process_root/triton"
  export CUDA_CACHE_PATH="$process_root/cuda"
}

run_compute_benchmark() {
  local mode="$1"
  local repetition="$2"
  local flags=()
  if [[ "$mode" == compiled ]]; then
    flags+=(--compile --compile-strict --require-compiled)
  else
    flags+=(--no-compile --no-compile-strict)
  fi
  configure_process_cache "compute-$mode-rep$repetition"
  echo "Starting exact-shape compute benchmark: $mode repetition $repetition"
  srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" \
    "$COMPUTE_BENCHMARK" \
    --device cuda \
    --warmup 10 \
    --measured 50 \
    --optimizer-backend auto \
    --max-projection-residual 1e-6 \
    --output "$JOB_ROOT/compute-$mode-rep$repetition.json" \
    "${flags[@]}" \
    | tee "$JOB_ROOT/compute-$mode-rep$repetition.log"
}

run_training_timing() {
  local mode="$1"
  local index expected_compile start_ns end_ns
  if [[ "$mode" == eager ]]; then
    index=0
    expected_compile=false
  else
    index=1
    expected_compile=true
  fi
  configure_process_cache "canary-$mode"
  echo "Starting 1,502-step end-to-end timing arm: $mode"
  start_ns="$(date +%s%N)"
  TDMPC2_COMPUTE_TIMING_OUTPUT="$JOB_ROOT/training-$mode-compute.json" \
    srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" main.py \
    --run "$MANIFEST" \
    --alg-dir "$ALG_DIR" \
    --log-dir "$JOB_ROOT/training-$mode" \
    --alg-index "$index" \
    --trial-index 0 \
    --num-runs 1 \
    | tee "$JOB_ROOT/training-$mode.log"
  end_ns="$(date +%s%N)"
  [[ -f "$JOB_ROOT/training-$mode-compute.json" ]] || fail \
    "training compute timing artifact is missing for $mode"
  AMBIXQC_TIMING_OUTPUT="$JOB_ROOT/training-$mode-timing.json" \
  AMBIXQC_TIMING_MODE="$mode" \
  AMBIXQC_TIMING_COMPILE="$expected_compile" \
  AMBIXQC_TIMING_START_NS="$start_ns" \
  AMBIXQC_TIMING_END_NS="$end_ns" \
  AMBIXQC_TIMING_SHA="$EXPECTED_ACTION_MODES_SHA" \
  AMBIXQC_TIMING_JOB="$SLURM_JOB_ID" \
    "$PYTHON" -c '
import json
import os
from pathlib import Path

elapsed_seconds = (
    int(os.environ["AMBIXQC_TIMING_END_NS"])
    - int(os.environ["AMBIXQC_TIMING_START_NS"])
) / 1e9
payload = {
    "schema": "ambixqc-end-to-end-timing-v2",
    "mode": os.environ["AMBIXQC_TIMING_MODE"],
    "compile": os.environ["AMBIXQC_TIMING_COMPILE"] == "true",
    "compile_strict": os.environ["AMBIXQC_TIMING_COMPILE"] == "true",
    "cold_process_elapsed_seconds": elapsed_seconds,
    "total_steps": 1502,
    "seed_steps": 500,
    "planned_inner_actions": 1001,
    "outer_updates": 1002,
    "source_sha": os.environ["AMBIXQC_TIMING_SHA"],
    "slurm_job_id": os.environ["AMBIXQC_TIMING_JOB"],
}
with Path(os.environ["AMBIXQC_TIMING_OUTPUT"]).open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
'
}

validate_checkpoint() {
  local mode="$1"
  local expected_compile=false
  [[ "$mode" == eager ]] || expected_compile=true
  AMBIXQC_VALIDATE_ROOT="$JOB_ROOT/training-$mode" \
  AMBIXQC_VALIDATE_MODE="$mode" \
  AMBIXQC_VALIDATE_COMPILE="$expected_compile" \
  AMBIXQC_VALIDATE_OUTPUT="$JOB_ROOT/checkpoint-$mode.json" \
    srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" -c '
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

root = Path(os.environ["AMBIXQC_VALIDATE_ROOT"])
paths = [
    path for path in root.rglob("*_latest")
    if path.is_file() and not path.name.endswith(".metadata.json")
]
if len(paths) != 1:
    raise SystemExit(f"expected one final latest checkpoint, found {paths}")
path = paths[0]
metadata_path = Path(f"{path}.metadata.json")
if not metadata_path.is_file():
    raise SystemExit(f"missing checkpoint metadata: {metadata_path}")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
checkpoint = metadata.get("checkpoint", {})
if checkpoint.get("kind") != "latest" or checkpoint.get("step") != 1502:
    raise SystemExit(f"latest checkpoint is not final step 1502: {checkpoint}")
trial = metadata.get("trial_run_params", {})
resolved = trial.get("resolved_runtime", {})
expected_compile = os.environ["AMBIXQC_VALIDATE_COMPILE"] == "true"
if (
    trial.get("alg") != "AMBIXQC/AMBIXQC"
    or trial.get("seed") != 55
    or trial.get("total_steps") != 1502
    or resolved.get("compilation")
    != {"enabled": expected_compile, "strict": expected_compile}
):
    raise SystemExit(f"unexpected resolved timing trial: {trial}")

state = torch.load(path, map_location="cpu", weights_only=False)
counters = (
    state.get("num_updates"),
    state.get("outer_version"),
    state.get("xqc_workspace", {}).get("update_step"),
    state.get("xqc_workspace", {}).get("actor_optimizer_steps"),
    state.get("xqc_workspace", {}).get("temperature_optimizer_steps"),
    state.get("inner", {}).get("action_index"),
)
expected_counters = (1002, 1002, 1002, 334, 334, 1001)
if counters != expected_counters:
    raise SystemExit(f"unexpected timing counters: {counters} != {expected_counters}")
if state.get("reward_normalizer", {}).get("count") != 1502.0:
    raise SystemExit("reward normalizer did not observe all 1,502 real decisions")

def assert_finite(value, location="state"):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise SystemExit(f"non-finite tensor at {location}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not bool(np.isfinite(value).all()):
            raise SystemExit(f"non-finite array at {location}")
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise SystemExit(f"non-finite scalar at {location}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{location}[{index}]")

assert_finite(state)
projected = {
    key: value for key, value in state["module"].items()
    if (
        key.startswith("xqc_controller.actor.")
        or key.startswith("xqc_controller.critic.")
    )
    and key.endswith((".linear.weight", ".mean.weight", ".log_std.weight", ".value.weight"))
}
if len(projected) != 16:
    raise SystemExit(f"expected 16 projected online weights, found {sorted(projected)}")
residual = max(
    float((torch.linalg.vector_norm(weight, dim=1) - 1.0).abs().max())
    for weight in projected.values()
)
if residual > 1e-6:
    raise SystemExit(f"projection residual {residual} exceeds 1e-6")
payload = {
    "schema": "ambixqc-timing-checkpoint-validation-v1",
    "mode": os.environ["AMBIXQC_VALIDATE_MODE"],
    "checkpoint": str(path),
    "counters": counters,
    "projection_residual": residual,
    "all_finite": True,
}
with Path(os.environ["AMBIXQC_VALIDATE_OUTPUT"]).open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(payload, sort_keys=True))
'
}

configure_process_cache "cuda-correctness-gate"
echo "Starting mandatory AMBI-XQC CUDA compiler correctness gate"
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" -m pytest \
  -q -p no:cacheprovider \
  tests/test_ambixqc_compile.py \
  tests/test_compile_regions.py \
  | tee "$JOB_ROOT/cuda-correctness-gate.log"

for repetition in 1 2 3 4 5; do
  if ((repetition % 2 == 1)) && [[ "$PAIR_ORDER" == eager-first ]] \
    || ((repetition % 2 == 0)) && [[ "$PAIR_ORDER" == compiled-first ]]; then
    REP_MODES=(eager compiled)
  else
    REP_MODES=(compiled eager)
  fi
  for mode in "${REP_MODES[@]}"; do
    run_compute_benchmark "$mode" "$repetition"
  done
done

AMBIXQC_AGGREGATE_ROOT="$JOB_ROOT" "$PYTHON" -c '
import json
import math
import os
from pathlib import Path
import statistics

root = Path(os.environ["AMBIXQC_AGGREGATE_ROOT"])
records = {
    mode: [
        json.loads((root / f"compute-{mode}-rep{rep}.json").read_text(encoding="utf-8"))
        for rep in range(1, 6)
    ]
    for mode in ("eager", "compiled")
}
expected_counters = {
    "inner_action_index": 60,
    "outer_num_updates": 60,
    "outer_version": 60,
    "outer_xqc_update_step": 60,
    "last_inner_update_slots": 8,
    "last_inner_actor_steps": 3,
    "last_inner_temperature_steps": 3,
    "last_inner_model_steps": 192,
    "last_inner_replay_draws": 512,
}
source_shas = {record["source_sha"] for values in records.values() for record in values}
config_hashes = {record["config_sha256"] for values in records.values() for record in values}
if len(source_shas) != 1 or len(config_hashes) != 1:
    raise SystemExit("compute repetitions did not use one source/config identity")
for mode, values in records.items():
    for repetition, record in enumerate(values, start=1):
        if (
            record["warmup_cycles"] != 10
            or record["measured_cycles"] != 50
            or record["counters"] != expected_counters
            or not record["all_finite"]
            or record["compile_fallback"]
            or record["projection_residual"] > 1e-6
            or not record["outer_prior_preserved_by_inner_action"]
            or not record["outer_prior_changed_by_outer_update"]
            or not record["workspace_reused"]
            or not record["workspace_reset_counters_fresh"]
            or not record["global_torch_rng_preserved"]
        ):
            raise SystemExit(f"mandatory correctness failed for {mode} repetition {repetition}")
        expected_compile = mode == "compiled"
        if (
            record["compile_requested"] != expected_compile
            or record["compile_strict"] != expected_compile
            or record["all_four_regions_compiled"] != expected_compile
        ):
            raise SystemExit(f"compile status failed for {mode} repetition {repetition}")

def values(mode, key):
    return [float(record[key]) for record in records[mode]]

def coefficient_of_variation(samples):
    mean = statistics.fmean(samples)
    return statistics.pstdev(samples) / mean

p50 = {mode: values(mode, "iteration_p50_seconds") for mode in records}
p95 = {mode: values(mode, "iteration_p95_seconds") for mode in records}
peak = {mode: values(mode, "peak_cuda_allocated_bytes") for mode in records}
cold = {mode: values(mode, "cold_cycle_seconds") for mode in records}
median_p50 = {mode: statistics.median(samples) for mode, samples in p50.items()}
median_p95 = {mode: statistics.median(samples) for mode, samples in p95.items()}
cv = {mode: coefficient_of_variation(samples) for mode, samples in p50.items()}
paired_p50_ratios = [compiled / eager for eager, compiled in zip(p50["eager"], p50["compiled"])]
p50_ratio = median_p50["compiled"] / median_p50["eager"]
p95_ratio = median_p95["compiled"] / median_p95["eager"]
paired_max_ratio = max(paired_p50_ratios)
peak_ratio = statistics.median(peak["compiled"]) / statistics.median(peak["eager"])
max_cv = max(cv.values())
cold_delta = statistics.median(cold["compiled"]) - statistics.median(cold["eager"])
steady_savings = median_p50["eager"] - median_p50["compiled"]
break_even_actions = (
    max(0.0, cold_delta) / steady_savings
    if steady_savings > 0.0
    else None
)
checks = {
    "compiled_p50_ratio_at_most_0p90": p50_ratio <= 0.90,
    "compiled_p95_ratio_at_most_0p95": p95_ratio <= 0.95,
    "maximum_p50_cv_at_most_0p05": max_cv <= 0.05,
    "paired_compiled_p50_ratio_at_most_1p10": paired_max_ratio <= 1.10,
    "peak_allocation_ratio_at_most_1p10": peak_ratio <= 1.10,
}
misses = [name for name, passed in checks.items() if not passed]
payload = {
    "schema": "ambixqc-compute-five-repetition-comparison-v1",
    "source_sha": next(iter(source_shas)),
    "config_sha256": next(iter(config_hashes)),
    "repetitions_per_mode": 5,
    "warmup_cycles_per_process": 10,
    "measured_cycles_per_process": 50,
    "mandatory_correctness_pass": True,
    "median_p50_seconds": median_p50,
    "median_p95_seconds": median_p95,
    "p50_ratio_compiled_over_eager": p50_ratio,
    "p95_ratio_compiled_over_eager": p95_ratio,
    "p50_coefficient_of_variation": cv,
    "maximum_p50_coefficient_of_variation": max_cv,
    "paired_p50_compiled_over_eager": paired_p50_ratios,
    "paired_p50_max_ratio": paired_max_ratio,
    "median_peak_cuda_allocated_bytes": {
        mode: statistics.median(samples) for mode, samples in peak.items()
    },
    "peak_allocation_ratio_compiled_over_eager": peak_ratio,
    "median_cold_cycle_seconds": {
        mode: statistics.median(samples) for mode, samples in cold.items()
    },
    "compile_cold_delta_seconds": cold_delta,
    "steady_p50_savings_seconds_per_action": steady_savings,
    "compile_cost_break_even_actions": break_even_actions,
    "performance_thresholds": checks,
    "performance_misses": misses,
    "micro_performance_pass": not misses,
}
with (root / "compute-aggregate.json").open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
print(json.dumps(payload, sort_keys=True, allow_nan=False))
if misses:
    print(f"PERFORMANCE MISS (nonfatal): {misses}")
'

if [[ "$PAIR_ORDER" == eager-first ]]; then
  MODES=(eager compiled)
else
  MODES=(compiled eager)
fi
readonly MODES

for mode in "${MODES[@]}"; do
  run_training_timing "$mode"
  validate_checkpoint "$mode"
done

AMBIXQC_COMPARE_ROOT="$JOB_ROOT" "$PYTHON" -c '
import json
import math
import os
from pathlib import Path

root = Path(os.environ["AMBIXQC_COMPARE_ROOT"])
compute = json.loads((root / "compute-aggregate.json").read_text(encoding="utf-8"))
training = {
    mode: json.loads((root / f"training-{mode}-timing.json").read_text(encoding="utf-8"))
    for mode in ("eager", "compiled")
}
training_compute = {
    mode: json.loads((root / f"training-{mode}-compute.json").read_text(encoding="utf-8"))
    for mode in ("eager", "compiled")
}
for mode, record in training_compute.items():
    expected = {
        "schema": "tdmpc2-training-compute-timing-v1",
        "clock": "synchronized-host-wall",
        "warmup_rule": "exclude-first-planned-action-and-first-outer-update",
        "planned_action_count": 1001,
        "outer_update_count": 1002,
        "warmed_planned_action_count": 1000,
        "warmed_outer_update_count": 1001,
        "total_steps": 1502,
        "num_updates": 1002,
        "all_finite": True,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise SystemExit(
                f"unexpected {mode} warmed compute timing {key}: "
                f"{record.get(key)!r} != {value!r}"
            )
    recomputed = (
        sum(record["planned_action_seconds"][1:])
        + sum(record["outer_update_seconds"][1:])
    )
    if (
        len(record["planned_action_seconds"]) != 1001
        or len(record["outer_update_seconds"]) != 1002
        or not math.isclose(
            recomputed,
            record["warmed_compute_seconds"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or record["warmed_compute_seconds"] <= 0.0
    ):
        raise SystemExit(f"invalid {mode} warmed compute timing payload")

cold_process_runtime = {
    mode: float(record["cold_process_elapsed_seconds"])
    for mode, record in training.items()
}
warmed_canary_compute = {
    mode: float(record["warmed_compute_seconds"])
    for mode, record in training_compute.items()
}
if min(cold_process_runtime.values()) <= 0.0:
    raise SystemExit("cold process runtimes must be positive")
cold_process_speedup = (
    cold_process_runtime["eager"] / cold_process_runtime["compiled"]
)
warmed_canary_compute_speedup = (
    warmed_canary_compute["eager"] / warmed_canary_compute["compiled"]
)
canary_pass = warmed_canary_compute_speedup >= 1.05
performance_misses = list(compute["performance_misses"])
if not canary_pass:
    performance_misses.append("warmed_canary_compute_speedup_at_least_1p05")
performance_pass = not performance_misses
payload = {
    "schema": "ambixqc-eager-compiled-pair-v1",
    "source_sha": compute["source_sha"],
    "mandatory_correctness_pass": compute["mandatory_correctness_pass"],
    "micro_performance_pass": compute["micro_performance_pass"],
    "cold_process_runtime_seconds": cold_process_runtime,
    "cold_process_speedup": cold_process_speedup,
    "warmed_canary_compute_seconds": warmed_canary_compute,
    "warmed_canary_compute_speedup": warmed_canary_compute_speedup,
    "warmed_canary_compute_speedup_at_least_1p05": canary_pass,
    "compile_cost_break_even_actions": compute["compile_cost_break_even_actions"],
    "performance_misses": performance_misses,
    "performance_pass": performance_pass,
    "classification": (
        "correctness-pass-performance-pass"
        if performance_pass
        else "correctness-pass-performance-miss"
    ),
}
with (root / "comparison.json").open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(payload, sort_keys=True))
if not performance_pass:
    with (root / "PERFORMANCE_MISS").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    print("PERFORMANCE MISS is nonfatal because mandatory correctness passed")
'

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"
printf 'PASS\n' > "$JOB_ROOT/PASS"
echo "AMBI-XQC Oscar paired benchmark passed. Artifacts: $JOB_ROOT"
