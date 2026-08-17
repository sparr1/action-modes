#!/usr/bin/env bash
#SBATCH --job-name=ambi-inner-latency
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -Eeuo pipefail
umask 077

fail() {
  echo "AMBI latency launcher error: $*" >&2
  exit 2
}

: "${SLURM_ARRAY_JOB_ID:?This launcher must run as a Slurm array job.}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array job.}"
: "${AMBI_LATENCY_CHECKPOINT:?Set AMBI_LATENCY_CHECKPOINT to an absolute frozen-checkpoint path.}"
: "${AMBI_LATENCY_OUTPUT_ROOT:?Set AMBI_LATENCY_OUTPUT_ROOT to an absolute Oscar scratch directory.}"
: "${AMBI_LATENCY_EXPECTED_COMMIT:?Set AMBI_LATENCY_EXPECTED_COMMIT to the exact 40-character benchmark commit SHA.}"

PROJECT_DIR="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$PROJECT_DIR"

GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || fail "not a Git checkout"
GIT_ROOT="$(cd -- "$GIT_ROOT" && pwd -P)"
[[ "$GIT_ROOT" == "$PROJECT_DIR" ]] || fail "submit directory is not the Git root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail \
  "benchmark requires a clean dedicated checkout"
[[ "$AMBI_LATENCY_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail \
  "AMBI_LATENCY_EXPECTED_COMMIT must be a full lowercase 40-character commit SHA"
ACTUAL_COMMIT="$(git rev-parse HEAD 2>/dev/null)" || fail "cannot resolve checkout commit"
[[ "$ACTUAL_COMMIT" == "$AMBI_LATENCY_EXPECTED_COMMIT" ]] || fail \
  "checkout commit $ACTUAL_COMMIT does not match expected $AMBI_LATENCY_EXPECTED_COMMIT"

CONFIG="${AMBI_LATENCY_CONFIG:-$PROJECT_DIR/configs/research/ambi_latency_benchmark.json}"
BENCHMARK_PYTHON="${AMBI_BENCHMARK_PYTHON:-$PROJECT_DIR/environments/dmcontrol/.venv/bin/python}"
RUNNER="$PROJECT_DIR/benchmark_ambi_latency.py"

[[ -f "$CONFIG" ]] || fail "benchmark config does not exist: $CONFIG"
[[ -f "$RUNNER" ]] || fail "benchmark runner does not exist: $RUNNER"
[[ -x "$BENCHMARK_PYTHON" ]] || fail "locked DMControl Python is not executable: $BENCHMARK_PYTHON"
[[ "$AMBI_LATENCY_CHECKPOINT" == /* ]] || fail "AMBI_LATENCY_CHECKPOINT must be absolute"
[[ -f "$AMBI_LATENCY_CHECKPOINT" ]] || fail "checkpoint does not exist: $AMBI_LATENCY_CHECKPOINT"
[[ "$AMBI_LATENCY_OUTPUT_ROOT" == /* ]] || fail "AMBI_LATENCY_OUTPUT_ROOT must be absolute"
case "${AMBI_LATENCY_OUTPUT_ROOT%/}" in
  "$PROJECT_DIR"|"$PROJECT_DIR"/*)
    fail "AMBI_LATENCY_OUTPUT_ROOT must be outside the source checkout"
    ;;
esac

mkdir -p "$AMBI_LATENCY_OUTPUT_ROOT"
JOB_ROOT="${AMBI_LATENCY_OUTPUT_ROOT%/}/job_${SLURM_ARRAY_JOB_ID}"
BLOCK_ROOT="$JOB_ROOT/block_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_ROOT"
mkdir "$BLOCK_ROOT" || fail "block output already exists: $BLOCK_ROOT"

export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

CONFIG_CELL_LIST="$BLOCK_ROOT/cells.config-order.tsv"
CELL_LIST="$BLOCK_ROOT/cells.tsv"
"$BENCHMARK_PYTHON" "$RUNNER" \
  --config "$CONFIG" \
  --list-cells >"$CONFIG_CELL_LIST"
"$BENCHMARK_PYTHON" - \
  "$CONFIG" "$SLURM_ARRAY_TASK_ID" "$CONFIG_CELL_LIST" >"$CELL_LIST" <<'PY'
import json
import random
import re
import sys

config_path, raw_block, cell_list_path = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)

settings = config.get("settings", {})
blocks = settings.get("blocks")
block = int(raw_block)
if blocks != 3:
    raise SystemExit(f"settings.blocks must remain 3 for array 0-2, got {blocks!r}")
if not 0 <= block < blocks:
    raise SystemExit(f"array block {block} is outside [0, {blocks})")

rows = []
with open(cell_list_path, encoding="utf-8") as handle:
    raw_rows = [line.rstrip("\n").split("\t") for line in handle]
if len(raw_rows) != 15:
    raise SystemExit(f"expected exactly 15 deduplicated cells, got {len(raw_rows)}")
for index, row in enumerate(raw_rows):
    if len(row) != 4:
        raise SystemExit(f"malformed --list-cells row {index}: {row!r}")
    name, raw_J, raw_N, raw_G = row
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise SystemExit(f"unsafe cell name at row {index}: {name!r}")
    try:
        J, N, G = int(raw_J), int(raw_N), int(raw_G)
    except ValueError as exc:
        raise SystemExit(f"invalid J/N/G at row {index}: {row!r}") from exc
    rows.append((name, J, N, G))

# Every block repeats all cells. A deterministic block-specific permutation
# reduces systematic thermal/clock-order bias while keeping the run auditable.
order_seed = settings.get("block_order_seed", 20260817)
if isinstance(order_seed, bool) or not isinstance(order_seed, int):
    raise SystemExit("settings.block_order_seed must be an integer")
random.Random(order_seed + block).shuffle(rows)
for row in rows:
    print(*row, sep="\t")
PY

mapfile -t CELLS <"$CELL_LIST"
[[ "${#CELLS[@]}" -eq 15 ]] || fail "block must contain all 15 cells"

echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Git commit: $ACTUAL_COMMIT"
echo "Python: $BENCHMARK_PYTHON"
echo "Checkpoint: $AMBI_LATENCY_CHECKPOINT"
echo "Output: $BLOCK_ROOT"
echo "Cell order:"
sed 's/^/  /' "$CELL_LIST"

SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp}/ambi-latency-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_ROOT"

for row in "${CELLS[@]}"; do
  IFS=$'\t' read -r CELL_NAME J N G <<<"$row"
  [[ -n "$CELL_NAME" && -n "$J" && -n "$N" && -n "$G" ]] || fail "malformed cell row: $row"

  RESULT="$BLOCK_ROOT/${CELL_NAME}.json"
  STDOUT="$BLOCK_ROOT/${CELL_NAME}.stdout.log"
  STDERR="$BLOCK_ROOT/${CELL_NAME}.stderr.log"
  [[ ! -e "$RESULT" && ! -e "$STDOUT" && ! -e "$STDERR" ]] || fail \
    "refusing to overwrite output for $CELL_NAME"

  CELL_SCRATCH="$SCRATCH_ROOT/$CELL_NAME"
  mkdir "$CELL_SCRATCH"
  export TORCHINDUCTOR_CACHE_DIR="$CELL_SCRATCH/torchinductor"
  export TRITON_CACHE_DIR="$CELL_SCRATCH/triton"
  export CUDA_CACHE_PATH="$CELL_SCRATCH/cuda"

  echo "Starting $CELL_NAME (J=$J N=$N G=$G) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  srun --unbuffered --kill-on-bad-exit=1 \
    --output="$STDOUT" \
    --error="$STDERR" \
    "$BENCHMARK_PYTHON" "$RUNNER" \
      --config "$CONFIG" \
      --checkpoint "$AMBI_LATENCY_CHECKPOINT" \
      --device cuda \
      --cell "$J,$N,$G" \
      --output "$RESULT"
done

touch "$BLOCK_ROOT/COMPLETE"
echo "Completed all 15 cells in block $SLURM_ARRAY_TASK_ID at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
