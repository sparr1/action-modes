#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly TARGET_NODE="gpu2301"
readonly JOB_COUNT=2
readonly JOB_CPUS=8
readonly JOB_MEMORY_MIB=65536
readonly MIN_DURABLE_FREE_KIB=$((8 * 1024 * 1024))
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
readonly LAUNCHER="$SCRIPT_DIR/run_ambixqc_humanoid_walk_heavy_inner_pair_hydra.sbatch"

EXPECTED_ACTION_MODES_SHA=""
AMBIXQC_HEAVY_RESULTS_ROOT=""
AMBIXQC_HEAVY_RUN_ID=""
AMBIXQC_HEAVY_PYTHON=""
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  slurm/submit_ambixqc_humanoid_walk_heavy_inner_pair_hydra.sh \
    --expected-action-modes-sha SHA \
    --results-root ABSOLUTE_PATH \
    --run-id UNIQUE_ID \
    --python ABSOLUTE_PATH [--dry-run]

Submits one atomic two-task Hydra array pinned to gpu2301. The only cells are
the seed-55, 14-million-decision AMBI-XQC heavy-inner d512_g3_j6 and d512_g3
configurations. Each task requests one GPU, eight CPUs, and 64 GiB host RAM.
Submission requires two immediately available resource slots on gpu2301; there
is no fallback node. The helper never updates the checkout or environment.
EOF
}

fail() {
  echo "AMBI-XQC heavy-inner Hydra submission error: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --expected-action-modes-sha)
      EXPECTED_ACTION_MODES_SHA="${2:-}"
      shift 2
      ;;
    --results-root)
      AMBIXQC_HEAVY_RESULTS_ROOT="${2:-}"
      shift 2
      ;;
    --run-id)
      AMBIXQC_HEAVY_RUN_ID="${2:-}"
      shift 2
      ;;
    --python)
      AMBIXQC_HEAVY_PYTHON="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

for command in git sha256sum df awk sbatch scontrol; do
  command -v "$command" >/dev/null 2>&1 || fail "missing required command: $command"
done

[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail \
  "--expected-action-modes-sha must be a full lowercase 40-character SHA"
[[ "$AMBIXQC_HEAVY_RESULTS_ROOT" == /* ]] || fail \
  "--results-root must be absolute"
[[ -d "$AMBIXQC_HEAVY_RESULTS_ROOT" ]] || fail \
  "--results-root must be an existing directory"
[[ ! -L "$AMBIXQC_HEAVY_RESULTS_ROOT" ]] || fail \
  "--results-root must not be a symlink"
[[ -w "$AMBIXQC_HEAVY_RESULTS_ROOT" ]] || fail \
  "--results-root is not writable"
[[ "$AMBIXQC_HEAVY_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail \
  "--run-id must use only letters, digits, '.', '_', and '-'"
(( ${#AMBIXQC_HEAVY_RUN_ID} <= 64 )) || fail \
  "--run-id must contain at most 64 characters"
[[ "$AMBIXQC_HEAVY_PYTHON" == /* && -x "$AMBIXQC_HEAVY_PYTHON" ]] || fail \
  "--python is required and must be an executable at an absolute path"
[[ -f "$LAUNCHER" ]] || fail "missing launcher: $LAUNCHER"

for exported_value in \
  "$AMBIXQC_HEAVY_RESULTS_ROOT" \
  "$AMBIXQC_HEAVY_RUN_ID" \
  "$AMBIXQC_HEAVY_PYTHON"; do
  if [[ "$exported_value" == *','* || "$exported_value" == *$'\n'* ]]; then
    fail "exported paths and IDs cannot contain commas or newlines"
  fi
done

AMBIXQC_HEAVY_RESULTS_ROOT="$(cd "$AMBIXQC_HEAVY_RESULTS_ROOT" && pwd -P)"
AMBIXQC_HEAVY_PYTHON="$(cd "$(dirname "$AMBIXQC_HEAVY_PYTHON")" && pwd -P)/$(basename "$AMBIXQC_HEAVY_PYTHON")"
ENV_PROJECT="$(cd "$(dirname "$AMBIXQC_HEAVY_PYTHON")/../.." && pwd -P)"
readonly AMBIXQC_HEAVY_RESULTS_ROOT AMBIXQC_HEAVY_PYTHON ENV_PROJECT
readonly STUDY_ROOT="$AMBIXQC_HEAVY_RESULTS_ROOT/$AMBIXQC_HEAVY_RUN_ID"

require_clean_sha() {
  local checkout="$1"
  local expected="$2"
  local actual
  actual="$(git -C "$checkout" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] || fail \
    "Action Modes commit mismatch: expected $expected, found $actual"
  if [[ -n "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)" ]]; then
    git -C "$checkout" status --short >&2
    fail "submission requires a clean checkout"
  fi
}

require_lock_hash() {
  local lockfile="$1"
  local expected="$2"
  local label="$3"
  local actual
  [[ -f "$lockfile" ]] || fail "missing $label lock: $lockfile"
  actual="$(sha256sum "$lockfile" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail \
    "$label lock mismatch: expected $expected, found $actual"
}

require_free_kib() {
  local path="$1"
  local minimum="$2"
  local available
  available="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  [[ "$available" =~ ^[0-9]+$ ]] || fail \
    "could not determine free space for results root: $path"
  (( available >= minimum )) || fail \
    "results root has less than $((minimum / 1024 / 1024)) GiB free: $path"
}

node_job_slots() {
  local node="$1"
  local record state total_gpus allocated_gpus total_cpus allocated_cpus
  local real_memory allocated_memory gpu_slots cpu_slots memory_slots slots

  if ! record="$(scontrol show node -o "$node" 2>/dev/null)"; then
    echo 0
    return
  fi
  if [[ ! "$record" =~ State=([^[:space:]]+) ]]; then
    echo 0
    return
  fi
  state="${BASH_REMATCH[1]}"
  case "$state" in
    *DOWN*|*DRAIN*|*FAIL*|*MAINT*|*NOT_RESPONDING*|*POWERED_DOWN*)
      echo 0
      return
      ;;
  esac
  if [[ ! "$record" =~ CfgTRES=[^[:space:]]*gres/gpu=([0-9]+) ]]; then
    echo 0
    return
  fi
  total_gpus="${BASH_REMATCH[1]}"
  allocated_gpus=0
  if [[ "$record" =~ AllocTRES=[^[:space:]]*gres/gpu=([0-9]+) ]]; then
    allocated_gpus="${BASH_REMATCH[1]}"
  fi
  if [[ "$record" =~ CPUAlloc=([0-9]+) ]]; then
    allocated_cpus="${BASH_REMATCH[1]}"
  else
    echo 0
    return
  fi
  if [[ "$record" =~ CPUTot=([0-9]+) ]]; then
    total_cpus="${BASH_REMATCH[1]}"
  else
    echo 0
    return
  fi
  if [[ "$record" =~ RealMemory=([0-9]+) ]]; then
    real_memory="${BASH_REMATCH[1]}"
  else
    echo 0
    return
  fi
  if [[ "$record" =~ AllocMem=([0-9]+) ]]; then
    allocated_memory="${BASH_REMATCH[1]}"
  else
    echo 0
    return
  fi

  gpu_slots=$((total_gpus - allocated_gpus))
  cpu_slots=$(((total_cpus - allocated_cpus) / JOB_CPUS))
  memory_slots=$(((real_memory - allocated_memory) / JOB_MEMORY_MIB))
  slots="$gpu_slots"
  ((cpu_slots < slots)) && slots="$cpu_slots"
  ((memory_slots < slots)) && slots="$memory_slots"
  ((slots < 0)) && slots=0
  echo "$slots"
}

[[ "$AMBIXQC_HEAVY_PYTHON" == "$ENV_PROJECT/.venv/bin/python" ]] || fail \
  "--python must point to the locked environment's .venv/bin/python"
case "$AMBIXQC_HEAVY_RESULTS_ROOT/" in
  "$PROJECT_DIR/"*) fail "durable results must be outside the Git checkout" ;;
esac
require_clean_sha "$PROJECT_DIR" "$EXPECTED_ACTION_MODES_SHA"
require_lock_hash \
  "$PROJECT_DIR/environments/dmcontrol/uv.lock" "$ACTION_LOCK_SHA" \
  "Action Modes"
require_lock_hash "$ENV_PROJECT/uv.lock" "$ACTION_LOCK_SHA" \
  "DMControl environment"
require_free_kib "$AMBIXQC_HEAVY_RESULTS_ROOT" "$MIN_DURABLE_FREE_KIB"

if [[ -e "$STUDY_ROOT" || -L "$STUDY_ROOT" ]]; then
  fail "refusing to reuse study artifacts: $STUDY_ROOT"
fi

available_slots="$(node_job_slots "$TARGET_NODE")"
readonly available_slots
(( available_slots >= JOB_COUNT )) || fail \
  "need two immediate one-GPU/eight-CPU/64-GiB slots on $TARGET_NODE; found $available_slots"

if [[ "$DRY_RUN" == false ]]; then
  mkdir "$STUDY_ROOT"
fi

export_spec="ALL,EXPECTED_ACTION_MODES_SHA=$EXPECTED_ACTION_MODES_SHA,AMBIXQC_HEAVY_RESULTS_ROOT=$AMBIXQC_HEAVY_RESULTS_ROOT,AMBIXQC_HEAVY_RUN_ID=$AMBIXQC_HEAVY_RUN_ID,AMBIXQC_HEAVY_PYTHON=$AMBIXQC_HEAVY_PYTHON"
cd "$PROJECT_DIR"
submit_command=(
  sbatch
  --parsable
  --chdir="$PROJECT_DIR"
  --nodelist="$TARGET_NODE"
  --output="$STUDY_ROOT/slurm-%x-%A_%a.out"
  --error="$STUDY_ROOT/slurm-%x-%A_%a.err"
  --export="$export_spec"
  "$LAUNCHER"
)

printf 'Target node: %s (immediate matching slots: %s)\n' \
  "$TARGET_NODE" "$available_slots"
printf 'Submit command:'
printf ' %q' "${submit_command[@]}"
printf '\n'
if [[ "$DRY_RUN" == false ]]; then
  submission="$("${submit_command[@]}")"
  job_id="${submission%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || fail "unexpected sbatch response: $submission"
  echo "Submitted AMBI-XQC heavy-inner array $job_id on $TARGET_NODE."
  echo "Study results: $STUDY_ROOT"
fi
