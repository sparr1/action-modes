#!/usr/bin/env bash

set -Eeuo pipefail

readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly PREFERRED_NODE="gpu2301"
readonly FALLBACK_NODE="gpu2201"
readonly JOB_CPUS=8
readonly JOB_MEMORY_MIB=32768
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
readonly LAUNCHER="$SCRIPT_DIR/run_xqc_humanoid_walk_1m_hydra.sbatch"

EXPECTED_ACTION_MODES_SHA=""
XQC_BASELINE_RESULTS_ROOT=""
XQC_BASELINE_RUN_ID=""
XQC_BASELINE_PYTHON=""
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  slurm/submit_xqc_humanoid_walk_1m_hydra.sh \
    --expected-action-modes-sha SHA \
    --results-root ABSOLUTE_PATH \
    --run-id UNIQUE_ID \
    --python ABSOLUTE_PATH [--dry-run]

Submits exactly one seed-55, one-GPU, one-million-decision vanilla XQC
Humanoid Walk baseline. The helper prefers a free slot on gpu2301, falls back
to gpu2201, and queues on gpu2301 if neither node has an immediate free slot.
The locked DMControl Python path is required explicitly and may live in another
clean checkout with the identical lock.
EOF
}

while (($#)); do
  case "$1" in
    --expected-action-modes-sha)
      EXPECTED_ACTION_MODES_SHA="${2:-}"
      shift 2
      ;;
    --results-root)
      XQC_BASELINE_RESULTS_ROOT="${2:-}"
      shift 2
      ;;
    --run-id)
      XQC_BASELINE_RUN_ID="${2:-}"
      shift 2
      ;;
    --python)
      XQC_BASELINE_PYTHON="${2:-}"
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
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for command in git sha256sum sbatch scontrol; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    exit 1
  fi
done
if [[ ! "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "--expected-action-modes-sha must be a full lowercase 40-character SHA." >&2
  exit 1
fi
if [[ "$XQC_BASELINE_RESULTS_ROOT" != /* || ! -d "$XQC_BASELINE_RESULTS_ROOT" ]]; then
  echo "--results-root must be an existing absolute directory." >&2
  exit 1
fi
if [[ ! -w "$XQC_BASELINE_RESULTS_ROOT" ]]; then
  echo "--results-root is not writable: $XQC_BASELINE_RESULTS_ROOT" >&2
  exit 1
fi
if [[ ! "$XQC_BASELINE_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "--run-id must use only letters, digits, '.', '_', and '-'." >&2
  exit 1
fi
if ((${#XQC_BASELINE_RUN_ID} > 64)); then
  echo "--run-id must contain at most 64 characters." >&2
  exit 1
fi
if [[ "$XQC_BASELINE_PYTHON" != /* || ! -x "$XQC_BASELINE_PYTHON" ]]; then
  echo "--python is required and must be an executable at an absolute path." >&2
  exit 1
fi
if [[ ! -f "$LAUNCHER" ]]; then
  echo "Missing launcher: $LAUNCHER" >&2
  exit 1
fi
for exported_value in "$XQC_BASELINE_RESULTS_ROOT" "$XQC_BASELINE_RUN_ID" "$XQC_BASELINE_PYTHON"; do
  if [[ "$exported_value" == *','* || "$exported_value" == *$'\n'* ]]; then
    echo "Exported paths and IDs cannot contain commas or newlines." >&2
    exit 1
  fi
done

XQC_BASELINE_RESULTS_ROOT="$(cd "$XQC_BASELINE_RESULTS_ROOT" && pwd -P)"
XQC_BASELINE_PYTHON="$(cd "$(dirname "$XQC_BASELINE_PYTHON")" && pwd -P)/$(basename "$XQC_BASELINE_PYTHON")"
ENV_PROJECT="$(cd "$(dirname "$XQC_BASELINE_PYTHON")/../.." && pwd -P)"
readonly XQC_BASELINE_RESULTS_ROOT XQC_BASELINE_PYTHON ENV_PROJECT
readonly STUDY_ROOT="$XQC_BASELINE_RESULTS_ROOT/$XQC_BASELINE_RUN_ID"

require_clean_sha() {
  local checkout="$1"
  local expected="$2"
  local actual
  actual="$(git -C "$checkout" rev-parse HEAD)"
  if [[ "$actual" != "$expected" ]]; then
    echo "Action Modes commit mismatch: expected $expected, found $actual" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "Action Modes checkout is not clean: $checkout" >&2
    git -C "$checkout" status --short >&2
    exit 1
  fi
}

require_lock_hash() {
  local lockfile="$1"
  local expected="$2"
  local label="$3"
  local actual
  if [[ ! -f "$lockfile" ]]; then
    echo "Missing $label lock: $lockfile" >&2
    exit 1
  fi
  actual="$(sha256sum "$lockfile" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label lock hash mismatch: expected $expected, found $actual" >&2
    exit 1
  fi
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

if [[ "$XQC_BASELINE_PYTHON" != "$ENV_PROJECT/.venv/bin/python" ]]; then
  echo "--python must point to the locked environment's .venv/bin/python." >&2
  exit 1
fi
case "$XQC_BASELINE_RESULTS_ROOT/" in
  "$PROJECT_DIR/"*)
    echo "Durable results must be outside the Git checkout." >&2
    exit 1
    ;;
esac
require_clean_sha "$PROJECT_DIR" "$EXPECTED_ACTION_MODES_SHA"
require_lock_hash "$PROJECT_DIR/environments/dmcontrol/uv.lock" "$ACTION_LOCK_SHA" "Action Modes"
require_lock_hash "$ENV_PROJECT/uv.lock" "$ACTION_LOCK_SHA" "DMControl environment"

if [[ -e "$STUDY_ROOT" || -L "$STUDY_ROOT" ]]; then
  echo "Refusing to reuse study artifacts: $STUDY_ROOT" >&2
  exit 1
fi

preferred_slots="$(node_job_slots "$PREFERRED_NODE")"
fallback_slots="$(node_job_slots "$FALLBACK_NODE")"
readonly preferred_slots fallback_slots
if ((preferred_slots >= 1)); then
  readonly SELECTED_NODE="$PREFERRED_NODE"
elif ((fallback_slots >= 1)); then
  readonly SELECTED_NODE="$FALLBACK_NODE"
else
  readonly SELECTED_NODE="$PREFERRED_NODE"
  echo "No immediate slot on $PREFERRED_NODE or $FALLBACK_NODE; queueing on $PREFERRED_NODE." >&2
fi

if [[ "$DRY_RUN" == false ]]; then
  mkdir "$STUDY_ROOT"
fi

export_spec="ALL,EXPECTED_ACTION_MODES_SHA=$EXPECTED_ACTION_MODES_SHA,XQC_BASELINE_RESULTS_ROOT=$XQC_BASELINE_RESULTS_ROOT,XQC_BASELINE_RUN_ID=$XQC_BASELINE_RUN_ID,XQC_BASELINE_PYTHON=$XQC_BASELINE_PYTHON"
cd "$PROJECT_DIR"
submit_command=(
  sbatch
  --parsable
  --nodelist="$SELECTED_NODE"
  --output="$STUDY_ROOT/slurm-%x-%j.out"
  --error="$STUDY_ROOT/slurm-%x-%j.err"
  --export="$export_spec"
  "$LAUNCHER"
)

printf 'Selected node: %s (free slots: %s preferred, %s fallback)\n' \
  "$SELECTED_NODE" "$preferred_slots" "$fallback_slots"
printf 'Submit command:'
printf ' %q' "${submit_command[@]}"
printf '\n'
if [[ "$DRY_RUN" == false ]]; then
  job_id="$("${submit_command[@]}")"
  echo "Submitted vanilla XQC Humanoid Walk job $job_id on $SELECTED_NODE."
  echo "Study results: $STUDY_ROOT"
fi
