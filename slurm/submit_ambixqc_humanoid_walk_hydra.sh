#!/usr/bin/env bash

set -Eeuo pipefail

readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly PREFERRED_NODE="gpu2301"
readonly FALLBACK_NODE="gpu2201"
readonly JOB_CPUS=8
readonly JOB_MEMORY_MIB=49152
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
readonly LAUNCHER="$SCRIPT_DIR/run_ambixqc_humanoid_walk_hydra.sbatch"

EXPECTED_ACTION_MODES_SHA=""
AMBIXQC_RESULTS_ROOT=""
AMBIXQC_RUN_ID=""
AMBIXQC_PYTHON="$PROJECT_DIR/environments/dmcontrol/.venv/bin/python"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  slurm/submit_ambixqc_humanoid_walk_hydra.sh \
    --expected-action-modes-sha SHA \
    --results-root ABSOLUTE_PATH \
    --run-id UNIQUE_ID \
    [--python ABSOLUTE_PATH] [--dry-run]

Submits exactly one seed-55, one-GPU, one-million-decision AMBI-XQC
Humanoid Walk job. The helper prefers a free slot on gpu2301, falls back to
gpu2201, and queues on gpu2301 if neither node has an immediate free slot.
EOF
}

while (($#)); do
  case "$1" in
    --expected-action-modes-sha)
      EXPECTED_ACTION_MODES_SHA="${2:-}"
      shift 2
      ;;
    --results-root)
      AMBIXQC_RESULTS_ROOT="${2:-}"
      shift 2
      ;;
    --run-id)
      AMBIXQC_RUN_ID="${2:-}"
      shift 2
      ;;
    --python)
      AMBIXQC_PYTHON="${2:-}"
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
if [[ "$AMBIXQC_RESULTS_ROOT" != /* || ! -d "$AMBIXQC_RESULTS_ROOT" ]]; then
  echo "--results-root must be an existing absolute directory." >&2
  exit 1
fi
if [[ ! -w "$AMBIXQC_RESULTS_ROOT" ]]; then
  echo "--results-root is not writable: $AMBIXQC_RESULTS_ROOT" >&2
  exit 1
fi
if [[ ! "$AMBIXQC_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "--run-id must use only letters, digits, '.', '_', and '-'." >&2
  exit 1
fi
if ((${#AMBIXQC_RUN_ID} > 64)); then
  echo "--run-id must contain at most 64 characters." >&2
  exit 1
fi
if [[ "$AMBIXQC_PYTHON" != /* || ! -x "$AMBIXQC_PYTHON" ]]; then
  echo "--python must be an executable at an absolute path: $AMBIXQC_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$LAUNCHER" ]]; then
  echo "Missing launcher: $LAUNCHER" >&2
  exit 1
fi
for exported_value in "$AMBIXQC_RESULTS_ROOT" "$AMBIXQC_RUN_ID" "$AMBIXQC_PYTHON"; do
  if [[ "$exported_value" == *','* || "$exported_value" == *$'\n'* ]]; then
    echo "Exported paths and IDs cannot contain commas or newlines." >&2
    exit 1
  fi
done

AMBIXQC_RESULTS_ROOT="$(cd "$AMBIXQC_RESULTS_ROOT" && pwd -P)"
readonly AMBIXQC_RESULTS_ROOT
readonly STUDY_ROOT="$AMBIXQC_RESULTS_ROOT/$AMBIXQC_RUN_ID"

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
  local actual
  actual="$(sha256sum "$lockfile" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "DMControl lock hash mismatch: expected $expected, found $actual" >&2
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

require_clean_sha "$PROJECT_DIR" "$EXPECTED_ACTION_MODES_SHA"
require_lock_hash "$PROJECT_DIR/environments/dmcontrol/uv.lock" "$ACTION_LOCK_SHA"

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

export_spec="ALL,EXPECTED_ACTION_MODES_SHA=$EXPECTED_ACTION_MODES_SHA,AMBIXQC_RESULTS_ROOT=$AMBIXQC_RESULTS_ROOT,AMBIXQC_RUN_ID=$AMBIXQC_RUN_ID,AMBIXQC_PYTHON=$AMBIXQC_PYTHON"
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
  echo "Submitted AMBI-XQC Humanoid Walk job $job_id on $SELECTED_NODE."
  echo "Study results: $STUDY_ROOT"
fi
