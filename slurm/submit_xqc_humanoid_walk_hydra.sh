#!/usr/bin/env bash

set -Eeuo pipefail

readonly OFFICIAL_XQC_SHA="9a6832bb742ef01bbe9f1e06153a9338e612dae5"
readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly OFFICIAL_LOCK_SHA="bda38deffad85326e41382b44e06f2a2fc21396210a3232e17800bbaabf7bf85"
readonly PREFERRED_NODE="gpu2301"
readonly FALLBACK_NODE="gpu2201"
readonly JOB_CPUS=8
readonly JOB_MEMORY_MIB=32768
readonly JOB_COUNT=4
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
readonly LAUNCHER="$SCRIPT_DIR/run_xqc_humanoid_walk_hydra.sbatch"

EXPECTED_ACTION_MODES_SHA=""
XQC_OFFICIAL_DIR=""
XQC_RESULTS_ROOT=""
XQC_COMPARISON_ID=""
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  slurm/submit_xqc_humanoid_walk_hydra.sh \
    --expected-action-modes-sha SHA \
    --official-dir ABSOLUTE_PATH \
    --results-root ABSOLUTE_PATH \
    --comparison-id UNIQUE_ID [--dry-run]

Submits exactly four independent one-GPU jobs: official/action x seeds 0/1.
The helper prefers gpu2301 when it has four free job slots, otherwise gpu2201.
EOF
}

while (($#)); do
  case "$1" in
    --expected-action-modes-sha)
      EXPECTED_ACTION_MODES_SHA="${2:-}"
      shift 2
      ;;
    --official-dir)
      XQC_OFFICIAL_DIR="${2:-}"
      shift 2
      ;;
    --results-root)
      XQC_RESULTS_ROOT="${2:-}"
      shift 2
      ;;
    --comparison-id)
      XQC_COMPARISON_ID="${2:-}"
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
if [[ "$XQC_OFFICIAL_DIR" != /* || ! -d "$XQC_OFFICIAL_DIR" ]]; then
  echo "--official-dir must be an existing absolute directory." >&2
  exit 1
fi
if [[ "$XQC_RESULTS_ROOT" != /* || ! -d "$XQC_RESULTS_ROOT" ]]; then
  echo "--results-root must be an existing absolute directory." >&2
  exit 1
fi
if [[ ! -w "$XQC_RESULTS_ROOT" ]]; then
  echo "--results-root is not writable: $XQC_RESULTS_ROOT" >&2
  exit 1
fi
if [[ ! "$XQC_COMPARISON_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "--comparison-id must use only letters, digits, '.', '_', and '-'." >&2
  exit 1
fi
if [[ ! -f "$LAUNCHER" ]]; then
  echo "Missing launcher: $LAUNCHER" >&2
  exit 1
fi

for exported_value in "$XQC_OFFICIAL_DIR" "$XQC_RESULTS_ROOT" "$XQC_COMPARISON_ID"; do
  if [[ "$exported_value" == *','* || "$exported_value" == *$'\n'* ]]; then
    echo "Exported paths and IDs cannot contain commas or newlines." >&2
    exit 1
  fi
done

XQC_OFFICIAL_DIR="$(cd "$XQC_OFFICIAL_DIR" && pwd -P)"
XQC_RESULTS_ROOT="$(cd "$XQC_RESULTS_ROOT" && pwd -P)"
readonly XQC_OFFICIAL_DIR XQC_RESULTS_ROOT
readonly COMPARISON_ROOT="$XQC_RESULTS_ROOT/$XQC_COMPARISON_ID"

require_clean_sha() {
  local checkout="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(git -C "$checkout" rev-parse HEAD)"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label commit mismatch: expected $expected, found $actual" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "$label checkout is not clean: $checkout" >&2
    git -C "$checkout" status --short >&2
    exit 1
  fi
}

require_lock_hash() {
  local lockfile="$1"
  local expected="$2"
  local label="$3"
  local actual
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

require_clean_sha "$PROJECT_DIR" "$EXPECTED_ACTION_MODES_SHA" "Action Modes"
require_clean_sha "$XQC_OFFICIAL_DIR" "$OFFICIAL_XQC_SHA" "Official XQC"
require_lock_hash "$PROJECT_DIR/environments/dmcontrol/uv.lock" "$ACTION_LOCK_SHA" "Action Modes"
require_lock_hash "$XQC_OFFICIAL_DIR/uv.lock" "$OFFICIAL_LOCK_SHA" "Official XQC"

if [[ -e "$COMPARISON_ROOT" || -L "$COMPARISON_ROOT" ]]; then
  echo "Refusing to reuse comparison artifacts: $COMPARISON_ROOT" >&2
  exit 1
fi

preferred_slots="$(node_job_slots "$PREFERRED_NODE")"
fallback_slots="$(node_job_slots "$FALLBACK_NODE")"
readonly preferred_slots fallback_slots
declare -a JOB_NODES=()

if ((preferred_slots >= JOB_COUNT)); then
  JOB_NODES=("$PREFERRED_NODE" "$PREFERRED_NODE" "$PREFERRED_NODE" "$PREFERRED_NODE")
elif ((fallback_slots >= JOB_COUNT)); then
  JOB_NODES=("$FALLBACK_NODE" "$FALLBACK_NODE" "$FALLBACK_NODE" "$FALLBACK_NODE")
elif ((preferred_slots + fallback_slots >= JOB_COUNT)); then
  for ((index = 0; index < JOB_COUNT; index++)); do
    if ((index < preferred_slots)); then
      JOB_NODES+=("$PREFERRED_NODE")
    else
      JOB_NODES+=("$FALLBACK_NODE")
    fi
  done
else
  echo "Need four concurrent Hydra slots; $PREFERRED_NODE has $preferred_slots and $FALLBACK_NODE has $fallback_slots." >&2
  exit 1
fi

readonly -a RUN_IMPLEMENTATIONS=(official official action action)
readonly -a RUN_SEEDS=(0 1 0 1)
declare -a SUBMITTED_JOB_IDS=()

if [[ "$DRY_RUN" == false ]]; then
  mkdir "$COMPARISON_ROOT"
fi

cd "$PROJECT_DIR"
for ((index = 0; index < JOB_COUNT; index++)); do
  implementation="${RUN_IMPLEMENTATIONS[$index]}"
  seed="${RUN_SEEDS[$index]}"
  node="${JOB_NODES[$index]}"
  job_name="xqc-${implementation}-s${seed}"
  export_spec="ALL,IMPLEMENTATION=$implementation,SEED=$seed,EXPECTED_ACTION_MODES_SHA=$EXPECTED_ACTION_MODES_SHA,XQC_OFFICIAL_DIR=$XQC_OFFICIAL_DIR,XQC_RESULTS_ROOT=$XQC_RESULTS_ROOT,XQC_COMPARISON_ID=$XQC_COMPARISON_ID"
  submit_command=(
    sbatch
    --parsable
    --job-name="$job_name"
    --nodelist="$node"
    --output="$COMPARISON_ROOT/slurm-%x-%j.out"
    --error="$COMPARISON_ROOT/slurm-%x-%j.err"
    --export="$export_spec"
    "$LAUNCHER"
  )

  if [[ "$DRY_RUN" == true ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${submit_command[@]}"
    printf '\n'
    continue
  fi

  submission="$("${submit_command[@]}")"
  job_id="${submission%%;*}"
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Unexpected sbatch response for $implementation seed $seed: $submission" >&2
    exit 1
  fi
  SUBMITTED_JOB_IDS+=("$job_id")
  echo "Submitted $implementation seed $seed as job $job_id on $node"
done

if [[ "$DRY_RUN" == false ]]; then
  printf 'Submitted comparison %s job IDs:' "$XQC_COMPARISON_ID"
  printf ' %s' "${SUBMITTED_JOB_IDS[@]}"
  printf '\nArtifacts: %s\n' "$COMPARISON_ROOT"
fi
