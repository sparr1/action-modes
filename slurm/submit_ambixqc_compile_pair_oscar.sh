#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd -P)"
readonly LAUNCHER="$SCRIPT_DIR/run_ambixqc_compile_pair_oscar.sbatch"

EXPECTED_ACTION_MODES_SHA=""
AMBIXQC_BENCHMARK_RESULTS_ROOT=""
AMBIXQC_PYTHON="$PROJECT_DIR/environments/dmcontrol/.venv/bin/python"
AMBIXQC_PAIR_ORDER="eager-first"
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  slurm/submit_ambixqc_compile_pair_oscar.sh \
    --expected-action-modes-sha SHA \
    --results-root ABSOLUTE_PATH \
    [--python ABSOLUTE_PATH] \
    [--order eager-first|compiled-first] \
    [--dry-run]

Submits one Oscar L40S job. The job runs eager and strict-compiled exact-shape
compute benchmarks plus the paired 1,502-step Humanoid Walk timing canaries
sequentially on the same GPU. The helper never updates the checkout: first
transport the tested commit through Git and invoke this from that clean commit.
EOF
}

fail() {
  echo "AMBI-XQC Oscar paired benchmark submission error: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --expected-action-modes-sha)
      EXPECTED_ACTION_MODES_SHA="${2:-}"
      shift 2
      ;;
    --results-root)
      AMBIXQC_BENCHMARK_RESULTS_ROOT="${2:-}"
      shift 2
      ;;
    --python)
      AMBIXQC_PYTHON="${2:-}"
      shift 2
      ;;
    --order)
      AMBIXQC_PAIR_ORDER="${2:-}"
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

for command in git sha256sum sbatch; do
  command -v "$command" >/dev/null 2>&1 || fail "missing required command: $command"
done
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail \
  "--expected-action-modes-sha must be a full lowercase 40-character SHA"
[[ "$AMBIXQC_BENCHMARK_RESULTS_ROOT" == /* ]] || fail \
  "--results-root must be absolute"
[[ -d "$AMBIXQC_BENCHMARK_RESULTS_ROOT" ]] || fail \
  "--results-root must be an existing directory"
[[ ! -L "$AMBIXQC_BENCHMARK_RESULTS_ROOT" ]] || fail \
  "--results-root must not be a symlink"
[[ -w "$AMBIXQC_BENCHMARK_RESULTS_ROOT" ]] || fail \
  "--results-root is not writable"
[[ "$AMBIXQC_PYTHON" == /* && -x "$AMBIXQC_PYTHON" ]] || fail \
  "--python must be an executable at an absolute path: $AMBIXQC_PYTHON"
[[ "$AMBIXQC_PAIR_ORDER" == eager-first || "$AMBIXQC_PAIR_ORDER" == compiled-first ]] || fail \
  "--order must be eager-first or compiled-first"
[[ -f "$LAUNCHER" ]] || fail "missing launcher: $LAUNCHER"

for value in \
  "$AMBIXQC_BENCHMARK_RESULTS_ROOT" \
  "$AMBIXQC_PYTHON" \
  "$AMBIXQC_PAIR_ORDER"; do
  if [[ "$value" == *','* || "$value" == *$'\n'* ]]; then
    fail "exported paths, IDs, and order cannot contain commas or newlines"
  fi
done

AMBIXQC_BENCHMARK_RESULTS_ROOT="$(cd "$AMBIXQC_BENCHMARK_RESULTS_ROOT" && pwd -P)"
readonly AMBIXQC_BENCHMARK_RESULTS_ROOT
readonly EXPECTED_RESULTS_ROOT="/oscar/scratch/rgao48/ambi/benchmarks/ambixqc-compile/$EXPECTED_ACTION_MODES_SHA"
case "$AMBIXQC_BENCHMARK_RESULTS_ROOT/" in
  "$PROJECT_DIR/"*) fail "durable results must be outside the Git checkout" ;;
esac
[[ "$AMBIXQC_BENCHMARK_RESULTS_ROOT" == "$EXPECTED_RESULTS_ROOT" ]] || fail \
  "--results-root must be exactly $EXPECTED_RESULTS_ROOT"
[[ ! -L "$AMBIXQC_BENCHMARK_RESULTS_ROOT" ]] || fail \
  "--results-root must not be a symlink"

actual_sha="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
[[ "$actual_sha" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail \
  "Action Modes commit mismatch: expected $EXPECTED_ACTION_MODES_SHA, found $actual_sha"
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain=v1 --untracked-files=all)" ]]; then
  git -C "$PROJECT_DIR" status --short >&2
  fail "submission requires a clean checkout"
fi
actual_lock_sha="$(sha256sum "$PROJECT_DIR/environments/dmcontrol/uv.lock" | awk '{print $1}')"
[[ "$actual_lock_sha" == "$ACTION_LOCK_SHA" ]] || fail \
  "DMControl lock mismatch: expected $ACTION_LOCK_SHA, found $actual_lock_sha"
export_spec="ALL,EXPECTED_ACTION_MODES_SHA=$EXPECTED_ACTION_MODES_SHA,AMBIXQC_BENCHMARK_RESULTS_ROOT=$AMBIXQC_BENCHMARK_RESULTS_ROOT,AMBIXQC_PYTHON=$AMBIXQC_PYTHON,AMBIXQC_PAIR_ORDER=$AMBIXQC_PAIR_ORDER"
submit_command=(
  sbatch
  --parsable
  --chdir="$PROJECT_DIR"
  --export="$export_spec"
  "$LAUNCHER"
)

printf 'Submit command:'
printf ' %q' "${submit_command[@]}"
printf '\n'
if [[ "$DRY_RUN" == false ]]; then
  job_id="$("${submit_command[@]}")"
  echo "Submitted AMBI-XQC paired benchmark job $job_id to Oscar."
  echo "Job artifacts: $AMBIXQC_BENCHMARK_RESULTS_ROOT/$job_id"
fi
