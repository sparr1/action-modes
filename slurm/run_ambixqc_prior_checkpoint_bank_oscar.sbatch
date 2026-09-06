#!/usr/bin/env bash
#SBATCH --job-name=ambixqc-prior-bank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --output=/tmp/ambixqc-prior-bank-%j.out
#SBATCH --error=/tmp/ambixqc-prior-bank-%j.err

set -Eeuo pipefail
umask 077

# Set AMBIXQC_MODE=smoke and override sbatch --time=02:00:00 for the GPU gate.
# Submit production only after that SHA's smoke validation.json passes.
readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed commit SHA.}"
: "${AMBIXQC_ACTION_MODES_DIR:?Export the clean checkout path.}"
: "${AMBIXQC_RESULTS_ROOT:?Export an existing durable directory outside the checkout.}"
: "${AMBIXQC_PYTHON:?Export the existing locked DMControl Python path.}"
: "${SLURM_JOB_ID:?Run this launcher in an allocated Slurm job.}"

fail() { echo "AMBI-XQC prior-bank error: $*" >&2; exit 2; }
readonly MODE="${AMBIXQC_MODE:-production}"
[[ "$MODE" == smoke || "$MODE" == production ]] || fail "mode must be smoke or production"
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail "expected SHA must be full lowercase hexadecimal"
[[ "$AMBIXQC_ACTION_MODES_DIR" == /* && -d "$AMBIXQC_ACTION_MODES_DIR" ]] || fail "checkout must be an existing absolute path"
[[ "$AMBIXQC_RESULTS_ROOT" == /* && -d "$AMBIXQC_RESULTS_ROOT" ]] || fail "results root must be an existing absolute directory"
[[ "$AMBIXQC_PYTHON" == /* && -x "$AMBIXQC_PYTHON" ]] || fail "Python must be executable at an absolute path"

readonly PROJECT_DIR="$(cd "$AMBIXQC_ACTION_MODES_DIR" && pwd -P)"
readonly RESULTS_ROOT="$(cd "$AMBIXQC_RESULTS_ROOT" && pwd -P)"
readonly ENV_PROJECT="$(cd "$(dirname "$AMBIXQC_PYTHON")/../.." && pwd -P)"
readonly PYTHON="$AMBIXQC_PYTHON"
readonly JOB_ROOT="$RESULTS_ROOT/$MODE-seed55-job$SLURM_JOB_ID"
readonly CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/ambixqc-prior-bank-$SLURM_JOB_ID"
readonly RUN_NAME="AMBIXQC-humanoid-walk-prior-only-no-inner-checkpoint-bank-1p5m-seed55-${EXPECTED_ACTION_MODES_SHA:0:8}-$MODE-job$SLURM_JOB_ID"
[[ "$PYTHON" == "$ENV_PROJECT/.venv/bin/python" ]] || fail "Python must belong to a nested locked .venv"
case "$RESULTS_ROOT/" in "$PROJECT_DIR/"*) fail "results must be outside the checkout" ;; esac
[[ ! -e "$JOB_ROOT" && ! -L "$JOB_ROOT" ]] || fail "refusing to overwrite job artifacts"
[[ ! -e "$CACHE_ROOT" && ! -L "$CACHE_ROOT" ]] || fail "refusing to reuse node-local caches"

require_clean_sha() {
  [[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "checkout SHA mismatch"
  [[ -z "$(git -C "$PROJECT_DIR" status --porcelain=v1 --untracked-files=all)" ]] || fail "checkout must be clean"
}
require_lock() {
  [[ -f "$1" ]] || fail "missing runtime lock: $1"
  [[ "$(sha256sum "$1" | awk '{print $1}')" == "$ACTION_LOCK_SHA" ]] || fail "locked environment mismatch: $1"
}
for command in git sha256sum nvidia-smi srun; do
  command -v "$command" >/dev/null || fail "missing command: $command"
done
require_clean_sha
require_lock "$PROJECT_DIR/environments/dmcontrol/uv.lock"
require_lock "$ENV_PROJECT/uv.lock"
available_kb="$(df -Pk "$RESULTS_ROOT" | awk 'NR == 2 {print $4}')"
required_gib=32
[[ "$MODE" != smoke ]] || required_gib=4
((available_kb >= required_gib * 1024 * 1024)) || fail "insufficient durable free space ($required_gib GiB required)"

mkdir "$JOB_ROOT" "$CACHE_ROOT"
mkdir -p "$CACHE_ROOT/wandb/cache" "$CACHE_ROOT/wandb/data" "$CACHE_ROOT/wandb/artifacts" \
  "$CACHE_ROOT/xdg" "$CACHE_ROOT/torchinductor" "$CACHE_ROOT/triton" "$CACHE_ROOT/cuda" "$CACHE_ROOT/tmp"
exec > >(tee "$JOB_ROOT/job.log") 2>&1
trap 'status=$?; if ((status != 0)); then printf "FAILED exit=%s\n" "$status" > "$JOB_ROOT/FAILED"; fi' EXIT

export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export WANDB_DIR="$CACHE_ROOT/wandb" WANDB_CACHE_DIR="$CACHE_ROOT/wandb/cache"
export WANDB_DATA_DIR="$CACHE_ROOT/wandb/data" WANDB_ARTIFACT_DIR="$CACHE_ROOT/wandb/artifacts"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton" CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export TMPDIR="$CACHE_ROOT/tmp" WANDB_DISABLE_CODE=true
unset TDMPC2_EVAL_CSV TDMPC2_COMPUTE_TIMING_OUTPUT WANDB_NAME WANDB_RUN_ID WANDB_RESUME WANDB_DISABLED
if [[ "$MODE" == smoke ]]; then
  export WANDB_MODE=disabled WANDB_DISABLED=true
else
  export WANDB_MODE=online WANDB_RESUME=never
  export WANDB_RUN_ID="axqc-prior-${EXPECTED_ACTION_MODES_SHA:0:8}-$SLURM_JOB_ID"
fi

echo "Mode: $MODE; job: $SLURM_JOB_ID; node: ${SLURM_NODELIST:-unknown}"
echo "Checkout: $PROJECT_DIR @ $EXPECTED_ACTION_MODES_SHA"
echo "Runtime: $PYTHON; durable artifacts: $JOB_ROOT"
echo "W&B name: $RUN_NAME; project: rwgao_b-brown-university/ambi"
git -C "$PROJECT_DIR" show --no-patch --format=fuller HEAD
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader | tee "$JOB_ROOT/gpu.txt"
cd "$PROJECT_DIR"
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" -c '
import json, sys, torch
from importlib.metadata import version
from pathlib import Path
assert sys.version_info[:2] == (3, 10)
assert torch.__version__.split("+")[0] == "2.3.1"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert "L40S" in torch.cuda.get_device_name(0)
assert torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled()
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve() / ".venv"
assert torch.ones(1, device="cuda").sum().item() == 1
print(json.dumps({"python": sys.version, "prefix": sys.prefix, "cuda": torch.version.cuda,
"gpu": torch.cuda.get_device_name(0), "versions": {k: version(k) for k in
("torch", "dm-control", "mujoco", "gymnasium", "numpy", "torchrl", "wandb")}}, sort_keys=True))
' "$ENV_PROJECT" | tee "$JOB_ROOT/runtime.json"

if [[ "$MODE" == smoke ]]; then
  AMBI_XQC_TEST_DEVICE=cuda AMBI_XQC_REQUIRE_CUDA=1 \
    srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" -m pytest \
      -q -rs -p no:cacheprovider tests/test_ambixqc_core.py tests/test_ambixqc_controller.py \
      tests/test_ambixqc_prior_checkpoint.py tests/test_ambixqc_checkpoint_evaluation.py \
      | tee "$JOB_ROOT/pytest.log"
fi
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" \
  tests/benchmarks/ambixqc_prior_checkpoint_bank.py --mode "$MODE" \
  --output "$JOB_ROOT/run" --run-name "$RUN_NAME"
[[ -f "$JOB_ROOT/run/validation.json" ]] || fail "missing final validation"
require_clean_sha
printf 'PASS\n' > "$JOB_ROOT/PASS"
echo "AMBI-XQC prior-bank $MODE passed. Artifacts: $JOB_ROOT"
