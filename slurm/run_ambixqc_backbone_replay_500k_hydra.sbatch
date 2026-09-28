#!/usr/bin/env bash
#SBATCH --job-name=ambixqc-backbone-replay-500k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2501
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/tmp/ambixqc-backbone-replay-500k-%j.out
#SBATCH --error=/tmp/ambixqc-backbone-replay-500k-%j.err

set -Eeuo pipefail
umask 077

# Submit smoke and production as arrays 0-5%6 on gpu2501. Each task owns one
# cell. Aggregate completed smokes with the helper --mode summarize before
# production, which requires the immutable smoke-gate.json from this commit.
readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed commit SHA.}"
: "${AMBIXQC_ACTION_MODES_DIR:?Export the clean checkout path.}"
: "${AMBIXQC_RESULTS_ROOT:?Export an existing durable directory outside the checkout.}"
: "${AMBIXQC_PYTHON:?Export the existing locked DMControl Python path.}"
: "${SLURM_JOB_ID:?Run this launcher in an allocated Slurm job.}"

fail() { echo "AMBI-XQC backbone replay error: $*" >&2; exit 2; }
readonly MODE="${AMBIXQC_MODE:-production}"
readonly CELLS=(baseline_utd1 aux_shared_utd1 aux_detached_utd1 baseline_utd2 aux_shared_utd2 aux_detached_utd2)
: "${SLURM_ARRAY_TASK_ID:?Submit a six-cell array with --array=0-5%6.}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-5]$ ]] || fail "array task must be 0 through 5"
readonly ARM="${CELLS[$SLURM_ARRAY_TASK_ID]}"
[[ "${SLURM_NODELIST:-}" == gpu2501 ]] || fail "all jobs must run on gpu2501"
[[ "$(hostname -s)" == gpu2501 ]] || fail "runtime host must be gpu2501"
[[ "$MODE" == smoke || "$MODE" == production ]] || fail "mode must be smoke or production"
if [[ "$MODE" == production ]]; then
  : "${AMBIXQC_SMOKE_GATE:?Export the complete smoke-gate.json path.}"
  [[ "$AMBIXQC_SMOKE_GATE" == /* && -f "$AMBIXQC_SMOKE_GATE" ]] || fail "smoke gate must be an existing absolute file"
fi
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail "expected SHA must be full lowercase hexadecimal"
[[ "$AMBIXQC_ACTION_MODES_DIR" == /* && -d "$AMBIXQC_ACTION_MODES_DIR" ]] || fail "checkout must be an existing absolute path"
[[ "$AMBIXQC_RESULTS_ROOT" == /* && -d "$AMBIXQC_RESULTS_ROOT" ]] || fail "results root must be an existing absolute directory"
[[ "$AMBIXQC_PYTHON" == /* && -x "$AMBIXQC_PYTHON" ]] || fail "Python must be executable at an absolute path"
readonly PROJECT_DIR="$(cd "$AMBIXQC_ACTION_MODES_DIR" && pwd -P)"
readonly RESULTS_ROOT="$(cd "$AMBIXQC_RESULTS_ROOT" && pwd -P)"
readonly ENV_PROJECT="$(cd "$(dirname "$AMBIXQC_PYTHON")/../.." && pwd -P)"
readonly PYTHON="$AMBIXQC_PYTHON"
readonly TASK_SUFFIX="${SLURM_ARRAY_TASK_ID:-single}"
readonly JOB_ROOT="$RESULTS_ROOT/$MODE-$ARM-seed55-job$SLURM_JOB_ID-task$TASK_SUFFIX"
readonly CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/ambixqc-backbone-replay-500k-$SLURM_JOB_ID-$TASK_SUFFIX"
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
required_gib="${AMBIXQC_MIN_FREE_GIB:-8}"
[[ "$required_gib" =~ ^[1-9][0-9]*$ ]] || fail "minimum free GiB must be a positive integer"
[[ "$available_kb" =~ ^[0-9]+$ ]] || fail "could not determine durable free storage"
((available_kb >= required_gib * 1024 * 1024)) || fail "insufficient durable free space ($required_gib GiB required)"
mkdir "$JOB_ROOT" "$CACHE_ROOT"
mkdir -p "$CACHE_ROOT/wandb/cache" "$CACHE_ROOT/wandb/data" "$CACHE_ROOT/wandb/artifacts" \
  "$CACHE_ROOT/wandb/config" "$CACHE_ROOT/xdg" "$CACHE_ROOT/torchinductor" \
  "$CACHE_ROOT/triton" "$CACHE_ROOT/cuda" "$CACHE_ROOT/tmp" "$CACHE_ROOT/matplotlib"
exec > >(tee "$JOB_ROOT/job.log") 2>&1
trap 'status=$?; if ((status != 0)); then printf "FAILED exit=%s\n" "$status" > "$JOB_ROOT/FAILED"; fi' EXIT
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export WANDB_DIR="$CACHE_ROOT/wandb" WANDB_CACHE_DIR="$CACHE_ROOT/wandb/cache"
export WANDB_DATA_DIR="$CACHE_ROOT/wandb/data" WANDB_ARTIFACT_DIR="$CACHE_ROOT/wandb/artifacts"
export WANDB_CONFIG_DIR="$CACHE_ROOT/wandb/config" WANDB_CONFIG_PATHS=""
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" XDG_CONFIG_HOME="$CACHE_ROOT/xdg"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda" MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export TMPDIR="$CACHE_ROOT/tmp" WANDB_DISABLE_CODE=true
unset TDMPC2_EVAL_CSV TDMPC2_COMPUTE_TIMING_OUTPUT WANDB_NAME WANDB_RUN_ID WANDB_RESUME WANDB_DISABLED
if [[ "$MODE" == smoke ]]; then
  export WANDB_MODE=disabled WANDB_DISABLED=true
else
  export WANDB_MODE=online WANDB_RESUME=never
  export WANDB_RUN_ID="axqc-500k-$ARM-${EXPECTED_ACTION_MODES_SHA:0:8}-$SLURM_JOB_ID-$TASK_SUFFIX"
fi
echo "Mode: $MODE; arm: $ARM; job: $SLURM_JOB_ID; node: ${SLURM_NODELIST:-unknown}"
echo "Checkout: $PROJECT_DIR @ $EXPECTED_ACTION_MODES_SHA"
echo "Runtime: $PYTHON; artifacts: $JOB_ROOT"
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
  if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
    CUBLAS_WORKSPACE_CONFIG=:4096:8 AMBI_XQC_REQUIRE_CUDA=1 \
    srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" -m pytest \
      -q -rs -p no:cacheprovider tests/test_ambixqc_backbone_replay_500k_hydra.py \
      tests/test_ambixqc_update_ratio.py tests/test_tdmpc2_replay_capacity.py \
      tests/test_ambixqc_aux_training.py tests/test_ambixqc_aux_checkpoint.py \
      tests/test_ambixqc_replay_archive.py tests/test_checkpoint_replay_publication.py \
      tests/test_replay_archive.py | tee "$JOB_ROOT/pytest.log"
  fi
  srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" \
    tests/benchmarks/ambixqc_backbone_replay_500k_campaign.py --mode smoke --arm "$ARM" \
    --output "$JOB_ROOT/run-$ARM" \
    --run-name "AMBIXQC-humanoid-walk-backbone-replay-500k-$ARM-seed55-${EXPECTED_ACTION_MODES_SHA:0:8}-smoke-job$SLURM_JOB_ID"
  [[ -f "$JOB_ROOT/run-$ARM/validation.json" ]] || fail "missing smoke cell validation"
else
  srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" \
    tests/benchmarks/ambixqc_backbone_replay_500k_campaign.py --mode production --arm "$ARM" \
    --output "$JOB_ROOT/run-$ARM" --smoke-gate "$AMBIXQC_SMOKE_GATE" \
    --run-name "AMBIXQC-humanoid-walk-backbone-replay-500k-$ARM-seed55-${EXPECTED_ACTION_MODES_SHA:0:8}-job$SLURM_JOB_ID"
  [[ -f "$JOB_ROOT/run-$ARM/validation.json" ]] || fail "missing final production validation"
fi
require_clean_sha
printf 'PASS\n' > "$JOB_ROOT/PASS"
echo "AMBI-XQC backbone replay $MODE passed. Artifacts: $JOB_ROOT"
