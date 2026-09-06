#!/usr/bin/env bash
#SBATCH --job-name=ambixqc-inner-j6
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0
#SBATCH --no-requeue
#SBATCH --output=/tmp/ambixqc-inner-%A_%a.out
#SBATCH --error=/tmp/ambixqc-inner-%A_%a.err

set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the clean repository root.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the exact tested and pushed commit.}"
: "${CHECKPOINT_MANIFEST:?Supply the absolute checkpoint/reference manifest.}"
: "${RESULT_ROOT:?Supply an absolute output directory outside the checkout.}"
: "${SLURM_ARRAY_TASK_ID:?Missing checkpoint array index.}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
MODE="${AMBIXQC_EVAL_MODE:-production}"
VARIANT="${AMBIXQC_EVAL_VARIANT:-inner}"
[[ "$MODE" == smoke || "$MODE" == production ]] || { echo "Invalid evaluation mode" >&2; exit 2; }
[[ "$VARIANT" == inner || "$VARIANT" == outer_terminal ]] || { echo "Invalid evaluation variant" >&2; exit 2; }
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$CHECKPOINT_MANIFEST" == /* && "$RESULT_ROOT" == /* && "$PYTHON_BIN" == /* ]] || exit 2
[[ -f "$CHECKPOINT_MANIFEST" && -x "$PYTHON_BIN" ]] || exit 2
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || exit 2
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || { echo "Source checkout is dirty" >&2; exit 2; }
readonly LOCK_SHA=f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6
ENV_PROJECT="$(cd "$(dirname "$PYTHON_BIN")/../.." && pwd -P)"
[[ "$PYTHON_BIN" == "$ENV_PROJECT/.venv/bin/python" ]] || exit 2
for lock in environments/dmcontrol/uv.lock "$ENV_PROJECT/uv.lock"; do
  [[ "$(sha256sum "$lock" | awk '{print $1}')" == "$LOCK_SHA" ]] || { echo "Runtime lock mismatch" >&2; exit 2; }
done

readonly CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/ambixqc-${VARIANT}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$CACHE_ROOT" && ! -L "$CACHE_ROOT" ]] || exit 2
mkdir -p "$CACHE_ROOT/wandb/cache" "$CACHE_ROOT/wandb/data" "$CACHE_ROOT/wandb/artifacts" "$CACHE_ROOT/tmp"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DIR="$CACHE_ROOT/wandb" WANDB_CACHE_DIR="$CACHE_ROOT/wandb/cache"
export WANDB_DATA_DIR="$CACHE_ROOT/wandb/data" WANDB_ARTIFACT_DIR="$CACHE_ROOT/wandb/artifacts"
export TMPDIR="$CACHE_ROOT/tmp" WANDB_DISABLE_CODE=true
unset WANDB_NAME WANDB_RUN_ID WANDB_RESUME WANDB_DISABLED PYTEST_ADDOPTS
export WANDB_MODE=online
[[ "$MODE" != smoke ]] || export WANDB_MODE=disabled WANDB_DISABLED=true

echo "AMBI-XQC $VARIANT J6/N512/H3/G3/B512 $MODE: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}; node ${SLURM_NODELIST}; commit $EXPECTED_ACTION_MODES_SHA"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON_BIN" -c '
import sys, torch
assert sys.version_info[:2] == (3, 10)
assert torch.__version__.split("+")[0] == "2.3.1"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert "L40S" in torch.cuda.get_device_name(0)
assert torch.ones(1, device="cuda").sum().item() == 1
print("Verified locked CUDA runtime", torch.__version__, torch.cuda.get_device_name(0))
'
args=(--manifest "$CHECKPOINT_MANIFEST" --index "$SLURM_ARRAY_TASK_ID" --result-root "$RESULT_ROOT" --mode "$MODE" --variant "$VARIANT" --device cuda)
if [[ "$MODE" == smoke ]]; then
  mapfile -t smoke_reference < <("$PYTHON_BIN" -c 'import json,sys; row=json.load(open(sys.argv[1]))["checkpoints"][int(sys.argv[2])]; print(row["smoke_reference_bundle"]); print(row["smoke_reference_manifest_sha256"])' "$CHECKPOINT_MANIFEST" "$SLURM_ARRAY_TASK_ID")
  [[ "${#smoke_reference[@]}" == 2 && "${smoke_reference[0]}" == /* && "${smoke_reference[1]}" =~ ^[0-9a-f]{64}$ ]] || exit 2
  args+=(--smoke-reference-bundle "${smoke_reference[0]}" --smoke-reference-manifest-sha256 "${smoke_reference[1]}")
  srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON_BIN" -m pytest -q -rs -p no:cacheprovider \
    tests/test_ambixqc_inner_j6.py tests/test_ambixqc_inner.py tests/test_ambixqc_controller.py \
    tests/test_ambixqc_prior_checkpoint.py tests/test_ambixqc_checkpoint_evaluation.py \
    tests/test_ambixqc_outer_terminal.py tests/test_ambixqc_terminal_checkpoint.py
else
  args+=(--wandb)
fi
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON_BIN" run_ambixqc_inner_evaluation.py "${args[@]}"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || { echo "Source changed during evaluation" >&2; exit 2; }
echo "AMBI-XQC paired $VARIANT $MODE completed successfully."
