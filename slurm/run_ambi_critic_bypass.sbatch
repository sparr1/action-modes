#!/usr/bin/env bash
#SBATCH --job-name=ambi-800k-bypass
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=gpu3003,gpu3106
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/rgao48/ambi/critic-bypass-%A_%a.out
#SBATCH --error=/oscar/scratch/rgao48/ambi/critic-bypass-%A_%a.err

set -Eeuo pipefail
: "${BYPASS_CHECKOUT:?Set the synchronized checkout path}"
: "${BYPASS_CHECKPOINT:?Set the pinned checkpoint path}"
: "${BYPASS_ROOT:?Set the campaign scratch directory}"
: "${BYPASS_EXPECTED_COMMIT:?Set the tested full commit SHA}"
: "${SLURM_JOB_ID:?Run through Slurm}"
cd "$BYPASS_CHECKOUT"
[[ "$BYPASS_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git rev-parse HEAD)" == "$BYPASS_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ -f "$BYPASS_CHECKPOINT" && -f "${BYPASS_CHECKPOINT}.metadata.json" ]]
task_index="${SLURM_ARRAY_TASK_ID:-${BYPASS_TASK_INDEX:-}}"
[[ "$task_index" =~ ^[0-9]+$ ]]
[[ "${BYPASS_SMOKE:-0}" =~ ^[01]$ ]]
PYTHON_BIN=/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python
[[ -x "$PYTHON_BIN" ]]
runtime_root="$BYPASS_ROOT/runtime/${SLURM_JOB_ID}-${task_index}"
export TMPDIR="$runtime_root/tmp" XDG_CACHE_HOME="$runtime_root/cache"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor" TRITON_CACHE_DIR="$runtime_root/triton"
export WANDB_DIR="$runtime_root/wandb" WANDB_CACHE_DIR="$runtime_root/wandb-cache"
export WANDB_DATA_DIR="$runtime_root/wandb-data" WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
unset WANDB_RUN_ID WANDB_NAME WANDB_GROUP WANDB_SERVICE _WANDB_SERVICE WANDB_RESUME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
printf 'critic_bypass job=%s task=%s node=%s commit=%s tree=%s smoke=%s\n' \
  "$SLURM_JOB_ID" "$task_index" "$(hostname)" "$BYPASS_EXPECTED_COMMIT" \
  "$(git rev-parse 'HEAD^{tree}')" "${BYPASS_SMOKE:-0}"
args=(--checkpoint "$BYPASS_CHECKPOINT" --config configs/research/ambi_800k_critic_bypass.json
  --campaign-root "$BYPASS_ROOT" --task-index "$task_index" --device cuda)
if [[ "${BYPASS_SMOKE:-0}" == 1 ]]; then args+=(--smoke); fi
exec "$PYTHON_BIN" evaluate_ambi_critic_bypass.py "${args[@]}"
