#!/usr/bin/env bash
#SBATCH --job-name=aux-horizon-mppi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the synchronized source checkout}"
: "${EXPECTED_ACTION_MODES_SHA:?}"
: "${AMBI_CAMPAIGN:?Absolute preparation.json path}"
: "${AMBI_TASK_KIND:?prior or mppi}"
: "${AMBI_TASK_COUNT:?Number of tasks of this kind}"
: "${AMBI_RUNTIME_ROOT:?Scratch directory for temporary data}"
: "${SLURM_ARRAY_TASK_ID:?}"
[[ "$AMBI_CAMPAIGN" == /* && "$AMBI_RUNTIME_ROOT" == /oscar/scratch/rgao48/* ]]
[[ "$AMBI_TASK_KIND" == prior || "$AMBI_TASK_KIND" == mppi ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
case "${1:-}" in ''|--smoke) ;; *) exit 2 ;; esac
group_size="${AMBI_GROUP_SIZE:-1}"
[[ "$group_size" =~ ^[1-9][0-9]*$ && "$AMBI_TASK_COUNT" =~ ^[1-9][0-9]*$ ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]]
first=$((SLURM_ARRAY_TASK_ID * group_size))
[[ "$first" -lt "$AMBI_TASK_COUNT" ]]

PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]]
runtime="$AMBI_RUNTIME_ROOT/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$runtime"/{tmp,cache,wandb,wandb-cache,wandb-data,torch}
export TMPDIR="$runtime/tmp" XDG_CACHE_HOME="$runtime/cache" TORCH_HOME="$runtime/torch"
export WANDB_DIR="$runtime/wandb" WANDB_CACHE_DIR="$runtime/wandb-cache" WANDB_DATA_DIR="$runtime/wandb-data"
export WANDB_MODE=disabled
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TORCHINDUCTOR_COMPILE_THREADS=1
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
options=()
if [[ "${1:-}" == --smoke ]]; then options+=(--smoke); fi
for ((index=first; index<first+group_size && index<AMBI_TASK_COUNT; index++)); do
    "$PYTHON_BIN" -m slurm.ambi_aux_horizon_mppi_campaign worker \
        --campaign "$AMBI_CAMPAIGN" --kind "$AMBI_TASK_KIND" --index "$index" "${options[@]}"
done
