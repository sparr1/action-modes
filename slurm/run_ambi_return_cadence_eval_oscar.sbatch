#!/usr/bin/env bash
#SBATCH --job-name=ambi-return-cadence-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --no-requeue

# GPU, CPU, memory, walltime and concurrency are explicit measured submission choices.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${SLURM_CPUS_PER_TASK:?Specify measured CPU resources.}"
: "${AMBI_PROJECT_DIR:?Set the clean source checkout.}"
: "${AMBI_PYTHON:?Set the existing locked interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the tested commit.}"
: "${AMBI_EVAL_ROOT:?Set the campaign directory outside source.}"
: "${AMBI_EVAL_INVENTORY:?Set the frozen checkpoint inventory.}"
: "${AMBI_EVAL_REFERENCE_ROOT:?Set the completed original campaign root.}"
mode="${AMBI_EVAL_MODE:-run}"
index="${SLURM_ARRAY_TASK_ID:-${AMBI_EVAL_INDEX:?Set the entry index.}}"
[[ "$mode" == run || "$mode" == smoke ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_EVAL_ROOT" == /* && "$AMBI_EVAL_INVENTORY" == /* ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_EVAL_ROOT" != "$PWD" && "$AMBI_EVAL_ROOT" != "$PWD/"* ]]
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-return-cadence-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/tmp"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root/tmp"
options=(--reference-root "$AMBI_EVAL_REFERENCE_ROOT" --inventory "$AMBI_EVAL_INVENTORY" --index "$index" --expected-commit "$AMBI_EXPECTED_COMMIT")
if [[ "$mode" == smoke ]]; then
  options+=(--output "$AMBI_EVAL_ROOT/smoke/cell-$index" --smoke-steps "${AMBI_EVAL_SMOKE_STEPS:-6}")
else
  options+=(--output "$AMBI_EVAL_ROOT/production/cell-$index" --eval-run-map "$AMBI_EVAL_ROOT/publication/cell-$index/eval-run-map.json")
fi
exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_return_cadence_eval.py "$mode" "${options[@]}"
