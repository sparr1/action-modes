#!/usr/bin/env bash
#SBATCH --job-name=q10-walk-pathwise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus
#SBATCH --array=0-1
#SBATCH --no-requeue

set -Eeuo pipefail
umask 077
: "${AMBI_EXPECTED_COMMIT:?Set the tested evaluator commit.}"
: "${AMBI_DMCONTROL_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EVAL_ROOT:?Set a fresh campaign directory outside the checkout.}"
: "${AMBI_EVAL_CHECKPOINT:?Set the pinned Q10 checkpoint path.}"
: "${SLURM_JOB_ID:?Submit this script through Slurm.}"
: "${SLURM_CPUS_PER_TASK:?Request CPUs for the evaluator.}"

case "${SLURM_ARRAY_TASK_ID:-}" in
    0) arm=pathwise ;;
    1) arm=detached ;;
    *) echo "Use array task 0 for pathwise and 1 for detached." >&2; exit 2 ;;
esac

[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_EVAL_ROOT" == /* && "$AMBI_EVAL_CHECKPOINT" == /* && "$AMBI_DMCONTROL_PYTHON" == /* ]]
[[ -x "$AMBI_DMCONTROL_PYTHON" ]]
root="$(realpath -m "$AMBI_EVAL_ROOT")"
[[ "$root" != "$(pwd -P)" && "$root" != "$(pwd -P)/"* ]]
arm_dir="$root/$arm"
budget=()
case "${AMBI_EVAL_SMOKE:-0}" in
    0) ;;
    1) arm_dir="$root/smoke/$arm"; budget=(--num-episodes 1 --max-steps 8) ;;
    *) echo "AMBI_EVAL_SMOKE must be 0 or 1." >&2; exit 2 ;;
esac

sidecar="$AMBI_EVAL_CHECKPOINT.metadata.json"
printf '%s  %s\n' b4acfbf81ffb3c8315d8b71ca900b10ffaf69e8a5a934e6f7b5e94f4ff8b3e57 "$sidecar" |
    sha256sum --check --status
mkdir -p "$(dirname "$arm_dir")"
mkdir "$arm_dir"
mkdir "$arm_dir/tmp" "$arm_dir/cache"
export TMPDIR="$arm_dir/tmp" XDG_CACHE_HOME="$arm_dir/cache"
export CUDA_CACHE_PATH="$arm_dir/cache/cuda"
export WANDB_MODE=disabled MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE

exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" -B evaluate_ambi_checkpoint.py \
    --matrix configs/research/ambi_humanoid_walk_q10_pathwise_eval_consecutive.json \
    --checkpoint "$AMBI_EVAL_CHECKPOINT" --preset "controller/$arm" --device cuda \
    --bundle-dir "$arm_dir/bundle" --output "$arm_dir/results.json" "${budget[@]}"
