#!/usr/bin/env bash
#SBATCH --job-name=ambi-posttanh-distill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --hint=nomultithread
#SBATCH --no-requeue

# Supply one GPU, CPU count, memory and measured walltime explicitly to sbatch.
# Use the same GPU model and CPU count for gate and train. No arrays or resumes.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SLURM_CPUS_PER_TASK:?Request explicit CPU resources}"
: "${AMBI_PROJECT_DIR:?Set the clean source checkout}"
: "${AMBI_DMCONTROL_PYTHON:?Set the locked interpreter}"
: "${AMBI_EXPECTED_COMMIT:?Pin the tested source revision}"
: "${AMBI_DISTILLATION_CONFIG:?Set the versioned single-run config}"
: "${AMBI_DISTILLATION_OUTPUT:?Set a fresh output path outside source}"
: "${AMBI_DISTILLATION_DEADLINE_UTC:?Set the original completion deadline with UTC offset}"
mode="${1:?gate or train}"
[[ "$mode" == gate || "$mode" == train ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_DMCONTROL_PYTHON" == /* && "$AMBI_DISTILLATION_OUTPUT" == /* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ ! -e "$AMBI_DISTILLATION_OUTPUT" ]]
[[ "$AMBI_DISTILLATION_OUTPUT" != "$PWD" && "$AMBI_DISTILLATION_OUTPUT" != "$PWD/"* ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
local_cache="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-distill-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_cache/inductor" "$local_cache/triton" "$local_cache/cache" "$local_cache/tmp" "$local_cache/data" "$local_cache/artifacts"
export TMPDIR="$local_cache/tmp" TORCHINDUCTOR_CACHE_DIR="$local_cache/inductor" TRITON_CACHE_DIR="$local_cache/triton"
export XDG_CACHE_HOME="$local_cache/cache" WANDB_CACHE_DIR="$local_cache/cache"
export WANDB_DATA_DIR="$local_cache/data" WANDB_ARTIFACT_DIR="$local_cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
arguments=("$mode" --config "$AMBI_DISTILLATION_CONFIG" --output "$AMBI_DISTILLATION_OUTPUT"
  --deadline-utc "$AMBI_DISTILLATION_DEADLINE_UTC" --max-wall-hours "${AMBI_DISTILLATION_MAX_WALL_HOURS:-46}")
if [[ "$mode" == gate ]]; then
  export WANDB_MODE=disabled WANDB_DIR="$local_cache"
  arguments+=(--max-gate-seconds "${AMBI_DISTILLATION_MAX_GATE_SECONDS:-600}")
else
  : "${AMBI_DISTILLATION_GATE:?Set the passing exact-source gate receipt}"
  : "${AMBI_DISTILLATION_STATE:?Set the existing comparison publication state}"
  : "${WANDB_RUN_ID:?Set a new publication identity before launching}"
  if [[ -n "${AMBI_WORKSPACE_PYTHON:-}" ]]; then
    [[ "$AMBI_WORKSPACE_PYTHON" == /* && -x "$AMBI_WORKSPACE_PYTHON" ]]
    publisher=("$AMBI_WORKSPACE_PYTHON")
  else
    : "${AMBI_WORKSPACE_UV:?Set an installed uv executable for the isolated publisher}"
    : "${AMBI_WORKSPACE_UV_CACHE:?Set the existing cache containing the exact publisher packages}"
    [[ "$AMBI_WORKSPACE_UV" == /* && -x "$AMBI_WORKSPACE_UV" ]]
    [[ "$AMBI_WORKSPACE_UV_CACHE" == /* && -d "$AMBI_WORKSPACE_UV_CACHE" ]]
    publisher=("$AMBI_WORKSPACE_UV" run --offline --no-project --cache-dir "$AMBI_WORKSPACE_UV_CACHE"
      --python "$AMBI_DMCONTROL_PYTHON" --with wandb==0.28.1 --with wandb-workspaces==0.4.5 python)
  fi
  [[ "$AMBI_DISTILLATION_STATE" == /* && -f "$AMBI_DISTILLATION_STATE" ]]
  export WANDB_MODE=online WANDB_RESUME=never
  export WANDB_DIR="${AMBI_DISTILLATION_OUTPUT}.wandb-${SLURM_JOB_ID}"
  mkdir "$WANDB_DIR"
  arguments+=(--gate "$AMBI_DISTILLATION_GATE")
fi
if [[ "$mode" == gate ]]; then
  exec srun --input=/dev/null --unbuffered --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" \
    slurm/ambi_distillation_posttanh_oscar.py "${arguments[@]}"
fi
# Establish and verify the visible comparison before training; update the same
# panels after success or failure. Publication failures remain visible to Slurm.
"${publisher[@]}" slurm/ambi_distillation_posttanh_workspace.py "$AMBI_DISTILLATION_STATE"
training_status=0
srun --input=/dev/null --unbuffered --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" \
  slurm/ambi_distillation_posttanh_oscar.py "${arguments[@]}" || training_status=$?
publisher_status=0
"${publisher[@]}" slurm/ambi_distillation_posttanh_workspace.py "$AMBI_DISTILLATION_STATE" || publisher_status=$?
if [[ "$training_status" != 0 ]]; then
  exit "$training_status"
fi
exit "$publisher_status"
