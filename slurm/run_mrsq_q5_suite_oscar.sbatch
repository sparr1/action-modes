#!/usr/bin/env bash
#SBATCH --job-name=mrsq-q5-suite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Use Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin source}"
: "${AMBI_ENTROPY_CAMPAIGN:?Set campaign}"
: "${AMBI_DMCONTROL_PYTHON:?Set locked training Python}"
: "${AMBI_WORKSPACE_PYTHON:?Set publisher Python}"
: "${SLURM_CPUS_PER_TASK:?Use measured CPUs}"
mode="${1:?gate or train}"
task="${2:?Set suite task}"
[[ "$mode" == gate || "$mode" == train ]]
case "$task" in dog-trot|dog-run|humanoid-stand|humanoid-walk|humanoid-run) ;; *) exit 2 ;; esac
[[ "$AMBI_ENTROPY_CAMPAIGN" == "/oscar/scratch/rgao48/ambi/mrsq-q5-suite-20261118/$task" ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME WANDB_RUN_ID AMBI_DOG_TASK
export AMBI_ENTROPY_VARIANT=mrsq_q5_suite AMBI_MRSQ_TASK="$task" WANDB_MODE=disabled
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
scratch_cache="$AMBI_ENTROPY_CAMPAIGN/cache/$SLURM_JOB_ID"
mkdir "$scratch_cache"
mkdir -p "$scratch_cache/"{inductor,triton,cache,wandb,data,artifacts,tmp}
export TMPDIR="$scratch_cache/tmp" TORCHINDUCTOR_CACHE_DIR="$scratch_cache/inductor" TRITON_CACHE_DIR="$scratch_cache/triton"
export XDG_CACHE_HOME="$scratch_cache/cache" WANDB_DIR="$scratch_cache/wandb" WANDB_CACHE_DIR="$scratch_cache/cache"
export WANDB_DATA_DIR="$scratch_cache/data" WANDB_ARTIFACT_DIR="$scratch_cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
publish() {
  flock "$AMBI_ENTROPY_CAMPAIGN/workspace.lock" "$AMBI_WORKSPACE_PYTHON" \
    slurm/inner_entropy_workspace.py "$AMBI_ENTROPY_CAMPAIGN/submission.json"
}
if [[ "$mode" == train ]]; then
  publish
  export WANDB_RUN_ID="$("$AMBI_DMCONTROL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runs"][0]["id"])' "$AMBI_ENTROPY_CAMPAIGN/submission.json")"
fi
srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/mrsq_q5_suite.py "$mode" --task "$task"
if [[ "$mode" == train ]]; then publish; fi
