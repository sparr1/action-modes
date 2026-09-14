#!/usr/bin/env bash
#SBATCH --job-name=reward-prior-study
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-5
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/rgao48/ambi/reward-prior-study/slurm/%x-%A_%a.out
#SBATCH --error=/oscar/scratch/rgao48/ambi/reward-prior-study/slurm/%x-%A_%a.err

set -Eeuo pipefail
umask 077

# Submit these six independent cells after checking the live account allowance
# and at least 36 GiB of aggregate output headroom. Each task retains its own
# 80 portable checkpoints (about 5.4 GiB); exact-resume replay is not enabled.
# Create the scheduler log directory before submission, or override --output/--error.
# Production deliberately has no smoke-test dependency: submit validation afterward.
# Resource defaults may be overridden after checking live Oscar limits. Prefer
# gpu:l40s:1, then gpu:l40:1 when Oscar exposes that type.
fail() { echo "Oscar reward-prior study launcher error: $*" >&2; exit 2; }

: "${SLURM_JOB_ID:?Run this launcher through Slurm.}"
: "${SLURM_ARRAY_JOB_ID:?This launcher requires a Slurm array job ID.}"
: "${SLURM_ARRAY_TASK_ID:?This launcher requires an six-cell Slurm array.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full intended and pushed Git commit SHA.}"
: "${AMBI_STUDY_OUTPUT_ROOT:?Set an absolute checkpoint output directory.}"
[[ "$AMBI_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "invalid expected commit SHA"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-5]$ ]] || fail "array task must be in 0..5"
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || fail "portable-checkpoint runs cannot resume after requeue"
[[ "$AMBI_STUDY_OUTPUT_ROOT" == /* ]] || fail "output root must be absolute"

project_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$project_dir"
git_root="$(git rev-parse --show-toplevel)"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$project_dir" ]] || fail "submit from the repository root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail "checkout is dirty"
source_commit="$(git rev-parse HEAD)"
[[ "$source_commit" == "$AMBI_EXPECTED_COMMIT" ]] || fail "HEAD differs from the intended commit"

: "${AMBI_DMCONTROL_PYTHON:?Set the absolute existing locked DMControl interpreter.}"
[[ "$AMBI_DMCONTROL_PYTHON" == /* ]] || fail "interpreter path must be absolute"
ambi_python="$AMBI_DMCONTROL_PYTHON"
[[ -x "$ambi_python" ]] || fail "missing locked DMControl interpreter: $ambi_python"

configs=(
  ambi_prior_reward_clip_target21
  ambi_prior_reward_clip_target10p5
  ambi_prior_reward_smooth_target21
  ambi_prior_reward_smooth_target10p5
  ambi_prior_reward_clip_fixed0p0001
  ambi_prior_reward_smooth_fixed0p0001
)
alg_index=$SLURM_ARRAY_TASK_ID
trial_index=0
seed=55
config="${configs[$alg_index]}"
manifest="configs/dmcontrol/experiments/ambi_prior_reward_critic_study.json"
[[ -f "$manifest" && -f "configs/dmcontrol/algs/$config.json" ]] || fail "missing study configuration"

mkdir -p "$AMBI_STUDY_OUTPUT_ROOT"
output_root="$(cd -- "$AMBI_STUDY_OUTPUT_ROOT" && pwd -P)"
[[ "$output_root" != "$project_dir" && "$output_root" != "$project_dir/"* ]] || fail \
  "checkpoint output must be outside the source checkout"
available_kb="$(df -Pk "$output_root" | awk 'NR == 2 {print $4}')"
[[ "$available_kb" =~ ^[0-9]+$ ]] || fail "could not determine output space"
(( available_kb >= 6 * 1024 * 1024 )) || fail "less than 6 GiB free for this run's checkpoint bank"
run_root="$output_root/$config/seed$seed/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$output_root/$config/seed$seed"
mkdir "$run_root" || fail "run output already exists; refusing a duplicate run"

# Compilation remains enabled in the configs. Its caches and online W&B
# staging belong on the allocated node, not the quota-limited home filesystem.
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/reward-prior-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/wandb" "$local_root/cache" "$local_root/data" \
  "$local_root/artifacts" "$local_root/inductor" "$local_root/triton"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor"
export TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache"
export WANDB_MODE=online WANDB_DISABLE_CODE=true
export WANDB_DIR="$local_root/wandb"
export WANDB_CACHE_DIR="$local_root/cache"
export WANDB_DATA_DIR="$local_root/data"
export WANDB_ARTIFACT_DIR="$local_root/artifacts"

echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURM_NODELIST:-unknown}"
echo "Source commit: $source_commit"
echo "Configuration: $config; seed: $seed; decisions: 2000000"
echo "Outer reward-only critic: H3/B256/UTD1; stochastic prior; no inner optimization"
echo "Actor: corrected squashed entropy; no Q scaling; see configuration for alpha"
echo "Checkpoints: every 25000 decisions, retain all 80"
echo "Checkpoint output: $run_root"
echo "Output available KiB: $available_kb"
echo "Node-local caches: $local_root"
"$ambi_python" -c 'import torch; assert torch.cuda.is_available(); print("torch:", torch.__version__); print("CUDA device:", torch.cuda.get_device_name()); print("CUDA devices:", torch.cuda.device_count())'

exec srun --kill-on-bad-exit=1 "$ambi_python" main.py \
  --run "$manifest" \
  --alg-dir configs/dmcontrol/algs \
  --log-dir "$run_root" \
  --alg-index "$alg_index" \
  --trial-index "$trial_index" \
  --num-runs 1
