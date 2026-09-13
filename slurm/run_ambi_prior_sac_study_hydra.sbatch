#!/usr/bin/env bash
#SBATCH --job-name=sac-prior-study
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-7
#SBATCH --no-requeue
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -Eeuo pipefail
umask 077

# Submit all eight independent cells after checking the live account allowance
# and at least 48 GiB of aggregate output headroom. Each task retains its own
# 80 portable checkpoints (about 5.4 GiB); exact-resume replay is not enabled.
# Use sbatch --output/--error to put scheduler logs outside the checkout too.
fail() { echo "Hydra SAC-prior study launcher error: $*" >&2; exit 2; }

: "${SLURM_JOB_ID:?Run this launcher through Slurm.}"
: "${SLURM_ARRAY_JOB_ID:?This launcher requires a Slurm array job ID.}"
: "${SLURM_ARRAY_TASK_ID:?This launcher requires an eight-cell Slurm array.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested and pushed Git commit SHA.}"
: "${AMBI_STUDY_OUTPUT_ROOT:?Set an absolute checkpoint output directory.}"
[[ "$AMBI_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "invalid expected commit SHA"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-7]$ ]] || fail "array task must be in 0..7"
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || fail "portable-checkpoint runs cannot resume after requeue"
[[ "$AMBI_STUDY_OUTPUT_ROOT" == /* ]] || fail "output root must be absolute"

project_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$project_dir"
git_root="$(git rev-parse --show-toplevel)"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$project_dir" ]] || fail "submit from the repository root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail "checkout is dirty"
source_commit="$(git rev-parse HEAD)"
[[ "$source_commit" == "$AMBI_EXPECTED_COMMIT" ]] || fail "HEAD differs from the tested commit"

ambi_python="${AMBI_DMCONTROL_PYTHON:-$project_dir/environments/dmcontrol/.venv/bin/python}"
[[ -x "$ambi_python" ]] || fail "missing locked DMControl interpreter: $ambi_python"

configs=(
  ambi_prior_sac_clip_target21
  ambi_prior_sac_clip_target10p5
  ambi_prior_sac_smooth_target21
  ambi_prior_sac_smooth_target10p5
)
alg_index=$(( SLURM_ARRAY_TASK_ID / 2 ))
trial_index=$(( SLURM_ARRAY_TASK_ID % 2 ))
seed=$(( 55 + trial_index ))
config="${configs[$alg_index]}"
manifest="configs/dmcontrol/experiments/ambi_prior_sac_parameterization_study.json"
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
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/sac-prior-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/wandb" "$local_root/cache" "$local_root/data" \
  "$local_root/artifacts" "$local_root/inductor" "$local_root/triton"
export MUJOCO_GL=egl
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
echo "Outer SAC: H3/B256/UTD1; stochastic prior; no inner optimization"
echo "Checkpoints: every 25000 decisions, retain all 80"
echo "Checkpoint output: $run_root"
echo "Output available KiB: $available_kb"
echo "Node-local caches: $local_root"
"$ambi_python" -c 'import torch; print("torch:", torch.__version__); print("CUDA device:", torch.cuda.get_device_name()); print("CUDA devices:", torch.cuda.device_count())'

exec srun --kill-on-bad-exit=1 "$ambi_python" main.py \
  --run "$manifest" \
  --alg-dir configs/dmcontrol/algs \
  --log-dir "$run_root" \
  --alg-index "$alg_index" \
  --trial-index "$trial_index" \
  --num-runs 1
