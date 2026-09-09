#!/usr/bin/env bash
#SBATCH --job-name=td-ambi-prior-study
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --array=0-3
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/rgao48/ambi/td-ambi-study/slurm/%x-%A_%a.out
#SBATCH --error=/oscar/scratch/rgao48/ambi/td-ambi-study/slurm/%x-%A_%a.err

# Submit from the clean repository root after checking the rgao48 scratch
# quota has at least 30 GiB headroom for this four-run checkpoint bank:
#   mkdir -p /oscar/scratch/rgao48/ambi/td-ambi-study/slurm
#   sbatch --export=ALL,EXPECTED_ACTION_MODES_SHA=<tested-sha>,TD_AMBI_CAMPAIGN=<unique-name>,AMBI_DMC_PYTHON=<locked-python> \
#     slurm/run_td_ambi_prior_study_oscar.sbatch
# These are portable 25k model snapshots, not replay-containing exact-resume
# generations. The ordinary training path honors --log-dir. The exact-resume
# launcher would place model snapshots under its home-based lineage instead.

set -Eeuo pipefail
umask 077
fail() { printf 'Oscar TD-AMBI study launcher error: %s\n' "$*" >&2; exit 2; }

: "${SLURM_JOB_ID:?Run this launcher through Slurm.}"
: "${SLURM_ARRAY_JOB_ID:?Run this launcher as the four-cell array.}"
: "${SLURM_ARRAY_TASK_ID:?Run this launcher as the four-cell array.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested full Git commit.}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-3]$ ]] || fail "array index must be 0--3"
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || fail "this portable-checkpoint run cannot resume after requeue"

project_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$project_dir"
git_root="$(git rev-parse --show-toplevel)" || fail "not a Git checkout"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$project_dir" ]] || fail "submit from the Git root"
source_commit="$(git rev-parse HEAD)"
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ && "$source_commit" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "checkout does not match the tested commit"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail "production launch requires a clean checkout"

ambi_python="${AMBI_DMC_PYTHON:-$project_dir/environments/dmcontrol/.venv/bin/python}"
[[ -x "$ambi_python" ]] || fail "missing locked DMControl interpreter: $ambi_python"
algorithm_configs=(
  td_ambi_prior_reward_qscale
  td_ambi_prior_entropy_qscale
  td_ambi_prior_entropy_autotemp
  td_ambi_prior_reward_autotemp
)
algorithm_config="${algorithm_configs[$SLURM_ARRAY_TASK_ID]}"
manifest="configs/dmcontrol/experiments/$algorithm_config.json"
algorithm_dir="configs/dmcontrol/algs"
[[ -f "$manifest" && -f "$algorithm_dir/$algorithm_config.json" ]] || fail "missing study configuration"

approved_output_root="/oscar/scratch/rgao48/ambi/td-ambi-study"
output_root="${TD_AMBI_OUTPUT_ROOT:-$approved_output_root}"
[[ "$output_root" == /* ]] || fail "TD_AMBI_OUTPUT_ROOT must be absolute"
if [[ -d /oscar/home/rgao48 ]]; then
  [[ "$output_root" == "$approved_output_root" ]] || fail "Oscar checkpoints must use $approved_output_root"
fi
campaign="${TD_AMBI_CAMPAIGN:-job_${SLURM_ARRAY_JOB_ID}}"
[[ "$campaign" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "campaign must be a filesystem-safe basename"

quota_output="$(checkquota)" || fail "checkquota failed"
printf '%s\n' "$quota_output" | "$ambi_python" -m utils.oscar_resume_launcher quota \
  --allocation rgao48 --filesystem-path /oscar/scratch
mkdir -p "$output_root/$campaign"
output_root="$(cd -- "$output_root" && pwd -P)"
if [[ -d /oscar/home/rgao48 ]]; then
  [[ "$output_root" == "$approved_output_root" ]] || fail "resolved checkpoint root is outside the approved scratch location"
fi
available_kb="$(df -Pk "$output_root" | awk 'NR == 2 {print $4}')"
[[ "$available_kb" =~ ^[0-9]+$ ]] || fail "cannot determine filesystem free space"
(( available_kb >= 30 * 1024 * 1024 )) || fail "less than 30 GiB filesystem space available"
run_root="$output_root/$campaign/task_${SLURM_ARRAY_TASK_ID}"
mkdir "$run_root" || fail "cell output already exists; refusing a duplicate run"

local_root="${SLURM_TMPDIR:-/tmp}/td-ambi-prior-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$local_root" "$local_root/wandb" "$local_root/wandb/cache" "$local_root/wandb/data" "$local_root/wandb/artifacts" "$local_root/torchinductor" "$local_root/triton"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/torchinductor" TRITON_CACHE_DIR="$local_root/triton"
export WANDB_MODE=online WANDB_DISABLE_CODE=true
export WANDB_DIR="$local_root/wandb" WANDB_CACHE_DIR="$local_root/wandb/cache"
export WANDB_DATA_DIR="$local_root/wandb/data" WANDB_ARTIFACT_DIR="$local_root/wandb/artifacts"

printf 'Job: %s_%s\nNode: %s\nConfig: %s\nSource commit: %s\nRun root: %s\nAgent decisions: 2000000\nCheckpoints: every 25000, retain all 80\n' \
  "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "${SLURM_NODELIST:-unknown}" "$algorithm_config" "$source_commit" "$run_root"
"$ambi_python" -c 'import torch; assert torch.cuda.is_available(); print("torch:", torch.__version__, "GPU:", torch.cuda.get_device_name())'
srun --unbuffered --kill-on-bad-exit=1 \
  "$ambi_python" main.py --run "$manifest" --alg-dir "$algorithm_dir" \
  --log-dir "$run_root" --alg-index 0 --trial-index 0 --num-runs 1
