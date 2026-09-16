#!/usr/bin/env bash
#SBATCH --job-name=ambi-aux-return-sac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/rgao48/ambi/aux-return-sac-study/slurm/%x-%A_%a.out
#SBATCH --error=/oscar/scratch/rgao48/ambi/aux-return-sac-study/slurm/%x-%A_%a.err

# Supply account/resources with sbatch after checking live allocation headroom.
# First submit one AMBI_AUX_MODE=smoke job (recommend --time=02:00:00), then
# AMBI_AUX_MODE=train --array=0-3 --dependency=afterok:<gate job> with its receipt.
# AMBI_AUX_RECIPE=original_prior selects the two-cell historical-policy study
# instead; submit its training with --array=0-1 and its own smoke receipt.
# Create scheduler-log directories before submission; override --output/--error
# when choosing another output root. Never submit the smoke as an array.
set -Eeuo pipefail
umask 077
fail() { printf 'Oscar auxiliary SAC campaign error: %s\n' "$*" >&2; exit 2; }
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Supply the clean absolute source checkout.}"
: "${AMBI_PYTHON:?Supply the existing absolute locked DMControl interpreter.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested full Git SHA.}"
: "${AMBI_AUX_OUTPUT_ROOT:?Supply absolute scratch/project output storage.}"
: "${AMBI_AUX_CAMPAIGN:?Supply a unique campaign basename.}"
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || fail "portable checkpoint banks cannot resume after requeue"
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail "invalid expected commit SHA"
[[ "$AMBI_PROJECT_DIR" == /* && -d "$AMBI_PROJECT_DIR" ]] || fail "project directory must exist and be absolute"
[[ "$AMBI_PYTHON" == /* && -x "$AMBI_PYTHON" ]] || fail "interpreter must exist and be absolute"
[[ "$AMBI_AUX_OUTPUT_ROOT" == /* ]] || fail "output root must be absolute"
[[ "$AMBI_AUX_CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "invalid campaign basename"
mode="${AMBI_AUX_MODE:-train}"
[[ "$mode" == smoke || "$mode" == train ]] || fail "mode must be smoke or train"
recipe="${AMBI_AUX_RECIPE:-standard}"
case "$recipe" in
  standard)
    manifest="configs/dmcontrol/experiments/ambi_aux_return_sac_study.json"
    configs=(ambi_aux_return_sac_clip_target10p5_shared ambi_aux_return_sac_clip_target10p5_detached ambi_aux_return_sac_clip_target21_shared ambi_aux_return_sac_clip_target21_detached)
    decisions=2000000
    ;;
  original_prior)
    manifest="configs/dmcontrol/experiments/ambi_aux_return_original_prior_study.json"
    configs=(ambi_aux_return_original_prior_shared ambi_aux_return_original_prior_detached)
    decisions=1500000
    ;;
  *) fail "recipe must be standard or original_prior" ;;
esac
export AMBI_AUX_RECIPE="$recipe"
if [[ "$mode" == smoke ]]; then
  [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "submit the smoke as a single job"
else
  : "${SLURM_ARRAY_JOB_ID:?Train as a recipe array.}"
  : "${SLURM_ARRAY_TASK_ID:?Train as a recipe array.}"
  : "${AMBI_AUX_GATE_RECEIPT:?Supply the successful GPU smoke receipt.}"
  [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-3]$ ]] || fail "array index must be 0..3"
  (( SLURM_ARRAY_TASK_ID < ${#configs[@]} )) || fail "array index exceeds recipe cell count"
  [[ "$AMBI_AUX_GATE_RECEIPT" == /* && -f "$AMBI_AUX_GATE_RECEIPT" ]] || fail "receipt must exist at an absolute path"
fi
cd "$AMBI_PROJECT_DIR"
project_dir="$(pwd -P)"
[[ "$(git rev-parse --show-toplevel)" == "$project_dir" ]] || fail "project directory must be the Git root"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "HEAD differs from the tested commit"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail "checkout is dirty"
mkdir -p "$AMBI_AUX_OUTPUT_ROOT"
output_root="$(cd "$AMBI_AUX_OUTPUT_ROOT" && pwd -P)"
[[ "$output_root" != "$project_dir" && "$output_root" != "$project_dir/"* ]] || fail "outputs must be outside source"
campaign_root="$output_root/$AMBI_AUX_CAMPAIGN"
mkdir -p "$campaign_root"
[[ ! -L "$campaign_root" ]] || fail "campaign directory must not be a symlink"
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-aux-return-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/wandb" "$local_root/data" "$local_root/artifacts"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache"
export WANDB_DISABLE_CODE=true WANDB_DIR="$local_root/wandb" WANDB_CACHE_DIR="$local_root/cache"
export WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"
export AMBI_AUX_CAMPAIGN EXPECTED_ACTION_MODES_SHA
helper="slurm/ambi_aux_return_sac_campaign.py"
if [[ "$mode" == smoke ]]; then
  export AMBI_AUX_GATE_OUTPUT_ROOT="$campaign_root/gate"
  mkdir "$AMBI_AUX_GATE_OUTPUT_ROOT" || fail "gate output exists; use a fresh campaign"
  export AMBI_RUN_AUX_RETURN_SAC_CUDA_GATE=1 WANDB_MODE=offline
  # The legacy regression suite deliberately tests the unchanged default recipe.
  srun --unbuffered --kill-on-bad-exit=1 /usr/bin/env AMBI_AUX_RECIPE=standard "$AMBI_PYTHON" -m pytest -q -s -p no:cacheprovider \
    tests/test_ambi_aux_return_sac_campaign.py tests/test_aux_return_checkpoint.py \
    tests/test_ambi_aux_return_original_prior_campaign.py \
    tests/test_ambi_aux_return_original_prior_configs.py --basetemp="$local_root/pytest-regression"
  srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m pytest -q -s -p no:cacheprovider \
    tests/test_ambi_aux_return_sac_cuda_gate.py --basetemp="$local_root/pytest-cuda"
  "$AMBI_PYTHON" "$helper" receipt --source-commit "$EXPECTED_ACTION_MODES_SHA" --gate-dir "$AMBI_AUX_GATE_OUTPUT_ROOT"
else
  config="${configs[$SLURM_ARRAY_TASK_ID]}"
  run_root="$campaign_root/$config-seed55"
  mkdir "$run_root" || fail "run output exists; refusing a duplicate run"
  export WANDB_MODE=online WANDB_RESUME=never WANDB_RUN_ID="aux${SLURM_ARRAY_JOB_ID}x${SLURM_ARRAY_TASK_ID}"
  "$AMBI_PYTHON" "$helper" train-metadata --source-commit "$EXPECTED_ACTION_MODES_SHA" \
    --receipt "$AMBI_AUX_GATE_RECEIPT" --index "$SLURM_ARRAY_TASK_ID" --run-root "$run_root"
  printf 'Job: %s_%s\nSource: %s\nConfig: %s; seed: 55; decisions: %s\nOutput: %s\n' \
    "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$EXPECTED_ACTION_MODES_SHA" "$config" "$decisions" "$run_root"
  exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" main.py \
    --run "$manifest" --alg-dir configs/dmcontrol/algs --log-dir "$run_root" \
    --alg-index "$SLURM_ARRAY_TASK_ID" --trial-index 0 --num-runs 1
fi
