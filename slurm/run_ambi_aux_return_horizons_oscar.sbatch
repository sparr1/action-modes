#!/usr/bin/env bash
#SBATCH --job-name=ambi-aux-horizons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/rgao48/ambi/aux-return-sac-horizons/slurm/%x-%A_%a.out
#SBATCH --error=/oscar/scratch/rgao48/ambi/aux-return-sac-horizons/slurm/%x-%A_%a.err

# Supply account/resources with sbatch after checking live allocation headroom.
# Submit AMBI_AUX_MODE=smoke --array=0-2 (recommend --time=02:00:00), then
# AMBI_AUX_MODE=train --array=0-2 --dependency=aftercorr:<gate array>. Each H
# requires its own successful immutable receipt. Override output/error paths.
set -Eeuo pipefail
umask 077
fail() { printf 'Oscar auxiliary SAC horizon campaign error: %s\n' "$*" >&2; exit 2; }
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
: "${SLURM_ARRAY_JOB_ID:?Submit as a three-cell array.}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a three-cell array.}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-2]$ ]] || fail "array index must be 0..2"
configs=(ambi_aux_return_sac_clip_target10p5_shared_h1 ambi_aux_return_sac_clip_target10p5_shared_h5 ambi_aux_return_sac_clip_target10p5_shared_h7)
config="${configs[$SLURM_ARRAY_TASK_ID]}"

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
mkdir -p "$campaign_root/runtime"
local_root="$(mktemp -d "$campaign_root/runtime/job-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/wandb" "$local_root/data" "$local_root/artifacts"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root" TMP="$local_root" TEMP="$local_root"
export WANDB_DISABLE_CODE=true WANDB_DIR="$local_root/wandb" WANDB_CACHE_DIR="$local_root/cache"
export WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"
export AMBI_AUX_CAMPAIGN EXPECTED_ACTION_MODES_SHA
manifest="configs/dmcontrol/experiments/ambi_aux_return_sac_horizons.json"
helper="slurm/ambi_aux_return_horizon_campaign.py"
if [[ "$mode" == smoke ]]; then
  mkdir -p "$campaign_root/gate"
  export AMBI_AUX_GATE_OUTPUT_ROOT="$campaign_root/gate/$config"
  mkdir "$AMBI_AUX_GATE_OUTPUT_ROOT" || fail "gate output exists; use a fresh campaign"
  export AMBI_RUN_AUX_RETURN_HORIZON_CUDA_GATE=1 WANDB_MODE=offline
  srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m pytest -q -s -p no:cacheprovider \
    tests/test_ambi_aux_return_horizon_cuda_gate.py -k "$config" --basetemp="$local_root/pytest"
  "$AMBI_PYTHON" "$helper" receipt --source-commit "$EXPECTED_ACTION_MODES_SHA" --gate-dir "$AMBI_AUX_GATE_OUTPUT_ROOT" --index "$SLURM_ARRAY_TASK_ID"
else
  AMBI_AUX_GATE_RECEIPT="$campaign_root/gate/$config/receipt.json"
  [[ -f "$AMBI_AUX_GATE_RECEIPT" ]] || fail "matching horizon GPU receipt is missing"
  run_root="$campaign_root/$config-seed55"
  mkdir "$run_root" || fail "run output exists; refusing a duplicate run"
  export WANDB_MODE=online WANDB_RESUME=never WANDB_RUN_ID="auxh${SLURM_ARRAY_JOB_ID}x${SLURM_ARRAY_TASK_ID}"
  "$AMBI_PYTHON" "$helper" train-metadata --source-commit "$EXPECTED_ACTION_MODES_SHA" \
    --receipt "$AMBI_AUX_GATE_RECEIPT" --index "$SLURM_ARRAY_TASK_ID" --run-root "$run_root"
  printf 'Job: %s_%s\nSource: %s\nConfig: %s; seed: 55; decisions: 2000000\nOutput: %s\n' \
    "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$EXPECTED_ACTION_MODES_SHA" "$config" "$run_root"
  exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" main.py \
    --run "$manifest" --alg-dir configs/dmcontrol/algs --log-dir "$run_root" \
    --alg-index "$SLURM_ARRAY_TASK_ID" --trial-index 0 --num-runs 1
fi
