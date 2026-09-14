#!/usr/bin/env bash
#SBATCH --job-name=reward-prior-validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

# Submit after production, using --output/--error paths outside the checkout.
# No production job depends on this independent validation allocation.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_EXPECTED_COMMIT:?Set the intended full Git SHA.}"
cd "${SLURM_SUBMIT_DIR:?Submit from the clean repository root.}"
[[ "$PWD" == "$(git rev-parse --show-toplevel)" ]]
[[ "$AMBI_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
: "${AMBI_DMCONTROL_PYTHON:?Set the absolute existing locked DMControl interpreter.}"
[[ "$AMBI_DMCONTROL_PYTHON" == /* ]]
ambi_python="$AMBI_DMCONTROL_PYTHON"
[[ -x "$ambi_python" ]]
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/reward-prior-validation-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/wandb"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 AMBI_RUN_REAL_DMCONTROL_TESTS=1 WANDB_MODE=offline
export WANDB_DIR="$local_root/wandb" XDG_CACHE_HOME="$local_root/cache"
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "CUDA validation: commit=$AMBI_EXPECTED_COMMIT node=${SLURM_NODELIST:-unknown}"
# Fail explicitly instead of reporting CUDA skips as successful GPU validation.
"$ambi_python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable on allocated node"; print(torch.__version__, torch.cuda.get_device_name())'
exec srun --kill-on-bad-exit=1 "$ambi_python" -m pytest -q -s -p no:cacheprovider \
  tests/test_compile_regions.py \
  tests/test_ambi_prior_reward_hydra_launcher.py \
  tests/test_ambi_prior_reward_critic_study_configs.py \
  tests/test_ambi_prior_reward_cuda_gate.py \
  tests/test_ambi_outer_policy_diagnostics.py \
  tests/test_outer_policy_diagnostic_recorder.py \
  tests/test_ambi_outer_diagnostics_integration.py \
  tests/test_ambi_outer_diagnostics_resume.py \
  tests/test_wandb_event_indexed.py
