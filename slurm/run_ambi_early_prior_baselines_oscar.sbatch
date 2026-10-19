#!/usr/bin/env bash
# Submit explicit, measured resources; one complete baseline cell per array task.
set -euo pipefail
: "${CAMPAIGN_ROOT:?Required original campaign}"
: "${BASELINE_ROOT:?Required separate output root}"
: "${EXPECTED_ACTION_MODES_SHA:?Required}"
: "${PYTHON_BIN:?Required}"
: "${SLURM_CPUS_PER_TASK:?Required}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MUJOCO_GL=egl WANDB_MODE=disabled
srun --unbuffered "$PYTHON_BIN" slurm/ambi_early_prior_baselines.py \
  --campaign "$CAMPAIGN_ROOT/campaign.json" --index "$SLURM_ARRAY_TASK_ID" \
  --output-root "$BASELINE_ROOT"
