#!/usr/bin/env bash
# Resources are explicit sbatch arguments derived from the CUDA gate.
set -euo pipefail
: "${CAMPAIGN_ROOT:?Required}"
: "${EXPECTED_ACTION_MODES_SHA:?Required}"
: "${PYTHON_BIN:?Required}"
: "${SLURM_CPUS_PER_TASK:?Submit with an explicit CPU count}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MUJOCO_GL=egl
export WANDB_MODE=disabled
export TORCHINDUCTOR_CACHE_DIR="$CAMPAIGN_ROOT/cache/job-${SLURM_JOB_ID}"
args=(--campaign "$CAMPAIGN_ROOT/campaign.json" --index "${CELL_INDEX:-${SLURM_ARRAY_TASK_ID:?Required cell index}}")
if [[ -n "${SMOKE_STEPS:-}" ]]; then args+=(--smoke-steps "$SMOKE_STEPS"); fi
srun --unbuffered "$PYTHON_BIN" slurm/ambi_early_h3_j8_campaign.py run "${args[@]}"
