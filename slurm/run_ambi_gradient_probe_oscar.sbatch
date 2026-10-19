#!/usr/bin/env bash
# Supply measured CPU/GPU/memory/walltime requirements as sbatch arguments.
set -euo pipefail
: "${CAMPAIGN_ROOT:?Required}"
: "${EXPECTED_ACTION_MODES_SHA:?Required}"
: "${PYTHON_BIN:?Required}"
: "${STAGE:?Required: banks, benchmark, or production}"
: "${SLURM_JOB_ID:?Run inside a Slurm compute allocation}"
: "${SLURM_CPUS_PER_TASK:?Submit with an explicit CPU count}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${TORCH_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MUJOCO_GL=disable
export WANDB_MODE=disabled
args=(--campaign "$CAMPAIGN_ROOT/campaign.json" --stage "$STAGE"
      --index "${CELL_INDEX:-${SLURM_ARRAY_TASK_ID:?Required cell index}}"
      --expected-sha "$EXPECTED_ACTION_MODES_SHA" --python "$PYTHON_BIN")
if [[ -n "${DEVICE:-}" ]]; then args+=(--device "$DEVICE"); fi
if [[ -n "${TORCH_THREADS:-}" ]]; then args+=(--torch-threads "$TORCH_THREADS"); fi
srun --unbuffered "$PYTHON_BIN" slurm/ambi_gradient_probe_campaign.py run "${args[@]}"
