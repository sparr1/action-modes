#!/usr/bin/env bash
set -euo pipefail
: "${EXPECTED_ACTION_MODES_SHA:?Set the tested source commit}"
: "${PYTHON_BIN:?Set the locked DMControl interpreter}"
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${SLURM_CPUS_PER_TASK:?Request an explicit CPU count}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain)" ]]
[[ "$SLURM_CPUS_PER_TASK" == 1 ]]
[[ "$PYTHON_BIN" = /* && -f "$PYTHON_BIN" ]]
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MUJOCO_GL=egl
export WANDB_MODE=disabled
srun --unbuffered "$PYTHON_BIN" slurm/ambi_alpha_long_episodes.py run "$@"
