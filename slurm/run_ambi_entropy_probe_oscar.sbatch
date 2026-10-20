#!/usr/bin/env bash
# Supply measured GPU/CPU/memory/walltime and log paths as sbatch arguments.
# One worker owns one checkpoint/environment-seed cell and one fresh output.
set -euo pipefail
: "${EXPECTED_ACTION_MODES_SHA:?Set the tested source commit}"
: "${PYTHON_BIN:?Set the locked DMControl interpreter}"
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${SLURM_CPUS_PER_TASK:?Request an explicit measured CPU count}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${TORCH_THREADS:-$SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MUJOCO_GL=disable
export WANDB_MODE=disabled
srun --unbuffered "$PYTHON_BIN" evaluate_ambi_entropy_probe.py "$@" --torch-threads "$OMP_NUM_THREADS"
