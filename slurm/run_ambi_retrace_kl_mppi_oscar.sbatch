#!/usr/bin/env bash
set -euo pipefail
: "${EXPECTED_ACTION_MODES_SHA:?Set the tested source commit}"
: "${PYTHON_BIN:?Set the locked DMControl interpreter}"
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${SLURM_CPUS_PER_TASK:?Request an explicit CPU count}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain)" ]]
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MUJOCO_GL=egl
export WANDB_MODE=disabled
if [[ "${RETRACE_CPU_GATE:-0}" == 1 ]]; then
  [[ "$SLURM_CPUS_PER_TASK" -ge 2 ]]
  srun --unbuffered "$PYTHON_BIN" slurm/ambi_retrace_kl_mppi.py run "$@" --index 0 --smoke-steps 32 --cpu-threads 1
  srun --unbuffered "$PYTHON_BIN" slurm/ambi_retrace_kl_mppi.py run "$@" --index 0 --smoke-steps 32 --cpu-threads 2
  srun --unbuffered "$PYTHON_BIN" slurm/ambi_retrace_kl_mppi.py run "$@" --index 3 --smoke-steps 8 --cpu-threads 1
else
  srun --unbuffered "$PYTHON_BIN" slurm/ambi_retrace_kl_mppi.py run "$@"
fi
