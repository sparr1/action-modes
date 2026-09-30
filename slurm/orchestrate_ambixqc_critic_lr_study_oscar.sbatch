#!/usr/bin/env bash
#SBATCH --job-name=axqc-critic-lr-coordinator
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${AMBIXQC_ACTION_MODES_DIR:?Original clean scientific checkout required}"
: "${AMBIXQC_TOOLING_DIR:?Clean tooling checkout required}"
: "${AMBIXQC_PYTHON:?Existing locked Python required}"
: "${AMBIXQC_RESULTS_ROOT:?Separate durable result root required}"
: "${EXPECTED_ACTION_MODES_SHA:?Original full source commit required}"
: "${EXPECTED_TOOLING_SHA:?Full tooling commit required}"
: "${SLURM_JOB_ID:?Use a CPU scheduler allocation}"
[[ "$(git -C "$AMBIXQC_ACTION_MODES_DIR" rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ "$(git -C "$AMBIXQC_TOOLING_DIR" rev-parse HEAD)" == "$EXPECTED_TOOLING_SHA" ]]
[[ -z "$(git -C "$AMBIXQC_ACTION_MODES_DIR" status --porcelain=v1 --untracked-files=all)" ]]
[[ -z "$(git -C "$AMBIXQC_TOOLING_DIR" status --porcelain=v1 --untracked-files=all)" ]]
readonly ENV_PROJECT="$(cd "$(dirname "$AMBIXQC_PYTHON")/../.." && pwd -P)"
readonly LOCK_SHA=f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6
for lock in "$AMBIXQC_ACTION_MODES_DIR/environments/dmcontrol/uv.lock" "$ENV_PROJECT/uv.lock"; do
  [[ "$(sha256sum "$lock" | awk '{print $1}')" == "$LOCK_SHA" ]]
done
readonly TASK_CACHE="$AMBIXQC_RESULTS_ROOT/coordinator-cache/job${SLURM_JOB_ID}"
[[ ! -e "$TASK_CACHE" ]]
mkdir -p "$TASK_CACHE/wandb" "$TASK_CACHE/cache" "$TASK_CACHE/data" "$TASK_CACHE/artifacts" "$TASK_CACHE/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR="$TASK_CACHE/tmp"
export WANDB_DIR="$TASK_CACHE/wandb" WANDB_CACHE_DIR="$TASK_CACHE/cache" WANDB_DATA_DIR="$TASK_CACHE/data"
export WANDB_ARTIFACT_DIR="$TASK_CACHE/artifacts" WANDB_MODE=online
unset WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_CONFIG_PATHS
cd "$AMBIXQC_ACTION_MODES_DIR"
exec "$AMBIXQC_PYTHON" "$AMBIXQC_TOOLING_DIR/run_ambixqc_critic_lr_study.py" coordinate "$@"
