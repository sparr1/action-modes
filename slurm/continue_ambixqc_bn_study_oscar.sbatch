#!/usr/bin/env bash
#SBATCH --job-name=axqc-bn-stage3-coordinator
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${AMBIXQC_ACTION_MODES_DIR:?Original clean experiment checkout required}"
: "${AMBIXQC_TOOLING_DIR:?Clean continuation tooling checkout required}"
: "${AMBIXQC_PYTHON:?Existing locked Python required}"
: "${AMBIXQC_RESULTS_ROOT:?Separate extension result root required}"
: "${EXPECTED_ACTION_MODES_SHA:?Original experiment SHA required}"
: "${EXPECTED_TOOLING_SHA:?Continuation tooling SHA required}"
: "${SLURM_JOB_ID:?Use a CPU scheduler allocation}"
[[ "$(git -C "$AMBIXQC_TOOLING_DIR" rev-parse HEAD)" == "$EXPECTED_TOOLING_SHA" ]]
[[ -z "$(git -C "$AMBIXQC_TOOLING_DIR" status --porcelain=v1 --untracked-files=all)" ]]
cd "$AMBIXQC_ACTION_MODES_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]]
readonly ENV_PROJECT="$(cd "$(dirname "$AMBIXQC_PYTHON")/../.." && pwd -P)"
for lock in environments/dmcontrol/uv.lock "$ENV_PROJECT/uv.lock"; do
  [[ "$(sha256sum "$lock" | awk '{print $1}')" == f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6 ]]
done
readonly CACHE="$AMBIXQC_RESULTS_ROOT/coordinator-cache/job${SLURM_JOB_ID}"
[[ ! -e "$CACHE" ]]
mkdir -p "$CACHE/wandb" "$CACHE/cache" "$CACHE/data" "$CACHE/artifacts" "$CACHE/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR="$CACHE/tmp"
export WANDB_DIR="$CACHE/wandb" WANDB_CACHE_DIR="$CACHE/cache" WANDB_DATA_DIR="$CACHE/data"
export WANDB_ARTIFACT_DIR="$CACHE/artifacts" WANDB_MODE=online
unset WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_CONFIG_PATHS
exec "$AMBIXQC_PYTHON" "$AMBIXQC_TOOLING_DIR/continue_ambixqc_bn_study.py" "$@"
