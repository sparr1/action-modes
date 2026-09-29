#!/usr/bin/env bash
#SBATCH --job-name=axqc-h1-publisher
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -Eeuo pipefail
umask 077
: "${EXPECTED_ACTION_MODES_SHA:?Export the exact tested source SHA.}"
: "${AMBIXQC_ACTION_MODES_DIR:?Export the clean checkout path.}"
: "${AMBIXQC_PYTHON:?Export the existing locked DMControl Python.}"
: "${AMBIXQC_RESULTS_ROOT:?Export the durable campaign directory.}"
: "${EVAL_RUN_DIR:?Export the explicit New registry for this condition.}"
: "${EVAL_CONDITION_INDEX:?Export its index 0 through 3.}"
: "${EVAL_COMPUTE_JOBS:?Export the complete four-task GPU array ID.}"
: "${SLURM_JOB_ID:?Use a CPU scheduler allocation.}"
cd "$AMBIXQC_ACTION_MODES_DIR"
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]]
[[ "$AMBIXQC_PYTHON" == /* && -x "$AMBIXQC_PYTHON" ]]
[[ "$AMBIXQC_RESULTS_ROOT" == /* && -d "$AMBIXQC_RESULTS_ROOT" ]]
[[ "$EVAL_RUN_DIR" == /* && -d "$EVAL_RUN_DIR" ]]
[[ "$EVAL_CONDITION_INDEX" =~ ^[0-3]$ ]]
readonly ENV_PROJECT="$(cd "$(dirname "$AMBIXQC_PYTHON")/../.." && pwd -P)"
for lock in environments/dmcontrol/uv.lock "$ENV_PROJECT/uv.lock"; do
 [[ "$(sha256sum "$lock" | awk '{print $1}')" == f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6 ]]
done
readonly CACHE_ROOT="$AMBIXQC_RESULTS_ROOT/publisher-cache/job${SLURM_JOB_ID}"
[[ ! -e "$CACHE_ROOT" && ! -L "$CACHE_ROOT" ]]
mkdir -p "$CACHE_ROOT/wandb" "$CACHE_ROOT/cache" "$CACHE_ROOT/data" "$CACHE_ROOT/artifacts" "$CACHE_ROOT/tmp"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export WANDB_DIR="$CACHE_ROOT/wandb" WANDB_CACHE_DIR="$CACHE_ROOT/cache"
export WANDB_DATA_DIR="$CACHE_ROOT/data" WANDB_ARTIFACT_DIR="$CACHE_ROOT/artifacts"
export TMPDIR="$CACHE_ROOT/tmp" WANDB_MODE=online
unset WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_CONFIG_PATHS
read -r -a jobs <<< "$EVAL_COMPUTE_JOBS"
exec "$AMBIXQC_PYTHON" publish_ambixqc_inner_475k_screen.py "$EVAL_RUN_DIR" \
 --index "$EVAL_CONDITION_INDEX" --jobs "${jobs[@]}" --owner "${EVAL_PUBLICATION_OWNER:-oscar-rgao48}" \
 --poll-seconds 15 --visibility-seconds 180
