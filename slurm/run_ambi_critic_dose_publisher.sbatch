#!/usr/bin/env bash
#SBATCH --job-name=ambi-800k-critic-dose-publish
#SBATCH --account=default
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/rgao48/ambi/critic-dose-publish-%j.out
#SBATCH --error=/oscar/scratch/rgao48/ambi/critic-dose-publish-%j.err

set -Eeuo pipefail
: "${DOSE_CHECKOUT:?Set the synchronized checkout path}"
: "${DOSE_ROOT:?Set the prepared campaign scratch directory}"
: "${DOSE_PUBLICATION_ROOT:?Set the separate publication scratch directory}"
: "${DOSE_EXPECTED_COMMIT:?Set the full tested commit SHA}"
: "${DOSE_REFERENCE_ROOT:?Set the completed C16 scientific campaign directory}"
: "${DOSE_REFERENCE_SNAPSHOT:?Set the verified completed C16 reference snapshot}"
: "${DOSE_GPU_JOB_ID:?Set the critic-dose GPU job or array ID}"
: "${SLURM_JOB_ID:?Run through Slurm}"

cd "$DOSE_CHECKOUT"
[[ "$DOSE_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git rev-parse HEAD)" == "$DOSE_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$DOSE_GPU_JOB_ID" =~ ^[0-9]+$ ]]
[[ "${DOSE_PUBLISH_ONCE:-0}" =~ ^[01]$ ]]
[[ -f "$DOSE_ROOT/campaign.json" && -f "$DOSE_REFERENCE_SNAPSHOT" && -f "$DOSE_REFERENCE_ROOT/campaign.json" ]]
[[ "$DOSE_ROOT" != "$DOSE_PUBLICATION_ROOT" ]]
PYTHON_BIN=/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python
[[ -x "$PYTHON_BIN" ]]

runtime_root="$DOSE_PUBLICATION_ROOT/runtime/$SLURM_JOB_ID"
export TMPDIR="$runtime_root/tmp" XDG_CACHE_HOME="$runtime_root/cache"
export WANDB_DIR="$runtime_root/wandb" WANDB_CACHE_DIR="$runtime_root/wandb-cache"
export WANDB_DATA_DIR="$runtime_root/wandb-data" WANDB_MODE=online
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
unset WANDB_RUN_ID WANDB_NAME WANDB_GROUP WANDB_SERVICE _WANDB_SERVICE WANDB_RESUME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# The publisher resolves authentication through the existing cluster runtime;
# never read, copy, or print credentials in the launcher.
printf 'critic_dose_publish job=%s gpu_job=%s node=%s commit=%s tree=%s\n' \
  "$SLURM_JOB_ID" "$DOSE_GPU_JOB_ID" "$(hostname)" "$DOSE_EXPECTED_COMMIT" \
  "$(git rev-parse 'HEAD^{tree}')"

args=(--root "$DOSE_ROOT" --publication-root "$DOSE_PUBLICATION_ROOT"
  --reference-snapshot "$DOSE_REFERENCE_SNAPSHOT" --reference-root "$DOSE_REFERENCE_ROOT"
  --gpu-job-id "$DOSE_GPU_JOB_ID" --poll-seconds 60)
if [[ "${DOSE_PUBLISH_ONCE:-0}" == 1 ]]; then args+=(--once); fi
exec "$PYTHON_BIN" slurm/publish_ambi_critic_dose.py "${args[@]}"
