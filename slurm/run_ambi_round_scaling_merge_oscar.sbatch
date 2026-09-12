#!/usr/bin/env bash
#SBATCH --job-name=ambi-round-scaling-merge
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${AMBI_ROUND_SCALING_CAMPAIGN:?Supply the immutable campaign manifest}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || { echo "Unexpected commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty checkout" >&2; exit 1; }
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked Python" >&2; exit 1; }
[[ "${AMBI_ROUND_SCALING_SMOKE:-0}" == 0 || "${AMBI_ROUND_SCALING_SMOKE:-0}" == 1 ]] || exit 1
arguments=(merge --campaign "$AMBI_ROUND_SCALING_CAMPAIGN" --cell-index "${SLURM_ARRAY_TASK_ID:?Supply cell index}")
if [[ "${AMBI_ROUND_SCALING_SMOKE:-0}" == 1 ]]; then
    arguments+=(--smoke)
    if [[ -n "${AMBI_ROUND_SCALING_SMOKE_SHARDS:-}" ]]; then
        read -r -a smoke_shards <<< "$AMBI_ROUND_SCALING_SMOKE_SHARDS"
        arguments+=(--shard-indices "${smoke_shards[@]}")
    fi
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 WANDB_MODE=offline
exec "$PYTHON_BIN" slurm/ambi_round_scaling_campaign.py "${arguments[@]}"
