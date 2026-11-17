#!/usr/bin/env bash
#SBATCH --job-name=gsde8-dog-gate
#SBATCH --partition=gpus
#SBATCH --account=default
#SBATCH --nodelist=gpu2501
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${AMBI_GSDE_SOURCE:?Set the new clean node-local source path}"
: "${AMBI_EXPECTED_COMMIT:?Pin the approved source}"
[[ "$(hostname -s)" == gpu2501 && "$AMBI_GSDE_SOURCE" == /ltmp/rgao48/ambi/source/* ]]
mkdir -p "$(dirname "$AMBI_GSDE_SOURCE")" "$AMBI_GSDE_CAMPAIGN"
if [[ ! -e "$AMBI_GSDE_SOURCE" ]]; then
  git clone --no-checkout --shared /home/rgao48/projects/action-modes-eval-d6b046c "$AMBI_GSDE_SOURCE"
  git -C "$AMBI_GSDE_SOURCE" remote set-url origin https://github.com/sparr1/action-modes.git
  git -C "$AMBI_GSDE_SOURCE" fetch origin "$AMBI_EXPECTED_COMMIT"
  git -C "$AMBI_GSDE_SOURCE" checkout --detach "$AMBI_EXPECTED_COMMIT"
fi
cd "$AMBI_GSDE_SOURCE"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
export AMBI_GSDE_MODE=gate
exec bash slurm/run_gsde8_dog_hydra.sbatch
