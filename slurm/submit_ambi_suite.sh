#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

launchers=(
  "slurm/run_ambi_anchor.sbatch"
  "slurm/run_ambi_horizon_h6.sbatch"
  "slurm/run_ambi_branch_n64.sbatch"
  "slurm/run_ambi_horizon_h1.sbatch"
  "slurm/run_ambi_branch_n16.sbatch"
)

for launcher in "${launchers[@]}"; do
  echo "Submitting $launcher"
  sbatch "$launcher"
done
