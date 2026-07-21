#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

launchers=(
  "slurm/run_ambi_full_copy.sbatch"
  "slurm/run_ambi_breadth_depth_inner6_rollouts32.sbatch"
  "slurm/run_ambi_fixed_budget_inner2_rollouts96.sbatch"
  "slurm/run_ambi_updates64_per_round.sbatch"
  "slurm/run_ambi_rounds8_full_dose.sbatch"
  "slurm/run_ambi_horizon_train6_inner3.sbatch"
)

for launcher in "${launchers[@]}"; do
  echo "Submitting $launcher"
  sbatch "$launcher"
done
