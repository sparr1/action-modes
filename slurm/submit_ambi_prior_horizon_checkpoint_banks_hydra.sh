#!/usr/bin/env bash

set -Eeuo pipefail

# Run this once from the clean Hydra action-modes checkout after the smoke job
# succeeds. It reserves campaign capacity conceptually with one aggregate free-
# space gate; each submitted job independently requires one bank plus headroom.
readonly PROJECT_DIR="$PWD"
readonly SHA_PREFLIGHT="slurm/require_expected_action_modes_sha.sh"
readonly STORAGE_PREFLIGHT="slurm/preflight_ambi_prior_horizon_checkpoint_banks_storage_hydra.sh"
readonly COMMON_RUNNER="slurm/run_prior_horizon_checkpoint_bank_1p5m_hydra_common.sh"
readonly CONFIG_STEMS=(
  "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h5"
  "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h10"
  "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h3_tdmpc2_tanh"
)
readonly LAUNCHERS=(
  "slurm/run_ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h5_hydra.sbatch"
  "slurm/run_ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h10_hydra.sbatch"
  "slurm/run_ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h3_tdmpc2_tanh_hydra.sbatch"
)

if [[ ! -f "$PROJECT_DIR/main.py" ]] || {
  [[ ! -d "$PROJECT_DIR/.git" ]] && [[ ! -f "$PROJECT_DIR/.git" ]]
}; then
  echo "Run this submission gate from the action-modes repository root." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to submit production jobs from a dirty checkout." >&2
  git status --short --untracked-files=normal >&2
  exit 1
fi
for required_path in \
  "$SHA_PREFLIGHT" \
  "$STORAGE_PREFLIGHT" \
  "$COMMON_RUNNER" \
  "${LAUNCHERS[@]}"; do
  if [[ ! -x "$required_path" ]]; then
    echo "Missing executable campaign launcher input: $required_path" >&2
    exit 1
  fi
  bash -n "$required_path"
done
for stem in "${CONFIG_STEMS[@]}"; do
  for config_path in \
    "configs/dmcontrol/algs/${stem}.json" \
    "configs/dmcontrol/experiments/${stem}.json"; do
    if [[ ! -f "$config_path" ]]; then
      echo "Missing production campaign config: $config_path" >&2
      exit 1
    fi
  done
done
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this gate on Hydra's login node." >&2
  exit 1
fi

readonly SOURCE_COMMIT="$("$SHA_PREFLIGHT")"
"$STORAGE_PREFLIGHT" --aggregate

submitted_job_ids=()
for launcher in "${LAUNCHERS[@]}"; do
  job_id="$(
    sbatch --parsable \
      --export=ALL,EXPECTED_ACTION_MODES_SHA="$SOURCE_COMMIT" \
      "$launcher"
  )"
  submitted_job_ids+=("$job_id")
  echo "Submitted $launcher as job $job_id"
done

echo "Submitted all three production banks from commit $SOURCE_COMMIT."
echo "Job IDs: ${submitted_job_ids[*]}"
