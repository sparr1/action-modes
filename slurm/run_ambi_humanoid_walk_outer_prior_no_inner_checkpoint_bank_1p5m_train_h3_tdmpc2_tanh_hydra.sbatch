#!/usr/bin/env bash
#SBATCH --job-name=ambi-hw-td2std-h3-bank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2501
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%j.out
#SBATCH --error=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
readonly CONFIG_STEM="ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h3_tdmpc2_tanh"
readonly BASELINE_STEM="ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m"
readonly BANK_FAMILY="ambi-no-inner-tdmpc2-tanh-prior"
readonly EXPECTED_TRAIN_HORIZON=3
readonly EXPECTED_NODE="gpu2501"
readonly ARTIFACT_SLUG="ambi-no-inner-train-h3-actor-tdmpc2-tanh-upstream-bounds"
readonly WANDB_STAGING_SLUG="ambi-no-inner-h3-tdmpc2-tanh-ckptbank-wandb"
readonly CONDITION_LABEL="AMBI no-inner H=3 with TD-MPC2 tanh policy log-std mapping and [-10, 2] bounds"

# shellcheck source=slurm/run_prior_horizon_checkpoint_bank_1p5m_hydra_common.sh
source "$PROJECT_DIR/slurm/run_prior_horizon_checkpoint_bank_1p5m_hydra_common.sh"
