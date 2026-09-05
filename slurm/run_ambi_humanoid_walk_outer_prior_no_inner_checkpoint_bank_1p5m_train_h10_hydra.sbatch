#!/usr/bin/env bash
#SBATCH --job-name=ambi-hw-none-h10-bank
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
readonly CONFIG_STEM="ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h10"
readonly BASELINE_STEM="ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m"
readonly BANK_FAMILY="ambi-no-inner"
readonly EXPECTED_TRAIN_HORIZON=10
readonly EXPECTED_NODE="gpu2501"
readonly ARTIFACT_SLUG="ambi-no-inner-train-h10"
readonly WANDB_STAGING_SLUG="ambi-no-inner-h10-ckptbank-wandb"
readonly CONDITION_LABEL="AMBI stochastic outer-policy prior, no inner controller, training unroll H=10"

# shellcheck source=slurm/run_prior_horizon_checkpoint_bank_1p5m_hydra_common.sh
source "$PROJECT_DIR/slurm/run_prior_horizon_checkpoint_bank_1p5m_hydra_common.sh"
