#!/usr/bin/env bash
#SBATCH --job-name=ambi-hw-none-bank
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
umask 077

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
readonly PYTHON="${AMBI_DMC_PYTHON:-/cs/home/rgao48/projects/action-modes/environments/dmcontrol/.venv/bin/python}"
readonly CONFIG_STEM="ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m"
readonly ALGORITHM_CONFIG="configs/dmcontrol/algs/${CONFIG_STEM}.json"
readonly MANIFEST="configs/dmcontrol/experiments/${CONFIG_STEM}.json"
readonly SEED=55
readonly ARTIFACT_ROOT="/cs/home/rgao48/projects/ambi-runs/ambi-no-inner-checkpoint-bank-1p5m"
readonly RUN_ROOT="${ARTIFACT_ROOT}/seed_${SEED}/job_${SLURM_JOB_ID}"
readonly WANDB_LOCAL_ROOT="${SLURM_TMPDIR:-/tmp}/ambi-no-inner-ckptbank-wandb-${SLURM_JOB_ID}"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing locked DMControl interpreter: $PYTHON" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to train from a checkout with uncommitted files." >&2
  git status --short --untracked-files=normal >&2
  exit 1
fi
if [[ "${SLURM_NODELIST:-}" != "gpu2501" ]]; then
  echo "Expected gpu2501, found ${SLURM_NODELIST:-unset}." >&2
  exit 1
fi
if [[ ! -f "$ALGORITHM_CONFIG" || ! -f "$MANIFEST" ]]; then
  echo "Missing checkpoint-bank algorithm config or manifest." >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
  echo "Refusing to reuse artifact root: $RUN_ROOT" >&2
  exit 1
fi

"$PYTHON" - "$ALGORITHM_CONFIG" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path


def load(path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SystemExit(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=reject_duplicates)


algorithm = load(sys.argv[1])
manifest = load(sys.argv[2])
params = algorithm["alg_params"]
required = {
    "outer_critic_target": "entropy_augmented",
    "inner_sac_critic_target": "entropy_augmented",
    "ent_coef": "auto",
    "target_entropy": "auto",
    "inner_operator": "none",
    "mpc": False,
    "eval_freq": None,
    "eval_inner_comparison": False,
    "eval_value": False,
    "value_equivalence_diagnostics": False,
    "inner_diagnostic_rollouts": 0,
    "train_unroll_horizon": 3,
    "outer_planning_horizon": 3,
}
actual = {key: params.get(key) for key in required}
if actual != required:
    raise SystemExit(f"checkpoint-bank scientific contract drifted: {actual}")
if algorithm.get("seed") != 55 or algorithm.get("total_steps") != 1_500_000:
    raise SystemExit("unexpected algorithm seed or decision budget")
if manifest.get("configs") != [Path(sys.argv[1]).stem]:
    raise SystemExit("manifest does not select exactly the checkpoint-bank config")
if manifest.get("trials") != 1 or manifest.get("logs") != "none":
    raise SystemExit("unexpected trial or trajectory-logging policy")
if manifest.get("save_trials") != "none":
    raise SystemExit("unexpected final-model policy")
cadence = manifest.get("checkpoint_every")
if cadence != 25_000 or manifest.get("save_strat") != ["all"]:
    raise SystemExit("unexpected checkpoint retention policy")
if algorithm["total_steps"] // cadence != 60:
    raise SystemExit("checkpoint bank must contain exactly 60 numbered snapshots")
print("Scientific preflight: entropy-augmented outer critic, auto alpha, zero inner work")
PY

mkdir -p "$ARTIFACT_ROOT"
available_kib="$(df -Pk "$ARTIFACT_ROOT" | awk 'NR == 2 {print $4}')"
if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || (( available_kib < 8 * 1024 * 1024 )); then
  echo "Refusing to start with less than 8 GiB free for checkpoint artifacts." >&2
  exit 1
fi
mkdir -p \
  "$WANDB_LOCAL_ROOT/cache" \
  "$WANDB_LOCAL_ROOT/data" \
  "$WANDB_LOCAL_ROOT/artifacts"

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE=online
export WANDB_DIR="$WANDB_LOCAL_ROOT"
export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"
export WANDB_DISABLE_CODE=true

readonly SOURCE_COMMIT="$(git rev-parse HEAD)"
echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Source commit: $SOURCE_COMMIT"
echo "Task: humanoid-walk"
echo "Condition: AMBI outer policy training with no inner controller"
echo "Outer critic target: entropy_augmented"
echo "Outer entropy coefficient: auto"
echo "Seed: $SEED"
echo "Agent decisions: 1500000"
echo "Raw simulator control steps: 3000000"
echo "Training unroll horizon: 3"
echo "Outer planning horizon: 3"
echo "Inner model steps per action: 0"
echo "Checkpoint cadence: 25000"
echo "Expected numbered checkpoints: 60"
echo "Manifest: $MANIFEST"
echo "Artifact root: $RUN_ROOT"
echo "W&B local staging: $WANDB_LOCAL_ROOT"
echo "Python: $PYTHON"
"$PYTHON" --version
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
"$PYTHON" -c 'import os, torch; name=torch.cuda.get_device_name(); assert torch.cuda.device_count() == 1 and "L40S" in name, (torch.cuda.device_count(), name); print("torch:", torch.__version__); print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")); print("visible CUDA devices:", torch.cuda.device_count()); print("cuda:", name)'

srun --unbuffered --kill-on-bad-exit=1 \
  "$PYTHON" main.py \
  --run "$MANIFEST" \
  --alg-dir configs/dmcontrol/algs \
  --log-dir "$RUN_ROOT" \
  --alg-index 0 \
  --trial-index 0 \
  --num-runs 1
