#!/usr/bin/env bash

# Shared body for the three guarded checkpoint-bank launchers. The sbatch
# wrapper must define the immutable scientific and artifact identifiers below.
set -Eeuo pipefail
umask 077

for required_name in \
  PROJECT_DIR \
  CONFIG_STEM \
  BASELINE_STEM \
  BANK_FAMILY \
  EXPECTED_TRAIN_HORIZON \
  EXPECTED_NODE \
  ARTIFACT_SLUG \
  WANDB_STAGING_SLUG \
  CONDITION_LABEL; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "Missing launcher contract variable: $required_name" >&2
    exit 1
  fi
done

readonly PYTHON="${AMBI_DMC_PYTHON:-/cs/home/rgao48/projects/action-modes/environments/dmcontrol/.venv/bin/python}"
readonly ALGORITHM_CONFIG="configs/dmcontrol/algs/${CONFIG_STEM}.json"
readonly BASELINE_CONFIG="configs/dmcontrol/algs/${BASELINE_STEM}.json"
readonly MANIFEST="configs/dmcontrol/experiments/${CONFIG_STEM}.json"
readonly STORAGE_PREFLIGHT="slurm/preflight_ambi_prior_horizon_checkpoint_banks_storage_hydra.sh"
readonly SHA_PREFLIGHT="slurm/require_expected_action_modes_sha.sh"
readonly SEED=55
readonly CAMPAIGN_ROOT="/cs/home/rgao48/projects/ambi-runs/ambi-prior-horizon-checkpoint-banks-1p5m"
readonly ARTIFACT_ROOT="${CAMPAIGN_ROOT}/${ARTIFACT_SLUG}"
readonly RUN_ROOT="${ARTIFACT_ROOT}/seed_${SEED}/job_${SLURM_JOB_ID}"
readonly WANDB_LOCAL_ROOT="${SLURM_TMPDIR:-/tmp}/${WANDB_STAGING_SLUG}-${SLURM_JOB_ID}"

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
if [[ "${SLURM_NODELIST:-}" != "$EXPECTED_NODE" ]]; then
  echo "Expected $EXPECTED_NODE, found ${SLURM_NODELIST:-unset}." >&2
  exit 1
fi
for required_path in \
  "$ALGORITHM_CONFIG" \
  "$BASELINE_CONFIG" \
  "$MANIFEST" \
  "$STORAGE_PREFLIGHT" \
  "$SHA_PREFLIGHT"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing checkpoint-bank input: $required_path" >&2
    exit 1
  fi
done
readonly SOURCE_COMMIT="$("$SHA_PREFLIGHT")"
if [[ -L "$ARTIFACT_ROOT" ]]; then
  echo "Refusing symlinked condition artifact root: $ARTIFACT_ROOT" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
  echo "Refusing to reuse immutable artifact root: $RUN_ROOT" >&2
  exit 1
fi

"$PYTHON" - \
  "$ALGORITHM_CONFIG" \
  "$BASELINE_CONFIG" \
  "$MANIFEST" \
  "$BANK_FAMILY" \
  "$EXPECTED_TRAIN_HORIZON" <<'PY'
import copy
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

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )


algorithm_path, baseline_path, manifest_path = map(Path, sys.argv[1:4])
family = sys.argv[4]
train_horizon = int(sys.argv[5])
algorithm = load(algorithm_path)
baseline = load(baseline_path)
manifest = load(manifest_path)
params = algorithm["alg_params"]

if algorithm.get("seed") != 55 or algorithm.get("total_steps") != 1_500_000:
    raise SystemExit("unexpected algorithm seed or decision budget")
if algorithm.get("episodes") is not None or algorithm.get("env") != "DMControl-v0":
    raise SystemExit("unexpected algorithm environment or episode budget")
if params.get("train_unroll_horizon") != train_horizon:
    raise SystemExit("training-unroll horizon drifted")
if {
    "outer_planning_horizon": params.get("outer_planning_horizon"),
    "inner_rollout_horizon": params.get("inner_rollout_horizon"),
    "temporal_loss_normalization": params.get("temporal_loss_normalization"),
    "temporal_loss_reference_horizon": params.get(
        "temporal_loss_reference_horizon"
    ),
} != {
    "outer_planning_horizon": 3,
    "inner_rollout_horizon": 3,
    "temporal_loss_normalization": "reference_weighted_mean",
    "temporal_loss_reference_horizon": 3,
}:
    raise SystemExit("fixed planning/inner/reference horizon contract drifted")
if params.get("eval_freq") is not None:
    raise SystemExit("online evaluation must remain disabled")

if family == "ambi-no-inner":
    if algorithm.get("alg") != "AMBITDMPC2/AMBITDMPC2":
        raise SystemExit("AMBI no-inner bank selected the wrong architecture")
    expected = copy.deepcopy(baseline)
    expected["alg_params"]["train_unroll_horizon"] = train_horizon
    for metadata_key in ("wandb_run_name", "wandb_tags"):
        expected["alg_params"][metadata_key] = params.get(metadata_key)
    if algorithm != expected:
        raise SystemExit(
            "AMBI horizon bank is not an exact no-inner bank derivative"
        )
    required = {
        "inner_operator": "none",
        "mpc": False,
        "eval_inner_comparison": False,
        "eval_value": False,
        "value_equivalence_diagnostics": False,
        "inner_diagnostic_rollouts": 0,
        "outer_critic_target": "entropy_augmented",
        "inner_sac_critic_target": "entropy_augmented",
        "ent_coef": "auto",
        "target_entropy": "auto",
    }
    if {key: params.get(key) for key in required} != required:
        raise SystemExit("AMBI no-inner scientific contract drifted")
    expected_group = "ambi-humanoid-walk-no-inner-checkpoint-bank-1p5m"
    expected_run = (
        "AMBITDMPC2-humanoid-walk-outer-prior-no-inner-"
        f"checkpoint-bank-1p5m-train-h{train_horizon}-seed55"
    )
    required_tags = {
        "checkpoint-bank",
        "no-action-local-improvement",
        "zero-inner-model-steps",
        f"train-unroll-horizon-{train_horizon}",
        "outer-planning-horizon-3",
        "temporal-loss-reference-horizon-3",
        "no-online-evaluation",
        "seed55",
    }
    expected_overrides = {
        "seed": 55,
        "device": "cuda",
        "env": "DMControl-v0",
        "total_steps": 1_500_000,
        "episodes": None,
    }
elif family == "ambi-no-inner-tdmpc2-tanh-prior":
    if algorithm.get("alg") != "AMBITDMPC2/AMBITDMPC2":
        raise SystemExit("TD-MPC2-tanh prior bank must retain AMBITDMPC2")
    expected = copy.deepcopy(baseline)
    expected_params = expected["alg_params"]
    expected_params["log_std_mapping"] = "tdmpc2_tanh"
    expected_params["log_std_min"] = -10
    for metadata_key in ("wandb_run_name", "wandb_tags"):
        expected_params[metadata_key] = params.get(metadata_key)
    if algorithm != expected:
        raise SystemExit(
            "TD-MPC2-tanh policy prior is not an exact no-inner bank derivative"
        )
    required = {
        "inner_operator": "none",
        "mpc": False,
        "eval_inner_comparison": False,
        "eval_value": False,
        "value_equivalence_diagnostics": False,
        "inner_diagnostic_rollouts": 0,
        "outer_critic_target": "entropy_augmented",
        "inner_sac_critic_target": "entropy_augmented",
        "ent_coef": "auto",
        "target_entropy": "auto",
        "log_std_mapping": "tdmpc2_tanh",
        "log_std_min": -10,
        "log_std_max": 2,
    }
    if {key: params.get(key) for key in required} != required:
        raise SystemExit("TD-MPC2-tanh no-inner scientific contract drifted")
    expected_group = "ambi-humanoid-walk-no-inner-checkpoint-bank-1p5m"
    expected_run = (
        "AMBITDMPC2-humanoid-walk-outer-prior-no-inner-checkpoint-bank-"
        "1p5m-train-h3-actor-tdmpc2-tanh-upstream-bounds-seed55"
    )
    required_tags = {
        "checkpoint-bank",
        "no-action-local-improvement",
        "zero-inner-model-steps",
        "train-unroll-horizon-3",
        "outer-planning-horizon-3",
        "temporal-loss-reference-horizon-3",
        "actor-log-std-tdmpc2-tanh",
        "actor-log-std-bounds-neg10-2",
        "no-online-evaluation",
        "seed55",
    }
    expected_overrides = {
        "seed": 55,
        "device": "cuda",
        "env": "DMControl-v0",
        "total_steps": 1_500_000,
        "episodes": None,
    }
else:
    raise SystemExit(f"unknown checkpoint-bank family: {family}")

wandb_identity = {
    "wandb": params.get("wandb"),
    "wandb_entity": params.get("wandb_entity"),
    "wandb_project": params.get("wandb_project"),
    "wandb_mode": params.get("wandb_mode"),
    "wandb_group": params.get("wandb_group"),
    "wandb_run_name": params.get("wandb_run_name"),
}
if wandb_identity != {
    "wandb": True,
    "wandb_entity": "rwgao_b-brown-university",
    "wandb_project": "ambi",
    "wandb_mode": "online",
    "wandb_group": expected_group,
    "wandb_run_name": expected_run,
}:
    raise SystemExit(f"W&B identity drifted: {wandb_identity}")
if not required_tags.issubset(set(params.get("wandb_tags", []))):
    raise SystemExit("required scientifically accurate W&B tags are missing")

if manifest.get("configs") != [algorithm_path.stem]:
    raise SystemExit("manifest does not select exactly its checkpoint bank")
if manifest.get("overrides_alg") != expected_overrides:
    raise SystemExit("manifest algorithm overrides drifted")
if manifest.get("env_params") != {
    "task": "humanoid-walk",
    "obs": "state",
    "render_mode": None,
}:
    raise SystemExit("manifest environment drifted")
if manifest.get("trials") != 1 or manifest.get("logs") != "none":
    raise SystemExit("unexpected trial or trajectory-logging policy")
if manifest.get("save_trials") != "none":
    raise SystemExit("unexpected final-model policy")
cadence = manifest.get("checkpoint_every")
if cadence != 25_000 or manifest.get("save_strat") != ["all"]:
    raise SystemExit("unexpected checkpoint retention policy")
if algorithm["total_steps"] // cadence != 60:
    raise SystemExit("checkpoint bank must contain exactly 60 snapshots")
print(
    "Scientific preflight passed:",
    family,
    f"train_h={train_horizon}",
    "planning_h=3 reference_h=3",
)
PY

"$STORAGE_PREFLIGHT" --single-bank
mkdir -p "$ARTIFACT_ROOT"
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

echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Source commit: $SOURCE_COMMIT"
echo "Task: humanoid-walk"
echo "Condition: $CONDITION_LABEL"
echo "Seed: $SEED"
echo "Agent decisions: 1500000"
echo "Raw simulator control steps: 3000000"
echo "Training unroll horizon: $EXPECTED_TRAIN_HORIZON"
echo "Outer planning horizon: 3"
echo "Inner rollout horizon: 3"
echo "Temporal loss reference horizon: 3"
echo "Checkpoint cadence: 25000"
echo "Expected numbered checkpoints: 60"
echo "Manifest: $MANIFEST"
echo "Immutable artifact root: $RUN_ROOT"
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
