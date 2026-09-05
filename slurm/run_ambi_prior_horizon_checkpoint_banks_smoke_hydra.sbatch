#!/usr/bin/env bash
#SBATCH --job-name=ambi-prior-banks-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2501
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%j.out
#SBATCH --error=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%j.err

set -Eeuo pipefail
umask 077

readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
readonly PYTHON="${AMBI_DMC_PYTHON:-/cs/home/rgao48/projects/action-modes/environments/dmcontrol/.venv/bin/python}"
readonly EXPECTED_NODE="gpu2501"
readonly SMOKE_TOTAL_STEPS=3000
readonly SHA_PREFLIGHT="slurm/require_expected_action_modes_sha.sh"
readonly CONFIG_STEMS=(
  "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h5"
  "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h10"
  "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h3_tdmpc2_tanh"
)

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing locked DMControl interpreter: $PYTHON" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to smoke-test from a checkout with uncommitted files." >&2
  git status --short --untracked-files=normal >&2
  exit 1
fi
if [[ "${SLURM_NODELIST:-}" != "$EXPECTED_NODE" ]]; then
  echo "Expected $EXPECTED_NODE, found ${SLURM_NODELIST:-unset}." >&2
  exit 1
fi
if [[ ! -x "$SHA_PREFLIGHT" ]]; then
  echo "Missing source-commit preflight: $SHA_PREFLIGHT" >&2
  exit 1
fi
readonly SOURCE_COMMIT="$("$SHA_PREFLIGHT")"
if [[ -z "${SLURM_TMPDIR:-}" || ! -d "$SLURM_TMPDIR" ]]; then
  echo "A job-local SLURM_TMPDIR is required for this non-durable smoke run." >&2
  exit 1
fi
readonly SMOKE_ROOT="$(mktemp -d "${SLURM_TMPDIR%/}/ambi-prior-banks-smoke-${SLURM_JOB_ID}-XXXXXX")"
readonly SMOKE_ALGORITHM_ROOT="$SMOKE_ROOT/algs"
readonly SMOKE_MANIFEST_ROOT="$SMOKE_ROOT/experiments"
readonly SMOKE_RESULT_ROOT="$SMOKE_ROOT/results"
readonly WANDB_LOCAL_ROOT="$SMOKE_ROOT/wandb"
mkdir -p \
  "$SMOKE_ALGORITHM_ROOT" \
  "$SMOKE_MANIFEST_ROOT" \
  "$SMOKE_RESULT_ROOT" \
  "$WANDB_LOCAL_ROOT/cache" \
  "$WANDB_LOCAL_ROOT/data" \
  "$WANDB_LOCAL_ROOT/artifacts" \
  "$SMOKE_ROOT/tmp" \
  "$SMOKE_ROOT/cache" \
  "$SMOKE_ROOT/torchinductor" \
  "$SMOKE_ROOT/triton"

"$PYTHON" - \
  "$PROJECT_DIR" \
  "$SMOKE_ROOT" \
  "$SMOKE_TOTAL_STEPS" \
  "${CONFIG_STEMS[@]}" <<'PY'
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
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )


project_root = Path(sys.argv[1]).resolve(strict=True)
smoke_root = Path(sys.argv[2]).resolve(strict=True)
total_steps = int(sys.argv[3])
stems = sys.argv[4:]
expected_stems = [
    "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h5",
    "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h10",
    "ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_train_h3_tdmpc2_tanh",
]
if stems != expected_stems:
    raise SystemExit(f"unexpected smoke matrix: {stems}")
if total_steps != 3_000:
    raise SystemExit("smoke budget must remain exactly 3,000 agent decisions")

algorithm_root = project_root / "configs/dmcontrol/algs"
manifest_root = project_root / "configs/dmcontrol/experiments"
output_algorithm_root = smoke_root / "algs"
output_manifest_root = smoke_root / "experiments"

for stem in stems:
    source_algorithm = algorithm_root / f"{stem}.json"
    source_manifest = manifest_root / f"{stem}.json"
    if not source_algorithm.is_file() or not source_manifest.is_file():
        raise SystemExit(f"missing tracked smoke source for {stem}")
    algorithm = load(source_algorithm)
    manifest = load(source_manifest)
    params = algorithm.get("alg_params", {})

    if algorithm.get("alg") != "AMBITDMPC2/AMBITDMPC2":
        raise SystemExit(f"{stem} is not an AMBITDMPC2 bank")
    if algorithm.get("seed") != 55 or algorithm.get("total_steps") != 1_500_000:
        raise SystemExit(f"{stem} production seed or budget drifted")
    if params.get("inner_operator") != "none" or params.get("mpc") is not False:
        raise SystemExit(f"{stem} no-inner policy-collection contract drifted")
    if params.get("eval_freq") is not None:
        raise SystemExit(f"{stem} unexpectedly enables online evaluation")
    resolved_seed_steps = params.get("seed_steps", 2_500)
    if resolved_seed_steps != 2_500 or total_steps <= resolved_seed_steps:
        raise SystemExit(
            f"{stem} smoke budget does not cross its 2,500-step seed phase"
        )
    if manifest.get("configs") != [stem]:
        raise SystemExit(f"{stem} manifest selection drifted")
    if manifest.get("checkpoint_every") != 25_000:
        raise SystemExit(f"{stem} production checkpoint cadence drifted")
    if manifest.get("save_strat") != ["all"]:
        raise SystemExit(f"{stem} production retention policy drifted")

    smoke_algorithm = copy.deepcopy(algorithm)
    smoke_algorithm["total_steps"] = total_steps
    smoke_algorithm["alg_params"]["wandb"] = False

    smoke_manifest = copy.deepcopy(manifest)
    smoke_manifest["overrides_alg"]["total_steps"] = total_steps
    smoke_manifest["logs"] = "none"
    smoke_manifest["save_trials"] = "none"
    smoke_manifest["checkpoint_every"] = None
    smoke_manifest["save_strat"] = "none"

    (output_algorithm_root / f"{stem}.json").write_text(
        json.dumps(smoke_algorithm, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_manifest_root / f"{stem}.json").write_text(
        json.dumps(smoke_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

print(f"Materialized {len(stems)} temporary smoke cells under {smoke_root}")
PY

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="$SMOKE_ROOT/tmp"
export XDG_CACHE_HOME="$SMOKE_ROOT/cache"
export TORCHINDUCTOR_CACHE_DIR="$SMOKE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$SMOKE_ROOT/triton"
export WANDB_MODE=disabled
export WANDB_DIR="$WANDB_LOCAL_ROOT"
export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"
export WANDB_DISABLE_CODE=true

echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Source commit: $SOURCE_COMMIT"
echo "Smoke cells: ${#CONFIG_STEMS[@]} (sequential)"
echo "Agent decisions per cell: $SMOKE_TOTAL_STEPS"
echo "Checkpoints: disabled"
echo "W&B: disabled"
echo "Node-local smoke root: $SMOKE_ROOT"
echo "Python: $PYTHON"
"$PYTHON" --version
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
"$PYTHON" -c 'import os, torch; name=torch.cuda.get_device_name(); assert torch.cuda.device_count() == 1 and "L40S" in name, (torch.cuda.device_count(), name); print("torch:", torch.__version__); print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")); print("visible CUDA devices:", torch.cuda.device_count()); print("cuda:", name)'

for stem in "${CONFIG_STEMS[@]}"; do
  readonly_cell_result_root="$SMOKE_RESULT_ROOT/$stem"
  cell_stdout="$SMOKE_ROOT/${stem}.stdout.log"
  if [[ -e "$readonly_cell_result_root" || -L "$readonly_cell_result_root" ]]; then
    echo "Refusing to reuse smoke result root: $readonly_cell_result_root" >&2
    exit 1
  fi
  echo "Starting smoke cell: $stem"
  srun --unbuffered --kill-on-bad-exit=1 \
    "$PYTHON" main.py \
    --run "$SMOKE_MANIFEST_ROOT/$stem.json" \
    --alg-dir "$SMOKE_ALGORITHM_ROOT" \
    --log-dir "$readonly_cell_result_root" \
    --alg-index 0 \
    --trial-index 0 \
    --num-runs 1 2>&1 | tee "$cell_stdout"
  if ! grep -Fq "Pretraining TD-MPC2 on seed data..." "$cell_stdout"; then
    echo "Smoke cell did not cross into learned updates: $stem" >&2
    exit 1
  fi
  echo "Completed smoke cell: $stem"
done

unexpected_checkpoint="$(find "$SMOKE_ROOT" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.metadata.json' \) -print -quit)"
if [[ -n "$unexpected_checkpoint" ]]; then
  echo "Smoke run unexpectedly created checkpoint artifact: $unexpected_checkpoint" >&2
  exit 1
fi
echo "All three node-local smoke cells completed without checkpoint artifacts."
