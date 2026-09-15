#!/usr/bin/env bash
#SBATCH --job-name=ambi-prior-split-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/rgao48/ambi/prior-split-study/slurm/%x-%j.out
#SBATCH --error=/oscar/scratch/rgao48/ambi/prior-split-study/slurm/%x-%j.err

# This job exercises two production-shape outer updates per configuration,
# real DMControl transitions, exact checkpoint reload, and strict CUDA compile.
# It does not start the 2M-decision experiment or contact W&B.
set -Eeuo pipefail
umask 077
fail() { printf 'Oscar split-prior validation error: %s\n' "$*" >&2; exit 2; }
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested full Git SHA.}"
: "${AMBI_SPLIT_GATE_OUTPUT_ROOT:?Supply a fresh absolute scratch output directory.}"
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail "invalid expected commit SHA"
cd "${SLURM_SUBMIT_DIR:?Submit from the clean repository root.}"
project_dir="$(pwd -P)"
[[ "$project_dir" == "$(git rev-parse --show-toplevel)" ]] || fail "submit from the repository root"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "HEAD differs from the tested commit"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail "checkout is dirty"
[[ "$AMBI_SPLIT_GATE_OUTPUT_ROOT" == /* && ! -e "$AMBI_SPLIT_GATE_OUTPUT_ROOT" ]] || fail "gate output must be fresh and absolute"
ambi_python="${AMBI_PYTHON:-$project_dir/environments/dmcontrol/.venv/bin/python}"
[[ -x "$ambi_python" ]] || fail "missing locked DMControl interpreter"
mkdir -p "$(dirname -- "$AMBI_SPLIT_GATE_OUTPUT_ROOT")"
parent_dir="$(cd -- "$(dirname -- "$AMBI_SPLIT_GATE_OUTPUT_ROOT")" && pwd -P)"
[[ "$parent_dir" != "$project_dir" && "$parent_dir" != "$project_dir/"* ]] || fail "gate output must be outside the checkout"
mkdir "$AMBI_SPLIT_GATE_OUTPUT_ROOT"
export AMBI_SPLIT_GATE_OUTPUT_ROOT EXPECTED_ACTION_MODES_SHA
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-prior-split-gate-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/wandb"
export AMBI_PYTHON="$ambi_python" AMBI_RUN_SPLIT_PRIOR_CUDA_GATE=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WANDB_MODE=offline
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" WANDB_DIR="$local_root/wandb"
srun --unbuffered --kill-on-bad-exit=1 "$ambi_python" -m pytest -q -s -p no:cacheprovider \
  tests/test_ambi_prior_split_cuda_gate.py

# Missing case reports (including skipped GPU tests) prevent a passing receipt.
"$ambi_python" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['AMBI_SPLIT_GATE_OUTPUT_ROOT'])
names = ('ambi_prior_split_clip_fixed0p0021', 'ambi_prior_split_clip_target10p5', 'ambi_prior_split_clip_target21')
cases = [json.loads((root / (name + '.json')).read_text()) for name in names]
source_commit = os.environ['EXPECTED_ACTION_MODES_SHA']
for name, case in zip(names, cases):
    assert case['config'] == name and case['source_commit'] == source_commit
    assert case['passed'] and case['compile_strict'] and case['checkpoint_roundtrip']
    assert case['optimizer_updates'] >= 2 and not any(case['compile_fallbacks'])
receipt = root / 'receipt.json'
with receipt.open('x') as stream:
    json.dump({'schema': 1, 'passed': True, 'source_commit': source_commit, 'cases': cases}, stream, indent=2)
    stream.write('\n')
print('SPLIT_PRIOR_CUDA_GATE_RECEIPT', receipt)
PY
