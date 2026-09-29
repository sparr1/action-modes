#!/usr/bin/env bash
#SBATCH --job-name=axqc-bn-probe
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${EXPECTED_ACTION_MODES_SHA:?Exact tested commit required}"
: "${AMBIXQC_ACTION_MODES_DIR:?Clean source checkout required}"
: "${AMBIXQC_PYTHON:?Existing locked DMControl Python required}"
: "${AMBIXQC_RESULTS_ROOT:?Durable study directory required}"
: "${AMBIXQC_PROBE_CHECKPOINT:?Pinned checkpoint required}"
: "${SLURM_JOB_ID:?Scheduler allocation required}"
cd "$AMBIXQC_ACTION_MODES_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]]
[[ "$AMBIXQC_RESULTS_ROOT" == /* && -d "$AMBIXQC_RESULTS_ROOT" ]]
[[ "$AMBIXQC_PROBE_CHECKPOINT" == /* && -f "$AMBIXQC_PROBE_CHECKPOINT" ]]
readonly ENV_PROJECT="$(cd "$(dirname "$AMBIXQC_PYTHON")/../.." && pwd -P)"
for lock in environments/dmcontrol/uv.lock "$ENV_PROJECT/uv.lock"; do
  [[ "$(sha256sum "$lock" | awk '{print $1}')" == f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6 ]]
done
readonly CACHE="${SLURM_TMPDIR:-/tmp}/axqc-bn-probe-${SLURM_JOB_ID}"
[[ ! -e "$CACHE" && ! -e "$AMBIXQC_RESULTS_ROOT/probe" ]]
mkdir -p "$CACHE/tmp" "$CACHE/xdg" "$CACHE/inductor" "$CACHE/triton" "$CACHE/mpl" "$AMBIXQC_RESULTS_ROOT/probe"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" OPENBLAS_NUM_THREADS=1
export TMPDIR="$CACHE/tmp" XDG_CACHE_HOME="$CACHE/xdg" MPLCONFIGDIR="$CACHE/mpl"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton"
export WANDB_MODE=disabled WANDB_DISABLED=true
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME PYTEST_ADDOPTS
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader > "$AMBIXQC_RESULTS_ROOT/probe/gpu.txt"
srun --unbuffered "$AMBIXQC_PYTHON" -m pytest -q -rs -p no:cacheprovider \
  tests/test_ambixqc_bn_probe.py | tee "$AMBIXQC_RESULTS_ROOT/probe/pytest.log"
srun --unbuffered "$AMBIXQC_PYTHON" run_ambixqc_bn_probe.py \
  --checkpoint "$AMBIXQC_PROBE_CHECKPOINT" --output "$AMBIXQC_RESULTS_ROOT/probe/smoke" --device cuda --mode smoke
srun --unbuffered "$AMBIXQC_PYTHON" run_ambixqc_bn_probe.py \
  --checkpoint "$AMBIXQC_PROBE_CHECKPOINT" --output "$AMBIXQC_RESULTS_ROOT/probe/production" --device cuda --mode production
touch "$AMBIXQC_RESULTS_ROOT/probe/PASS"
