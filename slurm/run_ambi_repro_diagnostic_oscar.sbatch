#!/usr/bin/env bash
#SBATCH --job-name=ambi-repro-diagnostic
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -Eeuo pipefail
cd "${AMBI_PROJECT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_ACTION_MODES_SHA:?}" && -z "$(git status --porcelain)" ]]
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "${AMBI_DIAGNOSTIC_ROOT:?}"
if [[ "${1:-}" == --pin-kernels ]]; then
  for profile in pinned-current pinned-cold critic-softmax-8 critic-softmax-32; do
    reference="${AMBI_KERNEL_REFERENCE_ROOT:?}/current/report.json"
    extra=()
    [[ "$profile" != pinned-cold ]] || reference="$AMBI_KERNEL_REFERENCE_ROOT/cold/report.json"
    [[ "$profile" != critic-softmax-8 ]] || extra+=(--override-kernel bddca0b1e02f:8:8)
    [[ "$profile" != critic-softmax-32 ]] || extra+=(--override-kernel bddca0b1e02f:32:8)
    (
      export TORCHINDUCTOR_CACHE_DIR="$AMBI_DIAGNOSTIC_ROOT/$profile-inductor"
      export TRITON_CACHE_DIR="$AMBI_DIAGNOSTIC_ROOT/$profile-triton"
      "${AMBI_DMC_PYTHON:?}" diagnose_ambi_reproducibility.py \
        --bundle "${AMBI_DIAGNOSTIC_BUNDLE:?}" --reset-sequence 0 \
        --kernel-configs "$reference" --output "$AMBI_DIAGNOSTIC_ROOT/$profile" "${extra[@]}"
    )
  done
  exit
fi
if [[ "${1:-}" == --source-and-cache ]]; then
  for profile in historical current cold; do
    run_root="$AMBI_PROJECT_DIR"
    [[ "$profile" != historical ]] || run_root="${AMBI_HISTORICAL_SOURCE:?}"
    (
      cd "$run_root"
      [[ -z "$(git status --porcelain)" ]]
      if [[ "$profile" == cold ]]; then
        export TORCHINDUCTOR_CACHE_DIR="$AMBI_DIAGNOSTIC_ROOT/cold-inductor"
        export TRITON_CACHE_DIR="$AMBI_DIAGNOSTIC_ROOT/cold-triton"
      fi
      "${AMBI_DMC_PYTHON:?}" -c 'import runpy,sys; runpy.run_path(sys.argv.pop(1),run_name="__main__")' \
        "$AMBI_PROJECT_DIR/diagnose_ambi_reproducibility.py" \
        --bundle "${AMBI_DIAGNOSTIC_BUNDLE:?}" --reset-sequence 0 \
        --output "$AMBI_DIAGNOSTIC_ROOT/$profile"
    )
  done
  exit
fi
for profile in production compile-a compile-b deterministic eager; do
  extra=(--mode compile)
  [[ "$profile" != deterministic ]] || extra+=(--deterministic)
  [[ "$profile" != eager ]] || extra=(--mode eager)
  [[ "$profile" != production ]] || extra+=(--no-record)
  "${AMBI_DMC_PYTHON:?}" diagnose_ambi_reproducibility.py \
    --bundle "${AMBI_DIAGNOSTIC_BUNDLE:?}" \
    --output "$AMBI_DIAGNOSTIC_ROOT/$profile" "${extra[@]}"
done
