#!/usr/bin/env bash
#SBATCH --job-name=aux625k-step-smoke
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40|a5500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${AMBI_PROJECT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_ACTION_MODES_SHA:?}" && -z "$(git status --porcelain)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TORCHINDUCTOR_COMPILE_THREADS=1
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
"${AMBI_DMC_PYTHON:?}" -c 'import torch; assert torch.cuda.is_available()'
"$AMBI_DMC_PYTHON" -m pytest -q -p no:cacheprovider tests/test_ambi_component_step_updates.py \
    tests/test_ambi_inner_step_updates.py::test_inductor_parallel_step_matches_eager \
    tests/test_ambi_inner_decoupling.py::test_cuda_act_preserves_all_global_rng_streams_and_outer_state \
    --junitxml="${AMBI_CAMPAIGN_ROOT:?}/gpu-tests.xml"
"$AMBI_DMC_PYTHON" - "$AMBI_CAMPAIGN_ROOT/gpu-tests.xml" <<'PY'
import sys,xml.etree.ElementTree as ET
suites=ET.parse(sys.argv[1]).getroot().findall('testsuite')
assert suites and all(int(s.get('skipped','0'))==0 and int(s.get('failures','0'))==0 and int(s.get('errors','0'))==0 for s in suites)
PY
exec "$AMBI_DMC_PYTHON" slurm/ambi_aux_hj_sweep.py worker --root "$AMBI_CAMPAIGN_ROOT" --smoke
