#!/usr/bin/env bash
#SBATCH --job-name=aux-return-prior
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40|a5500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
: "${EXPECTED_ACTION_MODES_SHA:?}"
: "${AMBI_CHECKPOINT_PREFIX:?}"
: "${AMBI_PRIOR_OUTPUT_ROOT:?}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" && -z "$(git status --porcelain)" ]]
[[ "${SLURM_ARRAY_TASK_ID:?}" =~ ^[0-9]+$ && "$SLURM_ARRAY_TASK_ID" -ge 1 && "$SLURM_ARRAY_TASK_ID" -le 40 ]]
step=$((SLURM_ARRAY_TASK_ID * 25000))
output="$AMBI_PRIOR_OUTPUT_ROOT/step_$step"
mkdir -p "$AMBI_PRIOR_OUTPUT_ROOT"
mkdir "$output"
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
"$PYTHON_BIN" evaluate_ambi_checkpoint.py --matrix configs/research/ambi_aux_return_prior_reference.json \
    --checkpoint "${AMBI_CHECKPOINT_PREFIX}${step}" --device cuda --seeds 101 102 103 104 105 \
    --max-steps 500 --bundle-dir "$output/bundle" --output "$output/results.json"
"$PYTHON_BIN" - "$output/results.json" <<'PY'
import json,sys
r,=json.load(open(sys.argv[1]))['results']
assert r['outer_state_unchanged'] and not r['nonfinite_model_metrics']
assert r['resolved_config']['inner_operator']=='none'
assert r['resolved_config']['inner_actor_source']=='sac'
assert r['environment_seeds']==[101,102,103,104,105]
assert all(e['length']==500 and not e['truncated_by_evaluator'] for e in r['episodes'])
assert r['model_metrics']['inner_model_steps']['mean']==0
print('Verified full paired policy-mean reference with unchanged outer state.')
PY
