#!/usr/bin/env bash
#SBATCH --job-name=aux-return-mppi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the synchronized source checkout}"
: "${EXPECTED_ACTION_MODES_SHA:?}"
: "${AMBI_CHECKPOINT_PREFIX:?}"
: "${AMBI_BENCHMARK_OUTPUT_ROOT:?}"
: "${SLURM_ARRAY_TASK_ID:?Use checkpoint indices 1 through 40}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" && -z "$(git status --porcelain)" ]]
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ && "$SLURM_ARRAY_TASK_ID" -ge 1 && "$SLURM_ARRAY_TASK_ID" -le 40 ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
case "${1:-}" in ''|--smoke) ;; *) exit 2 ;; esac
step=$((SLURM_ARRAY_TASK_ID * 25000))
checkpoint="${AMBI_CHECKPOINT_PREFIX}${step}"
output="${AMBI_BENCHMARK_OUTPUT_ROOT}/step_${step}"
[[ "$checkpoint" == /* && "$output" == /* && -f "$checkpoint" && -f "$checkpoint.metadata.json" ]]
mkdir -p "$AMBI_BENCHMARK_OUTPUT_ROOT"
mkdir "$output"
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TORCHINDUCTOR_COMPILE_THREADS=1
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
common=(--matrix configs/research/ambi_aux_return_mppi.json --checkpoint "$checkpoint"
        --device cuda --bundle-dir "$output/bundle" --output "$output/results.json")
if [[ "${1:-}" == --smoke ]]; then
    "$PYTHON_BIN" -m pytest -q tests/test_aux_return_mppi.py tests/test_aux_return_evaluation.py
    "$PYTHON_BIN" evaluate_ambi_checkpoint.py "${common[@]}" --seeds 101 102 --max-steps 3
    "$PYTHON_BIN" - "$output/bundle" <<'PY'
import json, os, sys
from utils.eval_series_data import load_records
from utils.eval_series import load_run
assignments = json.load(open(os.environ['EVAL_RUN_MAP']))
for record in load_records(sys.argv[1], checkpoint_inventory=os.environ['CHECKPOINT_MANIFEST']):
    expected = load_run(assignments[record['selector']])['identity']
    for key in ('backbone', 'planner', 'science'):
        assert record['identity'][key] == expected[key], key
print('Smoke results match the prepared production curve identities.')
PY
else
    : "${CHECKPOINT_MANIFEST:?}"
    : "${EVAL_RUN_MAP:?}"
    "$PYTHON_BIN" evaluate_ambi_checkpoint.py "${common[@]}" --seeds 101 102 103 104 105 --max-steps 500 \
        --checkpoint-inventory "$CHECKPOINT_MANIFEST" --eval-run-map "$EVAL_RUN_MAP"
fi
"$PYTHON_BIN" - "$output/results.json" <<'PY'
import json, sys
results = json.load(open(sys.argv[1]))['results']
assert len(results) == 2
assert [r['value_routing']['inner_horizon_critic_source'] for r in results] == ['sac', 'aux_return']
for result in results:
    assert result['outer_state_unchanged'] and not result['nonfinite_model_metrics']
    cfg = result['resolved_config']
    assert cfg['inner_mppi_num_samples'] == 512 and cfg['inner_mppi_iterations'] == 8
    assert cfg['mppi_terminal_q_reduction'] == 'mean_pair'
    assert cfg['inner_actor_source'] == cfg['inner_horizon_actor_source'] == 'sac'
    assert result['model_metrics']['inner_model_steps']['mean'] == 12336
    for component in ('actor', 'critic', 'temperature'):
        assert result['model_metrics'][f'inner_{component}_optimizer_steps']['mean'] == 0
    assert result['resolved_device'].startswith('cuda')
print('Verified both CUDA routes, MPPI budget, finite diagnostics and complete frozen state.')
PY
"$PYTHON_BIN" report_ambi_benchmark.py --bundle "$output/bundle" --output "$output/comparison.html" \
    --title "Auxiliary SAC target -21 detached: MPPI soft-Q versus return-Q at $step decisions"
