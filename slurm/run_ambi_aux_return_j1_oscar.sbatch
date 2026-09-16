#!/usr/bin/env bash
#SBATCH --job-name=aux-return-j1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40|a5500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
: "${EXPECTED_ACTION_MODES_SHA:?}"
: "${AMBI_CHECKPOINT_PREFIX:?}"
: "${AMBI_BENCHMARK_OUTPUT_ROOT:?}"
: "${CHECKPOINT_MANIFEST:?}"
: "${EVAL_RUN_MAP:?}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" && -z "$(git status --porcelain)" ]]
[[ "${SLURM_ARRAY_TASK_ID:?}" =~ ^[0-9]+$ && "$SLURM_ARRAY_TASK_ID" -ge 1 && "$SLURM_ARRAY_TASK_ID" -le 40 ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
case "${1:-}" in ''|--smoke) ;; *) exit 2 ;; esac
step=$((SLURM_ARRAY_TASK_ID * 25000))
output="$AMBI_BENCHMARK_OUTPUT_ROOT/step_$step"
mkdir -p "$AMBI_BENCHMARK_OUTPUT_ROOT"
mkdir "$output"
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TORCHINDUCTOR_COMPILE_THREADS=1
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
common=(--matrix configs/research/ambi_aux_return_j1.json --checkpoint "${AMBI_CHECKPOINT_PREFIX}${step}"
        --device cuda --bundle-dir "$output/bundle" --output "$output/results.json")
if [[ "${1:-}" == --smoke ]]; then
    "$PYTHON_BIN" evaluate_ambi_checkpoint.py "${common[@]}" --seeds 101 --max-steps 20
else
    : "${AMBI_PRIOR_REFERENCE_ROOT:?}"
    "$PYTHON_BIN" evaluate_ambi_checkpoint.py "${common[@]}" --seeds 101 102 103 104 105 --max-steps 500 \
        --reference-bundle "$AMBI_PRIOR_REFERENCE_ROOT/step_$step/bundle" \
        --checkpoint-inventory "$CHECKPOINT_MANIFEST" --eval-run-map "$EVAL_RUN_MAP"
fi
"$PYTHON_BIN" - "$output" "${1:-}" <<'PY'
import json, os, sys
from pathlib import Path
from utils.eval_series_data import load_records
from utils.eval_series import load_run
output, smoke = Path(sys.argv[1]), sys.argv[2] == '--smoke'
manifest = json.load(open(output/'bundle/manifest.json'))
results = json.load(open(output/'results.json'))['results']
assert len(results) == len(manifest['runs']) == 2
for result, source, run in zip(results, ('sac', 'aux_return'), manifest['runs']):
    assert result['outer_state_unchanged'] and not result['nonfinite_model_metrics']
    cfg = result['resolved_config']
    assert cfg['compile'] and cfg['compile_strict']
    assert cfg['inner_actor_source'] == cfg['inner_horizon_actor_source'] == 'sac'
    assert cfg['inner_critic_source'] == cfg['inner_horizon_critic_source'] == source
    assert cfg['inner_rounds'] == cfg['inner_rollout_horizon'] == 1
    assert cfg['inner_batch_size'] == 256 and cfg['inner_rollouts_per_round'] == 128
    assert cfg['inner_sac_critic_target'] == 'reward_only' and cfg['inner_finite_horizon']
    assert cfg['inner_execution_action'] == 'mean' and cfg['inner_log_std_mapping'] == 'direct_clamp'
    assert cfg['sac_actor_loss_scale_mode'] == 'none'
    assert cfg.get('aux_return_sac_actor_loss_scale_mode') in (None, 'none')
    metrics = result['model_metrics']
    for key, expected in [('inner_model_steps',128),('inner_actor_optimizer_steps',4),
                          ('inner_critic_optimizer_steps',32),('inner_temperature_optimizer_steps',0),
                          ('inner_alpha',0),('inner_compile_fallback',0)]:
        assert metrics[key]['mean'] == expected, (key, metrics[key])
    assert result['environment_seeds'] == ([101] if smoke else [101,102,103,104,105])
    assert all(e['length'] == (20 if smoke else 500) for e in result['episodes'])
    assert result['resolved_device'].startswith('cuda')
    if not smoke:
        assert all(not e['truncated_by_evaluator'] for e in result['episodes'])
        assert all('paired_return_delta' in e for e in run['episodes'])
    print(json.dumps({'selector': result['selector'], 'control_seconds':sum(e['control_seconds'] for e in result['episodes']),
                      'warmup_seconds':run['warmup_including_compile_seconds'], 'decisions':sum(e['length'] for e in result['episodes'])}))
assignments = json.load(open(os.environ['EVAL_RUN_MAP']))
for record in load_records(output/'bundle', inventory_path=os.environ['CHECKPOINT_MANIFEST']):
    expected = load_run(assignments[record['selector']])['identity']
    for key in ('backbone','planner','science'):
        assert record['identity'][key] == expected[key], key
    if not smoke:
        assert record['identity']['protocol'] == expected['protocol']
        assert record['metrics']['eval/paired_episodes'] == 5
print('Verified both J1 critic routes, optimizer work, paired gains, strict CUDA and frozen outer state.')
PY
"$PYTHON_BIN" report_ambi_benchmark.py --bundle "$output/bundle" --output "$output/comparison.html" \
    --title "Auxiliary SAC target -21 detached: J1 soft-Q versus return-Q at $step decisions"
