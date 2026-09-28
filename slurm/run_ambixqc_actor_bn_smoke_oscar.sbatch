#!/usr/bin/env bash
# Functional validation only: one CUDA test job, then eight short episode cells.
# Episode indices: shared/detached checkpoint x return-J1/soft-J1/return-J8/soft-J8.
# Supply CHECKPOINT_MANIFEST with two hash-pinned rows in that backbone order.
#SBATCH --job-name=axqc-actor-bn-smoke
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the clean repository root.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit.}"
: "${RESULT_ROOT:?Supply an absolute scratch directory.}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
MODE="${AMBIXQC_SMOKE_MODE:-episodes}"
[[ "$MODE" == tests || "$MODE" == episodes ]] || exit 2
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ && "$RESULT_ROOT" == /* && "$PYTHON_BIN" == /* ]] || exit 2
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" && -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || exit 2
readonly LOCK_SHA=f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6
ENV_PROJECT="$(cd "$(dirname "$PYTHON_BIN")/../.." && pwd -P)"
for lock in environments/dmcontrol/uv.lock "$ENV_PROJECT/uv.lock"; do
  [[ "$(sha256sum "$lock" | awk '{print $1}')" == "$LOCK_SHA" ]] || exit 2
done
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=disabled WANDB_DISABLED=true
unset PYTEST_ADDOPTS WANDB_RUN_ID WANDB_RESUME
export TMPDIR="${SLURM_TMPDIR:-/tmp}/axqc-actor-bn-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$RESULT_ROOT"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor" TRITON_CACHE_DIR="$TMPDIR/triton"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
srun --unbuffered "$PYTHON_BIN" -c 'import sys,torch; assert sys.version_info[:2]==(3,10); assert torch.__version__.split("+")[0]=="2.3.1"; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))'
if [[ "$MODE" == tests ]]; then
  srun --unbuffered "$PYTHON_BIN" -m pytest -q -rs -p no:cacheprovider \
    tests/test_ambixqc_actor_bn.py \
    tests/test_ambixqc_compile.py::test_cuda_complete_compiled_inner_action_matches_eager_and_reuses_graphs \
    --junitxml="$RESULT_ROOT/cuda-tests.xml"
else
  : "${CHECKPOINT_MANIFEST:?Supply two immutable auxiliary checkpoint rows.}"
  : "${SLURM_ARRAY_TASK_ID:?Supply an episode cell index from zero through seven.}"
  mapfile -t selected < <("$PYTHON_BIN" - "$CHECKPOINT_MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import hashlib,json,sys
from pathlib import Path
rows=json.loads(Path(sys.argv[1]).read_text())["checkpoints"]
index=int(sys.argv[2])
assert len(rows)==2 and 0<=index<8
assert [r['arm'] for r in rows]==['aux_shared','aux_detached']
row=rows[index//4]
assert row['step']==500000
for key,hash_key in [('path','sha256'),('metadata_path','metadata_sha256')]:
    path=Path(row[key]); assert path.is_absolute()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row[hash_key]
metadata=json.loads(Path(row['metadata_path']).read_text())
assert metadata['checkpoint']['step']==500000
params=metadata['trial_run_params']['alg_params']
assert params.get('compile',False) is False  # Episode smoke intentionally uses eager execution.
assert params['aux_return_mode']=='xqc'
assert params['aux_return_detach_representation']==(row['arm']=='aux_detached')
print(row['path'])
print(('controller/return_return_j1','controller/soft_soft_j1','controller/return_return_j8','controller/soft_soft_j8')[index%4])
PY
)
  [[ "${#selected[@]}" == 2 ]] || exit 2
  CELL_ROOT="$RESULT_ROOT/cell-$SLURM_ARRAY_TASK_ID"
  [[ ! -e "$CELL_ROOT" ]] || exit 2
  srun --unbuffered "$PYTHON_BIN" evaluate_ambi_checkpoint.py \
    --matrix configs/research/ambixqc_humanoid_h1_actor_bn_screen.json \
    --checkpoint "${selected[0]}" --preset "${selected[1]}" \
    --device cuda --seeds 101 --controller-seed 12345 --max-steps 3 \
    --bundle-dir "$CELL_ROOT/bundle" --output "$CELL_ROOT/results.json"
  "$PYTHON_BIN" - "$CELL_ROOT/bundle/manifest.json" <<'PY'
import gzip,json,math,sys
from pathlib import Path
path=Path(sys.argv[1]); manifest=json.loads(path.read_text())
assert manifest['status']=='complete' and len(manifest['runs'])==1
run=manifest['runs'][0]; result=run['result']; cfg=result['resolved_config']
assert result['outer_state_unchanged'] and result['outer_updates_before']==result['outer_updates_after']
assert not result.get('nonfinite_model_metrics') and not result.get('nonfinite_trace_metrics')
assert cfg['inner_actor_bn_mode']=='running' and cfg['inner_temperature_mode']=='auto'
assert cfg['inner_rollout_horizon']==1 and cfg['inner_rounds'] in (1,8)
assert len(run['episodes'])==1 and run['episodes'][0]['length']==3 and run['episodes'][0]['seed']==101
assert math.isfinite(run['episodes'][0]['return'])
events=[]
for relative in run['trace_files']:
    with gzip.open(path.parent/relative,'rt') as stream: events.extend(json.loads(line) for line in stream)
assert len(events)==3
for event in events:
    assert not event.get('nonfinite')
    assert event['critic_updates']==3*cfg['inner_rounds']
    assert event['actor_updates']==event['temperature_updates']==cfg['inner_rounds']
    assert event['metrics']['decision/inner_model_steps']==256*cfg['inner_rounds']
print('Validated H1 actor-BN smoke:',run['selector'],'three finite decisions; frozen outer state; exact optimizer dose')
PY
fi
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || exit 2
