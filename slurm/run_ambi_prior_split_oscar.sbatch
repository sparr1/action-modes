#!/usr/bin/env bash
#SBATCH --job-name=ambi-prior-split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/rgao48/ambi/prior-split-study/slurm/%x-%A_%a.out
#SBATCH --error=/oscar/scratch/rgao48/ambi/prior-split-study/slurm/%x-%A_%a.err

# Submit all three independent backbones after the validation job succeeds.
# Create the scheduler-log directory above before sbatch; overrides must also
# point outside the checkout. Check live scratch quota for at least 24 GiB of
# aggregate checkpoint headroom. These portable checkpoints do not resume a
# preempted/requeued training run.
set -Eeuo pipefail
umask 077
fail() { printf 'Oscar split-prior launcher error: %s\n' "$*" >&2; exit 2; }

: "${SLURM_JOB_ID:?Run through Slurm.}"
: "${SLURM_ARRAY_JOB_ID:?Run as the three-cell Slurm array.}"
: "${SLURM_ARRAY_TASK_ID:?Run as the three-cell Slurm array.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested full Git SHA.}"
: "${AMBI_SPLIT_GATE_RECEIPT:?Supply the successful validation receipt.}"
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail "invalid expected commit SHA"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-2]$ ]] || fail "array task must be in 0..2"
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || fail "portable checkpoints cannot resume after requeue"
[[ "$AMBI_SPLIT_GATE_RECEIPT" == /* && -f "$AMBI_SPLIT_GATE_RECEIPT" ]] || fail "gate receipt must exist at an absolute path"

project_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$project_dir"
git_root="$(git rev-parse --show-toplevel)"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$project_dir" ]] || fail "submit from the repository root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail "checkout is dirty"
source_commit="$(git rev-parse HEAD)"
[[ "$source_commit" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "HEAD differs from the tested commit"

ambi_python="${AMBI_PYTHON:-$project_dir/environments/dmcontrol/.venv/bin/python}"
[[ -x "$ambi_python" ]] || fail "missing locked DMControl interpreter: $ambi_python"
configs=(ambi_prior_split_clip_fixed0p0021 ambi_prior_split_clip_target10p5 ambi_prior_split_clip_target21)
config="${configs[$SLURM_ARRAY_TASK_ID]}"
manifest="configs/dmcontrol/experiments/ambi_prior_split_study.json"
[[ -f "$manifest" && -f "configs/dmcontrol/algs/$config.json" ]] || fail "missing split study configuration"

output_root="${AMBI_SPLIT_OUTPUT_ROOT:-/oscar/scratch/rgao48/ambi/prior-split-study}"
[[ "$output_root" == /* ]] || fail "output root must be absolute"
campaign="${AMBI_SPLIT_CAMPAIGN:-job_${SLURM_ARRAY_JOB_ID}}"
[[ "$campaign" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "campaign must be a filesystem-safe basename"
mkdir -p "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"
[[ "$output_root" != "$project_dir" && "$output_root" != "$project_dir/"* ]] || fail "checkpoint output must be outside the source checkout"
available_kb="$(df -Pk "$output_root" | awk 'NR == 2 {print $4}')"
[[ "$available_kb" =~ ^[0-9]+$ ]] || fail "could not determine output space"
(( available_kb >= 8 * 1024 * 1024 )) || fail "less than 8 GiB free for this checkpoint bank"

local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-prior-split-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" \
  "$local_root/wandb" "$local_root/data" "$local_root/artifacts"
export AMBI_PYTHON="$ambi_python"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache"
export WANDB_MODE=online WANDB_DISABLE_CODE=true
export WANDB_DIR="$local_root/wandb" WANDB_CACHE_DIR="$local_root/cache"
export WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"

# Read existing credentials without printing or copying their contents. The
# W&B SDK uses the same netrc; no login or credential installation occurs here.
"$ambi_python" - "$AMBI_SPLIT_GATE_RECEIPT" "$source_commit" "$config" <<'PY'
import json, netrc, pathlib, sys
receipt_path, source_commit, name = sys.argv[1:]
receipt = json.loads(pathlib.Path(receipt_path).read_text())
assert receipt.get('schema') == 1 and receipt.get('passed') is True, 'invalid CUDA gate receipt'
assert receipt['source_commit'] == source_commit, 'CUDA gate tested another commit'
cases = {case['config']: case for case in receipt['cases']}
expected = {'ambi_prior_split_clip_fixed0p0021', 'ambi_prior_split_clip_target10p5', 'ambi_prior_split_clip_target21'}
assert set(cases) == expected, 'CUDA gate did not exercise all three backbones'
for case in cases.values():
    assert case['passed'] and case['compile_strict'] and case['optimizer_updates'] >= 2
    assert case['checkpoint_roundtrip'] and not any(case['compile_fallbacks']), 'CUDA compilation fell back'
params = json.loads(pathlib.Path('configs/dmcontrol/algs', name + '.json').read_text())['alg_params']
assert params['compile'] and params['compile_strict'], 'production must fail instead of falling back'
assert params['critic_value_mode'] == 'return_entropy' and params['inner_operator'] == 'none'
authentication = netrc.netrc().authenticators('api.wandb.ai')
assert authentication and authentication[2], 'existing api.wandb.ai netrc credentials are required'
import torch
assert torch.cuda.is_available(), 'production requires CUDA'
print('Validated gate:', receipt_path)
print('Torch:', torch.__version__, 'CUDA device:', torch.cuda.get_device_name())
PY

mkdir -p "$output_root/$campaign"
run_root="$output_root/$campaign/$config-seed55"
mkdir "$run_root" || fail "run output already exists; refusing a duplicate run"
printf 'Job: %s_%s\nSource commit: %s\nConfiguration: %s; seed: 55; decisions: 2000000\nCheckpoint output: %s\n' \
  "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$source_commit" "$config" "$run_root"
exec srun --unbuffered --kill-on-bad-exit=1 "$ambi_python" main.py \
  --run "$manifest" --alg-dir configs/dmcontrol/algs --log-dir "$run_root" \
  --alg-index "$SLURM_ARRAY_TASK_ID" --trial-index 0 --num-runs 1
