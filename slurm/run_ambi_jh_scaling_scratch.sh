#!/usr/bin/env bash
# Called only after the reward-training gate validates one approved recipe.
# This explicit campaign opt-in does not change the canonical home launcher.
set -Eeuo pipefail
umask 077
batch_started=$SECONDS
fail() { echo "J/H scaling scratch launcher error: $*" >&2; exit 2; }
approved_root="/oscar/scratch/rgao48/ambi/jh-scaling-training/20261028"
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${SLURM_CPUS_PER_TASK:?Specify the measured CPU allocation.}"
: "${AMBI_PROJECT_DIR:?Set the clean training checkout.}"
: "${AMBI_EXPECTED_COMMIT:?Set its full tested commit.}"
: "${AMBI_PYTHON:?Set the locked interpreter.}"
: "${AMBI_JH_LAUNCH_RECEIPT:?Run the reward-training gate dispatch first.}"
: "${AMBI_LINEAGE_DIR:?Set the recipe-specific scratch lineage.}"
: "${AMBI_RUN_CONFIG:?Set the single-cell manifest.}"
: "${AMBI_ALG_DIR:?Set its algorithm directory.}"
: "${AMBI_DRAIN_AFTER_SECONDS:?Set a measured segment budget.}"
[[ "${AMBI_RETURN_STORAGE:-}" == scratch-jh-scaling ]] || fail "explicit scratch-jh-scaling selection is required"
[[ "${AMBI_DURABLE_ROOT:-}" == "$approved_root" ]] || fail "campaign root changed"
[[ "${AMBI_DURABLE_QUOTA_LABEL:-}" == rgao48 && "${AMBI_DURABLE_QUOTA_PATH:-}" == /oscar/scratch ]] || fail "scratch quota must be selected exactly"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$AMBI_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "job ID or expected SHA is invalid"
[[ "$SLURM_CPUS_PER_TASK" == 1 || "$SLURM_CPUS_PER_TASK" == 2 ]] || fail "only measured one- or two-CPU allocations are supported"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "${AMBI_RESUME_GENERATION:-}" ]] || fail "arrays and rollback selection are not supported by this campaign"
[[ "$AMBI_DRAIN_AFTER_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "drain budget must be positive"
checkpoint_minutes="${AMBI_RESUME_CHECKPOINT_MINUTES:-30}"
[[ "$checkpoint_minutes" =~ ^[1-9][0-9]*$ ]] || fail "checkpoint cadence must be positive"
restart_count="${SLURM_RESTART_COUNT:-0}"
[[ "$restart_count" =~ ^[0-9]+$ ]] || fail "restart count must be non-negative"
case "$AMBI_LINEAGE_DIR" in
  "$approved_root/lineages/h3-j12-k3-seed55") case_name=h3_j12_k3 ;;
  "$approved_root/lineages/h4-j10-k4-seed55") case_name=h4_j10_k4 ;;
  "$approved_root/lineages/h5-j10-k5-seed55") case_name=h5_j10_k5 ;;
  *) fail "lineage is not one of the three approved fresh runs" ;;
esac
[[ -d "$approved_root" && "$(cd "$approved_root" && pwd -P)" == "$approved_root" ]] || fail "scratch root must exist without symlink ancestors"
[[ ! -L "$approved_root/lineages" && ! -L "$AMBI_LINEAGE_DIR" ]] || fail "lineage paths must not be symlinks"
if (( restart_count == 0 )); then
  [[ "${AMBI_RESUME_MODE:-new}" == new && ! -e "$AMBI_LINEAGE_DIR" ]] || fail "first launch requires a new lineage"
  resume_mode=new
else
  [[ -f "$AMBI_LINEAGE_DIR/LINEAGE.json" && -f "$AMBI_LINEAGE_DIR/LATEST" ]] || fail "restart requires an existing lineage"
  resume_mode=required
fi
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && -x "$AMBI_PYTHON" ]] || fail "source and Python must be absolute"
cd "$AMBI_PROJECT_DIR"
project_dir="$(pwd -P)"
[[ "$(git rev-parse --show-toplevel)" == "$project_dir" ]] || fail "training directory must be the Git root"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]] || fail "source commit changed"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail "training checkout must be clean"
[[ "$approved_root" != "$project_dir" && "$approved_root" != "$project_dir/"* ]] || fail "scratch output must be outside source"
[[ "$AMBI_RUN_CONFIG" == "$project_dir/configs/dmcontrol/experiments/ambi_return_only_sac_${case_name}_training.json" ]] || fail "manifest does not match the lineage"
[[ "$AMBI_ALG_DIR" == "$project_dir/configs/dmcontrol/algs" ]] || fail "algorithm directory changed"
[[ -f "$AMBI_RUN_CONFIG" && -d "$AMBI_ALG_DIR" ]] || fail "recipe is missing"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
# Dispatch proof links this segment to the already-validated immutable gate.
"$AMBI_PYTHON" - "$AMBI_JH_LAUNCH_RECEIPT" "$case_name" "$resume_mode" <<'PY'
import hashlib, json, os, pathlib, sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert record['binding']['commit'] == os.environ['AMBI_EXPECTED_COMMIT']
assert record['config'] == 'ambi_full_' + sys.argv[2] + '_reward_only'
assert record['lineage_dir'] == os.environ['AMBI_LINEAGE_DIR']
assert record['segment'] == os.environ['SLURM_JOB_ID'] + '.' + os.environ.get('SLURM_RESTART_COUNT', '0')
assert record['resume_mode'] == sys.argv[3]
assert record['storage'] == {'mode': 'scratch-jh-scaling', 'root': os.environ['AMBI_DURABLE_ROOT'], 'quota_path': '/oscar/scratch'}
gate = pathlib.Path(record['gate_receipt'])
assert hashlib.sha256(gate.read_bytes()).hexdigest() == record['gate_sha256']
for path, checksum in json.loads(gate.read_text())['artifacts'].items():
    assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == checksum
PY
quota_output="$(checkquota)" || fail "checkquota failed"
printf '%s\n' "$quota_output" | "$AMBI_PYTHON" -m utils.oscar_resume_launcher quota \
  --allocation rgao48 --filesystem-path /oscar/scratch
"$AMBI_PYTHON" -m utils.oscar_resume_launcher storage \
  --durable-root "$approved_root" --lineage-dir "$AMBI_LINEAGE_DIR"
if [[ -e "$AMBI_LINEAGE_DIR/DONE" ]]; then
  "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
  echo "Verified DONE; training target is already complete."
  exit 0
fi
if (( restart_count > 0 )); then
  "$AMBI_PYTHON" -m utils.oscar_resume_launcher handoff \
    --lineage-dir "$AMBI_LINEAGE_DIR" --slurm-job-id "$SLURM_JOB_ID" \
    --segment-id "${SLURM_JOB_ID}.$((restart_count - 1))"
fi
export AMBI_SEGMENT_ID="${SLURM_JOB_ID}.${restart_count}"
segment_console_dir="$approved_root/segment-console/$AMBI_SEGMENT_ID"
mkdir -p "$(dirname "$segment_console_dir")"
mkdir "$segment_console_dir" || fail "segment output already exists"
export AMBI_SEGMENT_CONSOLE_DIR="$segment_console_dir"
# The generic reward wrapper already configured job-local caches and W&B paths.
export WANDB_MODE=online WANDB__REQUIRE_CORE=true
unset WANDB_RUN_ID WANDB_RESUME
remaining=$((AMBI_DRAIN_AFTER_SECONDS - (SECONDS - batch_started)))
(( remaining > 0 )) || fail "drain budget expired during startup"
set +e
srun --unbuffered --kill-on-bad-exit=1 \
  --output="$segment_console_dir/stdout.log" --error="$segment_console_dir/stderr.log" \
  "$AMBI_PYTHON" main.py --run "$AMBI_RUN_CONFIG" --alg-dir "$AMBI_ALG_DIR" \
  --num-runs 1 --lineage-dir "$AMBI_LINEAGE_DIR" \
  --resume-mode "$resume_mode" --resume-wandb-mode online \
  --resume-checkpoint-minutes "$checkpoint_minutes" --drain-after-seconds "$remaining"
status=$?
set -e
case "$status" in
  0)
    "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
    echo "Training target complete; DONE verified against LATEST."
    ;;
  75)
    "$AMBI_PYTHON" -m utils.oscar_resume_launcher handoff \
      --lineage-dir "$AMBI_LINEAGE_DIR" --slurm-job-id "$SLURM_JOB_ID" --segment-id "$AMBI_SEGMENT_ID"
    scontrol requeue "$SLURM_JOB_ID"
    ;;
  *) echo "Training failed with status $status; automatic requeue suppressed." >&2; exit "$status" ;;
esac
