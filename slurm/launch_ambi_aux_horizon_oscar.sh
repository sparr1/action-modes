#!/usr/bin/env bash
# Run from a clean Git-synchronized Oscar checkout. Submission is explicit.
set -Eeuo pipefail
umask 077
cd "${AMBI_PROJECT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_ACTION_MODES_SHA:?}" && -z "$(git status --porcelain)" ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
"${AMBI_DMC_PYTHON:?}" slurm/ambi_aux_hj_sweep.py prepare --root "${AMBI_CAMPAIGN_ROOT:?}" \
  --checkpoint "${AMBI_CHECKPOINT:?}" --inventory "${CHECKPOINT_MANIFEST:?}" \
  --reference "${AMBI_PRIOR_REFERENCE:?}" --registry /oscar/scratch/rgao48/ambi/evaluation-series/runs \
  --matrix configs/research/ambi_aux_horizon_conditioning_625k.json \
  --baseline-campaign /oscar/scratch/rgao48/ambi/aux-soft-critic-budget-625k/20260919-fe87ae0 \
  --group aux625k-softsoft-h23-j1248-conditioned-20260919 \
  --label '625k soft/soft horizon conditioning: H2/H3 J1/J2/J4/J8'
mkdir -p "$AMBI_CAMPAIGN_ROOT/slurm"
declare -p EXPECTED_ACTION_MODES_SHA AMBI_PROJECT_DIR AMBI_CAMPAIGN_ROOT AMBI_CHECKPOINT \
  AMBI_PRIOR_REFERENCE CHECKPOINT_MANIFEST AMBI_DMC_PYTHON > "$AMBI_CAMPAIGN_ROOT/campaign.env"
git rev-parse HEAD HEAD^{tree} > "$AMBI_CAMPAIGN_ROOT/source-sha.txt"
git status --porcelain > "$AMBI_CAMPAIGN_ROOT/source-status.txt"
mkdir "$AMBI_CAMPAIGN_ROOT/submission-lock"
smoke=$(sbatch --parsable --exclude=gpu2708,gpu3003,gpu3105 --time=00:15:00 --job-name=aux625k-hcond-smoke \
  --output="$AMBI_CAMPAIGN_ROOT/slurm/smoke-%j.out" --error="$AMBI_CAMPAIGN_ROOT/slurm/smoke-%j.err" \
  slurm/run_ambi_aux_hj_sweep_oscar.sbatch --smoke)
printf '%s\n' "$smoke" > "$AMBI_CAMPAIGN_ROOT/smoke-job.txt"
production=$(sbatch --parsable --exclude=gpu2708,gpu3003,gpu3105 --array=0-9%10 --time=03:00:00 --job-name=aux625k-hcond \
  --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
  --output="$AMBI_CAMPAIGN_ROOT/slurm/production-%A_%a.out" --error="$AMBI_CAMPAIGN_ROOT/slurm/production-%A_%a.err" \
  slurm/run_ambi_aux_hj_sweep_oscar.sbatch)
printf '%s\n' "$production" > "$AMBI_CAMPAIGN_ROOT/production-job.txt"
watcher=$(sbatch --parsable --cpus-per-task=8 --mem=64G --time=06:00:00 --job-name=aux625k-hcond-publish \
  --output="$AMBI_CAMPAIGN_ROOT/slurm/watcher-%j.out" --error="$AMBI_CAMPAIGN_ROOT/slurm/watcher-%j.err" \
  slurm/run_ambi_aux_hj_watch_oscar.sbatch)
printf '%s\n' "$watcher" > "$AMBI_CAMPAIGN_ROOT/watcher-job.txt"
"$AMBI_DMC_PYTHON" - "$AMBI_CAMPAIGN_ROOT" "$smoke" "$production" "$watcher" <<'PY'
import datetime,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);c=json.loads((root/'campaign.json').read_text())
v=dict(submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=c['source_commit'],
       smoke_job=sys.argv[2],production_job=sys.argv[3],watcher_job=sys.argv[4],gpu_job_ids=sys.argv[2:4],
       conditions=10,conditioned_settings=8,new_unconditioned_controls=2,reused_controls=6,new_episodes=50,
       concurrency_limit=10,cpus_per_gpu=4,memory_gib_per_gpu=32,publisher_workers=4,
       gpu_pool=['l40s','a6000','a40','a5500'],H=[2,3],J=[1,2,4,8],C=16,A=4,N=128,B=256,
       environment_seeds=list(range(101,106)),controller_seed=55,full_replay=True,
       overview_run_id=c['overview_run_id'],checkpoint_sha256=c['checkpoint_sha256'])
(root/'submission.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
PY
squeue -u rgao48 -o '%.24i %.25j %.10T %.7C %.16b %.35R'
