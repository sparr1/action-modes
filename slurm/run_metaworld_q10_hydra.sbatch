#!/usr/bin/env bash
#SBATCH --job-name=mw-q10-hydra-s55
#SBATCH --array=2-3
#SBATCH --partition=gpus
#SBATCH --account=default
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=USR1@900
# Supply measured-comparison walltime, memory, and external log paths at submission.
set -Eeuo pipefail
umask 077
: "${AMBI_PROJECT_DIR:?Set the clean pinned checkout}"
: "${AMBI_EXPECTED_COMMIT:?Pin the source commit}"
: "${AMBI_METAWORLD_ROOT:?Set the Hydra shared campaign directory}"
: "${AMBI_METAWORLD_PYTHON:?Set the verified MetaWorld runtime}"
[[ "$AMBI_METAWORLD_ROOT" == /data/public/rgao48/ambi/* ]]
case "${SLURM_ARRAY_TASK_ID:?Submit as an array}" in
    2) task=push ;;
    3) task=stick-pull ;;
    *) echo "Select Push or Stick Pull" >&2; exit 1 ;;
esac
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]

export AMBI_METAWORLD_VARIANT=mrsq_q10 AMBI_METAWORLD_SEED=55 AMBI_METAWORLD_RESUME_MINUTES=15
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
unset WANDB_RUN_ID WANDB_RESUME CUDA_LAUNCH_BLOCKING TDMPC2_EVAL_CSV AMBI_METAWORLD_NATIVE_ENV

gates="$AMBI_METAWORLD_ROOT/$task/gates"
mkdir -p "$gates"
gate="$gates/results.json"
cache="${SLURM_TMPDIR:-/tmp}/mw-q10-gate-$UID-${SLURM_JOB_ID:?}-${SLURM_RESTART_COUNT:-0}"
mkdir -p "$cache"/{inductor,triton,tmp,xdg,wandb,data,config}
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton" TMPDIR="$cache/tmp"
export XDG_CACHE_HOME="$cache/xdg" WANDB_DIR="$cache/wandb" WANDB_CACHE_DIR="$cache/xdg"
export WANDB_DATA_DIR="$cache/data" WANDB_CONFIG_DIR="$cache/config"
export WANDB_MODE=disabled
if [[ ! -s "$gate" ]]; then
    out="$gates/attempt-$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_METAWORLD_PYTHON" -B \
        slurm/smoke_metaworld_mrsq_q5.py --variant mrsq_q10 --task "$task" --gpu-name "NVIDIA L40" \
        --output "$out"
    ln -- "$out/results.json" "$gate"
fi
"$AMBI_METAWORLD_PYTHON" -B - "$gate" "$task" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
assert report["passed"] and report["source"]["commit"] == os.environ["AMBI_EXPECTED_COMMIT"]
assert report["task"] == sys.argv[2] and report["runtime"]["gpu"] == "NVIDIA L40"
assert report["num_q"] == report["q_pair_size"] == 10 and report["wandb"] is False
assert report["critic_geometry"]["passed"] and report["graph_native_graph_continuation"]
assert report["inner_cuda_graph_used"] and not report["compile_fallback"]
assert report["replay_storage_device"] == "cpu" and report["measured"]["decisions"] == 100
rate = report["measured"]["seconds_per_decision"]
eval_seconds = max(row["seconds"] for row in report["evaluations"][1:])
assert math.isfinite(rate) and rate > 0 and math.isfinite(eval_seconds) and eval_seconds > 0
hours = (1_000_000 * rate + 410 * eval_seconds) / 3600
print(f"HYDRA_MW_Q10_GATE_PASSED {sys.argv[2]} {rate:.4f} s/decision, projected {hours:.1f}h plus I/O",
      flush=True)
PY

# The shared launcher uses the array index to select the same task and lineage.
exec bash slurm/run_metaworld_oscar.sbatch
