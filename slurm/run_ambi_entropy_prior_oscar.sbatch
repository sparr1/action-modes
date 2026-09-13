#!/usr/bin/env bash
#SBATCH --job-name=ambi-entropy-prior
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# Submit one task per campaign.tasks entry; choose concurrency from live limits.
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${AMBI_ENTROPY_CAMPAIGN:?Supply the absolute campaign.json path}"
: "${SLURM_ARRAY_TASK_ID:?Supply the campaign task index}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || { echo "Unexpected commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty checkout" >&2; exit 1; }
[[ "$AMBI_ENTROPY_CAMPAIGN" == /* && -f "$AMBI_ENTROPY_CAMPAIGN" ]] || { echo "Campaign must be an existing absolute file" >&2; exit 1; }
[[ "${AMBI_ENTROPY_SMOKE:-0}" == 0 || "${AMBI_ENTROPY_SMOKE:-0}" == 1 ]] || { echo "AMBI_ENTROPY_SMOKE must be 0 or 1" >&2; exit 1; }
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked Python" >&2; exit 1; }
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ambi-entropy-${SLURM_JOB_ID:?Supply job ID}"
export WANDB_MODE=disabled
exec "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def absolute_file(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_file():
        raise ValueError(f"{label} must be an existing absolute file")
    return Path(value)


campaign = read_json(Path(os.environ["AMBI_ENTROPY_CAMPAIGN"]))
study_path = absolute_file(campaign["study"], "Study")
study = read_json(study_path)
checkpoints = study["checkpoints"]
ids = [row["id"] for row in checkpoints]
if not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
    raise ValueError("Study checkpoint IDs must be distinct nonempty strings")
tasks = campaign["tasks"]
if not isinstance(tasks, list) or not tasks:
    raise ValueError("Campaign tasks must be a nonempty list")
output_paths = set()
cells = set()
for task in tasks:
    index, seed = task["checkpoint_index"], task["episode_seed"]
    if type(index) is not int or not 0 <= index < len(checkpoints):
        raise ValueError("Invalid checkpoint_index in campaign task")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid episode_seed in campaign task")
    output = task["output_dir"]
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise ValueError("Task output_dir must be absolute")
    canonical_output = Path(output).resolve()
    cell = (index, seed)
    if canonical_output in output_paths or cell in cells:
        raise ValueError("Campaign tasks must have unique outputs and checkpoint/seed cells")
    output_paths.add(canonical_output)
    cells.add(cell)
task_text = os.environ["SLURM_ARRAY_TASK_ID"]
if not task_text.isdecimal() or not 0 <= int(task_text) < len(tasks):
    raise ValueError("Task index is outside the campaign")
task = tasks[int(task_text)]
checkpoint_id = ids[task["checkpoint_index"]]
checkpoint_path = absolute_file(campaign["checkpoint_paths"][checkpoint_id], "Checkpoint")
absolute_file(str(checkpoint_path) + ".metadata.json", "Checkpoint sidecar")
output = Path(task["output_dir"])
if output.exists() or output.with_name(output.name + ".work").exists():
    raise FileExistsError(f"Choose a fresh task output directory: {output}")

# Run only inside the Slurm allocation, after inexpensive campaign preflight.
import torch
threads = int(os.environ["OMP_NUM_THREADS"])
if threads < 1:
    raise ValueError("Allocated CPU count must be positive")
torch.set_num_threads(threads)
torch.set_num_interop_threads(1)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in the entropy worker allocation")
print(json.dumps({"task_index": int(task_text), "checkpoint_id": checkpoint_id,
                  "episode_seed": task["episode_seed"], "torch_threads": threads,
                  "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0)}), flush=True)
smoke = os.environ.get("AMBI_ENTROPY_SMOKE", "0") == "1"
if smoke:
    smoke_env = dict(os.environ, AMBI_RUN_REAL_DMCONTROL_TESTS="1")
    subprocess.run([sys.executable, "-m", "pytest", "-q",
                    "tests/test_ambi_entropy_probe.py",
                    "tests/test_ambi_entropy_reporting.py",
                    "tests/test_ambi_entropy_evaluator.py",
                    "tests/test_ambi_real_calibration.py"], env=smoke_env, check=True)
sys.argv = ["evaluate_ambi_entropy.py", "run", "--study", str(study_path),
            "--checkpoint", str(checkpoint_path),
            "--checkpoint-index", str(task["checkpoint_index"]),
            "--episode-seed", str(task["episode_seed"]), "--output-dir", str(output)]
if smoke:
    sys.argv.append("--smoke")
runpy.run_path("evaluate_ambi_entropy.py", run_name="__main__")
PY
