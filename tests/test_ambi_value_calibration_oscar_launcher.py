import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "slurm/run_ambi_humanoid_walk_value_calibration_outer_policy_ablation_oscar.sbatch"
)
MANIFEST = (
    ROOT
    / "configs/dmcontrol/experiments/ambi_humanoid_walk_value_calibration_outer_policy_ablation.json"
)
CONFIGS = [
    "ambi_humanoid_walk_base_min_all_reward_only_value_calibration",
    "ambi_humanoid_walk_base_min_all_reward_only_value_calibration_outer_policy_50",
]


def _executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _argument_after(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def test_oscar_array_resources_and_online_logging_are_explicit():
    contents = LAUNCHER.read_text(encoding="utf-8")

    for directive in (
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --cpus-per-task=6",
        "#SBATCH --mem=32G",
        "#SBATCH --time=96:00:00",
        "#SBATCH --array=0-5%6",
        "#SBATCH --output=slurm/%x-%A_%a.out",
        "#SBATCH --error=slurm/%x-%A_%a.err",
    ):
        assert directive in contents
    assert "#SBATCH --requeue" not in contents
    assert "#SBATCH --signal" not in contents
    assert "--nodelist" not in contents
    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert "uv run" not in contents
    assert "conda activate" not in contents
    assert 'ambi_python="${AMBI_DMC_PYTHON:-$default_python}"' in contents
    assert "production launch requires a clean checkout" in contents
    assert (
        'approved_output_root="/oscar/home/rgao48/ambi-durable/value-calibration"'
        in contents
    )
    assert 'run_root="$cell_output_root"' in contents
    assert (
        'cell_output_root="$output_root/$campaign/task_${SLURM_ARRAY_TASK_ID}"'
        in contents
    )
    assert "logs/dmcontrol/ambi_value_calibration" not in contents
    assert "export MUJOCO_GL=egl" in contents
    assert "export PYTHONUNBUFFERED=1" in contents
    assert "export WANDB_MODE=online" in contents
    assert 'export WANDB_DIR="$wandb_local_root"' in contents
    assert 'export WANDB_CACHE_DIR="$wandb_local_root/cache"' in contents
    assert 'export WANDB_DATA_DIR="$wandb_local_root/data"' in contents
    assert 'export WANDB_ARTIFACT_DIR="$wandb_local_root/artifacts"' in contents
    assert "export WANDB_DISABLE_CODE=true" in contents
    assert "srun --unbuffered --kill-on-bad-exit=1" in contents


def test_launcher_contract_matches_the_combined_three_seed_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contents = LAUNCHER.read_text(encoding="utf-8")

    assert manifest["configs"] == CONFIGS
    assert manifest["trials"] == 3
    assert manifest["overrides_alg"]["seed"] == 55
    assert 'alg_index=$((SLURM_ARRAY_TASK_ID / 3))' in contents
    assert 'trial_index=$((SLURM_ARRAY_TASK_ID % 3))' in contents
    assert 'seed=$((55 + trial_index))' in contents
    assert '--alg-index "$alg_index"' in contents
    assert '--trial-index "$trial_index"' in contents
    assert "--num-runs 1" in contents
    for config in CONFIGS:
        assert config in contents


def test_each_array_cell_launches_exactly_one_config_trial(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = tmp_path / "python"
    _executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -eu
if [[ "$1" == "rev-parse" && "$2" == "--show-toplevel" ]]; then
  printf '%s\n' "$FAKE_GIT_ROOT"
elif [[ "$1" == "rev-parse" && "$2" == "HEAD" ]]; then
  printf '%040d\n' 0
elif [[ "$1" == "status" ]]; then
  :
else
  exit 4
fi
""",
    )
    _executable(
        fake_bin / "srun",
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$@" > "$FAKE_SRUN_ARGS"
printf '%s\n' "$WANDB_MODE" > "$FAKE_WANDB_MODE"
""",
    )
    _executable(fake_python, "#!/usr/bin/env bash\nexit 0\n")

    base_env = os.environ.copy()
    base_env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{base_env['PATH']}",
            "SLURM_ARRAY_JOB_ID": "24680",
            "SLURM_JOB_ID": "24680_0",
            "SLURM_NODELIST": "fake-gpu",
            "SLURM_SUBMIT_DIR": str(ROOT),
            "AMBI_DMC_PYTHON": str(fake_python),
            "AMBI_VALUE_CALIBRATION_ROOT": str(tmp_path / "durable-output"),
            "AMBI_VALUE_CALIBRATION_CAMPAIGN": "test-campaign",
            "FAKE_GIT_ROOT": str(ROOT),
        }
    )

    for task_id in range(6):
        cell_root = tmp_path / f"cell-{task_id}"
        cell_root.mkdir()
        args_path = cell_root / "srun-args"
        wandb_mode_path = cell_root / "wandb-mode"
        env = base_env.copy()
        env.update(
            {
                "SLURM_ARRAY_TASK_ID": str(task_id),
                "SLURM_JOB_ID": f"24680_{task_id}",
                "SLURM_TMPDIR": str(cell_root / "slurm-tmp"),
                "FAKE_SRUN_ARGS": str(args_path),
                "FAKE_WANDB_MODE": str(wandb_mode_path),
            }
        )

        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        arguments = args_path.read_text(encoding="utf-8").splitlines()
        alg_index, trial_index = divmod(task_id, 3)
        seed = 55 + trial_index
        assert _argument_after(arguments, "--run") == str(
            MANIFEST.relative_to(ROOT)
        )
        assert _argument_after(arguments, "--alg-index") == str(alg_index)
        assert _argument_after(arguments, "--trial-index") == str(trial_index)
        assert _argument_after(arguments, "--num-runs") == "1"
        assert _argument_after(arguments, "--log-dir") == str(
            tmp_path / "durable-output" / "test-campaign" / f"task_{task_id}"
        )
        assert f"Seed: {seed}" in result.stdout
        assert f"Algorithm config: {CONFIGS[alg_index]}" in result.stdout
        assert wandb_mode_path.read_text(encoding="utf-8").strip() == "online"


def test_launcher_rejects_broad_home_output_root():
    contents = LAUNCHER.read_text(encoding="utf-8")

    assert '[[ "$output_root" != "/oscar/home" ]]' in contents
    assert '[[ "$output_root" != "/oscar/home/rgao48" ]]' in contents
