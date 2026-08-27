import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_ambi_anchor_kl_oscar.sbatch"
CONFIGS = [
    "ambi_anchor_kl_smooth",
    "ambi_anchor_kl_quantile",
    "ambi_anchor_kl_dual",
]
SCHEDULES = ["smooth", "quantile_gate", "dual"]


def _executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _argument_after(arguments: list[str], option: str) -> str:
    return arguments[arguments.index(option) + 1]


def test_oscar_behavior_kl_array_maps_all_nine_guarded_cells(tmp_path):
    contents = LAUNCHER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --cpus-per-task=6",
        "#SBATCH --mem=32G",
        "#SBATCH --time=96:00:00",
        "#SBATCH --array=0-8%6",
        "#SBATCH --output=slurm/%x-%A_%a.out",
        "#SBATCH --error=slurm/%x-%A_%a.err",
    ):
        assert directive in contents
    assert "#SBATCH --requeue" not in contents
    assert "#SBATCH --signal" not in contents
    assert "--nodelist" not in contents
    assert "production launch requires a clean checkout" in contents
    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert (
        'approved_output_root="/oscar/home/rgao48/ambi-durable/value-calibration"'
        in contents
    )
    assert '[[ "$output_root" != "/oscar/home" ]]' in contents
    assert '[[ "$output_root" != "/oscar/home/rgao48" ]]' in contents
    assert 'run_root="$cell_output_root"' in contents
    assert "export MUJOCO_GL=egl" in contents
    assert "export WANDB_MODE=online" in contents
    assert 'export WANDB_DIR="$wandb_local_root"' in contents
    assert "srun --unbuffered --kill-on-bad-exit=1" in contents

    for config, schedule in zip(CONFIGS, SCHEDULES):
        manifest = (
            ROOT / f"configs/dmcontrol/experiments/{config}.json"
        )
        algorithm = ROOT / f"configs/dmcontrol/algs/{config}.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["configs"] == [config]
        assert payload["trials"] == 3
        assert payload["overrides_alg"]["seed"] == 55
        assert (
            json.loads(algorithm.read_text(encoding="utf-8"))["alg_params"][
                "outer_behavior_policy_kl_schedule"
            ]
            == schedule
        )
        assert config in contents

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

    for task_id in range(9):
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
        trial_index, variant_index = divmod(task_id, 3)
        config = CONFIGS[variant_index]
        assert _argument_after(arguments, "--run") == (
            f"configs/dmcontrol/experiments/{config}.json"
        )
        assert _argument_after(arguments, "--alg-index") == "0"
        assert _argument_after(arguments, "--trial-index") == str(trial_index)
        assert _argument_after(arguments, "--num-runs") == "1"
        assert _argument_after(arguments, "--log-dir") == str(
            tmp_path / "durable-output" / "test-campaign" / f"task_{task_id}"
        )
        assert f"Seed: {55 + trial_index}" in result.stdout
        assert f"Algorithm config: {config}" in result.stdout
        assert f"Behavior-KL schedule: {SCHEDULES[variant_index]}" in result.stdout
        assert wandb_mode_path.read_text(encoding="utf-8").strip() == "online"
