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


def test_oscar_behavior_kl_array_maps_all_nine_guarded_cells(tmp_path):
    contents = LAUNCHER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --cpus-per-task=6",
        "#SBATCH --mem=32G",
        "#SBATCH --time=96:00:00",
        "#SBATCH --array=0-8%6",
        "#SBATCH --requeue",
        "#SBATCH --signal=USR1@3600",
        "#SBATCH --output=slurm/%x-%A_%a.out",
        "#SBATCH --error=slurm/%x-%A_%a.err",
    ):
        assert directive in contents
    assert "--nodelist" not in contents
    assert "production launch requires a clean checkout" in contents
    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert (
        'approved_durable_root="/oscar/home/rgao48/ambi-durable"'
        in contents
    )
    assert 'export AMBI_DURABLE_QUOTA_LABEL="rgao48"' in contents
    assert 'export AMBI_DURABLE_QUOTA_PATH="/oscar/home"' in contents
    assert 'exec bash "$project_dir/run_ambi_oscar.sh"' in contents
    assert "export MUJOCO_GL=egl" in contents
    assert "export WANDB_MODE=online" in contents
    assert 'export WANDB_DIR="$wandb_local_root"' in contents
    assert "Agent decisions: 14000000" in contents
    assert "srun --unbuffered --kill-on-bad-exit=1" not in contents

    for config, schedule in zip(CONFIGS, SCHEDULES):
        manifest = (
            ROOT / f"configs/dmcontrol/experiments/{config}.json"
        )
        algorithm = ROOT / f"configs/dmcontrol/algs/{config}.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["configs"] == [config]
        assert payload["trials"] == 3
        assert payload["overrides_alg"]["seed"] == 55
        assert payload["overrides_alg"]["total_steps"] == 14_000_000
        algorithm_payload = json.loads(algorithm.read_text(encoding="utf-8"))
        assert algorithm_payload["total_steps"] == 14_000_000
        assert "14m-decisions" in algorithm_payload["alg_params"]["wandb_tags"]
        assert "1m-decisions" not in algorithm_payload["alg_params"]["wandb_tags"]
        assert algorithm_payload["alg_params"][
            "outer_behavior_policy_kl_schedule"
        ] == schedule
        assert algorithm_payload["alg_params"]["eval_inner_comparison"] is True
        assert algorithm_payload["alg_params"]["eval_inner_comparison_episodes"] == 5
        assert algorithm_payload["alg_params"]["eval_inner_comparison_seed"] == 12345
        assert algorithm_payload["alg_params"]["inner_diagnostic_rollouts"] == 0
        assert config in contents
        for seed in (55, 56, 57):
            cell_manifest = ROOT / (
                "configs/dmcontrol/experiments/"
                f"{config}_14m_seed{seed}.json"
            )
            cell = json.loads(cell_manifest.read_text(encoding="utf-8"))
            assert cell["configs"] == [config]
            assert cell["trials"] == 1
            assert cell["overrides_alg"]["seed"] == seed
            assert cell["overrides_alg"]["total_steps"] == 14_000_000
            assert cell["logs"] == "none"
            assert cell["checkpoint_every"] == 100_000
            assert cell["save_strat"] == ["best", "latest"]

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = tmp_path / "python"
    _executable(
        fake_bin / "git",
        """#!/bin/bash
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
        fake_bin / "bash",
        """#!/bin/bash
set -eu
printf '%s\n' "$@" > "$FAKE_BASH_ARGS"
{
  printf 'AMBI_RUN_CONFIG=%s\n' "$AMBI_RUN_CONFIG"
  printf 'AMBI_ALG_DIR=%s\n' "$AMBI_ALG_DIR"
  printf 'AMBI_PYTHON=%s\n' "$AMBI_PYTHON"
  printf 'AMBI_DURABLE_ROOT=%s\n' "$AMBI_DURABLE_ROOT"
  printf 'AMBI_LINEAGE_DIR=%s\n' "$AMBI_LINEAGE_DIR"
  printf 'AMBI_DURABLE_QUOTA_LABEL=%s\n' "$AMBI_DURABLE_QUOTA_LABEL"
  printf 'AMBI_DURABLE_QUOTA_PATH=%s\n' "$AMBI_DURABLE_QUOTA_PATH"
  printf 'WANDB_MODE=%s\n' "$WANDB_MODE"
} > "$FAKE_EXPORTED_ENV"
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
            "AMBI_VALUE_CALIBRATION_CAMPAIGN": "test-campaign",
            "FAKE_GIT_ROOT": str(ROOT),
        }
    )

    for task_id in range(9):
        cell_root = tmp_path / f"cell-{task_id}"
        cell_root.mkdir()
        args_path = cell_root / "bash-args"
        exported_env_path = cell_root / "exported-env"
        env = base_env.copy()
        env.update(
            {
                "SLURM_ARRAY_TASK_ID": str(task_id),
                "SLURM_JOB_ID": f"24680_{task_id}",
                "SLURM_TMPDIR": str(cell_root / "slurm-tmp"),
                "FAKE_BASH_ARGS": str(args_path),
                "FAKE_EXPORTED_ENV": str(exported_env_path),
            }
        )

        result = subprocess.run(
            ["/bin/bash", str(LAUNCHER)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        arguments = args_path.read_text(encoding="utf-8").splitlines()
        exported = dict(
            line.split("=", 1)
            for line in exported_env_path.read_text(encoding="utf-8").splitlines()
        )
        trial_index, variant_index = divmod(task_id, 3)
        config = CONFIGS[variant_index]
        seed = 55 + trial_index
        manifest = f"configs/dmcontrol/experiments/{config}_14m_seed{seed}.json"
        assert arguments == [str(ROOT / "run_ambi_oscar.sh")]
        assert exported["AMBI_RUN_CONFIG"] == manifest
        assert exported["AMBI_ALG_DIR"] == "configs/dmcontrol/algs"
        assert exported["AMBI_PYTHON"] == str(fake_python)
        assert exported["AMBI_DURABLE_ROOT"] == (
            "/oscar/home/rgao48/ambi-durable"
        )
        assert exported["AMBI_LINEAGE_DIR"] == (
            "/oscar/home/rgao48/ambi-durable/value-calibration/"
            f"test-campaign/task_{task_id}"
        )
        assert exported["AMBI_DURABLE_QUOTA_LABEL"] == "rgao48"
        assert exported["AMBI_DURABLE_QUOTA_PATH"] == "/oscar/home"
        assert exported["WANDB_MODE"] == "online"
        assert f"Seed: {seed}" in result.stdout
        assert f"Algorithm config: {config}" in result.stdout
        assert f"Behavior-KL schedule: {SCHEDULES[variant_index]}" in result.stdout
        assert "Agent decisions: 14000000" in result.stdout
