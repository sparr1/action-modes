import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "slurm/run_ambi_humanoid_walk_outer_prior_no_inner_checkpoint_bank_1p5m_hydra.sbatch"
)


def test_bank_launcher_is_executable_and_has_valid_bash_syntax():
    assert os.access(LAUNCHER, os.X_OK)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
