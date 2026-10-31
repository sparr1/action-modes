import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("original_repro", ROOT / "slurm/reproduce_original_prior.py")
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


def test_production_recipe_changes_only_publication():
    original = json.loads((ROOT / repro.ALG).read_text())
    original_manifest = json.loads((ROOT / repro.MANIFEST).read_text())
    run, manifest = repro.recipe("test reproduction")
    for key in repro.PUBLICATION_KEYS:
        run["alg_params"][key] = original["alg_params"][key]
    assert run == original
    assert manifest == original_manifest
    assert manifest["checkpoint_every"] == 25000
    assert manifest["save_strat"] == ["all"]
    assert run["total_steps"] // manifest["checkpoint_every"] == 60


def test_cpu_choice_uses_smallest_within_five_percent_and_margin():
    choice = repro.choose_resources([
        {"cores": 1, "seconds_per_decision": .0207},
        {"cores": 2, "seconds_per_decision": .0200},
        {"cores": 4, "seconds_per_decision": .0199},
    ])
    assert choice["cpus"] == 1
    assert choice["walltime_minutes"] * 60 >= .0207 * 1_500_000 * 1.2 + 600


def test_output_refuses_source_and_existing_directory(tmp_path):
    with pytest.raises(ValueError):
        repro.require_output(ROOT / "results")
    with pytest.raises(FileExistsError):
        repro.require_output(tmp_path)


def test_storage_preflight(tmp_path, monkeypatch):
    class Disk:
        free = 7 * 1024**3
    monkeypatch.setattr(repro.shutil, "disk_usage", lambda _: Disk())
    with pytest.raises(RuntimeError, match="Insufficient storage"):
        repro.require_output(tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_original_launcher_requires_unchanged_scientific_files(monkeypatch):
    result = subprocess.check_output(["git", "diff", "--name-only", repro.ORIGINAL, "--", *repro.PROTECTED], cwd=ROOT, text=True)
    # A later scientific variant must never silently use the exact-baseline
    # launcher. Ignore working-tree dirt here to exercise its source guard.
    if result:
        real_git = repro.git
        monkeypatch.setattr(repro, "git", lambda *args: "" if args[0] == "status" else real_git(*args))
        with pytest.raises(RuntimeError, match="Original scientific source changed"):
            repro.source_binding()


def test_launcher_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "slurm/run_original_prior_reproduction_hydra.sbatch")], check=True)
