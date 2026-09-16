"""Keep campaign profiles isolated and receipts bound to the selected recipes."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.test_ambi_aux_return_sac_campaign import SHA, _receipt
from tests.test_ambi_prior_sac_study_configs import _HumanoidSpaces


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "slurm/ambi_aux_return_sac_campaign.py"
ORIGINAL_CASES = ("ambi_aux_return_original_prior_shared", "ambi_aux_return_original_prior_detached")


def _profile(monkeypatch, name):
    if name is None:
        monkeypatch.delenv("AMBI_AUX_RECIPE", raising=False)
    else:
        monkeypatch.setenv("AMBI_AUX_RECIPE", name)
    spec = importlib.util.spec_from_file_location("isolated_campaign_profile", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_profile_preserves_existing_inventory_binding_and_bank(monkeypatch):
    default = _profile(monkeypatch, None)
    explicit = _profile(monkeypatch, "standard")
    assert default.RECIPE == "standard"
    assert default.MANIFEST == Path("configs/dmcontrol/experiments/ambi_aux_return_sac_study.json")
    assert len(default.CASES) == 4 and default.CASES == explicit.CASES
    assert default.binding(SHA) == explicit.binding(SHA)
    assert set(default.checkpoint_counts().values()) == {80}
    assert sum(default.checkpoint_counts().values()) == 320
    assert set(default.binding(SHA)) == {
        "source_commit", "manifest_sha256", "config_sha256", "environment_lock_sha256",
    }


@pytest.mark.parametrize("name", ("", "original", "Original_prior", "../original_prior"))
def test_unknown_profile_fails_before_receipt_or_training(monkeypatch, name):
    with pytest.raises(ValueError, match="Unknown AMBI_AUX_RECIPE"):
        _profile(monkeypatch, name)


def test_original_profile_has_two_cells_sixty_checkpoints_and_complete_receipt(monkeypatch):
    campaign = _profile(monkeypatch, "original_prior")
    assert campaign.CASES == ORIGINAL_CASES
    assert campaign.MANIFEST == Path("configs/dmcontrol/experiments/ambi_aux_return_original_prior_study.json")
    assert campaign.checkpoint_counts() == dict.fromkeys(ORIGINAL_CASES, 60)
    receipt = _receipt(campaign)
    assert campaign.validate_receipt(receipt, SHA) is receipt


@pytest.mark.parametrize("source,target", (("standard", "original_prior"), ("original_prior", "standard")))
def test_receipt_cannot_cross_campaign_profiles(monkeypatch, source, target):
    receipt = _receipt(_profile(monkeypatch, source))
    campaign = _profile(monkeypatch, target)
    with pytest.raises(AssertionError, match="receipt does not bind"):
        campaign.validate_receipt(receipt, SHA)


@pytest.mark.parametrize("failure", ("missing", "duplicate", "reordered", "config_hash", "detached_gradient", "fallback"))
def test_original_receipt_rejects_missing_or_incompatible_evidence(monkeypatch, failure):
    campaign = _profile(monkeypatch, "original_prior")
    receipt = _receipt(campaign)
    if failure == "missing":
        receipt["cases"].pop()
    elif failure == "duplicate":
        receipt["cases"][1] = deepcopy(receipt["cases"][0])
    elif failure == "reordered":
        receipt["cases"].reverse()
    elif failure == "config_hash":
        receipt["cases"][1]["config_sha256"] = "f" * 64
    elif failure == "detached_gradient":
        receipt["cases"][1]["auxiliary_gradient_l1"]["encoder"] = 1.
    elif failure == "fallback":
        receipt["cases"][1]["compile_status"][0]["aux_online"] = True
    with pytest.raises(AssertionError):
        campaign.validate_receipt(receipt, SHA)


def test_receipt_estimate_uses_selected_periodic_bank(monkeypatch, tmp_path, capsys):
    campaign = _profile(monkeypatch, "original_prior")
    receipt = _receipt(campaign)
    receipt["cases"][1]["checkpoint_size_bytes"] = 84
    for case in receipt["cases"]:
        campaign.write_new(tmp_path / (case["config"] + ".json"), case)
    campaign.write_new(tmp_path / "fresh-training.json", receipt["fresh_training"])
    monkeypatch.setattr(campaign, "runtime", lambda: {"gpu": "fixture"})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    campaign.collect_receipt(tmp_path, SHA)
    actual = campaign.read_json(tmp_path / "receipt.json")
    assert actual["estimated_checkpoint_bank_bytes"] == 60 * (42 + 84)
    assert "120-checkpoint bank" in capsys.readouterr().out
    campaign.validate_receipt(actual, SHA)


@pytest.mark.parametrize("name,count", (("standard", 4), ("original_prior", 2)))
def test_cuda_gate_collects_the_selected_profile(name, count):
    command = (
        "import json; from tests import test_ambi_aux_return_sac_cuda_gate as gate; "
        "print(json.dumps({'manifest': str(gate.MANIFEST), 'cases': gate.CASES}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], cwd=ROOT,
        env={**os.environ, "AMBI_AUX_RECIPE": name}, capture_output=True, text=True, check=True,
    )
    actual = json.loads(result.stdout)
    assert len(actual["cases"]) == count
    if name == "original_prior":
        assert tuple(actual["cases"]) == ORIGINAL_CASES
        assert actual["manifest"].endswith("ambi_aux_return_original_prior_study.json")
    else:
        assert actual["manifest"].endswith("ambi_aux_return_sac_study.json")


def test_original_launch_metadata_resolves_selected_recipe_and_command(monkeypatch, tmp_path):
    campaign = _profile(monkeypatch, "original_prior")
    receipt = _receipt(campaign)
    current_runtime = {"python_executable": sys.executable, "python": sys.version,
                       "torch": "test", "cuda": "test", "gpu": "test", "compute_capability": [0, 0]}
    receipt["runtime"] = current_runtime
    receipt_path = tmp_path / "receipt.json"
    campaign.write_new(receipt_path, receipt)
    monkeypatch.setattr(campaign, "runtime", lambda: current_runtime)
    monkeypatch.setenv("WANDB_API_KEY", "fixture-placeholder-not-a-real-key")
    monkeypatch.setenv("WANDB_RUN_ID", "aux123x1")
    monkeypatch.setenv("AMBI_AUX_CAMPAIGN", "original-prior-fixture")
    import gymnasium as gym
    monkeypatch.setattr(gym, "make", lambda *args, **kwargs: _HumanoidSpaces())
    campaign.prepare_training(receipt_path, SHA, 1, tmp_path)
    actual = campaign.read_json(tmp_path / "launch.json")
    cfg = actual["resolved_config"]
    assert actual["config"] == ORIGINAL_CASES[1]
    assert actual["binding"] == campaign.binding(SHA)
    assert str(campaign.MANIFEST) in actual["command"]
    assert cfg["steps"] == 1_500_000 and cfg["seed"] == 55
    assert cfg["outer_q_actor_reduction"] == "min_pair"
    assert cfg["target_entropy"] == -21 and cfg["log_std_min"] == -20
    assert cfg["aux_return_detach_representation"] is True
    assert cfg["aux_return_mode"] == "sac" and cfg["inner_operator"] == "none"
    assert "fixture-placeholder" not in (tmp_path / "launch.json").read_text()


@pytest.mark.parametrize("recipe,index,error", (
    ("unknown", "0", "recipe must be standard or original_prior"),
    ("original_prior", "2", "array index exceeds recipe cell count"),
))
def test_launcher_rejects_wrong_recipe_or_array_index_before_output(tmp_path, recipe, index, error):
    launcher = ROOT / "slurm/run_ambi_aux_return_sac_oscar.sbatch"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    output = tmp_path / "must-not-create"
    env = {**os.environ, "SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "0",
           "SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": index,
           "AMBI_PROJECT_DIR": str(ROOT), "AMBI_PYTHON": sys.executable,
           "EXPECTED_ACTION_MODES_SHA": SHA, "AMBI_AUX_OUTPUT_ROOT": str(output),
           "AMBI_AUX_CAMPAIGN": "test", "AMBI_AUX_RECIPE": recipe, "AMBI_AUX_MODE": "train",
           "AMBI_AUX_GATE_RECEIPT": str(tmp_path / "missing-receipt.json")}
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and error in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("recipe", ("standard", "original_prior"))
def test_launcher_recipe_selection_matches_receipt_and_gate(monkeypatch, tmp_path, recipe):
    campaign = _profile(monkeypatch, recipe)
    launcher = ROOT / "slurm/run_ambi_aux_return_sac_oscar.sbatch"
    # Execute the launcher's real profile-selection prefix, stopping before any
    # directory creation, cluster command, credentials, or training.
    prefix = launcher.read_text().split('export AMBI_AUX_RECIPE="$recipe"', 1)[0]
    command = prefix + '\nprintf "%s\\n" "$manifest" "${configs[@]}"\n'
    env = {**os.environ, "SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": "0",
           "AMBI_PROJECT_DIR": str(ROOT), "AMBI_PYTHON": sys.executable,
           "EXPECTED_ACTION_MODES_SHA": SHA, "AMBI_AUX_OUTPUT_ROOT": str(tmp_path),
           "AMBI_AUX_CAMPAIGN": "test", "AMBI_AUX_RECIPE": recipe, "AMBI_AUX_MODE": "smoke"}
    result = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(campaign.MANIFEST), *campaign.CASES]
