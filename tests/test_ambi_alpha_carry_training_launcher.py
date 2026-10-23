"""Fail-closed launch provenance and the temperature lifetime gate contract."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from slurm import ambi_alpha_carry_training as campaign


def action(*, t0=False, initial=0.7, final=0.69, first_step=32, outer=0.8,
           alpha_id=1, optimizer_id=2, episode_reset=0):
    return {"t0": t0, "alpha_initial": initial, "alpha_final": final,
            "optimizer_step_initial": first_step, "optimizer_step_final": first_step + 32,
            "outer_alpha": outer, "alpha_identity": alpha_id, "optimizer_identity": optimizer_id,
            "episode_reset": episode_reset}


@pytest.mark.parametrize("scope", ["episode", "run"])
def test_both_modes_preserve_alpha_and_adam_across_decisions(scope):
    previous = action(initial=.72, final=.7, first_step=0)
    current = action()
    campaign.check_temperature_transition(previous, current, scope)
    for key, wrong in [("alpha_initial", .8), ("optimizer_step_initial", 0),
                       ("alpha_identity", 5), ("optimizer_identity", 7)]:
        broken = {**current, key: wrong}
        with pytest.raises(AssertionError):
            campaign.check_temperature_transition(previous, broken, scope)


def test_episode_reset_uses_current_outer_alpha_and_fresh_adam():
    previous = action(initial=.72, final=.7, first_step=0)
    reset = action(t0=True, initial=.8, first_step=0, alpha_id=3, optimizer_id=4, episode_reset=1)
    campaign.check_temperature_transition(previous, reset, "episode")
    with pytest.raises(AssertionError):
        campaign.check_temperature_transition(previous, reset, "run")


def test_run_temperature_survives_episode_boundary():
    previous = action(initial=.72, final=.7, first_step=0)
    current = action(t0=True)
    campaign.check_temperature_transition(previous, current, "run")
    with pytest.raises(AssertionError):
        campaign.check_temperature_transition(previous, current, "episode")


def receipt():
    bound = campaign.binding()
    runtime = {"python": "3.10.19", "torch": "2.3.1+cu121", "cuda": "12.1",
               "gpu": "NVIDIA L40S", "executable": "/locked/python"}
    cases = [{"passed": True, "index": index, "name": campaign.recipe(index)[0],
              "binding": bound, "runtime": runtime, "alpha_and_adam_lifecycle": True,
              "checkpoint_sidecars": True, "checkpoint_roundtrip": True, "checkpoint_bytes": 100}
             for index in range(8)]
    return {"passed": True, "binding": bound, "runtime": runtime, "cases": cases}


@pytest.mark.parametrize("corruption", ["commit", "config", "missing", "duplicate", "lifecycle"])
def test_invalid_gpu_gate_cannot_launch(tmp_path, corruption):
    report = deepcopy(receipt())
    if corruption == "commit":
        report["binding"]["commit"] = "different"
    elif corruption == "config":
        name = next(iter(report["binding"]["config_sha256"]))
        report["binding"]["config_sha256"][name] = "different"
    elif corruption == "missing":
        report["cases"].pop()
    elif corruption == "duplicate":
        report["cases"][7] = report["cases"][0]
    else:
        report["cases"][0]["alpha_and_adam_lifecycle"] = False
    gate = tmp_path / "receipt.json"
    gate.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        campaign.launch(0, tmp_path / "train", gate)
    assert not (tmp_path / "train").exists()


def test_seal_requires_all_eight_distinct_matching_cases(tmp_path):
    report = receipt()
    for row in report["cases"][:-1]:
        (tmp_path / f"case-{row['index']}.json").write_text(json.dumps(row))
    with pytest.raises(FileNotFoundError):
        campaign.seal(tmp_path)
    last = report["cases"][-1]
    (tmp_path / "case-7.json").write_text(json.dumps(last))
    campaign.seal(tmp_path)
    combined = json.loads((tmp_path / "receipt.json").read_text())
    assert combined["passed"] and len(combined["cases"]) == 8
    assert combined["estimated_all_25k_checkpoint_bytes"] == 40 * 8 * 100


def test_verified_launch_selects_exact_cell_and_production_budget(tmp_path, monkeypatch):
    report = receipt()
    monkeypatch.setattr(campaign, "runtime", lambda: report["runtime"])
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_RUN_ID", "test-only")
    monkeypatch.setenv("AMBI_LINEAGE_DIR", str(tmp_path / "lineage"))
    monkeypatch.setenv("AMBI_DRAIN_AFTER_SECONDS", "341700")
    calls = []
    monkeypatch.setattr(campaign.os, "execv", lambda executable, command: calls.append(command))
    gate = tmp_path / "receipt.json"
    gate.write_text(json.dumps(report))
    output = tmp_path / "train"
    campaign.launch(7, output, gate)
    command = calls[0]
    assert "--alg-index" not in command
    assert command[command.index("--resume-checkpoint-minutes") + 1] == "15"
    assert command[command.index("--resume-mode") + 1] == "new"
    selected = json.loads(Path(command[command.index("--run") + 1]).read_text())
    assert selected["configs"] == [campaign.recipe(7)[0]] and selected["trials"] == 1
    record = json.loads((output / "launch.json").read_text())
    assert record["config"] == campaign.recipe(7)[0]
    assert record["run_params"]["total_steps"] == 1_000_000
    assert record["run_params"]["checkpoint_every"] == 25_000


def test_oscar_launcher_resource_and_source_contract():
    launcher = (campaign.ROOT / "slurm/run_ambi_alpha_carry_training_oscar.sbatch").read_text()
    assert "#SBATCH --time=4-00:00:00" in launcher
    assert "#SBATCH --gres=gpu:l40s:1" in launcher
    assert "#SBATCH --cpus-per-task=2" in launcher
    assert "#SBATCH --mem=16G" in launcher
    assert "git status --porcelain --untracked-files=all" in launcher
    assert "AMBI_EXPECTED_COMMIT" in launcher
    assert "gate/receipt.json" in launcher
    assert '[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-7]$ ]]' in launcher
    assert "--allocation rgao48 --filesystem-path /oscar/scratch" in launcher
    assert "scontrol requeue" not in launcher
    assert 'smoke_indices="${AMBI_ALPHA_CARRY_INDICES:-${SLURM_ARRAY_TASK_ID:-}}"' in launcher
