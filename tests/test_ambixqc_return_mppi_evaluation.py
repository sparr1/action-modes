"""Four return-critic curves reuse the matching completed prior episode bundles."""
import json
from pathlib import Path
import subprocess

import pytest

import run_ambixqc_return_mppi_evaluation as campaign
from test_ambixqc_backbone_mppi_evaluation import inventory, write
from test_ambixqc_mppi_launcher import bundle


def test_return_grid_excludes_baselines_and_keeps_twenty_per_cell(inventory):
    assert len(campaign.INDICES) == 80
    assert campaign.INDICES == tuple(range(20, 60)) + tuple(range(80, 120))
    for index in (0, 19, 20, 39, 40, 59, 60, 79):
        row = campaign.select_checkpoint(inventory[1], index)
        assert row["cell"] == campaign.CELLS[index // 20]
        assert row["step"] == campaign.banks.STEPS[index % 20]


@pytest.mark.parametrize("index", [-1, 80, True, 0.5])
def test_bad_index_fails_before_checkpoint_reads(index):
    with pytest.raises(ValueError, match="index"):
        campaign.select_checkpoint("missing", index)


def references(tmp_path, inventory):
    _, manifest, data = inventory
    refs = []
    for index in campaign.INDICES:
        row = data["checkpoints"][index]
        path = tmp_path / "prior" / row["cell"] / str(row["step"])
        path.mkdir(parents=True)
        write(path / "manifest.json", {"checkpoint": {"sha256": row["sha256"]}})
        refs.append({"cell": row["cell"], "step": row["step"], "checkpoint_sha256": row["sha256"],
                     "bundle_path": str(path), "manifest_sha256": campaign.file_sha256(path / "manifest.json")})
    path = tmp_path / "references.json"
    write(path, {"schema": campaign.REFERENCE_SCHEMA,
                 "checkpoint_manifest_sha256": campaign.file_sha256(manifest), "references": refs})
    return path


@pytest.mark.parametrize("problem", [None, "grid", "inventory", "checkpoint", "manifest", "missing"])
def test_reference_hash_and_association(inventory, tmp_path, problem):
    path = references(tmp_path, inventory)
    data = campaign.banks.read_json(path)
    row = campaign.select_checkpoint(inventory[1], 0)
    selected = Path(data["references"][0]["bundle_path"])
    if problem == "grid":
        data["references"].pop()
    elif problem == "inventory":
        data["checkpoint_manifest_sha256"] = "f" * 64
    elif problem == "checkpoint":
        data["references"][0]["checkpoint_sha256"] = "f" * 64
    elif problem == "manifest":
        (selected / "manifest.json").write_text("{}")
    elif problem == "missing":
        (selected / "manifest.json").unlink()
    write(path, data)
    if problem is not None:
        with pytest.raises(ValueError):
            campaign.select_reference(path, row, inventory[1])
    else:
        assert campaign.select_reference(path, row, inventory[1]) == selected


def test_production_runs_only_return_mppi_and_reuses_reference(inventory, tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    import report_ambi_benchmark as report
    import utils.ambi_benchmark as benchmark
    path = references(tmp_path, inventory)
    assigned = {cell: {campaign.SELECTOR: str(tmp_path / f"curve-{cell}")} for cell in campaign.CELLS}
    mapping = tmp_path / "map.json"
    write(mapping, {"schema": campaign.MAP_SCHEMA, "cells": assigned})
    monkeypatch.setattr(benchmark, "resolve_eval_run_map", lambda selectors, run_map: run_map)
    monkeypatch.setattr(benchmark, "code_identity", lambda: {"commit": "a" * 40, "dirty": False})
    monkeypatch.setattr(report, "load_bundles", lambda paths: paths)
    monkeypatch.setattr(report, "write_report", lambda bundles, path, **kwargs: path.write_text("report"))
    seen = []
    def evaluate(matrix, checkpoint, **kwargs):
        seen.append(kwargs)
        kwargs["bundle_dir"].mkdir()
        return {"checkpoint_sha256": inventory[2]["checkpoints"][119]["sha256"]}
    monkeypatch.setattr(evaluator, "evaluate_matrix", evaluate)
    monkeypatch.setattr(campaign, "validate_return_bundle", lambda *a, **kw:
                        {"outer_state_unchanged": True, "optimizer_updates": 0})
    monkeypatch.setattr(benchmark, "stage_completed_bundle", lambda *a, **kw:
                        {campaign.SELECTOR: {"status": "queued"}})
    result = campaign.run(inventory[1], 79, tmp_path / "results", eval_run_map=mapping,
                          reference_index=path, device="cpu")
    assert (result / "PASS").is_file()
    assert seen[0]["selectors"] == [campaign.SELECTOR]
    assert seen[0]["seeds"] == [101, 102, 103, 104, 105] and seen[0]["max_steps"] == 500
    assert seen[0]["reference_bundle"] == tmp_path / "prior/aux_detached_utd2/500000"
    assert seen[0]["eval_run_map"] == assigned["aux_detached_utd2"]
    with pytest.raises(FileExistsError):
        campaign.run(inventory[1], 79, tmp_path / "results", eval_run_map=mapping,
                     reference_index=path, device="cpu")


def test_return_acceptance_rejects_soft_tail_and_incomplete_pair(tmp_path):
    bundle(tmp_path)
    path = tmp_path / "manifest.json"
    data = json.loads(path.read_text())
    data["runs"][1].update(selector=campaign.SELECTOR,
                           evaluation_controller={"protocol": {"terminal_value_source": campaign.TERMINAL_SOURCE}})
    write(path, data)
    actual = campaign.validate_return_bundle(tmp_path, seeds=[101, 102], max_steps=3)
    assert actual["decision_counts"] == {"controller/prior": 6, campaign.SELECTOR: 6}
    data["runs"][1]["evaluation_controller"]["protocol"]["terminal_value_source"] = "online_xqc_twin_mean"
    write(path, data)
    with pytest.raises(ValueError, match="wrong terminal"):
        campaign.validate_return_bundle(tmp_path, seeds=[101, 102], max_steps=3)


def test_smoke_gate_rejects_missing_smokes(tmp_path, inventory):
    with pytest.raises(ValueError, match="four successful"):
        campaign.verify_smokes(tmp_path, inventory[1], "a" * 40)


def test_return_matrix_changes_only_terminal_critic():
    soft = campaign.banks.read_json(campaign.banks.MATRIX)
    ret = campaign.banks.read_json(campaign.MATRIX)
    old = soft["comparisons"]["controller"]["variants"]["mppi"]["evaluation_controller"]
    new = ret["comparisons"]["controller"]["variants"]["mppi_return"]["evaluation_controller"]
    assert new == {**old, "terminal_critic_source": "aux_return"}
    for key in ("seeds", "max_steps", "controller_seed"):
        assert ret["evaluation"][key] == soft["evaluation"][key]


def test_launcher_requires_verified_references_and_completed_smokes():
    path = campaign.ROOT / "slurm/run_ambixqc_return_mppi_eval_oscar.sbatch"
    text = path.read_text()
    for expected in ("--array=0", "#SBATCH --gres=gpu:l40s:1", "#SBATCH --cpus-per-task=6",
                     "SLURM_ARRAY_TASK_ID < 80", "0|39|40|79", "--reference-index",
                     "--verify-smoke-root", "LOCK_SHA", "--untracked-files=all",
                     "test_xqc_return_mppi.py", "WANDB_MODE=disabled", "--color=no"):
        assert expected in text
    subprocess.run(["bash", "-n", str(path)], check=True, close_fds=False)
