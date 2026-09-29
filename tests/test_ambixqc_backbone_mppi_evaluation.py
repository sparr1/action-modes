"""Source identity, relocation, protocol and publication of six frozen banks."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

import run_ambixqc_backbone_mppi_evaluation as campaign
from test_ambixqc_mppi_launcher import bundle


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def inventory(tmp_path):
    root = tmp_path / "training"
    root.mkdir()
    for index, cell in enumerate(campaign.CELLS):
        run = root / f"production-{cell}-seed55-job{index}-task{index}" / f"run-{cell}"
        weights = run / "training" / "experiment" / "models"
        weights.mkdir(parents=True)
        (run.parent / "PASS").write_text("PASS\n")
        algorithm = campaign.read_json(campaign.ROOT / "configs/dmcontrol/algs" /
                                       f"ambixqc_humanoid_walk_backbone_replay_500k_{cell}.json")
        experiment = campaign.read_json(campaign.ROOT / "configs/dmcontrol/experiments" /
                                        f"ambixqc_humanoid_walk_backbone_replay_500k_{cell}.json")
        algorithm["resolved_runtime"] = {"observation": {
            "mode": "state", "shape": [67], "dtype": "float32", "action_dim": 21,
            "latent_dim": 512, "episode_length": 500,
        }}
        records = []
        for step in campaign.STEPS:
            path = weights / f"checkpoint_{step}.pt"
            path.write_bytes(f"weights-{cell}-{step}".encode())
            digest = campaign.file_sha256(path)
            write(Path(str(path) + ".metadata.json"), {
                "schema_version": 1, "checkpoint": {"step": step},
                "trial_run_params": algorithm, "experiment_params": experiment,
                "replay": {"schema": "ambi-replay-reference", "version": 1,
                           "step": step, "checkpoint_sha256": digest},
            })
            records.append({"path": str(path), "step": step, "sha256": digest,
                            "replay_hashes_verified": True})
        write(run / "validation.json", {
            "schema": "ambixqc-backbone-replay-500k-validation-v1", "mode": "production",
            "source_sha": campaign.TRAINING_SOURCE_SHA, "arm": cell, "total_steps": 500_000,
            "seed": 55, "xqc_utd": int(cell[-1]), "replay_capacity": 1_000_000,
            "lr_transition_steps": 500_000 * int(cell[-1]), "all_finite": True,
            "final_raw_replay_matches_training": True, "checkpoints": records,
        })
    manifest = tmp_path / "inventory.json"
    data = campaign.build_inventory(root, manifest)
    return root, manifest, data


def test_inventory_covers_all_cells_and_both_ends(inventory):
    root, manifest, data = inventory
    assert len(data["checkpoints"]) == 120
    for index in (0, 19, 20, 39, 40, 59, 60, 79, 80, 99, 100, 119):
        row = campaign.select_checkpoint(manifest, index)
        assert row["cell"] == campaign.CELLS[index // 20]
        assert row["step"] == campaign.STEPS[index % 20]
        assert row["source_run"] == campaign.SOURCE_RUNS[row["cell"]]
        assert Path(row["path"]).is_relative_to(root)


def test_inventory_relocates_without_rewriting_hashes(inventory, tmp_path):
    root, manifest, original = inventory
    copied = tmp_path / "oscar"
    shutil.move(str(root), copied)
    row = campaign.select_checkpoint(manifest, 119, checkpoint_root=copied)
    assert Path(row["path"]).is_relative_to(copied)
    rebuilt = campaign.build_inventory(copied, tmp_path / "rebuilt.json")
    assert rebuilt["checkpoints"] == original["checkpoints"]
    assert rebuilt["training_validations"] == original["training_validations"]
    # A relative root is resolved against the manifest, never the process cwd.
    original["checkpoint_root"] = "oscar"
    write(manifest, original)
    assert campaign.select_checkpoint(manifest, 0)["path"].startswith(str(copied))


@pytest.mark.parametrize("change", ["grid", "order", "cell", "duplicate", "traversal", "absolute", "source", "source_sha", "contract", "digest", "bool_step", "evidence"])
def test_inventory_rejects_bad_grid_or_identity_before_output(inventory, tmp_path, change):
    _, manifest, data = inventory
    if change == "grid":
        data["checkpoints"].pop()
    elif change == "order":
        data["checkpoints"][0], data["checkpoints"][1] = data["checkpoints"][1], data["checkpoints"][0]
    elif change == "cell":
        data["checkpoints"][0]["cell"] = "foreign"
    elif change == "duplicate":
        data["checkpoints"][1]["path"] = data["checkpoints"][0]["path"]
    elif change in {"traversal", "absolute"}:
        data["checkpoints"][0]["path"] = "../escape.pt" if change == "traversal" else "/escape.pt"
    elif change == "source":
        data["checkpoints"][0]["source_run"] = campaign.SOURCE_RUNS["baseline_utd2"]
    elif change == "source_sha":
        data["training_source_sha"] = "a" * 40
    elif change == "contract":
        data["source_contract"]["seed"] = 56
    elif change == "digest":
        data["checkpoints"][0]["sha256"] = "x"
    elif change == "bool_step":
        data["checkpoints"][0]["step"] = True
    else:
        del data["training_validations"]["baseline_utd1"]
    write(manifest, data)
    with pytest.raises(ValueError):
        campaign.run(manifest, 0, tmp_path / "results", mode="smoke")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("index", [-1, 120, True, 1.5])
def test_index_strictly_validated(inventory, index):
    with pytest.raises(ValueError, match="index"):
        campaign.select_checkpoint(inventory[1], index)


@pytest.mark.parametrize("kind", ["weights", "metadata", "validation", "pass", "missing"])
def test_tampered_or_incomplete_bank_rejected(inventory, kind):
    root, manifest, data = inventory
    row = data["checkpoints"][0]
    path = root / row["path"]
    evidence = root / data["training_validations"][row["cell"]]["path"]
    if kind == "weights":
        path.write_bytes(b"changed")
    elif kind == "metadata":
        Path(str(path) + ".metadata.json").write_text("{}")
    elif kind == "validation":
        evidence.write_text("{}")
    elif kind == "pass":
        (evidence.parent.parent / "PASS").unlink()
    else:
        path.unlink()
    with pytest.raises(ValueError):
        campaign.select_checkpoint(manifest, 0)


@pytest.mark.parametrize("field,value", [("xqc_utd", 1), ("aux_return_mode", "off"),
                                          ("aux_return_detach_representation", True),
                                          ("inner_operator", "xqc"), ("replay_capacity", 500_000)])
def test_valid_hash_cannot_hide_wrong_training_configuration(inventory, field, value):
    root, manifest, data = inventory
    row = data["checkpoints"][80]  # Shared UTD2
    path = Path(str(root / row["path"]) + ".metadata.json")
    metadata = campaign.read_json(path)
    metadata["trial_run_params"]["alg_params"][field] = value
    write(path, metadata)
    row["metadata_sha256"] = campaign.file_sha256(path)
    write(manifest, data)
    with pytest.raises(ValueError, match="metadata"):
        campaign.select_checkpoint(manifest, 80)


def test_build_inventory_preserves_existing_file_and_rejects_incomplete_sources(inventory, tmp_path):
    root, manifest, _ = inventory
    before = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        campaign.build_inventory(root, manifest)
    assert manifest.read_bytes() == before
    next(root.glob("production-*/PASS")).unlink()
    with pytest.raises(ValueError, match="PASS"):
        campaign.build_inventory(root, tmp_path / "incomplete.json")
    assert not (tmp_path / "incomplete.json").exists()


def run_map(tmp_path):
    cells = {cell: {selector: str(tmp_path / f"curve-{cell}-{selector.split('/')[-1]}")
                    for selector in campaign.SELECTORS} for cell in campaign.CELLS}
    path = tmp_path / "run-map.json"
    write(path, {"schema": campaign.RUN_MAP_SCHEMA, "cells": cells})
    return path, cells


def test_smoke_forbids_assignments_production_requires_explicit_map(tmp_path):
    path, _ = run_map(tmp_path)
    assert campaign.resolve_cell_run_map(None, campaign.CELLS[0], mode="smoke") == {}
    with pytest.raises(ValueError, match="Smoke"):
        campaign.resolve_cell_run_map(path, campaign.CELLS[0], mode="smoke")
    with pytest.raises(ValueError, match="Production"):
        campaign.resolve_cell_run_map(None, campaign.CELLS[0], mode="production")


@pytest.mark.parametrize("problem", ["missing_cell", "missing_selector", "duplicate", "relative"])
def test_map_cannot_mix_cells_or_planners(tmp_path, problem):
    path, cells = run_map(tmp_path)
    if problem == "missing_cell":
        del cells[campaign.CELLS[-1]]
    elif problem == "missing_selector":
        del cells[campaign.CELLS[-1]][campaign.SELECTORS[-1]]
    elif problem == "duplicate":
        cells[campaign.CELLS[-1]][campaign.SELECTORS[-1]] = cells[campaign.CELLS[0]][campaign.SELECTORS[0]]
    else:
        cells[campaign.CELLS[-1]][campaign.SELECTORS[-1]] = "relative"
    write(path, {"schema": campaign.RUN_MAP_SCHEMA, "cells": cells})
    with pytest.raises(ValueError):
        campaign.resolve_cell_run_map(path, campaign.CELLS[0], mode="production")


def test_run_records_protocol_provenance_and_preserves_original_driver(inventory, tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    import report_ambi_benchmark as report
    import utils.ambi_benchmark as benchmark
    _, manifest, data = inventory
    calls = []
    def evaluate(matrix, checkpoint, **kwargs):
        calls.append((matrix, checkpoint, kwargs))
        kwargs["bundle_dir"].mkdir()
        bundle(kwargs["bundle_dir"])
        return {"checkpoint_sha256": data["checkpoints"][80]["sha256"]}
    monkeypatch.setattr(evaluator, "evaluate_matrix", evaluate)
    monkeypatch.setattr(benchmark, "code_identity", lambda: {"commit": "a" * 40, "dirty": False})
    monkeypatch.setattr(report, "load_bundles", lambda paths: paths)
    monkeypatch.setattr(report, "write_report", lambda data, path, **kwargs: path.write_text("report"))
    monkeypatch.setattr(benchmark, "stage_completed_bundle", lambda *args, **kwargs: pytest.fail("smoke publication"))
    output = campaign.run(manifest, 80, tmp_path / "results", mode="smoke", device="cpu")
    assert output == tmp_path / "results/aux_shared_utd2/step_25000"
    assert (output / "PASS").is_file()
    validation = campaign.read_json(output / "validation.json")
    assert validation["index"] == 80 and validation["cell"] == "aux_shared_utd2"
    assert validation["manifest_sha256"] == campaign.file_sha256(manifest)
    assert validation["matrix_sha256"] == campaign.file_sha256(campaign.MATRIX)
    assert validation["source_run"] == campaign.SOURCE_RUNS["aux_shared_utd2"]
    assert validation["training_source_sha"] == campaign.TRAINING_SOURCE_SHA
    assert validation["replay_used"] is False
    assert validation["optimizer_updates"] == 0 and validation["outer_state_unchanged"]
    assert calls[0][2]["seeds"] == [101, 102]
    assert calls[0][2]["max_steps"] == 3 and calls[0][2]["controller_seed"] == 12345
    assert calls[0][2]["stage_results"] is False
    with pytest.raises(FileExistsError):
        campaign.run(manifest, 80, tmp_path / "results", mode="smoke")


def test_metadata_only_specs_select_actual_bank_source(inventory, tmp_path, monkeypatch):
    import evaluate_ambi_checkpoint as evaluator
    seen = {}
    monkeypatch.setattr(evaluator, "evaluate_matrix", lambda *args, **kwargs: seen.update(kwargs) or {"specs": {}})
    campaign.prepare_specs(inventory[1], 100, tmp_path / "specs")
    assert seen["source_run"] == campaign.SOURCE_RUNS["aux_detached_utd2"]
    assert seen["seeds"] == [101, 102, 103, 104, 105] and seen["max_steps"] == 500
    assert seen["controller_seed"] == 12345 and "bundle_dir" not in seen


def test_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": 1, "schema": 2}')
    with pytest.raises(ValueError, match="Duplicate"):
        campaign.read_json(path)


@pytest.mark.parametrize("staging_succeeds", [True, False])
def test_production_uses_own_cell_curves_and_reports_staging_failure(inventory, tmp_path, monkeypatch, staging_succeeds):
    import evaluate_ambi_checkpoint as evaluator
    import report_ambi_benchmark as report
    import utils.ambi_benchmark as benchmark
    _, manifest, data = inventory
    map_path, assignments = run_map(tmp_path)
    calls, staged = [], []
    monkeypatch.setattr(benchmark, "resolve_eval_run_map", lambda selectors, run_map: run_map)
    monkeypatch.setattr(benchmark, "code_identity", lambda: {"commit": "a" * 40, "dirty": False})
    monkeypatch.setattr(report, "load_bundles", lambda paths: paths)
    monkeypatch.setattr(report, "write_report", lambda data, path, **kwargs: path.write_text("report"))
    def evaluate(matrix, checkpoint, **kwargs):
        calls.append(kwargs)
        kwargs["bundle_dir"].mkdir()
        bundle(kwargs["bundle_dir"])
        return {"checkpoint_sha256": data["checkpoints"][119]["sha256"]}
    def validate(path, **kwargs):
        assert kwargs == {"seeds": [101, 102, 103, 104, 105], "max_steps": 500}
        return {"outer_state_unchanged": True, "optimizer_updates": 0}
    def stage(path, mapping, **kwargs):
        staged.append((path, mapping, kwargs))
        return {selector: {"status": "queued" if staging_succeeds else "failed"}
                for selector in campaign.SELECTORS}
    monkeypatch.setattr(evaluator, "evaluate_matrix", evaluate)
    monkeypatch.setattr(campaign, "validate_bundle", validate)
    monkeypatch.setattr(benchmark, "stage_completed_bundle", stage)
    output = tmp_path / "results/aux_detached_utd2/step_500000"
    if staging_succeeds:
        campaign.run(manifest, 119, tmp_path / "results", mode="production", eval_run_map=map_path)
        assert (output / "PASS").is_file()
    else:
        with pytest.raises(RuntimeError, match="stage all curves"):
            campaign.run(manifest, 119, tmp_path / "results", mode="production", eval_run_map=map_path)
        assert not (output / "PASS").exists()
    assert (output / "paired.json").is_file() and (output / "validation.json").is_file()
    assert calls[0]["eval_run_map"] == assignments["aux_detached_utd2"]
    assert calls[0]["max_steps"] == 500 and calls[0]["seeds"] == [101, 102, 103, 104, 105]
    assert calls[0]["source_run"] == campaign.SOURCE_RUNS["aux_detached_utd2"]
    assert staged[0][1] == assignments["aux_detached_utd2"]
    assert staged[0][2]["source_run"] == campaign.SOURCE_RUNS["aux_detached_utd2"]


def test_symlink_cannot_escape_training_root(inventory, tmp_path):
    root, manifest, data = inventory
    row = data["checkpoints"][0]
    path = root / row["path"]
    outside = tmp_path / "outside.pt"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        campaign.select_checkpoint(manifest, 0)
