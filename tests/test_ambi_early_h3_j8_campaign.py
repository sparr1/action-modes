import json
from copy import deepcopy
from pathlib import Path

import pytest

from slurm import ambi_early_h3_j8_campaign as campaign


ORIGINAL_SELECTION = campaign.REPO / "configs/research/ambi_original_bank_h3_j8_campaign.json"


def original_inventory():
    selection, _ = campaign.selection_definition(ORIGINAL_SELECTION)
    rows = [{**pin, "checkpoint": f"/original/weights-{pin['step']}",
             "metadata": f"/original/weights-{pin['step']}.metadata.json"}
            for pin in selection["checkpoint_pins"]]
    # The historical 80-checkpoint inventory stores source identity at its
    # top level rather than repeating it on every checkpoint.
    for row in rows:
        row.pop("backbone_id")
    rows.append({"step": 25000, "checkpoint": "/original/unselected",
                 "checkpoint_sha256": "0" * 64, "metadata_sha256": "1" * 64})
    return {"source_run": campaign.SOURCE_DEFINITIONS["original_aux"]["source_run"],
            "checkpoints": rows}


def test_commands_preserve_science_and_keep_smoke_out_of_publication(tmp_path):
    plan = {"seeds": campaign.SEEDS, "max_steps": 500, "controller_seed": 55,
            "matrix": campaign.MATRIX}
    cell = {"checkpoint": "/weights", "metadata": "/sidecar", "selector": "early_h3_j8/sample",
            "inventory_path": "/inventory"}
    command, seeds, limit = campaign.evaluator_command(plan, cell, tmp_path, smoke_steps=32)
    assert seeds == [101] and limit == 32
    assert "--eval-run-dir" not in command
    command, seeds, limit = campaign.evaluator_command(plan, cell, tmp_path, eval_run_dir="/registry")
    assert seeds == [101, 102, 103, 104, 105] and limit == 500
    assert command[command.index("--preset") + 1] == "early_h3_j8/sample"
    assert command[command.index("--eval-run-dir") + 1] == "/registry"
    assert "--wandb" not in command


def test_prepare_rejects_missing_or_wrong_checkpoint_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "verify_source", lambda: "a" * 40)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"checkpoints": []}))
    with pytest.raises(ValueError, match="eight checkpoints"):
        campaign.prepare(inventory, tmp_path / "out")
    rows = [{"backbone_id": b, "step": s, "source_run": "wrong"}
            for b in campaign.SOURCES for s in campaign.STEPS]
    inventory.write_text(json.dumps({"checkpoints": rows}))
    with pytest.raises(ValueError, match="source run"):
        campaign.prepare(inventory, tmp_path / "out")


def test_write_new_never_overwrites_completed_work(tmp_path):
    path = tmp_path / "receipt.json"
    campaign.write_new(path, {"status": "complete"})
    with pytest.raises(FileExistsError):
        campaign.write_new(path, {"status": "replaced"})
    assert json.loads(path.read_text()) == {"status": "complete"}


def test_original_selection_pins_only_authorized_source_steps_and_hashes():
    selection, provenance = campaign.selection_definition(ORIGINAL_SELECTION)
    assert selection["campaign_id"] == "original-bank-h3-j8-20261018"
    assert selection["sources"] == {"original_aux": campaign.SOURCE_DEFINITIONS["original_aux"]}
    assert selection["steps"] == [200000, 400000, 475000, 500000]
    assert provenance == {"path": str(ORIGINAL_SELECTION.relative_to(campaign.REPO)),
                          "sha256": campaign.sha256(ORIGINAL_SELECTION)}
    inventory = original_inventory()
    selected = campaign.select_inventory_rows(inventory, selection)
    assert [row["step"] for row in selected] == selection["steps"]
    for row, pin in zip(selected, selection["checkpoint_pins"]):
        assert row["backbone_id"] == "original_aux"
        assert row["source_run"] == "rwgao_b-brown-university/ambi/aux6428346x0"
        assert row["checkpoint_sha256"] == pin["checkpoint_sha256"]
        assert row["metadata_sha256"] == pin["metadata_sha256"]


@pytest.mark.parametrize("corruption", ["source", "checkpoint", "metadata", "missing", "duplicate"])
def test_original_selection_rejects_wrong_or_incomplete_inputs(corruption):
    selection, _ = campaign.selection_definition(ORIGINAL_SELECTION)
    inventory = original_inventory()
    if corruption == "source":
        inventory["source_run"] = campaign.SOURCES["sac_aux"]
        inventory["checkpoints"][0]["backbone_id"] = "original_aux"
    elif corruption in {"checkpoint", "metadata"}:
        inventory["checkpoints"][0][corruption + "_sha256"] = "f" * 64
    elif corruption == "missing":
        inventory["checkpoints"].pop(0)
    else:
        inventory["checkpoints"].append(deepcopy(inventory["checkpoints"][0]))
    with pytest.raises(ValueError):
        campaign.select_inventory_rows(inventory, selection)


def test_prepare_original_manifest_has_eight_selected_cells_and_no_other_backbones(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "verify_source", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "git", lambda *args: "b" * 40)
    verified = []
    monkeypatch.setattr(campaign, "verify_checkpoint", lambda row: verified.append(deepcopy(row)))
    inventory = tmp_path / "inventory.json"
    campaign.write_new(inventory, original_inventory())
    output = tmp_path / "campaign"
    result = campaign.prepare(inventory, output, selection_path=ORIGINAL_SELECTION)
    assert len(verified) == 4
    assert len(result["cells"]) == 8
    assert result["steps"] == [200000, 400000, 475000, 500000]
    assert result["seeds"] == [101, 102, 103, 104, 105]
    assert result["controller_seed"] == 55 and result["max_steps"] == 500
    assert {(cell["backbone_id"], cell["step"], cell["execution"]) for cell in result["cells"]} == {
        ("original_aux", step, mode) for step in result["steps"] for mode in ("mean", "sample")}
    assert [cell["index"] for cell in result["cells"]] == list(range(8))
    for cell in result["cells"]:
        assert cell["source_run"] == "rwgao_b-brown-university/ambi/aux6428346x0"
        assert cell["selector"] == "early_h3_j8/" + cell["execution"]
        assert Path(cell["output_dir"]).parent == output / "production"
    saved_inventory = campaign.read(output / "inputs/original_aux.json")
    assert [row["step"] for row in saved_inventory["checkpoints"]] == result["steps"]
    assert campaign.read(output / "campaign.json") == result
    campaign.validate_campaign_selection(result)
    for corruption in ("source", "step", "hash", "mode", "missing", "selection"):
        wrong = deepcopy(result)
        if corruption == "source":
            wrong["cells"][0].update(backbone_id="sac_aux", source_run=campaign.SOURCES["sac_aux"])
        elif corruption == "step":
            wrong["cells"][0]["step"] = 575000
        elif corruption == "hash":
            wrong["cells"][0]["checkpoint_sha256"] = "f" * 64
        elif corruption == "mode":
            wrong["cells"][0]["selector"] = "early_prior/mean"
        elif corruption == "missing":
            wrong["cells"].pop()
        else:
            wrong["selection"]["sha256"] = "f" * 64
        with pytest.raises(ValueError):
            campaign.validate_campaign_selection(wrong)


@pytest.mark.parametrize("operator", ["sac", "mppi"])
def test_original_checkpoint_rejects_training_backbone_even_with_self_consistent_hashes(tmp_path, operator):
    weights = tmp_path / "checkpoint"
    weights.write_bytes(b"weights")
    metadata = tmp_path / "metadata.json"
    campaign.write_new(metadata, {"checkpoint": {"step": 200000}, "trial_run_params": {
        "alg_params": {"inner_operator": operator, "aux_return_mode": "sac"}}})
    row = {"backbone_id": "original_aux", "source_run": "rwgao_b-brown-university/ambi/aux6428346x0",
           "step": 200000, "checkpoint": str(weights), "metadata": str(metadata),
           "checkpoint_sha256": campaign.sha256(weights), "metadata_sha256": campaign.sha256(metadata)}
    with pytest.raises(ValueError, match="Unexpected source backbone"):
        campaign.verify_checkpoint(row)
    contents = campaign.read(metadata)
    contents["trial_run_params"]["alg_params"]["inner_operator"] = "none"
    metadata.write_text(json.dumps(contents))
    row["metadata_sha256"] = campaign.sha256(metadata)
    campaign.verify_checkpoint(row)
