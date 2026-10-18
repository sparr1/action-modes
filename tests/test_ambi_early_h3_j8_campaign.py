import json
from pathlib import Path

import pytest

from slurm import ambi_early_h3_j8_campaign as campaign


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
