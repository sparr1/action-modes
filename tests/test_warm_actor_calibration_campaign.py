import json
from pathlib import Path

import pytest

from slurm.warm_actor_calibration_campaign import CHECKPOINT_SHA, CELLS, command_for, prepare


def source_campaign(path):
    cells = [dict(name=f"actor_warm_h{h}_j{j}_c16", selector=f"sweep/actor_warm_h{h}_j{j}_c16",
                  checkpoint="/frozen/checkpoint", metadata_sha256="metadata", expected_config={})
             for h, j in CELLS]
    path.write_text(json.dumps(dict(checkpoint_sha256=CHECKPOINT_SHA,
                                    study_protocol="actor-transfer-v2", cells=cells)))
    return path


def test_complete_panel_and_dependency_mapping(tmp_path):
    campaign = prepare(tmp_path / "campaign", source_campaign(tmp_path / "source.json"))
    assert len(campaign["captures"]) == 20
    assert len(campaign["prefixes"]) == 120
    assert len(campaign["replans"]) == 40
    assert {(x["H"], x["J"]) for x in campaign["cells"]} == set(CELLS)
    for kind in ("prefixes", "replans"):
        for task in campaign[kind]:
            capture = campaign["captures"][task["capture_index"]]
            assert (task["source_cell"], task["seed"]) == (capture["source_cell"], capture["seed"])
            assert task["decision"] in capture["decisions"]
            assert Path(task["root_file"]).parent.name == f'decision-{task["decision"]}'
    assert campaign["tail_steps"] == 1000 and campaign["rollouts"] == 32
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "campaign", tmp_path / "source.json")


def test_smoke_is_explicit_and_covers_h1_h3(tmp_path):
    c = prepare(tmp_path / "smoke", source_campaign(tmp_path / "source.json"), smoke=True)
    assert c["smoke"] and len(c["captures"]) == 2
    assert {(x["H"], x["J"]) for x in c["captures"]} == {(3, 10), (1, 8)}
    for kind in ("capture", "prefix", "replan"):
        command = command_for(c, kind, 0)
        assert "--checkpoint" in command and "--matrix" in command
    assert c["wandb_project"].endswith("validation")


def test_wrong_checkpoint_rejected(tmp_path):
    p = source_campaign(tmp_path / "source.json")
    d = json.loads(p.read_text()); d["checkpoint_sha256"] = "wrong"; p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="corrected"):
        prepare(tmp_path / "campaign", p)


def test_provenance_guard_uses_actual_bytes_and_resolved_science(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from slurm.warm_actor_calibration_campaign import verify_source_provenance, file_sha256
    from utils import checkpoint_context, ambi_research, eval_series_data

    checkpoint, sidecar, matrix = [tmp_path / name for name in ("weights", "metadata.json", "matrix.json")]
    checkpoint.write_bytes(b"actual weights")
    sidecar.write_text("{}")
    matrix.write_text("{}")
    expected = {"device": "cpu", "compile": True, "inner_rounds": 8}
    actual = {**expected, "device": "cuda"}
    cell = dict(source_cell="warm", selector="sweep/warm", checkpoint=str(checkpoint),
                checkpoint_sha256=file_sha256(checkpoint), metadata_sha256=file_sha256(sidecar),
                expected_config=expected)
    campaign = dict(cells=[cell], checkpoint_sha256=cell["checkpoint_sha256"], matrix=str(matrix))
    monkeypatch.setattr(checkpoint_context, "load_checkpoint_context", lambda p:
                        SimpleNamespace(source=sidecar, metadata={}))
    monkeypatch.setattr(ambi_research, "load_preset_matrix", lambda p: {})
    monkeypatch.setattr(ambi_research, "resolve_preset", lambda *a, **k: {})
    monkeypatch.setattr(eval_series_data, "resolved_checkpoint_config", lambda *a: actual)
    assert verify_source_provenance(campaign, cell)["config_verified"]
    actual["inner_rounds"] = 6
    with pytest.raises(ValueError, match="configuration differs"):
        verify_source_provenance(campaign, cell)
    actual["inner_rounds"] = 8
    actual["compile"] = False
    with pytest.raises(ValueError, match="configuration differs"):
        verify_source_provenance(campaign, cell)
    actual["compile"] = True
    sidecar.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="metadata checksum"):
        verify_source_provenance(campaign, cell)
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        verify_source_provenance(campaign, cell)


def test_branch_dispatch_requires_complete_matching_capture(tmp_path):
    from slurm.warm_actor_calibration_campaign import verify_captured_source, file_sha256
    matrix = tmp_path / "matrix.json"
    matrix.write_text("{}")
    expected = {"device": "cpu", "compile": True, "inner_rounds": 8}
    cell = dict(source_cell="warm", selector="sweep/warm", checkpoint_sha256="weights",
                expected_config=expected)
    task = dict(source_cell="warm", seed=101, capture_directory=str(tmp_path))
    campaign = dict(cells=[cell], controller_seed=55, matrix=str(matrix))
    manifest = dict(status="complete", checkpoint_sha256="weights", selector="sweep/warm",
                    seed=101, controller_seed=55, matrix_sha256=file_sha256(matrix),
                    resolved_config={**expected, "device": "cuda"})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    verify_captured_source(campaign, task)
    manifest["resolved_config"]["inner_rounds"] = 6
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="configuration differs"):
        verify_captured_source(campaign, task)
