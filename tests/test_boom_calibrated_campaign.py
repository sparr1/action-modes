import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm.boom_likelihood_calibrated_oscar import verify_training_recipe, evaluation_plan
from slurm.graph_checkpoint_campaign import checkpoint_steps, inventory
from slurm import graph_checkpoint_campaign as campaign
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.eval_series_data import resolved_checkpoint_config

ROOT = Path(__file__).resolve().parents[1]


def test_calibrated_recipe_changes_only_strength_and_publication_labels():
    verify_training_recipe()


def test_reverse_inventory_rejects_missing_and_duplicate_checkpoints(tmp_path):
    steps = checkpoint_steps(400000, 25000, True)
    assert steps == list(range(400000, 0, -25000)) and len(steps) == 16
    folder = tmp_path / "experiment" / "models"
    folder.mkdir(parents=True)
    for step in steps:
        weight = folder / f"model:{step}"
        weight.write_bytes(str(step).encode())
        Path(str(weight) + ".metadata.json").write_text(json.dumps(
            {"checkpoint": {"step": step}, "trial_run_params": {"name": "calibrated"}}))
    assert [x["step"] for x in inventory(tmp_path, "calibrated", steps)] == steps
    weight.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        inventory(tmp_path, "calibrated", steps)
    weight.write_bytes(b"last")
    duplicate = folder / "model:duplicate"
    duplicate.write_bytes(b"duplicate")
    Path(str(duplicate) + ".metadata.json").write_text(Path(str(weight) + ".metadata.json").read_text())
    with pytest.raises(ValueError, match="duplicates"):
        inventory(tmp_path, "calibrated", steps)


def checkpoint_fixture(root, step):
    folder = root / "experiment" / "models"
    folder.mkdir(parents=True, exist_ok=True)
    weight = folder / f"model:{step}"
    weight.write_bytes(str(step).encode())
    Path(str(weight) + ".metadata.json").write_text(json.dumps(
        {"checkpoint": {"step": step}, "trial_run_params": {"name": "calibrated"}}))
    return weight


def test_streaming_inventory_keeps_indices_and_published_inventory_immutable(tmp_path, monkeypatch):
    steps = [75000, 50000, 25000]
    checkpoint_fixture(tmp_path, 25000)
    rows = inventory(tmp_path, "calibrated", steps, allow_missing=True)
    grid = campaign.complete_grid(steps, rows)
    assert [r["step"] for r in grid] == steps
    assert [i for i, r in enumerate(grid) if "path" in r] == [2]
    pinned = tmp_path / "result/checkpoint-inventory.json"
    campaign.write_inventory(pinned, {"source_run": "run", "training_root": str(tmp_path)}, rows)
    published_digest = campaign.series._file_digest(pinned)
    checkpoint_fixture(tmp_path, 50000)
    real_digest = campaign.series._file_digest
    reads = []
    def digest(path):
        reads.append(Path(path).name)
        return real_digest(path)
    monkeypatch.setattr(campaign.series, "_file_digest", digest)
    newer = inventory(tmp_path, "calibrated", steps, allow_missing=True, known=grid)
    assert [r["step"] for r in newer] == [50000, 25000]
    assert "model:25000" not in reads
    assert "model:50000" in reads
    assert real_digest(pinned) == published_digest
    assert [i for i, r in enumerate(campaign.complete_grid(steps, newer)) if "path" in r] == [1, 2]
    Path(rows[0]["metadata_path"]).write_text(json.dumps(
        {"checkpoint": {"step": 25000}, "trial_run_params": {"name": "calibrated"}, "changed": True}))
    with pytest.raises(ValueError, match="changed"):
        inventory(tmp_path, "calibrated", steps, allow_missing=True, known=grid)


def test_streaming_release_only_ready_tasks_in_reverse_order_and_is_restart_safe(monkeypatch):
    manifest = {"checkpoints": [{"step": 75000}, {"step": 50000, "path": "new"},
                                {"step": 25000, "path": "old"}]}
    calls, saves, released = [], [], set()
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw:
                        "JobState=PENDING Reason=JobHeldUser Priority=0 Dependency=(null) ")
    monkeypatch.setattr(campaign.subprocess, "run", lambda args, **kw: calls.append(args))
    campaign.release_available(manifest, "123", released, lambda: saves.append(sorted(released)))
    assert calls == [["scontrol", "release", "123_1"], ["scontrol", "release", "123_2"]]
    assert saves == [[1], [1, 2]]
    campaign.release_available(manifest, "123", released, lambda: saves.append(sorted(released)))
    assert len(calls) == 2
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw:
                        "JobState=RUNNING Reason=None Priority=100 Dependency=(null) ")
    campaign.release_available(manifest, "123", set(), lambda: None)
    assert len(calls) == 2
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw:
                        "JobState=PENDING Reason=JobHeldAdmin Priority=0 Dependency=(null) ")
    with pytest.raises(RuntimeError, match="not releasable"):
        campaign.release_available(manifest, "123", set(), lambda: None)


def test_release_controller_handles_final_save_race(tmp_path, monkeypatch):
    partial = {"checkpoints": [{"step": 50000}, {"step": 25000, "path": "old"}]}
    complete = {"checkpoints": [{"step": 50000, "path": "new"}, {"step": 25000, "path": "old"}]}
    snapshots = iter([partial, complete])
    monkeypatch.setattr(campaign, "refresh_inventory", lambda path: next(snapshots))
    def release_ready(manifest, job, released, save):
        released.update(i for i, cp in enumerate(manifest["checkpoints"]) if "path" in cp)
        save()
    monkeypatch.setattr(campaign, "release_available", release_ready)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "")
    monkeypatch.setattr(campaign.signal, "signal", lambda *a: None)
    campaign.release(SimpleNamespace(campaign=tmp_path, compute_job="123", training_job="456", timeout_seconds=60))
    receipt = json.loads((tmp_path / "release-state.json").read_text())
    assert receipt["released"] == [0, 1] and receipt["complete"]


def test_release_controller_on_training_failure_cancels_only_future_user_holds(tmp_path, monkeypatch):
    partial = {"checkpoints": [{"step": 75000}, {"step": 50000}, {"step": 25000, "path": "old"}]}
    monkeypatch.setattr(campaign, "refresh_inventory", lambda path: partial)
    monkeypatch.setattr(campaign, "load_campaign", lambda path: partial)
    def release_ready(manifest, job, released, save):
        released.add(2)
        save()
    monkeypatch.setattr(campaign, "release_available", release_ready)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "")
    monkeypatch.setattr(campaign.signal, "signal", lambda *a: None)
    cancelled = []
    def command(args, **kwargs):
        if args[0] == "scancel":
            cancelled.append(args[1])
        reason = "JobHeldUser" if "123_0" in args else "JobHeldAdmin"
        return SimpleNamespace(returncode=0, stdout=f"JobState=PENDING Reason={reason} ")
    monkeypatch.setattr(campaign.subprocess, "run", command)
    with pytest.raises(RuntimeError, match="Training ended"):
        campaign.release(SimpleNamespace(campaign=tmp_path, compute_job="123", training_job="456", timeout_seconds=60))
    assert cancelled == ["123_0"]
    receipt = json.loads((tmp_path / "release-state.json").read_text())
    assert receipt["released"] == [2] and not receipt["complete"]


def test_boom_prior_preflight_uses_same_frozen_context_as_model_load():
    saved = json.loads((ROOT / "configs/dmcontrol/algs/ambi_boom_likelihood_outer_adapt_calibrated.json").read_text())
    matrix_path = ROOT / "configs/research/ambi_checkpoint_graph_eval.json"
    context = SimpleNamespace(source=Path("metadata.json"), trial_run_params=saved,
        experiment_params={"env_params": {"task": "humanoid-walk", "obs": "state", "render_mode": None}})
    matrix = load_preset_matrix(matrix_path)
    checkpoint = {"metadata": {"trial_run_params": {"resolved_runtime": {"observation": {
        "shape": [67], "mode": "state", "action_dim": 21, "episode_length": 500}}}}}
    for selector in ("controller/prior", "controller/graph"):
        resolved = resolve_preset(matrix_path, selector, matrix=matrix, checkpoint_context=context)
        cfg = resolved_checkpoint_config(checkpoint, resolved)
        assert cfg["outer_actor_objective"] == "boom"
        assert cfg["outer_boom_alignment"] == "replay_likelihood"
        assert cfg["outer_boom_coef"] == 0.000042
        assert cfg["inner_operator"] == ("none" if selector.endswith("prior") else "sac")


def test_evaluation_sizing_uses_smallest_cpu_within_five_percent():
    def row(cpus, seconds):
        return {"cpus": cpus, "seconds": 120, "results": [
            {"episodes": [{"control_seconds": 32 * seconds, "length": 32}]}]}
    plan = evaluation_plan([row(1, .15), row(2, .147)])
    assert plan["cpus"] == 1
    assert plan["wall_minutes"] == 15
    assert evaluation_plan([row(1, .20), row(2, .15)])["cpus"] == 2
