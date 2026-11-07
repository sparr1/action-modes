import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm.boom_likelihood_calibrated_oscar import verify_training_recipe, evaluation_plan
from slurm.graph_checkpoint_campaign import checkpoint_steps, inventory
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
        assert cfg["outer_boom_coef"] == 1e-5
        assert cfg["inner_operator"] == ("none" if selector.endswith("prior") else "sac")


def test_evaluation_sizing_uses_smallest_cpu_within_five_percent():
    def row(cpus, seconds):
        return {"cpus": cpus, "seconds": 120, "results": [
            {"episodes": [{"control_seconds": 32 * seconds, "length": 32}]}]}
    plan = evaluation_plan([row(1, .15), row(2, .147)])
    assert plan["cpus"] == 1
    assert plan["wall_minutes"] == 15
    assert evaluation_plan([row(1, .20), row(2, .15)])["cpus"] == 2
