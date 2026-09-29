"""Actor BN prelaunch identities agree with frozen episode publication records."""
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import pytest

import evaluate_ambi_checkpoint as evaluator
from test_ambixqc_checkpoint_evaluation import checkpoint_case
from utils import ambi_benchmark as storage
from utils import eval_series_data as data
from utils.ambi_research import PresetMatrixError, resolve_preset
from utils.checkpoint_context import load_checkpoint_context
from RL.AMBIXQC import AMBIXQC


@pytest.mark.parametrize("mode", ["batch_update", "running"])
def test_actor_bn_metadata_spec_matches_executed_identity(checkpoint_case, tmp_path, monkeypatch, mode):
    matrix_path, checkpoint = checkpoint_case
    matrix = json.loads(matrix_path.read_text())
    matrix["source_run"] = "entity/training/xqc"
    matrix["comparisons"]["controller"]["variants"]["xqc"]["alg_params"]["inner_actor_bn_mode"] = mode
    matrix_path.write_text(json.dumps(matrix))
    metadata = Path(str(checkpoint) + ".metadata.json")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"source_run": matrix["source_run"], "checkpoints": [{
        "step": 4, "path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }]}))
    monkeypatch.setattr(data, "scientific_identity", lambda *a, **kw: {"fixture": "science"})
    monkeypatch.setattr(storage, "code_identity", lambda: {"commit": "fixture", "dirty": False})
    # The fixture uses Pendulum. Supply its existing spaces to configuration
    # resolution; production DMControl metadata reconstructs them itself.
    env = gym.make("Pendulum-v1", max_episode_steps=3)
    resolve_config = data.resolved_checkpoint_config
    monkeypatch.setattr(data, "resolved_checkpoint_config", lambda *a, **kw: resolve_config(*a, **{**kw, "env": env}))
    try:
        with monkeypatch.context() as preflight:
            preflight.setattr(evaluator, "_make_env", lambda *a: pytest.fail("specification created an environment"))
            preflight.setattr(evaluator, "_initialize_frozen_model", lambda *a, **kw: pytest.fail("specification loaded a model"))
            prepared = evaluator.evaluate_matrix(
                matrix_path, checkpoint, selectors=["controller/xqc"],
                checkpoint_inventory=inventory, eval_series_spec_dir=tmp_path / "specs")
        spec = json.loads(Path(prepared["specs"]["controller/xqc"]).read_text())
        bundle = tmp_path / "episodes"
        result = evaluator.evaluate_matrix(
            matrix_path, checkpoint, selectors=["controller/xqc"],
            checkpoint_inventory=inventory, bundle_dir=bundle)["results"][0]
        record, = data.load_records(bundle, checkpoint_inventory=inventory)
        assert record["identity"] == spec["identity"]
        assert result["resolved_config"]["inner_actor_bn_mode"] == mode
        assert result["outer_state_unchanged"]
        settings = spec["identity"]["planner"]["settings"]
        assert settings.get("inner_actor_bn_mode", "batch_update") == mode
        assert ("inner_actor_bn_mode" in settings) is (mode == "running")
        run = json.loads((bundle / "manifest.json").read_text())["runs"][0]
        assert f"actor-bn:{mode}" in run["wandb_tags"]
    finally:
        env.close()


@pytest.mark.parametrize("mode", [True, [], "batch", "eval"])
def test_invalid_actor_bn_mode_rejected_by_checkpoint_matrix(mode):
    matrix = {"schema_version": 1, "base_alg_config": "checkpoint",
              "shared_alg_params": {"inner_actor_bn_mode": mode},
              "comparisons": {"controller": {"variants": {"xqc": {"alg_params": {"inner_operator": "xqc"}}}}}}
    from utils.ambi_research import validate_preset_matrix
    with pytest.raises(PresetMatrixError, match="inner_actor_bn_mode"):
        validate_preset_matrix(matrix)


@pytest.mark.parametrize("checkpoint_case", [{"inner_actor_bn_mode": "running"}], indirect=True)
def test_null_preset_resets_inherited_running_mode_to_default(checkpoint_case):
    matrix_path, checkpoint = checkpoint_case
    context = load_checkpoint_context(checkpoint)
    assert context.trial_run_params["alg_params"]["inner_actor_bn_mode"] == "running"
    matrix = json.loads(matrix_path.read_text())
    matrix["comparisons"]["controller"]["variants"]["xqc"]["alg_params"]["inner_actor_bn_mode"] = None
    resolved = resolve_preset(matrix_path, "controller/xqc", matrix, checkpoint_context=context)
    params = resolved["algorithm_config"]["alg_params"]
    assert "inner_actor_bn_mode" not in params
    assert context.trial_run_params["alg_params"]["inner_actor_bn_mode"] == "running"
    env = gym.make("Pendulum-v1", max_episode_steps=3)
    try:
        model = AMBIXQC.__new__(AMBIXQC)
        model.env = env
        model.run_params = resolved["algorithm_config"]
        model.custom_params = params
        assert model._build_cfg(params).inner_actor_bn_mode == "batch_update"
        # Null is supported matrix syntax only, not a direct algorithm value.
        with pytest.raises(ValueError, match="inner_actor_bn_mode"):
            model._build_cfg({**params, "inner_actor_bn_mode": None})
    finally:
        env.close()
