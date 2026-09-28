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


@pytest.mark.parametrize("checkpoint_case", [{"aux_return_mode": "xqc"}], indirect=True)
@pytest.mark.parametrize("route", ["return_return", "soft_soft"])
def test_j8_low_actor_lr_changes_only_actor_and_tied_temperature_rates(checkpoint_case, route):
    from utils.ambi_research import load_preset_matrix, normalize_selectors

    root = Path(__file__).resolve().parents[1] / "configs/research"
    original_path = root / "ambixqc_humanoid_h1_actor_bn_screen.json"
    followup_path = root / "ambixqc_humanoid_h1_j8_low_actor_lr.json"
    original = load_preset_matrix(original_path)
    followup = load_preset_matrix(followup_path)
    assert normalize_selectors(followup) == [
        "controller/return_return_j8", "controller/soft_soft_j8",
    ]
    assert {key: value for key, value in original["evaluation"].items()
            if key != "default_presets"} == {
        key: value for key, value in followup["evaluation"].items()
        if key != "default_presets"
    }
    _, checkpoint = checkpoint_case
    context = load_checkpoint_context(checkpoint)
    selector = f"controller/{route}_j8"
    old = resolve_preset(original_path, selector, checkpoint_context=context)
    new = resolve_preset(followup_path, selector, checkpoint_context=context)
    before = old["algorithm_config"]["alg_params"]
    after = new["algorithm_config"]["alg_params"]
    assert {key for key in set(before) | set(after)
            if before.get(key) != after.get(key)} == {"inner_actor_lr"}
    env = gym.make("Pendulum-v1", max_episode_steps=3)
    try:
        configs = [data.resolved_checkpoint_config(
            {"metadata": context.metadata}, resolved, env=env,
        ) for resolved in (old, new)]
    finally:
        env.close()
    before, after = configs
    assert {key for key in set(before) | set(after)
            if before.get(key) != after.get(key)} == {
        "inner_actor_lr", "inner_temperature_lr",
    }
    assert before["inner_actor_lr"] == before["inner_temperature_lr"] == 5e-5
    assert after["inner_actor_lr"] == after["inner_temperature_lr"] == 6.25e-6
    assert after["inner_critic_lr"] == 5e-5
    assert (after["inner_model_step_budget"], after["inner_critic_updates_per_action"],
            after["inner_actor_updates_per_action"], after["inner_temperature_updates_per_action"]) == (2048, 24, 8, 8)
    planners = [data.planner_identity(config, {}, "AMBIXQC/AMBIXQC", "tanh_mean")
                for config in configs]
    assert planners[0] != planners[1]
    assert planners[1]["settings"]["inner_temperature_lr"] == 6.25e-6
