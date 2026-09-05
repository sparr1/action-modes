"""Checkpoint-derived presets keep the learned model and environment fixed."""

import json
import warnings
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2
from utils.ambi_research import (
    PresetMatrixError,
    list_preset_selectors,
    load_preset_matrix,
    materialize_presets,
    normalize_selectors,
    resolve_preset,
)
from utils.checkpoint_context import CheckpointContextError, load_checkpoint_context


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_humanoid_inner_benchmark.json"


@pytest.fixture
def checkpoint_context(tmp_path):
    run = json.loads(
        (ROOT / "configs/dmcontrol/algs/ambi_humanoid_walk_base_v2_d512_2.json").read_text()
    )
    # A no-inner source can retain zeroed legacy or shared schedule controls.
    run["alg_params"].update(
        inner_operator="none", inner_rounds=None, inner_rollouts_per_round=None,
        inner_updates_per_round=0, inner_critic_updates_per_action=0,
        inner_actor_updates_per_action=0, inner_temperature_updates_per_action=0,
        inner_temperature_mode="inherit_outer",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"configuration-only fixture")
    metadata = {
        "schema_version": 1,
        "checkpoint": {
            "kind": "step", "step": 500000, "episode": 1000,
            "best_score": None, "best_window": 100,
        },
        "trial_run_params": run,
        "experiment_params": {
            "env_params": {"task": "humanoid-walk", "obs": "state", "render_mode": None},
        },
    }
    Path(f"{checkpoint}.metadata.json").write_text(json.dumps(metadata))
    return load_checkpoint_context(checkpoint)


def _build_cfg(run):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {**run, "device": "cpu"}
    algorithm.custom_params = run["alg_params"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return algorithm._build_cfg({**run["alg_params"], "device": "cpu"})
    finally:
        algorithm.env.close()


def test_starter_budgets_resolve_from_saved_model_and_preserve_source(checkpoint_context):
    matrix = load_preset_matrix(MATRIX)
    before_matrix = deepcopy(matrix)
    before_context = deepcopy(checkpoint_context)
    assert normalize_selectors(matrix) == ["inner_budget/prior"]
    assert matrix["source_run"] == "rwgao_b-brown-university/ambi/u13m14st"
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["max_steps"] == 500
    expected = {"prior": (0, 0, 0), "sac_1x": (24, 8, 8),
                "sac_2x": (48, 16, 16), "sac_4x": (96, 32, 32)}
    for selector in list_preset_selectors(matrix, comparisons=["inner_budget"]):
        resolved = resolve_preset(MATRIX, selector, matrix, checkpoint_context=checkpoint_context)
        run = resolved["algorithm_config"]
        params = run["alg_params"]
        cfg = _build_cfg(run)
        assert resolved["environment"] == {
            "id": "DMControl-v0", "params": checkpoint_context.experiment_params["env_params"],
        }
        for key, value in checkpoint_context.trial_run_params["alg_params"].items():
            if not key.startswith("inner_"):
                assert params[key] == value
        for key in ("inner_actor_lr", "inner_critic_lr", "inner_temperature_lr"):
            assert params[key] == checkpoint_context.trial_run_params["alg_params"][key]
        for component in ("actor", "critic", "temperature", "replay",
                          "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
            assert getattr(cfg, f"inner_{component}_scope") == "action"
        assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
                cfg.inner_temperature_updates_per_action) == expected[resolved["variant"]]
        if resolved["variant"] != "prior":
            assert cfg.inner_rounds == 8
            assert cfg.inner_rollouts_per_round == 512
            assert cfg.inner_rollout_horizon == 3
            assert cfg.inner_model_step_budget == 12288
            assert cfg.inner_batch_size == 512
            assert cfg.inner_temperature_mode == "auto"
            assert cfg.inner_temperature_initialization == "inherit_outer"
            assert "inner_updates_per_round" not in params
        else:
            assert cfg.inner_model_step_budget == 0
    assert matrix == before_matrix
    assert checkpoint_context == before_context


def test_named_run_preserves_joint_schedule_and_all_source_inner_settings(checkpoint_context):
    source = json.loads((ROOT / "configs/dmcontrol/algs/ambi_humanoid_walk_base_v2_d512_4_j6.json").read_text())
    checkpoint_context.trial_run_params["alg_params"].update(
        inner_actor_lr=0.9, inner_critic_lr=0.8, inner_temperature_lr=0.7,
    )
    resolved = resolve_preset(MATRIX, "named_run/d512_4_j6", checkpoint_context=checkpoint_context)
    params = resolved["algorithm_config"]["alg_params"]
    for key, value in source["alg_params"].items():
        if key.startswith("inner_"):
            assert params[key] == value
    cfg = _build_cfg(resolved["algorithm_config"])
    assert not cfg.inner_component_update_schedule
    assert cfg.inner_updates_per_round == 3
    assert cfg.inner_rounds == 6
    assert cfg.inner_model_step_budget == cfg.inner_replay_capacity == 9216
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
            cfg.inner_temperature_updates_per_action) == (18, 18, 18)
    assert cfg.inner_finite_horizon is False
    assert cfg.inner_steps_per_update is None
    assert cfg.inner_outer_replay_fraction == 0.0


def test_fixed_outer_q_variant_changes_only_bootstrap_critic(checkpoint_context):
    base = resolve_preset(MATRIX, "named_run/d512_4_j6", checkpoint_context=checkpoint_context)
    fixed = resolve_preset(MATRIX, "named_run/d512_4_j6_outer_target", checkpoint_context=checkpoint_context)
    expected = deepcopy(base["algorithm_config"])
    expected["alg_params"]["inner_bootstrap_source"] = "outer_target"
    assert fixed["algorithm_config"] == expected
    cfg = _build_cfg(fixed["algorithm_config"])
    assert cfg.inner_bootstrap_source == "outer_target"
    assert cfg.inner_actor_adaptation == cfg.inner_critic_adaptation == "clone"
    assert cfg.inner_updates_per_round == 3
    assert cfg.inner_finite_horizon is False
    assert cfg.inner_temperature_mode == "auto"


@pytest.mark.parametrize("key", ["model_size", "obs", "num_q", "outer_critic_target", "actor_lr"])
def test_checkpoint_matrix_rejects_outer_or_architecture_overrides(checkpoint_context, key):
    matrix = load_preset_matrix(MATRIX)
    matrix["shared_alg_params"][key] = None
    with pytest.raises(PresetMatrixError, match="incompatible overrides"):
        resolve_preset(MATRIX, "inner_budget/prior", matrix, checkpoint_context=checkpoint_context)


@pytest.mark.parametrize("key", ["alg", "env", "seed", "total_steps"])
def test_checkpoint_matrix_rejects_non_runtime_run_overrides(checkpoint_context, key):
    matrix = load_preset_matrix(MATRIX)
    matrix["comparisons"]["inner_budget"]["variants"]["prior"]["run_params"] = {key: "changed"}
    with pytest.raises(PresetMatrixError, match="incompatible overrides"):
        resolve_preset(MATRIX, "inner_budget/prior", matrix, checkpoint_context=checkpoint_context)


def test_checkpoint_environment_is_assertion_not_override(checkpoint_context):
    matrix = load_preset_matrix(MATRIX)
    matrix["environment"] = {
        "id": "DMControl-v0", "params": deepcopy(checkpoint_context.experiment_params["env_params"]),
    }
    resolve_preset(MATRIX, "inner_budget/prior", matrix, checkpoint_context=checkpoint_context)
    matrix["environment"]["params"]["task"] = "walker-walk"
    with pytest.raises(PresetMatrixError, match="saved environment"):
        resolve_preset(MATRIX, "inner_budget/prior", matrix, checkpoint_context=checkpoint_context)


def test_checkpoint_runtime_overrides_and_variant_precedence(checkpoint_context):
    matrix = load_preset_matrix(MATRIX)
    matrix["shared_alg_params"].update(compile=False, wandb=False, inner_actor_lr=0.01)
    variant = matrix["comparisons"]["inner_budget"]["variants"]["sac_1x"]
    variant["alg_params"]["inner_actor_lr"] = 0.02
    variant["run_params"] = {"device": "cpu"}
    result = resolve_preset(MATRIX, "inner_budget/sac_1x", matrix, checkpoint_context=checkpoint_context)
    assert result["algorithm_config"]["device"] == "cpu"
    assert result["algorithm_config"]["alg_params"]["compile"] is False
    assert result["algorithm_config"]["alg_params"]["wandb"] is False
    assert result["algorithm_config"]["alg_params"]["inner_actor_lr"] == 0.02


def test_checkpoint_materialization_requires_context_before_writing(tmp_path, checkpoint_context):
    output = tmp_path / "generated"
    with pytest.raises(PresetMatrixError, match="requires a checkpoint metadata context"):
        materialize_presets(MATRIX, output)
    assert not output.exists()
    paths = materialize_presets(MATRIX, output, checkpoint_context=checkpoint_context)
    assert [path.name for path in paths] == ["inner_budget__prior.json"]
    experiment = json.loads((output / "AMBIResearchExperiment.json").read_text())
    assert experiment["env_params"] == checkpoint_context.experiment_params["env_params"]


def test_sidecar_reader_is_shared_with_renderer_and_has_no_legacy_fallback(tmp_path, checkpoint_context):
    import render_checkpoint as renderer

    checkpoint = Path(str(checkpoint_context.source).removesuffix(".metadata.json"))
    assert renderer.resolve_render_context(checkpoint) == checkpoint_context
    assert isinstance(checkpoint_context, renderer.RenderContext)
    explicit = load_checkpoint_context(tmp_path / "copied.pt", checkpoint_context.source)
    assert explicit == checkpoint_context
    missing = tmp_path / "missing.pt"
    (tmp_path / "settings.json").write_text("{}")
    with pytest.raises(CheckpointContextError, match="checkpoint metadata does not exist"):
        load_checkpoint_context(missing)
