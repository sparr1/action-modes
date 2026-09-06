"""Checkpoint matrices derive the learned problem and limit evaluation overrides."""

import copy
import json
from pathlib import Path

import pytest

from utils.ambi_research import PresetMatrixError, materialize_presets, resolve_preset
from utils.checkpoint_context import CheckpointContext, CheckpointContextError, load_checkpoint_context


@pytest.fixture
def case(tmp_path):
    context = CheckpointContext(
        trial_run_params={
            "alg": "AMBIXQC/AMBIXQC", "env": "Pendulum-v1", "seed": 55,
            "total_steps": 1_500_000,
            "alg_params": {"inner_operator": "none", "inner_rounds": 2,
                           "inner_rollouts_per_round": 32, "inner_rollout_horizon": 3,
                           "inner_updates_per_round": 4, "inner_batch_size": 64,
                           "inner_reward_normalization": "frozen_real_scale",
                           "xqc_actor_net_arch": [256] * 4, "train_unroll_horizon": 3},
        },
        experiment_params={"env_params": {"max_episode_steps": 500}},
        source=tmp_path / "prior.pt.metadata.json",
    )
    matrix = {
        "schema_version": 1, "base_alg_config": "checkpoint",
        "evaluation": {"seeds": [101], "default_presets": ["controller/prior"]},
        "comparisons": {"controller": {"reference": "prior", "variants": {
            "prior": {"alg_params": {"inner_operator": "none"}},
            "xqc": {"alg_params": {"inner_operator": "xqc"}},
        }}},
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    return path, matrix, context


def test_checkpoint_presets_keep_source_settings_and_dormant_budget(case):
    path, matrix, context = case
    original = copy.deepcopy(context)
    resolved = resolve_preset(path, "controller/xqc", checkpoint_context=context)
    params = resolved["algorithm_config"]["alg_params"]
    expected = copy.deepcopy(context.trial_run_params)
    expected["alg_params"]["inner_operator"] = "xqc"
    assert resolved["algorithm_config"] == expected
    assert params["inner_updates_per_round"] == 4
    assert resolved["saved_algorithm_config"] == context.trial_run_params
    assert context == original
    assert resolved["environment"] == {"id": "Pendulum-v1", "params": {"max_episode_steps": 500}}


@pytest.mark.parametrize("key,value", [("xqc_policy_delay", 9), ("train_unroll_horizon", 9),
                                     ("discount", 0.8), ("latent_dim", 16)])
def test_outer_semantics_cannot_be_overridden(case, key, value):
    path, matrix, context = case
    matrix["comparisons"]["controller"]["variants"]["xqc"]["alg_params"][key] = value
    path.write_text(json.dumps(matrix))
    with pytest.raises(PresetMatrixError, match="incompatible overrides"):
        resolve_preset(path, "controller/xqc", checkpoint_context=context)


@pytest.mark.parametrize("key,value", [("inner_actor_scope", "episode"),
                                     ("inner_diagnostic_rollouts", 8),
                                     ("inner_bootstrap_source", "outer_target")])
def test_unsupported_xqc_inner_options_fail_during_resolution(case, key, value):
    path, matrix, context = case
    matrix["comparisons"]["controller"]["variants"]["xqc"]["alg_params"][key] = value
    path.write_text(json.dumps(matrix))
    with pytest.raises(PresetMatrixError, match="unsupported inner controls/probes"):
        resolve_preset(path, "controller/xqc", checkpoint_context=context)


def test_environment_and_wrappers_are_frozen(case):
    path, matrix, context = case
    matrix["environment"] = {"id": "Pendulum-v1", "params": {"max_episode_steps": 10}}
    path.write_text(json.dumps(matrix))
    with pytest.raises(PresetMatrixError, match="saved environment"):
        resolve_preset(path, "controller/xqc", checkpoint_context=context)
    matrix.pop("environment")
    matrix["comparisons"]["controller"]["variants"]["xqc"]["run_params"] = {"env_wrappers": []}
    path.write_text(json.dumps(matrix))
    with pytest.raises(PresetMatrixError, match="incompatible overrides"):
        resolve_preset(path, "controller/xqc", checkpoint_context=context)


def test_materialization_requires_context_before_creating_output(case, tmp_path):
    path, _, context = case
    output = tmp_path / "missing"
    with pytest.raises(PresetMatrixError, match="metadata context"):
        materialize_presets(path, output)
    assert not output.exists()
    files = materialize_presets(path, output, comparisons=["controller"], checkpoint_context=context)
    assert len(files) == 2
    experiment = json.loads((output / "AMBIResearchExperiment.json").read_text())
    assert experiment["env_params"] == context.experiment_params["env_params"]


def test_legacy_ambi_checkpoint_context_still_resolves(case):
    path, _, context = case
    context.trial_run_params["alg"] = "AMBITDMPC2/AMBITDMPC2"
    resolved = resolve_preset(path, "controller/prior", checkpoint_context=context)
    assert resolved["algorithm_config"]["alg"] == "AMBITDMPC2/AMBITDMPC2"


def test_missing_or_malformed_metadata_does_not_fall_back(tmp_path):
    checkpoint = tmp_path / "prior.pt"
    with pytest.raises(CheckpointContextError, match="does not exist"):
        load_checkpoint_context(checkpoint)
    Path(f"{checkpoint}.metadata.json").write_text('{"schema_version": 999}')
    with pytest.raises(CheckpointContextError, match="schema_version"):
        load_checkpoint_context(checkpoint)
