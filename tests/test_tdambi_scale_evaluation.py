"""Checkpoint-scale provenance and frozen TDAMBI evaluation identity."""

import copy
import json
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from evaluate_ambi_checkpoint import _outer_state_digest, evaluate_matrix
from RL.TDAMBI import TDAMBI
from RL.TDMPC2 import TDMPC2Baseline
from tests.test_ambi_inner_decoupling import _fixed_reward_model
from tests.test_tdambi_checkpoint import MATRIX, _native_context, make_tdambi, tiny_native_params
from tests.test_tdambi_publication import _config
from utils.ambi_research import load_preset_matrix, normalize_selectors, resolve_preset
from utils.eval_series_data import planner_identity, resolved_checkpoint_config


def _planner(params):
    return planner_identity(params, {}, "TDAMBI/TDAMBI", "tanh_mean")


def test_scale_initialization_policy_identifies_curves_without_learned_values():
    legacy = _planner(_config())
    calibrated = _planner({**_config(), "tdambi_scale_initialization": "calibrate"})
    assert calibrated == legacy
    identities = [legacy]
    for policy in ("checkpoint_or_calibrate", "checkpoint"):
        params = {**_config(), "tdambi_scale_initialization": policy}
        identity = _planner(params)
        assert identity["settings"]["tdambi_scale_initialization"] == policy
        assert identity not in identities
        identities.append(identity)
        params.update(tdambi_checkpoint_scale=17.0, tdambi_scale_source="checkpoint")
        assert _planner(params) == identity
        params.update(tdambi_checkpoint_scale=91.0, tdambi_scale_source="first_collection_calibration")
        assert _planner(params) == identity


def test_scale_comparison_preserves_default_and_resolves_matching_budget(tmp_path):
    context = _native_context(tmp_path)
    matrix = load_preset_matrix(MATRIX)
    assert normalize_selectors(matrix) == ["inner_budget/tdambi_3"]
    configs = {}
    for selector in ("inner_budget/tdambi_3", "q_scale/checkpoint", "q_scale/calibrate"):
        resolved = resolve_preset(MATRIX, selector, matrix, checkpoint_context=context)
        params = resolved["algorithm_config"]["alg_params"]
        algorithm = object.__new__(TDAMBI)
        algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
        algorithm.run_params = {"device": "cpu"}
        try:
            cfg = algorithm._build_cfg(params)
            preflight = resolved_checkpoint_config({}, resolved, env=algorithm.env)
        finally:
            algorithm.env.close()
        assert _planner(preflight) == _planner(vars(cfg))
        configs[selector] = copy.deepcopy(params)
    assert configs["inner_budget/tdambi_3"]["tdambi_scale_initialization"] == "checkpoint_or_calibrate"
    for mode in ("checkpoint", "calibrate"):
        selected = configs[f"q_scale/{mode}"]
        assert selected.pop("tdambi_scale_initialization") == mode
    configs["inner_budget/tdambi_3"].pop("tdambi_scale_initialization")
    assert configs["q_scale/checkpoint"] == configs["q_scale/calibrate"] == configs["inner_budget/tdambi_3"]


def test_frozen_digest_includes_native_saved_scale():
    model = make_tdambi()
    try:
        model.agent.tdambi_checkpoint_scale = torch.tensor([17.0])
        before = _outer_state_digest(model)
        model.agent.tdambi_checkpoint_scale.add_(1)
        assert _outer_state_digest(model) != before
    finally:
        model.close()
        model.env.close()


def test_frozen_digest_includes_ambi_outer_scale():
    model = _fixed_reward_model(sac_actor_loss_scale_mode="tdmpc2_percentile_range")
    try:
        before = _outer_state_digest(model)
        model.agent.actor_loss_scale.add_(1)
        assert _outer_state_digest(model) != before
    finally:
        model.close()
        model.env.close()


@pytest.mark.parametrize(
    "policy,saved,source",
    [
        ("checkpoint", True, "checkpoint"),
        ("checkpoint_or_calibrate", False, "first_collection_calibration"),
        ("calibrate", True, "first_collection_calibration"),
    ],
)
def test_evaluator_records_actual_scale_source_without_changing_outer_state(
    tmp_path, monkeypatch, policy, saved, source,
):
    monkeypatch.setattr("utils.ambi_benchmark.code_identity",
                        lambda: {"commit": "test", "dirty": False})
    monkeypatch.setattr("evaluate_ambi_checkpoint._make_env", lambda resolved: gym.make(
        resolved["environment"]["id"], **resolved["environment"]["params"]
    ))
    env = gym.make("Pendulum-v1", max_episode_steps=2)
    native = TDMPC2Baseline(
        "TDMPC2", env, tiny_native_params(),
        {"seed": 55, "device": "cpu", "env": "Pendulum-v1", "total_steps": 10}, {},
    )
    checkpoint = tmp_path / "native.pt"
    payload = native.agent.checkpoint_state()
    payload.pop("scale", None)
    if saved:
        payload["scale"] = {"value": torch.tensor([17.0]), "percentiles": torch.tensor([5.0, 95.0])}
    torch.save(payload, checkpoint)
    native.close()
    env.close()
    metadata = {
        "schema_version": 1,
        "checkpoint": {"kind": "step", "step": 25000, "episode": 5,
                       "best_score": None, "best_window": 100},
        "trial_run_params": {"alg": "TDMPC2/TDMPC2Baseline", "env": "Pendulum-v1",
                             "seed": 55, "alg_params": tiny_native_params()},
        "experiment_params": {"env_params": {"max_episode_steps": 2}},
    }
    Path(f"{checkpoint}.metadata.json").write_text(json.dumps(metadata))
    matrix = load_preset_matrix(MATRIX)
    matrix["shared_alg_params"].update(
        inner_rounds=1, inner_rollouts_per_round=4, inner_rollout_horizon=2,
        inner_batch_size=4, inner_replay_capacity=16, tdambi_scale_initialization=policy,
    )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    bundle = tmp_path / "bundle"
    result = evaluate_matrix(
        matrix_path, checkpoint, seeds=[101], max_steps=2, bundle_dir=bundle,
    )["results"][0]
    assert result["outer_state_unchanged"] is True
    expected = {"initialization": policy, "source": source,
                "initial_value": 17.0 if source == "checkpoint" else None}
    assert result["q_scale"] == expected
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["runs"][0]["q_scale"] == expected
    assert result["resolved_config"]["tdambi_scale_initialization"] == policy
    metrics = result["episodes"][0]["model_metrics"]
    assert metrics["inner_tdambi_scale_from_checkpoint"] == float(source == "checkpoint")
    if source == "checkpoint":
        assert metrics["inner_tdambi_q_scale_initial"] == 17.0
        assert metrics["inner_tdambi_calibration_samples"] == 0.0
    else:
        assert metrics["inner_tdambi_calibration_samples"] == 4.0
