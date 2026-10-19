"""Frozen early-checkpoint recipe, execution semantics and probe isolation."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params, _tiny_params
from utils.ambi_benchmark import BenchmarkBundle, frozen_execution_action, protocol_for
from utils.ambi_research import PresetMatrixError, load_preset_matrix, resolve_preset


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_early_h3_j8.json"
SOURCE_CONFIGS = ("ambi_full_h3_j8_fresh", "ambi_full_h3_mppi_aux_return",
                  "ambi_aux_return_sac_clip_target10p5_shared")


def _resolve(config, mode="mean"):
    context = SimpleNamespace(
        source="checkpoint.metadata.json", trial_run_params=config,
        experiment_params={"env_params": {"domain_name": "humanoid", "task_name": "walk"}},
    )
    return resolve_preset(MATRIX, f"early_h3_j8/{mode}", checkpoint_context=context)


@pytest.mark.parametrize("source_name", SOURCE_CONFIGS)
def test_production_recipe_preserves_each_source_outer_state_contract(source_name):
    source = json.loads((ROOT / f"configs/dmcontrol/algs/{source_name}.json").read_text())
    before = deepcopy(source)
    production = json.loads((ROOT / "configs/dmcontrol/algs/ambi_full_h3_j8_fresh.json").read_text())
    expected = _build_cfg(**production["alg_params"])
    for mode in ("mean", "sample"):
        resolved = _resolve(source, mode)
        params = resolved["algorithm_config"]["alg_params"]
        actual = _build_cfg(**params)
        for key in production["alg_params"]:
            if key.startswith("inner_"):
                assert getattr(actual, key) == getattr(expected, key), key
        for key, value in source["alg_params"].items():
            if not key.startswith("inner_") and key not in {"wandb", "outer_policy_diagnostics"}:
                assert params[key] == value, key
        assert actual.inner_operator == "sac"
        assert actual.outer_policy_diagnostics is False
        assert (actual.inner_rounds, actual.inner_rollout_horizon) == (8, 3)
        assert (actual.inner_critic_updates_per_round, actual.inner_actor_updates_per_round) == (16, 4)
        assert actual.inner_model_step_budget == 3072
        assert actual.inner_critic_bernoulli_p == 0.0
        assert actual.inner_sac_critic_target == "reward_only"
        assert actual.inner_critic_source == actual.inner_horizon_critic_source == "aux_return"
        assert actual.inner_temperature_mode == "auto"
        assert actual.inner_temperature_initialization == "inherit_outer"
        assert not any(key.startswith("inner_mppi_") for key in params)
        assert "inner_model_step_budget" not in params
        assert frozen_execution_action(resolved) == mode
        assert resolved["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert source == before


def test_checkpoint_preset_can_only_disable_training_collector_recorder():
    from utils.ambi_research import validate_preset_matrix
    matrix = load_preset_matrix(MATRIX)
    assert matrix["shared_alg_params"]["outer_policy_diagnostics"] is False
    matrix["shared_alg_params"]["outer_policy_diagnostics"] = True
    with pytest.raises(PresetMatrixError, match="incompatible overrides"):
        validate_preset_matrix(matrix)


def test_execution_identity_is_explicit_and_old_matrices_remain_mean():
    source = json.loads((ROOT / "configs/dmcontrol/algs/ambi_full_h3_j8_fresh.json").read_text())
    mean, sample = _resolve(source), _resolve(source, "sample")
    assert mean["algorithm_config"] == sample["algorithm_config"]
    assert protocol_for(mean, 55, 500)["action_rule"] == "tanh_mean"
    assert protocol_for(sample, 55, 500)["action_rule"] == "tanh_sample"
    historical = deepcopy(sample)
    historical.pop("evaluation")
    assert protocol_for(historical, 55, 500) == protocol_for(mean, 55, 500)
    for key, value in (("inner_execution_action", "mean"), ("inner_execution_std_scale", 0.0),
                       ("inner_operator", "mppi")):
        invalid = deepcopy(sample)
        invalid["algorithm_config"]["alg_params"][key] = value
        with pytest.raises(ValueError, match="Sampled frozen evaluation requires"):
            frozen_execution_action(invalid)


def test_variant_execution_does_not_allow_unpaired_seed_or_budget_overrides():
    matrix = load_preset_matrix(MATRIX)
    matrix["comparisons"]["early_h3_j8"]["variants"]["sample"]["evaluation"]["seeds"] = [1]
    from utils.ambi_research import validate_preset_matrix
    with pytest.raises(PresetMatrixError, match="episode seeds and budgets must remain paired"):
        validate_preset_matrix(matrix)


def _tiny_checkpoint(tmp_path, source_operator):
    params = _tiny_params(aux_return_mode="sac", aux_return_detach_representation=False,
                          train_unroll_horizon=3,
                          ent_coef="auto_0.37", inner_operator=source_operator,
                          inner_actor_source="sac", inner_horizon_actor_source="sac",
                          inner_critic_source="aux_return", inner_horizon_critic_source="aux_return")
    if source_operator == "none":
        params["outer_policy_diagnostics"] = True
        params["mpc"] = False
        for key in ("inner_rounds", "inner_rollouts_per_round", "inner_updates_per_round"):
            params.pop(key, None)
    if source_operator == "mppi":
        for key in ("inner_rounds", "inner_rollouts_per_round", "inner_updates_per_round"):
            params.pop(key)
        params.update(inner_mppi_iterations=1, inner_model_step_budget=8)
    source = _model_from_params(params)
    path = tmp_path / f"{source_operator}.pt"
    torch.save(source.agent.checkpoint_state(), path)
    source.close()
    source.env.close()
    config = {"alg": "AMBITDMPC2/AMBITDMPC2", "env": "Pendulum-v1", "device": "cpu",
              "seed": 55, "total_steps": 10, "alg_params": params}
    resolved = _resolve(config)
    # A small real network/solve checks loading and execution. The full
    # production schedule is checked above without spending a benchmark budget.
    resolved["algorithm_config"]["alg_params"].update(
        inner_rounds=2, inner_rollouts_per_round=2, inner_batch_size=4,
        inner_critic_updates_per_round=2, inner_actor_updates_per_round=1,
        inner_replay_capacity=12, compile=False, compile_strict=False,
    )
    resolved["algorithm_config"]["alg_params"].pop("inner_updates_per_round", None)
    return path, resolved


def _evaluate(monkeypatch, path, resolved, *, bundle_dir=None, rollouts=0, cadence=1, steps=3):
    actions = []
    def make_env(_resolved):
        env = gym.make("Pendulum-v1", max_episode_steps=steps)
        original = env.step
        def step(action):
            actions.append(np.asarray(action).copy())
            return original(action)
        env.step = step
        return env
    monkeypatch.setattr(evaluator, "_make_env", make_env)
    bundle = None
    run = None
    if bundle_dir is not None:
        bundle = BenchmarkBundle(bundle_dir, checkpoint={"sha256": "test"},
                                 protocol=protocol_for(resolved, 55, steps))
        run = bundle.start_run(resolved, "episodes")
    result = evaluator.evaluate_preset(
        resolved, path, [101], controller_seed=55, max_steps=steps, device="cpu",
        bundle=bundle, bundle_run=run, togo_return_rollouts=rollouts,
        togo_return_every=cadence,
    )
    return result, actions, bundle, run


@pytest.mark.parametrize("source_operator", ["sac", "mppi", "none"])
def test_real_portable_checkpoint_execution_samples_and_keeps_outer_frozen(
        tmp_path, monkeypatch, source_operator):
    path, mean = _tiny_checkpoint(tmp_path, source_operator)
    sample = deepcopy(mean)
    sample["evaluation"]["execution_action"] = "sample"
    mean_result, mean_actions, _, _ = _evaluate(monkeypatch, path, mean, steps=1)
    sample_result, sample_actions, _, _ = _evaluate(monkeypatch, path, sample, steps=1)
    repeated, repeated_actions, _, _ = _evaluate(monkeypatch, path, sample, steps=1)
    assert not np.array_equal(mean_actions[0], sample_actions[0])
    np.testing.assert_array_equal(sample_actions, repeated_actions)
    assert sample_result["return"] == repeated["return"]
    for result in (mean_result, sample_result):
        assert result["outer_state_unchanged"]
        assert result["outer_updates_before"] == result["outer_updates_after"] == 0
        assert result["resolved_config"]["inner_critic_source"] == "aux_return"
        assert result["model_metrics"]["inner_critic_optimizer_steps"]["mean"] == 4
    assert mean_result["execution_action"] == "mean"
    assert sample_result["execution_action"] == "sample"


def test_sparse_togo_probes_leave_sampled_actions_identical_and_record_rounds(tmp_path, monkeypatch):
    path, resolved = _tiny_checkpoint(tmp_path, "mppi")
    resolved["evaluation"]["execution_action"] = "sample"
    base, base_actions, _, _ = _evaluate(monkeypatch, path, resolved, bundle_dir=tmp_path / "plain")
    result, actions, bundle, run = _evaluate(
        monkeypatch, path, resolved, bundle_dir=tmp_path / "probed", rollouts=4, cadence=2,
    )
    np.testing.assert_array_equal(actions, base_actions)
    assert result["return"] == base["return"]
    assert result["outer_state_unchanged"]
    assert result["togo_return_probe"]["decision_cadence"] == 2
    rows = [json.loads(line) for relative in run["trace_files"]
            for line in gzip.decompress((bundle.path / relative).read_bytes()).decode().splitlines()]
    probes = [row for row in rows if row["phase"] == "probe"]
    assert {row["decision_index"] for row in probes} == {0, 2}
    assert {row["round_index"] for row in probes} == {0, 1, 2}
    assert len(probes) == 6
    assert len([row for row in rows if row["phase"] == "decision"]) == 3
    assert result["episodes"][0]["togo_probe_model_steps"] > 0
    assert result["episodes"][0]["togo_probe_seconds"] > 0
    assert all(row["metrics"]["togo_return_mean"] is not None for row in probes)


@pytest.mark.parametrize("cadence", [0, -1, True, 1.5])
def test_invalid_probe_cadence_fails_before_environment_creation(tmp_path, monkeypatch, cadence):
    path = tmp_path / "placeholder.pt"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(evaluator, "_make_env", lambda _: pytest.fail("environment constructed"))
    with pytest.raises(ValueError, match="togo_return_every"):
        evaluator.evaluate_preset({}, path, [101], controller_seed=55, togo_return_every=cadence)
