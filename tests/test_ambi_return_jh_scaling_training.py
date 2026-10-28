"""Bind the J/H/K training screen to its complete J10 reference recipe."""

from copy import deepcopy
import json
import warnings

import pytest

from slurm import ambi_reward_training as campaign
from slurm.benchmark_ambi_training_cpu import action_work
from tests.test_ambi_root_local_sac import _build_cfg
from tests.test_ambi_return_only_j10_training import _receipt

CASES = list(campaign.SCALING_CASES.items())


@pytest.mark.parametrize("path,spec", CASES, ids=lambda value: value.name if hasattr(value, "name") else str(value))
def test_scaling_recipe_changes_only_requested_axes_budget_and_publication(path, spec):
    name, operator, rounds, interval, horizon, decisions = spec
    selected, manifest, run = campaign.validate_campaign(path)
    assert selected == name and operator == "sac" and decisions == 1000000
    raw = campaign.read(campaign.ROOT / "configs/dmcontrol/algs" / (name + ".json"))
    assert raw["total_steps"] == manifest["overrides_alg"]["total_steps"] == decisions
    assert raw["seed"] == run["seed"] == 55
    assert (run["checkpoint_every"], run["save_strat"]) == (25000, "all")
    params = run["alg_params"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = _build_cfg(**params)
    assert (cfg.train_unroll_horizon, cfg.outer_planning_horizon) == (3, 3)
    assert (cfg.inner_rounds, cfg.inner_rollout_horizon, cfg.inner_solve_interval) == (rounds, horizon, interval)
    assert cfg.inner_hold_encoder == "solve_snapshot"
    assert cfg.inner_replay_capacity == cfg.inner_model_step_budget == rounds * 128 * horizon
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
            cfg.inner_temperature_updates_per_action) == (16 * rounds, 4 * rounds, 4 * rounds)
    assert cfg.outer_critic_target == cfg.inner_sac_critic_target == "reward_only"
    assert cfg.aux_return_mode == "off" and cfg.target_entropy == -10.5
    assert any("exceeds train_unroll_horizon=3" in str(row.message) for row in caught) == (horizon > 3)
    assert "500k-decisions" not in params["wandb_tags"]
    assert "1m-environment-steps" not in params["wandb_tags"]


@pytest.mark.parametrize("path,spec", CASES)
def test_smoke_contains_complete_cycles_and_preserves_production_shape(path, spec):
    _, _, production = campaign.validate_campaign(path)
    _, _, smoke, overrides = campaign.smoke_recipe(60, 120, path)
    expected = deepcopy(production["alg_params"])
    expected.update(overrides)
    assert smoke["alg_params"] == expected
    assert smoke["total_steps"] == smoke["checkpoint_every"] == 681
    with pytest.raises(AssertionError, match="cycles"):
        campaign.smoke_recipe(61, 120, path)
    with pytest.raises(AssertionError, match="cycles"):
        campaign.smoke_recipe(60, 121, path)


@pytest.mark.parametrize("key,value", [
    ("inner_rounds", 8), ("inner_solve_interval", 1), ("inner_rollout_horizon", 2),
    ("inner_replay_capacity", 3072), ("inner_critic_updates_per_round", 8),
    ("train_unroll_horizon", 4), ("outer_planning_horizon", 4),
    ("aux_return_mode", "sac"), ("inner_temperature_scope", "run"),
    ("discount", .97), ("inner_execution_action", "policy_mean"),
])
def test_scaling_gate_rejects_unrequested_scientific_changes(monkeypatch, key, value):
    path, _ = CASES[1]
    original = campaign.recipe
    def changed(index, selected):
        result = original(index, selected)
        if selected == path:
            result[2]["alg_params"][key] = value
        return result
    monkeypatch.setattr(campaign, "recipe", changed)
    with pytest.raises(AssertionError):
        campaign.validate_campaign(path)


@pytest.mark.parametrize("path,spec", CASES)
def test_exact_solve_work_and_zero_work_between_solves(path, spec):
    _, operator, rounds, interval, horizon, _ = spec
    metrics = {**campaign.sac_work(rounds, horizon), "inner_compile_fallback": 0.,
               "inner_solve_performed": 1., "inner_policy_held": 0.,
               "inner_action_age": 0., "inner_episode_decision_index": 0.}
    assert campaign.validate_action(metrics, operator=operator, decision=0, interval=interval,
                                    rounds=rounds, horizon=horizon)
    for key in campaign.sac_work(rounds, horizon):
        with pytest.raises(AssertionError):
            action_work({**metrics, key: metrics[key] - 1}, decision=0, interval=interval,
                        probability=0., rounds=rounds, horizon=horizon)
    held = {"inner_solve_performed": 0., "inner_policy_held": 1., "inner_action_age": 1.,
            "inner_episode_decision_index": 1., "inner_compile_fallback": 0., "inner_policy_evaluations": 1.}
    held.update(dict.fromkeys((*campaign.sac_work(rounds, horizon), "inner_updates", "inner_steps",
                "inner_rollouts", "inner_replay_draws", "inner_realized_model_steps", "inner_total_model_steps",
                "inner_q_evaluations"), 0.))
    assert not campaign.validate_action(held, operator=operator, decision=1, interval=interval,
                                        rounds=rounds, horizon=horizon)
    with pytest.raises(AssertionError):
        campaign.validate_action({**held, "inner_model_steps": 1}, operator=operator, decision=1,
                                 interval=interval, rounds=rounds, horizon=horizon)


def scaling_receipt(spec):
    name, operator, rounds, interval, horizon, _ = spec
    receipt = _receipt(rounds)
    for row in receipt["cases"]:
        row.update(config=name, inner_solve_interval=interval, inner_rollout_horizon=horizon,
                   inner_work_per_solve=campaign.sac_work(rounds, horizon), warmup_active_decisions=60,
                   outer_actor_optimizer_steps=182, outer_temperature_optimizer_steps=182)
        row["measured"].update(solve_decisions=120 // interval, held_decisions=120 - 120 // interval)
    return receipt, dict(name=name, operator=operator, interval=interval)


@pytest.mark.parametrize("path,spec", CASES)
def test_receipt_binds_horizon_cadence_and_work_for_both_cpu_counts(path, spec):
    receipt, identity = scaling_receipt(spec)
    def validate(value):
        campaign.validate_receipt(value, receipt["binding"], receipt["runtime"], 1, **identity)
    validate(receipt)
    for cores_index in (0, 1):
        for key in campaign.sac_work(spec[2], spec[4]):
            broken = deepcopy(receipt)
            broken["cases"][cores_index]["inner_work_per_solve"][key] -= 1
            with pytest.raises(AssertionError):
                validate(broken)
        for key, value in (("inner_rounds", 8), ("inner_rollout_horizon", 2),
                           ("inner_solve_interval", 1), ("warmup_active_decisions", 61)):
            broken = deepcopy(receipt)
            broken["cases"][cores_index][key] = value
            with pytest.raises(AssertionError):
                validate(broken)
        broken = deepcopy(receipt)
        del broken["cases"][cores_index]["inner_rollout_horizon"]
        with pytest.raises(KeyError):
            validate(broken)


def test_three_configs_have_distinct_identities_and_preserve_reference_files():
    assert len(CASES) == len({spec[0] for _, spec in CASES}) == 3
    for path in (campaign.MANIFESTS["sac"], campaign.J10_MANIFEST, campaign.MANIFESTS["mppi"]):
        _, _, run = campaign.validate_campaign(path)
        assert run["total_steps"] == 500000


@pytest.mark.parametrize("path,expected", [(campaign.J10_MANIFEST, 12), *[(path, 60) for path, _ in CASES]])
def test_cli_selects_recipe_warmup_without_wrapper_override(monkeypatch, tmp_path, path, expected):
    import sys
    selected = []
    monkeypatch.setattr(sys, "argv", ["campaign", "smoke", "--manifest", str(path), "--output", str(tmp_path)])
    monkeypatch.setattr(campaign, "binding", lambda path: {"commit": "test"})
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(campaign, "smoke", lambda args: selected.append((args.warmup, args.decisions)))
    monkeypatch.setenv("AMBI_EXPECTED_COMMIT", "test")
    campaign.main()
    assert selected == [(expected, 120)]


@pytest.mark.parametrize("override", [None, "60"])
def test_shell_wrapper_defers_warmup_unless_explicit(tmp_path, override):
    import os
    import subprocess
    path = campaign.ROOT / "slurm/run_ambi_reward_training_oscar.sbatch"
    source = path.read_text()
    fragment = source[source.index('  smoke_args='):source.index('\nelse\n', source.index('  smoke_args='))]
    srun = tmp_path / "srun"
    srun.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    srun.chmod(0o755)
    environment = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                   "AMBI_PYTHON": "/locked/python", "AMBI_RETURN_MANIFEST": str(CASES[2][0]),
                   "AMBI_RETURN_OUTPUT": str(tmp_path / "output")}
    environment.pop("AMBI_RETURN_GATE_WARMUP", None)
    environment.pop("AMBI_RETURN_GATE_DECISIONS", None)
    if override is not None:
        environment["AMBI_RETURN_GATE_WARMUP"] = override
    arguments = subprocess.check_output(["bash", "-eu", "-c", fragment], env=environment, text=True).splitlines()
    assert arguments[arguments.index("--decisions") + 1] == "120"
    assert ("--warmup" in arguments) == (override is not None)
    if override is not None:
        assert arguments[arguments.index("--warmup") + 1] == override


@pytest.mark.parametrize("horizon", [4, 5])
def test_h4_h5_held_actor_clears_at_boundary_and_resumes_exactly(horizon):
    import torch
    from tests.test_ambi_root_local_sac import _tiny_component_model
    from tests.test_ambi_latency_contract import _assert_tree_equal
    models = []
    try:
        for _ in range(2):
            with pytest.warns(UserWarning, match="exceeds train_unroll_horizon"):
                models.append(_tiny_component_model(inner_solve_interval=horizon, inner_rounds=1,
                    inner_rollout_horizon=horizon, train_unroll_horizon=3,
                    outer_planning_horizon=3, outer_critic_target="reward_only",
                    inner_sac_critic_target="reward_only",
                    inner_critic_updates_per_round=2, inner_actor_updates_per_round=1,
                    inner_replay_capacity=2 * horizon, inner_execution_action="policy_sample",
                    inner_critic_bernoulli_p=0.))
        source, restored = [model.agent for model in models]
        source.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        source.act(torch.ones(3) * .1, collect_diagnostics=False)
        assert source.last_inner_metrics["inner_policy_held"] == 1.
        source.prepare_training_resume_boundary()
        payload = deepcopy(source.training_state_dict())
        assert source.inner_engine._held_actor is source.inner_engine._held_encoder is None
        assert payload["inner"]["held_policy_spec"]["solve_interval"] == horizon
        restored.load_training_state_dict(payload)
        _assert_tree_equal(restored.training_state_dict(), payload)
        for agent in (source, restored):
            agent.reset()
        for index in range(horizon + 1):
            observation = torch.tensor([.1 * index, .2, -.3])
            expected = source.act(observation, t0=index == 0, collect_diagnostics=False)
            actual = restored.act(observation, t0=index == 0, collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert restored.last_inner_metrics["inner_solve_performed"] == float(index % horizon == 0)
        for agent in (source, restored):
            agent.prepare_training_resume_boundary()
        _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())
    finally:
        for model in models:
            model.close()
            model.env.close()
