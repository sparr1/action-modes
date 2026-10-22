"""A frozen temperature stream crosses episodes without carrying policy state."""

from copy import deepcopy
from dataclasses import fields
import gzip
import json

import gymnasium as gym
import numpy as np
import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from RL.tdmpc2_core.inner_improvement import InnerWorkspace
from tests.test_ambi_early_h3_j8 import _tiny_checkpoint
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_temperature_adaptation import _model
from utils.ambi_benchmark import BenchmarkBundle, protocol_for
from utils.eval_series_data import _episode_protocol


def _chain_model():
    return _model(
        aux_return_mode="sac", inner_actor_source="sac",
        inner_critic_source="aux_return", inner_horizon_actor_source="sac",
        inner_horizon_critic_source="aux_return", inner_sac_critic_target="reward_only",
        inner_temperature_scope="run", inner_temperature_optimizer_scope="run",
    )


def test_only_alpha_adam_and_lifetime_survive_evaluation_episode_boundary():
    model = _chain_model()
    try:
        engine = model.agent.inner_engine
        outer = _clone_tree(model.agent.model.state_dict())
        outer_alpha = model.agent.alpha.detach().clone()
        engine.reset_for_evaluation(1001)
        prepare = engine._prepare_workspace

        def prepare_and_check(*, t0):
            prepare(t0=t0)
            state = engine.state
            _assert_tree_equal(state.actor.state_dict(), engine._actor_owner.model._pi.state_dict())
            _assert_tree_equal(state.critic.state_dict(), model.agent.model._aux_return_Qs.state_dict())
            assert state.replay.size == 0
            assert state.actor_lifetime_steps == state.critic_lifetime_steps == 0

        engine._prepare_workspace = prepare_and_check
        previous_final = None
        for global_decision in range(6):
            if global_decision in (2, 4):
                alpha_parameter = engine.state.log_alpha
                optimizer = engine.state.temperature_optim
                optimizer_before = _clone_tree(optimizer.state_dict())
                engine.reset_for_evaluation(1001 + global_decision, preserve_run_temperature=True)
                assert engine.state.log_alpha is alpha_parameter
                assert engine.state.temperature_optim is optimizer
                assert optimizer.param_groups[0]["params"][0] is alpha_parameter
                _assert_tree_equal(optimizer.state_dict(), optimizer_before)
                blank = InnerWorkspace()
                for field in fields(blank):
                    if field.name not in {"log_alpha", "temperature_optim", "temperature_lifetime_steps"}:
                        _assert_tree_equal(getattr(engine.state, field.name), getattr(blank, field.name))
                _assert_tree_equal(engine._action_pool, blank)
                expected_rng = engine._new_rng(1001 + global_decision).training_state_dict()
                _assert_tree_equal(engine.rng.training_state_dict(), expected_rng)
                assert engine.action_index == engine.episode_index == 0
            model.agent.act(torch.zeros(3), t0=global_decision % 2 == 0, eval_mode=False)
            metrics = model.agent.last_inner_metrics
            assert metrics["inner_temperature_chain_reset"] == float(global_decision == 0)
            assert metrics["inner_temperature_episode_reset"] == 0
            for prefix in ("inner_temperature_optimizer_step", "inner_temperature_lifetime_steps"):
                assert metrics[prefix + "_initial"] == global_decision * 4
                assert metrics[prefix + "_final"] == (global_decision + 1) * 4
            assert metrics["inner_alpha_initial"] == pytest.approx(
                outer_alpha.item() if previous_final is None else previous_final, rel=0, abs=0)
            previous_final = metrics["inner_alpha_final"]
        _assert_tree_equal(model.agent.model.state_dict(), outer)
        torch.testing.assert_close(model.agent.alpha, outer_alpha, rtol=0, atol=0)
        # Starting another stream discards alpha and Adam despite run scopes.
        engine.reset_for_evaluation(1001)
        model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_temperature_chain_reset"] == 1
        assert metrics["inner_temperature_optimizer_step_initial"] == 0
        assert metrics["inner_alpha_initial"] == pytest.approx(outer_alpha.item(), rel=0, abs=0)
    finally:
        model.close()


def test_online_episode_reset_preserves_run_alpha_and_adam():
    model = _chain_model()
    try:
        model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
        engine = model.agent.inner_engine
        alpha, optimizer = engine.state.log_alpha, engine.state.temperature_optim
        previous_final = model.agent.last_inner_metrics["inner_alpha_final"]
        model.agent.reset()
        assert engine.state.log_alpha is alpha
        assert engine.state.temperature_optim is optimizer
        model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
        assert model.agent.last_inner_metrics["inner_alpha_initial"] == previous_final
        assert model.agent.last_inner_metrics["inner_temperature_optimizer_step_initial"] == 4
        assert model.agent.last_inner_metrics["inner_temperature_chain_reset"] == 0
    finally:
        model.close()


@pytest.mark.parametrize("scope", ["action", "episode", "run"])
def test_chain_reset_requires_supported_initialized_state(scope):
    model = _model(inner_temperature_scope=scope, inner_temperature_optimizer_scope=scope)
    try:
        engine = model.agent.inner_engine
        with pytest.raises(ValueError, match="initialized" if scope == "run" else "paired run-scoped"):
            engine.reset_for_evaluation(101, preserve_run_temperature=True)
        if scope == "run":
            model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
            engine.state.temperature_optim.param_groups[0]["params"] = [torch.nn.Parameter(torch.zeros(()))]
            with pytest.raises(RuntimeError, match="does not own"):
                engine.reset_for_evaluation(102, preserve_run_temperature=True)
    finally:
        model.close()


def test_frozen_chain_is_ordered_reproducible_and_keeps_outer_frozen(tmp_path, monkeypatch):
    path, resolved = _tiny_checkpoint(tmp_path, "mppi")
    resolved["evaluation"]["execution_action"] = "sample"
    resolved["algorithm_config"]["alg_params"].update(
        inner_temperature_scope="run", inner_temperature_optimizer_scope="run",
    )
    seeds = [103, 101, 102]
    observed = []
    for repetition in range(2):
        actions = []

        def make_env(_resolved):
            env = gym.make("Pendulum-v1", max_episode_steps=2)
            original_step = env.step

            def step(action):
                actions.append(np.asarray(action).copy())
                return original_step(action)

            env.step = step
            return env

        monkeypatch.setattr(evaluator, "_make_env", make_env)
        protocol = protocol_for(resolved, 55, 2)
        protocol["temperature_chain"] = {
            "kind": "run-temperature-chain-v1", "ordered_environment_seeds": seeds,
        }
        bundle = BenchmarkBundle(tmp_path / f"chain-{repetition}",
                                 checkpoint={"sha256": "test"}, protocol=protocol)
        run = bundle.start_run(resolved, "episodes")
        result = evaluator.evaluate_preset(
            resolved, path, seeds, controller_seed=55, max_steps=2, device="cpu",
            bundle=bundle, bundle_run=run, temperature_chain=True,
        )
        assert result["outer_state_unchanged"]
        assert result["outer_updates_before"] == result["outer_updates_after"] == 0
        assert result["temperature_chain"] is True
        assert result["temperature_chain_seeds"] == seeds
        assert result["temperature_chain_kind"] == "run-temperature-chain-v1"
        assert "dependent" in result["episode_variation_semantics"]
        assert [episode["seed"] for episode in result["episodes"]] == seeds
        assert [episode["temperature_chain_episode_index"] for episode in result["episodes"]] == [0, 1, 2]
        rows = [json.loads(line) for relative in run["trace_files"]
                for line in gzip.decompress((bundle.path / relative).read_bytes()).decode().splitlines()]
        decisions = [row for row in rows if row["phase"] == "decision"]
        assert len(decisions) == 6
        previous = None
        for index, row in enumerate(decisions):
            metric = row["metrics"]
            assert metric["decision/inner_temperature_optimizer_step_initial"] == index * 2
            assert metric["decision/inner_temperature_optimizer_step_final"] == (index + 1) * 2
            assert metric["decision/inner_temperature_lifetime_steps_initial"] == index * 2
            assert metric["decision/inner_temperature_chain_reset"] == float(index == 0)
            assert metric["decision/inner_temperature_episode_reset"] == 0
            if previous is not None:
                assert metric["decision/inner_alpha_initial"] == previous
            previous = metric["decision/inner_alpha_final"]
        observed.append((actions, [episode["return"] for episode in result["episodes"]]))
    np.testing.assert_array_equal(observed[0][0], observed[1][0])
    assert observed[0][1] == observed[1][1]
    # Run carry changes nothing before the first episode boundary, so its first
    # episode remains exactly paired with the earlier episode-carry protocol.
    independent = deepcopy(resolved)
    independent["algorithm_config"]["alg_params"].update(
        inner_temperature_scope="episode", inner_temperature_optimizer_scope="episode",
    )
    actions = []
    baseline_bundle = BenchmarkBundle(tmp_path / "episode-baseline", checkpoint={"sha256": "test"},
                                     protocol=protocol_for(independent, 55, 2))
    baseline_run = baseline_bundle.start_run(independent, "episodes")
    baseline = evaluator.evaluate_preset(
        independent, path, seeds[:1], controller_seed=55, max_steps=2, device="cpu",
        bundle=baseline_bundle, bundle_run=baseline_run,
    )
    np.testing.assert_array_equal(actions, observed[0][0][:2])
    assert baseline["episodes"][0]["return"] == observed[0][1][0]
    assert "temperature_chain" not in baseline
    with pytest.raises(ValueError, match="explicit --temperature-chain"):
        evaluator.evaluate_preset(resolved, path, seeds, controller_seed=55)


def test_chain_scopes_and_publication_require_explicit_protocol():
    params = dict(inner_operator="sac", inner_temperature_mode="auto", inner_entropy_enabled=True,
                  inner_temperature_scope="run", inner_temperature_optimizer_scope="run",
                  inner_actor_scope="action", inner_critic_scope="action", inner_replay_scope="action")
    evaluator._validate_benchmark_scopes(params, episode_evaluation=True, temperature_chain=True)
    for override in ({"inner_actor_scope": "run"}, {"inner_temperature_optimizer_scope": "episode"},
                     {"inner_explorer_mode": "independent"}, {"inner_actor_writeback_coef": .1},
                     {"inner_entropy_enabled": False}, {"inner_temperature_mode": "fixed"}):
        with pytest.raises(ValueError):
            evaluator._validate_benchmark_scopes({**params, **override}, episode_evaluation=True,
                                                 temperature_chain=True)
    with pytest.raises(ValueError):
        evaluator._validate_benchmark_scopes(params, episode_evaluation=False, temperature_chain=True)
    for option in ("reference_bundle", "eval_run_dir", "eval_run_map", "eval_series_spec_dir",
                   "root_bank_path", "save_root_bank", "wandb_options"):
        with pytest.raises(ValueError, match="own protocol"):
            evaluator.evaluate_matrix("missing", "missing", temperature_chain=True, **{option: "set"})
    with pytest.raises(ValueError, match="dependent"):
        _episode_protocol({"temperature_chain": {"kind": "run-temperature-chain-v1"}}, [103, 101, 102])
    assert evaluator.build_parser().parse_args(["--matrix", "matrix.json", "--checkpoint", "c.pt",
                                                "--temperature-chain"]).temperature_chain
