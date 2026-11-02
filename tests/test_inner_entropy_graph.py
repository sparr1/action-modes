"""Graph capture preserves entropy definitions and all three Q-scale lifetimes."""

from copy import deepcopy
import math

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.whole_inner_graph import (
    WholeInnerGraph, preflight_training_graph_state, training_graph_act,
    training_graph_state, validate_training_graph,
)
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_params
from utils.resume_identity import scientific_trial_parameters


MODES = (("action", "per_action"), ("action", "per_update"), ("run", "per_update"))
ENTROPIES = ("tdmpc2_scaled", "squashed")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires scheduled CUDA GPU.")


def make_agent(entropy="squashed", scope="action", update="per_action", *, device="cpu", compiled=False):
    params = _tiny_params(
        compile=compiled, compile_strict=compiled, critic_ensemble_backend="vmap",
        outer_actor_entropy_mode=entropy, inner_actor_entropy_mode=entropy,
        log_std_mapping="tdmpc2_tanh", inner_log_std_mapping="tdmpc2_tanh",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range", sac_actor_loss_scale_tau=.25,
        inner_actor_loss_scale_scope=scope, inner_actor_loss_scale_update=update,
        outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
        ent_coef=.0001, inner_temperature_mode="inherit_outer", aux_return_mode="off",
        inner_rounds=1, inner_rollouts_per_round=4, inner_batch_size=4,
        inner_critic_updates_per_round=2, inner_actor_updates_per_round=2,
        inner_replay_capacity=8, inner_finite_horizon=True, inner_solve_interval=1,
        inner_root_entropy_samples=0, critic_geometry_diagnostics=False,
        inner_rebase_persistent=False,
    )
    params.pop("inner_updates_per_round", None)
    cfg = _build_cfg(**params)
    cfg.device = device
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    agent.actor_loss_scale.fill_(4.)
    return agent


def assert_tree_close(actual, expected, *, rtol=0, atol=0):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_close(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_close(left, right, rtol=rtol, atol=atol)
    elif isinstance(actual, float):
        assert actual == pytest.approx(expected, rel=rtol, abs=atol, nan_ok=True)
    else:
        assert actual == expected


def inner_snapshot(engine):
    pool = engine._action_pool
    result = {}
    for name in ("actor", "critic", "critic_target", "actor_target"):
        module = getattr(pool, name)
        result[name] = None if module is None else deepcopy(module.state_dict())
    result["alpha_fixed"] = deepcopy(pool.alpha_fixed)
    result["log_alpha"] = deepcopy(pool.log_alpha)
    result["replay"] = pool.replay._storage[:pool.replay.size].detach().clone()
    result["replay_metadata"] = (pool.replay.size, pool.replay.pos, pool.replay.full, pool.replay.next_sample_id)
    result["optimizers"] = {
        name: None if getattr(pool, name) is None else deepcopy(getattr(pool, name).state_dict())
        for name in ("actor_optim", "critic_optim", "temperature_optim")
    }
    return result


def outer_snapshot(agent):
    return {
        "model": deepcopy(agent.model.state_dict()),
        "scale": agent.actor_loss_scale.detach().clone(),
        "alpha": agent.alpha.detach().clone(),
        "optimizers": {name: None if getattr(agent, name) is None else deepcopy(getattr(agent, name).state_dict())
                       for name in ("optim", "pi_optim", "ent_coef_optim")},
    }


@pytest.mark.parametrize("entropy", ENTROPIES)
@pytest.mark.parametrize("scope,update", MODES)
def test_graph_validation_accepts_six_normalization_contracts(entropy, scope, update):
    agent = make_agent(entropy, scope, update)
    agent.cfg.compile = agent.cfg.compile_strict = True
    agent.inner_engine.device = torch.device("cuda")
    validate_training_graph(agent.inner_engine)


@pytest.mark.parametrize("key,value", [
    ("inner_solve_interval", 3), ("aux_return_mode", "sac"),
    ("inner_temperature_scope", "run"), ("compile_strict", False),
    ("inner_actor_loss_scale_scope", "episode"), ("inner_actor_loss_scale_update", "unknown"),
])
def test_graph_validation_rejects_unsupported_protocols(key, value):
    agent = make_agent()
    agent.cfg.compile = agent.cfg.compile_strict = True
    agent.inner_engine.device = torch.device("cuda")
    setattr(agent.cfg, key, value)
    with pytest.raises(ValueError):
        validate_training_graph(agent.inner_engine)


def test_graph_mode_has_distinct_scientific_identity():
    native = {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": {}}
    disabled = {**native, "alg_params": {"inner_cuda_graph": False}}
    graph = {**native, "alg_params": {"inner_cuda_graph": True}}
    assert scientific_trial_parameters(native) == scientific_trial_parameters(disabled)
    assert scientific_trial_parameters(native) != scientific_trial_parameters(graph)


def test_graph_state_rejects_incomplete_rng_and_native_cross_resume():
    agent = make_agent()
    engine = agent.inner_engine
    agent.cfg.inner_cuda_graph = True
    agent.cfg.compile = agent.cfg.compile_strict = True
    engine.device = torch.device("cuda")
    saved = training_graph_state(engine)
    assert preflight_training_graph_state(engine, saved) == saved
    for bad in (None, {}, {**saved, "calls": 1}, {**saved, "seed": -1}):
        with pytest.raises(ValueError):
            preflight_training_graph_state(engine, bad)
    agent.cfg.inner_cuda_graph = False
    with pytest.raises(ValueError, match="native"):
        preflight_training_graph_state(engine, saved)


def test_graph_carry_bridge_owns_scalar_storage_and_reads_latest_native_value():
    agent = make_agent(scope="run", update="per_update")
    engine = agent.inner_engine
    inputs = []
    final = torch.tensor(2.5)
    def runner(root, *, initial_actor_loss_scale):
        inputs.append(initial_actor_loss_scale.clone())
        return root, {"inner_actor_loss_scale_final": final}, []
    engine._whole_graph_runner = runner
    root = torch.zeros(1, agent.cfg.latent_dim)
    training_graph_act(engine, root)
    torch.testing.assert_close(inputs[-1], torch.tensor([4.]))
    assert engine._carried_actor_loss_scale.shape == agent.actor_loss_scale.shape
    final.fill_(9.)
    torch.testing.assert_close(engine._carried_actor_loss_scale, torch.tensor([2.5]))
    # Stand in for a successful intervening native diagnostic solve.
    engine._carried_actor_loss_scale.fill_(3.25)
    agent.actor_loss_scale.fill_(100.)
    training_graph_act(engine, root)
    torch.testing.assert_close(inputs[-1], torch.tensor([3.25]))
    assert engine.action_index == 2
    assert engine.state.outer_version == agent.outer_version
    torch.testing.assert_close(agent.actor_loss_scale, torch.tensor([100.]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., .5])
def test_graph_scale_preflight_is_atomic(value):
    # No CUDA allocation is needed to exercise preflight before replay.
    runner = WholeInnerGraph.__new__(WholeInnerGraph)
    runner.root = torch.zeros(1, 4)
    runner.actor_loss_scale_input = torch.tensor([4.])
    runner.alpha_input = torch.tensor(.1)
    runner.calls = 7
    runner._initial_alpha = lambda: pytest.fail("Invalid Q scale reached alpha refresh")
    with pytest.raises(ValueError, match="finite and at least one"):
        runner(torch.ones_like(runner.root), initial_actor_loss_scale=torch.tensor([value]))
    torch.testing.assert_close(runner.root, torch.zeros_like(runner.root))
    torch.testing.assert_close(runner.actor_loss_scale_input, torch.tensor([4.]))
    torch.testing.assert_close(runner.alpha_input, torch.tensor(.1))
    assert runner.calls == 7


def assert_graph_resume_continuation(agent):
    """Production gate: graph/native/graph, changed priors, episode reset, recapture.

    Restores the original boundary after checking so this adds no training dose.
    Both graph randomness and the run-scoped Q estimator are part of the saved
    state. Normal scalar diagnostics, including entropy and saturation, must
    survive graph replay and agree after recapture.
    """
    engine = agent.inner_engine
    assert engine._whole_graph_runner is not None and engine._whole_graph_runner.calls > 0
    agent.prepare_training_resume_boundary()
    saved = deepcopy(agent.training_state_dict())
    original_runner = engine._whole_graph_runner
    results = []
    for attempt in range(2):
        if attempt:
            agent.load_training_state_dict(saved)
            assert agent.inner_engine._whole_graph_runner is None
        engine = agent.inner_engine
        agent.reset()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        sequence = []
        previous = (None if engine._carried_actor_loss_scale is None
                    else engine._carried_actor_loss_scale.detach().clone())
        for index, diagnostic in enumerate((False, True, False)):
            # This real outer quantity changes independently between solves.
            agent.actor_loss_scale.fill_(4. + index)
            if index:
                with torch.no_grad():
                    next(agent.model._pi.parameters()).add_(.0001)
            observation = torch.full(agent.cfg.obs_shape["state"], .1 * index)
            action = agent.act(observation, t0=index == 0, collect_diagnostics=diagnostic)
            metrics = deepcopy(agent.last_inner_metrics)
            assert all(math.isfinite(float(value)) for value in metrics.values())
            assert metrics["inner_cuda_graph_used"] == float(not diagnostic)
            assert metrics["inner_critic_optimizer_steps"] == agent.cfg.inner_critic_updates_per_action
            assert metrics["inner_actor_optimizer_steps"] == agent.cfg.inner_actor_updates_per_action
            assert metrics["inner_temperature_optimizer_steps"] == 0
            expected_initial = (float(previous.item()) if engine._carry_actor_loss_scale and previous is not None
                                else 4. + index)
            assert metrics["inner_actor_loss_scale_initial"] == pytest.approx(expected_initial)
            assert metrics["inner_actor_loss_scale_outer"] == pytest.approx(4. + index)
            if agent.cfg.inner_actor_loss_scale_update == "per_action":
                assert metrics["inner_actor_loss_scale_final"] == pytest.approx(expected_initial)
            torch.testing.assert_close(agent.actor_loss_scale, torch.full_like(agent.actor_loss_scale, 4. + index))
            if engine._carry_actor_loss_scale:
                previous = engine._carried_actor_loss_scale.detach().clone()
                assert previous.item() == pytest.approx(metrics["inner_actor_loss_scale_final"])
            if diagnostic:
                assert metrics["inner_diagnostics_sampled"] == 1
            sequence.append((action.clone(), {k: v for k, v in metrics.items() if not k.endswith("seconds")}))
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
        agent.prepare_training_resume_boundary()
        results.append((sequence, deepcopy(agent.training_state_dict())))
    assert agent.inner_engine._whole_graph_runner is not original_runner
    assert_tree_close(results[0], results[1])
    agent.load_training_state_dict(saved)


@CUDA
@pytest.mark.parametrize("entropy", ENTROPIES)
@pytest.mark.parametrize("scope,update", MODES)
def test_cuda_capture_matches_uncaptured_normalized_solve(entropy, scope, update):
    native = make_agent(entropy, scope, update, device="cuda", compiled=True)
    graph = make_agent(entropy, scope, update, device="cuda", compiled=True)
    graph.load(deepcopy(native.checkpoint_state()))
    root = torch.zeros(1, native.cfg.latent_dim, device="cuda")
    control = WholeInnerGraph(native.inner_engine, root, capture=False)
    captured = WholeInnerGraph(graph.inner_engine, root)
    initial = torch.full_like(native.actor_loss_scale, 4.)
    for index in range(3):
        native.actor_loss_scale.fill_(5. + index)
        graph.actor_loss_scale.fill_(5. + index)
        if scope == "action":
            initial = native.actor_loss_scale.clone()
        # Both paths use the same isolated graph-protocol RNG; native AMBI uses
        # a different per-phase stream and is deliberately not claimed bit-equal.
        outer_before = [outer_snapshot(agent) for agent in (native, graph)]
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        private_rng = captured.rng_state.clone()
        expected = control(root + .1 * index, initial_actor_loss_scale=initial)
        actual = captured(root + .1 * index, initial_actor_loss_scale=initial)
        assert_tree_close(actual, expected, rtol=5e-5, atol=5e-7)
        assert_tree_close(inner_snapshot(graph.inner_engine), inner_snapshot(native.inner_engine), rtol=5e-5, atol=5e-7)
        for agent, before in zip((native, graph), outer_before):
            assert_tree_close(outer_snapshot(agent), before)
            assert all(parameter.grad is None for parameter in agent.model.parameters())
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
        torch.testing.assert_close(control.rng_state, captured.rng_state, rtol=0, atol=0)
        assert not torch.equal(captured.rng_state, private_rng)
        if update == "per_action":
            torch.testing.assert_close(actual[1]["inner_actor_loss_scale_final"], initial.reshape(()))
        initial = expected[1]["inner_actor_loss_scale_final"].reshape_as(initial).clone()
        for left, right in zip(native.model.parameters(), graph.model.parameters()):
            with torch.no_grad():
                left.add_(.0001)
                right.add_(.0001)


@CUDA
@pytest.mark.parametrize("entropy", ENTROPIES)
@pytest.mark.parametrize("scope,update", MODES)
def test_cuda_graph_native_graph_scale_and_resume(entropy, scope, update):
    agent = make_agent(entropy, scope, update, device="cuda", compiled=True)
    agent.cfg.inner_cuda_graph = True
    agent.act(torch.zeros(agent.cfg.obs_shape["state"]), collect_diagnostics=False)
    assert_graph_resume_continuation(agent)
