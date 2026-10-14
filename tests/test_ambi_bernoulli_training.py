"""Online training, evaluation isolation and exact resume for critic transfer."""

from copy import deepcopy

import gymnasium as gym
import pytest
import torch

from RL.tdmpc2_core.paired_controller_evaluation import PairedControllerEvaluator
from tests.test_ambi_bernoulli_transfer import models
from tests.test_ambi_inner_initialization import _snapshot
from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_pave_integration import _batch
from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


def test_disabled_transfer_preserves_old_trajectory_and_resume_payload(models):
    implicit = _tiny_model(inner_rounds=1, inner_updates_per_round=2)
    explicit = models(inner_critic_bernoulli_p=0.)
    try:
        _assert_tree_equal(implicit.agent.training_state_dict(), explicit.agent.training_state_dict())
        for index in range(3):
            expected = implicit.agent.act(torch.zeros(3), t0=index == 0, collect_diagnostics=False)
            actual = explicit.agent.act(torch.zeros(3), t0=index == 0, collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _assert_tree_equal(_snapshot(implicit.agent.inner_engine),
                               _snapshot(explicit.agent.inner_engine))
        for agent in (implicit.agent, explicit.agent):
            agent.prepare_training_resume_boundary()
        _assert_tree_equal(implicit.agent.training_state_dict(), explicit.agent.training_state_dict())
        assert "critic_transfer_spec" not in explicit.agent.inner_engine.training_state_dict()
        assert "critic_bernoulli" not in explicit.agent.inner_engine.rng.generators
    finally:
        implicit.close()
        implicit.env.close()


@pytest.mark.parametrize("auxiliary", [False, True])
def test_exact_resume_reproduces_masks_actions_and_outer_learning(models, auxiliary):
    options = {}
    if auxiliary:
        options.update(aux_return_mode="return_actor", inner_actor_source="return_actor",
                       inner_critic_source="aux_return")
    source = models(**options).agent
    source.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    source._update(*_batch(source))
    source.act(torch.zeros(3), collect_diagnostics=False)
    source.prepare_training_resume_boundary()
    payload = deepcopy(source.training_state_dict())
    assert payload["inner"]["critic_transfer_spec"]["probability"] == .5
    assert payload["outer"]["critic_target_spec"]["inner_solve"]["critic_bernoulli_p"] == .5

    restored = models(**options).agent
    global_rng = torch.random.get_rng_state().clone()
    restored.load_training_state_dict(payload)
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    _assert_tree_equal(restored.training_state_dict(), payload)
    for agent in (source, restored):
        agent.reset()
    for index in range(3):
        # Outer updates use global RNG; feed the same state to both continuations.
        outer_rng = torch.random.get_rng_state().clone()
        for agent in (source, restored):
            torch.random.set_rng_state(outer_rng)
            agent._update(*_batch(agent))
        expected = source.act(torch.ones(3) * .1 * index, collect_diagnostics=False)
        actual = restored.act(torch.ones(3) * .1 * index, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(_snapshot(source.inner_engine), _snapshot(restored.inner_engine))
        assert restored.last_inner_metrics["inner_critic_bernoulli_applied"] == float(index > 0)
    for agent in (source, restored):
        agent.prepare_training_resume_boundary()
    _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())


@pytest.mark.parametrize("source_p,target_p", [(0., .5), (.5, 0.), (.5, 1.)])
@pytest.mark.parametrize("direct", [False, True])
def test_resume_rejects_different_transfer_before_mutation(models, source_p, target_p, direct):
    source = models(inner_critic_bernoulli_p=source_p).agent
    target = models(inner_critic_bernoulli_p=target_p).agent
    if direct:
        source, target = source.inner_engine, target.inner_engine
    before = deepcopy(target.training_state_dict())
    global_rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="incompatible|does not match"):
        target.load_training_state_dict(deepcopy(source.training_state_dict()))
    _assert_tree_equal(target.training_state_dict(), before)
    assert torch.equal(torch.random.get_rng_state(), global_rng)


@pytest.mark.parametrize("corruption", ["version", "missing_spec", "missing_rng", "extra"])
def test_corrupt_transfer_state_rejected_transactionally(models, corruption):
    engine = models().agent.inner_engine
    before = deepcopy(engine.training_state_dict())
    payload = deepcopy(before)
    if corruption == "version":
        payload["critic_transfer_spec"]["version"] = True
    elif corruption == "missing_spec":
        del payload["critic_transfer_spec"]
    elif corruption == "missing_rng":
        del payload["rng"]["streams"]["critic_bernoulli"]
    else:
        payload["critic_transfer_spec"]["scale"] = 2.
    with pytest.raises((ValueError, TypeError)):
        engine.load_training_state_dict(payload)
    _assert_tree_equal(engine.training_state_dict(), before)


@pytest.mark.parametrize("source_p,target_p", [(0., .5), (.5, 0.)])
def test_portable_weights_allow_new_controller_and_clear_live_donor(models, source_p, target_p):
    source = models(inner_critic_bernoulli_p=source_p).agent
    target = models(inner_critic_bernoulli_p=target_p).agent
    target.act(torch.zeros(3))
    target.load(deepcopy(source.checkpoint_state()))
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())
    assert not target.inner_engine._critic_transfer_ready
    target.act(torch.zeros(3))
    assert target.last_inner_metrics.get("inner_critic_bernoulli_applied", 0.) == 0.


@pytest.mark.parametrize("source,horizon", [("sac", 2), ("aux_return", 1),
                                          ("aux_return", 2), ("aux_return", 3)])
def test_real_training_loop_transfers_with_outer_updates_and_resets_each_episode(
    models, monkeypatch, source, horizon,
):
    options = dict(eval_freq=None, inner_rollout_horizon=horizon)
    if source == "aux_return":
        options.update(aux_return_mode="return_actor", inner_actor_source="return_actor",
                       inner_critic_source="aux_return", inner_horizon_actor_source="return_actor",
                       inner_horizon_critic_source="aux_return", q_representation="distributional")
    model = models(**options)
    seen = []
    act = model.agent.act

    def record(*args, **kwargs):
        result = act(*args, **kwargs)
        seen.append((kwargs["t0"], model.agent.outer_version,
                     model.agent.last_inner_metrics["inner_critic_bernoulli_applied"]))
        return result

    monkeypatch.setattr(model.agent, "act", record)
    model.learn(total_timesteps=15)
    assert model._global_step == 15
    assert model.agent.num_updates > 0
    assert len(seen) >= 10
    assert sum(first for first, _, _ in seen) >= 2
    assert all(applied == float(not first) for first, _, applied in seen)
    assert len({version for _, version, _ in seen}) > 1


def test_paired_evaluation_has_its_own_donor_and_correct_metric_names(models, monkeypatch):
    model = models()
    agent, live = model.agent, model.agent.inner_engine
    agent.act(torch.zeros(3))
    before = _snapshot(live)
    old_metrics = deepcopy(agent.last_inner_metrics)
    evaluator = PairedControllerEvaluator(
        agent, lambda: gym.make("Pendulum-v1", max_episode_steps=3),
        lambda obs: torch.as_tensor(obs, dtype=torch.float32), lambda action: action * 2,
        episodes=2, seed=18, device="cpu",
    )
    seen = []
    act = agent.act

    def record(*args, **kwargs):
        result = act(*args, **kwargs)
        seen.append(agent.last_inner_metrics["inner_critic_bernoulli_applied"])
        return result

    monkeypatch.setattr(agent, "act", record)
    try:
        metrics = evaluator.evaluate()
        assert seen == [0., 1., 1., 0., 1., 1.]
        assert "eval/paired_bernoulli_inner_episode_reward" in metrics
        assert "eval/paired_outer_episode_reward" in metrics
        assert not any("paired_fresh_inner" in key for key in metrics)
        assert agent.inner_engine is live and live._critic_transfer_ready
        _assert_tree_equal(_snapshot(live), before)
        _assert_tree_equal(agent.last_inner_metrics, old_metrics)
        agent.act(torch.zeros(3))
        assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 1.
    finally:
        evaluator.close()


def test_primary_evaluation_restores_training_boundary_and_rng(models):
    model = models(eval_freq=5, eval_episodes=1)
    agent = model.agent
    agent.act(torch.zeros(3))
    agent.prepare_training_resume_boundary()
    before = deepcopy(agent.inner_engine.training_state_dict())
    obs, _ = model.env.reset(seed=13)
    model._evaluate_policy(5, initial_obs=obs)
    _assert_tree_equal(agent.inner_engine.training_state_dict(), before)
    agent.reset()
    agent.act(torch.zeros(3))
    assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 0.


def test_transfer_is_recorded_in_runtime_and_scientific_identity(models):
    import main as training_main

    base = {"alg": "AMBITDMPC2/AMBITDMPC2", "seed": 13,
            "alg_params": {"inner_operator": "sac"}}
    disabled = {**base, "alg_params": {**base["alg_params"], "inner_critic_bernoulli_p": 0.}}
    active = {**base, "alg_params": {**base["alg_params"], "inner_critic_bernoulli_p": .5}}
    assert scientific_trial_parameters(base) == scientific_trial_parameters(disabled)
    assert scientific_trial_parameters(base) != scientific_trial_parameters(active)
    identities = [planner_identity(config["alg_params"], {}, base["alg"], "mean")
                  for config in (base, disabled, active)]
    assert identities[0] == identities[1] and identities[0] != identities[2]
    metadata = training_main._resolved_runtime_metadata(models(), trial_run_params=active)
    assert metadata["inner_budget"]["inner_critic_bernoulli_p"] == .5


def _assert_compiled_state_close(actual, expected, key=None):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            _assert_compiled_state_close(actual[name], expected[name], key=name)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_compiled_state_close(left, right, key=key)
    elif torch.is_tensor(expected) and expected.is_floating_point() and key != "step":
        # Successive transferred weights accumulate float32 vmap/graph rounding.
        # RNG state, optimizer steps and other counters remain bitwise identical.
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-9)
    else:
        _assert_tree_equal(actual, expected)


@pytest.mark.parametrize("backend", ["loop", "vmap"])
def test_compiled_sac_reuses_graphs_with_transferred_weights(models, monkeypatch, backend):
    torch._dynamo.reset()
    compile_original = torch.compile
    graphs = []

    def compile_for_test(function, **kwargs):
        def capture(graph, inputs):
            graphs.append(function.__name__)
            return graph.forward
        return compile_original(function, backend=capture, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_for_test)
    options = dict(critic_ensemble_backend=backend, dropout=.2)
    eager, compiled = models(**options).agent, models(**options, compile=True).agent
    try:
        for index in range(3):
            if index:
                # Change current priors without replacing module/parameter identities.
                with torch.no_grad():
                    for agent in (eager, compiled):
                        for parameter in agent.model.parameters():
                            parameter.add_(.001)
                        agent.outer_version += 1
            expected = eager.act(torch.zeros(3), collect_diagnostics=False)
            actual = compiled.act(torch.zeros(3), collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=5e-7, atol=1e-10)
            actual_state, expected_state = _snapshot(compiled.inner_engine), _snapshot(eager.inner_engine)
            if backend == "loop":
                _assert_tree_equal(actual_state, expected_state)
            else:
                _assert_tree_equal(actual_state["rng"], expected_state["rng"])
                _assert_tree_equal(actual_state["replay"], expected_state["replay"])
                _assert_compiled_state_close(actual_state, expected_state)
            assert compiled.last_inner_metrics["inner_compile_fallback"] == 0.
            if index == 0:
                count = len(graphs)
                assert count > 0
            else:
                assert len(graphs) == count
        assert {"_dense_rollout_kernel", "_sac_critic_kernel", "_sac_actor_kernel"} <= set(graphs)
    finally:
        torch._dynamo.reset()
