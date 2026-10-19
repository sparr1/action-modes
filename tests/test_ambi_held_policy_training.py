"""Held policies still observe, collect, and train at every real decision."""

from copy import deepcopy

import gymnasium as gym
import pytest
import torch

from RL.tdmpc2_core.paired_controller_evaluation import PairedControllerEvaluator
from tests.test_ambi_inner_initialization import _snapshot
from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_pave_integration import _batch


HELD_ZERO_WORK = (
    "inner_rollouts", "inner_steps", "inner_updates", "inner_model_steps",
    "inner_realized_model_steps", "inner_total_model_steps", "inner_replay_draws",
    "inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
    "inner_temperature_optimizer_steps", "inner_q_evaluations",
)


@pytest.fixture
def held_models():
    opened = []

    def create(*, interval=3, auxiliary=False, probability=0., **overrides):
        options = dict(
            inner_rounds=1, inner_updates_per_round=2,
            inner_solve_interval=interval, inner_critic_bernoulli_p=probability,
            inner_finite_horizon=True, inner_execution_action="policy_sample",
            outer_critic_target="entropy_augmented", eval_freq=None,
        )
        if auxiliary:
            options.update(
                aux_return_mode="sac", inner_critic_source="aux_return",
                inner_horizon_critic_source="aux_return",
                inner_sac_critic_target="reward_only", inner_terminal_entropy="none",
            )
        options.update(overrides)
        model = _tiny_model(**options)
        opened.append(model)
        return model

    yield create
    for model in opened:
        model.close()
        model.env.close()


@pytest.mark.parametrize("interval", [2, 3])
@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("probability", [0., .5])
def test_online_loop_keeps_real_replay_and_outer_updates_on_held_decisions(
    held_models, monkeypatch, interval, auxiliary, probability,
):
    model = held_models(interval=interval, auxiliary=auxiliary, probability=probability)
    seen, after_steps, solve_calls = [], [], []
    original_act = model.agent.act
    original_solve = model.agent.inner_engine._act_rl
    original_log = model._log_wandb_step

    def record_act(*args, **kwargs):
        result = original_act(*args, **kwargs)
        seen.append({
            "step": model._global_step, "t0": kwargs["t0"],
            "outer_updates": model.agent.num_updates,
            "metrics": deepcopy(model.agent.last_inner_metrics),
        })
        return result

    def record_solve(*args, **kwargs):
        solve_calls.append(model._global_step)
        return original_solve(*args, **kwargs)

    def record_log(*args, **kwargs):
        after_steps.append((model._global_step, model.agent.num_updates))
        return original_log(*args, **kwargs)

    monkeypatch.setattr(model.agent, "act", record_act)
    monkeypatch.setattr(model.agent.inner_engine, "_act_rl", record_solve)
    monkeypatch.setattr(model, "_log_wandb_step", record_log)
    model.learn(total_timesteps=17)

    # The first complete five-step episode provides seed replay. Two further
    # complete episodes and a two-decision partial episode exercise both kinds
    # of block boundary; the last partial episode is not added as a full one.
    assert model._global_step == 17
    assert model.buffer.num_eps == 3
    assert model.buffer.total_transitions == 15
    assert model.agent.num_updates == model._num_updates == 13
    assert len(seen) == 12
    assert [row["step"] for row in seen if row["t0"]] == [5, 10, 15]
    assert [row["outer_updates"] for row in seen] == list(range(1, 13))
    assert dict(after_steps)[17] == 13

    expected_solves, episode_index = [], -1
    for row in seen:
        episode_index = 0 if row["t0"] else episode_index + 1
        solve = episode_index % interval == 0
        metrics = row["metrics"]
        assert metrics["inner_solve_performed"] == float(solve)
        assert metrics["inner_policy_held"] == float(not solve)
        assert metrics["inner_action_age"] == episode_index % interval
        assert metrics["inner_episode_decision_index"] == episode_index
        assert metrics["inner_solve_interval"] == interval
        if solve:
            expected_solves.append(row["step"])
            assert metrics["inner_critic_optimizer_steps"] == 2
            assert metrics["inner_actor_optimizer_steps"] == 2
            if probability:
                assert metrics["inner_critic_bernoulli_applied"] == float(episode_index > 0)
        else:
            assert all(metrics[key] == 0 for key in HELD_ZERO_WORK)
            assert metrics["inner_policy_evaluations"] == 1
    assert solve_calls == expected_solves
    # The shared-G schedule counts update slots, separately from each
    # optimizer's two actual steps.
    assert model._inner_updates_total == 2 * len(expected_solves)


@pytest.mark.parametrize("auxiliary", [False, True])
def test_held_behavior_uses_new_observation_and_fixed_actor_encoder(
    held_models, monkeypatch, auxiliary,
):
    agent = held_models(auxiliary=auxiliary, probability=.5).agent
    engine = agent.inner_engine
    agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    encoder = engine._held_encoder
    actor = engine._held_actor
    old_encoder, old_actor = deepcopy(encoder.state_dict()), deepcopy(actor.state_dict())
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert all(not parameter.requires_grad for parameter in actor.parameters())
    assert not encoder.training and not actor.training
    assert all(p.data_ptr() != dict(agent.model._encoder["state"].named_parameters())[name].data_ptr()
               for name, p in encoder.named_parameters())

    # Actual outer learning, followed by a deliberately large representation
    # mutation, must not change the in-flight observation-to-action mapping.
    agent._update(*_batch(agent))
    with torch.no_grad():
        for parameter in agent.model._encoder["state"].parameters():
            parameter.add_(.7)
        for parameter in agent.model._pi.parameters():
            parameter.add_(.4)
    observation = torch.tensor([.8, -.3, .2])
    with torch.no_grad():
        root = encoder(observation.unsqueeze(0))
        live_root = agent.model.encode(observation.unsqueeze(0))
        assert not torch.equal(root, live_root)
        _, expected = agent.model.pi(
            root, policy=actor, noise=torch.zeros(1, 1),
            log_std_mapping=agent.cfg.inner_log_std_mapping,
            log_std_min=agent.cfg.inner_log_std_min,
            log_std_max=agent.cfg.inner_log_std_max,
            std_scale=agent.cfg.inner_execution_std_scale,
        )
    outer_before = deepcopy(agent.model.state_dict())
    solve_before = _snapshot(engine)
    global_before = torch.random.get_rng_state().clone()

    def no_live_encoder(*_args, **_kwargs):
        raise AssertionError("A held action must use the solve-time encoder snapshot")

    monkeypatch.setattr(agent.model, "encode", no_live_encoder)
    actions = []
    for age in (1, 2):
        action, behavior = agent.act(
            observation, collect_diagnostics=False, return_behavior_policy=True,
        )
        actions.append(action)
        assert agent.last_inner_metrics["inner_action_age"] == age
        torch.testing.assert_close(behavior["pre_tanh_mean"], expected["pre_tanh_mean"][0], rtol=0, atol=0)
        torch.testing.assert_close(behavior["log_std"], expected["log_std"][0], rtol=0, atol=0)
        assert all(agent.last_inner_metrics[key] == 0 for key in HELD_ZERO_WORK)
    assert not torch.equal(actions[0], actions[1]), "Held policy must draw fresh execution noise"
    _assert_tree_equal(encoder.state_dict(), old_encoder)
    _assert_tree_equal(actor.state_dict(), old_actor)
    _assert_tree_equal(agent.model.state_dict(), outer_before)
    torch.testing.assert_close(torch.random.get_rng_state(), global_before, rtol=0, atol=0)
    solve_after = _snapshot(engine)
    for state in (solve_before, solve_after):
        state.pop("rng")
    _assert_tree_equal(solve_after, solve_before)


def test_held_behavior_metadata_reaches_each_real_replay_row(held_models):
    model = held_models(
        outer_behavior_policy_kl_schedule="smooth",
        outer_behavior_policy_kl_min_valid_count=1,
    )
    observation, _ = model.env.reset(seed=31)
    obs_t = model._obs_to_tensor(observation)
    row = model._start_episode_staging(obs_t)
    means = []
    for decision in range(3):
        action = model._act_agent(obs_t, t0=decision == 0, eval_mode=False)
        behavior = deepcopy(model._pending_behavior_policy)
        observation, reward, _, _, _ = model.env.step(model._unscale_action(action.numpy()))
        obs_t = model._obs_to_tensor(observation)
        model._stage_transition(row, obs_t, action.numpy(), reward, False)
        assert model._pending_behavior_policy is None
        assert bool(model._episode_staging["behavior_policy_valid"][row])
        torch.testing.assert_close(model._episode_staging["behavior_pre_tanh_mean"][row], behavior["pre_tanh_mean"])
        torch.testing.assert_close(model._episode_staging["behavior_log_std"][row], behavior["log_std"])
        means.append(behavior["pre_tanh_mean"])
        row += 1
    assert any(not torch.equal(mean, means[0]) for mean in means[1:])


@pytest.mark.parametrize("probability", [0., .5])
def test_paired_eval_does_not_consume_training_hold_or_rng(held_models, probability):
    observed = held_models(probability=probability)
    control = held_models(probability=probability)
    for model in (observed, control):
        model.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    agent, live = observed.agent, observed.agent.inner_engine
    before = _snapshot(live)
    actor_before = deepcopy(live._held_actor.state_dict())
    encoder_before = deepcopy(live._held_encoder.state_dict())
    old_metrics = deepcopy(agent.last_inner_metrics)
    evaluator = PairedControllerEvaluator(
        agent, lambda: gym.make("Pendulum-v1", max_episode_steps=4),
        lambda obs: torch.as_tensor(obs, dtype=torch.float32), lambda action: action * 2,
        episodes=2, seed=18, device="cpu",
    )
    try:
        metrics = evaluator.evaluate()
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
        prefix = "eval/paired_bernoulli_inner" if probability else "eval/paired_fresh_inner"
        assert metrics[f"{prefix}_actions"] == 8
        assert metrics[f"{prefix}_solve_decisions"] == 4
        assert metrics[f"{prefix}_held_decisions"] == 4
        assert metrics[f"{prefix}_solve_interval"] == 3
        assert metrics[f"{prefix}_fixed_target_q_action_gain_count"] == 4
        assert agent.inner_engine is live
        _assert_tree_equal(_snapshot(live), before)
        _assert_tree_equal(live._held_actor.state_dict(), actor_before)
        _assert_tree_equal(live._held_encoder.state_dict(), encoder_before)
        _assert_tree_equal(agent.last_inner_metrics, old_metrics)
        for index in range(1, 5):
            obs = torch.ones(3) * index / 10
            expected = control.agent.act(obs, collect_diagnostics=False)
            actual = agent.act(obs, collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert agent.last_inner_metrics["inner_action_age"] == index % 3
    finally:
        evaluator.close()


def test_mixed_training_window_counts_solves_holds_and_actual_work(held_models):
    model = held_models()
    model._reset_wandb_window()
    metrics = []
    for index in range(5):
        model.agent.act(torch.zeros(3), t0=index == 0)
        metrics.append(deepcopy(model.agent.last_inner_metrics))
        model._record_action_metrics(planned=True, action_seconds=.1)
    payload = model._wandb_train_window.pop()
    assert payload["train/inner_solve_performed"] == 2
    assert payload["train/inner_policy_held"] == 3
    assert payload["train/inner_critic_optimizer_steps"] == 4
    assert payload["train/inner_actor_optimizer_steps"] == 4
    assert payload["train/inner_updates"] == 4
    for source in ("inner_solve_seconds", "inner_held_seconds", "inner_encoder_snapshot_seconds"):
        assert payload[f"time/{source}"] == pytest.approx(sum(row.get(source, 0.) for row in metrics))


def test_explicit_interval_one_preserves_actions_and_exact_inner_state(held_models):
    baseline = _tiny_model(inner_rounds=1, inner_updates_per_round=2, inner_finite_horizon=True,
                           inner_execution_action="policy_sample", outer_critic_target="entropy_augmented")
    explicit = held_models(interval=1)
    try:
        for index in range(4):
            obs = torch.ones(3) * index / 10
            expected = baseline.agent.act(obs, t0=index == 0, collect_diagnostics=False)
            actual = explicit.agent.act(obs, t0=index == 0, collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _assert_tree_equal(_snapshot(explicit.agent.inner_engine), _snapshot(baseline.agent.inner_engine))
    finally:
        baseline.close()
        baseline.env.close()


@pytest.mark.parametrize("auxiliary", [False, True])
def test_strict_compiled_solves_reuse_graphs_across_holds_and_outer_changes(
    held_models, monkeypatch, auxiliary,
):
    """Recording backend checks capture boundaries; CUDA gate checks Inductor."""
    torch._dynamo.reset()
    compile_original = torch.compile
    captures = []

    def compile_for_test(function, **kwargs):
        def backend(graph, _inputs):
            captures.append(function.__name__)
            return graph.forward
        return compile_original(function, backend=backend, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_for_test)
    options = dict(auxiliary=auxiliary, probability=.5, dropout=.1,
                   critic_ensemble_backend="vmap")
    eager = held_models(**options).agent
    compiled = held_models(**options, compile=True, compile_strict=True).agent
    try:
        for index in range(7):
            if index:
                # Real full updates change the shared encoder, actor, and
                # critics during held blocks as well as between solves.
                outer_rng = torch.random.get_rng_state().clone()
                for agent in (eager, compiled):
                    torch.random.set_rng_state(outer_rng)
                    agent._update(*_batch(agent))
            before_capture = len(captures)
            observation = torch.ones(3) * index / 10
            expected = eager.act(observation, t0=index == 0, collect_diagnostics=False)
            actual = compiled.act(observation, t0=index == 0, collect_diagnostics=False)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
            assert compiled.last_inner_metrics["inner_compile_fallback"] == 0
            assert compiled.last_inner_metrics["inner_action_age"] == index % 3
            if index:
                assert len(captures) == before_capture, "A hold or subsequent solve recaptured a graph"
            else:
                assert {"_dense_rollout_kernel", "_sac_critic_kernel", "_sac_actor_kernel"} <= set(captures)
    finally:
        torch._dynamo.reset()
