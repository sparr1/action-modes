"""SAC frequency scheduling exercises real solves, gradients, replay and traces."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_inner_step_updates import _snapshot
from tests.test_ambi_root_local_sac import _tiny_model


def _frequency_model(**overrides):
    options = dict(
        inner_rounds=3, inner_rollouts_per_round=4, inner_rollout_horizon=2,
        inner_replay_capacity=24, inner_batch_size=4, inner_updates_per_round=3,
        inner_actor_update_interval=2, inner_critic_target_update_interval=3,
        inner_finite_horizon=True, inner_actor_lr=.001, inner_temperature_lr=.001,
    )
    options.update(overrides)
    return _tiny_model(**options)


def _record_updates(monkeypatch, engine):
    """Observe completed updates while leaving the actual optimizer work intact."""
    observed = dict(events=[], critics=[], actors=[], samples=[], temperatures=[])
    critic = engine._sac_critic_step
    policy = engine._sac_policy_step
    targets = engine._maybe_update_targets
    sample = engine._sample_batch

    def sample_batch(*args, **kwargs):
        batch = sample(*args, **kwargs)
        observed["samples"].append(batch)
        return batch

    def critic_step(batch, alpha, **kwargs):
        observed["critics"].append(dict(
            batch=batch, alpha=float(alpha),
            actor=torch.cat([p.detach().flatten() for p in engine.state.actor.parameters()]).clone(),
        ))
        result = critic(batch, alpha, **kwargs)
        observed["events"].append(("critic", engine.state.critic_lifetime_steps))
        return result

    def policy_step(batch, **kwargs):
        observed["actors"].append((engine.state.critic_lifetime_steps, batch))
        assert kwargs["update_actor"]
        before = engine.state.temperature_steps
        result = policy(batch, **kwargs)
        observed["events"].append(("actor", engine.state.critic_lifetime_steps))
        if engine.state.temperature_steps > before:
            observed["temperatures"].append(engine.state.critic_lifetime_steps)
        return result

    def target_step(**kwargs):
        before = engine.state.critic_target_steps
        result = targets(**kwargs)
        if engine.state.critic_target_steps > before:
            observed["events"].append(("target", engine.state.critic_lifetime_steps))
        return result

    monkeypatch.setattr(engine, "_sample_batch", sample_batch)
    monkeypatch.setattr(engine, "_sac_critic_step", critic_step)
    monkeypatch.setattr(engine, "_sac_policy_step", policy_step)
    monkeypatch.setattr(engine, "_maybe_update_targets", target_step)
    return observed


@pytest.mark.parametrize("sampling", ["with_replacement", "without_replacement"])
@pytest.mark.parametrize("critic_variant", ["scalar", "aux_return_distributional"])
def test_frequency_carries_across_rounds_resets_each_action_and_shares_batches(
    monkeypatch, sampling, critic_variant,
):
    options = dict(inner_replay_sampling=sampling)
    if critic_variant == "aux_return_distributional":
        options.update(
            q_representation="distributional", aux_return_mode="sac",
            inner_critic_source="aux_return", inner_horizon_critic_source="aux_return",
            inner_sac_critic_target="reward_only",
        )
    holder = _frequency_model(**options)
    engine = holder.agent.inner_engine
    recorded = _record_updates(monkeypatch, engine)
    outer = deepcopy(holder.agent.model.state_dict())
    global_rng = torch.random.get_rng_state().clone()
    expected_events = [
        ("critic", 1), ("critic", 2), ("actor", 2), ("critic", 3), ("target", 3),
        ("critic", 4), ("actor", 4), ("critic", 5),
        ("critic", 6), ("actor", 6), ("target", 6),
        ("critic", 7), ("critic", 8), ("actor", 8), ("critic", 9), ("target", 9),
    ]
    try:
        # A second decision in the same episode must restart cadence at critic 1.
        for decision in range(2):
            for values in recorded.values():
                values.clear()
            trace = InnerActionTrace()
            holder.agent.act(torch.tensor([.7, .3, -.2]), t0=decision == 0,
                             eval_mode=True, trace=trace)
            assert recorded["events"] == expected_events
            assert recorded["temperatures"] == [2, 4, 6, 8]
            assert len(recorded["samples"]) == len(recorded["critics"]) == 9
            assert len({id(batch) for batch in recorded["samples"]}) == 9
            for step, batch in recorded["actors"]:
                assert batch is recorded["critics"][step - 1]["batch"]
            for index, item in enumerate(recorded["critics"]):
                assert item["batch"] is recorded["samples"][index]
            sample_ids = [item["batch"]["sample_ids"] for item in recorded["critics"]]
            assert any(not torch.equal(a, b) for a, b in zip(sample_ids, sample_ids[1:]))

            # Two critic steps see each policy/alpha; adaptation affects the next step.
            critics = recorded["critics"]
            assert torch.equal(critics[0]["actor"], critics[1]["actor"])
            assert not torch.equal(critics[1]["actor"], critics[2]["actor"])
            assert critics[0]["alpha"] == critics[1]["alpha"]
            assert critics[1]["alpha"] != critics[2]["alpha"]

            updates = [event for event in trace.events if event["phase"] == "update"]
            assert [event["round_index"] for event in updates] == [1] * 3 + [2] * 3 + [3] * 3
            assert [event["critic_updates"] for event in updates] == list(range(1, 10))
            assert [event["actor_updates"] for event in updates] == [0, 1, 1, 2, 2, 3, 3, 4, 4]
            assert [event["temperature_updates"] for event in updates] == [0, 1, 1, 2, 2, 3, 3, 4, 4]
            assert [event["updated_actor"] for event in updates] == [False, True, False, True, False, True, False, True, False]
            for index, event in enumerate(updates):
                assert event["updated_critic"] is True
                assert event["updated_temperature"] == event["updated_actor"]
                assert event["metrics"]["alpha_used"] == pytest.approx(critics[index]["alpha"])
                assert ("actor_loss" in event["metrics"]) == event["updated_actor"]
                assert ("temperature_loss" in event["metrics"]) == event["updated_temperature"]

            metrics = holder.agent.last_inner_metrics
            assert metrics["inner_critic_optimizer_steps"] == 9
            assert metrics["inner_actor_optimizer_steps"] == metrics["inner_temperature_optimizer_steps"] == 4
            assert metrics["inner_critic_target_updates"] == 3
            assert metrics["inner_replay_draws"] == 9 * 4
            assert metrics["inner_requested_update_slots"] == 9
            assert metrics["inner_model_steps"] == metrics["inner_buffer_size"] == 24
            replay = engine._action_pool.replay
            assert replay.size == replay.next_sample_id == 24
            torch.testing.assert_close(replay.sample_id[:24], torch.arange(24))
            _assert_tree_equal(holder.agent.model.state_dict(), outer)
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        holder.close()


@pytest.mark.parametrize("critic_count,actor_interval,expected_actors,expected_targets", [
    (20, 1, 20, 10), (20, 5, 4, 10), (10, 1, 10, 5),
])
def test_article_schedule_examples_run_exact_update_counts(
    monkeypatch, critic_count, actor_interval, expected_actors, expected_targets,
):
    holder = _frequency_model(
        inner_rounds=1, inner_updates_per_round=critic_count,
        inner_actor_update_interval=actor_interval, inner_critic_target_update_interval=2,
    )
    recorded = _record_updates(monkeypatch, holder.agent.inner_engine)
    try:
        holder.agent.act(torch.zeros(3), t0=True, eval_mode=True)
        metrics = holder.agent.last_inner_metrics
        assert metrics["inner_critic_optimizer_steps"] == critic_count
        assert metrics["inner_actor_optimizer_steps"] == expected_actors
        assert metrics["inner_temperature_optimizer_steps"] == expected_actors
        assert metrics["inner_critic_target_updates"] == expected_targets
        assert metrics["inner_replay_draws"] == critic_count * 4
        assert [step for kind, step in recorded["events"] if kind == "actor"] == list(range(actor_interval, critic_count + 1, actor_interval))
        assert [step for kind, step in recorded["events"] if kind == "target"] == list(range(2, critic_count + 1, 2))
    finally:
        holder.close()


def test_frequency_one_preserves_default_shared_sac_exactly_and_trace_is_observational():
    default = _frequency_model(inner_actor_update_interval=None)
    frequency = _frequency_model(inner_actor_update_interval=1)
    try:
        for decision in range(2):
            kwargs = dict(t0=decision == 0, eval_mode=True)
            expected = default.agent.act(torch.tensor([.3, -.4, .2]), **kwargs)
            actual = frequency.agent.act(torch.tensor([.3, -.4, .2]),
                                         trace=InnerActionTrace(), **kwargs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _assert_tree_equal(_snapshot(frequency.agent), _snapshot(default.agent))
    finally:
        default.close()
        frequency.close()


@pytest.mark.parametrize("temperature_mode", ["fixed", "inherit_outer"])
def test_actor_frequency_does_not_enable_fixed_temperature_updates(temperature_mode):
    holder = _frequency_model(inner_temperature_mode=temperature_mode, inner_temperature=.1)
    try:
        trace = InnerActionTrace()
        holder.agent.act(torch.zeros(3), trace=trace)
        metrics = holder.agent.last_inner_metrics
        assert metrics["inner_actor_optimizer_steps"] == 4
        assert metrics["inner_temperature_optimizer_steps"] == 0
        assert all(not event["updated_temperature"] for event in trace.events if event["phase"] == "update")
    finally:
        holder.close()


def test_entropy_disabled_keeps_frequency_actor_updates_without_temperature():
    holder = _frequency_model(
        aux_return_mode="sac", inner_critic_source="aux_return",
        inner_horizon_critic_source="aux_return", inner_entropy_enabled=False,
    )
    try:
        trace = InnerActionTrace()
        holder.agent.act(torch.zeros(3), trace=trace)
        metrics = holder.agent.last_inner_metrics
        assert metrics["inner_actor_optimizer_steps"] == 4
        assert metrics["inner_temperature_optimizer_steps"] == 0
        assert all(not event["updated_temperature"] for event in trace.events if event["phase"] == "update")
    finally:
        holder.close()


def test_actor_interval_longer_than_solve_updates_only_critic_and_target(monkeypatch):
    holder = _frequency_model(inner_actor_update_interval=10)
    recorded = _record_updates(monkeypatch, holder.agent.inner_engine)
    try:
        trace = InnerActionTrace()
        holder.agent.act(torch.zeros(3), trace=trace)
        metrics = holder.agent.last_inner_metrics
        assert metrics["inner_critic_optimizer_steps"] == 9
        assert metrics["inner_critic_target_updates"] == 3
        assert metrics["inner_actor_optimizer_steps"] == metrics["inner_temperature_optimizer_steps"] == 0
        assert metrics["inner_replay_draws"] == 9 * 4
        assert recorded["actors"] == recorded["temperatures"] == []
        assert all(torch.equal(item["actor"], recorded["critics"][0]["actor"])
                   for item in recorded["critics"])
        assert all(not event["updated_actor"] and not event["updated_temperature"]
                   for event in trace.events if event["phase"] == "update")
    finally:
        holder.close()


def test_delayed_actor_is_used_in_later_critic_bootstraps(monkeypatch):
    holder = _frequency_model()
    engine = holder.agent.inner_engine
    critic = engine._sac_critic_step
    pi = engine.model.pi
    initial_actor = None
    in_critic = False
    bootstraps = []

    def critic_step(batch, alpha, **kwargs):
        nonlocal in_critic, initial_actor
        if initial_actor is None:
            initial_actor = deepcopy(engine.state.actor)
        in_critic = True
        try:
            return critic(batch, alpha, **kwargs)
        finally:
            in_critic = False

    def policy(z, **kwargs):
        result = pi(z, **kwargs)
        if in_critic and kwargs.get("policy") is engine.state.actor:
            # Same actual bootstrap states and sampled noise isolate the effect
            # of adaptation from changes to replay sampling or policy randomness.
            before = pi(z, **{**kwargs, "policy": initial_actor})
            bootstraps.append((engine.state.actor_steps,
                               result[0].detach().clone(), before[0].detach().clone()))
        return result

    monkeypatch.setattr(engine, "_sac_critic_step", critic_step)
    monkeypatch.setattr(engine.model, "pi", policy)
    try:
        holder.agent.act(torch.tensor([.7, .3, -.2]), t0=True, eval_mode=True)
        assert [steps for steps, _, _ in bootstraps] == [0, 0, 1, 1, 2, 2, 3, 3, 4]
        for steps, actual, initial in bootstraps:
            if steps == 0:
                torch.testing.assert_close(actual, initial, rtol=0, atol=0)
            else:
                assert not torch.equal(actual, initial)
    finally:
        holder.close()


def test_frequency_compiled_graphs_reuse_and_preserve_events(monkeypatch):
    torch._dynamo.reset()
    graphs = []
    compile_function = torch.compile

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    monkeypatch.setattr(torch, "compile", lambda function, **kwargs:
                        compile_function(function, backend=backend, **kwargs))
    eager = _frequency_model()
    compiled = _frequency_model(compile=True, compile_strict=True)
    try:
        count = None
        for decision in range(2):
            eager_trace, compiled_trace = InnerActionTrace(), InnerActionTrace()
            expected = eager.agent.act(torch.zeros(3), t0=decision == 0, eval_mode=True,
                                       trace=eager_trace)
            actual = compiled.agent.act(torch.zeros(3), t0=decision == 0, eval_mode=True,
                                        trace=compiled_trace)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert [event for event in compiled_trace.events] == eager_trace.events
            assert compiled.agent.last_inner_metrics["inner_compile_fallback"] == 0
            assert graphs
            if count is not None:
                assert len(graphs) == count
            count = len(graphs)
    finally:
        eager.close()
        compiled.close()
        torch._dynamo.reset()
