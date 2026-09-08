"""Causal depth-wise XQC updates and fresh frozen-checkpoint evaluation."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.inner_xqc import InnerXQCEngine
from RL.tdmpc2_core.xqc_controller import LatentXQCConfig
from test_ambixqc_core import _batch, _tree_equal
from test_ambixqc_inner import _agent, _FakeWorkspace
from test_ambixqc_inner_j6 import BUDGET, deterministic_xqc_numerics
from test_ambixqc_prior_checkpoint import _wrapper


def _step_engine(*, horizon=3, updates=3, episodic=False, reward_mode="frozen_real_scale"):
    agent, outer = _agent(episodic=episodic, reward_mode=reward_mode)
    cfg = agent.cfg
    cfg.inner_update_timing = "step"
    cfg.inner_policy_delay = 1
    cfg.inner_rollout_horizon = horizon
    cfg.inner_updates_per_round = updates
    cfg.inner_replay_capacity = cfg.inner_model_step_budget = 2 * 2 * horizon
    outer.config = LatentXQCConfig(policy_delay=3)

    def clone(**kwargs):
        outer.clone_calls += 1
        assert kwargs["transition_steps"] == 2 * updates
        workspace = _FakeWorkspace(outer, deepcopy(outer.config))
        update = workspace.update

        def checked_update(batch, *, outer_terminal_mask=None, outer_controller=None, **kwargs):
            if outer_terminal_mask is not None:
                assert outer_controller is outer
                assert not (outer_terminal_mask & ~batch.bootstrap_mask.bool().reshape(-1)).any()
            return update(batch, **kwargs)

        workspace.update = checked_update
        return workspace

    outer.clone_for_inner = clone
    return InnerXQCEngine(agent), outer


@pytest.mark.parametrize("updates", [3, 9])
def test_each_depth_commits_replay_then_updates_actor_before_continuing_branches(monkeypatch, updates):
    engine, outer = _step_engine(updates=updates)
    sample, update, sample_batch = engine._sample_actor, engine._update_slot, engine._sample_batch
    events, collections = [], []

    def collect(z, *, stream, **kwargs):
        action = sample(z, stream=stream, **kwargs)
        if stream == "collection":
            workspace = engine.state.workspace
            events.append(("collect", workspace.update_step))
            collections.append((z.clone(), action.clone(), workspace.actor_optimizer_steps))
        return action

    def slot():
        assert torch.is_grad_enabled()  # No rollout no_grad context spans optimization.
        events.append(("update", engine.state.replay.next_sample_id))
        return update()

    def available_batch():
        batch = sample_batch()
        assert (batch["sample_ids"] < engine.state.replay.next_sample_id).all()
        assert not batch["z"].requires_grad and not batch["next_z"].requires_grad
        return batch

    monkeypatch.setattr(engine, "_sample_actor", collect)
    monkeypatch.setattr(engine, "_update_slot", slot)
    monkeypatch.setattr(engine, "_sample_batch", available_batch)
    outer_before = deepcopy(outer.actor.state_dict())
    action, metrics, _ = engine.act(torch.zeros(1, 2), eval_mode=True)
    slots_per_depth = updates // 3
    expected = []
    for depth in range(6):
        expected.append(("collect", depth * slots_per_depth))
        expected.extend([("update", 2 * (depth + 1))] * slots_per_depth)
    assert events == expected
    for depth, (z, collected_action, actor_steps) in enumerate(collections):
        assert actor_steps == depth * slots_per_depth
        torch.testing.assert_close(collected_action, torch.full_like(collected_action, np.tanh(.1 * actor_steps)))
        # New rounds restart at the real root; intermediate depths continue the
        # same branches' next states, rather than generating new root rollouts.
        assert torch.equal(z, torch.full_like(z, .25 * (depth % 3)))
    assert action.item() == pytest.approx(np.tanh(.1 * 2 * updates))
    assert metrics["inner_policy_delay"] == 1
    assert metrics["inner_update_timing_step"] == 1
    assert metrics["inner_updates_per_rollout_step"] == slots_per_depth
    assert metrics["inner_collection_steps"] == 6
    assert tuple(metrics[f"inner_{name}_optimizer_steps"] for name in ("critic", "actor", "temperature")) == (2 * updates,) * 3
    assert outer.config.policy_delay == 3
    assert _tree_equal(outer_before, outer.actor.state_dict())


@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("all_terminated", [False, True])
def test_step_terminal_sidecar_tracks_actual_appends_and_stops_dead_branches(monkeypatch, horizon, all_terminated):
    engine, _ = _step_engine(horizon=horizon, updates=horizon, episodic=True)
    engine.cfg.inner_terminal_bootstrap = "outer"
    calls = []

    def termination(z):
        result = z.new_zeros(z.shape[0], 1)
        if z.shape[0] == 2:  # First depth of each fresh-root round.
            result[0] = 1
        if all_terminated:
            result.fill_(1)
        calls.append(z.shape[0])
        return result

    monkeypatch.setattr(engine.model, "termination", termination)
    sample_batch = engine._sample_batch
    sampled_masks = []
    all_flags = torch.tensor(
        [False] * 4 if all_terminated else ([False] * horizon + [True]) * 2
    )

    def checked_sample():
        raw = sample_batch()
        size = engine.state.replay.next_sample_id
        assert torch.equal(engine.state.outer_terminal_flags[:size], all_flags[:size])
        expected = all_flags[raw["sample_ids"]]
        assert torch.equal(raw["outer_terminal_mask"], expected)
        sampled_masks.append(expected.clone())
        return raw

    monkeypatch.setattr(engine, "_sample_batch", checked_sample)
    _, metrics, _ = engine.act(torch.zeros(1, 2), eval_mode=True)
    expected_calls = [2] * 2 if all_terminated else ([2] + [1] * (horizon - 1)) * 2
    assert calls == expected_calls
    steps = 2 if all_terminated else 2 * horizon
    rows = 4 if all_terminated else 2 * (horizon + 1)
    assert metrics["inner_collection_steps"] == steps
    assert metrics["inner_model_steps"] == rows
    assert metrics["inner_critic_optimizer_steps"] == metrics["inner_actor_optimizer_steps"] == steps
    assert metrics["inner_requested_update_slots"] == 2 * horizon
    assert metrics["inner_outer_terminal_boundary_rows"] == (0 if all_terminated else 2)
    assert metrics["inner_outer_terminal_bootstrap_rows"] == sum(int(mask.sum()) for mask in sampled_masks)
    replay = engine._replay_pool
    assert replay.size == replay.next_sample_id == rows
    assert int(replay.terminated[:rows].sum()) == (4 if all_terminated else 2)


def test_adaptive_return_scale_consumes_each_depth_before_its_update(monkeypatch):
    engine, _ = _step_engine(reward_mode="action_local_imagined")
    real_before = deepcopy(vars(engine.agent.reward_normalizer))
    seen = []
    update = engine._update_slot

    def slot():
        normalizer = engine.state.reward_normalizer
        seen.append((float(normalizer.count), float(normalizer.mean)))
        return update()

    monkeypatch.setattr(engine, "_update_slot", slot)
    _, metrics, _ = engine.act(torch.zeros(1, 2), eval_mode=True)
    # Each round's independent branches start at the frozen real accumulator4.
    branch_returns = [4.6, 5.14, 5.626] * 2
    assert [count for count, _ in seen] == [2, 4, 6, 8, 10, 12]
    np.testing.assert_allclose([mean for _, mean in seen], np.cumsum(branch_returns) / np.arange(1, 7), rtol=1e-6)
    assert metrics["inner_reward_normalizer_imagined_updates"] == 12
    assert vars(engine.agent.reward_normalizer) == real_before


def test_collection_timer_excludes_interleaved_optimization(monkeypatch):
    engine, _ = _step_engine()
    clock, elapsed = [0.0], {}
    dynamics, update = engine.model.next_from_joint, engine._update_slot

    def model_step(joint):
        clock[0] += 1
        return dynamics(joint)

    def slot():
        clock[0] += 100
        return update()

    monkeypatch.setattr(engine, "_timer_start", lambda: clock[0])
    monkeypatch.setattr(engine, "_timer_stop", lambda key, start: elapsed.setdefault(key, []).append(clock[0] - start))
    monkeypatch.setattr(engine.model, "next_from_joint", model_step)
    monkeypatch.setattr(engine, "_update_slot", slot)
    engine.act(torch.zeros(1, 2), eval_mode=True)
    assert elapsed["inner_rollout_seconds"] == [1.0] * 6
    assert elapsed["inner_update_seconds"] == [100.0] * 6


def test_missing_timing_retains_exact_round_order_and_inherited_delay():
    implicit_agent, implicit_outer = _agent(oracle_rollout=True)
    explicit_agent, explicit_outer = _agent(oracle_rollout=True)
    explicit_agent.cfg.inner_update_timing = "round"
    implicit, explicit = InnerXQCEngine(implicit_agent), InnerXQCEngine(explicit_agent)
    first, first_metrics, first_ids = implicit.act(torch.zeros(1, 2), eval_mode=True)
    second, second_metrics, second_ids = explicit.act(torch.zeros(1, 2), eval_mode=True)
    assert torch.equal(first, second)
    assert _tree_equal(implicit_outer.records, explicit_outer.records)
    assert _tree_equal(first_ids, second_ids)
    assert _tree_equal(implicit.rng.training_state_dict(), explicit.rng.training_state_dict())
    for metrics in (first_metrics, second_metrics):
        assert metrics["inner_policy_delay"] == 3
        assert tuple(metrics[f"inner_{name}_optimizer_steps"] for name in ("critic", "actor", "temperature")) == (4, 2, 2)
        assert "inner_update_timing_step" not in metrics


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware is unavailable"))])
def test_full_j6_step_budget_has_fresh_learners_and_exact_seeded_frozen_actions(
    device, tmp_path, monkeypatch, deterministic_xqc_numerics
):
    source = _wrapper(device=device, inner_operator="none", xqc_optimizer_backend="auto", train_unroll_horizon=3)
    target = _wrapper(device=device, xqc_optimizer_backend="auto", train_unroll_horizon=3,
                      inner_terminal_bootstrap="outer", inner_update_timing="step",
                      inner_policy_delay=1, **BUDGET)
    try:
        source.agent.observe_reward(2., False, False)
        source.agent.observe_reward(3., False, True)
        source.agent._update(*(tensor.to(device) for tensor in _batch(source.agent)))
        checkpoint = tmp_path / "trained-prior.pt"
        source.agent.save(str(checkpoint))
        target.load(str(checkpoint), frozen_evaluation=True)
        agent, engine = target.agent, target.agent.inner_engine
        before = agent.frozen_outer_state()
        observation, _ = target.env.reset(seed=101)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(agent.device).clone() if device == "cuda" else None
        prepare, dynamics = engine._prepare_action, agent.model.next_from_joint
        preparations, chronology, model_rows = [], [], []
        instrumented = set()

        def checked_prepare():
            prepare()
            state, workspace = engine.state, engine.state.workspace
            local = workspace.controller
            assert local.config.policy_delay == 1 and agent.xqc_controller.config.policy_delay == 3
            assert workspace.update_step == workspace.actor_optimizer_steps == workspace.temperature_optimizer_steps == 0
            assert state.replay.size == state.replay.next_sample_id == 0
            assert not state.outer_terminal_flags.any()
            assert state.reward_normalizer is None
            assert state.reward_scale == agent.reward_normalizer.scale
            for name in ("actor", "critic"):
                assert _tree_equal(getattr(local, name).state_dict(), getattr(agent.xqc_controller, name).state_dict())
            assert _tree_equal(local.critic_target.state_dict(), agent.xqc_controller.critic.state_dict())
            assert torch.equal(local.log_temperature, agent.xqc_controller.log_temperature)
            preparations.append(workspace)
            for component, optimizer in (("critic", workspace.critic_optimizer),
                                         ("actor", workspace.actor_optimizer),
                                         ("temperature", workspace.temperature_optimizer)):
                assert not optimizer.state
                if id(optimizer) not in instrumented:
                    instrumented.add(id(optimizer))
                    original = optimizer.step

                    def counted_step(*args, _step=original, _component=component, **kwargs):
                        chronology.append(_component)
                        return _step(*args, **kwargs)

                    monkeypatch.setattr(optimizer, "step", counted_step)

        def counted_dynamics(joint):
            chronology.append("model")
            model_rows.append(joint.shape[0])
            return dynamics(joint)

        monkeypatch.setattr(engine, "_prepare_action", checked_prepare)
        monkeypatch.setattr(agent.model, "next_from_joint", counted_dynamics)

        def decision():
            chronology.clear()
            model_rows.clear()
            action, _ = target.predict(observation, deterministic=True)
            metrics = agent.last_inner_metrics
            assert chronology == ["model", "critic", "actor", "temperature"] * 18
            assert model_rows == [512] * 18
            assert metrics["inner_model_steps"] == metrics["inner_replay_draws"] == metrics["inner_buffer_size"] == 9216
            assert metrics["inner_collection_steps"] == 18
            assert metrics["inner_policy_delay"] == metrics["inner_updates_per_rollout_step"] == metrics["inner_update_timing_step"] == 1
            assert tuple(metrics[f"inner_{name}_optimizer_steps"] for name in ("critic", "actor", "temperature")) == (18, 18, 18)
            assert metrics["inner_outer_terminal_boundary_rows"] == 3072
            assert metrics["inner_outer_terminal_policy_evaluations"] == metrics["inner_outer_terminal_q_evaluations"] == 9216
            assert metrics["inner_reward_normalizer_imagined_updates"] == metrics["inner_reward_scale_delta"] == 0
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
            pool = engine._workspace_pool
            assert engine.state.workspace is None and engine.state.replay is None
            assert not _tree_equal(pool.controller.actor.state_dict(), agent.xqc_controller.actor.state_dict())
            with torch.no_grad():
                z = agent.model.encode(target._obs_to_tensor(observation).to(agent.device).unsqueeze(0))
                mean, _ = pool.controller.actor.distribution(z, bn_mode="running")
            np.testing.assert_array_equal(action, target._unscale_action(mean.tanh()[0].cpu().numpy()))
            agent.observe_reward(100., True, False)
            assert _tree_equal(before, agent.frozen_outer_state())
            return action

        decision()  # Allocation warmup cannot affect the seeded scored sequence.

        def episode(seed, *, reuse=True):
            target.reset_for_evaluation(seed, reuse_action_pool=reuse)
            assert engine.rng.generator("collection").device.type == device
            assert engine.action_index == 0
            result = np.stack([decision(), decision()])
            assert engine.action_index == 2
            return result

        first = episode(12345)
        first_rng = deepcopy(engine.rng.training_state_dict())
        assert len({id(workspace) for workspace in preparations}) == 1
        alternate = episode(12346)
        assert not np.array_equal(first, alternate)
        np.testing.assert_array_equal(first, episode(12345))
        assert _tree_equal(first_rng, engine.rng.training_state_dict())
        np.testing.assert_array_equal(first, episode(12345, reuse=False))
        assert _tree_equal(first_rng, engine.rng.training_state_dict())
        assert len({id(workspace) for workspace in preparations}) == 2
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(agent.device))
    finally:
        source.env.close()
        target.env.close()
