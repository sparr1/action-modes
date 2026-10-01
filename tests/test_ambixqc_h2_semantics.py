"""H2 activates the inner target while retaining the frozen return-only tail."""

from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.xqc_controller import LatentXQCBatch
from test_ambixqc_core import _tree_equal
from test_ambixqc_inner_j6 import deterministic_xqc_numerics
from test_ambixqc_outer_terminal import _controller, _target_oracle
from test_ambixqc_prior_checkpoint import _wrapper
from test_ambixqc_update_ratio import _Replay


def _buffers(module):
    return deepcopy(dict(module.named_buffers()))


def _mixed_case():
    torch.manual_seed(14)
    inner, outer, auxiliary = _controller(), _controller(), _controller().critic
    batch = LatentXQCBatch(
        latents=torch.tensor([[.1, .3], [-.2, .7], [.9, -.4], [.5, .5]]),
        actions=torch.tensor([[.2], [-.8], [.5], [0.]]),
        rewards=torch.tensor([.8, -.6, .5, 6.]),
        next_latents=torch.tensor([[.4, .1], [.1, .8], [-.6, .2], [.3, -.1]]),
        bootstrap_mask=torch.tensor([1., 1., 0., 0.]), discount=.9,
    )
    noise = torch.tensor([[.2], [-.9], [.1], [.8]])
    kwargs = dict(next_noise=noise, reward_scale=2.5,
                  outer_terminal_mask=torch.tensor([False, True, False, True]),
                  outer_controller=outer, outer_critic=auxiliary,
                  outer_critic_is_return=True, critic_target_kind="reward_only",
                  critic_bn_mode="running")
    return inner, outer, auxiliary, batch, kwargs


def test_h2_mixed_reward_only_target_matches_independent_categorical_oracle(monkeypatch):
    inner, outer, auxiliary, batch, kwargs = _mixed_case()
    before = deepcopy((inner.state_dict(), outer.state_dict(), auxiliary.state_dict()))
    rng = torch.get_rng_state().clone()
    count, support = batch.actions.shape[0], inner.critic.support
    # Independent actions and scalar-loop atom projection, rather than calling
    # the learner's target builder to construct its own expected result.
    with torch.no_grad():
        mean, log_std = inner.actor.distribution(batch.next_latents, bn_mode="running")
        inner_next_actions = (mean + log_std.exp() * kwargs["next_noise"]).tanh()
        mean, log_std = outer.actor.distribution(batch.next_latents, bn_mode="running")
        outer_next_actions = (mean + log_std.exp() * kwargs["next_noise"]).tanh()
        joined_z = torch.cat((batch.latents, batch.next_latents))
        joined_a = torch.cat((batch.actions, inner_next_actions))
        inner_log_q = inner.critic_target.log_probs(joined_z, joined_a, bn_mode="batch_no_update")[:, count:]
        outer_log_q = auxiliary.log_probs(batch.next_latents, outer_next_actions, bn_mode="running")
        expected_current = inner.critic.log_probs(batch.latents, batch.actions, bn_mode="running")
        arguments = (batch.rewards / 2.5, batch.bootstrap_mask, .9, 0., torch.zeros(count), support)
        projected_inner, values_inner, heads_inner, clipped = _target_oracle(inner_log_q, *arguments)
        projected_outer, values_outer, heads_outer, _ = _target_oracle(outer_log_q, *arguments)
    mask = kwargs["outer_terminal_mask"] & batch.bootstrap_mask.bool()
    expected = torch.where(mask[:, None], projected_outer, projected_inner)
    calls = []
    target_log_probs = inner.critic_target.log_probs

    def target_query(z, actions, **options):
        calls.append(options["bn_mode"])
        torch.testing.assert_close(z, joined_z, rtol=0, atol=0)
        torch.testing.assert_close(actions, joined_a, rtol=0, atol=0)
        return target_log_probs(z, actions, **options)

    monkeypatch.setattr(inner.critic_target, "log_probs", target_query)
    monkeypatch.setattr(outer.critic, "log_probs", lambda *a, **k: pytest.fail("Used the primary soft tail"))
    monkeypatch.setattr(outer.critic_target, "log_probs", lambda *a, **k: pytest.fail("Used the outer target network"))
    actual = inner.critic_objective(batch, **kwargs)
    assert calls == ["batch_no_update"]
    torch.testing.assert_close(actual.target_probabilities, expected, atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(actual.target_values, torch.where(mask, values_outer, values_inner))
    assert torch.equal(actual.target_head, torch.where(mask, heads_outer, heads_inner))
    torch.testing.assert_close(actual.current_log_probs, expected_current, atol=0, rtol=0)
    torch.testing.assert_close(actual.loss, -(expected.unsqueeze(0) * expected_current).sum(-1).sum(0).mean())
    assert float(actual.clip_fraction) == pytest.approx(clipped)
    assert clipped > 0  # The terminal boundary row exceeds the upper support.
    # Terminal rows reduce to immediate reward, even if marked as horizon ends.
    torch.testing.assert_close((actual.target_probabilities[2:] * support).sum(-1), torch.tensor([.2, 2.]))
    actual.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in inner.critic.parameters())
    assert all(p.grad is None for module in (inner.actor, inner.critic_target, outer, auxiliary)
               for p in module.parameters())
    assert _tree_equal(before, (inner.state_dict(), outer.state_dict(), auxiliary.state_dict()))
    assert torch.equal(rng, torch.get_rng_state())


def test_h2_inner_target_parameters_matter_but_its_running_buffers_and_alpha_do_not():
    inner, outer, auxiliary, batch, kwargs = _mixed_case()
    before = inner.critic_objective(batch, **kwargs)
    with torch.no_grad():
        # batch_no_update must ignore these inherited moments, without writing
        # over them. Reward-only targets must also ignore either temperature.
        for name, buffer in inner.critic_target.named_buffers():
            if name.endswith("running_mean"):
                buffer.add_(100.)
            elif name.endswith("running_var"):
                buffer.mul_(7.)
        inner.log_temperature.add_(4.)
        outer.log_temperature.sub_(3.)
    target_buffers = _buffers(inner.critic_target)
    changed_buffers = inner.critic_objective(batch, **kwargs)
    torch.testing.assert_close(before.target_probabilities, changed_buffers.target_probabilities, rtol=0, atol=0)
    assert _tree_equal(target_buffers, _buffers(inner.critic_target))
    with torch.no_grad():
        for head in inner.critic_target.q_networks:
            head.value.bias.add_(torch.linspace(-2., 2., inner.config.num_atoms))
    changed_parameters = inner.critic_objective(batch, **kwargs)
    assert not torch.allclose(before.target_probabilities[0], changed_parameters.target_probabilities[0])
    # The active frozen outer tail and both terminal rows ignore the inner target.
    torch.testing.assert_close(before.target_probabilities[1], changed_parameters.target_probabilities[1], rtol=0, atol=0)
    # Terminal projection sums a changed categorical distribution into the
    # same reward bin(s); permit the final float32 accumulation roundoff.
    torch.testing.assert_close(before.target_probabilities[2:], changed_parameters.target_probabilities[2:], rtol=0, atol=1e-7)
    torch.testing.assert_close(before.current_log_probs, changed_parameters.current_log_probs, rtol=0, atol=0)
    assert _tree_equal(target_buffers, _buffers(inner.critic_target))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware is unavailable"))])
@pytest.mark.parametrize("updates_per_round,policy_delay,accepted_slots", [
    (6, 3, (0, 3, 6, 9)),
    (12, 6, (0, 6, 12, 18)),
    (12, 3, (0, 3, 6, 9, 12, 15, 18, 21)),
])
def test_frozen_h2_j2_actual_budget_masks_bn_and_fresh_action_resets(
    device, updates_per_round, policy_delay, accepted_slots,
    tmp_path, monkeypatch, deterministic_xqc_numerics,
):
    # Real learner/model/optimizers, with the production N/B/J/G budget. Only
    # network widths, action dimension and training minibatch are kept tiny.
    common = dict(device=device, train_unroll_horizon=3, xqc_utd=2,
                  aux_return_mode="xqc", aux_return_detach_representation=False)
    critic_steps = 2 * updates_per_round
    actor_steps = len(accepted_slots)
    replay_draws = 256 * critic_steps
    source = _wrapper(inner_operator="none", **common)
    target = _wrapper(**common, inner_rounds=2, inner_rollout_horizon=2,
        inner_rollouts_per_round=256, inner_batch_size=256, inner_updates_per_round=updates_per_round,
        inner_replay_capacity=1024, inner_policy_delay=policy_delay, inner_update_timing="round",
        inner_critic_source="aux_return", inner_horizon_critic_source="aux_return",
        inner_critic_target="reward_only", inner_terminal_bootstrap="outer",
        inner_actor_bn_mode="running", inner_critic_bn_mode="running",
        inner_actor_lr=5e-5, inner_critic_lr=5e-5, inner_temperature_lr=5e-5,
        inner_reward_normalization="frozen_real_scale", inner_diagnostics_every=1)
    try:
        source.agent.update(_Replay(source.agent))
        source.agent.observe_reward(2., False, False)
        source.agent.observe_reward(3., False, True)
        checkpoint = tmp_path / "prior.pt"
        source.agent.save(str(checkpoint))
        target.load(str(checkpoint), frozen_evaluation=True)
        agent, engine = target.agent, target.agent.inner_engine
        provenance = agent.checkpoint_evaluation_provenance
        assert provenance["saved_semantic_signature"]["inner_policy_delay"] == 3
        assert provenance["evaluated_semantic_signature"]["inner_policy_delay"] == policy_delay
        frozen = agent.frozen_outer_state()
        observation, _ = target.env.reset(seed=101)
        global_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        cuda_rng = torch.cuda.get_rng_state(agent.device).clone() if device == "cuda" else None
        preparations, rounds, sampled_masks, steps, target_modes = [], [], [], [], []
        instrumented = set()
        prepare, collect, sample = engine._prepare_action, engine._collect_round, engine._sample_batch

        def checked_prepare():
            prepare()
            state, workspace = engine.state, engine.state.workspace
            local = workspace.controller
            assert local.config.policy_delay == policy_delay
            assert agent.xqc_controller.config.policy_delay == 3
            assert workspace.update_step == workspace.actor_optimizer_steps == workspace.temperature_optimizer_steps == 0
            assert state.replay.size == state.replay.next_sample_id == 0
            assert not state.outer_terminal_flags.any() and state.reward_normalizer is None
            assert _tree_equal(local.actor.state_dict(), agent.xqc_controller.actor.state_dict())
            assert torch.equal(local.log_temperature, agent.xqc_controller.log_temperature)
            for module in (local.critic, local.critic_target):
                assert _tree_equal(module.state_dict(), agent.aux_return.critic.state_dict())
            preparations.append(id(workspace))
            for name, optimizer in (("critic", workspace.critic_optimizer), ("actor", workspace.actor_optimizer),
                                    ("temperature", workspace.temperature_optimizer)):
                assert not optimizer.state
                if id(optimizer) not in instrumented:
                    instrumented.add(id(optimizer))
                    original = optimizer.step
                    def counted(*args, _name=name, _step=original, **kwargs):
                        steps.append(_name)
                        return _step(*args, **kwargs)
                    monkeypatch.setattr(optimizer, "step", counted)
            if id(local.critic_target) not in instrumented:
                instrumented.add(id(local.critic_target))
                original_target = local.critic_target.log_probs
                def checked_target(z, a, **kwargs):
                    target_modes.append(kwargs["bn_mode"])
                    assert z.shape[0] == a.shape[0] == 512
                    return original_target(z, a, **kwargs)
                monkeypatch.setattr(local.critic_target, "log_probs", checked_target)

        def checked_collect(root_z):
            start = engine.state.replay.size
            rounds.append((start, engine.state.workspace.update_step))
            result = collect(root_z)
            replay = engine.state.replay
            assert replay.size == start + 512
            assert torch.equal(replay.z[start:start+256], root_z.expand(256, -1))
            assert torch.equal(replay.z[start+256:start+512], replay.next_z[start:start+256])
            expected = torch.tensor(([False]*256 + [True]*256)*2, device=agent.device)
            assert torch.equal(engine.state.outer_terminal_flags[:replay.size], expected[:replay.size])
            return result

        def checked_sample():
            batch = sample()
            expected = batch["sample_ids"].div(256, rounding_mode="floor").remainder(2).bool()
            assert torch.equal(batch["outer_terminal_mask"], expected)
            sampled_masks.append(expected.clone())
            return batch

        monkeypatch.setattr(engine, "_prepare_action", checked_prepare)
        monkeypatch.setattr(engine, "_collect_round", checked_collect)
        monkeypatch.setattr(engine, "_sample_batch", checked_sample)

        def episode():
            target.reset_for_evaluation(12345, reuse_action_pool=True)
            actions = []
            for _ in range(2):
                rounds.clear(); sampled_masks.clear(); steps.clear(); target_modes.clear()
                actions.append(target.predict(observation, deterministic=True)[0])
                metrics = agent.last_inner_metrics
                assert rounds == [(0, 0), (512, updates_per_round)]
                assert steps == [component for slot in range(critic_steps)
                                 for component in (("critic", "actor", "temperature")
                                     if slot in accepted_slots else ("critic",))]
                expected = {"inner_model_steps": 1024, "inner_buffer_size": 1024,
                    "inner_buffer_capacity": 1024, "inner_rollout_count": 512, "inner_rollout_len_mean": 2,
                    "inner_replay_draws": replay_draws, "inner_critic_optimizer_steps": critic_steps,
                    "inner_actor_optimizer_steps": actor_steps, "inner_temperature_optimizer_steps": actor_steps,
                    "inner_policy_delay": policy_delay,
                    "inner_target_updates": critic_steps, "inner_outer_terminal_boundary_rows": 512,
                    "inner_reward_scale_delta": 0, "inner_reward_normalizer_imagined_updates": 0,
                    "inner_critic_source_aux_return": 1, "inner_horizon_critic_source_aux_return": 1,
                    "inner_critic_target_reward_only": 1, "inner_compile_fallback": 0}
                assert all(metrics[key] == value for key, value in expected.items())
                assert target_modes == ["batch_no_update"]*critic_steps
                assert metrics["inner_outer_terminal_bootstrap_rows"] == sum(int(mask.sum()) for mask in sampled_masks)
                assert 0 < metrics["inner_outer_terminal_bootstrap_rows"] < replay_draws
                assert all(torch.isfinite(torch.as_tensor(value)).all() for value in metrics.values())
                local = engine._workspace_pool.controller
                assert _tree_equal(_buffers(local.actor), _buffers(agent.xqc_controller.actor))
                assert _tree_equal(_buffers(local.critic), _buffers(agent.aux_return.critic))
                assert _tree_equal(_buffers(local.critic_target), _buffers(agent.aux_return.critic))
                assert not _tree_equal(local.critic.state_dict(), agent.aux_return.critic.state_dict())
                assert not _tree_equal(local.critic_target.state_dict(), agent.aux_return.critic.state_dict())
                assert engine.state.workspace is None and engine.state.replay is None
                assert _tree_equal(frozen, agent.frozen_outer_state())
            return np.stack(actions)

        first = episode()
        np.testing.assert_array_equal(first, episode())
        assert len(preparations) == 4 and len(set(preparations)) == 1
        assert torch.equal(global_rng[0], torch.get_rng_state())
        assert global_rng[1] == random.getstate()
        current_numpy = np.random.get_state()
        assert global_rng[2][0] == current_numpy[0] and global_rng[2][2:] == current_numpy[2:]
        np.testing.assert_array_equal(global_rng[2][1], current_numpy[1])
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(agent.device))
    finally:
        source.env.close()
        target.env.close()
