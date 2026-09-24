"""Action Monte Carlo averages change only Retrace's state-value estimates."""

import pytest
import torch

from RL.tdmpc2_core.common.entropy import policy_entropy
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_retrace_inner import _collect, _prepare, make_model
from tests.test_ambi_retrace_checkpoint_compile import _snapshot
from tests.test_aux_return_inner import _prepared as _prepared_aux


def _noises(engine, batch):
    count, horizon, action_dim = batch["action"].shape
    samples = engine.cfg.inner_retrace_value_samples
    boundary_samples = engine.cfg.inner_retrace_boundary_value_samples
    noise = torch.linspace(-1.1, 0.8, count * horizon * samples * action_dim)
    prior = torch.linspace(-0.6, 1.2, count * boundary_samples * action_dim)
    shape = (count, horizon, action_dim) if samples == 1 else (count, horizon, samples, action_dim)
    boundary_shape = (count, action_dim) if boundary_samples == 1 else (count, boundary_samples, action_dim)
    return noise.reshape(shape), prior.reshape(boundary_shape)


@torch.no_grad()
def _manual_targets(engine, batch, alpha, noise, prior_noise, pair, scale):
    """Evaluate one state/action at a time, then apply a scalar backward return."""
    cfg = engine.cfg
    count, horizon = batch["valid"].shape[:2]
    noise = noise.reshape(count, horizon, cfg.inner_retrace_value_samples, cfg.action_dim)
    prior_noise = prior_noise.reshape(count, cfg.inner_retrace_boundary_value_samples, cfg.action_dim)
    values, stored_q, coefficients = (torch.zeros(count, horizon, 1) for _ in range(3))
    q_options = dict(qs=engine.state.critic_target, reduction=cfg.inner_q_target_reduction,
                     pair_indices=pair, trusted_pair_indices=True)
    for b in range(count):
        for t in range(horizon):
            if not batch["valid"][b, t].item():
                continue
            z = batch["z"][b, t:t + 1]
            stored_q[b, t] = engine.model.Q(z, batch["action"][b, t:t + 1], **q_options)
            stats = engine.model.policy_stats(
                z, policy=engine.state.actor, log_std_mapping=cfg.inner_log_std_mapping,
                log_std_min=cfg.inner_log_std_min, log_std_max=cfg.inner_log_std_max,
            )
            log_pi = engine.model.squashed_component_log_prob(
                batch["pre_tanh_action"][b, t:t + 1], stats["pre_tanh_mean"], stats["log_std"],
            )
            coefficients[b, t] = cfg.inner_retrace_lambda * torch.exp(
                (log_pi - batch["behavior_log_prob"][b, t]).clamp_max(0),
            )
            if batch["terminated"][b, t].item():
                continue
            next_z = batch["next_z"][b, t:t + 1]
            action_values = []
            if batch["horizon_end"][b, t].item():
                for draw in prior_noise[b]:
                    action, _ = engine.model.pi(next_z, noise=draw[None])
                    action_values.append(engine.model.Q(next_z, action, reduction="mean_all"))
            else:
                for draw in noise[b, t]:
                    action, info = engine.model.pi(
                        next_z, policy=engine.state.actor, noise=draw[None],
                        log_std_mapping=cfg.inner_log_std_mapping,
                        log_std_min=cfg.inner_log_std_min, log_std_max=cfg.inner_log_std_max,
                        **engine.agent._inner_critic_entropy_kwargs(),
                    )
                    value = engine.model.Q(next_z, action, **q_options)
                    if cfg.inner_sac_critic_target == "entropy_augmented":
                        value = value + alpha * scale * policy_entropy(info, cfg.inner_actor_entropy_mode)
                    action_values.append(value)
            values[b, t] = sum(action_values) / len(action_values)
    targets = torch.zeros_like(values)
    for b in range(count):
        for t in reversed(range(horizon)):
            if not batch["valid"][b, t].item():
                continue
            targets[b, t] = batch["reward"][b, t]
            if batch["terminated"][b, t].item():
                continue
            targets[b, t] += engine.agent.discount * values[b, t]
            if t + 1 < horizon and batch["valid"][b, t + 1].item():
                targets[b, t] += engine.agent.discount * coefficients[b, t + 1] * (
                    targets[b, t + 1] - stored_q[b, t + 1]
                )
    return targets.reshape(-1, 1), coefficients


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("target,entropy", [
    ("reward_only", "squashed"), ("entropy_augmented", "squashed"),
    ("entropy_augmented", "tdmpc2_scaled"),
])
def test_multi_action_targets_and_gradients_match_scalar_reference(make_model, representation, target, entropy):
    scaled = target == "entropy_augmented"
    holder = make_model(
        q_representation=representation, num_q=3 if representation == "distributional" else 2,
        inner_retrace_value_samples=4, inner_retrace_boundary_value_samples=3,
        inner_sac_critic_target=target, inner_actor_entropy_mode=entropy,
        inner_retrace_lambda=0.7, inner_behavior_std_scale=0.6,
        mppi_terminal_q_reduction="mean_all",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range" if scaled else "none",
        ent_coef=0.2, inner_temperature_mode="fixed", inner_temperature=0.2,
    )
    with torch.no_grad():
        for i, head in enumerate(holder.agent.model._Qs):
            head[-1].weight.copy_(torch.linspace(-0.2, 0.3, head[-1].weight.numel()).reshape_as(head[-1].weight) + i * 0.1)
    engine = _prepare(holder)
    batch, _ = _collect(engine)
    batch["reward"][:] = torch.tensor([0.2, 0.7, -0.3])[None, :, None]
    noise, prior = _noises(engine, batch)
    pair = torch.tensor([0, 2]) if representation == "distributional" else torch.tensor([0, 1])
    alpha = torch.tensor(0.31, requires_grad=True)
    scale = torch.tensor(2.3 if scaled else 1.0, requires_grad=True)
    actual = engine._retrace_critic_kernel(batch, alpha, noise, prior, pair, actor_loss_scale=scale)
    target_values, coefficients = _manual_targets(engine, batch, alpha, noise, prior, pair, scale)
    torch.testing.assert_close(actual[2], target_values, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(actual[4], coefficients, rtol=3e-5, atol=3e-6)
    predictions = engine.model.q_predictions(batch["z"].flatten(0, 1), batch["action"].flatten(0, 1), qs=engine.state.critic)
    expected_loss = engine.model.critic_loss(predictions, target_values) * engine.cfg.inner_critic_loss_coef
    torch.testing.assert_close(actual[0], expected_loss, rtol=3e-5, atol=3e-6)
    actual[0].backward()
    gradients = [p.grad.clone() for p in engine.state.critic_params]
    engine.state.critic_optim.zero_grad(set_to_none=True)
    expected_loss.backward()
    for parameter, expected in zip(engine.state.critic_params, gradients):
        torch.testing.assert_close(parameter.grad, expected, rtol=3e-5, atol=3e-6)
    assert alpha.grad is scale.grad is None
    assert all(p.grad is None for p in engine.state.actor.parameters())
    assert all(p.grad is None for p in engine.state.critic_target.parameters())
    assert all(p.grad is None for p in engine.model.parameters())


def test_action_average_keeps_minimum_inside_expectation(make_model, monkeypatch):
    engine = _prepare(make_model(
        inner_rollout_horizon=2, inner_retrace_lambda=0, inner_sac_critic_target="reward_only",
        inner_retrace_value_samples=2, inner_log_std_min=-4., inner_log_std_max=1.,
    ))
    with torch.no_grad():
        engine.state.actor[-1].weight.zero_()
        engine.state.actor[-1].bias.zero_()
    batch, _ = _collect(engine)
    monkeypatch.setattr(engine, "_bootstrap_q", lambda z, a, **kw: torch.minimum(a, -a))
    monkeypatch.setattr(engine, "_prior_bootstrap", lambda z, noise: z.new_zeros(z.shape[0], 1))
    noise = torch.tensor([-1., 1.]).reshape(1, 1, 2, 1).expand(2, 2, -1, -1)
    result = engine._retrace_critic_kernel(batch, torch.tensor(0.), noise, torch.zeros(2, 1), torch.tensor([0, 1]))
    expected = batch["reward"][:, 0] - engine.agent.discount * torch.tanh(torch.tensor(1.))
    torch.testing.assert_close(result[2].reshape(2, 2, 1)[:, 0], expected)


@pytest.mark.parametrize("samples,boundary_samples", [(4, 1), (1, 4), (4, 16)])
def test_rng_pairing_and_evaluation_counters(make_model, monkeypatch, samples, boundary_samples):
    engines = [_prepare(make_model(q_representation="distributional", num_q=3, **options)) for options in ({}, {
        "inner_retrace_value_samples": samples, "inner_retrace_boundary_value_samples": boundary_samples,
    })]
    captured = []
    for engine in engines:
        batch, _ = _collect(engine)
        original = engine._compile_regions["critic"]
        def capture(*args, _original=original, **kwargs):
            captured.append(tuple(value.clone() if torch.is_tensor(value) else value for value in args[2:5]))
            return _original(*args, **kwargs)
        monkeypatch.setitem(engine._compile_regions, "critic", capture)
        policy_before, q_before = engine.state.policy_evaluations, engine.state.q_evaluations
        with engine.rng.fork("bootstrap"):
            engine._retrace_critic_step(batch, torch.tensor(0.2))
        count, horizon = batch["valid"].shape[:2]
        k, kb = engine.cfg.inner_retrace_value_samples, engine.cfg.inner_retrace_boundary_value_samples
        assert engine.state.policy_evaluations - policy_before == count * ((k + 1) * horizon + kb)
        assert engine.state.q_evaluations - q_before == count * ((k + 2) * horizon + kb)
    baseline, expanded = captured
    actual_noise = expanded[0] if samples == 1 else expanded[0][:, :, 0]
    actual_prior = expanded[1] if boundary_samples == 1 else expanded[1][:, 0]
    torch.testing.assert_close(actual_noise, baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_prior, baseline[1], rtol=0, atol=0)
    if baseline[2] is not None:
        torch.testing.assert_close(expanded[2], baseline[2], rtol=0, atol=0)
    base_rng, multi_rng = [engine.rng.training_state_dict() for engine in engines]
    assert len(multi_rng["streams"]) == len(base_rng["streams"]) + 2
    for group in ("streams", "phase_streams"):
        for key, value in base_rng[group].items():
            torch.testing.assert_close(multi_rng[group][key], value, rtol=0, atol=0)
    if samples > 1:
        assert not torch.equal(expanded[0][:, :, 0], expanded[0][:, :, 1])
    if boundary_samples > 1:
        assert not torch.equal(expanded[1][:, 0], expanded[1][:, 1])


def test_changing_one_sample_count_does_not_shift_the_other_noise_stream(make_model, monkeypatch):
    captures = {}
    for samples, boundary_samples in ((4, 1), (1, 16), (4, 16)):
        engine = _prepare(make_model(
            inner_retrace_value_samples=samples, inner_retrace_boundary_value_samples=boundary_samples,
        ))
        batch, _ = _collect(engine)
        draws = captures[samples, boundary_samples] = []
        original = engine._compile_regions["critic"]
        def capture(*args, _original=original, _draws=draws, **kwargs):
            _draws.append((args[2].clone(), args[3].clone()))
            return _original(*args, **kwargs)
        monkeypatch.setitem(engine._compile_regions, "critic", capture)
        for _ in range(2):
            with engine.rng.fork("bootstrap"):
                engine._retrace_critic_step(batch, torch.tensor(.2))
    for step in range(2):
        torch.testing.assert_close(captures[4, 1][step][0], captures[4, 16][step][0], rtol=0, atol=0)
        torch.testing.assert_close(captures[1, 16][step][1], captures[4, 16][step][1], rtol=0, atol=0)


def test_h1_uses_only_boundary_sample_count(make_model):
    targets = []
    for count in (1, 4):
        engine = _prepare(make_model(inner_rollout_horizon=1, inner_retrace_value_samples=count))
        batch, _ = _collect(engine)
        noise, _ = _noises(engine, batch)
        with engine.rng.fork("bootstrap"):
            outputs = engine._retrace_critic_kernel(batch, torch.tensor(.2), noise, torch.zeros(2, 1), torch.tensor([0, 1]))
        targets.append(outputs[2])
        assert torch.count_nonzero(outputs[6]) == 0
    torch.testing.assert_close(*targets, rtol=0, atol=0)


@pytest.mark.parametrize("horizon_critic", ["sac", "aux_return"])
def test_sampled_boundary_preserves_source_and_masks_terminal_padding(horizon_critic):
    with _prepared_aux(
        inner_sac_return_estimator="retrace", inner_finite_horizon=True,
        inner_rollout_horizon=3, train_unroll_horizon=3, inner_retrace_lambda=0.,
        inner_horizon_critic_source=horizon_critic,
        inner_retrace_value_samples=4, inner_retrace_boundary_value_samples=3,
    ) as (_, engine):
        batch, _ = _collect(engine)
        batch["reward"][:] = torch.tensor([1., 2., 3.])[None, :, None]
        batch["terminated"][1, 1] = 1
        batch["next_z"][1, 1] = float("nan")
        batch["valid"][1, 2] = False
        for value in batch.values():
            if value.is_floating_point() and value.ndim == 3:
                value[1, 2] = float("nan")
        noise, prior = _noises(engine, batch)
        cold, hot = [engine._retrace_critic_kernel(batch, torch.tensor(alpha), noise, prior, torch.tensor([0, 1]))
                     for alpha in (0., 100.)]
        expected = 3 + engine.agent.discount * (2 if horizon_critic == "sac" else 9)
        assert hot[2].reshape(2, 3)[0, 2].item() == pytest.approx(expected)
        torch.testing.assert_close(hot[2].reshape(2, 3)[0, 2], cold[2].reshape(2, 3)[0, 2], rtol=0, atol=0)
        torch.testing.assert_close(hot[2].reshape(2, 3)[1, 1:], torch.tensor([2., 0.]), rtol=0, atol=0)
        assert torch.isfinite(hot[0]) and torch.isfinite(hot[2]).all()


def test_default_one_sample_is_bitwise_equivalent_to_explicit_one(make_model):
    implicit, explicit = [make_model(**options).agent for options in ({}, {
        "inner_retrace_value_samples": 1, "inner_retrace_boundary_value_samples": 1,
    })]
    global_rng = torch.random.get_rng_state().clone()
    for observation in (torch.zeros(3), torch.tensor([.2, -.3, .1])):
        expected = implicit.act(observation, collect_diagnostics=False)
        actual = explicit.act(observation, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(_snapshot(implicit.inner_engine), _snapshot(explicit.inner_engine))
        for metric in ("inner_policy_evaluations", "inner_q_evaluations", "inner_model_steps",
                       "inner_critic_optimizer_steps", "inner_actor_optimizer_steps", "inner_temperature_optimizer_steps"):
            assert implicit.last_inner_metrics[metric] == explicit.last_inner_metrics[metric]
    assert "value_samples" not in implicit.inner_engine._retrace_spec()
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)


def test_expanded_targets_share_inner_pair_and_use_one_boundary_q_call(make_model, monkeypatch):
    engine = _prepare(make_model(q_representation="distributional", num_q=5,
                               inner_retrace_value_samples=4, inner_retrace_boundary_value_samples=3))
    batch, _ = _collect(engine)
    noise, prior = _noises(engine, batch)
    original, calls = engine.model.Q, []
    def q(z, action, **kwargs):
        calls.append((z.shape[0], kwargs))
        return original(z, action, **kwargs)
    monkeypatch.setattr(engine.model, "Q", q)
    pair = torch.tensor([1, 3])
    engine._retrace_critic_kernel(batch, torch.tensor(.2), noise, prior, pair)
    inner = [(rows, kw) for rows, kw in calls if kw.get("qs") is engine.state.critic_target]
    outer = [(rows, kw) for rows, kw in calls if kw.get("qs") is not engine.state.critic_target]
    assert [rows for rows, _ in inner] == [24, 6]
    assert all(kw["pair_indices"] is pair and kw["trusted_pair_indices"] for _, kw in inner)
    assert len(outer) == 1 and outer[0][0] == 6
    assert outer[0][1].get("pair_indices") is None


def test_zero_work_reports_configured_samples_without_value_evaluations(make_model):
    agent = make_model(
        inner_rollouts_per_round=0, inner_updates_per_round=0, inner_replay_capacity=None,
        inner_retrace_value_samples=4, inner_retrace_boundary_value_samples=16,
    ).agent
    action = agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    metrics = agent.last_inner_metrics
    assert torch.isfinite(action).all()
    assert metrics["inner_retrace_value_samples"] == 4
    assert metrics["inner_retrace_boundary_value_samples"] == 16
    assert metrics["inner_q_evaluations"] == metrics["inner_model_steps"] == 0
    assert metrics["inner_critic_optimizer_steps"] == metrics["inner_actor_optimizer_steps"] == 0
