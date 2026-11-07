"""Paper-likelihood oracles, collection/replay contracts, and objective identity."""

from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict

from RL.tdmpc2_core.boom import boom_replay_likelihood_loss
from RL.tdmpc2_core.common.buffer import Buffer
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.whole_inner_graph import training_graph_act
from tests.test_ambi_boom import _agent, _batch, _labels
from tests.test_ambi_boom_config import _boom_cfg, _identity
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_behavior_policy_buffer import _cfg, _episode, _assert_batches_equal
from tests.test_ambi_root_local_sac import _tiny_model


def _inputs():
    shape = (2, 4, 3)
    return dict(
        actor_pre_tanh_mean=torch.linspace(-0.2, 0.4, 24, dtype=torch.float64).reshape(shape).requires_grad_(),
        actor_log_std=torch.linspace(-2, 0.5, 24, dtype=torch.float64).reshape(shape).requires_grad_(),
        replay_pre_tanh_action=torch.linspace(-1.1, 1.2, 24, dtype=torch.float64).reshape(shape).requires_grad_(),
        valid=torch.tensor([[True, False, True, True], [True, True, True, False]]),
        q_values=torch.tensor([[1., 8., 3., 7.], [-1., -3., 2., 9.]], dtype=torch.float64, requires_grad=True),
        scale=torch.tensor(4., dtype=torch.float64, requires_grad=True),
        rho=0.6, coefficient=0.17,
    )


def _reference(x, temperature):
    result = x["actor_pre_tanh_mean"].sum() * 0
    for t in range(2):
        valid = x["valid"][t]
        if not valid.any():
            continue
        dist = torch.distributions.TransformedDistribution(
            torch.distributions.Normal(
                x["actor_pre_tanh_mean"][t, valid], x["actor_log_std"][t, valid].exp(),
            ), [torch.distributions.TanhTransform()],
        )
        target = x["replay_pre_tanh_action"][t, valid].detach().tanh()
        nll = -dist.log_prob(target).sum(-1)
        weights = (x["q_values"][t, valid].detach() / temperature).softmax(0)
        result = result + x["rho"] ** t * (weights * nll).sum()
    return result * x["coefficient"] / (2 * x["scale"].detach())


@pytest.mark.parametrize("temperature", ["scale", 1.0, 0.3])
def test_matches_transformed_distribution_value_and_both_density_gradients(temperature):
    x = _inputs()
    loss, metrics = boom_replay_likelihood_loss(**x, q_temperature=temperature)
    expected = _reference(x, x["scale"].detach() if temperature == "scale" else temperature)
    leaves = [x[key] for key in ("actor_pre_tanh_mean", "actor_log_std", "replay_pre_tanh_action", "q_values", "scale")]
    actual_grads = torch.autograd.grad(loss, leaves, allow_unused=True)
    reference_grads = torch.autograd.grad(expected, leaves[:2])
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    for actual, reference in zip(actual_grads[:2], reference_grads):
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    assert actual_grads[2:] == (None, None, None)
    assert all(not value.requires_grad for value in metrics.values())


@pytest.mark.parametrize("std", [1., 0.01])
@pytest.mark.parametrize("residual_in_std", [0., 1., 2.])
def test_variance_can_expand_and_is_not_floored(std, residual_in_std):
    mean = torch.zeros(1, 1, 1, dtype=torch.float64, requires_grad=True)
    log_std = torch.full_like(mean, std).log().requires_grad_()
    loss, _ = boom_replay_likelihood_loss(
        mean, log_std, torch.full_like(mean, residual_in_std * std),
        torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1), torch.tensor(4.),
        rho=1., coefficient=4.,
    )
    mean_grad, std_grad = torch.autograd.grad(loss, (mean, log_std))
    torch.testing.assert_close(mean_grad, torch.full_like(mean, -residual_in_std / std))
    torch.testing.assert_close(std_grad, torch.full_like(mean, 1 - residual_in_std ** 2))


@pytest.mark.parametrize("scale", [1., 2., 2.00001, 4.])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_gate_missing_labels_and_saturation_remain_finite(scale, all_invalid):
    x = _inputs()
    x["scale"] = torch.tensor(scale, dtype=torch.float64)
    if all_invalid:
        x["valid"].zero_()
    else:
        x["valid"][1].zero_()  # Empty depth must not poison the softmax.
    for key in ("actor_pre_tanh_mean", "actor_log_std", "replay_pre_tanh_action"):
        value = x[key].detach().clone()
        if key == "replay_pre_tanh_action":
            value.fill_(12.)  # tanh is rounded to one in float32; retain its preimage.
        x[key] = value.masked_fill(~x["valid"].unsqueeze(-1), torch.nan).requires_grad_()
    x["q_values"] = x["q_values"].detach().masked_fill(~x["valid"], torch.nan)
    loss, metrics = boom_replay_likelihood_loss(**x)
    loss.backward()
    assert all(torch.isfinite(value) for value in metrics.values())
    for key in ("actor_pre_tanh_mean", "actor_log_std"):
        assert torch.isfinite(x[key].grad).all()
        assert torch.count_nonzero(x[key].grad[~x["valid"]]) == 0
        if scale <= 2 or all_invalid:
            assert torch.count_nonzero(x[key].grad) == 0
            assert loss.item() == 0


def test_likelihood_fullgraph_handles_gate_and_masks_without_retracing():
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(boom_replay_likelihood_loss, backend=backend, fullgraph=True)
    x = _inputs()
    actual, _ = compiled(**x)
    expected, _ = boom_replay_likelihood_loss(**x)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    with torch.no_grad():
        x["scale"].fill_(2)
        x["valid"].zero_()
    assert compiled(**x)[0].item() == 0
    assert len(graphs) == 1


def test_actor_kernel_weights_replay_actions_and_matches_independent_policy_gradient(monkeypatch):
    agent = _agent(outer_boom_alignment="replay_likelihood", outer_boom_q_temperature=1.)
    agent.actor_loss_scale.fill_(4)
    cfg = agent.cfg
    shape = (cfg.train_unroll_horizon, cfg.batch_size, cfg.action_dim)
    zs = torch.randn(*shape[:2], cfg.latent_dim)
    noise = torch.linspace(-.8, .9, zs.shape[0] * zs.shape[1]).reshape(shape)
    targets = torch.linspace(-1.1, 1.2, noise.numel()).reshape(shape).requires_grad_()
    replay_actions = targets.detach().tanh().requires_grad_()
    valid = torch.ones(*shape[:2], 1, dtype=torch.bool)
    valid[:, -1] = False
    reference_pi = deepcopy(agent.model._pi)
    original_pi = agent.model.pi
    sample, info = original_pi(zs, policy=reference_pi, noise=noise)
    offsets = torch.arange(cfg.batch_size).reshape(1, -1, 1)
    def q_value(action):
        return offsets + 2 * action.sum(-1, keepdim=True) + .3 * action.square().sum(-1, keepdim=True)
    sampled_q = q_value(sample)
    quantiles = torch.quantile(sampled_q[0].detach(), torch.tensor([.05, .95]), dim=0)
    scale = torch.tensor([4.]).lerp((quantiles[1] - quantiles[0]).clamp_min(1), cfg.sac_actor_loss_scale_tau)
    temporal = cfg.rho ** torch.arange(shape[0])
    rl_loss = ((cfg.ent_coef * info["log_prob"] - sampled_q / scale).mean((1, 2)) * temporal).mean()
    distribution = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(info["pre_tanh_mean"], info["log_std"].exp()),
        [torch.distributions.TanhTransform()],
    )
    nll = -distribution.log_prob(replay_actions.detach()).sum(-1)
    weights = q_value(replay_actions.detach()).squeeze(-1).masked_fill(~valid.squeeze(-1), -torch.inf).softmax(1)
    alignment = cfg.outer_boom_coef / scale.squeeze() * ((weights * nll).sum(1) * temporal).mean()
    reference_grads = torch.autograd.grad(rl_loss + alignment, tuple(reference_pi.parameters()))
    q_actions = []
    def critic(z, action, **kwargs):
        q_actions.append(action.detach().clone())
        q = q_value(action)
        return torch.stack([q + head for head in range(cfg.num_q)])
    monkeypatch.setattr(agent.model, "pi", lambda z: original_pi(z, noise=noise))
    monkeypatch.setattr(agent.model, "Q", critic)
    # Stored behavior means are unrelated: changing them cannot change NLL.
    outputs = agent._boom_actor_kernel(
        zs, torch.full_like(targets, 100.), valid, agent.alpha.detach(), agent.actor_loss_scale,
        replay_action=replay_actions, behavior_pre_tanh_action=targets,
    )
    outputs["actor_loss"].backward()
    assert len(q_actions) == 2
    torch.testing.assert_close(q_actions[0], sample.detach())
    torch.testing.assert_close(q_actions[1], torch.where(valid, replay_actions.detach(), 0.))
    torch.testing.assert_close(outputs["actor_loss"], rl_loss + alignment)
    torch.testing.assert_close(outputs["boom_metrics"]["boom_alignment_loss"], alignment)
    torch.testing.assert_close(outputs["proposed_scale"], scale)
    for actual, expected in zip(agent.model._pi.parameters(), reference_grads):
        torch.testing.assert_close(actual.grad, expected, rtol=3e-5, atol=1e-6)
    assert targets.grad is None and replay_actions.grad is None
    assert all(parameter.grad is None for parameter in agent.model._Qs.parameters())


def test_new_configuration_is_explicit_and_old_identities_are_preserved():
    cfg = _boom_cfg(outer_boom_alignment="REPLAY_LIKELIHOOD")
    assert cfg.outer_boom_alignment == "replay_likelihood"
    assert cfg.store_behavior_pre_tanh_action and cfg.store_behavior_policy
    assert cfg.outer_boom_q_temperature == "scale"
    base = {"outer_actor_objective": "boom"}
    likelihood = {**base, "outer_boom_alignment": "replay_likelihood"}
    assert _identity(base) == _identity({**base, "outer_boom_alignment": "sampled_action", "outer_boom_q_temperature": "scale"})
    assert _identity(base) != _identity(likelihood)
    assert _identity(likelihood) == _identity({**likelihood, "outer_boom_min_std": .9, "store_behavior_pre_tanh_action": True})
    assert _identity(likelihood) != _identity({**likelihood, "outer_boom_q_temperature": 1.})
    assert _identity({}) == _identity({"outer_boom_alignment": "replay_likelihood", "outer_boom_q_temperature": 1.})


@pytest.mark.parametrize("temperature", [0, -1, True, None, float("nan"), float("inf"), "1"])
def test_fixed_temperature_requires_finite_positive_number(temperature):
    with pytest.raises(ValueError, match="outer_boom_q_temperature"):
        _boom_cfg(outer_boom_alignment="replay_likelihood", outer_boom_q_temperature=temperature)


def test_fixed_temperature_does_not_silently_change_legacy_and_clipped_noise_is_rejected():
    with pytest.raises(ValueError, match="requires outer_boom_alignment"):
        _boom_cfg(outer_boom_q_temperature=1.)
    with pytest.raises(ValueError, match="no exact pre-tanh"):
        _boom_cfg(outer_boom_alignment="replay_likelihood", inner_execution_action="mean_plus_gaussian")


def test_replay_preimages_align_with_executed_transitions_and_roundtrip_exactly():
    cfg = _cfg(store_behavior_policy=True)
    cfg.store_behavior_pre_tanh_action = True
    source = Buffer(cfg, resumable=True)
    episode = _episode(include_behavior_policy=True)
    preimage = torch.tensor([[0., 0.], [12., -12.], [.2, -.3], [.4, .5]])
    episode["behavior_pre_tanh_action"] = preimage
    episode["action"] = preimage.tanh()
    batch = source._prepare_batch(episode.unsqueeze(1), include_behavior_policy=True, include_behavior_action=True)
    assert len(batch) == 9
    torch.testing.assert_close(batch[8].squeeze(1), preimage[1:])
    torch.testing.assert_close(batch[1], batch[8].tanh())
    with pytest.raises(RuntimeError, match="lacks behavior-policy"):
        source._prepare_batch(episode.exclude("behavior_pre_tanh_action").unsqueeze(1), include_behavior_policy=True, include_behavior_action=True)
    source.add(episode)
    metadata = source.training_state_metadata()
    assert metadata["version"] == 3
    shards = list(source.iter_training_state_shards(max_rows=2))
    restored = Buffer(deepcopy(cfg), resumable=True)
    restored.load_training_state_shards(metadata, iter(shards))
    rng = torch.get_rng_state()
    expected = source.sample(include_behavior_policy=True, include_behavior_action=True)
    torch.set_rng_state(rng)
    _assert_batches_equal(restored.sample(include_behavior_policy=True, include_behavior_action=True), expected)
    legacy = Buffer(_cfg(store_behavior_policy=True), resumable=True)
    with pytest.raises(ValueError, match="replay version"):
        legacy.load_training_state_shards(metadata, iter(shards))
    assert legacy.size == 0


@pytest.mark.parametrize("execution,interval", [("policy_sample", 1), ("mean", 1), ("policy_sample", 2)])
def test_collection_captures_actual_sample_and_seed_actions_are_invalid(execution, interval):
    model = _tiny_model(
        outer_actor_objective="boom", outer_boom_alignment="replay_likelihood",
        inner_execution_action=execution, inner_solve_interval=interval,
        sac_actor_loss_scale_mode="tdmpc2_percentile_range", ent_coef=.2,
        inner_temperature_mode="inherit_outer", outer_critic_target="reward_only",
        inner_sac_critic_target="reward_only",
    )
    try:
        obs = torch.zeros(3)
        model._start_episode_staging(obs)
        for row in (1, 2):
            action = model._act_agent(obs, t0=row == 1, eval_mode=False)
            captured = model._pending_behavior_policy["pre_tanh_action"].clone()
            torch.testing.assert_close(action, captured.tanh())
            model._stage_transition(row, obs, action, 1., False)
            torch.testing.assert_close(model._episode_staging["behavior_pre_tanh_action"][row], captured)
            assert model._episode_staging["behavior_policy_valid"][row]
        model._random_action_norm()
        model._stage_transition(3, obs, torch.zeros(1), 0., False)
        assert not model._episode_staging["behavior_policy_valid"][3]
        generator = torch.Generator().manual_seed(56)
        action, behavior = model.agent.act_outer_policy(obs, generator=generator, return_behavior_policy=True)
        torch.testing.assert_close(action, behavior["pre_tanh_action"].tanh())
    finally:
        model.env.close()


def test_buffer_to_learner_and_checkpoint_exact_next_update():
    source = _agent(outer_boom_alignment="replay_likelihood")
    rows = 8
    preimage = torch.linspace(-2, 2, rows).reshape(rows, 1)
    replay = Buffer(source.cfg)
    replay.add(TensorDict({
        "obs": torch.randn(rows, 3), "action": preimage.tanh(), "reward": torch.ones(rows),
        "terminated": torch.zeros(rows), "behavior_pre_tanh_action": preimage,
        "behavior_pre_tanh_mean": torch.zeros_like(preimage), "behavior_log_std": torch.ones_like(preimage),
        "behavior_policy_valid": torch.ones(rows, dtype=torch.bool),
    }, batch_size=[rows]))
    source.actor_loss_scale.fill_(4)
    metrics = source.update(replay)
    assert metrics["boom_valid_fraction"] == 1 and metrics["boom_gate_active"] == 1
    assert torch.isfinite(metrics["actor_grad_norm"])
    restored = _agent(outer_boom_alignment="replay_likelihood")
    training = _clone_tree(source.training_state_dict())
    restored.load_training_state_dict(training)
    data = _batch(source)
    mean, log_std, valid = _labels(source)
    labels = dict(behavior_pre_tanh_mean=mean, behavior_log_std=log_std, behavior_policy_valid=valid,
                  behavior_pre_tanh_action=data[1].atanh())
    rng = torch.get_rng_state()
    expected = source._update(*data, **labels)
    torch.set_rng_state(rng)
    actual = restored._update(*data, **labels)
    _assert_tree_equal(actual, expected)
    _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())


def test_training_loop_collects_numpy_actions_and_updates_from_actual_replay(monkeypatch):
    model = _tiny_model(
        outer_actor_objective="boom", outer_boom_alignment="replay_likelihood",
        outer_boom_coef=.0042, sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        ent_coef=.0021, inner_temperature_mode="inherit_outer",
        outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
        inner_rounds=1, wandb=False,
    )
    metrics = []
    update = model.agent.update
    def capture_update(buffer):
        result = update(buffer)
        metrics.append({key: float(result[key]) for key in (
            "actor_loss", "actor_grad_norm", "boom_gate_active", "boom_valid_fraction",
        )})
        return result
    monkeypatch.setattr(model.agent, "update", capture_update)
    try:
        model.agent.actor_loss_scale.fill_(4.)
        model.learn(total_timesteps=15)
        assert model._global_step == 15 and metrics
        assert any(row["boom_gate_active"] and row["boom_valid_fraction"] > 0 for row in metrics)
        assert all(torch.isfinite(torch.tensor(list(row.values()))).all() for row in metrics)
        assert model.buffer._training_state_version() == 3
        rows = model.buffer._buffer._storage._storage
        valid = rows["behavior_policy_valid"][:model.buffer.size]
        torch.testing.assert_close(
            rows["action"][:model.buffer.size][valid],
            rows["behavior_pre_tanh_action"][:model.buffer.size][valid].tanh(),
        )
    finally:
        model.env.close()


def test_graph_bridge_materializes_preimage_without_aliasing_reusable_storage():
    agent = _agent(outer_boom_alignment="replay_likelihood")
    engine = agent.inner_engine
    preimage = torch.tensor([12.])
    action = preimage.tanh()
    behavior = {"pre_tanh_mean": torch.zeros(1), "log_std": torch.ones(1), "pre_tanh_action": preimage}
    engine._whole_graph_runner = lambda root, **kwargs: (
        action, {"inner_actor_loss_scale_final": torch.tensor(4.)}, [], behavior,
    )
    output = training_graph_act(engine, torch.zeros(1, agent.cfg.latent_dim), return_behavior_policy=True)
    copied_action, _, _, copied_behavior = agent._materialize_action_metrics(
        *output[:3], behavior_policy=output[3],
    )
    preimage.fill_(-12.)
    action.fill_(-1.)
    torch.testing.assert_close(copied_action, torch.ones(1))
    torch.testing.assert_close(copied_behavior["pre_tanh_action"], torch.tensor([12.]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_whole_inner_graph_preserves_executed_preimages_through_native_diagnostics():
    from tests.test_inner_entropy_graph import make_agent
    cfg = deepcopy(make_agent(scope="action", update="per_update").cfg)
    cfg.device = "cuda"
    cfg.compile = cfg.compile_strict = cfg.inner_cuda_graph = True
    cfg.outer_actor_objective = "boom"
    cfg.outer_boom_alignment = "replay_likelihood"
    cfg.store_behavior_policy = cfg.store_behavior_pre_tanh_action = True
    agent = AMBITDMPC2Agent(cfg)
    saved = []
    for index, diagnostic in enumerate((False, False, True, False)):
        action, behavior = agent.act(
            torch.full(cfg.obs_shape["state"], index / 10.),
            collect_diagnostics=diagnostic, return_behavior_policy=True,
        )
        torch.testing.assert_close(action, behavior["pre_tanh_action"].tanh(), rtol=1e-6, atol=1e-6)
        assert bool(agent.last_inner_metrics.get("inner_cuda_graph_used", 0)) is not diagnostic
        for live, snapshot in saved:
            _assert_tree_equal(live, snapshot)
        saved.append((behavior, _clone_tree(behavior)))
    assert agent.inner_engine._whole_graph_runner.calls == 3


@pytest.mark.parametrize("alignment,temperature", [("sampled_action", "scale"), ("replay_likelihood", 1.)])
def test_checkpoint_rejects_changed_likelihood_semantics_before_mutation(alignment, temperature):
    source = _agent(outer_boom_alignment="replay_likelihood")
    target = _agent(outer_boom_alignment=alignment, outer_boom_q_temperature=temperature)
    before = _clone_tree(target.training_state_dict())
    with pytest.raises(ValueError, match="(?i)boom"):
        target.load_training_state_dict(_clone_tree(source.training_state_dict()))
    _assert_tree_equal(target.training_state_dict(), before)
