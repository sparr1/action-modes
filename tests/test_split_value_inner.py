"""Split inner values retain exact seeds and obey a reward-prior handoff."""

from contextlib import contextmanager
from copy import deepcopy

import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _tiny_model


@contextmanager
def _prepared(**overrides):
    params = dict(
        critic_value_mode="return_entropy", q_representation="distributional",
        inner_finite_horizon=True, inner_entropy_enabled=False,
        inner_value_initialization="return", ent_coef=0.2,
        inner_temperature_mode="inherit_outer", inner_rounds=1,
        inner_updates_per_round=1, num_q=3, q_num_bins=11, num_bins=11,
    )
    params.update(overrides)
    if params.get("inner_critic_adaptation") == "lora_rl":
        params.setdefault("inner_critic_lora_rank", 4)
    model = _tiny_model(**params)
    try:
        engine = model.agent.inner_engine
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        yield model, engine
    finally:
        model.env.close()


def _batch(engine, count=4):
    return dict(
        z=torch.randn(count, engine.cfg.latent_dim),
        action=torch.randn(count, engine.cfg.action_dim).tanh(),
        reward=torch.linspace(-0.5, 1.0, count).unsqueeze(-1),
        next_z=torch.randn(count, engine.cfg.latent_dim),
        terminated=torch.tensor([[0.], [0.], [1.], [1.]]),
        horizon_end=torch.tensor([[0.], [1.], [0.], [1.]]),
    )


def _nonzero_outputs(engine):
    with torch.no_grad():
        for index, critic in enumerate(engine.model._Qs):
            critic[-1].weight.copy_(torch.linspace(
                -.04, .06, critic[-1].weight.numel()
            ).reshape_as(critic[-1].weight) + index * .002)
            critic[-1].bias.copy_(torch.linspace(-.2, .3, critic[-1].bias.numel()))
        engine.model.soft_update_target_Q(1.0)
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)


@pytest.mark.parametrize("initialization", ["return", "soft"])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_exact_initial_composition_survives_cloning_and_lora(initialization, adaptation):
    with _prepared(inner_value_initialization=initialization,
                   inner_critic_adaptation=adaptation,
                   sac_actor_loss_scale_mode="tdmpc2_percentile_range") as (model, engine):
        model.agent.actor_loss_scale.fill_(7.0)
        _nonzero_outputs(engine)
        z = torch.randn(5, model.cfg.latent_dim)
        a = torch.randn(5, model.cfg.action_dim).tanh()
        outer = model.agent.model.q_values(z, a)
        inner = model.agent.model.q_values(z, a, qs=engine.state.critic)
        c = 1.4 if initialization == "soft" else 0.0
        torch.testing.assert_close(inner, outer, rtol=1e-6, atol=1e-6)
        actual = engine.model.project_values(inner, weights=engine.state.value_composition)
        expected = outer[..., 0, :] + c * outer[..., 1, :]
        torch.testing.assert_close(actual, expected)
        frozen = engine.state.value_composition.clone()
        model.agent.actor_loss_scale.fill_(19.0)
        model.agent.fixed_ent_coef.fill_(.9)
        torch.testing.assert_close(engine.state.value_composition, frozen, rtol=0, atol=0)


@pytest.mark.parametrize("entropy", [False, True])
@pytest.mark.parametrize("initialization", ["return", "soft"])
def test_component_targets_equal_scalar_sac_and_terminal_return(entropy, initialization):
    with _prepared(inner_entropy_enabled=entropy, inner_value_initialization=initialization,
                   sac_actor_loss_scale_mode="tdmpc2_percentile_range") as (_, engine):
        reward = torch.tensor([[1.], [2.], [3.], [4.]])
        terminated = torch.tensor([[0.], [0.], [1.], [1.]])
        boundary = torch.tensor([[0.], [1.], [0.], [1.]])
        components = torch.tensor([[[5.], [-2.]], [[float('nan')], [float('nan')]],
                                   [[float('inf')], [float('inf')]],
                                   [[float('nan')], [float('nan')]]])
        tail = torch.tensor([[float('nan')], [11.], [float('nan')], [float('inf')]])
        h = torch.tensor([[-3.], [float('nan')], [float('inf')], [float('nan')]])
        targets = engine._split_sac_targets(
            reward, terminated, components, h, torch.tensor(.25), boundary, tail,
            actor_loss_scale=torch.tensor([4.]),
        )
        gamma = engine.agent.discount
        primary = reward + gamma * torch.tensor([[2. if entropy else 5.], [11.], [0.], [0.]])
        residual = gamma * torch.tensor([[-2.], [0.], [0.], [0.]])
        torch.testing.assert_close(targets[..., 0, :], primary)
        torch.testing.assert_close(targets[..., 1, :], residual)
        c = engine.state.value_composition[1]
        expected = primary + c * residual
        torch.testing.assert_close(engine.model.project_values(targets, weights=engine.state.value_composition), expected)


def test_kernel_selects_both_components_from_same_member_and_frozen_online_prior(monkeypatch):
    with _prepared(inner_value_initialization="soft", inner_entropy_enabled=True) as (_, engine):
        engine.state.value_composition.copy_(torch.tensor([1., .5]))
        batch = _batch(engine)
        # Member 0 has the smallest return, member 1 the smallest composed Q;
        # independently minimizing the two components would make a fictitious Q.
        candidates = torch.tensor([[[1.], [10.]], [[3.], [-4.]], [[-50.], [0.]]])
        def values(z, a, **kwargs):
            assert kwargs.get("qs") is engine.state.critic_target
            return candidates[:, None].expand(-1, len(z), -1, -1)
        def policy(z, **kwargs):
            return z.new_full((len(z), engine.cfg.action_dim), .7 if "policy" not in kwargs else -.3), {"log_prob": z.new_full((len(z), 1), -2.)}
        def outer_q(z, a, **kwargs):
            assert kwargs == {"reduction": "min_pair", "projection": "return"}
            torch.testing.assert_close(a, torch.full_like(a, .7))
            return z.new_full((len(z), 1), 11.)
        monkeypatch.setattr(engine.model, "q_values", values)
        monkeypatch.setattr(engine.model, "pi", policy)
        monkeypatch.setattr(engine.model, "Q", outer_q)
        original = engine.model.critic_loss
        captured = []
        def loss(prediction, target, **kwargs):
            captured.append(target.clone())
            return original(prediction, target, **kwargs)
        monkeypatch.setattr(engine.model, "critic_loss", loss)
        result = engine._sac_critic_kernel(
            batch["z"], batch["action"], batch["reward"], batch["next_z"], batch["terminated"],
            torch.tensor(.25, requires_grad=True), torch.zeros(4, 1), torch.tensor([0, 1]),
            batch["horizon_end"], torch.zeros(4, 1),
        )
        expected_u = batch["reward"] + engine.agent.discount * torch.tensor([[3.5], [11.], [0.], [0.]])
        expected_w = engine.agent.discount * torch.tensor([[-4.], [0.], [0.], [0.]])
        torch.testing.assert_close(captured[0], torch.stack((expected_u, expected_w), -2))
        assert not result[2].requires_grad
        result[0].backward()
        assert all(p.grad is None for p in engine.model.parameters())


@pytest.mark.parametrize("initialization", ["return", "soft"])
def test_component_loss_weights_and_actor_gradients(initialization):
    with _prepared(inner_value_initialization=initialization, inner_entropy_enabled=True) as (_, engine):
        _nonzero_outputs(engine)
        batch = _batch(engine)
        args = (batch["z"], batch["action"], batch["reward"], batch["next_z"], batch["terminated"],
                torch.tensor(.25), torch.zeros(4, 1), torch.tensor([0, 1]),
                batch["horizon_end"], torch.zeros(4, 1))
        output = engine._sac_critic_kernel(*args)
        expected_loss = output[4]["primary_critic_loss"]
        if initialization == "soft":
            expected_loss = .5 * (expected_loss + output[4]["initialization_residual_critic_loss"])
        torch.testing.assert_close(output[0], expected_loss)
        output[0].backward()
        for critic in engine.state.critic:
            head_grad = critic[-1].weight.grad.reshape(2, engine.cfg.q_num_bins, -1)
            assert head_grad[0].abs().sum() > 0
            if initialization == "return":
                assert head_grad[1].count_nonzero() == 0
            else:
                assert head_grad[1].abs().sum() > 0
        engine.state.critic_optim.zero_grad(set_to_none=True)
        actor = engine._sac_actor_kernel(batch["z"], torch.tensor(.25), torch.zeros(4, 1),
                                         torch.tensor([0, 1]), True)
        torch.testing.assert_close(actor[2], (.25 * actor[0]).mean() - actor[3])
        actor[2].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in engine.state.actor_params)
        assert all(p.grad is None for p in engine.state.critic_params)
        assert all(p.grad is None for p in engine.model.parameters())


@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
@pytest.mark.parametrize("entropy", [False, True])
def test_action_updates_temperature_only_when_enabled_and_keeps_outer_private(entropy, adaptation):
    with _prepared(inner_entropy_enabled=entropy, inner_temperature_mode="auto",
                   inner_value_initialization="soft", inner_updates_per_round=2,
                   inner_critic_adaptation=adaptation) as (model, engine):
        before = _clone_tree(engine.model.state_dict())
        rng = torch.random.get_rng_state().clone()
        model.agent.act(torch.zeros(3))
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_temperature_optimizer_steps"] == (2 if entropy else 0)
        assert metrics["inner_alpha"] > 0 if entropy else metrics["inner_alpha"] == 0
        _assert_tree_equal(engine.model.state_dict(), before)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        if not entropy:
            assert engine.state.log_alpha is None and engine.state.temperature_optim is None


def test_inner_scale_is_frozen_across_updates_and_outer_storage_is_private():
    with _prepared(inner_entropy_enabled=True, sac_actor_loss_scale_mode="tdmpc2_percentile_range",
                   inner_value_initialization="soft", inner_updates_per_round=2) as (model, engine):
        model.agent.actor_loss_scale.fill_(7.)
        model.agent.act(torch.zeros(3))
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_actor_loss_scale"] == 7.
        assert model.agent.actor_loss_scale.item() == 7.
        assert metrics["inner_value_composition_coefficient"] == pytest.approx(1.4)


def test_inner_semantic_checkpoint_roundtrip_and_transactional_rejection():
    with _prepared(inner_value_initialization="soft") as (model, engine):
        model.agent.act(torch.zeros(3))
        model.agent.prepare_training_resume_boundary()
        payload = _clone_tree(engine.training_state_dict())
        assert payload["version"] == 5
        assert payload["split_value_spec"]["inner_roles"] == ["primary", "initialization_residual"]
        assert payload["workspace"]["value_composition"] is None
        assert payload["workspace"]["value_roles"] is None
        engine.load_training_state_dict(payload)
        _assert_tree_equal(engine.training_state_dict(), payload)
        invalid = deepcopy(payload)
        invalid["split_value_spec"]["outer_roles"] = ["return", "soft"]
        with pytest.raises(ValueError, match="semantics"):
            engine.load_training_state_dict(invalid)
        _assert_tree_equal(engine.training_state_dict(), payload)
        invalid = deepcopy(payload)
        invalid["workspace"]["value_composition"] = torch.tensor([1., .2])
        with pytest.raises(ValueError, match="inventory"):
            engine.load_training_state_dict(invalid)
        _assert_tree_equal(engine.training_state_dict(), payload)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("initialization", ["return", "soft"])
def test_inner_compiled_kernels_match_eager_values_and_gradients(initialization, backend):
    with _prepared(inner_value_initialization=initialization, inner_entropy_enabled=True) as (_, engine):
        _nonzero_outputs(engine)
        batch = _batch(engine)
        critic_args = (batch["z"], batch["action"], batch["reward"], batch["next_z"], batch["terminated"],
                       torch.tensor(.25), torch.zeros(4, 1), torch.tensor([0, 1]),
                       batch["horizon_end"], torch.zeros(4, 1))
        # Supply the prior pair too, so a compiled call cannot redraw a different tail.
        engine._prior_bootstrap = lambda z, noise: engine.model.Q(
            z, engine.model.pi(z, noise=noise)[0], projection="return", reduction="min_pair",
            pair_indices=torch.tensor([0, 1]), trusted_pair_indices=True,
        )
        actor_args = (batch["z"], torch.tensor(.25), torch.zeros(4, 1), torch.tensor([0, 1]), True)
        for kernel, args, params, loss_index in (
            (engine._sac_critic_kernel, critic_args, engine.state.critic_params, 0),
            (engine._sac_actor_kernel, actor_args, engine.state.actor_params, 2),
        ):
            eager = kernel(*args)
            eager_grads = torch.autograd.grad(eager[loss_index], params)
            compiled = torch.compile(kernel, backend=backend, fullgraph=True)(*args)
            torch.testing.assert_close(compiled, eager, rtol=3e-5, atol=5e-6)
            actual_grads = torch.autograd.grad(compiled[loss_index], params)
            for expected, actual in zip(eager_grads, actual_grads):
                torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-6)


def test_automatic_alpha_snapshot_is_shared_by_critic_and_actor(monkeypatch):
    with _prepared(inner_entropy_enabled=True, inner_temperature_mode="auto",
                   inner_value_initialization="soft", inner_updates_per_round=2) as (model, engine):
        seen = []
        original_critic = engine._sac_critic_step
        original_actor = engine._sac_policy_step
        def critic(batch, alpha, **kwargs):
            seen.append(("critic", alpha.detach().clone(), engine.state.value_composition.clone()))
            return original_critic(batch, alpha, **kwargs)
        def actor(batch, **kwargs):
            seen.append(("actor", kwargs["alpha"].detach().clone(), engine.state.value_composition.clone()))
            return original_actor(batch, **kwargs)
        monkeypatch.setattr(engine, "_sac_critic_step", critic)
        monkeypatch.setattr(engine, "_sac_policy_step", actor)
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert [item[0] for item in seen] == ["critic", "actor", "critic", "actor"]
        torch.testing.assert_close(seen[0][1], seen[1][1], rtol=0, atol=0)
        torch.testing.assert_close(seen[2][1], seen[3][1], rtol=0, atol=0)
        assert not torch.equal(seen[0][1], seen[2][1])
        for _, _, composition in seen:
            torch.testing.assert_close(composition, torch.tensor([1., .2]))


@pytest.mark.parametrize("kernel", ["step", "dense"])
@pytest.mark.parametrize("use_compile_support", [False, True])
def test_imagined_rewards_use_mean_preserving_decoder(kernel, use_compile_support):
    with _prepared() as (_, engine):
        with torch.no_grad():
            engine.model._reward[-1].bias.copy_(torch.linspace(-1., 2., engine.cfg.num_bins))
        root = torch.zeros(1, engine.cfg.latent_dim)
        raw = engine.model.reward_codec.decode(engine.model._reward[-1].bias)
        # This law is deliberately asymmetric; symexp(mean symlog) would differ.
        engine.cfg.compile = use_compile_support
        support = torch.linspace(engine.cfg.vmin, engine.cfg.vmax, engine.cfg.num_bins)
        if kernel == "step":
            _, reward, _ = engine._rollout_step_kernel(root, torch.zeros(1, 1), support)
            torch.testing.assert_close(reward, raw.reshape(1, 1))
        else:
            noise = torch.zeros(engine.cfg.inner_rollout_horizon, engine.cfg.inner_rollouts_per_round, 1)
            outputs = engine._dense_rollout_kernel(root, noise, support)
            # The dense kernel returns flattened transitions followed by sums.
            transition_tensor = outputs[0]
            reward_index = engine.cfg.latent_dim + engine.cfg.action_dim
            torch.testing.assert_close(transition_tensor[..., reward_index],
                                       torch.full_like(transition_tensor[..., reward_index], raw.item()))


@pytest.mark.parametrize("probe_mode", ["legacy", "outer_tail"])
def test_split_diagnostic_probes_preserve_private_rng_and_return_tail(probe_mode):
    from RL.tdmpc2_core.inner_trace import InnerActionTrace
    with _prepared(inner_value_initialization="soft", inner_entropy_enabled=True,
                   inner_diagnostic_rollouts=2) as (model, engine):
        _nonzero_outputs(engine)
        before = _clone_tree(engine.model.state_dict())
        rng = torch.random.get_rng_state().clone()
        trace = InnerActionTrace(probes=True, probe_mode=probe_mode, probe_rollouts=2, probe_horizon=2)
        model.agent.act(torch.zeros(3), trace=trace)
        assert any(event["phase"] == "probe" for event in trace.events)
        assert all(torch.isfinite(torch.tensor(value)) for event in trace.events for value in event["metrics"].values())
        _assert_tree_equal(engine.model.state_dict(), before)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
