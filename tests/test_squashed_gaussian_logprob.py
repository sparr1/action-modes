from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from RL.tdmpc2_core.common.world_model import WorldModel
from tests.test_ambi_inner_decoupling import _model as _inner_model


def _model_cfg():
    return SimpleNamespace(
        multitask=False,
        obs_shape={"state": (3,)},
        obs="state",
        task_dim=0,
        num_enc_layers=2,
        enc_dim=16,
        latent_dim=8,
        simnorm_dim=4,
        action_dim=2,
        mlp_dim=16,
        num_bins=1,
        vmin=-5,
        vmax=5,
        episodic=False,
        dropout=0.0,
        log_std_min=-5,
        log_std_max=2,
        log_std_mapping="direct_clamp",
        tau=0.005,
        q_representation="scalar",
        num_q=2,
        q_pair_size=2,
        q_num_bins=1,
        q_vmin=-2,
        q_vmax=2,
    )


def _set_saturated_policy_head(model):
    with torch.no_grad():
        for parameter in model._pi.parameters():
            parameter.zero_()
        model._pi[-1].bias[: model.cfg.action_dim].copy_(
            torch.linspace(
                20.0,
                -20.0,
                model.cfg.action_dim,
                device=model._pi[-1].bias.device,
                dtype=model._pi[-1].bias.dtype,
            )
        )


def _clone_generator(generator):
    clone = torch.Generator(device=generator.device)
    clone.set_state(generator.get_state())
    return clone


def _constant_q(model, value):
    def critic(z, action, **kwargs):
        scalar = action[..., :1] * 0.0 + float(value)
        if kwargs.get("reduction") == "all":
            return scalar.unsqueeze(0).expand(model.q_backend.num_q, *scalar.shape)
        return scalar

    return critic


def _inner_sac_model(**overrides):
    return _inner_model(
        inner_rounds=1,
        inner_model_step_budget=8,
        inner_sac_critic_target="entropy_augmented",
        inner_temperature_mode="auto",
        inner_temperature_updates_per_action=1,
        inner_temperature_initialization="fixed",
        inner_temperature=0.25,
        inner_actor_grad_clip_norm=1e6,
        inner_temperature_grad_clip_norm=1e6,
        **overrides,
    )


def _prepare_saturated_inner_actor(model):
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    _set_saturated_policy_head(
        SimpleNamespace(_pi=engine.state.actor, cfg=model.cfg)
    )
    return engine


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_squash_matches_exact_tanh_transform_at_extremes_and_keeps_gradients(
    dtype,
):
    pre_tanh = torch.tensor(
        [[-100.0, -20.0, -3.0, 0.0, 3.0, 20.0, 100.0]],
        dtype=dtype,
        requires_grad=True,
    )
    mean = torch.linspace(-2.0, 2.0, pre_tanh.shape[-1], dtype=dtype)
    gaussian_log_prob = torch.zeros(1, 1, dtype=dtype)
    rng_before = torch.random.get_rng_state().clone()

    squashed_mean, action, actual = td_math.squash(
        mean,
        pre_tanh,
        gaussian_log_prob,
    )
    transform = torch.distributions.transforms.TanhTransform()
    expected = gaussian_log_prob - transform.log_abs_det_jacobian(
        pre_tanh,
        torch.tanh(pre_tanh),
    ).sum(dim=-1, keepdim=True)
    stable_identity = 2.0 * (
        0.6931471805599453
        - pre_tanh
        - torch.nn.functional.softplus(-2.0 * pre_tanh)
    )
    expected_from_identity = gaussian_log_prob - stable_identity.sum(
        dim=-1,
        keepdim=True,
    )

    torch.testing.assert_close(squashed_mean, torch.tanh(mean), rtol=0, atol=0)
    torch.testing.assert_close(action, torch.tanh(pre_tanh), rtol=0, atol=0)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
    torch.testing.assert_close(
        actual,
        expected_from_identity,
        rtol=0,
        atol=tolerance,
    )
    expected_u20_log_jacobian = torch.tensor(-38.61370563888011, dtype=dtype)
    torch.testing.assert_close(
        td_math.tanh_log_abs_det_jacobian(
            torch.tensor([[20.0]], dtype=dtype)
        ).squeeze(),
        expected_u20_log_jacobian,
        rtol=0,
        atol=tolerance,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(
        pre_tanh.grad,
        2.0 * torch.tanh(pre_tanh.detach()),
        rtol=0,
        atol=tolerance,
    )


def test_squash_compiled_forward_backward_matches_eager():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable in this PyTorch build")

    def forward(pre_tanh):
        gaussian_log_prob = pre_tanh.new_zeros(*pre_tanh.shape[:-1], 1)
        return td_math.squash(pre_tanh * 0.25, pre_tanh, gaussian_log_prob)

    try:
        compiled_forward = torch.compile(forward, backend="eager", fullgraph=True)
        compiled_input = torch.tensor(
            [[-100.0, -20.0, 0.0, 20.0, 100.0]],
            requires_grad=True,
        )
        compiled_outputs = compiled_forward(compiled_input)
    except PermissionError as exc:
        pytest.skip(f"sandbox prevents torch.compile initialization: {exc}")
    except RuntimeError as exc:
        message = str(exc).lower()
        if "operation not permitted" in message or "not supported" in message:
            pytest.skip(f"host cannot initialize torch.compile: {exc}")
        raise

    eager_input = compiled_input.detach().clone().requires_grad_(True)
    eager_outputs = forward(eager_input)
    for compiled_output, eager_output in zip(compiled_outputs, eager_outputs):
        torch.testing.assert_close(compiled_output, eager_output, rtol=0, atol=0)

    compiled_outputs[-1].sum().backward()
    eager_outputs[-1].sum().backward()
    torch.testing.assert_close(compiled_input.grad, eager_input.grad, rtol=0, atol=0)


def test_inner_sac_actor_region_compiled_forward_backward_matches_eager():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable in this PyTorch build")

    model = _inner_sac_model()
    try:
        engine = _prepare_saturated_inner_actor(model)
        z = torch.zeros(4, model.cfg.latent_dim)
        alpha = engine.alpha.detach()
        policy_noise = torch.linspace(
            -1.0,
            1.0,
            4 * model.cfg.action_dim,
        ).reshape(4, model.cfg.action_dim)

        def forward(z_value, alpha_value, noise_value):
            return engine._sac_actor_kernel(
                z_value,
                alpha_value,
                noise_value,
                None,
                True,
            )

        eager_outputs = forward(z, alpha, policy_noise)
        eager_gradients = torch.autograd.grad(
            eager_outputs[2],
            tuple(engine.state.actor_params),
            allow_unused=True,
        )
        try:
            compiled_forward = torch.compile(
                forward,
                backend="eager",
                fullgraph=False,
                dynamic=False,
            )
            compiled_outputs = compiled_forward(z, alpha, policy_noise)
        except PermissionError as exc:
            pytest.skip(f"sandbox prevents torch.compile initialization: {exc}")
        except RuntimeError as exc:
            message = str(exc).lower()
            if "operation not permitted" in message or "not supported" in message:
                pytest.skip(f"host cannot initialize torch.compile: {exc}")
            raise

        compiled_gradients = torch.autograd.grad(
            compiled_outputs[2],
            tuple(engine.state.actor_params),
            allow_unused=True,
        )
        assert len(compiled_outputs) == 10
        for compiled_output, eager_output in zip(compiled_outputs, eager_outputs):
            torch.testing.assert_close(
                compiled_output,
                eager_output,
                rtol=0,
                atol=0,
            )
        for compiled_gradient, eager_gradient in zip(
            compiled_gradients,
            eager_gradients,
        ):
            if eager_gradient is None:
                assert compiled_gradient is None
            else:
                torch.testing.assert_close(
                    compiled_gradient,
                    eager_gradient,
                    rtol=0,
                    atol=0,
                )
    finally:
        model.env.close()


def test_tanh_saturation_statistics_are_detached_and_follow_public_contract():
    pre_tanh = torch.tensor(
        [-8.0, -7.600902, 0.0, 3.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    action = torch.tensor(
        [-1.0, -0.999999, 1.0, 0.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    abs_mean, abs_max, floor_fraction, exact_fraction = (
        td_math.tanh_saturation_statistics(pre_tanh, action)
    )

    torch.testing.assert_close(abs_mean, pre_tanh.detach().abs().mean())
    torch.testing.assert_close(abs_max, torch.tensor(8.0, dtype=torch.float64))
    torch.testing.assert_close(floor_fraction, torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(exact_fraction, torch.tensor(0.5, dtype=torch.float64))
    assert not any(
        statistic.requires_grad
        for statistic in (abs_mean, abs_max, floor_fraction, exact_fraction)
    )


def test_soft_world_model_pi_uses_exact_correction_and_preserves_action_rng():
    model = SoftWorldModel(_model_cfg())
    _set_saturated_policy_head(model)
    latent = torch.zeros(3, model.cfg.latent_dim)
    global_rng_before = torch.random.get_rng_state().clone()

    action, info = model.pi(latent, deterministic=True)
    transform = torch.distributions.transforms.TanhTransform()
    gaussian = torch.distributions.Normal(
        info["pre_tanh_mean"],
        info["log_std"].exp(),
    )
    expected = gaussian.log_prob(info["pre_tanh_action"]).sum(
        dim=-1,
        keepdim=True,
    ) - transform.log_abs_det_jacobian(
        info["pre_tanh_action"],
        action,
    ).sum(dim=-1, keepdim=True)

    torch.testing.assert_close(action, torch.tanh(info["pre_tanh_action"]), rtol=0, atol=0)
    torch.testing.assert_close(info["log_prob"], expected)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before, rtol=0, atol=0)

    info["log_prob"].sum().backward()
    mean_bias_grad = model._pi[-1].bias.grad[: model.cfg.action_dim]
    expected_grad = 3.0 * 2.0 * torch.tanh(torch.tensor([20.0, -20.0]))
    torch.testing.assert_close(mean_bias_grad, expected_grad)

    full_generator = torch.Generator().manual_seed(41)
    sampled_action, _ = model.pi(latent, generator=full_generator)
    full_generator_state = full_generator.get_state()
    action_generator = torch.Generator().manual_seed(41)
    action_only = model.pi_action(latent, generator=action_generator)
    torch.testing.assert_close(action_only, sampled_action, rtol=0, atol=0)
    torch.testing.assert_close(
        action_generator.get_state(),
        full_generator_state,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng_before, rtol=0, atol=0)


def test_world_model_pi_uses_exact_correction_without_extra_rng_draws():
    model = WorldModel(_model_cfg())
    _set_saturated_policy_head(model)
    latent = torch.zeros(3, model.cfg.latent_dim)

    torch.manual_seed(73)
    raw_mean, raw_log_std = model._pi(latent).chunk(2, dim=-1)
    log_std = td_math.log_std(raw_log_std, model.log_std_min, model.log_std_dif)
    eps = torch.randn_like(raw_mean)
    pre_tanh_action = raw_mean + eps * log_std.exp()
    expected_action = torch.tanh(pre_tanh_action)
    expected_log_prob = td_math.gaussian_logprob(
        eps,
        log_std,
    ) - td_math.tanh_log_abs_det_jacobian(pre_tanh_action)
    expected_rng_after = torch.random.get_rng_state().clone()

    torch.manual_seed(73)
    action, info = model.pi(latent, None)

    torch.testing.assert_close(action, expected_action, rtol=0, atol=0)
    torch.testing.assert_close(info["mean"], torch.tanh(raw_mean), rtol=0, atol=0)
    torch.testing.assert_close(info["entropy"], -expected_log_prob)
    torch.testing.assert_close(
        torch.random.get_rng_state(),
        expected_rng_after,
        rtol=0,
        atol=0,
    )


def test_canonical_inner_sac_critic_target_uses_exact_saturated_density(
    monkeypatch,
):
    model = _inner_sac_model()
    engine = _prepare_saturated_inner_actor(model)
    batch = {
        "z": torch.zeros(4, model.cfg.latent_dim),
        "action": torch.zeros(4, model.cfg.action_dim),
        "reward": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        "next_z": torch.zeros(4, model.cfg.latent_dim),
        "terminated": torch.tensor([[0.0], [1.0], [0.0], [1.0]]),
    }
    alpha = torch.tensor(0.25)
    bootstrap_q = 5.0
    policy_generator = _clone_generator(engine.rng.generator("bootstrap"))
    policy_noise = torch.randn(
        (*batch["next_z"].shape[:-1], model.cfg.action_dim),
        generator=policy_generator,
    )
    with torch.no_grad():
        next_action, next_info = model.agent.model.pi(
            batch["next_z"],
            policy=engine.state.actor,
            noise=policy_noise,
            log_std_mapping=model.cfg.inner_log_std_mapping,
            log_std_min=model.cfg.inner_log_std_min,
            log_std_max=model.cfg.inner_log_std_max,
        )
        gaussian = torch.distributions.Normal(
            next_info["pre_tanh_mean"],
            next_info["log_std"].exp(),
        )
        expected_log_prob = gaussian.log_prob(
            next_info["pre_tanh_action"]
        ).sum(dim=-1, keepdim=True) - td_math.tanh_log_abs_det_jacobian(
            next_info["pre_tanh_action"]
        )
        expected_target = batch["reward"] + model.agent.discount * (
            1.0 - batch["terminated"]
        ) * (bootstrap_q - alpha * expected_log_prob)

    assert torch.all(next_action.abs() == 1.0)
    torch.testing.assert_close(next_info["log_prob"], expected_log_prob)
    monkeypatch.setattr(
        engine,
        "_bootstrap_q",
        lambda z, action: z.new_full((z.shape[0], 1), bootstrap_q),
    )
    captured = {}
    original_loss = model.agent.model.critic_loss

    def capture_target(predictions, scalar_target, **kwargs):
        captured["target"] = scalar_target.detach().clone()
        return original_loss(predictions, scalar_target, **kwargs)

    monkeypatch.setattr(model.agent.model, "critic_loss", capture_target)
    with engine.rng.fork("bootstrap"):
        engine._sac_critic_step(batch, alpha)

    torch.testing.assert_close(captured["target"], expected_target)
    torch.testing.assert_close(
        engine.rng.generator("bootstrap").get_state(),
        policy_generator.get_state(),
        rtol=0,
        atol=0,
    )


def test_canonical_inner_sac_actor_and_auto_alpha_use_exact_desaturating_density(
    monkeypatch,
):
    model = _inner_sac_model()
    engine = _prepare_saturated_inner_actor(model)
    batch = {"z": torch.zeros(4, model.cfg.latent_dim)}
    alpha = engine.alpha.detach()
    monkeypatch.setattr(model.agent.model, "Q", _constant_q(model.agent.model, 0.0))

    policy_generator = _clone_generator(engine.rng.generator("gradient_policy"))
    policy_noise = torch.randn(
        (*batch["z"].shape[:-1], model.cfg.action_dim),
        generator=policy_generator,
    )
    action, info = model.agent.model.pi(
        batch["z"],
        policy=engine.state.actor,
        noise=policy_noise,
        log_std_mapping=model.cfg.inner_log_std_mapping,
        log_std_min=model.cfg.inner_log_std_min,
        log_std_max=model.cfg.inner_log_std_max,
    )
    gaussian = torch.distributions.Normal(
        info["pre_tanh_mean"],
        info["log_std"].exp(),
    )
    expected_log_prob = gaussian.log_prob(info["pre_tanh_action"]).sum(
        dim=-1,
        keepdim=True,
    ) - td_math.tanh_log_abs_det_jacobian(info["pre_tanh_action"])
    torch.testing.assert_close(info["log_prob"], expected_log_prob)
    assert torch.all(action.abs() == 1.0)

    actor_outputs = engine._sac_actor_kernel(
        batch["z"],
        alpha,
        policy_noise,
        None,
        True,
    )
    actor_loss = actor_outputs[2]
    final_bias = engine.state.actor[-1].bias
    actor_bias_grad = torch.autograd.grad(actor_loss, final_bias)[0][
        : model.cfg.action_dim
    ]
    expected_bias_grad = alpha * (
        2.0 * torch.tanh(info["pre_tanh_action"].detach())
    ).mean(dim=0)
    torch.testing.assert_close(actor_loss, alpha * expected_log_prob.mean())
    torch.testing.assert_close(actor_bias_grad, expected_bias_grad)
    assert torch.all(actor_bias_grad > 0.0)

    mean_bias_before = final_bias[: model.cfg.action_dim].detach().clone()
    log_alpha_before = engine.state.log_alpha.detach().clone()
    expected_temperature_loss = -(
        log_alpha_before
        * (expected_log_prob + engine._resolved_inner_target_entropy()).detach()
    ).mean()
    global_rng_before = torch.random.get_rng_state().clone()

    metrics = engine._sac_policy_step(
        batch,
        update_temperature=True,
        update_actor=True,
        alpha=alpha,
    )

    torch.testing.assert_close(metrics["actor_loss"], alpha * expected_log_prob.mean())
    torch.testing.assert_close(metrics["temperature_loss"], expected_temperature_loss)
    assert torch.all(
        engine.state.actor[-1].bias[: model.cfg.action_dim] < mean_bias_before
    )
    assert engine.state.log_alpha > log_alpha_before
    torch.testing.assert_close(
        engine.rng.generator("gradient_policy").get_state(),
        policy_generator.get_state(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.random.get_rng_state(),
        global_rng_before,
        rtol=0,
        atol=0,
    )
