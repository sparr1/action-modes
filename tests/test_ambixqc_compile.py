from copy import deepcopy

import gymnasium as gym
import pytest
import torch

from RL.AMBIXQC import AMBIXQC
from RL.tdmpc2_core.xqc_controller import (
    LatentXQCBatch,
    LatentXQCConfig,
    LatentXQCController,
)


def _build_cfg(**params):
    algorithm = object.__new__(AMBIXQC)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 12,
    }
    algorithm.experiment_params = {}
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def _controller(*, device="cpu"):
    torch.manual_seed(17)
    return LatentXQCController(
        3,
        1,
        LatentXQCConfig(
            actor_net_arch=(8, 8),
            critic_net_arch=(8, 8),
            num_atoms=11,
            vmin=-2.0,
            vmax=2.0,
            tau=0.005,
            target_update_interval=1,
            policy_delay=3,
            init_temperature=0.01,
            target_entropy=-0.5,
            optimizer_backend="single_tensor",
        ),
    ).to(device)


def _batch(*, device="cpu", leading=(4,), latent_requires_grad=False):
    generator = torch.Generator(device=device)
    generator.manual_seed(23)
    latents = torch.randn(
        leading + (3,), device=device, generator=generator
    ).requires_grad_(latent_requires_grad)
    return LatentXQCBatch(
        latents=latents,
        actions=torch.randn(
            leading + (1,), device=device, generator=generator
        ).tanh(),
        rewards=torch.randn(leading + (1,), device=device, generator=generator),
        next_latents=torch.randn(
            leading + (3,), device=device, generator=generator
        ),
        bootstrap_mask=torch.ones(leading + (1,), device=device),
        discount=torch.full(leading + (1,), 0.99, device=device),
    )


def _noise(*, device="cpu", leading=(4,), seed=29):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(leading + (1,), device=device, generator=generator)


def _assert_objective_close(left, right, *, atol=1e-6, rtol=1e-5):
    assert vars(left).keys() == vars(right).keys()
    for name, left_value in vars(left).items():
        torch.testing.assert_close(
            left_value, vars(right)[name], atol=atol, rtol=rtol
        )


def _assert_module_close(left, right, *, atol=1e-6, rtol=1e-5):
    assert left.state_dict().keys() == right.state_dict().keys()
    for key, value in left.state_dict().items():
        torch.testing.assert_close(
            value, right.state_dict()[key], atol=atol, rtol=rtol
        )


def test_compile_flags_default_off_and_accept_explicit_booleans():
    defaults = _build_cfg()
    requested = _build_cfg(compile=True, compile_strict=True)

    assert defaults.compile is False
    assert defaults.compile_strict is False
    assert requested.compile is True
    assert requested.compile_strict is True


@pytest.mark.parametrize("key", ("compile", "compile_strict"))
def test_compile_flags_reject_non_booleans(key):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key: 1})


def test_cpu_compile_request_is_inactive_and_propagates_to_inner_clone():
    controller = _controller()
    controller.configure_compile(enabled=True, strict=True)

    assert controller.compile_status == {
        "requested": True,
        "enabled": False,
        "strict": True,
        "critic_compiled": False,
        "actor_compiled": False,
        "fallback": False,
    }

    workspace = controller.clone_for_inner(
        actor_lr=3e-4,
        critic_lr=2e-4,
        transition_steps=8,
    )
    assert workspace.controller.compile_status == controller.compile_status


def test_mocked_compile_is_lazy_and_each_objective_reuses_one_cached_graph(
    monkeypatch,
):
    controller = _controller()
    controller.configure_compile(enabled=True, strict=False)
    # CPU requests are intentionally inactive. Force only the region execution
    # bit so this test can exercise the compiler lifecycle without a GPU.
    controller._critic_loss_region.enabled = True
    controller._actor_loss_region.enabled = True
    compile_calls = []

    def fake_compile(function, **kwargs):
        compile_calls.append((function.__name__, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    batch = _batch()
    next_noise = _noise(seed=31)
    actor_noise = _noise(seed=37)

    for _ in range(2):
        controller.critic_objective(batch, next_noise=next_noise)
        controller.actor_objective(
            batch.latents, actor_noise=actor_noise
        )

    assert compile_calls == [
        (
            "_critic_loss_components",
            {"fullgraph": False, "dynamic": False},
        ),
        (
            "_actor_loss_components",
            {"fullgraph": False, "dynamic": False},
        ),
    ]
    assert controller.compile_status == {
        "requested": True,
        "enabled": True,
        "strict": False,
        "critic_compiled": True,
        "actor_compiled": True,
        "fallback": False,
    }


def test_post_device_apply_rebuilds_regions_around_live_bn_buffers():
    controller = _controller()
    controller.configure_compile(enabled=True, strict=True)
    old_critic_region = controller._critic_loss_region
    old_actor_region = controller._actor_loss_region
    old_critic_region._compiled = object()
    old_actor_region._compiled = object()

    controller.to(dtype=torch.float64)

    assert controller._critic_loss_region is not old_critic_region
    assert controller._actor_loss_region is not old_actor_region
    assert controller._critic_loss_region._compiled is None
    assert controller._actor_loss_region._compiled is None
    assert all(
        captured is live
        for captured, live in zip(
            controller._critic_loss_region.mutable_buffers,
            controller.critic.buffers(),
        )
    )
    assert all(
        captured is live
        for captured, live in zip(
            controller._actor_loss_region.mutable_buffers,
            controller.actor.buffers(),
        )
    )
    assert controller.compile_status == {
        "requested": True,
        "enabled": False,
        "strict": True,
        "critic_compiled": False,
        "actor_compiled": False,
        "fallback": False,
    }


def test_non_strict_failure_rolls_back_bn_before_eager_fallback(monkeypatch):
    reference = _controller()
    candidate = _controller()
    candidate.load_state_dict(deepcopy(reference.state_dict()))
    reference.configure_compile(enabled=False, strict=False)
    candidate.configure_compile(enabled=True, strict=False)
    candidate._critic_loss_region.enabled = True
    mutated_buffer = candidate._critic_loss_region.mutable_buffers[0]

    def fake_compile(_function, **_kwargs):
        def broken(*_args):
            with torch.no_grad():
                mutated_buffer.add_(100.0)
            raise RuntimeError("injected compiler failure")

        return broken

    monkeypatch.setattr(torch, "compile", fake_compile)
    batch = _batch()
    next_noise = _noise(seed=41)
    expected = reference.critic_objective(batch, next_noise=next_noise)
    with pytest.warns(RuntimeWarning, match="Falling back"):
        actual = candidate.critic_objective(batch, next_noise=next_noise)

    _assert_objective_close(actual, expected)
    _assert_module_close(candidate.critic, reference.critic)
    assert candidate.compile_status == {
        "requested": True,
        "enabled": False,
        "strict": False,
        "critic_compiled": False,
        "actor_compiled": False,
        "fallback": True,
    }


def test_strict_first_runtime_failure_propagates_without_eager_retry(monkeypatch):
    controller = _controller()
    controller.configure_compile(enabled=True, strict=True)
    region = controller._critic_loss_region
    region.enabled = True
    eager = region.eager
    eager_calls = 0

    def tracked_eager(*args):
        nonlocal eager_calls
        eager_calls += 1
        return eager(*args)

    def fake_compile(_function, **_kwargs):
        def broken(*_args):
            raise RuntimeError("injected strict compiler failure")

        return broken

    region.eager = tracked_eager
    monkeypatch.setattr(torch, "compile", fake_compile)
    with pytest.raises(
        RuntimeError, match="Compiled AMBI-XQC critic loss failed at runtime"
    ):
        controller.critic_objective(
            _batch(), next_noise=_noise(seed=67)
        )

    assert eager_calls == 0
    assert region.enabled is True
    assert region.failed is False
    assert controller.compile_status["fallback"] is False


def test_success_then_runtime_failure_never_retries_eager(monkeypatch):
    controller = _controller()
    controller.configure_compile(enabled=True, strict=False)
    region = controller._critic_loss_region
    region.enabled = True
    eager = region.eager
    eager_calls = 0
    compiled_calls = 0

    def tracked_eager(*args):
        nonlocal eager_calls
        eager_calls += 1
        return eager(*args)

    def fake_compile(function, **_kwargs):
        def compiled(*args):
            nonlocal compiled_calls
            compiled_calls += 1
            if compiled_calls == 2:
                raise RuntimeError("injected late compiler failure")
            return function(*args)

        return compiled

    region.eager = tracked_eager
    monkeypatch.setattr(torch, "compile", fake_compile)
    batch = _batch()
    next_noise = _noise(seed=71)
    controller.critic_objective(batch, next_noise=next_noise)
    assert eager_calls == 1

    with pytest.raises(
        RuntimeError, match="Compiled AMBI-XQC critic loss failed at runtime"
    ):
        controller.critic_objective(batch, next_noise=next_noise)

    assert compiled_calls == 2
    assert eager_calls == 1
    assert region.enabled is True
    assert region.failed is False
    assert controller.compile_status == {
        "requested": True,
        "enabled": True,
        "strict": False,
        "critic_compiled": True,
        "actor_compiled": False,
        "fallback": False,
    }


def test_compile_runtime_state_is_excluded_from_module_checkpoint_state():
    controller = _controller()
    before = deepcopy(controller.state_dict())
    controller.configure_compile(enabled=True, strict=True)
    controller._critic_loss_region._compiled = object()
    controller._actor_loss_region.failed = True
    after = controller.state_dict()

    assert after.keys() == before.keys()
    assert all("compile" not in key for key in after)
    for key, value in before.items():
        torch.testing.assert_close(after[key], value, atol=0, rtol=0)


def _paired_cuda_controllers():
    eager = _controller(device="cuda")
    compiled = _controller(device="cuda")
    compiled.load_state_dict(deepcopy(eager.state_dict()))
    eager.configure_compile(enabled=False, strict=False)
    compiled.configure_compile(enabled=True, strict=True)
    return eager, compiled


def _assert_parameter_grads_close(left, right, *, atol=1e-4, rtol=1e-4):
    for (left_name, left_parameter), (
        right_name,
        right_parameter,
    ) in zip(left.named_parameters(), right.named_parameters()):
        assert left_name == right_name
        assert (left_parameter.grad is None) == (right_parameter.grad is None)
        if left_parameter.grad is not None:
            torch.testing.assert_close(
                left_parameter.grad,
                right_parameter.grad,
                atol=atol,
                rtol=rtol,
            )


def _assert_nested_close(left, right, *, atol=1e-5, rtol=1e-4):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
        return
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_close(left[key], right[key], atol=atol, rtol=rtol)
        return
    if isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_close(
                left_item, right_item, atol=atol, rtol=rtol
            )
        return
    if isinstance(left, float):
        assert left == pytest.approx(right, abs=atol, rel=rtol)
        return
    assert left == right


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_compiled_latent_objectives_match_eager_outputs_gradients_and_bn():
    eager, compiled = _paired_cuda_controllers()
    eager_batch = _batch(device="cuda", latent_requires_grad=True)
    compiled_batch = _batch(device="cuda", latent_requires_grad=True)
    next_noise = _noise(device="cuda", seed=43)

    eager_critic = eager.critic_objective(
        eager_batch, next_noise=next_noise, reward_scale=2.0
    )
    compiled_critic = compiled.critic_objective(
        compiled_batch, next_noise=next_noise, reward_scale=2.0
    )
    _assert_objective_close(eager_critic, compiled_critic)
    _assert_module_close(eager.critic, compiled.critic)

    eager_critic.loss.backward()
    compiled_critic.loss.backward()
    _assert_parameter_grads_close(eager.critic, compiled.critic)
    torch.testing.assert_close(
        eager_batch.latents.grad,
        compiled_batch.latents.grad,
        atol=1e-4,
        rtol=1e-4,
    )

    eager.zero_grad(set_to_none=True)
    compiled.zero_grad(set_to_none=True)
    eager_latents = _batch(device="cuda").latents
    compiled_latents = eager_latents.detach().clone()
    actor_noise = _noise(device="cuda", seed=47)
    eager_actor = eager.actor_objective(
        eager_latents, actor_noise=actor_noise
    )
    compiled_actor = compiled.actor_objective(
        compiled_latents, actor_noise=actor_noise
    )
    _assert_objective_close(eager_actor, compiled_actor)
    _assert_module_close(eager.actor, compiled.actor)

    eager_actor.loss.backward()
    compiled_actor.loss.backward()
    _assert_parameter_grads_close(eager.actor, compiled.actor)
    assert compiled.compile_status == {
        "requested": True,
        "enabled": True,
        "strict": True,
        "critic_compiled": True,
        "actor_compiled": True,
        "fallback": False,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_compiled_workspace_matches_four_eager_state_machine_slots():
    eager_controller, compiled_controller = _paired_cuda_controllers()
    eager = eager_controller.make_workspace(
        actor_lr=3e-4,
        critic_lr=2e-4,
        actor_lr_end=3e-5,
        critic_lr_end=3e-5,
        transition_steps=8,
        optimizer_backend="single_tensor",
    )
    compiled = compiled_controller.make_workspace(
        actor_lr=3e-4,
        critic_lr=2e-4,
        actor_lr_end=3e-5,
        critic_lr_end=3e-5,
        transition_steps=8,
        optimizer_backend="single_tensor",
    )
    batch = _batch(device="cuda")
    next_noise = _noise(device="cuda", seed=53)
    actor_noise = _noise(device="cuda", seed=59)

    for _ in range(4):
        eager_metrics = eager.update(
            batch, next_noise=next_noise, actor_noise=actor_noise
        )
        compiled_metrics = compiled.update(
            batch, next_noise=next_noise, actor_noise=actor_noise
        )

    for key in eager_metrics:
        atol, rtol = (
            (2e-4, 1e-3)
            if key in {"actor_loss", "q_policy_mean"}
            else (1e-5, 1e-4)
        )
        _assert_nested_close(
            eager_metrics[key], compiled_metrics[key], atol=atol, rtol=rtol
        )
    for eager_module, compiled_module in (
        (eager_controller.actor, compiled_controller.actor),
        (eager_controller.critic, compiled_controller.critic),
        (eager_controller.critic_target, compiled_controller.critic_target),
    ):
        for key, eager_value in eager_module.state_dict().items():
            if "batch_norm.bias" in key:
                atol, rtol = 1e-3, 1e-3
            elif "batch_norm.running_mean" in key:
                atol, rtol = 1e-4, 1e-3
            else:
                atol, rtol = 1e-5, 1e-4
            torch.testing.assert_close(
                compiled_module.state_dict()[key],
                eager_value,
                atol=atol,
                rtol=rtol,
            )
    _assert_nested_close(
        eager.state_dict(), compiled.state_dict(), atol=1e-3, rtol=1e-3
    )
    assert eager.update_step == compiled.update_step == 4
    assert eager.actor_optimizer_steps == compiled.actor_optimizer_steps == 2
    assert (
        eager.temperature_optimizer_steps
        == compiled.temperature_optimizer_steps
        == 2
    )


def _tiny_params(*, device, compile):
    return {
        "device": device,
        "model_size": None,
        "enc_dim": 16,
        "mlp_dim": 16,
        "latent_dim": 8,
        "num_enc_layers": 2,
        "simnorm_dim": 4,
        "num_bins": 5,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "train_unroll_horizon": 2,
        "buffer_size": 32,
        "seed_steps": 4,
        "pretrain_steps": 1,
        "utd": 1,
        "compile": compile,
        "compile_strict": compile,
        "episodic": False,
        "discount": 0.99,
        "wandb": False,
        "xqc_actor_net_arch": [8, 8],
        "xqc_critic_net_arch": [8, 8],
        "xqc_num_atoms": 11,
        "xqc_vmin": -2,
        "xqc_vmax": 2,
        "xqc_optimizer_backend": "single_tensor",
        "inner_rounds": 1,
        "inner_rollouts_per_round": 2,
        "inner_rollout_horizon": 2,
        "inner_updates_per_round": 1,
        "inner_batch_size": 2,
        "inner_replay_capacity": 4,
    }


def _tiny_wrapper(*, device, compile):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    return AMBIXQC(
        "AMBIXQC",
        env,
        _tiny_params(device=device, compile=compile),
        {"seed": 3, "device": device, "env": "test", "total_steps": 10},
        {},
    )


def test_outer_and_inner_fallback_status_reads_live_compile_regions():
    wrapper = _tiny_wrapper(device="cpu", compile=True)
    try:
        agent = wrapper.agent
        agent.xqc_controller._actor_loss_region.failed = True
        assert agent.xqc_controller.compile_status["fallback"] is True

        engine = agent.inner_engine
        engine._prepare_action()
        try:
            controller = engine.state.workspace.controller
            controller._critic_loss_region.failed = True
            assert engine._compile_fallback_metrics() == {
                "inner_compile_rollout_fallback": 0.0,
                "inner_compile_critic_fallback": 1.0,
                "inner_compile_actor_fallback": 0.0,
                "inner_compile_fallback": 1.0,
            }
        finally:
            engine._release_action()
    finally:
        wrapper.env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_compiled_outer_critic_preserves_deep_recurrent_world_gradients():
    eager_wrapper = _tiny_wrapper(device="cuda", compile=False)
    compiled_wrapper = _tiny_wrapper(device="cuda", compile=True)
    try:
        eager = eager_wrapper.agent
        compiled = compiled_wrapper.agent
        compiled.load_state_dict(deepcopy(eager.state_dict()))
        generator = torch.Generator(device="cuda")
        generator.manual_seed(61)
        horizon = int(eager.cfg.train_unroll_horizon)
        batch_size = int(eager.cfg.batch_size)
        obs_dim = int(eager.cfg.obs_shape["state"][0])
        obs = torch.randn(
            horizon + 1,
            batch_size,
            obs_dim,
            device="cuda",
            generator=generator,
        )
        action = torch.randn(
            horizon,
            batch_size,
            eager.cfg.action_dim,
            device="cuda",
            generator=generator,
        ).tanh()
        reward = torch.randn(
            horizon, batch_size, 1, device="cuda", generator=generator
        )
        terminated = torch.zeros(horizon, batch_size, 1, device="cuda")
        with torch.no_grad():
            eager_targets = eager.model.encode(obs[1:])
            compiled_targets = compiled.model.encode(obs[1:])
        torch.testing.assert_close(eager_targets, compiled_targets, atol=0, rtol=0)
        noise_state = eager._outer_generator.get_state()
        eager._outer_generator.set_state(noise_state)
        compiled._outer_generator.set_state(noise_state)

        eager_losses = eager._recurrent_world_and_value_losses(
            obs, action, reward, terminated, eager_targets
        )
        compiled_losses = compiled._recurrent_world_and_value_losses(
            obs, action, reward, terminated, compiled_targets
        )
        _assert_objective_close(
            eager_losses["critic"], compiled_losses["critic"]
        )
        torch.testing.assert_close(
            eager_losses["total_loss"],
            compiled_losses["total_loss"],
            atol=1e-6,
            rtol=1e-5,
        )

        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        eager_losses["critic"].per_sample_loss[-1].mean().backward()
        compiled_losses["critic"].per_sample_loss[-1].mean().backward()
        _assert_parameter_grads_close(
            eager.model._encoder, compiled.model._encoder
        )
        _assert_parameter_grads_close(
            eager.model._dynamics, compiled.model._dynamics
        )
        _assert_parameter_grads_close(
            eager.xqc_controller.critic,
            compiled.xqc_controller.critic,
        )
        assert sum(
            parameter.grad.abs().sum()
            for parameter in compiled.model._encoder.parameters()
            if parameter.grad is not None
        ) > 0
        assert sum(
            parameter.grad.abs().sum()
            for parameter in compiled.model._dynamics.parameters()
            if parameter.grad is not None
        ) > 0
        assert compiled.xqc_controller.compile_status["critic_compiled"] is True
    finally:
        eager_wrapper.env.close()
        compiled_wrapper.env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_compiled_full_outer_update_keeps_encoder_and_dynamics_gradients():
    wrapper = _tiny_wrapper(device="cuda", compile=True)
    try:
        agent = wrapper.agent
        generator = torch.Generator(device="cuda")
        generator.manual_seed(73)
        horizon = int(agent.cfg.train_unroll_horizon)
        batch_size = int(agent.cfg.batch_size)
        obs_dim = int(agent.cfg.obs_shape["state"][0])
        obs = torch.randn(
            horizon + 1,
            batch_size,
            obs_dim,
            device="cuda",
            generator=generator,
        )
        action = torch.randn(
            horizon,
            batch_size,
            agent.cfg.action_dim,
            device="cuda",
            generator=generator,
        ).tanh()
        reward = torch.randn(
            horizon, batch_size, 1, device="cuda", generator=generator
        )
        terminated = torch.zeros(horizon, batch_size, 1, device="cuda")

        metrics = agent._update(obs, action, reward, terminated)

        def gradient_mass(module):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            assert gradients
            assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
            return sum(gradient.abs().sum() for gradient in gradients)

        assert gradient_mass(agent.model._encoder) > 0
        assert gradient_mass(agent.model._dynamics) > 0
        assert agent.num_updates == agent.outer_version == 1
        assert agent.xqc_workspace.update_step == 1
        assert metrics["compile_fallback"] == 0.0
        assert agent.xqc_controller.compile_status == {
            "requested": True,
            "enabled": True,
            "strict": True,
            "critic_compiled": True,
            "actor_compiled": True,
            "fallback": False,
        }
    finally:
        wrapper.env.close()


def _eight_slot_inner_wrapper(*, compile):
    device = "cuda"
    params = _tiny_params(device=device, compile=compile)
    params.update(
        inner_rounds=2,
        inner_rollouts_per_round=2,
        inner_rollout_horizon=2,
        inner_updates_per_round=4,
        inner_batch_size=2,
        inner_replay_capacity=8,
    )
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    return AMBIXQC(
        "AMBIXQC",
        env,
        params,
        {"seed": 3, "device": device, "env": "test", "total_steps": 10},
        {},
    )


def _assert_eight_slot_local_state_close(eager, compiled):
    assert eager.update_step == compiled.update_step == 8
    assert eager.actor_optimizer_steps == compiled.actor_optimizer_steps == 3
    assert (
        eager.temperature_optimizer_steps
        == compiled.temperature_optimizer_steps
        == 3
    )
    for key, eager_value in eager.controller.state_dict().items():
        if "batch_norm.bias" in key:
            atol, rtol = 1e-3, 1e-3
        elif "batch_norm.running_mean" in key:
            atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 1e-5, 1e-4
        torch.testing.assert_close(
            compiled.controller.state_dict()[key],
            eager_value,
            atol=atol,
            rtol=rtol,
        )
    _assert_nested_close(
        eager.state_dict(), compiled.state_dict(), atol=1e-5, rtol=1e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_complete_compiled_inner_action_matches_eager_and_reuses_graphs(
    monkeypatch,
):
    from RL.tdmpc2_core.common.latent_buffer import LatentReplayBuffer

    eager_wrapper = _eight_slot_inner_wrapper(compile=False)
    compiled_wrapper = _eight_slot_inner_wrapper(compile=True)
    original_sample = LatentReplayBuffer.sample
    draw_log = {}

    def tracked_sample(replay, *args, **kwargs):
        result = original_sample(replay, *args, **kwargs)
        draw_log.setdefault(id(replay), []).append(
            (
                result["indices"].detach().clone(),
                result["sample_ids"].detach().clone(),
            )
        )
        return result

    monkeypatch.setattr(LatentReplayBuffer, "sample", tracked_sample)
    try:
        eager = eager_wrapper.agent
        compiled = compiled_wrapper.agent
        compiled.load_state_dict(deepcopy(eager.state_dict()))
        compiled.inner_engine.load_training_state_dict(
            deepcopy(eager.inner_engine.training_state_dict())
        )
        eager.model.eval()
        compiled.model.eval()
        root_z = torch.linspace(
            -0.75,
            0.75,
            int(eager.cfg.latent_dim),
            device="cuda",
        ).unsqueeze(0)

        def run_and_capture(agent):
            engine = agent.inner_engine
            outer_before = deepcopy(agent.xqc_controller.state_dict())
            cpu_rng_before = torch.random.get_rng_state().clone()
            cuda_rng_before = torch.cuda.get_rng_state().clone()
            result = engine.act(root_z, collect_diagnostics=True)
            torch.testing.assert_close(
                torch.random.get_rng_state(), cpu_rng_before, atol=0, rtol=0
            )
            torch.testing.assert_close(
                torch.cuda.get_rng_state(), cuda_rng_before, atol=0, rtol=0
            )
            _assert_nested_close(
                agent.xqc_controller.state_dict(),
                outer_before,
                atol=0,
                rtol=0,
            )
            outer_critic_buffers = dict(
                agent.xqc_controller.critic.named_buffers()
            )
            for name, target_buffer in (
                engine._workspace_pool.controller.critic_target.named_buffers()
            ):
                torch.testing.assert_close(
                    target_buffer,
                    outer_critic_buffers[name],
                    atol=0,
                    rtol=0,
                )
            return result

        first_eager = run_and_capture(eager)
        first_compiled = run_and_capture(compiled)
        eager_engine = eager.inner_engine
        compiled_engine = compiled.inner_engine
        eager_workspace = eager_engine._workspace_pool
        compiled_workspace = compiled_engine._workspace_pool
        compiled_critic_region = (
            compiled_workspace.controller._critic_loss_region
        )
        compiled_actor_region = compiled_workspace.controller._actor_loss_region
        compiled_critic_callable = compiled_critic_region._compiled
        compiled_actor_callable = compiled_actor_region._compiled
        assert compiled_critic_callable is not None
        assert compiled_actor_callable is not None

        def assert_action_pair(eager_result, compiled_result, *, draw_start):
            eager_action, eager_metrics, eager_lengths = eager_result
            compiled_action, compiled_metrics, compiled_lengths = compiled_result
            assert eager_metrics.keys() == compiled_metrics.keys()
            torch.testing.assert_close(
                compiled_action, eager_action, atol=1e-5, rtol=1e-4
            )
            assert compiled_lengths == eager_lengths == [2, 2, 2, 2]
            exact_metric_keys = {
                "inner_active",
                "inner_algorithm_xqc",
                "inner_diagnostics_sampled",
                "inner_diagnostics_sample_count",
                "inner_diagnostics_step",
                "inner_rounds",
                "inner_iterations",
                "inner_rollouts",
                "inner_requested_rollouts",
                "inner_rollout_count",
                "inner_steps",
                "inner_model_steps",
                "inner_model_steps_budget",
                "inner_nominal_model_steps",
                "inner_realized_model_steps",
                "inner_total_model_steps",
                "inner_updates",
                "inner_update_slots",
                "inner_requested_update_slots",
                "inner_critic_optimizer_steps",
                "inner_actor_optimizer_steps",
                "inner_temperature_optimizer_steps",
                "inner_target_updates",
                "inner_critic_target_updates",
                "inner_actor_target_updates",
                "inner_policy_evaluations",
                "inner_q_evaluations",
                "inner_replay_draws",
                "inner_buffer_size",
                "inner_buffer_capacity",
                "inner_replay_unique_fraction",
                "inner_compile_rollout_fallback",
                "inner_compile_critic_fallback",
                "inner_compile_actor_fallback",
                "inner_compile_fallback",
            }
            for key in eager_metrics:
                if key in exact_metric_keys:
                    _assert_nested_close(
                        eager_metrics[key], compiled_metrics[key], atol=0, rtol=0
                    )
                else:
                    atol, rtol = (
                        (2e-4, 1e-3)
                        if key in {"inner_actor_loss", "inner_q_policy_mean"}
                        else (1e-5, 1e-4)
                    )
                    _assert_nested_close(
                        eager_metrics[key],
                        compiled_metrics[key],
                        atol=atol,
                        rtol=rtol,
                    )
            assert eager_metrics["inner_update_slots"] == 8.0
            assert eager_metrics["inner_actor_optimizer_steps"] == 3.0
            assert eager_metrics["inner_temperature_optimizer_steps"] == 3.0
            assert eager_metrics["inner_replay_draws"] == 16.0
            eager_replay = eager_engine._replay_pool
            compiled_replay = compiled_engine._replay_pool
            eager_replay_state = eager_replay.state_dict()
            compiled_replay_state = compiled_replay.state_dict()
            for key in ("capacity", "pos", "full", "next_sample_id", "sample_id"):
                _assert_nested_close(
                    eager_replay_state[key],
                    compiled_replay_state[key],
                    atol=0,
                    rtol=0,
                )
            for key in ("z", "action", "reward", "next_z", "terminated"):
                torch.testing.assert_close(
                    eager_replay_state[key],
                    compiled_replay_state[key],
                    atol=1e-5,
                    rtol=1e-4,
                )
            eager_draws = draw_log[id(eager_replay)][draw_start : draw_start + 8]
            compiled_draws = draw_log[id(compiled_replay)][
                draw_start : draw_start + 8
            ]
            assert len(eager_draws) == len(compiled_draws) == 8
            for eager_draw, compiled_draw in zip(eager_draws, compiled_draws):
                _assert_nested_close(eager_draw, compiled_draw, atol=0, rtol=0)
            _assert_nested_close(
                eager_engine.training_state_dict()["rng"],
                compiled_engine.training_state_dict()["rng"],
                atol=0,
                rtol=0,
            )
            _assert_eight_slot_local_state_close(
                eager_engine._workspace_pool,
                compiled_engine._workspace_pool,
            )

        assert_action_pair(first_eager, first_compiled, draw_start=0)
        assert eager_engine.action_index == compiled_engine.action_index == 1

        first_outer_state = deepcopy(eager.xqc_controller.state_dict())
        with torch.no_grad():
            eager.xqc_controller.actor.mean.bias.add_(0.1)
            eager.xqc_controller.critic.q1.value.bias.add_(0.05)
            eager.xqc_controller.critic.q2.value.bias.sub_(0.025)
            eager.xqc_controller.actor.input_batch_norm.running_mean.add_(0.03)
            eager.xqc_controller.log_temperature.add_(0.02)
        compiled.xqc_controller.load_state_dict(
            deepcopy(eager.xqc_controller.state_dict())
        )
        assert any(
            not torch.equal(value, first_outer_state[key])
            for key, value in eager.xqc_controller.state_dict().items()
        )

        first_workspace_id = id(compiled_workspace)
        second_eager = run_and_capture(eager)
        second_compiled = run_and_capture(compiled)
        assert_action_pair(second_eager, second_compiled, draw_start=8)
        assert eager_engine.action_index == compiled_engine.action_index == 2
        assert id(compiled_engine._workspace_pool) == first_workspace_id
        assert (
            compiled_engine._workspace_pool.controller._critic_loss_region
            is compiled_critic_region
        )
        assert (
            compiled_engine._workspace_pool.controller._actor_loss_region
            is compiled_actor_region
        )
        assert compiled_critic_region._compiled is compiled_critic_callable
        assert compiled_actor_region._compiled is compiled_actor_callable
        assert compiled_engine._workspace_pool.controller.compile_status == {
            "requested": True,
            "enabled": True,
            "strict": True,
            "critic_compiled": True,
            "actor_compiled": True,
            "fallback": False,
        }
    finally:
        eager_wrapper.env.close()
        compiled_wrapper.env.close()
