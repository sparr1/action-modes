import inspect

import pytest
import torch

from RL.tdmpc2_core.xqc_controller import (
    LatentXQCBatch,
    LatentXQCConfig,
    LatentXQCController,
)
from RL.xqc_core import XQCAgent, XQCConfig


def _assert_module_close(left, right):
    assert left.state_dict().keys() == right.state_dict().keys()
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], atol=1e-7, rtol=1e-6)


def test_controller_refreshes_projection_and_target_caches_after_conversion():
    controller = LatentXQCController(
        3,
        1,
        LatentXQCConfig(
            actor_net_arch=(8,),
            critic_net_arch=(8,),
            num_atoms=11,
            target_entropy=-0.5,
            optimizer_backend="single_tensor",
        ),
    )

    controller.to(dtype=torch.float64)

    live_actor_weights = tuple(
        module.weight
        for module in controller.actor.modules()
        if isinstance(module, torch.nn.Linear)
    )
    live_critic_weights = tuple(
        module.weight
        for module in controller.critic.modules()
        if isinstance(module, torch.nn.Linear)
    )
    assert all(
        cached is live
        for cached, live in zip(controller._actor_linear_weights, live_actor_weights)
    )
    assert all(
        cached is live
        for cached, live in zip(controller._critic_linear_weights, live_critic_weights)
    )
    assert all(
        cached is live
        for cached, live in zip(
            controller._critic_parameters, controller.critic.parameters()
        )
    )
    assert all(
        cached is live
        for cached, live in zip(
            controller._target_parameters, controller.critic_target.parameters()
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_workspace_projects_live_weights_and_moves_the_live_target():
    device = torch.device("cuda")
    controller = LatentXQCController(
        3,
        1,
        LatentXQCConfig(
            actor_net_arch=(8,),
            critic_net_arch=(8,),
            num_atoms=11,
            tau=0.5,
            target_entropy=-0.5,
            optimizer_backend="single_tensor",
        ),
    ).to(device)
    workspace = controller.make_workspace(
        actor_lr=3e-4,
        critic_lr=3e-4,
        transition_steps=10,
        optimizer_backend="single_tensor",
    )
    target_before = tuple(
        parameter.detach().clone() for parameter in controller.critic_target.parameters()
    )
    batch = LatentXQCBatch(
        latents=torch.randn(4, 3, device=device),
        actions=torch.randn(4, 1, device=device).tanh(),
        rewards=torch.randn(4, 1, device=device),
        next_latents=torch.randn(4, 3, device=device),
        bootstrap_mask=torch.ones(4, 1, device=device),
        discount=0.99,
    )

    workspace.update(
        batch,
        next_noise=torch.randn(4, 1, device=device),
        actor_noise=torch.randn(4, 1, device=device),
    )
    torch.cuda.synchronize()

    for module in (controller.actor, controller.critic):
        for child in module.modules():
            if isinstance(child, torch.nn.Linear):
                assert (child.weight.norm(dim=1) - 1.0).abs().max() < 1e-6
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, controller.critic_target.parameters())
    )


def test_latent_workspace_one_step_matches_standalone_xqc_state_machine():
    config = XQCConfig(
        actor_net_arch=(8, 8),
        critic_net_arch=(8, 8),
        num_atoms=11,
        vmin=-2.0,
        vmax=2.0,
        actor_lr=3e-4,
        critic_lr=2e-4,
        lr_end=3e-5,
        num_interactions=10,
        updates_per_step=1,
        gradient_steps=1,
        batch_size=4,
        tau=0.005,
        target_update_interval=1,
        policy_delay=3,
        init_temperature=0.01,
        target_entropy=-0.5,
        reward_normalization=False,
        optimizer_backend="single_tensor",
        device="cpu",
        seed=7,
        compile=False,
    )
    standalone = XQCAgent(3, 1, config)
    latent = LatentXQCController(
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
    )
    latent.actor.load_state_dict(standalone.actor.state_dict())
    latent.critic.load_state_dict(standalone.critic.state_dict())
    latent.critic_target.load_state_dict(standalone.critic_target.state_dict())
    latent.log_temperature.data.copy_(standalone.log_temperature.data)
    workspace = latent.make_workspace(
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        actor_lr_end=config.lr_end,
        critic_lr_end=config.lr_end,
        transition_steps=config.transition_steps,
        optimizer_backend="single_tensor",
    )

    torch.manual_seed(23)
    observations = torch.randn(4, 3)
    actions = torch.randn(4, 1).tanh()
    rewards = torch.randn(4, 1)
    next_observations = torch.randn(4, 3)
    masks = torch.tensor([[1.0], [1.0], [0.0], [1.0]])
    next_noise = torch.randn(4, 1)
    actor_noise = torch.randn(4, 1)
    standalone_metrics = standalone._update_once(
        {
            "obs": observations,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_observations,
            "masks": masks,
            "discount": torch.tensor(0.99),
        },
        next_noise=next_noise,
        actor_noise=actor_noise,
    )
    latent_metrics = workspace.update(
        LatentXQCBatch(
            latents=observations,
            actions=actions,
            rewards=rewards,
            next_latents=next_observations,
            bootstrap_mask=masks,
            discount=torch.tensor(0.99),
        ),
        next_noise=next_noise,
        actor_noise=actor_noise,
        reward_scale=1.0,
    )

    _assert_module_close(latent.actor, standalone.actor)
    _assert_module_close(latent.critic, standalone.critic)
    _assert_module_close(latent.critic_target, standalone.critic_target)
    torch.testing.assert_close(
        latent.log_temperature, standalone.log_temperature, atol=1e-7, rtol=1e-6
    )
    assert workspace.update_step == standalone.update_step == 1
    assert workspace.actor_optimizer_steps == standalone.actor_optimizer_steps == 1
    assert (
        workspace.temperature_optimizer_steps
        == standalone.temperature_optimizer_steps
        == 1
    )
    metric_pairs = {
        "critic_loss": "critic_loss",
        "actor_loss": "actor_loss",
        "policy_entropy": "policy_entropy",
        "q1_mean": "q1_mean",
        "q2_mean": "q2_mean",
        "q_target_mean": "q_target_mean",
        "q_policy_mean": "q_policy_mean",
        "q_disagreement_mean": "q_disagreement_mean",
        "q_target_clip_fraction": "q_target_clip_fraction",
    }
    for left_key, right_key in metric_pairs.items():
        assert float(latent_metrics[left_key]) == pytest.approx(
            standalone_metrics[right_key], abs=1e-6, rel=1e-5
        )


def test_latent_critic_restores_arbitrary_leading_dimensions_without_detaching():
    controller = LatentXQCController(
        4,
        2,
        LatentXQCConfig(
            actor_net_arch=(8,),
            critic_net_arch=(8,),
            num_atoms=11,
            vmin=-2,
            vmax=2,
            target_entropy=-1,
            optimizer_backend="single_tensor",
        ),
    )
    latent = torch.randn(3, 5, 4, requires_grad=True)
    objective = controller.critic_objective(
        LatentXQCBatch(
            latents=latent,
            actions=torch.randn(3, 5, 2).tanh(),
            rewards=torch.randn(3, 5, 1),
            next_latents=torch.randn(3, 5, 4),
            bootstrap_mask=torch.ones(3, 5, 1),
            discount=0.99,
        ),
        next_noise=torch.randn(3, 5, 2),
        reward_scale=torch.tensor(2.0),
    )

    assert objective.per_sample_loss.shape == (3, 5)
    assert objective.current_values.shape == (2, 3, 5)
    objective.per_sample_loss[-1].mean().backward()
    assert latent.grad is not None
    assert latent.grad[-1].abs().sum() > 0


@pytest.mark.parametrize(
    "reward_scale",
    (
        torch.tensor(0.0),
        torch.tensor(-1.0),
        torch.tensor(torch.inf),
        torch.tensor(torch.nan),
        torch.ones(2),
    ),
)
def test_tensor_reward_scale_cpu_validation_preserves_value_errors(reward_scale):
    controller = LatentXQCController(
        3,
        1,
        LatentXQCConfig(
            actor_net_arch=(8,),
            critic_net_arch=(8,),
            num_atoms=11,
            target_entropy=-0.5,
            optimizer_backend="single_tensor",
        ),
    )
    batch = LatentXQCBatch(
        latents=torch.randn(2, 3),
        actions=torch.randn(2, 1).tanh(),
        rewards=torch.randn(2, 1),
        next_latents=torch.randn(2, 3),
        bootstrap_mask=torch.ones(2, 1),
        discount=0.99,
    )

    with pytest.raises(
        ValueError, match="reward_scale must be one positive finite scalar"
    ):
        controller.critic_objective(
            batch,
            next_noise=torch.randn(2, 1),
            reward_scale=reward_scale,
        )


def test_cuda_tensor_reward_scale_validation_uses_async_assertion():
    source = inspect.getsource(LatentXQCController.critic_objective)

    assert 'if scale.device.type == "cuda":' in source
    assert "torch._assert_async(" in source
    assert "bool(torch.isfinite(scale)" not in source
    assert "bool(scale > 0)" not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_tensor_reward_scale_does_not_materialize_a_host_scalar(monkeypatch):
    device = torch.device("cuda")
    controller = LatentXQCController(
        3,
        1,
        LatentXQCConfig(
            actor_net_arch=(8,),
            critic_net_arch=(8,),
            num_atoms=11,
            target_entropy=-0.5,
            optimizer_backend="single_tensor",
        ),
    ).to(device)
    batch = LatentXQCBatch(
        latents=torch.randn(2, 3, device=device),
        actions=torch.randn(2, 1, device=device).tanh(),
        rewards=torch.randn(2, 1, device=device),
        next_latents=torch.randn(2, 3, device=device),
        bootstrap_mask=torch.ones(2, 1, device=device),
        discount=0.99,
    )
    original_item = torch.Tensor.item
    original_bool = torch.Tensor.__bool__

    def guarded_item(tensor, *args, **kwargs):
        if tensor.device.type == "cuda":
            raise AssertionError("CUDA reward-scale validation called Tensor.item()")
        return original_item(tensor, *args, **kwargs)

    def guarded_bool(tensor):
        if tensor.device.type == "cuda":
            raise AssertionError("CUDA reward-scale validation converted a tensor to bool")
        return original_bool(tensor)

    monkeypatch.setattr(torch.Tensor, "item", guarded_item)
    monkeypatch.setattr(torch.Tensor, "__bool__", guarded_bool)

    objective = controller.critic_objective(
        batch,
        next_noise=torch.randn(2, 1, device=device),
        reward_scale=torch.tensor(2.0, device=device),
    )
    torch.cuda.synchronize()

    assert objective.loss.device == device
