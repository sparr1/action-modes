"""Allocated-GPU parity gate for joint soft and return-only critic updates."""

from argparse import Namespace
from copy import deepcopy

import pytest
import torch

from tests.benchmark_aux_return_joint_vmap import make_agent


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires an allocated CUDA GPU.",
)


def _close_tree(actual, expected):
    if torch.is_tensor(expected):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-6)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _close_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _close_tree(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=5e-4, abs=5e-6)
    else:
        assert actual == expected


def _agent(joint):
    args = Namespace(
        device="cuda", compile=False, aux_mode="sac", operator="mppi",
        representation="distributional", heads=5, encoder_width=64, width=64,
        latent_dim=32, bins=101, batch_size=8, horizon=3, dropout=0.,
        obs_dim=67, action_dim=21, mppi_horizon=3, mppi_iterations=2,
        mppi_elites=4, mppi_pi_trajs=0, mppi_samples=16,
    )
    agent = make_agent(args, joint)
    # Set unequal weights/rates before saving any initial state or updating.
    for cfg in (agent.cfg, agent.aux_return.cfg):
        cfg.critic_coef = .37
        cfg.aux_return_critic_coef = .23
        cfg.critic_lr = 3e-4
        cfg.aux_return_critic_lr = 7e-4
        cfg.grad_clip_norm = .01
    primary = {id(parameter) for parameter in agent.model._Qs.parameters()}
    auxiliary = {id(parameter) for parameter in agent.model._aux_return_Qs.parameters()}
    for group in agent.optim.param_groups:
        if any(id(parameter) in primary for parameter in group["params"]):
            group["lr"] = agent.cfg.critic_lr
        elif any(id(parameter) in auxiliary for parameter in group["params"]):
            group["lr"] = agent.cfg.aux_return_critic_lr
    return agent


def _capture_clipped_gradients(monkeypatch, agent):
    snapshots = []
    step = agent.optim.step
    world_parameters = {id(parameter) for parameter in agent._world_critic_params}

    def capture(*args, **kwargs):
        snapshots.append({
            name: parameter.grad.detach().clone()
            for name, parameter in agent.model.named_parameters()
            if id(parameter) in world_parameters and parameter.grad is not None
        })
        return step(*args, **kwargs)

    monkeypatch.setattr(agent.optim, "step", capture)
    return snapshots


def test_cuda_two_updates_preserve_losses_gradients_adam_and_target_ema(monkeypatch):
    torch.manual_seed(1247)
    source = _agent(False)
    with torch.no_grad():
        for name in ("_Qs", "_aux_return_Qs", "_target_Qs", "_target_aux_return_Qs"):
            for head in getattr(source.model, name):
                head[-1].weight.normal_(std=.13)
                head[-1].bias.normal_(std=.07)
    checkpoint = deepcopy(source.checkpoint_state())
    separate, joint = _agent(False), _agent(True)
    separate.load(deepcopy(checkpoint))
    joint.load(deepcopy(checkpoint))
    del source
    separate_gradients = _capture_clipped_gradients(monkeypatch, separate)
    joint_gradients = _capture_clipped_gradients(monkeypatch, joint)
    generator = torch.Generator().manual_seed(7654)
    batch = (
        torch.randn(4, 8, 67, generator=generator, device="cpu").cuda(),
        torch.randn(3, 8, 21, generator=generator, device="cpu").tanh().cuda(),
        torch.randn(3, 8, 1, generator=generator, device="cpu").cuda(),
        torch.zeros(3, 8, 1, device="cuda"),
    )
    latent = torch.randn(3, 8, 32, generator=generator).cuda()
    targets_before = {
        name: deepcopy(getattr(joint.model, name).state_dict())
        for name in ("_target_Qs", "_target_aux_return_Qs")
    }

    for completed in (1, 2):
        with torch.no_grad():
            for target in (False, True):
                expected = (
                    separate.model.q_predictions(latent, batch[1], target=target),
                    separate.model.aux_return_q_predictions(latent, batch[1], target=target),
                )
                actual = joint.model.joint_q_predictions(
                    latent, batch[1], latent, batch[1], target=target,
                )
                _close_tree(actual, expected)
                assert all(prediction.std() > 0 for prediction in actual)

        data = [value.detach().clone() for value in batch]
        joint_data = [value.detach().clone() for value in batch]
        data[1].requires_grad_()
        joint_data[1].requires_grad_()
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all()
        expected = separate._update(*data)
        expected_cpu_rng = torch.random.get_rng_state().clone()
        expected_cuda_rng = torch.cuda.get_rng_state_all()
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        actual = joint._update(*joint_data)

        assert actual["aux_return_joint_vmap_online"] == actual["aux_return_joint_vmap_target"] == 1.
        assert expected["aux_return_joint_vmap_online"] == expected["aux_return_joint_vmap_target"] == 0.
        assert not actual["compile_fallback"] and not expected["compile_fallback"]
        for key in (
            "consistency_loss", "reward_loss", "critic_loss", "aux_return_critic_loss",
            "total_loss", "grad_norm", "q_mean", "q_target_mean", "aux_return_q_mean",
            "aux_return_q_target_mean",
        ):
            _close_tree(actual[key], expected[key])
        assert actual["grad_norm"] > joint.cfg.grad_clip_norm
        _close_tree(joint_data[1].grad, data[1].grad)
        assert joint_data[1].grad.abs().sum() > 0
        _close_tree(joint_gradients[-1], separate_gradients[-1])
        gradient_norm = torch.stack([
            gradient.square().sum() for gradient in joint_gradients[-1].values()
        ]).sum().sqrt()
        torch.testing.assert_close(
            gradient_norm, gradient_norm.new_tensor(joint.cfg.grad_clip_norm),
            rtol=1e-4, atol=1e-7,
        )
        for prefix in ("_encoder.", "_dynamics.", "_Qs.", "_aux_return_Qs."):
            assert any(
                name.startswith(prefix) and gradient.abs().sum() > 0
                for name, gradient in joint_gradients[-1].items()
            )
        _close_tree(joint.model.state_dict(), separate.model.state_dict())
        for name in ("optim", "pi_optim"):
            _close_tree(getattr(joint, name).state_dict(), getattr(separate, name).state_dict())
        for name in ("_Qs", "_aux_return_Qs"):
            for parameter in getattr(joint.model, name).parameters():
                state = joint.optim.state[parameter]
                assert state["step"] == completed
                assert state["exp_avg"].abs().sum() > 0
                assert state["exp_avg_sq"].sum() > 0
        for name, before in targets_before.items():
            assert any(
                not torch.equal(value, before[key])
                for key, value in getattr(joint.model, name).state_dict().items()
            )
        assert joint.num_updates == separate.num_updates == completed
        assert joint.aux_return.num_updates == separate.aux_return.num_updates == completed
        assert torch.equal(torch.random.get_rng_state(), expected_cpu_rng)
        assert all(torch.equal(left, right) for left, right in zip(
            torch.cuda.get_rng_state_all(), expected_cuda_rng,
        ))
        for name in ("_cpu_generator", "_device_generator"):
            assert torch.equal(
                getattr(joint.aux_return, name).get_state(),
                getattr(separate.aux_return, name).get_state(),
            )
