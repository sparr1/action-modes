"""Two complete learner updates agree with the pre-fusion PAVE calculation."""

from copy import deepcopy
from argparse import Namespace

import pytest
import torch

from RL.tdmpc2_core import pave
from tests.benchmark_aux_return_joint_vmap import make_agent
from tests.pave_reference import separate_critic_pave_loss
from tests.test_pave_view_batching import DEVICES, close_tree


@pytest.fixture
def agents():
    def create(**options):
        args = Namespace(
            device=options.pop("device"), compile=options.pop("compile"), aux_mode="sac",
            operator="sac", representation=options.pop("q_representation"),
            heads=options.pop("num_q"), encoder_width=16, width=16, latent_dim=8,
            bins=5, batch_size=2, horizon=2, dropout=options.pop("dropout"),
            obs_dim=3, action_dim=1,
        )
        agent = make_agent(args, joint=True)
        for key, value in dict(critic_coef=.37, aux_return_critic_coef=.23,
                              grad_clip_norm=.01, **options).items():
            setattr(agent.cfg, key, value)
        primary = {id(parameter) for parameter in agent.model._Qs.parameters()}
        auxiliary = {id(parameter) for parameter in agent.model._aux_return_Qs.parameters()}
        for group in agent.optim.param_groups:
            if any(id(parameter) in primary for parameter in group["params"]):
                group["lr"] = 3e-4
            elif any(id(parameter) in auxiliary for parameter in group["params"]):
                group["lr"] = 7e-4
        return agent
    return create


def _weights(*prefixes):
    return {f"{prefix}_{term}_coef": value for prefix in prefixes
            for term, value in (("mpr", .1), ("vfc", .1), ("curvature", .01))}


def _batch(agent):
    h, b = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    generator = torch.Generator().manual_seed(991)
    return (torch.randn(h + 1, b, 3, generator=generator),
            torch.randn(h, b, 1, generator=generator).tanh(),
            torch.randn(h, b, 1, generator=generator), torch.zeros(h, b, 1))


def _nonconstant_all_critics(agent):
    generator = torch.Generator().manual_seed(746)
    with torch.no_grad():
        for name in ("_Qs", "_target_Qs", "_aux_return_Qs", "_target_aux_return_Qs"):
            for head in getattr(agent.model, name):
                head[-1].weight.copy_(torch.randn(head[-1].weight.shape, generator=generator) * .13)
                head[-1].bias.copy_(torch.randn(head[-1].bias.shape, generator=generator) * .07)


def _record_world_gradients(monkeypatch, agent):
    snapshots = []
    step = agent.optim.step

    def capture(*args, **kwargs):
        snapshots.append({name: parameter.grad.detach().clone()
                          for name, parameter in agent.model.named_parameters()
                          if parameter.grad is not None})
        return step(*args, **kwargs)

    monkeypatch.setattr(agent.optim, "step", capture)
    return snapshots


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("prefixes", [
    ("outer_pave",), ("aux_return_pave",), ("outer_pave", "aux_return_pave"),
])
def test_two_updates_preserve_clipped_gradients_adam_targets_and_rng(
    agents, monkeypatch, device, compiled, representation, prefixes,
):
    actual_compile = torch.compile
    if compiled and device == "cpu":
        monkeypatch.setattr(torch, "compile", lambda function, **kwargs:
                            actual_compile(function, backend="aot_eager", **kwargs))
    if compiled:
        torch._dynamo.reset()
    options = dict(
        q_representation=representation, num_q=5 if representation == "distributional" else 2,
        compile=compiled, compile_strict=compiled, dropout=.2, device=device,
        outer_pave_batch_size=3, **_weights(*prefixes),
    )
    separate, fused = agents(**options), agents(**options)
    _nonconstant_all_critics(separate)
    fused.load(deepcopy(separate.checkpoint_state()))
    parameter_ids = tuple(map(id, fused.model.parameters()))
    state_keys = tuple(fused.model.state_dict())
    expected_gradients = _record_world_gradients(monkeypatch, separate)
    actual_gradients = _record_world_gradients(monkeypatch, fused)
    before_targets = {name: deepcopy(getattr(fused.model, name).state_dict())
                      for name in ("_target_Qs", "_target_aux_return_Qs")}
    try:
        for completed in (1, 2):
            data = [value.to(device) for value in _batch(separate)]
            data[1].requires_grad_()
            fused_data = [value.detach().clone().requires_grad_(value.requires_grad) for value in data]
            rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else []
            with monkeypatch.context() as oracle:
                oracle.setattr(pave, "critic_pave_loss", separate_critic_pave_loss)
                expected = separate._update(*data)
            expected_rng = torch.get_rng_state().clone()
            expected_cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else []
            torch.set_rng_state(rng)
            if device == "cuda":
                torch.cuda.set_rng_state_all(cuda_rng)
            actual = fused._update(*fused_data)
            for key in ("total_loss", "critic_loss", "aux_return_critic_loss", "grad_norm",
                        *(name for name in expected if "pave" in name)):
                close_tree(actual[key], expected[key], rtol=5e-5, atol=5e-7)
            assert actual["grad_norm"] > fused.cfg.grad_clip_norm
            close_tree(actual_gradients[-1], expected_gradients[-1], rtol=5e-5, atol=1e-7)
            close_tree(fused_data[1].grad, data[1].grad, rtol=5e-5, atol=1e-7)
            close_tree(fused.model.state_dict(), separate.model.state_dict(), rtol=5e-5, atol=5e-7)
            for name in ("optim", "pi_optim"):
                close_tree(getattr(fused, name).state_dict(), getattr(separate, name).state_dict(),
                           rtol=5e-5, atol=5e-7)
            assert fused.num_updates == separate.num_updates == completed
            assert fused.aux_return.num_updates == separate.aux_return.num_updates == completed
            assert torch.equal(torch.get_rng_state(), expected_rng)
            if device == "cuda":
                assert all(torch.equal(left, right) for left, right in zip(
                    torch.cuda.get_rng_state_all(), expected_cuda_rng,
                ))
            for name in ("_cpu_generator", "_device_generator"):
                left, right = getattr(fused.aux_return, name), getattr(separate.aux_return, name)
                if left is None:
                    assert right is None
                else:
                    assert torch.equal(left.get_state(), right.get_state())
            assert not actual["compile_fallback"] and not expected["compile_fallback"]
            if compiled:
                assert fused._outer_update_region._compiled is not None
                assert separate._outer_update_region._compiled is not None
        assert tuple(map(id, fused.model.parameters())) == parameter_ids
        assert tuple(fused.model.state_dict()) == state_keys
        for name, before in before_targets.items():
            assert any(not torch.equal(value, before[key])
                       for key, value in getattr(fused.model, name).state_dict().items())
    finally:
        if compiled:
            torch._dynamo.reset()
