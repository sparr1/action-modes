"""Independent critic architecture, target lifecycle and compiled gradients."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_params


def model(mode, **kwargs):
    return SoftWorldModel(_build_cfg(**_tiny_params(aux_return_mode=mode, **kwargs)))


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_auxiliary_networks_are_independent_single_values(mode, representation):
    world = model(mode, q_representation=representation)
    params = {id(p) for p in world._Qs.parameters()}
    auxiliary = {id(p) for p in world._aux_return_Qs.parameters()}
    target = {id(p) for p in world._target_aux_return_Qs.parameters()}
    assert not params.intersection(auxiliary | target)
    assert not auxiliary.intersection(target)
    assert all(not p.requires_grad for p in world._target_aux_return_Qs.parameters())
    predictions = world.aux_return_q_predictions(torch.randn(4, world.cfg.latent_dim), torch.zeros(4, world.cfg.action_dim))
    assert predictions.shape == (world.cfg.num_q, 4, world.q_backend.output_dim)
    assert hasattr(world, "_return_pi") == (mode == "return_actor")
    if mode == "return_actor":
        actor = {id(p) for p in world._pi.parameters()}
        assert not actor.intersection(id(p) for p in world._return_pi.parameters())


def test_auxiliary_construction_preserves_original_parameters_and_rng():
    torch.manual_seed(37)
    original = model("off")
    rng = torch.random.get_rng_state().clone()
    torch.manual_seed(37)
    augmented = model("return_actor")
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, augmented.state_dict()[key], rtol=0, atol=0)
    assert not any("aux_return" in key or "return_pi" in key for key in original.state_dict())


def test_both_target_ensembles_update_and_stay_frozen_in_train_mode():
    world = model("sac")
    before = deepcopy(world._target_aux_return_Qs.state_dict())
    with torch.no_grad():
        for param in world._aux_return_Qs.parameters():
            param.add_(2.)
    world.soft_update_target_Q(tau=.25)
    for key, value in world._target_aux_return_Qs.state_dict().items():
        torch.testing.assert_close(value, .75 * before[key] + .25 * world._aux_return_Qs.state_dict()[key])
    world.train()
    assert world._aux_return_Qs.training
    assert not world._target_aux_return_Qs.training
    assert not world._target_Qs.training


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_auxiliary_fullgraph_eager_compile_values_and_gradients(representation, backend):
    world = model("sac", q_representation=representation)
    with torch.no_grad():
        for member in world._aux_return_Qs:
            member[-1].weight.normal_(std=.1)
    z = torch.randn(4, world.cfg.latent_dim, requires_grad=True)
    action = torch.randn(4, world.cfg.action_dim, requires_grad=True)
    targets = torch.randn(4, 1)
    def loss(z, action, targets):
        return world.q_backend.loss(world.aux_return_q_predictions(z, action), targets)
    eager = loss(z, action, targets)
    eager_grads = torch.autograd.grad(eager, (z, action) + tuple(world._aux_return_Qs.parameters()))
    compiled = torch.compile(loss, backend=backend, fullgraph=True)(z, action, targets)
    compiled_grads = torch.autograd.grad(compiled, (z, action) + tuple(world._aux_return_Qs.parameters()))
    torch.testing.assert_close(eager, compiled)
    for actual, expected in zip(compiled_grads, eager_grads):
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable on this host")
def test_auxiliary_cpu_initialization_preserves_all_cuda_generator_states():
    before = torch.cuda.get_rng_state_all()
    model("return_actor")
    for left, right in zip(before, torch.cuda.get_rng_state_all()):
        assert torch.equal(left, right)
