"""Preserve official TD-MPC2 critic initialization in the module-list port."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from RL.tdmpc2_core.common import init, layers
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from RL.tdmpc2_core.common.world_model import WorldModel


MODEL_CASES = [
    pytest.param(WorldModel, "distributional", 2, id="tdmpc2-two-heads"),
    pytest.param(WorldModel, "distributional", 5, id="tdmpc2-five-heads"),
    pytest.param(SoftWorldModel, "scalar", 2, id="ambi-scalar-twins"),
    pytest.param(SoftWorldModel, "distributional", 5, id="ambi-five-heads"),
    pytest.param(SoftWorldModel, "distributional", 10, id="ambi-ten-heads"),
]


def model_cfg(representation, num_q):
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
        num_bins=7,
        vmin=-5,
        vmax=5,
        episodic=True,
        dropout=0.01,
        log_std_min=-10,
        log_std_max=2,
        tau=0.005,
        q_representation=representation,
        num_q=num_q,
        q_pair_size=2,
        q_num_bins=7,
        q_vmin=-5,
        q_vmax=5,
    )


@pytest.mark.parametrize("model_type,representation,num_q", MODEL_CASES)
def test_critics_retain_pytorch_constructor_initialization(
    monkeypatch, model_type, representation, num_q
):
    constructor_samples = {}
    original_reset = nn.Linear.reset_parameters

    def record_constructor_samples(module):
        assert module not in constructor_samples
        original_reset(module)
        constructor_samples[module] = (
            module.weight.detach().clone(),
            module.bias.detach().clone(),
        )

    monkeypatch.setattr(nn.Linear, "reset_parameters", record_constructor_samples)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        model = model_type(model_cfg(representation, num_q))

    assert len(model._Qs) == num_q
    for critic in model._Qs:
        for index, linear in enumerate(critic):
            initial_weight, initial_bias = constructor_samples[linear]
            expected_weight = (
                torch.zeros_like(initial_weight)
                if index == len(critic) - 1
                else initial_weight
            )
            torch.testing.assert_close(linear.weight, expected_weight, rtol=0, atol=0)
            torch.testing.assert_close(linear.bias, initial_bias, rtol=0, atol=0)
        for module in critic.modules():
            if isinstance(module, nn.LayerNorm):
                torch.testing.assert_close(
                    module.weight, torch.ones_like(module.weight), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    module.bias, torch.zeros_like(module.bias), rtol=0, atol=0
                )

    online_parameters = dict(model._Qs.named_parameters())
    target_parameters = dict(model._target_Qs.named_parameters())
    assert online_parameters.keys() == target_parameters.keys()
    for name, online in online_parameters.items():
        target = target_parameters[name]
        torch.testing.assert_close(target, online, rtol=0, atol=0)
        assert target.data_ptr() != online.data_ptr()
        assert online.requires_grad
        assert not target.requires_grad
    model.train()
    assert all(not module.training for module in model._target_Qs.modules())


@pytest.mark.parametrize("model_type,representation,num_q", MODEL_CASES)
def test_noncritic_initialization_matches_historical_recursive_initialization(
    monkeypatch, model_type, representation, num_q
):
    construction_rng = []
    original_ensemble_init = layers.Ensemble.__init__

    def record_rng_after_construction(ensemble, *args, **kwargs):
        original_ensemble_init(ensemble, *args, **kwargs)
        construction_rng.append(torch.random.get_rng_state())

    monkeypatch.setattr(layers.Ensemble, "__init__", record_rng_after_construction)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(81)
        model = model_type(model_cfg(representation, num_q))
        assert len(construction_rng) == 1
        historical = deepcopy(model)
        # Before the fix both constructors recursively initialized every child.
        # Replay that operation from the exact pre-initialization RNG state.
        torch.random.set_rng_state(construction_rng[0])
        historical.apply(init.weight_init)
        init.zero_([historical._reward[-1].weight])

    expected_parameters = dict(historical.named_parameters())
    for name, parameter in model.named_parameters():
        if name.startswith(("_Qs.", "_target_Qs.")):
            continue
        torch.testing.assert_close(
            parameter, expected_parameters[name], rtol=0, atol=0, msg=name
        )
