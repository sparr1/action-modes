from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from RL.tdmpc2_core.common.lora import LoRALinear, lorafy_copy
from RL.tdmpc2_core.common.q_representation import QRepresentation
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel


def model_cfg(**overrides):
    values = {
        "multitask": False,
        "obs_shape": {"state": (3,)},
        "obs": "state",
        "task_dim": 0,
        "num_enc_layers": 2,
        "enc_dim": 16,
        "latent_dim": 8,
        "simnorm_dim": 4,
        "action_dim": 2,
        "mlp_dim": 16,
        "num_bins": 7,
        "vmin": -5,
        "vmax": 5,
        "episodic": False,
        "dropout": 0.0,
        "log_std_min": -20,
        "log_std_max": 2,
        "tau": 0.005,
        "q_representation": "scalar",
        "num_q": 2,
        "q_pair_size": 2,
        "q_num_bins": 5,
        "q_vmin": -2,
        "q_vmax": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_scalar_backend_preserves_twin_q_values_keys_and_mse():
    model = SoftWorldModel(model_cfg())
    assert model.critic_signature == {
        "q_representation": "scalar",
        "num_q": 2,
        "q_num_bins": 1,
        "q_vmin": None,
        "q_vmax": None,
    }

    with torch.no_grad():
        for index, critic in enumerate(model._Qs):
            critic[-1].weight.zero_()
            critic[-1].bias.fill_(1.0 + 2.0 * index)

    z = torch.zeros(4, model.cfg.latent_dim)
    action = torch.zeros(4, model.cfg.action_dim)
    raw = model.q_predictions(z, action)
    values = model.q_values(z, action)
    torch.testing.assert_close(values, raw)
    assert raw.shape == (2, 4, 1)
    torch.testing.assert_close(model.Q(z, action), torch.ones(4, 1))
    torch.testing.assert_close(
        model.Q(z, action, return_type="avg"),
        torch.full((4, 1), 2.0),
    )
    torch.testing.assert_close(model.Q(z, action, return_type="all"), raw)
    assert model.critic_loss(raw, torch.zeros(4, 1)).item() == pytest.approx(5.0)

    state_keys = set(model.state_dict())
    assert any(key.startswith("_Qs.modules_list.0.") for key in state_keys)
    assert any(key.startswith("_target_Qs.modules_list.0.") for key in state_keys)
    assert all("q_backend" not in key for key in state_keys)


def test_distributional_backend_separates_q_bins_and_decodes_every_head():
    model = SoftWorldModel(
        model_cfg(
            q_representation="distributional",
            num_q=4,
            q_pair_size=2,
            num_bins=7,
            q_num_bins=5,
            q_vmin=-2,
            q_vmax=2,
        )
    )
    z = torch.zeros(3, model.cfg.latent_dim)
    action = torch.zeros(3, model.cfg.action_dim)
    raw = model.q_predictions(z, action)
    values = model.q_values(z, action)

    assert model._reward[-1].out_features == 7
    assert raw.shape == (4, 3, 5)
    assert values.shape == (4, 3, 1)
    torch.testing.assert_close(values, torch.zeros_like(values), atol=1e-7, rtol=0)
    assert model.critic_signature == {
        "q_representation": "distributional",
        "num_q": 4,
        "q_num_bins": 5,
        "q_vmin": -2.0,
        "q_vmax": 2.0,
    }


def test_policy_sampling_accepts_isolated_generator_std_scale_and_inner_bounds():
    model = SoftWorldModel(model_cfg())
    with torch.no_grad():
        for parameter in model._pi.parameters():
            parameter.zero_()
    z = torch.zeros(3, model.cfg.latent_dim)

    global_before = torch.random.get_rng_state().clone()
    generator = torch.Generator().manual_seed(19)
    first_action, first_info = model.pi(
        z,
        generator=generator,
        std_scale=0.5,
        log_std_min=-1.0,
        log_std_max=1.0,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), global_before, rtol=0, atol=0)

    generator.manual_seed(19)
    second_action, second_info = model.pi(
        z,
        generator=generator,
        std_scale=0.5,
        log_std_min=-1.0,
        log_std_max=1.0,
    )
    torch.testing.assert_close(first_action, second_action, rtol=0, atol=0)
    torch.testing.assert_close(first_info["pre_tanh_mean"], torch.zeros_like(z[:, :2]))
    torch.testing.assert_close(first_info["log_prob"], second_info["log_prob"], rtol=0, atol=0)
    torch.testing.assert_close(
        first_info["log_std"],
        torch.full_like(first_info["log_std"], float(torch.log(torch.tensor(0.5)))),
    )

    with pytest.raises(ValueError, match="std_scale must be positive"):
        model.pi(z, std_scale=0.0)
    with pytest.raises(ValueError, match="smaller than"):
        model.pi(z, log_std_min=1.0, log_std_max=1.0)


def test_distributional_soft_target_loss_trains_all_ensemble_heads():
    backend = QRepresentation(
        "distributional",
        num_q=4,
        pair_size=2,
        num_bins=5,
        vmin=-2,
        vmax=2,
    )
    predictions = torch.zeros(4, 3, 5, requires_grad=True)
    target = torch.tensor([[-3.0], [0.0], [3.0]])

    encoded = backend.encode_target(target)
    torch.testing.assert_close(encoded.sum(dim=-1), torch.ones(3))
    assert backend.loss(predictions, target).item() == pytest.approx(torch.log(torch.tensor(5.0)).item())

    backend.loss(predictions, target).backward()
    assert predictions.grad is not None
    assert torch.count_nonzero(predictions.grad.reshape(4, -1).abs().sum(dim=1)) == 4


def test_q_reductions_make_pair_sampling_explicit_and_all_reductions_deterministic():
    backend = QRepresentation(
        "distributional",
        num_q=4,
        pair_size=2,
        num_bins=5,
        vmin=-2,
        vmax=2,
    )
    values = torch.tensor([1.0, 4.0, -2.0, 3.0]).reshape(4, 1, 1)

    torch.testing.assert_close(
        backend.reduce(values, "min_pair", pair_indices=[1, 3]),
        torch.tensor([[3.0]]),
    )
    torch.testing.assert_close(
        backend.reduce(values, "mean_pair", pair_indices=[1, 3]),
        torch.tensor([[3.5]]),
    )
    torch.testing.assert_close(backend.reduce(values, "min_all"), torch.tensor([[-2.0]]))
    torch.testing.assert_close(backend.reduce(values, "mean_all"), torch.tensor([[1.5]]))
    with pytest.raises(ValueError, match="unique"):
        backend.reduce(values, "min_pair", pair_indices=[1, 1])


@pytest.mark.parametrize(
    ("representation", "num_q"),
    [("scalar", 2), ("distributional", 4)],
)
def test_decoded_q_values_preserve_actor_gradients_while_critic_is_detached(
    representation, num_q
):
    model = SoftWorldModel(
        model_cfg(q_representation=representation, num_q=num_q)
    )
    with torch.no_grad():
        for critic in model._Qs:
            critic[-1].weight.normal_(std=0.1)

    z = torch.randn(3, model.cfg.latent_dim)
    action = torch.randn(3, model.cfg.action_dim, requires_grad=True)
    q_value = model.Q(z, action, detach=True, reduction="mean_all")
    q_value.sum().backward()

    assert action.grad is not None
    assert torch.count_nonzero(action.grad) > 0
    assert all(parameter.grad is None for parameter in model._Qs.parameters())
    assert all(parameter.requires_grad for parameter in model._Qs.parameters())
    assert all(not parameter.requires_grad for parameter in model._target_Qs.parameters())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"q_representation": "scalar", "num_q": 3}, "exactly num_q=2"),
        ({"q_representation": "distributional", "num_q": 1}, "num_q>=2"),
        (
            {"q_representation": "distributional", "num_q": 2, "q_num_bins": 1},
            "at least 2",
        ),
    ],
)
def test_invalid_q_architectures_fail_before_network_construction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SoftWorldModel(model_cfg(**kwargs))


@pytest.mark.parametrize("rank", [2, 4, 8])
def test_lora_scale_is_the_direct_multiplier_independent_of_requested_rank(rank):
    # The one-unit output also verifies that effective-rank clipping does not
    # accidentally change the configured multiplier.
    adapted = lorafy_copy(nn.Sequential(nn.Linear(8, 1)), rank=rank, scale=0.75)
    layer = next(module for module in adapted.modules() if isinstance(module, LoRALinear))
    assert layer.rank == 1
    assert layer.scaling == pytest.approx(0.75)
