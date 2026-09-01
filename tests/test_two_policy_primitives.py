from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.common.latent_buffer import LatentReplayBuffer
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel


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
        num_bins=7,
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
        q_num_bins=5,
        q_vmin=-2,
        q_vmax=2,
    )


def _transition_rows(count):
    z = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2)
    return (
        z,
        torch.arange(count, dtype=torch.float32).reshape(count, 1),
        torch.ones(count, 1),
        z + 1,
        torch.zeros(count, 1),
    )


def test_policy_stats_is_rng_free_and_matches_zero_noise_policy_parameters():
    torch.manual_seed(11)
    model = SoftWorldModel(_model_cfg())
    latent = torch.randn(4, model.cfg.latent_dim)
    global_state = torch.random.get_rng_state().clone()

    stats = model.policy_stats(latent)

    torch.testing.assert_close(torch.random.get_rng_state(), global_state, rtol=0, atol=0)
    _, sampled = model.pi(latent, noise=torch.zeros(4, model.cfg.action_dim))
    assert set(stats) == {"mean", "pre_tanh_mean", "log_std"}
    torch.testing.assert_close(stats["mean"], sampled["mean"], rtol=0, atol=0)
    torch.testing.assert_close(
        stats["pre_tanh_mean"], sampled["pre_tanh_mean"], rtol=0, atol=0
    )
    torch.testing.assert_close(stats["log_std"], sampled["log_std"], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), global_state, rtol=0, atol=0)


def test_arbitrary_squashed_component_density_matches_transformed_normal():
    mean = torch.tensor([[0.3, -0.7], [1.1, 0.2]], dtype=torch.float64)
    log_std = torch.tensor([[-0.4, 0.2], [0.1, -0.8]], dtype=torch.float64)
    pre_tanh_action = torch.tensor(
        [[-0.2, 0.9], [1.8, -1.3]], dtype=torch.float64
    )
    expected = torch.distributions.Normal(mean, log_std.exp()).log_prob(
        pre_tanh_action
    ).sum(-1, keepdim=True) - td_math.tanh_log_abs_det_jacobian(pre_tanh_action)

    actual = SoftWorldModel.squashed_component_log_prob(
        pre_tanh_action,
        mean,
        log_std,
    )

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 1)


def test_exact_mixture_density_uses_one_shared_jacobian_and_matches_identity():
    pre_tanh_action = torch.tensor(
        [[0.25, -0.5], [1.0, 0.75]], dtype=torch.float64
    )
    primary = {
        "pre_tanh_mean": torch.tensor(
            [[0.1, -0.3], [0.7, 0.4]], dtype=torch.float64
        ),
        "log_std": torch.tensor(
            [[-0.2, 0.1], [-0.6, 0.3]], dtype=torch.float64
        ),
    }
    explorer = {
        "pre_tanh_mean": torch.tensor(
            [[-0.8, 0.6], [1.3, -0.2]], dtype=torch.float64
        ),
        "log_std": torch.tensor(
            [[0.2, -0.4], [0.1, -0.1]], dtype=torch.float64
        ),
    }
    weight = 0.35
    primary_gaussian = torch.distributions.Normal(
        primary["pre_tanh_mean"], primary["log_std"].exp()
    ).log_prob(pre_tanh_action).sum(-1, keepdim=True)
    explorer_gaussian = torch.distributions.Normal(
        explorer["pre_tanh_mean"], explorer["log_std"].exp()
    ).log_prob(pre_tanh_action).sum(-1, keepdim=True)
    expected = torch.logsumexp(
        torch.stack(
            (
                primary_gaussian + torch.log(torch.tensor(weight)),
                explorer_gaussian + torch.log(torch.tensor(1.0 - weight)),
            )
        ),
        dim=0,
    ) - td_math.tanh_log_abs_det_jacobian(pre_tanh_action)

    actual = SoftWorldModel.mixture_log_prob(
        pre_tanh_action,
        primary,
        explorer,
        weight,
    )
    identical = SoftWorldModel.mixture_log_prob(
        pre_tanh_action,
        primary,
        primary,
        0.73,
    )
    component = SoftWorldModel.squashed_component_log_prob(
        pre_tanh_action,
        primary["pre_tanh_mean"],
        primary["log_std"],
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(identical, component)


def test_exact_mixture_density_preserves_cross_policy_and_sample_gradients():
    primary_mean = torch.tensor([[0.2, -0.4]], requires_grad=True)
    primary_log_std = torch.tensor([[-0.3, 0.1]], requires_grad=True)
    explorer_mean = torch.tensor([[-0.7, 0.6]], requires_grad=True)
    explorer_log_std = torch.tensor([[0.2, -0.5]], requires_grad=True)
    noise = torch.tensor([[0.4, -1.1]])
    pre_tanh_action = primary_mean + noise * primary_log_std.exp()

    log_prob = SoftWorldModel.mixture_log_prob(
        pre_tanh_action,
        {
            "pre_tanh_mean": primary_mean,
            "log_std": primary_log_std,
        },
        {
            "pre_tanh_mean": explorer_mean,
            "log_std": explorer_log_std,
        },
        0.6,
    )
    gradients = torch.autograd.grad(
        -log_prob.sum(),
        (primary_mean, primary_log_std, explorer_mean, explorer_log_std),
    )

    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_disabled_source_storage_preserves_v1_contract_and_sample_fields():
    replay = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    replay.add_batch(*_transition_rows(3))
    sampled = replay.sample(
        2,
        include_ids=False,
        indices=torch.tensor([2, 0]),
    )
    state = replay.training_state_dict()

    assert replay.store_source is False
    assert replay.source is None
    assert replay._storage.shape == (4, 7)
    assert set(sampled) == {"z", "action", "reward", "next_z", "terminated"}
    assert set(state) == {"schema", "version", "latent_dim", "action_dim", "state"}
    assert state["version"] == 1
    assert "source" not in state["state"]
    with pytest.raises(ValueError, match="store_source=True"):
        replay.add_batch(*_transition_rows(1), source=0)


def test_enabled_source_storage_tracks_dense_and_wrapped_rows_and_samples():
    replay = LatentReplayBuffer(
        5,
        latent_dim=2,
        action_dim=1,
        device="cpu",
        store_source=True,
    )
    rows = _transition_rows(6)
    source = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.uint8)
    replay.add_round(*rows, source=source)
    physical = torch.arange(replay.capacity)
    sampled = replay.sample(replay.capacity, indices=physical)

    assert replay.size == 5
    assert replay.pos == 1
    assert replay.source.dtype == torch.uint8
    torch.testing.assert_close(sampled["source"], replay.source)
    # A six-row bulk append into capacity five writes retained logical rows
    # 1..5 exactly as sequential ring appends would.
    torch.testing.assert_close(
        replay.source,
        torch.tensor([[0], [1], [0], [1], [1]], dtype=torch.uint8),
    )

    replay.add_batch(*_transition_rows(2), source=1)
    assert torch.all(replay.source[[1, 2]] == 1)
    with pytest.raises(ValueError, match="required"):
        replay.add_batch(*_transition_rows(1))
    with pytest.raises(ValueError, match="0 or 1"):
        replay.add_packed(torch.zeros(1, 7), source=2)


def test_source_training_state_v2_restores_exactly_and_rejects_cross_mode():
    source = LatentReplayBuffer(
        4,
        latent_dim=2,
        action_dim=1,
        device="cpu",
        store_source=True,
    )
    source.add_batch(
        *_transition_rows(6),
        source=torch.tensor([0, 0, 1, 1, 0, 1]),
    )
    state = deepcopy(source.training_state_dict())
    assert state["version"] == 2
    assert state["store_source"] is True
    assert state["state"]["source"].dtype == torch.uint8

    restored = LatentReplayBuffer(
        4,
        latent_dim=2,
        action_dim=1,
        device="cpu",
        store_source=True,
    ).load_training_state_dict(state)
    indices = torch.tensor([3, 0, 2, 2, 1])
    restored_batch = restored.sample(len(indices), indices=indices)
    source_batch = source.sample(len(indices), indices=indices)
    for key in source_batch:
        torch.testing.assert_close(restored_batch[key], source_batch[key])

    disabled = LatentReplayBuffer(4, latent_dim=2, action_dim=1, device="cpu")
    with pytest.raises(ValueError, match="incompatible schema"):
        disabled.load_training_state_dict(state)
    assert disabled.size == 0

    invalid = deepcopy(state)
    invalid["state"]["source"][0, 0] = 2
    pristine = deepcopy(restored.training_state_dict())
    with pytest.raises(ValueError, match="0 or 1"):
        restored.load_training_state_dict(invalid)
    assert restored.pos == pristine["state"]["pos"]
    torch.testing.assert_close(
        restored.training_state_dict()["state"]["source"],
        pristine["state"]["source"],
    )
