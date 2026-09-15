"""Numeric and architectural contracts for opt-in component-valued critics."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from RL.tdmpc2_core.common import layers, math as td_math
from RL.tdmpc2_core.common.lora import make_lora_rl_critic
from RL.tdmpc2_core.common.q_representation import SymexpTwoHotCodec
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from RL.tdmpc2_core.common.value_semantics import ValueSpecification


def split_cfg(**overrides):
    cfg = dict(
        multitask=False, obs_shape={"state": (3,)}, obs="state", task_dim=0,
        num_enc_layers=2, enc_dim=16, latent_dim=8, simnorm_dim=4,
        action_dim=2, mlp_dim=16, num_bins=7, vmin=-3., vmax=3., bin_size=1.,
        episodic=False, dropout=0., log_std_min=-5., log_std_max=2., tau=.01,
        q_representation="distributional", num_q=3, q_pair_size=2,
        q_num_bins=7, q_vmin=-3., q_vmax=3., critic_value_mode="return_entropy",
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_symexp_two_hot_preserves_signed_means_endpoints_and_mixtures():
    codec = SymexpTwoHotCodec(101, -10, 10)
    target = torch.tensor([-1e9, -100., -1.23, 0., .031, 40., 1e9], dtype=torch.float64)[:, None]
    support = codec.support(target)
    encoded = codec.encode_target(target)
    clipped = target.clamp(support[0], support[-1])
    torch.testing.assert_close(encoded.sum(-1), torch.ones(len(target), dtype=target.dtype))
    torch.testing.assert_close((encoded * support).sum(-1, keepdim=True), clipped)
    torch.testing.assert_close(codec.decode(encoded.log()), clipped)
    torch.testing.assert_close(codec.decode(encoded.mean(0).log()), clipped.mean(0))
    assert codec.clipping_fraction(target).item() == pytest.approx(2 / 7)
    point_targets = support[:, None]
    torch.testing.assert_close(codec.encode_target(point_targets), torch.eye(101, dtype=target.dtype))


def test_component_sparse_cross_entropy_matches_dense_values_and_gradients():
    codec = SymexpTwoHotCodec(7, -3, 3)
    target = torch.tensor([[[-9.], [2.]], [[0.], [100.]], [[-1.], [.4]]])
    predictions = torch.randn(4, 3, 2, 7, requires_grad=True)
    losses = codec.loss(predictions, target, reduction="none")
    reference = -(codec.encode_target(target) * predictions.log_softmax(-1)).sum(-1, keepdim=True)
    assert losses.shape == (4, 3, 2, 1)
    torch.testing.assert_close(losses, reference)
    grad, = torch.autograd.grad(losses.sum(), predictions, retain_graph=True)
    reference_grad, = torch.autograd.grad(reference.sum(), predictions)
    torch.testing.assert_close(grad, reference_grad)


def test_packed_heads_have_separate_component_softmax_and_symmetric_initialization():
    model = SoftWorldModel(split_cfg())
    z, action = torch.randn(4, 8), torch.randn(4, 2)
    predictions = model.q_predictions(z, action)
    assert predictions.shape == (3, 4, 2, 7)
    assert model.q_values(z, action).shape == (3, 4, 2, 1)
    assert model._Qs[0][-1].out_features == 14
    assert torch.count_nonzero(predictions) == 0
    assert torch.count_nonzero(model._reward[-1].bias) == 0
    packed = model._Qs(model.joint_input(z, action))
    torch.testing.assert_close(predictions, packed.reshape(3, 4, 2, 7))
    modified = predictions.clone()
    modified[..., 1, :] += torch.arange(7)
    torch.testing.assert_close(model.q_backend.decode(modified)[..., 0, :], model.q_values(z, action)[..., 0, :])
    with pytest.raises(ValueError, match="require"):
        model.Q(z, action)
    for projection, beta in (("return", None), ("policy", .1)):
        assert model.Q(z, action, projection=projection, beta=beta).shape == (4, 1)


def test_min_reduction_gathers_complete_member_with_batched_independent_indices():
    model = SoftWorldModel(split_cfg())
    # The first batch chooses member 1 for beta=2, the second chooses member 0.
    values = torch.tensor([
        [[[0.], [10.]], [[4.], [0.]]],
        [[[5.], [0.]], [[0.], [10.]]],
        [[[-100.], [-100.]], [[-100.], [-100.]]],
    ])
    selected = model.reduce_components(values, "min_pair", projection="policy", beta=2., pair_indices=[0, 1])
    torch.testing.assert_close(selected, torch.stack((values[1, 0], values[0, 1])))
    torch.testing.assert_close(
        model.project_values(selected, weights=(1., 2.)), torch.tensor([[5.], [4.]])
    )
    return_selected = model.reduce_components(values, "min_pair", projection="return", pair_indices=[0, 1])
    torch.testing.assert_close(return_selected, torch.stack((values[0, 0], values[1, 1])))
    with pytest.raises(ValueError, match="unique"):
        model.reduce_components(values, "min_pair", weights=(1., 2.), pair_indices=[1, 1])


@pytest.mark.parametrize("terminal_value", [9., float("nan"), float("inf"), float("-inf")])
def test_named_outer_targets_start_entropy_at_next_action_and_detach(terminal_value):
    spec = ValueSpecification.from_config(split_cfg())
    next_components = torch.tensor(
        [[[3.], [7.]], [[terminal_value], [terminal_value]]], requires_grad=True,
    )
    target = spec.outer_targets(
        torch.tensor([[2.], [4.]], requires_grad=True), .9,
        torch.tensor([[0.], [1.]]), next_components,
        torch.tensor([[5.], [terminal_value]], requires_grad=True),
    )
    torch.testing.assert_close(target, torch.tensor([[[4.7], [10.8]], [[4.], [0.]]]))
    assert not target.requires_grad


def test_component_targets_and_decoders_share_mean_preserving_reward_codec():
    model = SoftWorldModel(split_cfg())
    target = torch.tensor([[-8.], [.3], [4.]])
    prediction = model.encode_reward(target).log()
    torch.testing.assert_close(model.decode_reward(prediction), target)
    torch.testing.assert_close(model.reward_loss(prediction, target), model.reward_codec.loss(prediction, target, reduction="none"))
    q_prediction = model.q_backend.encode_target(target).log().expand(3, 3, 7)
    torch.testing.assert_close(model.q_backend.decode(q_prediction), target.expand(3, 3, 1))


def test_single_mode_reward_wrappers_preserve_legacy_arithmetic_and_signature():
    model = SoftWorldModel(split_cfg(critic_value_mode="single"))
    predictions, target = torch.randn(3, 7), torch.randn(3, 1)
    torch.testing.assert_close(model.decode_reward(predictions), td_math.two_hot_inv(predictions, model.cfg), rtol=0, atol=0)
    torch.testing.assert_close(model.reward_loss(predictions, target), td_math.soft_ce(predictions, target, model.cfg), rtol=0, atol=0)
    assert set(model.critic_signature) == {"q_representation", "num_q", "q_num_bins", "q_vmin", "q_vmax"}
    split = SoftWorldModel(split_cfg())
    assert split.critic_signature["value_components"] == ["return", "entropy"]
    assert split.critic_signature["q_value_codec"] == split.critic_signature["reward_value_codec"]


@pytest.mark.parametrize("component", [0, 1])
def test_each_component_loss_trains_encoder_and_recurrent_dynamics(component):
    torch.manual_seed(8)
    model = SoftWorldModel(split_cfg())
    with torch.no_grad():
        for critic in model._Qs:
            critic[-1].weight.normal_(std=.08)
    latent = model.encode(torch.randn(5, 3))
    latent_next = model.next(latent, torch.randn(5, 2))
    prediction = model.q_predictions(latent_next, torch.randn(5, 2))
    target = torch.full((5, 2, 1), 2.)
    model.critic_loss(prediction, target, reduction="none")[..., component, :].mean().backward()
    for module in (model._encoder, model._dynamics):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model._target_Qs.parameters())


@pytest.mark.parametrize("component", [0, 1])
def test_each_component_preserves_action_gradient_with_detached_critic(component):
    torch.manual_seed(2)
    model = SoftWorldModel(split_cfg())
    with torch.no_grad():
        for critic in model._Qs:
            critic[-1].weight.normal_(std=.08)
    action = torch.randn(5, 2, requires_grad=True)
    values = model.q_values(torch.randn(5, 8), action, detach=True)
    values[..., component, :].sum().backward()
    assert action.grad.abs().sum() > 0
    assert all(p.grad is None for p in model._Qs.parameters())


def test_clone_lora_and_target_ema_preserve_packed_component_layout():
    model = SoftWorldModel(split_cfg())
    with torch.no_grad():
        for critic in model._Qs:
            critic[-1].weight.normal_(std=.1)
    dense = deepcopy(model._Qs)
    lora = layers.Ensemble([make_lora_rl_critic(q, rank=4, scale=1.) for q in model._Qs])
    z, action = torch.randn(4, 8), torch.randn(4, 2)
    for qs in (dense, lora):
        torch.testing.assert_close(model.q_predictions(z, action, qs=qs), model.q_predictions(z, action))
        assert model.q_predictions(z, action, qs=qs, detach=True).shape == (3, 4, 2, 7)
    old = [p.clone() for p in model._target_Qs.parameters()]
    model.soft_update_target_Q(.25)
    for before, target, online in zip(old, model._target_Qs.parameters(), model._Qs.parameters()):
        torch.testing.assert_close(target, before.lerp(online, .25))


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_compiled_codec_and_shared_selection_match_eager_values_and_gradients(backend):
    model = SoftWorldModel(split_cfg())
    codec = model.q_backend.value_codec
    pair = torch.tensor([0, 2])
    target = torch.randn(4, 2, 1)
    prediction = torch.randn(3, 4, 2, 7, requires_grad=True)
    # Populate support outside the compiled region, as model kernels do.
    codec.support(prediction)

    def calculation(logits, targets, indices):
        values = codec.decode(logits)
        selected = model.reduce_components(values, "min_pair", weights=(1., .2), pair_indices=indices, trusted_pair_indices=True)
        return codec.loss(logits, targets) + model.project_values(selected, weights=(1., .2)).mean()

    compiled = torch.compile(calculation, backend=backend, fullgraph=True)
    eager_loss = calculation(prediction, target, pair)
    compiled_loss = compiled(prediction, target, pair)
    torch.testing.assert_close(compiled_loss, eager_loss)
    eager_grad, = torch.autograd.grad(eager_loss, prediction)
    compiled_grad, = torch.autograd.grad(compiled_loss, prediction)
    torch.testing.assert_close(compiled_grad, eager_grad)
