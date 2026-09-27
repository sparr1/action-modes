"""Reward-return C51 targets, state ownership, and inner controller primitives."""

from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.xqc_auxiliary_return import XQCAuxiliaryReturnLearner
from RL.tdmpc2_core.xqc_controller import (
    LatentXQCBatch, LatentXQCConfig, LatentXQCController,
)


def _controller(device="cpu"):
    return LatentXQCController(2, 1, LatentXQCConfig(
        actor_net_arch=(8,), critic_net_arch=(8,), num_atoms=7,
        vmin=-2., vmax=2., init_temperature=.3, tau=.2,
        optimizer_backend="single_tensor",
    )).to(device)


def _aux(primary, *, compile=False):
    cfg = SimpleNamespace(
        seed=17, xqc_critic_lr=3e-4, xqc_lr_end=3e-5,
        xqc_lr_transition_steps=100, compile=compile, compile_strict=False,
    )
    return XQCAuxiliaryReturnLearner(primary, cfg, next(primary.parameters()).device)


def _batch(device="cpu"):
    return LatentXQCBatch(
        latents=torch.tensor([[.1, .3], [-.2, .7], [.9, -.4], [.5, .5]], device=device),
        actions=torch.tensor([[.2], [-.8], [.5], [0.]], device=device),
        rewards=torch.tensor([.4, -.6, 1., -.1], device=device),
        next_latents=torch.tensor([[.4, .1], [.1, .8], [-.6, .2], [.3, -.1]], device=device),
        bootstrap_mask=torch.tensor([1., 1., 0., 1.], device=device), discount=.9,
    )


def _same(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _oracle(log_q, batch, support, *, reward_scale, entropy):
    """Scalar C51 interpolation, independently selecting the lower twin head."""
    q_values = (log_q.exp() * support).sum(-1)
    heads = q_values.argmin(0)
    result = torch.zeros_like(log_q[0])
    spacing = float((support[-1] - support[0]) / (len(support) - 1))
    clipped = 0
    for row in range(batch.rewards.numel()):
        for atom, value in enumerate(support):
            target = float(batch.rewards[row] / reward_scale + batch.discount
                           * batch.bootstrap_mask[row] * (value - entropy[row]))
            target = min(max(target, float(support[0])), float(support[-1]))
            clipped += target in (float(support[0]), float(support[-1]))
            index = min(max((target - float(support[0])) / spacing, 0), len(support) - 1)
            low, high = math.floor(index), math.ceil(index)
            probability = log_q[heads[row], row, atom].exp()
            result[row, low] += probability * (high + (low == high) - index)
            result[row, high] += probability * (index - low)
    return result, clipped / result.numel()


def test_auxiliary_owns_only_critics_and_preserves_primary_and_global_rng():
    torch.manual_seed(7)
    primary = _controller()
    before, rng = deepcopy(primary.state_dict()), torch.get_rng_state().clone()
    aux = _aux(primary)
    assert torch.equal(torch.get_rng_state(), rng)
    assert set(aux._modules) == {"critic", "critic_target"}
    assert not set(map(id, primary.parameters())) & set(map(id, aux.parameters()))
    assert not any("actor" in name or "temperature" in name for name in aux.state_dict())
    objective = aux.critic_objective(_batch(), actor=primary.actor, reward_scale=2.5)
    aux.zero_grad()
    objective.loss.backward()
    aux.step()
    assert _same(primary.state_dict(), before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(parameter.grad is None for parameter in primary.parameters())


def test_auxiliary_reward_only_oracle_raw_scale_and_bn_ownership(monkeypatch):
    torch.manual_seed(12)
    primary = _controller()
    aux = _aux(primary)
    batch = _batch()
    incoming_rewards = batch.rewards.clone()
    online_before = deepcopy(dict(aux.critic.named_buffers()))
    target_before = deepcopy(dict(aux.critic_target.named_buffers()))
    actions = []
    sample = primary.actor.sample

    def capture(z, **kwargs):
        assert kwargs["bn_mode"] == "running"
        result = sample(z, **kwargs)
        actions.append(result[0].clone())
        return result

    monkeypatch.setattr(primary.actor, "sample", capture)
    actual = aux.critic_objective(batch, actor=primary.actor, reward_scale=2.5)
    with torch.no_grad():
        target_log = aux.critic_target.log_probs(
            torch.cat((batch.latents, batch.next_latents)),
            torch.cat((batch.actions, actions[0])), bn_mode="batch_no_update",
        )[:, batch.rewards.numel():]
        expected, clipped = _oracle(
            target_log, batch, aux.critic.support,
            reward_scale=2.5, entropy=torch.zeros_like(batch.rewards),
        )
    torch.testing.assert_close(actual.target_probabilities, expected, atol=2e-7, rtol=1e-6)
    assert actual.clip_fraction.item() == pytest.approx(clipped)
    torch.testing.assert_close(
        actual.loss, -(expected.unsqueeze(0) * actual.current_log_probs).sum(-1).sum(0).mean()
    )
    assert torch.equal(batch.rewards, incoming_rewards)
    assert not _same(dict(aux.critic.named_buffers()), online_before)
    assert _same(dict(aux.critic_target.named_buffers()), target_before)
    old_target = deepcopy(dict(aux.critic_target.named_parameters()))
    aux.zero_grad()
    actual.loss.backward()
    info = aux.step()
    assert info["aux_return_target_updated"] == 1.
    assert info["aux_return_num_updates"] == 1.
    assert info["aux_return_learning_rate"] == pytest.approx(3e-4)
    for name, target in aux.critic_target.named_parameters():
        torch.testing.assert_close(target, old_target[name].lerp(dict(aux.critic.named_parameters())[name], .2))
    assert _same(dict(aux.critic_target.named_buffers()), target_before)
    for weight in aux._critic_linear_weights:
        torch.testing.assert_close(weight.norm(dim=1), torch.ones(weight.shape[0]))


def test_auxiliary_training_state_round_trip_continues_private_sampling_exactly():
    torch.manual_seed(34)
    primary = _controller()
    aux = _aux(primary)
    for _ in range(2):
        aux.zero_grad()
        aux.critic_objective(_batch(), actor=primary.actor).loss.backward()
        aux.step()
    module, state = deepcopy(aux.state_dict()), deepcopy(aux.training_state_dict())
    restored = _aux(primary)
    restored.load_state_dict(module)
    candidate = restored.preflight_training_state(state, expected_updates=2)
    restored.load_training_state(candidate)
    assert _same(restored.training_state_dict(), aux.training_state_dict())
    for learner in (aux, restored):
        learner.zero_grad()
        learner.critic_objective(_batch(), actor=primary.actor).loss.backward()
        learner.step()
    assert _same(restored.state_dict(), aux.state_dict())
    assert _same(restored.training_state_dict(), aux.training_state_dict())


@pytest.mark.parametrize("corruption", ["counter", "missing", "step", "nonfinite", "rng", "schema"])
def test_auxiliary_preflight_rejects_corrupt_state_without_mutation(corruption):
    primary = _controller()
    aux = _aux(primary)
    aux.zero_grad()
    aux.critic_objective(_batch(), actor=primary.actor).loss.backward()
    aux.step()
    before = deepcopy(aux.training_state_dict())
    corrupt = deepcopy(before)
    if corruption == "counter":
        corrupt["update_step"] += 1
    elif corruption == "missing":
        corrupt["critic_optimizer"]["state"].pop(next(iter(corrupt["critic_optimizer"]["state"])))
    elif corruption in {"step", "nonfinite"}:
        item = next(iter(corrupt["critic_optimizer"]["state"].values()))
        if corruption == "step":
            item["step"].zero_()
        else:
            item["exp_avg"].reshape(-1)[0] = float("nan")
    elif corruption == "rng":
        corrupt["generator"] = torch.zeros(2, dtype=torch.uint8)
    else:
        corrupt["unexpected"] = 3
    with pytest.raises((ValueError, TypeError)):
        aux.preflight_training_state(corrupt, expected_updates=1)
    assert _same(aux.training_state_dict(), before)


def test_selected_critic_reset_copies_auxiliary_bn_but_primary_actor_and_temperature():
    primary = _controller()
    aux = _aux(primary)
    aux.critic_objective(_batch(), actor=primary.actor)
    workspace = primary.clone_for_inner(actor_lr=1e-4, critic_lr=1e-4, critic_source=aux.critic)
    assert _same(workspace.controller.actor.state_dict(), primary.actor.state_dict())
    assert torch.equal(workspace.controller.log_temperature, primary.log_temperature)
    assert _same(workspace.controller.critic.state_dict(), aux.critic.state_dict())
    assert _same(workspace.controller.critic_target.state_dict(), aux.critic.state_dict())
    workspace.update(
        _batch(), next_noise=torch.zeros(4, 1), actor_noise=torch.zeros(4, 1),
        critic_target_kind="reward_only",
    )
    assert workspace.critic_optimizer.state
    workspace.reset_from_(primary, critic_source=aux.critic)
    assert workspace.update_step == 0 and not workspace.critic_optimizer.state
    assert _same(workspace.controller.critic.state_dict(), aux.critic.state_dict())
    assert _same(workspace.controller.critic_target.state_dict(), aux.critic.state_dict())
    incompatible = deepcopy(aux.critic)
    incompatible.support.add_(1)
    before = deepcopy(workspace.controller.state_dict())
    with pytest.raises(ValueError, match="support"):
        workspace.reset_from_(primary, critic_source=incompatible)
    assert _same(workspace.controller.state_dict(), before)


@pytest.mark.parametrize("kind", ["entropy_augmented", "reward_only"])
@pytest.mark.parametrize("return_tail", [False, True])
def test_inner_targets_and_return_tail_have_independent_entropy_rules(kind, return_tail):
    torch.manual_seed(9)
    inner, outer = _controller(), _controller()
    aux = _aux(outer)
    batch, noise = _batch(), torch.tensor([[.2], [-.9], [.1], [.8]])
    mask = torch.tensor([False, True, True, False])
    selected_outer = aux.critic if return_tail else outer.critic
    original = deepcopy(outer.state_dict()), deepcopy(aux.state_dict())
    with torch.no_grad():
        actions, inner_log_prob = inner.actor.sample(batch.next_latents, bn_mode="running", noise=noise)
        target_log = inner.critic_target.log_probs(
            torch.cat((batch.latents, batch.next_latents)),
            torch.cat((batch.actions, actions)), bn_mode="batch_no_update",
        )[:, 4:]
        outer_actions, outer_log_prob = outer.actor.sample(batch.next_latents, bn_mode="running", noise=noise)
        outer_log = selected_outer.log_probs(batch.next_latents, outer_actions, bn_mode="running")
        inner_entropy = inner.temperature * inner_log_prob if kind == "entropy_augmented" else torch.zeros(4)
        outer_entropy = inner.temperature * outer_log_prob if kind == "entropy_augmented" and not return_tail else torch.zeros(4)
        expected_inner, _ = _oracle(target_log, batch, inner.critic.support, reward_scale=2.5, entropy=inner_entropy)
        expected_outer, _ = _oracle(outer_log, batch, inner.critic.support, reward_scale=2.5, entropy=outer_entropy)
        expected = torch.where((mask & batch.bootstrap_mask.bool())[:, None], expected_outer, expected_inner)
    actual = inner.critic_objective(
        batch, next_noise=noise, reward_scale=2.5, critic_target_kind=kind,
        outer_terminal_mask=mask, outer_controller=outer,
        outer_critic=selected_outer if return_tail else None,
        outer_critic_is_return=return_tail,
    )
    torch.testing.assert_close(actual.target_probabilities, expected, atol=2e-7, rtol=1e-6)
    actual.loss.backward()
    assert _same(outer.state_dict(), original[0]) and _same(aux.state_dict(), original[1])
    assert all(p.grad is None for module in (outer, aux) for p in module.parameters())


def test_auxiliary_compile_failure_restores_bn_and_draws_target_noise_once(monkeypatch):
    primary = _controller()
    baseline, failing = _aux(primary), _aux(primary)
    expected = baseline.critic_objective(_batch(), actor=primary.actor)

    def compile_then_fail(fn, **kwargs):
        def partial(*args):
            fn(*args)
            raise RuntimeError("simulated first graph failure")
        return partial

    monkeypatch.setattr(torch, "compile", compile_then_fail)
    failing._critic_loss_region.enabled = True
    with pytest.warns(RuntimeWarning, match="Falling back"):
        actual = failing.critic_objective(_batch(), actor=primary.actor)
    assert torch.equal(actual.loss, expected.loss)
    assert _same(failing.state_dict(), baseline.state_dict())
    assert _same(failing.training_state_dict(), baseline.training_state_dict())
    assert failing.compile_status["fallback"]


def test_auxiliary_cpu_compile_request_stays_eager_and_supports_latent_gradients():
    primary = _controller()
    aux = _aux(primary, compile=True)
    assert aux.compile_status["requested"] and not aux.compile_status["enabled"]
    batch = _batch()
    batch.latents.requires_grad_(True)
    aux.critic_objective(batch, actor=primary.actor).loss.backward()
    assert batch.latents.grad is not None and batch.latents.grad.abs().sum() > 0


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only CUDA wire preflight")
def test_auxiliary_cuda_checkpoint_rng_requires_frozen_evaluation_and_valid_wire_state():
    aux = _aux(_controller())
    saved = deepcopy(aux.training_state_dict())
    saved["device_type"] = "cuda"
    saved["generator"] = torch.zeros(16, dtype=torch.uint8)
    before = deepcopy(aux.training_state_dict())
    with pytest.raises(ValueError, match="frozen evaluation"):
        aux.preflight_training_state(saved, expected_updates=0)
    candidate = aux.preflight_training_state(saved, expected_updates=0, frozen_evaluation=True)
    assert candidate["device_type"] == "cpu"
    aux.load_training_state(candidate, frozen_evaluation=True)
    assert _same(aux.training_state_dict(), before)
    saved["generator"][8] = 1
    with pytest.raises(ValueError, match="divisible by four"):
        aux.preflight_training_state(saved, expected_updates=0, frozen_evaluation=True)
    assert _same(aux.training_state_dict(), before)


@pytest.mark.parametrize("kwargs", [
    {"critic_target_kind": "return"},
    {"outer_critic_is_return": 1},
    {"outer_critic_is_return": True},
])
def test_critic_target_options_reject_invalid_or_incomplete_routing(kwargs):
    controller = _controller()
    before = deepcopy(controller.state_dict())
    with pytest.raises(ValueError):
        controller.critic_objective(_batch(), next_noise=torch.zeros(4, 1), **kwargs)
    assert _same(controller.state_dict(), before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_auxiliary_cuda_compile_matches_eager():
    primary = _controller("cuda")
    eager, compiled = _aux(primary), _aux(primary, compile=True)
    actual = compiled.critic_objective(_batch("cuda"), actor=primary.actor)
    expected = eager.critic_objective(_batch("cuda"), actor=primary.actor)
    torch.testing.assert_close(actual.loss, expected.loss, atol=2e-5, rtol=2e-5)
    for learner, objective in ((eager, expected), (compiled, actual)):
        learner.zero_grad()
        objective.loss.backward()
        learner.step()
    for key, value in eager.state_dict().items():
        torch.testing.assert_close(value, compiled.state_dict()[key], atol=2e-5, rtol=2e-5)
