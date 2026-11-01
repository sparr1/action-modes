"""Frozen separate-view PAVE oracle for parity tests and bounded benchmarks.

This retains the pre-view-batching calculation, extended with a separate
action-perturbation forward. Keep its independent forwards and derivatives:
sharing the optimized implementation would defeat the oracle.
"""

from contextlib import contextmanager

import torch

from RL.tdmpc2_core.common.layers import Ensemble


@contextmanager
def _geometry_mode(critic):
    modes = tuple((module, module.training) for module in critic.modules())
    try:
        critic.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def _action_gradient(value, action):
    if not value.requires_grad:
        return action * 0.0
    result = torch.autograd.grad(value.sum(), action, create_graph=True, allow_unused=True)[0]
    if result is None:
        return action * 0.0 + value.sum() * 0.0
    return result + action * 0.0


def separate_critic_pave_loss(
    critic, decode, z, action, *, noisy_z=None, next_z=None, probe=None, noisy_action=None,
    mpr_coef=0.0, vfc_coef=0.0, curvature_coef=0.0, curvature_margin=1.0,
    action_smoothness_coef=0.0, return_components=False,
):
    coefficients = dict(mpr=mpr_coef, vfc=vfc_coef, curvature=curvature_coef,
                        action_smoothness=action_smoothness_coef)
    if not any(coefficient > 0.0 for coefficient in coefficients.values()):
        result = (action.new_zeros(()), {})
        return (*result, {}) if return_components else result
    actions = action.detach().unsqueeze(0).expand(len(critic), -1, -1).clone()
    actions.requires_grad_(True)

    def gradient(latents, coordinates=actions):
        latents = latents.detach()
        if isinstance(critic, Ensemble):
            expanded = latents.unsqueeze(0).expand(coordinates.shape[0], *latents.shape)
            predictions = critic.forward_heads_eager(torch.cat((expanded, coordinates), dim=-1))
        else:
            predictions = torch.stack([
                head(torch.cat((latents, coordinates[index]), dim=-1))
                for index, head in enumerate(critic)
            ])
        return _action_gradient(decode(predictions), coordinates)

    losses, metrics = {}, {}
    with _geometry_mode(critic):
        original = gradient(z)
        if return_components:
            metrics["action_gradient_rms"] = original.detach().square().mean().sqrt()
        if mpr_coef > 0.0:
            losses["mpr"] = (original - gradient(noisy_z)).square().mean(dim=(1, 2)).sum()
        if vfc_coef > 0.0:
            losses["vfc"] = (original - gradient(next_z)).square().mean(dim=(1, 2)).sum()
        if curvature_coef > 0.0:
            vectors = probe.detach().unsqueeze(0)
            hvp = _action_gradient((original * vectors).sum(), actions)
            margin_error = (hvp * vectors).sum(dim=-1) + curvature_margin
            losses["curvature"] = margin_error.relu().mean(dim=1).sum()
            metrics["curvature_violation_fraction"] = (margin_error.detach() > 0).float().mean()
        if action_smoothness_coef > 0.0:
            perturbed = noisy_action.detach().unsqueeze(0).expand(len(critic), -1, -1).clone()
            perturbed.requires_grad_(True)
            losses["action_smoothness"] = (original - gradient(z, perturbed)).square().mean(dim=(1, 2)).sum()

    total = action.new_zeros(())
    for term, loss in losses.items():
        weighted = float(coefficients[term]) * loss
        total = total + weighted
        metrics[f"{term}_loss"] = loss.detach()
        metrics[f"{term}_weighted_loss"] = weighted.detach()
    metrics["weighted_loss"] = total.detach()
    return (total, metrics, losses) if return_components else (total, metrics)
