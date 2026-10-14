"""Weight-only Bernoulli initialization from the previous inner critic."""

import math

import torch


@torch.no_grad()
def reset_critic_with_bernoulli_(critic, prior, probability, *, generator, measure=False):
    """Retain each old scalar parameter with p, otherwise copy the current prior.

    ``critic`` owns the previous solve's parameters. Updating them in place
    preserves optimizer ownership and compiled module identities. Buffers always
    come from the current prior; neither dropout scaling nor delta rebasing is
    applied. The caller resets optimizer moments and the target afterwards.
    """
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("Bernoulli probability must be finite and in [0, 1].")
    if 0 < probability < 1 and generator is None:
        raise ValueError("Bernoulli copying requires an isolated generator.")
    parameters, source = dict(critic.named_parameters()), dict(prior.named_parameters())
    buffers, source_buffers = dict(critic.named_buffers()), dict(prior.named_buffers())
    if not parameters or parameters.keys() != source.keys() or buffers.keys() != source_buffers.keys():
        raise ValueError("Bernoulli critic and prior must have matching parameters and buffers.")
    # Validate the entire layout before changing any tensor or drawing a mask.
    for targets, sources in ((parameters, source), (buffers, source_buffers)):
        for name, target in targets.items():
            base = sources[name]
            if target.shape != base.shape or target.dtype != base.dtype or target.device != base.device:
                raise ValueError(f"Incompatible Bernoulli transfer tensor: {name}.")
    if any(not value.is_floating_point() for value in parameters.values()):
        raise ValueError("Bernoulli copying requires floating-point parameters.")

    retained = []
    for name, target in parameters.items():
        if probability == 0:
            target.copy_(source[name])
        elif probability != 1:
            mask = torch.rand(target.shape, device=target.device, generator=generator) < probability
            target.copy_(torch.where(mask, target, source[name]))
            if measure:
                retained.append(mask.sum())
    for name, target in buffers.items():
        target.copy_(source_buffers[name])
    if not measure:
        return None
    if probability in (0, 1):
        return float(probability)
    return torch.stack(retained).sum() / sum(value.numel() for value in parameters.values())
