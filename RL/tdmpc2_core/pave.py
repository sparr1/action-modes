"""PAVE geometry losses on real observations for the outer AMBI critics.

Reference: https://github.com/AIRLABkhu/PAVE (SAC implementation). The
regularizers sum independent critic heads, use recorded actions in all three
state views, and leave the ordinary Bellman objective and Q decoder intact.
"""

from contextlib import contextmanager

import torch


_TERMS = ("mpr", "vfc", "curvature")
_CRITICS = (("outer_pave", "_Qs"), ("aux_return_pave", "_aux_return_Qs"))


def pave_enabled(cfg):
    return any(
        float(getattr(cfg, f"{prefix}_{term}_coef", 0.0)) > 0.0
        for prefix, _ in _CRITICS for term in _TERMS
    )


@contextmanager
def _geometry_mode(critic):
    """Disable dropout without losing mixed train/eval flags on restoration."""
    modes = tuple((module, module.training) for module in critic.modules())
    try:
        critic.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def _action_gradient(value, action):
    # Constant/affine critics legitimately have zero first/second derivatives.
    # Keep the zero connected so a curvature-only loss remains backwardable.
    if not value.requires_grad:
        return action * 0.0
    gradient = torch.autograd.grad(
        value.sum(), action, create_graph=True, allow_unused=True,
    )[0]
    if gradient is None:
        return action * 0.0 + value.sum() * 0.0
    return gradient + action * 0.0


def _head_predictions(critic, latents, actions):
    """Eager, independent head forwards with head-specific action coordinates."""
    return torch.stack([
        member(torch.cat((latents, actions[index]), dim=-1))
        for index, member in enumerate(critic)
    ], dim=0)


def critic_pave_loss(
    critic, decode, z, action, *, noisy_z=None, next_z=None, probe=None,
    mpr_coef=0.0, vfc_coef=0.0, curvature_coef=0.0, curvature_margin=1.0,
    return_components=False,
):
    """Return weighted loss and detached metrics for one independent ensemble.

    Inputs are flat transition batches. ``decode`` receives every head at once
    and must return [heads, rows, 1]. Separate action coordinates for each head
    let a single autograd call compute all head gradients without cancellation.
    Encoder graphs and replay actions are deliberately detached here as well as
    at the outer call site. Only the selected online critic is differentiated.
    """
    coefficients = dict(mpr=mpr_coef, vfc=vfc_coef, curvature=curvature_coef)
    if not any(coefficient > 0.0 for coefficient in coefficients.values()):
        result = (action.new_zeros(()), {})
        return (*result, {}) if return_components else result
    if mpr_coef > 0.0 and noisy_z is None:
        raise ValueError("PAVE MPR requires perturbed-observation latents.")
    if vfc_coef > 0.0 and next_z is None:
        raise ValueError("PAVE VFC requires successor-observation latents.")
    if curvature_coef > 0.0 and (probe is None or probe.shape != action.shape):
        raise ValueError("PAVE curvature requires one action-shaped probe per transition.")

    actions = action.detach().unsqueeze(0).expand(len(critic), -1, -1).clone()
    actions.requires_grad_(True)

    def gradient(latents):
        latents = latents.detach()
        # Call the members directly: Ensemble.forward may invoke a compiled
        # wrapper whose AOT backward does not support the higher derivatives.
        predictions = _head_predictions(critic, latents, actions)
        values = decode(predictions)
        if values.shape != (*actions.shape[:-1], 1):
            raise ValueError("PAVE requires a single decoded scalar per critic head and row.")
        return _action_gradient(values, actions)

    losses = {}
    metrics = {}
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

    total = action.new_zeros(())
    for term, loss in losses.items():
        weighted = float(coefficients[term]) * loss
        total = total + weighted
        metrics[f"{term}_loss"] = loss.detach()
        metrics[f"{term}_weighted_loss"] = weighted.detach()
    metrics["weighted_loss"] = total.detach()
    return (total, metrics, losses) if return_components else (total, metrics)


def prepare_pave_inputs(
    model, obs, action, *, seed, update_number, batch_size=64,
    obs_noise_std=0.01, needs_noise=True, needs_next=True, needs_probe=True,
):
    """Prepare one shared geometry sample using the training RNG convention.

    This is shared by training and frozen diagnostics so a probe need not
    approximate transition selection, observation perturbations, or SimNorm.
    All returned tensors are detached. Only requested views consume draws.
    """
    if obs.ndim != 3 or action.ndim != 3 or obs.shape[:2] != (action.shape[0] + 1, action.shape[1]):
        raise ValueError("PAVE expects aligned state observations [H+1,B,S] and actions [H,B,A].")
    generator = torch.Generator(device=action.device)
    generator.manual_seed(
        (int(seed) + 1_000_003 * int(update_number) + 0x50415645) % (2**63 - 1)
    )
    with torch.no_grad():
        current_obs = obs[:-1].flatten(0, 1)
        successor_obs = obs[1:].flatten(0, 1)
        actions = action.detach().flatten(0, 1)
        if batch_size is not None and int(batch_size) < actions.shape[0]:
            rows = torch.randperm(actions.shape[0], device=action.device, generator=generator)[:int(batch_size)]
            current_obs, successor_obs, actions = current_obs[rows], successor_obs[rows], actions[rows]
        z = model.encode(current_obs).detach()
        noisy_z = None
        if needs_noise:
            noise = torch.randn(
                current_obs.shape, dtype=current_obs.dtype, device=current_obs.device, generator=generator,
            ) * float(obs_noise_std)
            noisy_z = model.encode(current_obs + noise).detach()
        next_z = model.encode(successor_obs).detach() if needs_next else None
        probe = (
            torch.randint(0, 2, actions.shape, device=actions.device, generator=generator)
            .to(actions.dtype).mul_(2).sub_(1)
            if needs_probe else None
        )
    return dict(z=z, action=actions, noisy_z=noisy_z, next_z=next_z, probe=probe)


def outer_pave_loss(model, obs, action, cfg, update_number):
    """Regularize selected critics on shared, paired real replay transitions.

    No persistent RNG state is needed: the training seed and saved outer update
    counter determine these draws independently of actor/replay/dropout RNG.
    The caller adds the result to the existing total before its single backward.
    """
    if not pave_enabled(cfg):
        return action.new_zeros(()), {}
    if obs.ndim != 3 or action.ndim != 3 or obs.shape[:2] != (action.shape[0] + 1, action.shape[1]):
        raise ValueError("PAVE expects aligned state observations [H+1,B,S] and actions [H,B,A].")

    selected = [
        (prefix, getattr(model, attribute), {
            f"{term}_coef": float(getattr(cfg, f"{prefix}_{term}_coef", 0.0))
            for term in _TERMS
        })
        for prefix, attribute in _CRITICS
        if any(float(getattr(cfg, f"{prefix}_{term}_coef", 0.0)) > 0.0 for term in _TERMS)
    ]
    needs_noise = any(coefficients["mpr_coef"] > 0.0 for _, _, coefficients in selected)
    needs_next = any(coefficients["vfc_coef"] > 0.0 for _, _, coefficients in selected)
    needs_probe = any(coefficients["curvature_coef"] > 0.0 for _, _, coefficients in selected)
    inputs = prepare_pave_inputs(
        model, obs, action, seed=cfg.seed, update_number=update_number,
        batch_size=getattr(cfg, "outer_pave_batch_size", 64),
        obs_noise_std=getattr(cfg, "outer_pave_obs_noise_std", 0.01),
        needs_noise=needs_noise, needs_next=needs_next, needs_probe=needs_probe,
    )

    total = action.new_zeros(())
    metrics = {}
    for prefix, critic, coefficients in selected:
        loss, critic_metrics = critic_pave_loss(
            critic, model.q_backend.decode, inputs["z"], inputs["action"],
            noisy_z=inputs["noisy_z"], next_z=inputs["next_z"], probe=inputs["probe"],
            curvature_margin=float(getattr(cfg, "outer_pave_curvature_margin", 1.0)),
            **coefficients,
        )
        total = total + loss
        metrics.update({f"{prefix}_{key}": value for key, value in critic_metrics.items()})
        metrics[f"{prefix}_sample_count"] = action.new_tensor(inputs["action"].shape[0])
    return total, metrics
