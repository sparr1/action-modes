"""PAVE geometry losses on real observations for the outer AMBI critics.

Reference: https://github.com/AIRLABkhu/PAVE (SAC implementation). The
regularizers sum independent critic heads, use recorded actions in all three
state views, and leave the ordinary Bellman objective and Q decoder intact.
AMBI additionally supports action-gradient consistency under action noise.
"""

from contextlib import contextmanager

import torch

from .common import ensemble_vmap
from .common.layers import Ensemble, NormedLinear
from .common.q_representation import QRepresentation


_TERMS = ("mpr", "vfc", "curvature", "action_smoothness")
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
    # Keep higher derivatives outside compiled ensemble wrappers. Independent
    # input/action coordinates prevent opposing critic gradients cancelling.
    if isinstance(critic, Ensemble):
        expanded = latents.unsqueeze(0).expand(actions.shape[0], *latents.shape)
        return critic.forward_heads_eager(torch.cat((expanded, actions), dim=-1))
    return torch.stack([
        member(torch.cat((latents, actions[index]), dim=-1))
        for index, member in enumerate(critic)
    ], dim=0)


def _view_batch_spec(critic, decode):
    """Only fuse known row-independent critics and decoders.

    Custom heads, hooks, forwards or decoders can couple rows or observe call
    boundaries. Preserve their existing per-view execution, including errors.
    Check this inside geometry mode, after dropout has been disabled.
    """
    if (
        type(critic) is Ensemble
        and getattr(critic.forward_heads_eager, "__func__", None) is Ensemble.forward_heads_eager
        and type(getattr(decode, "__self__", None)) is QRepresentation
        and getattr(decode, "__func__", None) is QRepresentation.decode
    ):
        return ensemble_vmap.grouped_vmap_spec((critic,), NormedLinear)
    return None


def _view_gradients(critic, decode, views, action, *, curvature, noisy_action=None):
    """Share parameter stacks without making curvature traverse unused views.

    When supplied, ``noisy_action`` belongs to the last view only; preceding
    views keep the recorded action. Each gradient uses its own coordinates.
    """
    spec = _view_batch_spec(critic, decode) if len(views) > 1 else None
    parameters = ensemble_vmap.stack_parameters(critic.modules_list) if spec is not None else None

    def gradient(latents, actions):
        latents = latents.detach()
        if spec is None:
            predictions = _head_predictions(critic, latents, actions)
        else:
            expanded = latents.unsqueeze(0).expand(len(critic), *latents.shape)
            predictions = ensemble_vmap._tensor_mlp_vmap(
                parameters, torch.cat((expanded, actions), dim=-1), spec, True,
            )
        values = decode(predictions)
        if values.shape != (*actions.shape[:-1], 1):
            raise ValueError("PAVE requires a single decoded scalar per critic head and row.")
        return _action_gradient(values, actions)

    if spec is None:
        # Preserve custom call boundaries and shared action coordinates exactly.
        actions = action.detach().unsqueeze(0).expand(len(critic), -1, -1).clone()
        actions.requires_grad_(True)
        view_actions = [actions] * len(views)
        if noisy_action is not None:
            # Differentiate Q at the perturbed point, never through the noise
            # or clamp operation that produced it (especially at +/-1).
            perturbed = noisy_action.detach().unsqueeze(0).expand(len(critic), -1, -1).clone()
            view_actions[-1] = perturbed.requires_grad_(True)
        return tuple(gradient(view, coordinates) for view, coordinates in zip(views, view_actions)), actions

    # With curvature, keep the current-state graph small. Merely slicing its
    # gradient out of a fully fused forward still makes higher derivatives
    # traverse all views (including zero cotangents), wasting compute/memory.
    view_actions = [action] * len(views)
    if noisy_action is not None:
        view_actions[-1] = noisy_action
    pairs = list(zip(views, view_actions))
    groups = (pairs[:1], pairs[1:]) if curvature else (pairs,)
    gradients = []
    current_actions = None
    for group in groups:
        # Every head, view and transition gets independent coordinates. Never
        # differentiate expanded aliases of the same underlying action leaf.
        actions = torch.cat([coordinates.detach() for _, coordinates in group], dim=0)
        actions = actions.unsqueeze(0).repeat(len(critic), 1, 1)
        actions.requires_grad_(True)
        if current_actions is None:
            current_actions = actions
        latents = torch.cat([view.detach() for view, _ in group], dim=0)
        gradients.extend(gradient(latents, actions).split(action.shape[0], dim=1))
    return tuple(gradients), current_actions


def critic_pave_loss(
    critic, decode, z, action, *, noisy_z=None, next_z=None, probe=None, noisy_action=None,
    mpr_coef=0.0, vfc_coef=0.0, curvature_coef=0.0, curvature_margin=1.0,
    action_smoothness_coef=0.0, return_components=False,
):
    """Return weighted loss and detached metrics for one independent ensemble.

    Inputs are flat transition batches. ``decode`` receives every head at once
    and must return [heads, rows, 1]. Supported dense vmap critics concatenate
    state/action views along the row axis, stacking parameters only once. Curvature
    keeps a separate current-state graph; without it, all views share a forward.
    Independent action coordinates preserve each head/view's action gradient.
    Encoder graphs and replay actions are deliberately detached here as well as
    at the outer call site. Only the selected online critic is differentiated.
    """
    coefficients = dict(mpr=mpr_coef, vfc=vfc_coef, curvature=curvature_coef,
                        action_smoothness=action_smoothness_coef)
    if not any(coefficient > 0.0 for coefficient in coefficients.values()):
        result = (action.new_zeros(()), {})
        return (*result, {}) if return_components else result
    if mpr_coef > 0.0 and noisy_z is None:
        raise ValueError("PAVE MPR requires perturbed-observation latents.")
    if vfc_coef > 0.0 and next_z is None:
        raise ValueError("PAVE VFC requires successor-observation latents.")
    if curvature_coef > 0.0 and (probe is None or probe.shape != action.shape):
        raise ValueError("PAVE curvature requires one action-shaped probe per transition.")
    if action_smoothness_coef > 0.0 and (noisy_action is None or noisy_action.shape != action.shape):
        raise ValueError("Action smoothness requires one perturbed action per transition.")

    losses = {}
    metrics = {}
    with _geometry_mode(critic):
        views = [z]
        if mpr_coef > 0.0:
            views.append(noisy_z)
        if vfc_coef > 0.0:
            views.append(next_z)
        if action_smoothness_coef > 0.0:
            views.append(z)
        gradients, actions = _view_gradients(
            critic, decode, views, action, curvature=curvature_coef > 0.0,
            noisy_action=noisy_action if action_smoothness_coef > 0.0 else None,
        )
        original, *other_gradients = gradients
        other_gradients = iter(other_gradients)
        if return_components:
            metrics["action_gradient_rms"] = original.detach().square().mean().sqrt()
        if mpr_coef > 0.0:
            losses["mpr"] = (original - next(other_gradients)).square().mean(dim=(1, 2)).sum()
        if vfc_coef > 0.0:
            losses["vfc"] = (original - next(other_gradients)).square().mean(dim=(1, 2)).sum()
        if curvature_coef > 0.0:
            vectors = probe.detach().unsqueeze(0)
            hvp = _action_gradient((original * vectors).sum(), actions)
            margin_error = (hvp * vectors).sum(dim=-1) + curvature_margin
            losses["curvature"] = margin_error.relu().mean(dim=1).sum()
            metrics["curvature_violation_fraction"] = (margin_error.detach() > 0).float().mean()
        if action_smoothness_coef > 0.0:
            losses["action_smoothness"] = (original - next(other_gradients)).square().mean(dim=(1, 2)).sum()

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
    action_noise_std=0.02, needs_action_noise=False,
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
        noisy_action = None
        if needs_action_noise:
            # Draw last so enabling this extension preserves the existing
            # observation noise and curvature probes on the same sample.
            noise = torch.randn(
                actions.shape, dtype=actions.dtype, device=actions.device, generator=generator,
            ) * float(action_noise_std)
            noisy_action = (actions + noise).clamp(-1., 1.)
    return dict(z=z, action=actions, noisy_z=noisy_z, next_z=next_z, probe=probe,
                noisy_action=noisy_action)


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
    needs_action_noise = any(coefficients["action_smoothness_coef"] > 0.0 for _, _, coefficients in selected)
    inputs = prepare_pave_inputs(
        model, obs, action, seed=cfg.seed, update_number=update_number,
        batch_size=getattr(cfg, "outer_pave_batch_size", 64),
        obs_noise_std=getattr(cfg, "outer_pave_obs_noise_std", 0.01),
        action_noise_std=getattr(cfg, "outer_pave_action_noise_std", 0.02),
        needs_noise=needs_noise, needs_next=needs_next, needs_probe=needs_probe,
        needs_action_noise=needs_action_noise,
    )

    total = action.new_zeros(())
    metrics = {}
    for prefix, critic, coefficients in selected:
        loss, critic_metrics = critic_pave_loss(
            critic, model.q_backend.decode, inputs["z"], inputs["action"],
            noisy_z=inputs["noisy_z"], next_z=inputs["next_z"], probe=inputs["probe"],
            noisy_action=inputs["noisy_action"],
            curvature_margin=float(getattr(cfg, "outer_pave_curvature_margin", 1.0)),
            **coefficients,
        )
        total = total + loss
        metrics.update({f"{prefix}_{key}": value for key, value in critic_metrics.items()})
        metrics[f"{prefix}_sample_count"] = action.new_tensor(inputs["action"].shape[0])
    return total, metrics
