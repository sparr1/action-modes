"""Periodic action-gradient telemetry, independent of training regularizers."""

import torch

from .pave import _geometry_mode, _view_gradients, prepare_pave_inputs


_GRADIENT_FLOOR = 1e-8


def geometry_diagnostics_due(cfg, update_number):
    return (bool(getattr(cfg, "critic_geometry_diagnostics", False))
            and update_number > 0
            and update_number % int(cfg.critic_geometry_every) == 0)


def gradient_consistency_metrics(original, perturbed, term):
    """Match training reductions; report direction only for nonzero vectors.

    Gradients have shape [heads, rows, action coordinates]. Undefined relative
    or angular measurements are NaN and accompanied by explicit validity
    metrics; the training logger omits nonfinite values instead of fabricating
    perfect agreement for a flat critic.
    """
    original, perturbed = original.detach(), perturbed.detach()
    difference_squared = (original - perturbed).square()
    rms = original.square().mean().sqrt()
    delta_rms = difference_squared.mean().sqrt()
    original_norm = original.norm(dim=-1)
    perturbed_norm = perturbed.norm(dim=-1)
    valid = (original_norm > _GRADIENT_FLOOR) & (perturbed_norm > _GRADIENT_FLOOR)
    cosine = ((original * perturbed).sum(dim=-1)
              / (original_norm * perturbed_norm).clamp_min(_GRADIENT_FLOOR ** 2)).clamp(-1, 1)
    count = valid.sum()
    undefined = rms.new_full((), float("nan"))
    return {
        f"{term}_loss": difference_squared.mean(dim=(1, 2)).sum(),
        f"{term}_gradient_change_rms": delta_rms,
        f"{term}_perturbed_action_gradient_rms": perturbed.square().mean().sqrt(),
        f"{term}_relative_gradient_change": torch.where(
            rms > _GRADIENT_FLOOR, delta_rms / rms.clamp_min(_GRADIENT_FLOOR), undefined,
        ),
        f"{term}_cosine_similarity": torch.where(
            count > 0, torch.where(valid, cosine, 0.).sum() / count.clamp_min(1), undefined,
        ),
        f"{term}_cosine_valid_fraction": valid.float().mean(),
    }


def critic_geometry_diagnostics(model, obs, action, cfg, update_number):
    """Measure both perturbations on the same real replay rows for all critics.

    This is a separate pre-update measurement: no penalty is added to the
    objective, no parameter gradients are accumulated, and no higher-order
    graph is constructed. Its seed/scales/cap do not depend on enabled losses.
    The changing replay batch is not a fixed cross-run evaluation bank.
    """
    if not geometry_diagnostics_due(cfg, update_number):
        return {}
    devices = [action.device.index] if action.is_cuda else []
    with torch.random.fork_rng(devices=devices), _geometry_mode(model), torch.enable_grad():
        inputs = prepare_pave_inputs(
            model, obs, action, seed=cfg.critic_geometry_seed, update_number=update_number,
            batch_size=cfg.critic_geometry_batch_size,
            obs_noise_std=cfg.critic_geometry_obs_noise_std,
            action_noise_std=cfg.critic_geometry_action_noise_std,
            needs_noise=True, needs_action_noise=True, needs_next=False, needs_probe=False,
        )
        metrics = {
            "critic_geometry_sample_count": action.new_tensor(inputs["action"].shape[0]),
            "critic_geometry_update": action.new_tensor(update_number),
            "critic_geometry_action_perturbation_rms": (
                inputs["noisy_action"] - inputs["action"]
            ).square().mean().sqrt(),
            "critic_geometry_action_boundary_fraction": (inputs["noisy_action"].abs() == 1).float().mean(),
        }
        for name, attribute in (("primary", "_Qs"), ("aux_return", "_aux_return_Qs")):
            critic = getattr(model, attribute, None)
            if critic is None:
                continue
            (clean, state_noisy, action_noisy), _ = _view_gradients(
                critic, model.q_backend.decode,
                [inputs["z"], inputs["noisy_z"], inputs["z"]], inputs["action"],
                noisy_action=inputs["noisy_action"], curvature=False, create_graph=False,
            )
            rms = clean.square().mean().sqrt()
            values = {
                "action_gradient_rms": rms,
                "gradient_scale_valid": (rms > _GRADIENT_FLOOR).float(),
                **gradient_consistency_metrics(clean, state_noisy, "mpr"),
                **gradient_consistency_metrics(clean, action_noisy, "action_smoothness"),
            }
            metrics.update({f"critic_geometry_{name}_{key}": value.detach()
                            for key, value in values.items()})
    return metrics
