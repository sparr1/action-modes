"""Observational BOOM gradient probes and actual optimizer-step measurements.

Component probes use the current training minibatch and committed S, but their
own reproducible policy/dropout/pair draws. They estimate gradient balance;
they are not a decomposition of the compiled step's exact random realization.
No probe changes parameters, .grad, optimizer state, module mode, or RNG.
"""
from contextlib import contextmanager

import torch

from .boom import boom_replay_likelihood_loss
from .common import math as td_math
from .common.compile_regions import _preserve_host_rng_state


CANDIDATE_COEFFICIENTS = (.000042, .00042, .0042, .042, .42)


def diagnostics_due(cfg, update):
    if not getattr(cfg, "outer_boom_diagnostics", False):
        return False
    every = cfg.outer_boom_diagnostics_early_every if update <= cfg.outer_boom_diagnostics_early_until else cfg.outer_boom_diagnostics_every
    return update == 1 or update % every == 0


@contextmanager
def isolated_probe(agent):
    devices = [agent.device.index if agent.device.index is not None else torch.cuda.current_device()] if agent.device.type == "cuda" else []
    ensemble = agent.model._Qs
    compiled = ensemble._compile_enabled
    try:
        ensemble._compile_enabled = False
        with _preserve_host_rng_state(), torch.random.fork_rng(devices=devices):
            seed = int(agent.cfg.outer_boom_diagnostics_seed) + int(agent.num_updates) + 1
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            yield
    finally:
        ensemble._compile_enabled = compiled


def _gradient(loss, parameters, *, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1)
                      for p, g in zip(parameters, gradients)])


def _norm(vector):
    return vector.double().norm()


def _ratio(numerator, denominator):
    valid = torch.isfinite(numerator) & torch.isfinite(denominator) & (denominator > 1e-12)
    return torch.where(valid, numerator / denominator.clamp_min(1e-12), 0.), valid.double()


def before_step(agent, zs, replay_action, pre_tanh_action, valid):
    """Called after the real backward/clip and before its one Adam step."""
    cfg = agent.cfg
    parameters = tuple(agent.model._pi.parameters())
    scale = agent.actor_loss_scale.detach().clone()
    with isolated_probe(agent), torch.enable_grad():
        action, info = agent.model.pi(zs.detach(), include_scaled_entropy=agent._actor_entropy_mode == "tdmpc2_scaled")
        q = agent.model.q_backend.reduce(
            agent.model.Q(zs.detach(), action, reduction="all", detach=True), cfg.outer_q_actor_reduction,
        )
        def reduce(value):
            return td_math.reduce_temporal_loss(value.mean((1, 2)), cfg.rho, include_terminal=False, legacy_order="vector_mean")
        q_loss = reduce(-q / scale)
        entropy_loss = reduce(-agent.alpha.detach() * info["scaled_entropy"] if agent._actor_entropy_mode == "tdmpc2_scaled"
                              else agent.alpha.detach() * info["log_prob"])
        mask = valid.reshape(*action.shape[:2], 1)
        with torch.no_grad():
            replay_q = agent.model.q_backend.reduce(
                agent.model.Q(zs.detach(), torch.where(mask, replay_action.detach(), 0.), reduction="all", detach=True),
                cfg.outer_q_actor_reduction,
            )
        unit_loss, _ = boom_replay_likelihood_loss(
            info["pre_tanh_mean"], info["log_std"], pre_tanh_action, valid, replay_q, scale,
            rho=cfg.rho, coefficient=1., q_temperature=cfg.outer_boom_q_temperature, apply_gate=False,
        )
        gq = _gradient(q_loss, parameters, retain_graph=True)
        gh = _gradient(entropy_loss, parameters, retain_graph=True)
        ga = _gradient(unit_loss, parameters, retain_graph=False)
        gsac = gq + gh
        nq, nh, na, nsac = map(_norm, (gq, gh, ga, gsac))
        coefficient = float(cfg.outer_boom_coef)
        ratio, ratio_valid = _ratio(coefficient * na, nsac)
        cosine, cosine_valid = _ratio(torch.dot(ga.double(), gsac.double()), na * nsac)
        gate = (scale > 2).double().reshape(())
        metrics = {
            "q_grad_norm": nq, "entropy_grad_norm": nh, "sac_grad_norm": nsac,
            "alignment_unit_grad_norm": na,
            "alignment_weighted_grad_norm": coefficient * na * gate,
            "alignment_prospective_grad_norm": coefficient * na,
            "alignment_over_sac": ratio * gate,
            "prospective_alignment_over_sac": ratio,
            "ratio_valid": ratio_valid, "alignment_sac_cosine": cosine,
            "cosine_valid": cosine_valid, "scale": scale.reshape(()),
            "gate_active": gate, "valid_fraction": valid.double().mean(),
            "update": scale.new_tensor(agent.num_updates + 1),
        }
        for candidate in CANDIDATE_COEFFICIENTS:
            label = format(candidate, '.6f').rstrip('0').replace('.', 'p')
            metrics[f"prospective_ratio_lambda_{label}"] = _ratio(candidate * na, nsac)[0]
        for target, label in ((.1, "0p1"), (.3, "0p3")):
            metrics[f"lambda_for_ratio_{label}"], metrics["coefficient_estimate_valid"] = _ratio(target * nsac, na)
        states = zs.detach().reshape(-1, zs.shape[-1])[:64].clone()
        with torch.no_grad():
            stats = {key: value.detach().clone() for key, value in agent.model.policy_stats(states).items()}
            saved = {"states": states, "stats": stats, "parameters": [p.detach().clone() for p in parameters]}
    return {"boom_diag_" + key: value.detach() for key, value in metrics.items()}, saved


@torch.no_grad()
def after_step(agent, saved):
    """Actual Adam delta and analytic KL(old || new) on the same 64 latents."""
    with isolated_probe(agent):
        old = saved["stats"]
        new = agent.model.policy_stats(saved["states"])
        delta = torch.cat([(p.detach() - before).reshape(-1) for p, before in zip(agent.model._pi.parameters(), saved["parameters"])])
        parameter_norm = _norm(torch.cat([p.reshape(-1) for p in saved["parameters"]]))
        relative, relative_valid = _ratio(_norm(delta), parameter_norm)
        log_ratio = new["log_std"] - old["log_std"]
        kl = (log_ratio + .5 * (torch.exp(-2 * log_ratio)
              + ((old["pre_tanh_mean"] - new["pre_tanh_mean"]) * torch.exp(-new["log_std"])).square() - 1)).sum(-1)
        return {"boom_diag_" + key: value.detach() for key, value in {
            "adam_step_norm": _norm(delta), "adam_step_relative_norm": relative,
            "adam_step_relative_valid": relative_valid, "parameter_norm": parameter_norm,
            "policy_kl_before_after": kl.mean(), "policy_kl_p95": torch.quantile(kl, .95),
            "mean_action_change_rms": (new["mean"] - old["mean"]).square().mean().sqrt(),
            "log_std_change_rms": log_ratio.square().mean().sqrt(),
            "std_before": old["log_std"].exp().mean(), "std_after": new["log_std"].exp().mean(),
        }.items()}
