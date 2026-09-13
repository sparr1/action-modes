"""Entropy-only actor experiments against one immutable root-local fitted Q.

This diagnostic deliberately separates one H1 collection/critic-fit phase from
actor optimization. It never runs a normal controller action or changes the
live inner engine. Actor minibatches draw replay states with replacement and
fresh policy noise; for H1 all those states are the same encoded root.

The callback borrows a read-only fitted critic and root tensor, and receives an
immutable actor snapshot. It may perform synchronous external measurements;
its time and random draws are excluded from optimization. Snapshot diagnostics
use eval mode, a private paired policy sample and Q pair, and no Q-scale update.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import io
import math
import random
import time

import numpy as np
import torch

from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.common.inner_utils import InnerRNG
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.inner_trace import FrozenActorSnapshot


MODES = {"off", "squashed", "tdmpc2_scaled", "gaussian"}


def entropy_arms(model, *, include_gaussian=False):
    """Resolve the checkpoint's actual alpha and preserve its literal recipe.

    TD-MPC2's statistic is approximately D times joint Gaussian entropy, so
    matching its Gaussian coefficient requires D*alpha for true squashed
    entropy. An ordinary squashed recipe already matches and becomes an alias.
    ``include_gaussian`` is a panel-level choice; per-root preflight flags never
    silently create different arm coverage at different roots.
    """
    mode = str(model.cfg.outer_actor_entropy_mode)
    if mode not in {"squashed", "tdmpc2_scaled"}:
        raise ValueError(f"Unsupported saved outer entropy mode: {mode!r}")
    alpha = float(model.agent.alpha.detach().cpu().item())
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Saved outer alpha must be finite and nonnegative.")
    dimension = int(model.cfg.action_dim)
    matched = alpha * (dimension if mode == "tdmpc2_scaled" else 1)
    arms = [dict(name="off", mode="off", alpha=0.),
            dict(name="prior_recipe", mode=mode, alpha=alpha),
            dict(name="squashed_matched", mode="squashed", alpha=matched)]
    aliases = {}
    if mode == "squashed":
        arms[-1]["alias_of"] = "prior_recipe"
        aliases["squashed_matched"] = "prior_recipe"
    if include_gaussian:
        arms.append(dict(name="gaussian_control", mode="gaussian", alpha=matched))
    return arms, dict(saved_mode=mode, saved_alpha=alpha, action_dim=dimension,
                      matched_gaussian_coefficient=matched, aliases=aliases,
                      gaussian_control_included=bool(include_gaussian),
                      native_recipe="literal_tdmpc2_scaled" if mode == "tdmpc2_scaled" else "stable_squashed",
                      preflight="Each root reports native versus D*Gaussian value and parameter-gradient error.")


def _digest(value):
    # Reuse the evaluator's canonical tensor/state encoding, not pickle storage IDs.
    from evaluate_ambi_checkpoint import _digest_update
    digest = hashlib.sha256()
    _digest_update(digest, value)
    return digest.hexdigest()


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _preserve_globals(*modules):
    """Also isolate callback failures and heterogeneous child module modes."""
    modes = {part: part.training for module in modules for part in module.modules()}
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        for part, mode in modes.items():
            part.training = mode
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _bounds(cfg):
    return {name: getattr(cfg, f"inner_{name}")
            for name in ("log_std_mapping", "log_std_min", "log_std_max")}


def _entropy(info, noise, mode):
    if mode == "tdmpc2_scaled":
        return info["scaled_entropy"]
    if mode == "squashed":
        return info["entropy"]
    if mode == "gaussian":
        return -td_math.gaussian_logprob(noise, info["log_std"])
    if mode == "off":
        return info["entropy"] * 0.
    raise ValueError(f"Unknown entropy mode: {mode!r}")


def _losses(world, cfg, actor, critic, z, noise, arm, scale):
    action, info = world.pi(z, policy=actor, noise=noise,
                            include_scaled_entropy=True, **_bounds(cfg))
    # Preserve ordinary actor all-head -> configured pair reduction draw order.
    q_all = world.Q(z, action, qs=critic, detach=True, reduction="all")
    q = world.q_backend.reduce(q_all, cfg.inner_q_actor_reduction)
    q_scaled = q if scale is None else q / scale
    entropy = _entropy(info, noise, arm["mode"])
    return -q_scaled.mean(), -float(arm["alpha"]) * entropy.mean(), q, action, info


def _grads(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True,
                                 allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, values)])


def _finite_numbers(metrics):
    result = {key: float(value.detach().cpu().item()) if torch.is_tensor(value)
              else float(value) for key, value in metrics.items()}
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite entropy diagnostic; refusing incomplete scientific rows.")
    return result


def _preflight(info, noise, parameters):
    native = info["scaled_entropy"].mean()
    gaussian = -td_math.gaussian_logprob(noise, info["log_std"]).mean() * noise.shape[-1]
    native_grad, gaussian_grad = _grads(native, parameters), _grads(gaussian, parameters)
    error = _finite_numbers(dict(
        native_gaussian_value_abs_error=(native - gaussian).abs(),
        native_gaussian_value_relative_error=(native - gaussian).abs() / gaussian.abs().clamp_min(1.),
        native_gaussian_gradient_l2_error=(native_grad - gaussian_grad).norm(),
        native_gaussian_gradient_relative_error=(native_grad - gaussian_grad).norm() / gaussian_grad.norm().clamp_min(1e-8),
    ))
    error["needs_control"] = bool(error["native_gaussian_value_relative_error"] > 1e-5
                                  or error["native_gaussian_gradient_relative_error"] > 1e-5)
    return error


def _probe(world, cfg, actor, critic, root_z, arm, scale, *, seed, count,
           initial_parameters, initial_mean, initial_pre_tanh_mean, preflight=False):
    parameters = tuple(p for p in actor.parameters() if p.requires_grad)
    rng = InnerRNG(int(seed), root_z.device)
    with _preserve_globals(world, actor, critic), rng.fork("diagnostics") as generator:
        world.eval()
        actor.eval()
        critic.eval()
        z = root_z.expand(count, -1)
        noise = torch.randn((count, int(cfg.action_dim)), device=z.device,
                            dtype=z.dtype, generator=generator)
        q_loss, entropy_loss, q, action, info = _losses(
            world, cfg, actor, critic, z, noise, arm, scale)
        q_gradient = _grads(q_loss, parameters)
        entropy_gradient = _grads(entropy_loss, parameters)
        q_norm, entropy_norm = q_gradient.norm(), entropy_gradient.norm()
        dot = torch.dot(q_gradient, entropy_gradient)
        q_pre, q_std = torch.autograd.grad(q_loss, (info["pre_tanh_mean"], info["log_std"]),
                                          retain_graph=True, allow_unused=True)
        e_pre, e_std = torch.autograd.grad(entropy_loss, (info["pre_tanh_mean"], info["log_std"]),
                                          retain_graph=True, allow_unused=True)
        def norm(value):
            return root_z.new_zeros(()) if value is None else value.norm()
        parameter_vector = torch.cat([p.reshape(-1) for p in parameters])
        metrics = dict(
            true_entropy=info["entropy"].mean(),
            gaussian_entropy=-td_math.gaussian_logprob(noise, info["log_std"]).mean(),
            native_entropy=info["scaled_entropy"].mean(),
            entropy_objective=_entropy(info, noise, arm["mode"]).mean(),
            entropy_bonus=-entropy_loss, q_loss=q_loss, entropy_loss=entropy_loss,
            actor_loss=q_loss + entropy_loss, fitted_q_mean=q.mean(),
            fitted_q_scaled_mean=-q_loss,
            sampled_exact_saturation=(action.abs() == 1).float().mean(),
            sampled_near_saturation=(action.abs() >= .99).float().mean(),
            mean_exact_saturation=(info["mean"].abs() == 1).float().mean(),
            mean_near_saturation=(info["mean"].abs() >= .99).float().mean(),
            sampled_action_abs_mean=action.abs().mean(),
            mean_action_abs_mean=info["mean"].abs().mean(),
            pre_tanh_mean_abs=info["pre_tanh_mean"].abs().mean(),
            pre_tanh_mean_abs_max=info["pre_tanh_mean"].abs().max(),
            pre_tanh_action_abs=info["pre_tanh_action"].abs().mean(),
            pre_tanh_action_abs_max=info["pre_tanh_action"].abs().max(),
            log_std_mean=info["log_std"].mean(), log_std_min=info["log_std"].min(),
            log_std_max=info["log_std"].max(), std_mean=info["log_std"].exp().mean(),
            jacobian_mean=(1. - action.square()).mean(),
            mean_jacobian_mean=(1. - info["mean"].square()).mean(),
            log_jacobian_mean=td_math.tanh_log_abs_det_jacobian(info["pre_tanh_action"]).mean(),
            q_gradient_norm=q_norm, entropy_gradient_norm=entropy_norm,
            total_gradient_norm=(q_gradient + entropy_gradient).norm(),
            gradient_dot=dot, gradient_cosine=dot / (q_norm * entropy_norm).clamp_min(1e-20),
            q_pre_tanh_mean_gradient_norm=norm(q_pre),
            entropy_pre_tanh_mean_gradient_norm=norm(e_pre),
            q_log_std_gradient_norm=norm(q_std), entropy_log_std_gradient_norm=norm(e_std),
            parameter_displacement_l2=(parameter_vector - initial_parameters).norm(),
            mean_action_displacement_l2=(info["mean"][0] - initial_mean).norm(),
            pre_tanh_mean_displacement_l2=(info["pre_tanh_mean"][0] - initial_pre_tanh_mean).norm(),
        )
        check = _preflight(info, noise, parameters) if preflight else None
        return _finite_numbers(metrics), check


def _snapshot(actor, cfg, updates, critic_updates):
    stream = io.BytesIO()
    torch.save(copy.deepcopy(actor).cpu().eval().requires_grad_(False), stream)
    return FrozenActorSnapshot(round_index=int(updates), actor_updates=int(updates),
        critic_updates=int(critic_updates), temperature_updates=0, inner=True,
        bounds=tuple(sorted(_bounds(cfg).items())), payload=stream.getvalue())


def _validate(model, arms, snapshot_updates, probe_rollouts, fit_engine):
    cfg = model.cfg
    for field, value in dict(inner_operator="sac", inner_rollout_horizon=1,
                             inner_finite_horizon=True, inner_actor_adaptation="clone",
                             inner_critic_adaptation="clone", inner_actor_initialization="prior",
                             inner_critic_initialization="prior", inner_temperature_mode="fixed",
                             inner_replay_sampling="with_replacement",
                             inner_actor_loss_scale_update="per_action",
                             inner_behavior_action="policy_sample", inner_behavior_std_scale=1.,
                             inner_explorer_mode="none", inner_outer_replay_fraction=0.).items():
        if getattr(cfg, field) != value:
            raise ValueError(f"Entropy probe requires {field}={value!r}.")
    if float(getattr(cfg, "inner_outer_policy_kl_coef", 0.)) != 0.:
        raise ValueError("Entropy probe requires zero auxiliary outer-policy KL.")
    for name, value in _bounds(cfg).items():
        if value is not None and value != getattr(cfg, name):
            raise ValueError("Entropy probe must inherit checkpoint policy mapping and bounds.")
    if float(cfg.inner_temperature) != 0.:
        raise ValueError("Shared fit must use fixed inner_temperature=0.")
    if bool(getattr(cfg, "compile", False)):
        raise ValueError("Entropy probe requires explicit compile=False.")
    if int(cfg.inner_rounds) != 1 or int(cfg.inner_rollouts_per_round) < 1:
        raise ValueError("Entropy probe requires one nonempty collection round.")
    if any(str(getattr(cfg, f"inner_{component}_scope")) != "action"
           for component in ("actor", "critic", "temperature", "replay", "actor_optimizer",
                             "critic_optimizer", "temperature_optimizer")):
        raise ValueError("Entropy probe requires fully action-local fitting.")
    if fit_engine is not None and (fit_engine.agent is not model.agent
                                  or fit_engine is model.agent.inner_engine):
        raise ValueError("fit_engine must be dedicated to this loaded model, never its live inner engine.")
    updates = tuple(snapshot_updates)
    if (not updates or updates[0] != 0 or tuple(sorted(set(updates))) != updates
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in updates)):
        raise ValueError("snapshot_updates must be increasing nonnegative integers starting at zero.")
    if isinstance(probe_rollouts, bool) or not isinstance(probe_rollouts, int) or probe_rollouts < 1:
        raise ValueError("probe_rollouts must be positive.")
    by_name = {}
    for arm in arms:
        if not isinstance(arm.get("name"), str) or not arm["name"] or arm["name"] in by_name:
            raise ValueError("Entropy arm names must be unique nonempty strings.")
        if arm.get("mode") not in MODES or not math.isfinite(float(arm.get("alpha", -1))) or float(arm["alpha"]) < 0:
            raise ValueError("Each entropy arm needs a supported mode and finite nonnegative alpha.")
        if arm["mode"] == "off" and float(arm["alpha"]) != 0.:
            raise ValueError("The off arm must have alpha zero.")
        by_name[arm["name"]] = arm
    if not by_name:
        raise ValueError("At least one entropy arm is required.")
    for arm in arms:
        if "alias_of" in arm:
            target = by_name.get(arm["alias_of"])
            if (target is None or "alias_of" in target or target is arm
                    or (target["mode"], target["alpha"]) != (arm["mode"], arm["alpha"])):
                raise ValueError("Aliases must point directly to an identical canonical objective.")
    return updates


@torch.enable_grad()
def run_root_entropy_probe(model, observation, *, arms, fit_seed, actor_seed, probe_seed,
                           snapshot_updates=(0, 1, 4, 16), probe_rollouts=32,
                           on_snapshot=None, fit_engine=None):
    """Collect once, fit once, then independently update paired actor clones.

    Counts N/C/B are taken from the resolved H1 configuration (production:
    128/32/256); A is the last requested snapshot count. Old squashed objective
    aliases emit their own callback identities using the same saved actor.
    Diagnostics report gradients of the two loss terms before clipping, not a
    fictitious additive attribution of Adam's actual parameter displacement.
    """
    from evaluate_ambi_checkpoint import _outer_state_digest
    updates = _validate(model, arms, snapshot_updates, probe_rollouts, fit_engine)
    for seed in (fit_seed, actor_seed, probe_seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Diagnostic seeds must be nonnegative integers.")
    started = time.perf_counter()
    cfg, world, device = model.cfg, model.agent.model, model.agent.device
    engine = fit_engine if fit_engine is not None else InnerImprovementEngine(model.agent)
    before_outer = _outer_state_digest(model)
    timing = dict(fit_seconds=0., actor_optimization_seconds=0., model_probe_seconds=0.,
                  serialization_seconds=0., callback_seconds=0.)
    rows, initial_check = [], None
    result = None
    with _preserve_globals(world):
        try:
            world.eval()
            engine.reset_for_evaluation(int(fit_seed), reuse_action_pool=True)
            engine._collect_diagnostics = False
            _sync(device)
            phase_start = time.perf_counter()
            with engine.rng.action_fork():
                with engine.rng.fork("observation"), torch.no_grad():
                    obs = torch.as_tensor(observation, device=device, dtype=torch.float32)
                    root_z = world.encode(obs.unsqueeze(0)).detach()
                with engine.rng.fork("initialization"):
                    engine._prepare_workspace(t0=True)
                if (_digest(engine.state.actor.state_dict()) != _digest(world._pi.state_dict())
                        or _digest(engine.state.critic.state_dict()) != _digest(world._Qs.state_dict())):
                    raise RuntimeError("Shared fit must initialize both networks from the saved prior.")
                scale = (model.agent.actor_loss_scale.detach().clone()
                         if engine._sac_actor_loss_scale_enabled else None)
                if scale is not None and (not torch.isfinite(scale).all() or (scale <= 0).any()):
                    raise ValueError("Saved Q scale must be finite and positive.")
                collection = engine._collect_round(root_z)
                critic_updates = engine._resolved_primary_round_count("critic")
                if critic_updates < 1:
                    raise ValueError("Entropy probe requires a nonempty shared critic fit.")
                engine._run_component_update_counts(critic_count=critic_updates, actor_count=0,
                                                    actor_loss_scale=scale)
                if engine.state.actor_steps != 0 or engine.state.critic_steps != critic_updates:
                    raise RuntimeError("Shared fit executed unexpected optimizer counts.")
            _sync(device)
            timing["fit_seconds"] = time.perf_counter() - phase_start
            initial_actor = copy.deepcopy(engine.state.actor)
            critic = copy.deepcopy(engine.state.critic).requires_grad_(False)
            replay = engine.state.replay
            replay_hash, critic_hash = _digest(replay.state_dict()), _digest(critic.state_dict())
            initial_hash = _digest(initial_actor.state_dict())
            if initial_hash != _digest(world._pi.state_dict()):
                raise RuntimeError("Actual actor initialization must equal the saved outer policy.")
            root_hash = _digest(root_z)
            if not torch.equal(replay.z[:replay.size], root_z.expand(replay.size, -1)):
                raise RuntimeError("H1 replay must contain only the encoded root as current state.")
            if not bool((replay.horizon_end[:replay.size] == 1).all()):
                raise RuntimeError("Every H1 fit target must hand off to the frozen outer bootstrap.")
            with torch.no_grad(), _preserve_globals(initial_actor):
                initial_actor.eval()
                _, initial_info = world.pi(root_z, policy=initial_actor,
                    noise=root_z.new_zeros(1, int(cfg.action_dim)), **_bounds(cfg))
            initial_parameters = torch.cat([p.detach().reshape(-1) for p in initial_actor.parameters() if p.requires_grad])
            canonical = [arm for arm in arms if "alias_of" not in arm]
            for arm in canonical:
                actor = copy.deepcopy(initial_actor)
                optimizer = engine._new_optimizer(actor, "actor")
                parameters = tuple(p for p in actor.parameters() if p.requires_grad)
                rng = InnerRNG(int(actor_seed), device)
                # Exactly the standard actor-phase batched replay-index draw.
                replay_indices = torch.randint(replay.size, (updates[-1], int(cfg.inner_batch_size)),
                    device=device, generator=rng.generator("replay"))
                aliases = [item for item in arms if item.get("alias_of") == arm["name"]]
                with rng.action_fork():
                    for update in range(updates[-1] + 1):
                        if update in updates:
                            _sync(device)
                            probe_start = time.perf_counter()
                            metrics, check = _probe(world, cfg, actor, critic, root_z, arm, scale,
                                seed=probe_seed, count=probe_rollouts, initial_parameters=initial_parameters,
                                initial_mean=initial_info["mean"][0],
                                initial_pre_tanh_mean=initial_info["pre_tanh_mean"][0],
                                preflight=initial_check is None)
                            if check is not None:
                                initial_check = check
                                initial_check["relevant_to_saved_recipe"] = cfg.outer_actor_entropy_mode == "tdmpc2_scaled"
                                initial_check["needs_control"] = bool(check["needs_control"] and initial_check["relevant_to_saved_recipe"])
                            _sync(device)
                            timing["model_probe_seconds"] += time.perf_counter() - probe_start
                            serialization_start = time.perf_counter()
                            snapshot = _snapshot(actor, cfg, update, critic_updates)
                            actor_hash = _digest(actor.state_dict())
                            timing["serialization_seconds"] += time.perf_counter() - serialization_start
                            for identity in [arm, *aliases]:
                                metadata = dict(arm=identity["name"], mode=identity["mode"], alpha=float(identity["alpha"]),
                                    actor_updates=update, critic_updates=critic_updates,
                                    actor_sha256=snapshot.sha256, actor_state_sha256=actor_hash,
                                    critic_state_sha256=critic_hash, replay_sha256=replay_hash,
                                    metrics=copy.deepcopy(metrics), probe_rollouts=probe_rollouts,
                                    probe_mode="eval_fixed_paired_noise_and_q_pair")
                                if "alias_of" in identity:
                                    metadata["alias_of"] = identity["alias_of"]
                                rows.append(copy.deepcopy(metadata))
                                if on_snapshot is not None:
                                    callback_start = time.perf_counter()
                                    with _preserve_globals(world, critic):
                                        on_snapshot(copy.deepcopy(metadata), snapshot, critic, root_z)
                                    _sync(device)
                                    timing["callback_seconds"] += time.perf_counter() - callback_start
                                    if _digest(critic.state_dict()) != critic_hash or _digest(root_z) != root_hash:
                                        raise RuntimeError("Snapshot callback mutated the shared fitted critic or root.")
                            del snapshot
                        if update == updates[-1]:
                            break
                        _sync(device)
                        actor_start = time.perf_counter()
                        batch = replay.sample(int(cfg.inner_batch_size), replacement=True,
                            include_ids=False, indices=replay_indices[update])
                        with rng.fork("gradient_policy") as generator:
                            noise = torch.randn((*batch["z"].shape[:-1], int(cfg.action_dim)),
                                device=device, dtype=batch["z"].dtype, generator=generator)
                            q_loss, entropy_loss, *_ = _losses(world, cfg, actor, critic,
                                                              batch["z"], noise, arm, scale)
                            loss = q_loss + entropy_loss
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError("Nonfinite actor objective.")
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(cfg.inner_actor_grad_clip_norm))
                            if not bool(torch.isfinite(grad_norm)):
                                raise FloatingPointError("Nonfinite actor gradient.")
                            optimizer.step()
                        _sync(device)
                        timing["actor_optimization_seconds"] += time.perf_counter() - actor_start
                if _digest(critic.state_dict()) != critic_hash or _digest(replay.state_dict()) != replay_hash:
                    raise RuntimeError("Actor optimization mutated the shared critic or replay.")
                del actor, optimizer
            result = dict(schema="ambi-fixed-critic-entropy-probe", version=1, arms=copy.deepcopy(arms),
                snapshots=rows, preflight=initial_check, fit_seed=fit_seed, actor_seed=actor_seed, probe_seed=probe_seed,
                collection_transitions=int(collection["transition_count"]), critic_updates=critic_updates,
                actor_updates_per_arm=updates[-1], optimizer_arms=len(canonical),
                actor_optimizer_updates=len(canonical) * updates[-1], inner_batch_size=int(cfg.inner_batch_size),
                replay_size=int(replay.size), replay_sha256=replay_hash, critic_state_sha256=critic_hash,
                initial_actor_state_sha256=initial_hash, root_sha256=root_hash,
                actor_q_reduction=str(cfg.inner_q_actor_reduction), terminal_q_reduction=str(cfg.mppi_terminal_q_reduction),
                q_scale=None if scale is None else float(scale.cpu().item()),
                q_scale_source="saved_checkpoint" if scale is not None else "unscaled", policy_bounds=_bounds(cfg),
                timing=timing, optimization_seconds=timing["fit_seconds"] + timing["actor_optimization_seconds"])
        finally:
            # Return private fitting allocations to its pool, including on failure.
            engine._clear_expired(t0=False, include_action=True)
            if _outer_state_digest(model) != before_outer:
                raise RuntimeError("Entropy diagnostic changed frozen outer weights or optimizer state.")
    result["total_seconds"] = time.perf_counter() - started
    return result
