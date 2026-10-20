"""Paired, frozen-checkpoint diagnostics for inner-SAC entropy drift.

The full engine is observed without replacing its stochastic forwards. Separate
counterfactuals freeze the first imagined replay and post-first-phase return Q;
they are actor interventions, not closed-loop return evaluations.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math
import random

import numpy as np
import torch

from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from utils.ambi_gradient_probe import frozen_parameters


def _json(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hash(value):
    # Reuse the checkpoint evaluator's deterministic nested tensor encoding.
    from evaluate_ambi_checkpoint import _digest_update
    digest = hashlib.sha256()
    _digest_update(digest, value)
    return digest.hexdigest()


def _parameters(actor):
    return [p for p in actor.parameters() if p.requires_grad]


def _vector(actor):
    return torch.cat([p.detach().reshape(-1) for p in _parameters(actor)])


def _gradient(value, actor, *, retain_graph=True):
    params = _parameters(actor)
    if not value.requires_grad:
        return torch.cat([torch.zeros_like(p).reshape(-1) for p in params])
    gradients = torch.autograd.grad(value, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                      for p, g in zip(params, gradients)])


def _cosine(a, b):
    denom = a.norm() * b.norm()
    return (a @ b / denom).item() if denom.item() > 1e-20 else None


def _gradient_summary(q_return, entropy, *, alpha, q_soft=None, delta=None):
    values = {
        "return_q_gradient_norm": q_return.norm(),
        "entropy_gradient_norm": entropy.norm(),
        "entropy_q_gradient_cosine": _cosine(entropy, q_return),
        "weighted_entropy_to_return_gradient_norm": (
            float(alpha) * entropy.norm() / q_return.norm()
            if q_return.norm().item() > 1e-20 else None),
    }
    if q_soft is not None:
        difference = q_soft - q_return
        values.update(soft_q_gradient_norm=q_soft.norm(),
                      soft_return_gradient_cosine=_cosine(q_soft, q_return),
                      entropy_soft_minus_return_cosine=_cosine(entropy, difference),
                      soft_minus_return_gradient_norm=difference.norm())
    if delta is not None:
        values.update(actual_adam_step_norm=delta.norm(),
                      entropy_first_order_change=entropy @ delta,
                      return_q_first_order_change=q_return @ delta,
                      entropy_adam_step_cosine=_cosine(entropy, delta))
    return _json(values)


@contextmanager
def _preserve_environment(model, device):
    """Preserve global RNGs and every module's mode, including mixed modes."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(m, m.training) for m in model.modules()]
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, mode in modes:
            module.training = mode


def _policy_options(cfg):
    return dict(log_std_mapping=cfg.inner_log_std_mapping,
                log_std_min=cfg.inner_log_std_min, log_std_max=cfg.inner_log_std_max)


def entropy_components(info):
    """Joint differential entropy, with analytic Gaussian and sampled Jacobian."""
    gaussian = (info["log_std"] + .5 * math.log(2 * math.pi * math.e)).sum(-1)
    jacobian = td_math.tanh_log_abs_det_jacobian(info["pre_tanh_action"]).squeeze(-1)
    action = info["pre_tanh_action"].tanh()
    return _json({
        "entropy_mc": -info["log_prob"].mean(),
        "entropy_gaussian_sampled": (-info["log_prob"].squeeze(-1) - jacobian).mean(),
        "entropy_gaussian_analytic": gaussian.mean(),
        "entropy_tanh_correction_mc": jacobian.mean(),
        "entropy_analytic_gaussian_plus_mc_jacobian": (gaussian + jacobian).mean(),
        "pre_tanh_mean_abs_mean": info["pre_tanh_mean"].abs().mean(),
        "pre_tanh_std_mean": info["log_std"].exp().mean(),
        "log_std_mean": info["log_std"].mean(),
        "log_std_min": info["log_std"].min(),
        "log_std_max": info["log_std"].max(),
        "action_abs_ge_0p99_fraction": (action.abs() >= .99).float().mean(),
        "action_exact_saturation_fraction": (action.abs() == 1).float().mean(),
    })


def _anchor_readout(engine, actor, anchors, noises):
    values = {}
    mode = actor.training
    try:
        actor.eval()
        with torch.no_grad():
            for name, z in anchors.items():
                _, info = engine.model.pi(z, policy=actor, noise=noises[name],
                                          **_policy_options(engine.cfg))
                values[name] = entropy_components(info)
    finally:
        actor.train(mode)
    return values


def _q(model, critic, z, action, reduction, pair):
    eager = getattr(critic, "_forward_eager", critic)
    return model.Q(z, action, qs=eager, reduction=reduction,
                   pair_indices=pair, trusted_pair_indices=True)


def _anchor_gradients(engine, actor, return_critic, anchors, noises, pairs, alpha):
    """Fixed-anchor eval gradients; distinct from actual training-step gradients."""
    result = {}
    actor_mode = actor.training
    soft = engine.model._Qs
    try:
        actor.eval()
        with torch.enable_grad(), frozen_parameters(engine.model), frozen_parameters(return_critic):
            for name, z in anchors.items():
                action, info = engine.model.pi(z, policy=actor, noise=noises[name],
                                              **_policy_options(engine.cfg))
                entropy = -info["log_prob"].mean()
                qr = _q(engine.model, return_critic, z, action,
                        engine.cfg.inner_q_actor_reduction, pairs[name])
                qs = _q(engine.model, soft, z, action,
                        engine.cfg.inner_q_actor_reduction, pairs[name])
                gr = _gradient(qr.mean(), actor)
                gh = _gradient(entropy, actor)
                gs = _gradient(qs.mean(), actor)
                result[name] = {
                    "gradient_mode": "fixed_anchor_eval_critics_common_actions_and_head_pair",
                    "return_q_mean": qr.mean().item(),
                    "soft_q_mean": qs.mean().item(),
                    "return_q_std_across_anchor_samples": qr.std(unbiased=False).item(),
                    "soft_q_std_across_anchor_samples": qs.std(unbiased=False).item(),
                    **_gradient_summary(gr, gh, alpha=alpha, q_soft=gs),
                }
    finally:
        actor.train(actor_mode)
    return result


def _temperature_state(parameter, optimizer):
    state = optimizer.state.get(parameter, {}) if optimizer is not None else {}
    return _json({"alpha": parameter.detach().exp(),
                  "log_alpha": parameter.detach(),
                  "adam_step": state.get("step", 0),
                  "adam_first_moment": state.get("exp_avg", 0.),
                  "adam_second_moment": state.get("exp_avg_sq", 0.)})


class _ObservedActorRegion:
    """Inspect the actual connected outputs, without a second stochastic forward."""
    def __init__(self, region, engine):
        self.region, self.engine = region, engine

    def __getattr__(self, name):
        return getattr(self.region, name)

    def __call__(self, *args, **kwargs):
        outputs = self.region(*args, **kwargs)
        if self.engine._observe_actor:
            self.engine._actual_gradients = (
                _gradient(outputs[3], self.engine.state.actor),
                _gradient(outputs[1].mean(), self.engine.state.actor),
            )
        return outputs


class _FirstPhaseComplete(Exception):
    pass


class _EntropyProbeEngine(InnerImprovementEngine):
    def __init__(self, agent, *, root_z, seed, anchor_samples, full_solve):
        self.probe_seed = int(seed)
        self.root_z = root_z
        self.anchor_samples = int(anchor_samples)
        self.full_solve = bool(full_solve)
        self.anchors = None
        self.first_critic = None
        self.initial_actor = None
        self.actor_updates = []
        self._observe_actor = False
        self._actual_gradients = None
        super().__init__(agent)
        self.rng = self._new_rng(seed)
        self._compile_regions["actor"] = _ObservedActorRegion(self._compile_regions["actor"], self)

    def _capture_first_replay(self):
        count = int(self.cfg.inner_rollouts_per_round)
        horizon = int(self.cfg.inner_rollout_horizon)
        replay = self.state.replay
        if replay.size != count * horizon:
            raise ValueError("Entropy probe requires one complete dense first-round replay.")
        self.initial_actor = deepcopy(self.state.actor)
        self.first_replay_z = replay.z[:replay.size].detach().clone()
        self.initial_alpha = float(self.alpha.detach())
        generator = torch.Generator(device=self.device).manual_seed(self.probe_seed + 200003)
        self.anchors = {"real_root": self.root_z.expand(self.anchor_samples, -1).clone()}
        # Dense collection is depth-major. Include the terminal latent explicitly.
        for depth in range(1, horizon + 1):
            source = (replay.z[depth * count:(depth + 1) * count] if depth < horizon
                      else replay.next_z[(horizon - 1) * count:horizon * count])
            indices = torch.randint(count, (self.anchor_samples,), generator=generator, device=self.device)
            self.anchors[f"imagined_depth_{depth}"] = source[indices].detach().clone()
        self.anchor_noise = {name: torch.randn((len(z), int(self.cfg.action_dim)),
                                             generator=generator, device=self.device)
                             for name, z in self.anchors.items()}
        self.anchor_pairs = {
            name: self.model.q_backend.sample_pair_indices(self.device, generator=generator)
            for name in self.anchors}
        self.anchors_initial = _anchor_readout(self, self.state.actor, self.anchors, self.anchor_noise)
        self.gradients_before = _anchor_gradients(
            self, self.state.actor, self.state.critic, self.anchors,
            self.anchor_noise, self.anchor_pairs, self.initial_alpha)

    def _run_component_update_counts(self, **kwargs):
        if self.anchors is None:
            self._capture_first_replay()
        return super()._run_component_update_counts(**kwargs)

    def _sac_policy_step(self, batch, **kwargs):
        if self.first_critic is None:
            expected = int(self.cfg.inner_critic_updates_per_round)
            if self.state.critic_steps != expected or self.state.actor_steps:
                raise RuntimeError("First actor phase did not follow the configured first critic phase.")
            self.first_critic = deepcopy(self.state.critic).eval().requires_grad_(False)
            self.gradients_after = _anchor_gradients(
                self, self.state.actor, self.first_critic, self.anchors,
                self.anchor_noise, self.anchor_pairs, self.initial_alpha)
            if not self.full_solve:
                raise _FirstPhaseComplete
        if not kwargs["update_actor"]:
            return super()._sac_policy_step(batch, **kwargs)
        before = _vector(self.state.actor).clone()
        entropy_before = _anchor_readout(self, self.state.actor, self.anchors, self.anchor_noise)
        temperature_before = _temperature_state(self.state.log_alpha, self.state.temperature_optim)
        self._observe_actor = True
        try:
            metrics = super()._sac_policy_step(batch, **kwargs)
        finally:
            self._observe_actor = False
        gr, gh = self._actual_gradients
        self._actual_gradients = None
        delta = _vector(self.state.actor) - before
        self.actor_updates.append({
            "step": int(self.state.actor_steps),
            "alpha_before": float(kwargs["alpha"]),
            "alpha_after": float(self.alpha.detach()),
            "temperature_before": temperature_before,
            "temperature_after": _temperature_state(self.state.log_alpha, self.state.temperature_optim),
            "temperature_entropy_residual": float(metrics["actor_entropy"]) - self._resolved_inner_target_entropy(),
            "metrics": _json(metrics),
            "anchors_before": entropy_before,
            "anchors_after": _anchor_readout(self, self.state.actor, self.anchors, self.anchor_noise),
            "gradient_mode": "actual_actor_training_forward_same_noise_dropout_and_head_pair",
            "actual_update_critic_training": bool(self.state.critic.training),
            "gradient_projections": _gradient_summary(gr, gh, alpha=float(kwargs["alpha"]), delta=delta),
        })
        return metrics


def _paired_tapes(engine, steps):
    generator = torch.Generator(device=engine.device).manual_seed(engine.probe_seed + 300007)
    batch = int(engine.cfg.inner_batch_size)
    indices = torch.randint(len(engine.first_replay_z), (steps, batch),
                            device=engine.device, generator=generator)
    noise = torch.randn((steps, batch, int(engine.cfg.action_dim)),
                        device=engine.device, generator=generator)
    # Scalar twin critics return None because their pair is the whole ensemble.
    pairs = [engine.model.q_backend.sample_pair_indices(engine.device, generator=generator)
             for _ in range(steps)]
    return indices, noise, pairs


def _actor_arm(engine, multiplier, steps, tapes):
    """Frozen-Q actor-only intervention; alpha zero keeps stochastic sampling."""
    cfg = engine.cfg
    actor = deepcopy(engine.initial_actor).train().requires_grad_(True)
    params = _parameters(actor)
    eps = cfg.inner_actor_adam_eps if cfg.inner_actor_adam_eps is not None else cfg.inner_adam_eps
    optimizer = torch.optim.Adam(params, lr=float(cfg.inner_actor_lr), eps=float(eps),
                                 foreach=engine.device.type == "cuda")
    initial_alpha = engine.initial_alpha * multiplier
    log_alpha = (torch.nn.Parameter(torch.tensor(initial_alpha, device=engine.device).log())
                 if multiplier else None)
    temperature_optimizer = (torch.optim.Adam([log_alpha], lr=float(cfg.inner_temperature_lr),
                                             eps=float(cfg.inner_adam_eps),
                                             foreach=engine.device.type == "cuda")
                             if multiplier else None)
    target = engine._resolved_inner_target_entropy()
    indices, noises, pairs = tapes
    updates = []
    with frozen_parameters(engine.model), frozen_parameters(engine.first_critic):
        for index in range(steps):
            z = engine.first_replay_z[indices[index]]
            action, info = engine.model.pi(z, policy=actor, noise=noises[index], **_policy_options(cfg))
            q = _q(engine.model, engine.first_critic, z, action, cfg.inner_q_actor_reduction, pairs[index])
            entropy = -info["log_prob"].mean()
            alpha = log_alpha.detach().exp() if multiplier else z.new_zeros(())
            loss = -q.mean() - alpha * entropy
            gr, gh = _gradient(q.mean(), actor), _gradient(entropy, actor)
            before = _vector(actor).clone()
            temp_before = _temperature_state(log_alpha, temperature_optimizer) if multiplier else None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.inner_actor_grad_clip_norm))
            optimizer.step()
            if multiplier:
                alpha_loss = log_alpha * (entropy.detach() - target)
                temperature_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                clip = cfg.inner_temperature_grad_clip_norm
                temp_norm = torch.nn.utils.clip_grad_norm_([log_alpha], math.inf if clip is None else float(clip))
                temperature_optimizer.step()
                with torch.no_grad():
                    log_alpha.clamp_(min=math.log(1e-8))
            else:
                alpha_loss, temp_norm = z.new_zeros(()), z.new_zeros(())
            updates.append({
                "step": index + 1,
                "alpha_before": float(alpha),
                "alpha_after": float(log_alpha.detach().exp()) if multiplier else 0.,
                "temperature_before": temp_before,
                "temperature_after": _temperature_state(log_alpha, temperature_optimizer) if multiplier else None,
                "temperature_entropy_residual": float(entropy.detach()) - target,
                "temperature_loss": float(alpha_loss.detach()),
                "temperature_grad_norm_pre_clip": float(temp_norm),
                "actor_grad_norm_pre_clip": float(actor_norm),
                "actor_loss": float(loss.detach()),
                "training_batch_entropy": float(entropy.detach()),
                "training_batch_return_q": float(q.detach().mean()),
                "anchors_after": _anchor_readout(engine, actor, engine.anchors, engine.anchor_noise),
                "gradient_projections": _gradient_summary(gr, gh, alpha=float(alpha), delta=_vector(actor) - before),
            })
    return {
        "alpha_multiplier": multiplier,
        "temperature_mode": "auto" if multiplier else "disabled",
        "critic_mode": "frozen_post_first_critic_phase_eval_dropout_disabled",
        "policy_sampling": "stochastic_with_common_presampled_standard_normal_noise",
        "paired_sample_id": _hash(tapes),
        "initial_actor_sha256": _hash(engine.initial_actor.state_dict()),
        "frozen_critic_sha256": _hash(engine.first_critic.state_dict()),
        "fixed_replay_sha256": _hash(engine.first_replay_z),
        "updates": updates,
        "final_anchor_entropy": _anchor_readout(engine, actor, engine.anchors, engine.anchor_noise),
        "final_actor_sha256": _hash(actor.state_dict()),
    }


def _validate(cfg):
    expected = {
        "obs": "state", "inner_operator": "sac", "critic_value_mode": "single",
        "aux_return_mode": "sac", "inner_actor_source": "sac",
        "inner_critic_source": "aux_return", "inner_sac_critic_target": "reward_only",
        "outer_actor_entropy_mode": "squashed", "inner_actor_entropy_mode": "squashed",
        "inner_actor_adaptation": "clone", "inner_critic_adaptation": "clone",
        "inner_temperature_mode": "auto", "inner_temperature_initialization": "inherit_outer",
        "inner_replay_sampling": "with_replacement", "inner_schedule_mode": "canonical",
        "inner_update_timing": "round", "inner_horizon_conditioning": "none",
        "inner_explorer_mode": "none", "sac_actor_loss_scale_mode": "none",
    }
    for key, value in expected.items():
        if getattr(cfg, key, None) != value:
            raise ValueError(f"Entropy probe requires {key}={value!r}; got {getattr(cfg, key, None)!r}.")
    if (cfg.episodic or not cfg.inner_component_update_schedule
            or not cfg.inner_entropy_enabled or not cfg.inner_finite_horizon
            or cfg.inner_actor_writeback_coef or cfg.inner_critic_writeback_coef
            or cfg.inner_critic_bernoulli_p or cfg.inner_outer_policy_kl_coef
            or cfg.inner_outer_replay_fraction):
        raise ValueError("Entropy probe requires fresh finite-horizon component SAC without writeback, transfer, KL, or real replay.")
    for component in ("actor", "critic", "temperature", "replay", "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
        if getattr(cfg, f"inner_{component}_scope") != "action":
            raise ValueError("Entropy probe requires action-local components and optimizer state.")
    if min(cfg.inner_rounds, cfg.inner_rollouts_per_round, cfg.inner_rollout_horizon,
           cfg.inner_actor_updates_per_round, cfg.inner_critic_updates_per_round) <= 0:
        raise ValueError("Entropy probe requires positive rollout and component update counts.")


def probe_entropy_root(wrapper, observation, *, seed, actor_steps=32, anchor_samples=128, exact_engine=True):
    """Return JSON-safe paired entropy diagnostics without touching outer state.

    A new engine uses ``seed``; the wrapper's continuing-episode engine is never
    advanced. Its returned mean action therefore belongs to this standalone
    root solve. ``exact_engine=False`` stops after the first critic phase and
    runs only the fixed-replay counterfactuals.
    """
    from evaluate_ambi_checkpoint import _outer_state_digest
    if actor_steps < 1 or anchor_samples < 2:
        raise ValueError("actor_steps must be positive and anchor_samples at least two.")
    agent, cfg = wrapper.agent, wrapper.cfg
    _validate(cfg)
    before = _outer_state_digest(wrapper)
    global_before = torch.random.get_rng_state().clone()
    with _preserve_environment(agent.model, agent.device), torch.enable_grad():
        obs = torch.as_tensor(observation, device=agent.device, dtype=torch.float32)
        with torch.no_grad():
            root_z = agent.model.encode(obs.unsqueeze(0)).detach()
        engine = _EntropyProbeEngine(agent, root_z=root_z, seed=seed,
                                     anchor_samples=anchor_samples, full_solve=exact_engine)
        exact = {"performed": bool(exact_engine)}
        try:
            action, metrics, lengths = engine.act(root_z, t0=True, eval_mode=True,
                                                  collect_diagnostics=False)
        except _FirstPhaseComplete:
            if exact_engine:
                raise
        else:
            exact.update(action=_json(action), metrics=_json(metrics), rollout_lengths=_json(lengths),
                         final_actor_sha256=_hash(engine._action_pool.actor.state_dict()),
                         final_critic_sha256=_hash(engine._action_pool.critic.state_dict()),
                         final_private_rng_sha256=_hash(engine.rng.training_state_dict()))
        if engine.first_critic is None:
            raise RuntimeError("Entropy probe did not capture the first critic phase.")
        exact["actor_updates"] = engine.actor_updates
        tapes = _paired_tapes(engine, actor_steps)
        arms = {name: _actor_arm(engine, multiplier, actor_steps, tapes)
                for name, multiplier in (("inherited_auto", 1.), ("tenth_auto", .1), ("zero_entropy", 0.))}
        result = {
            "schema_version": 1, "seed": int(seed),
            "protocol": {
                "inner_horizon": int(cfg.inner_rollout_horizon), "inner_rounds": int(cfg.inner_rounds),
                "critic_updates_per_round": int(cfg.inner_critic_updates_per_round),
                "actor_updates_per_round": int(cfg.inner_actor_updates_per_round),
                "counterfactual_actor_steps": int(actor_steps), "anchor_samples_per_group": int(anchor_samples),
                "target_entropy": engine._resolved_inner_target_entropy(),
                "initial_alpha": engine.initial_alpha,
                "actor_lr": float(cfg.inner_actor_lr), "temperature_lr": float(cfg.inner_temperature_lr),
                "counterfactuals_are_closed_loop_evaluations": False,
                "anchor_noise_is_fixed_across_updates": True,
                "critic_gradient_comparisons_use_eval_mode": True,
                "actual_step_projections_use_original_stochastic_forward": True,
                "timing_includes_observer_overhead": True,
            },
            "anchors_sha256": _hash(engine.anchors),
            "anchor_noise_sha256": _hash(engine.anchor_noise),
            "anchors_initial": engine.anchors_initial,
            "gradients_before_critic": engine.gradients_before,
            "gradients_after_first_critic_phase": engine.gradients_after,
            "exact_engine": exact, "actor_arms": arms,
        }
    after = _outer_state_digest(wrapper)
    if before != after:
        raise RuntimeError("Entropy probe mutated outer checkpoint/optimizer state.")
    result.update(outer_state_unchanged=True, outer_state_sha256=before,
                  global_rng_unchanged=bool(torch.equal(global_before, torch.random.get_rng_state())))
    return _json(result)
