"""Paired prior-KL versus entropy inner-SAC interventions on fixed roots."""
from copy import copy, deepcopy
import math

import torch

from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from utils.ambi_entropy_probe import (
    _EntropyProbeEngine, _gradient, _gradient_summary, _hash, _json,
    _paired_tapes, _parameters, _policy_options, _preserve_environment, _q,
    _temperature_state, _validate, _vector, entropy_components,
)
from utils.ambi_gradient_probe import frozen_parameters


def _kl(engine, actor, z):
    _, info = engine.model.pi(z, policy=actor, deterministic=True,
                              **_policy_options(engine.cfg))
    with torch.no_grad():
        _, prior = engine.model.pi(z, policy=engine._actor_base, deterministic=True,
                                    **engine._actor_options)
    return engine._gaussian_kl(info, prior).mean()


def _readout(engine, actor):
    result = {}
    with torch.no_grad():
        for name, z in engine.anchors.items():
            _, info = engine.model.pi(z, policy=actor, noise=engine.anchor_noise[name],
                                      **_policy_options(engine.cfg))
            result[name] = {**entropy_components(info), "kl_to_prior": float(_kl(engine, actor, z))}
    return result


def _projections(gq, gh, gkl, alpha, beta, delta):
    result = _gradient_summary(gq, gh, alpha=alpha, delta=delta)
    result.update(kl_gradient_norm=float(gkl.norm()),
                  weighted_kl_to_q_gradient_norm=(float(beta * gkl.norm() / gq.norm())
                                                 if float(gq.norm()) > 1e-20 else None),
                  kl_first_order_change=float(gkl @ delta),
                  regularized_objective_first_order_change=float((gq + alpha * gh - beta * gkl) @ delta))
    return result


class _RegularizerRegion:
    def __init__(self, region, engine):
        self.region, self.engine = region, engine

    def __getattr__(self, name):
        return getattr(self.region, name)

    def __call__(self, *args, **kwargs):
        output = self.region(*args, **kwargs)
        if self.engine._observe_actor:
            actor = self.engine.state.actor
            # Gaussian KL is analytic and independent of sampled actions. This
            # deterministic forward adds no random draws or dropout (actor has none).
            kl = _kl(self.engine, actor, args[0])
            self.engine._actual_gradients = (
                _gradient(output[3], actor), _gradient(output[1].mean(), actor),
                _gradient(kl, actor), float(kl.detach()),
            )
        return output


class _PriorKLEngine(_EntropyProbeEngine):
    def __init__(self, *args, beta, **kwargs):
        self.beta = float(beta)
        super().__init__(*args, **kwargs)
        # Replace the existing observer, keeping its untouched CompileRegion.
        self._compile_regions["actor"] = _RegularizerRegion(self._compile_regions["actor"].region, self)

    def _capture_first_replay(self):
        super()._capture_first_replay()
        self.anchors_initial = _readout(self, self.state.actor)

    def _sac_policy_step(self, batch, **kwargs):
        if self.first_critic is None:
            if self.state.critic_steps != self.cfg.inner_critic_updates_per_round or self.state.actor_steps:
                raise RuntimeError("Unexpected first critic/actor schedule.")
            self.first_critic = deepcopy(self.state.critic).eval().requires_grad_(False)
        before = _vector(self.state.actor).clone()
        self._observe_actor = True
        try:
            metrics = InnerImprovementEngine._sac_policy_step(self, batch, **kwargs)
        finally:
            self._observe_actor = False
        gq, gh, gkl, kl = self._actual_gradients
        self._actual_gradients = None
        alpha = float(kwargs["alpha"])
        self.actor_updates.append(dict(
            step=int(self.state.actor_steps), round=(int(self.state.actor_steps) - 1) // int(self.cfg.inner_actor_updates_per_round) + 1,
            alpha_before=alpha, alpha_after=float(self.alpha.detach()), beta=self.beta,
            temperature_steps=int(self.state.temperature_steps), kl_before=kl,
            metrics=_json(metrics), anchors_after=_readout(self, self.state.actor),
            gradient_mode="actual_Q_entropy_forward_and_analytic_deterministic_KL",
            gradient_projections=_projections(gq, gh, gkl, alpha, self.beta,
                                             _vector(self.state.actor) - before)))
        return metrics


def _fixed_arm(engine, *, beta, entropy_enabled, steps, tapes, initial_alpha):
    actor = deepcopy(engine.initial_actor).train().requires_grad_(True)
    cfg, params = engine.cfg, _parameters(actor)
    eps = cfg.inner_actor_adam_eps if cfg.inner_actor_adam_eps is not None else cfg.inner_adam_eps
    optimizer = torch.optim.Adam(params, lr=cfg.inner_actor_lr, eps=eps,
                                 foreach=engine.device.type == "cuda")
    log_alpha = torch.nn.Parameter(torch.tensor(initial_alpha, device=engine.device).log()) if entropy_enabled else None
    temp_optim = (torch.optim.Adam([log_alpha], lr=cfg.inner_temperature_lr, eps=cfg.inner_adam_eps,
                                   foreach=engine.device.type == "cuda") if entropy_enabled else None)
    indices, noises, pairs = tapes
    updates = []
    with frozen_parameters(engine.model), frozen_parameters(engine.first_critic):
        for index in range(steps):
            z = engine.first_replay_z[indices[index]]
            action, info = engine.model.pi(z, policy=actor, noise=noises[index], **_policy_options(cfg))
            q = _q(engine.model, engine.first_critic, z, action, cfg.inner_q_actor_reduction, pairs[index]).mean()
            entropy, kl = -info["log_prob"].mean(), _kl(engine, actor, z)
            alpha = log_alpha.detach().exp() if entropy_enabled else z.new_zeros(())
            loss = -q - alpha * entropy + beta * kl
            gq, gh, gkl = _gradient(q, actor), _gradient(entropy, actor), _gradient(kl, actor)
            before = _vector(actor).clone()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.inner_actor_grad_clip_norm)
            optimizer.step()
            if entropy_enabled:
                temp_optim.zero_grad(set_to_none=True)
                (log_alpha * (entropy.detach() - engine._resolved_inner_target_entropy())).backward()
                torch.nn.utils.clip_grad_norm_([log_alpha], engine._temperature_grad_clip_maximum)
                temp_optim.step()
                with torch.no_grad():
                    log_alpha.clamp_(min=math.log(1e-8))
            updates.append(dict(step=index + 1, alpha_before=float(alpha), beta=beta,
                kl_before=float(kl.detach()), entropy_before=float(entropy.detach()),
                anchors_after=_readout(engine, actor),
                gradient_projections=_projections(gq, gh, gkl, float(alpha), beta, _vector(actor) - before)))
    return dict(paired_sample_id=_hash(tapes), initial_actor_sha256=_hash(engine.initial_actor.state_dict()),
                frozen_critic_sha256=_hash(engine.first_critic.state_dict()),
                fixed_replay_sha256=_hash(engine.first_replay_z), beta=beta,
                entropy_enabled=entropy_enabled, temperature_steps=steps if entropy_enabled else 0,
                critic_mode="frozen_post_first_critic_phase_eval_dropout_disabled",
                anchors_initial=_readout(engine, engine.initial_actor), updates=updates,
                final_anchor_metrics=_readout(engine, actor))


def probe_prior_kl_root(wrapper, observation, *, seed, anchor_samples=128, actor_steps=32):
    """Full paired solves plus fixed-Q actor interventions; never outer learning."""
    from evaluate_ambi_checkpoint import _outer_state_digest
    _validate(wrapper.cfg)
    if anchor_samples < 2 or actor_steps < 1:
        raise ValueError("Need at least two anchor samples and one actor step.")
    agent = wrapper.agent
    before = _outer_state_digest(wrapper)
    initial_alpha = float(agent.alpha.detach())
    if not math.isfinite(initial_alpha) or initial_alpha <= 0:
        raise ValueError("Inherited outer alpha must be finite and positive.")
    rng_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(agent.device).clone() if agent.device.type == "cuda" else None
    full, engines = {}, {}
    with _preserve_environment(agent.model, agent.device), torch.enable_grad():
        with torch.no_grad():
            root_z = agent.model.encode(torch.as_tensor(observation, device=agent.device, dtype=torch.float32)[None]).detach()
        for name, beta, entropy in (("baseline_entropy", 0., True), ("prior_kl", initial_alpha, False)):
            proxy = copy(agent)
            proxy.cfg = deepcopy(agent.cfg)
            proxy.cfg.inner_outer_policy_kl_coef = beta
            proxy.cfg.inner_entropy_enabled = entropy
            if not entropy:
                proxy.cfg.inner_temperature_mode = "inherit_outer"
            engine = _PriorKLEngine(proxy, root_z=root_z, seed=seed, anchor_samples=anchor_samples,
                                    full_solve=True, beta=beta)
            action, metrics, _ = engine.act(root_z, t0=True, eval_mode=True, collect_diagnostics=False)
            final = engine._action_pool
            full[name] = dict(action=_json(action), metrics=_json(metrics), beta=beta,
                entropy_enabled=entropy, actor_steps=len(engine.actor_updates),
                temperature_steps=engine.actor_updates[-1]["temperature_steps"],
                temperature_allocated=final.log_alpha is not None,
                anchors_initial=engine.anchors_initial, actor_updates=engine.actor_updates,
                anchors_final=_readout(engine, final.actor),
                first_replay_sha256=_hash(engine.first_replay_z),
                first_critic_sha256=_hash(engine.first_critic.state_dict()),
                initial_actor_sha256=_hash(engine.initial_actor.state_dict()),
                anchors_sha256=_hash(engine.anchors), anchor_noise_sha256=_hash(engine.anchor_noise),
                final_actor_sha256=_hash(final.actor.state_dict()), final_critic_sha256=_hash(final.critic.state_dict()),
                final_private_rng_sha256=_hash(engine.rng.training_state_dict()))
            engines[name] = engine
        for key in ("first_replay_sha256", "first_critic_sha256", "initial_actor_sha256", "anchors_sha256", "anchor_noise_sha256"):
            if full["baseline_entropy"][key] != full["prior_kl"][key]:
                raise RuntimeError(f"Unpaired full-solve initialization: {key}")
        base = engines["baseline_entropy"]
        tapes = _paired_tapes(base, actor_steps)
        fixed = {name: _fixed_arm(base, beta=beta, entropy_enabled=entropy, steps=actor_steps,
                                  tapes=tapes, initial_alpha=initial_alpha)
                 for name, beta, entropy in (("baseline_entropy", 0., True), ("prior_kl", initial_alpha, False))}
    if _outer_state_digest(wrapper) != before:
        raise RuntimeError("Prior-KL probe mutated outer state.")
    rng_ok = torch.equal(rng_before, torch.random.get_rng_state()) and (
        cuda_before is None or torch.equal(cuda_before, torch.cuda.get_rng_state(agent.device)))
    if not rng_ok:
        raise RuntimeError("Prior-KL probe mutated global RNG.")
    return _json(dict(schema_version=1, seed=seed, beta=initial_alpha,
        kl_direction="adapted_policy_to_frozen_inherited_primary_actor", kl_scope="per_state_joint_action",
        outer_state_unchanged=True, outer_state_sha256=before, global_rng_unchanged=True,
        paired_first_phase=True, full_solves=full, fixed_replay_arms=fixed,
        counterfactuals_are_closed_loop_evaluations=False))
