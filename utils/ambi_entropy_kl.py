"""Observe automatic-entropy SAC with a fixed penalty to its frozen prior.

All arms execute the ordinary complete inner solve. Beta is selected once from
the checkpoint's alpha and is held fixed while inner alpha continues adapting.
This is a soft KL penalty, not a KL budget or an enforced entropy constraint.
"""
from copy import copy, deepcopy
import math

import torch

from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from utils.ambi_entropy_probe import (
    _EntropyProbeEngine, _anchor_gradients, _cosine, _hash, _json,
    _preserve_environment, _temperature_state, _validate, _vector,
)
from utils.ambi_prior_kl import _RegularizerRegion, _projections, _readout


DEFAULT_BETA_MULTIPLIERS = (0., 1., 3.)
PAIRING_FIELDS = (
    "first_replay_sha256", "first_critic_sha256", "initial_actor_sha256",
    "anchors_sha256", "anchor_noise_sha256", "anchor_pairs_sha256",
    "non_beta_config_sha256", "final_private_rng_sha256",
)


def beta_multiplier_key(multiplier):
    return "beta_" + format(float(multiplier), ".12g").replace(".", "p").replace("-", "m").replace("+", "") + "x"


def validate_beta_multipliers(values):
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Beta multipliers must be finite and nonnegative.")
    if len({beta_multiplier_key(value) for value in values}) != len(values):
        raise ValueError("Beta multipliers must have distinct arm identifiers.")
    return values


class _EntropyKLProbeEngine(_EntropyProbeEngine):
    def __init__(self, *args, beta, **kwargs):
        self.beta = float(beta)
        super().__init__(*args, **kwargs)
        self._compile_regions["actor"] = _RegularizerRegion(
            self._compile_regions["actor"].region, self)

    def _capture_first_replay(self):
        super()._capture_first_replay()
        self.anchors_initial = _readout(self, self.state.actor)

    def _sac_policy_step(self, batch, **kwargs):
        if self.first_critic is None:
            if (self.state.critic_steps != int(self.cfg.inner_critic_updates_per_round)
                    or self.state.actor_steps):
                raise RuntimeError("Unexpected first critic/actor schedule.")
            self.first_critic = deepcopy(self.state.critic).eval().requires_grad_(False)
            self.gradients_after = _anchor_gradients(
                self, self.state.actor, self.first_critic, self.anchors,
                self.anchor_noise, self.anchor_pairs, self.initial_alpha)
        if not kwargs["update_actor"] or not kwargs["update_temperature"]:
            raise RuntimeError("Joint probe requires paired actor/temperature updates.")
        before = _vector(self.state.actor).clone()
        anchors_before = _readout(self, self.state.actor)
        temperature_before = _temperature_state(self.state.log_alpha, self.state.temperature_optim)
        self._observe_actor = True
        try:
            metrics = InnerImprovementEngine._sac_policy_step(self, batch, **kwargs)
        finally:
            self._observe_actor = False
        gq, gh, gkl, kl = self._actual_gradients
        self._actual_gradients = None
        alpha = float(kwargs["alpha"])
        delta = _vector(self.state.actor) - before
        projections = _projections(gq, gh, gkl, alpha, self.beta, delta)
        projections.update(
            entropy_kl_gradient_cosine=_cosine(gh, gkl),
            return_q_kl_gradient_cosine=_cosine(gq, gkl),
            regularized_objective_gradient_norm=float((gq + alpha * gh - self.beta * gkl).norm()),
            weighted_kl_to_entropy_gradient_norm=(
                float(self.beta * gkl.norm() / (alpha * gh.norm()))
                if alpha * float(gh.norm()) > 1e-20 else None))
        self.actor_updates.append(dict(
            step=int(self.state.actor_steps),
            round=(int(self.state.actor_steps) - 1) // int(self.cfg.inner_actor_updates_per_round) + 1,
            alpha_before=alpha, alpha_after=float(self.alpha.detach()), beta=self.beta,
            temperature_before=temperature_before,
            temperature_after=_temperature_state(self.state.log_alpha, self.state.temperature_optim),
            temperature_entropy_residual=float(metrics["actor_entropy"]) - self._resolved_inner_target_entropy(),
            temperature_steps=int(self.state.temperature_steps), kl_before=kl,
            metrics=_json(metrics), anchors_before=anchors_before,
            anchors_after=_readout(self, self.state.actor),
            gradient_mode="actual_Q_entropy_forward_and_analytic_deterministic_KL",
            actual_update_critic_training=bool(self.state.critic.training),
            gradient_projections=projections))
        return metrics


def probe_entropy_kl_root(wrapper, observation, *, seed, anchor_samples=128,
                          beta_multipliers=DEFAULT_BETA_MULTIPLIERS):
    """Paired full solves changing only a fixed prior-KL penalty coefficient."""
    from evaluate_ambi_checkpoint import _outer_state_digest
    _validate(wrapper.cfg)
    if anchor_samples < 2:
        raise ValueError("Need at least two anchor samples.")
    multipliers = validate_beta_multipliers(beta_multipliers)
    agent, cfg = wrapper.agent, wrapper.cfg
    if float(cfg.inner_temperature_lr) != .0003:
        raise ValueError("Joint entropy-KL probe keeps inner_temperature_lr=0.0003.")
    initial_alpha = float(agent.alpha.detach())
    if not math.isfinite(initial_alpha) or initial_alpha <= 0:
        raise ValueError("Inherited outer alpha must be finite and positive.")
    expected_actor = int(cfg.inner_actor_updates_per_action)
    expected_temperature = int(cfg.inner_temperature_updates_per_action)
    if expected_actor != expected_temperature:
        raise ValueError("Need one temperature update per actor update.")
    before = _outer_state_digest(wrapper)
    global_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(agent.device).clone() if agent.device.type == "cuda" else None
    full = {}
    with _preserve_environment(agent.model, agent.device), torch.enable_grad():
        with torch.no_grad():
            root_z = agent.model.encode(torch.as_tensor(
                observation, device=agent.device, dtype=torch.float32)[None]).detach()
        for multiplier in multipliers:
            beta = initial_alpha * multiplier
            if not math.isfinite(beta):
                raise ValueError("Beta is not finite.")
            proxy = copy(agent)
            proxy.cfg = deepcopy(agent.cfg)
            proxy.cfg.inner_outer_policy_kl_coef = beta
            engine = _EntropyKLProbeEngine(proxy, root_z=root_z, seed=seed,
                anchor_samples=anchor_samples, full_solve=True, beta=beta)
            action, metrics, lengths = engine.act(root_z, t0=True, eval_mode=True,
                                                  collect_diagnostics=False)
            final = engine._action_pool
            if (len(engine.actor_updates) != expected_actor
                    or engine.state.temperature_steps != expected_temperature
                    or engine.state.critic_steps != int(cfg.inner_critic_updates_per_action)):
                raise RuntimeError("Joint entropy-KL probe did not complete its full solve.")
            if final.log_alpha is None or final.temperature_optim is None:
                raise RuntimeError("Joint entropy-KL probe disabled automatic temperature.")
            full[beta_multiplier_key(multiplier)] = dict(
                beta_multiplier=multiplier, beta=beta, entropy_enabled=True,
                inner_temperature_lr=float(cfg.inner_temperature_lr),
                config_overrides={"inner_outer_policy_kl_coef": beta},
                full_solve_performed=True, action=_json(action), metrics=_json(metrics),
                rollout_lengths=_json(lengths), initial_alpha=engine.initial_alpha,
                target_entropy=engine._resolved_inner_target_entropy(),
                actor_steps=len(engine.actor_updates), critic_steps=int(engine.state.critic_steps),
                temperature_steps=int(engine.state.temperature_steps), temperature_allocated=True,
                temperature_final=_temperature_state(final.log_alpha, final.temperature_optim),
                anchors_initial=engine.anchors_initial, actor_updates=engine.actor_updates,
                anchors_final=_readout(engine, final.actor),
                gradients_before_critic=engine.gradients_before,
                gradients_after_first_critic_phase=engine.gradients_after,
                first_replay_sha256=_hash(engine.first_replay_z),
                first_critic_sha256=_hash(engine.first_critic.state_dict()),
                initial_actor_sha256=_hash(engine.initial_actor.state_dict()),
                anchors_sha256=_hash(engine.anchors), anchor_noise_sha256=_hash(engine.anchor_noise),
                anchor_pairs_sha256=_hash(engine.anchor_pairs),
                non_beta_config_sha256=_hash({k: v for k, v in vars(proxy.cfg).items()
                                              if k != "inner_outer_policy_kl_coef"}),
                final_actor_sha256=_hash(final.actor.state_dict()),
                final_critic_sha256=_hash(final.critic.state_dict()),
                final_private_rng_sha256=_hash(engine.rng.training_state_dict()))
        reference = next(iter(full.values()))
        for name, arm in full.items():
            for key in PAIRING_FIELDS:
                if arm[key] != reference[key]:
                    raise RuntimeError(f"Unpaired joint entropy-KL solve {name}: {key}")
    if _outer_state_digest(wrapper) != before:
        raise RuntimeError("Joint entropy-KL probe mutated outer state.")
    rng_ok = torch.equal(global_before, torch.random.get_rng_state()) and (
        cuda_before is None or torch.equal(cuda_before, torch.cuda.get_rng_state(agent.device)))
    if not rng_ok:
        raise RuntimeError("Joint entropy-KL probe mutated global RNG.")
    return _json(dict(
        schema_version=1, kind="ambi_entropy_kl_root_probe", seed=int(seed),
        initial_alpha=initial_alpha, target_entropy=reference["target_entropy"],
        beta_multipliers=multipliers, full_solves=full, paired_first_phase=True,
        paired_private_rng_draw_counts=True, outer_state_unchanged=True,
        outer_state_sha256=before, global_rng_unchanged=True,
        kl_direction="adapted_policy_to_frozen_inherited_primary_actor",
        kl_scope="per_state_joint_action", hard_kl_budget_enforced=False,
        counterfactuals_are_closed_loop_evaluations=False,
        protocol=dict(changed_parameter="inner_outer_policy_kl_coef",
            beta_selection="multiplier_times_inherited_outer_alpha_held_fixed",
            inner_temperature_lr=float(cfg.inner_temperature_lr),
            inner_horizon=int(cfg.inner_rollout_horizon), inner_rounds=int(cfg.inner_rounds),
            actor_steps=expected_actor, critic_steps=int(cfg.inner_critic_updates_per_action),
            temperature_steps=expected_temperature, anchor_samples_per_group=int(anchor_samples),
            temperature_uses_pre_actor_update_entropy=True,
            anchor_noise_is_fixed_across_updates=True,
            actual_step_projections_use_original_stochastic_forward=True,
            subsequent_replay_may_diverge_with_policy=True, timing_includes_observer_overhead=True)))
