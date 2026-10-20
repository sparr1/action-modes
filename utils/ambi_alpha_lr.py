"""Paired complete inner-SAC solves varying only the temperature learning rate.

These are frozen-root diagnostics, not episode return evaluations. All arms
start from the same actor, critic, temperature, private RNG, and first replay.
Their later imagined replay is allowed to change with the adapted policy.
"""
from copy import copy, deepcopy
import math

import torch

from utils.ambi_entropy_probe import (
    _EntropyProbeEngine, _anchor_readout, _hash, _json,
    _preserve_environment, _temperature_state, _validate,
)


DEFAULT_TEMPERATURE_LRS = (.0003, .003, .03, .3)
PAIRING_FIELDS = (
    "first_replay_sha256", "first_critic_sha256", "initial_actor_sha256",
    "anchors_sha256", "anchor_noise_sha256", "anchor_pairs_sha256",
    "non_lr_config_sha256", "final_private_rng_sha256",
)


def temperature_lr_key(rate):
    return "lr_" + format(float(rate), ".12g").replace(".", "p").replace("-", "m").replace("+", "")


def validate_temperature_lrs(rates):
    rates = tuple(float(rate) for rate in rates)
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Temperature learning rates must be finite and positive.")
    if len({temperature_lr_key(rate) for rate in rates}) != len(rates):
        raise ValueError("Temperature learning rates must have distinct arm identifiers.")
    return rates


def probe_alpha_lr_root(wrapper, observation, *, seed, anchor_samples=128,
                        temperature_lrs=DEFAULT_TEMPERATURE_LRS):
    """Return observed full solves without advancing any live or outer state."""
    from evaluate_ambi_checkpoint import _outer_state_digest
    _validate(wrapper.cfg)
    if anchor_samples < 2:
        raise ValueError("Need at least two anchor samples.")
    rates = validate_temperature_lrs(temperature_lrs)
    agent, cfg = wrapper.agent, wrapper.cfg
    initial_alpha = float(agent.alpha.detach())
    if not math.isfinite(initial_alpha) or initial_alpha <= 0:
        raise ValueError("Inherited outer alpha must be finite and positive.")
    expected_actor = int(cfg.inner_actor_updates_per_action)
    expected_temperature = int(cfg.inner_temperature_updates_per_action)
    if expected_temperature != expected_actor:
        raise ValueError("This observer requires one temperature update per actor update.")
    before = _outer_state_digest(wrapper)
    global_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(agent.device).clone() if agent.device.type == "cuda" else None
    full = {}
    with _preserve_environment(agent.model, agent.device), torch.enable_grad():
        with torch.no_grad():
            root_z = agent.model.encode(torch.as_tensor(
                observation, device=agent.device, dtype=torch.float32)[None]).detach()
        for rate in rates:
            proxy = copy(agent)
            proxy.cfg = deepcopy(agent.cfg)
            proxy.cfg.inner_temperature_lr = rate
            engine = _EntropyProbeEngine(proxy, root_z=root_z, seed=seed,
                                         anchor_samples=anchor_samples, full_solve=True)
            action, metrics, lengths = engine.act(root_z, t0=True, eval_mode=True,
                                                  collect_diagnostics=False)
            final = engine._action_pool
            if (len(engine.actor_updates) != expected_actor
                    or engine.state.temperature_steps != expected_temperature
                    or engine.state.critic_steps != int(cfg.inner_critic_updates_per_action)):
                raise RuntimeError("Temperature-LR probe did not complete the configured full solve.")
            if final.log_alpha is None or final.temperature_optim is None:
                raise RuntimeError("Temperature-LR arm did not allocate its automatic temperature.")
            actual_lr = float(final.temperature_optim.param_groups[0]["lr"])
            if actual_lr != rate:
                raise RuntimeError("Temperature optimizer did not use the requested learning rate.")
            full[temperature_lr_key(rate)] = dict(
                inner_temperature_lr=rate, optimizer_temperature_lr=actual_lr,
                config_overrides={"inner_temperature_lr": rate}, full_solve_performed=True,
                action=_json(action), metrics=_json(metrics), rollout_lengths=_json(lengths),
                initial_alpha=engine.initial_alpha,
                target_entropy=engine._resolved_inner_target_entropy(),
                actor_steps=len(engine.actor_updates), critic_steps=int(engine.state.critic_steps),
                temperature_steps=int(engine.state.temperature_steps), temperature_allocated=True,
                temperature_final=_temperature_state(final.log_alpha, final.temperature_optim),
                anchors_initial=engine.anchors_initial, actor_updates=engine.actor_updates,
                anchors_final=_anchor_readout(engine, final.actor, engine.anchors, engine.anchor_noise),
                gradients_before_critic=engine.gradients_before,
                gradients_after_first_critic_phase=engine.gradients_after,
                first_replay_sha256=_hash(engine.first_replay_z),
                first_critic_sha256=_hash(engine.first_critic.state_dict()),
                initial_actor_sha256=_hash(engine.initial_actor.state_dict()),
                anchors_sha256=_hash(engine.anchors), anchor_noise_sha256=_hash(engine.anchor_noise),
                anchor_pairs_sha256=_hash(engine.anchor_pairs),
                non_lr_config_sha256=_hash({k: v for k, v in vars(proxy.cfg).items()
                                            if k != "inner_temperature_lr"}),
                final_actor_sha256=_hash(final.actor.state_dict()),
                final_critic_sha256=_hash(final.critic.state_dict()),
                final_private_rng_sha256=_hash(engine.rng.training_state_dict()))
        reference = next(iter(full.values()))
        for name, arm in full.items():
            for key in PAIRING_FIELDS:
                if arm[key] != reference[key]:
                    raise RuntimeError(f"Unpaired temperature-LR solve {name}: {key}")
            # Alpha is changed after the first actor step. Its objective and
            # actual Adam displacement therefore must agree across all arms.
            for key in ("alpha_before", "metrics", "gradient_projections", "anchors_after"):
                left, right = arm["actor_updates"][0], reference["actor_updates"][0]
                if _hash(left[key]) != _hash(right[key]):
                    raise RuntimeError(f"Unpaired first actor update in {name}: {key}")
    if _outer_state_digest(wrapper) != before:
        raise RuntimeError("Temperature-LR probe mutated outer state.")
    rng_ok = torch.equal(global_before, torch.random.get_rng_state()) and (
        cuda_before is None or torch.equal(cuda_before, torch.cuda.get_rng_state(agent.device)))
    if not rng_ok:
        raise RuntimeError("Temperature-LR probe mutated global RNG.")
    return _json(dict(
        schema_version=1, kind="ambi_alpha_lr_root_probe", seed=int(seed),
        initial_alpha=initial_alpha, target_entropy=reference["target_entropy"],
        temperature_lrs=rates, full_solves=full, paired_first_phase=True,
        paired_first_actor_update=True, paired_private_rng_draw_counts=True,
        outer_state_unchanged=True, outer_state_sha256=before, global_rng_unchanged=True,
        counterfactuals_are_closed_loop_evaluations=False,
        protocol=dict(
            changed_parameter="inner_temperature_lr", inner_horizon=int(cfg.inner_rollout_horizon),
            inner_rounds=int(cfg.inner_rounds), actor_steps=expected_actor,
            critic_steps=int(cfg.inner_critic_updates_per_action), temperature_steps=expected_temperature,
            anchor_samples_per_group=int(anchor_samples), actor_lr=float(cfg.inner_actor_lr),
            critic_lr=float(cfg.inner_critic_lr), initial_temperature="inherited_outer_each_solve",
            temperature_optimizer="fresh_Adam_each_solve",
            temperature_uses_pre_actor_update_entropy=True,
            anchor_noise_is_fixed_across_updates=True,
            actual_step_projections_use_original_stochastic_forward=True,
            subsequent_replay_may_diverge_with_policy=True,
            timing_includes_observer_overhead=True)))
