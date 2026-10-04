"""Controlled, checkpoint-frozen inner-SAC transfer diagnostics.

This module owns experiment orchestration, not production controller semantics.
Every fork starts with fresh replay/Adam/temperature, explicit weight donors,
and an identical private solver RNG. Diagnostic Monte Carlo uses separate RNG.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace
from utils.ambi_benchmark import solver_seed
from utils.transfer_diagnostic_metrics import (
    action_value_metrics, forced_action_mc_returns, fit_stationary_targets,
    paired_directional_metrics,
)


PROTOCOL = "inner-sac-transfer-diagnostics-v1"
BRANCHES = {
    "fresh": (False, False), "actor": (True, False),
    "critic": (False, True), "joint": (True, True),
}


def positive_ints(values, name):
    result = tuple(values)
    if not result or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in result):
        raise ValueError(f"{name} must contain positive integers.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def json_value(value):
    if torch.is_tensor(value):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite diagnostic output.")
    return value


def write_json(path, value):
    """Exclusive output creation: completed evidence is never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(json_value(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


@contextmanager
def evaluating(*modules):
    modes = {m: m.training for root in modules if root is not None for m in root.modules()}
    try:
        for m in modes:
            m.training = False
        yield
    finally:
        for m, mode in modes.items():
            m.training = mode


def expected_reduction(values, reduction, pair_size=2):
    """Integrate random head choice analytically; preserve action gradients."""
    if reduction in {"mean", "mean_all", "mean_pair"}:
        return values.mean(0)
    if reduction == "min_all":
        return values.min(0).values
    if reduction in {"min", "min_pair"}:
        return torch.stack([values[list(ids)].min(0).values
                            for ids in combinations(range(len(values)), pair_size)]).mean(0)
    raise ValueError(f"Unsupported diagnostic Q reduction: {reduction}")


def validate_controller(wrapped):
    cfg = wrapped.cfg
    expected = dict(inner_operator="sac", inner_actor_adaptation="clone",
                    inner_critic_adaptation="clone", inner_explorer_mode="none",
                    inner_update_timing="round", inner_sac_return_estimator="one_step",
                    inner_bootstrap_source="inner_target",
                    inner_solve_interval=1, inner_finite_horizon=True,
                    inner_actor_writeback_coef=0., inner_critic_writeback_coef=0.,
                    inner_actor_entropy_mode="squashed", inner_component_update_order="critic_first",
                    inner_eval_execution_action="mean")
    expected.update({f"inner_{name}_scope": "action" for name in (
        "actor", "critic", "replay", "temperature", "actor_optimizer",
        "critic_optimizer", "temperature_optimizer")})
    mismatch = {key: (getattr(cfg, key, None), value) for key, value in expected.items()
                if getattr(cfg, key, None) != value}
    if mismatch:
        raise ValueError(f"Diagnostic controller requires explicit supported semantics: {mismatch}")
    if getattr(cfg, "obs", "state") != "state":
        raise ValueError("Transfer root capture currently requires state observations.")
    if getattr(cfg, "inner_horizon_conditioning", "none") not in {"none", None, False}:
        raise ValueError("This evaluator currently probes unconditioned actors/critics.")
    if getattr(cfg, "critic_value_mode", "single") != "single":
        raise ValueError("Split return/entropy heads require a separate diagnostic projection.")


class Reference:
    """Eager frozen-model reference with explicit policy, noise and tail semantics."""

    def __init__(self, wrapped, *, rollouts=32):
        self.wrapped = wrapped
        self.model, self.engine, self.cfg = wrapped.agent.model, wrapped.agent.inner_engine, wrapped.cfg
        self.device = wrapped.agent.device
        self.rollouts = int(rollouts)
        self.discount = float(wrapped.agent.discount)
        self.bounds = dict(log_std_mapping=self.cfg.inner_log_std_mapping,
                           log_std_min=self.cfg.inner_log_std_min,
                           log_std_max=self.cfg.inner_log_std_max)

    @torch.no_grad()
    def encode(self, observations):
        return self.model.encode(torch.as_tensor(observations, device=self.device, dtype=torch.float32))

    def policy(self, actor, bounds=None):
        def apply(z, noise):
            action, info = self.model.pi(z, policy=actor, noise=noise, **(bounds or self.bounds))
            return action, info["log_prob"]
        return apply

    def q_heads(self, critic, z, actions):
        return self.model.Q(z, actions, qs=critic._forward_eager, reduction="all")

    def q(self, critic, z, actions, reduction=None):
        return expected_reduction(self.q_heads(critic, z, actions),
            reduction or self.cfg.inner_q_actor_reduction, self.model.q_backend.pair_size)

    def transition(self, z, action):
        joint = self.model.joint_input(z, action)
        reward = self.model.decode_reward(self.model.reward_from_joint(joint))
        next_z = self.model.next_from_joint(joint)
        done = (self.model.termination(next_z) > self.cfg.inner_termination_threshold
                if self.cfg.episodic else torch.zeros_like(reward, dtype=torch.bool))
        return reward, next_z, done

    def tail(self, z, noise):
        # Match _prior_bootstrap, including terminal entropy for soft/soft.
        if self.cfg.inner_terminal_entropy == "outer":
            from RL.tdmpc2_core.common.entropy import policy_entropy
            mode = self.cfg.outer_actor_entropy_mode
            action, info = self.model.pi(z, noise=noise,
                **({"include_scaled_entropy": True} if mode == "tdmpc2_scaled" else {}))
            heads = self.q_heads(self.model._Qs, z, action)
            coefficient = self.wrapped.agent.alpha.detach()
            if self.wrapped.agent.actor_loss_scale_enabled:
                coefficient = coefficient * self.wrapped.agent.actor_loss_scale.detach().reshape(())
            bonus = coefficient * policy_entropy(info, mode)
        else:
            action, _ = self.model.pi(z, policy=self.engine._horizon_actor,
                noise=noise, **self.engine._horizon_actor_options)
            heads = self.q_heads(self.engine._horizon_critic, z, action)
            bonus = 0.
        return expected_reduction(heads, self.cfg.mppi_terminal_q_reduction,
                                  self.model.q_backend.pair_size) + bonus

    def noise(self, shape, seed):
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.randn(shape, generator=generator, device=self.device)

    @torch.no_grad()
    def returns(self, z, actions, actor, horizons, *, seed, coefficient=0., rollouts=None):
        horizons = positive_ints(tuple(horizons), "horizons")
        count, maximum = self.rollouts if rollouts is None else int(rollouts), max(horizons)
        batch, dim = actions.shape
        # All candidate actions share continuation noise; terminal draws use a
        # separate stream so extending H cannot alter earlier horizon labels.
        def depth_noise(length, stream_seed):
            if length == 0:
                return z.new_empty((0, count, 1, dim))
            return torch.stack([self.noise((count, 1, dim), stream_seed if depth == 0
                else solver_seed(stream_seed, "depth", depth)) for depth in range(length)])
        policy_noise = depth_noise(maximum - 1, seed).expand(-1, -1, batch, -1)
        tail_noise = depth_noise(maximum, solver_seed(seed, "tail")).expand(-1, -1, batch, -1)
        with evaluating(self.model, actor):
            return forced_action_mc_returns(z.expand(batch, -1), actions, horizons=horizons,
                transition=self.transition, policy=self.policy(actor), tail=self.tail,
                discount=self.discount, policy_noise=policy_noise, tail_noise=tail_noise,
                critic_target=self.cfg.inner_sac_critic_target, entropy_coefficient=coefficient)

    @torch.no_grad()
    def action_bank(self, z, prior, carried, *, seed, count=8):
        if count < 4:
            raise ValueError("action_count must be at least four.")
        with evaluating(self.model, prior, carried):
            means = [self.model.policy_stats(z, policy=p, **self.bounds)["mean"] for p in (prior, carried)]
            n = (count - 2 + 1) // 2
            noise = self.noise((n, self.cfg.action_dim), seed)
            samples = [self.policy(p)(z.expand(n, -1), noise)[0] for p in (prior, carried)]
        return torch.cat([*means, *samples])[:count].detach()


def raw_entropy_coefficient(snapshot):
    return float(snapshot.alpha) * float(1. if snapshot.actor_loss_scale is None else snapshot.actor_loss_scale)


def solve_fork(wrapped, observation, rng, *, donor=None, branch="fresh", target="online",
               collection_actor=None, capture_rounds=None):
    if branch not in BRANCHES:
        raise ValueError(f"Unknown component branch: {branch}")
    actor_carry, critic_carry = BRANCHES[branch]
    if donor is None and (actor_carry or critic_carry or target == "carried"):
        raise ValueError("Transfer branches require a donor from the preceding solve.")
    engine = wrapped.agent.inner_engine
    engine.reset_for_evaluation(0, reuse_action_pool=True)
    engine.rng.load_training_state_dict(deepcopy(rng))
    actor = donor.state_dict("actor") if actor_carry else None
    critic = donor.state_dict("critic") if critic_carry else None
    target_state = (donor.state_dict("critic") if target == "carried" else
                    engine._critic_base if target == "prior" else target)
    trace = InnerActionTrace(capture_learners=True, learner_rounds=capture_rounds)
    options = dict(actor=actor, critic=critic, target=target_state)
    if collection_actor is not None:
        options["collection_actor"] = collection_actor
    with engine.diagnostic_initialization(**options):
        action, _ = wrapped.predict(observation, deterministic=True, episode_start=True, trace=trace)
    final = [s for s in trace.learner_snapshots if s.stage == "pre_reset"][-1]
    return np.asarray(action).copy(), trace, final


def audit_snapshot(reference, snapshot, z, actions, continuations, *, horizon, seed,
                   evaluation_coefficient=None, epsilons=(.03, .1)):
    actor = snapshot.make_module("actor", reference.device)
    critic = snapshot.make_module("critic", reference.device)
    target = snapshot.make_module("critic_target", reference.device)
    coefficient = raw_entropy_coefficient(snapshot)
    rows = []
    with evaluating(reference.model, actor, critic, target):
        with torch.no_grad():
            predictions = {name: reference.q(module, z.expand(len(actions), -1), actions,
                           reference.cfg.inner_q_target_reduction if name == "target" else
                           reference.cfg.inner_q_actor_reduction).flatten()
                           for name, module in (("online", critic), ("target", target))}
            heads = reference.q_heads(critic, z.expand(len(actions), -1), actions)[..., 0]
        for name, policy in {**continuations, "current": actor}.items():
            samples = reference.returns(z, actions, policy, (horizon,), seed=seed,
                                        coefficient=coefficient)[horizon]
            for kind, values in predictions.items():
                rows.append(dict(continuation=name, critic=kind,
                    reduction=reference.cfg.inner_q_target_reduction if kind == "target" else reference.cfg.inner_q_actor_reduction,
                    **json_value(action_value_metrics(values, samples))))
        def direction_probe(anchor, support):
            # Pre-tanh coordinates respect the bounded action geometry.
            u = torch.atanh(anchor.clamp(-1 + 1e-6, 1 - 1e-6)).detach().requires_grad_(True)
            value = reference.q(critic, z, torch.tanh(u)).sum()
            gradient, = torch.autograd.grad(value, u)
            norm = torch.linalg.vector_norm(gradient)
            direction = gradient / norm.clamp_min(1e-12)
            checks = []
            for epsilon in epsilons:
                pair = torch.cat([torch.tanh(u.detach() + epsilon * direction),
                                  torch.tanh(u.detach() - epsilon * direction)])
                samples = reference.returns(z, pair, actor, (horizon,), seed=seed,
                                            coefficient=coefficient)[horizon]
                checks.append(dict(epsilon=epsilon, support=support, predicted_derivative=float(norm),
                    zero_gradient=bool(norm <= 1e-12),
                    **json_value(paired_directional_metrics(samples[:, 0], samples[:, 1], epsilon))))
            return checks
        directional = direction_probe(actions[:1], "prior_mean")
        with torch.no_grad():
            current_mean = reference.model.policy_stats(z, policy=actor, **reference.bounds)["mean"]
            mean_prediction = float(reference.q(critic, z, current_mean).item())
        current_directional = direction_probe(current_mean, "current_mean")
        mean_samples = reference.returns(z, current_mean, actor, (horizon,), seed=seed,
                                         coefficient=coefficient)[horizon].flatten()
        first_noise = reference.noise((8, reference.cfg.action_dim), solver_seed(seed, "first-action"))
        with torch.no_grad():
            first_actions, logp = reference.policy(actor)(z.expand(8, -1), first_noise)
            predicted = reference.q(critic, z.expand(8, -1), first_actions).flatten()
        # Actor objective includes first-action entropy; Q_h itself excludes it.
        scoring_coefficient = coefficient if evaluation_coefficient is None else evaluation_coefficient
        samples = reference.returns(z, first_actions, actor, (horizon,),
            seed=solver_seed(seed, "actor-objective"), coefficient=scoring_coefficient)[horizon]
        entropy_bonus = -scoring_coefficient * logp.flatten() if reference.cfg.inner_entropy_enabled else 0.
        objective = (samples + entropy_bonus).mean()
        predicted_objective = (predicted + entropy_bonus).mean()
    return dict(stage=snapshot.stage, round=snapshot.round_index,
        actor_updates=snapshot.actor_updates, critic_updates=snapshot.critic_updates,
        alpha=snapshot.alpha, actor_loss_scale=snapshot.actor_loss_scale,
        model_objective_entropy_coefficient=scoring_coefficient,
        critic_checks=rows, directional=directional, q_heads=json_value(heads),
        current_action_directional=current_directional,
        current_mean_action=current_mean.flatten().tolist(),
        current_mean_critic_value=mean_prediction,
        current_mean_reference_return=float(mean_samples.mean()),
        current_mean_reference_se=float(mean_samples.std(unbiased=True) / math.sqrt(len(mean_samples)))
            if len(mean_samples) > 1 else None,
        model_actor_objective=float(objective), critic_actor_objective=float(predicted_objective),
        model_actor_objective_conditional_mc_se=float(samples.mean(1).std(unbiased=True) / math.sqrt(len(samples)))
            if len(samples) > 1 else None,
        model_actor_objective_se_semantics="Conditional on eight paired first-action samples; excludes their sampling uncertainty.")


def portability_audit(reference, donor, current_z, previous_z, previous_action, *, horizon, seed, count):
    prior = deepcopy(reference.engine._actor_base).eval()
    actor = donor.make_module("actor", reference.device)
    critic = donor.make_module("critic", reference.device)
    with torch.no_grad(), evaluating(reference.model):
        _, imagined, _ = reference.transition(previous_z, previous_action)
    rows = []
    for support, z in (("previous_root", previous_z), ("imagined_successor", imagined),
                       ("actual_successor", current_z)):
        actions = reference.action_bank(z, prior, actor, seed=seed, count=count)
        for policy_name, policy in (("prior", prior), ("carried", actor)):
            samples = reference.returns(z, actions, policy, (horizon,), seed=seed,
                                        coefficient=raw_entropy_coefficient(donor))[horizon]
            with torch.no_grad(), evaluating(reference.model, critic):
                values = reference.q(critic, z.expand(len(actions), -1), actions).flatten()
            rows.append(dict(support=support, continuation=policy_name,
                             **json_value(action_value_metrics(values, samples))))
    if horizon == 1:
        return dict(support=rows, horizon_shift={"applicable": False,
            "reason": "H=1 has no positive-horizon suffix after the first transition."})
    actions = reference.action_bank(imagined, prior, actor, seed=seed, count=count)
    samples = reference.returns(imagined, actions, actor, (horizon - 1, horizon),
        seed=seed, coefficient=raw_entropy_coefficient(donor))
    difference = samples[horizon] - samples[horizon - 1]
    with torch.no_grad(), evaluating(reference.model, critic):
        values = reference.q(critic, imagined.expand(len(actions), -1), actions).flatten()
    return dict(support=rows, horizon_shift=dict(applicable=True, old_horizon=horizon - 1,
        new_horizon=horizon, actionwise_target_change=json_value(difference.mean(0)),
        paired_mc_se=json_value(difference.std(0, unbiased=True) / math.sqrt(len(difference)))
            if len(difference) > 1 else None,
        old_slice=json_value(action_value_metrics(values, samples[horizon - 1])),
        new_slice=json_value(action_value_metrics(values, samples[horizon]))))


def stationary_audit(reference, donor, root_z, *, horizon, seed, steps=32, batch_size=32,
                     state_count=16, action_count=8):
    """Common fixed targets and held-out states, with no bootstrap feedback."""
    if state_count < 4 or state_count % 2:
        raise ValueError("stationary state_count must be even and >=4.")
    prior = deepcopy(reference.engine._actor_base).eval()
    carried = donor.make_module("actor", reference.device)
    states = [root_z.detach().clone()]
    with torch.no_grad(), evaluating(reference.model, prior, carried):
        trajectory_count = max(2, math.ceil((state_count - 1) / max(2, horizon)))
        z = root_z.expand(trajectory_count, -1).clone()
        noise = reference.noise((trajectory_count, reference.cfg.action_dim), seed)
        for step in range(max(2, horizon)):
            policy = prior if step % 2 == 0 else carried
            action, _ = reference.policy(policy)(z, noise.roll(step, 0))
            _, z, _ = reference.transition(z, action)
            states.extend(z.split(1))
    states = torch.cat(states)[:state_count]
    # Interleave depths across train/held-out sets; no outcome-based split.
    order = torch.cat([torch.arange(0, state_count, 2), torch.arange(1, state_count, 2)]).to(states.device)
    states = states[order]
    features, labels, target_actions = [], [], []
    for index, z in enumerate(states.split(1)):
        actions = reference.action_bank(z, prior, carried, seed=solver_seed(seed, index), count=action_count)
        samples = reference.returns(z, actions, prior, (horizon,), seed=solver_seed(seed, "label", index),
                                    coefficient=raw_entropy_coefficient(donor))[horizon]
        means = samples.mean(0)
        features.append(torch.cat([z.expand(len(actions), -1), actions], -1))
        labels.append(means[:, None])
        target_actions.append(actions[means.argmax()].detach())
    features, labels = torch.cat(features), torch.cat(labels)
    labels = labels.expand(-1, reference.model.q_backend.num_q)
    target_actions = torch.stack(target_actions)
    split = state_count // 2
    generator = torch.Generator().manual_seed(solver_seed(seed, "batches"))
    critic_batches = torch.randint(split * action_count, (steps, batch_size), generator=generator)
    actor_batches = torch.randint(split, (steps, batch_size), generator=generator)
    results = {}
    for component in ("critic", "actor"):
        modules = dict(prior=reference.engine._critic_base if component == "critic" else prior,
                       carried=donor.make_module(component, reference.device))
        for name, module in modules.items():
            if component == "critic":
                def predict(net, data):
                    latent = reference.cfg.latent_dim
                    return reference.q_heads(net, data[:, :latent], data[:, latent:])[..., 0].transpose(0, 1)
                def features_fn(net, data):
                    return torch.cat([head[:-1](data) for head in net.modules_list], dim=-1)
                x, y, boundary, batches = features, labels, split * action_count, critic_batches
            else:
                def predict(net, data):
                    return reference.model.policy_stats(data, policy=net, **reference.bounds)["mean"]
                def features_fn(net, data):
                    return net[:-1](data)
                x, y, boundary, batches = states, target_actions, split, actor_batches
            fitted = fit_stationary_targets(module, predict, x[:boundary], y[:boundary],
                x[boundary:], y[boundary:], batch_indices=batches,
                learning_rate=float(getattr(reference.cfg, f"inner_{component}_lr")),
                seed=solver_seed(seed, "fit"), trainable_selector=lambda name, parameter: True,
                feature_fn=features_fn)
            results[f"{component}_{name}"] = fitted["curve"]
    return dict(curves=results, semantics="Decoded per-head Q regression; actor mean-action imitation. "
        "Targets fixed from independent prior-continuation model rollouts. Held-out states never optimized.",
        state_count=state_count, action_count=action_count, steps=steps)


def audit_root(wrapped, observation, rng, donor, *, previous_observation, previous_action,
               horizon, seed, options, env=None, simulator_snapshot=None):
    reference = Reference(wrapped, rollouts=options["mc_rollouts"])
    z = reference.encode(np.asarray(observation)[None])
    prior = deepcopy(reference.engine._actor_base).eval().requires_grad_(False)
    carried = donor.make_module("actor", reference.device)
    actions = reference.action_bank(z, prior, carried, seed=seed, count=options["action_count"])
    results, finals = [], {}
    evaluation_coefficient = None
    for lane in options["data_lanes"]:
        collection = prior if lane == "common" else None
        for branch in BRANCHES:
            action, trace, final = solve_fork(wrapped, observation, rng, donor=donor, branch=branch,
                collection_actor=collection, capture_rounds=options.get("capture_rounds"))
            if evaluation_coefficient is None:
                evaluation_coefficient = raw_entropy_coefficient(next(
                    snap for snap in trace.learner_snapshots if snap.stage == "initial"))
            checks = [audit_snapshot(reference, snap, z, actions, {"prior": prior, "carried": carried},
                horizon=horizon, seed=seed, evaluation_coefficient=evaluation_coefficient)
                for snap in trace.learner_snapshots if snap.stage != "pre_reset"]
            # Always score the final snapshot even when a custom round filter
            # omitted the final post_round capture.
            final_check = audit_snapshot(reference, final, z, actions, {"prior": prior, "carried": carried},
                                         horizon=horizon, seed=seed,
                                         evaluation_coefficient=evaluation_coefficient)
            hashes = [event["replay_sha256"] for event in trace.events if "replay_sha256" in event]
            if lane == "common" and branch != "fresh":
                baseline = next(row["replay_sha256"] for row in results
                                if row["data_lane"] == lane and row["branch"] == "fresh")
                if not hashes or hashes != baseline:
                    raise RuntimeError("Common-data diagnostic branches received different replay.")
            row = dict(branch=branch, data_lane=lane, target_initialization="selected_online",
                       action=action.tolist(), stages=checks, final=final_check,
                       replay_sha256=hashes,
                       solver_metrics=deepcopy(wrapped.agent.last_inner_metrics))
            results.append(row)
            finals[(lane, branch)] = (action, final)
    targets = []
    if options.get("target_cross", True):
        for branch in ("fresh", "critic"):
            for target in ("prior", "carried"):
                action, trace, final = solve_fork(wrapped, observation, rng, donor=donor,
                    branch=branch, target=target, collection_actor=prior,
                    capture_rounds=options.get("capture_rounds"))
                targets.append(dict(online="carried" if branch == "critic" else "prior", target=target,
                    action=action.tolist(), stages=[audit_snapshot(reference, snap, z, actions,
                        {"prior": prior, "carried": carried}, horizon=horizon, seed=seed,
                        evaluation_coefficient=evaluation_coefficient)
                        for snap in trace.learner_snapshots if snap.stage != "pre_reset"],
                    final=audit_snapshot(reference, final, z, actions,
                        {"prior": prior, "carried": carried}, horizon=horizon, seed=seed,
                        evaluation_coefficient=evaluation_coefficient)))
    portability = portability_audit(reference, donor, z,
        reference.encode(np.asarray(previous_observation)[None]),
        torch.as_tensor(previous_action, device=reference.device, dtype=torch.float32)[None],
        horizon=horizon, seed=seed, count=options["action_count"])
    stationary = stationary_audit(reference, donor, z, horizon=horizon, seed=seed,
        steps=options["fit_steps"], state_count=options["fit_states"], action_count=options["action_count"])
    real = []
    if options.get("real_rollouts", 0):
        if env is None or simulator_snapshot is None:
            raise ValueError("Real calibration requires the saved simulator root.")
        real = audit_real(reference, env, simulator_snapshot, z, prior, finals,
                          horizon=horizon, seed=seed, options=options)
    replanning = []
    if options.get("replan_steps", 0):
        replanning = audit_replanning(reference, env, simulator_snapshot, finals, rng,
            horizon=horizon, seed=seed, options=options)
    return dict(horizon=horizon, seed=seed, actions=actions.tolist(), branches=results,
                target_cross=targets, portability=portability, stationary=stationary,
                real=real, replanning=replanning)


def audit_real(reference, env, simulator_snapshot, z, prior, finals, *, horizon, seed, options):
    """Matched model A, real-prefix/bootstrap B, and finite-real-tail C.

    Every model/real pair uses the same forced first action and Gaussian draws.
    All model actions and entropies use normalized action units; only simulator
    actions are rescaled. A-B includes both reward and endpoint-state model
    errors. B-C includes finite-tail truncation; C is not replanning performance.
    """
    from utils.transfer_diagnostic_real import evaluate_real_prefix, PolicyOutput
    wrapped, cfg, model, engine = reference.wrapped, reference.cfg, reference.model, reference.engine
    boundary_entropy = cfg.inner_terminal_entropy == "outer"
    terminal_source = "sac" if boundary_entropy else cfg.inner_horizon_critic_source
    terminal_soft = terminal_source == "sac" and cfg.outer_critic_target == "entropy_augmented"
    if (boundary_entropy or terminal_soft) and cfg.outer_actor_entropy_mode != "squashed":
        raise ValueError("Real soft-tail calibration currently requires squashed outer entropy.")
    outer_bounds = {name: getattr(cfg, name) for name in ("log_std_mapping", "log_std_min", "log_std_max")}
    tail_actor = model._pi if boundary_entropy else engine._horizon_actor
    tail_bounds = outer_bounds if boundary_entropy else {**outer_bounds, **engine._horizon_actor_options}
    tail_critic = model._Qs if boundary_entropy else engine._horizon_critic
    terminal_alpha = 0.0
    if terminal_soft or boundary_entropy:
        terminal_alpha = float(wrapped.agent.alpha.detach())
        if wrapped.agent.actor_loss_scale_enabled:
            terminal_alpha *= float(wrapped.agent.actor_loss_scale.detach().reshape(()))

    def actor_callback(policy, bounds):
        def apply(obs, noise, remaining_horizon):
            with torch.no_grad(), evaluating(model, policy):
                latent = reference.encode(obs)
                actions, info = model.pi(latent, policy=policy,
                    noise=torch.as_tensor(noise, device=reference.device, dtype=torch.float32), **bounds)
                env_actions = np.stack([wrapped._unscale_action(a) for a in actions.cpu().numpy()])
                return PolicyOutput(env_actions, info["log_prob"].cpu().numpy())
        return apply

    def terminal_q(obs, actions):
        with torch.no_grad(), evaluating(model, tail_critic):
            latent = reference.encode(obs)
            normalized = np.stack([wrapped._scale_action(a) for a in actions])
            acts = torch.as_tensor(normalized, device=reference.device, dtype=torch.float32)
            return reference.q(tail_critic, latent, acts, cfg.mppi_terminal_q_reduction).flatten().cpu().numpy()

    rows = []
    for (lane, branch), (action, final) in finals.items():
        if lane != "natural":
            continue
        actor = final.make_module("actor", reference.device)
        coefficient = raw_entropy_coefficient(final)
        normalized_first = torch.as_tensor(wrapped._scale_action(action), device=reference.device,
                                           dtype=torch.float32)[None]
        for replicate in range(options["real_rollouts"]):
            # Separate streams keep earlier prefix/tail draws unchanged when H
            # or tail length is changed in a matched-horizon diagnostic.
            prefix_rng = np.random.default_rng(solver_seed(seed, "real-prefix", replicate))
            tail_rng = np.random.default_rng(solver_seed(seed, "real-tail", replicate))
            prefix_noise = prefix_rng.standard_normal((horizon, cfg.action_dim)).astype(np.float32)
            tail_noise = tail_rng.standard_normal((options["real_tail_steps"], cfg.action_dim)).astype(np.float32)
            policy_noise = torch.as_tensor(prefix_noise[1:], device=reference.device)[:, None, None, :]
            model_tail_noise = torch.as_tensor(tail_noise[:1], device=reference.device)
            model_tail_noise = model_tail_noise[None, :, None, :].expand(horizon, -1, -1, -1)
            with torch.no_grad(), evaluating(model, actor, tail_actor, tail_critic):
                estimate = forced_action_mc_returns(z, normalized_first, horizons=(horizon,),
                    transition=reference.transition, policy=reference.policy(actor, final.policy_bounds),
                    tail=reference.tail, discount=reference.discount,
                    policy_noise=policy_noise, tail_noise=model_tail_noise,
                    critic_target=cfg.inner_sac_critic_target, entropy_coefficient=coefficient)[horizon]
                predicted = float(estimate.item())
            row = evaluate_real_prefix(env, simulator_snapshot, first_action=action,
                prefix_actor=actor_callback(actor, final.policy_bounds),
                prior_actor=actor_callback(tail_actor, tail_bounds),
                terminal_q=terminal_q, prefix_noise=prefix_noise, tail_noise=tail_noise,
                discount=reference.discount,
                objective="soft" if cfg.inner_sac_critic_target == "entropy_augmented" else "reward",
                alpha=coefficient, terminal_objective="soft" if terminal_soft else "reward",
                terminal_alpha=terminal_alpha, terminal_first_action_entropy=boundary_entropy,
                continuing=True)
            bootstrapped, measured = row["real_bootstrapped_return"], row["real_mc_return"]
            rows.append(dict(branch=branch, data_lane=lane, replicate=replicate,
                predicted_model_return=predicted,
                model_prefix_error=None if bootstrapped is None else predicted - bootstrapped,
                terminal_value_error=None if bootstrapped is None or measured is None else bootstrapped - measured,
                total_prediction_error=None if measured is None else predicted - measured,
                terminal_critic_source=terminal_source,
                terminal_actor_source="sac" if boundary_entropy else getattr(cfg, "inner_horizon_actor_source", "sac"),
                terminal_head_reduction="exact_expectation_over_head_selection",
                entropy_action_units="normalized", **json_value(row)))
    return rows


def audit_replanning(reference, env, simulator_snapshot, finals, rng, *, horizon, seed, options):
    """Separate first-action and carried-memory effects in actual replanning.

    Action-only branches all use a fresh future controller. Memory-only branches
    all execute the fresh arm's first action and subsequently use the same joint
    carry rule, installing the respective arm's complete final actor/critic.
    Full branches execute each arm's action and retain its own component rule.
    These three estimands are intentionally distinct. Future solver streams are
    paired within replicate, not counted as independent environment episodes.
    """
    from utils.transfer_diagnostic_real import evaluate_continuation
    wrapped = reference.wrapped
    steps, repeats = int(options["replan_steps"]), int(options["replan_repeats"])
    if steps < 1 or repeats < 1:
        raise ValueError("Replanning requires positive steps and repeats.")
    natural = {branch: value for (lane, branch), value in finals.items() if lane == "natural"}
    if set(natural) != set(BRANCHES):
        raise ValueError("Replanning requires all four natural-data final branches.")
    engine = wrapped.agent.inner_engine
    saved_rng = deepcopy(engine.rng.training_state_dict())
    rows = []
    try:
        for repeat in range(repeats):
            # The supplied rng belongs before the current solve. A future
            # intervention instead starts from its own named stream for every
            # repeat, shared across arms and the three intervention types.
            future_seed = solver_seed(seed, "replanning-future", repeat)
            initial_rng = engine._new_rng(future_seed).training_state_dict()
            for intervention in ("first_action_only", "memory_only", "full"):
                for branch, (action, final) in natural.items():
                    first = natural["fresh"][0] if intervention == "memory_only" else action
                    rule = ("fresh" if intervention == "first_action_only" else
                            "joint" if intervention == "memory_only" else branch)
                    initial_donor = None if rule == "fresh" else final

                    def factory(donor=initial_donor, carry_rule=rule):
                        private_rng = deepcopy(initial_rng)
                        current_donor = donor

                        def controller(observation, offset):
                            nonlocal private_rng, current_donor
                            next_action, _, current_donor = solve_fork(wrapped, observation,
                                private_rng, donor=current_donor, branch=carry_rule,
                                capture_rounds=())
                            private_rng = deepcopy(engine.rng.training_state_dict())
                            return next_action
                        return controller

                    row = evaluate_continuation(env, simulator_snapshot, first_action=first,
                        controller_factory=factory, steps=steps, discount=float(wrapped.agent.discount),
                        intervention=intervention, controller_identity=f"{rule}_component_rule",
                        memory_identity="none" if initial_donor is None else f"natural_{branch}_final")
                    rows.append(dict(branch=branch, data_lane="natural", horizon=horizon,
                        future_replicate=repeat, future_solver_seed=future_seed,
                        future_rng_source="independent_named_rng",
                        future_component_rule=rule, first_action_source="fresh" if intervention == "memory_only" else branch,
                        memory_source="none" if initial_donor is None else branch,
                        remaining_state_lifecycle="fresh_replay_optimizer_temperature_target_from_online",
                        **json_value(row)))
    finally:
        engine.rng.load_training_state_dict(saved_rng)
    return rows
