"""Read-only, root-matched action scoring for the 800k return-Q audit.

These helpers never run SAC updates or choose source trajectories. All model
scores force the candidate first action, then use the same frozen SAC prior.
Thus the H3 score is an action diagnostic, not MPPI's optimized sequence value.
Real tails have an explicit finite cutoff and no appended value bootstrap.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from numbers import Integral

import numpy as np
import torch

from RL.tdmpc2_core.mppi import MPPIModelCallbacks, mppi_plan
from utils.ambi_benchmark import solver_seed
from utils.transfer_diagnostic_metrics import action_value_metrics, forced_action_mc_returns
from utils.transfer_diagnostic_real import PolicyOutput, evaluate_real_prefix
from utils.transfer_diagnostics import evaluating, json_value


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _horizons(values):
    values = tuple(_integer(value, "horizon") for value in values)
    if not values or len(set(values)) != len(values):
        raise ValueError("Horizons must be nonempty and unique.")
    return values


def _reference(reference):
    cfg = reference.cfg
    if cfg.inner_sac_critic_target != "reward_only" or cfg.inner_terminal_entropy != "none":
        raise ValueError("This action audit requires return-only targets without terminal entropy.")
    if getattr(cfg, "inner_horizon_critic_source", None) != "aux_return":
        raise ValueError("This action audit requires the frozen auxiliary return critic.")


def _bank(reference, observation, actions, labels):
    _reference(reference)
    labels = tuple(labels)
    values = torch.as_tensor(actions, dtype=torch.float32, device=reference.device).detach().clone()
    if (values.ndim != 2 or values.shape != (len(labels), int(reference.cfg.action_dim))
            or not labels or any(not isinstance(name, str) or not name for name in labels)
            or len(set(labels)) != len(labels) or not torch.isfinite(values).all()
            or bool((values.abs() > 1).any())):
        raise ValueError("Actions require unique nonempty labels and finite normalized [N,A] values in [-1,1].")
    with _frozen(reference):
        root = reference.encode(np.asarray(observation)[None]).detach()
    if root.ndim != 2 or len(root) != 1 or not torch.isfinite(root).all():
        raise ValueError("One finite encoded root is required.")
    return root, values, labels


@contextmanager
def _frozen(reference, *modules):
    device = torch.device(reference.device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), evaluating(
            reference.model, reference.engine._actor_base,
            reference.engine._horizon_actor, reference.engine._horizon_critic, *modules):
        yield


def _mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Scoring samples must be finite nonempty vectors.")
    return dict(mean=float(values.mean()),
                se=float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None,
                samples=len(values))


def _prior_policy(reference):
    """Use the frozen owner's bounds, not adapted inner-actor bounds."""
    def call(z, noise):
        action, info = reference.model.pi(z, policy=reference.engine._actor_base,
            noise=noise, **reference.engine._actor_options)
        return action, info["log_prob"]
    return call


def _prior_returns(reference, root, actions, horizons, seed, count):
    maximum, batch, dimension = max(horizons), len(actions), int(reference.cfg.action_dim)
    def noise(length, stream_seed):
        if length == 0:
            return root.new_empty((0, count, batch, dimension))
        return torch.stack([reference.noise((count, 1, dimension), stream_seed if depth == 0
            else solver_seed(stream_seed, "depth", depth)) for depth in range(length)]).expand(-1, -1, batch, -1)
    return forced_action_mc_returns(root.expand(batch, -1), actions, horizons=horizons,
        transition=reference.transition, policy=_prior_policy(reference), tail=reference.tail,
        discount=reference.discount, policy_noise=noise(maximum - 1, seed),
        tail_noise=noise(maximum, solver_seed(seed, "tail")), critic_target="reward_only")


@torch.no_grad()
def mppi_candidate(reference, observation, *, horizon, seed, planner_options,
                   previous_mean=None):
    """Return one mean-executed MPPI candidate without a persistent planner.

    All population/iteration settings are explicit. The default is a cold
    solve; a supplied previous_mean is read-only and recorded as a warm start.
    Planning uses actual random-pair reduction, while bank scoring integrates
    that pair choice exactly through Reference.q/Reference.tail.
    """
    _reference(reference)
    horizon, seed = _integer(horizon, "horizon"), _integer(seed, "seed", 0)
    required = {"iterations", "num_samples", "num_elites", "num_pi_trajs",
                "temperature", "min_std", "max_std"}
    if set(planner_options) != required:
        raise ValueError(f"Planner options must contain exactly {sorted(required)}.")
    model, engine = reference.model, reference.engine

    def transition(z, actions):
        reward, successor, _ = reference.transition(z, actions)
        return successor, reward

    callbacks = MPPIModelCallbacks(
        action_dim=int(reference.cfg.action_dim),
        dynamics=lambda z, a: reference.transition(z, a)[1],
        reward=lambda z, a: reference.transition(z, a)[0], transition=transition,
        policy=lambda z, *, generator: model.pi_action(
            z, policy=engine._actor_base, generator=generator, **engine._actor_options),
        terminal_policy=lambda z, *, generator: model.pi_action(
            z, policy=engine._horizon_actor, generator=generator, **engine._horizon_actor_options),
        terminal_q=lambda z, a, *, reduction, generator: model.Q(
            z, a, qs=engine._horizon_critic._forward_eager,
            reduction=reduction, generator=generator),
        termination=(lambda z: model.termination(z)) if reference.cfg.episodic else None)
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    with _frozen(reference):
        root = reference.encode(np.asarray(observation)[None])
        previous_mean = (None if previous_mean is None else
            torch.as_tensor(previous_mean, device=reference.device, dtype=root.dtype).detach().clone())
        result = mppi_plan(root_z=root, callbacks=callbacks, horizon=horizon,
            discount=reference.discount, q_reduction=reference.cfg.mppi_terminal_q_reduction,
            termination_threshold=reference.cfg.inner_termination_threshold,
            generator=generator, previous_mean=previous_mean, t0=previous_mean is None,
            eval_mode=True, **dict(planner_options))
    return json_value(dict(normalized_action=result.action, proposal_mean=result.next_mean,
        next_mean=result.next_mean,
        metrics=result.metrics, model_steps=result.model_steps, horizon=horizon, seed=seed,
        warm_start=previous_mean is not None, execution="optimized_proposal_mean",
        planner_options=dict(planner_options)))


@torch.no_grad()
def score_action_bank(reference, observation, *, actions, labels, critics, seed,
                      mc_rollouts=32, horizons=(1, 3), max_expanded_batch=512,
                      baseline_index=0):
    """Score a common root action bank, with noise invariant to chunk size.

    Use distinct caller-supplied seeds for selection and held-out scoring. The
    empirical bank winner in critic summaries uses these scoring draws and is
    descriptive; it must not be presented as an independently validated optimum.
    Returned per-action draws permit paired MC differences without treating
    model draws as independent environment episodes.
    """
    root, actions, labels = _bank(reference, observation, actions, labels)
    if reference.cfg.inner_q_actor_reduction not in {"mean", "mean_pair", "mean_all"}:
        raise ValueError("This audit's learned-Q comparison requires mean-pair actor reduction.")
    horizons = _horizons(horizons)
    seed, mc_rollouts = _integer(seed, "seed", 0), _integer(mc_rollouts, "mc_rollouts", 2)
    max_expanded_batch = _integer(max_expanded_batch, "max_expanded_batch", mc_rollouts)
    baseline_index = _integer(baseline_index, "baseline_index", 0)
    if baseline_index >= len(actions) or not critics or any(not isinstance(k, str) or not k for k in critics):
        raise ValueError("Critics and a valid baseline action are required.")
    chunks = max(1, max_expanded_batch // mc_rollouts)
    samples = {h: [] for h in horizons}
    head_values = {name: [] for name in critics}
    with _frozen(reference, *critics.values()):
        for start in range(0, len(actions), chunks):
            batch = actions[start:start + chunks]
            values = _prior_returns(reference, root, batch, horizons, seed, mc_rollouts)
            for h in horizons:
                if values[h].shape != (mc_rollouts, len(batch)) or not torch.isfinite(values[h]).all():
                    raise ValueError("Invalid model return samples.")
                samples[h].append(values[h].detach().cpu())
            for name, critic in critics.items():
                heads = reference.q_heads(critic, root.expand(len(batch), -1), batch)[..., 0]
                if heads.ndim != 2 or heads.shape[1] != len(batch) or not torch.isfinite(heads).all():
                    raise ValueError("Invalid critic head values.")
                head_values[name].append(heads.detach().cpu())
    samples = {h: torch.cat(values, dim=1).numpy() for h, values in samples.items()}
    heads = {name: torch.cat(values, dim=1) for name, values in head_values.items()}
    # The configured actor mean-pair has the same expectation as all-head mean.
    means = {name: values.mean(0).numpy() for name, values in heads.items()}
    rows = []
    for index, label in enumerate(labels):
        models = {}
        for h in horizons:
            draws = samples[h][:, index]
            gain = _mean_se(draws - samples[h][:, baseline_index])
            models[f"h{h}"] = dict(**_mean_se(draws), draws=draws.tolist(),
                gain_vs_baseline_mean=gain["mean"], gain_vs_baseline_se=gain["se"])
        rows.append(dict(label=label, action=actions[index].cpu().tolist(),
            q={name: float(value[index]) for name, value in means.items()},
            q_head_std={name: float(value[:, index].std(unbiased=False)) for name, value in heads.items()},
            model=models))
    summaries = []
    for name, values in means.items():
        for h in horizons:
            metrics = action_value_metrics(values, samples[h], baseline_index=baseline_index)
            # Per-action predictions/draws live above; retain only summary metrics here.
            metrics = {k: v for k, v in metrics.items() if k not in {
                "reference_mean", "reference_se", "reference_relative_se", "prediction_mean", "prediction_draw_std"}}
            summaries.append(dict(name=name, horizon=h,
                top_action_label=labels[metrics["top_action_index"]],
                reference_top_action_label=labels[metrics["reference_top_action_index"]], **metrics))
    return dict(actions=rows, critics=summaries, noise_seed=seed, mc_rollouts=mc_rollouts,
        baseline_label=labels[baseline_index], horizons=list(horizons),
        max_expanded_batch=max_expanded_batch,
        semantics=dict(states="one shared real root; no successor-state averaging",
            continuation="frozen SAC prior after the forced first action",
            terminal="frozen auxiliary return Q; exact expectation over critic-pair selection",
            entropy="none", mc_uncertainty="conditional on fixed root and actions",
            winner="empirical bank winner; caller must separate selection and validation seeds",
            h3="fixed-prior continuation, not the optimized MPPI sequence"))


@torch.no_grad()
def score_real_candidates(reference, env, snapshot, *, actions, labels, seed,
                          real_rollouts=8, tail_steps=500, horizons=(1, 3),
                          max_expanded_batch=512, baseline_index=0):
    """Paired model A, real-prefix/frozen-Q B, and finite-real-prior-tail C.

    The same post-first-action noise sequence is used across candidates and
    horizons. Model and real prefix/terminal actions receive identical draws.
    H has H+tail_steps real decisions before its explicit cutoff. All real
    branches enable continuing calibration and restore the caller's simulator.
    """
    from utils.transfer_diagnostic_real import capture_simulator_snapshot
    # Restore only within a preserving context to obtain the root observation;
    # callers need not expose simulator-specific snapshot internals.
    from utils.transfer_diagnostic_real import preserve_simulator, restore_simulator_snapshot
    with preserve_simulator(env):
        observation = restore_simulator_snapshot(env, snapshot)
    root, actions, labels = _bank(reference, observation, actions, labels)
    horizons = _horizons(horizons)
    seed, count = _integer(seed, "seed", 0), _integer(real_rollouts, "real_rollouts", 2)
    tail_steps = _integer(tail_steps, "tail_steps")
    max_expanded_batch = _integer(max_expanded_batch, "max_expanded_batch", count)
    baseline_index = _integer(baseline_index, "baseline_index", 0)
    if baseline_index >= len(actions):
        raise ValueError("Invalid baseline action.")
    maximum, dimension = max(horizons), int(reference.cfg.action_dim)
    noise = np.stack([np.random.default_rng(solver_seed(seed, "real-continuation", i)).standard_normal(
        (maximum + tail_steps, dimension)).astype(np.float32) for i in range(count)])
    r, prior, tail_actor = reference, reference.engine._actor_base, reference.engine._horizon_actor

    def actor_callback(policy, bounds):
        def call(observations, draws, remaining):
            latent = r.encode(observations)
            action, info = r.model.pi(latent, policy=policy,
                noise=torch.as_tensor(draws, device=r.device, dtype=torch.float32), **bounds)
            return PolicyOutput(np.stack([r.wrapped._unscale_action(a) for a in action.cpu().numpy()]),
                                info["log_prob"].cpu().numpy())
        return call

    def terminal_q(observations, env_actions):
        normalized = np.stack([r.wrapped._scale_action(a) for a in env_actions])
        return r.q(r.engine._horizon_critic, r.encode(observations),
            torch.as_tensor(normalized, device=r.device, dtype=torch.float32),
            r.cfg.mppi_terminal_q_reduction).flatten().cpu().numpy()

    before = capture_simulator_snapshot(env).sha256
    model_parts = {h: [] for h in horizons}
    replicates = []
    with _frozen(r):
        chunk = max(1, max_expanded_batch // count)
        for start in range(0, len(actions), chunk):
            batch = actions[start:start + chunk]
            prefix = torch.as_tensor(noise[:, 1:maximum], device=r.device).transpose(0, 1)[:, :, None, :].expand(-1, -1, len(batch), -1)
            terminal = torch.as_tensor(noise[:, 1:maximum + 1], device=r.device).transpose(0, 1)[:, :, None, :].expand(-1, -1, len(batch), -1)
            values = forced_action_mc_returns(root.expand(len(batch), -1), batch, horizons=horizons,
                transition=r.transition, policy=_prior_policy(r), tail=r.tail,
                discount=r.discount, policy_noise=prefix, tail_noise=terminal,
                critic_target="reward_only", entropy_coefficient=0.)
            for h in horizons:
                model_parts[h].append(values[h].cpu().numpy())
        model_values = {h: np.concatenate(parts, axis=1) for h, parts in model_parts.items()}
        prior_callback = actor_callback(prior, r.engine._actor_options)
        tail_callback = actor_callback(tail_actor, r.engine._horizon_actor_options)
        for index, label in enumerate(labels):
            first = r.wrapped._unscale_action(actions[index].cpu().numpy())
            for h in horizons:
                for replicate in range(count):
                    row = evaluate_real_prefix(env, snapshot, first_action=first,
                        prefix_actor=prior_callback, prior_actor=tail_callback, terminal_q=terminal_q,
                        prefix_noise=noise[replicate, :h], tail_noise=noise[replicate, h:h + tail_steps],
                        discount=r.discount, objective="reward", terminal_objective="reward",
                        continuing=True)
                    a, b, c = float(model_values[h][replicate, index]), row["real_bootstrapped_return"], row["real_mc_return"]
                    replicates.append(dict(label=label, replicate=replicate,
                        model_return=a, model_prefix_error=None if b is None else a-b,
                        terminal_error=None if b is None or c is None else b-c,
                        total_error=None if c is None else a-c, **row))
    if capture_simulator_snapshot(env).sha256 != before:
        raise RuntimeError("Real action scoring changed the caller's simulator.")
    summaries = []
    fields = {"model": "model_return", "real_prefix_value": "real_bootstrapped_return",
              "real_tail": "real_mc_return", "model_prefix_bias": "model_prefix_error",
              "terminal_bias": "terminal_error", "total_bias": "total_error"}
    for label in labels:
        for h in horizons:
            selected = [row for row in replicates if row["label"] == label and row["horizon"] == h]
            baseline = [row for row in replicates if row["label"] == labels[baseline_index] and row["horizon"] == h]
            result = dict(label=label, horizon=h, requested_replicates=count,
                complete=all(row["mc_complete"] and row["real_bootstrapped_return"] is not None for row in selected),
                terminated_replicates=sum(row["terminated"] for row in selected),
                truncated_replicates=sum(row["truncated"] for row in selected),
                real_decisions_requested=h+tail_steps)
            for name, key in fields.items():
                values = [row[key] for row in selected if row[key] is not None]
                stats = _mean_se(values) if values else dict(mean=None, se=None, samples=0)
                result.update({name+"_"+k: v for k, v in stats.items()})
            for name, key in (("model", "model_return"), ("real_prefix_value", "real_bootstrapped_return"), ("real_tail", "real_mc_return")):
                values = [row[key]-base[key] for row, base in zip(selected, baseline) if row[key] is not None and base[key] is not None]
                stats = _mean_se(values) if values else dict(mean=None, se=None, samples=0)
                result.update({name+"_gain_vs_baseline_"+k: v for k, v in stats.items()})
            summaries.append(result)
    return dict(actions=summaries, replicates=json_value(replicates), noise_seed=seed,
        baseline_label=labels[baseline_index], real_rollouts=count, tail_steps=tail_steps,
        horizons=list(horizons), continuing=True, simulator_unchanged=True,
        semantics=dict(model="forced first action, frozen-prior continuation, frozen auxiliary return-Q boundary",
            real_prefix_value="real H-step prefix plus the same frozen return-Q at its actual endpoint",
            real_tail="real H-step prefix plus finite sampled prior tail, no appended Q",
            uncertainty="paired continuation-noise MC at one fixed simulator root; not independent episodes",
            cutoff="H+tail_steps decisions; finite cutoff is not infinite-horizon ground truth",
            cross_horizon_noise="same absolute post-first-action noise; different H+tail_steps cutoffs"))
