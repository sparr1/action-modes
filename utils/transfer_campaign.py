"""Lean full-episode discovery of transfer between successive inner SAC solves.

The previous solve is the only weight donor. Networks blend toward frozen
priors or retain individual parameters with a Bernoulli mask; replay, optimizer
and temperature reuse are explicit arm properties. Optional observational
diagnostics report their overhead separately from controller work.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import torch

from utils.ambi_benchmark import read_json, solver_seed


PROTOCOL = "inner-sac-transfer-discovery-v1"
METRIC_POLICIES = ("legacy", "all_scalars")
METRICS = (
    "inner_actor_loss", "inner_critic_loss", "inner_alpha", "inner_alpha_initial",
    "inner_alpha_final", "inner_actor_grad_norm", "inner_critic_grad_norm",
    "inner_actor_grad_norm_mean", "inner_critic_grad_norm_mean", "inner_actor_loss_scale",
    "inner_actor_optimizer_steps", "inner_critic_optimizer_steps", "inner_model_steps_budget",
    "inner_actor_prior_kl", "inner_critic_prior_l2", "inner_diagnostic_replay_fraction",
    "inner_diagnostic_old_replay_samples", "inner_diagnostic_actor_prior_kl",
    "inner_diagnostic_critic_prior_l2", "inner_diagnostic_actor_anchor_loss",
    "inner_diagnostic_critic_anchor_loss", "inner_replay_size", "inner_solve_performed",
    "inner_previous_replay_samples", "inner_current_replay_samples", "inner_previous_replay_fraction",
    "inner_transfer_full_state", "inner_actor_prior_anchor_coef", "inner_critic_prior_anchor_coef",
    "inner_actor_prior_anchor_kl", "inner_actor_prior_anchor_penalty",
    "inner_critic_prior_anchor_loss", "inner_critic_prior_anchor_penalty",
    "inner_critic_prior_anchor_grad_norm", "inner_critic_prior_anchor_to_total_grad_ratio",
)

SELECTION_METRICS = (
    "retained_fraction", "projection_coefficient", "gradient_squared_norm",
    "donor_gradient_inner_product", "predicted_benefit", "transferred_energy_fraction",
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def load_campaign(path):
    path = Path(path).resolve()
    campaign = read_json(path)
    if campaign.get("schema_version") != 1 or campaign.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported transfer discovery campaign.")
    for key in ("horizons", "rounds", "seeds"):
        values = campaign[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{key} must be nonempty and unique.")
        for value in values:
            _positive_integer(value, key)
    for key in ("max_steps", "critic_updates", "actor_updates", "rollouts", "batch_size"):
        _positive_integer(campaign[key], key)
    if not campaign.get("arms"):
        raise ValueError("Campaign must name at least one transfer arm.")
    for name, arm in campaign["arms"].items():
        validate_arm(arm)
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name):
            raise ValueError("Arm names must use lowercase letters, digits and underscores.")
    base_dir = (Path(__file__).resolve().parents[1] / "configs/research"
                if campaign.get("family") == "spectral_transfer" else path.parent)
    campaign["base_matrix_path"] = str((base_dir / campaign["base_matrix"]).resolve())
    return campaign


def validate_arm(arm):
    allowed = {"actor_rho", "critic_rho", "collection", "replay_fraction", "full_state",
               "actor_prior_kl_coef", "critic_prior_l2_coef", "description", "compile",
               "actor_bernoulli_p", "critic_bernoulli_p", "parameter_scope",
               "actor_spectral", "critic_spectral"}
    if set(arm) - allowed:
        raise ValueError(f"Unknown transfer arm fields: {sorted(set(arm) - allowed)}")
    for key in ("actor_rho", "critic_rho", "replay_fraction", "actor_bernoulli_p", "critic_bernoulli_p"):
        value = arm.get(key, 0.)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be finite in [0,1].")
    for component in ("actor", "critic"):
        if f"{component}_bernoulli_p" in arm and f"{component}_rho" in arm:
            raise ValueError(f"{component} Bernoulli copying and deterministic blending are exclusive.")
        if f"{component}_spectral" in arm:
            from utils.spectral_transfer import validate_spectral_spec
            validate_spectral_spec(arm[f"{component}_spectral"])
            if any(f"{component}_{key}" in arm for key in ("rho", "bernoulli_p")):
                raise ValueError(f"{component} spectral, Bernoulli and rho transfer are exclusive.")
            if arm.get("parameter_scope") != "matrices":
                raise ValueError("Spectral arms require explicit parameter_scope='matrices'.")
            if (arm.get("full_state") or arm.get("replay_fraction", 0.) or
                    arm.get("collection", "learner") != "learner" or
                    arm.get("actor_prior_kl_coef", 0.) or arm.get("critic_prior_l2_coef", 0.)):
                raise ValueError("Spectral transfer requires fresh replay/optimizers and no extra intervention.")
    if arm.get("parameter_scope", "all") not in {"all", "matrices"}:
        raise ValueError("parameter_scope must be all or matrices.")
    if arm.get("parameter_scope") == "matrices" and arm.get("full_state"):
        raise ValueError("Matrix-only transfer cannot carry full learner state.")
    if arm.get("full_state", False) and any(f"{name}_bernoulli_p" in arm for name in ("actor", "critic")):
        raise ValueError("Bernoulli copying is weight-only and cannot carry full learner state.")
    for key in ("actor_prior_kl_coef", "critic_prior_l2_coef"):
        value = arm.get(key, 0.)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if arm.get("collection", "learner") not in {"learner", "previous_actor"}:
        raise ValueError("Collection must use learner or previous_actor.")
    if not isinstance(arm.get("full_state", False), bool) or not isinstance(arm.get("compile", True), bool):
        raise ValueError("full_state and compile must be booleans.")
    if arm.get("full_state", False) and (arm.get("actor_rho") != 1. or arm.get("critic_rho") != 1.):
        raise ValueError("Full-state carry requires actor_rho=critic_rho=1.")


def cells(campaign):
    return [dict(horizon=h, rounds=j, arm=arm,
                 cell_id=f"h{h}_j{j}_{arm}")
            for h in campaign["horizons"] for j in campaign["rounds"] for arm in campaign["arms"]]


def resolved_cell(base, campaign, *, horizon, rounds, arm, no_compile=False):
    if horizon not in campaign["horizons"] or rounds not in campaign["rounds"] or arm not in campaign["arms"]:
        raise ValueError("Cell is not in the configured discovery matrix.")
    result = deepcopy(base)
    params = result["algorithm_config"]["alg_params"]
    params.update(inner_rollout_horizon=horizon, inner_rounds=rounds,
        inner_first_action_rounds=None, inner_solve_interval=1,
        inner_rollouts_per_round=campaign["rollouts"], inner_batch_size=campaign["batch_size"],
        inner_critic_updates_per_round=campaign["critic_updates"],
        inner_actor_updates_per_round=campaign["actor_updates"],
        inner_actor_writeback_coef=0., inner_critic_writeback_coef=0.,
        inner_critic_transfer_head="retain", inner_rebase_persistent=False,
        inner_diagnostic_rollouts=0, outer_policy_diagnostics=False, wandb=False,
        compile=bool(campaign["arms"][arm].get("compile", True)) and not no_compile,
        compile_strict=bool(campaign["arms"][arm].get("compile", True)) and not no_compile)
    for name in ("actor", "critic", "replay", "temperature", "actor_optimizer",
                 "critic_optimizer", "temperature_optimizer"):
        params[f"inner_{name}_scope"] = "action"
    params["inner_replay_capacity"] = max(int(params.get("inner_replay_capacity") or 3072),
        horizon * rounds * campaign["rollouts"])
    result["selector"] = f"transfer_discovery/{arm}/H{horizon}/J{rounds}"
    return result


def blend_state(prior, donor, rho):
    """Blend floating tensors toward the immutable prior; preserve exact ends."""
    if set(prior) != set(donor):
        raise ValueError("Transfer donor and prior state keys differ.")
    result = {}
    for key, base in prior.items():
        value = donor[key]
        if not torch.is_tensor(base) or not torch.is_tensor(value) or base.shape != value.shape or base.dtype != value.dtype:
            raise ValueError(f"Incompatible transfer tensor {key}.")
        # The engine atomically validates all donor tensors before applying the
        # intervention. Avoid duplicating a GPU synchronization per tensor here.
        if rho == 0.:
            result[key] = base.detach().clone()
        elif rho == 1.:
            result[key] = value.detach().clone()
        elif base.is_floating_point() or base.is_complex():
            result[key] = torch.lerp(base.detach(), value.detach(), rho)
        elif torch.equal(base, value):
            result[key] = base.detach().clone()
        else:
            raise ValueError(f"Cannot partially blend differing nonfloating buffer {key}.")
    return result


def needs_donor(arm):
    return bool(any(arm.get(f"{name}_{method}", 0.) for name in ("actor", "critic")
                    for method in ("rho", "bernoulli_p"))
        or arm.get("replay_fraction", 0.) or arm.get("full_state", False)
        or arm.get("collection", "learner") == "previous_actor"
        or any(f"{name}_spectral" in arm for name in ("actor", "critic")))


def make_transfer_generators(engine, *, controller_seed, episode_seed):
    """Episode-local mask RNGs, separate from learner/probe/global RNG streams.

    Component-specific seeds pair the actor mask in actor-only and joint arms
    (and likewise the critic), regardless of other components' mask draws.
    Calling this again starts a fresh reproducible episode, never cross-episode
    carry. No global generator is seeded or sampled here.
    """
    return {name: torch.Generator(device=engine.device).manual_seed(
        solver_seed(controller_seed, "bernoulli_transfer", name, int(episode_seed)))
        for name in ("actor", "critic")}


@torch.no_grad()
def bernoulli_state(prior, donor, probability, *, parameter_names, generator=None, metrics=None):
    """Copy each scalar parameter from donor with p, otherwise from prior.

    This includes biases and normalization affine parameters. Module buffers
    retain the prior values. Values are selected exactly, without 1/p scaling,
    interpolation, in-place mutations, or new random parameter initialization.
    Endpoints do not draw masks. Optional metrics synchronize only once after
    all parameter reductions have been accumulated on device.
    """
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("Bernoulli keep probability must be finite in [0,1].")
    if set(prior) != set(donor):
        raise ValueError("Transfer donor and prior state keys differ.")
    names = set(parameter_names)
    if not names <= set(prior):
        raise ValueError("Transfer parameter names are absent from the prior.")
    for key, base in prior.items():
        value = donor[key]
        if (not torch.is_tensor(base) or not torch.is_tensor(value)
                or base.shape != value.shape or base.dtype != value.dtype or base.device != value.device):
            raise ValueError(f"Incompatible transfer tensor {key}.")
        if key in names and not base.is_floating_point():
            raise ValueError(f"Bernoulli transfer requires real floating parameters: {key}.")
    if 0 < probability < 1 and generator is None:
        raise ValueError("Bernoulli transfer requires an explicit isolated generator.")
    result, reductions, total = {}, [], 0
    for key, base in prior.items():
        value = donor[key]
        if key not in names or probability == 0:
            selected = base.detach().clone()
        elif probability == 1:
            selected = value.detach().clone()
        else:
            mask = torch.rand(base.shape, device=base.device, generator=generator) < probability
            selected = torch.where(mask, value.detach(), base.detach())
        result[key] = selected
        if metrics is not None and key in names:
            total += base.numel()
            kept = (mask.sum(dtype=torch.float64) if 0 < probability < 1 else
                    base.new_tensor(base.numel() * probability, dtype=torch.float64))
            reductions.append(torch.stack((kept,
                (selected - base).square().sum(dtype=torch.float64),
                base.square().sum(dtype=torch.float64))))
    if metrics is not None:
        kept, delta_squared, prior_squared = (torch.stack(reductions).sum(0).cpu().tolist()
                                             if reductions else (0., 0., 0.))
        metrics.update(retained_fraction=kept / total if total else 0.,
            relative_parameter_delta_l2=math.sqrt(delta_squared) / max(math.sqrt(prior_squared), 1e-12))
    return result


def arm_initialization(engine, arm, donor, *, transfer_generators=None, transfer_metrics=None,
                       spectral_context=None):
    """Translate one reviewed arm into an explicit engine intervention."""
    validate_arm(arm)
    options = dict(allow_compile=bool(engine.cfg.compile),
        actor_prior_kl_coef=float(arm.get("actor_prior_kl_coef", 0.)),
        critic_prior_l2_coef=float(arm.get("critic_prior_l2_coef", 0.)))
    if not arm.get("full_state", False):
        # Initialization overrides run after the ordinary fresh target copy.
        # Explicitly recopy the selected (possibly blended) online critic.
        options["target"] = "online"
    if transfer_metrics is not None:
        for component in ("actor", "critic"):
            if f"{component}_spectral" in arm:
                transfer_metrics[f"inner_{component}_spectral_applied"] = 0.
            if f"{component}_bernoulli_p" in arm:
                transfer_metrics.update({f"inner_{component}_bernoulli_retained_fraction": 0.,
                    f"inner_{component}_relative_parameter_delta_l2": 0.,
                    f"inner_{component}_bernoulli_applied": 0.})
    if donor is None:
        return options
    if arm.get("full_state", False):
        options.update(learner_state=donor, replay_fraction=float(arm["replay_fraction"]))
    else:
        for component in ("actor", "critic"):
            rho = float(arm.get(f"{component}_rho", 0.))
            probability = arm.get(f"{component}_bernoulli_p")
            module = getattr(engine, f"_{component}_base")
            names = {key for key, value in module.named_parameters()
                     if arm.get("parameter_scope", "all") == "all" or value.ndim == 2}
            spec = arm.get(f"{component}_spectral")
            if spec is not None:
                from utils.spectral_transfer import spectral_state
                sampled_metrics = {} if transfer_metrics is not None else None
                context = spectral_context or {}
                options[component] = spectral_state(module.state_dict(), donor["modules"][component],
                    parameter_names=names, spec=spec,
                    inputs=context.get("inputs", {}).get(component),
                    gradients=context.get("gradients", {}).get(component), metrics=sampled_metrics,
                    include_layer_metrics=False)
                if transfer_metrics is not None:
                    transfer_metrics[f"inner_{component}_spectral_applied"] = float(spec.get("strength", 1.) > 0)
                    transfer_metrics.update({f"inner_{component}_spectral_{key}": value
                        for key, value in sampled_metrics.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)})
            elif probability is not None and probability > 0:
                sampled_metrics = {} if transfer_metrics is not None else None
                options[component] = bernoulli_state(module.state_dict(), donor["modules"][component],
                    probability, parameter_names=names,
                    generator=None if transfer_generators is None else transfer_generators[component],
                    metrics=sampled_metrics)
                if transfer_metrics is not None:
                    transfer_metrics.update({
                        f"inner_{component}_bernoulli_retained_fraction": sampled_metrics["retained_fraction"],
                        f"inner_{component}_relative_parameter_delta_l2": sampled_metrics["relative_parameter_delta_l2"],
                        f"inner_{component}_bernoulli_applied": 1.})
            elif rho:
                base = getattr(engine, f"_{component}_base").state_dict()
                options[component] = blend_state(base, donor["modules"][component], rho)
                if arm.get("parameter_scope") == "matrices":
                    for key in set(base) - names:
                        options[component][key] = base[key].detach().clone()
        if arm.get("replay_fraction", 0.):
            options.update(replay=donor["replay"], replay_fraction=float(arm["replay_fraction"]))
    if arm.get("collection", "learner") == "previous_actor":
        options["collection_actor"] = donor["modules"]["actor"]
    return options


def selected_metrics(metrics, *, policy="legacy"):
    """Retain computed scalar measurements without extra solver probes."""
    if policy not in METRIC_POLICIES:
        raise ValueError(f"Unknown inner metric policy: {policy}")
    result = {}
    keys = METRICS if policy == "legacy" else (
        key for key in metrics if isinstance(key, str) and key.startswith("inner_"))
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        if isinstance(value, (bool, int, float, np.number)):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite controller metric {key}.")
            result[key] = value
    return result


def evaluate_episode(wrapped, env, arm, *, episode_seed, controller_seed, max_steps,
                     on_step=None, smoke=False, diagnostics=None, spectral_probe=None,
                     metric_policy="legacy"):
    """One paired episode with optional separately timed observational probes."""
    from evaluate_ambi_checkpoint import _seed_spaces
    if metric_policy not in METRIC_POLICIES:
        raise ValueError(f"Unknown inner metric policy: {metric_policy}")
    validate_arm(arm)
    _positive_integer(max_steps, "max_steps")
    engine = wrapped.agent.inner_engine
    episode_solver_seed = solver_seed(controller_seed, "episode", int(episode_seed))
    engine.reset_for_evaluation(episode_solver_seed, reuse_action_pool=True)
    _seed_spaces(env, int(episode_seed))
    observation, _ = env.reset(seed=int(episode_seed))
    donor = None
    carry_donor = needs_donor(arm)
    transfer_generators = make_transfer_generators(engine, controller_seed=controller_seed,
                                                   episode_seed=episode_seed)
    rewards, latencies, transfer_times, diagnostic_times = [], [], [], []
    spectral_probe_times, spectral_filter_times, export_times = [], [], []
    selection_components = tuple(name for name in ("actor", "critic")
        if arm.get(f"{name}_spectral", {}).get("method") in {"gradient", "gradient_projection", "gradient_gate"}
        and arm[f"{name}_spectral"].get("strength", 1.) > 0)
    selection_values = {}
    terminated = truncated = False
    started = time.perf_counter()
    for decision in range(max_steps):
        if wrapped.agent.device.type == "cuda":
            torch.cuda.synchronize(wrapped.agent.device)
        transfer_started = time.perf_counter()
        transfer_metrics = {}
        context = None
        components = tuple(name for name in ("actor", "critic")
            if arm.get(f"{name}_spectral", {}).get("method") in {"activation", "gradient", "gradient_projection", "gradient_gate"}
            and arm[f"{name}_spectral"].get("strength", 1.) > 0)
        if donor is not None and components:
            from utils.spectral_transfer_probes import build_spectral_context
            context = build_spectral_context(wrapped, observation, controller_seed=controller_seed,
                episode_seed=episode_seed, decision=decision, settings=spectral_probe,
                components=components, compute_gradients=any(
                    arm[f"{name}_spectral"]["method"] in {"gradient", "gradient_projection", "gradient_gate"}
                    for name in components))
        if wrapped.agent.device.type == "cuda":
            torch.cuda.synchronize(wrapped.agent.device)
        probe_seconds = time.perf_counter() - transfer_started if context is not None else 0.
        filter_started = time.perf_counter()
        options = arm_initialization(engine, arm, donor, transfer_generators=transfer_generators,
                                     transfer_metrics=transfer_metrics, spectral_context=context)
        if wrapped.agent.device.type == "cuda":
            torch.cuda.synchronize(wrapped.agent.device)
        initialization_finished = time.perf_counter()
        initialization_seconds = initialization_finished - transfer_started
        filter_seconds = (initialization_finished - filter_started if donor is not None and
            any(f"{name}_spectral" in arm for name in ("actor", "critic")) else 0.)
        trace = None if diagnostics is None else diagnostics.begin(wrapped, observation, decision, donor)
        predict_started = time.perf_counter()
        with engine.diagnostic_initialization(**options):
            extra = {} if trace is None else {"trace": trace}
            action, _ = wrapped.predict(observation, deterministic=True, episode_start=decision == 0, **extra)
        if wrapped.agent.device.type == "cuda":
            torch.cuda.synchronize(wrapped.agent.device)
        prediction_seconds = time.perf_counter() - predict_started
        diagnostic_record = None if diagnostics is None else diagnostics.finish(trace)
        diagnostic_seconds = 0. if diagnostics is None else diagnostics.decision_seconds
        if diagnostics is not None:
            prediction_seconds -= diagnostics.in_prediction_seconds
        transfer_started = time.perf_counter()
        diagnostic_donor = diagnostics is not None and getattr(diagnostics, "needs_donor", False)
        if carry_donor or diagnostic_donor:
            donor = engine.export_diagnostic_state(include_optimizers=bool(arm.get("full_state", False)),
                include_replay=bool(arm.get("replay_fraction", 0.)))
        if wrapped.agent.device.type == "cuda":
            torch.cuda.synchronize(wrapped.agent.device)
        export_seconds = time.perf_counter() - transfer_started
        # A fresh arm exports a donor only for observational measurements.
        if diagnostic_donor and not carry_donor:
            diagnostic_seconds += export_seconds
            export_seconds = 0.
        transfer_seconds = initialization_seconds + export_seconds
        action = np.asarray(action)
        if not np.isfinite(action).all():
            raise ValueError("Controller emitted a nonfinite action.")
        observation, reward, terminated, truncated, _ = env.step(action)
        reward = float(reward)
        if not math.isfinite(reward) or not np.isfinite(observation).all():
            raise ValueError("Environment emitted a nonfinite observation or reward.")
        rewards.append(reward)
        latencies.append(prediction_seconds)
        transfer_times.append(transfer_seconds)
        diagnostic_times.append(diagnostic_seconds)
        spectral_probe_times.append(probe_seconds)
        spectral_filter_times.append(filter_seconds)
        export_times.append(export_seconds)
        row = dict(seed=int(episode_seed), decision=decision, reward=reward,
            cumulative_reward=float(sum(rewards)), action=action.tolist(),
            prediction_seconds=prediction_seconds, transfer_seconds=transfer_seconds,
            control_seconds=prediction_seconds + transfer_seconds,
            diagnostic_seconds=diagnostic_seconds,
            spectral_probe_seconds=probe_seconds, spectral_filter_seconds=filter_seconds,
            donor_export_seconds=export_seconds,
            metrics={**selected_metrics(wrapped.agent.last_inner_metrics, policy=metric_policy), **transfer_metrics},
            terminated=bool(terminated), truncated=bool(truncated))
        if diagnostic_record is not None:
            row["diagnostics"] = diagnostic_record
        if context is not None:
            row["spectral_selection"] = context["metadata"]
            # These are every-decision selection-bank measurements, separate
            # from sparse held-out diagnostics. No donor or zero strength means
            # no gradient probe and therefore no contributing observation.
            for component in selection_components:
                for quantity in SELECTION_METRICS:
                    source = f"inner_{component}_spectral_{quantity}"
                    if source not in transfer_metrics:
                        continue
                    value = transfer_metrics[source]
                    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
                        raise ValueError(f"Nonfinite or nonnumeric selection metric {source}.")
                    key = f"selection_{component}_{quantity}"
                    selection_values.setdefault(key, []).append(float(value))
        if on_step is not None:
            on_step(row)
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - started
    result = dict(seed=int(episode_seed), episode_solver_seed=episode_solver_seed,
        return_value=float(sum(rewards)), reward=float(sum(rewards)), steps=len(rewards),
        terminated=bool(terminated), truncated=bool(truncated),
        truncated_by_evaluator=not (terminated or truncated) and len(rewards) == max_steps,
        smoke=bool(smoke), wall_seconds=elapsed,
        prediction_seconds=float(sum(latencies)), transfer_seconds=float(sum(transfer_times)),
        diagnostic_seconds=float(sum(diagnostic_times)),
        spectral_probe_seconds=float(sum(spectral_probe_times)),
        spectral_filter_seconds=float(sum(spectral_filter_times)),
        donor_export_seconds=float(sum(export_times)),
        control_seconds=float(sum(latencies) + sum(transfer_times)),
        mean_prediction_seconds=float(np.mean(latencies)),
        p50_prediction_seconds=float(np.percentile(latencies, 50)),
        p90_prediction_seconds=float(np.percentile(latencies, 90)),
        first_decision_seconds=latencies[0],
        subsequent_mean_prediction_seconds=float(np.mean(latencies[1:])) if len(latencies) > 1 else None,
        rng_protocol="solver_seed(controller_seed, 'episode', environment_seed); persistent private streams")
    if any(f"{name}_bernoulli_p" in arm for name in ("actor", "critic")):
        result["transfer_rng_protocol"] = (
            "solver_seed(controller_seed, 'bernoulli_transfer', component, environment_seed); "
            "independent episode-local torch.Generator per component")
    if diagnostics is not None:
        result["diagnostics"] = diagnostics.coverage(len(rewards))
        result["diagnostic_samples"] = result["diagnostics"]["samples"]
    if selection_values:
        # Scale before summing so individually finite measurements cannot
        # overflow merely because an episode has many contributing decisions.
        summary = {key: math.fsum(value / len(values) for value in values)
                   for key, values in selection_values.items()}
        if not all(math.isfinite(value) for value in summary.values()):
            raise ValueError("Nonfinite selection diagnostic episode mean.")
        counts = {key: len(values) for key, values in selection_values.items()}
        result["selection_diagnostics"] = {"summary": summary, "summary_counts": counts}
        if diagnostics is not None:
            result["diagnostics"].setdefault("summary", {}).update(summary)
            result["diagnostics"].setdefault("summary_counts", {}).update(counts)
    result.update({"return": result["reward"], "length": result["steps"], "solver_seed": episode_solver_seed})
    return result


def summarize_episodes(episodes):
    values = np.asarray([episode["reward"] for episode in episodes], dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("At least one complete finite episode record is required.")
    return dict(episodes=len(values), mean_return=float(values.mean()),
        std_return=float(values.std(ddof=1)) if len(values) > 1 else None,
        se_return=float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None,
        min_return=float(values.min()), max_return=float(values.max()),
        total_steps=sum(row["steps"] for row in episodes),
        control_seconds=sum(row["control_seconds"] for row in episodes))
