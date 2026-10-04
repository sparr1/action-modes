"""Lean full-episode discovery of transfer between successive inner SAC solves.

The previous solve is the only weight donor. Networks blend toward frozen
priors; replay, optimizer and temperature reuse are explicit arm properties.
No expensive model probes or learner-module serializations run in this loop.
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
    campaign["base_matrix_path"] = str((path.parent / campaign["base_matrix"]).resolve())
    return campaign


def validate_arm(arm):
    allowed = {"actor_rho", "critic_rho", "collection", "replay_fraction", "full_state",
               "actor_prior_kl_coef", "critic_prior_l2_coef", "description", "compile"}
    if set(arm) - allowed:
        raise ValueError(f"Unknown transfer arm fields: {sorted(set(arm) - allowed)}")
    for key in ("actor_rho", "critic_rho", "replay_fraction"):
        value = arm.get(key, 0.)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be finite in [0,1].")
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


def arm_initialization(engine, arm, donor):
    """Translate one reviewed arm into an explicit engine intervention."""
    validate_arm(arm)
    options = dict(allow_compile=bool(engine.cfg.compile),
        actor_prior_kl_coef=float(arm.get("actor_prior_kl_coef", 0.)),
        critic_prior_l2_coef=float(arm.get("critic_prior_l2_coef", 0.)))
    if not arm.get("full_state", False):
        # Initialization overrides run after the ordinary fresh target copy.
        # Explicitly recopy the selected (possibly blended) online critic.
        options["target"] = "online"
    if donor is None:
        return options
    if arm.get("full_state", False):
        options.update(learner_state=donor, replay_fraction=float(arm["replay_fraction"]))
    else:
        for component in ("actor", "critic"):
            rho = float(arm.get(f"{component}_rho", 0.))
            if rho:
                base = getattr(engine, f"_{component}_base").state_dict()
                options[component] = blend_state(base, donor["modules"][component], rho)
        if arm.get("replay_fraction", 0.):
            options.update(replay=donor["replay"], replay_fraction=float(arm["replay_fraction"]))
    if arm.get("collection", "learner") == "previous_actor":
        options["collection_actor"] = donor["modules"]["actor"]
    return options


def selected_metrics(metrics):
    result = {}
    for key in METRICS:
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
                     on_step=None, smoke=False):
    """One paired environment episode, with no learner snapshots or probe trace."""
    from evaluate_ambi_checkpoint import _seed_spaces
    validate_arm(arm)
    _positive_integer(max_steps, "max_steps")
    engine = wrapped.agent.inner_engine
    episode_solver_seed = solver_seed(controller_seed, "episode", int(episode_seed))
    engine.reset_for_evaluation(episode_solver_seed, reuse_action_pool=True)
    _seed_spaces(env, int(episode_seed))
    observation, _ = env.reset(seed=int(episode_seed))
    donor = None
    needs_donor = bool(arm.get("actor_rho", 0.) or arm.get("critic_rho", 0.)
        or arm.get("replay_fraction", 0.) or arm.get("full_state", False)
        or arm.get("collection", "learner") == "previous_actor")
    rewards, latencies, transfer_times = [], [], []
    terminated = truncated = False
    started = time.perf_counter()
    for decision in range(max_steps):
        if wrapped.agent.device.type == "cuda":
            torch.cuda.synchronize(wrapped.agent.device)
        transfer_started = time.perf_counter()
        options = arm_initialization(engine, arm, donor)
        if wrapped.agent.device.type == "cuda":
            torch.cuda.synchronize(wrapped.agent.device)
        initialization_seconds = time.perf_counter() - transfer_started
        predict_started = time.perf_counter()
        with engine.diagnostic_initialization(**options):
            action, _ = wrapped.predict(observation, deterministic=True, episode_start=decision == 0)
        if wrapped.agent.device.type == "cuda":
            torch.cuda.synchronize(wrapped.agent.device)
        prediction_seconds = time.perf_counter() - predict_started
        transfer_started = time.perf_counter()
        if needs_donor:
            donor = engine.export_diagnostic_state(include_optimizers=bool(arm.get("full_state", False)),
                include_replay=bool(arm.get("replay_fraction", 0.)))
        if wrapped.agent.device.type == "cuda":
            torch.cuda.synchronize(wrapped.agent.device)
        transfer_seconds = initialization_seconds + time.perf_counter() - transfer_started
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
        row = dict(seed=int(episode_seed), decision=decision, reward=reward,
            cumulative_reward=float(sum(rewards)), action=action.tolist(),
            prediction_seconds=prediction_seconds, transfer_seconds=transfer_seconds,
            control_seconds=prediction_seconds + transfer_seconds,
            metrics=selected_metrics(wrapped.agent.last_inner_metrics),
            terminated=bool(terminated), truncated=bool(truncated))
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
        control_seconds=float(sum(latencies) + sum(transfer_times)),
        mean_prediction_seconds=float(np.mean(latencies)),
        p50_prediction_seconds=float(np.percentile(latencies, 50)),
        p90_prediction_seconds=float(np.percentile(latencies, 90)),
        first_decision_seconds=latencies[0],
        subsequent_mean_prediction_seconds=float(np.mean(latencies[1:])) if len(latencies) > 1 else None,
        rng_protocol="solver_seed(controller_seed, 'episode', environment_seed); persistent private streams")
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
