"""Evaluation-only native TD-MPC2 MPPI over a frozen AMBI prior.

Uses raw TOLD rewards and the learned online AMBI soft-Q tail. No rollout or
terminal entropy correction is added; AMBI Q was trained with a soft target,
so this is an MPPI adaptation to that learned critic, not a reward-only critic.
"""
from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real
import time

import torch

from .mppi import mppi_plan

TD_MPC2_MPPI_DEFAULTS = {
    "horizon": 3,
    "iterations": 6,
    "num_samples": 512,
    "num_elites": 64,
    "num_pi_trajs": 24,
    "min_std": 0.05,
    "max_std": 2.0,
    "temperature": 0.5,
}


def resolve_mppi_settings(settings, *, action_dim):
    if settings is not None and not isinstance(settings, Mapping):
        raise TypeError("MPPI settings must be a mapping.")
    settings = dict(settings or {})
    unknown = set(settings) - set(TD_MPC2_MPPI_DEFAULTS)
    if unknown:
        raise ValueError(f"Unsupported AMBI MPPI settings: {sorted(unknown)}")
    resolved = {**TD_MPC2_MPPI_DEFAULTS, **settings}
    for key in ("horizon", "iterations", "num_samples", "num_elites", "num_pi_trajs"):
        value = resolved[key]
        minimum = 0 if key == "num_pi_trajs" else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"MPPI {key} must be an integer >= {minimum}.")
        resolved[key] = int(value)
    if resolved["num_elites"] > resolved["num_samples"]:
        raise ValueError("MPPI num_elites cannot exceed num_samples.")
    if resolved["num_pi_trajs"] > resolved["num_samples"]:
        raise ValueError("MPPI num_pi_trajs cannot exceed num_samples.")
    for key in ("min_std", "max_std", "temperature"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"MPPI {key} must be positive and finite.")
        resolved[key] = float(value)
    if resolved["min_std"] > resolved["max_std"]:
        raise ValueError("MPPI min_std cannot exceed max_std.")
    # Preserve the official large-action-space heuristic, including overrides
    # to the authored base iteration count. Humanoid has 21 actions: 6 + 2.
    resolved["effective_iterations"] = resolved["iterations"] + 2 * int(action_dim >= 20)
    return resolved


class FrozenAMBIMPPIController:
    """Private episode-scoped search state; owns no learner or optimizer."""

    def __init__(self, agent, settings=None):
        if str(agent.cfg.inner_operator) != "none":
            raise ValueError("Native AMBI MPPI requires loading the checkpoint as a frozen prior.")
        if str(agent.cfg.obs) != "state":
            raise ValueError("Native AMBI MPPI supports state observations only.")
        if any(module.training for module in agent.model.modules()):
            raise ValueError("Native AMBI MPPI requires every frozen model module in eval mode.")
        self.agent = agent
        self.model = agent.model
        self.device = torch.device(agent.device)
        self._settings = resolve_mppi_settings(settings, action_dim=agent.cfg.action_dim)
        self.discount = float(agent.discount)
        if not math.isfinite(self.discount) or not 0 <= self.discount <= 1:
            raise ValueError("MPPI requires a finite discount in [0, 1].")
        self.reset(int(agent.cfg.seed))

    @property
    def settings(self):
        return deepcopy(self._settings)

    @property
    def protocol(self):
        return {
            "algorithm": "tdmpc2_mppi_over_frozen_ambi",
            "action_rule": "weighted_elite_gumbel_no_execution_noise",
            "terminal_value_source": "online_ambi_q_mean_pair",
            "terminal_action": "frozen_prior_sample",
            "terminal_value_units": "raw_reward_soft_q",
            "terminal_value_semantics": "learned_soft_q_tail_without_entropy_correction",
            "reward_units": "raw_environment_reward",
            "discount": self.discount,
            "warm_start": "shift_previous_mean_within_episode_reset_before_episode",
            "rng": "private_device_generator_sha256_episode_seed",
            "value_finite_guard": "torch.nan_to_num(nan=0)",
            "upstream_tdmpc2_commit": "8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe",
        }

    def reset(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("MPPI episode seed must be an integer.")
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.previous_mean = None
        self.action_index = 0
        return self

    @torch.no_grad()
    def act(self, observation):
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("Native AMBI MPPI cannot run with training-mode model modules.")
        started = time.perf_counter()
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        shape = tuple(self.agent.cfg.obs_shape["state"])
        if tuple(observation.shape) != shape:
            raise ValueError(f"MPPI observation must have shape {shape}.")
        root_z = self.model.encode(observation.unsqueeze(0))
        settings = self.settings
        settings["iterations"] = settings.pop("effective_iterations")
        result = mppi_plan(root_z, model=self.model, **settings,
            discount=self.discount, q_reduction="mean_pair", generator=self.generator,
            previous_mean=self.previous_mean, t0=self.previous_mean is None,
            eval_mode=True, action_selection="tdmpc2", materialize_metrics=False)
        self.previous_mean = result.next_mean.detach()
        self.action_index += 1
        metrics = dict(result.metrics)
        metrics.update(inner_active=1.0, inner_algorithm_mppi=1.0,
            inner_mppi_iterations=settings["iterations"], inner_iterations=settings["iterations"],
            inner_model_steps=result.model_steps, inner_total_model_steps=result.model_steps,
            inner_steps=result.model_steps,
            inner_policy_evaluations=metrics["planner_policy_evaluations"],
            inner_q_evaluations=metrics["planner_q_evaluations"],
            inner_updates=0, inner_update_slots=0, inner_critic_optimizer_steps=0,
            inner_actor_optimizer_steps=0, inner_temperature_optimizer_steps=0,
            inner_critic_target_updates=0, inner_actor_target_updates=0,
            inner_diagnostics_sampled=0.0, inner_diagnostics_sample_count=0.0)
        action, metrics, _ = self.agent._materialize_action_metrics(result.action, metrics, [])
        if not bool(torch.isfinite(action).all()):
            raise ValueError("MPPI produced a nonfinite action.")
        elapsed = time.perf_counter() - started
        metrics.update(inner_mppi_seconds=elapsed, inner_action_seconds=elapsed)
        self.agent.last_inner_metrics = metrics
        self.agent.last_inner_rollout_lengths = []
        return action
