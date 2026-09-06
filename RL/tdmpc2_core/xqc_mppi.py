"""Evaluation-only TD-MPC2 MPPI over a frozen AMBI-XQC checkpoint.

The population defaults and action selection follow official TD-MPC2 commit
8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe. The terminal value is deliberately an
XQC bootstrap approximation: its learned soft-Q tail is multiplied by the
frozen real reward scale and averaged over both online critics. It remains a
soft-Q tail; no entropy correction makes it a raw-reward value function.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Integral, Real
import time

import torch

from .common import math as td_math
from .mppi import MPPIModelCallbacks, mppi_plan


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
        raise ValueError(f"Unsupported XQC MPPI settings: {sorted(unknown)}")
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


class FrozenXQCMPPIController:
    """A private planner workspace; never installed into the training wrapper."""

    def __init__(self, agent, settings=None):
        if not getattr(agent, "_frozen_evaluation", False):
            raise ValueError("XQC MPPI requires a frozen-evaluation checkpoint load.")
        if str(agent.cfg.obs) != "state":
            raise NotImplementedError("XQC MPPI evaluation supports state observations only.")
        self.agent = agent
        self.model = agent.model
        self.controller = agent.xqc_controller
        self.device = torch.device(agent.device)
        self._settings = resolve_mppi_settings(settings, action_dim=agent.cfg.action_dim)
        self.reward_scale = float(agent.reward_normalizer.scale)
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("XQC MPPI requires a positive finite frozen real reward scale.")
        self.discount = float(agent.discount)
        if not math.isfinite(self.discount) or not 0 <= self.discount <= 1:
            raise ValueError("XQC MPPI requires a finite checkpoint discount in [0, 1].")
        self.callbacks = MPPIModelCallbacks(
            action_dim=int(agent.cfg.action_dim),
            dynamics=self._dynamics,
            reward=self._reward,
            transition=self._transition,
            policy=self._policy,
            terminal_q=self._terminal_q,
            termination=self.model.termination if bool(agent.cfg.episodic) else None,
        )
        self.reset(int(agent.cfg.seed))

    @property
    def settings(self):
        return deepcopy(self._settings)

    @property
    def protocol(self):
        return {
            "algorithm": "tdmpc2_mppi_over_frozen_xqc",
            "action_rule": "weighted_elite_gumbel_no_execution_noise",
            "terminal_value_source": "online_xqc_twin_mean",
            "terminal_value_units": "normalized_xqc_soft_q_times_frozen_real_reward_scale",
            "terminal_value_semantics": "learned_soft_q_tail_without_entropy_correction",
            "reward_units": "raw_environment_reward",
            "reward_scale": self.reward_scale,
            "discount": self.discount,
            "batchnorm_mode": "running",
            "warm_start": "shift_previous_mean_within_episode_reset_before_episode",
            "rng": "private_device_generator",
            "value_finite_guard": "torch.nan_to_num(nan=0)",
            "upstream_tdmpc2_commit": "8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe",
        }

    def reset(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("The MPPI episode seed must be an integer.")
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.previous_mean = None
        self.action_index = 0
        self._model_steps = 0
        self._policy_evaluations = 0
        self._q_evaluations = 0
        return self

    def _dynamics(self, z, action):
        self._model_steps += int(z.shape[0])
        return self.model.next(z, action)

    def _reward(self, z, action):
        return td_math.two_hot_inv(self.model.reward(z, action), self.agent.cfg)

    def _transition(self, z, action):
        self._model_steps += int(z.shape[0])
        joint = self.model.joint_input(z, action)
        return (
            self.model.next_from_joint(joint),
            td_math.two_hot_inv(self.model.reward_from_joint(joint), self.agent.cfg),
        )

    def _policy(self, z, *, generator):
        self._policy_evaluations += int(z.shape[0])
        noise = torch.randn(
            (*z.shape[:-1], int(self.agent.cfg.action_dim)),
            dtype=z.dtype, device=z.device, generator=generator,
        )
        return self.controller.sample_action(z, deterministic=False, noise=noise)

    def _terminal_q(self, z, action, *, reduction, generator):
        del generator
        if reduction != "mean_all":
            raise ValueError("Frozen XQC MPPI uses the mean of both online critics.")
        self._q_evaluations += int(z.shape[0])
        normalized_values = self.controller.critic.values(z, action, bn_mode="running")
        return normalized_values.mean(dim=0).unsqueeze(-1) * self.reward_scale

    @torch.no_grad()
    def act(self, observation):
        if not self.agent._frozen_evaluation:
            raise RuntimeError("XQC MPPI cannot run after the agent returns to training.")
        started = time.perf_counter()
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        expected_shape = tuple(self.agent.cfg.obs_shape["state"])
        if tuple(observation.shape) != expected_shape:
            raise ValueError(f"MPPI observation must have shape {expected_shape}.")
        self._model_steps = self._policy_evaluations = self._q_evaluations = 0
        was_training = self.model.training
        self.model.eval()
        try:
            root_z = self.model.encode(observation.unsqueeze(0))
            settings = dict(self._settings)
            settings["iterations"] = settings.pop("effective_iterations")
            result = mppi_plan(
                root_z, callbacks=self.callbacks, **settings,
                discount=self.discount, q_reduction="mean_all",
                generator=self.generator, previous_mean=self.previous_mean,
                t0=self.previous_mean is None, eval_mode=True,
                action_selection="tdmpc2", materialize_metrics=False,
            )
        finally:
            self.model.train(was_training)
        if self._model_steps != result.model_steps:
            raise RuntimeError("MPPI model-step accounting does not match actual callbacks.")
        if not bool(torch.isfinite(result.action).all()) or not bool(torch.isfinite(result.next_mean).all()):
            raise ValueError("MPPI produced a non-finite action or warm-start state.")
        self.previous_mean = result.next_mean.detach()
        self.action_index += 1
        metrics = dict(result.metrics)
        metrics.update({
            "inner_active": 1.0, "inner_algorithm_mppi": 1.0,
            "inner_algorithm_xqc": 0.0, "inner_iterations": self._settings["effective_iterations"],
            "inner_mppi_iterations": self._settings["effective_iterations"],
            "inner_model_steps": self._model_steps, "inner_steps": self._model_steps,
            "inner_realized_model_steps": self._model_steps,
            "inner_total_model_steps": self._model_steps,
            "inner_model_steps_budget": self._model_steps,
            "inner_nominal_model_steps": self._model_steps,
            "inner_policy_evaluations": self._policy_evaluations,
            "inner_q_evaluations": self._q_evaluations,
            "planner_q_head_evaluations": 2 * self._q_evaluations,
            "inner_updates": 0, "inner_update_slots": 0,
            "inner_critic_optimizer_steps": 0, "inner_actor_optimizer_steps": 0,
            "inner_temperature_optimizer_steps": 0, "inner_replay_draws": 0,
            "inner_reward_normalizer_imagined_updates": 0,
            "inner_reward_scale": self.reward_scale,
            "inner_reward_scale_initial": self.reward_scale,
            "inner_reward_scale_final": self.reward_scale,
            "inner_reward_scale_delta": 0.0,
            "inner_diagnostics_sampled": 0.0, "inner_diagnostics_sample_count": 0.0,
            "inner_target_updates": 0, "inner_critic_target_updates": 0,
            "inner_actor_target_updates": 0,
        })
        action, metrics, _ = self.agent._materialize_action_metrics(result.action, metrics, [])
        elapsed = time.perf_counter() - started
        metrics.update(inner_mppi_seconds=elapsed, inner_action_seconds=elapsed)
        self.agent.last_inner_metrics = metrics
        self.agent.last_inner_rollout_lengths = []
        return action


__all__ = ["FrozenXQCMPPIController", "TD_MPC2_MPPI_DEFAULTS", "resolve_mppi_settings"]
