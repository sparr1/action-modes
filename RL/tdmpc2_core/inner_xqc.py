"""Fresh per-decision XQC improvement over imagined TOLD transitions.

This engine is intentionally separate from :mod:`inner_improvement`.  Native
AMBI's SAC/TD3/MPPI experiment surface has different network, normalization,
target, and lifecycle semantics.  AMBI-XQC always makes one action-local clone
of the persistent XQC priors, improves it with an action-local latent replay,
executes it once, and then logically discards it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import time
from typing import Any

import torch

from .common import math as td_math
from .common.inner_utils import InnerRNG
from .common.latent_buffer import LatentReplayBuffer
from .common.training_state import require_exact_keys
from .xqc_controller import LatentXQCBatch, LatentXQCWorkspace


@dataclass
class InnerXQCState:
    """Only the logically live state of the current decision."""

    workspace: LatentXQCWorkspace | None = None
    replay: LatentReplayBuffer | None = None
    reward_scale: float = 1.0
    replay_draws: int = 0
    policy_evaluations: int = 0
    sampled_ids: list[torch.Tensor] = field(default_factory=list)


class InnerXQCEngine:
    """Run a fresh root-local XQC solve without mutating persistent priors."""

    _STATE_SCHEMA = "ambi-xqc-inner-training-state"
    _STATE_VERSION = 1

    def __init__(self, agent):
        self.agent = agent
        self.cfg = agent.cfg
        self.model = agent.model
        self.outer_controller = agent.xqc_controller
        self.device = torch.device(agent.device)
        self.rng = InnerRNG(self.cfg.seed, self.device)
        self.state = InnerXQCState()
        # These allocations are scratch. Resetting them from the persistent
        # priors before every action makes reuse semantically indistinguishable
        # from constructing a new learner while avoiding allocator overhead.
        self._workspace_pool: LatentXQCWorkspace | None = None
        self._replay_pool: LatentReplayBuffer | None = None
        self.action_index = 0
        self.episode_index = 0
        self._collect_diagnostics = True
        self._pending_timers: dict[str, list[Any]] = {}

    def clear_all(self):
        self.state = InnerXQCState()
        self._workspace_pool = None
        self._replay_pool = None
        self.rng = InnerRNG(self.cfg.seed, self.device)
        self.action_index = 0
        self.episode_index = 0
        self._pending_timers = {}

    def reset_episode(self):
        # All learned inner state is action-local, so an episode boundary only
        # advances the serialized diagnostic/lifecycle counter.
        self.episode_index += 1

    def prepare_training_resume_boundary(self):
        if self.state.workspace is not None or self.state.replay is not None:
            raise RuntimeError("AMBI-XQC cannot checkpoint during an inner action.")

    def training_state_dict(self):
        self.prepare_training_resume_boundary()
        return {
            "schema": self._STATE_SCHEMA,
            "version": self._STATE_VERSION,
            "action_index": int(self.action_index),
            "episode_index": int(self.episode_index),
            "rng": self.rng.training_state_dict(),
        }

    @staticmethod
    def _nonnegative_index(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"AMBI-XQC inner {name} must be a non-negative integer.")
        return value

    def _preflight_training_state_dict(self, state):
        state = require_exact_keys(
            state,
            {"schema", "version", "action_index", "episode_index", "rng"},
            "AMBI-XQC inner training state",
        )
        if state["schema"] != self._STATE_SCHEMA or state["version"] != self._STATE_VERSION:
            raise ValueError("Unsupported AMBI-XQC inner training-state version.")
        candidate = {
            "action_index": self._nonnegative_index(
                state["action_index"], "action_index"
            ),
            "episode_index": self._nonnegative_index(
                state["episode_index"], "episode_index"
            ),
            "rng": self.rng._preflight_training_state_dict(state["rng"]),
        }
        return candidate

    def _commit_training_state_candidate(self, candidate):
        self.state = InnerXQCState()
        self._workspace_pool = None
        self._replay_pool = None
        self.action_index = candidate["action_index"]
        self.episode_index = candidate["episode_index"]
        self.rng.load_training_state_dict(candidate["rng"])
        self._pending_timers = {}

    def load_training_state_dict(self, state):
        candidate = self._preflight_training_state_dict(state)
        self._commit_training_state_candidate(candidate)
        return self

    def mark_outer_update(self, version):
        # Action-local state never survives long enough to need rebasing.
        del version

    def _timer_start(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def _timer_stop(self, key, start):
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            value = (start, end)
        else:
            value = time.perf_counter() - start
        self._pending_timers.setdefault(key, []).append(value)

    def finalize_timing_metrics(self, metrics):
        for key, measurements in self._pending_timers.items():
            if self.device.type == "cuda":
                metrics[key] = sum(
                    start.elapsed_time(end) / 1000.0 for start, end in measurements
                )
            else:
                metrics[key] = sum(measurements)
        self._pending_timers = {}
        return metrics

    def _real_reward_scale(self) -> float:
        normalizer = getattr(self.agent, "reward_normalizer", None)
        scale = 1.0 if normalizer is None else float(normalizer.scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "AMBI-XQC's real-return reward scale must be finite and positive."
            )
        return scale

    def _prepare_action(self):
        cfg = self.cfg
        if self._workspace_pool is None:
            workspace = self.outer_controller.clone_for_inner(
                actor_lr=float(cfg.inner_actor_lr),
                critic_lr=float(cfg.inner_critic_lr),
                # The endpoints equal the starts inside clone_for_inner, so
                # this value only bounds a constant local schedule.
                transition_steps=int(
                    cfg.inner_rounds * cfg.inner_updates_per_round
                ),
            )
        else:
            workspace = self._workspace_pool
            self._workspace_pool = None
            workspace.reset_from_(self.outer_controller)

        if self._replay_pool is None:
            replay = LatentReplayBuffer(
                capacity=int(cfg.inner_replay_capacity),
                latent_dim=int(cfg.latent_dim),
                action_dim=int(cfg.action_dim),
                device=self.device,
            )
        else:
            replay = self._replay_pool
            self._replay_pool = None
            replay.clear()

        self.state = InnerXQCState(
            workspace=workspace,
            replay=replay,
            reward_scale=self._real_reward_scale(),
        )

    def _release_action(self):
        if self.state.workspace is not None:
            self._workspace_pool = self.state.workspace
        if self.state.replay is not None:
            self._replay_pool = self.state.replay
        self.state = InnerXQCState()

    @contextmanager
    def _action_lifetime(self):
        """Release logical action state on both success and failure."""
        try:
            yield
        finally:
            self._release_action()

    @torch.no_grad()
    def _sample_actor(self, z, *, stream, deterministic=False):
        generator = self.rng.generator(stream)
        # Draw explicitly so all inner randomness stays on its named stream and
        # evaluation advances the stream exactly once, as released XQC does.
        noise = torch.randn(
            (*z.shape[:-1], int(self.cfg.action_dim)),
            dtype=z.dtype,
            device=z.device,
            generator=generator,
        )
        action, _ = self.state.workspace.controller.sample_action(
            z,
            deterministic=deterministic,
            noise=noise,
        )
        self.state.policy_evaluations += int(z.shape[0])
        return action

    @torch.no_grad()
    def _collect_round(self, root_z):
        cfg = self.cfg
        count = int(cfg.inner_rollouts_per_round)
        horizon = int(cfg.inner_rollout_horizon)
        z = root_z.expand(count, -1).clone()
        alive = torch.ones(count, dtype=torch.bool, device=self.device)
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        reward_sums = torch.zeros(count, dtype=root_z.dtype, device=self.device)
        discounted_rewards = torch.zeros_like(reward_sums)
        discount_weight = torch.ones_like(reward_sums)
        terminated_rollout = torch.zeros_like(alive)
        transition_fields = ([], [], [], [], [])

        for _ in range(horizon):
            active = torch.nonzero(alive, as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                break
            active_z = z.index_select(0, active)
            action = self._sample_actor(active_z, stream="collection")
            joint = self.model.joint_input(active_z, action)
            reward = td_math.two_hot_inv(self.model.reward_from_joint(joint), cfg)
            next_z = self.model.next_from_joint(joint)
            if bool(cfg.episodic):
                terminated = (
                    self.model.termination(next_z)
                    > float(cfg.inner_termination_threshold)
                ).to(dtype=reward.dtype)
            else:
                terminated = reward.new_zeros((active.numel(), 1))

            transition_fields[0].append(active_z)
            transition_fields[1].append(action)
            transition_fields[2].append(reward)
            transition_fields[3].append(next_z)
            transition_fields[4].append(terminated)

            reward_vector = reward.squeeze(-1)
            lengths[active] += 1
            reward_sums[active] += reward_vector
            discounted_rewards[active] += discount_weight[active] * reward_vector
            discount_weight[active] *= float(self.agent.discount)
            z[active] = next_z
            just_terminated = terminated.squeeze(-1) >= 0.5
            terminated_rollout[active] |= just_terminated
            alive[active] = ~just_terminated

        if transition_fields[0]:
            self.state.replay.add_batch(
                *(torch.cat(values, dim=0) for values in transition_fields)
            )
        return {
            "lengths": lengths,
            "reward_sums": reward_sums,
            "discounted_rewards": discounted_rewards,
            "terminated": terminated_rollout,
        }

    def _sample_batch(self):
        replacement = self.cfg.inner_replay_sampling == "with_replacement"
        batch = self.state.replay.sample(
            int(self.cfg.inner_batch_size),
            replacement=replacement,
            generator=self.rng.generator("replay"),
            include_ids=self._collect_diagnostics,
        )
        self.state.replay_draws += int(batch["z"].shape[0])
        if self._collect_diagnostics:
            self.state.sampled_ids.append(batch["sample_ids"].detach())
        return batch

    def _update_slot(self):
        raw = self._sample_batch()
        batch_size = int(raw["z"].shape[0])
        next_noise = torch.randn(
            (batch_size, int(self.cfg.action_dim)),
            dtype=raw["z"].dtype,
            device=self.device,
            generator=self.rng.generator("bootstrap"),
        )
        actor_noise = torch.randn(
            (batch_size, int(self.cfg.action_dim)),
            dtype=raw["z"].dtype,
            device=self.device,
            generator=self.rng.generator("gradient_policy"),
        )
        batch = LatentXQCBatch(
            latents=raw["z"],
            actions=raw["action"],
            rewards=raw["reward"],
            next_latents=raw["next_z"],
            bootstrap_mask=1.0 - raw["terminated"],
            discount=torch.as_tensor(
                float(self.agent.discount),
                dtype=raw["reward"].dtype,
                device=self.device,
            ),
        )
        return self.state.workspace.update(
            batch,
            next_noise=next_noise,
            actor_noise=actor_noise,
            reward_scale=self.state.reward_scale,
        )

    @staticmethod
    def _stats(groups, reference):
        if not groups:
            zero = reference.new_zeros(())
            return zero, zero, zero, zero
        values = torch.cat(groups).float()
        return (
            values.mean(),
            values.std(unbiased=False),
            values.min(),
            values.max(),
        )

    @staticmethod
    def _average_updates(history, reference):
        if not history:
            return {}
        result = {}
        keys = set().union(*(metrics.keys() for metrics in history))
        for key in keys:
            values = [metrics[key] for metrics in history if key in metrics]
            if not values:
                continue
            tensors = [torch.as_tensor(value, device=reference.device) for value in values]
            result[f"inner_{key}"] = torch.stack(tensors).float().mean()
        return result

    def _base_metrics(self):
        return {
            "inner_active": 1.0,
            "inner_algorithm_xqc": 1.0,
            "inner_diagnostics_sampled": 0.0,
            "inner_diagnostics_sample_count": 0.0,
        }

    def act(self, root_z, *, t0=False, eval_mode=False, collect_diagnostics=True):
        del t0  # The v1 learner is fresh at every action, including episode starts.
        self._pending_timers = {}
        action_start = self._timer_start()
        self.action_index += 1
        self._collect_diagnostics = bool(collect_diagnostics)

        with self.rng.action_fork(), self._action_lifetime():
            setup_start = self._timer_start()
            # Module initialization is implicit-random only on the first use;
            # this private fork prevents it from advancing outer learner RNG.
            with self.rng.fork("initialization"):
                self._prepare_action()
            self._timer_stop("inner_setup_seconds", setup_start)

            alpha_initial = (
                self.state.workspace.controller.temperature.detach().clone()
            )
            all_lengths = []
            reward_sums = []
            discounted_rewards = []
            terminated = []
            update_history = []

            for _ in range(int(self.cfg.inner_rounds)):
                rollout_start = self._timer_start()
                rollout = self._collect_round(root_z)
                self._timer_stop("inner_rollout_seconds", rollout_start)
                all_lengths.append(rollout["lengths"])
                reward_sums.append(rollout["reward_sums"])
                discounted_rewards.append(rollout["discounted_rewards"])
                terminated.append(rollout["terminated"])

                update_start = self._timer_start()
                for _ in range(int(self.cfg.inner_updates_per_round)):
                    update_history.append(self._update_slot())
                self._timer_stop("inner_update_seconds", update_start)

            execution_start = self._timer_start()
            action = self._sample_actor(
                root_z, stream="execution", deterministic=bool(eval_mode)
            )
            self._timer_stop("inner_execution_seconds", execution_start)

            workspace = self.state.workspace
            alpha_final = workspace.controller.temperature.detach().clone()
            length_values = torch.cat(all_lengths)
            termination_values = torch.cat(terminated).float()
            reward_stats = self._stats(reward_sums, root_z)
            discounted_stats = self._stats(discounted_rewards, root_z)
            length_stats = self._stats(all_lengths, root_z)
            termination_stats = self._stats(terminated, root_z)
            update_slots = int(workspace.update_step)
            actor_steps = int(workspace.actor_optimizer_steps)
            temperature_steps = int(workspace.temperature_optimizer_steps)
            target_interval = int(
                workspace.controller.config.target_update_interval
            )
            target_steps = update_slots // target_interval
            update_batch_work = update_slots * int(self.cfg.inner_batch_size)
            realized_steps = length_values.sum()
            utd_denominator = realized_steps.clamp_min(1)

            metrics = self._base_metrics()
            metrics.update(
                inner_rounds=float(self.cfg.inner_rounds),
                inner_iterations=float(self.cfg.inner_rounds),
                inner_rollouts=float(length_values.numel()),
                inner_requested_rollouts=float(
                    self.cfg.inner_rounds * self.cfg.inner_rollouts_per_round
                ),
                inner_rollout_count=float(length_values.numel()),
                inner_steps=length_values.sum(),
                inner_model_steps=length_values.sum(),
                inner_model_steps_budget=float(self.cfg.inner_model_step_budget),
                inner_nominal_model_steps=float(self.cfg.inner_model_step_budget),
                inner_realized_model_steps=length_values.sum(),
                inner_total_model_steps=length_values.sum(),
                inner_updates=float(update_slots),
                inner_update_slots=float(update_slots),
                inner_requested_update_slots=float(
                    self.cfg.inner_rounds * self.cfg.inner_updates_per_round
                ),
                inner_critic_optimizer_steps=float(update_slots),
                inner_actor_optimizer_steps=float(actor_steps),
                inner_temperature_optimizer_steps=float(temperature_steps),
                inner_critic_utd=(
                    torch.as_tensor(float(update_slots), device=self.device)
                    / utd_denominator
                ),
                inner_actor_utd=(
                    torch.as_tensor(float(actor_steps), device=self.device)
                    / utd_denominator
                ),
                inner_temperature_utd=(
                    torch.as_tensor(float(temperature_steps), device=self.device)
                    / utd_denominator
                ),
                inner_target_updates=float(target_steps),
                inner_critic_target_updates=float(target_steps),
                inner_actor_target_updates=0.0,
                inner_policy_evaluations=float(
                    self.state.policy_evaluations + 2 * update_batch_work
                ),
                inner_q_evaluations=float(5 * update_batch_work),
                inner_replay_draws=float(self.state.replay_draws),
                inner_buffer_size=float(self.state.replay.size),
                inner_buffer_capacity=float(self.state.replay.capacity),
                inner_buffer_fill_ratio=float(
                    self.state.replay.size / self.state.replay.capacity
                ),
                inner_reward_scale=float(self.state.reward_scale),
                inner_alpha=alpha_final.mean(),
                inner_alpha_initial=alpha_initial.mean(),
                inner_alpha_final=alpha_final.mean(),
                inner_alpha_delta=(alpha_final - alpha_initial).mean(),
                inner_return_mean=reward_stats[0],
                inner_return_std=reward_stats[1],
                inner_return_min=reward_stats[2],
                inner_return_max=reward_stats[3],
                inner_behavior_reward_sum_mean=reward_stats[0],
                inner_behavior_reward_sum_std=reward_stats[1],
                inner_behavior_reward_sum_min=reward_stats[2],
                inner_behavior_reward_sum_max=reward_stats[3],
                inner_behavior_discounted_reward_mean=discounted_stats[0],
                inner_behavior_discounted_reward_std=discounted_stats[1],
                inner_behavior_discounted_reward_min=discounted_stats[2],
                inner_behavior_discounted_reward_max=discounted_stats[3],
                inner_rollout_len_mean=length_stats[0],
                inner_rollout_len_std=length_stats[1],
                inner_rollout_len_min=length_stats[2],
                inner_rollout_len_max=length_stats[3],
                inner_termination_rate=termination_stats[0],
                inner_termination_rate_std=termination_stats[1],
                inner_termination_rate_min=termination_stats[2],
                inner_termination_rate_max=termination_stats[3],
            )
            metrics.update(self._average_updates(update_history, root_z))
            if self._collect_diagnostics:
                metrics["inner_diagnostics_sampled"] = 1.0
                metrics["inner_diagnostics_sample_count"] = 1.0
                metrics["inner_diagnostics_step"] = float(self.action_index)
                if self.state.sampled_ids:
                    sampled = torch.cat(self.state.sampled_ids)
                    metrics["inner_replay_unique_fraction"] = (
                        sampled.unique().numel() / sampled.numel()
                    )

            if bool(self.cfg.episodic):
                rollout_lengths = length_values.detach()
            else:
                rollout_lengths = [
                    int(self.cfg.inner_rollout_horizon)
                ] * int(length_values.numel())
            result = action[0], metrics, rollout_lengths

        self._timer_stop("inner_action_seconds", action_start)
        return result
