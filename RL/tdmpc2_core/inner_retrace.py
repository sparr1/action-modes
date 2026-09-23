"""Complete-trajectory Retrace for the action-local, finite-horizon SAC solve.

The one-step engine remains the default. This mixin isolates the different
collection and minibatch contracts of the optional trajectory learner.
"""

import torch

from .common import math as td_math
from .common.entropy import policy_entropy
from .common.latent_trajectory_buffer import LatentTrajectoryReplayBuffer
from .common.retrace import retrace_targets


def _masked_mean(values, valid):
    mask = torch.broadcast_to(valid, values.shape)
    return torch.where(mask, values, 0.0).sum() / mask.sum().clamp_min(1)


class RetraceInnerMixin:
    # This bounded port supports ordinary latent policies only. The resolver
    # rejects time-conditioned networks instead of introducing that separate
    # architecture feature into an execution/return-estimator experiment.
    _horizon_conditioning_horizon = None

    @staticmethod
    def _horizon_kwargs(remaining_horizon):
        return {}

    @staticmethod
    def _remaining_horizon(z, step=0):
        return None

    @property
    def _retrace_enabled(self):
        return getattr(self.cfg, "inner_sac_return_estimator", "one_step") == "retrace"

    def _retrace_spec(self):
        if not self._retrace_enabled:
            return None
        return {
            "protocol_version": 1,
            "estimator": "retrace",
            "lambda": float(self.cfg.inner_retrace_lambda),
            "horizon": int(self.cfg.inner_rollout_horizon),
            "batch_trajectories": int(self.cfg.inner_retrace_batch_trajectories),
            "loss_positions": "all_valid_suffixes",
            "boundary": "frozen_outer_no_extra_entropy",
        }

    def _new_trajectory_replay(self):
        return LatentTrajectoryReplayBuffer(
            capacity=self.cfg.inner_replay_capacity,
            latent_dim=self.cfg.latent_dim,
            action_dim=self.cfg.action_dim,
            device=self.device,
            horizon=self.cfg.inner_rollout_horizon,
            horizon_conditioning_horizon=self._horizon_conditioning_horizon,
        )

    def _retrace_policy(self, z, *, noise=None, generator=None, remaining_horizon=None):
        return self.model.pi(
            z, policy=self.state.actor, noise=noise, generator=generator,
            std_scale=max(float(self.cfg.inner_behavior_std_scale), 1e-12),
            log_std_mapping=self.cfg.inner_log_std_mapping,
            log_std_min=self.cfg.inner_log_std_min,
            log_std_max=self.cfg.inner_log_std_max,
            **self._horizon_kwargs(remaining_horizon),
        )

    def _retrace_dense_rollout_kernel(self, root_z, policy_noise, reward_support):
        """Pure fixed-H collection, with no additional random draws for densities."""
        count, horizon = int(self.cfg.inner_rollouts_per_round), int(self.cfg.inner_rollout_horizon)
        z = root_z.expand(count, -1)
        records = []
        for step in range(horizon):
            remaining = self._remaining_horizon(z, step)
            action, info = self._retrace_policy(z, noise=policy_noise[step], remaining_horizon=remaining)
            joint = self.model.joint_input(z, action)
            prediction = self.model.reward_from_joint(joint)
            reward = (self.model.decode_reward(prediction, support=reward_support)
                      if bool(getattr(self.cfg, "compile", False))
                      else self.model.decode_reward(prediction))
            next_z = self.model.next_from_joint(joint)
            row = {
                "z": z, "action": action, "reward": reward, "next_z": next_z,
                "terminated": torch.zeros_like(reward),
                "horizon_end": torch.full_like(reward, float(step == horizon - 1)),
                "pre_tanh_action": info["pre_tanh_action"],
                "behavior_log_prob": info["log_prob"],
                "valid": torch.ones_like(reward, dtype=torch.bool),
            }
            if remaining is not None:
                row["remaining_horizon"] = remaining
            records.append(row)
            z = next_z
        return {name: torch.stack([row[name] for row in records], dim=1) for name in records[0]}

    @torch.no_grad()
    def _collect_retrace_round(self, root_z):
        cfg, state = self.cfg, self.state
        count, horizon = int(cfg.inner_rollouts_per_round), int(cfg.inner_rollout_horizon)
        state.actor.eval()
        self.model.eval()
        with self.rng.fork("collection") as generator:
            if not cfg.episodic:
                noise = torch.stack([
                    torch.randn((count, int(cfg.action_dim)), device=self.device,
                                dtype=root_z.dtype, generator=generator)
                    for _ in range(horizon)
                ])
                batch = self._compile_regions["rollout"](
                    root_z, noise, td_math.categorical_support(root_z, cfg),
                )
                transition_count = count * horizon
            else:
                widths = {"z": cfg.latent_dim, "action": cfg.action_dim, "reward": 1,
                          "next_z": cfg.latent_dim, "terminated": 1, "horizon_end": 1,
                          "pre_tanh_action": cfg.action_dim, "behavior_log_prob": 1}
                batch = {name: root_z.new_zeros(count, horizon, width) for name, width in widths.items()}
                batch["valid"] = torch.zeros(count, horizon, 1, device=self.device, dtype=torch.bool)
                if self._horizon_conditioning_horizon is not None:
                    batch["remaining_horizon"] = torch.ones(count, horizon, 1, device=self.device, dtype=torch.long)
                z = root_z.expand(count, -1).clone()
                alive = torch.ones(count, device=self.device, dtype=torch.bool)
                transition_count = 0
                for step in range(horizon):
                    active = torch.nonzero(alive, as_tuple=False).squeeze(-1)
                    if not active.numel():
                        break
                    active_z = z[active]
                    remaining = self._remaining_horizon(active_z, step)
                    action, info = self._retrace_policy(active_z, generator=generator, remaining_horizon=remaining)
                    joint = self.model.joint_input(active_z, action)
                    reward = self.model.decode_reward(self.model.reward_from_joint(joint))
                    next_z = self.model.next_from_joint(joint)
                    done = (self.model.termination(next_z) > float(cfg.inner_termination_threshold)).float()
                    values = dict(z=active_z, action=action, reward=reward, next_z=next_z,
                                  terminated=done, horizon_end=torch.full_like(done, float(step == horizon - 1)),
                                  pre_tanh_action=info["pre_tanh_action"], behavior_log_prob=info["log_prob"])
                    for name, value in values.items():
                        batch[name][active, step] = value
                    batch["valid"][active, step] = True
                    if remaining is not None:
                        batch["remaining_horizon"][active, step] = remaining
                    transition_count += active.numel()
                    z[active] = next_z
                    alive[active] = done.squeeze(-1) < 0.5
        state.replay.add_trajectories(batch, validate=False)
        state.policy_evaluations += transition_count
        if cfg.inner_actor_adaptation != "frozen":
            state.actor.train()
        discounts = float(self.agent.discount) ** torch.arange(horizon, device=self.device)
        rewards = batch["reward"].squeeze(-1)
        return {
            "lengths": batch["valid"].squeeze(-1).sum(dim=1),
            "reward_sums": rewards.sum(dim=1),
            "discounted_rewards": (rewards * discounts).sum(dim=1),
            "terminated": batch["terminated"].squeeze(-1).bool().any(dim=1),
            "transition_count": transition_count,
        }

    def _sample_retrace_batch(self, indices):
        batch = self.state.replay.sample_trajectories(
            self.cfg.inner_retrace_batch_trajectories,
            replacement=self.cfg.inner_replay_sampling == "with_replacement",
            generator=self.rng.generator("replay"), indices=indices,
            include_ids=self._collect_diagnostics,
        )
        self.state.replay_draws += batch["valid"].sum().detach()
        if self._collect_diagnostics:
            ids = batch["sample_ids"].reshape(-1)
            self.state.sampled_ids.append(ids[batch["valid"].reshape(-1)].detach())
        return batch

    def _retrace_critic_kernel(self, batch, alpha, policy_noise, prior_noise, pair_indices,
                               *, actor_loss_scale=None):
        """One detached target snapshot; gradients update only the online critic."""
        cfg, state = self.cfg, self.state
        valid = batch["valid"].bool()
        batch_size, horizon = valid.shape[:2]
        flat_valid = valid.reshape(-1, 1)
        z = torch.where(valid, batch["z"], 0.0).reshape(-1, cfg.latent_dim)
        action = torch.where(valid, batch["action"], 0.0).reshape(-1, cfg.action_dim)
        remaining = batch.get("remaining_horizon")
        if remaining is not None:
            remaining = torch.where(valid, remaining, 1).reshape(-1, 1)
        horizon_kwargs = self._horizon_kwargs(remaining)
        with torch.no_grad():
            continuing = valid & ~batch["terminated"].bool()
            next_z = torch.where(continuing, batch["next_z"], 0.0).reshape(-1, cfg.latent_dim)
            next_kwargs = self._horizon_kwargs(None if remaining is None else (remaining - 1).clamp_min(1))
            next_action, info = self.model.pi(
                next_z, policy=state.actor, noise=policy_noise.reshape(-1, cfg.action_dim),
                log_std_mapping=cfg.inner_log_std_mapping,
                log_std_min=cfg.inner_log_std_min, log_std_max=cfg.inner_log_std_max,
                **self.agent._inner_critic_entropy_kwargs(), **next_kwargs,
            )
            q_kwargs = {} if pair_indices is None else {"pair_indices": pair_indices, "trusted_pair_indices": True}
            value = self._bootstrap_q(next_z, next_action, **q_kwargs, **next_kwargs)
            if cfg.inner_sac_critic_target == "entropy_augmented":
                coefficient = alpha
                if self._sac_actor_loss_scale_enabled:
                    if actor_loss_scale is None:
                        raise RuntimeError("Scaled Retrace requires the action-local actor loss scale.")
                    coefficient = coefficient * actor_loss_scale.detach().reshape(())
                value = value + coefficient * policy_entropy(info, cfg.inner_actor_entropy_mode)
            value = value.reshape(batch_size, horizon, 1)
            # Only the actual H boundary uses the frozen outer continuation.
            outer = self._prior_bootstrap(next_z.reshape(batch_size, horizon, -1)[:, -1], prior_noise)
            outer = outer[:, None, :].expand(-1, horizon, -1)
            value = torch.where(batch["horizon_end"].bool(), outer, value)
            value = torch.where(continuing, value, 0.0)
            q = self._bootstrap_q(z, action, **q_kwargs, **horizon_kwargs).reshape(batch_size, horizon, 1)
            stats = self.model.policy_stats(
                z, policy=state.actor, log_std_mapping=cfg.inner_log_std_mapping,
                log_std_min=cfg.inner_log_std_min, log_std_max=cfg.inner_log_std_max,
                **horizon_kwargs,
            )
            pre_tanh = torch.where(valid, batch["pre_tanh_action"], 0.0).reshape(-1, cfg.action_dim)
            log_pi = self.model.squashed_component_log_prob(
                pre_tanh, stats["pre_tanh_mean"], stats["log_std"],
            ).reshape(batch_size, horizon, 1)
            log_mu = torch.where(valid, batch["behavior_log_prob"], 0.0)
            c = float(cfg.inner_retrace_lambda) * torch.exp((log_pi - log_mu).clamp_max(0.0))
            c = torch.where(valid, c, 0.0)
            discount = continuing.to(value.dtype) * float(self.agent.discount)
            targets, lengths, corrections = retrace_targets(
                batch["reward"], discount, value, q, c, valid,
            )
        flat_target = targets.reshape(-1, 1)
        predictions = self.model.q_predictions(z, action, qs=state.critic, **horizon_kwargs)
        losses = self.model.critic_loss(predictions, flat_target, reduction="none")
        loss = _masked_mean(losses, flat_valid.unsqueeze(0)) * float(cfg.inner_critic_loss_coef)
        values = self.model.q_backend.decode(predictions.detach())
        clip_fraction = targets.new_zeros(())
        if cfg.q_representation == "distributional":
            symlog_target = td_math.symlog(targets)
            clipped = (symlog_target <= float(cfg.q_vmin)) | (symlog_target >= float(cfg.q_vmax))
            clip_fraction = _masked_mean(clipped.to(targets.dtype), valid)
        return loss, values, flat_target, clip_fraction, c, lengths, corrections

    def _retrace_critic_step(self, batch, alpha, *, actor_loss_scale=None):
        state, cfg = self.state, self.cfg
        count, horizon = batch["valid"].shape[:2]
        generator = self.rng.generator("bootstrap")
        noise = torch.randn((count, horizon, cfg.action_dim), device=self.device,
                            dtype=batch["z"].dtype, generator=generator)
        prior_noise = self._prior_noise(batch["next_z"][:, -1])
        pair = self._sample_pair_indices(generator) if cfg.inner_q_target_reduction.endswith("_pair") else None
        outputs = self._compile_regions["critic"](
            batch, alpha, noise, prior_noise, pair, actor_loss_scale=actor_loss_scale,
        )
        loss, values, targets, clip_fraction, c, lengths, corrections = outputs
        state.policy_evaluations += 2 * count * horizon
        state.q_evaluations += 3 * count * horizon
        state.critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(state.critic_params, float(cfg.inner_critic_grad_clip_norm))
        state.critic_optim.step()
        state.critic_steps += 1
        state.critic_lifetime_steps += 1
        valid = batch["valid"].bool()
        flat_valid = valid.reshape(-1, 1)
        # c_0 is never a trace weight; only real successor edges enter diagnostics.
        edges = valid[:, 1:] & valid[:, :-1] & ~batch["terminated"][:, :-1].bool()
        return {
            "critic_loss": loss.detach(), "critic_grad_norm": torch.as_tensor(grad_norm).detach(),
            "q_mean": _masked_mean(values, flat_valid.unsqueeze(0)),
            "q_abs_mean": _masked_mean(values.abs(), flat_valid.unsqueeze(0)),
            "q_target_mean": _masked_mean(targets, flat_valid),
            "q_target_clip_fraction": clip_fraction.detach(),
            "td_error_abs_mean": _masked_mean((values - targets.unsqueeze(0)).abs(), flat_valid.unsqueeze(0)),
            "retrace_trajectory_draws": targets.new_tensor(count),
            "retrace_critic_rows": valid.sum().to(targets.dtype).detach(),
            "retrace_trace_coefficient_mean": _masked_mean(c[:, 1:], edges),
            "retrace_effective_trace_length": _masked_mean(lengths, valid),
            "retrace_correction_abs_mean": _masked_mean(corrections.abs(), valid),
        }

    def _run_retrace_update_counts(self, *, critic_count, actor_count, temperature_count,
                                   actor_loss_scale=None):
        """Keep actor row batches independent from complete-trajectory critic draws."""
        state, cfg = self.state, self.cfg
        generator = self.rng.generator("replay")
        replacement = cfg.inner_replay_sampling == "with_replacement"

        def indices(slots, size, batch_size):
            if not slots:
                return None
            if not replacement and batch_size > size:
                raise ValueError(f"Cannot sample Retrace replay without replacement: batch_size={batch_size}, size={size}.")
            if replacement:
                return torch.randint(size, (slots, batch_size), device=self.device, generator=generator)
            return torch.stack([torch.randperm(size, device=self.device, generator=generator)[:batch_size]
                                for _ in range(slots)])

        critic_indices = indices(critic_count, state.replay.trajectory_count, cfg.inner_retrace_batch_trajectories)
        policy_count = max(actor_count, temperature_count if self._inner_entropy_enabled else 0)
        policy_indices = indices(policy_count, state.replay.size, cfg.inner_batch_size)
        metrics = []
        for slot in range(max(critic_count, actor_count, temperature_count)):
            do_critic, do_actor = slot < critic_count, slot < actor_count
            do_temperature = slot < temperature_count and self._inner_entropy_enabled
            alpha = self.alpha.detach().clone()
            slot_metrics = {}
            if do_critic:
                batch = self._sample_retrace_batch(critic_indices[slot])
                with self.rng.fork("bootstrap"):
                    slot_metrics.update(self._retrace_critic_step(batch, alpha, actor_loss_scale=actor_loss_scale))
            if do_actor or do_temperature:
                batch = self._sample_batch(policy_indices[slot])
                slot_metrics.update(self._sac_policy_step(
                    batch, update_temperature=do_temperature, update_actor=do_actor,
                    alpha=alpha, actor_loss_scale=actor_loss_scale,
                ))
            self._maybe_update_targets(critic_updated=do_critic, actor_updated=do_actor)
            if self._active_trace is not None:
                self._active_trace.record(
                    "update", state, {**slot_metrics, "alpha_used": alpha},
                    updated_critic=do_critic, updated_actor=do_actor,
                    updated_temperature=do_temperature, measurement="pre_update_minibatch",
                )
            metrics.append(slot_metrics)
        return metrics
