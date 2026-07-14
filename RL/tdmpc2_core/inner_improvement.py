"""Componentized AMBI inner-loop improvement strategies.

All state in this module is private to action selection. The outer world model,
optimizers, target critics, and entropy coefficient are never mutated here.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import time

import torch

from .common import math as td_math
from .common.inner_utils import (
    InnerRNG,
    allocate_across_rounds,
    rebase_clone_,
    rebase_lora_base_,
    trainable_parameter_count,
)
from .common.latent_buffer import LatentReplayBuffer
from .common.lora import lorafy_copy, trainable_parameters


@torch.no_grad()
def polyak_update(source, target, tau):
    tau = float(tau)
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        target_parameter.data.lerp_(source_parameter.data, tau)
    for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
        target_buffer.data.copy_(source_buffer.data)


@dataclass
class InnerWorkspace:
    actor: torch.nn.Module | None = None
    actor_anchor: torch.nn.Module | None = None
    critic: torch.nn.Module | None = None
    critic_anchor: torch.nn.Module | None = None
    critic_target: torch.nn.Module | None = None
    actor_target: torch.nn.Module | None = None
    actor_optim: torch.optim.Optimizer | None = None
    critic_optim: torch.optim.Optimizer | None = None
    log_alpha: torch.nn.Parameter | None = None
    alpha_fixed: torch.Tensor | None = None
    temperature_optim: torch.optim.Optimizer | None = None
    replay: LatentReplayBuffer | None = None
    outer_version: int = -1
    critic_steps: int = 0
    critic_lifetime_steps: int = 0
    actor_steps: int = 0
    actor_lifetime_steps: int = 0
    temperature_steps: int = 0
    target_steps: int = 0
    critic_target_steps: int = 0
    actor_target_steps: int = 0
    replay_draws: int = 0
    policy_evaluations: int = 0
    q_evaluations: int = 0
    sampled_indices: list[torch.Tensor] = field(default_factory=list)
    sampled_ids: list[torch.Tensor] = field(default_factory=list)


class InnerImprovementEngine:
    """Run none/SAC/TD3 inner improvement behind ``AMBI.agent.act``."""

    def __init__(self, agent):
        self.agent = agent
        self.cfg = agent.cfg
        self.model = agent.model
        self.device = agent.device
        self.rng = InnerRNG(self.cfg.seed, self.device)
        self.state = InnerWorkspace()
        self.action_index = 0
        self.episode_index = 0
        self._mppi_prev_mean = None

    @property
    def alpha(self):
        if self.state.log_alpha is not None:
            return self.state.log_alpha.exp()
        if self.state.alpha_fixed is not None:
            return self.state.alpha_fixed
        return self.agent.alpha.detach()

    def clear_all(self):
        self.state = InnerWorkspace()
        self.rng = InnerRNG(self.cfg.seed, self.device)
        self.action_index = 0
        self.episode_index = 0
        self._mppi_prev_mean = None

    def reset_episode(self):
        """Clear action/episode state while preserving explicitly run-scoped state."""
        self.episode_index += 1
        self._clear_expired(t0=True, include_action=True)
        if str(self.cfg.inner_mppi_warm_start_scope) != "run":
            self._mppi_prev_mean = None

    def mark_outer_update(self, version):
        # Workspaces are refreshed lazily at the next action, after all outer
        # updates in the current real environment step have completed.
        del version

    def _scope_expires(self, scope, *, t0, include_action=True):
        scope = str(scope)
        return (include_action and scope == "action") or (t0 and scope == "episode")

    def _clear_expired(self, *, t0, include_action=True):
        state, cfg = self.state, self.cfg
        if self._scope_expires(cfg.inner_actor_scope, t0=t0, include_action=include_action):
            state.actor = state.actor_anchor = state.actor_target = None
            state.actor_optim = None
            state.actor_lifetime_steps = 0
        elif self._scope_expires(
            cfg.inner_actor_optimizer_scope, t0=t0, include_action=include_action
        ):
            state.actor_optim = None

        if self._scope_expires(cfg.inner_critic_scope, t0=t0, include_action=include_action):
            state.critic = state.critic_anchor = state.critic_target = None
            state.critic_optim = None
            state.critic_lifetime_steps = 0
        elif self._scope_expires(
            cfg.inner_critic_optimizer_scope, t0=t0, include_action=include_action
        ):
            state.critic_optim = None

        if self._scope_expires(
            cfg.inner_temperature_scope, t0=t0, include_action=include_action
        ):
            state.log_alpha = state.alpha_fixed = state.temperature_optim = None
        elif self._scope_expires(
            cfg.inner_temperature_optimizer_scope,
            t0=t0,
            include_action=include_action,
        ):
            state.temperature_optim = None

        if self._scope_expires(cfg.inner_replay_scope, t0=t0, include_action=include_action):
            state.replay = None

    def _adapt_module(self, base, component):
        mode = str(getattr(self.cfg, f"inner_{component}_adaptation"))
        if mode == "frozen":
            module = deepcopy(base).to(self.device)
            module.requires_grad_(False)
        elif mode == "clone":
            module = deepcopy(base).to(self.device)
            module.requires_grad_(True)
        elif mode == "lora":
            module = lorafy_copy(
                base,
                rank=getattr(self.cfg, f"inner_{component}_lora_rank"),
                scale=getattr(self.cfg, f"inner_{component}_lora_scale"),
                dropout=getattr(self.cfg, f"inner_{component}_lora_dropout"),
            ).to(self.device)
        else:
            raise ValueError(f"Unknown {component} adaptation mode: {mode!r}")
        return module

    def _new_optimizer(self, module, component):
        params = trainable_parameters(module)
        if not params:
            return None
        return torch.optim.Adam(
            params,
            lr=float(getattr(self.cfg, f"inner_{component}_lr")),
            eps=float(self.cfg.inner_adam_eps),
            capturable=self.device.type == "cuda",
        )

    def _refresh_persistent_component(self, component, outer):
        state = self.state
        adapted = getattr(state, component)
        anchor = getattr(state, f"{component}_anchor")
        target = getattr(state, f"{component}_target", None)
        mode = str(getattr(self.cfg, f"inner_{component}_adaptation"))
        if adapted is None or anchor is None:
            return
        if mode == "lora":
            rebase_lora_base_(adapted, outer)
            if target is not None:
                rebase_lora_base_(target, outer)
            anchor.load_state_dict(outer.state_dict())
        elif mode == "clone":
            old_anchor = deepcopy(anchor)
            rebase_clone_(adapted, anchor, outer)
            if target is not None:
                rebase_clone_(target, old_anchor, outer)
        else:
            adapted.load_state_dict(outer.state_dict())
            anchor.load_state_dict(outer.state_dict())
            if target is not None:
                target.load_state_dict(outer.state_dict())

    def _prepare_workspace(self, *, t0):
        state, cfg = self.state, self.cfg
        self._clear_expired(t0=t0, include_action=True)

        actor_was_missing = state.actor is None
        critic_was_missing = state.critic is None
        if actor_was_missing:
            state.actor = self._adapt_module(self.model._pi, "actor")
            state.actor_anchor = deepcopy(self.model._pi).to(self.device).requires_grad_(False)
            state.actor_lifetime_steps = 0
        if critic_was_missing:
            state.critic = self._adapt_module(self.model._Qs, "critic")
            state.critic_anchor = deepcopy(self.model._Qs).to(self.device).requires_grad_(False)
            state.critic_lifetime_steps = 0

        outer_changed = state.outer_version >= 0 and state.outer_version != self.agent.outer_version
        if outer_changed and bool(cfg.inner_rebase_persistent):
            if not actor_was_missing:
                self._refresh_persistent_component("actor", self.model._pi)
            if not critic_was_missing:
                self._refresh_persistent_component("critic", self.model._Qs)

        if state.critic_target is None or critic_was_missing:
            state.critic_target = deepcopy(state.critic).to(self.device).requires_grad_(False)
        if cfg.inner_operator == "td3" and (
            state.actor_target is None or actor_was_missing
        ):
            state.actor_target = deepcopy(state.actor).to(self.device).requires_grad_(False)

        if (
            state.actor_optim is None
            and cfg.inner_actor_adaptation != "frozen"
            and cfg.inner_actor_updates_per_action > 0
        ):
            state.actor_optim = self._new_optimizer(state.actor, "actor")
        if (
            state.critic_optim is None
            and cfg.inner_critic_adaptation != "frozen"
            and cfg.inner_critic_updates_per_action > 0
        ):
            state.critic_optim = self._new_optimizer(state.critic, "critic")

        mode = str(cfg.inner_temperature_mode)
        if mode == "inherit_outer":
            state.log_alpha = state.temperature_optim = None
            state.alpha_fixed = self.agent.alpha.detach().clone()
        elif mode == "fixed":
            state.log_alpha = state.temperature_optim = None
            state.alpha_fixed = torch.tensor(
                float(cfg.inner_temperature), device=self.device
            )
        elif cfg.inner_operator == "sac":
            if state.log_alpha is None:
                state.log_alpha = torch.nn.Parameter(
                    torch.log(torch.tensor(float(cfg.inner_temperature), device=self.device))
                )
            state.alpha_fixed = None
            if (
                state.temperature_optim is None
                and cfg.inner_temperature_updates_per_action > 0
            ):
                state.temperature_optim = torch.optim.Adam(
                    [state.log_alpha],
                    lr=float(cfg.inner_temperature_lr),
                    eps=float(cfg.inner_adam_eps),
                    capturable=self.device.type == "cuda",
                )

        if state.replay is None:
            state.replay = LatentReplayBuffer(
                capacity=cfg.inner_replay_capacity,
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                device=self.device,
            )
        elif outer_changed:
            # Latents are coordinates of the current encoder/dynamics and may
            # never survive a representation update.
            state.replay.clear()

        state.outer_version = self.agent.outer_version
        state.critic_steps = state.actor_steps = state.temperature_steps = 0
        state.target_steps = state.critic_target_steps = state.actor_target_steps = 0
        state.replay_draws = 0
        state.policy_evaluations = state.q_evaluations = 0
        state.sampled_indices.clear()
        state.sampled_ids.clear()
        state.actor.train(cfg.inner_actor_adaptation != "frozen")
        state.critic.train(cfg.inner_critic_adaptation != "frozen")
        state.critic_target.eval()
        if state.actor_target is not None:
            state.actor_target.eval()

    def make_modules_for_compatibility(self):
        """Legacy test/debug hook returning freshly created inner modules."""
        actor = self._adapt_module(self.model._pi, "actor")
        critic = self._adapt_module(self.model._Qs, "critic")
        target = deepcopy(critic).to(self.device).requires_grad_(False)
        actor_optim = self._new_optimizer(actor, "actor")
        critic_optim = self._new_optimizer(critic, "critic")
        return (
            actor,
            critic,
            target,
            actor_optim,
            critic_optim,
            trainable_parameters(actor),
            trainable_parameters(critic),
        )

    def _policy_action(
        self,
        z,
        policy,
        *,
        mode,
        generator,
        std_scale=1.0,
        noise_std=0.0,
        inner_bounds=True,
    ):
        kwargs = {}
        if inner_bounds:
            kwargs.update(
                log_std_min=self.cfg.inner_log_std_min,
                log_std_max=self.cfg.inner_log_std_max,
            )
        if mode == "policy_sample":
            return self.model.pi(
                z,
                policy=policy,
                generator=generator,
                std_scale=std_scale,
                **kwargs,
            )
        mean, info = self.model.pi(z, policy=policy, deterministic=True, **kwargs)
        if mode == "mean":
            return mean, info
        if mode != "mean_plus_gaussian":
            raise ValueError(f"Unknown action sampling mode: {mode!r}")
        noise = torch.randn(
            mean.shape,
            device=mean.device,
            dtype=mean.dtype,
            generator=generator,
        ) * float(noise_std)
        return (mean + noise).clamp(-1.0, 1.0), info

    @torch.no_grad()
    def _collect_round(self, root_z):
        cfg, state = self.cfg, self.state
        count = int(cfg.inner_rollouts_per_round)
        horizon = int(cfg.inner_rollout_horizon)
        if count == 0:
            return {"lengths": [], "reward_sums": [], "discounted_rewards": [], "terminated": []}

        z = root_z.expand(count, -1).clone()
        alive = torch.ones(count, dtype=torch.bool, device=self.device)
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        reward_sums = torch.zeros(count, device=self.device)
        discounted_rewards = torch.zeros(count, device=self.device)
        terminated_rollout = torch.zeros(count, dtype=torch.bool, device=self.device)
        discounts = torch.ones(count, device=self.device)

        state.actor.eval()
        self.model.eval()
        with self.rng.fork("collection") as generator:
            for _ in range(horizon):
                active = torch.nonzero(alive, as_tuple=False).squeeze(-1)
                if active.numel() == 0:
                    break
                active_z = z[active]
                std_scale = float(cfg.inner_behavior_std_scale)
                behavior_mode = str(cfg.inner_behavior_action)
                if behavior_mode == "policy_sample" and std_scale == 0.0:
                    behavior_mode = "mean"
                action, _ = self._policy_action(
                    active_z,
                    state.actor,
                    mode=behavior_mode,
                    generator=generator,
                    std_scale=max(std_scale, 1e-12),
                    noise_std=cfg.inner_behavior_noise_std,
                )
                state.policy_evaluations += int(active.numel())
                reward = td_math.two_hot_inv(self.model.reward(active_z, action), cfg)
                next_z = self.model.next(active_z, action)
                if cfg.episodic:
                    terminated = (
                        self.model.termination(next_z)
                        > float(cfg.inner_termination_threshold)
                    ).float()
                else:
                    terminated = torch.zeros(active.numel(), 1, device=self.device)

                state.replay.add_batch(active_z, action, reward, next_z, terminated)
                reward_vector = reward.squeeze(-1)
                lengths[active] += 1
                reward_sums[active] += reward_vector
                discounted_rewards[active] += discounts[active] * reward_vector
                discounts[active] *= float(self.agent.discount)
                z[active] = next_z
                just_terminated = terminated.squeeze(-1) >= 0.5
                terminated_rollout[active] |= just_terminated
                alive[active] = ~just_terminated

        if cfg.inner_actor_adaptation != "frozen":
            state.actor.train()
        return {
            "lengths": lengths.tolist(),
            "reward_sums": reward_sums.tolist(),
            "discounted_rewards": discounted_rewards.tolist(),
            "terminated": terminated_rollout.tolist(),
        }

    def _sample_batch(self):
        replacement = self.cfg.inner_replay_sampling == "with_replacement"
        batch = self.state.replay.sample(
            self.cfg.inner_batch_size,
            replacement=replacement,
            generator=self.rng.generator("replay"),
        )
        self.state.replay_draws += int(batch["indices"].numel())
        self.state.sampled_indices.append(batch["indices"].detach())
        self.state.sampled_ids.append(batch["sample_ids"].detach())
        return batch

    def _bootstrap_q(self, z, action):
        source = str(self.cfg.inner_bootstrap_source)
        kwargs = {"reduction": self.cfg.inner_q_target_reduction}
        if source == "inner_target":
            kwargs["qs"] = self.state.critic_target
        elif source == "outer_target":
            kwargs["target"] = True
        elif source == "outer_online":
            pass
        else:
            raise ValueError(f"Unknown inner bootstrap source: {source!r}")
        return self.model.Q(z, action, **kwargs)

    def _sac_critic_step(self, batch, alpha):
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        with torch.no_grad(), self.rng.fork("bootstrap") as generator:
            next_action, next_info = self.model.pi(
                batch["next_z"],
                policy=state.actor,
                generator=generator,
                log_std_min=cfg.inner_log_std_min,
                log_std_max=cfg.inner_log_std_max,
            )
            next_q = self._bootstrap_q(batch["next_z"], next_action)
            target_q = batch["reward"] + float(self.agent.discount) * (
                1.0 - batch["terminated"]
            ) * (next_q - alpha * next_info["log_prob"])
        state.policy_evaluations += batch_size
        state.q_evaluations += batch_size

        predictions = self.model.q_predictions(
            batch["z"], batch["action"], qs=state.critic
        )
        state.q_evaluations += batch_size
        critic_loss = self.model.critic_loss(predictions, target_q)
        state.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_params = trainable_parameters(state.critic)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_params, float(cfg.inner_critic_grad_clip_norm)
        )
        state.critic_optim.step()
        state.critic_optim.zero_grad(set_to_none=True)
        state.critic_steps += 1
        state.critic_lifetime_steps += 1

        values = self.model.q_backend.decode(predictions.detach())
        clip_fraction = 0.0
        if self.cfg.q_representation == "distributional":
            symlog_target = td_math.symlog(target_q.detach())
            clip_fraction = float(
                (
                    (symlog_target <= float(self.cfg.q_vmin))
                    | (symlog_target >= float(self.cfg.q_vmax))
                )
                .float()
                .mean()
                .cpu()
            )
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "critic_grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "q_mean": float(values.mean().cpu()),
            "q_abs_mean": float(values.abs().mean().cpu()),
            "q_target_mean": float(target_q.mean().cpu()),
            "q_target_clip_fraction": clip_fraction,
            "td_error_abs_mean": float(
                (values - target_q.unsqueeze(0)).abs().mean().cpu()
            ),
        }

    @staticmethod
    def _gaussian_kl(inner_info, outer_info):
        inner_log_std = inner_info["log_std"]
        outer_log_std = outer_info["log_std"]
        mean_delta = inner_info["pre_tanh_mean"] - outer_info["pre_tanh_mean"]
        ratio = (
            inner_log_std.mul(2).exp() + mean_delta.square()
        ) / outer_log_std.mul(2).exp().clamp_min(1e-12)
        return (outer_log_std - inner_log_std + 0.5 * ratio - 0.5).sum(
            dim=-1, keepdim=True
        )

    def _sac_policy_step(self, batch, *, update_temperature, update_actor, alpha):
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        with self.rng.fork("gradient_policy") as generator:
            action, info = self.model.pi(
                batch["z"],
                policy=state.actor,
                generator=generator,
                log_std_min=cfg.inner_log_std_min,
                log_std_max=cfg.inner_log_std_max,
            )
            state.policy_evaluations += batch_size

            metrics = {}
            if update_temperature:
                target_entropy = (
                    -float(cfg.action_dim)
                    if cfg.inner_target_entropy == "auto"
                    else float(cfg.inner_target_entropy)
                )
                temperature_loss = -(
                    state.log_alpha * (info["log_prob"] + target_entropy).detach()
                ).mean()
                state.temperature_optim.zero_grad(set_to_none=True)
                temperature_loss.backward()
                temperature_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [state.log_alpha], float(cfg.inner_temperature_grad_clip_norm)
                )
                state.temperature_optim.step()
                state.temperature_steps += 1
                metrics.update(
                    temperature_loss=float(temperature_loss.detach().cpu()),
                    temperature_grad_norm=float(
                        torch.as_tensor(temperature_grad_norm).detach().cpu()
                    ),
                )

            if update_actor:
                requires_grad = [parameter.requires_grad for parameter in state.critic.parameters()]
                for parameter in state.critic.parameters():
                    parameter.requires_grad_(False)
                try:
                    q_pi = self.model.Q(
                        batch["z"],
                        action,
                        qs=state.critic,
                        reduction=cfg.inner_q_actor_reduction,
                    )
                    state.q_evaluations += batch_size
                    actor_loss_values = alpha * info["log_prob"] - q_pi
                    kl = torch.zeros_like(actor_loss_values)
                    if float(cfg.inner_outer_policy_kl_coef) > 0.0:
                        _, outer_info = self.model.pi(
                            batch["z"],
                            policy=state.actor_anchor,
                            deterministic=True,
                        )
                        state.policy_evaluations += batch_size
                        kl = self._gaussian_kl(info, outer_info)
                        actor_loss_values = actor_loss_values + float(
                            cfg.inner_outer_policy_kl_coef
                        ) * kl
                    actor_loss = actor_loss_values.mean()
                finally:
                    for parameter, flag in zip(state.critic.parameters(), requires_grad):
                        parameter.requires_grad_(flag)

                state.actor_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_params = trainable_parameters(state.actor)
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor_params, float(cfg.inner_actor_grad_clip_norm)
                )
                state.actor_optim.step()
                state.actor_steps += 1
                state.actor_lifetime_steps += 1
                metrics.update(
                    actor_loss=float(actor_loss.detach().cpu()),
                    actor_grad_norm=float(torch.as_tensor(actor_grad_norm).detach().cpu()),
                    actor_q_mean=float(q_pi.detach().mean().cpu()),
                    actor_entropy=float(info["entropy"].detach().mean().cpu()),
                    outer_policy_kl=float(kl.detach().mean().cpu()),
                )
            return metrics

    def _td3_critic_step(self, batch):
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        with torch.no_grad(), self.rng.fork("bootstrap") as generator:
            next_action, _ = self.model.pi(
                batch["next_z"], policy=state.actor_target, deterministic=True
            )
            noise = torch.randn(
                next_action.shape,
                device=self.device,
                dtype=next_action.dtype,
                generator=generator,
            ) * float(cfg.inner_td3_target_noise_std)
            noise = noise.clamp(
                -float(cfg.inner_td3_target_noise_clip),
                float(cfg.inner_td3_target_noise_clip),
            )
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            next_q = self._bootstrap_q(batch["next_z"], next_action)
            target_q = batch["reward"] + float(self.agent.discount) * (
                1.0 - batch["terminated"]
            ) * next_q
        state.policy_evaluations += batch_size
        state.q_evaluations += batch_size

        predictions = self.model.q_predictions(
            batch["z"], batch["action"], qs=state.critic
        )
        state.q_evaluations += batch_size
        critic_loss = self.model.critic_loss(predictions, target_q)
        state.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_params = trainable_parameters(state.critic)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_params, float(cfg.inner_critic_grad_clip_norm)
        )
        state.critic_optim.step()
        state.critic_optim.zero_grad(set_to_none=True)
        state.critic_steps += 1
        state.critic_lifetime_steps += 1
        values = self.model.q_backend.decode(predictions.detach())
        clip_fraction = 0.0
        if self.cfg.q_representation == "distributional":
            symlog_target = td_math.symlog(target_q.detach())
            clip_fraction = float(
                (
                    (symlog_target <= float(self.cfg.q_vmin))
                    | (symlog_target >= float(self.cfg.q_vmax))
                )
                .float()
                .mean()
                .cpu()
            )
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "critic_grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "q_mean": float(values.mean().cpu()),
            "q_abs_mean": float(values.abs().mean().cpu()),
            "q_target_mean": float(target_q.mean().cpu()),
            "q_target_clip_fraction": clip_fraction,
            "td_error_abs_mean": float(
                (values - target_q.unsqueeze(0)).abs().mean().cpu()
            ),
        }

    def _td3_actor_step(self, batch):
        state, cfg = self.state, self.cfg
        action, _ = self.model.pi(batch["z"], policy=state.actor, deterministic=True)
        batch_size = int(batch["z"].shape[0])
        state.policy_evaluations += batch_size
        requires_grad = [parameter.requires_grad for parameter in state.critic.parameters()]
        for parameter in state.critic.parameters():
            parameter.requires_grad_(False)
        try:
            q_pi = self.model.Q(
                batch["z"],
                action,
                qs=state.critic,
                reduction=cfg.inner_q_actor_reduction,
                generator=self.rng.generator("gradient_policy"),
            )
            state.q_evaluations += batch_size
            anchor_l2 = torch.zeros_like(q_pi)
            if float(cfg.inner_outer_action_l2_coef) > 0.0:
                with torch.no_grad():
                    outer_action, _ = self.model.pi(
                        batch["z"], policy=state.actor_anchor, deterministic=True
                    )
                state.policy_evaluations += batch_size
                anchor_l2 = (action - outer_action).square().sum(dim=-1, keepdim=True)
            actor_loss = (
                -q_pi + float(cfg.inner_outer_action_l2_coef) * anchor_l2
            ).mean()
        finally:
            for parameter, flag in zip(state.critic.parameters(), requires_grad):
                parameter.requires_grad_(flag)
        state.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_params = trainable_parameters(state.actor)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_params, float(cfg.inner_actor_grad_clip_norm)
        )
        state.actor_optim.step()
        state.actor_steps += 1
        state.actor_lifetime_steps += 1
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "actor_q_mean": float(q_pi.detach().mean().cpu()),
            "outer_action_l2": float(anchor_l2.detach().mean().cpu()),
        }

    def _maybe_update_targets(self, *, critic_updated, actor_updated):
        state, cfg = self.state, self.cfg
        if (
            critic_updated
            and state.critic_lifetime_steps > 0
            and state.critic_lifetime_steps
            % int(cfg.inner_critic_target_update_interval)
            == 0
            and cfg.inner_bootstrap_source == "inner_target"
        ):
            polyak_update(state.critic, state.critic_target, cfg.inner_critic_target_tau)
            state.target_steps += 1
            state.critic_target_steps += 1
        if (
            actor_updated
            and cfg.inner_operator == "td3"
            and state.actor_target is not None
            and state.actor_lifetime_steps > 0
            and state.actor_lifetime_steps
            % int(cfg.inner_actor_target_update_interval)
            == 0
        ):
            polyak_update(state.actor, state.actor_target, cfg.inner_actor_target_tau)
            state.target_steps += 1
            state.actor_target_steps += 1

    def _run_updates(self, round_index, allocations):
        critic_count = allocations["critic"][round_index]
        actor_count = allocations["actor"][round_index]
        temperature_count = allocations["temperature"][round_index]
        slots = max(critic_count, actor_count, temperature_count)
        metrics = []
        for slot in range(slots):
            do_critic = slot < critic_count
            do_actor = slot < actor_count
            do_temperature = slot < temperature_count
            batch = self._sample_batch()
            alpha = self.alpha.detach()
            slot_metrics = {}
            if self.cfg.inner_operator == "sac":
                if do_critic:
                    # Critic dropout and pair selection are isolated from the
                    # stochastic policy-gradient stream.
                    with self.rng.fork("bootstrap"):
                        slot_metrics.update(self._sac_critic_step(batch, alpha))
                if do_temperature or do_actor:
                    slot_metrics.update(
                        self._sac_policy_step(
                            batch,
                            update_temperature=do_temperature,
                            update_actor=do_actor,
                            alpha=alpha,
                        )
                    )
            else:
                if do_critic:
                    with self.rng.fork("bootstrap"):
                        slot_metrics.update(self._td3_critic_step(batch))
                if do_actor:
                    with self.rng.fork("gradient_policy"):
                        slot_metrics.update(self._td3_actor_step(batch))
            self._maybe_update_targets(
                critic_updated=do_critic,
                actor_updated=do_actor,
            )
            metrics.append(slot_metrics)
        return metrics

    @torch.no_grad()
    def _execute_policy(self, root_z, policy, *, eval_mode):
        mode = "mean" if eval_mode else str(self.cfg.inner_execution_action)
        std_scale = float(self.cfg.inner_execution_std_scale)
        if mode == "policy_sample" and std_scale == 0.0:
            mode = "mean"
        was_training = policy.training
        policy.eval()
        with self.rng.fork("execution") as generator:
            action, _ = self._policy_action(
                root_z,
                policy,
                mode=mode,
                generator=generator,
                std_scale=max(std_scale, 1e-12),
                noise_std=self.cfg.inner_execution_noise_std,
            )
        self.state.policy_evaluations += int(root_z.shape[0])
        policy.train(was_training)
        return action

    @torch.no_grad()
    def _evaluate_policy_trajectory(self, root_z, policy, generator, *, stochastic):
        count = int(getattr(self.cfg, "inner_diagnostic_rollouts", 0))
        if count <= 0:
            return None
        z = root_z.expand(count, -1).clone()
        continuation = torch.ones(count, 1, device=self.device)
        discount = torch.ones(count, 1, device=self.device)
        score = torch.zeros(count, 1, device=self.device)
        soft_score = torch.zeros(count, 1, device=self.device)
        alpha = self.agent.alpha.detach()
        for _ in range(int(self.cfg.inner_rollout_horizon)):
            action, info = self.model.pi(
                z,
                policy=policy,
                deterministic=not stochastic,
                generator=generator,
                log_std_min=self.cfg.inner_log_std_min,
                log_std_max=self.cfg.inner_log_std_max,
            )
            self.state.policy_evaluations += count
            reward = td_math.two_hot_inv(self.model.reward(z, action), self.cfg)
            score += discount * continuation * reward
            soft_score += discount * continuation * (
                reward - alpha * info["log_prob"]
            )
            z = self.model.next(z, action)
            if self.cfg.episodic:
                terminated = (
                    self.model.termination(z)
                    > float(self.cfg.inner_termination_threshold)
                ).float()
                continuation *= 1.0 - terminated
            discount *= float(self.agent.discount)

        terminal_action, terminal_info = self.model.pi(
            z,
            policy=policy,
            deterministic=not stochastic,
            generator=generator,
            log_std_min=self.cfg.inner_log_std_min,
            log_std_max=self.cfg.inner_log_std_max,
        )
        self.state.policy_evaluations += count
        terminal_q = self.model.Q(
            z,
            terminal_action,
            target=True,
            reduction="mean_all",
        )
        self.state.q_evaluations += count
        score += discount * continuation * terminal_q
        soft_score += discount * continuation * (
            terminal_q - alpha * terminal_info["log_prob"]
        )
        return {
            "score": float(score.mean().cpu()),
            "soft_score": float(soft_score.mean().cpu()),
            "model_steps": count * int(self.cfg.inner_rollout_horizon),
        }

    @torch.no_grad()
    def _diagnostics(self, root_z, improved_policy):
        start = time.perf_counter()
        with self.rng.fork("diagnostics") as generator:
            outer_action, _ = self.model.pi(root_z, deterministic=True)
            improved_action, _ = self.model.pi(
                root_z, policy=improved_policy, deterministic=True
            )
            self.state.policy_evaluations += 2 * int(root_z.shape[0])
            outer_q = self.model.Q(
                root_z,
                outer_action,
                target=True,
                reduction="mean_all",
            )
            self.state.q_evaluations += 2 * int(root_z.shape[0])
            improved_q = self.model.Q(
                root_z,
                improved_action,
                target=True,
                reduction="mean_all",
            )
            action_delta = torch.linalg.vector_norm(
                improved_action - outer_action, dim=-1
            ).mean()

            state_before = generator.get_state()
            outer_eval = self._evaluate_policy_trajectory(
                root_z,
                self.model._pi,
                generator,
                stochastic=self.cfg.inner_operator == "sac",
            )
            generator.set_state(state_before)
            improved_eval = self._evaluate_policy_trajectory(
                root_z,
                improved_policy,
                generator,
                stochastic=self.cfg.inner_operator == "sac",
            )

        metrics = {
            "inner_policy_mean_delta_l2": float(action_delta.cpu()),
            "inner_fixed_target_q_action_gain": float((improved_q - outer_q).mean().cpu()),
            # One-release compatibility alias. The evaluator is now fixed and
            # cannot be changed by the adapted critic.
            "inner_outer_q_gain": float((improved_q - outer_q).mean().cpu()),
            "inner_fixed_target_q_outer": float(outer_q.mean().cpu()),
            "inner_fixed_target_q_improved": float(improved_q.mean().cpu()),
            "inner_fixed_target_q_abs_mean": float(
                torch.stack((outer_q.abs().mean(), improved_q.abs().mean())).mean().cpu()
            ),
            "inner_fixed_evaluator_alpha": float(self.agent.alpha.detach().cpu()),
            "inner_diagnostic_seconds": time.perf_counter() - start,
        }
        if outer_eval is not None and improved_eval is not None:
            metrics.update(
                inner_predicted_j_outer=outer_eval["score"],
                inner_predicted_j_improved=improved_eval["score"],
                inner_predicted_j_gain=improved_eval["score"] - outer_eval["score"],
                inner_predicted_soft_j_outer=outer_eval["soft_score"],
                inner_predicted_soft_j_improved=improved_eval["soft_score"],
                inner_predicted_soft_j_gain=(
                    improved_eval["soft_score"] - outer_eval["soft_score"]
                ),
                inner_fixed_alpha_soft_j_outer=outer_eval["soft_score"],
                inner_fixed_alpha_soft_j_improved=improved_eval["soft_score"],
                inner_fixed_alpha_soft_j_gain=(
                    improved_eval["soft_score"] - outer_eval["soft_score"]
                ),
                inner_diagnostic_model_steps=(
                    outer_eval["model_steps"] + improved_eval["model_steps"]
                ),
            )
        else:
            metrics["inner_diagnostic_model_steps"] = 0.0
        return metrics

    @staticmethod
    def _stats(values):
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        tensor = torch.as_tensor(values, dtype=torch.float64)
        return (
            float(tensor.mean()),
            float(tensor.std(unbiased=False)),
            float(tensor.min()),
            float(tensor.max()),
        )

    @staticmethod
    def _average_update_metrics(history):
        grouped = {}
        for item in history:
            for key, value in item.items():
                grouped.setdefault(key, []).append(float(value))
        return {
            f"inner_{key}": sum(values) / len(values)
            for key, values in grouped.items()
            if values
        }

    def _base_metrics(self, *, active, action_seconds=0.0):
        return {
            "inner_active": float(active),
            "inner_actions": 1.0,
            "inner_model_steps_budget": float(self.cfg.inner_model_step_budget),
            "inner_steps": 0.0,
            "inner_model_steps": 0.0,
            "inner_total_model_steps": 0.0,
            "inner_rounds": 0.0,
            "inner_iterations": 0.0,
            "inner_rollout_horizon": float(self.cfg.inner_rollout_horizon),
            "inner_horizon_ratio": float(self.cfg.inner_horizon_ratio),
            "inner_requested_rollouts": 0.0,
            "inner_rollouts": 0.0,
            "inner_updates": 0.0,
            "inner_update_slots": 0.0,
            "inner_critic_optimizer_steps": 0.0,
            "inner_actor_optimizer_steps": 0.0,
            "inner_temperature_optimizer_steps": 0.0,
            "inner_target_updates": 0.0,
            "inner_critic_target_updates": 0.0,
            "inner_actor_target_updates": 0.0,
            "inner_policy_evaluations": 0.0,
            "inner_q_evaluations": 0.0,
            "inner_replay_draws": 0.0,
            "inner_replay_unique_fraction": 0.0,
            "inner_buffer_size": 0.0,
            "inner_buffer_capacity": 0.0,
            "inner_buffer_fill_ratio": 0.0,
            "inner_return_mean": 0.0,
            "inner_return_std": 0.0,
            "inner_return_min": 0.0,
            "inner_return_max": 0.0,
            "inner_behavior_reward_sum_mean": 0.0,
            "inner_behavior_discounted_reward_mean": 0.0,
            "inner_rollout_len_mean": 0.0,
            "inner_rollout_len_std": 0.0,
            "inner_rollout_len_min": 0.0,
            "inner_rollout_len_max": 0.0,
            "inner_termination_rate": 0.0,
            "inner_alpha": float(self.agent.alpha.detach().cpu()),
            "inner_q_abs_mean": 0.0,
            "inner_alpha_to_abs_q": 0.0,
            "inner_action_seconds": float(action_seconds),
        }

    def _act_none(self, root_z, *, eval_mode, start):
        action = self._execute_policy(root_z, self.model._pi, eval_mode=eval_mode)
        metrics = self._base_metrics(active=False)
        metrics["inner_policy_evaluations"] = 1.0
        metrics["inner_action_seconds"] = time.perf_counter() - start
        return action[0], metrics, []

    def _act_rl(self, root_z, *, t0, eval_mode, start):
        cfg, state = self.cfg, self.state
        setup_start = time.perf_counter()
        # LoRA adapter initialization uses ordinary PyTorch initializers; fork
        # it onto the private optimization stream so act() cannot advance the
        # outer learner's global RNG.
        with self.rng.fork("initialization"):
            self._prepare_workspace(t0=t0)
        setup_seconds = time.perf_counter() - setup_start
        allocations = {
            "critic": allocate_across_rounds(
                cfg.inner_critic_updates_per_action, cfg.inner_rounds
            ),
            "actor": allocate_across_rounds(
                cfg.inner_actor_updates_per_action, cfg.inner_rounds
            ),
            "temperature": allocate_across_rounds(
                cfg.inner_temperature_updates_per_action, cfg.inner_rounds
            ),
        }
        all_lengths, reward_sums, discounted_rewards, terminated = [], [], [], []
        update_history = []
        rollout_seconds = update_seconds = 0.0
        update_slots = 0
        for round_index in range(int(cfg.inner_rounds)):
            rollout_start = time.perf_counter()
            rollout = self._collect_round(root_z)
            rollout_seconds += time.perf_counter() - rollout_start
            all_lengths.extend(rollout["lengths"])
            reward_sums.extend(rollout["reward_sums"])
            discounted_rewards.extend(rollout["discounted_rewards"])
            terminated.extend(rollout["terminated"])

            update_start = time.perf_counter()
            round_metrics = self._run_updates(round_index, allocations)
            update_seconds += time.perf_counter() - update_start
            update_history.extend(round_metrics)
            update_slots += len(round_metrics)

        execution_start = time.perf_counter()
        action = self._execute_policy(root_z, state.actor, eval_mode=eval_mode)
        execution_seconds = time.perf_counter() - execution_start
        diagnostic_metrics = self._diagnostics(root_z, state.actor)

        reward_stats = self._stats(reward_sums)
        discounted_stats = self._stats(discounted_rewards)
        length_stats = self._stats(all_lengths)
        metrics = self._base_metrics(active=True)
        metrics.update(
            inner_rounds=float(cfg.inner_rounds),
            inner_iterations=float(cfg.inner_rounds),
            inner_rollouts=float(len(all_lengths)),
            inner_requested_rollouts=float(
                cfg.inner_rounds * cfg.inner_rollouts_per_round
            ),
            inner_rollout_count=float(len(all_lengths)),
            inner_steps=float(sum(all_lengths)),
            inner_model_steps=float(sum(all_lengths)),
            inner_updates=float(update_slots),
            inner_update_slots=float(update_slots),
            inner_critic_optimizer_steps=float(state.critic_steps),
            inner_actor_optimizer_steps=float(state.actor_steps),
            inner_temperature_optimizer_steps=float(state.temperature_steps),
            inner_target_updates=float(state.target_steps),
            inner_critic_target_updates=float(state.critic_target_steps),
            inner_actor_target_updates=float(state.actor_target_steps),
            inner_policy_evaluations=float(state.policy_evaluations),
            inner_q_evaluations=float(state.q_evaluations),
            inner_replay_draws=float(state.replay_draws),
            inner_buffer_size=float(state.replay.size),
            inner_buffer_capacity=float(state.replay.capacity),
            inner_buffer_fill_ratio=float(state.replay.size / state.replay.capacity),
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
            inner_termination_rate=(
                float(sum(terminated) / len(terminated)) if terminated else 0.0
            ),
            inner_alpha=float(self.alpha.detach().cpu()),
            inner_actor_trainable_params=float(trainable_parameter_count(state.actor)),
            inner_critic_trainable_params=float(trainable_parameter_count(state.critic)),
            inner_temperature_trainable_params=float(
                state.log_alpha.numel() if state.log_alpha is not None else 0
            ),
            inner_setup_seconds=setup_seconds,
            inner_rollout_seconds=rollout_seconds,
            inner_update_seconds=update_seconds,
            inner_execution_seconds=execution_seconds,
        )
        if state.sampled_ids:
            sampled = torch.cat(state.sampled_ids)
            metrics["inner_replay_unique_fraction"] = float(
                sampled.unique().numel() / sampled.numel()
            )
        metrics.update(self._average_update_metrics(update_history))
        metrics.update(diagnostic_metrics)
        metrics["inner_total_model_steps"] = float(
            metrics["inner_model_steps"]
            + metrics.get("inner_diagnostic_model_steps", 0.0)
        )
        # Counters include the fixed-evaluator calls performed above.
        metrics["inner_policy_evaluations"] = float(state.policy_evaluations)
        metrics["inner_q_evaluations"] = float(state.q_evaluations)
        q_scale = float(metrics.get("inner_q_abs_mean", 0.0))
        if q_scale <= 0.0:
            q_scale = float(metrics.get("inner_fixed_target_q_abs_mean", 0.0))
        metrics["inner_q_abs_mean"] = q_scale
        metrics["inner_alpha_to_abs_q"] = float(self.alpha.detach().cpu()) / max(
            q_scale, 1e-8
        )
        metrics["inner_action_seconds"] = time.perf_counter() - start
        return action[0], metrics, all_lengths

    def _act_mppi(self, root_z, *, t0, eval_mode, start):
        from .mppi import mppi_plan

        scope = str(self.cfg.inner_mppi_warm_start_scope)
        if scope == "action" or (scope == "episode" and t0):
            previous_mean = None
        else:
            previous_mean = self._mppi_prev_mean
        planner_start = time.perf_counter()
        with self.rng.fork("mppi") as generator:
            result = mppi_plan(
                model=self.model,
                root_z=root_z,
                horizon=self.cfg.inner_rollout_horizon,
                iterations=self.cfg.inner_rounds,
                num_samples=self.cfg.inner_mppi_num_samples,
                num_elites=self.cfg.inner_mppi_num_elites,
                num_pi_trajs=self.cfg.inner_mppi_num_pi_trajs,
                temperature=self.cfg.inner_mppi_temperature,
                min_std=self.cfg.inner_mppi_min_std,
                max_std=self.cfg.inner_mppi_max_std,
                discount=float(self.agent.discount),
                q_reduction=self.cfg.mppi_terminal_q_reduction,
                termination_threshold=self.cfg.inner_termination_threshold,
                generator=generator,
                previous_mean=previous_mean,
                t0=t0,
                eval_mode=eval_mode,
            )
        planner_seconds = time.perf_counter() - planner_start
        self._mppi_prev_mean = None if scope == "action" else result.next_mean.detach()
        metrics = self._base_metrics(active=True)
        metrics.update(result.metrics)
        metrics.update(
            inner_rounds=float(self.cfg.inner_rounds),
            inner_iterations=float(self.cfg.inner_rounds),
            inner_rollouts=float(
                self.cfg.inner_rounds * self.cfg.inner_mppi_num_samples
                + self.cfg.inner_mppi_num_pi_trajs
            ),
            inner_requested_rollouts=float(
                self.cfg.inner_rounds * self.cfg.inner_mppi_num_samples
                + self.cfg.inner_mppi_num_pi_trajs
            ),
            inner_steps=float(result.model_steps),
            inner_model_steps=float(result.model_steps),
            inner_total_model_steps=float(result.model_steps),
            inner_policy_evaluations=float(
                result.metrics["planner_policy_evaluations"] + 1
            ),
            inner_q_evaluations=float(result.metrics["planner_q_evaluations"] + 2),
            inner_mppi_seconds=planner_seconds,
            inner_action_seconds=time.perf_counter() - start,
        )
        with torch.no_grad():
            outer_action, _ = self.model.pi(root_z, deterministic=True)
            outer_q = self.model.Q(
                root_z, outer_action, target=True, reduction="mean_all"
            )
            improved_q = self.model.Q(
                root_z,
                result.action.unsqueeze(0),
                target=True,
                reduction="mean_all",
            )
        gain = float((improved_q - outer_q).mean().cpu())
        metrics["inner_fixed_target_q_action_gain"] = gain
        metrics["inner_outer_q_gain"] = gain
        metrics["inner_fixed_target_q_outer"] = float(outer_q.mean().cpu())
        metrics["inner_fixed_target_q_improved"] = float(improved_q.mean().cpu())
        q_scale = float(
            torch.stack((outer_q.abs().mean(), improved_q.abs().mean())).mean().cpu()
        )
        metrics["inner_fixed_target_q_abs_mean"] = q_scale
        metrics["inner_q_abs_mean"] = q_scale
        metrics["inner_alpha_to_abs_q"] = float(self.agent.alpha.detach().cpu()) / max(
            q_scale, 1e-8
        )
        proposal_action = result.next_mean[0].clamp(-1.0, 1.0).unsqueeze(0)
        proposal_delta = float(
            torch.linalg.vector_norm(proposal_action - outer_action, dim=-1)
            .mean()
            .cpu()
        )
        metrics["inner_proposal_mean_delta_l2"] = proposal_delta
        metrics["inner_policy_mean_delta_l2"] = float(
            proposal_delta
        )
        candidate_lengths = [int(self.cfg.inner_rollout_horizon)] * int(
            self.cfg.inner_rounds * self.cfg.inner_mppi_num_samples
        )
        policy_lengths = [max(0, int(self.cfg.inner_rollout_horizon) - 1)] * int(
            self.cfg.inner_mppi_num_pi_trajs
        )
        return result.action, metrics, candidate_lengths + policy_lengths

    def act(self, root_z, *, t0=False, eval_mode=False):
        start = time.perf_counter()
        self.action_index += 1
        operator = str(self.cfg.inner_operator)
        if operator == "none" or self.cfg.inner_rounds == 0:
            action, metrics, lengths = self._act_none(
                root_z, eval_mode=eval_mode, start=start
            )
        elif operator == "mppi":
            action, metrics, lengths = self._act_mppi(
                root_z, t0=t0, eval_mode=eval_mode, start=start
            )
        else:
            action, metrics, lengths = self._act_rl(
                root_z, t0=t0, eval_mode=eval_mode, start=start
            )

        # Action-scoped tensors are explicitly released after producing the
        # action; episode/run scopes survive according to configuration.
        self._clear_expired(t0=False, include_action=True)
        return action, metrics, lengths
