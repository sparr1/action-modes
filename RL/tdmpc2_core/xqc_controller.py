"""Reusable XQC control heads for latent-state learners.

This module contains no replay or environment ownership.  It keeps the exact
XQC actor/critic, BatchNorm, categorical target, projection, target-update, and
delayed actor/temperature rules available to both AMBI-XQC's persistent outer
priors and its freshly reset per-action learner.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from RL.xqc_core import (
    XQCActor,
    XQCTwinCritic,
    _global_grad_norm,
    _optimizer_execution_kwargs,
    _polyak_update_parameter_lists_,
    _project_unit_weights_,
    _set_optimizer_lr,
    categorical_td_projection,
    linear_learning_rate,
    project_unit_rows_,
    select_lower_distribution,
)


def _positive_float(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number.")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@dataclass(frozen=True)
class LatentXQCConfig:
    actor_net_arch: tuple[int, ...] = (256, 256, 256, 256)
    critic_net_arch: tuple[int, ...] = (512, 512, 512, 512)
    num_atoms: int = 101
    vmin: float = -5.0
    vmax: float = 5.0
    tau: float = 0.005
    target_update_interval: int = 1
    policy_delay: int = 3
    init_temperature: float = 0.01
    target_entropy: float = -1.0
    adam_eps: float = 1e-8
    optimizer_backend: str = "auto"

    def __post_init__(self):
        actor_arch = tuple(
            _positive_int(width, "actor_net_arch") for width in self.actor_net_arch
        )
        critic_arch = tuple(
            _positive_int(width, "critic_net_arch") for width in self.critic_net_arch
        )
        if not actor_arch or not critic_arch:
            raise ValueError("XQC actor and critic architectures cannot be empty.")
        object.__setattr__(self, "actor_net_arch", actor_arch)
        object.__setattr__(self, "critic_net_arch", critic_arch)
        object.__setattr__(self, "num_atoms", _positive_int(self.num_atoms, "num_atoms"))
        if self.num_atoms < 2:
            raise ValueError("num_atoms must be at least two.")
        vmin, vmax = float(self.vmin), float(self.vmax)
        if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin >= vmax:
            raise ValueError("XQC support bounds must be finite and increasing.")
        object.__setattr__(self, "vmin", vmin)
        object.__setattr__(self, "vmax", vmax)
        tau = _positive_float(self.tau, "tau")
        if tau > 1.0:
            raise ValueError("tau must be in (0, 1].")
        object.__setattr__(self, "tau", tau)
        object.__setattr__(
            self,
            "target_update_interval",
            _positive_int(self.target_update_interval, "target_update_interval"),
        )
        object.__setattr__(
            self, "policy_delay", _positive_int(self.policy_delay, "policy_delay")
        )
        object.__setattr__(
            self,
            "init_temperature",
            _positive_float(self.init_temperature, "init_temperature"),
        )
        target_entropy = float(self.target_entropy)
        if not math.isfinite(target_entropy):
            raise ValueError("target_entropy must be finite.")
        object.__setattr__(self, "target_entropy", target_entropy)
        object.__setattr__(self, "adam_eps", _positive_float(self.adam_eps, "adam_eps"))
        backend = str(self.optimizer_backend).lower()
        if backend not in {"auto", "single_tensor", "foreach", "fused"}:
            raise ValueError("Unsupported XQC optimizer backend.")
        object.__setattr__(self, "optimizer_backend", backend)


@dataclass
class LatentXQCBatch:
    latents: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_latents: torch.Tensor
    bootstrap_mask: torch.Tensor
    discount: float | torch.Tensor

    @property
    def leading_shape(self):
        return tuple(self.latents.shape[:-1])

    def flattened(self, latent_dim: int, action_dim: int):
        if self.latents.shape[-1] != latent_dim:
            raise ValueError("Latent batch has the wrong feature width.")
        if self.next_latents.shape != self.latents.shape:
            raise ValueError("Current and next latent shapes must match.")
        if self.actions.shape[:-1] != self.leading_shape or self.actions.shape[-1] != action_dim:
            raise ValueError("Action batch does not align with the latent batch.")
        count = self.latents.numel() // latent_dim

        def scalar_field(value, name):
            value = torch.as_tensor(
                value, device=self.latents.device, dtype=self.latents.dtype
            )
            if value.numel() == 1:
                return value.reshape(1).expand(count)
            if tuple(value.shape) == self.leading_shape + (1,):
                return value.reshape(count)
            if tuple(value.shape) == self.leading_shape:
                return value.reshape(count)
            raise ValueError(f"{name} must contain one scalar per latent transition.")

        return {
            "latents": self.latents.reshape(count, latent_dim),
            "actions": self.actions.reshape(count, action_dim),
            "rewards": scalar_field(self.rewards, "rewards"),
            "next_latents": self.next_latents.reshape(count, latent_dim),
            "bootstrap_mask": scalar_field(self.bootstrap_mask, "bootstrap_mask"),
            "discount": scalar_field(self.discount, "discount"),
        }


@dataclass
class LatentXQCCriticObjective:
    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    current_log_probs: torch.Tensor
    target_probabilities: torch.Tensor
    current_values: torch.Tensor
    target_values: torch.Tensor
    target_head: torch.Tensor
    clip_fraction: torch.Tensor


@dataclass
class LatentXQCActorObjective:
    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    q_values: torch.Tensor
    minimum_q: torch.Tensor


class LatentXQCController(nn.Module):
    """XQC actor, online twin critic, target critic, and temperature."""

    def __init__(self, latent_dim, action_dim, config: LatentXQCConfig):
        super().__init__()
        self.latent_dim = _positive_int(latent_dim, "latent_dim")
        self.action_dim = _positive_int(action_dim, "action_dim")
        if not isinstance(config, LatentXQCConfig):
            raise TypeError("config must be LatentXQCConfig.")
        self.config = config
        self.actor = XQCActor(
            self.latent_dim, self.action_dim, config.actor_net_arch
        )
        self.critic = XQCTwinCritic(
            self.latent_dim,
            self.action_dim,
            config.critic_net_arch,
            config.num_atoms,
            vmin=config.vmin,
            vmax=config.vmax,
        )
        self.critic_target = XQCTwinCritic(
            self.latent_dim,
            self.action_dim,
            config.critic_net_arch,
            config.num_atoms,
            vmin=config.vmin,
            vmax=config.vmax,
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(config.init_temperature), dtype=torch.float32)
        )
        project_unit_rows_(self.actor)
        project_unit_rows_(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)
        self._refresh_cached_tensors()

    def _refresh_cached_tensors(self):
        self._actor_linear_weights = tuple(
            module.weight
            for module in self.actor.modules()
            if isinstance(module, nn.Linear)
        )
        self._critic_linear_weights = tuple(
            module.weight
            for module in self.critic.modules()
            if isinstance(module, nn.Linear)
        )
        self._critic_parameters = tuple(self.critic.parameters())
        self._target_parameters = tuple(self.critic_target.parameters())

    def _apply(self, fn):
        # ``Module.to``/``cuda``/dtype conversion may replace Parameter
        # objects. Projection and Polyak updates must always reference the
        # live tensors after that conversion, never the pre-move objects.
        result = super()._apply(fn)
        self._refresh_cached_tensors()
        return result

    @property
    def temperature(self):
        return self.log_temperature.exp()

    @property
    def target_entropy(self):
        return self.config.target_entropy

    @property
    def critic_signature(self):
        return {
            "q_representation": "xqc_c51",
            "num_q": 2,
            "num_atoms": self.config.num_atoms,
            "vmin": self.config.vmin,
            "vmax": self.config.vmax,
        }

    @torch.no_grad()
    def reset_prior_from_(self, source: "LatentXQCController"):
        if not isinstance(source, LatentXQCController):
            raise TypeError("source must be a LatentXQCController.")
        if self.critic_signature != source.critic_signature:
            raise ValueError("Cannot copy incompatible latent XQC controllers.")
        self.actor.load_state_dict(source.actor.state_dict())
        self.critic.load_state_dict(source.critic.state_dict())
        # A fresh inner target starts from the copied online critic, including
        # its BN buffers.  Subsequent target EMA updates parameters only.
        self.critic_target.load_state_dict(source.critic.state_dict())
        self.log_temperature.copy_(source.log_temperature)
        self._refresh_cached_tensors()
        return self

    def critic_objective(
        self,
        batch: LatentXQCBatch,
        *,
        next_noise: torch.Tensor,
        reward_scale: float | torch.Tensor = 1.0,
    ) -> LatentXQCCriticObjective:
        flat = batch.flattened(self.latent_dim, self.action_dim)
        leading = batch.leading_shape
        count = flat["latents"].shape[0]
        next_noise = torch.as_tensor(
            next_noise, device=flat["latents"].device, dtype=flat["latents"].dtype
        ).reshape(count, self.action_dim)
        if torch.is_tensor(reward_scale):
            scale = reward_scale.to(
                device=flat["latents"].device,
                dtype=flat["latents"].dtype,
            )
            if (
                scale.numel() != 1
                or not bool(torch.isfinite(scale).all())
                or not bool(scale > 0)
            ):
                raise ValueError("reward_scale must be one positive finite scalar.")
        else:
            scale_value = float(reward_scale)
            if not math.isfinite(scale_value) or scale_value <= 0.0:
                raise ValueError("reward_scale must be one positive finite scalar.")
            # Normal operation supplies the real-return scale as a Python
            # scalar. Validate it on the host so every inner update does not
            # introduce an otherwise unnecessary GPU synchronization.
            scale = flat["latents"].new_tensor(scale_value)

        alpha = self.temperature.detach()
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(
                flat["next_latents"], bn_mode="running", noise=next_noise
            )
            target_latents = torch.cat(
                (flat["latents"].detach(), flat["next_latents"].detach()), dim=0
            )
            target_actions = torch.cat((flat["actions"], next_actions), dim=0)
            target_joined = self.critic_target.log_probs(
                target_latents, target_actions, bn_mode="batch_no_update"
            )
            target_next = target_joined[:, count:]
            selected, target_values, target_head = select_lower_distribution(
                target_next, self.critic_target.support
            )
            target_probabilities, clip_fraction = categorical_td_projection(
                selected,
                flat["rewards"] / scale,
                flat["bootstrap_mask"],
                flat["discount"],
                alpha * next_log_prob,
                self.critic.support,
                validate_support=False,
            )

        joined_latents = torch.cat(
            (flat["latents"], flat["next_latents"].detach()), dim=0
        )
        joined_actions = torch.cat((flat["actions"], next_actions), dim=0)
        joined_log_probs = self.critic.log_probs(
            joined_latents, joined_actions, bn_mode="batch_update"
        )
        current_log_probs = joined_log_probs[:, :count]
        per_head = -(
            target_probabilities.unsqueeze(0) * current_log_probs
        ).sum(dim=-1)
        per_sample = per_head.sum(dim=0)
        current_values = self.critic.values_from_log_probs(current_log_probs)
        return LatentXQCCriticObjective(
            loss=per_sample.mean(),
            per_sample_loss=per_sample.reshape(leading),
            current_log_probs=current_log_probs.reshape(
                (2,) + leading + (self.config.num_atoms,)
            ),
            target_probabilities=target_probabilities.reshape(
                leading + (self.config.num_atoms,)
            ),
            current_values=current_values.reshape((2,) + leading),
            target_values=target_values.reshape(leading),
            target_head=target_head.reshape(leading),
            clip_fraction=clip_fraction,
        )

    def actor_objective(
        self,
        latents: torch.Tensor,
        *,
        actor_noise: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> LatentXQCActorObjective:
        if latents.shape[-1] != self.latent_dim:
            raise ValueError("Actor latents have the wrong feature width.")
        leading = tuple(latents.shape[:-1])
        count = latents.numel() // self.latent_dim
        flat_latents = latents.reshape(count, self.latent_dim)
        actor_noise = torch.as_tensor(
            actor_noise, device=latents.device, dtype=latents.dtype
        ).reshape(count, self.action_dim)
        alpha = self.temperature.detach() if alpha is None else alpha.detach()

        critic_requires_grad = tuple(
            parameter.requires_grad for parameter in self.critic.parameters()
        )
        self.critic.requires_grad_(False)
        try:
            actions, log_prob = self.actor.sample(
                flat_latents, bn_mode="batch_update", noise=actor_noise
            )
            log_q = self.critic.log_probs(
                flat_latents, actions, bn_mode="running"
            )
            q_values = self.critic.values_from_log_probs(log_q)
            minimum_q = q_values.min(dim=0).values
            per_sample = alpha * log_prob - minimum_q
        finally:
            for parameter, requires_grad in zip(
                self.critic.parameters(), critic_requires_grad
            ):
                parameter.requires_grad_(requires_grad)
        return LatentXQCActorObjective(
            loss=per_sample.mean(),
            per_sample_loss=per_sample.reshape(leading),
            log_prob=log_prob.reshape(leading),
            entropy=(-log_prob).reshape(leading),
            q_values=q_values.reshape((2,) + leading),
            minimum_q=minimum_q.reshape(leading),
        )

    @torch.no_grad()
    def sample_action(
        self,
        latents: torch.Tensor,
        *,
        deterministic=False,
        noise: torch.Tensor | None = None,
    ):
        if latents.shape[-1] != self.latent_dim:
            raise ValueError("Action latents have the wrong feature width.")
        leading = latents.shape[:-1]
        flat = latents.reshape(-1, self.latent_dim)
        if noise is not None:
            noise = torch.as_tensor(
                noise, dtype=flat.dtype, device=flat.device
            ).reshape(-1, self.action_dim)
        action, log_prob = self.actor.sample(
            flat,
            deterministic=bool(deterministic),
            bn_mode="running",
            noise=noise,
        )
        return (
            action.reshape(leading + (self.action_dim,)),
            log_prob.reshape(leading),
        )

    def make_workspace(
        self,
        *,
        actor_lr,
        critic_lr,
        actor_lr_end=None,
        critic_lr_end=None,
        transition_steps=1,
        optimizer_backend=None,
    ):
        return LatentXQCWorkspace(
            self,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_lr_end=actor_lr if actor_lr_end is None else actor_lr_end,
            critic_lr_end=critic_lr if critic_lr_end is None else critic_lr_end,
            transition_steps=transition_steps,
            optimizer_backend=(
                self.config.optimizer_backend
                if optimizer_backend is None
                else optimizer_backend
            ),
        )

    def clone_for_inner(self, *, actor_lr, critic_lr, transition_steps=1):
        clone = LatentXQCController(
            self.latent_dim, self.action_dim, copy.deepcopy(self.config)
        ).to(next(self.parameters()).device)
        clone.reset_prior_from_(self)
        return clone.make_workspace(
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_lr_end=actor_lr,
            critic_lr_end=critic_lr,
            transition_steps=transition_steps,
        )


class LatentXQCWorkspace:
    """Optimizers and local counters around one latent XQC controller."""

    def __init__(
        self,
        controller,
        *,
        actor_lr,
        critic_lr,
        actor_lr_end,
        critic_lr_end,
        transition_steps,
        optimizer_backend,
    ):
        self.controller = controller
        self.actor_lr = _positive_float(actor_lr, "actor_lr")
        self.critic_lr = _positive_float(critic_lr, "critic_lr")
        self.actor_lr_end = _positive_float(actor_lr_end, "actor_lr_end")
        self.critic_lr_end = _positive_float(critic_lr_end, "critic_lr_end")
        self.transition_steps = _positive_int(transition_steps, "transition_steps")
        device = next(controller.parameters()).device
        execution = _optimizer_execution_kwargs(device, str(optimizer_backend).lower())
        self.actor_optimizer = torch.optim.AdamW(
            controller.actor.parameters(),
            lr=self.actor_lr,
            betas=(0.9, 0.999),
            eps=controller.config.adam_eps,
            weight_decay=0.0,
            **execution,
        )
        self.critic_optimizer = torch.optim.AdamW(
            controller.critic.parameters(),
            lr=self.critic_lr,
            betas=(0.9, 0.999),
            eps=controller.config.adam_eps,
            weight_decay=0.0,
            **execution,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [controller.log_temperature],
            lr=self.actor_lr,
            betas=(0.9, 0.999),
            eps=controller.config.adam_eps,
            **execution,
        )
        self.update_step = 0
        self.actor_optimizer_steps = 0
        self.temperature_optimizer_steps = 0

    def reset_from_(self, source: LatentXQCController):
        self.controller.reset_prior_from_(source)
        for optimizer in (
            self.actor_optimizer,
            self.critic_optimizer,
            self.temperature_optimizer,
        ):
            optimizer.state.clear()
        self.update_step = 0
        self.actor_optimizer_steps = 0
        self.temperature_optimizer_steps = 0
        return self

    def zero_critic_grad(self):
        self.critic_optimizer.zero_grad(set_to_none=True)

    def step_critic(self):
        lr = linear_learning_rate(
            self.critic_lr,
            self.critic_lr_end,
            self.update_step,
            self.transition_steps,
        )
        _set_optimizer_lr(self.critic_optimizer, lr)
        self.critic_optimizer.step()
        _project_unit_weights_(self.controller._critic_linear_weights)
        target_updated = (
            (self.update_step + 1)
            % self.controller.config.target_update_interval
            == 0
        )
        if target_updated:
            _polyak_update_parameter_lists_(
                self.controller._critic_parameters,
                self.controller._target_parameters,
                self.controller.config.tau,
            )
        return lr, target_updated

    def step_actor_and_temperature(
        self,
        actor_loss: torch.Tensor,
        entropy: torch.Tensor,
    ):
        accepted = self.update_step % self.controller.config.policy_delay == 0
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = _global_grad_norm(self.controller.actor.parameters())
        actor_lr = linear_learning_rate(
            self.actor_lr,
            self.actor_lr_end,
            self.actor_optimizer_steps,
            self.transition_steps,
        )
        if accepted:
            _set_optimizer_lr(self.actor_optimizer, actor_lr)
            self.actor_optimizer.step()
            self.actor_optimizer_steps += 1
        _project_unit_weights_(self.controller._actor_linear_weights)

        temperature = self.controller.temperature
        temperature_loss = temperature * (
            entropy.detach() - self.controller.target_entropy
        )
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        temperature_grad_norm = _global_grad_norm(
            [self.controller.log_temperature]
        )
        temperature_lr = linear_learning_rate(
            self.actor_lr,
            self.actor_lr_end,
            self.temperature_optimizer_steps,
            self.transition_steps,
        )
        if accepted:
            _set_optimizer_lr(self.temperature_optimizer, temperature_lr)
            self.temperature_optimizer.step()
            self.temperature_optimizer_steps += 1
        self.update_step += 1
        return {
            "actor_update_accepted": float(accepted),
            "actor_learning_rate": float(actor_lr),
            "temperature_learning_rate": float(temperature_lr),
            "actor_grad_norm": actor_grad_norm.detach(),
            "temperature_grad_norm": temperature_grad_norm.detach(),
            "temperature": temperature.detach(),
            "temperature_loss": temperature_loss.detach(),
        }

    def update(
        self,
        batch: LatentXQCBatch,
        *,
        next_noise: torch.Tensor,
        actor_noise: torch.Tensor,
        reward_scale=1.0,
    ) -> dict[str, Any]:
        objective = self.controller.critic_objective(
            batch, next_noise=next_noise, reward_scale=reward_scale
        )
        self.zero_critic_grad()
        objective.loss.backward()
        critic_grad_norm = _global_grad_norm(self.controller.critic.parameters())
        critic_lr, target_updated = self.step_critic()

        actor = self.controller.actor_objective(
            batch.latents.detach(), actor_noise=actor_noise
        )
        step_info = self.step_actor_and_temperature(
            actor.loss, actor.entropy.mean()
        )
        values = objective.current_values.detach()
        metrics = {
            "critic_loss": objective.loss.detach(),
            "actor_loss": actor.loss.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "critic_learning_rate": float(critic_lr),
            "target_updated": float(target_updated),
            "policy_entropy": actor.entropy.detach().mean(),
            "policy_log_prob": actor.log_prob.detach().mean(),
            "q1_mean": values[0].mean(),
            "q2_mean": values[1].mean(),
            "q_target_mean": objective.target_values.detach().mean(),
            "q_policy_mean": actor.minimum_q.detach().mean(),
            "q_disagreement_mean": (values[0] - values[1]).abs().mean(),
            "q_target_clip_fraction": objective.clip_fraction.detach(),
        }
        metrics.update(step_info)
        return metrics

    def state_dict(self):
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "update_step": self.update_step,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "temperature_optimizer_steps": self.temperature_optimizer_steps,
        }

    def restore_learning_rate_phase_(self):
        """Restore the rates that were active after the recorded last steps."""

        critic_index = max(0, self.update_step - 1)
        actor_index = max(0, self.actor_optimizer_steps - 1)
        temperature_index = max(0, self.temperature_optimizer_steps - 1)
        _set_optimizer_lr(
            self.critic_optimizer,
            linear_learning_rate(
                self.critic_lr,
                self.critic_lr_end,
                critic_index,
                self.transition_steps,
            ),
        )
        _set_optimizer_lr(
            self.actor_optimizer,
            linear_learning_rate(
                self.actor_lr,
                self.actor_lr_end,
                actor_index,
                self.transition_steps,
            ),
        )
        _set_optimizer_lr(
            self.temperature_optimizer,
            linear_learning_rate(
                self.actor_lr,
                self.actor_lr_end,
                temperature_index,
                self.transition_steps,
            ),
        )
        return self

    def load_state_dict(self, state):
        expected = {
            "actor_optimizer",
            "critic_optimizer",
            "temperature_optimizer",
            "update_step",
            "actor_optimizer_steps",
            "temperature_optimizer_steps",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("Latent XQC workspace state has incompatible fields.")
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        for key in (
            "update_step",
            "actor_optimizer_steps",
            "temperature_optimizer_steps",
        ):
            value = state[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Latent XQC workspace {key} is invalid.")
            setattr(self, key, value)
        self.restore_learning_rate_phase_()
        return self
