"""Independent reward-return evaluation of AMBI-XQC's primary actor.

The learner owns only two categorical critics, their target copies, an
optimizer, and a private sampling stream. The existing XQC actor supplies
bootstrap actions read-only; no actor, temperature, or reward normalizer is
duplicated here.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from RL.xqc_core import (
    XQCTwinCritic,
    _MutableCompileRegion,
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
from .common.training_state import (
    load_optimizer_state_preserving_hyperparameters,
    preflight_optimizer_state,
    require_exact_keys,
    require_tensor,
)
from .xqc_controller import LatentXQCBatch, LatentXQCCriticObjective, LatentXQCController


class XQCAuxiliaryReturnLearner(nn.Module):
    """Reward-only twin C51 evaluator with the primary XQC numerical rules."""

    _STATE_SCHEMA = "ambi-xqc-auxiliary-return-training-state"
    _STATE_VERSION = 1
    _SEED_OFFSET = 13_036_363

    def __init__(self, primary_controller: LatentXQCController, cfg, device):
        super().__init__()
        if not isinstance(primary_controller, LatentXQCController):
            raise TypeError("Auxiliary returns require a latent XQC controller.")
        self.config = copy.deepcopy(primary_controller.config)
        self.latent_dim = primary_controller.latent_dim
        self.action_dim = primary_controller.action_dim
        self.critic_lr = float(cfg.xqc_critic_lr)
        self.critic_lr_end = float(cfg.xqc_lr_end)
        self.transition_steps = int(cfg.xqc_lr_transition_steps)
        self._seed = int(cfg.seed) + self._SEED_OFFSET
        device = torch.device(device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Auxiliary XQC returns support CPU and CUDA only.")
        # Construction is CPU-only inside a scoped CPU RNG. Do not seed CUDA
        # globally or consume the primary learner's initialization stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self._seed)
            self.critic = XQCTwinCritic(
                self.latent_dim,
                self.action_dim,
                self.config.critic_net_arch,
                self.config.num_atoms,
                vmin=self.config.vmin,
                vmax=self.config.vmax,
            )
            project_unit_rows_(self.critic)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)
        self.critic.to(device)
        self.critic_target.to(device)
        self._refresh_cached_tensors()
        execution = _optimizer_execution_kwargs(device, self.config.optimizer_backend)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_lr,
            betas=(0.9, 0.999),
            eps=self.config.adam_eps,
            weight_decay=0.0,
            **execution,
        )
        self.update_step = 0
        self._generator = torch.Generator(device=device).manual_seed(self._seed + 1)
        self.configure_compile(
            enabled=bool(getattr(cfg, "compile", False)),
            strict=bool(getattr(cfg, "compile_strict", False)),
        )

    @property
    def device(self):
        return next(self.critic.parameters()).device

    @property
    def critic_signature(self):
        return {
            "q_representation": "xqc_c51",
            "num_q": 2,
            "num_atoms": self.config.num_atoms,
            "vmin": self.config.vmin,
            "vmax": self.config.vmax,
        }

    def _refresh_cached_tensors(self):
        self._critic_linear_weights = tuple(
            module.weight for module in self.critic.modules() if isinstance(module, nn.Linear)
        )
        self._critic_parameters = tuple(self.critic.parameters())
        self._target_parameters = tuple(self.critic_target.parameters())

    def _apply(self, fn):
        configured = hasattr(self, "_critic_loss_region")
        result = super()._apply(fn)
        self._refresh_cached_tensors()
        if configured:
            self.configure_compile(self._compile_requested, self._compile_strict)
        return result

    def configure_compile(self, enabled, strict):
        if type(enabled) is not bool or type(strict) is not bool:
            raise TypeError("Auxiliary XQC compile flags must be booleans.")
        self._compile_requested = enabled
        self._compile_strict = strict
        self._critic_loss_region = _MutableCompileRegion(
            "AMBI-XQC auxiliary return critic loss",
            self._critic_loss_components,
            self.critic.buffers(),
            enabled=enabled and self.device.type == "cuda",
            strict=strict,
        )
        return self

    @property
    def compile_status(self):
        region = self._critic_loss_region
        return {
            "requested": self._compile_requested,
            "enabled": region.enabled,
            "strict": self._compile_strict,
            "critic_compiled": region._compiled is not None and not region.failed,
            "fallback": region.failed,
        }

    def critic_objective(self, batch: LatentXQCBatch, *, actor, reward_scale=1.0):
        flat = batch.flattened(self.latent_dim, self.action_dim)
        count = flat["latents"].shape[0]
        if torch.is_tensor(reward_scale):
            scale = reward_scale.to(flat["latents"])
            if scale.numel() != 1:
                raise ValueError("reward_scale must be one positive finite scalar.")
            valid = (torch.isfinite(scale) & (scale > 0)).reshape(())
            if scale.device.type == "cuda":
                torch._assert_async(valid, "reward_scale must be one positive finite scalar.")
            elif not bool(valid):
                raise ValueError("reward_scale must be one positive finite scalar.")
        else:
            value = float(reward_scale)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("reward_scale must be one positive finite scalar.")
            scale = flat["latents"].new_tensor(value)
        with torch.no_grad():
            noise = torch.randn(
                (count, self.action_dim),
                dtype=flat["latents"].dtype,
                device=flat["latents"].device,
                generator=self._generator,
            )
            next_actions, _ = actor.sample(
                flat["next_latents"], bn_mode="running", noise=noise
            )
        outputs = self._critic_loss_region(
            flat["latents"], flat["actions"], flat["rewards"],
            flat["next_latents"], flat["bootstrap_mask"], flat["discount"],
            next_actions, scale.reshape(()),
        )
        loss, per_sample, log_probs, targets, values, target_values, head, clipped = outputs
        leading = batch.leading_shape
        return LatentXQCCriticObjective(
            loss=loss,
            per_sample_loss=per_sample.reshape(leading),
            current_log_probs=log_probs.reshape((2,) + leading + (self.config.num_atoms,)),
            target_probabilities=targets.reshape(leading + (self.config.num_atoms,)),
            current_values=values.reshape((2,) + leading),
            target_values=target_values.reshape(leading),
            target_head=head.reshape(leading),
            clip_fraction=clipped,
        )

    def _critic_loss_components(
        self, latents, actions, rewards, next_latents, bootstrap_mask,
        discount, next_actions, reward_scale,
    ):
        count = latents.shape[0]
        joined_actions = torch.cat((actions, next_actions), dim=0)
        with torch.no_grad():
            target_latents = torch.cat((latents.detach(), next_latents.detach()), dim=0)
            target_joined = self.critic_target.log_probs(
                target_latents, joined_actions, bn_mode="batch_no_update"
            )
            selected, target_values, target_head = select_lower_distribution(
                target_joined[:, count:], self.critic_target.support
            )
            targets, clipped = categorical_td_projection(
                selected, rewards / reward_scale, bootstrap_mask, discount,
                torch.zeros_like(rewards), self.critic.support, validate_support=False,
            )
        joined_latents = torch.cat((latents, next_latents.detach()), dim=0)
        joined_log_probs = self.critic.log_probs(
            joined_latents, joined_actions, bn_mode="batch_update"
        )
        current_log_probs = joined_log_probs[:, :count]
        per_sample = -(targets.unsqueeze(0) * current_log_probs).sum(-1).sum(0)
        current_values = self.critic.values_from_log_probs(current_log_probs)
        return (
            per_sample.mean(), per_sample, current_log_probs, targets,
            current_values, target_values, target_head, clipped,
        )

    def zero_grad(self, set_to_none=True):
        self.critic_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        grad_norm = _global_grad_norm(self.critic.parameters()).detach()
        lr = linear_learning_rate(
            self.critic_lr, self.critic_lr_end, self.update_step, self.transition_steps
        )
        _set_optimizer_lr(self.critic_optimizer, lr)
        self.critic_optimizer.step()
        _project_unit_weights_(self._critic_linear_weights)
        target_updated = (self.update_step + 1) % self.config.target_update_interval == 0
        if target_updated:
            _polyak_update_parameter_lists_(
                self._critic_parameters, self._target_parameters, self.config.tau
            )
        self.update_step += 1
        return {
            "aux_return_grad_norm": grad_norm,
            "aux_return_learning_rate": float(lr),
            "aux_return_target_updated": float(target_updated),
            "aux_return_num_updates": float(self.update_step),
        }

    def training_state_dict(self):
        return {
            "schema": self._STATE_SCHEMA,
            "version": self._STATE_VERSION,
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
            "device_type": self._generator.device.type,
            "generator": self._generator.get_state(),
        }

    def preflight_training_state(self, state, expected_updates, *, frozen_evaluation=False):
        state = require_exact_keys(
            state,
            {"schema", "version", "critic_optimizer", "update_step", "device_type", "generator"},
            "AMBI-XQC auxiliary return state",
        )
        if (state["schema"] != self._STATE_SCHEMA or type(state["version"]) is not int
                or state["version"] != self._STATE_VERSION):
            raise ValueError("Unsupported AMBI-XQC auxiliary return state version.")
        if (type(expected_updates) is not int or expected_updates < 0
                or type(state["update_step"]) is not int
                or state["update_step"] != expected_updates):
            raise ValueError("Auxiliary return update counter must match outer updates.")
        optimizer = preflight_optimizer_state(
            self.critic_optimizer, state["critic_optimizer"], "Auxiliary return optimizer"
        )
        inventory = optimizer["state"]
        parameter_count = sum(len(group["params"]) for group in self.critic_optimizer.param_groups)
        if (expected_updates == 0 and inventory) or (expected_updates > 0 and len(inventory) != parameter_count):
            raise ValueError("Auxiliary return optimizer state inventory is incomplete.")
        for values in inventory.values():
            step = values["step"]
            if float(step.item() if torch.is_tensor(step) else step) != expected_updates:
                raise ValueError("Auxiliary return optimizer step does not match its counter.")
            if any(torch.is_tensor(value) and not bool(torch.isfinite(value).all()) for value in values.values()):
                raise ValueError("Auxiliary return optimizer state must be finite.")
        device_type = state["device_type"]
        if device_type not in {"cpu", "cuda"}:
            raise ValueError("Auxiliary return RNG device type is invalid.")
        generator = require_tensor(state["generator"], "Auxiliary return RNG", dtype=torch.uint8)
        if generator.ndim != 1 or generator.layout != torch.strided or not generator.is_contiguous():
            raise ValueError("Auxiliary return RNG must be a contiguous one-dimensional tensor.")
        if device_type == "cuda" and not torch.cuda.is_available():
            if generator.numel() != 16:
                raise ValueError("Auxiliary return CUDA RNG state is invalid.")
            offset = int.from_bytes(bytes(generator.detach().cpu().tolist())[8:], "little")
            if offset % 4:
                raise ValueError("Auxiliary return CUDA RNG offset must be divisible by four.")
        else:
            probe = torch.Generator(device=device_type)
            try:
                probe.set_state(generator.detach().cpu())
            except RuntimeError as exc:
                raise ValueError("Auxiliary return RNG state is invalid.") from exc
        cross_device = device_type != self.device.type
        if cross_device and not frozen_evaluation:
            raise ValueError("Auxiliary return RNG device type requires frozen evaluation for transfer.")
        candidate = dict(state)
        if cross_device:
            candidate["device_type"] = self.device.type
            candidate["generator"] = self._generator.get_state()
        return candidate

    def load_training_state(self, state, *, frozen_evaluation=False):
        state = self.preflight_training_state(
            state, state["update_step"], frozen_evaluation=frozen_evaluation
        )
        load_optimizer_state_preserving_hyperparameters(
            self.critic_optimizer, state["critic_optimizer"]
        )
        self.update_step = state["update_step"]
        self._generator.set_state(state["generator"].detach().cpu())
        _set_optimizer_lr(
            self.critic_optimizer,
            linear_learning_rate(
                self.critic_lr, self.critic_lr_end,
                max(0, self.update_step - 1), self.transition_steps,
            ),
        )
        return self
