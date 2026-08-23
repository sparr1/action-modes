"""Faithful, state-based PyTorch port of the official XQC learner.

The reference implementation is the XQC repository at commit
``9a6832bb742ef01bbe9f1e06153a9338e612dae5``.  This module deliberately
preserves the released implementation's less-obvious behavior: target critic
batch statistics are computed but never stored, policy and temperature Adam
states advance only on delayed updates, and the temperature optimizer uses the
actor learning-rate schedule.

XQC is Copyright (c) 2026 Daniel Palenicek and was released under the MIT
license. This is an independent PyTorch port rather than copied JAX source.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from RL.tdmpc2_core.common.training_state import (
    load_optimizer_state_preserving_hyperparameters,
    preflight_module_state,
    preflight_optimizer_state,
    require_exact_keys,
    require_mapping,
    require_tensor,
)


OFFICIAL_XQC_COMMIT = "9a6832bb742ef01bbe9f1e06153a9338e612dae5"
LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0
BN_MOMENTUM = 0.99
BN_EPSILON = 0.001
LOG_2 = math.log(2.0)
LOG_2PI = math.log(2.0 * math.pi)
_BN_MODES = {"running", "batch_update", "batch_no_update"}


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve an Action Modes device string, falling back cleanly to CPU."""

    requested = str(device)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested for XQC, but CUDA is unavailable. Falling back to CPU.")
        requested = "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    return torch.device(requested)


def _finite_float(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not a boolean.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _positive_int(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (Integral, Real))
        or not np.isfinite(value)
        or int(value) != value
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _nonnegative_int(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (Integral, Real))
        or not np.isfinite(value)
        or int(value) != value
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


@dataclass
class XQCConfig:
    """The released XQC defaults, expressed in Action Modes terminology."""

    learning_rate: float = 3e-4
    actor_lr: Optional[float] = None
    critic_lr: Optional[float] = None
    lr_end: float = 3e-5
    num_interactions: int = 500_000
    updates_per_step: int = 2
    buffer_size: int = 1_000_000
    learning_starts: int = 5_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 2
    target_update_interval: int = 1
    policy_delay: int = 3
    actor_net_arch: Tuple[int, ...] = (256, 256, 256, 256)
    critic_net_arch: Tuple[int, ...] = (512, 512, 512, 512)
    num_atoms: int = 101
    vmin: float = -5.0
    vmax: float = 5.0
    init_temperature: float = 0.01
    target_entropy: str | float = "auto"
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    reward_normalization: bool = True
    debug_checks: bool = False
    compile: bool = False
    compile_strict: bool = False
    optimizer_backend: str = "auto"
    seed: Optional[int] = None
    device: str = "auto"
    verbose: int = 1

    def __post_init__(self) -> None:
        self.learning_rate = _finite_float(
            self.learning_rate, "learning_rate", positive=True
        )
        if self.actor_lr is None:
            self.actor_lr = self.learning_rate
        if self.critic_lr is None:
            self.critic_lr = self.learning_rate
        self.actor_lr = _finite_float(self.actor_lr, "actor_lr", positive=True)
        self.critic_lr = _finite_float(self.critic_lr, "critic_lr", positive=True)
        self.lr_end = _finite_float(self.lr_end, "lr_end", positive=True)
        self.tau = _finite_float(self.tau, "tau", positive=True)
        self.gamma = _finite_float(self.gamma, "gamma", positive=True)
        self.init_temperature = _finite_float(
            self.init_temperature, "init_temperature", positive=True
        )
        self.adam_eps = _finite_float(self.adam_eps, "adam_eps", positive=True)
        self.weight_decay = _finite_float(self.weight_decay, "weight_decay")
        self.vmin = _finite_float(self.vmin, "vmin")
        self.vmax = _finite_float(self.vmax, "vmax")

        if self.tau > 1.0:
            raise ValueError("tau must be in (0, 1].")
        if self.gamma > 1.0:
            raise ValueError("gamma must be in (0, 1].")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.vmin >= self.vmax:
            raise ValueError("vmin must be smaller than vmax.")

        for field_name in (
            "num_interactions",
            "updates_per_step",
            "buffer_size",
            "batch_size",
            "train_freq",
            "target_update_interval",
            "policy_delay",
            "num_atoms",
        ):
            setattr(self, field_name, _positive_int(getattr(self, field_name), field_name))
        self.learning_starts = _nonnegative_int(
            self.learning_starts, "learning_starts"
        )
        if (
            isinstance(self.gradient_steps, (bool, np.bool_))
            or not isinstance(self.gradient_steps, (Integral, Real))
            or not np.isfinite(self.gradient_steps)
            or int(self.gradient_steps) != self.gradient_steps
            or int(self.gradient_steps) == 0
            or int(self.gradient_steps) < -1
        ):
            raise ValueError("gradient_steps must be -1 or a positive integer.")
        self.gradient_steps = int(self.gradient_steps)
        if self.num_atoms < 2:
            raise ValueError("num_atoms must be at least two.")

        self.actor_net_arch = self._validate_architecture(
            self.actor_net_arch, "actor_net_arch"
        )
        self.critic_net_arch = self._validate_architecture(
            self.critic_net_arch, "critic_net_arch"
        )
        if not isinstance(self.reward_normalization, (bool, np.bool_)):
            raise ValueError("reward_normalization must be a boolean.")
        self.reward_normalization = bool(self.reward_normalization)
        if not isinstance(self.debug_checks, (bool, np.bool_)):
            raise ValueError("debug_checks must be a boolean.")
        self.debug_checks = bool(self.debug_checks)
        for field_name in ("compile", "compile_strict"):
            if not isinstance(getattr(self, field_name), (bool, np.bool_)):
                raise ValueError(f"{field_name} must be a boolean.")
            setattr(self, field_name, bool(getattr(self, field_name)))
        self.optimizer_backend = str(self.optimizer_backend).lower()
        if self.optimizer_backend not in {
            "auto",
            "single_tensor",
            "foreach",
            "fused",
        }:
            raise ValueError(
                "optimizer_backend must be auto, single_tensor, foreach, or fused."
            )
        if self.seed is not None:
            self.seed = _nonnegative_int(self.seed, "seed")
        self.verbose = int(self.verbose)

        if self.target_entropy != "auto":
            self.target_entropy = _finite_float(
                self.target_entropy, "target_entropy"
            )

    @staticmethod
    def _validate_architecture(value, name: str) -> Tuple[int, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be a sequence of positive widths.")
        try:
            widths = tuple(_positive_int(width, name) for width in value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a sequence of positive widths.") from exc
        if not widths:
            raise ValueError(f"{name} must contain at least one hidden layer.")
        return widths

    @property
    def alpha_lr(self) -> float:
        """The release ignores ``temp_lr`` and uses the actor schedule."""

        return float(self.actor_lr)

    @property
    def transition_steps(self) -> int:
        return int(self.num_interactions * self.updates_per_step)


class FlaxBatchNorm1d(nn.Module):
    """BatchNorm with the three state behaviors needed by released XQC.

    ``momentum`` follows Flax notation: 0.99 retains 99% of the old running
    value.  PyTorch's equivalent constructor value would therefore be 0.01.
    The variance deliberately uses Flax's fast population formula E[x^2]-E[x]^2.
    """

    def __init__(
        self,
        num_features: int,
        *,
        momentum: float = BN_MOMENTUM,
        eps: float = BN_EPSILON,
    ) -> None:
        super().__init__()
        self.num_features = _positive_int(num_features, "num_features")
        self.momentum = _finite_float(momentum, "momentum")
        self.eps = _finite_float(eps, "eps", positive=True)
        if not 0.0 <= self.momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1].")
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = nn.Parameter(torch.zeros(self.num_features))
        self.register_buffer("running_mean", torch.zeros(self.num_features))
        self.register_buffer("running_var", torch.ones(self.num_features))

    def _batch_statistics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean = stats_x.mean(dim=0)
        variance = torch.clamp((stats_x * stats_x).mean(dim=0) - mean.square(), min=0.0)
        return mean, variance

    def forward(self, x: torch.Tensor, mode: str = "running") -> torch.Tensor:
        if mode not in _BN_MODES:
            raise ValueError(
                f"BatchNorm mode must be one of {sorted(_BN_MODES)}, got {mode!r}."
            )
        if x.ndim != 2 or x.shape[-1] != self.num_features:
            raise ValueError(
                "FlaxBatchNorm1d expects [batch, features] input with "
                f"{self.num_features} features, got {tuple(x.shape)}."
            )

        if mode == "running":
            mean = self.running_mean
            variance = self.running_var
        else:
            mean, variance = self._batch_statistics(x)
            if mode == "batch_update":
                with torch.no_grad():
                    retain = self.momentum
                    update = 1.0 - retain
                    self.running_mean.mul_(retain).add_(mean.detach(), alpha=update)
                    self.running_var.mul_(retain).add_(variance.detach(), alpha=update)

        normalized = (x - mean.to(dtype=x.dtype)) * torch.rsqrt(
            variance.to(dtype=x.dtype) + self.eps
        )
        return normalized * self.weight.to(dtype=x.dtype) + self.bias.to(dtype=x.dtype)


class XQCBlock(nn.Module):
    """The released XQC Dense(no bias) -> BatchNorm -> ReLU block."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.batch_norm = FlaxBatchNorm1d(output_dim)
        nn.init.orthogonal_(self.linear.weight, gain=math.sqrt(2.0))

    def forward(self, x: torch.Tensor, bn_mode: str = "running") -> torch.Tensor:
        return F.relu(self.batch_norm(self.linear(x), mode=bn_mode))


class XQCActor(nn.Module):
    """Feature-vector XQC policy with a tanh-transformed diagonal Gaussian."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256, 256),
        *,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        super().__init__()
        self.obs_dim = _positive_int(obs_dim, "obs_dim")
        self.action_dim = _positive_int(action_dim, "action_dim")
        self.log_std_min = _finite_float(log_std_min, "log_std_min")
        self.log_std_max = _finite_float(log_std_max, "log_std_max")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max.")
        hidden_dims = XQCConfig._validate_architecture(hidden_dims, "hidden_dims")

        self.input_batch_norm = FlaxBatchNorm1d(self.obs_dim)
        blocks = []
        input_dim = self.obs_dim
        for hidden_dim in hidden_dims:
            blocks.append(XQCBlock(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.mean = nn.Linear(input_dim, self.action_dim, bias=True)
        self.log_std = nn.Linear(input_dim, self.action_dim, bias=True)
        nn.init.orthogonal_(self.mean.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.mean.bias)
        nn.init.orthogonal_(self.log_std.weight, gain=1.0)
        nn.init.zeros_(self.log_std.bias)

    def distribution(
        self, obs: torch.Tensor, bn_mode: str = "running"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_batch_norm(obs, mode=bn_mode)
        for block in self.blocks:
            x = block(x, bn_mode=bn_mode)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    @staticmethod
    def _log_prob_from_pre_tanh(
        pre_tanh: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        standardized = (pre_tanh - mean) * torch.exp(-log_std)
        gaussian_log_prob = -0.5 * standardized.square() - log_std - 0.5 * LOG_2PI
        log_abs_det_jacobian = 2.0 * (
            LOG_2 - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        return (gaussian_log_prob - log_abs_det_jacobian).sum(dim=-1)

    def sample(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        bn_mode: str = "running",
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temperature = _finite_float(temperature, "temperature")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        mean, log_std = self.distribution(obs, bn_mode=bn_mode)
        if deterministic or temperature == 0.0:
            pre_tanh = mean
        else:
            if noise is None:
                noise_device = mean.device if generator is None else generator.device
                noise = torch.randn(
                    mean.shape,
                    dtype=mean.dtype,
                    device=noise_device,
                    generator=generator,
                ).to(mean.device)
            else:
                noise = torch.as_tensor(noise, dtype=mean.dtype, device=mean.device)
                if noise.shape != mean.shape:
                    raise ValueError(
                        f"noise must have shape {tuple(mean.shape)}, got {tuple(noise.shape)}."
                    )
            pre_tanh = mean + torch.exp(log_std) * temperature * noise
        action = torch.tanh(pre_tanh)
        effective_log_std = log_std + math.log(temperature) if temperature > 0 else log_std
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, effective_log_std)
        return action, log_prob

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        *,
        bn_mode: str = "running",
    ) -> torch.Tensor:
        return self.sample(
            obs, deterministic=deterministic, bn_mode=bn_mode
        )[0]


class XQCCriticHead(nn.Module):
    """One independent categorical Q head, including independent BN state."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 512, 512),
        num_atoms: int = 101,
    ) -> None:
        super().__init__()
        obs_dim = _positive_int(obs_dim, "obs_dim")
        action_dim = _positive_int(action_dim, "action_dim")
        num_atoms = _positive_int(num_atoms, "num_atoms")
        hidden_dims = XQCConfig._validate_architecture(hidden_dims, "hidden_dims")
        input_dim = obs_dim + action_dim
        self.input_batch_norm = FlaxBatchNorm1d(input_dim)
        blocks = []
        for hidden_dim in hidden_dims:
            blocks.append(XQCBlock(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.value = nn.Linear(input_dim, num_atoms, bias=True)
        nn.init.orthogonal_(self.value.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.value.bias)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        bn_mode: str = "running",
    ) -> torch.Tensor:
        if obs.ndim != 2 or actions.ndim != 2 or obs.shape[0] != actions.shape[0]:
            raise ValueError("critic observations/actions must be aligned rank-two batches.")
        x = torch.cat((obs, actions), dim=-1)
        x = self.input_batch_norm(x, mode=bn_mode)
        for block in self.blocks:
            x = block(x, bn_mode=bn_mode)
        return self.value(x)


class XQCTwinCritic(nn.Module):
    """Two fully independent categorical critics."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 512, 512, 512),
        num_atoms: int = 101,
        *,
        vmin: float = -5.0,
        vmax: float = 5.0,
    ) -> None:
        super().__init__()
        if float(vmin) >= float(vmax):
            raise ValueError("vmin must be smaller than vmax.")
        self.q1 = XQCCriticHead(obs_dim, action_dim, hidden_dims, num_atoms)
        self.q2 = XQCCriticHead(obs_dim, action_dim, hidden_dims, num_atoms)
        self.register_buffer(
            "support", torch.linspace(float(vmin), float(vmax), int(num_atoms))
        )

    @property
    def q_networks(self) -> Tuple[XQCCriticHead, XQCCriticHead]:
        return self.q1, self.q2

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        bn_mode: str = "running",
    ) -> torch.Tensor:
        return torch.stack(
            tuple(head(obs, actions, bn_mode=bn_mode) for head in self.q_networks),
            dim=0,
        )

    def log_probs(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        bn_mode: str = "running",
    ) -> torch.Tensor:
        return F.log_softmax(self(obs, actions, bn_mode=bn_mode), dim=-1)

    def values_from_log_probs(self, log_probs: torch.Tensor) -> torch.Tensor:
        return (log_probs.exp() * self.support).sum(dim=-1)

    def values(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        bn_mode: str = "running",
    ) -> torch.Tensor:
        return self.values_from_log_probs(
            self.log_probs(obs, actions, bn_mode=bn_mode)
        )


def select_lower_distribution(
    log_probs: torch.Tensor, support: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the complete distribution of the lower-expectation twin head.

    Ties select head zero, matching ``jnp.argmin`` in the official code.
    Returns the selected log probabilities, selected scalar values, and indices.
    """

    if log_probs.ndim != 3 or log_probs.shape[0] != 2:
        raise ValueError("log_probs must have shape [2, batch, atoms].")
    support = torch.as_tensor(support, dtype=log_probs.dtype, device=log_probs.device)
    if support.ndim != 1 or support.shape[0] != log_probs.shape[-1]:
        raise ValueError("support must be one-dimensional and match the atom count.")
    values = (log_probs.exp() * support).sum(dim=-1)
    indices = values.argmin(dim=0)
    batch_indices = torch.arange(log_probs.shape[1], device=log_probs.device)
    selected = log_probs[indices, batch_indices]
    selected_values = values[indices, batch_indices]
    return selected, selected_values, indices


def categorical_td_projection(
    target_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    masks: torch.Tensor,
    discount: float | torch.Tensor,
    actor_entropy: torch.Tensor,
    support: torch.Tensor,
    *,
    validate_support: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project the released XQC categorical Bellman target onto ``support``."""

    if target_log_probs.ndim != 2:
        raise ValueError("target_log_probs must have shape [batch, atoms].")
    batch_size, num_atoms = target_log_probs.shape
    support = torch.as_tensor(
        support, dtype=target_log_probs.dtype, device=target_log_probs.device
    )
    if support.ndim != 1 or support.shape[0] != num_atoms:
        raise ValueError("support must be one-dimensional and match the atom count.")
    if num_atoms < 2:
        raise ValueError("support must contain at least two strictly increasing atoms.")
    # Match the reference expression ``(max_v - min_v) / (num_bins - 1)``.
    # Using the first torch.linspace difference accumulates enough float32
    # rounding error for the top atom to ceil to ``num_atoms`` at 101 atoms.
    spacing = (support[-1] - support[0]) / (num_atoms - 1)
    if validate_support:
        if not bool(torch.all(support[1:] > support[:-1]).item()):
            raise ValueError(
                "support must contain at least two strictly increasing atoms."
            )
        if not bool(
            torch.allclose(
                support[1:] - support[:-1],
                spacing.expand_as(support[1:]),
                rtol=1e-5,
                atol=1e-7,
            )
        ):
            raise ValueError("XQC categorical support must be evenly spaced.")

    def batch_column(value, name: str) -> torch.Tensor:
        value = torch.as_tensor(
            value, dtype=target_log_probs.dtype, device=target_log_probs.device
        )
        if value.numel() == 1:
            return value.reshape(1, 1).expand(batch_size, 1)
        value = value.reshape(batch_size, -1)
        if value.shape[1] != 1:
            raise ValueError(f"{name} must contain one scalar per batch element.")
        return value

    rewards = batch_column(rewards, "rewards")
    masks = batch_column(masks, "masks")
    discount = batch_column(discount, "discount")
    actor_entropy = batch_column(actor_entropy, "actor_entropy")
    transformed = rewards + discount * masks * (support.reshape(1, -1) - actor_entropy)
    # Tensor bounds avoid a device-to-host synchronization on CUDA. The public
    # function validates arbitrary supports by default; the learner can skip
    # that repeated validation because its registered support is immutable.
    transformed = torch.maximum(
        torch.minimum(transformed, support[-1]), support[0]
    )
    clip_fraction = (
        (transformed == support[0]) | (transformed == support[-1])
    ).to(target_log_probs.dtype).mean()

    positions = ((transformed - support[0]) / spacing).clamp(0.0, num_atoms - 1)
    lower = torch.floor(positions).to(torch.long)
    upper = torch.ceil(positions).to(torch.long)
    lower_weight = upper.to(positions.dtype) + (lower == upper).to(positions.dtype) - positions
    upper_weight = positions - lower.to(positions.dtype)
    target_probabilities = target_log_probs.exp()
    projected = torch.zeros_like(target_probabilities)
    projected.scatter_add_(1, lower, target_probabilities * lower_weight)
    projected.scatter_add_(1, upper, target_probabilities * upper_weight)
    return projected.detach(), clip_fraction.detach()


def categorical_cross_entropy(
    pred_log_probs: torch.Tensor, target_probs: torch.Tensor
) -> torch.Tensor:
    """Mean categorical CE, summing heads when a twin-head axis is present."""

    if pred_log_probs.ndim == 2:
        if target_probs.shape != pred_log_probs.shape:
            raise ValueError("target_probs must match pred_log_probs.")
        return -(target_probs * pred_log_probs).sum(dim=-1).mean()
    if pred_log_probs.ndim != 3 or pred_log_probs.shape[0] != 2:
        raise ValueError(
            "pred_log_probs must have shape [batch, atoms] or [2, batch, atoms]."
        )
    if target_probs.ndim == 2:
        target_probs = target_probs.unsqueeze(0).expand_as(pred_log_probs)
    if target_probs.shape != pred_log_probs.shape:
        raise ValueError("target_probs must match pred_log_probs or omit the head axis.")
    return -(target_probs * pred_log_probs).sum(dim=-1).mean(dim=-1).sum()


@torch.no_grad()
def project_unit_rows_(module: nn.Module) -> nn.Module:
    """Apply Flax XQC's column projection to PyTorch Linear weight rows.

    Final-layer biases are intentionally untouched.  XQC hidden layers have no
    bias, so all learned Dense kernels are normalized independently per output.
    """

    weights = tuple(
        child.weight for child in module.modules() if isinstance(child, nn.Linear)
    )
    _project_unit_weights_(weights)
    return module


@torch.no_grad()
def _project_unit_weights_(weights: Iterable[torch.Tensor]) -> None:
    """Project a cached collection of linear weights with one foreach divide."""

    weights = tuple(weights)
    if not weights:
        return
    row_norms = [
        torch.linalg.vector_norm(weight, dim=1, keepdim=True)
        for weight in weights
    ]
    torch._foreach_div_(weights, row_norms)


def polyak_update_parameters(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak-average parameters only; target BN running buffers stay frozen."""

    with torch.no_grad():
        source_parameters = tuple(source.named_parameters())
        target_parameters = tuple(target.named_parameters())
        if tuple(name for name, _ in source_parameters) != tuple(
            name for name, _ in target_parameters
        ):
            raise ValueError("source and target parameter layouts do not match.")
        _polyak_update_parameter_lists_(
            tuple(parameter for _, parameter in source_parameters),
            tuple(parameter for _, parameter in target_parameters),
            tau,
        )


@torch.no_grad()
def _polyak_update_parameter_lists_(
    source_parameters: Iterable[torch.Tensor],
    target_parameters: Iterable[torch.Tensor],
    tau: float,
) -> None:
    """Update cached parameter lists in a single multi-tensor operation."""

    source_parameters = tuple(source_parameters)
    target_parameters = tuple(target_parameters)
    if len(source_parameters) != len(target_parameters):
        raise ValueError("source and target parameter layouts do not match.")
    # Retain the official/source expression order p*tau + tp*(1-tau), while
    # reducing it to two multi-tensor launches.
    tau = float(tau)
    torch._foreach_mul_(target_parameters, 1.0 - tau)
    torch._foreach_add_(target_parameters, source_parameters, alpha=tau)


class DiscountedReturnNormalizer:
    """One-stream port of the official discounted-return reward normalizer."""

    def __init__(self, gamma: float, epsilon: float = 1e-8) -> None:
        self.gamma = _finite_float(gamma, "gamma", positive=True)
        if self.gamma > 1.0:
            raise ValueError("gamma must be in (0, 1].")
        self.epsilon = _finite_float(epsilon, "epsilon", positive=True)
        self.return_accumulator = 0.0
        self.mean = 0.0
        self.mean_squared = 0.0
        self.var = 1.0
        self.count = 0.0
        self.maximum = -math.inf
        self.minimum = math.inf

    def update(self, reward, done: bool) -> None:
        reward_array = np.asarray(reward, dtype=np.float64)
        if reward_array.size != 1 or not np.isfinite(reward_array).all():
            raise ValueError("reward normalizer expects one finite reward scalar.")
        reward_value = float(reward_array.reshape(-1)[0])
        self.return_accumulator = (
            self.gamma * (1.0 - float(bool(done))) * self.return_accumulator
            + reward_value
        )

        batch_mean = self.return_accumulator
        batch_mean_squared = self.return_accumulator * self.return_accumulator
        total_count = self.count + 1.0
        delta = batch_mean - self.mean
        delta_squared = batch_mean_squared - self.mean_squared
        old_m2 = self.var * self.count
        new_m2 = old_m2 + delta * delta * self.count / total_count
        self.mean += delta / total_count
        self.mean_squared += delta_squared / total_count
        self.var = new_m2 / total_count
        self.count = total_count
        absolute_return = abs(self.return_accumulator)
        self.maximum = max(self.maximum, absolute_return)
        self.minimum = min(self.minimum, absolute_return)

    @property
    def scale(self) -> float:
        return math.sqrt(self.var) + self.epsilon

    def normalize(self, rewards):
        if torch.is_tensor(rewards):
            return rewards / rewards.new_tensor(self.scale)
        return np.asarray(rewards) / self.scale

    def state_dict(self) -> Dict[str, float]:
        return {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "return_accumulator": self.return_accumulator,
            "mean": self.mean,
            "mean_squared": self.mean_squared,
            "var": self.var,
            "count": self.count,
            "maximum": self.maximum,
            "minimum": self.minimum,
        }

    def _validated_state(self, state) -> Dict[str, float]:
        expected_keys = {
            "gamma",
            "epsilon",
            "return_accumulator",
            "mean",
            "mean_squared",
            "var",
            "count",
            "maximum",
            "minimum",
        }
        state = require_exact_keys(state, expected_keys, "XQC reward normalizer")
        if float(state["gamma"]) != self.gamma or float(state["epsilon"]) != self.epsilon:
            raise ValueError("XQC reward-normalizer configuration does not match.")
        validated = {key: float(value) for key, value in state.items()}
        for key in (
            "gamma",
            "epsilon",
            "return_accumulator",
            "mean",
            "mean_squared",
            "var",
            "count",
        ):
            if not np.isfinite(validated[key]):
                raise ValueError(f"XQC reward-normalizer {key} must be finite.")
        if validated["var"] < 0.0 or validated["count"] < 0.0:
            raise ValueError("XQC reward-normalizer variance/count cannot be negative.")
        if validated["count"] == 0.0:
            if not (
                validated["maximum"] == -math.inf
                and validated["minimum"] == math.inf
            ):
                raise ValueError("Empty XQC reward-normalizer extrema are invalid.")
        elif not (
            np.isfinite(validated["maximum"])
            and np.isfinite(validated["minimum"])
            and 0.0 <= validated["minimum"] <= validated["maximum"]
        ):
            raise ValueError("XQC reward-normalizer extrema are invalid.")
        return validated

    def load_state_dict(self, state) -> None:
        validated = self._validated_state(state)
        for key, value in validated.items():
            if key not in {"gamma", "epsilon"}:
                setattr(self, key, value)


def _global_grad_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(norms).norm(2)


def linear_learning_rate(
    start: float, end: float, step: int, transition_steps: int
) -> float:
    """Optax-compatible clipped linear schedule value for one inner step."""

    fraction = min(max(float(step) / float(transition_steps), 0.0), 1.0)
    return float(start + fraction * (end - start))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def _optimizer_execution_kwargs(
    device: torch.device, backend: str
) -> Dict[str, bool]:
    """Resolve a numerically equivalent optimizer implementation.

    The single-tensor path stays the CPU reference. CUDA defaults to the fused
    implementation, which removes dozens of per-parameter kernel launches. A
    foreach path remains available for installations where fused Adam is not
    desired, without changing optimizer state or checkpoint contents.
    """

    backend = "fused" if backend == "auto" and device.type == "cuda" else backend
    backend = "single_tensor" if backend == "auto" else backend
    if backend == "fused":
        if device.type != "cuda":
            raise ValueError("The XQC fused optimizer backend requires CUDA.")
        return {"fused": True}
    if backend == "foreach":
        return {"foreach": True}
    return {"foreach": False}


class _MutableCompileRegion:
    """Lazy fixed-shape compilation with atomic eager fallback for BN buffers."""

    def __init__(
        self,
        name: str,
        eager,
        mutable_buffers: Iterable[torch.Tensor],
        *,
        enabled: bool,
        strict: bool,
    ) -> None:
        self.name = str(name)
        self.eager = eager
        self.mutable_buffers = tuple(mutable_buffers)
        self.enabled = bool(enabled)
        self.strict = bool(strict)
        self.failed = False
        self._compiled = None
        self._warned = False
        if self.enabled and not hasattr(torch, "compile"):
            error = RuntimeError("torch.compile is unavailable in this PyTorch build.")
            if self.strict:
                raise error
            self._fallback(error)

    def _fallback(self, error: BaseException) -> None:
        self.enabled = False
        self.failed = True
        self._compiled = None
        if not self._warned:
            warnings.warn(
                f"Falling back to eager {self.name} after compile failure: {error}",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned = True

    def __call__(self, *args):
        if not self.enabled:
            return self.eager(*args)
        constructing = self._compiled is None
        snapshots = (
            tuple(buffer.detach().clone() for buffer in self.mutable_buffers)
            if constructing and not self.strict
            else None
        )
        if self._compiled is None:
            try:
                self._compiled = torch.compile(
                    self.eager,
                    # ``strict`` controls failure policy only. Keep the graph
                    # mode identical between validation and production so the
                    # gate exercises exactly what long runs execute.
                    fullgraph=False,
                    dynamic=False,
                )
            except Exception as error:
                if self.strict:
                    raise
                self._fallback(error)
                return self.eager(*args)
        try:
            return self._compiled(*args)
        except Exception as error:
            if snapshots is None:
                # A graph which previously executed successfully must never be
                # retried after partial state mutation.
                raise RuntimeError(f"Compiled {self.name} failed at runtime.") from error
            with torch.no_grad():
                torch._foreach_copy_(self.mutable_buffers, snapshots)
            if self.strict:
                raise
            self._fallback(error)
            return self.eager(*args)


class XQCAgent:
    """Standalone released-XQC learner over flat feature vectors."""

    _STATE_VERSION = 1

    def __init__(self, obs_dim: int, action_dim: int, config: XQCConfig) -> None:
        self.obs_dim = _positive_int(obs_dim, "obs_dim")
        self.action_dim = _positive_int(action_dim, "action_dim")
        if not isinstance(config, XQCConfig):
            raise TypeError("config must be an XQCConfig.")
        self.config = config
        self.device = resolve_device(config.device)

        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        self.actor = XQCActor(
            self.obs_dim,
            self.action_dim,
            config.actor_net_arch,
        ).to(self.device)
        self.critic = XQCTwinCritic(
            self.obs_dim,
            self.action_dim,
            config.critic_net_arch,
            config.num_atoms,
            vmin=config.vmin,
            vmax=config.vmax,
        ).to(self.device)
        self.critic_target = XQCTwinCritic(
            self.obs_dim,
            self.action_dim,
            config.critic_net_arch,
            config.num_atoms,
            vmin=config.vmin,
            vmax=config.vmax,
        ).to(self.device)

        project_unit_rows_(self.actor)
        project_unit_rows_(self.critic)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        # These identities never change across optimizer or checkpoint loads.
        # Caching them avoids repeated module walks and dictionary construction
        # on every learner step, and enables multi-tensor CUDA operations.
        self._actor_linear_weights = tuple(
            child.weight
            for child in self.actor.modules()
            if isinstance(child, nn.Linear)
        )
        self._critic_linear_weights = tuple(
            child.weight
            for child in self.critic.modules()
            if isinstance(child, nn.Linear)
        )
        self._critic_parameters = tuple(self.critic.parameters())
        self._critic_target_parameters = tuple(self.critic_target.parameters())
        optimizer_execution = _optimizer_execution_kwargs(
            self.device, config.optimizer_backend
        )

        # With the released configuration, the Optax AdamW decay mask is empty:
        # hidden kernels are projected, predictor kernels are projected, and BN
        # decay is disabled.  Preserve that behavior even if the retained
        # diagnostic weight_decay value is non-zero.
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=config.actor_lr,
            betas=(0.9, 0.999),
            eps=config.adam_eps,
            weight_decay=0.0,
            **optimizer_execution,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.critic_lr,
            betas=(0.9, 0.999),
            eps=config.adam_eps,
            weight_decay=0.0,
            **optimizer_execution,
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(config.init_temperature), device=self.device)
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature],
            lr=config.actor_lr,
            betas=(0.9, 0.999),
            eps=config.adam_eps,
            **optimizer_execution,
        )

        self.target_entropy = (
            -self.action_dim / 2.0
            if config.target_entropy == "auto"
            else float(config.target_entropy)
        )
        self.reward_normalizer = DiscountedReturnNormalizer(config.gamma)
        self.update_step = 0
        self.actor_optimizer_steps = 0
        self.temperature_optimizer_steps = 0

        # Keep the small action-noise stream on CPU so checkpoints remain
        # portable between Hydra CUDA training and CPU evaluation machines.
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(0 if config.seed is None else config.seed)
        self._training_generator = (
            torch.Generator(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        # Debug mode deliberately retains eager validation fences, including
        # scalar support checks which are incompatible with a fixed graph.
        compile_enabled = (
            config.compile and not config.debug_checks and self.device.type == "cuda"
        )
        self._critic_loss_region = _MutableCompileRegion(
            "XQC critic loss",
            self._critic_loss_components,
            self.critic.buffers(),
            enabled=compile_enabled,
            strict=config.compile_strict,
        )
        self._actor_loss_region = _MutableCompileRegion(
            "XQC actor loss",
            self._actor_loss_components,
            self.actor.buffers(),
            enabled=compile_enabled,
            strict=config.compile_strict,
        )

    @property
    def num_updates(self) -> int:
        """Compatibility alias used by the shared off-policy wrapper."""

        return self.update_step

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    @property
    def semantic_signature(self) -> Dict[str, object]:
        return {
            "algorithm": "XQC",
            "official_commit": OFFICIAL_XQC_COMMIT,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "actor_net_arch": tuple(self.config.actor_net_arch),
            "critic_net_arch": tuple(self.config.critic_net_arch),
            "num_atoms": self.config.num_atoms,
            "vmin": self.config.vmin,
            "vmax": self.config.vmax,
            "log_std_min": LOG_STD_MIN,
            "log_std_max": LOG_STD_MAX,
            "bn_momentum": BN_MOMENTUM,
            "bn_epsilon": BN_EPSILON,
            "tau": self.config.tau,
            "gamma": self.config.gamma,
            "target_update_interval": self.config.target_update_interval,
            "policy_delay": self.config.policy_delay,
            "actor_lr": self.config.actor_lr,
            "critic_lr": self.config.critic_lr,
            "lr_end": self.config.lr_end,
            "transition_steps": self.config.transition_steps,
            "adam_eps": self.config.adam_eps,
            "target_entropy": self.target_entropy,
            "init_temperature": self.config.init_temperature,
            "reward_normalization": self.config.reward_normalization,
            "weight_projection": "all_linear_weight_rows",
            "temperature_lr_source": "actor_lr",
            "training_noise_rng": "portable_cpu_seed_device_local_v1",
        }

    def observe_reward(self, reward, terminated: bool, truncated: bool) -> None:
        if self.config.reward_normalization:
            self.reward_normalizer.update(
                reward, bool(terminated) or bool(truncated)
            )

    @staticmethod
    def _as_batch_tensor(value, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        observation = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        action, _ = self.actor.sample(
            observation,
            deterministic=deterministic,
            bn_mode="running",
            generator=self.generator,
        )
        if deterministic:
            # Official evaluation samples a zero-temperature distribution. The
            # action is exactly tanh(mean), but JAX still splits the learner RNG
            # and draws one action-shaped normal sample. Preserve that state
            # transition because evaluation therefore affects later train RNG.
            torch.randn(
                (1, self.action_dim),
                dtype=observation.dtype,
                device=self.generator.device,
                generator=self.generator,
            )
        return np.clip(action.cpu().numpy()[0], -1.0, 1.0)

    def q_predictions(self, obs, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        observations = self._as_batch_tensor(obs, self.device)
        actions = self._as_batch_tensor(actions, self.device)
        logits = self.critic(observations, actions, bn_mode="running")
        return tuple(logits.unbind(0))

    def q_values(self, obs, actions) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.stack(self.q_predictions(obs, actions), dim=0)
        values = self.critic.values_from_log_probs(F.log_softmax(logits, dim=-1))
        return tuple(values.unsqueeze(-1).unbind(0))

    def _prepared_batch(
        self,
        batch: Mapping[str, object],
        *,
        validate_finite: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch = require_mapping(batch, "XQC replay batch")
        required = {"obs", "actions", "rewards", "next_obs"}
        missing = sorted(required - set(batch))
        if missing:
            raise ValueError(f"XQC replay batch is missing fields: {missing}.")
        if "masks" not in batch and "dones" not in batch:
            raise ValueError("XQC replay batch must contain masks or dones.")
        prepared = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch.items()
            if key in required | {"masks", "dones", "discount"}
        }
        if prepared["obs"].ndim != 2 or prepared["obs"].shape[-1] != self.obs_dim:
            raise ValueError("XQC observations have an incompatible shape.")
        if prepared["next_obs"].shape != prepared["obs"].shape:
            raise ValueError("XQC next observations must match observations.")
        if (
            prepared["actions"].ndim != 2
            or prepared["actions"].shape[0] != prepared["obs"].shape[0]
            or prepared["actions"].shape[-1] != self.action_dim
        ):
            raise ValueError("XQC actions have an incompatible shape.")
        batch_size = prepared["obs"].shape[0]
        for name in ("rewards", "masks", "dones", "discount"):
            if name in prepared and prepared[name].numel() not in (1, batch_size):
                raise ValueError(f"XQC batch field {name!r} has an incompatible shape.")
        if validate_finite and not all(
            bool(torch.isfinite(value).all().item()) for value in prepared.values()
        ):
            raise ValueError("XQC replay batches must contain only finite values.")
        if "masks" not in prepared:
            prepared["masks"] = 1.0 - prepared["dones"]
        if "discount" not in prepared:
            prepared["discount"] = torch.tensor(self.config.gamma, device=self.device)
        if self.config.reward_normalization:
            prepared["rewards"] = self.reward_normalizer.normalize(prepared["rewards"])
        return prepared

    @staticmethod
    def _freeze_parameters(module: nn.Module, frozen: bool) -> None:
        module.requires_grad_(not frozen)

    def _critic_loss_components(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        discount: torch.Tensor,
        alpha: torch.Tensor,
        next_noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fixed-shape critic graph, including the official joined BN batch."""

        batch_size = observations.shape[0]
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(
                next_observations,
                bn_mode="running",
                noise=next_noise,
            )
            joined_observations = torch.cat((observations, next_observations), dim=0)
            joined_actions = torch.cat((actions, next_actions), dim=0)
            target_joined_log_probs = self.critic_target.log_probs(
                joined_observations,
                joined_actions,
                bn_mode="batch_no_update",
            )
            target_next_log_probs = target_joined_log_probs[:, batch_size:]
            selected_target_log_probs, target_q, _ = select_lower_distribution(
                target_next_log_probs, self.critic_target.support
            )
            projected_targets, clip_fraction = categorical_td_projection(
                selected_target_log_probs,
                rewards,
                masks,
                discount,
                alpha * next_log_probs,
                self.critic.support,
                validate_support=self.config.debug_checks,
            )

        online_joined_log_probs = self.critic.log_probs(
            joined_observations,
            joined_actions,
            bn_mode="batch_update",
        )
        current_log_probs = online_joined_log_probs[:, :batch_size]
        critic_loss = categorical_cross_entropy(current_log_probs, projected_targets)
        return critic_loss, current_log_probs, target_q, clip_fraction

    def _actor_loss_components(
        self,
        observations: torch.Tensor,
        alpha: torch.Tensor,
        actor_noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fixed-shape actor graph evaluated against the newly updated critic."""

        policy_actions, policy_log_probs = self.actor.sample(
            observations,
            bn_mode="batch_update",
            noise=actor_noise,
        )
        policy_logit_probs = self.critic.log_probs(
            observations, policy_actions, bn_mode="running"
        )
        policy_q_values = self.critic.values_from_log_probs(policy_logit_probs)
        minimum_policy_q = policy_q_values.min(dim=0).values
        actor_loss = (alpha * policy_log_probs - minimum_policy_q).mean()
        return actor_loss, policy_log_probs, minimum_policy_q

    def _update_once(
        self,
        batch: Mapping[str, object],
        *,
        next_noise: Optional[torch.Tensor] = None,
        actor_noise: Optional[torch.Tensor] = None,
        prepared: bool = False,
        collect_metrics: bool = True,
    ) -> Dict[str, float]:
        if not prepared:
            batch = self._prepared_batch(
                batch, validate_finite=self.config.debug_checks or collect_metrics
            )
        observations = batch["obs"]
        actions = batch["actions"]
        next_observations = batch["next_obs"]
        batch_size = observations.shape[0]
        rewards = batch["rewards"].reshape(batch_size)
        masks = batch["masks"].reshape(batch_size)
        discount = batch["discount"]
        alpha = self.temperature.detach()

        # Critic: infer the next action from stored actor statistics, then run
        # current and next samples together through each critic's batch moments.
        if next_noise is None:
            next_noise = torch.randn(
                (batch_size, self.action_dim),
                dtype=observations.dtype,
                device=self.generator.device,
                generator=self.generator,
            ).to(self.device)
        (
            critic_loss,
            current_log_probs,
            target_q,
            clip_fraction,
        ) = self._critic_loss_region(
            observations,
            actions,
            next_observations,
            rewards,
            masks,
            discount,
            alpha,
            next_noise,
        )
        critic_lr = linear_learning_rate(
            self.config.critic_lr,
            self.config.lr_end,
            self.update_step,
            self.config.transition_steps,
        )
        _set_optimizer_lr(self.critic_optimizer, critic_lr)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = (
            _global_grad_norm(self.critic.parameters())
            if collect_metrics
            else None
        )
        self.critic_optimizer.step()
        _project_unit_weights_(self._critic_linear_weights)

        # The official target gate reads the pre-update critic Model.step, which
        # starts at one.  Hence interval N fires on attempts N-1, 2N-1, ... .
        if (self.update_step + 1) % self.config.target_update_interval == 0:
            _polyak_update_parameter_lists_(
                self._critic_parameters,
                self._critic_target_parameters,
                self.config.tau,
            )

        # Actor: always evaluate the loss, gradient, BN update, and projection.
        # The conditional Optax wrapper masks only the Adam transformation.
        self._freeze_parameters(self.critic, frozen=True)
        try:
            if actor_noise is None:
                actor_noise = torch.randn(
                    (batch_size, self.action_dim),
                    dtype=observations.dtype,
                    device=self.generator.device,
                    generator=self.generator,
                ).to(self.device)
            actor_loss, policy_log_probs, minimum_policy_q = self._actor_loss_region(
                observations,
                alpha,
                actor_noise,
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = (
                _global_grad_norm(self.actor.parameters())
                if collect_metrics
                else None
            )
            actor_update_accepted = self.update_step % self.config.policy_delay == 0
            if actor_update_accepted:
                actor_lr = linear_learning_rate(
                    self.config.actor_lr,
                    self.config.lr_end,
                    self.actor_optimizer_steps,
                    self.config.transition_steps,
                )
                _set_optimizer_lr(self.actor_optimizer, actor_lr)
                self.actor_optimizer.step()
                self.actor_optimizer_steps += 1
            else:
                actor_lr = linear_learning_rate(
                    self.config.actor_lr,
                    self.config.lr_end,
                    self.actor_optimizer_steps,
                    self.config.transition_steps,
                )
            _project_unit_weights_(self._actor_linear_weights)
        finally:
            self._freeze_parameters(self.critic, frozen=False)

        # Temperature uses entropy from the pre-optimizer actor forward and the
        # old alpha value.  Its condition and inner schedule match the actor's.
        entropy = -policy_log_probs.detach().mean()
        temperature_value = self.temperature
        temperature_loss = temperature_value * (entropy - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        temperature_grad_norm = (
            _global_grad_norm([self.log_temperature])
            if collect_metrics
            else None
        )
        if actor_update_accepted:
            temperature_lr = linear_learning_rate(
                self.config.actor_lr,
                self.config.lr_end,
                self.temperature_optimizer_steps,
                self.config.transition_steps,
            )
            _set_optimizer_lr(self.temperature_optimizer, temperature_lr)
            self.temperature_optimizer.step()
            self.temperature_optimizer_steps += 1
        else:
            temperature_lr = linear_learning_rate(
                self.config.actor_lr,
                self.config.lr_end,
                self.temperature_optimizer_steps,
                self.config.transition_steps,
            )

        self.update_step += 1
        if not collect_metrics:
            return {}
        current_values = self.critic.values_from_log_probs(current_log_probs.detach())
        tensor_metric_names = (
            "actor_loss",
            "critic_loss",
            "temperature",
            "temperature_loss",
            "policy_entropy",
            "policy_log_prob",
            "q1_mean",
            "q2_mean",
            "q_target_mean",
            "q_policy_mean",
            "q_disagreement_mean",
            "q_target_clip_fraction",
            "actor_grad_norm",
            "critic_grad_norm",
            "temperature_grad_norm",
        )
        tensor_metrics = torch.stack(
            (
                actor_loss.detach(),
                critic_loss.detach(),
                temperature_value.detach(),
                temperature_loss.detach(),
                entropy,
                policy_log_probs.detach().mean(),
                current_values[0].mean(),
                current_values[1].mean(),
                target_q.mean(),
                minimum_policy_q.detach().mean(),
                (current_values[0] - current_values[1]).abs().mean(),
                clip_fraction,
                actor_grad_norm.detach(),
                critic_grad_norm.detach(),
                temperature_grad_norm.detach(),
            )
        )
        # One packed transfer replaces fifteen CUDA synchronization points.
        host_metrics = tensor_metrics.cpu().tolist()
        if self.config.debug_checks and not np.isfinite(host_metrics).all():
            raise FloatingPointError("XQC learner produced a non-finite metric.")
        metrics = dict(zip(tensor_metric_names, map(float, host_metrics)))
        metrics.update(
            {
                "actor_update_accepted": float(actor_update_accepted),
                "actor_learning_rate": float(actor_lr),
                "critic_learning_rate": float(critic_lr),
                "temperature_learning_rate": float(temperature_lr),
                "reward_scale": float(self.reward_normalizer.scale),
            }
        )
        return metrics

    def _sample_update_noises(
        self, gradient_steps: int, batch_size: int
    ) -> torch.Tensor:
        """Draw all UTD policy noise in one operation on the learner device.

        CUDA uses a short-lived seed from the checkpointed CPU RNG stream. The
        CUDA generator is reseeded for each public ``update`` call, so its
        opaque device-specific state never needs to enter a checkpoint. This
        keeps checkpoints loadable on CPU while eliminating pageable-CPU noise
        copies from the training hot path. CPU retains the original generator
        directly.
        """

        shape = (gradient_steps, 2, batch_size, self.action_dim)
        if self._training_generator is None:
            # CPU is the exact portable reference stream. Separate calls retain
            # the historical draw boundaries used by ``_update_once``.
            return torch.stack(
                [
                    torch.stack(
                        [
                            torch.randn(
                                (batch_size, self.action_dim),
                                dtype=torch.float32,
                                device=self.device,
                                generator=self.generator,
                            )
                            for _ in range(2)
                        ]
                    )
                    for _ in range(gradient_steps)
                ]
            )
        seed = int(
            torch.randint(
                0,
                torch.iinfo(torch.int64).max,
                (1,),
                dtype=torch.int64,
                device="cpu",
                generator=self.generator,
            )[0]
        )
        self._training_generator.manual_seed(seed)
        return torch.randn(
            shape,
            dtype=torch.float32,
            device=self.device,
            generator=self._training_generator,
        )

    def _sample_replay_batch(
        self, replay_buffer, total_batch_size: int
    ) -> Mapping[str, object]:
        """Sample UTD data once and use one packed host-to-device transfer."""

        sample_device = (
            torch.device("cpu") if self.device.type == "cuda" else self.device
        )
        batch = require_mapping(
            replay_buffer.sample(total_batch_size, sample_device),
            "XQC replay batch",
        )
        if self.device.type != "cuda":
            return batch

        relevant = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in batch.items()
            if key
            in {
                "obs",
                "actions",
                "rewards",
                "next_obs",
                "masks",
                "dones",
                "discount",
            }
        }
        batched = [
            (key, value)
            for key, value in relevant.items()
            if value.device.type == "cpu"
            and value.ndim > 0
            and value.shape[0] == total_batch_size
        ]
        if not batched:
            return batch

        packed = torch.cat(
            [value.reshape(total_batch_size, -1) for _, value in batched], dim=1
        ).to(self.device)
        transferred = dict(batch)
        offset = 0
        for key, value in batched:
            width = value.numel() // total_batch_size
            transferred[key] = packed[:, offset : offset + width].reshape(value.shape)
            offset += width
        return transferred

    @staticmethod
    def _slice_prepared_batch(
        batch: Mapping[str, torch.Tensor],
        index: int,
        batch_size: int,
        total_batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        start = index * batch_size
        stop = start + batch_size
        return {
            key: (
                value[start:stop]
                if value.ndim > 0 and value.shape[0] == total_batch_size
                else value
            )
            for key, value in batch.items()
        }

    def update(
        self, replay_buffer, gradient_steps: int, batch_size: int
    ) -> Dict[str, float]:
        gradient_steps = _positive_int(gradient_steps, "gradient_steps")
        batch_size = _positive_int(batch_size, "batch_size")
        total_batch_size = gradient_steps * batch_size
        batch = self._prepared_batch(
            self._sample_replay_batch(replay_buffer, total_batch_size),
            validate_finite=self.config.debug_checks,
        )
        noises = self._sample_update_noises(gradient_steps, batch_size)
        metrics = None
        for index in range(gradient_steps):
            collect_metrics = self.config.debug_checks or index == gradient_steps - 1
            metrics = self._update_once(
                self._slice_prepared_batch(
                    batch, index, batch_size, total_batch_size
                ),
                next_noise=noises[index, 0],
                actor_noise=noises[index, 1],
                prepared=True,
                collect_metrics=collect_metrics,
            )
        # JAX's fori_loop threads only the most recent ``info`` mapping through
        # the state, so the released learner returns the final minibatch rather
        # than an average over UTD updates.
        return metrics

    @staticmethod
    def _validate_counter(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"XQC checkpoint {name} must be a non-negative integer.")
        return value

    @staticmethod
    def _validate_optimizer_steps(
        optimizer: torch.optim.Optimizer,
        incoming: Mapping[str, object],
        name: str,
        expected_steps: int,
    ) -> None:
        state = incoming["state"]
        parameter_count = sum(len(group["params"]) for group in optimizer.param_groups)
        if expected_steps == 0:
            if state:
                raise ValueError(f"{name} must be uninitialized at step zero.")
            return
        if len(state) != parameter_count:
            raise ValueError(f"{name} optimizer state inventory is incomplete.")
        for parameter_state in state.values():
            step = parameter_state["step"]
            numeric_step = float(step.item() if torch.is_tensor(step) else step)
            if numeric_step != expected_steps:
                raise ValueError(f"{name} optimizer step does not match its counter.")
            if any(
                torch.is_tensor(value)
                and not bool(torch.isfinite(value).all().item())
                for value in parameter_state.values()
            ):
                raise ValueError(f"{name} optimizer state must be finite.")

    @staticmethod
    def _validate_finite_module_state(state: Mapping[str, object], name: str) -> None:
        for key, value in state.items():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} tensor {key!r} must be finite.")

    def state_dict(self) -> Dict[str, object]:
        return {
            "state_version": self._STATE_VERSION,
            "semantic_spec": copy.deepcopy(self.semantic_signature),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "update_step": self.update_step,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "temperature_optimizer_steps": self.temperature_optimizer_steps,
            "reward_normalizer": self.reward_normalizer.state_dict(),
            "rng_state": self.generator.get_state().cpu(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected_keys = {
            "state_version",
            "semantic_spec",
            "actor",
            "critic",
            "critic_target",
            "actor_optimizer",
            "critic_optimizer",
            "log_temperature",
            "temperature_optimizer",
            "update_step",
            "actor_optimizer_steps",
            "temperature_optimizer_steps",
            "reward_normalizer",
            "rng_state",
        }
        state = require_exact_keys(state, expected_keys, "XQC checkpoint state")
        if (
            isinstance(state["state_version"], bool)
            or not isinstance(state["state_version"], int)
            or state["state_version"] != self._STATE_VERSION
        ):
            raise ValueError("XQC checkpoint state version is incompatible.")
        if state["semantic_spec"] != self.semantic_signature:
            raise ValueError(
                "XQC checkpoint semantic specification does not match this agent."
            )

        update_step = self._validate_counter(state["update_step"], "update_step")
        actor_steps = self._validate_counter(
            state["actor_optimizer_steps"], "actor_optimizer_steps"
        )
        temperature_steps = self._validate_counter(
            state["temperature_optimizer_steps"], "temperature_optimizer_steps"
        )
        expected_delayed_steps = (
            0
            if update_step == 0
            else (update_step - 1) // self.config.policy_delay + 1
        )
        if actor_steps != expected_delayed_steps or temperature_steps != actor_steps:
            raise ValueError("XQC delayed optimizer counters are inconsistent.")

        preflight_module_state(self.actor, state["actor"], "XQC checkpoint actor")
        preflight_module_state(self.critic, state["critic"], "XQC checkpoint critic")
        preflight_module_state(
            self.critic_target,
            state["critic_target"],
            "XQC checkpoint target critic",
        )
        preflight_optimizer_state(
            self.actor_optimizer,
            state["actor_optimizer"],
            "XQC checkpoint actor optimizer",
        )
        preflight_optimizer_state(
            self.critic_optimizer,
            state["critic_optimizer"],
            "XQC checkpoint critic optimizer",
        )
        preflight_optimizer_state(
            self.temperature_optimizer,
            state["temperature_optimizer"],
            "XQC checkpoint temperature optimizer",
        )
        self._validate_optimizer_steps(
            self.actor_optimizer,
            state["actor_optimizer"],
            "XQC checkpoint actor",
            actor_steps,
        )
        self._validate_optimizer_steps(
            self.critic_optimizer,
            state["critic_optimizer"],
            "XQC checkpoint critic",
            update_step,
        )
        self._validate_optimizer_steps(
            self.temperature_optimizer,
            state["temperature_optimizer"],
            "XQC checkpoint temperature",
            temperature_steps,
        )

        saved_log_temperature = require_tensor(
            state["log_temperature"],
            "XQC checkpoint log_temperature",
            shape=self.log_temperature.shape,
            dtype=self.log_temperature.dtype,
        )
        if not bool(torch.isfinite(saved_log_temperature).all().item()):
            raise ValueError("XQC checkpoint log_temperature must be finite.")
        rng_state = require_tensor(
            state["rng_state"], "XQC checkpoint RNG state", dtype=torch.uint8
        )
        if rng_state.ndim != 1 or rng_state.numel() == 0:
            raise ValueError("XQC checkpoint RNG state is incompatible.")
        try:
            rng_probe = torch.Generator(device="cpu")
            rng_probe.set_state(rng_state.cpu())
        except RuntimeError as exc:
            raise ValueError("XQC checkpoint RNG state is incompatible.") from exc
        self.reward_normalizer._validated_state(state["reward_normalizer"])
        self._validate_finite_module_state(state["actor"], "XQC actor")
        self._validate_finite_module_state(state["critic"], "XQC critic")
        self._validate_finite_module_state(
            state["critic_target"], "XQC target critic"
        )

        # Nothing above mutates live state.  Install the fully preflighted
        # payload only after every component has passed validation.
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        load_optimizer_state_preserving_hyperparameters(
            self.actor_optimizer, state["actor_optimizer"]
        )
        load_optimizer_state_preserving_hyperparameters(
            self.critic_optimizer, state["critic_optimizer"]
        )
        self.log_temperature.data.copy_(saved_log_temperature.to(self.device))
        load_optimizer_state_preserving_hyperparameters(
            self.temperature_optimizer, state["temperature_optimizer"]
        )
        self.reward_normalizer.load_state_dict(state["reward_normalizer"])
        self.generator.set_state(rng_state.cpu())
        self.update_step = update_step
        self.actor_optimizer_steps = actor_steps
        self.temperature_optimizer_steps = temperature_steps

        def last_used_lr(start: float, completed_steps: int) -> float:
            schedule_step = max(0, completed_steps - 1)
            return linear_learning_rate(
                start, self.config.lr_end, schedule_step, self.config.transition_steps
            )

        _set_optimizer_lr(
            self.critic_optimizer,
            last_used_lr(self.config.critic_lr, self.update_step),
        )
        _set_optimizer_lr(
            self.actor_optimizer,
            last_used_lr(self.config.actor_lr, self.actor_optimizer_steps),
        )
        _set_optimizer_lr(
            self.temperature_optimizer,
            last_used_lr(self.config.actor_lr, self.temperature_optimizer_steps),
        )
