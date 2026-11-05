"""State-dependent Gaussian exploration with explicit, caller-owned noise.

The actor owns learned matrix scales, never a sampled matrix.  Collection can
hold a standard-normal matrix while the actor changes; independent SAC samples
use the same fixed-input Gaussian marginal.  No variance jitter or independent
action noise is added to that marginal.
"""

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

from .layers import NormedLinear


class GsdeActor(nn.Module):
    """AMBI's ordinary actor trunk, mean head, and full gSDE scale matrix."""

    is_gsde = True

    def __init__(self, latent_dim, hidden_dim, action_dim, *, feature_source="actor",
                 log_std_init=-3.0, mean_clip=2.0):
        super().__init__()
        if feature_source not in {"actor", "latent"}:
            raise ValueError("gsde_feature_source must be 'actor' or 'latent'.")
        if (isinstance(log_std_init, bool) or not isinstance(log_std_init, Real)
                or not math.isfinite(log_std_init)):
            raise ValueError("gsde_log_std_init must be finite.")
        if mean_clip is not None and (
            isinstance(mean_clip, bool) or not isinstance(mean_clip, Real)
            or not math.isfinite(mean_clip) or mean_clip <= 0
        ):
            raise ValueError("gsde_mean_clip must be positive and finite, or None.")
        self.feature_source = feature_source
        self.sde_feature_dim = int(hidden_dim if feature_source == "actor" else latent_dim)
        self.sde_action_dim = int(action_dim)
        self.mean_clip = None if mean_clip is None else float(mean_clip)
        self.trunk = nn.Sequential(
            NormedLinear(latent_dim, hidden_dim),
            NormedLinear(hidden_dim, hidden_dim),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_scale = nn.Parameter(torch.full(
            (self.sde_feature_dim, self.sde_action_dim), float(log_std_init),
        ))

    def _distribution_terms(self, x, *, detach_parameters=False):
        """Evaluate current features and validated marginal statistics once."""
        if detach_parameters:
            # Explicit tensor operations support Torch 2.3 full-graph probes;
            # functional_call(module, ...) is not traceable there. These layers
            # have no dropout or trainable activation parameters.
            hidden = x
            for layer in self.trunk:
                hidden = F.linear(hidden, layer.weight.detach(), layer.bias.detach())
                hidden = F.layer_norm(
                    hidden, layer.ln.normalized_shape,
                    layer.ln.weight.detach(), layer.ln.bias.detach(), layer.ln.eps,
                )
                hidden = layer.act(hidden)
            mean = F.linear(
                hidden, self.mean_head.weight.detach(), self.mean_head.bias.detach(),
            )
            log_scale = self.log_scale.detach()
        else:
            hidden = self.trunk(x)
            mean = self.mean_head(hidden)
            log_scale = self.log_scale
        features = hidden if self.feature_source == "actor" else x
        torch._assert_async(torch.isfinite(mean).all(), "gSDE mean outputs must be finite.")
        if self.mean_clip is not None:
            mean = mean.clamp(-self.mean_clip, self.mean_clip)
        distribution_dtype = (
            torch.float32 if features.dtype in {torch.float16, torch.bfloat16}
            else features.dtype
        )
        features = features.to(dtype=distribution_dtype)
        mean = mean.to(dtype=distribution_dtype)
        log_scale = log_scale.to(dtype=distribution_dtype)
        scale = log_scale.exp()
        torch._assert_async(
            torch.isfinite(features).all() & torch.isfinite(mean).all()
            & torch.isfinite(scale).all() & (scale > 0).all(),
            "gSDE features, means, and matrix scales must be finite; scales must be positive.",
        )
        nonzero = features != 0
        torch._assert_async(
            nonzero.any(dim=-1).all(),
            "gSDE requires a nonzero feature vector; zero features give a singular Gaussian.",
        )
        log_std = self._marginal_log_std(features, log_scale)
        std = log_std.exp()
        torch._assert_async(
            torch.isfinite(std).all() & (std > 0).all(),
            "gSDE marginal standard deviations must be finite and positive.",
        )
        return mean, features, scale, log_std, std

    @staticmethod
    def _marginal_log_std(features, log_scale):
        """Exact variance with a measured CPU matrix-multiplication path.

        Every product of squared finite float32 features and positive finite
        float32 matrix scales fits float64, even when float32 variance would
        underflow or overflow. The GEMM therefore avoids the batch-by-feature-
        by-action broadcast and its backward without a variance floor. Native
        float64 retains the log-domain reduction for its wider exponent range.
        CUDA retains that implementation until its FP64 tradeoff is measured.
        """
        if features.device.type == "cpu" and features.dtype == torch.float32:
            variance = features.double().square() @ (2.0 * log_scale.double()).exp()
            return (0.5 * variance.log()).to(dtype=features.dtype)
        nonzero = features != 0
        # Replacing only exact zeros before log preserves subnormal nonzero
        # features, and avoids a hidden log(0) backward through torch.where.
        safe_abs = torch.where(nonzero, features.abs(), torch.ones_like(features))
        log_weight = safe_abs.log().unsqueeze(-1) + log_scale
        log_variance_terms = torch.where(
            nonzero.unsqueeze(-1), 2.0 * log_weight,
            torch.full_like(log_weight, -torch.inf),
        )
        return 0.5 * torch.logsumexp(log_variance_terms, dim=-2)

    def _validated_noise(self, features, sde_noise):
        expected_shape = (*features.shape[:-1], self.sde_feature_dim, self.sde_action_dim)
        if not torch.is_tensor(sde_noise) or sde_noise.shape != expected_shape:
            raise ValueError(f"gSDE noise must have shape {expected_shape}.")
        sde_noise = sde_noise.to(device=features.device, dtype=features.dtype)
        torch._assert_async(torch.isfinite(sde_noise).all(), "gSDE noise must be finite.")
        return sde_noise

    @staticmethod
    def _matrix_perturbation(features, scale, sde_noise):
        perturbation = (features.unsqueeze(-1) * scale * sde_noise).sum(dim=-2)
        torch._assert_async(
            torch.isfinite(perturbation).all(), "gSDE matrix perturbations must be finite.",
        )
        return perturbation

    def forward(self, x, *, sde_noise=None, detach_parameters=False):
        """Return mean, marginal log std, and optional standardized matrix noise.

        Half-precision inputs use float32 distribution math. Zero feature
        coordinates contribute exactly zero to the variance and its derivative;
        an entirely zero feature row is a singular distribution and is rejected.
        """
        mean, features, scale, log_std, std = self._distribution_terms(
            x, detach_parameters=detach_parameters,
        )
        standardized_noise = None
        if sde_noise is not None:
            sde_noise = self._validated_noise(features, sde_noise)
            # Keep this linear in features, including at individual zero
            # coordinates. The standardized representation reuses the common
            # Gaussian density and squashing implementation without resampling.
            perturbation = self._matrix_perturbation(features, scale, sde_noise)
            standardized_noise = perturbation / std
            torch._assert_async(
                torch.isfinite(standardized_noise).all(),
                "gSDE matrix perturbations must be finite.",
            )
        return mean, log_std, standardized_noise

    def action(self, x, *, sde_noise=None, deterministic=False,
               detach_parameters=False, std_scale=1.0):
        """Execute a mean or matrix action without constructing Gaussian noise.

        Keep the same distribution validity checks as the statistics path.
        The common unit-scale matrix case uses its perturbation directly;
        non-unit scaling retains log-space handling of extreme finite scalars.
        """
        std_scale = float(std_scale)
        if not 0.0 < std_scale < float("inf"):
            raise ValueError(f"std_scale must be positive, got {std_scale}.")
        mean, features, scale, log_std, std = self._distribution_terms(
            x, detach_parameters=detach_parameters,
        )
        if std_scale != 1.0:
            scaled_std = (log_std + math.log(std_scale)).exp()
            torch._assert_async(
                torch.isfinite(scaled_std).all() & (scaled_std > 0).all(),
                "Scaled gSDE marginal standard deviations must be finite and positive.",
            )
        if sde_noise is not None:
            sde_noise = self._validated_noise(features, sde_noise)
        if deterministic:
            return mean.tanh()
        if sde_noise is None:
            raise ValueError("gSDE action-only sampling requires an explicit matrix or deterministic=True.")
        perturbation = self._matrix_perturbation(features, scale, sde_noise)
        torch._assert_async(
            (perturbation.abs() <= std * torch.finfo(std.dtype).max).all(),
            "gSDE matrix perturbations must be finite.",
        )
        if std_scale != 1.0:
            perturbation = (perturbation / std) * scaled_std
        return (mean + perturbation).tanh()
