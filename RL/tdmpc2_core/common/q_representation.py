"""Backend-neutral scalar and distributional critic representations.

This module intentionally does not inherit from :class:`torch.nn.Module`.
The critic networks remain owned by ``SoftWorldModel._Qs`` and
``SoftWorldModel._target_Qs`` so existing checkpoint keys stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
import torch.nn.functional as F


_REDUCTIONS = {"min_pair", "mean_pair", "min_all", "mean_all", "all"}
_REDUCTION_ALIASES = {"min": "min_pair", "avg": "mean_pair"}


def _symlog(value):
    return torch.sign(value) * torch.log1p(torch.abs(value))


def _symexp(value):
    return torch.sign(value) * torch.expm1(torch.abs(value))


class SymexpTwoHotCodec:
    """Mean-preserving categorical regression over symexp-spaced raw support.

    Leading axes may denote batches, times, ensembles, or value components. The
    codec has no policy/value semantics and only interprets the final bin axis.
    """

    name = "symexp_two_hot_mean_v1"

    def __init__(self, num_bins, vmin, vmax):
        self.num_bins = int(num_bins)
        self.vmin, self.vmax = float(vmin), float(vmax)
        if self.num_bins < 2:
            raise ValueError("Mean-preserving symexp regression requires at least 2 bins.")
        if not (isfinite(self.vmin) and isfinite(self.vmax) and self.vmin < self.vmax):
            raise ValueError("Symexp support bounds must be finite and increasing.")
        self._support_cache = {}

    def support(self, reference):
        key = (reference.device, reference.dtype)
        support = self._support_cache.get(key)
        if support is None:
            support = _symexp(torch.linspace(
                self.vmin, self.vmax, self.num_bins,
                device=reference.device, dtype=reference.dtype,
            ))
            self._support_cache[key] = support
        return support

    def _target(self, target):
        if target.ndim == 0:
            return target.reshape(1, 1)
        if target.shape[-1] != 1:
            raise ValueError("Regression targets require a trailing singleton dimension.")
        return target

    def bin_weights(self, target):
        target = self._target(target)
        support = self.support(target)
        bounded = target.clamp(min=support[0], max=support[-1])
        # Searching raw support avoids symlog roundoff choosing a neighbouring
        # interval at an exact support point. Both interpolation and decoding
        # therefore use the very same floating-point support values.
        upper = torch.searchsorted(support, bounded.contiguous()).clamp(1, self.num_bins - 1)
        lower = upper - 1
        low_value, high_value = support[lower], support[upper]
        high_weight = (bounded - low_value) / (high_value - low_value)
        return lower, upper, 1.0 - high_weight, high_weight

    def encode_target(self, target):
        target = self._target(target)
        lower, upper, low_weight, high_weight = self.bin_weights(target)
        encoded = target.new_zeros(*target.shape[:-1], self.num_bins)
        encoded.scatter_add_(-1, lower, low_weight)
        encoded.scatter_add_(-1, upper, high_weight)
        return encoded

    def decode(self, predictions):
        if predictions.shape[-1] != self.num_bins:
            raise ValueError(f"Expected {self.num_bins} categorical logits.")
        return (predictions.softmax(dim=-1) * self.support(predictions)).sum(-1, keepdim=True)

    def loss(self, predictions, target, *, reduction="mean"):
        if predictions.shape[-1] != self.num_bins:
            raise ValueError(f"Expected {self.num_bins} categorical logits.")
        lower, upper, low_weight, high_weight = self.bin_weights(target)
        index_shape = predictions.shape[:-1] + (1,)
        lower = torch.broadcast_to(lower, index_shape)
        upper = torch.broadcast_to(upper, index_shape)
        log_probabilities = F.log_softmax(predictions, dim=-1)
        losses = -(
            low_weight * log_probabilities.gather(-1, lower)
            + high_weight * log_probabilities.gather(-1, upper)
        )
        if reduction == "none":
            return losses
        if reduction == "mean":
            return losses.mean()
        if reduction == "sum":
            return losses.sum()
        raise ValueError(f"Unknown regression loss reduction: {reduction!r}.")

    def clipping_fraction(self, target):
        support = self.support(target)
        return ((target < support[0]) | (target > support[-1])).to(target.dtype).mean()


@dataclass(frozen=True)
class CriticSignature:
    """Architecture metadata needed to preflight critic checkpoints."""

    q_representation: str
    num_q: int
    q_num_bins: int
    q_vmin: float | None
    q_vmax: float | None

    def as_dict(self):
        return {
            "q_representation": self.q_representation,
            "num_q": self.num_q,
            "q_num_bins": self.q_num_bins,
            "q_vmin": self.q_vmin,
            "q_vmax": self.q_vmax,
        }


class QRepresentation:
    """Translate between critic network predictions and scalar Q-values.

    Scalar critics emit one value per head and use mean squared error.
    Distributional critics emit categorical logits over symlog-spaced bins,
    decode them to scalar expectations, and use soft two-hot cross entropy.
    """

    def __init__(
        self,
        representation,
        *,
        num_q,
        pair_size=2,
        num_bins=None,
        vmin=None,
        vmax=None,
        codec="symlog",
    ):
        representation = str(representation).lower()
        if representation not in {"scalar", "distributional"}:
            raise ValueError(
                "q_representation must be 'scalar' or 'distributional', "
                f"got {representation!r}."
            )

        num_q = int(num_q)
        if representation == "scalar" and num_q != 2:
            raise ValueError("Scalar SAC critics require exactly num_q=2.")
        if representation == "distributional" and num_q < 2:
            raise ValueError("Distributional Q ensembles require num_q>=2.")

        pair_size = int(pair_size)
        if pair_size <= 0 or pair_size > num_q:
            raise ValueError(
                f"q_pair_size must be in [1, num_q={num_q}], got {pair_size}."
            )

        if representation == "distributional":
            if num_bins is None:
                raise ValueError("q_num_bins is required for distributional Q critics.")
            num_bins = int(num_bins)
            if num_bins < 2:
                raise ValueError("q_num_bins must be at least 2 for distributional Q critics.")
            if vmin is None or vmax is None:
                raise ValueError("q_vmin and q_vmax are required for distributional Q critics.")
            vmin, vmax = float(vmin), float(vmax)
            if not (isfinite(vmin) and isfinite(vmax) and vmin < vmax):
                raise ValueError(f"q_vmin must be smaller than q_vmax, got {vmin} >= {vmax}.")
        else:
            # Scalar signatures describe the actual one-unit output head. Q-bin
            # settings are deliberately irrelevant to the scalar architecture.
            num_bins, vmin, vmax = 1, None, None

        self.representation = representation
        self.num_q = num_q
        self.pair_size = pair_size
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        if codec not in {"symlog", "symexp_mean"}:
            raise ValueError(f"Unknown categorical codec: {codec!r}.")
        if codec == "symexp_mean" and representation != "distributional":
            raise ValueError("Mean-preserving symexp regression requires distributional critics.")
        self.codec = codec
        self.value_codec = (
            SymexpTwoHotCodec(num_bins, vmin, vmax) if codec == "symexp_mean" else None
        )
        # QRepresentation intentionally is not an nn.Module, so this cache does
        # not add checkpoint entries. A model normally uses one device/dtype;
        # retaining the uncommon alternatives keeps device moves correct too.
        self._support_cache = {}

    @classmethod
    def from_config(cls, cfg):
        """Build a representation while accepting legacy scalar configs."""
        representation = str(getattr(cfg, "q_representation", "scalar")).lower()
        if representation == "distributional":
            num_bins = getattr(cfg, "q_num_bins", getattr(cfg, "num_bins", None))
            vmin = getattr(cfg, "q_vmin", getattr(cfg, "vmin", None))
            vmax = getattr(cfg, "q_vmax", getattr(cfg, "vmax", None))
        else:
            num_bins = vmin = vmax = None
        return cls(
            representation,
            num_q=getattr(cfg, "num_q"),
            pair_size=getattr(cfg, "q_pair_size", 2),
            num_bins=num_bins,
            vmin=vmin,
            vmax=vmax,
            codec=(
                "symexp_mean"
                if str(getattr(cfg, "critic_value_mode", "single")).lower() != "single"
                else "symlog"
            ),
        )

    @property
    def output_dim(self):
        return 1 if self.representation == "scalar" else self.num_bins

    @property
    def signature(self):
        return CriticSignature(
            q_representation=self.representation,
            num_q=self.num_q,
            q_num_bins=self.num_bins,
            q_vmin=self.vmin,
            q_vmax=self.vmax,
        )

    def _validate_predictions(self, predictions):
        if predictions.ndim < 2:
            raise ValueError(
                "Critic predictions must have a leading ensemble dimension and "
                f"an output dimension, got shape {tuple(predictions.shape)}."
            )
        if predictions.shape[0] != self.num_q:
            raise ValueError(
                f"Expected {self.num_q} Q heads, got {predictions.shape[0]}."
            )
        if predictions.shape[-1] != self.output_dim:
            raise ValueError(
                f"Expected critic output dimension {self.output_dim}, "
                f"got {predictions.shape[-1]}."
            )

    def encode_target(self, scalar_target):
        """Encode scalar targets using the distributional symlog bins."""
        if self.value_codec is not None:
            return self.value_codec.encode_target(scalar_target)
        if self.representation == "scalar":
            return scalar_target
        if scalar_target.ndim == 0:
            scalar_target = scalar_target.reshape(1, 1)
        elif scalar_target.shape[-1] != 1:
            raise ValueError(
                "Scalar Q targets must have a trailing singleton dimension, "
                f"got shape {tuple(scalar_target.shape)}."
            )

        symlog_target = _symlog(scalar_target).clamp(self.vmin, self.vmax)
        position = (symlog_target - self.vmin) / (self.vmax - self.vmin)
        position = position * (self.num_bins - 1)
        lower = position.floor().long()
        upper = (lower + 1).clamp(max=self.num_bins - 1)
        upper_weight = position - lower.to(position.dtype)
        lower_weight = 1.0 - upper_weight

        encoded = scalar_target.new_zeros(*scalar_target.shape[:-1], self.num_bins)
        encoded.scatter_add_(-1, lower, lower_weight)
        encoded.scatter_add_(-1, upper, upper_weight)
        return encoded

    def _target_bin_weights(self, scalar_target):
        """Return the two occupied bins and their interpolation weights."""
        if self.value_codec is not None:
            return self.value_codec.bin_weights(scalar_target)
        symlog_target = _symlog(scalar_target).clamp(self.vmin, self.vmax)
        position = (symlog_target - self.vmin) / (self.vmax - self.vmin)
        position = position * (self.num_bins - 1)
        lower = position.floor().long()
        upper = (lower + 1).clamp(max=self.num_bins - 1)
        upper_weight = position - lower.to(position.dtype)
        return lower, upper, 1.0 - upper_weight, upper_weight

    def _support(self, reference):
        """Return a cached categorical support matching ``reference``."""
        key = (reference.device, reference.dtype)
        support = self._support_cache.get(key)
        if support is None:
            support = torch.linspace(
                self.vmin,
                self.vmax,
                self.num_bins,
                device=reference.device,
                dtype=reference.dtype,
            )
            self._support_cache[key] = support
        return support

    def decode(self, predictions):
        """Decode every critic head to a scalar Q expectation."""
        self._validate_predictions(predictions)
        if self.value_codec is not None:
            return self.value_codec.decode(predictions)
        if self.representation == "scalar":
            return predictions

        symlog_value = (F.softmax(predictions, dim=-1) * self._support(predictions)).sum(
            dim=-1, keepdim=True
        )
        return _symexp(symlog_value)

    def loss(self, predictions, scalar_target, *, reduction="mean"):
        """Compute a per-head scalar or categorical critic loss."""
        self._validate_predictions(predictions)
        if self.value_codec is not None:
            return self.value_codec.loss(predictions, scalar_target, reduction=reduction)
        if scalar_target.ndim == 0:
            scalar_target = scalar_target.reshape(1, 1)
        elif scalar_target.shape[-1] != 1:
            raise ValueError(
                "Scalar Q targets must have a trailing singleton dimension, "
                f"got shape {tuple(scalar_target.shape)}."
            )

        if self.representation == "scalar":
            losses = (predictions - scalar_target) ** 2
        else:
            lower, upper, lower_weight, upper_weight = self._target_bin_weights(
                scalar_target
            )
            index_shape = predictions.shape[:-1] + (1,)
            lower = torch.broadcast_to(lower, index_shape)
            upper = torch.broadcast_to(upper, index_shape)
            lower_weight = torch.broadcast_to(lower_weight, index_shape)
            upper_weight = torch.broadcast_to(upper_weight, index_shape)
            log_probabilities = F.log_softmax(predictions, dim=-1)
            losses = -(
                lower_weight * log_probabilities.gather(-1, lower)
                + upper_weight * log_probabilities.gather(-1, upper)
            )

        if reduction == "none":
            return losses
        if reduction == "mean":
            return losses.mean()
        if reduction == "sum":
            return losses.sum()
        raise ValueError(f"Unknown critic loss reduction: {reduction!r}.")

    def reduce(
        self,
        values,
        reduction,
        *,
        pair_indices=None,
        generator=None,
        trusted_pair_indices=False,
    ):
        """Reduce decoded scalar values across the ensemble dimension."""
        if values.ndim < 2 or values.shape[0] != self.num_q or values.shape[-1] != 1:
            raise ValueError(
                "Q reduction requires decoded values with shape "
                f"[{self.num_q}, ..., 1], got {tuple(values.shape)}."
            )

        reduction = _REDUCTION_ALIASES.get(reduction, reduction)
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f"Unknown Q reduction {reduction!r}; expected one of "
                f"{sorted(_REDUCTIONS - {'all'})}."
            )
        if reduction == "all":
            if pair_indices is not None:
                raise ValueError("pair_indices cannot be supplied with reduction='all'.")
            return values
        if reduction.endswith("_all"):
            if pair_indices is not None:
                raise ValueError(
                    f"pair_indices cannot be supplied with reduction={reduction!r}."
                )
            selected = values
        else:
            # The default pair is the whole ensemble for scalar twin critics
            # (and for full-size distributional pairs), so selecting an arange
            # would only launch an extra gather kernel.
            if pair_indices is None and self.pair_size == self.num_q:
                selected = values
            else:
                selected = values.index_select(
                    0,
                    self._pair_indices(
                        values.device,
                        pair_indices=pair_indices,
                        generator=generator,
                        trusted=trusted_pair_indices,
                    ),
                )

        if reduction.startswith("min_"):
            return selected.min(dim=0).values
        return selected.mean(dim=0)

    def sample_pair_indices(self, device, *, generator=None):
        """Sample an explicit critic pair, or return ``None`` for identity pairs.

        Sampling before entering a compiled region keeps generator objects out
        of the graph while preserving the configured without-replacement law.
        """
        if self.pair_size == self.num_q:
            return None
        return self._pair_indices(device, generator=generator)

    def _pair_indices(
        self,
        device,
        *,
        pair_indices=None,
        generator=None,
        trusted=False,
    ):
        if pair_indices is None:
            if self.pair_size == self.num_q:
                return torch.arange(self.num_q, device=device)
            return torch.randperm(
                self.num_q,
                device=device,
                generator=generator,
            )[: self.pair_size]

        indices = torch.as_tensor(pair_indices, device=device, dtype=torch.long).flatten()
        if trusted:
            # Only internally sampled randperm prefixes use this path. They are
            # already unique/in-range, so avoid unique()/any() scalar reads in
            # compiled critic and actor regions.
            return indices
        if indices.numel() != self.pair_size:
            raise ValueError(
                f"Expected {self.pair_size} pair indices, got {indices.numel()}."
            )
        if indices.unique().numel() != indices.numel():
            raise ValueError("pair_indices must be unique.")
        if bool(((indices < 0) | (indices >= self.num_q)).any()):
            raise ValueError(f"pair_indices must be in [0, {self.num_q - 1}].")
        return indices
