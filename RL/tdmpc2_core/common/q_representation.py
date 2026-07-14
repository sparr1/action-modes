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

    def decode(self, predictions):
        """Decode every critic head to a scalar Q expectation."""
        self._validate_predictions(predictions)
        if self.representation == "scalar":
            return predictions

        bins = torch.linspace(
            self.vmin,
            self.vmax,
            self.num_bins,
            device=predictions.device,
            dtype=predictions.dtype,
        )
        symlog_value = (F.softmax(predictions, dim=-1) * bins).sum(
            dim=-1, keepdim=True
        )
        return _symexp(symlog_value)

    def loss(self, predictions, scalar_target, *, reduction="mean"):
        """Compute a per-head scalar or categorical critic loss."""
        self._validate_predictions(predictions)
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
            encoded_target = self.encode_target(scalar_target)
            losses = -(
                encoded_target * F.log_softmax(predictions, dim=-1)
            ).sum(dim=-1, keepdim=True)

        if reduction == "none":
            return losses
        if reduction == "mean":
            return losses.mean()
        if reduction == "sum":
            return losses.sum()
        raise ValueError(f"Unknown critic loss reduction: {reduction!r}.")

    def reduce(self, values, reduction, *, pair_indices=None, generator=None):
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
            selected = values.index_select(
                0,
                self._pair_indices(
                    values.device,
                    pair_indices=pair_indices,
                    generator=generator,
                ),
            )

        if reduction.startswith("min_"):
            return selected.min(dim=0).values
        return selected.mean(dim=0)

    def _pair_indices(self, device, *, pair_indices=None, generator=None):
        if pair_indices is None:
            if self.pair_size == self.num_q:
                return torch.arange(self.num_q, device=device)
            return torch.randperm(
                self.num_q,
                device=device,
                generator=generator,
            )[: self.pair_size]

        indices = torch.as_tensor(pair_indices, device=device, dtype=torch.long).flatten()
        if indices.numel() != self.pair_size:
            raise ValueError(
                f"Expected {self.pair_size} pair indices, got {indices.numel()}."
            )
        if indices.unique().numel() != indices.numel():
            raise ValueError("pair_indices must be unique.")
        if bool(((indices < 0) | (indices >= self.num_q)).any()):
            raise ValueError(f"pair_indices must be in [0, {self.num_q - 1}].")
        return indices
