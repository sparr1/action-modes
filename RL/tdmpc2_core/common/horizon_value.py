"""Inner-only finite-horizon Q and V function layouts.

The outer :class:`SoftWorldModel` intentionally continues to own an ordinary
``layers.Ensemble``.  Search creates one of the modules in this file from that
ensemble, so portable outer checkpoint keys and architectures remain stable.

Every public forward takes a remaining horizon in ``[1, H]``.  A shared module
ignores the value after validating it, a depth-conditioned module appends an
``H``-way one-hot encoding, and a stage-head module selects one of ``H`` output
heads after a shared trunk.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import init, layers


HORIZON_MODES = frozenset({"shared", "depth_conditioned", "stage_heads"})


def normalize_horizon_mode(mode):
    """Validate and canonicalize a horizon-value layout name."""
    if not isinstance(mode, str):
        raise ValueError(
            f"horizon mode must be one of {sorted(HORIZON_MODES)}, got {mode!r}."
        )
    mode = mode.lower()
    if mode not in HORIZON_MODES:
        raise ValueError(
            f"horizon mode must be one of {sorted(HORIZON_MODES)}, got {mode!r}."
        )
    return mode


def _validate_horizon_count(horizon):
    if isinstance(horizon, bool) or not isinstance(horizon, Integral):
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}.")
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    return horizon


def _is_compiling():
    return bool(
        hasattr(torch, "compiler")
        and hasattr(torch.compiler, "is_compiling")
        and torch.compiler.is_compiling()
    )


def _normalize_remaining_horizon(reference, remaining_horizon, horizon):
    """Return an integer depth tensor broadcast over ``reference`` rows."""
    leading_shape = reference.shape[:-1]
    if isinstance(remaining_horizon, bool):
        raise ValueError("remaining_horizon must contain integer depths, not bools.")
    if isinstance(remaining_horizon, Integral):
        remaining_horizon = torch.tensor(
            int(remaining_horizon), device=reference.device, dtype=torch.long
        )
    elif torch.is_tensor(remaining_horizon):
        remaining_horizon = remaining_horizon.to(device=reference.device)
        if remaining_horizon.dtype == torch.bool:
            raise ValueError(
                "remaining_horizon must contain integer depths, not bools."
            )
        if remaining_horizon.is_floating_point():
            if not _is_compiling() and not bool(
                torch.eq(remaining_horizon, remaining_horizon.round()).all().item()
            ):
                raise ValueError("remaining_horizon must contain integer depths.")
        remaining_horizon = remaining_horizon.to(dtype=torch.long)
    else:
        raise TypeError(
            "remaining_horizon must be an integer or tensor, "
            f"got {type(remaining_horizon).__name__}."
        )

    if (
        remaining_horizon.ndim == len(leading_shape) + 1
        and remaining_horizon.shape[-1] == 1
    ):
        remaining_horizon = remaining_horizon.squeeze(-1)
    try:
        remaining_horizon = torch.broadcast_to(
            remaining_horizon, leading_shape
        )
    except RuntimeError as exc:
        raise ValueError(
            "remaining_horizon is not broadcastable to the input rows: "
            f"{tuple(remaining_horizon.shape)} vs {tuple(leading_shape)}."
        ) from exc

    # one_hot/gather also enforce the range inside compiled regions.  The
    # eager check provides a precise user-facing error without adding a graph
    # break to compiled search kernels.
    if not _is_compiling() and remaining_horizon.numel():
        invalid = (remaining_horizon < 1) | (remaining_horizon > horizon)
        if bool(invalid.any().item()):
            raise ValueError(
                f"remaining_horizon entries must lie in [1, {horizon}]."
            )
    return remaining_horizon


def _network_endpoints(network):
    if not isinstance(network, nn.Sequential) or len(network) == 0:
        raise TypeError(
            "Finite-horizon critic conversion requires non-empty Sequential "
            "ensemble members."
        )
    first, last = network[0], network[-1]
    if not isinstance(first, nn.Linear) or not isinstance(last, nn.Linear):
        raise TypeError(
            "Finite-horizon critic conversion requires Linear-compatible first "
            "and final layers."
        )
    return first, last


def _widen_first_layer(network, extra_inputs):
    """Clone ``network`` and append exactly-zero columns to its first layer."""
    converted = deepcopy(network)
    first, _ = _network_endpoints(converted)
    old_weight = first.weight.detach()
    widened = old_weight.new_zeros(old_weight.shape[0], old_weight.shape[1] + extra_inputs)
    widened[:, : old_weight.shape[1]].copy_(old_weight)
    first.weight = nn.Parameter(widened, requires_grad=first.weight.requires_grad)
    first.in_features += int(extra_inputs)
    return converted


class _SharedNetwork(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = deepcopy(network)

    def forward(self, value_input, remaining_horizon):
        del remaining_horizon
        return self.network(value_input)


class _DepthConditionedNetwork(nn.Module):
    def __init__(self, network, horizon):
        super().__init__()
        self.horizon = int(horizon)
        self.network = _widen_first_layer(network, self.horizon)

    def forward(self, value_input, remaining_horizon):
        depth = F.one_hot(
            remaining_horizon - 1, num_classes=self.horizon
        ).to(dtype=value_input.dtype)
        return self.network(torch.cat((value_input, depth), dim=-1))


class _StageHeadNetwork(nn.Module):
    def __init__(self, network, horizon):
        super().__init__()
        _, last = _network_endpoints(network)
        children = list(network.children())
        self.trunk = nn.Sequential(*deepcopy(children[:-1]))
        self.heads = nn.ModuleList([deepcopy(last) for _ in range(int(horizon))])
        self.horizon = int(horizon)

    def forward(self, value_input, remaining_horizon):
        features = self.trunk(value_input)
        leading_shape = features.shape[:-1]
        flat_features = features.reshape(-1, features.shape[-1])
        flat_depth = remaining_horizon.reshape(-1)
        all_outputs = torch.stack(
            [head(flat_features) for head in self.heads], dim=1
        )
        selected = all_outputs.gather(
            1,
            (flat_depth - 1)
            .reshape(-1, 1, 1)
            .expand(-1, 1, all_outputs.shape[-1]),
        ).squeeze(1)
        return selected.reshape(*leading_shape, selected.shape[-1])

    def parameters_for_stage(self, remaining_horizon):
        remaining_horizon = int(remaining_horizon)
        if not 1 <= remaining_horizon <= self.horizon:
            raise ValueError(
                f"remaining_horizon must lie in [1, {self.horizon}]."
            )
        return tuple(self.trunk.parameters()) + tuple(
            self.heads[remaining_horizon - 1].parameters()
        )


def _make_member(network, horizon, mode):
    if mode == "shared":
        return _SharedNetwork(network)
    if mode == "depth_conditioned":
        return _DepthConditionedNetwork(network, horizon)
    return _StageHeadNetwork(network, horizon)


def _check_matching_layout(source, target):
    if type(source) is not type(target):
        raise ValueError(
            "Horizon-value source and target must have the same concrete type."
        )
    if source.mode != target.mode or source.horizon != target.horizon:
        raise ValueError(
            "Horizon-value source and target layouts do not match: "
            f"{source.mode}/H={source.horizon} vs "
            f"{target.mode}/H={target.horizon}."
        )


@torch.no_grad()
def _update_tensors(source_tensors, target_tensors, tau):
    source_tensors = tuple(source_tensors)
    target_tensors = tuple(target_tensors)
    if len(source_tensors) != len(target_tensors):
        raise ValueError("Horizon-value source and target tensor layouts differ.")
    for source, target in zip(source_tensors, target_tensors):
        if source.shape != target.shape:
            raise ValueError("Horizon-value source and target tensor shapes differ.")
    if not source_tensors:
        return
    if tau == 1.0:
        torch._foreach_copy_(target_tensors, [value.detach() for value in source_tensors])
    elif tau != 0.0:
        torch._foreach_lerp_(
            target_tensors, [value.detach() for value in source_tensors], tau
        )


@torch.no_grad()
def _copy_buffers(source_buffers, target_buffers):
    source_buffers = tuple(source_buffers)
    target_buffers = tuple(target_buffers)
    if len(source_buffers) != len(target_buffers):
        raise ValueError("Horizon-value source and target buffer layouts differ.")
    for source, target in zip(source_buffers, target_buffers):
        if source.shape != target.shape:
            raise ValueError("Horizon-value source and target buffer shapes differ.")
        target.copy_(source)


class _HorizonModuleBase(nn.Module):
    """Synchronization and depth-selection behavior shared by Q and V."""

    def __init__(self, horizon, mode):
        super().__init__()
        self.horizon = _validate_horizon_count(horizon)
        self.mode = normalize_horizon_mode(mode)

    def _depth(self, value_input, remaining_horizon):
        if not torch.is_tensor(value_input) or value_input.ndim == 0:
            raise ValueError("value input must be a tensor with a feature dimension.")
        return _normalize_remaining_horizon(
            value_input, remaining_horizon, self.horizon
        )

    def make_target(self):
        """Return an exact frozen copy suitable for target bootstrapping."""
        target = deepcopy(self).requires_grad_(False)
        target.train(False)
        return target

    def parameters_for_stage(self, remaining_horizon):
        """Return parameters updated by one depth stage.

        Shared and one-hot layouts share every parameter across depths.  A
        stage-head layout returns its shared trunk and only the selected head.
        """
        remaining_horizon = int(remaining_horizon)
        if not 1 <= remaining_horizon <= self.horizon:
            raise ValueError(
                f"remaining_horizon must lie in [1, {self.horizon}]."
            )
        if self.mode != "stage_heads":
            return tuple(self.parameters())
        return self._stage_parameters(remaining_horizon)

    @torch.no_grad()
    def update_from(self, source, *, tau=1.0, remaining_horizon=None):
        """Hard/Polyak update all parameters or one depth stage from ``source``.

        For stage heads a stage update synchronizes the shared trunk and the
        selected output head while retaining all other target heads.  Because
        depth-conditioned parameters are fully shared, selecting one of their
        stages necessarily updates the complete module.
        """
        _check_matching_layout(source, self)
        tau = float(tau)
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must lie in [0, 1], got {tau}.")
        if remaining_horizon is None or self.mode != "stage_heads":
            _update_tensors(source.parameters(), self.parameters(), tau)
            _copy_buffers(source.buffers(), self.buffers())
            return self

        remaining_horizon = int(remaining_horizon)
        if not 1 <= remaining_horizon <= self.horizon:
            raise ValueError(
                f"remaining_horizon must lie in [1, {self.horizon}]."
            )
        self._update_stage_from(source, tau, remaining_horizon)
        return self

    def hard_update_from(self, source, *, remaining_horizon=None):
        return self.update_from(
            source, tau=1.0, remaining_horizon=remaining_horizon
        )

    def polyak_update_from(self, source, tau, *, remaining_horizon=None):
        return self.update_from(
            source, tau=tau, remaining_horizon=remaining_horizon
        )


class HorizonQEnsemble(_HorizonModuleBase):
    """An inner Q ensemble indexed by remaining search horizon."""

    def __init__(self, outer_qs, horizon, mode="shared"):
        super().__init__(horizon, mode)
        if not isinstance(outer_qs, layers.Ensemble) or len(outer_qs) == 0:
            raise TypeError("outer_qs must be a non-empty layers.Ensemble.")
        # The shared approximation is deliberately a direct ensemble clone.
        # Besides matching RL Search's projection of every Q_h into one
        # function, this retains compatibility with the existing Q/LoRA tools.
        self.ensemble = (
            deepcopy(outer_qs)
            if self.mode == "shared"
            else layers.Ensemble(
                [
                    _make_member(network, self.horizon, self.mode)
                    for network in outer_qs
                ]
            )
        )
        # A deepcopy of the shared outer ensemble intentionally preserves
        # model weights, but its compile-enabled/failed flags are process-local
        # operational state. Search explicitly enables its online and target
        # critics after construction, so neither may inherit an outer
        # ensemble's wrappers, failure state, or strictness policy.
        self.ensemble.disable_compile(reset_failure=True)
        self.output_dim = int(_network_endpoints(outer_qs[0])[1].out_features)
        self.num_q = len(outer_qs)
        for network in outer_qs:
            if int(_network_endpoints(network)[1].out_features) != self.output_dim:
                raise ValueError("All outer critic members must share an output size.")

    def __len__(self):
        return self.num_q

    def __iter__(self):
        return iter(self.ensemble)

    def __getitem__(self, index):
        return self.ensemble[index]

    def forward(self, q_input, remaining_horizon):
        depth = self._depth(q_input, remaining_horizon)
        if self.mode == "shared":
            return self.ensemble(q_input)
        return self.ensemble(q_input, depth)

    def forward_detached(self, q_input, remaining_horizon):
        """Freeze Q parameters while retaining gradients into ``q_input``."""
        depth = self._depth(q_input, remaining_horizon)
        if self.mode == "shared":
            return self.ensemble.forward_detached(q_input)
        return self.ensemble.forward_detached(q_input, depth)

    def enable_compile(self, *, strict=False):
        self.ensemble.enable_compile(strict=strict)
        return self

    @property
    def compile_failed(self):
        return self.ensemble.compile_failed

    def reset_from_outer(self, outer_qs):
        """Restore the transformed critic from ``outer_qs`` in-place.

        Root-local search reuses this object so Dynamo can keep the graph it
        compiled for the previous real action.  Constructing a replacement
        transformed critic here would both allocate another potentially large
        Humanoid Q ensemble and obscure the requirement that every depth is
        reset to the *current* outer prior.  Copy the three layouts directly
        while preserving every parameter and module identity.
        """
        if not isinstance(outer_qs, layers.Ensemble) or len(outer_qs) != self.num_q:
            raise ValueError("Outer critic ensemble does not match this horizon Q.")
        for outer_network in outer_qs:
            if int(_network_endpoints(outer_network)[1].out_features) != self.output_dim:
                raise ValueError("Outer critic output size does not match horizon Q.")

        with torch.no_grad():
            if self.mode == "shared":
                self.ensemble.load_state_dict(outer_qs.state_dict(), strict=True)
                return self

            for outer_network, member in zip(outer_qs, self.ensemble):
                outer_children = tuple(outer_network.children())
                if self.mode == "depth_conditioned":
                    converted_children = tuple(member.network.children())
                    if len(outer_children) != len(converted_children):
                        raise ValueError("Outer and depth-conditioned Q layouts differ.")
                    outer_first = outer_children[0]
                    converted_first = converted_children[0]
                    if (
                        converted_first.weight.shape[0] != outer_first.weight.shape[0]
                        or converted_first.weight.shape[1]
                        != outer_first.weight.shape[1] + self.horizon
                    ):
                        raise ValueError("Depth-conditioned Q input layout is incompatible.")
                    converted_first.weight[:, : outer_first.weight.shape[1]].copy_(
                        outer_first.weight
                    )
                    converted_first.weight[:, outer_first.weight.shape[1] :].zero_()
                    if (converted_first.bias is None) != (outer_first.bias is None):
                        raise ValueError("Depth-conditioned Q bias layout is incompatible.")
                    if outer_first.bias is not None:
                        converted_first.bias.copy_(outer_first.bias)
                    # NormedLinear subclasses Linear and owns trainable
                    # LayerNorm tensors in addition to weight/bias.  Those
                    # must be restored too after inner adaptation.
                    outer_first_state = outer_first.state_dict()
                    converted_first_state = converted_first.state_dict()
                    for name, value in outer_first_state.items():
                        if name not in {"weight", "bias"}:
                            converted_first_state[name].copy_(value)
                    for source, target in zip(
                        outer_children[1:], converted_children[1:]
                    ):
                        target.load_state_dict(source.state_dict(), strict=True)
                    continue

                trunk_children = tuple(member.trunk.children())
                if len(trunk_children) != len(outer_children) - 1:
                    raise ValueError("Outer and stage-head Q trunk layouts differ.")
                for source, target in zip(outer_children[:-1], trunk_children):
                    target.load_state_dict(source.state_dict(), strict=True)
                for head in member.heads:
                    head.load_state_dict(outer_children[-1].state_dict(), strict=True)
        return self

    def _stage_parameters(self, remaining_horizon):
        parameters = []
        for member in self.ensemble:
            parameters.extend(member.parameters_for_stage(remaining_horizon))
        return tuple(parameters)

    @torch.no_grad()
    def _update_stage_from(self, source, tau, remaining_horizon):
        for source_member, target_member in zip(source.ensemble, self.ensemble):
            _update_tensors(
                source_member.trunk.parameters(),
                target_member.trunk.parameters(),
                tau,
            )
            _copy_buffers(
                source_member.trunk.buffers(), target_member.trunk.buffers()
            )
            source_head = source_member.heads[remaining_horizon - 1]
            target_head = target_member.heads[remaining_horizon - 1]
            _update_tensors(source_head.parameters(), target_head.parameters(), tau)
            _copy_buffers(source_head.buffers(), target_head.buffers())


class HorizonValue(_HorizonModuleBase):
    """A scalar inner state-value function indexed by remaining horizon."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        horizon,
        mode="shared",
        *,
        dropout=0.0,
    ):
        super().__init__(horizon, mode)
        input_dim = int(input_dim)
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        self.input_dim = input_dim
        self.hidden_dims = (
            (int(hidden_dims),)
            if isinstance(hidden_dims, Integral) and not isinstance(hidden_dims, bool)
            else tuple(int(width) for width in hidden_dims)
        )
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")
        self.dropout = float(dropout)
        base = layers.mlp(
            self.input_dim, list(self.hidden_dims), 1, dropout=self.dropout
        )
        base.apply(init.weight_init)
        self.value = _make_member(base, self.horizon, self.mode)

    def forward(self, latent, remaining_horizon):
        depth = self._depth(latent, remaining_horizon)
        return self.value(latent, depth)

    def forward_detached(self, latent, remaining_horizon):
        """Freeze V parameters while retaining gradients into ``latent``."""
        depth = self._depth(latent, remaining_horizon)
        return layers.detached_module_forward(self.value, latent, depth)

    def reset_parameters(self):
        """Reinitialize online V parameters while preserving object identity."""
        replacement = type(self)(
            self.input_dim,
            self.hidden_dims,
            self.horizon,
            self.mode,
            dropout=self.dropout,
        )
        self.load_state_dict(replacement.state_dict(), strict=True)
        return self

    def reset_from(self, source):
        """Hard-copy a compatible V function, preserving optimizer references."""
        return self.hard_update_from(source)

    def _stage_parameters(self, remaining_horizon):
        return self.value.parameters_for_stage(remaining_horizon)

    @torch.no_grad()
    def _update_stage_from(self, source, tau, remaining_horizon):
        _update_tensors(
            source.value.trunk.parameters(), self.value.trunk.parameters(), tau
        )
        _copy_buffers(source.value.trunk.buffers(), self.value.trunk.buffers())
        source_head = source.value.heads[remaining_horizon - 1]
        target_head = self.value.heads[remaining_horizon - 1]
        _update_tensors(source_head.parameters(), target_head.parameters(), tau)
        _copy_buffers(source_head.buffers(), target_head.buffers())


def build_horizon_q(outer_qs, horizon, mode="shared"):
    """Build a finite-horizon inner critic from an outer Q ensemble."""
    return HorizonQEnsemble(outer_qs, horizon, mode)


def build_horizon_value(
    input_dim, hidden_dims, horizon, mode="shared", *, dropout=0.0
):
    """Build a scalar finite-horizon V function for V-trace search."""
    return HorizonValue(
        input_dim,
        hidden_dims,
        horizon,
        mode,
        dropout=dropout,
    )
