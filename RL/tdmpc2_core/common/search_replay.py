"""Trajectory-safe replay for finite-horizon AMBI search.

The legacy :mod:`latent_buffer` stores independent transitions and is kept
unchanged.  Search estimators need linked suffixes, so this buffer allocates a
fixed ``[trajectory, time, feature]`` tensor for every field and advances its
ring cursor only in whole-trajectory units.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


_FLOAT_FIELDS = (
    "z",
    "action",
    "pre_tanh_action",
    "reward",
    "next_z",
    "behavior_log_prob",
)
_BOOL_FIELDS = ("terminated", "valid")
_LONG_FIELDS = ("round_id", "remaining_horizon")
_ALL_FIELDS = _FLOAT_FIELDS + _BOOL_FIELDS + _LONG_FIELDS


class SearchTrajectoryReplayBuffer:
    """A fixed-horizon ring whose indivisible storage unit is a trajectory.

    Parameters
    ----------
    capacity:
        Capacity in *transitions*.  It must be an exact multiple of ``horizon``
        so no partial trajectory can be represented or evicted.
    horizon:
        Number of allocated time slots per trajectory.  Early termination is
        represented by ``valid=False`` in the unused suffix.
    """

    schema = "ambi-search-trajectory-replay-training-state"
    schema_version = 2

    def __init__(
        self,
        capacity,
        horizon,
        latent_dim,
        action_dim,
        device,
        *,
        dtype=torch.float32,
    ):
        self.capacity = int(capacity)
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        if self.capacity <= 0:
            raise ValueError("Search replay capacity must be positive.")
        if self.horizon <= 0:
            raise ValueError("Search replay horizon must be positive.")
        if self.capacity % self.horizon:
            raise ValueError(
                "Search replay capacity must be a multiple of the horizon so "
                "the ring never stores a partial trajectory."
            )
        if self.latent_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Latent and action dimensions must be positive.")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("Search replay floating-point dtype must be floating.")

        self.trajectory_capacity = self.capacity // self.horizon
        shape = (self.trajectory_capacity, self.horizon)
        self.z = torch.empty(*shape, self.latent_dim, device=self.device, dtype=dtype)
        self.action = torch.empty(
            *shape, self.action_dim, device=self.device, dtype=dtype
        )
        self.pre_tanh_action = torch.empty_like(self.action)
        self.reward = torch.empty(*shape, 1, device=self.device, dtype=dtype)
        self.next_z = torch.empty_like(self.z)
        self.terminated = torch.empty(*shape, 1, device=self.device, dtype=torch.bool)
        self.valid = torch.empty_like(self.terminated)
        self.behavior_log_prob = torch.empty(
            *shape, 1, device=self.device, dtype=dtype
        )
        self.round_id = torch.empty(*shape, 1, device=self.device, dtype=torch.long)
        self.remaining_horizon = torch.empty_like(self.round_id)

        self.pos = 0
        self.full = False
        # ``next_trajectory_id`` is local to the current clear-to-clear
        # generation, which lets ``size`` and the physical ring layout retain
        # their simple relationship.  The offset makes externally reported
        # IDs monotonic for the buffer lifetime.  That distinction matters for
        # round-local replay: samples collected before and after ``clear()``
        # must not alias in aggregate replay-coverage diagnostics.
        self.trajectory_id_offset = 0
        self.next_trajectory_id = 0

    @property
    def size(self):
        """Number of complete trajectories currently stored."""
        return self.trajectory_capacity if self.full else self.pos

    @property
    def transition_size(self):
        """Number of valid transitions, excluding padded suffix slots."""
        if self.size == 0:
            return 0
        return int(self.valid[: self.size].sum().item())

    @property
    def trajectory_id(self):
        """Monotonic IDs associated with physical trajectory slots."""
        ids = torch.empty(
            self.trajectory_capacity, dtype=torch.long, device=self.device
        )
        if self.size:
            indices = torch.arange(self.size, dtype=torch.long, device=self.device)
            ids[: self.size] = self._trajectory_ids(indices)
        return ids

    def clear(self, *, reset_identity=False):
        """Discard all trajectories without reallocating device storage.

        By default, future trajectory IDs continue after every ID allocated
        before the clear.  This keeps sampled identities unique across the
        fresh-round replay phases of one root solve.  ``reset_identity=True``
        is reserved for loading an explicitly empty training state.
        """
        if reset_identity:
            self.trajectory_id_offset = 0
        else:
            self.trajectory_id_offset += self.next_trajectory_id
        self.pos = 0
        self.full = False
        self.next_trajectory_id = 0

    def _trajectory_ids(self, indices):
        oldest_id = self.next_trajectory_id - self.size
        if self.full:
            logical_offset = torch.remainder(
                indices - self.pos, self.trajectory_capacity
            )
        else:
            logical_offset = indices
        return logical_offset + oldest_id + self.trajectory_id_offset

    def _float_field(self, name, value, count, width):
        if not torch.is_tensor(value):
            raise TypeError(f"Search replay field {name!r} must be a tensor.")
        value = value.detach()
        expected = (count, self.horizon, width)
        if value.shape != expected:
            raise ValueError(
                f"Search replay field {name!r} must have shape {expected}, "
                f"got {tuple(value.shape)}."
            )
        return value.to(device=self.device, dtype=self.dtype)

    def _mask_field(self, name, value, count):
        if not torch.is_tensor(value):
            raise TypeError(f"Search replay field {name!r} must be a tensor.")
        value = value.detach()
        if value.shape == (count, self.horizon):
            value = value.unsqueeze(-1)
        expected = (count, self.horizon, 1)
        if value.shape != expected:
            raise ValueError(
                f"Search replay field {name!r} must have shape {expected}, "
                f"got {tuple(value.shape)}."
            )
        if value.dtype is not torch.bool:
            if not bool(torch.logical_or(value == 0, value == 1).all().item()):
                raise ValueError(f"Search replay field {name!r} must be binary.")
        return value.to(device=self.device, dtype=torch.bool)

    def _round_ids(self, value, count):
        if isinstance(value, bool):
            raise TypeError("round_id must be an integer or integer tensor.")
        if isinstance(value, int):
            return torch.full(
                (count, self.horizon, 1),
                value,
                device=self.device,
                dtype=torch.long,
            )
        if not torch.is_tensor(value):
            raise TypeError("round_id must be an integer or integer tensor.")
        value = value.detach()
        if value.numel() == 1:
            value = value.reshape(1, 1, 1).expand(count, self.horizon, 1)
        elif value.shape == (count,):
            value = value[:, None, None].expand(count, self.horizon, 1)
        elif value.shape == (count, 1):
            value = value[:, None, :].expand(count, self.horizon, 1)
        elif value.shape == (count, self.horizon):
            value = value.unsqueeze(-1)
        if value.shape != (count, self.horizon, 1):
            raise ValueError("round_id has an incompatible shape.")
        if value.is_floating_point() or value.dtype is torch.bool:
            raise TypeError("round_id must have an integer dtype.")
        value = value.to(device=self.device, dtype=torch.long)
        if not bool((value == value[:, :1]).all().item()):
            raise ValueError("round_id must be constant within each trajectory.")
        return value

    def _remaining_horizons(self, value, count):
        canonical = torch.arange(
            self.horizon,
            0,
            -1,
            device=self.device,
            dtype=torch.long,
        ).reshape(1, self.horizon, 1).expand(count, -1, -1)
        if value is None:
            return canonical
        if not torch.is_tensor(value):
            raise TypeError("remaining_horizon must be a tensor when provided.")
        value = value.detach()
        if value.shape == (count, self.horizon):
            value = value.unsqueeze(-1)
        if value.shape != (count, self.horizon, 1):
            raise ValueError("remaining_horizon has an incompatible shape.")
        if value.is_floating_point() or value.dtype is torch.bool:
            raise TypeError("remaining_horizon must have an integer dtype.")
        value = value.to(device=self.device, dtype=torch.long)
        if not torch.equal(value, canonical):
            raise ValueError(
                "remaining_horizon must descend from horizon to one in every "
                "stored root trajectory."
            )
        return value

    def _prepare_batch(
        self,
        z,
        action,
        pre_tanh_action,
        reward,
        next_z,
        terminated,
        valid,
        behavior_log_prob,
        round_id,
        remaining_horizon,
    ):
        if not torch.is_tensor(z) or z.ndim != 3:
            raise TypeError("z must be a [trajectory, time, latent] tensor.")
        count = int(z.shape[0])
        if count == 0:
            return None, 0
        values = {
            "z": self._float_field("z", z, count, self.latent_dim),
            "action": self._float_field(
                "action", action, count, self.action_dim
            ),
            "pre_tanh_action": self._float_field(
                "pre_tanh_action", pre_tanh_action, count, self.action_dim
            ),
            "reward": self._float_field("reward", reward, count, 1),
            "next_z": self._float_field("next_z", next_z, count, self.latent_dim),
            "terminated": self._mask_field("terminated", terminated, count),
            "valid": self._mask_field("valid", valid, count),
            "behavior_log_prob": self._float_field(
                "behavior_log_prob", behavior_log_prob, count, 1
            ),
            "round_id": self._round_ids(round_id, count),
            "remaining_horizon": self._remaining_horizons(
                remaining_horizon, count
            ),
        }
        if bool(torch.logical_and(values["terminated"], ~values["valid"]).any()):
            raise ValueError("A padded search transition cannot be terminated.")
        # Padding must be a suffix.  This guarantees that every sampled anchor
        # has one contiguous sequence and makes fixed-shape masking unambiguous.
        valid = values["valid"].squeeze(-1)
        if self.horizon > 1 and bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("Search replay valid masks must be prefix masks.")
        terminated = values["terminated"].squeeze(-1)
        terminated_long = terminated.to(torch.long)
        terminated_before = torch.cumsum(terminated_long, dim=1) - terminated_long
        if bool(((terminated_before > 0) & valid).any()):
            raise ValueError(
                "Search replay transitions after termination must be padded invalid."
            )
        return values, count

    def add_trajectories(
        self,
        z,
        action,
        pre_tanh_action,
        reward,
        next_z,
        terminated,
        valid,
        behavior_log_prob,
        round_id,
        remaining_horizon=None,
    ):
        """Append trajectory-major tensors, evicting only complete trajectories."""
        values, original_count = self._prepare_batch(
            z,
            action,
            pre_tanh_action,
            reward,
            next_z,
            terminated,
            valid,
            behavior_log_prob,
            round_id,
            remaining_horizon,
        )
        if original_count == 0:
            return

        old_pos = self.pos
        self.next_trajectory_id += original_count
        count = original_count
        if count >= self.trajectory_capacity:
            values = {
                name: value[-self.trajectory_capacity :]
                for name, value in values.items()
            }
            count = self.trajectory_capacity
            write_pos = (old_pos + original_count) % self.trajectory_capacity
        else:
            write_pos = old_pos

        first = min(count, self.trajectory_capacity - write_pos)
        second = count - first
        for name, value in values.items():
            storage = getattr(self, name)
            storage[write_pos : write_pos + first].copy_(value[:first])
            if second:
                storage[:second].copy_(value[first:])

        self.pos = (old_pos + original_count) % self.trajectory_capacity
        self.full = self.full or (
            old_pos + original_count >= self.trajectory_capacity
        )

    def add_round(self, *, horizon_major=False, **fields):
        """Append a rollout round, optionally transposing ``[time, trajectory]``.

        ``round_id`` may be a scalar and ``remaining_horizon`` may be omitted;
        every other field is transposed over its first two dimensions when
        ``horizon_major=True``.
        """
        if horizon_major:
            converted = {}
            for name, value in fields.items():
                if name == "round_id" and not torch.is_tensor(value):
                    converted[name] = value
                elif name == "remaining_horizon" and value is None:
                    converted[name] = value
                elif torch.is_tensor(value) and value.ndim >= 2:
                    converted[name] = value.transpose(0, 1)
                else:
                    converted[name] = value
            fields = converted
        self.add_trajectories(**fields)

    def _validate_sample_request(self, batch_size, replacement):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty search replay buffer.")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Search replay batch_size must be positive.")
        if not replacement and batch_size > self.size:
            raise ValueError(
                "Cannot sample search trajectories without replacement: "
                f"batch_size={batch_size} exceeds size={self.size}."
            )
        return batch_size

    def _draw(self, population, batch_size, replacement, generator):
        if replacement:
            return torch.randint(
                population,
                (batch_size,),
                device=self.device,
                generator=generator,
            )
        return torch.randperm(
            population, device=self.device, generator=generator
        )[:batch_size]

    def _validate_indices(self, indices, batch_size, upper, owner):
        if not torch.is_tensor(indices):
            raise TypeError(f"Pre-generated {owner} indices must be a tensor.")
        if indices.shape != (batch_size,) or indices.dtype != torch.long:
            raise ValueError(
                f"Pre-generated {owner} indices must be a length-{batch_size} "
                "long tensor."
            )
        if indices.device != self.device:
            raise ValueError(f"Pre-generated {owner} indices must be on replay device.")
        if bool(torch.logical_or(indices < 0, indices >= upper).any()):
            raise ValueError(f"Pre-generated {owner} indices are out of range.")
        return indices

    def sample_trajectories(
        self,
        batch_size,
        *,
        replacement=True,
        generator=None,
        indices=None,
        include_ids=True,
    ):
        """Sample complete physical trajectory slots."""
        batch_size = self._validate_sample_request(batch_size, replacement)
        if indices is None:
            indices = self._draw(self.size, batch_size, replacement, generator)
        else:
            indices = self._validate_indices(
                indices, batch_size, self.size, "trajectory"
            )
        batch = {
            name: getattr(self, name).index_select(0, indices)
            for name in _ALL_FIELDS
        }
        if include_ids:
            batch["trajectory_indices"] = indices
            batch["trajectory_ids"] = self._trajectory_ids(indices)
        return batch

    def sample_anchors(
        self,
        batch_size,
        *,
        replacement=True,
        generator=None,
        candidate_indices=None,
        include_ids=True,
    ):
        """Sample valid transition anchors and return fixed-shape suffixes.

        Returned transition tensors all have shape ``[batch, horizon, ...]``.
        Time slots before the anchor are omitted; out-of-range suffix padding is
        zeroed and marked invalid.  ``candidate_indices`` addresses the flattened
        list returned by ``nonzero(valid)`` and permits RNG-free compiled callers.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty search replay buffer.")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Search replay batch_size must be positive.")
        candidates = torch.nonzero(
            self.valid[: self.size, :, 0], as_tuple=False
        )
        population = int(candidates.shape[0])
        if population == 0:
            raise ValueError("Cannot sample anchors when search replay has no valid rows.")
        if not replacement and batch_size > population:
            raise ValueError(
                "Cannot sample search anchors without replacement: "
                f"batch_size={batch_size} exceeds valid rows={population}."
            )
        if candidate_indices is None:
            candidate_indices = self._draw(
                population, batch_size, replacement, generator
            )
        else:
            candidate_indices = self._validate_indices(
                candidate_indices, batch_size, population, "anchor candidate"
            )
        selected = candidates.index_select(0, candidate_indices)
        trajectory_indices, anchor_time = selected[:, 0], selected[:, 1]
        offsets = torch.arange(self.horizon, device=self.device)
        time = anchor_time[:, None] + offsets[None, :]
        in_bounds = time < self.horizon
        safe_time = time.clamp(max=self.horizon - 1)
        suffix_valid = (
            self.valid[trajectory_indices[:, None], safe_time]
            & in_bounds.unsqueeze(-1)
        )

        batch = {}
        for name in _ALL_FIELDS:
            value = getattr(self, name)[trajectory_indices[:, None], safe_time]
            batch[name] = torch.where(
                suffix_valid, value, torch.zeros_like(value)
            )
        batch["valid"] = suffix_valid
        batch["anchor_time"] = anchor_time
        batch["suffix_offset"] = offsets.expand(batch_size, -1)
        if include_ids:
            batch["trajectory_indices"] = trajectory_indices
            batch["trajectory_ids"] = self._trajectory_ids(trajectory_indices)
            batch["candidate_indices"] = candidate_indices
        return batch

    # ``sample`` is intentionally the estimator-facing anchor operation.
    sample = sample_anchors

    def state_dict(self):
        """Return a device-portable physical-ring snapshot."""
        size = self.size
        indices = torch.arange(size, device=self.device, dtype=torch.long)
        state = {
            "capacity": self.capacity,
            "pos": self.pos,
            "full": self.full,
            "trajectory_id_offset": self.trajectory_id_offset,
            "next_trajectory_id": self.next_trajectory_id,
            "trajectory_id": self._trajectory_ids(indices),
        }
        state.update(
            {name: getattr(self, name)[:size].clone() for name in _ALL_FIELDS}
        )
        return state

    def training_state_dict(self):
        return {
            "schema": self.schema,
            "version": self.schema_version,
            "capacity": self.capacity,
            "horizon": self.horizon,
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "dtype": str(self.dtype),
            "state": self.state_dict(),
        }

    def _validated_state(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("Search replay state must be a mapping.")
        expected_keys = {
            "capacity",
            "pos",
            "full",
            "trajectory_id_offset",
            "next_trajectory_id",
            "trajectory_id",
            *_ALL_FIELDS,
        }
        if set(state) != expected_keys:
            missing = sorted(expected_keys - set(state))
            extra = sorted(set(state) - expected_keys)
            raise ValueError(
                f"Search replay state keys differ (missing={missing}, extra={extra})."
            )
        if state["capacity"] != self.capacity:
            raise ValueError("Search replay state capacity is incompatible.")
        pos, full = state["pos"], state["full"]
        id_offset = state["trajectory_id_offset"]
        next_id = state["next_trajectory_id"]
        if isinstance(pos, bool) or not isinstance(pos, int):
            raise TypeError("Search replay state pos must be an integer.")
        if not 0 <= pos < self.trajectory_capacity:
            raise ValueError("Search replay state pos is out of range.")
        if not isinstance(full, bool):
            raise TypeError("Search replay state full must be bool.")
        if (
            isinstance(id_offset, bool)
            or not isinstance(id_offset, int)
            or id_offset < 0
        ):
            raise ValueError("Search replay trajectory_id_offset is invalid.")
        if isinstance(next_id, bool) or not isinstance(next_id, int) or next_id < 0:
            raise ValueError("Search replay next_trajectory_id is invalid.")
        if pos != next_id % self.trajectory_capacity:
            raise ValueError("Search replay cursor is inconsistent with its IDs.")
        if full != (next_id >= self.trajectory_capacity):
            raise ValueError("Search replay full flag is inconsistent with its IDs.")
        size = min(next_id, self.trajectory_capacity)

        converted = {}
        widths = {
            "z": self.latent_dim,
            "action": self.action_dim,
            "pre_tanh_action": self.action_dim,
            "reward": 1,
            "next_z": self.latent_dim,
            "behavior_log_prob": 1,
            "terminated": 1,
            "valid": 1,
            "round_id": 1,
            "remaining_horizon": 1,
        }
        for name in _ALL_FIELDS:
            value = state[name]
            expected = (size, self.horizon, widths[name])
            if not torch.is_tensor(value) or value.shape != expected:
                raise ValueError(
                    f"Search replay state field {name!r} must have shape {expected}."
                )
            expected_dtype = (
                torch.bool
                if name in _BOOL_FIELDS
                else torch.long
                if name in _LONG_FIELDS
                else self.dtype
            )
            if value.dtype != expected_dtype:
                raise ValueError(
                    f"Search replay state field {name!r} has the wrong dtype."
                )
            converted[name] = value.detach().to(self.device)

        valid = converted["valid"].squeeze(-1)
        terminated = converted["terminated"].squeeze(-1)
        if self.horizon > 1 and bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("Search replay state valid masks are not prefixes.")
        if bool(torch.logical_and(terminated, ~valid).any()):
            raise ValueError("Search replay state terminates a padded transition.")
        terminated_long = terminated.to(torch.long)
        terminated_before = torch.cumsum(terminated_long, dim=1) - terminated_long
        if bool(((terminated_before > 0) & valid).any()):
            raise ValueError("Search replay state contains rows after termination.")
        if size and not bool(
            (converted["round_id"] == converted["round_id"][:, :1]).all()
        ):
            raise ValueError("Search replay state round IDs vary within a trajectory.")
        canonical_horizon = torch.arange(
            self.horizon,
            0,
            -1,
            device=self.device,
            dtype=torch.long,
        ).reshape(1, self.horizon, 1)
        if size and not torch.equal(
            converted["remaining_horizon"],
            canonical_horizon.expand(size, -1, -1),
        ):
            raise ValueError("Search replay state remaining horizons are invalid.")

        ids = state["trajectory_id"]
        if not torch.is_tensor(ids) or ids.shape != (size,) or ids.dtype != torch.long:
            raise ValueError("Search replay trajectory_id has the wrong shape or dtype.")
        physical = torch.arange(size, dtype=torch.long)
        oldest = next_id - size
        expected_ids = (
            torch.remainder(physical - pos, self.trajectory_capacity)
            + oldest
            + id_offset
            if full
            else physical + oldest + id_offset
        )
        if not torch.equal(ids.detach().cpu(), expected_ids):
            raise ValueError("Search replay trajectory IDs are inconsistent.")
        return converted, size, pos, full, id_offset, next_id

    def load_state_dict(self, state):
        if not state:
            self.clear(reset_identity=True)
            return self
        converted, size, pos, full, id_offset, next_id = self._validated_state(state)
        for name, value in converted.items():
            getattr(self, name)[:size].copy_(value)
        self.pos = pos
        self.full = full
        self.trajectory_id_offset = id_offset
        self.next_trajectory_id = next_id
        return self

    def load_training_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("Search replay training state must be a mapping.")
        expected = {
            "schema",
            "version",
            "capacity",
            "horizon",
            "latent_dim",
            "action_dim",
            "dtype",
            "state",
        }
        if set(state) != expected:
            raise ValueError("Search replay training-state keys are incompatible.")
        if state["schema"] != self.schema or state["version"] != self.schema_version:
            raise ValueError("Unsupported search replay training-state version.")
        signature = {
            "capacity": self.capacity,
            "horizon": self.horizon,
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "dtype": str(self.dtype),
        }
        if {name: state[name] for name in signature} != signature:
            raise ValueError("Search replay training-state configuration is incompatible.")
        return self.load_state_dict(state["state"])


# A shorter alias makes call sites readable while retaining the descriptive
# public class name in checkpoints and documentation.
SearchReplayBuffer = SearchTrajectoryReplayBuffer
