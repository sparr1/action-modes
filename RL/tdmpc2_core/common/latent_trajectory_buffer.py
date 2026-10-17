"""Device-resident whole-trajectory replay for finite-horizon inner Retrace.

The ordinary transition buffer deliberately stays separate: a trajectory is
the unit of eviction here, and padding never becomes actor training data.
"""

from numbers import Integral

import torch

from .training_state import require_exact_keys


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


class LatentTrajectoryReplayBuffer:
    """Packed ring of completed, padded trajectories with stable sample IDs.

    ``capacity`` is requested in transition slots. Its effective value is the
    largest multiple of ``horizon`` no greater than the requested capacity.
    ``size`` counts live valid transitions, whereas ``trajectory_count`` counts
    complete trajectories. Transition indices address the valid rows in
    physical-slot order; trajectory indices address physical trajectory slots.
    """

    _STATE_SCHEMA = "ambi-latent-trajectory-replay-state"
    _TRAINING_SCHEMA = "ambi-latent-trajectory-replay-training-state"
    _VERSION = 1
    _STATE_METADATA = {
        "schema", "version", "requested_capacity", "capacity", "latent_dim", "action_dim",
        "horizon", "horizon_conditioning_horizon", "pos", "full", "next_sample_id",
        "next_trajectory_id", "valid", "sample_id", "trajectory_id",
    }

    def __init__(self, capacity, latent_dim, action_dim, device, *, horizon,
                 horizon_conditioning_horizon=None):
        self.requested_capacity = _positive_integer(capacity, "capacity")
        self.horizon = _positive_integer(horizon, "horizon")
        self.latent_dim = _positive_integer(latent_dim, "latent_dim")
        self.action_dim = _positive_integer(action_dim, "action_dim")
        self.device = torch.device(device)
        self.trajectory_capacity = self.requested_capacity // self.horizon
        if self.trajectory_capacity == 0:
            raise ValueError("Trajectory replay capacity must accommodate one full horizon.")
        self.capacity = self.trajectory_capacity * self.horizon
        if horizon_conditioning_horizon is not None:
            horizon_conditioning_horizon = _positive_integer(
                horizon_conditioning_horizon, "horizon_conditioning_horizon"
            )
            if horizon_conditioning_horizon != self.horizon:
                raise ValueError("Horizon conditioning must match the trajectory horizon.")
        self.horizon_conditioning_horizon = horizon_conditioning_horizon
        self.store_horizon = True
        self.store_source = False
        self.source = None

        widths = (
            ("z", self.latent_dim), ("action", self.action_dim),
            ("reward", 1), ("next_z", self.latent_dim), ("terminated", 1),
            ("horizon_end", 1), ("pre_tanh_action", self.action_dim),
            ("behavior_log_prob", 1),
        )
        if horizon_conditioning_horizon is not None:
            widths += (("remaining_horizon", 1),)
        self._field_slices = {}
        offset = 0
        for name, width in widths:
            self._field_slices[name] = slice(offset, offset + width)
            offset += width
        shape = (self.trajectory_capacity, self.horizon)
        self._storage = torch.zeros(*shape, offset, device=self.device)
        for name, field_slice in self._field_slices.items():
            setattr(self, name, self._storage[..., field_slice])
        if horizon_conditioning_horizon is None:
            self.remaining_horizon = None
        self.valid = torch.zeros(*shape, 1, dtype=torch.bool, device=self.device)
        self.sample_id = torch.full(shape, -1, dtype=torch.long, device=self.device)
        self.trajectory_id = torch.full(
            (self.trajectory_capacity,), -1, dtype=torch.long, device=self.device,
        )
        self.clear()

    @property
    def size(self):
        return self._size

    @property
    def trajectory_count(self):
        return self.trajectory_capacity if self.full else self.pos

    def clear(self):
        """Reset scientific contents and identities without reallocating storage."""
        self.pos = 0
        self.full = False
        self.next_sample_id = 0
        self.next_trajectory_id = 0
        self._size = 0
        self._lengths = [0] * self.trajectory_capacity
        self._valid_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.valid.zero_()
        self.sample_id.fill_(-1)
        self.trajectory_id.fill_(-1)

    def _prepare_batch(self, batch, *, validate, exact_dtype=False):
        keys = set(self._field_slices) | {"valid"}
        batch = require_exact_keys(batch, keys, "trajectory replay batch")
        valid = batch["valid"]
        if (not torch.is_tensor(valid) or valid.ndim != 3
                or tuple(valid.shape[1:]) != (self.horizon, 1)
                or valid.dtype != torch.bool):
            raise ValueError("Trajectory valid must be bool with shape [N, H, 1].")
        n = int(valid.shape[0])
        valid = valid.detach().to(self.device)
        fields = {}
        for name, field_slice in self._field_slices.items():
            value = batch[name]
            width = field_slice.stop - field_slice.start
            if (not torch.is_tensor(value)
                    or tuple(value.shape) != (n, self.horizon, width)):
                raise ValueError(f"Trajectory field {name!r} must have shape [N, H, {width}].")
            integer_horizon = name == "remaining_horizon" and value.dtype in (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            )
            if not value.is_floating_point() and not integer_horizon:
                raise TypeError(f"Trajectory field {name!r} must be floating point.")
            if exact_dtype and value.dtype != self._storage.dtype:
                raise ValueError(f"Trajectory field {name!r} has the wrong dtype.")
            fields[name] = value.detach().to(device=self.device, dtype=self._storage.dtype)
        lengths_tensor = valid.squeeze(-1).sum(dim=1)
        # Collection is eager. One transfer maintains Python ring counters;
        # trusted generated batches skip all subsequent value-check syncs.
        lengths = lengths_tensor.cpu().tolist()
        if validate and n:
            steps = torch.arange(self.horizon, device=self.device).view(1, -1, 1)
            expected_valid = steps < lengths_tensor.view(-1, 1, 1)
            if not all(length > 0 for length in lengths) or not torch.equal(valid, expected_valid):
                raise ValueError("Trajectory valid masks must be nonempty contiguous prefixes.")
            for name, value in fields.items():
                if not bool(torch.isfinite(value[valid.expand_as(value)]).all().item()):
                    raise ValueError(f"Valid trajectory field {name!r} must be finite.")
            terminal = fields["terminated"]
            if not bool(((terminal == 0) | (terminal == 1) | ~valid).all().item()):
                raise ValueError("Trajectory terminated flags must be binary.")
            last = (steps == lengths_tensor.view(-1, 1, 1) - 1) & valid
            if bool(((terminal != 0) & valid & ~last).any().item()):
                raise ValueError("A trajectory may terminate only at its last valid step.")
            short_last = last & (lengths_tensor.view(-1, 1, 1) < self.horizon)
            if bool(((terminal != 1) & short_last).any().item()):
                raise ValueError("A trajectory shorter than H must end in termination.")
            boundary = fields["horizon_end"]
            expected_boundary = (steps == self.horizon - 1) & valid
            if not torch.equal(boundary[valid], expected_boundary[valid].to(boundary.dtype)):
                raise ValueError("Trajectory horizon_end must mark exactly the valid H boundary.")
            if self.horizon_conditioning_horizon is not None:
                remaining = fields["remaining_horizon"]
                expected = (self.horizon - steps).expand(n, -1, -1)
                if not torch.equal(remaining[valid], expected[valid].to(remaining.dtype)):
                    raise ValueError("Trajectory remaining_horizon must decrease from H by one.")
            action = fields["action"]
            unsquashed = fields["pre_tanh_action"]
            action_valid = valid.expand_as(action)
            if not torch.allclose(action[action_valid], unsquashed.tanh()[action_valid],
                                  atol=1e-6, rtol=1e-5):
                raise ValueError("Trajectory action must equal tanh(pre_tanh_action).")

        # Neutralize even nonfinite caller padding before it can reach a model.
        # A conditioned network has no h=0 slot, so its padding uses h=1.
        fields = {
            name: torch.where(valid, value, 1.0 if name == "remaining_horizon" else 0.0)
            for name, value in fields.items()
        }
        return torch.cat(tuple(fields.values()), dim=-1), valid, lengths

    def _refresh_valid_indices(self):
        self._size = sum(self._lengths[:self.trajectory_count])
        self._valid_indices = torch.nonzero(
            self.valid[:self.trajectory_count].reshape(-1), as_tuple=False,
        ).squeeze(-1)

    def add_trajectories(self, batch, *, validate=True):
        """Append completed trajectories atomically, preserving whole suffixes.

        ``validate=False`` is reserved for trusted collectors that generate the
        masks and horizon metadata. Tensor/shape checks still run in that mode.
        """
        if not isinstance(validate, bool):
            raise TypeError("validate must be bool.")
        packed, valid, lengths = self._prepare_batch(batch, validate=validate)
        n = len(lengths)
        if n == 0:
            return
        total_valid = sum(lengths)
        trajectory_ids = torch.arange(
            self.next_trajectory_id, self.next_trajectory_id + n,
            dtype=torch.long, device=self.device,
        )
        ids = valid.squeeze(-1).to(torch.long).reshape(-1).cumsum(0).reshape(n, self.horizon)
        ids = torch.where(valid.squeeze(-1), ids + self.next_sample_id - 1, -1)
        original_n = n
        old_pos = self.pos
        if n >= self.trajectory_capacity:
            packed, valid, ids, trajectory_ids = (
                value[-self.trajectory_capacity:]
                for value in (packed, valid, ids, trajectory_ids)
            )
            lengths = lengths[-self.trajectory_capacity:]
            n = self.trajectory_capacity
            write_pos = (old_pos + original_n) % self.trajectory_capacity
        else:
            write_pos = old_pos
        first = min(n, self.trajectory_capacity - write_pos)
        for destination, source in (
            (self._storage, packed), (self.valid, valid),
            (self.sample_id, ids), (self.trajectory_id, trajectory_ids),
        ):
            destination[write_pos:write_pos + first].copy_(source[:first])
            if n > first:
                destination[:n - first].copy_(source[first:])
        self._lengths[write_pos:write_pos + first] = lengths[:first]
        if n > first:
            self._lengths[:n - first] = lengths[first:]
        self.pos = (old_pos + original_n) % self.trajectory_capacity
        self.full = self.full or old_pos + original_n >= self.trajectory_capacity
        self.next_trajectory_id += original_n
        self.next_sample_id += total_valid
        self._refresh_valid_indices()

    def _indices(self, size, batch_size, *, replacement, generator, indices):
        batch_size = _positive_integer(batch_size, "batch_size")
        if size == 0:
            raise ValueError("Cannot sample from an empty trajectory replay buffer.")
        if not isinstance(replacement, bool):
            raise TypeError("replacement must be bool.")
        if not replacement and batch_size > size:
            raise ValueError("Cannot sample trajectory replay without replacement beyond its size.")
        if indices is None:
            if replacement:
                return torch.randint(size, (batch_size,), device=self.device, generator=generator)
            return torch.randperm(size, device=self.device, generator=generator)[:batch_size]
        if not torch.is_tensor(indices):
            raise TypeError("Pre-generated replay indices must be a tensor.")
        if indices.dtype != torch.long:
            raise TypeError("Pre-generated replay indices must have dtype long.")
        if indices.ndim != 1 or indices.numel() != batch_size:
            raise ValueError("Pre-generated replay indices must have shape [batch_size].")
        if indices.device != self._storage.device:
            raise ValueError("Pre-generated replay indices must be on the replay device.")
        # index_select below checks bounds without an extra explicit device sync.
        return indices

    def sample_trajectories(self, batch_size, *, replacement=True, generator=None,
                            indices=None, include_ids=False):
        """Sample full padded trajectories; explicit indices consume no RNG."""
        indices = self._indices(self.trajectory_count, batch_size,
                                replacement=replacement, generator=generator, indices=indices)
        packed = self._storage[:self.trajectory_count].index_select(0, indices)
        batch = {name: packed[..., field_slice] for name, field_slice in self._field_slices.items()}
        batch["valid"] = self.valid[:self.trajectory_count].index_select(0, indices)
        if include_ids:
            batch["indices"] = indices
            batch["sample_ids"] = self.sample_id[:self.trajectory_count].index_select(0, indices)
            batch["trajectory_ids"] = self.trajectory_id[:self.trajectory_count].index_select(0, indices)
        return batch

    def sample(self, batch_size, *, replacement=True, generator=None,
               include_ids=True, indices=None):
        """Uniformly sample valid transitions for unchanged actor objectives."""
        indices = self._indices(self.size, batch_size, replacement=replacement,
                                generator=generator, indices=indices)
        physical = self._valid_indices.index_select(0, indices)
        packed = self._storage.flatten(0, 1).index_select(0, physical)
        batch = {name: packed[:, field_slice] for name, field_slice in self._field_slices.items()}
        if include_ids:
            batch["indices"] = indices
            batch["sample_ids"] = self.sample_id.reshape(-1).index_select(0, physical)
            batch["trajectory_ids"] = self.trajectory_id.index_select(
                0, torch.div(physical, self.horizon, rounding_mode="floor"),
            )
        return batch

    def state_dict(self):
        """Return a detached physical-ring snapshot, excluding unused slots."""
        count = self.trajectory_count
        state = {
            "schema": self._STATE_SCHEMA, "version": self._VERSION,
            "requested_capacity": self.requested_capacity, "capacity": self.capacity,
            "latent_dim": self.latent_dim, "action_dim": self.action_dim,
            "horizon": self.horizon,
            "horizon_conditioning_horizon": self.horizon_conditioning_horizon,
            "pos": self.pos, "full": self.full,
            "next_sample_id": self.next_sample_id,
            "next_trajectory_id": self.next_trajectory_id,
            "valid": self.valid[:count].clone(),
            "sample_id": self.sample_id[:count].clone(),
            "trajectory_id": self.trajectory_id[:count].clone(),
        }
        state.update({name: self._storage[:count, :, field_slice].clone()
                      for name, field_slice in self._field_slices.items()})
        return state

    def preflight_state_dict(self, state):
        """Validate and copy a snapshot without changing the live buffer."""
        state = require_exact_keys(state, self._STATE_METADATA | set(self._field_slices),
                                   "trajectory replay state")
        if (state["schema"] != self._STATE_SCHEMA or isinstance(state["version"], bool)
                or state["version"] != self._VERSION):
            raise ValueError("Unsupported trajectory replay state schema/version.")
        for name in ("requested_capacity", "capacity", "latent_dim", "action_dim", "horizon",
                     "horizon_conditioning_horizon"):
            saved, expected = state[name], getattr(self, name)
            if (expected is None and saved is not None) or (
                expected is not None and
                (isinstance(saved, bool) or not isinstance(saved, Integral) or saved != expected)
            ):
                raise ValueError(f"Trajectory replay {name} is incompatible.")
        full, pos = state["full"], state["pos"]
        next_trajectory, next_sample = state["next_trajectory_id"], state["next_sample_id"]
        if not isinstance(full, bool):
            raise TypeError("Trajectory replay full must be bool.")
        for name, value in (("pos", pos), ("next_trajectory_id", next_trajectory),
                            ("next_sample_id", next_sample)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"Trajectory replay {name} is invalid.")
        if pos != next_trajectory % self.trajectory_capacity:
            raise ValueError("Trajectory replay position is inconsistent with trajectory IDs.")
        if full != (next_trajectory >= self.trajectory_capacity):
            raise ValueError("Trajectory replay full flag is inconsistent with trajectory IDs.")
        batch = {name: state[name] for name in self._field_slices}
        batch["valid"] = state["valid"]
        packed, valid, lengths = self._prepare_batch(batch, validate=True, exact_dtype=True)
        count = min(next_trajectory, self.trajectory_capacity)
        if len(lengths) != count:
            raise ValueError("Trajectory replay row count is inconsistent with ring metadata.")
        trajectory_ids, sample_ids = state["trajectory_id"], state["sample_id"]
        for name, value, shape in (
            ("trajectory_id", trajectory_ids, (count,)),
            ("sample_id", sample_ids, (count, self.horizon)),
        ):
            if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != torch.long:
                raise ValueError(f"Trajectory replay {name} has the wrong shape or dtype.")
        physical = torch.arange(count, device=self.device)
        logical = torch.remainder(physical - pos, self.trajectory_capacity) if full else physical
        expected_trajectory = logical + next_trajectory - count
        trajectory_ids = trajectory_ids.detach().to(self.device).clone()
        sample_ids = sample_ids.detach().to(self.device).clone()
        if not torch.equal(trajectory_ids, expected_trajectory):
            raise ValueError("Trajectory IDs are inconsistent with ring metadata.")
        if next_sample < sum(lengths) or not next_trajectory <= next_sample <= next_trajectory * self.horizon:
            raise ValueError("Transition sample counter is inconsistent with trajectory lengths.")
        if not full and next_sample != sum(lengths):
            raise ValueError("Transition sample counter is inconsistent with retained trajectories.")
        evicted_trajectories = next_trajectory - count
        evicted_samples = next_sample - sum(lengths)
        if not evicted_trajectories <= evicted_samples <= evicted_trajectories * self.horizon:
            raise ValueError("Transition sample counter is inconsistent with evicted trajectories.")
        # Retention evicts complete oldest trajectories, so valid IDs must be
        # exactly one consecutive suffix in chronological trajectory order.
        order = torch.argsort(trajectory_ids)
        ordered_valid = valid.index_select(0, order).squeeze(-1)
        ordered_ids = sample_ids.index_select(0, order)
        expected_samples = torch.arange(next_sample - sum(lengths), next_sample, device=self.device)
        if (not torch.equal(ordered_ids[ordered_valid], expected_samples)
                or not bool((sample_ids[~valid.squeeze(-1)] == -1).all().item())):
            raise ValueError("Transition sample IDs are inconsistent with trajectory retention.")
        return {
            "packed": packed.clone(), "valid": valid.clone(), "lengths": list(lengths),
            "sample_id": sample_ids, "trajectory_id": trajectory_ids,
            "pos": int(pos), "full": full,
            "next_sample_id": int(next_sample), "next_trajectory_id": int(next_trajectory),
        }

    def _commit_state_candidate(self, candidate):
        self.clear()
        count = len(candidate["lengths"])
        self._storage[:count].copy_(candidate["packed"])
        self.valid[:count].copy_(candidate["valid"])
        self.sample_id[:count].copy_(candidate["sample_id"])
        self.trajectory_id[:count].copy_(candidate["trajectory_id"])
        self._lengths[:count] = candidate["lengths"]
        for name in ("pos", "full", "next_sample_id", "next_trajectory_id"):
            setattr(self, name, candidate[name])
        self._refresh_valid_indices()

    def load_state_dict(self, state):
        if not state:
            self.clear()
            return self
        candidate = self.preflight_state_dict(state)
        self._commit_state_candidate(candidate)
        return self

    def training_state_dict(self):
        return {"schema": self._TRAINING_SCHEMA, "version": self._VERSION,
                "state": self.state_dict()}

    def load_training_state_dict(self, state):
        state = require_exact_keys(state, {"schema", "version", "state"},
                                   "trajectory replay training state")
        if (state["schema"] != self._TRAINING_SCHEMA or isinstance(state["version"], bool)
                or state["version"] != self._VERSION):
            raise ValueError("Unsupported trajectory replay training-state schema/version.")
        candidate = self.preflight_state_dict(state["state"])
        self._commit_state_candidate(candidate)
        return self
