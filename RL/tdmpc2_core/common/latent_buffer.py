"""Small device-resident replay buffer used by AMBI's per-state inner SAC."""

from numbers import Integral

import torch

from .training_state import require_exact_keys


class LatentReplayBuffer:
    """Fixed-size packed replay storage for AMBI's latent transitions.

    All floating-point transition fields share one allocation. This lets an
    append use at most two ring-buffer copies and a sample use one gather,
    while the public field views and sample dictionary remain unchanged.
    """

    def __init__(
        self,
        capacity,
        latent_dim,
        action_dim,
        device,
        *,
        store_source=False,
        store_horizon=False,
        remaining_horizon_max=None,
    ):
        self.capacity = max(1, int(capacity))
        self.device = torch.device(device)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        if not isinstance(store_source, bool):
            raise TypeError("store_source must be bool.")
        self.store_source = store_source
        if not isinstance(store_horizon, bool):
            raise TypeError("store_horizon must be bool.")
        self.store_horizon = store_horizon
        if remaining_horizon_max is not None:
            if (isinstance(remaining_horizon_max, bool)
                    or not isinstance(remaining_horizon_max, Integral)
                    or remaining_horizon_max < 1):
                raise ValueError("remaining_horizon_max must be a positive integer or None.")
            if not store_horizon:
                raise ValueError("Horizon-labelled replay requires store_horizon=True.")
            remaining_horizon_max = int(remaining_horizon_max)
        self.remaining_horizon_max = remaining_horizon_max
        if self.latent_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Latent and action dimensions must be positive.")

        # Keep named views for compatibility, but gather/copy the packed rows.
        self._field_slices = {}
        offset = 0
        field_widths = (
            ("z", self.latent_dim),
            ("action", self.action_dim),
            ("reward", 1),
            ("next_z", self.latent_dim),
            ("terminated", 1),
        )
        if self.store_horizon:
            field_widths += (("horizon_end", 1),)
        if self.remaining_horizon_max is not None:
            field_widths += (("remaining_horizon", 1),)
        for name, width in field_widths:
            self._field_slices[name] = slice(offset, offset + width)
            offset += width
        self._storage = torch.empty(self.capacity, offset, device=self.device)
        self.z = self._storage[:, self._field_slices["z"]]
        self.action = self._storage[:, self._field_slices["action"]]
        self.reward = self._storage[:, self._field_slices["reward"]]
        self.next_z = self._storage[:, self._field_slices["next_z"]]
        self.terminated = self._storage[:, self._field_slices["terminated"]]
        self.horizon_end = (
            self._storage[:, self._field_slices["horizon_end"]]
            if self.store_horizon else None
        )
        self.remaining_horizon = (
            self._storage[:, self._field_slices["remaining_horizon"]]
            if self.remaining_horizon_max is not None else None
        )
        self.source = (
            torch.empty(
                self.capacity,
                1,
                dtype=torch.uint8,
                device=self.device,
            )
            if self.store_source
            else None
        )

        self.pos = 0
        self.full = False
        self.next_sample_id = 0

    @property
    def size(self):
        return self.capacity if self.full else self.pos

    @property
    def sample_id(self):
        """Materialize physical-slot IDs for legacy state/debug access.

        IDs are derivable from ring metadata, so maintaining a device tensor on
        every append is unnecessary. Unused slots, as before, are unspecified.
        """
        ids = torch.empty(self.capacity, dtype=torch.long, device=self.device)
        if self.size:
            indices = torch.arange(self.size, dtype=torch.long, device=self.device)
            ids[: self.size] = self._sample_ids(indices)
        return ids

    def clear(self, *, preserve_sample_ids=False):
        """Discard all stored transitions without reallocating device storage."""
        self.pos = 0
        self.full = False
        if not preserve_sample_ids:
            self.next_sample_id = 0

    def _reshape_fields(self, z, action, reward, next_z, terminated, horizon_end=None,
                        remaining_horizon=None):
        values = (
            z.detach().reshape(-1, self.latent_dim),
            action.detach().reshape(-1, self.action_dim),
            reward.detach().reshape(-1, 1),
            next_z.detach().reshape(-1, self.latent_dim),
            terminated.detach().reshape(-1, 1),
        )
        if self.store_horizon:
            if horizon_end is None:
                raise ValueError("Latent replay requires horizon_end when store_horizon=True.")
            values += (horizon_end.detach().reshape(-1, 1),)
        elif horizon_end is not None:
            raise ValueError("Latent replay horizon_end requires store_horizon=True.")
        if self.remaining_horizon_max is not None:
            if remaining_horizon is None:
                raise ValueError("Horizon-labelled replay requires remaining_horizon.")
            values += (remaining_horizon.detach().reshape(-1, 1),)
        elif remaining_horizon is not None:
            raise ValueError("Latent replay remaining_horizon requires remaining_horizon_max.")
        n = values[0].shape[0]
        if any(value.shape[0] != n for value in values[1:]):
            raise ValueError(
                "Latent replay batch fields must have the same leading dimension."
            )
        return values, n

    def _pack_fields(self, values, n):
        # Normal collection produces same-device tensors, making this one cat.
        # Retain the old cross-device copy behavior for lifecycle/test callers.
        devices = {value.device for value in values}
        if len(devices) == 1:
            return torch.cat(values, dim=-1)

        packed = torch.empty(
            n,
            self._storage.shape[-1],
            dtype=self._storage.dtype,
            device=self.device,
        )
        for value, field_slice in zip(values, self._field_slices.values()):
            packed[:, field_slice].copy_(value)
        return packed

    def _reshape_source(self, source, n):
        """Validate and materialize primary/explorer labels for one append."""
        if not self.store_source:
            if source is not None:
                raise ValueError(
                    "Latent replay source labels require store_source=True."
                )
            return None
        if source is None:
            raise ValueError(
                "Source labels are required when latent replay store_source=True."
            )
        if isinstance(source, bool):
            source = int(source)
        if isinstance(source, int):
            if source not in (0, 1):
                raise ValueError("Latent replay source labels must be 0 or 1.")
            return torch.full(
                (n, 1),
                source,
                dtype=torch.uint8,
                device=self.device,
            )
        if not torch.is_tensor(source):
            raise TypeError("Latent replay source labels must be an integer or tensor.")
        source = source.detach()
        if source.numel() == 1:
            source = source.reshape(1, 1).expand(n, 1)
        else:
            source = source.reshape(-1, 1)
            if source.shape[0] != n:
                raise ValueError(
                    "Latent replay source labels must match the transition row count."
                )
        if not bool(torch.logical_or(source == 0, source == 1).all().item()):
            raise ValueError("Latent replay source labels must be 0 or 1.")
        return source.to(device=self.device, dtype=torch.uint8)

    def _append_packed(self, packed, original_n, source=None):
        self.next_sample_id += original_n
        old_pos = self.pos
        n = original_n
        if n >= self.capacity:
            packed = packed[-self.capacity :]
            if source is not None:
                source = source[-self.capacity :]
            n = self.capacity
            # ``packed`` now starts at the oldest retained transition. A
            # sequence of smaller appends would place that row at the final
            # write cursor, not necessarily at the cursor that preceded this
            # bulk append. Preserve that exact physical ring layout so seeded
            # physical-index sampling has unchanged transition semantics.
            write_pos = (old_pos + original_n) % self.capacity
        else:
            write_pos = old_pos

        first = min(n, self.capacity - write_pos)
        second = n - first
        self._storage[write_pos : write_pos + first].copy_(packed[:first])
        if self.source is not None:
            self.source[write_pos : write_pos + first].copy_(source[:first])
        if second:
            self._storage[:second].copy_(packed[first:])
            if self.source is not None:
                self.source[:second].copy_(source[first:])

        self.pos = (old_pos + original_n) % self.capacity
        self.full = self.full or (old_pos + original_n >= self.capacity)

    def add_batch(self, z, action, reward, next_z, terminated, *, source=None,
                  horizon_end=None, remaining_horizon=None, validate=True):
        """Append transitions; internal collectors may skip generated horizon checks."""
        values, n = self._reshape_fields(
            z, action, reward, next_z, terminated, horizon_end, remaining_horizon,
        )
        if n == 0:
            return
        if validate and self.remaining_horizon_max is not None:
            self._validate_remaining_horizon(values[-1], values[-2])
        source = self._reshape_source(source, n)
        self._append_packed(self._pack_fields(values, n), n, source)

    def add_packed(self, packed, *, source=None, validate=True):
        """Append rows already laid out like the packed replay allocation."""
        if not torch.is_tensor(packed) or packed.ndim < 2:
            raise TypeError("Packed latent replay rows must be a tensor with rows.")
        if packed.shape[-1] != self._storage.shape[-1]:
            raise ValueError(
                "Packed latent replay width does not match the configured fields."
            )
        packed = packed.detach().reshape(-1, self._storage.shape[-1])
        if validate and self.remaining_horizon_max is not None:
            self._validate_remaining_horizon(
                packed[:, self._field_slices["remaining_horizon"]],
                packed[:, self._field_slices["horizon_end"]],
            )
        if packed.device != self.device:
            packed = packed.to(self.device)
        if packed.dtype != self._storage.dtype:
            packed = packed.to(dtype=self._storage.dtype)
        if packed.shape[0]:
            n = int(packed.shape[0])
            source = self._reshape_source(source, n)
            self._append_packed(packed, n, source)

    def add_round(self, z, action, reward, next_z, terminated, *, source=None,
                  horizon_end=None, remaining_horizon=None, validate=True):
        """Append a dense rollout round in horizon-major order.

        Inputs may have any common leading dimensions (normally ``H x N``).
        Flattening keeps the last dimension as the field width, so transitions
        are stored as ``h0/n0, h0/n1, ..., h1/n0, ...`` in one append.
        """
        self.add_batch(z, action, reward, next_z, terminated, source=source,
                       horizon_end=horizon_end, remaining_horizon=remaining_horizon,
                       validate=validate)

    def _draw_indices(self, batch_size, replacement, generator):
        if replacement:
            return torch.randint(
                self.size,
                (batch_size,),
                device=self.device,
                generator=generator,
            )
        return torch.randperm(
            self.size,
            device=self.device,
            generator=generator,
        )[:batch_size]

    def _sample_ids(self, indices):
        """Map physical ring indices to monotonically assigned sample IDs."""
        oldest_id = self.next_sample_id - self.size
        if self.full:
            logical_offset = torch.remainder(indices - self.pos, self.capacity)
        else:
            logical_offset = indices
        return logical_offset + oldest_id

    def sample(
        self,
        batch_size,
        *,
        replacement=True,
        generator=None,
        include_ids=True,
        indices=None,
    ):
        """Sample transitions, optionally using pre-generated physical indices.

        The default return value contains the legacy ``indices`` and
        ``sample_ids`` fields. Passing ``include_ids=False`` avoids both ID
        construction and retaining index tensors for non-diagnostic updates.
        Supplying ``indices`` performs no random draw and therefore does not
        advance ``generator``.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty latent replay buffer.")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Latent replay batch_size must be positive.")
        if not replacement and batch_size > self.size:
            raise ValueError(
                "Cannot sample latent replay without replacement: "
                f"batch_size={batch_size} exceeds replay size={self.size}."
            )

        if indices is None:
            indices = self._draw_indices(batch_size, replacement, generator)
        else:
            if not torch.is_tensor(indices):
                raise TypeError("Pre-generated latent replay indices must be a tensor.")
            if indices.ndim != 1 or indices.numel() != batch_size:
                raise ValueError(
                    "Pre-generated latent replay indices must be one-dimensional "
                    "and match batch_size."
                )
            if indices.device != self._storage.device:
                raise ValueError(
                    "Pre-generated latent replay indices must be on the replay device."
                )
            if indices.dtype != torch.long:
                raise TypeError("Pre-generated latent replay indices must have dtype long.")

        packed = self._storage.index_select(0, indices)
        # Observational labels must not change the tensor strides seen by the
        # learner. In particular, Inductor may choose different reduction
        # schedules for widened replay rows and introduce rounding drift even
        # when the unconditioned networks never consume the labels. Preserve
        # the historical packed minibatch layout with one device-local copy.
        horizon_slice = self._field_slices.get("remaining_horizon")
        learner_packed = (packed if horizon_slice is None
                          else packed[:, :horizon_slice.start].contiguous())
        batch = {
            name: learner_packed[:, field_slice]
            for name, field_slice in self._field_slices.items()
            if name != "remaining_horizon"
        }
        if horizon_slice is not None:
            batch["remaining_horizon"] = packed[:, horizon_slice]
        if self.source is not None:
            batch["source"] = self.source.index_select(0, indices)
        if include_ids:
            batch["indices"] = indices
            batch["sample_ids"] = self._sample_ids(indices)
        return batch

    def state_dict(self):
        """Return the live replay contents for in-process lifecycle management."""
        size = self.size
        indices = torch.arange(size, dtype=torch.long, device=self.device)
        state = {
            "capacity": self.capacity,
            "pos": self.pos,
            "full": self.full,
            "next_sample_id": self.next_sample_id,
            "z": self.z[:size].clone(),
            "action": self.action[:size].clone(),
            "reward": self.reward[:size].clone(),
            "next_z": self.next_z[:size].clone(),
            "terminated": self.terminated[:size].clone(),
            "sample_id": self._sample_ids(indices),
        }
        if self.source is not None:
            state["source"] = self.source[:size].clone()
        if self.store_horizon:
            state["horizon_end"] = self.horizon_end[:size].clone()
        if self.remaining_horizon_max is not None:
            state["remaining_horizon_max"] = self.remaining_horizon_max
            state["remaining_horizon"] = self.remaining_horizon[:size].clone()
        return state

    def training_state_dict(self):
        """Wrap the existing physical-ring state in a versioned contract."""
        state = {
            "schema": "ambi-latent-replay-training-state",
            "version": self._training_state_version,
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "state": self.state_dict(),
        }
        if self.store_source:
            state["store_source"] = True
        if self.store_horizon:
            state["store_horizon"] = True
        if self.remaining_horizon_max is not None:
            state["remaining_horizon_max"] = self.remaining_horizon_max
        return state

    @property
    def _training_state_version(self):
        if self.remaining_horizon_max is not None:
            return 4
        return 3 if self.store_horizon else (2 if self.store_source else 1)

    def _validate_ring_metadata(self, *, pos, full, next_sample_id, owner):
        if not isinstance(full, bool):
            raise TypeError(f"{owner} full flag must be bool.")
        if isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos < self.capacity:
            raise ValueError(f"{owner} write position is invalid.")
        if (
            isinstance(next_sample_id, bool)
            or not isinstance(next_sample_id, int)
            or next_sample_id < 0
        ):
            raise ValueError(f"{owner} next_sample_id is invalid.")
        expected_pos = next_sample_id % self.capacity
        if pos != expected_pos:
            raise ValueError(
                f"{owner} write position is inconsistent with next_sample_id: "
                f"checkpoint={pos}, expected={expected_pos}."
            )
        expected_full = next_sample_id >= self.capacity
        if full is not expected_full:
            raise ValueError(
                f"{owner} full flag is inconsistent with next_sample_id."
            )
        return min(next_sample_id, self.capacity)

    def load_training_state_dict(self, state):
        """Restore an exact, device-portable physical ring snapshot."""
        top_level_keys = {
            "schema",
            "version",
            "latent_dim",
            "action_dim",
            "state",
        }
        if self.store_source:
            top_level_keys.add("store_source")
        if self.store_horizon:
            top_level_keys.add("store_horizon")
        if self.remaining_horizon_max is not None:
            top_level_keys.add("remaining_horizon_max")
        state = require_exact_keys(
            state,
            top_level_keys,
            "latent replay training state",
        )
        expected_version = self._training_state_version
        if (
            state["schema"] != "ambi-latent-replay-training-state"
            or state["version"] != expected_version
        ):
            raise ValueError("Unsupported latent replay training-state version.")
        if self.store_source and state["store_source"] is not True:
            raise ValueError("Latent replay source-label configuration is incompatible.")
        if self.store_horizon and state["store_horizon"] is not True:
            raise ValueError("Latent replay horizon configuration is incompatible.")
        self._validate_horizon_configuration(state)
        expected = {
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
        }
        actual = {key: state[key] for key in expected}
        if actual != expected:
            raise ValueError(
                "Latent replay training-state configuration is incompatible: "
                f"checkpoint={actual}, configured={expected}."
            )
        candidate = self._preflight_exact_state_dict(state["state"])
        self._commit_state_candidate(candidate)
        return self

    def _preflight_exact_state_dict(self, state):
        physical_keys = {
            "capacity",
            "pos",
            "full",
            "next_sample_id",
            "z",
            "action",
            "reward",
            "next_z",
            "terminated",
            "sample_id",
        }
        if self.store_source:
            physical_keys.add("source")
        if self.store_horizon:
            physical_keys.add("horizon_end")
        if self.remaining_horizon_max is not None:
            physical_keys.update(("remaining_horizon_max", "remaining_horizon"))
        state = require_exact_keys(
            state,
            physical_keys,
            "latent replay physical state",
        )
        self._validate_horizon_configuration(state)
        capacity = state["capacity"]
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity != self.capacity
        ):
            raise ValueError("Latent replay state capacity does not match this buffer.")

        widths = {
            "z": self.latent_dim,
            "action": self.action_dim,
            "reward": 1,
            "next_z": self.latent_dim,
            "terminated": 1,
        }
        if self.store_horizon:
            widths["horizon_end"] = 1
        if self.remaining_horizon_max is not None:
            widths["remaining_horizon"] = 1
        values = []
        size = None
        for name, width in widths.items():
            value = state[name]
            if not torch.is_tensor(value) or value.ndim != 2:
                raise TypeError(f"Latent replay field {name!r} must be a matrix tensor.")
            if value.shape[1] != width:
                raise ValueError(f"Latent replay field {name!r} has the wrong width.")
            if value.dtype != self._storage.dtype:
                raise ValueError(f"Latent replay field {name!r} has the wrong dtype.")
            if size is None:
                size = int(value.shape[0])
            elif value.shape[0] != size:
                raise ValueError("Latent replay fields have different row counts.")
            if name == "horizon_end":
                self._validate_horizon_flags(value)
            values.append(value)
        if self.remaining_horizon_max is not None:
            self._validate_remaining_horizon(state["remaining_horizon"], state["horizon_end"])
        if not 0 <= size <= self.capacity:
            raise ValueError("Latent replay state has an invalid size.")

        full, pos, next_sample_id = (
            state["full"],
            state["pos"],
            state["next_sample_id"],
        )
        expected_size = self._validate_ring_metadata(
            pos=pos,
            full=full,
            next_sample_id=next_sample_id,
            owner="Latent replay state",
        )
        if size != expected_size:
            raise ValueError("Latent replay size is inconsistent with ring metadata.")

        sample_id = state["sample_id"]
        if (
            not torch.is_tensor(sample_id)
            or sample_id.shape != torch.Size([size])
            or sample_id.dtype != torch.long
        ):
            raise ValueError("Latent replay sample_id has the wrong shape or dtype.")
        indices = torch.arange(size, dtype=torch.long)
        oldest = next_sample_id - size
        expected_ids = (
            torch.remainder(indices - pos, self.capacity) + oldest
            if full
            else indices + oldest
        )
        if not torch.equal(sample_id.detach().cpu(), expected_ids):
            raise ValueError("Latent replay sample_id is inconsistent with ring metadata.")

        source = None
        if self.store_source:
            source = state["source"]
            if (
                not torch.is_tensor(source)
                or source.shape != torch.Size([size, 1])
                or source.dtype != torch.uint8
            ):
                raise ValueError("Latent replay source has the wrong shape or dtype.")
            if not bool(torch.logical_or(source == 0, source == 1).all().item()):
                raise ValueError("Latent replay source labels must be 0 or 1.")
            source = source.detach().to(self.device)

        packed = self._pack_fields(tuple(values), size) if size else None
        return packed, source, size, full, pos, next_sample_id

    def _commit_state_candidate(self, candidate):
        packed, source, size, full, pos, next_sample_id = candidate
        if size:
            self._storage[:size].copy_(packed)
            if self.source is not None:
                self.source[:size].copy_(source)
        self.full = full
        self.pos = pos
        self.next_sample_id = next_sample_id

    @staticmethod
    def _validate_horizon_flags(value):
        # Restore-time validation only; collection and compiled kernels do not
        # synchronize the device to validate flags generated by our collectors.
        if not bool(((value == 0) | (value == 1)).all().item()):
            raise ValueError("Latent replay horizon_end flags must be 0 or 1.")

    def _validate_horizon_configuration(self, state):
        saved = state.get("remaining_horizon_max")
        configured = self.remaining_horizon_max
        if configured is None:
            if "remaining_horizon_max" in state or "remaining_horizon" in state:
                raise ValueError("Latent replay remaining-horizon metadata is incompatible.")
        elif (isinstance(saved, bool) or not isinstance(saved, Integral)
              or int(saved) != configured):
            raise ValueError("Latent replay remaining_horizon_max is incompatible.")

    def _validate_remaining_horizon(self, value, horizon_end):
        # Public appends and restores validate data; trusted internal collectors
        # pass validate=False to avoid synchronizing generated device tensors.
        if value.dtype == torch.bool or value.is_complex():
            raise ValueError("Latent replay remaining_horizon must contain integers in [1, H].")
        valid = torch.isfinite(value) & (value >= 1) & (value <= self.remaining_horizon_max)
        valid &= value == value.floor()
        if not bool(valid.all().item()):
            raise ValueError("Latent replay remaining_horizon must contain integers in [1, H].")
        if not bool((horizon_end.to(value.device) == (value == 1)).all().item()):
            raise ValueError("Latent replay horizon_end must equal (remaining_horizon == 1).")

    def load_state_dict(self, state):
        """Restore replay contents into this buffer, respecting its capacity."""
        if not state:
            self.clear()
            return
        self._validate_horizon_configuration(state)
        if int(state.get("capacity", self.capacity)) != self.capacity:
            raise ValueError("Latent replay state capacity does not match this buffer.")
        size = int(state["z"].shape[0])
        if not 0 <= size <= self.capacity:
            raise ValueError("Latent replay state has an invalid size.")

        values, packed_size = self._reshape_fields(
            state["z"],
            state["action"],
            state["reward"],
            state["next_z"],
            state["terminated"],
            state.get("horizon_end"),
            state.get("remaining_horizon"),
        )
        if self.store_horizon:
            self._validate_horizon_flags(state["horizon_end"])
        if self.remaining_horizon_max is not None:
            self._validate_remaining_horizon(values[-1], values[-2])
        if packed_size != size:
            raise ValueError("Latent replay state fields have incompatible shapes.")
        full = state.get("full", size == self.capacity)
        pos = state.get("pos", 0 if full else size)
        next_sample_id = state.get("next_sample_id", size)
        expected_size = self._validate_ring_metadata(
            pos=pos,
            full=full,
            next_sample_id=next_sample_id,
            owner="Latent replay state",
        )
        if size != expected_size:
            raise ValueError("Latent replay size is inconsistent with ring metadata.")
        packed = self._pack_fields(values, size) if size else None
        if self.store_source:
            source = self._reshape_source(state.get("source"), size)
        else:
            source = None
        if not self.store_source and "source" in state:
            raise ValueError(
                "Latent replay source labels require store_source=True."
            )

        # All validation finishes before live metadata or storage is changed.
        self._commit_state_candidate(
            (packed, source, size, full, pos, next_sample_id)
        )
