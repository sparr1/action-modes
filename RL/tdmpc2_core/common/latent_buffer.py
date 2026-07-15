"""Small device-resident replay buffer used by AMBI's per-state inner SAC."""

import torch


class LatentReplayBuffer:
    """Fixed-size packed replay storage for AMBI's latent transitions.

    All floating-point transition fields share one allocation. This lets an
    append use at most two ring-buffer copies and a sample use one gather,
    while the public field views and sample dictionary remain unchanged.
    """

    def __init__(self, capacity, latent_dim, action_dim, device):
        self.capacity = max(1, int(capacity))
        self.device = torch.device(device)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        if self.latent_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Latent and action dimensions must be positive.")

        # Keep named views for compatibility, but gather/copy the packed rows.
        self._field_slices = {}
        offset = 0
        for name, width in (
            ("z", self.latent_dim),
            ("action", self.action_dim),
            ("reward", 1),
            ("next_z", self.latent_dim),
            ("terminated", 1),
        ):
            self._field_slices[name] = slice(offset, offset + width)
            offset += width
        self._storage = torch.empty(self.capacity, offset, device=self.device)
        self.z = self._storage[:, self._field_slices["z"]]
        self.action = self._storage[:, self._field_slices["action"]]
        self.reward = self._storage[:, self._field_slices["reward"]]
        self.next_z = self._storage[:, self._field_slices["next_z"]]
        self.terminated = self._storage[:, self._field_slices["terminated"]]

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

    def clear(self):
        """Discard all stored transitions without reallocating device storage."""
        self.pos = 0
        self.full = False
        self.next_sample_id = 0

    def _reshape_fields(self, z, action, reward, next_z, terminated):
        values = (
            z.detach().reshape(-1, self.latent_dim),
            action.detach().reshape(-1, self.action_dim),
            reward.detach().reshape(-1, 1),
            next_z.detach().reshape(-1, self.latent_dim),
            terminated.detach().reshape(-1, 1),
        )
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

    def _append_packed(self, packed, original_n):
        self.next_sample_id += original_n
        old_pos = self.pos
        n = original_n
        if n >= self.capacity:
            packed = packed[-self.capacity :]
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
        if second:
            self._storage[:second].copy_(packed[first:])

        self.pos = (old_pos + original_n) % self.capacity
        self.full = self.full or (old_pos + original_n >= self.capacity)

    def add_batch(self, z, action, reward, next_z, terminated):
        """Append a flat batch of transitions in its existing row order."""
        values, n = self._reshape_fields(z, action, reward, next_z, terminated)
        if n == 0:
            return
        self._append_packed(self._pack_fields(values, n), n)

    def add_packed(self, packed):
        """Append rows already laid out like the packed replay allocation."""
        if not torch.is_tensor(packed) or packed.ndim < 2:
            raise TypeError("Packed latent replay rows must be a tensor with rows.")
        if packed.shape[-1] != self._storage.shape[-1]:
            raise ValueError(
                "Packed latent replay width does not match the configured fields."
            )
        packed = packed.detach().reshape(-1, self._storage.shape[-1])
        if packed.device != self.device:
            packed = packed.to(self.device)
        if packed.dtype != self._storage.dtype:
            packed = packed.to(dtype=self._storage.dtype)
        if packed.shape[0]:
            self._append_packed(packed, int(packed.shape[0]))

    def add_round(self, z, action, reward, next_z, terminated):
        """Append a dense rollout round in horizon-major order.

        Inputs may have any common leading dimensions (normally ``H x N``).
        Flattening keeps the last dimension as the field width, so transitions
        are stored as ``h0/n0, h0/n1, ..., h1/n0, ...`` in one append.
        """
        self.add_batch(z, action, reward, next_z, terminated)

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
        batch = {
            name: packed[:, field_slice]
            for name, field_slice in self._field_slices.items()
        }
        if include_ids:
            batch["indices"] = indices
            batch["sample_ids"] = self._sample_ids(indices)
        return batch

    def state_dict(self):
        """Return the live replay contents for in-process lifecycle management."""
        size = self.size
        indices = torch.arange(size, dtype=torch.long, device=self.device)
        return {
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

    def load_state_dict(self, state):
        """Restore replay contents into this buffer, respecting its capacity."""
        self.clear()
        if not state:
            return
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
        )
        if packed_size != size:
            raise ValueError("Latent replay state fields have incompatible shapes.")
        if size:
            self._storage[:size].copy_(self._pack_fields(values, size))

        self.full = bool(state.get("full", size == self.capacity))
        self.pos = int(state.get("pos", 0 if self.full else size))
        if not 0 <= self.pos < self.capacity:
            raise ValueError("Latent replay state has an invalid write position.")
        self.next_sample_id = int(state.get("next_sample_id", size))
