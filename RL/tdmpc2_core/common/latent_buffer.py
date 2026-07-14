"""Small device-resident replay buffer used by AMBI's per-state inner SAC."""

import torch


class LatentReplayBuffer:
    def __init__(self, capacity, latent_dim, action_dim, device):
        self.capacity = max(1, int(capacity))
        self.device = torch.device(device)
        self.z = torch.empty(self.capacity, latent_dim, device=self.device)
        self.action = torch.empty(self.capacity, action_dim, device=self.device)
        self.reward = torch.empty(self.capacity, 1, device=self.device)
        self.next_z = torch.empty(self.capacity, latent_dim, device=self.device)
        self.terminated = torch.empty(self.capacity, 1, device=self.device)
        self.sample_id = torch.empty(
            self.capacity, dtype=torch.long, device=self.device
        )
        self.pos = 0
        self.full = False
        self.next_sample_id = 0

    @property
    def size(self):
        return self.capacity if self.full else self.pos

    def clear(self):
        """Discard all stored transitions without reallocating device storage."""
        self.pos = 0
        self.full = False
        self.next_sample_id = 0

    def add_batch(self, z, action, reward, next_z, terminated):
        z = z.detach().reshape(-1, self.z.shape[-1])
        action = action.detach().reshape(-1, self.action.shape[-1])
        reward = reward.detach().reshape(-1, 1)
        next_z = next_z.detach().reshape(-1, self.next_z.shape[-1])
        terminated = terminated.detach().reshape(-1, 1)
        n = z.shape[0]
        if not (action.shape[0] == reward.shape[0] == next_z.shape[0] == terminated.shape[0] == n):
            raise ValueError("Latent replay batch fields must have the same leading dimension.")
        if n == 0:
            return

        sample_ids = torch.arange(
            self.next_sample_id,
            self.next_sample_id + n,
            dtype=torch.long,
            device=self.device,
        )
        self.next_sample_id += n
        if n >= self.capacity:
            z = z[-self.capacity:]
            action = action[-self.capacity:]
            reward = reward[-self.capacity:]
            next_z = next_z[-self.capacity:]
            terminated = terminated[-self.capacity:]
            sample_ids = sample_ids[-self.capacity:]
            n = self.capacity

        old_pos = self.pos
        first = min(n, self.capacity - old_pos)
        second = n - first
        fields = (
            (self.z, z),
            (self.action, action),
            (self.reward, reward),
            (self.next_z, next_z),
            (self.terminated, terminated),
        )
        for storage, values in fields:
            storage[old_pos:old_pos + first].copy_(values[:first])
            if second:
                storage[:second].copy_(values[first:])
        self.sample_id[old_pos:old_pos + first].copy_(sample_ids[:first])
        if second:
            self.sample_id[:second].copy_(sample_ids[first:])

        self.pos = (old_pos + n) % self.capacity
        self.full = self.full or (old_pos + n >= self.capacity)

    def sample(self, batch_size, *, replacement=True, generator=None):
        """Sample transitions and expose indices for compute/diversity accounting.

        Sampling with replacement is the legacy AMBI behavior. Without-replacement
        sampling is deliberately strict: callers must collect enough unique data
        instead of silently shrinking or partially duplicating a requested batch.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty latent replay buffer.")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("Latent replay batch_size must be positive.")
        if replacement:
            indices = torch.randint(
                self.size,
                (batch_size,),
                device=self.device,
                generator=generator,
            )
        else:
            if batch_size > self.size:
                raise ValueError(
                    "Cannot sample latent replay without replacement: "
                    f"batch_size={batch_size} exceeds replay size={self.size}."
                )
            indices = torch.randperm(
                self.size,
                device=self.device,
                generator=generator,
            )[:batch_size]
        return {
            "z": self.z[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "next_z": self.next_z[indices],
            "terminated": self.terminated[indices],
            "indices": indices,
            "sample_ids": self.sample_id[indices],
        }

    def state_dict(self):
        """Return the live replay contents for in-process lifecycle management."""
        size = self.size
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
            "sample_id": self.sample_id[:size].clone(),
        }

    def load_state_dict(self, state):
        """Restore replay contents into this buffer, respecting its capacity."""
        self.clear()
        if not state:
            return
        if int(state.get("capacity", self.capacity)) != self.capacity:
            raise ValueError(
                "Latent replay state capacity does not match this buffer."
            )
        size = int(state["z"].shape[0])
        if not 0 <= size <= self.capacity:
            raise ValueError("Latent replay state has an invalid size.")
        for storage, key in (
            (self.z, "z"),
            (self.action, "action"),
            (self.reward, "reward"),
            (self.next_z, "next_z"),
            (self.terminated, "terminated"),
        ):
            storage[:size].copy_(state[key].to(self.device))
        if "sample_id" in state:
            self.sample_id[:size].copy_(state["sample_id"].to(self.device))
        else:
            self.sample_id[:size].copy_(
                torch.arange(size, device=self.device, dtype=torch.long)
            )
        self.full = bool(state.get("full", size == self.capacity))
        self.pos = int(state.get("pos", 0 if self.full else size))
        if not 0 <= self.pos < self.capacity:
            raise ValueError("Latent replay state has an invalid write position.")
        self.next_sample_id = int(
            state.get("next_sample_id", size)
        )
