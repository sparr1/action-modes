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
        self.pos = 0
        self.full = False

    @property
    def size(self):
        return self.capacity if self.full else self.pos

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

        if n >= self.capacity:
            z = z[-self.capacity:]
            action = action[-self.capacity:]
            reward = reward[-self.capacity:]
            next_z = next_z[-self.capacity:]
            terminated = terminated[-self.capacity:]
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

        self.pos = (old_pos + n) % self.capacity
        self.full = self.full or (old_pos + n >= self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty latent replay buffer.")
        indices = torch.randint(self.size, (int(batch_size),), device=self.device)
        return {
            "z": self.z[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "next_z": self.next_z[indices],
            "terminated": self.terminated[indices],
        }
