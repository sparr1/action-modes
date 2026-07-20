from collections import deque

import torch
from tensordict.tensordict import TensorDict
from .device import cuda_mem_get_info, resolve_device
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = resolve_device(getattr(cfg, 'device', None))
		self.cfg.device = str(self._device)
		self._capacity = min(cfg.buffer_size, getattr(cfg, 'steps', cfg.buffer_size))
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.train_unroll_horizon+1)
		self._num_eps = 0
		self._num_transitions = 0
		self._total_transitions = 0
		self._resident_episode_rows = deque()
		self._resident_rows = 0
		self._pin_memory = False

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	@property
	def num_transitions(self):
		"""Return real transitions currently resident in replay storage."""
		return self._num_transitions

	@property
	def total_transitions(self):
		"""Return the cumulative number of real transitions submitted."""
		return self._total_transitions

	@property
	def size(self):
		"""Return the number of TensorDict rows currently resident in storage."""
		if self._num_eps == 0:
			return 0
		return len(self._buffer)

	@property
	def fill_fraction(self):
		"""Return the fraction of replay storage currently occupied."""
		return self.size / self._capacity if self._capacity else 0.0

	def _track_episode_rows(self, num_rows):
		"""Mirror FIFO storage eviction while excluding each episode's initial row."""
		num_rows = max(0, int(num_rows))
		transitions = max(0, num_rows - 1)
		self._resident_episode_rows.append([num_rows, 1 if num_rows else 0, transitions])
		self._resident_rows += num_rows
		self._num_transitions += transitions
		self._total_transitions += transitions

		overflow = max(0, self._resident_rows - self._capacity)
		while overflow > 0 and self._resident_episode_rows:
			rows, initial_rows, resident_transitions = self._resident_episode_rows[0]
			removed = min(rows, overflow)
			removed_initial = min(initial_rows, removed)
			removed_transitions = removed - removed_initial
			rows -= removed
			initial_rows -= removed_initial
			resident_transitions -= removed_transitions
			self._num_transitions -= removed_transitions
			self._resident_rows -= removed
			overflow -= removed
			if rows == 0:
				self._resident_episode_rows.popleft()
			else:
				self._resident_episode_rows[0] = [rows, initial_rows, resident_transitions]

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			# TorchRL pins the sampled TensorDict, providing the staging memory
			# required for the non-blocking CPU-to-CUDA transfer below. Pinning
			# CUDA storage (or CPU-only training) has no benefit and is unsafe.
			pin_memory=self._pin_memory,
			prefetch=0,
			batch_size=self._batch_size,
		)

	@staticmethod
	def _uses_pinned_staging(storage_device, target_device):
		"""Whether sampled CPU rows need pinning for an asynchronous transfer."""
		return (
			torch.device(storage_device).type == 'cpu'
			and torch.device(target_device).type == 'cuda'
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = cuda_mem_get_info(self._device)
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = str(self._device) if self._device.type == 'cuda' and 2.5*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		self._pin_memory = self._uses_pinned_staging(
			self._storage_device, self._device
		)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td)
		rows_per_episode = int(td.shape[1])
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		for _ in range(num_new_eps):
			self._track_episode_rows(rows_per_episode)
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		self._track_episode_rows(len(td))
		return self._num_eps

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, terminated, task

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(
			-1, self.cfg.train_unroll_horizon+1
		).permute(1, 0)
		return self._prepare_batch(td)
