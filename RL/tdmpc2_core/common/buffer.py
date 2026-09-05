from collections import deque

import torch
from tensordict.tensordict import TensorDict
from .device import cuda_mem_get_info, resolve_device
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

from .training_state import require_exact_keys


_BEHAVIOR_POLICY_FIELDS = (
	"behavior_pre_tanh_mean",
	"behavior_log_std",
	"behavior_policy_valid",
)
_BEHAVIOR_POLICY_REPLAY_SIGNATURE = "pre-tanh-diagonal-gaussian-v1"


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg, *, resumable=False):
		self.cfg = cfg
		self._device = resolve_device(getattr(cfg, 'device', None))
		self.cfg.device = str(self._device)
		self._capacity = min(cfg.buffer_size, getattr(cfg, 'steps', cfg.buffer_size))
		self._sampler = self._make_sampler()
		self._batch_size = cfg.batch_size * (cfg.train_unroll_horizon+1)
		self._num_eps = 0
		self._num_transitions = 0
		self._total_transitions = 0
		self._resident_episode_rows = deque()
		self._resident_rows = 0
		self._transition_index_cache = {}
		self._num_sampleable_transitions = 0
		self._pin_memory = False
		self._resumable_storage = False
		if resumable:
			self.enable_resumable_storage()

	def _make_sampler(self):
		return SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=self.cfg.multitask,
		)

	@property
	def resumable_storage(self):
		return self._resumable_storage

	def enable_resumable_storage(self):
		"""Force CPU replay placement before the first row is allocated.

		Legacy callers retain adaptive CPU/CUDA placement. Exact resume callers
		must opt in before collection so a later GPU change never changes replay
		placement or the TorchRL writer/sampler contract.
		"""
		if getattr(self.cfg, 'obs', 'state') != 'state':
			raise NotImplementedError(
				"Exact replay resume supports state observations only; RGB is unsupported."
			)
		if self._num_eps != 0 or hasattr(self, '_buffer'):
			raise RuntimeError(
				"Resumable replay storage must be enabled before the first episode."
			)
		self._resumable_storage = True
		return self

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
		self._invalidate_transition_index()

	def _invalidate_transition_index(self):
		self._transition_index_cache.clear()
		self._num_sampleable_transitions = None

	def _reserve_buffer(self, storage, *, sampler=None, pin_memory=None):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler if sampler is None else sampler,
			# TorchRL pins the sampled TensorDict, providing the staging memory
			# required for the non-blocking CPU-to-CUDA transfer below. Pinning
			# CUDA storage (or CPU-only training) has no benefit and is unsafe.
			pin_memory=self._pin_memory if pin_memory is None else bool(pin_memory),
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
		storage_device = (
			'cpu'
			if self._resumable_storage
			else (
				str(self._device)
				if self._device.type == 'cuda' and 2.5*total_bytes < mem_free
				else 'cpu'
			)
		)
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

	def _prepare_batch(self, td, *, include_behavior_policy=False):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		if include_behavior_policy and not self._behavior_policy_enabled():
			raise RuntimeError(
				"Behavior-policy replay is not enabled for this buffer."
			)
		selected = ("obs", "action", "reward", "terminated", "task")
		if include_behavior_policy:
			selected += _BEHAVIOR_POLICY_FIELDS
		td = td.select(*selected, strict=False).to(
			self._device, non_blocking=True
		)
		if include_behavior_policy:
			missing = set(_BEHAVIOR_POLICY_FIELDS).difference(td.keys())
			if missing:
				raise RuntimeError(
					"Replay sample lacks behavior-policy fields: "
					f"{sorted(missing)}."
				)
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
		batch = (obs, action, reward, terminated, task)
		if not include_behavior_policy:
			return batch
		behavior_mean = td.get("behavior_pre_tanh_mean")[1:].contiguous()
		behavior_log_std = td.get("behavior_log_std")[1:].contiguous()
		behavior_valid = (
			td.get("behavior_policy_valid")[1:].unsqueeze(-1).contiguous()
		)
		return batch + (behavior_mean, behavior_log_std, behavior_valid)

	def sample(self, *, include_behavior_policy=False):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(
			-1, self.cfg.train_unroll_horizon+1
		).permute(1, 0)
		if include_behavior_policy:
			return self._prepare_batch(td, include_behavior_policy=True)
		# Keep the historical one-argument seam used by tests and downstream
		# instrumentation when behavior-policy replay is disabled.
		return self._prepare_batch(td)

	@property
	def num_sampleable_transitions(self):
		"""Count real transitions with both observations still resident.

		An evicted predecessor makes the oldest remaining row of a partial
		episode unusable, even though that row still contains a recorded action.
		"""
		if self._num_sampleable_transitions is None:
			self._num_sampleable_transitions = sum(
				max(0, rows - 1) for rows, _, _ in self._resident_episode_rows
			)
		return self._num_sampleable_transitions

	def _transition_index(self, device):
		"""Build episode lookup tensors once per replay mutation and device."""
		device = torch.device(device)
		if device not in self._transition_index_cache:
			counts, starts = [], []
			row_offset = 0
			for rows, _, _ in self._resident_episode_rows:
				if rows > 1:
					counts.append(rows - 1)
					starts.append(row_offset)
				row_offset += rows
			cumulative = torch.tensor(counts, dtype=torch.long, device=device).cumsum(0)
			preceding = torch.cat((cumulative.new_zeros(1), cumulative[:-1]))
			self._transition_index_cache[device] = (
				cumulative, preceding, torch.tensor(starts, dtype=torch.long, device=device)
			)
		return self._transition_index_cache[device]

	def sample_transitions(self, batch_size, *, generator):
		"""Sample independent real transitions without touching the outer sampler.

		Sampling is uniform with replacement over consecutive resident rows
		within each episode. Actions, rewards, and true termination flags belong
		to the successor row. Episode timeouts still retain their successor.
		The caller supplies an isolated generator; neither TorchRL's sequence
		sampler nor PyTorch's default RNG is consumed.
		"""
		if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
			raise ValueError("Real replay transition batch_size must be a positive integer.")
		if not isinstance(generator, torch.Generator):
			raise TypeError("Real replay transition sampling requires a torch.Generator.")
		total = self.num_sampleable_transitions
		if total == 0:
			raise ValueError("Cannot sample real replay without a resident transition pair.")
		draw_device = generator.device
		cumulative, preceding, starts = self._transition_index(draw_device)
		draws = torch.randint(total, (batch_size,), device=draw_device, generator=generator)
		episodes = torch.searchsorted(cumulative, draws, right=True)
		logical = starts[episodes] + draws - preceding[episodes]
		cursor = int(self._buffer._writer._cursor) if self.size == self.capacity else 0
		indices = (logical + cursor).remainder(self.capacity).to(self._storage_device)
		next_indices = (indices + 1).remainder(self.capacity)
		storage = self._buffer._storage
		current = storage[indices].select("obs", "task", strict=False)
		successor = storage[next_indices].select("obs", "action", "reward", "terminated", strict=False)
		if self._pin_memory:
			current = current.pin_memory()
			successor = successor.pin_memory()
		current = current.to(self._device, non_blocking=True)
		successor = successor.to(self._device, non_blocking=True)
		reward = successor["reward"].reshape(batch_size, 1).contiguous()
		terminated = successor.get("terminated", None)
		terminated = (
			torch.zeros_like(reward)
			if terminated is None
			else terminated.reshape(batch_size, 1).contiguous()
		)
		return (
			current["obs"].contiguous(),
			successor["action"].contiguous(),
			reward,
			successor["obs"].contiguous(),
			terminated,
			current.get("task", None),
		)

	def _behavior_policy_enabled(self):
		return bool(getattr(self.cfg, "store_behavior_policy", False))

	def _training_state_version(self):
		return 2 if self._behavior_policy_enabled() else 1

	def _training_signature(self):
		mode = str(getattr(self.cfg, "obs", "state"))
		obs_shapes = getattr(self.cfg, "obs_shape", None)
		observation_shape = None
		if obs_shapes is not None:
			observation_shape = list(obs_shapes[mode])
		signature = {
			"capacity": int(self._capacity),
			"batch_size": int(self.cfg.batch_size),
			"train_unroll_horizon": int(self.cfg.train_unroll_horizon),
			"multitask": bool(self.cfg.multitask),
			"observation_mode": str(getattr(self.cfg, "obs", "state")),
			"observation_shape": observation_shape,
			"observation_dtype": str(
				getattr(self.cfg, "obs_dtype", "float32")
			),
			"action_dim": (
				None
				if not hasattr(self.cfg, "action_dim")
				else int(self.cfg.action_dim)
			),
		}
		if self._behavior_policy_enabled():
			signature["behavior_policy_replay"] = (
				_BEHAVIOR_POLICY_REPLAY_SIGNATURE
			)
		return signature

	def _configured_field_specs(self):
		"""Return the fixed state-observation replay tensor contract."""
		mode = str(getattr(self.cfg, "obs", "state"))
		observation_shapes = getattr(self.cfg, "obs_shape", None)
		if observation_shapes is None or mode not in observation_shapes:
			raise ValueError("Replay configuration lacks an observation shape.")
		action_dim = getattr(self.cfg, "action_dim", None)
		if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
			raise ValueError("Replay configuration lacks a positive action dimension.")
		observation_dtype = str(getattr(self.cfg, "obs_dtype", "float32"))
		if not observation_dtype.startswith("torch."):
			observation_dtype = f"torch.{observation_dtype}"
		specs = {
			"obs": {
				"name": "obs",
				"shape": [int(value) for value in observation_shapes[mode]],
				"dtype": observation_dtype,
			},
			"action": {
				"name": "action",
				"shape": [action_dim],
				"dtype": "torch.float32",
			},
			"reward": {"name": "reward", "shape": [], "dtype": "torch.float32"},
			"terminated": {
				"name": "terminated",
				"shape": [],
				"dtype": "torch.float32",
			},
			"episode": {"name": "episode", "shape": [], "dtype": "torch.int64"},
		}
		if bool(self.cfg.multitask):
			specs["task"] = {
				"name": "task",
				"shape": [],
				"dtype": "torch.int64",
			}
		if self._behavior_policy_enabled():
			specs.update(
				{
					"behavior_pre_tanh_mean": {
						"name": "behavior_pre_tanh_mean",
						"shape": [action_dim],
						"dtype": "torch.float32",
					},
					"behavior_log_std": {
						"name": "behavior_log_std",
						"shape": [action_dim],
						"dtype": "torch.float32",
					},
					"behavior_policy_valid": {
						"name": "behavior_policy_valid",
						"shape": [],
						"dtype": "torch.bool",
					},
				}
			)
		return specs

	def _accounting_state(self):
		return {
			"num_eps": int(self._num_eps),
			"num_transitions": int(self._num_transitions),
			"total_transitions": int(self._total_transitions),
			"resident_episode_rows": [
				list(entry) for entry in self._resident_episode_rows
			],
			"resident_rows": int(self._resident_rows),
		}

	@classmethod
	def _clone_tree_to_cpu(cls, value):
		if torch.is_tensor(value):
			return value.detach().cpu().clone()
		if isinstance(value, dict):
			return type(value)(
				(key, cls._clone_tree_to_cpu(item)) for key, item in value.items()
			)
		if isinstance(value, list):
			return [cls._clone_tree_to_cpu(item) for item in value]
		if isinstance(value, tuple):
			return tuple(cls._clone_tree_to_cpu(item) for item in value)
		return value

	def _storage_tensor_state(self):
		"""Return live CPU storage tensor references without cloning them."""
		if not hasattr(self, "_buffer"):
			return None, 0, None
		# ReplayBuffer.state_dict() clones LazyTensorStorage in full. Calling it
		# here would therefore materialize one capacity-sized copy for metadata and
		# another for every shard. Read the live physical storage directly and only
		# clone the bounded slice in training_state_shard().
		storage_state = self._buffer._storage._storage
		rows = int(len(self._buffer))
		fields = {
			key: value
			for key, value in storage_state.items()
			if not str(key).startswith("__")
		}
		if (
			not fields
			or any(not isinstance(key, str) for key in fields)
			or any(not torch.is_tensor(value) for value in fields.values())
		):
			raise TypeError(
				"Sharded replay serialization requires flat tensor-backed state fields."
			)
		expected_fields = {"obs", "action", "reward", "terminated", "episode"}
		if bool(self.cfg.multitask):
			expected_fields.add("task")
		if self._behavior_policy_enabled():
			expected_fields.update(_BEHAVIOR_POLICY_FIELDS)
		if set(fields) != expected_fields:
			raise ValueError(
				"Replay storage fields are incompatible with the state-observation "
				f"contract: storage={sorted(fields)}, expected={sorted(expected_fields)}."
			)
		if any(value.device.type != "cpu" for value in fields.values()):
			raise RuntimeError("Resumable replay unexpectedly uses non-CPU storage.")
		torchrl = {
			"sampler": self._clone_tree_to_cpu(self._buffer._sampler.state_dict()),
			"writer": self._clone_tree_to_cpu(self._buffer._writer.state_dict()),
			"transforms": self._clone_tree_to_cpu(
				self._buffer._transform.state_dict()
			),
			"batch_size": self._buffer._batch_size,
		}
		return fields, rows, torchrl

	def training_state_metadata(self):
		"""Return replay metadata without copying the capacity-sized storage.

		Pair this with :meth:`iter_training_state_shards`. The returned mapping has
		no resident replay tensor, so a checkpoint writer can serialize it with
		bounded additional memory.
		"""
		if not self._resumable_storage:
			raise RuntimeError(
				"Replay training state requires enable_resumable_storage() before use."
			)
		fields, rows, torchrl = self._storage_tensor_state()
		initialized = torchrl is not None
		if initialized and self._storage_device.type != "cpu":
			raise RuntimeError("Resumable replay unexpectedly uses non-CPU storage.")
		field_specs = []
		if fields is not None:
			for name, value in fields.items():
				field_specs.append(
					{
						"name": str(name),
						"shape": list(value.shape[1:]),
						"dtype": str(value.dtype),
					}
				)
			if {spec["name"]: spec for spec in field_specs} != (
				self._configured_field_specs()
			):
				raise RuntimeError(
					"Live replay tensors do not match the configured exact-resume schema."
				)
		return {
			"schema": "tdmpc2-replay-sharded-training-state",
			"version": self._training_state_version(),
			"signature": self._training_signature(),
			"storage_device": "cpu",
			"initialized": initialized,
			"storage_rows": rows,
			"field_specs": field_specs,
			"torchrl": torchrl,
			**self._accounting_state(),
		}

	def iter_training_state_shards(self, *, max_rows):
		"""Yield capacity-bounded CPU slices in physical storage order."""
		if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
			raise ValueError("Replay shard max_rows must be a positive integer.")
		metadata = self.training_state_metadata()
		rows = metadata["storage_rows"]
		for shard_index in range((rows + max_rows - 1) // max_rows):
			yield self.training_state_shard(shard_index, max_rows=max_rows)

	def training_state_shard(self, index, *, max_rows):
		"""Materialize exactly one shard for a deferred lineage file writer."""
		if not self._resumable_storage:
			raise RuntimeError(
				"Replay training state requires enable_resumable_storage() before use."
			)
		if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
			raise ValueError("Replay shard max_rows must be a positive integer.")
		if isinstance(index, bool) or not isinstance(index, int) or index < 0:
			raise ValueError("Replay shard index must be a non-negative integer.")
		fields, rows, _ = self._storage_tensor_state()
		start = index * max_rows
		if fields is None or start >= rows:
			raise IndexError(
				f"Replay shard index {index} is outside {rows} resident rows."
			)
		stop = min(rows, start + max_rows)
		return {
			"schema": "tdmpc2-replay-shard",
			"version": self._training_state_version(),
			"index": index,
			"start": start,
			"stop": stop,
			"fields": {
				name: value[start:stop].clone()
				for name, value in fields.items()
			},
		}

	def _preflight_sharded_metadata(self, metadata):
		metadata = require_exact_keys(
			metadata,
			{
				"schema",
				"version",
				"signature",
				"storage_device",
				"initialized",
				"storage_rows",
				"field_specs",
				"torchrl",
				"num_eps",
				"num_transitions",
				"total_transitions",
				"resident_episode_rows",
				"resident_rows",
			},
			"TD-MPC2 replay sharded metadata",
		)
		if (
			metadata["schema"] != "tdmpc2-replay-sharded-training-state"
			or metadata["version"] != self._training_state_version()
		):
			raise ValueError("Unsupported TD-MPC2 sharded replay version.")
		if metadata["signature"] != self._training_signature():
			raise ValueError("Sharded replay signature is incompatible.")
		if metadata["storage_device"] != "cpu":
			raise ValueError("Exact resume requires CPU replay storage.")
		if not isinstance(metadata["initialized"], bool):
			raise TypeError("Replay initialized flag must be bool.")
		rows = metadata["storage_rows"]
		if isinstance(rows, bool) or not isinstance(rows, int) or not 0 <= rows <= self._capacity:
			raise ValueError("Sharded replay storage_rows is invalid.")
		specs = metadata["field_specs"]
		if not isinstance(specs, list):
			raise TypeError("Sharded replay field_specs must be a list.")
		field_specs = {}
		for index, spec in enumerate(specs):
			spec = require_exact_keys(
				spec,
				{"name", "shape", "dtype"},
				f"replay field specification {index}",
			)
			name, shape, dtype = spec["name"], spec["shape"], spec["dtype"]
			if not isinstance(name, str) or not isinstance(dtype, str):
				raise TypeError("Replay field names and dtypes must be strings.")
			if (
				not isinstance(shape, list)
				or any(
					isinstance(value, bool)
					or not isinstance(value, int)
					or value < 0
					for value in shape
				)
			):
				raise ValueError("Replay field shapes must contain non-negative integers.")
			field_specs[name] = {"name": name, "shape": shape, "dtype": dtype}
		if len(field_specs) != len(specs):
			raise ValueError("Replay field names must be unique.")
		if metadata["initialized"]:
			if rows <= 0 or not field_specs or metadata["torchrl"] is None:
				raise ValueError("Initialized sharded replay metadata is incomplete.")
			expected_specs = self._configured_field_specs()
			if field_specs != expected_specs:
				raise ValueError(
					"Sharded replay field specifications are incompatible with the "
					f"configured contract: checkpoint={field_specs}, "
					f"configured={expected_specs}."
				)
			torchrl = require_exact_keys(
				metadata["torchrl"],
				{"sampler", "writer", "transforms", "batch_size"},
				"sharded replay TorchRL metadata",
			)
			if torchrl["batch_size"] != self._batch_size:
				raise ValueError("Sharded replay TorchRL batch size is incompatible.")
			writer = require_exact_keys(
				torchrl["writer"], {"_cursor"}, "sharded replay writer state"
			)
			cursor = writer["_cursor"]
			if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor < self._capacity:
				raise ValueError("Sharded replay writer cursor is invalid.")
		else:
			if rows != 0 or field_specs or metadata["torchrl"] is not None:
				raise ValueError("Uninitialized sharded replay must contain no storage.")
		self._preflight_accounting(metadata, candidate_size=rows)
		if metadata["initialized"]:
			self._preflight_writer_cursor(
				metadata["torchrl"]["writer"]["_cursor"], metadata
			)
		return metadata, field_specs

	def load_training_state_shards(self, metadata, shards):
		"""Transactionally stream validated physical replay shards into storage."""
		if not self._resumable_storage:
			raise RuntimeError(
				"Replay restore requires enable_resumable_storage() before use."
			)
		metadata, field_specs = self._preflight_sharded_metadata(metadata)
		candidate = None
		expected_start = 0
		expected_index = 0
		if metadata["initialized"]:
			pin_memory = self._uses_pinned_staging("cpu", self._device)
			candidate = self._reserve_buffer(
				LazyTensorStorage(self._capacity, device="cpu"),
				sampler=self._make_sampler(),
				pin_memory=pin_memory,
			)

		for shard in shards:
			if candidate is None:
				raise ValueError("Uninitialized replay must not have data shards.")
			shard = require_exact_keys(
				shard,
				{"schema", "version", "index", "start", "stop", "fields"},
				f"replay shard {expected_index}",
			)
			if (
				shard["schema"] != "tdmpc2-replay-shard"
				or shard["version"] != metadata["version"]
			):
				raise ValueError("Unsupported replay shard schema/version.")
			if shard["index"] != expected_index or shard["start"] != expected_start:
				raise ValueError("Replay shards are missing, duplicated, or out of order.")
			stop = shard["stop"]
			if isinstance(stop, bool) or not isinstance(stop, int) or not expected_start < stop <= metadata["storage_rows"]:
				raise ValueError("Replay shard row range is invalid.")
			row_count = stop - expected_start
			fields = require_exact_keys(
				shard["fields"], field_specs, f"replay shard {expected_index} fields"
			)
			td_fields = {}
			for name, spec in field_specs.items():
				value = fields[name]
				if not torch.is_tensor(value):
					raise TypeError(f"Replay shard field {name!r} must be a tensor.")
				if value.device.type != "cpu":
					raise ValueError(f"Replay shard field {name!r} must be on CPU.")
				if list(value.shape) != [row_count, *spec["shape"]]:
					raise ValueError(f"Replay shard field {name!r} has the wrong shape.")
				if str(value.dtype) != spec["dtype"]:
					raise ValueError(f"Replay shard field {name!r} has the wrong dtype.")
				td_fields[name] = value
			candidate.extend(TensorDict(td_fields, batch_size=[row_count]))
			expected_start = stop
			expected_index += 1

		if expected_start != metadata["storage_rows"]:
			raise ValueError("Replay shard sequence ended before all rows were restored.")
		if candidate is not None:
			torchrl = metadata["torchrl"]
			candidate._sampler.load_state_dict(torchrl["sampler"])
			candidate._writer.load_state_dict(torchrl["writer"])
			candidate._transform.load_state_dict(torchrl["transforms"])
			candidate._batch_size = torchrl["batch_size"]
			if len(candidate) != metadata["storage_rows"]:
				raise ValueError("Streamed replay size differs from metadata.")

		self._commit_restored_state(metadata, candidate)
		return self

	def _preflight_accounting(self, state, *, candidate_size):
		integer_fields = (
			"num_eps",
			"num_transitions",
			"total_transitions",
			"resident_rows",
		)
		for key in integer_fields:
			value = state[key]
			if isinstance(value, bool) or not isinstance(value, int) or value < 0:
				raise ValueError(f"Replay {key} must be a non-negative integer.")
		rows = state["resident_episode_rows"]
		if not isinstance(rows, list):
			raise TypeError("Replay resident_episode_rows must be a list.")
		row_sum = transition_sum = 0
		for index, entry in enumerate(rows):
			if not isinstance(entry, (list, tuple)) or len(entry) != 3:
				raise ValueError(
					f"Replay resident episode entry {index} must contain three integers."
				)
			if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in entry):
				raise ValueError(
					f"Replay resident episode entry {index} contains invalid counters."
				)
			episode_rows, initial_rows, transitions = entry
			if initial_rows not in {0, 1} or transitions != episode_rows - initial_rows:
				raise ValueError(
					f"Replay resident episode entry {index} is internally inconsistent."
				)
			row_sum += episode_rows
			transition_sum += transitions
		if row_sum != state["resident_rows"] or row_sum != candidate_size:
			raise ValueError("Replay resident row accounting does not match storage.")
		if transition_sum != state["num_transitions"]:
			raise ValueError("Replay resident transition accounting is inconsistent.")
		if state["total_transitions"] < state["num_transitions"]:
			raise ValueError("Replay cumulative transitions precede resident transitions.")
		if len(rows) > state["num_eps"]:
			raise ValueError("Replay contains more resident episodes than submitted episodes.")
		if bool(state["initialized"]) != (state["num_eps"] > 0):
			raise ValueError("Replay initialization and episode counters disagree.")

	def _preflight_writer_cursor(self, cursor, state):
		"""Tie TorchRL's physical write cursor to cumulative submitted rows."""
		if self._capacity <= 0:
			raise ValueError("Initialized replay cannot have zero capacity.")
		expected = (state["total_transitions"] + state["num_eps"]) % self._capacity
		if cursor != expected:
			raise ValueError(
				"Replay writer cursor is inconsistent with cumulative submitted rows: "
				f"checkpoint={cursor}, expected={expected}."
			)

	def _commit_restored_state(self, state, candidate):
		if candidate is None:
			if hasattr(self, "_buffer"):
				del self._buffer
			self._sampler = self._make_sampler()
		else:
			self._buffer = candidate
			self._sampler = candidate._sampler
			self._storage_device = torch.device("cpu")
			self._pin_memory = self._uses_pinned_staging("cpu", self._device)
		self._num_eps = int(state["num_eps"])
		self._num_transitions = int(state["num_transitions"])
		self._total_transitions = int(state["total_transitions"])
		self._resident_episode_rows = deque(
			list(entry) for entry in state["resident_episode_rows"]
		)
		self._resident_rows = int(state["resident_rows"])
		self._invalidate_transition_index()
