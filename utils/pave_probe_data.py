"""Portable, episode-aware real transition banks for frozen PAVE probes.

NPZ banks contain only numeric arrays and one JSON Unicode scalar; loading never
enables pickle. Episode blocks may be resident replay fragments. Their explicit
boundaries, rather than a guessed timeout flag, delimit valid training windows.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from numbers import Integral
import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import torch


_ARRAY_FIELDS = (
    "observations", "actions", "next_observations", "rewards",
    "terminated", "truncated", "episode_ids", "episode_steps",
)
_SOURCE_KINDS = {"training_replay", "stochastic_prior_proxy"}


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _seed(value, name="seed"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _provenance(value):
    if not isinstance(value, dict):
        raise ValueError("Bank provenance must be a JSON object.")
    value = json.loads(json.dumps(value, allow_nan=False))
    version = value.setdefault("schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported PAVE probe-bank schema_version.")
    if value.get("source_kind") not in _SOURCE_KINDS:
        raise ValueError(f"source_kind must be one of {sorted(_SOURCE_KINDS)}.")
    if not isinstance(value.get("env_signature"), dict) or not value["env_signature"]:
        raise ValueError("Bank provenance requires a nonempty env_signature object.")
    checkpoint = value.get("source_checkpoint_sha256")
    if checkpoint is not None and (
        not isinstance(checkpoint, str) or re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint) is None
    ):
        raise ValueError("source_checkpoint_sha256 must be a SHA-256 hex digest.")
    return value


@dataclass(frozen=True)
class PaveProbeBank:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_ids: np.ndarray
    episode_steps: np.ndarray
    provenance: dict

    def __post_init__(self):
        for name in _ARRAY_FIELDS:
            array = np.asarray(getattr(self, name))
            if name in {"episode_ids", "episode_steps"}:
                if array.dtype.kind not in "iu" or np.any(array < 0):
                    raise ValueError(f"{name} must contain non-negative integers.")
                if np.any(array > np.iinfo(np.int64).max):
                    raise ValueError(f"{name} exceeds int64 range.")
                array = np.array(array, dtype=np.int64, copy=True, order="C")
            elif name in {"terminated", "truncated"}:
                if array.dtype.kind not in "bifu" or not np.all((array == 0) | (array == 1)):
                    raise ValueError(f"{name} must contain boolean flags.")
                array = np.array(array, dtype=bool, copy=True, order="C")
            else:
                if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                    raise ValueError(f"{name} must contain finite numeric values.")
                with np.errstate(over="ignore"):
                    array = np.array(array, dtype=np.float32, copy=True, order="C")
                if not np.isfinite(array).all():
                    raise ValueError(f"{name} must be finite in float32.")
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        object.__setattr__(self, "provenance", _provenance(self.provenance))
        n = len(self.observations) if self.observations.ndim else 0
        if n == 0 or self.observations.ndim != 2 or self.observations.shape[1] == 0:
            raise ValueError("observations must be nonempty [T,S].")
        if self.next_observations.shape != self.observations.shape:
            raise ValueError("next_observations must match observations [T,S].")
        if self.actions.ndim != 2 or self.actions.shape[0] != n or self.actions.shape[1] == 0:
            raise ValueError("actions must be nonempty [T,A].")
        if np.any(np.abs(self.actions) > 1):
            raise ValueError("actions must be normalized to [-1,1].")
        for name in ("rewards", "terminated", "truncated"):
            if getattr(self, name).shape != (n, 1):
                raise ValueError(f"{name} must have shape [T,1].")
        for name in ("episode_ids", "episode_steps"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape [T].")
        same_episode = self.episode_ids[1:] == self.episode_ids[:-1]
        starts = np.r_[True, ~same_episode]
        if np.unique(self.episode_ids[starts]).size != int(starts.sum()):
            raise ValueError("Each episode_id must occupy one contiguous block.")
        if np.any((self.terminated[:-1, 0] | self.truncated[:-1, 0]) & same_episode):
            raise ValueError("Episode boundary required after terminated or truncated rows.")
        if np.any((self.episode_steps[1:] - self.episode_steps[:-1])[same_episode] != 1):
            raise ValueError("episode_steps must advance by one within each block.")
        if not np.array_equal(self.next_observations[:-1][same_episode], self.observations[1:][same_episode]):
            raise ValueError("Observations must be contiguous within each episode block.")

    def __len__(self):
        return self.observations.shape[0]


def save_probe_bank(path, bank):
    """Atomically publish an NPZ without replacing any existing path."""
    path = Path(path)
    # Revalidate provenance, which is a caller-visible JSON mapping.
    provenance = _provenance(bank.provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pave-bank-", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **{name: getattr(bank, name) for name in _ARRAY_FIELDS},
                                provenance_json=np.asarray(json.dumps(provenance, sort_keys=True, allow_nan=False)))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_probe_bank(path):
    with np.load(path, allow_pickle=False) as archive:
        expected = {*_ARRAY_FIELDS, "provenance_json"}
        if set(archive.files) != expected or len(archive.files) != len(expected):
            raise ValueError("PAVE probe bank has missing, extra, or duplicate fields.")
        raw = archive["provenance_json"]
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("provenance_json must be a Unicode JSON scalar.")
        provenance = json.loads(raw.item())
        return PaveProbeBank(**{name: archive[name] for name in _ARRAY_FIELDS}, provenance=provenance)


def valid_window_starts(bank, horizon):
    """Find all H-transition windows wholly contained in one episode block."""
    horizon = _positive_int(horizon, "horizon")
    starts = np.arange(max(0, len(bank) - horizon + 1), dtype=np.int64)
    return starts[bank.episode_ids[starts] == bank.episode_ids[starts + horizon - 1]]


def sample_probe_windows(bank, *, horizon, batch_size, seed, device="cpu"):
    """Uniform windows with replacement, isolated from NumPy/Torch global RNG.

    Returns the ordinary four-tensor AMBI update input. Timeouts end windows but
    remain bootstrap-eligible; only genuine termination enters the final mask.
    """
    horizon = _positive_int(horizon, "horizon")
    batch_size = _positive_int(batch_size, "batch_size")
    starts = valid_window_starts(bank, horizon)
    if not len(starts):
        raise ValueError(f"Probe bank has no valid {horizon}-transition windows.")
    generator = np.random.default_rng(_seed(seed))
    draws = generator.choice(starts, size=batch_size, replace=True)
    rows = draws[None, :] + np.arange(horizon)[:, None]
    obs = np.concatenate((bank.observations[rows], bank.next_observations[rows[-1]][None]), axis=0)
    values = (obs, bank.actions[rows], bank.rewards[rows], bank.terminated[rows])
    return tuple(torch.tensor(value, dtype=torch.float32, device=device) for value in values)


@contextmanager
def _isolated_collection(model, device, seed):
    modes = [(module, module.training) for module in model.modules()]
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            # Some legacy wrappers use the global generators despite reset(seed).
            np.random.seed(seed % (2**32))
            random.seed(seed)
            model.eval()
            with torch.no_grad():
                yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        for module, mode in modes:
            module.training = mode


def collect_prior_bank(wrapper, env, *, seeds, max_steps, controller_seed, provenance):
    """Collect one stochastic-primary-prior episode per seed in a dedicated env.

    ``max_steps`` is a per-episode cap; forced caps are marked truncated and
    counted in provenance. The environment is advanced, but learner state,
    global RNG, mixed module modes, and optimizer state are left untouched.
    """
    max_steps = _positive_int(max_steps, "max_steps")
    controller_seed = _seed(controller_seed, "controller_seed")
    seeds = [_seed(seed, "episode seed") for seed in seeds]
    if not seeds:
        raise ValueError("At least one episode seed is required.")
    provenance = {**provenance, "source_kind": "stochastic_prior_proxy"}
    provenance = _provenance(provenance)
    if str(wrapper.cfg.obs) != "state":
        raise ValueError("PAVE probe collection requires state observations.")
    if not (np.array_equal(np.asarray(env.action_space.low).reshape(-1), wrapper._action_low)
            and np.array_equal(np.asarray(env.action_space.high).reshape(-1), wrapper._action_high)):
        raise ValueError("Collection environment action bounds differ from the checkpoint wrapper.")
    model = wrapper.agent.model
    device = torch.device(wrapper.agent.device)
    records = {name: [] for name in _ARRAY_FIELDS}
    forced = 0
    with _isolated_collection(model, device, controller_seed):
        for episode_id, episode_seed in enumerate(seeds):
            observation, _ = env.reset(seed=episode_seed)
            for step in range(max_steps):
                # CPU conversion may alias an environment's reused observation
                # array, which step() is allowed to mutate in place.
                current = wrapper._obs_to_tensor(observation).to(device).clone()
                action, _ = model.pi(model.encode(current.unsqueeze(0)))
                normalized = action[0].detach().cpu().numpy().copy()
                successor, reward, terminated, truncated, info = env.step(wrapper._unscale_action(normalized))
                if (terminated or truncated) and info.get("final_observation") is not None:
                    successor = info["final_observation"]
                if step + 1 == max_steps and not (terminated or truncated):
                    truncated = True
                    forced += 1
                values = (current.detach().cpu().numpy(), normalized,
                          wrapper._obs_to_tensor(successor).cpu().numpy(), [reward],
                          [terminated], [truncated], episode_id, step)
                for name, value in zip(_ARRAY_FIELDS, values):
                    records[name].append(np.array(value, copy=True))
                observation = successor
                if terminated or truncated:
                    break
    provenance.update(collection={"policy": "stochastic_primary_prior", "episode_seeds": seeds,
                                  "controller_seed": controller_seed,
                                  "max_steps_per_episode": max_steps,
                                  "forced_cap_truncations": forced})
    return PaveProbeBank(**{name: np.asarray(values) for name, values in records.items()}, provenance=provenance)


def export_buffer_bank(buffer, *, provenance):
    """Copy resident Buffer transitions without sampling or changing its RNG.

    The Buffer stores actions/rewards on successor rows and omits timeout flags.
    Chronological resident blocks preserve boundaries even after ring eviction;
    their first remaining row supplies an observation, never a fabricated action.
    """
    if getattr(buffer.cfg, "obs", "state") != "state" or getattr(buffer.cfg, "multitask", False):
        raise ValueError("Probe export supports single-task state replay only.")
    if buffer.size == 0:
        raise ValueError("Cannot export an empty replay buffer.")
    provenance = _provenance({**provenance, "source_kind": "training_replay"})
    cursor = int(buffer._buffer._writer._cursor) if buffer.size == buffer.capacity else 0
    records = {name: [] for name in _ARRAY_FIELDS}
    offset = 0
    for rows, _, _ in buffer._resident_episode_rows:
        indices = (torch.arange(rows, device=buffer._storage_device) + offset + cursor) % buffer.capacity
        offset += rows
        if rows < 2:
            continue
        resident = buffer._buffer._storage[indices]
        arrays = {key: resident[key].detach().cpu().numpy() for key in
                  ("obs", "action", "reward", "terminated", "episode")}
        ids = arrays["episode"].reshape(-1)
        if not np.all(ids == ids[0]):
            raise ValueError("Replay resident episode accounting does not match stored IDs.")
        n = rows - 1
        values = (arrays["obs"][:-1], arrays["action"][1:], arrays["obs"][1:],
                  arrays["reward"][1:].reshape(n, 1), arrays["terminated"][1:].reshape(n, 1),
                  np.zeros((n, 1), dtype=bool), np.full(n, ids[0], dtype=np.int64), np.arange(n))
        for name, value in zip(_ARRAY_FIELDS, values):
            records[name].append(value)
    if offset != buffer.size or not records["observations"]:
        raise ValueError("Replay contains no valid resident transition pairs or invalid row accounting.")
    provenance.update(export={"format": "tdmpc2_buffer_resident_pairs_v1",
                              "episode_steps": "relative_to_resident_block",
                              "truncation_flags": "not_stored_by_source_buffer"})
    return PaveProbeBank(**{name: np.concatenate(values) for name, values in records.items()}, provenance=provenance)
