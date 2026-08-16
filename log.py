import csv
import json
import os
import queue
import threading

import numpy as np


# We have a few different settings for logs: "none", "overwrite", "warn", and
# "timestamp". See main.py for the directory-handling policy. This module owns
# only the per-trial log streams.

_STEP_FIELDS = [
    "global_step", "episode", "episode_step", "reward", "episode_return",
    "done", "terminated", "truncated", "inner_steps",
]


def _fsync_regular_file(path):
    """Make one already-written log file durable without changing its contents."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path):
    """Persist directory entries created by the segment log writers."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_float(value, default=0.0):
    if isinstance(value, (list, tuple, np.ndarray)):
        values = _as_list(value)
        return _as_float(values[0], default) if values else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _done_value(dones):
    dones = _as_list(dones)
    if not dones:
        return False
    total = np.sum(dones)
    return bool(total.item() if hasattr(total, "item") else total)


def _sum_numeric(value):
    if isinstance(value, np.ndarray):
        try:
            return float(np.sum(value, dtype=np.float64))
        except (TypeError, ValueError):
            return sum(_sum_numeric(item) for item in value)
    if isinstance(value, (list, tuple)):
        return sum(_sum_numeric(item) for item in value)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_values(value):
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return True


def _convert_arrays_recursively(obj):
    """Convert trajectory values only when detailed logging needs them."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _convert_arrays_recursively(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_arrays_recursively(value) for value in obj]
    return obj


def _trajectory_list(value):
    """Match utils.listify's historical detailed-trajectory representation."""
    if isinstance(value, dict):
        return [{key: _convert_arrays_recursively(item) for key, item in value.items()}]
    converted = _convert_arrays_recursively(value)
    return converted if isinstance(converted, list) else [converted]


def _info_dict(item):
    if isinstance(item, list) and item and isinstance(item[0], dict):
        return item[0]
    if isinstance(item, dict):
        return item
    return {}


def _write_basic_summary(
    summary_file,
    episode_count,
    total_reward,
    num_steps,
    inner_steps=None,
    cumulative_steps=None,
):
    with open(summary_file, "a") as stream:
        line = (
            f"episode_{episode_count}: "
            f"Total Reward = {total_reward}, "
            f"Steps = {num_steps}"
        )
        if cumulative_steps is not None:
            line += f", Cumulative Steps = {cumulative_steps}"
        if inner_steps is not None:
            line += f", Inner Steps = {inner_steps}"
        stream.write(line + ",\n")


def _write_reward_info_summary(
    summary_file,
    episode_count,
    total_reward,
    num_steps,
    episode_info,
    cumulative_steps=None,
):
    first_info = _info_dict(episode_info[0])
    goal_key = [key for key in first_info if "desired" in key][0]
    goal = first_info[goal_key]
    base_reward = sum(_info_dict(item)["base"] for item in episode_info)
    healthy_bonus = sum(_info_dict(item)["healthy_bonus"] for item in episode_info)
    control_cost = sum(_info_dict(item)["control cost"] for item in episode_info)
    contact_cost = sum(_info_dict(item)["contact cost"] for item in episode_info)
    with open(summary_file, "a") as stream:
        stream.write(
            f"episode_{episode_count}: "
            f"Total Reward = {total_reward}, "
            f"Total Base = {base_reward}, "
            f"Total Healthy = {healthy_bonus}, "
            f"Total Control = {control_cost}, "
            f"Total Contact = {contact_cost}, "
            f"Goal = {goal}, Steps = {num_steps}"
            f"{', Cumulative Steps = ' + str(cumulative_steps) if cumulative_steps is not None else ''},\n"
        )


class _BufferedStepWriter:
    """Keep the step CSV open instead of reopening it for every environment step."""

    def __init__(self, path, buffer_size=64 * 1024):
        self.path = path
        needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
        self._stream = open(path, "a", newline="", buffering=buffer_size)
        self._writer = csv.DictWriter(self._stream, fieldnames=_STEP_FIELDS)
        if needs_header:
            self._writer.writeheader()

    def writerow(self, row):
        self._writer.writerow(row)

    def flush(self, *, durable=False):
        if not self._stream.closed:
            self._stream.flush()
            if durable:
                _fsync_regular_file(self.path)

    def close(self):
        if not self._stream.closed:
            error = None
            try:
                self._stream.flush()
            except BaseException as exc:
                error = exc
            try:
                self._stream.close()
            except BaseException as exc:
                if error is None:
                    error = exc
            if error is not None:
                raise error


class _BackgroundJSONWriter:
    """Serialize detailed episodes on one bounded background worker."""

    _STOP = object()

    def __init__(self, max_pending=2):
        self._queue = queue.Queue(maxsize=max_pending)
        self._error = None
        self._closed = False
        self._dirty_paths = set()
        self._dirty_paths_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="ambi-trajectory-writer",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                path, payload = item
                if self._error is None:
                    try:
                        with open(path, "w") as stream:
                            json.dump(payload, stream)
                        with self._dirty_paths_lock:
                            self._dirty_paths.add(os.path.abspath(path))
                    except BaseException as exc:  # surfaced on flush/close
                        self._error = exc
            finally:
                self._queue.task_done()

    def _raise_error(self):
        if self._error is not None:
            raise RuntimeError("Failed to write a detailed trajectory log.") from self._error

    def submit(self, path, payload):
        if self._closed:
            raise RuntimeError("Cannot submit to a closed trajectory writer.")
        self._raise_error()
        self._queue.put((path, payload))
        self._raise_error()

    def flush(self, *, durable=False):
        self._queue.join()
        self._raise_error()
        if not durable:
            return
        with self._dirty_paths_lock:
            dirty_paths = tuple(sorted(self._dirty_paths))
        for path in dirty_paths:
            _fsync_regular_file(path)
        for directory in sorted({os.path.dirname(path) for path in dirty_paths}):
            _fsync_directory(directory)
        with self._dirty_paths_lock:
            self._dirty_paths.difference_update(dirty_paths)

    def close(self):
        if self._closed:
            return
        error = None
        try:
            self.flush()
        except BaseException as exc:
            error = exc
        self._closed = True
        self._queue.put(self._STOP)
        self._thread.join()
        if error is not None:
            raise error
        self._raise_error()


class _BaseTrainingLogger:
    _include_inner_steps = False
    # ``setup_logs`` defaults to its historical materialized contract. These
    # built-in loggers can explicitly accept native numpy trajectory values and
    # defer conversion until detailed output actually consumes them.
    accepts_native_step_payload = True

    def __init__(self, log_dir=None, log_info=True, log_type="detailed"):
        if log_type not in ("detailed", "summary"):
            raise ValueError("log_type must be 'detailed' or 'summary'.")
        self._log_info = bool(log_info)
        self._log_type = log_type
        self._step_writer = None
        self._trajectory_writer = (
            _BackgroundJSONWriter() if self._log_type == "detailed" else None
        )
        self._closed = False
        if log_dir:
            self.set_log_dir(log_dir)
        self.reset()

    @property
    def retains_trajectories(self):
        return self._log_type == "detailed"

    def reset(self):
        self.reset_episode()
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.inner_step_count = 0.0

    def reset_episode(self):
        # These lists remain available for compatibility, but summary mode never
        # appends to them. High-level summaries use the running totals below.
        self.episode_rewards = []
        self.episode_observations = []
        self.episode_actions = []
        self.episode_info = []
        self.episode_inner_steps = []
        self.episode_step_count = 0
        self.episode_return = 0.0
        self.episode_num_steps = 0
        self.episode_inner_step_count = 0.0
        self._episode_has_inner_steps = False
        self._reward_info_seen = False
        self._reward_info_valid = True
        self._reward_info_goal = None
        self._reward_info_totals = {
            "base": 0.0,
            "healthy_bonus": 0.0,
            "control cost": 0.0,
            "contact cost": 0.0,
        }

    def resume_state_dict(self):
        """Return dynamic counters; the lineage fingerprint owns logger config."""
        if self.episode_step_count != 0 or self.episode_num_steps != 0:
            raise ValueError(
                "Training logger resume state can only be captured between episodes."
            )
        return {
            "schema_version": 2,
            "step_count": int(self.step_count),
            "episode_count": int(self.episode_count),
            "total_reward": float(self.total_reward),
            "inner_step_count": float(self.inner_step_count),
        }

    def load_resume_state_dict(self, state):
        """Restore absolute counters into a fresh, empty segment logger."""
        normalized = self.validate_resume_state_dict(state)
        if self.step_count or self.episode_count or self.episode_step_count:
            raise ValueError("Training logger must be fresh before resume state is loaded.")
        self.step_count = normalized["step_count"]
        self.episode_count = normalized["episode_count"]
        self.total_reward = normalized["total_reward"]
        self.inner_step_count = normalized["inner_step_count"]

    def validate_resume_state_dict(self, state):
        """Return normalized between-episode counters without mutating the logger."""
        fields = {
            "schema_version",
            "step_count",
            "episode_count",
            "total_reward",
            "inner_step_count",
        }
        if (
            not isinstance(state, dict)
            or set(state) != fields
            or state.get("schema_version") != 2
        ):
            raise ValueError("Unsupported training logger resume schema.")
        for key in ("step_count", "episode_count"):
            value = state.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Training logger {key} is invalid.")
        try:
            total_reward = float(state["total_reward"])
            inner_step_count = float(state["inner_step_count"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Training logger cumulative values must be numeric."
            ) from exc
        if not np.isfinite(total_reward) or not np.isfinite(inner_step_count):
            raise ValueError("Training logger cumulative values must be finite.")
        if inner_step_count < 0:
            raise ValueError("Training logger inner_step_count must be non-negative.")
        return {
            "step_count": int(state["step_count"]),
            "episode_count": int(state["episode_count"]),
            "total_reward": total_reward,
            "inner_step_count": inner_step_count,
        }

    def set_log_dir(self, log_dir):
        if self._step_writer is not None:
            self._step_writer.close()
        if self._log_type == "detailed" and self._trajectory_writer is None:
            self._trajectory_writer = _BackgroundJSONWriter()
        self.log_dir = log_dir
        self.summary_file = os.path.join(log_dir, "stats.txt")
        self.step_file = os.path.join(log_dir, "step_stats.csv")
        self.train_episodes_dir = os.path.join(log_dir, "train_episodes")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.train_episodes_dir, exist_ok=True)
        self._step_writer = _BufferedStepWriter(self.step_file)
        self._closed = False
        print("log directory set to ", log_dir)

    def _accumulate_reward_info(self, info):
        if not self._log_info:
            return
        info = _info_dict(info)
        if not info:
            self._reward_info_seen = True
            self._reward_info_valid = False
            return
        self._reward_info_seen = True
        if not self._reward_info_valid:
            return
        try:
            if self._reward_info_goal is None:
                goal_key = next(key for key in info if "desired" in key)
                self._reward_info_goal = info[goal_key]
            for key in self._reward_info_totals:
                self._reward_info_totals[key] += float(info[key])
        except (StopIteration, KeyError, TypeError, ValueError):
            self._reward_info_valid = False

    def _write_running_reward_info_summary(self):
        totals = self._reward_info_totals
        with open(self.summary_file, "a") as stream:
            stream.write(
                f"episode_{self.episode_count}: "
                f"Total Reward = {self.episode_return}, "
                f"Total Base = {totals['base']}, "
                f"Total Healthy = {totals['healthy_bonus']}, "
                f"Total Control = {totals['control cost']}, "
                f"Total Contact = {totals['contact cost']}, "
                f"Goal = {self._reward_info_goal}, Steps = {self.episode_num_steps}, "
                f"Cumulative Steps = {self.step_count},\n"
            )

    def _trajectory_payload(self):
        payload = {
            "rewards": self.episode_rewards,
            "observations": self.episode_observations,
            "actions": self.episode_actions,
            "info": self.episode_info,
        }
        if self._include_inner_steps:
            payload["inner_steps"] = self.episode_inner_steps
        payload["cumulative_step"] = self.step_count
        return payload

    def on_episode(self):
        self.episode_count += 1

        if self._log_type == "detailed":
            path = os.path.join(
                self.train_episodes_dir,
                f"episode_{self.episode_count}.json",
            )
            self._trajectory_writer.submit(path, self._trajectory_payload())

        inner_steps = None
        if self._include_inner_steps and self._episode_has_inner_steps:
            inner_steps = self.episode_inner_step_count

        if (
            self._log_info
            and self._reward_info_seen
            and self._reward_info_valid
            and self._reward_info_goal is not None
        ):
            self._write_running_reward_info_summary()
        else:
            _write_basic_summary(
                self.summary_file,
                self.episode_count,
                self.episode_return,
                self.episode_num_steps,
                inner_steps,
                cumulative_steps=self.step_count,
            )

        # Episode rows and summaries must be durable even during long runs.
        if self._step_writer is not None:
            self._step_writer.flush()
        self.reset_episode()

    def on_step(self, data):
        assert "dones" in data  # used for episode termination in the logger
        self.step_count += 1
        self.episode_step_count += 1

        rewards = _as_list(data.get("rewards"))
        reward_sum = _sum_numeric(rewards)
        self.episode_return += reward_sum
        self.total_reward += reward_sum
        self.episode_num_steps += len(rewards)

        if self._log_type == "detailed":
            self.episode_rewards.extend(_convert_arrays_recursively(rewards))
            self.episode_observations.extend(_trajectory_list(data.get("obs")))
            self.episode_actions.extend(_trajectory_list(data.get("actions")))

        infos = data.get("infos")
        if self._log_info and infos is not None:
            info_for_episode = _convert_arrays_recursively(infos)
            self._accumulate_reward_info(info_for_episode)
            if self._log_type == "detailed":
                self.episode_info.append(info_for_episode)

        raw_inner_steps = data.get("inner_steps")
        inner_step_sum = _sum_numeric(raw_inner_steps)
        if self._include_inner_steps and _has_values(raw_inner_steps):
            self._episode_has_inner_steps = True
            self.episode_inner_step_count += inner_step_sum
            self.inner_step_count += inner_step_sum
            if self._log_type == "detailed":
                self.episode_inner_steps.extend(
                    _as_list(_convert_arrays_recursively(raw_inner_steps))
                )

        done = _done_value(data["dones"])
        info = _info_dict(_as_list(infos)[0]) if infos else {}
        if self._step_writer is not None:
            self._step_writer.writerow({
                "global_step": self.step_count,
                "episode": self.episode_count + 1,
                "episode_step": self.episode_step_count,
                "reward": _as_float(rewards),
                "episode_return": self.episode_return,
                "done": int(done),
                "terminated": int(bool(info.get("terminated", done))),
                "truncated": int(bool(info.get("truncated", info.get("TimeLimit.truncated", False)))),
                "inner_steps": inner_step_sum if self._include_inner_steps else "",
            })
        if done:
            self.on_episode()

    def flush(self):
        if self._step_writer is not None:
            self._step_writer.flush()
        if self._trajectory_writer is not None:
            self._trajectory_writer.flush()

    def flush_durable(self):
        """Flush and fsync every segment-local training-log prefix."""

        if self._step_writer is not None:
            self._step_writer.flush(durable=True)
        if self._trajectory_writer is not None:
            self._trajectory_writer.flush(durable=True)
        if hasattr(self, "summary_file") and os.path.exists(self.summary_file):
            _fsync_regular_file(self.summary_file)
        if hasattr(self, "train_episodes_dir"):
            _fsync_directory(self.train_episodes_dir)
        if hasattr(self, "log_dir"):
            _fsync_directory(self.log_dir)

    def close(self):
        if self._closed:
            return
        error = None
        try:
            self.flush()
        except BaseException as exc:
            error = exc
        if self._step_writer is not None:
            try:
                self._step_writer.close()
            except BaseException as exc:
                if error is None:
                    error = exc
            finally:
                self._step_writer = None
        if self._trajectory_writer is not None:
            try:
                self._trajectory_writer.close()
            except BaseException as exc:
                if error is None:
                    error = exc
            finally:
                self._trajectory_writer = None
        self._closed = True
        if error is not None:
            raise error

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TrainingLogger(_BaseTrainingLogger):
    pass


class AMBITrainingLogger(_BaseTrainingLogger):
    _include_inner_steps = True
