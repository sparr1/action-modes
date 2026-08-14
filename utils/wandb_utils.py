"""Optional W&B helpers used by AMBI-native algorithms.

Importing this module does not import wandb. W&B is imported only when a run is
explicitly enabled with `wandb: true` in an algorithm config.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np


DEFAULT_WANDB_ENTITY = "rwgao_b-brown-university"
DEFAULT_WANDB_PROJECT = "ambi"


def _finite_float(value) -> float | None:
    """Convert scalar-like values to finite floats without importing torch."""
    if isinstance(value, (str, bytes)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


@dataclass
class _WeightedMean:
    weighted_sum: float = 0.0
    weight: float = 0.0


@dataclass
class _Moments:
    count: float
    mean: float
    m2: float
    minimum: float
    maximum: float


class WandbAccumulator:
    """Accumulate scalar metrics between W&B emissions.

    The accumulator supports four intentionally distinct aggregation modes:

    * :meth:`add_weighted` computes a weighted arithmetic mean.
    * :meth:`add_sum` sums interval work counters.
    * :meth:`set_last` retains the latest gauge or cumulative counter.
    * :meth:`add_stats` pools population moments and emits ``_count``,
      ``_mean``, ``_std``, ``_min``, and ``_max`` keys for a metric prefix.

    Non-finite or non-scalar observations are ignored. A zero sum is retained,
    which lets callers explicitly report meaningful zero-work intervals.
    """

    def __init__(self):
        self.clear()

    @property
    def empty(self) -> bool:
        return not (self._weighted or self._sums or self._last or self._stats)

    def __bool__(self) -> bool:
        return not self.empty

    def clear(self) -> None:
        """Discard every accumulated observation."""
        self._weighted: dict[str, _WeightedMean] = {}
        self._sums: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._stats: dict[str, _Moments] = {}
        self._claimed_keys: dict[str, str] = {}

    def _claim(self, keys: tuple[str, ...], aggregation: str) -> None:
        for key in keys:
            previous = self._claimed_keys.get(key)
            if previous is not None and previous != aggregation:
                raise ValueError(
                    f"Metric {key!r} is already using {previous} aggregation; "
                    f"it cannot also use {aggregation} aggregation."
                )
        for key in keys:
            self._claimed_keys[key] = aggregation

    def add_weighted(self, key: str, value, weight=1.0) -> None:
        """Add one finite value to a weighted interval mean."""
        value = _finite_float(value)
        weight = _finite_float(weight)
        if value is None or weight is None or weight <= 0.0:
            return
        self._claim((key,), "weighted mean")
        metric = self._weighted.setdefault(key, _WeightedMean())
        metric.weighted_sum += value * weight
        metric.weight += weight

    def update_weighted(self, metrics: Mapping[str, object] | None, *, weight=1.0) -> None:
        """Add a mapping of values that all have the same positive weight."""
        for key, value in (metrics or {}).items():
            self.add_weighted(key, value, weight)

    def add_sum(self, key: str, value) -> None:
        """Add a finite value to an interval sum, retaining explicit zeros."""
        value = _finite_float(value)
        if value is None:
            return
        self._claim((key,), "sum")
        self._sums[key] = self._sums.get(key, 0.0) + value

    def update_sums(self, metrics: Mapping[str, object] | None) -> None:
        """Add each value in a mapping to its interval sum."""
        for key, value in (metrics or {}).items():
            self.add_sum(key, value)

    def set_last(self, key: str, value) -> None:
        """Retain the latest finite value for a gauge or cumulative counter."""
        value = _finite_float(value)
        if value is None:
            return
        self._claim((key,), "last value")
        self._last[key] = value

    def update_last(self, metrics: Mapping[str, object] | None) -> None:
        """Set the latest value for each metric in a mapping."""
        for key, value in (metrics or {}).items():
            self.set_last(key, value)

    @staticmethod
    def _summarize_values(values) -> _Moments | None:
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "numpy"):
            values = values.numpy()
        elif not np.isscalar(values) and not isinstance(values, np.ndarray):
            try:
                values = list(values)
            except TypeError:
                pass

        try:
            array = np.asarray(values, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            return None
        array = array[np.isfinite(array)]
        if array.size == 0:
            return None
        mean = float(array.mean())
        centered = array - mean
        return _Moments(
            count=float(array.size),
            mean=mean,
            m2=float(np.dot(centered, centered)),
            minimum=float(array.min()),
            maximum=float(array.max()),
        )

    @staticmethod
    def _summarize_moments(*, count, mean, std, min_value, max_value) -> _Moments | None:
        count = _finite_float(count)
        mean = _finite_float(mean)
        std = 0.0 if std is None else _finite_float(std)
        min_value = mean if min_value is None else _finite_float(min_value)
        max_value = mean if max_value is None else _finite_float(max_value)
        if (
            count is None
            or count <= 0.0
            or mean is None
            or std is None
            or std < 0.0
            or min_value is None
            or max_value is None
            or min_value > max_value
        ):
            return None
        return _Moments(
            count=count,
            mean=mean,
            m2=std * std * count,
            minimum=min_value,
            maximum=max_value,
        )

    def add_stats(
        self,
        prefix: str,
        values=None,
        *,
        count=None,
        mean=None,
        std=None,
        min_value=None,
        max_value=None,
    ) -> None:
        """Pool raw values or precomputed population moments.

        Pass either ``values`` or a ``count``/``mean`` summary. ``std`` is the
        population standard deviation (``ddof=0``); omitted summary standard
        deviations are treated as zero. Non-finite raw values are filtered.
        """
        summary_supplied = any(
            value is not None for value in (count, mean, std, min_value, max_value)
        )
        if values is not None and summary_supplied:
            raise ValueError("add_stats accepts raw values or summary moments, not both.")
        if values is None:
            if count is None or mean is None:
                return
            incoming = self._summarize_moments(
                count=count,
                mean=mean,
                std=std,
                min_value=min_value,
                max_value=max_value,
            )
        else:
            incoming = self._summarize_values(values)
        if incoming is None:
            return

        output_keys = tuple(
            f"{prefix}_{suffix}"
            for suffix in ("count", "mean", "std", "min", "max")
        )
        self._claim(output_keys, "pooled statistics")
        current = self._stats.get(prefix)
        if current is None:
            self._stats[prefix] = incoming
            return

        total_count = current.count + incoming.count
        delta = incoming.mean - current.mean
        combined_mean = current.mean + delta * incoming.count / total_count
        combined_m2 = (
            current.m2
            + incoming.m2
            + delta * delta * current.count * incoming.count / total_count
        )
        self._stats[prefix] = _Moments(
            count=total_count,
            mean=combined_mean,
            m2=combined_m2,
            minimum=min(current.minimum, incoming.minimum),
            maximum=max(current.maximum, incoming.maximum),
        )

    def snapshot(self, *, clear: bool = False) -> dict[str, float]:
        """Return the current finite metric payload, optionally clearing it."""
        payload = {
            key: metric.weighted_sum / metric.weight
            for key, metric in self._weighted.items()
            if metric.weight > 0.0
        }
        payload.update(self._sums)
        payload.update(self._last)
        for prefix, metric in self._stats.items():
            variance = max(0.0, metric.m2 / metric.count)
            payload.update(
                {
                    f"{prefix}_count": metric.count,
                    f"{prefix}_mean": metric.mean,
                    f"{prefix}_std": math.sqrt(variance),
                    f"{prefix}_min": metric.minimum,
                    f"{prefix}_max": metric.maximum,
                }
            )
        if clear:
            self.clear()
        return payload

    def pop(self) -> dict[str, float]:
        """Return and clear the current metric payload."""
        return self.snapshot(clear=True)


_COMPONENT_SEPARATOR = re.compile(r"[^0-9A-Za-z]+")


def _reward_component_name(key) -> str | None:
    component = _COMPONENT_SEPARATOR.sub("_", str(key)).strip("_").lower()
    if component.startswith("reward_"):
        component = component[len("reward_") :]
    return component or None


def _finite_reward_scalar(value) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    return _finite_float(value)


def extract_reward_components(info: Mapping | None) -> dict[str, float]:
    """Extract finite scalar reward components from one environment info dict.

    Numeric values in nested ``reward_info`` and top-level ``reward_*`` fields
    become ``rollout/reward_<component>`` metrics. Top-level fields take
    precedence on normalized-name collisions. Booleans are not rewards.
    """
    if not isinstance(info, Mapping):
        return {}

    payload: dict[str, float] = {}
    nested = info.get("reward_info")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            value = _finite_reward_scalar(value)
            component = _reward_component_name(key)
            if value is not None and component is not None:
                payload[f"rollout/reward_{component}"] = value

    for key, value in info.items():
        if (
            not isinstance(key, str)
            or not key.lower().startswith("reward_")
            or key.lower() == "reward_info"
        ):
            continue
        value = _finite_reward_scalar(value)
        component = _reward_component_name(key)
        if value is not None and component is not None:
            payload[f"rollout/reward_{component}"] = value
    return payload


def wandb_enabled(params: dict | None) -> bool:
    return bool(params and params.get("wandb", False))


def init_wandb(params: dict | None, *, default_project: str, run_name: str | None, config: dict | None = None):
    if not wandb_enabled(params):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb=True but the wandb package is not installed. Run `pip install wandb` or set wandb=false.") from exc

    project = params.get("wandb_project", default_project or DEFAULT_WANDB_PROJECT)
    entity = params.get("wandb_entity", DEFAULT_WANDB_ENTITY)
    name = params.get("wandb_run_name", run_name)
    mode = params.get("wandb_mode", None)
    tags = params.get("wandb_tags", None)
    group = params.get("wandb_group", None)

    kwargs = {"project": project, "entity": entity, "name": name, "config": config or {}}
    if mode:
        kwargs["mode"] = mode
    if tags:
        kwargs["tags"] = tags
    if group:
        kwargs["group"] = group

    run = wandb.init(**kwargs)
    wandb.define_metric("env_step")
    wandb.define_metric("train/*", step_metric="env_step")
    wandb.define_metric("rollout/*", step_metric="env_step")
    wandb.define_metric("episode/*", step_metric="env_step")
    wandb.define_metric("time/*", step_metric="env_step")
    wandb.define_metric("eval/*", step_metric="env_step")
    return run


def log_wandb(run, payload: dict, *, step: int) -> None:
    if run is None:
        return
    payload = dict(payload)
    payload.setdefault("env_step", int(step))
    run.log(payload, step=int(step))


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()
