"""Observational action/return diagnostics shared by AMBI and TD-MPC2.

Only real executed actions and completed training episodes enter this state.
Reading/logging a snapshot never advances it or consumes random numbers.
"""

from collections import deque
from collections.abc import Mapping
import math

import numpy as np


class TrainingDiagnostics:
    RETURN_WINDOW = 20

    def __init__(self):
        self._previous_action = None
        self._returns = deque(maxlen=2 * self.RETURN_WINDOW)
        self._episodes = 0
        self._peak_window_mean = None

    def record_action(self, action, accumulator, *, episode_start=False):
        """Collect executed normalized actions without crossing episode resets.

        Fractions pool finite coordinates; nonfinite fraction uses all coordinates.
        Delta RMS is computed per decision and averaged over eligible decisions.
        Both flip fractions use all finite within-episode coordinate pairs as their
        denominator, including zeros (which do not count as sign flips).
        """
        action = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        if episode_start:
            self._previous_action = None
        if not action.size:
            return
        prefix = "train/executed_action_"
        finite = np.isfinite(action)
        coordinates = int(finite.sum())
        accumulator.add_sum(prefix + "samples", 1)
        accumulator.add_sum(prefix + "finite_coordinates", coordinates)
        accumulator.add_sum(prefix + "transition_samples", 0)
        accumulator.add_sum(prefix + "transition_coordinates", 0)
        accumulator.add_weighted(
            prefix + "nonfinite_fraction", float((~finite).mean()), action.size
        )
        if coordinates:
            absolute = np.abs(action[finite])
            accumulator.add_weighted(
                prefix + "abs_ge_0p99_fraction", float((absolute >= .99).mean()),
                coordinates,
            )
            accumulator.add_weighted(
                prefix + "exact_saturation_fraction", float((absolute == 1.).mean()),
                coordinates,
            )
        previous = self._previous_action
        if previous is not None and previous.shape == action.shape:
            paired = finite & np.isfinite(previous)
            pairs = int(paired.sum())
            if pairs:
                current, old = action[paired], previous[paired]
                flips = np.signbit(current) != np.signbit(old)
                flips &= (current != 0) & (old != 0)
                bound_flips = flips & (np.abs(current) >= .99) & (np.abs(old) >= .99)
                accumulator.add_sum(prefix + "transition_samples", 1)
                accumulator.add_sum(prefix + "transition_coordinates", pairs)
                accumulator.add_weighted(
                    prefix + "delta_rms", float(np.sqrt(np.mean((current - old) ** 2)))
                )
                accumulator.add_weighted(prefix + "sign_flip_fraction", float(flips.mean()), pairs)
                accumulator.add_weighted(prefix + "bound_flip_fraction", float(bound_flips.mean()), pairs)
        self._previous_action = action

    def complete_episode(self, episode_return):
        """Record once at completion, irrespective of W&B enablement/cadence."""
        self._previous_action = None
        episode_return = float(episode_return)
        if not math.isfinite(episode_return):
            # Invalid returns cannot define a contiguous rolling window.
            self._returns.clear()
            self._episodes = 0
            self._peak_window_mean = None
            return
        self._episodes += 1
        self._returns.append(episode_return)
        if len(self._returns) >= self.RETURN_WINDOW:
            mean = float(np.mean(list(self._returns)[-self.RETURN_WINDOW:]))
            self._peak_window_mean = mean if self._peak_window_mean is None else max(
                self._peak_window_mean, mean
            )

    def episode_metrics(self):
        prefix = "episode/return_"
        latest = np.asarray(list(self._returns)[-self.RETURN_WINDOW:], dtype=np.float64)
        payload = {
            prefix + "diagnostic_episodes": self._episodes,
            prefix + "window20_count": len(latest),
            prefix + "window20_ready": int(len(latest) == self.RETURN_WINDOW),
            prefix + "disjoint20_ready": int(len(self._returns) == 2 * self.RETURN_WINDOW),
        }
        if not len(latest):
            return payload
        mean = float(latest.mean())
        payload.update({
            prefix + "window20_mean": mean,
            prefix + "window20_std": float(latest.std()),
            prefix + "window20_min": float(latest.min()),
        })
        if self._peak_window_mean is not None:
            drawdown = max(0., self._peak_window_mean - mean)
            payload[prefix + "window20_peak_mean"] = self._peak_window_mean
            payload[prefix + "window20_drawdown"] = drawdown
            if self._peak_window_mean > 0:
                payload[prefix + "window20_drawdown_fraction"] = drawdown / self._peak_window_mean
        if len(self._returns) == 2 * self.RETURN_WINDOW:
            previous = float(np.mean(list(self._returns)[:self.RETURN_WINDOW]))
            payload[prefix + "previous20_mean"] = previous
            payload[prefix + "disjoint20_change"] = mean - previous
            if previous > 0:
                payload[prefix + "disjoint20_drop_fraction"] = max(0., previous - mean) / previous
        return payload

    def state_dict(self):
        if self._previous_action is not None:
            raise RuntimeError("Training diagnostics require an episode boundary.")
        return {
            "version": 1,
            "episodes": self._episodes,
            "recent_returns": list(self._returns),
            "peak_window_mean": self._peak_window_mean,
        }

    @classmethod
    def from_state_dict(cls, state):
        """Validate into a fresh candidate; never mutate live training state."""
        if not isinstance(state, Mapping) or set(state) != {
            "version", "episodes", "recent_returns", "peak_window_mean"
        } or state["version"] != 1:
            raise ValueError("Unsupported training diagnostics state.")
        episodes = state["episodes"]
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 0:
            raise ValueError("Training diagnostics episode count is invalid.")
        returns = state["recent_returns"]
        if not isinstance(returns, list) or len(returns) != min(episodes, 2 * cls.RETURN_WINDOW):
            raise ValueError("Training diagnostics return history length is invalid.")
        def finite_number(value):
            return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        if not all(finite_number(value) for value in returns):
            raise ValueError("Training diagnostics return history must be finite.")
        peak = state["peak_window_mean"]
        if episodes < cls.RETURN_WINDOW:
            if peak is not None:
                raise ValueError("Training diagnostics peak precedes a complete window.")
        elif not finite_number(peak) or peak < float(np.mean(returns[-cls.RETURN_WINDOW:])):
            raise ValueError("Training diagnostics rolling peak is invalid.")
        candidate = cls()
        candidate._episodes = episodes
        candidate._returns.extend(float(value) for value in returns)
        candidate._peak_window_mean = None if peak is None else float(peak)
        return candidate
