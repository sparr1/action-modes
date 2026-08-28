"""Isolated paired outer-versus-fresh-inner controller evaluation.

The evaluator runs deterministic outer-only and fresh-inner AMBI episodes in
two independently constructed environments with paired reset seeds.  It owns a
dedicated inner engine and temporarily installs that engine on the supplied
agent, so evaluation cannot consume or mutate the live training engine.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .inner_improvement import InnerImprovementEngine


_TORCH_SEED_MAX = (1 << 63) - 1
_QUANTILES = (
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
)
_Q_GAIN_KEY = "inner_fixed_target_q_action_gain"
_Q_GAIN_STEM = "eval/paired_fresh_inner_fixed_target_q_action_gain"


@dataclass(frozen=True)
class _GlobalRNGState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...] | None


def _capture_global_rng() -> _GlobalRNGState:
    cuda_state = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_state = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return _GlobalRNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch_cpu=torch.random.get_rng_state().clone(),
        torch_cuda=cuda_state,
    )


def _restore_global_rng(state: _GlobalRNGState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        torch.cuda.set_rng_state_all(list(state.torch_cuda))


def _namespaced_seed(seed: int, *parts: object) -> int:
    """Derive a stable, process-independent evaluator substream seed."""

    digest = hashlib.blake2b(digest_size=8, person=b"AMBI-paired-eval")
    digest.update(str(int(seed)).encode("ascii"))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little") & _TORCH_SEED_MAX


def _finite_float(value: object, name: str) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise TypeError(f"{name} must be a scalar finite number.")
        value = value.detach().cpu().item()
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a scalar finite number.") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


def _nonnegative_float(value: object, name: str) -> float:
    resolved = _finite_float(value, name)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return resolved


def _mean_std(values: list[float], name: str) -> tuple[float, float]:
    if not values:
        raise ValueError(f"{name} must contain at least one observation.")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must contain only finite scalar observations.")
    return float(array.mean()), float(array.std(ddof=0))


class PairedControllerEvaluator:
    """Compare deterministic outer control with deterministic fresh-inner AMBI.

    Every episode index receives one explicit environment reset seed shared by
    the two auxiliary environments.  The evaluator-owned inner engine is reset
    from a distinct namespaced controller seed before the corresponding
    fresh-inner episode.  Episode returns are undiscounted.
    """

    def __init__(
        self,
        agent: Any,
        env_factory: Callable[[], Any],
        observation_to_tensor: Callable[[Any], torch.Tensor],
        unscale_action: Callable[[np.ndarray], Any],
        episodes: int,
        seed: int,
        device: torch.device | str,
    ) -> None:
        if not callable(env_factory):
            raise TypeError("env_factory must be callable.")
        if not callable(observation_to_tensor):
            raise TypeError("observation_to_tensor must be callable.")
        if not callable(unscale_action):
            raise TypeError("unscale_action must be callable.")
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
            raise ValueError("episodes must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        for attribute in (
            "act",
            "act_outer_policy",
            "eval",
            "modules",
        ):
            if not callable(getattr(agent, attribute, None)):
                raise TypeError(f"agent must provide callable {attribute}().")
        for attribute in (
            "inner_engine",
            "last_inner_metrics",
            "last_inner_rollout_lengths",
        ):
            if not hasattr(agent, attribute):
                raise TypeError(f"agent must expose {attribute}.")

        self.agent = agent
        self.env_factory = env_factory
        self.observation_to_tensor = observation_to_tensor
        self.unscale_action = unscale_action
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise NotImplementedError(
                "Paired controller evaluation supports CPU and CUDA agents only."
            )

        # Construction is observational too.  Keep a defensive RNG boundary in
        # case a future engine constructor gains randomized setup.
        rng_state = _capture_global_rng()
        try:
            self._evaluation_inner_engine = InnerImprovementEngine(agent)
        finally:
            _restore_global_rng(rng_state)
        if self._evaluation_inner_engine is agent.inner_engine:
            raise RuntimeError("The evaluation and live inner engines must be distinct.")

        self._outer_env: Any | None = None
        self._inner_env: Any | None = None
        self._closed = False
        self._evaluating = False

    def _environment(self, controller: str) -> Any:
        if self._closed:
            raise RuntimeError("PairedControllerEvaluator is closed.")
        if controller == "outer":
            attribute = "_outer_env"
            other = self._inner_env
        elif controller == "fresh_inner":
            attribute = "_inner_env"
            other = self._outer_env
        else:  # Internal call sites use only the two fixed controller names.
            raise AssertionError(f"Unknown paired controller {controller!r}.")

        env = getattr(self, attribute)
        if env is None:
            env = self.env_factory()
            if env is None:
                raise TypeError("env_factory returned None.")
            if env is other:
                raise ValueError(
                    "env_factory must return independent outer and fresh-inner "
                    "environment instances."
                )
            setattr(self, attribute, env)
        return env

    def _seed(self, namespace: str, episode: int) -> int:
        return _namespaced_seed(self.seed, "paired_controller", namespace, episode)

    def _reset(self, controller: str, episode: int) -> Any:
        # Gymnasium reset seeds are unsigned 32-bit values.  Both paired envs
        # receive exactly the same explicitly namespaced episode seed.
        reset_seed = self._seed("environment_reset", episode) & 0xFFFFFFFF
        result = self._environment(controller).reset(seed=reset_seed)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("Paired controller environments must use Gymnasium reset().")
        return result[0]

    def _observation(self, observation: Any) -> torch.Tensor:
        tensor = self.observation_to_tensor(observation)
        if not torch.is_tensor(tensor):
            raise TypeError("observation_to_tensor must return a torch.Tensor.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("The paired controller observation must be finite.")
        return tensor.detach()

    def _paired_initial_observations(self, episode: int) -> tuple[Any, Any]:
        outer = self._reset("outer", episode)
        fresh_inner = self._reset("fresh_inner", episode)
        outer_tensor = self._observation(outer)
        inner_tensor = self._observation(fresh_inner)
        if (
            outer_tensor.shape != inner_tensor.shape
            or outer_tensor.dtype != inner_tensor.dtype
            or not torch.equal(outer_tensor, inner_tensor)
        ):
            raise ValueError(
                "Paired controller environments produced different initial "
                "observations for the same reset seed."
            )
        return outer, fresh_inner

    def _environment_action(self, action: Any) -> Any:
        if not torch.is_tensor(action):
            raise TypeError("AMBI controllers must return torch.Tensor actions.")
        if action.numel() == 0 or not bool(torch.isfinite(action).all()):
            raise ValueError("AMBI controllers must return non-empty finite actions.")
        normalized = action.detach().cpu().numpy()
        env_action = self.unscale_action(normalized)
        try:
            array = np.asarray(env_action, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("unscale_action must return a numeric action.") from exc
        if array.size == 0 or not bool(np.isfinite(array).all()):
            raise ValueError("unscale_action must return a non-empty finite action.")
        return env_action

    @staticmethod
    def _transition(env: Any, action: Any) -> tuple[Any, float, bool]:
        result = env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("Paired controller environments must use Gymnasium step().")
        observation, reward, terminated, truncated, _ = result
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
            truncated, (bool, np.bool_)
        ):
            raise TypeError("Environment termination and truncation flags must be bool.")
        return (
            observation,
            _finite_float(reward, "environment reward"),
            bool(terminated or truncated),
        )

    def _outer_episode(self, observation: Any) -> float:
        env = self._environment("outer")
        episode_reward = 0.0
        while True:
            action = self.agent.act_outer_policy(
                self._observation(observation),
                deterministic=True,
            )
            observation, reward, done = self._transition(
                env, self._environment_action(action)
            )
            episode_reward = _finite_float(
                episode_reward + reward, "outer episode reward"
            )
            if done:
                return episode_reward

    def _fresh_inner_episode(
        self,
        episode: int,
        observation: Any,
        q_gains: list[float],
        model_steps: list[float],
        control_seconds: list[float],
        diagnostic_seconds: list[float],
    ) -> float:
        env = self._environment("fresh_inner")
        self._evaluation_inner_engine.reset_for_evaluation(
            self._seed("fresh_inner_engine", episode)
        )
        episode_reward = 0.0
        first_action = True
        while True:
            action = self.agent.act(
                self._observation(observation),
                t0=first_action,
                eval_mode=True,
                collect_diagnostics=True,
                apply_inner_writeback=False,
            )
            first_action = False
            metrics = self.agent.last_inner_metrics
            if not isinstance(metrics, dict):
                raise TypeError("agent.last_inner_metrics must be a dictionary.")
            if _Q_GAIN_KEY not in metrics:
                raise KeyError(
                    "Fresh-inner evaluation requires diagnostic metric "
                    f"{_Q_GAIN_KEY!r}."
                )
            for key in (
                "inner_model_steps",
                "inner_action_seconds",
                "inner_diagnostic_seconds",
            ):
                if key not in metrics:
                    raise KeyError(
                        f"Fresh-inner evaluation requires cost metric {key!r}."
                    )

            q_gains.append(_finite_float(metrics[_Q_GAIN_KEY], _Q_GAIN_KEY))
            model_steps.append(
                _nonnegative_float(
                    metrics["inner_model_steps"], "inner_model_steps"
                )
            )
            action_seconds = _nonnegative_float(
                metrics["inner_action_seconds"], "inner_action_seconds"
            )
            diagnostics = _nonnegative_float(
                metrics["inner_diagnostic_seconds"], "inner_diagnostic_seconds"
            )
            control_seconds.append(max(0.0, action_seconds - diagnostics))
            diagnostic_seconds.append(diagnostics)

            observation, reward, done = self._transition(
                env, self._environment_action(action)
            )
            episode_reward = _finite_float(
                episode_reward + reward, "fresh-inner episode reward"
            )
            if done:
                return episode_reward

    @staticmethod
    def _q_gain_metrics(q_gains: list[float]) -> dict[str, float]:
        if not q_gains:
            raise ValueError("Fresh-inner evaluation produced no root-Q gains.")
        values = np.asarray(q_gains, dtype=np.float64)
        if values.ndim != 1 or not bool(np.isfinite(values).all()):
            raise ValueError("Fresh-inner root-Q gains must all be finite scalars.")
        mean = float(values.mean())
        metrics = {
            _Q_GAIN_STEM: mean,
            f"{_Q_GAIN_STEM}_count": float(values.size),
            f"{_Q_GAIN_STEM}_mean": mean,
            f"{_Q_GAIN_STEM}_std": float(values.std(ddof=0)),
            f"{_Q_GAIN_STEM}_min": float(values.min()),
            f"{_Q_GAIN_STEM}_max": float(values.max()),
            f"{_Q_GAIN_STEM}_positive_fraction": float(np.mean(values > 0.0)),
        }
        for suffix, quantile in _QUANTILES:
            metrics[f"{_Q_GAIN_STEM}_{suffix}"] = float(
                np.quantile(values, quantile, method="linear")
            )
        return metrics

    def evaluate(self) -> dict[str, float]:
        """Run the fixed paired episode bank without altering live agent state."""

        if self._closed:
            raise RuntimeError("PairedControllerEvaluator is closed.")
        if self._evaluating:
            raise RuntimeError("PairedControllerEvaluator.evaluate() is not reentrant.")

        self._evaluating = True
        started = time.perf_counter()
        rng_state = _capture_global_rng()
        module_modes = tuple(
            (module, bool(module.training)) for module in self.agent.modules()
        )
        live_inner_engine = self.agent.inner_engine
        previous_metrics = self.agent.last_inner_metrics
        previous_lengths = self.agent.last_inner_rollout_lengths
        try:
            self.agent.inner_engine = self._evaluation_inner_engine
            self.agent.eval()

            outer_rewards: list[float] = []
            inner_rewards: list[float] = []
            q_gains: list[float] = []
            model_steps: list[float] = []
            control_seconds: list[float] = []
            diagnostic_seconds: list[float] = []
            for episode in range(self.episodes):
                outer_observation, inner_observation = (
                    self._paired_initial_observations(episode)
                )
                outer_rewards.append(self._outer_episode(outer_observation))
                inner_rewards.append(
                    self._fresh_inner_episode(
                        episode,
                        inner_observation,
                        q_gains,
                        model_steps,
                        control_seconds,
                        diagnostic_seconds,
                    )
                )

            outer_mean, outer_std = _mean_std(
                outer_rewards, "paired outer episode rewards"
            )
            inner_mean, inner_std = _mean_std(
                inner_rewards, "paired fresh-inner episode rewards"
            )
            deltas = [
                inner_reward - outer_reward
                for outer_reward, inner_reward in zip(outer_rewards, inner_rewards)
            ]
            delta_mean, delta_std = _mean_std(
                deltas, "paired fresh-inner-minus-outer rewards"
            )
            action_count = len(q_gains)
            if not (
                action_count
                == len(model_steps)
                == len(control_seconds)
                == len(diagnostic_seconds)
            ):
                raise RuntimeError("Fresh-inner action diagnostics became misaligned.")

            metrics = {
                "eval/paired_outer_episode_reward": outer_mean,
                "eval/paired_outer_episode_reward_std": outer_std,
                "eval/paired_fresh_inner_episode_reward": inner_mean,
                "eval/paired_fresh_inner_episode_reward_std": inner_std,
                "eval/paired_fresh_inner_minus_outer": delta_mean,
                "eval/paired_fresh_inner_minus_outer_std": delta_std,
                "eval/paired_fresh_inner_win_fraction": float(
                    np.mean(np.asarray(deltas, dtype=np.float64) > 0.0)
                ),
                "eval/paired_episodes": float(self.episodes),
                "eval/paired_fresh_inner_model_steps_per_action": float(
                    np.mean(np.asarray(model_steps, dtype=np.float64))
                ),
                "time/paired_fresh_inner_control_seconds_per_action": float(
                    np.mean(np.asarray(control_seconds, dtype=np.float64))
                ),
                "time/paired_fresh_inner_diagnostic_seconds_per_action": float(
                    np.mean(np.asarray(diagnostic_seconds, dtype=np.float64))
                ),
            }
            metrics.update(self._q_gain_metrics(q_gains))
            metrics["time/paired_inner_comparison_seconds"] = max(
                0.0, time.perf_counter() - started
            )
            return {
                key: _finite_float(value, f"paired controller metric {key}")
                for key, value in metrics.items()
            }
        finally:
            try:
                self.agent.inner_engine = live_inner_engine
                self.agent.last_inner_metrics = previous_metrics
                self.agent.last_inner_rollout_lengths = previous_lengths
                for module, was_training in module_modes:
                    module.training = was_training
            finally:
                try:
                    _restore_global_rng(rng_state)
                finally:
                    self._evaluating = False

    def close(self) -> None:
        """Close both auxiliary environments at most once."""

        if self._closed:
            return
        if self._evaluating:
            raise RuntimeError("Cannot close PairedControllerEvaluator during evaluation.")
        self._closed = True
        environments = (self._outer_env, self._inner_env)
        self._outer_env = None
        self._inner_env = None
        primary_error: BaseException | None = None
        for env in environments:
            if env is None:
                continue
            try:
                env.close()
            except BaseException as exc:
                if primary_error is None:
                    primary_error = exc
                else:
                    try:
                        primary_error.add_note(
                            "Additional paired environment close failure: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    except AttributeError:  # Python 3.10 lacks BaseException.add_note.
                        pass
        if primary_error is not None:
            raise primary_error


__all__ = ["PairedControllerEvaluator"]
