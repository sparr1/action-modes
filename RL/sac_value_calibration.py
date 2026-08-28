"""Isolated real-environment value calibration for native SAC.

The paper-compatible deterministic protocol is retained as a deliberately
unmatched reference. The stochastic protocol is the correctness probe: it
pairs each online-Q query with the exact sampled action executed in the real
environment and accumulates SAC's entropy-augmented action return. At a
Gymnasium time limit, it also reports a target-critic tail correction because
native SAC masks true termination, not truncation, in its Bellman targets.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch


PAPER_DETERMINISTIC = "paper_deterministic"
STOCHASTIC_SOFT_BELLMAN = "stochastic_soft_bellman"
_PROTOCOLS = frozenset({PAPER_DETERMINISTIC, STOCHASTIC_SOFT_BELLMAN})
_Q_REDUCTIONS = frozenset({"min_pair", "mean_pair", "min_all", "mean_all"})
_TORCH_SEED_MAX = (1 << 63) - 1


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
    digest = hashlib.blake2b(digest_size=8, person=b"SAC-value-eval")
    digest.update(str(int(seed)).encode("ascii"))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little") & _TORCH_SEED_MAX


def _finite_float(value: object, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be one finite scalar.") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


class SACValueCalibrationEvaluator:
    """Compare native SAC critics with deterministic and matched soft returns."""

    def __init__(
        self,
        agent: Any,
        env_factory: Callable[[], Any],
        observation_to_array: Callable[[Any], Any],
        unscale_action: Callable[[np.ndarray], Any],
        discount: float,
        samples: int,
        seed: int,
        protocols: Iterable[str] | str,
        device: torch.device | str,
    ) -> None:
        if not callable(env_factory):
            raise TypeError("env_factory must be callable.")
        if not callable(observation_to_array):
            raise TypeError("observation_to_array must be callable.")
        if not callable(unscale_action):
            raise TypeError("unscale_action must be callable.")
        discount = _finite_float(discount, "discount")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1].")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise ValueError("samples must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        if isinstance(protocols, str):
            normalized_protocols = (protocols,)
        else:
            try:
                normalized_protocols = tuple(protocols)
            except TypeError as exc:
                raise TypeError("protocols must be a protocol name or iterable.") from exc
        if not normalized_protocols:
            raise ValueError("At least one SAC value-calibration protocol is required.")
        if any(not isinstance(protocol, str) for protocol in normalized_protocols):
            raise TypeError("SAC value-calibration protocol names must be strings.")
        if len(set(normalized_protocols)) != len(normalized_protocols):
            raise ValueError("SAC value-calibration protocols must be unique.")
        unknown = set(normalized_protocols) - _PROTOCOLS
        if unknown:
            raise ValueError(
                f"Unknown SAC value-calibration protocol(s) {sorted(unknown)}; "
                f"expected {sorted(_PROTOCOLS)}."
            )

        for attribute in ("sample_action_log_prob", "q_values"):
            if not callable(getattr(agent, attribute, None)):
                raise TypeError(f"agent must provide callable {attribute}().")
        for attribute in ("obs_dim", "action_dim"):
            dimension = getattr(agent, attribute, None)
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise ValueError(f"agent.{attribute} must be a positive integer.")
        for attribute in ("actor", "critic", "critic_target"):
            module = getattr(agent, attribute, None)
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"agent.{attribute} must be a torch.nn.Module.")
        if not hasattr(agent, "entropy_coefficient"):
            raise TypeError("agent must expose entropy_coefficient.")

        q_backend = getattr(agent, "q_backend", None)
        if q_backend is None:
            raise TypeError("agent must expose q_backend.")
        num_q = getattr(q_backend, "num_q", None)
        pair_size = getattr(q_backend, "pair_size", None)
        if (
            isinstance(num_q, bool)
            or not isinstance(num_q, int)
            or num_q < 2
        ):
            raise ValueError("SAC calibration requires at least two Q heads.")
        if (
            isinstance(pair_size, bool)
            or not isinstance(pair_size, int)
            or not 1 <= pair_size <= num_q
        ):
            raise ValueError("agent.q_backend has an invalid critic pair size.")
        if PAPER_DETERMINISTIC in normalized_protocols and pair_size != 2:
            raise ValueError("paper_deterministic requires q_pair_size=2.")

        config = getattr(agent, "config", None)
        target_reduction = getattr(config, "q_target_reduction", None)
        if target_reduction not in _Q_REDUCTIONS:
            raise ValueError(
                "agent.config.q_target_reduction must be one of "
                f"{sorted(_Q_REDUCTIONS)}."
            )

        self.agent = agent
        self.env_factory = env_factory
        self.observation_to_array = observation_to_array
        self.unscale_action = unscale_action
        self.discount = discount
        self.samples = int(samples)
        self.seed = int(seed)
        self.protocols = normalized_protocols
        self.device = torch.device(device)
        agent_device = torch.device(getattr(agent, "device", self.device))
        if agent_device != self.device:
            raise ValueError(
                "Evaluator and SAC agent devices must match, got "
                f"{self.device} and {agent_device}."
            )
        self.num_q = int(num_q)
        self.pair_size = int(pair_size)
        self.obs_dim = int(agent.obs_dim)
        self.action_dim = int(agent.action_dim)
        self.target_reduction = str(target_reduction)
        self._env: Any | None = None
        self._closed = False

    def _environment(self) -> Any:
        if self._closed:
            raise RuntimeError("SACValueCalibrationEvaluator is closed.")
        if self._env is None:
            env = self.env_factory()
            if env is None:
                raise TypeError("env_factory returned None.")
            self._env = env
        return self._env

    def _seed(self, protocol: str, namespace: str, sample: int) -> int:
        return _namespaced_seed(self.seed, protocol, namespace, int(sample))

    def _generator(
        self,
        protocol: str,
        namespace: str,
        sample: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Generator:
        generator = torch.Generator(device=self.device if device is None else device)
        generator.manual_seed(self._seed(protocol, namespace, sample))
        return generator

    def _reset(self, protocol: str, namespace: str, sample: int) -> Any:
        if sample == 0:
            reset_seed = self._seed(protocol, f"{namespace}/reset_batch", 0)
            result = self._environment().reset(seed=reset_seed & 0xFFFFFFFF)
        else:
            result = self._environment().reset()
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("SAC calibration environments must use Gymnasium reset().")
        return result[0]

    def _observation_tensor(self, observation: Any) -> torch.Tensor:
        converted = self.observation_to_array(observation)
        tensor = torch.as_tensor(converted, dtype=torch.float32, device=self.device)
        if tensor.ndim != 1:
            tensor = tensor.reshape(-1)
        if tensor.numel() != self.obs_dim:
            raise ValueError(
                "The flattened calibration observation must contain "
                f"{self.obs_dim} values, got {tensor.numel()}."
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("The calibration observation must be finite.")
        return tensor.unsqueeze(0)

    def _validated_action(self, action: Any) -> torch.Tensor:
        if not torch.is_tensor(action):
            raise TypeError("The SAC policy must return a torch.Tensor action.")
        if action.ndim != 2 or tuple(action.shape) != (1, self.action_dim):
            raise ValueError(
                "The SAC policy must return one batched action with shape "
                f"(1, {self.action_dim}), got "
                f"shape {tuple(action.shape)}."
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError("The SAC policy produced a non-finite action.")
        return action

    def _deterministic_action(self, observation: Any) -> torch.Tensor:
        action = self.agent.actor(
            self._observation_tensor(observation), deterministic=True
        )
        return self._validated_action(action)

    def _stochastic_action(
        self,
        observation: Any,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action, log_prob = self.agent.sample_action_log_prob(
            self._observation_tensor(observation), generator=generator
        )
        action = self._validated_action(action)
        if not torch.is_tensor(log_prob):
            raise TypeError("The SAC policy must return a torch.Tensor log-probability.")
        flattened = log_prob.reshape(-1)
        if flattened.numel() != 1 or not bool(torch.isfinite(flattened).all()):
            raise ValueError("The SAC policy log-probability must be one finite scalar.")
        return action, flattened[0]

    def _q_heads(
        self,
        observation: Any,
        action: torch.Tensor,
        *,
        target: bool,
    ) -> torch.Tensor:
        values = self.agent.q_values(
            self._observation_tensor(observation), action, target=target
        )
        if torch.is_tensor(values):
            if values.ndim < 2:
                raise ValueError("SAC Q values must have a leading head dimension.")
            stacked = values
        else:
            try:
                values = tuple(values)
            except TypeError as exc:
                raise TypeError("agent.q_values() must return Q-head tensors.") from exc
            if len(values) != self.num_q or any(
                not torch.is_tensor(value) for value in values
            ):
                raise ValueError(
                    f"agent.q_values() must return {self.num_q} tensor heads."
                )
            stacked = torch.stack(values, dim=0)
        if stacked.shape[0] != self.num_q:
            raise ValueError(f"Expected {self.num_q} Q heads, got {stacked.shape[0]}.")
        flattened = stacked.detach().reshape(self.num_q, -1)
        if flattened.shape[1] != 1:
            raise ValueError("Each Q head must produce one scalar for a single probe.")
        heads = flattened[:, 0]
        if not bool(torch.isfinite(heads).all()):
            raise ValueError("The SAC critic produced non-finite Q values.")
        return heads

    def _reduce_heads(
        self,
        heads: torch.Tensor,
        reduction: str,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if reduction.endswith("_all") or self.pair_size == self.num_q:
            selected = heads
        else:
            indices = torch.randperm(
                self.num_q, device=heads.device, generator=generator
            )[: self.pair_size]
            selected = heads.index_select(0, indices)
        if reduction.startswith("min_"):
            return selected.min()
        return selected.mean()

    def _env_action(self, action: torch.Tensor) -> Any:
        normalized = action[0].detach().cpu().numpy()
        env_action = self.unscale_action(normalized)
        try:
            finite = bool(np.isfinite(np.asarray(env_action)).all())
        except TypeError as exc:
            raise TypeError("unscale_action must return a numeric action.") from exc
        if not finite:
            raise ValueError("unscale_action produced a non-finite action.")
        return env_action

    def _step(self, action: torch.Tensor) -> tuple[Any, float, bool, bool]:
        transition = self._environment().step(self._env_action(action))
        if not isinstance(transition, tuple) or len(transition) != 5:
            raise TypeError("SAC calibration environments must use Gymnasium step().")
        observation, reward, terminated, truncated, _ = transition
        return (
            observation,
            _finite_float(reward, "environment reward"),
            bool(terminated),
            bool(truncated),
        )

    def _deterministic_rollout(self, observation: Any) -> float:
        discounted_return = 0.0
        discount = 1.0
        while True:
            action = self._deterministic_action(observation)
            observation, reward, terminated, truncated = self._step(action)
            discounted_return += discount * reward
            if not math.isfinite(discounted_return):
                raise ValueError("The deterministic Monte Carlo return became non-finite.")
            if terminated or truncated:
                return float(discounted_return)
            discount *= self.discount

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        return float(array.mean()), float(array.std(ddof=0))

    def _evaluate_paper_deterministic(self) -> dict[str, float]:
        mc_values = [
            self._deterministic_rollout(
                self._reset(PAPER_DETERMINISTIC, "mc", sample)
            )
            for sample in range(self.samples)
        ]

        q_values: list[float] = []
        for sample in range(self.samples):
            observation = self._reset(PAPER_DETERMINISTIC, "q", sample)
            action = self._deterministic_action(observation)
            heads = self._q_heads(observation, action, target=False)
            head_generator = self._generator(
                PAPER_DETERMINISTIC,
                "q/head_pair",
                sample,
                device=heads.device,
            )
            indices = torch.randperm(
                heads.numel(), device=heads.device, generator=head_generator
            )[:2]
            q_values.append(
                _finite_float(
                    heads.index_select(0, indices).mean().cpu(),
                    "paper deterministic Q estimate",
                )
            )

        mc_mean, mc_std = self._mean_std(mc_values)
        q_mean, q_std = self._mean_std(q_values)
        return {
            "eval/mc_value": mc_mean,
            "eval/mc_value_std": mc_std,
            "eval/q_value": q_mean,
            "eval/q_value_std": q_std,
            "eval/q_minus_mc": q_mean - mc_mean,
        }

    def _resolved_alpha(self) -> float:
        alpha = getattr(self.agent, "entropy_coefficient")
        if torch.is_tensor(alpha):
            flattened = alpha.detach().reshape(-1)
            if flattened.numel() != 1:
                raise ValueError("SAC entropy_coefficient must contain one scalar.")
            alpha = flattened[0].cpu()
        alpha = _finite_float(alpha, "SAC entropy coefficient")
        if alpha <= 0.0:
            raise ValueError("SAC entropy coefficient must be positive.")
        return alpha

    def _soft_rollout(
        self,
        observation: Any,
        initial_action: torch.Tensor,
        *,
        alpha: float,
        action_generator: torch.Generator,
        pair_generator: torch.Generator,
    ) -> dict[str, float | bool]:
        reward_return = 0.0
        soft_return = 0.0
        discount = 1.0
        action = initial_action
        log_prob: torch.Tensor | None = None

        while True:
            observation, reward, terminated, truncated = self._step(action)
            reward_return += discount * reward
            soft_reward = reward
            if log_prob is not None:
                soft_reward -= alpha * _finite_float(
                    log_prob.cpu(), "SAC rollout log-probability"
                )
            soft_return += discount * soft_reward
            if not (math.isfinite(reward_return) and math.isfinite(soft_return)):
                raise ValueError("The stochastic SAC Monte Carlo return became non-finite.")

            next_discount = discount * self.discount
            if terminated or truncated:
                tail = 0.0
                if truncated and not terminated:
                    terminal_action, terminal_log_prob = self._stochastic_action(
                        observation, action_generator
                    )
                    target_heads = self._q_heads(
                        observation, terminal_action, target=True
                    )
                    terminal_q = self._reduce_heads(
                        target_heads,
                        self.target_reduction,
                        generator=pair_generator,
                    )
                    terminal_soft_value = _finite_float(
                        terminal_q.cpu(), "terminal target Q"
                    ) - alpha * _finite_float(
                        terminal_log_prob.cpu(), "terminal policy log-probability"
                    )
                    tail = next_discount * terminal_soft_value
                corrected = soft_return + tail
                if not (math.isfinite(tail) and math.isfinite(corrected)):
                    raise ValueError("The truncation-corrected SAC return became non-finite.")
                return {
                    "reward": float(reward_return),
                    "soft_finite": float(soft_return),
                    "soft_bootstrapped": float(corrected),
                    "tail": float(tail),
                    "truncated": bool(truncated and not terminated),
                }

            discount = next_discount
            action, log_prob = self._stochastic_action(
                observation, action_generator
            )

    def _evaluate_stochastic_soft_bellman(self) -> dict[str, float]:
        alpha = self._resolved_alpha()
        reward_values: list[float] = []
        finite_values: list[float] = []
        bootstrapped_values: list[float] = []
        tails: list[float] = []
        truncations: list[float] = []
        mean_values: list[float] = []
        min_values: list[float] = []
        head_spreads: list[float] = []

        for sample in range(self.samples):
            observation = self._reset(
                STOCHASTIC_SOFT_BELLMAN, "paired", sample
            )
            action_generator = self._generator(
                STOCHASTIC_SOFT_BELLMAN, "paired/action", sample
            )
            pair_generator = self._generator(
                STOCHASTIC_SOFT_BELLMAN, "paired/target_pair", sample
            )
            initial_action, _ = self._stochastic_action(
                observation, action_generator
            )
            heads = self._q_heads(observation, initial_action, target=False)
            head_array = heads.cpu().to(torch.float64).numpy()
            mean_values.append(float(head_array.mean()))
            min_values.append(float(head_array.min()))
            head_spreads.append(float(head_array.std(ddof=0)))

            rollout = self._soft_rollout(
                observation,
                initial_action,
                alpha=alpha,
                action_generator=action_generator,
                pair_generator=pair_generator,
            )
            reward_values.append(float(rollout["reward"]))
            finite_values.append(float(rollout["soft_finite"]))
            bootstrapped_values.append(float(rollout["soft_bootstrapped"]))
            tails.append(float(rollout["tail"]))
            truncations.append(float(bool(rollout["truncated"])))

        reward_array = np.asarray(reward_values, dtype=np.float64)
        finite_array = np.asarray(finite_values, dtype=np.float64)
        bootstrapped_array = np.asarray(bootstrapped_values, dtype=np.float64)
        mean_array = np.asarray(mean_values, dtype=np.float64)
        min_array = np.asarray(min_values, dtype=np.float64)
        mean_error = mean_array - bootstrapped_array
        min_error = min_array - bootstrapped_array

        return {
            "eval/stochastic_reward_mc_value": float(reward_array.mean()),
            "eval/stochastic_reward_mc_value_std": float(reward_array.std(ddof=0)),
            "eval/stochastic_soft_mc_finite_value": float(finite_array.mean()),
            "eval/stochastic_soft_mc_finite_value_std": float(
                finite_array.std(ddof=0)
            ),
            "eval/stochastic_soft_mc_bootstrapped_value": float(
                bootstrapped_array.mean()
            ),
            "eval/stochastic_soft_mc_bootstrapped_value_std": float(
                bootstrapped_array.std(ddof=0)
            ),
            "eval/stochastic_soft_truncation_tail": float(
                np.asarray(tails, dtype=np.float64).mean()
            ),
            "eval/stochastic_soft_truncation_fraction": float(
                np.asarray(truncations, dtype=np.float64).mean()
            ),
            "eval/stochastic_soft_q_mean_all": float(mean_array.mean()),
            "eval/stochastic_soft_q_mean_all_std": float(mean_array.std(ddof=0)),
            "eval/stochastic_soft_q_min_all": float(min_array.mean()),
            "eval/stochastic_soft_q_head_std": float(
                np.asarray(head_spreads, dtype=np.float64).mean()
            ),
            "eval/stochastic_soft_q_minus_mc_bootstrapped_mean_all": float(
                mean_error.mean()
            ),
            "eval/stochastic_soft_q_rmse_bootstrapped_mean_all": float(
                np.sqrt(np.mean(np.square(mean_error)))
            ),
            "eval/stochastic_soft_q_minus_mc_bootstrapped_min_all": float(
                min_error.mean()
            ),
            "eval/stochastic_soft_q_rmse_bootstrapped_min_all": float(
                np.sqrt(np.mean(np.square(min_error)))
            ),
            "eval/stochastic_soft_alpha": float(alpha),
        }

    def evaluate(self) -> dict[str, float]:
        """Run configured protocols without advancing training/global RNG state."""

        if self._closed:
            raise RuntimeError("SACValueCalibrationEvaluator is closed.")
        rng_state = _capture_global_rng()
        modes = {
            "actor": bool(self.agent.actor.training),
            "critic": bool(self.agent.critic.training),
            "critic_target": bool(self.agent.critic_target.training),
        }
        start = time.perf_counter()
        try:
            self.agent.actor.eval()
            self.agent.critic.eval()
            self.agent.critic_target.eval()
            with torch.inference_mode():
                metrics: dict[str, float] = {}
                for protocol in self.protocols:
                    if protocol == PAPER_DETERMINISTIC:
                        protocol_metrics = self._evaluate_paper_deterministic()
                    elif protocol == STOCHASTIC_SOFT_BELLMAN:
                        protocol_metrics = self._evaluate_stochastic_soft_bellman()
                    else:  # Constructor validation makes this unreachable.
                        raise AssertionError(f"Unhandled protocol {protocol!r}.")
                    protocol_metrics = {
                        key: _finite_float(value, f"calibration metric {key}")
                        for key, value in protocol_metrics.items()
                    }
                    collisions = set(metrics).intersection(protocol_metrics)
                    if collisions:
                        raise RuntimeError(
                            "SAC value-calibration protocols emitted duplicate metrics: "
                            f"{sorted(collisions)}."
                        )
                    metrics.update(protocol_metrics)
                metrics["eval/value_samples"] = float(self.samples)
                metrics["time/value_eval_seconds"] = _finite_float(
                    max(0.0, time.perf_counter() - start),
                    "SAC value evaluation duration",
                )
                return metrics
        finally:
            try:
                self.agent.actor.train(modes["actor"])
                self.agent.critic.train(modes["critic"])
                self.agent.critic_target.train(modes["critic_target"])
            finally:
                _restore_global_rng(rng_state)

    def close(self) -> None:
        """Close the lazily-created auxiliary environment exactly once."""

        if self._closed:
            return
        self._closed = True
        env, self._env = self._env, None
        if env is not None:
            env.close()
