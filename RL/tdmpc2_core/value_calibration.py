"""Isolated real-environment calibration for AMBI's outer value prior.

The evaluator deliberately owns neither an agent nor an inner-improvement
engine.  It probes a supplied world model directly in a lazily-created
auxiliary environment, so callers can attach it to evaluation without changing
the training environment or AMBI's root-local state.
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
STOCHASTIC_BELLMAN = "stochastic_bellman"
_PROTOCOLS = frozenset({PAPER_DETERMINISTIC, STOCHASTIC_BELLMAN})
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
    """Derive one stable, process-independent seed for an evaluator substream."""

    digest = hashlib.blake2b(digest_size=8, person=b"AMBI-value-eval")
    digest.update(str(int(seed)).encode("ascii"))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little") & _TORCH_SEED_MAX


def _finite_float(value: object, name: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number.") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    return resolved


class ValueCalibrationEvaluator:
    """Measure outer-policy Monte Carlo returns against online critic values.

    ``paper_deterministic`` reproduces the public-paper protocol: independent
    initial-state samples for deterministic mean-policy rollouts and Q probes,
    with each Q probe averaging a seeded pair of online critic heads.

    ``stochastic_bellman`` uses a stricter paired estimator.  It samples one
    outer-policy action at an initial state, evaluates every online head at that
    exact action, executes the same action, and follows the stochastic outer
    policy for the remainder of the discounted real-environment rollout.
    """

    def __init__(
        self,
        model: Any,
        env_factory: Callable[[], Any],
        observation_to_tensor: Callable[[Any], torch.Tensor],
        unscale_action: Callable[[np.ndarray], Any],
        discount: float,
        samples: int,
        seed: int,
        protocols: Iterable[str] | str,
        device: torch.device | str,
    ) -> None:
        if not callable(env_factory):
            raise TypeError("env_factory must be callable.")
        if not callable(observation_to_tensor):
            raise TypeError("observation_to_tensor must be callable.")
        if not callable(unscale_action):
            raise TypeError("unscale_action must be callable.")
        discount = _finite_float(discount, "discount")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1].")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise ValueError("samples must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        if isinstance(protocols, str):
            normalized_protocols = (protocols,)
        else:
            try:
                normalized_protocols = tuple(protocols)
            except TypeError as exc:
                raise TypeError("protocols must be a protocol name or iterable.") from exc
        if not normalized_protocols:
            raise ValueError("At least one value-calibration protocol is required.")
        if any(not isinstance(protocol, str) for protocol in normalized_protocols):
            raise TypeError("Value-calibration protocol names must be strings.")
        if len(set(normalized_protocols)) != len(normalized_protocols):
            raise ValueError("Value-calibration protocols must be unique.")
        unknown = set(normalized_protocols) - _PROTOCOLS
        if unknown:
            raise ValueError(
                f"Unknown value-calibration protocol(s) {sorted(unknown)}; "
                f"expected {sorted(_PROTOCOLS)}."
            )

        for attribute in ("encode", "q_values", "eval", "train"):
            if not callable(getattr(model, attribute, None)):
                raise TypeError(f"model must provide callable {attribute}().")
        if not callable(getattr(model, "pi_action", None)) and not callable(
            getattr(model, "pi", None)
        ):
            raise TypeError("model must provide callable pi_action() or pi().")
        if not hasattr(model, "training"):
            raise TypeError("model must expose its training mode.")

        self.model = model
        self.env_factory = env_factory
        self.observation_to_tensor = observation_to_tensor
        self.unscale_action = unscale_action
        self.discount = discount
        self.samples = int(samples)
        self.seed = int(seed)
        self.protocols = normalized_protocols
        self.device = torch.device(device)
        self._env: Any | None = None
        self._closed = False

    def _environment(self) -> Any:
        if self._closed:
            raise RuntimeError("ValueCalibrationEvaluator is closed.")
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
        # DMControl's adapter rebuilds the raw task when a seed is supplied.
        # Seed the start of each protocol batch once, then draw the remaining
        # initial states from that reset stream without rebuilding the task.
        if sample == 0:
            # Gymnasium accepts unsigned 32-bit reset seeds.  The full
            # namespaced seed remains available to the Torch substreams above.
            reset_seed = self._seed(protocol, f"{namespace}/reset_batch", 0)
            result = self._environment().reset(seed=reset_seed & 0xFFFFFFFF)
        else:
            result = self._environment().reset()
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("Calibration environments must use Gymnasium reset().")
        return result[0]

    def _encode(self, observation: Any) -> torch.Tensor:
        observation_tensor = self.observation_to_tensor(observation)
        if not torch.is_tensor(observation_tensor):
            raise TypeError("observation_to_tensor must return a torch.Tensor.")
        # The evaluator is intentionally state-observation-only.  The wrapper's
        # conversion hook returns one unbatched state vector; model.encode uses
        # the same leading batch dimension as training and acting.
        batched = observation_tensor.detach().to(self.device).unsqueeze(0)
        return self.model.encode(batched)

    def _policy_action(
        self,
        latent: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator,
    ) -> torch.Tensor:
        pi_action = getattr(self.model, "pi_action", None)
        if callable(pi_action):
            action = pi_action(
                latent,
                deterministic=deterministic,
                generator=generator,
            )
        else:
            output = self.model.pi(
                latent,
                deterministic=deterministic,
                generator=generator,
            )
            action = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(action):
            raise TypeError("The outer policy must return a torch.Tensor action.")
        if action.ndim < 2 or action.shape[0] != 1:
            raise ValueError(
                "The outer policy must return one batched action, got "
                f"shape {tuple(action.shape)}."
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError("The outer policy produced a non-finite action.")
        return action

    @staticmethod
    def _decoded_heads(values: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(values):
            raise TypeError("model.q_values() must return a torch.Tensor.")
        if values.ndim < 2 or values.shape[0] < 1:
            raise ValueError(
                "Online Q values must have a leading critic-head dimension."
            )
        flattened = values.detach().reshape(values.shape[0], -1)
        if flattened.shape[1] != 1:
            raise ValueError(
                "Each online critic head must produce one scalar for the single "
                f"probe, got shape {tuple(values.shape)}."
            )
        if not bool(torch.isfinite(flattened).all()):
            raise ValueError("The online critic produced non-finite Q head values.")
        return flattened[:, 0]

    def _online_q_heads(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        values = self.model.q_values(latent, action, target=False)
        return self._decoded_heads(values)

    def _env_action(self, batched_action: torch.Tensor) -> Any:
        normalized = batched_action[0].detach().cpu().numpy()
        env_action = self.unscale_action(normalized)
        try:
            finite = bool(np.isfinite(np.asarray(env_action)).all())
        except TypeError as exc:
            raise TypeError("unscale_action must return a numeric action.") from exc
        if not finite:
            raise ValueError("unscale_action produced a non-finite environment action.")
        return env_action

    def _rollout(
        self,
        observation: Any,
        *,
        deterministic: bool,
        action_generator: torch.Generator,
        initial_action: torch.Tensor | None = None,
    ) -> float:
        discounted_return = 0.0
        discount = 1.0
        action = initial_action
        while True:
            if action is None:
                latent = self._encode(observation)
                action = self._policy_action(
                    latent,
                    deterministic=deterministic,
                    generator=action_generator,
                )
            transition = self._environment().step(self._env_action(action))
            if not isinstance(transition, tuple) or len(transition) != 5:
                raise TypeError("Calibration environments must use Gymnasium step().")
            observation, reward, terminated, truncated, _ = transition
            reward = _finite_float(reward, "environment reward")
            discounted_return += discount * reward
            if not math.isfinite(discounted_return):
                raise ValueError("The discounted Monte Carlo return became non-finite.")
            if bool(terminated or truncated):
                return float(discounted_return)
            discount *= self.discount
            action = None

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        return float(array.mean()), float(array.std(ddof=0))

    def _evaluate_paper_deterministic(self) -> dict[str, float]:
        mc_values: list[float] = []
        for sample in range(self.samples):
            observation = self._reset(PAPER_DETERMINISTIC, "mc", sample)
            action_generator = self._generator(
                PAPER_DETERMINISTIC, "mc/action", sample
            )
            mc_values.append(
                self._rollout(
                    observation,
                    deterministic=True,
                    action_generator=action_generator,
                )
            )

        q_values: list[float] = []
        for sample in range(self.samples):
            observation = self._reset(PAPER_DETERMINISTIC, "q", sample)
            latent = self._encode(observation)
            action = self._policy_action(
                latent,
                deterministic=True,
                generator=self._generator(
                    PAPER_DETERMINISTIC, "q/action", sample
                ),
            )
            heads = self._online_q_heads(latent, action)
            if heads.numel() < 2:
                raise ValueError(
                    "paper_deterministic requires at least two online Q heads."
                )
            head_generator = self._generator(
                PAPER_DETERMINISTIC,
                "q/head_pair",
                sample,
                device=heads.device,
            )
            indices = torch.randperm(
                heads.numel(),
                device=heads.device,
                generator=head_generator,
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
            "eval/q_value": q_mean,
            "eval/q_minus_mc": q_mean - mc_mean,
            "eval/mc_value_std": mc_std,
            "eval/q_value_std": q_std,
        }

    def _evaluate_stochastic_bellman(self) -> dict[str, float]:
        mc_values: list[float] = []
        mean_values: list[float] = []
        min_values: list[float] = []
        head_spreads: list[float] = []

        for sample in range(self.samples):
            observation = self._reset(STOCHASTIC_BELLMAN, "paired", sample)
            action_generator = self._generator(
                STOCHASTIC_BELLMAN, "paired/action", sample
            )
            latent = self._encode(observation)
            initial_action = self._policy_action(
                latent,
                deterministic=False,
                generator=action_generator,
            )
            heads = self._online_q_heads(latent, initial_action)
            head_array = heads.cpu().to(torch.float64).numpy()
            mean_values.append(float(head_array.mean()))
            min_values.append(float(head_array.min()))
            head_spreads.append(float(head_array.std(ddof=0)))
            mc_values.append(
                self._rollout(
                    observation,
                    deterministic=False,
                    action_generator=action_generator,
                    initial_action=initial_action,
                )
            )

        mc_array = np.asarray(mc_values, dtype=np.float64)
        mean_array = np.asarray(mean_values, dtype=np.float64)
        min_array = np.asarray(min_values, dtype=np.float64)
        mean_error = mean_array - mc_array
        min_error = min_array - mc_array

        return {
            "eval/stochastic_mc_value": float(mc_array.mean()),
            "eval/stochastic_mc_value_std": float(mc_array.std(ddof=0)),
            "eval/stochastic_q_mean_all": float(mean_array.mean()),
            "eval/stochastic_q_mean_all_std": float(mean_array.std(ddof=0)),
            "eval/stochastic_q_min_all": float(min_array.mean()),
            "eval/stochastic_q_minus_mc_mean_all": float(mean_error.mean()),
            "eval/stochastic_q_rmse_mean_all": float(
                np.sqrt(np.mean(np.square(mean_error)))
            ),
            "eval/stochastic_q_head_std": float(
                np.asarray(head_spreads, dtype=np.float64).mean()
            ),
            "eval/stochastic_q_minus_mc_min_all": float(min_error.mean()),
            "eval/stochastic_q_rmse_min_all": float(
                np.sqrt(np.mean(np.square(min_error)))
            ),
        }

    def evaluate(self) -> dict[str, float]:
        """Run configured protocols without advancing training/global RNG state."""

        if self._closed:
            raise RuntimeError("ValueCalibrationEvaluator is closed.")
        rng_state = _capture_global_rng()
        was_training = bool(self.model.training)
        start = time.perf_counter()
        try:
            self.model.eval()
            with torch.inference_mode():
                metrics: dict[str, float] = {}
                for protocol in self.protocols:
                    if protocol == PAPER_DETERMINISTIC:
                        protocol_metrics = self._evaluate_paper_deterministic()
                    elif protocol == STOCHASTIC_BELLMAN:
                        protocol_metrics = self._evaluate_stochastic_bellman()
                    else:  # Constructor validation makes this unreachable.
                        raise AssertionError(f"Unhandled protocol {protocol!r}.")
                    protocol_metrics = {
                        key: _finite_float(value, f"calibration metric {key}")
                        for key, value in protocol_metrics.items()
                    }
                    collisions = set(metrics) & set(protocol_metrics)
                    if collisions:
                        raise RuntimeError(
                            "Value-calibration protocols emitted duplicate metrics: "
                            f"{sorted(collisions)}."
                        )
                    metrics.update(protocol_metrics)
                metrics["eval/value_samples"] = float(self.samples)
                metrics["time/value_eval_seconds"] = _finite_float(
                    max(0.0, time.perf_counter() - start),
                    "value evaluation duration",
                )
                return metrics
        finally:
            try:
                self.model.train(was_training)
            finally:
                _restore_global_rng(rng_state)

    def close(self) -> None:
        """Close the auxiliary environment once; construction remains lazy."""

        if self._closed:
            return
        self._closed = True
        env, self._env = self._env, None
        if env is not None:
            env.close()
