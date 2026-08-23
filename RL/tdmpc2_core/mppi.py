"""Stateless, backend-neutral MPPI comparison operator for AMBI ablations.

The helper in this module mirrors the MPPI update used by the vendored
TD-MPC2 agent, but makes every source of compute explicit.  It operates on an
already-encoded root latent and never changes a world model, optimizer, or
warm-start buffer.  Callers own the returned mean and may persist it at any
desired lifecycle scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Callable, Mapping, Optional

import torch

from .common import math as td_math


_Q_REDUCTIONS = {"min_pair", "mean_pair", "min_all", "mean_all"}


@dataclass(frozen=True)
class MPPIModelCallbacks:
    """Minimal decoded world-model interface consumed by :func:`mppi_plan`.

    Callback mode is useful for tests and for planners whose policy or value
    function is not owned by one module. ``reward`` and ``terminal_q`` must
    return decoded scalar columns with shape ``[batch, 1]``. ``policy`` may
    return either an action tensor or ``(action, info)``.

    The expected signatures are::

        dynamics(z, action) -> next_z
        reward(z, action) -> scalar_reward
        policy(z, *, generator) -> action | (action, info)
        terminal_q(z, action, *, reduction, generator) -> scalar_q
        termination(z) -> probability  # optional
        transition(z, action) -> (next_z, scalar_reward)  # optional fused path
    """

    action_dim: int
    dynamics: Callable[..., torch.Tensor]
    reward: Callable[..., torch.Tensor]
    policy: Callable[..., Any]
    terminal_q: Callable[..., torch.Tensor]
    termination: Optional[Callable[..., torch.Tensor]] = None
    transition: Optional[Callable[..., tuple[torch.Tensor, torch.Tensor]]] = None


@dataclass(frozen=True)
class MPPIResult:
    """Output of a stateless MPPI solve."""

    action: torch.Tensor
    next_mean: torch.Tensor
    metrics: Mapping[str, float | int | torch.Tensor]
    model_steps: int


def _model_callbacks(model, task=None):
    """Adapt ``SoftWorldModel`` to the decoded callback interface."""

    def dynamics(z, action):
        return model.next(z, action, task)

    def reward(z, action):
        prediction = model.reward(z, action, task)
        return td_math.two_hot_inv(prediction, model.cfg)

    def transition(z, action):
        # Reward and dynamics consume the same latent/action pair. Pack it once
        # for the dense candidate population rather than launching two cats at
        # every horizon step.
        joint = model.joint_input(z, action)
        prediction = model.reward_from_joint(joint)
        return (
            model.next_from_joint(joint),
            td_math.two_hot_inv(prediction, model.cfg),
        )

    def policy(z, *, generator):
        if hasattr(model, "pi_action"):
            return model.pi_action(z, task, generator=generator)
        return model.pi(z, task, generator=generator)

    def terminal_q(z, action, *, reduction, generator):
        return model.Q(
            z,
            action,
            task,
            reduction=reduction,
            generator=generator,
        )

    termination = None
    if bool(getattr(model.cfg, "episodic", False)):
        def termination(z):
            return model.termination(z, task)

    return MPPIModelCallbacks(
        action_dim=int(model.cfg.action_dim),
        dynamics=dynamics,
        reward=reward,
        policy=policy,
        terminal_q=terminal_q,
        termination=termination,
        transition=transition,
    )


def _as_scalar_column(value, batch_size, *, name, like):
    value = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if value.ndim == 1:
        value = value.unsqueeze(-1)
    expected = (batch_size, 1)
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}.")
    return value


def _policy_action(callbacks, z, generator):
    output = callbacks.policy(z, generator=generator)
    action = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(action):
        raise TypeError("MPPI policy callback must return a tensor or (tensor, info).")
    expected = (z.shape[0], callbacks.action_dim)
    if tuple(action.shape) != expected:
        raise ValueError(
            f"MPPI policy action must have shape {expected}, got {tuple(action.shape)}."
        )
    return action


def _validate_generator(generator, device):
    if not isinstance(generator, torch.Generator):
        raise TypeError("MPPI requires an explicit torch.Generator for RNG isolation.")
    generator_device = torch.device(generator.device)
    device = torch.device(device)
    if generator_device.type != device.type:
        raise ValueError(
            "MPPI generator and root latent must use the same device type, "
            f"got {generator_device} and {device}."
        )
    if device.type == "cuda":
        generator_index = generator_device.index
        device_index = device.index
        if generator_index is not None and device_index is not None and generator_index != device_index:
            raise ValueError(
                "MPPI generator and root latent must use the same CUDA device, "
                f"got {generator_device} and {device}."
            )


def _validate_inputs(
    root_z,
    callbacks,
    *,
    horizon,
    iterations,
    num_samples,
    num_elites,
    num_pi_trajs,
    temperature,
    min_std,
    max_std,
    discount,
    q_reduction,
    termination_threshold,
    generator,
    previous_mean,
):
    if not torch.is_tensor(root_z) or not root_z.is_floating_point():
        raise TypeError("root_z must be a floating-point torch.Tensor.")
    if root_z.ndim == 1:
        root_z = root_z.unsqueeze(0)
    if root_z.ndim != 2 or root_z.shape[0] != 1:
        raise ValueError(
            "MPPI plans one root state at a time; root_z must have shape [latent] "
            f"or [1, latent], got {tuple(root_z.shape)}."
        )

    horizon, iterations = int(horizon), int(iterations)
    num_samples, num_elites = int(num_samples), int(num_elites)
    num_pi_trajs = int(num_pi_trajs)
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not 1 <= num_elites <= num_samples:
        raise ValueError("num_elites must be in [1, num_samples].")
    if not 0 <= num_pi_trajs <= num_samples:
        raise ValueError("num_pi_trajs must be in [0, num_samples].")

    action_dim = int(callbacks.action_dim)
    if action_dim <= 0:
        raise ValueError("action_dim must be positive.")
    temperature, min_std, max_std = float(temperature), float(min_std), float(max_std)
    if not all(isfinite(value) for value in (temperature, min_std, max_std)):
        raise ValueError("temperature, min_std, and max_std must be finite.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if min_std <= 0.0 or max_std <= 0.0:
        raise ValueError("min_std and max_std must be positive.")
    if min_std > max_std:
        raise ValueError("min_std cannot exceed max_std.")

    # AMBI supplies the configured Python scalar. Validate it before creating
    # the device tensor so planning never performs a scalar GPU read merely to
    # check a constant hyperparameter.
    if isinstance(discount, Real):
        discount_value = float(discount)
        if not isfinite(discount_value) or not 0.0 <= discount_value <= 1.0:
            raise ValueError("discount must be finite and in [0, 1].")
        discount_tensor = root_z.new_tensor(discount_value)
    else:
        # Preserve the public helper's historical scalar-tensor support. This
        # branch may synchronize a caller-supplied accelerator scalar for input
        # validation; AMBI's hot path always takes the Python-scalar branch.
        discount_tensor = torch.as_tensor(
            discount, device=root_z.device, dtype=root_z.dtype
        )
        if discount_tensor.numel() != 1:
            raise ValueError("discount must be scalar for a single MPPI root state.")
        discount_tensor = discount_tensor.reshape(())
        if not bool(torch.isfinite(discount_tensor)) or not 0.0 <= float(
            discount_tensor
        ) <= 1.0:
            raise ValueError("discount must be finite and in [0, 1].")

    q_reduction = str(q_reduction).lower()
    if q_reduction not in _Q_REDUCTIONS:
        raise ValueError(
            f"q_reduction must be one of {sorted(_Q_REDUCTIONS)}, got {q_reduction!r}."
        )
    termination_threshold = float(termination_threshold)
    if not isfinite(termination_threshold):
        raise ValueError("termination_threshold must be finite.")
    if not 0.0 <= termination_threshold <= 1.0:
        raise ValueError("termination_threshold must be in [0, 1].")
    _validate_generator(generator, root_z.device)

    normalized_previous_mean = None
    if previous_mean is not None:
        if not torch.is_tensor(previous_mean):
            raise TypeError("previous_mean must be a torch.Tensor when provided.")
        expected = (horizon, action_dim)
        if tuple(previous_mean.shape) != expected:
            raise ValueError(
                f"previous_mean must have shape {expected}, got {tuple(previous_mean.shape)}."
            )
        normalized_previous_mean = previous_mean.to(
            device=root_z.device,
            dtype=root_z.dtype,
        )

    normalized = {
        "horizon": horizon,
        "iterations": iterations,
        "num_samples": num_samples,
        "num_elites": num_elites,
        "num_pi_trajs": num_pi_trajs,
        "temperature": temperature,
        "min_std": min_std,
        "max_std": max_std,
        "discount": discount_tensor,
        "q_reduction": q_reduction,
        "termination_threshold": termination_threshold,
        "previous_mean": normalized_previous_mean,
    }
    return root_z, normalized


def _estimate_value(
    root_z,
    actions,
    callbacks,
    *,
    discount,
    q_reduction,
    termination_threshold,
    generator,
):
    """Evaluate one candidate population without pruning masked transitions."""
    horizon, num_samples = actions.shape[:2]
    z = root_z.expand(num_samples, -1)
    value = root_z.new_zeros(num_samples, 1)
    continuation = root_z.new_ones(num_samples, 1)
    discount_power = root_z.new_ones(())

    for step in range(horizon):
        if callbacks.transition is None:
            reward = callbacks.reward(z, actions[step])
            next_z = callbacks.dynamics(z, actions[step])
        else:
            next_z, reward = callbacks.transition(z, actions[step])
        reward = _as_scalar_column(
            reward,
            num_samples,
            name="decoded reward",
            like=root_z,
        )
        value = value + discount_power * continuation * reward
        z = next_z
        if not torch.is_tensor(z) or tuple(z.shape) != (num_samples, root_z.shape[-1]):
            actual = tuple(z.shape) if torch.is_tensor(z) else type(z).__name__
            raise ValueError(
                "MPPI dynamics must preserve [batch, latent] shape; "
                f"expected {(num_samples, root_z.shape[-1])}, got {actual}."
            )
        discount_power = discount_power * discount
        if callbacks.termination is not None:
            termination = callbacks.termination(z)
            termination = _as_scalar_column(
                termination,
                num_samples,
                name="termination probability",
                like=root_z,
            )
            continuation = continuation * (termination <= termination_threshold).to(root_z.dtype)

    terminal_action = _policy_action(callbacks, z, generator)
    terminal_q = callbacks.terminal_q(
        z,
        terminal_action,
        reduction=q_reduction,
        generator=generator,
    )
    terminal_q = _as_scalar_column(
        terminal_q,
        num_samples,
        name="decoded terminal Q",
        like=root_z,
    )
    return value + discount_power * continuation * terminal_q


@torch.no_grad()
def mppi_plan(
    root_z,
    model=None,
    *,
    callbacks=None,
    horizon,
    iterations,
    num_samples,
    num_elites,
    num_pi_trajs,
    temperature,
    min_std,
    max_std,
    discount,
    q_reduction,
    termination_threshold=0.5,
    generator,
    previous_mean=None,
    t0=False,
    eval_mode=False,
    task=None,
    materialize_metrics=True,
):
    """Plan one action with model-predictive path integral control.

    Exactly one of ``model`` and ``callbacks`` must be supplied. Model mode
    accepts ``SoftWorldModel`` and decodes its reward/Q representations before
    optimization. Callback mode accepts :class:`MPPIModelCallbacks` and is
    representation agnostic.

    Model-step accounting follows the computation actually performed:

    ``num_pi_trajs * (horizon - 1) + iterations * num_samples * horizon``.

    Predicted termination masks subsequent value contributions but deliberately
    does not prune the batched model rollout, keeping compute exact and stable.
    ``previous_mean`` is only read, and only when ``t0`` is false. The caller
    may persist the returned ``next_mean`` without any planner-owned state.
    Set ``materialize_metrics=False`` to keep diagnostic scalars on-device for
    packing with another boundary transfer; the default preserves float metrics.
    """
    if (model is None) == (callbacks is None):
        raise ValueError("Supply exactly one of model or callbacks to MPPI.")
    if model is not None and bool(model.training):
        raise ValueError(
            "MPPI model mode requires model.eval() so critic dropout cannot leak "
            "implicit global randomness; callback implementations must provide the "
            "same deterministic contract."
        )
    if callbacks is None:
        callbacks = _model_callbacks(model, task=task)
    elif not isinstance(callbacks, MPPIModelCallbacks):
        raise TypeError("callbacks must be an MPPIModelCallbacks instance.")

    root_z, values = _validate_inputs(
        root_z,
        callbacks,
        horizon=horizon,
        iterations=iterations,
        num_samples=num_samples,
        num_elites=num_elites,
        num_pi_trajs=num_pi_trajs,
        temperature=temperature,
        min_std=min_std,
        max_std=max_std,
        discount=discount,
        q_reduction=q_reduction,
        termination_threshold=termination_threshold,
        generator=generator,
        previous_mean=previous_mean,
    )
    horizon = values["horizon"]
    iterations = values["iterations"]
    num_samples = values["num_samples"]
    num_elites = values["num_elites"]
    num_pi_trajs = values["num_pi_trajs"]
    temperature = values["temperature"]
    min_std, max_std = values["min_std"], values["max_std"]
    discount = values["discount"]
    q_reduction = values["q_reduction"]
    termination_threshold = values["termination_threshold"]
    previous_mean = values["previous_mean"]
    action_dim = callbacks.action_dim

    policy_actions = None
    if num_pi_trajs:
        policy_actions = root_z.new_empty(horizon, num_pi_trajs, action_dim)
        policy_z = root_z.expand(num_pi_trajs, -1)
        for step in range(horizon):
            policy_actions[step] = _policy_action(callbacks, policy_z, generator)
            if step + 1 < horizon:
                policy_z = callbacks.dynamics(policy_z, policy_actions[step])
                expected = (num_pi_trajs, root_z.shape[-1])
                if not torch.is_tensor(policy_z) or tuple(policy_z.shape) != expected:
                    actual = (
                        tuple(policy_z.shape)
                        if torch.is_tensor(policy_z)
                        else type(policy_z).__name__
                    )
                    raise ValueError(
                        "MPPI dynamics must preserve [batch, latent] shape; "
                        f"expected {expected}, got {actual}."
                    )

    mean = root_z.new_zeros(horizon, action_dim)
    if previous_mean is not None and not bool(t0) and horizon > 1:
        mean[:-1].copy_(previous_mean[1:])
    std = root_z.new_full((horizon, action_dim), max_std)
    actions = root_z.new_empty(horizon, num_samples, action_dim)
    if policy_actions is not None:
        actions[:, :num_pi_trajs].copy_(policy_actions)

    sampled_count = num_samples - num_pi_trajs
    for _ in range(iterations):
        if sampled_count:
            noise = torch.randn(
                (horizon, sampled_count, action_dim),
                device=root_z.device,
                dtype=root_z.dtype,
                generator=generator,
            )
            sampled_actions = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            actions[:, num_pi_trajs:].copy_(sampled_actions.clamp(-1.0, 1.0))

        candidate_value = _estimate_value(
            root_z,
            actions,
            callbacks,
            discount=discount,
            q_reduction=q_reduction,
            termination_threshold=termination_threshold,
            generator=generator,
        ).nan_to_num(0.0)
        elite_indices = torch.topk(
            candidate_value.squeeze(-1),
            num_elites,
            dim=0,
        ).indices
        elite_value = candidate_value.index_select(0, elite_indices)
        elite_actions = actions.index_select(1, elite_indices)

        max_elite_value = elite_value.max(dim=0).values
        weights = torch.exp(temperature * (elite_value - max_elite_value))
        weights = weights / weights.sum(dim=0).clamp_min(1e-9)
        mean = (weights.unsqueeze(0) * elite_actions).sum(dim=1)
        variance = (
            weights.unsqueeze(0)
            * (elite_actions - mean.unsqueeze(1)).square()
        ).sum(dim=1)
        std = variance.sqrt().clamp(min_std, max_std)

    if bool(eval_mode):
        # Evaluation is the deterministic optimized proposal. It deliberately
        # consumes no categorical-selection or execution-noise randomness.
        action = mean[0]
    else:
        selected_elite = torch.multinomial(
            weights.squeeze(-1),
            1,
            replacement=True,
            generator=generator,
        )
        selected_actions = elite_actions.index_select(1, selected_elite).squeeze(1)
        action = selected_actions[0]
        action_noise = torch.randn(
            (action_dim,),
            device=root_z.device,
            dtype=root_z.dtype,
            generator=generator,
        )
        action = action + std[0] * action_noise
    action = action.clamp(-1.0, 1.0)

    policy_model_steps = num_pi_trajs * max(0, horizon - 1)
    candidate_model_steps = iterations * num_samples * horizon
    model_steps = policy_model_steps + candidate_model_steps
    metrics = {
        # Keep scalar diagnostics on-device. AMBI packs them with the selected
        # action for its single host transfer at the action boundary.
        "planner_value_mean": candidate_value.mean().detach(),
        "planner_value_std": candidate_value.std(unbiased=False).detach(),
        "planner_value_max": candidate_value.max().detach(),
        "planner_elite_value_mean": elite_value.mean().detach(),
        "planner_elite_value_std": elite_value.std(unbiased=False).detach(),
        "planner_elite_value_max": elite_value.max().detach(),
        "planner_std_mean": std.mean().detach(),
        "planner_std_min": std.min().detach(),
        "planner_std_max": std.max().detach(),
        "planner_action_l2": torch.linalg.vector_norm(action).detach(),
        "planner_num_samples": num_samples,
        "planner_num_elites": num_elites,
        "planner_num_pi_trajs": num_pi_trajs,
        "planner_iterations": iterations,
        "planner_policy_model_steps": policy_model_steps,
        "planner_candidate_model_steps": candidate_model_steps,
        "planner_model_steps": model_steps,
        "planner_policy_evaluations": num_pi_trajs * horizon + iterations * num_samples,
        "planner_q_evaluations": iterations * num_samples,
    }
    if materialize_metrics:
        tensor_metrics = [
            (key, value) for key, value in metrics.items() if torch.is_tensor(value)
        ]
        if tensor_metrics:
            packed_metrics = torch.stack(
                [value.reshape(()) for _, value in tensor_metrics]
            ).cpu().tolist()
            for (key, _), value in zip(tensor_metrics, packed_metrics):
                metrics[key] = float(value)
    return MPPIResult(
        action=action,
        next_mean=mean.clone(),
        metrics=metrics,
        model_steps=model_steps,
    )


__all__ = ["MPPIModelCallbacks", "MPPIResult", "mppi_plan"]
