"""Independent numerical references for matched-root transfer diagnostics.

No learner, environment, or logging service is imported here. Callbacks must
evaluate frozen models; caller-supplied noise makes counterfactuals pairable.
Monte Carlo draws quantify reference noise, not independent environment trials.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import math
from numbers import Integral

import numpy as np
import torch
from torch import nn


def _finite_tensor(value, name):
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite tensor")
    return value


def _vector(value, count, like, name):
    value = torch.as_tensor(value, device=like.device)
    if value.numel() != count:
        raise ValueError(f"{name} must have one scalar per state")
    value = value.reshape(count)
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


@torch.no_grad()
def forced_action_mc_returns(
    roots, first_actions, *, horizons, transition, policy, tail, discount,
    policy_noise, tail_noise, critic_target="reward_only", entropy_coefficient=0.0,
):
    """Return ``{h: returns[M, B]}`` for forced-first-action finite-horizon Q.

    ``roots[B,Z]`` and ``first_actions[B,A]`` describe paired state/action
    queries. ``policy_noise[max(h)-1,M,B,A]`` supplies all continuation noise;
    ``tail_noise[max(h),M,B,A]`` supplies independent terminal-policy noise for
    each horizon. Reuse these tensors across candidates for common randomness.
    Callbacks receive flattened batches of *alive* states only:

    - ``transition(state, action) -> reward, next_state, terminated``;
    - ``policy(state, noise) -> action, log_probability``;
    - ``tail(state, noise) -> frozen_terminal_value``.

    Transition rewards and true-termination flags contain one scalar per row.
    A rollout cutoff is NOT a termination. The tail callback owns its complete
    value semantics, including any entropy already learned in a frozen critic.
    For ``entropy_augmented``, add ``-alpha*log_pi`` only at sampled interior
    actions t=1,...,h-1. Neither the forced first action nor the terminal tail
    receives an additional entropy term. The coefficient is ignored for
    reward-only targets. It is expressed in reward/value units; the caller
    supplies any Q-scale conversion. This reference supports a stationary continuation
    policy; horizon-conditioned policies need separate calls/callbacks.

    The function does not draw noise, mutate inputs, or change module modes.
    Stochastic transitions, if used, must have externally paired randomness.
    """
    roots = _finite_tensor(roots, "roots")
    first_actions = _finite_tensor(first_actions, "first_actions")
    if roots.ndim != 2 or first_actions.ndim != 2 or roots.shape[0] != first_actions.shape[0]:
        raise ValueError("roots and first_actions must be [B,features] with matching B")
    if not roots.is_floating_point() or roots.shape[0] == 0 or roots.shape[1] == 0:
        raise ValueError("roots must be a nonempty floating-point batch")
    if first_actions.device != roots.device or not first_actions.is_floating_point():
        raise ValueError("first_actions must be floating point on the roots device")
    horizons = tuple(horizons)
    if (not horizons or any(isinstance(h, bool) or not isinstance(h, Integral) or h < 1 for h in horizons)
            or len(set(horizons)) != len(horizons)):
        raise ValueError("horizons must be unique positive integers")
    maximum = int(max(horizons))
    if not math.isfinite(float(discount)) or not 0 <= float(discount) <= 1:
        raise ValueError("discount must lie in [0,1]")
    if critic_target not in {"reward_only", "entropy_augmented"}:
        raise ValueError("critic_target must be reward_only or entropy_augmented")
    alpha = float(entropy_coefficient)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("entropy_coefficient must be finite and nonnegative")
    if critic_target == "reward_only":
        alpha = 0.0
    policy_noise = _finite_tensor(policy_noise, "policy_noise")
    tail_noise = _finite_tensor(tail_noise, "tail_noise")
    batch, action_dim = first_actions.shape
    if (tail_noise.ndim != 4 or tail_noise.shape[0] != maximum
            or tail_noise.shape[1] < 1 or tuple(tail_noise.shape[2:]) != (batch, action_dim)):
        raise ValueError("tail_noise must have shape [max_horizon,M,B,A]")
    samples = tail_noise.shape[1]
    if tuple(policy_noise.shape) != (maximum - 1, samples, batch, action_dim):
        raise ValueError("policy_noise must have shape [max_horizon-1,M,B,A]")
    if policy_noise.device != roots.device or tail_noise.device != roots.device:
        raise ValueError("noise and roots must be on the same device")
    states = roots.unsqueeze(0).expand(samples, -1, -1).reshape(-1, roots.shape[-1]).clone()
    actions = first_actions.unsqueeze(0).expand(samples, -1, -1).reshape(-1, action_dim)
    total = roots.new_zeros(samples * batch)
    alive = torch.ones(samples * batch, dtype=torch.bool, device=roots.device)
    references = {}
    wanted = set(horizons)
    for depth in range(maximum):
        rows = torch.nonzero(alive, as_tuple=False).flatten()
        if rows.numel():
            if depth:
                action, logp = policy(states[rows], policy_noise[depth - 1].reshape(-1, action_dim)[rows])
                action = _finite_tensor(action, "policy action")
                if tuple(action.shape) != (rows.numel(), action_dim) or action.device != roots.device:
                    raise ValueError("policy action shape/device disagrees with first_actions")
                if alpha:
                    logp = _vector(logp, rows.numel(), roots, "policy log_probability")
                    total[rows] -= (float(discount) ** depth) * alpha * logp
            else:
                action = actions[rows]
            reward, successor, terminated = transition(states[rows], action)
            reward = _vector(reward, rows.numel(), roots, "reward")
            successor = _finite_tensor(successor, "successor")
            if successor.shape != states[rows].shape or successor.device != roots.device:
                raise ValueError("successor shape/device disagrees with roots")
            terminated = _vector(terminated, rows.numel(), roots, "terminated")
            if not torch.all((terminated == 0) | (terminated == 1)):
                raise ValueError("terminated must contain boolean/zero-one flags")
            total[rows] += (float(discount) ** depth) * reward
            states[rows] = successor
            alive[rows] = ~terminated.bool()
        horizon = depth + 1
        if horizon in wanted:
            result = total.clone()
            survivors = torch.nonzero(alive, as_tuple=False).flatten()
            if survivors.numel():
                terminal = tail(states[survivors], tail_noise[depth].reshape(-1, action_dim)[survivors])
                terminal = _vector(terminal, survivors.numel(), roots, "tail value")
                result[survivors] += (float(discount) ** horizon) * terminal
            references[horizon] = result.reshape(samples, batch)
    return {int(h): references[h] for h in horizons}


def _array(value, name):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return value


def _standard_error(samples):
    if samples.shape[0] < 2:
        return None
    return np.std(samples, axis=0, ddof=1) / math.sqrt(samples.shape[0])


def _json_number(value):
    if value is None:
        return None
    value = np.asarray(value)
    return float(value) if value.ndim == 0 else value.tolist()


def _ranks(values):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def _correlation(left, right):
    left, right = left - left.mean(), right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else None


def paired_action_difference(reference_samples, action_index, baseline_index=0):
    """Paired action difference and MC SE; common noise cancels before reduction."""
    samples = _array(reference_samples, "reference_samples")
    if samples.ndim != 2:
        raise ValueError("reference_samples must be [M,A]")
    for index in (action_index, baseline_index):
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < samples.shape[1]:
            raise ValueError("action indices must be valid integers")
    differences = samples[:, action_index] - samples[:, baseline_index]
    return {"mean": float(differences.mean()), "se": _json_number(_standard_error(differences)),
            "samples": int(samples.shape[0]), "action_index": int(action_index),
            "baseline_index": int(baseline_index)}


def action_value_metrics(predicted, reference_samples, *, baseline_index=0):
    """Audit one root's common action bank, keeping MC noise separate from error.

    Predictions are [A] or [D,A] (D paired critic/dropout evaluations). Ranking
    and selected action use the mean prediction. Reference samples are [M,A].
    ``top_action_regret`` uses the *empirical* reference winner, not a known
    population optimum; its SE is conditional on those selected indices and
    does not correct winner-selection bias. Use independent selection/scoring
    samples for an inferential winner claim. Correlations are None if undefined.
    """
    prediction = _array(predicted, "predicted")
    reference = _array(reference_samples, "reference_samples")
    if prediction.ndim == 1:
        prediction = prediction[None, :]
    if prediction.ndim != 2 or reference.ndim != 2 or prediction.shape[1] != reference.shape[1]:
        raise ValueError("predicted must be [A] or [D,A] and reference_samples [M,A]")
    if (isinstance(baseline_index, bool) or not isinstance(baseline_index, Integral)
            or not 0 <= baseline_index < prediction.shape[1]):
        raise ValueError("baseline_index must identify an action")
    q, target = prediction.mean(axis=0), reference.mean(axis=0)
    error = q - target
    relative_error = (q - q[baseline_index]) - (target - target[baseline_index])
    chosen, best = int(q.argmax()), int(target.argmax())
    regret = paired_action_difference(reference, best, chosen)
    gain = paired_action_difference(reference, chosen, baseline_index)
    return {
        "bias": float(error.mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "centered_rmse": float(np.sqrt(np.mean((error - error.mean()) ** 2))),
        "relative_bias": float(relative_error.mean()),
        "relative_rmse": float(np.sqrt(np.mean(relative_error ** 2))),
        "spearman": _correlation(_ranks(q), _ranks(target)),
        "pearson": _correlation(q, target),
        "top_action_index": chosen, "reference_top_action_index": best,
        "top_action_regret": regret["mean"], "top_action_regret_se": regret["se"],
        "selected_action_gain": gain["mean"], "selected_action_gain_se": gain["se"],
        "reference_mean": target.tolist(), "reference_se": _json_number(_standard_error(reference)),
        "reference_relative_se": _json_number(_standard_error(reference - reference[:, baseline_index:baseline_index + 1])),
        "prediction_mean": q.tolist(),
        "prediction_draw_std": _json_number(np.std(prediction, axis=0, ddof=1)) if len(prediction) > 1 else None,
        "samples": len(reference), "prediction_draws": len(prediction), "actions": len(q),
    }


def paired_directional_metrics(plus_samples, minus_samples, step):
    """Central directional derivative and paired uncertainty on MC axis zero.

    Inputs may be [M] or [M,...] for several directions/roots. ``step`` is the
    actual symmetric displacement length; clipped/asymmetric action changes
    must not be passed off as a central derivative.
    """
    plus, minus = _array(plus_samples, "plus_samples"), _array(minus_samples, "minus_samples")
    if plus.shape != minus.shape or plus.ndim < 1:
        raise ValueError("paired samples must have equal shapes with a sample axis")
    if not math.isfinite(float(step)) or float(step) <= 0:
        raise ValueError("step must be finite and positive")
    differences = plus - minus
    slopes = differences / (2 * float(step))
    return {"gain": _json_number(differences.mean(axis=0)),
            "gain_se": _json_number(_standard_error(differences)),
            "reference_slope": _json_number(slopes.mean(axis=0)),
            "reference_slope_se": _json_number(_standard_error(slopes)),
            "fraction_positive": _json_number((differences > 0).mean(axis=0)),
            "samples": len(plus), "step": float(step)}


def aggregate_paired_root_metrics(
    records, *, value_key="gain", reference_key=None, episode_key="episode", root_key="root",
):
    """Average paired differences: repeats -> roots -> episodes, with episode SE.

    Each row already contains a paired metric, or supplies candidate/reference
    scalars on the *same row* through value_key/reference_key. Repeated rows for
    a root are replicate measurements, not independent environment trials.
    Episode identifiers must include seed/trial identity when combining banks.
    No interval is reported with only one episode. Unequal numbers of roots do
    not give an episode more population weight. Missing/unpaired rows fail.
    """
    roots = defaultdict(list)
    count = 0
    for row in records:
        episode, root = row[episode_key], row[root_key]
        for value in (episode, root):
            if isinstance(value, bool) or not isinstance(value, (str, Integral)) or value == "":
                raise ValueError("episode/root IDs must be nonempty strings or integers")
        value = row[value_key]
        if isinstance(value, bool) or not np.isscalar(value) or not math.isfinite(float(value)):
            raise ValueError("record values must be finite scalars")
        if reference_key is not None:
            reference = row[reference_key]
            if isinstance(reference, bool) or not np.isscalar(reference) or not math.isfinite(float(reference)):
                raise ValueError("reference values must be finite scalars")
            value = float(value) - float(reference)
        roots[(episode, root)].append(float(value))
        count += 1
    if not roots:
        raise ValueError("records must not be empty")
    episodes = defaultdict(list)
    root_records = []
    for (episode, root), values in roots.items():
        mean = float(np.mean(values))
        episodes[episode].append(mean)
        root_records.append({episode_key: episode, root_key: root, "mean": mean, "replicates": len(values)})
    episode_records = [{episode_key: episode, "mean": float(np.mean(values)), "roots": len(values)}
                       for episode, values in episodes.items()]
    means = np.asarray([row["mean"] for row in episode_records])
    return {"mean": float(means.mean()), "episode_se": _json_number(_standard_error(means)),
            "episodes": len(episodes), "roots": len(roots), "records": count,
            "episode_means": episode_records, "root_means": root_records,
            "uncertainty_unit": "episode"}


def feature_metrics(features, *, center=True, inactive_tolerance=1e-12):
    """Descriptive feature diagnostics, not a causal claim about plasticity."""
    features = _finite_tensor(features, "features")
    if features.ndim < 2 or features.shape[0] == 0 or features.numel() == 0:
        raise ValueError("features must be [N,features...] and nonempty")
    if not math.isfinite(inactive_tolerance) or inactive_tolerance < 0:
        raise ValueError("inactive_tolerance must be finite and nonnegative")
    values = features.detach().reshape(features.shape[0], -1).double()
    spectrum_input = values - values.mean(dim=0) if center else values
    spectrum = torch.linalg.svdvals(spectrum_input).square()
    energy = spectrum.sum()
    if energy > 0:
        probabilities = spectrum[spectrum > 0] / energy
        effective_rank = torch.exp(-(probabilities * probabilities.log()).sum()).item()
        stable_rank = (energy / spectrum.max()).item()
    else:
        effective_rank = stable_rank = 0.0
    return {"effective_rank": effective_rank, "stable_rank": stable_rank,
            "inactive_fraction": (values.abs().amax(dim=0) <= inactive_tolerance).double().mean().item(),
            "constant_fraction": (values.std(dim=0, unbiased=False) <= inactive_tolerance).double().mean().item(),
            "samples": values.shape[0], "features": values.shape[1], "centered": bool(center)}


def _matching_target(prediction, target):
    if prediction.ndim == 0 or prediction.shape[0] != target.shape[0]:
        raise ValueError("predict must preserve the leading sample dimension")
    if prediction.ndim == 1 and target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    elif target.ndim == 1 and prediction.ndim > 1:
        target = target.reshape(target.shape[0], *([1] * (prediction.ndim - 1)))
    if target.ndim != prediction.ndim:
        raise ValueError("prediction and target shapes are incompatible")
    try:
        prediction, target = torch.broadcast_tensors(prediction, target)
    except RuntimeError as error:
        raise ValueError("prediction and target shapes are incompatible") from error
    return prediction, target


def fit_stationary_targets(
    module, predict, train_inputs, train_targets, heldout_inputs, heldout_targets, *,
    batch_indices, learning_rate=1e-3, evaluation_steps=None, seed=0,
    trainable_selector=None, loss_fn=None, feature_fn=None, train_mode=True,
):
    """Fit a deepcopy using fresh Adam, fixed labels/batches, and isolated RNG.

    ``predict(module, inputs[N,...])`` returns [N] or [N,K,...]; scalar labels
    broadcast over heads/features only, never across samples. Default loss is
    decoded-value/action MSE. An optional scalar ``loss_fn(pred,target)`` may
    implement a distributional loss. Inputs and targets are detached snapshots.
    ``batch_indices[updates,batch]`` is supplied by the caller and can be reused
    across prior/carried fits. Dropout receives the same seeded random stream.

    Existing requires_grad settings are preserved unless trainable_selector
    is supplied as ``(name, parameter)->bool``. Held-out/train curves evaluate
    in eval mode; training defaults to train mode. Evaluation cannot mutate the
    originals or consume training/dropout RNG. Feature rank is optional.
    Returns a fitted ``model`` clone and JSON-ready ``curve`` (including step 0).
    This measures stationary learning; it does not itself diagnose plasticity.
    """
    if not isinstance(module, nn.Module):
        raise ValueError("module must be a torch.nn.Module")
    for data, label in ((train_inputs, "train_inputs"), (train_targets, "train_targets"),
                        (heldout_inputs, "heldout_inputs"), (heldout_targets, "heldout_targets")):
        _finite_tensor(data, label)
        if data.ndim == 0 or data.shape[0] == 0:
            raise ValueError(f"{label} must have a nonempty sample axis")
    if train_inputs.shape[0] != train_targets.shape[0] or heldout_inputs.shape[0] != heldout_targets.shape[0]:
        raise ValueError("input/target sample counts disagree")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning_rate must be finite and positive")
    raw_indices = torch.as_tensor(batch_indices)
    if raw_indices.ndim != 2 or raw_indices.shape[1] == 0 or raw_indices.dtype == torch.bool or raw_indices.is_floating_point():
        raise ValueError("batch_indices must be an integer [updates,batch] array")
    indices = raw_indices.to(dtype=torch.long, device=train_inputs.device)
    if indices.numel() and (indices.min() < 0 or indices.max() >= len(train_inputs)):
        raise ValueError("batch_indices contains an out-of-range row")
    updates = len(indices)
    checkpoints = set(range(updates + 1)) if evaluation_steps is None else set(evaluation_steps)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i <= updates for i in checkpoints):
        raise ValueError("evaluation_steps must lie between zero and the update count")
    checkpoints.update((0, updates))
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    model = copy.deepcopy(module)
    if trainable_selector is not None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(bool(trainable_selector(name, parameter)))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("stationary fitting requires at least one trainable parameter")
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    initial = [parameter.detach().clone() for parameter in parameters]
    train_inputs, train_targets = train_inputs.detach().clone(), train_targets.detach().clone()
    heldout_inputs, heldout_targets = heldout_inputs.detach().clone(), heldout_targets.detach().clone()
    devices = sorted({tensor.device.index for tensor in list(model.parameters()) + list(model.buffers())
                      + [train_inputs, train_targets, heldout_inputs, heldout_targets] if tensor.is_cuda})
    curve = []

    def evaluate(step, gradient_norm=None):
        # fork_rng also prevents optional stochastic prediction/feature probes
        # from changing the fixed training random stream.
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            row = {"step": int(step), "gradient_norm": gradient_norm}
            for name, inputs, targets in (("train", train_inputs, train_targets),
                                          ("heldout", heldout_inputs, heldout_targets)):
                prediction = _finite_tensor(predict(model, inputs), f"{name} prediction")
                prediction, matched = _matching_target(prediction, targets)
                error = prediction - matched
                row[f"{name}_mse"] = error.square().mean().item()
                row[f"{name}_rmse"] = math.sqrt(row[f"{name}_mse"])
                row[f"{name}_bias"] = error.mean().item()
            row["parameter_drift_l2"] = math.sqrt(sum((p.detach() - p0).double().square().sum().item()
                                                      for p, p0 in zip(parameters, initial)))
            if feature_fn is not None:
                row["features"] = feature_metrics(feature_fn(model, heldout_inputs))
            curve.append(row)
        model.train(bool(train_mode))

    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(int(seed)).get_state())
        for device in devices:
            torch.cuda.set_rng_state(torch.Generator(device=f"cuda:{device}").manual_seed(int(seed)).get_state(), device)
        evaluate(0)
        for step, batch in enumerate(indices, 1):
            optimizer.zero_grad(set_to_none=True)
            prediction = _finite_tensor(predict(model, train_inputs[batch]), "training prediction")
            prediction, target = _matching_target(prediction, train_targets[batch])
            loss = (prediction - target).square().mean() if loss_fn is None else loss_fn(prediction, target)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("loss_fn must produce a finite scalar tensor")
            loss.backward()
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise ValueError("stationary fit has missing or non-finite gradients")
            gradient_norm = math.sqrt(sum(gradient.detach().double().square().sum().item() for gradient in gradients))
            optimizer.step()
            if step in checkpoints:
                evaluate(step, gradient_norm)
    model.eval()
    return {"model": model, "curve": curve, "updates": updates,
            "train_samples": len(train_inputs), "heldout_samples": len(heldout_inputs),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "seed": int(seed), "optimizer": "Adam", "learning_rate": float(learning_rate)}
