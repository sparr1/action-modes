#!/usr/bin/env python3
"""Regenerate the compact XQC oracle with the pinned official JAX checkout.

This script is intentionally not imported by the PyTorch test suite. Run it
with the official XQC environment, review the resulting JSON, and update the
tracked fixture explicitly:

  ../xqc/.venv/bin/python tests/oracles/generate_xqc_official_fixture.py \
    --official-repo ../xqc --output /tmp/xqc_official_fixture.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


OFFICIAL_COMMIT = "9a6832bb742ef01bbe9f1e06153a9338e612dae5"


def _nested_arrays(value):
    import numpy as np

    array = np.asarray(value)
    # Canonical decimal rounding makes the reviewed JSON byte-reproducible
    # across the official CPU XLA backends on arm64 macOS and x86_64 Linux.
    # The PyTorch parity assertions remain tighter than their 1e-6 tolerance.
    array = np.round(array.astype(np.float64), decimals=6)
    if array.ndim == 0:
        return float(array)
    return array.tolist()


def _head_distribution_fixture():
    import jax.numpy as jnp
    import numpy as np
    from flax.core import freeze, unfreeze
    from xqc.networks.critic_net import VMapCritic

    critic = VMapCritic(
        hidden_dims=(3,),
        n_outputs=5,
        min_v=-2.0,
        max_v=2.0,
        n_critics=2,
        pre_activation_bn=True,
        use_layer_norm=False,
        use_batch_norm=True,
        skip_connections=False,
    )
    observations = jnp.array(
        [[0.2, -0.5], [1.1, 0.3], [-0.7, 0.8]], dtype=jnp.float32
    )
    actions = jnp.array([[0.4], [-0.2], [0.6]], dtype=jnp.float32)
    variables = unfreeze(
        critic.init(
            __import__("jax").random.PRNGKey(19), observations, actions, False
        )
    )
    prefix = variables["params"]["VmapCritic_0"]
    stats = variables["batch_stats"]["VmapCritic_0"]

    input_bn = prefix["BatchNormEmbedder_0"]["BatchNorm_0"]
    input_bn["scale"] = jnp.array(
        [[1.1, 0.9, 1.2], [0.8, 1.3, 0.7]], dtype=jnp.float32
    )
    input_bn["bias"] = jnp.array(
        [[-0.1, 0.2, 0.05], [0.15, -0.2, 0.1]], dtype=jnp.float32
    )
    input_stats = stats["BatchNormEmbedder_0"]["BatchNorm_0"]
    input_stats["mean"] = jnp.array(
        [[0.2, -0.3, 0.1], [-0.4, 0.25, -0.15]], dtype=jnp.float32
    )
    input_stats["var"] = jnp.array(
        [[1.4, 0.7, 1.1], [0.8, 1.6, 0.9]], dtype=jnp.float32
    )

    block = prefix["MLP_0"]["XQCBlock_0"]
    block["Dense_0"]["kernel"] = jnp.array(
        [
            [[0.2, -0.4, 0.6], [0.8, 0.1, -0.3], [-0.5, 0.7, 0.25]],
            [[-0.3, 0.5, 0.2], [0.4, -0.6, 0.7], [0.9, 0.15, -0.45]],
        ],
        dtype=jnp.float32,
    )
    block_bn = block["BatchNorm_0"]
    block_bn["scale"] = jnp.array(
        [[1.0, 0.85, 1.15], [0.75, 1.2, 0.95]], dtype=jnp.float32
    )
    block_bn["bias"] = jnp.array(
        [[0.05, -0.1, 0.2], [-0.15, 0.25, 0.1]], dtype=jnp.float32
    )
    block_stats = stats["MLP_0"]["XQCBlock_0"]["BatchNorm_0"]
    block_stats["mean"] = jnp.array(
        [[0.1, -0.2, 0.3], [-0.25, 0.15, -0.05]], dtype=jnp.float32
    )
    block_stats["var"] = jnp.array(
        [[0.9, 1.3, 0.6], [1.1, 0.75, 1.4]], dtype=jnp.float32
    )

    predictor = prefix["predictor_scalar"]["value"]
    predictor["kernel"] = jnp.array(
        [
            [
                [0.2, -0.1, 0.3, -0.4, 0.5],
                [0.6, 0.25, -0.35, 0.15, -0.2],
                [-0.45, 0.55, 0.1, 0.35, -0.3],
            ],
            [
                [-0.25, 0.4, 0.15, -0.35, 0.2],
                [0.5, -0.3, 0.45, 0.1, -0.15],
                [0.2, 0.35, -0.5, 0.25, 0.4],
            ],
        ],
        dtype=jnp.float32,
    )
    predictor["bias"] = jnp.array(
        [[0.1, -0.2, 0.05, 0.15, -0.1], [-0.05, 0.1, -0.15, 0.2, 0.0]],
        dtype=jnp.float32,
    )
    variables = freeze(variables)

    (values, info) = critic.apply(variables, observations, actions, False)
    (_, train_info), updates = critic.apply(
        variables,
        observations,
        actions,
        True,
        mutable=["batch_stats"],
    )
    updated = updates["batch_stats"]["VmapCritic_0"]

    return {
        "observations": _nested_arrays(observations),
        "actions": _nested_arrays(actions),
        "input_bn_scale": _nested_arrays(input_bn["scale"]),
        "input_bn_bias": _nested_arrays(input_bn["bias"]),
        "input_bn_mean": _nested_arrays(input_stats["mean"]),
        "input_bn_var": _nested_arrays(input_stats["var"]),
        "hidden_kernel": _nested_arrays(block["Dense_0"]["kernel"]),
        "hidden_bn_scale": _nested_arrays(block_bn["scale"]),
        "hidden_bn_bias": _nested_arrays(block_bn["bias"]),
        "hidden_bn_mean": _nested_arrays(block_stats["mean"]),
        "hidden_bn_var": _nested_arrays(block_stats["var"]),
        "output_kernel": _nested_arrays(predictor["kernel"]),
        "output_bias": _nested_arrays(predictor["bias"]),
        "running_log_probs": _nested_arrays(info["log_probs"]),
        "running_values": _nested_arrays(values),
        "batch_log_probs": _nested_arrays(train_info["log_probs"]),
        "updated_input_mean": _nested_arrays(
            updated["BatchNormEmbedder_0"]["BatchNorm_0"]["mean"]
        ),
        "updated_input_var": _nested_arrays(
            updated["BatchNormEmbedder_0"]["BatchNorm_0"]["var"]
        ),
        "updated_hidden_mean": _nested_arrays(
            updated["MLP_0"]["XQCBlock_0"]["BatchNorm_0"]["mean"]
        ),
        "updated_hidden_var": _nested_arrays(
            updated["MLP_0"]["XQCBlock_0"]["BatchNorm_0"]["var"]
        ),
    }


def _actor_fixture():
    import jax
    import jax.numpy as jnp
    from flax.core import freeze, unfreeze
    from xqc.networks.policies import NormalTanhPolicy

    actor = NormalTanhPolicy(
        hidden_dims=(3,),
        action_dim=1,
        pre_activation_bn=True,
        use_layer_norm=False,
        use_batch_norm=True,
        skip_connections=False,
    )
    observations = jnp.array(
        [[0.2, -0.5], [1.1, 0.3], [-0.7, 0.8]], dtype=jnp.float32
    )
    variables = unfreeze(
        actor.init(jax.random.PRNGKey(7), observations, training=False)
    )
    params = variables["params"]
    stats = variables["batch_stats"]
    input_bn = params["BatchNormEmbedder_0"]["BatchNorm_0"]
    input_bn["scale"] = jnp.array([1.1, 0.9], dtype=jnp.float32)
    input_bn["bias"] = jnp.array([-0.2, 0.3], dtype=jnp.float32)
    input_stats = stats["BatchNormEmbedder_0"]["BatchNorm_0"]
    input_stats["mean"] = jnp.array([0.5, -0.25], dtype=jnp.float32)
    input_stats["var"] = jnp.array([1.5, 0.75], dtype=jnp.float32)
    block = params["MLP_0"]["XQCBlock_0"]
    block["Dense_0"]["kernel"] = jnp.array(
        [[0.2, -0.4, 0.6], [0.8, 0.1, -0.3]], dtype=jnp.float32
    )
    block_bn = block["BatchNorm_0"]
    block_bn["scale"] = jnp.array([1.0, 0.85, 1.15], dtype=jnp.float32)
    block_bn["bias"] = jnp.array([0.05, -0.1, 0.2], dtype=jnp.float32)
    block_stats = stats["MLP_0"]["XQCBlock_0"]["BatchNorm_0"]
    block_stats["mean"] = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    block_stats["var"] = jnp.array([0.9, 1.3, 0.6], dtype=jnp.float32)
    predictor = params["predictor_tanh_gauss"]
    predictor["mean"]["kernel"] = jnp.array(
        [[0.3], [-0.5], [0.7]], dtype=jnp.float32
    )
    predictor["mean"]["bias"] = jnp.array([0.1], dtype=jnp.float32)
    predictor["log_std"]["kernel"] = jnp.array(
        [[-0.2], [0.4], [0.15]], dtype=jnp.float32
    )
    predictor["log_std"]["bias"] = jnp.array([-0.35], dtype=jnp.float32)
    variables = freeze(variables)

    dist = actor.apply(variables, observations, training=False)
    means = dist.distribution.loc
    log_stds = jnp.log(dist.distribution.scale.diag)
    noise = jnp.array([[0.25], [-1.0], [0.6]], dtype=jnp.float32)
    actions = jnp.tanh(means + jnp.exp(log_stds) * noise)
    log_probs = dist.log_prob(actions)

    return {
        "observations": _nested_arrays(observations),
        "noise": _nested_arrays(noise),
        "input_bn_scale": _nested_arrays(input_bn["scale"]),
        "input_bn_bias": _nested_arrays(input_bn["bias"]),
        "input_bn_mean": _nested_arrays(input_stats["mean"]),
        "input_bn_var": _nested_arrays(input_stats["var"]),
        "hidden_kernel": _nested_arrays(block["Dense_0"]["kernel"]),
        "hidden_bn_scale": _nested_arrays(block_bn["scale"]),
        "hidden_bn_bias": _nested_arrays(block_bn["bias"]),
        "hidden_bn_mean": _nested_arrays(block_stats["mean"]),
        "hidden_bn_var": _nested_arrays(block_stats["var"]),
        "mean_kernel": _nested_arrays(predictor["mean"]["kernel"]),
        "mean_bias": _nested_arrays(predictor["mean"]["bias"]),
        "log_std_kernel": _nested_arrays(predictor["log_std"]["kernel"]),
        "log_std_bias": _nested_arrays(predictor["log_std"]["bias"]),
        "mean": _nested_arrays(means),
        "log_std": _nested_arrays(log_stds),
        "actions": _nested_arrays(actions),
        "log_probs": _nested_arrays(log_probs),
    }


def _categorical_fixture():
    import jax
    import jax.numpy as jnp
    from xqc.agents.xqc.critic import categorical_td_loss

    pred_logits = jnp.array(
        [[0.2, -0.1, 0.5, -0.4, 0.0], [-0.3, 0.4, 0.1, 0.2, -0.2]],
        dtype=jnp.float32,
    )
    target_logits = jnp.array(
        [[-0.2, 0.3, 0.1, -0.1, 0.4], [0.5, -0.4, 0.2, 0.0, -0.3]],
        dtype=jnp.float32,
    )
    pred_log_probs = jax.nn.log_softmax(pred_logits, axis=-1)
    target_log_probs = jax.nn.log_softmax(target_logits, axis=-1)
    rewards = jnp.array([0.35, -1.25], dtype=jnp.float32)
    masks = jnp.array([1.0, 0.0], dtype=jnp.float32)
    entropy_shift = jnp.array([-0.12, -0.08], dtype=jnp.float32)
    support = jnp.linspace(-2.0, 2.0, 5)

    target_values = rewards[:, None] + 0.9 * (
        support[None, :] - entropy_shift[:, None]
    ) * masks[:, None]
    clipped = jnp.clip(target_values, -2.0, 2.0)
    b = (clipped + 2.0) / 1.0
    lower = jnp.floor(b)
    upper = jnp.ceil(b)
    lower_mask = jax.nn.one_hot(lower.reshape(-1), 5).reshape(2, 5, 5)
    upper_mask = jax.nn.one_hot(upper.reshape(-1), 5).reshape(2, 5, 5)
    probs = jnp.exp(target_log_probs)
    lower_mass = (probs * (upper + (lower == upper) - b))[..., None]
    upper_mass = (probs * (b - lower))[..., None]
    projected = jnp.sum(
        lower_mass * lower_mask + upper_mass * upper_mask, axis=1
    )
    loss, info = categorical_td_loss(
        pred_log_probs,
        target_log_probs,
        rewards,
        masks,
        entropy_shift,
        0.9,
        5,
        -2.0,
        2.0,
    )

    def loss_from_logits(logits):
        candidate_log_probs = jax.nn.log_softmax(logits, axis=-1)
        candidate_loss, _ = categorical_td_loss(
            candidate_log_probs,
            target_log_probs,
            rewards,
            masks,
            entropy_shift,
            0.9,
            5,
            -2.0,
            2.0,
        )
        return candidate_loss

    return {
        "pred_logits": _nested_arrays(pred_logits),
        "pred_log_probs": _nested_arrays(pred_log_probs),
        "pred_logits_gradient": _nested_arrays(jax.grad(loss_from_logits)(pred_logits)),
        "target_log_probs": _nested_arrays(target_log_probs),
        "rewards": _nested_arrays(rewards),
        "masks": _nested_arrays(masks),
        "entropy_shift": _nested_arrays(entropy_shift),
        "support": _nested_arrays(support),
        "discount": 0.9,
        "projected_probs": _nested_arrays(projected),
        "clip_fraction": _nested_arrays(info["clip_percentage"]),
        "cross_entropy": _nested_arrays(loss),
    }


def _projection_and_polyak_fixture():
    import jax.numpy as jnp
    from xqc.networks.common import norm_dense_layer

    initial_kernel = jnp.array([[3.0, 0.0], [4.0, 5.0]], dtype=jnp.float32)
    hidden = {"hidden/kernel": initial_kernel.copy()}
    projected_hidden = norm_dense_layer(hidden, "hidden", norm_bias=True)
    final = {
        "final/kernel": initial_kernel.copy(),
        "final/bias": jnp.array([7.0, -2.0], dtype=jnp.float32),
    }
    projected_final = norm_dense_layer(final, "final", norm_bias=False)
    source = jnp.array([[1.0, -2.0], [3.0, 4.0]], dtype=jnp.float32)
    target = jnp.array([[-1.0, 2.0], [0.5, -0.5]], dtype=jnp.float32)
    tau = 0.2
    return {
        "hidden_kernel": _nested_arrays(initial_kernel),
        "projected_hidden_kernel": _nested_arrays(
            projected_hidden["hidden/kernel"]
        ),
        "final_kernel": _nested_arrays(initial_kernel),
        "final_bias": _nested_arrays(final["final/bias"]),
        "projected_final_kernel": _nested_arrays(
            projected_final["final/kernel"]
        ),
        "projected_final_bias": _nested_arrays(
            projected_final["final/bias"]
        ),
        "polyak_source": _nested_arrays(source),
        "polyak_target": _nested_arrays(target),
        "polyak_tau": tau,
        "polyak_result": _nested_arrays(tau * source + (1.0 - tau) * target),
    }


def _optimizer_fixture():
    import jax.numpy as jnp
    import optax

    params = jnp.array([[0.4, -0.2], [0.1, 0.7]], dtype=jnp.float32)
    gradients = (
        jnp.array([[0.3, -0.5], [0.2, 0.4]], dtype=jnp.float32),
        jnp.array([[-0.1, 0.25], [0.6, -0.2]], dtype=jnp.float32),
        jnp.array([[0.35, 0.15], [-0.45, 0.3]], dtype=jnp.float32),
    )
    schedule = optax.linear_schedule(3e-4, 3e-5, transition_steps=8)
    optimizer = optax.adamw(schedule, weight_decay=0.0, eps=1e-8)
    state = optimizer.init(params)
    values = [params]
    learning_rates = []
    for step, grad in enumerate(gradients):
        learning_rates.append(schedule(step))
        updates, state = optimizer.update(grad, state, params)
        params = optax.apply_updates(params, updates)
        values.append(params)
    return {
        "initial": _nested_arrays(values[0]),
        "gradients": [_nested_arrays(value) for value in gradients],
        "learning_rates": [_nested_arrays(value) for value in learning_rates],
        "parameters": [_nested_arrays(value) for value in values[1:]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.official_repo.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise SystemExit(
            f"official checkout must be {OFFICIAL_COMMIT}, found {commit}"
        )
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ):
        raise SystemExit("official checkout must be clean")
    sys.path.insert(0, str(repo))

    import flax
    import jax
    import optax

    fixture = {
        "metadata": {
            "official_commit": OFFICIAL_COMMIT,
            "jax": jax.__version__,
            "flax": flax.__version__,
            "optax": optax.__version__,
            "dtype": "float32",
        },
        "actor": _actor_fixture(),
        "critic": _head_distribution_fixture(),
        "categorical": _categorical_fixture(),
        "optimizer": _optimizer_fixture(),
        "projection_and_polyak": _projection_and_polyak_fixture(),
    }
    args.output.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
