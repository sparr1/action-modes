#!/usr/bin/env python3
"""Run the pinned official smoke and assert its completed learner state."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path


OFFICIAL_COMMIT = "9a6832bb742ef01bbe9f1e06153a9338e612dae5"


def _assert_finite_leaves(label, leaves) -> None:
    """Reject non-numeric or non-finite leaves from an official JAX state."""

    import numpy as np

    count = 0
    for index, leaf in enumerate(leaves):
        try:
            value = np.asarray(leaf)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"official XQC {label} leaf {index} is not array-like"
            ) from exc
        if value.dtype.kind not in "biufc":
            raise SystemExit(
                f"official XQC {label} leaf {index} is not numeric "
                f"(dtype={value.dtype})"
            )
        count += 1
        if not bool(np.isfinite(value).all()):
            raise SystemExit(
                f"official XQC {label} contains a non-finite value in leaf {index}"
            )
    if count == 0:
        raise SystemExit(f"official XQC {label} contains no numeric state")


def _projected_column_residual(flat_params) -> float:
    """Return the largest official Norm Network column-norm residual."""

    import numpy as np

    residual = 0.0
    projected_kernel_count = 0
    for raw_path, parameter in flat_params.items():
        path = "/".join(raw_path) if isinstance(raw_path, tuple) else str(raw_path)
        if not path.endswith("/kernel"):
            continue
        layer_path = path.rsplit("/", 1)[0]
        if not (
            ("MLP_0" in layer_path and "Dense" in layer_path)
            or "predictor" in layer_path
        ):
            continue

        value = np.asarray(parameter)
        if value.ndim < 2:
            raise SystemExit(
                f"official XQC projected kernel {path} has rank {value.ndim}"
            )
        if not bool(np.isfinite(value).all()):
            raise SystemExit(
                f"official XQC projected kernel {path} contains a non-finite value"
            )
        column_norms = np.linalg.norm(value, axis=-2)
        residual = max(
            residual,
            float(np.max(np.abs(column_norms - 1.0))),
        )
        projected_kernel_count += 1

    if projected_kernel_count == 0:
        raise SystemExit("official XQC state contains no projected kernels")
    return residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--expected-updates", type=int, required=True)
    parser.add_argument(
        "--max-projection-residual",
        type=float,
        required=True,
    )
    args, hydra_args = parser.parse_known_args()

    repo = args.official_repo.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise SystemExit(
            f"official checkout must be {OFFICIAL_COMMIT}, found {commit}"
        )
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    ):
        raise SystemExit("official checkout must be clean")
    if args.expected_updates < 1:
        raise SystemExit("--expected-updates must be positive")
    if (
        not math.isfinite(args.max_projection_residual)
        or args.max_projection_residual < 0.0
    ):
        raise SystemExit("--max-projection-residual must be finite and non-negative")

    sys.path.insert(0, str(repo))
    sys.argv = [sys.argv[0], *hydra_args]

    from xqc.agents import XQCLearner
    from xqc.normalization import RewardNormalizer
    from train_parallel import main as official_main

    completed_updates = 0
    last_learner = None
    last_info = None
    reward_normalizer = None
    original_update = XQCLearner.update
    original_reward_normalizer_init = RewardNormalizer.__init__

    def counted_update(self, batch, num_updates=1, time_to_intervene=False):
        nonlocal completed_updates, last_info, last_learner
        result = original_update(
            self,
            batch,
            num_updates=num_updates,
            time_to_intervene=time_to_intervene,
        )
        completed_updates += int(num_updates)
        last_learner = self
        last_info = result
        return result

    def captured_reward_normalizer_init(self, *init_args, **init_kwargs):
        nonlocal reward_normalizer
        original_reward_normalizer_init(self, *init_args, **init_kwargs)
        reward_normalizer = self

    XQCLearner.update = counted_update
    RewardNormalizer.__init__ = captured_reward_normalizer_init
    try:
        official_main()
    finally:
        XQCLearner.update = original_update
        RewardNormalizer.__init__ = original_reward_normalizer_init

    if completed_updates != args.expected_updates:
        raise SystemExit(
            "official XQC smoke completed "
            f"{completed_updates} learner updates; expected {args.expected_updates}"
        )
    if last_learner is None or last_info is None:
        raise SystemExit("official XQC smoke did not expose a completed learner state")
    if reward_normalizer is None:
        raise SystemExit("official XQC smoke did not construct its reward normalizer")

    import jax
    from flax.traverse_util import flatten_dict

    learned_state = {
        "actor": last_learner.actor,
        "critic": last_learner.critic,
        "target_critic": last_learner.target_critic,
        "temperature": last_learner.temperature,
        "rng": last_learner.rng,
        "step": last_learner.step,
    }
    reward_state = {
        "gamma": reward_normalizer.gamma,
        "g_max": reward_normalizer.g_max,
        "return": reward_normalizer.G,
        "epsilon": reward_normalizer.epsilon,
        "running_moments": vars(reward_normalizer.G_rms),
    }
    _assert_finite_leaves(
        "learner state", jax.tree_util.tree_leaves(learned_state)
    )
    _assert_finite_leaves(
        "reward-normalizer state", jax.tree_util.tree_leaves(reward_state)
    )
    _assert_finite_leaves(
        "final update diagnostics", jax.tree_util.tree_leaves(last_info)
    )

    actor_residual = _projected_column_residual(
        flatten_dict(last_learner.actor.params, sep="/")
    )
    critic_residual = _projected_column_residual(
        flatten_dict(last_learner.critic.params, sep="/")
    )
    max_residual = max(actor_residual, critic_residual)
    if max_residual > args.max_projection_residual:
        raise SystemExit(
            "official XQC projected-column residual exceeds limit: "
            f"actor={actor_residual:.9g}, critic={critic_residual:.9g}, "
            f"limit={args.max_projection_residual:.9g}"
        )

    print(
        f"Official XQC learner updates: {completed_updates}; "
        f"projected-column residuals: actor={actor_residual:.9g}, "
        f"critic={critic_residual:.9g}"
    )


if __name__ == "__main__":
    main()
