#!/usr/bin/env python3
"""Run the pinned official smoke and assert its completed learner state."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


OFFICIAL_COMMIT = "9a6832bb742ef01bbe9f1e06153a9338e612dae5"
EVALUATION_CSV_FIELDS = (
    "implementation",
    "source_commit",
    "base_seed",
    "num_seeds",
    "seed_index",
    "seed",
    "evaluation_index",
    "decision_step",
    "raw_frame",
    "paper_raw_frame",
    "action_repeat",
    "return",
)


def _validate_evaluation_capture_args(args) -> bool:
    """Validate the all-or-none arguments for durable evaluation capture."""

    values = {
        "--base-seed": args.base_seed,
        "--num-seeds": args.num_seeds,
        "--action-repeat": args.action_repeat,
        "--expected-evaluation-rows": args.expected_evaluation_rows,
    }
    if args.evaluation_csv is None:
        supplied = [name for name, value in values.items() if value is not None]
        if supplied:
            raise SystemExit(
                f"{', '.join(supplied)} require --evaluation-csv"
            )
        return False

    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise SystemExit(
            "--evaluation-csv requires " + ", ".join(missing)
        )
    if args.num_seeds < 1:
        raise SystemExit("--num-seeds must be positive")
    if args.action_repeat < 1:
        raise SystemExit("--action-repeat must be positive")
    if args.expected_evaluation_rows < 1:
        raise SystemExit("--expected-evaluation-rows must be positive")
    if args.expected_evaluation_rows % args.num_seeds:
        raise SystemExit(
            "--expected-evaluation-rows must be divisible by --num-seeds"
        )
    return True


class _EvaluationCsvCapture:
    """Durably record official evaluation returns without editing upstream."""

    def __init__(
        self,
        path: Path,
        *,
        base_seed: int,
        num_seeds: int,
        action_repeat: int,
        expected_rows: int,
    ) -> None:
        self.path = Path(path)
        self.base_seed = int(base_seed)
        self.num_seeds = int(num_seeds)
        self.action_repeat = int(action_repeat)
        self.expected_rows = int(expected_rows)
        self.row_count = 0
        self.evaluation_count = 0
        self._stream = self.path.open("x", encoding="utf-8", newline="")
        try:
            self._writer = csv.DictWriter(
                self._stream,
                fieldnames=EVALUATION_CSV_FIELDS,
            )
            self._writer.writeheader()
            self._sync()
        except Exception:
            self._stream.close()
            raise

    def _sync(self) -> None:
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def assert_expected_rows(self) -> None:
        if self.row_count != self.expected_rows:
            raise SystemExit(
                "official XQC evaluation capture wrote "
                f"{self.row_count} rows; expected {self.expected_rows}"
            )

    def record(self, step, infos) -> None:
        """Record one evaluation event; ignore ordinary learner logging."""

        if "return" not in infos:
            return

        import numpy as np

        try:
            raw_frame_value = float(step)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SystemExit(
                f"official XQC evaluation step is not numeric: {step!r}"
            ) from exc
        if not math.isfinite(raw_frame_value):
            raise SystemExit(
                "official XQC evaluation step must be a finite, non-negative "
                f"integer, found {step!r}"
            )
        raw_frame = int(raw_frame_value)
        if raw_frame_value != raw_frame or raw_frame < 0:
            raise SystemExit(
                "official XQC evaluation step must be a finite, non-negative "
                f"integer, found {step!r}"
            )
        if raw_frame % self.action_repeat:
            raise SystemExit(
                f"official XQC evaluation raw frame {raw_frame} is not divisible "
                f"by action repeat {self.action_repeat}"
            )

        try:
            returns = np.asarray(infos["return"])
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                "official XQC evaluation returns are not array-like"
            ) from exc
        if returns.ndim != 1 or returns.size == 0:
            raise SystemExit(
                "official XQC evaluation returns must be a non-empty vector, "
                f"found shape {returns.shape}"
            )
        if returns.size != self.num_seeds:
            raise SystemExit(
                "official XQC evaluation return count does not match "
                f"--num-seeds: found {returns.size}, expected {self.num_seeds}"
            )
        if returns.dtype.kind not in "biuf":
            raise SystemExit(
                "official XQC evaluation returns must be real numeric values, "
                f"found dtype {returns.dtype}"
            )
        if not bool(np.isfinite(returns).all()):
            raise SystemExit(
                "official XQC evaluation returns contain a non-finite value"
            )
        return_values = [float(value) for value in returns]
        if not all(math.isfinite(value) for value in return_values):
            raise SystemExit(
                "official XQC evaluation returns cannot be represented as "
                "finite CSV numbers"
            )

        evaluation_index = self.evaluation_count
        decision_step = raw_frame // self.action_repeat
        paper_raw_frame = 0 if evaluation_index == 0 else raw_frame
        for seed_index, value in enumerate(return_values):
            self._writer.writerow(
                {
                    "implementation": "official-jax",
                    "source_commit": OFFICIAL_COMMIT,
                    "base_seed": self.base_seed,
                    "num_seeds": self.num_seeds,
                    "seed_index": seed_index,
                    "seed": self.base_seed + seed_index,
                    "evaluation_index": evaluation_index,
                    "decision_step": decision_step,
                    "raw_frame": raw_frame,
                    "paper_raw_frame": paper_raw_frame,
                    "action_repeat": self.action_repeat,
                    "return": value,
                }
            )
            self.row_count += 1
        self.evaluation_count += 1
        self._sync()


@contextmanager
def _patched_evaluation_logging(logging_module, capture):
    """Patch only the official call window and always restore its logger."""

    original_logging = logging_module.log_multiple_seeds_to_wandb

    def captured_logging(step, infos, fps=30):
        capture.record(step, infos)
        return original_logging(step, infos, fps=fps)

    logging_module.log_multiple_seeds_to_wandb = captured_logging
    try:
        yield
    finally:
        logging_module.log_multiple_seeds_to_wandb = original_logging


def _run_official_main(
    official_main,
    *,
    logging_module=None,
    evaluation_capture=None,
) -> None:
    """Run upstream directly unless the optional capture was requested."""

    if evaluation_capture is None:
        official_main()
        return
    if logging_module is None:
        raise RuntimeError("evaluation capture requires the official logging module")
    with _patched_evaluation_logging(logging_module, evaluation_capture):
        official_main()
    evaluation_capture.assert_expected_rows()


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
    parser.add_argument("--evaluation-csv", type=Path)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--num-seeds", type=int)
    parser.add_argument("--action-repeat", type=int)
    parser.add_argument("--expected-evaluation-rows", type=int)
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
    capture_evaluations = _validate_evaluation_capture_args(args)

    sys.path.insert(0, str(repo))
    sys.argv = [sys.argv[0], *hydra_args]

    from xqc.agents import XQCLearner
    from xqc.normalization import RewardNormalizer
    from train_parallel import main as official_main

    evaluation_capture = None
    official_logging = None
    if capture_evaluations:
        import xqc.logging as official_logging

        try:
            evaluation_capture = _EvaluationCsvCapture(
                args.evaluation_csv,
                base_seed=args.base_seed,
                num_seeds=args.num_seeds,
                action_repeat=args.action_repeat,
                expected_rows=args.expected_evaluation_rows,
            )
        except FileExistsError as exc:
            raise SystemExit(
                f"official XQC evaluation CSV already exists: {args.evaluation_csv}"
            ) from exc

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
        _run_official_main(
            official_main,
            logging_module=official_logging,
            evaluation_capture=evaluation_capture,
        )
    finally:
        XQCLearner.update = original_update
        RewardNormalizer.__init__ = original_reward_normalizer_init
        if evaluation_capture is not None:
            evaluation_capture.close()

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
