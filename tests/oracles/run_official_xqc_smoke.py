#!/usr/bin/env python3
"""Run the pinned official smoke and assert its completed learner state."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path


OFFICIAL_COMMIT = "9a6832bb742ef01bbe9f1e06153a9338e612dae5"
CANONICAL_IMPLEMENTATION = "official-jax"
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


def _canonical_wandb_metadata(args):
    """Validate the opt-in single-seed comparison logging contract."""

    specific_values = {
        "--task": args.task,
        "--implementation": args.implementation,
        "--source-sha": args.source_sha,
    }
    if not args.canonical_wandb:
        supplied = [
            name for name, value in specific_values.items() if value is not None
        ]
        if supplied:
            raise SystemExit(
                f"{', '.join(supplied)} require --canonical-wandb"
            )
        return None

    required_values = {
        **specific_values,
        "--base-seed": args.base_seed,
        "--num-seeds": args.num_seeds,
        "--action-repeat": args.action_repeat,
    }
    missing = [name for name, value in required_values.items() if value is None]
    if missing:
        raise SystemExit("--canonical-wandb requires " + ", ".join(missing))
    if args.num_seeds != 1:
        raise SystemExit(
            "canonical W&B comparison requires --num-seeds 1 so each actual "
            "seed has its own run"
        )
    if args.action_repeat < 1:
        raise SystemExit("--action-repeat must be positive")
    task = str(args.task).strip()
    if not task:
        raise SystemExit("--task must be non-empty")
    if args.implementation != CANONICAL_IMPLEMENTATION:
        raise SystemExit(
            f"official wrapper requires --implementation {CANONICAL_IMPLEMENTATION}"
        )
    if args.source_sha != OFFICIAL_COMMIT:
        raise SystemExit(
            f"official wrapper requires --source-sha {OFFICIAL_COMMIT}"
        )
    comparison_id = os.environ.get("XQC_COMPARISON_ID")
    if comparison_id is None:
        raise SystemExit("--canonical-wandb requires XQC_COMPARISON_ID")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", comparison_id) is None:
        raise SystemExit(
            "XQC_COMPARISON_ID must use only letters, digits, '.', '_', and '-'"
        )
    return {
        "implementation": CANONICAL_IMPLEMENTATION,
        "seed": int(args.base_seed),
        "task": task,
        "source_sha": OFFICIAL_COMMIT,
        "comparison_id": comparison_id,
        "action_repeat": int(args.action_repeat),
    }


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


class _CanonicalWandbLogger:
    """Emit comparable single-seed return series on the raw-frame axis."""

    def __init__(self, wandb_module, *, action_repeat: int) -> None:
        self.wandb = wandb_module
        self.action_repeat = int(action_repeat)

    @staticmethod
    def _finite_float(value, label: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SystemExit(f"official XQC {label} is not numeric: {value!r}") from exc
        if not math.isfinite(result):
            raise SystemExit(f"official XQC {label} is not finite: {value!r}")
        return result

    def _log(self, raw_frame: int, decision_step: int, payload) -> None:
        raw_frame = int(raw_frame)
        decision_step = int(decision_step)
        if raw_frame != decision_step * self.action_repeat:
            raise SystemExit(
                "official XQC canonical W&B step does not match action repeat: "
                f"raw_frame={raw_frame}, decision_step={decision_step}, "
                f"action_repeat={self.action_repeat}"
            )
        # Training episode ends and evaluations can share one raw frame. W&B's
        # private monotonically increasing step must therefore remain free to
        # allocate separate rows; charts use comparison/raw_frame explicitly.
        self.wandb.log(
            {
                "comparison/raw_frame": raw_frame,
                "comparison/decision_step": decision_step,
                **payload,
            }
        )

    def record_training_episode(
        self,
        *,
        decision_step: int,
        episode_return,
        episode_length: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        episode_return = self._finite_float(episode_return, "training return")
        self._log(
            decision_step * self.action_repeat,
            decision_step,
            {
                "comparison/train_return": episode_return,
                "episode/return": episode_return,
                "episode/len": int(episode_length),
                "episode/terminated": int(bool(terminated)),
                "episode/truncated": int(bool(truncated)),
            },
        )

    def record_evaluation(self, raw_frame, infos) -> None:
        if "return" not in infos:
            return

        import numpy as np

        returns = np.asarray(infos["return"])
        if returns.shape != (1,):
            raise SystemExit(
                "official XQC canonical evaluation requires one return, "
                f"found shape {returns.shape}"
            )
        try:
            raw_frame_value = float(raw_frame)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SystemExit(
                f"official XQC evaluation step is not numeric: {raw_frame!r}"
            ) from exc
        raw_frame_int = int(raw_frame_value)
        if (
            not math.isfinite(raw_frame_value)
            or raw_frame_value != raw_frame_int
            or raw_frame_int < 0
            or raw_frame_int % self.action_repeat
        ):
            raise SystemExit(
                "official XQC evaluation raw frame must be a non-negative "
                f"multiple of action repeat, found {raw_frame!r}"
            )
        evaluation_return = self._finite_float(
            returns[0], "evaluation return"
        )
        self._log(
            raw_frame_int,
            raw_frame_int // self.action_repeat,
            {
                "comparison/eval_return": evaluation_return,
                "eval/episode_reward": evaluation_return,
            },
        )


@contextmanager
def _patched_wandb_initialization(wandb_module, metadata):
    """Label the upstream run and define the shared raw-frame metric axis."""

    original_init = wandb_module.init

    def canonical_init(*args, **kwargs):
        config = kwargs.get("config", {})
        if not isinstance(config, Mapping):
            raise SystemExit("official XQC resolved W&B config must be a mapping")
        if int(config.get("seed", -1)) != metadata["seed"]:
            raise SystemExit(
                "official XQC Hydra seed does not match canonical W&B seed"
            )
        if int(config.get("num_seeds", -1)) != 1:
            raise SystemExit("official XQC canonical W&B run must train one seed")
        env_config = config.get("env")
        if not isinstance(env_config, Mapping):
            raise SystemExit("official XQC Hydra env config must be a mapping")
        if env_config.get("name") != metadata["task"]:
            raise SystemExit(
                "official XQC Hydra task does not match canonical W&B task"
            )
        if int(env_config.get("action_repeat", -1)) != metadata["action_repeat"]:
            raise SystemExit(
                "official XQC Hydra action repeat does not match canonical W&B axis"
            )
        kwargs["config"] = {**dict(config), **metadata}
        kwargs["name"] = (
            f"xqc-{metadata['implementation']}-{metadata['task']}-"
            f"seed{metadata['seed']}"
        )
        kwargs["job_type"] = metadata["implementation"]
        group = os.environ.get("WANDB_RUN_GROUP")
        expected_group = (
            f"{metadata['comparison_id']}-{metadata['implementation']}"
        )
        if group != expected_group:
            raise SystemExit(
                "official XQC requires method-specific WANDB_RUN_GROUP="
                f"{expected_group!r}, found {group!r}"
            )
        kwargs["group"] = group
        run = original_init(*args, **kwargs)
        define_metric = getattr(run, "define_metric", None)
        if not callable(define_metric):
            raise SystemExit(
                "official XQC W&B run cannot define canonical comparison metrics"
            )
        define_metric("comparison/raw_frame")
        for name in (
            "comparison/decision_step",
            "comparison/train_return",
            "comparison/eval_return",
        ):
            define_metric(name, step_metric="comparison/raw_frame")
        return run

    wandb_module.init = canonical_init
    try:
        yield
    finally:
        wandb_module.init = original_init


@contextmanager
def _patched_training_returns(
    parallel_env_class,
    canonical_logger,
    *,
    num_seeds: int,
):
    """Observe upstream CPU environment outputs without modifying its checkout."""

    original_init = parallel_env_class.__init__
    original_step = parallel_env_class.step
    trackers = {}
    constructed = []

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        constructed.append(self)
        if len(constructed) == 1:
            trackers[id(self)] = {
                "decision_step": 0,
                "returns": [0.0] * num_seeds,
                "lengths": [0] * num_seeds,
            }

    def tracked_step(self, actions):
        result = original_step(self, actions)
        tracker = trackers.get(id(self))
        if tracker is None:
            return result

        import numpy as np

        _observations, rewards, terminals, truncations, _goals = result
        rewards = np.asarray(rewards)
        terminals = np.asarray(terminals, dtype=bool)
        truncations = np.asarray(truncations, dtype=bool)
        expected_shape = (num_seeds,)
        for label, values in (
            ("rewards", rewards),
            ("terminals", terminals),
            ("truncations", truncations),
        ):
            if values.shape != expected_shape:
                raise SystemExit(
                    f"official XQC training {label} has shape {values.shape}; "
                    f"expected {expected_shape}"
                )

        tracker["decision_step"] += 1
        for seed_index in range(num_seeds):
            reward = canonical_logger._finite_float(
                rewards[seed_index], "training reward"
            )
            tracker["returns"][seed_index] += reward
            tracker["lengths"][seed_index] += 1
            terminated = bool(terminals[seed_index])
            truncated = bool(truncations[seed_index])
            if terminated or truncated:
                canonical_logger.record_training_episode(
                    decision_step=tracker["decision_step"],
                    episode_return=tracker["returns"][seed_index],
                    episode_length=tracker["lengths"][seed_index],
                    terminated=terminated,
                    truncated=truncated,
                )
                tracker["returns"][seed_index] = 0.0
                tracker["lengths"][seed_index] = 0
        return result

    parallel_env_class.__init__ = tracked_init
    parallel_env_class.step = tracked_step
    try:
        yield
    finally:
        parallel_env_class.__init__ = original_init
        parallel_env_class.step = original_step


@contextmanager
def _patched_evaluation_logging(
    logging_module,
    capture=None,
    canonical_logger=None,
):
    """Patch only the official call window and always restore its logger."""

    original_logging = logging_module.log_multiple_seeds_to_wandb

    def captured_logging(step, infos, fps=30):
        if capture is not None:
            capture.record(step, infos)
        if canonical_logger is not None:
            canonical_logger.record_evaluation(step, infos)
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
    canonical_logger=None,
) -> None:
    """Run upstream directly unless the optional capture was requested."""

    if evaluation_capture is None and canonical_logger is None:
        official_main()
        return
    if logging_module is None:
        raise RuntimeError("logging capture requires the official logging module")
    with _patched_evaluation_logging(
        logging_module,
        evaluation_capture,
        canonical_logger,
    ):
        official_main()
    if evaluation_capture is not None:
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
    parser.add_argument("--canonical-wandb", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--implementation")
    parser.add_argument("--source-sha")
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
    canonical_metadata = _canonical_wandb_metadata(args)

    sys.path.insert(0, str(repo))
    sys.argv = [sys.argv[0], *hydra_args]

    from xqc.agents import XQCLearner
    from xqc.envs import ParallelEnv
    from xqc.normalization import RewardNormalizer
    from train_parallel import main as official_main

    evaluation_capture = None
    official_logging = None
    if capture_evaluations or canonical_metadata is not None:
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

    canonical_logger = None
    wandb_module = None
    if canonical_metadata is not None:
        import wandb as wandb_module

        canonical_logger = _CanonicalWandbLogger(
            wandb_module,
            action_repeat=args.action_repeat,
        )

    XQCLearner.update = counted_update
    RewardNormalizer.__init__ = captured_reward_normalizer_init
    try:
        if canonical_metadata is None:
            _run_official_main(
                official_main,
                logging_module=official_logging,
                evaluation_capture=evaluation_capture,
            )
        else:
            with _patched_wandb_initialization(
                wandb_module,
                canonical_metadata,
            ), _patched_training_returns(
                ParallelEnv,
                canonical_logger,
                num_seeds=args.num_seeds,
            ):
                _run_official_main(
                    official_main,
                    logging_module=official_logging,
                    evaluation_capture=evaluation_capture,
                    canonical_logger=canonical_logger,
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
