"""Oscar replay checkpoint benchmark used by the two-segment smoke job."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


class CanaryError(RuntimeError):
    pass


def verify_resumed_lineage(
    *,
    lineage_dir: str | os.PathLike[str],
    first_segment: str,
    second_segment: str,
    minimum_first_step: int,
) -> dict[str, Any]:
    """Prove that the real second segment advanced learned state on one W&B run."""

    if not first_segment or not second_segment or first_segment == second_segment:
        raise CanaryError("canary segment IDs must be distinct and non-empty")
    if type(minimum_first_step) is not int or minimum_first_step < 0:
        raise CanaryError("minimum first step must be a non-negative integer")
    try:
        from utils.resume_lineage import LineageStore

        with LineageStore.open(lineage_dir, mode="required") as store:
            latest = store.load()
            if latest.parent_generation is None:
                raise CanaryError("latest canary generation has no retained predecessor")
            predecessor = store.load(latest.parent_generation)
    except CanaryError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise CanaryError(f"could not validate the resumed canary lineage: {exc}") from exc

    first = predecessor.metadata
    second = latest.metadata
    for metadata, expected_segment, label in (
        (first, first_segment, "first"),
        (second, second_segment, "second"),
    ):
        if metadata.get("segment_id") != expected_segment:
            raise CanaryError(f"{label} generation has the wrong segment ID")
        for field in ("global_step", "num_updates"):
            value = metadata.get(field)
            if type(value) is not int or value < 0:
                raise CanaryError(f"{label} generation has invalid {field}")
        run_id = metadata.get("wandb_run_id")
        if not isinstance(run_id, str) or not run_id:
            raise CanaryError(f"{label} generation has no W&B run ID")

    if first["global_step"] <= minimum_first_step or first["num_updates"] <= 0:
        raise CanaryError("first canary segment did not reach learned-state training")
    if second["global_step"] <= first["global_step"]:
        raise CanaryError("required canary segment did not advance environment steps")
    if second["num_updates"] <= first["num_updates"]:
        raise CanaryError("required canary segment did not advance optimizer updates")
    if second["wandb_run_id"] != first["wandb_run_id"]:
        raise CanaryError("required canary segment created a different W&B run")

    return {
        "schema_version": 1,
        "first_generation": predecessor.generation_id,
        "second_generation": latest.generation_id,
        "first_step": first["global_step"],
        "second_step": second["global_step"],
        "first_updates": first["num_updates"],
        "second_updates": second["num_updates"],
        "wandb_run_id": second["wandb_run_id"],
        "verified": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, TypeError, ValueError):
        return None
    return value if sys.platform == "darwin" else value * 1024


def _tree_equal(left: object, right: object, torch_module) -> bool:
    if torch_module.is_tensor(left):
        return torch_module.is_tensor(right) and torch_module.equal(left, right)
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key], torch_module) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(
                _tree_equal(left_item, right_item, torch_module)
                for left_item, right_item in zip(left, right)
            )
        )
    return left == right


def _publish_json(path: Path, payload: Mapping[str, Any], durable_root: Path) -> None:
    if not path.is_absolute() or not durable_root.is_absolute():
        raise CanaryError("benchmark output and durable root must be absolute")
    try:
        root = durable_root.resolve(strict=True)
        parent = path.parent.resolve(strict=True)
        parent.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CanaryError("benchmark output must be below the durable root") from exc
    if path.exists():
        raise CanaryError(f"benchmark output already exists: {path}")
    temporary: Path | None = None
    try:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=parent
        )
        temporary = Path(raw_temporary)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_descriptor = os.open(
            parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise CanaryError(f"could not durably publish benchmark result: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def benchmark_full_replay(
    *,
    output_path: str | os.PathLike[str],
    durability_root: str | os.PathLike[str],
    capacity: int,
    observation_dim: int,
    action_dim: int,
    episode_rows: int,
    batch_size: int,
    train_unroll_horizon: int,
    shard_rows: int,
    maximum_estimated_bytes: int,
) -> dict[str, Any]:
    """Fill, shard, checksum, restore, and sample the real production Buffer."""

    integers = {
        "capacity": capacity,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "episode_rows": episode_rows,
        "batch_size": batch_size,
        "train_unroll_horizon": train_unroll_horizon,
        "shard_rows": shard_rows,
        "maximum_estimated_bytes": maximum_estimated_bytes,
    }
    if any(type(value) is not int or value <= 0 for value in integers.values()):
        raise CanaryError("all replay benchmark dimensions must be positive integers")
    if capacity < train_unroll_horizon + 1 or episode_rows < train_unroll_horizon + 1:
        raise CanaryError("capacity and episode rows must admit one training slice")

    # state observations, actions, reward, and termination are float32; the
    # physical episode index is int64.
    bytes_per_row = 4 * (observation_dim + action_dim + 2) + 8
    estimated_peak_bytes = (
        2 * bytes_per_row * capacity
        + 3 * bytes_per_row * min(capacity, shard_rows)
    )
    if estimated_peak_bytes > maximum_estimated_bytes:
        raise CanaryError(
            f"estimated peak {estimated_peak_bytes} exceeds limit {maximum_estimated_bytes}"
        )

    output = Path(output_path)
    root = Path(durability_root)
    if not output.is_absolute() or not output.parent.is_dir():
        raise CanaryError("benchmark output needs an absolute path in an existing directory")
    if output.exists():
        raise CanaryError(f"benchmark output already exists: {output}")
    checkpoint_dir: Path | None = None
    started = time.perf_counter()
    try:
        import torch
        from tensordict import TensorDict

        from RL.tdmpc2_core.common.buffer import Buffer

        def make_cfg():
            return SimpleNamespace(
                device="cpu",
                buffer_size=capacity,
                steps=capacity,
                batch_size=batch_size,
                train_unroll_horizon=train_unroll_horizon,
                multitask=False,
                obs="state",
                obs_shape={"state": [observation_dim]},
                obs_dtype="float32",
                action_dim=action_dim,
            )

        fill_started = time.perf_counter()
        source = Buffer(make_cfg(), resumable=True)
        episodes_needed = math.ceil(capacity / episode_rows)
        episodes_loaded = 0
        while episodes_loaded < episodes_needed:
            count = min(32, episodes_needed - episodes_loaded)
            values = torch.arange(
                count * episode_rows, dtype=torch.float32
            ).reshape(count, episode_rows)
            values.add_(episodes_loaded * episode_rows)
            obs = torch.zeros(count, episode_rows, observation_dim)
            action = torch.zeros(count, episode_rows, action_dim)
            obs[..., 0] = values
            action[..., 0] = -values
            source.load(
                TensorDict(
                    {
                        "obs": obs,
                        "action": action,
                        "reward": values.clone(),
                        "terminated": torch.zeros_like(values),
                    },
                    batch_size=[count, episode_rows],
                )
            )
            episodes_loaded += count
        if source.size != capacity:
            raise CanaryError("synthetic Buffer did not reach full physical capacity")
        fill_seconds = time.perf_counter() - fill_started

        metadata = source.training_state_metadata()
        if metadata.get("storage_rows") != capacity:
            raise CanaryError("production replay metadata does not describe full capacity")
        checkpoint_dir = Path(
            tempfile.mkdtemp(prefix=".replay-canary.", dir=output.parent)
        )
        records: list[tuple[Path, str, int]] = []

        def save(name: str, payload: object) -> None:
            path = checkpoint_dir / name
            with path.open("xb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            records.append((path, _sha256(path), path.stat().st_size))

        save_started = time.perf_counter()
        save("metadata.pt", metadata)
        shard_count = math.ceil(capacity / shard_rows)
        for index in range(shard_count):
            save(
                f"shard-{index:06d}.pt",
                source.training_state_shard(index, max_rows=shard_rows),
            )
        directory_descriptor = os.open(
            checkpoint_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        save_seconds = time.perf_counter() - save_started

        inventory = hashlib.sha256()
        for path, digest, size in records:
            inventory.update(f"{path.name}\0{size}\0{digest}\n".encode("ascii"))

        def checked_load(record: tuple[Path, str, int]):
            path, expected, _ = record
            if _sha256(path) != expected:
                raise CanaryError(f"checksum changed for {path.name}")
            return torch.load(path, map_location="cpu", weights_only=False)

        restore_started = time.perf_counter()
        restored = Buffer(make_cfg(), resumable=True)
        restored.load_training_state_shards(
            checked_load(records[0]),
            (checked_load(record) for record in records[1:]),
        )
        restore_seconds = time.perf_counter() - restore_started
        if restored.size != capacity:
            raise CanaryError("restored Buffer is not at full capacity")

        rng = torch.random.get_rng_state()
        sample_started = time.perf_counter()
        source_sample = source.sample()
        torch.random.set_rng_state(rng)
        restored_sample = restored.sample()
        if not _tree_equal(source_sample, restored_sample, torch):
            raise CanaryError("next replay sample changed after restore")
        sample_seconds = time.perf_counter() - sample_started

        metrics = {
            "schema_version": 1,
            "capacity_rows": capacity,
            "episode_rows": episode_rows,
            "shard_rows": shard_rows,
            "shard_count": shard_count,
            "bytes_per_row": bytes_per_row,
            "estimated_peak_bytes": estimated_peak_bytes,
            "checkpoint_bytes": sum(record[2] for record in records),
            "checkpoint_sha256": inventory.hexdigest(),
            "fill_seconds": float(fill_seconds),
            "save_seconds": float(save_seconds),
            "restore_seconds": float(restore_seconds),
            "sample_seconds": float(sample_seconds),
            "peak_rss_bytes": _rss_bytes(),
            "verified": True,
            "total_seconds": float(time.perf_counter() - started),
        }
        shutil.rmtree(checkpoint_dir)
        checkpoint_dir = None
        _publish_json(output, metrics, root)
        return metrics
    except CanaryError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise CanaryError(f"full-capacity replay benchmark failed: {exc}") from exc
    finally:
        if checkpoint_dir is not None:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _resolved_spec(run_path: Path, algorithm_path: Path) -> dict[str, int]:
    """Resolve the benchmark dimensions from the same experiment files as training."""

    try:
        experiment = json.loads(run_path.read_text(encoding="utf-8"))
        algorithm = json.loads(algorithm_path.read_text(encoding="utf-8"))
        resolved = dict(algorithm)
        resolved.update(experiment.get("overrides_alg", {}))
        params = resolved["alg_params"]
        if params.get("obs", "state") != "state":
            raise CanaryError("Oscar replay canary supports state observations only")
        import gymnasium as gym
        from gymnasium.spaces.utils import flatdim

        import domains  # noqa: F401
        from utils.core import build_env

        env = build_env(resolved, experiment, render_mode=None)
        try:
            if not isinstance(env.action_space, gym.spaces.Box):
                raise CanaryError("Oscar replay canary requires a Box action space")
            observation_dim = int(flatdim(env.observation_space))
            action_dim = int(math.prod(env.action_space.shape))
            episode_steps = getattr(getattr(env, "spec", None), "max_episode_steps", None)
            if type(episode_steps) is not int or episode_steps <= 0:
                raise CanaryError("environment has no fixed positive episode horizon")
        finally:
            env.close()
        capacity = min(int(params["buffer_size"]), int(resolved["total_steps"]))
        return {
            "capacity": capacity,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "episode_rows": episode_steps + 1,
            "batch_size": int(params["batch_size"]),
            "train_unroll_horizon": int(params["train_unroll_horizon"]),
        }
    except CanaryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CanaryError(f"could not resolve replay benchmark configuration: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-lineage")
    verify.add_argument("--lineage-dir", required=True)
    verify.add_argument("--first-segment", required=True)
    verify.add_argument("--second-segment", required=True)
    verify.add_argument("--minimum-first-step", type=int, required=True)
    benchmark = commands.add_parser("benchmark-replay")
    benchmark.add_argument("--run", required=True)
    benchmark.add_argument("--algorithm", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--durable-root", required=True)
    benchmark.add_argument("--shard-rows", type=int, default=100_000)
    benchmark.add_argument("--maximum-estimated-bytes", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-lineage":
            metrics = verify_resumed_lineage(
                lineage_dir=args.lineage_dir,
                first_segment=args.first_segment,
                second_segment=args.second_segment,
                minimum_first_step=args.minimum_first_step,
            )
        else:
            spec = _resolved_spec(Path(args.run), Path(args.algorithm))
            metrics = benchmark_full_replay(
                output_path=args.output,
                durability_root=args.durable_root,
                shard_rows=args.shard_rows,
                maximum_estimated_bytes=args.maximum_estimated_bytes,
                **spec,
            )
    except CanaryError as exc:
        print(f"Oscar resume canary failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metrics, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
