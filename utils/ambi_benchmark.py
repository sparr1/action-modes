"""Portable data and publication for frozen AMBI benchmarks.

This module owns no environment or optimizer loop. Recording stays on the CPU;
serialization happens only at completed episode/root boundaries; W&B is owned by a CPU publisher.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
import uuid
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


SCHEMA_VERSION = 1
SEED_SCHEME = "sha256-v1"
ROOT_DECISIONS = (0, 100, 200, 300, 400)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def solver_seed(base, *identity):
    return int(canonical_hash([SEED_SCHEME, int(base), *identity])[:8], 16)


def protocol_for(resolved, controller_seed, max_steps):
    config = resolved["algorithm_config"]
    return {
        "environment": copy.deepcopy(resolved["environment"]),
        "env_wrappers": copy.deepcopy(config.get("env_wrappers", [])),
        "env_wrapper": copy.deepcopy(config.get("env_wrapper")),
        "observation": config.get("alg_params", {}).get("obs", "state"),
        "action_rule": "tanh_mean",
        "max_steps": max_steps,
        "controller_seed": int(controller_seed),
        "seed_scheme": SEED_SCHEME,
    }


def episode_protocol(protocol):
    return {key: value for key, value in protocol.items() if key != "root_bank_id"}


def _nonnegative_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _legacy_update_labels(params):
    """Describe front-loaded total allocations without assuming uniform rounds."""
    totals = [(symbol, params[f"inner_{component}_updates_per_action"])
              for symbol, component in (("C", "critic"), ("A", "actor"), ("T", "temperature"))
              if params.get(f"inner_{component}_updates_per_action") is not None]
    rounds = params.get("inner_rounds")
    uniform = (_nonnegative_integer(rounds) and rounds > 0 and bool(totals)
               and all(_nonnegative_integer(value) and value % rounds == 0 for _, value in totals))
    tags = ["schedule:legacy-total-budget"]
    tags.extend(f"{symbol}-per-action:{value}" for symbol, value in totals)
    if uniform:
        counts = {symbol: value // rounds for symbol, value in totals}
        labels = [f"{symbol}{value}" for symbol, value in counts.items()]
        tags.extend(f"{symbol}:{value}" for symbol, value in counts.items())
        if set(counts) == {"C", "A", "T"} and counts["C"] > counts["A"] == counts["T"] > 0:
            return labels + ["(joint then critic)"], tags + ["update-order:joint-then-critic"]
        if set(counts) == {"C", "A", "T"} and counts["C"] == counts["A"] == counts["T"] > 0:
            return labels + ["(joint)"], tags + ["update-order:joint"]
    else:
        labels = [f"{symbol}/action{value}" for symbol, value in totals]
    # The legacy engine activates each component in the first allocated slots
    # of its round. Different totals can produce different overlap per round.
    return labels + ["(overlapping slots)"], tags + ["update-order:overlapping-slots"]


def benchmark_run_labels(checkpoint, protocol, config, kind, *, selector=None):
    """Describe a run from the same saved inputs used by the evaluator.

    ``config`` is the algorithm mapping stored as W&B ``inner_config``. The
    checkpoint step comes only from its sidecar metadata, never a filename or
    preset name. Calling this helper performs no I/O and does not mutate inputs.
    """
    if kind not in {"episodes", "bank", "both"}:
        raise ValueError("Benchmark kind must be episodes, bank, or both.")
    params = config.get("alg_params", {})
    environment = protocol.get("environment", {})
    task = environment.get("params", {}).get("task") or environment.get("id") or "task unknown"
    metadata = checkpoint.get("metadata") or {}
    step = metadata.get("checkpoint", {}).get("step")
    known_step = isinstance(step, int) and not isinstance(step, bool) and step >= 0
    digest = checkpoint.get("sha256")
    if known_step:
        step_label = f"{step // 1000}k" if step and step % 1000 == 0 else str(step)
        checkpoint_label = f"ckpt {step_label}"
    else:
        checkpoint_label = f"ckpt {digest[:12]}" if digest else "ckpt unknown"

    operator = params.get("inner_operator")
    controller = "prior" if operator == "none" else operator or "unknown"
    tags = ["frozen-inner-benchmark", kind, f"kind:{kind}", f"task:{task}", f"controller:{controller}"]
    if protocol.get("action_rule"):
        tags.append(f"action:{protocol['action_rule']}")
    if known_step:
        tags.append(f"checkpoint-step:{step}")
    if digest:
        tags.append(f"checkpoint-sha:{digest[:12]}")
    source = checkpoint.get("source_run")
    if isinstance(source, dict):
        source = source.get("id") or source.get("path") or source.get("url")
    if isinstance(source, str) and source.strip():
        source_path = urlsplit(source).path.rstrip("/")
        if source_path:
            tags.append(f"source-run:{source_path.rsplit('/', 1)[-1]}")
    if selector:
        tags.append(f"preset:{selector}")
    tags.append(f"config:{canonical_hash(config)[:12]}")

    parts = [str(task), checkpoint_label]
    if controller == "prior":
        parts.append("prior only")
        tags.append("bootstrap:none")
    else:
        schedule = []
        legacy = (not any(params.get(key) is not None for key in (
            "inner_steps_per_update", "inner_updates_per_round",
            "inner_critic_updates_per_round", "inner_actor_updates_per_round",
        )) and any(params.get(key) is not None for key in (
            "inner_model_step_budget", "inner_critic_updates_per_action",
            "inner_actor_updates_per_action", "inner_temperature_updates_per_action",
        )))
        rollouts = params.get("inner_rollouts_per_round")
        if legacy and rollouts is None:
            rounds, horizon, budget = (params.get(key) for key in (
                "inner_rounds", "inner_rollout_horizon", "inner_model_step_budget"))
            if (all(_nonnegative_integer(value) for value in (rounds, horizon, budget))
                    and rounds > 0 and horizon > 0 and budget % (rounds * horizon) == 0):
                rollouts = budget // (rounds * horizon)
        for symbol, key in (("J", "inner_rounds"), ("N", "inner_rollouts_per_round"),
                            ("H", "inner_rollout_horizon")):
            value = rollouts if symbol == "N" else params.get(key)
            if value is not None:
                schedule.append(f"{symbol}{value}")
                tags.append(f"{symbol}:{value}")
        if params.get("inner_steps_per_update") is not None:
            interval = params["inner_steps_per_update"]
            schedule.append(f"update/{interval} transitions")
            tags.extend(("schedule:transitions", f"steps-per-update:{interval}"))
            dimensions = [params.get(key) for key in (
                "inner_rounds", "inner_rollouts_per_round", "inner_rollout_horizon")]
            if (controller in {"sac", "td3"}
                    and params.get("inner_explorer_mode", "none") == "none"
                    and all(_nonnegative_integer(value) for value in dimensions)
                    and isinstance(interval, (int, float)) and not isinstance(interval, bool)
                    and math.isfinite(interval) and interval > 0):
                # These are planned totals: episodic model termination can
                # reduce generated transitions and therefore realized updates.
                total = int(Fraction(math.prod(dimensions)) / Fraction(str(interval)))
                temperature = total if controller == "sac" and params.get("inner_temperature_mode") == "auto" else 0
                schedule.append(f"nominal C{total}/A{total}/T{temperature} per action (joint)")
                tags.extend(("update-order:joint", f"C-nominal-per-action:{total}",
                             f"A-nominal-per-action:{total}", f"T-nominal-per-action:{temperature}"))
            if params.get("inner_actor_lr") is not None:
                actor_lr = params["inner_actor_lr"]
                schedule.append(f"actor LR {actor_lr:g}")
                tags.append(f"actor-lr:{actor_lr:g}")
        elif any(params.get(f"inner_{component}_updates_per_round") is not None
                 for component in ("critic", "actor")):
            tags.append("schedule:separate")
            for symbol, component in (("C", "critic"), ("A", "actor")):
                value = params.get(f"inner_{component}_updates_per_round")
                if value is not None:
                    schedule.append(f"{symbol}{value}")
                    tags.append(f"{symbol}:{value}")
        elif params.get("inner_updates_per_round") is not None:
            updates = params["inner_updates_per_round"]
            schedule.append(f"G{updates}")
            tags.extend(("schedule:joint", f"G:{updates}"))
        elif legacy:
            legacy_labels, legacy_tags = _legacy_update_labels(params)
            schedule.extend(legacy_labels)
            tags.extend(legacy_tags)
        controller_label = controller.upper() if controller != "unknown" else "controller unknown"
        parts.append(" ".join([controller_label, *schedule]))
        bootstrap = params.get("inner_bootstrap_source")
        parts.append(f"Q {bootstrap.replace('_', '-')}" if bootstrap else "Q unspecified")
        tags.append(f"bootstrap:{bootstrap or 'unknown'}")
        if params.get("inner_finite_horizon") is not None:
            tags.append(f"finite-horizon:{str(params['inner_finite_horizon']).lower()}")
        for key, tag in (("inner_batch_size", "batch"), ("inner_temperature_mode", "temperature"),
                         ("inner_sac_critic_target", "critic-target")):
            if params.get(key) is not None:
                tags.append(f"{tag}:{params[key]}")
    parts.append(kind)
    return {"name": " | ".join(parts), "tags": tags}


def atomic_json(path, value, *, overwrite=False):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
                 overwrite=overwrite)


def atomic_write(path, data, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Non-finite JSON value {value!r} in {path}")
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def make_bank(checkpoint_sha256, protocol, roots, *, complete):
    bank = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": episode_protocol(protocol),
        "roots": roots,
        "complete": bool(complete),
    }
    bank["id"] = canonical_hash(bank)
    return bank


def load_bank(path, checkpoint_sha256, protocol):
    bank = read_json(path)
    identity = bank.get("id")
    if bank.get("schema_version") != SCHEMA_VERSION or identity != canonical_hash(
        {key: value for key, value in bank.items() if key != "id"}
    ):
        raise ValueError("Unsupported or corrupted observation bank.")
    if not bank.get("complete") or not bank.get("roots"):
        raise ValueError("Observation bank must be complete and nonempty.")
    if bank.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Observation bank checkpoint does not match.")
    if bank.get("protocol") != episode_protocol(protocol):
        raise ValueError("Observation bank environment/action/seed protocol does not match.")
    seen = set()
    for root in bank["roots"]:
        key = root["root_id"]
        observation = np.asarray(root["observation"], dtype=root["dtype"])
        if key in seen or observation.ndim != 1 or not np.isfinite(observation).all():
            raise ValueError(f"Duplicate or invalid bank observation {key!r}.")
        if list(observation.shape) != root["shape"] or observation.dtype != np.float32:
            raise ValueError(f"Invalid state observation shape/dtype for {key!r}.")
        seen.add(key)
    return bank


def capture_root(observation, seed, decision_index, return_before):
    observation = np.asarray(observation)
    if observation.ndim != 1 or observation.dtype != np.float32 or not np.isfinite(observation).all():
        raise ValueError("Shared banks currently require finite float32 state observations.")
    return {
        "root_id": f"seed-{seed}-decision-{decision_index}",
        "episode_id": f"seed-{seed}", "seed": int(seed),
        "decision_index": int(decision_index), "return_before": float(return_before),
        "dtype": str(observation.dtype), "shape": list(observation.shape),
        "observation": observation.tolist(),
    }


def reference_returns(path, checkpoint_sha256, protocol):
    path = Path(path)
    manifest = read_json(path / "manifest.json" if path.is_dir() else path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise ValueError("Prior reference must be a completed benchmark bundle.")
    if manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ValueError("Prior reference checkpoint does not match.")
    if episode_protocol(manifest.get("protocol", {})) != episode_protocol(protocol):
        raise ValueError("Prior reference environment/action/seed protocol does not match.")
    runs = [run for run in manifest["runs"] if
            run.get("config", {}).get("alg_params", {}).get("inner_operator") == "none"
            and run.get("status") == "complete" and run.get("episodes")]
    if len(runs) != 1:
        raise ValueError("Prior reference must contain exactly one completed prior-only episode run.")
    episodes = runs[0]["episodes"]
    values = {episode["seed"]: episode["return"] for episode in episodes}
    if len(values) != len(episodes) or not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Prior reference has duplicate seeds or non-finite returns.")
    return values


def code_identity():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
    digest = hashlib.sha256()
    paths = {root / "evaluate_ambi_checkpoint.py", root / "report_ambi_benchmark.py"}
    for directory in ("RL", "utils", "domains", "configs/research"):
        paths.update(path for path in (root / directory).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".json", ".js", ".html"})
    for path in sorted(paths):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(path.read_bytes())
    from importlib.metadata import version
    result = {"source_sha256": digest.hexdigest(), "runtime": {
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": version("torch"), "gymnasium": version("gymnasium"),
    }}
    try:
        return {**result, "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                "diff_sha256": hashlib.sha256(git("diff", "HEAD", "--", ".").encode()).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {**result, "commit": None, "dirty": None, "diff_sha256": None}


def resolve_eval_run_map(selectors, *, run_dir=None, run_map=None, wandb=None):
    """Validate an explicitly prepared run assignment without contacting W&B."""
    if run_dir is not None and run_map is not None:
        raise ValueError("Use only one of --eval-run-dir and --eval-run-map.")
    if run_dir is not None:
        if len(selectors) != 1:
            raise ValueError("--eval-run-dir requires exactly one selected planner; use --eval-run-map.")
        run_map = {selectors[0]: run_dir}
    if isinstance(run_map, (str, Path)):
        run_map = read_json(run_map)
    if run_map is None:
        if wandb:
            raise ValueError("--wandb requires an explicit --eval-run-dir or --eval-run-map prepared by eval_series.py create/append.")
        return {}
    if not isinstance(run_map, dict) or any(selector not in run_map for selector in selectors):
        raise ValueError("The evaluation run map must assign every selected planner to an existing run directory.")
    from utils.eval_series import load_run
    assigned = {}
    for selector in selectors:
        path = Path(run_map[selector]).resolve()
        load_run(path)
        assigned[selector] = str(path)
    if len(set(assigned.values())) != len(assigned):
        raise ValueError("Distinct selected planners require distinct evaluation run directories.")
    return assigned


def preflight_eval_runs(run_map, checkpoint, resolved_presets, protocol, seeds, *,
                        result_path, inventory_path=None, source_run=None):
    """Reject an assignment whose actual resolved controller would change a curve."""
    if not run_map:
        return
    from utils.eval_series import validate_identity
    from utils.eval_series_data import identity_for_ambi_checkpoint
    code = code_identity()
    for resolved in resolved_presets:
        identity = identity_for_ambi_checkpoint(
            checkpoint, resolved, protocol, seeds, code, path=result_path,
            inventory_path=inventory_path, source_run=source_run,
        )
        validate_identity(run_map[resolved["selector"]], identity)


def write_eval_series_specs(directory, checkpoint, resolved_presets, protocol, seeds, *,
                            inventory_path=None, source_run=None):
    """Prepare reviewable New/Append identities without constructing a learner."""
    from utils.eval_series_data import identity_for_ambi_checkpoint, descriptive_label
    code = code_identity()
    if code.get("dirty") is not False:
        raise ValueError("Prepare evaluation series specifications from a clean checkout.")
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(f"Specification directory already exists: {directory}")
    prepared = {}
    for resolved in resolved_presets:
        selector = resolved["selector"]
        identity = identity_for_ambi_checkpoint(
            checkpoint, resolved, protocol, seeds, code, path=checkpoint["path"],
            inventory_path=inventory_path, source_run=source_run,
        )
        label = descriptive_label(identity, selector)
        prepared[selector] = {"identity": identity, "label": label, "selector": selector}
    directory.mkdir(parents=True, exist_ok=False)
    paths = {}
    for selector, spec in prepared.items():
        path = directory / (selector.replace("/", "__") + ".json")
        atomic_json(path, spec)
        paths[selector] = str(path)
    return {"mode": "evaluation_series_specifications", "specs": paths}


def stage_completed_bundle(path, run_map, *, source_run=None, inventory_path=None):
    """Queue completed local results; publication failures never change science status."""
    from utils.eval_series import stage_result
    path = Path(path)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = read_json(manifest_path)
    status = {}
    for run in manifest["runs"]:
        selector = run["selector"]
        if selector not in run_map or run["status"] != "complete":
            continue
        try:
            stage_result(run_map[selector], manifest_path, selector=selector,
                         format="ambi-bundle", source_run=source_run,
                         inventory_path=inventory_path)
            status[selector] = {"status": "queued", "run_dir": run_map[selector]}
        except Exception as error:
            status[selector] = {"status": "failed", "run_dir": run_map[selector],
                                "error": f"{type(error).__name__}: {error}"}
    atomic_json(manifest_path.parent / ".series-staging.json", status, overwrite=True)
    return status


class BenchmarkBundle:
    """One invocation's durable manifest and bounded, per-episode trace shards."""

    def __init__(self, path, *, checkpoint, protocol, wandb=None, reference=None, eval_run_map=None,
                 checkpoint_inventory=None):
        if wandb and not eval_run_map:
            raise ValueError("W&B publication requires explicitly prepared evaluation run directories.")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.perf_counter()
        self.eval_run_map = eval_run_map or {}
        self.checkpoint_inventory = checkpoint_inventory
        self.reference = reference or {}
        self.manifest = {
            "schema_version": SCHEMA_VERSION, "evaluation_id": uuid.uuid4().hex,
            "checkpoint": checkpoint, "code": code_identity(), "protocol": protocol,
            "metric_catalog": {}, "runs": [], "status": "running",
        }
        self.save()

    def save(self):
        self.manifest["elapsed_seconds"] = time.perf_counter() - self.started
        atomic_json(self.path / "manifest.json", self.manifest, overwrite=True)

    def start_run(self, resolved, kind):
        config = copy.deepcopy(resolved["algorithm_config"])
        labels = benchmark_run_labels(self.manifest["checkpoint"], self.manifest["protocol"],
                                      config, kind, selector=resolved["selector"])
        run = {"id": resolved["selector"].replace("/", "__"), "selector": resolved["selector"],
               "config": config, "config_hash": canonical_hash(config), "kind": kind,
               "wandb_name": labels["name"], "wandb_tags": labels["tags"],
               "episodes": [], "roots": [], "trace_files": [], "status": "running",
               "serialization_seconds": 0.0, "publication_seconds": 0.0}
        self.manifest["runs"].append(run)
        self.save()
        return run

    def write_trace(self, run, name, events):
        if not events:
            return
        from RL.tdmpc2_core.inner_trace import metric_catalog
        catalog = metric_catalog(key for event in events for key in event.get("metrics", {})
                                 if not key.startswith("decision/"))
        started = time.perf_counter()
        rows = []
        for event in events:
            row = {"run_id": run["id"], **event, "metrics": dict(event.get("metrics", {}))}
            nonfinite = dict(row.get("nonfinite", {}))
            for key, value in row["metrics"].items():
                if value is not None and not math.isfinite(value):
                    nonfinite[key] = repr(value)
                    row["metrics"][key] = None
                if key not in self.manifest["metric_catalog"]:
                    self.manifest["metric_catalog"][key] = catalog.get(key, {
                        "definition": key.removeprefix("decision/").replace("_", " "),
                        "unit": "scalar", "sampling_phase": row["phase"],
                        "preferred_axis": "decision_index" if row["phase"] == "decision" else "round_index",
                    })
            if nonfinite:
                row["nonfinite"] = nonfinite
                counts = run.setdefault("nonfinite_trace_metrics", {})
                for key in nonfinite:
                    counts[key] = counts.get(key, 0) + 1
            rows.append(json.dumps(row, separators=(",", ":"), allow_nan=False))
        relative = f"{run['id']}/{name}.jsonl.gz"
        atomic_write(self.path / relative, gzip.compress(("\n".join(rows) + "\n").encode(), mtime=0))
        run["trace_files"].append(relative)
        run["serialization_seconds"] += time.perf_counter() - started
        self.save()

    def episode(self, run, result, events):
        result = copy.deepcopy(result)
        result["episode_id"] = f"seed-{result['seed']}"
        result["capped"] = result["truncated_by_evaluator"]
        result["inner_metrics_mean"] = result["model_metrics"]
        if result["seed"] in self.reference:
            result["paired_return_delta"] = result["return"] - self.reference[result["seed"]]
        run["episodes"].append(result)
        self.write_trace(run, result["episode_id"], events)
        self.save()

    def finish_run(self, run, result=None, error=None):
        run["status"] = "failed" if error is not None else "complete"
        if error is not None:
            run["error"] = f"{type(error).__name__}: {error}"
        if result is not None:
            run["result"] = result
            deltas = [episode["paired_return_delta"] for episode in run["episodes"]
                      if "paired_return_delta" in episode]
            if deltas:
                result["paired_return_delta_vs_prior"] = {
                    "count": len(deltas), "mean": float(np.mean(deltas)),
                    "std": float(np.std(deltas)), "min": min(deltas), "max": max(deltas),
                }
        self.save()

    def finish(self, error=None):
        self.manifest["status"] = "failed" if error is not None else "complete"
        for run in self.manifest["runs"]:
            if run["status"] == "running":
                self.finish_run(run, error=error or RuntimeError("Evaluation did not finish."))
        self.save()
        if self.eval_run_map:
            stage_completed_bundle(self.path, self.eval_run_map, inventory_path=self.checkpoint_inventory)
