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
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


SCHEMA_VERSION = 1
SEED_SCHEME = "sha256-v1"
ROOT_DECISIONS = (0, 100, 200, 300, 400)
MPPI_ACTION_RULE = "weighted_elite_gumbel_no_execution_noise"


def _is_xqc(config):
    return (config.get("alg") == "AMBIXQC/AMBIXQC"
            or config.get("alg_params", {}).get("inner_operator") == "xqc")


def controller_type(config):
    """Identify the evaluation controller before inspecting collection settings."""
    explicit = config.get("evaluation_controller")
    if explicit is not None:
        if not isinstance(explicit, dict) or explicit.get("type") not in {"prior", "xqc", "mppi"}:
            raise ValueError("Invalid explicit evaluation controller.")
        return explicit["type"]
    operator = config.get("alg_params", {}).get("inner_operator")
    return "prior" if operator == "none" else operator or "unknown"


def run_controller_type(run):
    config = run.get("config", {})
    result = controller_type(config)
    explicit = run.get("evaluation_controller")
    if explicit is not None:
        recorded = controller_type({"evaluation_controller": explicit})
        if recorded != result:
            raise ValueError("Recorded evaluation controller conflicts with the run configuration.")
    return result


def validate_evaluation_controller(config, controller):
    """Check authored MPPI settings against the actual adapter metadata."""
    if controller_type({"evaluation_controller": controller}) != controller_type(config):
        raise ValueError("Recorded evaluation controller conflicts with the run configuration.")
    if controller["type"] != "mppi":
        return
    settings, protocol = controller.get("settings"), controller.get("protocol")
    if not isinstance(settings, dict) or not isinstance(protocol, dict):
        raise ValueError("MPPI requires resolved settings and controller protocol.")
    for key, value in config.get("evaluation_controller", {}).get("params", {}).items():
        if settings.get(key) != value:
            raise ValueError(f"Recorded MPPI setting {key!r} conflicts with the authored configuration.")
    for key in ("horizon", "iterations", "effective_iterations", "num_samples", "num_elites"):
        if isinstance(settings.get(key), bool) or not isinstance(settings.get(key), int) or settings[key] <= 0:
            raise ValueError(f"Invalid resolved MPPI setting {key!r}.")
    if settings["effective_iterations"] < settings["iterations"]:
        raise ValueError("Effective MPPI iterations cannot be below configured iterations.")
    if (not _nonnegative_integer(settings.get("num_pi_trajs"))
            or settings["num_pi_trajs"] > settings["num_samples"]
            or settings["num_elites"] > settings["num_samples"]):
        raise ValueError("MPPI policy trajectories and elites must fit the candidate count.")
    for key in ("min_std", "max_std", "temperature"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid resolved MPPI setting {key!r}.")
    if settings["min_std"] > settings["max_std"]:
        raise ValueError("MPPI minimum standard deviation exceeds its maximum.")
    if protocol.get("action_rule") != MPPI_ACTION_RULE:
        raise ValueError("MPPI controller action rule must describe weighted-elite execution.")
    if protocol.get("terminal_value_source") != "online_xqc_twin_mean":
        raise ValueError("MPPI terminal value source must use the online XQC twin mean.")
    if protocol.get("terminal_value_units") != "normalized_xqc_soft_q_times_frozen_real_reward_scale":
        raise ValueError("MPPI terminal value units must describe the frozen reward-scale conversion.")
    scale = protocol.get("reward_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("MPPI controller requires a finite positive frozen reward scale.")


def decision_metric_catalog(names, *, xqc=False):
    """Describe completed solves without implying per-update observations.

    XQC rewards in imagined replay remain raw; its critic predictions and actor
    objective use normalized reward units. Slot-averaged diagnostics are not
    measurements of the final policy or accepted optimizer steps.
    """
    result = {}
    for name in names:
        key = name.removeprefix("decision/")
        inner = key.removeprefix("inner_")
        definition = f"Completed decision: {key.replace('_', ' ')}."
        unit = "scalar"
        if key in {"reward", "return", "cumulative_return", "return_so_far"}:
            unit = "raw_environment_reward"
            definition = ("Raw reward from this real environment decision." if key == "reward"
                          else "Cumulative raw environment return through this decision.")
        elif key.endswith("seconds"):
            unit = "seconds"
        elif key.startswith(("planner_value_", "planner_elite_value_")):
            unit = "raw_return_score"
            definition = ("MPPI final-iteration score statistic: raw predicted rewards plus discounted "
                          "online mean XQC soft-Q tail times the frozen real reward scale; "
                          f"no entropy correction. Statistic: {key}." if xqc else
                          f"MPPI final-iteration predicted score statistic: {key}.")
        elif key.startswith("planner_std_") or key == "planner_action_l2":
            unit = "normalized_action"
        elif key.startswith("planner_"):
            unit = "count"
            definition = f"Actual MPPI search measurement for this decision: {key}."
        elif inner.startswith(("behavior_reward_", "behavior_discounted_reward_", "return_")):
            unit = "raw_predicted_reward"
            definition = f"Imagined collection statistic in raw model reward units: {inner}."
        elif inner.startswith("reward_scale"):
            unit = "raw_reward_scale"
            definition = f"XQC reward-normalization scale statistic: {inner}."
        elif inner.startswith("outer_terminal_") and inner.endswith(("rows", "evaluations")):
            unit = "count"
            definition = f"Measured frozen outer terminal-bootstrap work for this decision: {inner}."
        elif inner == "terminal_bootstrap_outer":
            unit = "indicator"
            definition = "One when the final imagined transition uses frozen outer actor/online critic bootstrap with the inner temperature."
        elif inner.endswith(("fraction", "ratio", "rate")) and "learning_rate" not in inner:
            unit = "fraction"
        elif "kl" in inner or inner in {"policy_entropy", "policy_log_prob"}:
            unit = "nats"
        elif (inner.startswith("q") and "evaluation" not in inner and xqc):
            unit = "normalized_value"
            definition = f"Mean across XQC optimizer slots in normalized reward units: {inner}."
        elif inner in {"actor_loss", "temperature_loss"} and xqc:
            unit = "normalized_objective"
            definition = (f"Mean XQC {inner.replace('_', ' ')} across all slots, including slots "
                          "where policy delay skips the optimizer step.")
        elif inner == "critic_loss" and xqc:
            unit = "cross_entropy"
            definition = "Mean XQC categorical critic cross-entropy across optimizer slots."
        elif inner.startswith("alpha") or inner == "temperature":
            unit = "normalized_temperature" if xqc else "temperature"
        elif inner.endswith(("steps", "updates", "slots", "rollouts", "count", "draws", "size", "capacity")):
            unit = "count"
        if inner in {"critic_optimizer_steps", "actor_optimizer_steps", "temperature_optimizer_steps"}:
            definition = f"Actual completed {inner.replace('_', ' ')} in this action-local solve."
        result[name] = {"definition": definition, "unit": unit,
                        "sampling_phase": "completed_decision", "preferred_axis": "decision_index"}
    return result


def _legacy_metric_semantic(name, phase):
    """Minimal fallback for imported traces; their saved catalogs remain authoritative."""
    axis = ("actor_updates" if name.startswith("actor_") else
            "temperature_updates" if name.startswith("temperature_") else "critic_updates")
    return {"definition": f"Raw pre-update sampled-minibatch metric: {name}.",
            "unit": "objective" if "loss" in name else "scalar",
            "sampling_phase": phase, "preferred_axis": axis}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def solver_seed(base, *identity):
    return int(canonical_hash([SEED_SCHEME, int(base), *identity])[:8], 16)


def protocol_for(resolved, controller_seed, max_steps):
    """Common pairing contract; action_rule is the prior-reference rule.

    Candidate execution rules live on each run's evaluation_controller. Keeping
    the reference contract unchanged permits reuse of existing prior bundles.
    """
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


def benchmark_run_labels(checkpoint, protocol, config, kind, *, selector=None,
                         evaluation_controller=None):
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

    controller = controller_type(config)
    explicit_mppi = config.get("evaluation_controller", {}).get("type") == "mppi"
    tags = ["frozen-inner-benchmark", kind, f"kind:{kind}", f"task:{task}", f"controller:{controller}"]
    action_rule = MPPI_ACTION_RULE if explicit_mppi else protocol.get("action_rule")
    if action_rule:
        tags.append(f"action:{action_rule}")
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
        if not _is_xqc(config):
            tags.append("bootstrap:none")
    elif explicit_mppi:
        settings = ((evaluation_controller or {}).get("settings")
                    or config["evaluation_controller"].get("params", {}))
        schedule = []
        for symbol, key in (("H", "horizon"), ("N", "num_samples"),
                            ("E", "num_elites"), ("pi", "num_pi_trajs")):
            if key in settings:
                schedule.append(f"{symbol}{settings[key]}")
                tags.append(f"{symbol}:{settings[key]}")
        if "effective_iterations" in settings:
            schedule.append(f"J{settings['effective_iterations']}")
            tags.append(f"J:{settings['effective_iterations']}")
        elif "iterations" in settings:
            schedule.append(f"configured iterations {settings['iterations']}")
        tags.extend(("algorithm:ambixqc", "schedule:mppi-search", "C:0", "A:0", "T:0",
                     "terminal-q:online-xqc-twin-mean", "reward-scale:frozen-real"))
        if "iterations" in settings:
            tags.append(f"configured-iterations:{settings['iterations']}")
        parts.extend((" ".join(["MPPI", *schedule]), "weighted elite", "online Q × frozen scale"))
    elif _is_xqc(config):
        schedule = []
        for symbol, key in (("J", "inner_rounds"), ("N", "inner_rollouts_per_round"),
                            ("H", "inner_rollout_horizon"), ("G", "inner_updates_per_round")):
            if params.get(key) is not None:
                schedule.append(f"{symbol}{params[key]}")
                tags.append(f"{symbol}:{params[key]}")
        delay = params.get("xqc_policy_delay", 3)
        schedule.append(f"policy delay {delay}")
        tags.extend(("algorithm:ambixqc", "schedule:xqc-slots", f"policy-delay:{delay}"))
        for key, tag in (("inner_batch_size", "batch"),
                         ("inner_reward_normalization", "reward-normalization")):
            if params.get(key) is not None:
                tags.append(f"{tag}:{params[key]}")
        parts.append(" ".join(["XQC", *schedule]))
        if params.get("inner_terminal_bootstrap", "inner") == "outer":
            parts.append("outer terminal bootstrap")
            tags.extend(("terminal-bootstrap:outer", "terminal-policy:frozen-outer",
                         "terminal-q:online-outer", "terminal-alpha:inner"))
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
            run_controller_type(run) == "prior"
            and run.get("status") == "complete" and run.get("episodes")]
    if len(runs) != 1:
        raise ValueError("Prior reference must contain exactly one completed prior-only episode run.")
    if runs[0].get("action_rule", protocol.get("action_rule")) != "tanh_mean":
        raise ValueError("Prior reference must execute the deterministic actor mean.")
    episodes = runs[0]["episodes"]
    values = {episode["seed"]: episode["return"] for episode in episodes}
    if len(values) != len(episodes) or not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Prior reference has duplicate seeds or non-finite returns.")
    return values


def code_identity():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        executable = shutil.which("git")
        if executable is None:
            raise FileNotFoundError("git executable is unavailable")
        # Keep Python 3.10 on its posix_spawn path. Forking after PyTorch has
        # initialized OpenMP can abort the child on macOS. Python-created FDs
        # are CLOEXEC by default; no credentials or input are passed to git.
        return subprocess.check_output(
            [str(Path(executable).resolve()), "-C", str(root), *args],
            stderr=subprocess.DEVNULL, close_fds=False,
        ).decode().strip()
    digest = hashlib.sha256()
    paths = {root / name for name in ("evaluate_ambi_checkpoint.py", "report_ambi_benchmark.py",
                                     "run_ambixqc_mppi_evaluation.py", "summarize_ambixqc_mppi_eval.py",
                                     "run_ambixqc_inner_evaluation.py", "summarize_ambixqc_inner_eval.py")}
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
            "protocol_semantics": {"action_rule": "prior_reference", "candidate_action_rule": "per_run"},
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
        run["action_rule"] = (MPPI_ACTION_RULE if config.get("evaluation_controller", {}).get("type") == "mppi"
                              else self.manifest["protocol"]["action_rule"])
        if _is_xqc(config):
            run["diagnostic_capabilities"] = {
                "decision_metrics": True, "optimizer_traces": False,
                "shared_observation_probes": False,
            }
        self.manifest["runs"].append(run)
        self.save()
        return run

    def set_evaluation_controller(self, run, controller):
        """Publish resolved search settings before any scored episode is saved."""
        validate_evaluation_controller(run["config"], controller)
        run["evaluation_controller"] = copy.deepcopy(controller)
        run["action_rule"] = controller.get("protocol", {}).get("action_rule", run["action_rule"])
        run["controller_hash"] = canonical_hash(controller)
        labels = benchmark_run_labels(
            self.manifest["checkpoint"], self.manifest["protocol"], run["config"],
            run["kind"], selector=run["selector"], evaluation_controller=controller,
        )
        run["wandb_name"], run["wandb_tags"] = labels["name"], labels["tags"]
        self.save()

    def write_trace(self, run, name, events):
        if not events:
            return
        catalog = decision_metric_catalog(
            (key for event in events if event.get("phase") == "decision"
             for key in event.get("metrics", {})), xqc=_is_xqc(run["config"]),
        )
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
                    self.manifest["metric_catalog"][key] = catalog.get(
                        key, _legacy_metric_semantic(key, row["phase"]),
                    )
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
        result.setdefault("status", "complete")
        result["capped"] = result["truncated_by_evaluator"]
        result["inner_metrics_mean"] = result["model_metrics"]
        decisions = [event for event in events if event.get("phase") == "decision"]
        if decisions:
            result["actual_optimizer_steps"] = {
                component: sum(int(event.get(f"{component}_updates", 0)) for event in decisions)
                for component in ("critic", "actor", "temperature")
            }
        if result["seed"] in self.reference:
            result["paired_return_delta"] = result["return"] - self.reference[result["seed"]]
        run["episodes"].append(result)
        self.write_trace(run, result["episode_id"], events)
        self.save()

    def finish_run(self, run, result=None, error=None):
        run["status"] = "failed" if error is not None else "complete"
        if error is not None:
            run["error"] = f"{type(error).__name__}: {error}"
        counted = [episode["actual_optimizer_steps"] for episode in run["episodes"]
                   if "actual_optimizer_steps" in episode
                   and episode.get("status", "complete") == "complete"]
        if counted:
            run["actual_optimizer_steps_scope"] = "completed_episodes"
            run["actual_optimizer_steps"] = {
                component: sum(counts[component] for counts in counted)
                for component in ("critic", "actor", "temperature")
            }
            if _is_xqc(run["config"]):
                counts = run["actual_optimizer_steps"]
                run["display_name"] = (run["wandb_name"] + " | completed episodes: " + " ".join(
                    f"total{symbol}{counts[component]}" for symbol, component in
                    (("C", "critic"), ("A", "actor"), ("T", "temperature"))
                ))
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
