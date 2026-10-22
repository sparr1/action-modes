"""Validate complete early H3/J8 results against their streamed trace bundles."""

from collections import Counter
import gzip
import json
import math
from pathlib import Path
import statistics

from utils.ambi_benchmark import solver_seed

SEEDS = (101, 102, 103, 104, 105)
COUNTERS = {"critic": 128, "actor": 32, "temperature": 32}
METRICS = {**{f"inner_{key}_optimizer_steps": value for key, value in COUNTERS.items()},
           "inner_compile_fallback": 0, "inner_model_steps": 3072}
PROBE = {"version": 2, "rollouts": 32, "horizon": 3,
         "cadence": "initial_and_after_each_round", "decision_cadence": 100,
         "first_decision": 0, "tail_actor": "outer", "tail_critic": "outer_online",
         "tail_q_reduction": "mean_pair", "entropy_bonus": False,
         "policy_actions": "sampled", "noise": "fixed_per_root_separate_from_learner",
         "critic_pair": "fixed_per_root_separate_from_learner"}
CONFIG = {"inner_operator": "sac", "inner_rollout_horizon": 3, "inner_rounds": 8,
          "inner_rollouts_per_round": 128, "inner_batch_size": 256,
          "inner_replay_capacity": 3072, "inner_critic_updates_per_round": 16,
          "inner_actor_updates_per_round": 4, "inner_sac_critic_target": "reward_only",
          "inner_temperature_mode": "auto", "inner_finite_horizon": True,
          "inner_actor_source": "sac", "inner_horizon_actor_source": "sac",
          "inner_critic_source": "aux_return", "inner_horizon_critic_source": "aux_return",
          "aux_return_mode": "sac", "inner_critic_bernoulli_p": 0.,
          "inner_actor_writeback_coef": 0., "inner_critic_writeback_coef": 0.,
          "inner_outer_replay_fraction": 0., "inner_execution_action": "policy_sample",
          "compile": True, "compile_strict": True, "critic_ensemble_backend": "vmap"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON literal: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _finite(value, message, *, nonnegative=False):
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and (not nonnegative or value >= 0), message)
    return float(value)


def _same(actual, expected, label):
    for key, value in expected.items():
        _require(actual.get(key) == value, f"{label}: expected {key}={value!r}, got {actual.get(key)!r}")


def _summary(summary, values, label):
    expected = {"count": len(values), "sum": sum(values), "mean": statistics.mean(values),
                "std": statistics.pstdev(values), "min": min(values), "max": max(values)}
    for key, value in expected.items():
        number = _finite(summary.get(key), f"{label}/{key} must be finite")
        _require(math.isclose(number, value, rel_tol=1e-7, abs_tol=1e-6), f"{label}/{key} disagrees with episodes")


def _metric_constants(metrics, *, prefix="", aggregate_count=None, expected_temperature_steps=32):
    constants = {**METRICS, "inner_temperature_optimizer_steps": expected_temperature_steps}
    for key, expected in constants.items():
        value = metrics.get(prefix + key)
        if aggregate_count is None:
            _require(value == expected, f"Unexpected {prefix + key}: {value!r}, expected {expected}")
        else:
            _require(isinstance(value, dict), f"Missing metric aggregate {key}")
            _same(value, {"count": aggregate_count, "mean": expected, "min": expected, "max": expected}, key)
    _require(metrics.get(prefix + "inner_critic_bernoulli_applied", 0.) == 0., "Unexpected Bernoulli transfer")


def _probe_summaries(rows, count):
    expected = [(round_index, 4 * round_index, 16 * round_index) for round_index in range(9)]
    actual = [(row["round_index"], row["actor_updates"], row["critic_updates"]) for row in rows]
    _require(actual == expected, "Probe summaries do not cover initialization and all eight rounds")
    for row in rows:
        metrics = row["metrics"]
        for key in ("togo_return_mean", "togo_reward_mean", "togo_bootstrap_mean"):
            _require(metrics[key]["count"] == count, "Probe summary has incorrect decision coverage")
            for value in metrics[key].values():
                _finite(value, f"Non-finite probe summary {key}")


def validate_result(result_path, bundle_dir, expected_seeds=SEEDS, max_steps=500, *,
                    expected_temperature_steps=32, config_overrides=None):
    """Require one complete controller result; return compact performance metrics.

    Smoke callers may select fewer seeds and decisions. Every requested decision
    must still execute the full H3/J8 solve. Trace shards are read one line at a
    time so validation does not duplicate the full episode trace in memory.

    Entropy-disabled variants select zero temperature updates and explicitly
    override their resolved configuration (including temperature mode). Actor,
    critic, rollout, and probe work retain the same H3/J8 contract. Default
    validation remains the original 32-temperature-update protocol. Explicit
    temperature-repeat variants retain the same actor slots but count each
    actual Adam step, with a matching resolved repetition configuration.
    """
    seeds = tuple(expected_seeds)
    _require(bool(seeds) and len(seeds) == len(set(seeds))
             and all(type(seed) is int for seed in seeds), "Expected seeds must be distinct integers")
    _require(type(max_steps) is int and 1 <= max_steps <= 500, "max_steps must be in [1, 500]")
    _require(type(expected_temperature_steps) is int and expected_temperature_steps >= 0
             and expected_temperature_steps % 32 == 0,
             "expected_temperature_steps must be zero or an integer multiple of 32")
    _require(config_overrides is None or isinstance(config_overrides, dict),
             "config_overrides must be a mapping or None")
    config = {**CONFIG, **(config_overrides or {})}
    temperature_repeats = expected_temperature_steps // 32
    if temperature_repeats > 1:
        _require(config.get("inner_temperature_updates_per_actor") == temperature_repeats,
                 "Repeated temperature work requires its explicit resolved configuration")
    counters = {**COUNTERS, "temperature": expected_temperature_steps}
    temperature_steps_per_round = expected_temperature_steps // 8
    payload = _read(result_path)
    root = Path(bundle_dir).resolve(strict=True)
    manifest = _read(root / "manifest.json")
    _same(payload, {"schema_version": 1, "frozen_outer_learning": True}, "Result")
    _same(manifest, {"schema_version": 1, "status": "complete"}, "Bundle")
    _require(len(payload["results"]) == len(manifest["runs"]) == 1, "Expected exactly one controller per bundle")
    result, run = payload["results"][0], manifest["runs"][0]
    _same(run, {"selector": result["selector"], "status": "complete"}, "Bundle run")
    _require(manifest["checkpoint"]["sha256"] == payload["checkpoint_sha256"], "Result/bundle checkpoint mismatch")
    _require(result["environment_seeds"] == list(seeds), "Result seed coverage mismatch")
    _require(result["outer_state_unchanged"] is True, "Frozen outer state changed")
    _require(result["outer_updates_before"] == result["outer_updates_after"], "Outer update count changed")
    _require(not result["nonfinite_model_metrics"] and not result["nonfinite_trace_metrics"], "Result contains non-finite metrics")
    _require(not run.get("nonfinite_trace_metrics", {}), "Bundle contains non-finite trace metrics")
    _same(result["resolved_config"], config, "Resolved config")
    _same(run["resolved_config"], config, "Bundle resolved config")
    _require(str(result["resolved_device"]).startswith("cuda"), "GPU result did not use CUDA")
    _same(result, {"controller_seed": 55, "seed_scheme": "sha256-v1"}, "Controller RNG")
    mode = result["execution_action"]
    _require(mode in {"mean", "sample"}, "Unknown execution mode")
    _require(payload["deterministic_execution"] is (mode == "mean"), "Result execution flag disagrees with mode")
    _same(manifest["protocol"], {"action_rule": "tanh_mean" if mode == "mean" else "tanh_sample",
                                "max_steps": max_steps, "controller_seed": 55, "seed_scheme": "sha256-v1"}, "Protocol")
    _same(result["togo_return_probe"], PROBE, "Result probes")
    _same(run["togo_return_probe"], PROBE, "Bundle probes")
    for key in ("episodes", "outer_updates_before", "outer_updates_after", "outer_state_unchanged", "execution_action"):
        _require(run["result"][key] == result[key], f"Saved bundle/result mismatch: {key}")

    episodes = result["episodes"]
    _require([episode["seed"] for episode in episodes] == list(seeds), "Episode seed coverage mismatch")
    _require([episode["seed"] for episode in run["episodes"]] == list(seeds), "Bundle episode coverage mismatch")
    probe_decisions = tuple(range(0, max_steps, 100))
    decisions = max_steps * len(seeds)
    control_seconds = probe_seconds = 0.
    for episode, saved in zip(episodes, run["episodes"]):
        _same(episode, {"length": max_steps, "terminated": False, "truncated": True,
                        "solver_seed": solver_seed(55, "episode", episode["seed"])}, "Episode")
        if max_steps == 500:
            _require(not episode["truncated_by_evaluator"], "Full episode was cut short by evaluator")
        _finite(episode["return"], "Non-finite episode return")
        control_seconds += _finite(episode["control_seconds"], "Invalid control time", nonnegative=True)
        probe_seconds += _finite(episode["togo_probe_seconds"], "Invalid probe time", nonnegative=True)
        _require(not episode["nonfinite_model_metrics"], "Episode contains non-finite metrics")
        _metric_constants(episode["model_metrics"], expected_temperature_steps=expected_temperature_steps)
        _require(episode["togo_probe_model_steps"] == len(probe_decisions) * 960, "Incorrect probe model-step count")
        _probe_summaries(episode["togo_round_summaries"], len(probe_decisions))
        for key in ("seed", "length", "return", "control_seconds", "solver_seed"):
            _require(saved[key] == episode[key], f"Saved episode differs: {key}")
    _metric_constants(result["model_metrics"], aggregate_count=decisions,
                      expected_temperature_steps=expected_temperature_steps)
    _summary(result["return"], [episode["return"] for episode in episodes], "Return summary")
    _summary(result["episode_length"], [episode["length"] for episode in episodes], "Length summary")
    _probe_summaries(result["togo_round_summaries"], len(probe_decisions) * len(seeds))

    expected_probe_keys = {(f"seed-{seed}", decision, round_index)
                           for seed in seeds for decision in probe_decisions for round_index in range(9)}
    recorded_probes = {}
    for row in run["togo_probe_rows"]:
        key = (row["episode_id"], row["decision_index"], row["round_index"])
        _require(key in expected_probe_keys and key not in recorded_probes, "Missing/duplicate/unexpected recorded probe boundary")
        _same(row, {"actor_updates": 4 * row["round_index"], "critic_updates": 16 * row["round_index"]}, "Recorded probe counters")
        recorded_probes[key] = row["metrics"]
    _require(set(recorded_probes) == expected_probe_keys, "Recorded probe coverage mismatch")

    expected_decisions = {(f"seed-{seed}", decision) for seed in seeds for decision in range(max_steps)}
    states, seen_probes, trace_rows, trace_bytes = {}, set(), 0, 0
    files = run["trace_files"]
    _require(bool(files) and len(files) == len(set(files)), "Missing/duplicate trace files")
    for relative in files:
        path = (root / relative).resolve(strict=True)
        _require(path.is_relative_to(root) and path.name.endswith(".jsonl.gz"), "Invalid trace path")
        trace_bytes += path.stat().st_size
        with gzip.open(path, "rt") as stream:
            for line in stream:
                event = json.loads(line)
                trace_rows += 1
                key = (event["episode_id"], event["decision_index"])
                _require(key in expected_decisions and event["run_id"] == run["id"], "Unexpected trace decision/run")
                state = states.setdefault(key, {"last": -1, "counts": Counter(), "complete": False})
                _require(not state["complete"] and event["event_index"] == state["last"] + 1, "Trace ordering/duplication failure")
                _require(state["last"] != -1 or event["phase"] == "initial", "Missing initial trace event")
                state["last"] = event["event_index"]
                _require(not event.get("nonfinite", {}), "Non-finite trace event")
                for name, value in event["metrics"].items():
                    _finite(value, f"Non-finite trace metric {name}")
                if event["phase"] == "update":
                    for name in counters:
                        flag = event[f"updated_{name}"]
                        _require(type(flag) is bool, "Invalid update flag")
                        increment = int(flag)
                        if name == "temperature":
                            _require(not flag or temperature_repeats > 0,
                                     "Unexpected temperature update flag")
                            increment *= temperature_repeats
                            actual = event["metrics"].get("temperature_updates", increment)
                            _require(actual == increment, "Trace temperature repeat count mismatch")
                        state["counts"][name] += increment
                for name in counters:
                    _require(event[f"{name}_updates"] == state["counts"][name], "Trace cumulative optimizer counter mismatch")
                if event["phase"] == "probe":
                    probe_key = (*key, event["round_index"])
                    _require(probe_key in expected_probe_keys and probe_key not in seen_probes, "Unexpected/duplicate probe trace")
                    seen_probes.add(probe_key)
                    _require(event["metrics"] == recorded_probes[probe_key], "Probe trace/summary row mismatch")
                    _same(event, {"critic_updates": 16 * event["round_index"], "actor_updates": 4 * event["round_index"],
                                  "temperature_updates": temperature_steps_per_round * event["round_index"]}, "Probe counters")
                    _same(event["metrics"], {"probe_model_steps": 192 if event["round_index"] == 0 else 96}, "Probe workload")
                elif event["phase"] == "decision":
                    _require(state["counts"] == counters, "Trace optimizer update count mismatch")
                    _metric_constants(event["metrics"], prefix="decision/",
                                      expected_temperature_steps=expected_temperature_steps)
                    _require(event["round_index"] == 8, "Incomplete inner rounds")
                    state["complete"] = True
    _require(set(states) == expected_decisions and all(state["complete"] for state in states.values()), "Incomplete decision trace coverage")
    _require(seen_probes == expected_probe_keys, "Probe trace coverage mismatch")
    returns = [episode["return"] for episode in episodes]
    return {"passed": True, "selector": result["selector"], "execution_action": mode,
            "seeds": list(seeds), "episodes": len(seeds), "decisions": decisions,
            "mean_return": statistics.mean(returns),
            "sample_std_return": statistics.stdev(returns) if len(returns) > 1 else None,
            "control_seconds": control_seconds, "control_seconds_per_decision": control_seconds / decisions,
            "probe_seconds": probe_seconds, "probe_boundaries": len(seen_probes),
            "trace_rows": trace_rows, "trace_compressed_bytes": trace_bytes,
            "bundle_elapsed_seconds": _finite(manifest["elapsed_seconds"], "Invalid bundle elapsed time", nonnegative=True),
            "initialization_seconds": _finite(run["initialization_seconds"], "Invalid initialization time", nonnegative=True)}
