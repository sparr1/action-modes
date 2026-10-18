"""Reject incomplete or scientifically mismatched frozen-evaluation artifacts."""

from copy import deepcopy
import gzip
import json
import statistics

import pytest

from slurm.ambi_early_h3_j8_validation import CONFIG, METRICS, PROBE, validate_result
from utils.ambi_benchmark import solver_seed


def summary(values):
    return {"count": len(values), "sum": sum(values), "mean": statistics.mean(values),
            "std": statistics.pstdev(values), "min": min(values), "max": max(values)}


def artifact(tmp_path, *, seeds=(101, 102), steps=2, mode="mean"):
    """Build complete real-schema traces, including every optimizer event."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    run_id = f"early_h3_j8__{mode}"
    (bundle / run_id).mkdir()
    episode_rows, bundle_episodes, probe_rows, trace_files = [], [], [], []
    probe_decisions = list(range(0, steps, 100))
    all_rounds = []
    for seed in seeds:
        episode_id = f"seed-{seed}"
        events, episode_probes = [], []
        for decision in range(steps):
            counters = {"critic": 0, "actor": 0, "temperature": 0}
            event_index = 0

            def event(phase, round_index, metrics, **extra):
                nonlocal event_index
                row = {"run_id": run_id, "episode_id": episode_id, "decision_index": decision,
                       "event_index": event_index, "phase": phase, "round_index": round_index,
                       **{f"{key}_updates": value for key, value in counters.items()},
                       "metrics": metrics, **extra}
                events.append(row)
                event_index += 1
                return row

            def probe(round_index):
                metrics = {"togo_return_mean": 3. + round_index, "togo_reward_mean": 1.,
                           "togo_bootstrap_mean": 2. + round_index,
                           "probe_model_steps": 192 if round_index == 0 else 96,
                           "probe_seconds": .01}
                row = event("probe", round_index, metrics)
                saved = {key: row[key] for key in ("episode_id", "decision_index", "round_index", "critic_updates", "actor_updates", "metrics")}
                probe_rows.append(saved)
                episode_probes.append(saved)

            event("initial", 0, {"alpha": 1.})
            if decision in probe_decisions:
                probe(0)
            for round_index in range(1, 9):
                event("collection", round_index, {"collection_transitions": 384})
                for critic, actor, temperature in [(True, False, False)] * 16 + [(False, True, True)] * 4:
                    for key, flag in zip(counters, (critic, actor, temperature)):
                        counters[key] += int(flag)
                    event("update", round_index, {"loss": .25},
                          updated_critic=critic, updated_actor=actor, updated_temperature=temperature)
                if decision in probe_decisions:
                    probe(round_index)
            event("decision", 8, {**{f"decision/{key}": value for key, value in METRICS.items()},
                                  "decision/reward": .1, "decision/control_seconds": .2})
        relative = f"{run_id}/{episode_id}.jsonl.gz"
        with gzip.open(bundle / relative, "wt") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in events)
        trace_files.append(relative)
        round_summaries = []
        for round_index in range(9):
            selected = [row for row in episode_probes if row["round_index"] == round_index]
            round_summaries.append({"round_index": round_index, "actor_updates": 4 * round_index,
                                    "critic_updates": 16 * round_index,
                                    "metrics": {key: summary([row["metrics"][key] for row in selected])
                                                for key in selected[0]["metrics"]}})
        all_rounds.extend(episode_probes)
        episode = {"seed": seed, "solver_seed": solver_seed(55, "episode", seed),
                   "length": steps, "return": float(seed), "terminated": False, "truncated": True,
                   "truncated_by_evaluator": steps < 500, "control_seconds": .2 * steps,
                   "togo_probe_seconds": .09 * len(probe_decisions),
                   "togo_probe_model_steps": 960 * len(probe_decisions),
                   "togo_round_summaries": round_summaries, "model_metrics": dict(METRICS),
                   "nonfinite_model_metrics": {}}
        episode_rows.append(episode)
        bundle_episodes.append({**episode, "episode_id": episode_id})
    round_summaries = []
    for round_index in range(9):
        selected = [row for row in all_rounds if row["round_index"] == round_index]
        round_summaries.append({"round_index": round_index, "actor_updates": 4 * round_index,
                               "critic_updates": 16 * round_index,
                               "metrics": {key: summary([row["metrics"][key] for row in selected])
                                           for key in selected[0]["metrics"]}})
    result = {"selector": f"early_h3_j8/{mode}", "environment_seeds": list(seeds),
              "controller_seed": 55, "seed_scheme": "sha256-v1", "execution_action": mode,
              "outer_updates_before": 99999, "outer_updates_after": 99999, "outer_state_unchanged": True,
              "resolved_config": dict(CONFIG), "resolved_device": "cuda:0", "episodes": episode_rows,
              "nonfinite_model_metrics": {}, "nonfinite_trace_metrics": {},
              "togo_return_probe": dict(PROBE), "togo_round_summaries": round_summaries,
              "return": summary([row["return"] for row in episode_rows]), "episode_length": summary([steps] * len(seeds)),
              "model_metrics": {key: summary([value] * (steps * len(seeds))) for key, value in METRICS.items()}}
    payload = {"schema_version": 1, "frozen_outer_learning": True, "checkpoint_sha256": "abc",
               "deterministic_execution": mode == "mean", "results": [result]}
    manifest = {"schema_version": 1, "status": "complete", "checkpoint": {"sha256": "abc"},
                "elapsed_seconds": 1., "protocol": {"action_rule": "tanh_mean" if mode == "mean" else "tanh_sample",
                "controller_seed": 55, "seed_scheme": "sha256-v1", "max_steps": steps},
                "runs": [{"selector": result["selector"], "status": "complete", "id": run_id,
                "initialization_seconds": .1, "resolved_config": dict(CONFIG), "result": deepcopy(result),
                "togo_return_probe": dict(PROBE), "togo_probe_rows": probe_rows,
                "episodes": bundle_episodes, "trace_files": trace_files}]}
    result_path = tmp_path / "result.json"
    save(result_path, payload)
    save(bundle / "manifest.json", manifest)
    return result_path, bundle, payload, manifest


def save(path, value):
    path.write_text(json.dumps(value))


def alter_trace(bundle, manifest, change):
    path = bundle / manifest["runs"][0]["trace_files"][0]
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    change(rows)
    with gzip.open(path, "wt") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)


@pytest.mark.parametrize("mode", ["mean", "sample"])
def test_complete_short_smoke_validates_actual_work_and_execution_protocol(tmp_path, mode):
    path, bundle, _, _ = artifact(tmp_path, mode=mode)
    result = validate_result(path, bundle, (101, 102), 2)
    assert result["passed"] and result["execution_action"] == mode
    assert result["decisions"] == 4 and result["probe_boundaries"] == 18
    assert result["control_seconds_per_decision"] == pytest.approx(.2)
    assert result["mean_return"] == 101.5
    assert result["sample_std_return"] == pytest.approx(2 ** -.5)


def test_probe_cadence_includes_decision_100_and_skips_intervening_decisions(tmp_path):
    path, bundle, _, _ = artifact(tmp_path, seeds=(101,), steps=101)
    result = validate_result(path, bundle, (101,), 101)
    assert result["probe_boundaries"] == 18 and result["decisions"] == 101


@pytest.mark.parametrize("change,match", [
    (lambda p: p["results"][0].update(outer_state_unchanged=False), "outer state"),
    (lambda p: p["results"][0].update(outer_updates_after=100000), "update count"),
    (lambda p: p.update(deterministic_execution=False), "execution flag"),
    (lambda p: p["results"][0]["resolved_config"].update(inner_sac_critic_target="entropy_augmented"), "inner_sac_critic_target"),
    (lambda p: p["results"][0].update(environment_seeds=[101]), "seed coverage"),
])
def test_rejects_wrong_controller_frozen_state_and_episode_selection(tmp_path, change, match):
    path, bundle, payload, _ = artifact(tmp_path)
    change(payload)
    save(path, payload)
    with pytest.raises(ValueError, match=match):
        validate_result(path, bundle, (101, 102), 2)


def test_full_production_defaults_reject_a_smoke_result(tmp_path):
    path, bundle, _, _ = artifact(tmp_path)
    with pytest.raises(ValueError, match="seed coverage"):
        validate_result(path, bundle)


@pytest.mark.parametrize("mutate,match", [
    (lambda rows: rows.pop(next(i for i, row in enumerate(rows) if row["phase"] == "update")), "ordering"),
    (lambda rows: next(row for row in rows if row["phase"] == "update").update(updated_critic=False), "counter"),
    (lambda rows: next(row for row in rows if row["phase"] == "decision")["metrics"].update({"decision/inner_compile_fallback": 1}), "compile_fallback"),
    (lambda rows: next(row for row in rows if row["phase"] == "probe").update(decision_index=1), "ordering|probe|initial"),
    (lambda rows: next(row for row in rows if row["phase"] == "probe")["metrics"].update(probe_model_steps=96), "Probe trace"),
])
def test_rejects_missing_work_compile_fallback_and_invalid_probe_traces(tmp_path, mutate, match):
    path, bundle, _, manifest = artifact(tmp_path)
    alter_trace(bundle, manifest, mutate)
    with pytest.raises(ValueError, match=match):
        validate_result(path, bundle, (101, 102), 2)


def test_rejects_missing_probe_boundary_even_if_trace_remains_complete(tmp_path):
    path, bundle, _, manifest = artifact(tmp_path)
    manifest["runs"][0]["togo_probe_rows"].pop()
    save(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="probe coverage"):
        validate_result(path, bundle, (101, 102), 2)
