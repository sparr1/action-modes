"""Fail-closed coverage receipt for full-suite diagnostic runtime measurements.

This checks saved evidence, not just requested flags. It intentionally requires
complete requested real rollouts: an early termination can be scientifically
valid, but is not a measurement of the full requested runtime workload.
Only JSON is read; trusted donor pickle payloads are never deserialized here.
"""
from __future__ import annotations

from collections import Counter
from itertools import product
import json
import math
from pathlib import Path


BRANCHES = ("fresh", "actor", "critic", "joint")
LANES = ("common", "natural")
INTERVENTIONS = ("first_action_only", "memory_only", "full")


def _require(condition, label):
    if not condition:
        raise ValueError(f"Full diagnostic coverage failed: {label}")


def _read(path):
    _require(path.is_file(), f"missing {path}")
    with path.open() as stream:
        value = json.load(stream)
    _require(isinstance(value, dict), f"expected JSON object in {path}")
    return value


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numbers(row, keys, label):
    for key in keys:
        _require(_finite(row.get(key)), f"{label}: missing/nonfinite {key}")


def _identities(rows, keys, expected, label):
    _require(isinstance(rows, list), f"{label}: expected list")
    actual = [tuple(row.get(key) for key in keys) for row in rows]
    _require(Counter(actual) == Counter(expected), f"{label}: incomplete or duplicate identities")


def _value_check(check, options, label):
    _numbers(check, ("bias", "rmse", "centered_rmse", "relative_rmse", "top_action_regret"), label)
    _require(check.get("samples") == options["mc_rollouts"], f"{label}: MC sample count")
    _require(check.get("actions") == options["action_count"], f"{label}: action count")
    for key in ("reference_mean", "prediction_mean", "reference_se"):
        values = check.get(key, [])
        _require(len(values) == options["action_count"] and all(map(_finite, values)), f"{label}: {key}")
    # Constant action values legitimately have undefined correlations.
    for key in ("spearman", "pearson"):
        _require(key in check and (check[key] is None or _finite(check[key])), f"{label}: {key}")


def _snapshot(snapshot, options, label, counts):
    checks = snapshot.get("critic_checks")
    _identities(checks, ("continuation", "critic"), product(("prior", "carried", "current"),
                ("online", "target")), f"{label}: critic checks")
    for check in checks:
        _value_check(check, options, label)
    counts["critic_checks"] += len(checks)
    for field, support in (("directional", "prior_mean"), ("current_action_directional", "current_mean")):
        directions = snapshot.get(field)
        _identities(directions, ("support", "epsilon"), ((support, .03), (support, .1)), f"{label}: {field}")
        for check in directions:
            _numbers(check, ("predicted_derivative", "reference_slope", "reference_slope_se", "gain", "gain_se"), label)
            _require(check.get("samples") == options["mc_rollouts"], f"{label}: directional sample count")
        counts["directional_checks"] += len(directions)
    _numbers(snapshot, ("model_actor_objective", "critic_actor_objective", "current_mean_critic_value",
                       "current_mean_reference_return", "current_mean_reference_se"), label)
    heads = snapshot.get("q_heads", [])
    _require(len(heads) >= 2 and all(len(head) == options["action_count"] and all(map(_finite, head))
                                   for head in heads), f"{label}: per-head Q evidence")
    counts["snapshots_scored"] += 1


def _fork(row, rounds, options, label, counts):
    stages = row.get("stages")
    stages_expected = [("initial", 0), ("after_first_critic_block", 1), ("after_first_actor_block", 1)]
    stages_expected.extend(("post_round", index) for index in range(1, rounds + 1))
    _identities(stages, ("stage", "round"), stages_expected, f"{label}: learner stages")
    final = row.get("final", {})
    _require((final.get("stage"), final.get("round")) == ("pre_reset", rounds), f"{label}: final snapshot")
    for snapshot in [*stages, final]:
        _snapshot(snapshot, options, label, counts)
    events = row.get("trace_events")
    _require(isinstance(events, list) and events, f"{label}: no trace events")
    updates = [event for event in events if event.get("phase") == "update"]
    _require(updates and any(event.get("updated_actor") is True for event in updates)
             and any(event.get("updated_critic") is True for event in updates), f"{label}: optimizer update events")
    for event in updates:
        _require(event.get("metrics") and all(map(_finite, event["metrics"].values())), f"{label}: update metrics")
    probe_stages = [("initial", 0), ("before_first_actor_block", 1), ("after_first_actor_block", 1)]
    probe_stages.extend(("post_round", index) for index in range(1, rounds + 1))
    for phase, measurement, fields in (
        ("transfer_probe", "transfer_boundary_root_probe", ("transfer_policy_kl_vs_prior",
            "transfer_mean_action_delta_l2", "transfer_actor_std_mean",
            "transfer_root_q_inner_actor_mean_all", "transfer_root_q_target_actor_mean_all",
            "transfer_root_q_frozen_actor_mean_all")),
        ("probe", "post_update_togo_probe", ("togo_return_mean", "togo_reward_mean", "togo_bootstrap_mean",
            "togo_return_gain_vs_initial", "togo_return_gain_vs_outer", "probe_model_steps")),
    ):
        selected = [event for event in events if event.get("phase") == phase and event.get("measurement") == measurement]
        _identities(selected, ("stage", "round_index"), probe_stages, f"{label}: {measurement}")
        for event in selected:
            _numbers(event.get("metrics", {}), fields, f"{label}: {measurement}")
        counts[measurement] += len(selected)
    counts["trace_events"] += len(events)
    counts["optimizer_update_events"] += len(updates)


def verify_full_bundle(output_dir):
    """Return compact verified counts, or raise ValueError on absent evidence.

    Requires both data lanes, all four target crossings, all-round snapshots,
    every full-trace probe, stationary/portability tests, real calibration,
    all three replanning interventions, and saved donor/simulator snapshots.
    """
    output = Path(output_dir)
    manifest = _read(output / "manifest.json")
    _require(manifest.get("status") == "complete", "manifest is not complete")
    _require(manifest.get("outer_state_unchanged") is True, "frozen outer state not verified")
    options = manifest.get("options", {})
    for name in ("full_trace_probes", "target_cross"):
        _require(options.get(name) is True, f"{name} must be enabled")
    _require(Counter(options.get("data_lanes", [])) == Counter(LANES), "both data lanes required")
    _require(options.get("capture_rounds") is None, "all-round capture must be enabled")
    for name, minimum in (("mc_rollouts", 2), ("action_count", 4), ("fit_steps", 1), ("fit_states", 4),
                          ("real_rollouts", 1), ("real_tail_steps", 1), ("replan_steps", 1), ("replan_repeats", 1)):
        value = options.get(name)
        _require(type(value) is int and value >= minimum, f"invalid/missing {name}")
    dimensions = [manifest.get(key, []) for key in ("horizons", "rounds", "histories", "episode_seeds", "decisions")]
    _require(all(dimensions), "nonempty requested root grid required")
    h_values, j_values, histories, seeds, decisions = dimensions
    expected_episodes = list(product(h_values, j_values, histories, seeds))
    _identities(manifest.get("episodes"), ("horizon", "rounds", "history", "seed"),
                expected_episodes, "manifest episodes")
    for episode in manifest["episodes"]:
        _require(episode.get("unreached_decisions") == [], "manifest has unreached diagnostic decisions")
        _require(Counter(episode.get("captured_decisions", [])) == Counter(decisions), "manifest captured decision count")
    expected_roots = list(product(h_values, j_values, histories, seeds, decisions))
    _require(manifest.get("root_count") == len(expected_roots) > 0, "manifest root count")
    counts = Counter(roots=0, episodes=len(expected_episodes))
    for h, j, history, seed in expected_episodes:
        episode = _read(output / f"h{h}-j{j}/{history}/seed-{seed}/episode.json")
        _require(episode.get("unreached_decisions") == [], "unreached diagnostic decisions")
        _require(Counter(episode.get("captured_decisions", [])) == Counter(decisions), "captured decision count")
    for h, j, history, seed, decision in expected_roots:
        root_id = f"h{h}-j{j}/{history}/seed-{seed}/decision-{decision}"
        directory = output / root_id
        root = _read(directory / "diagnostics.json")
        _require(tuple(root.get(key) for key in ("horizon", "rounds", "history", "episode_seed", "decision", "root_id"))
                 == (h, j, history, seed, decision, root_id), f"{root_id}: root identity")
        diagnostics = root.get("diagnostics", {})
        _require(diagnostics.get("horizon") == h, f"{root_id}: diagnostic horizon")
        branches = diagnostics.get("branches")
        _identities(branches, ("data_lane", "branch"), product(LANES, BRANCHES), f"{root_id}: branches")
        for row in branches:
            _fork(row, j, options, f"{root_id}/{row['data_lane']}/{row['branch']}", counts)
            hashes = row.get("replay_sha256", [])
            _require(len(hashes) == j and all(isinstance(item, str) and item for item in hashes), f"{root_id}: replay hashes")
        common = [row["replay_sha256"] for row in branches if row["data_lane"] == "common"]
        _require(all(value == common[0] for value in common), f"{root_id}: common-data replay mismatch")
        targets = diagnostics.get("target_cross")
        _identities(targets, ("online", "target"), product(("prior", "carried"), repeat=2), f"{root_id}: target crossing")
        for row in targets:
            _fork(row, j, options, f"{root_id}/target_cross/{row['online']}/{row['target']}", counts)
        portability = diagnostics.get("portability", {})
        support = portability.get("support")
        _identities(support, ("support", "continuation"), product(("previous_root", "imagined_successor", "actual_successor"),
                    ("prior", "carried")), f"{root_id}: portability")
        for check in support:
            _value_check(check, options, f"{root_id}: portability")
        shift = portability.get("horizon_shift", {})
        _require(shift.get("applicable") is (h > 1), f"{root_id}: horizon-shift applicability")
        if h > 1:
            _require((shift.get("old_horizon"), shift.get("new_horizon")) == (h - 1, h), f"{root_id}: horizon shift")
            values = shift.get("actionwise_target_change", [])
            _require(len(values) == options["action_count"] and all(map(_finite, values)), f"{root_id}: horizon target change")
            for key in ("old_slice", "new_slice"):
                _value_check(shift.get(key, {}), options, f"{root_id}: {key}")
        curves = diagnostics.get("stationary", {}).get("curves", {})
        _require(set(curves) == {"critic_prior", "critic_carried", "actor_prior", "actor_carried"}, f"{root_id}: stationary curves")
        for name, curve in curves.items():
            _identities(curve, ("step",), ((step,) for step in range(options["fit_steps"] + 1)), f"{root_id}: {name}")
            for point in curve:
                _numbers(point, ("train_rmse", "heldout_rmse", "parameter_drift_l2"), f"{root_id}: {name}")
                if point["step"]:
                    _numbers(point, ("gradient_norm",), f"{root_id}: {name}")
                _numbers(point.get("features", {}), ("effective_rank", "inactive_fraction"), f"{root_id}: {name} features")
            counts["stationary_points"] += len(curve)
        real = diagnostics.get("real")
        _identities(real, ("branch", "replicate"), product(BRANCHES, range(options["real_rollouts"])), f"{root_id}: real calibration")
        for row in real:
            _require(row.get("prefix_complete") is True and row.get("mc_complete") is True,
                     f"{root_id}: incomplete real calibration")
            _require((row.get("prefix_decisions"), row.get("tail_decisions")) == (h, options["real_tail_steps"]),
                     f"{root_id}: real calibration ended before full timing workload")
            _numbers(row, ("real_bootstrapped_return", "real_mc_return", "predicted_model_return",
                           "model_prefix_error", "terminal_value_error", "total_prediction_error"), f"{root_id}: real calibration")
        replans = diagnostics.get("replanning")
        _identities(replans, ("branch", "intervention", "future_replicate"),
                    product(BRANCHES, INTERVENTIONS, range(options["replan_repeats"])), f"{root_id}: replanning")
        for row in replans:
            _require(row.get("steps") == row.get("steps_requested") == options["replan_steps"]
                     and row.get("requested_horizon_complete") is True
                     and len(row.get("rewards", [])) == len(row.get("actions", [])) == options["replan_steps"],
                     f"{root_id}: replanning ended before full timing workload")
            _numbers(row, ("real_return", "real_discounted_return"), f"{root_id}: replanning")
        _require((directory / "donor.pt").is_file() and (directory / "donor.pt").stat().st_size > 0,
                 f"{root_id}: donor snapshot missing/empty")
        simulator = _read(directory / "simulator.json")
        _require(simulator.get("schema_version") == 1 and simulator.get("sha256") and simulator.get("state"),
                 f"{root_id}: simulator snapshot missing/empty")
        counts.update(roots=1, initialization_forks=len(branches), target_cross_forks=len(targets),
                      portability_checks=len(support), real_rollouts=len(real), replanning_rollouts=len(replans),
                      replanning_decisions=sum(row["steps"] for row in replans), saved_snapshots=2)
    _require(len(list(output.glob("h*-j*/*/seed-*/decision-*/diagnostics.json"))) == counts["roots"],
             "unexpected extra root files")
    return {"verified": True, "schema_version": 1, "full_requested_workload": True, **dict(counts)}
