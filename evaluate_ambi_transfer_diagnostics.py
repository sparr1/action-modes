"""Saved-state diagnostics for actor, critic and joint inner-SAC transfer.

Produces a local diagnostic bundle, never W&B curves or training updates.
Run --help for selectable horizons, update budgets and source histories.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np
import torch

from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _make_env,
    _outer_state_digest, _validate_checkpoint_contract,
)
from utils.ambi_benchmark import solver_seed
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.checkpoint_context import load_checkpoint_context
from utils.transfer_diagnostic_metrics import aggregate_paired_root_metrics
from utils.transfer_diagnostics import (
    BRANCHES, PROTOCOL, audit_root, positive_ints, solve_fork, validate_controller, write_json,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--matrix", type=Path, default=Path("configs/research/ambi_critic_transfer_575k.json"))
    p.add_argument("--preset", default="return_return/fresh")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 5, 7])
    p.add_argument("--rounds", type=int, nargs="+", default=[1, 8])
    p.add_argument("--histories", nargs="+", choices=tuple(BRANCHES), default=list(BRANCHES))
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    p.add_argument("--decisions", type=int, nargs="+", default=[25, 75, 150, 350],
                   help="Zero-based real decision indices; at least 1 to have a donor.")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--controller-seed", type=int, default=55)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mc-rollouts", type=int, default=32)
    p.add_argument("--action-count", type=int, default=8)
    p.add_argument("--data-lanes", nargs="+", choices=["common", "natural"], default=["common", "natural"])
    p.add_argument("--fit-steps", type=int, default=32)
    p.add_argument("--fit-states", type=int, default=16)
    p.add_argument("--capture-rounds", type=int, nargs="+", default=None,
                   help="Optional post-round snapshot filter; initial/block/final snapshots remain.")
    p.add_argument("--no-target-cross", action="store_true")
    p.add_argument("--real-rollouts", type=int, default=0,
                   help="Optional matched model/real prefix and finite prior-tail branches per root/arm.")
    p.add_argument("--real-tail-steps", type=int, default=1000)
    p.add_argument("--replan-steps", type=int, default=0,
                   help="Optional real first-action/memory/full interventions; 0 disables.")
    p.add_argument("--replan-repeats", type=int, default=3)
    p.add_argument("--save-snapshots", action="store_true",
                   help="Save trusted local donor module snapshots and exact simulator states.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve and validate the matrix/checkpoint contract without loading networks or creating output.")
    return p


def validate_options(args):
    for name in ("horizons", "rounds", "decisions"):
        positive_ints(tuple(getattr(args, name)), name)
    for name in ("mc_rollouts", "action_count", "fit_steps", "fit_states", "max_steps",
                 "real_tail_steps", "replan_repeats"):
        positive_ints((getattr(args, name),), name)
    if args.mc_rollouts < 2 or args.action_count < 4:
        raise ValueError("At least two MC rollouts and four candidate actions are required.")
    if args.fit_states < 4 or args.fit_states % 2:
        raise ValueError("fit_states must be even and >=4.")
    if max(args.decisions) >= args.max_steps:
        raise ValueError("Requested decisions must be smaller than max_steps.")
    if args.real_rollouts < 0 or args.replan_steps < 0:
        raise ValueError("Real branch counts must be nonnegative.")
    if (args.real_rollouts or args.replan_steps) and "natural" not in args.data_lanes:
        raise ValueError("Real branches require the natural collection lane.")
    for name in ("seeds", "histories", "data_lanes"):
        values = getattr(args, name)
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and unique.")
    if any(seed < 0 for seed in args.seeds) or args.controller_seed < 0:
        raise ValueError("Seeds must be nonnegative.")
    if args.capture_rounds is not None and (len(set(args.capture_rounds)) != len(args.capture_rounds)
            or any(isinstance(index, bool) or not isinstance(index, int) or index < 0
                   for index in args.capture_rounds)):
        raise ValueError("capture_rounds must contain unique nonnegative integers.")


def resolved_setting(base, horizon, rounds):
    resolved = deepcopy(base)
    params = resolved["algorithm_config"]["alg_params"]
    # Explicit evaluation overrides have their own protocol and manifest.
    params.update(inner_rollout_horizon=horizon, inner_rounds=rounds,
                  inner_first_action_rounds=None, inner_solve_interval=1,
                  compile=False, compile_strict=False, wandb=False,
                  inner_actor_writeback_coef=0., inner_critic_writeback_coef=0.,
                  inner_critic_transfer_head="retain",
                  inner_rebase_persistent=False)
    for name in ("actor", "critic", "replay", "temperature", "actor_optimizer",
                 "critic_optimizer", "temperature_optimizer"):
        params[f"inner_{name}_scope"] = "action"
    # Keep every imagined transition, so changing H/J does not silently change
    # the fraction of data discarded through replay capacity overflow.
    params["inner_replay_capacity"] = max(int(params.get("inner_replay_capacity", 1)),
        horizon * rounds * int(params["inner_rollouts_per_round"]))
    return resolved


def source_identity():
    root = Path(__file__).resolve().parent
    paths = [Path(__file__), root / "utils/transfer_diagnostics.py",
             root / "utils/transfer_diagnostic_metrics.py", root / "utils/transfer_diagnostic_real.py",
             root / "RL/tdmpc2_core/inner_improvement.py", root / "RL/tdmpc2_core/inner_trace.py"]
    files = {str(path.relative_to(root)): _file_sha256(path) for path in paths}
    # Absolute executable, no cwd change, and close_fds=False permit Python
    # 3.10's posix_spawn path. Forking an initialized macOS OpenMP runtime can
    # abort even for a read-only Git metadata query.
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to record the diagnostic source revision.")
    revision = subprocess.run([git, "-C", str(root), "rev-parse", "HEAD"],
        close_fds=False, capture_output=True, text=True, check=True).stdout.strip()
    return {"git_head": revision, "files": files,
            "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def summarize(roots):
    """Equal-root within episode, then episode-level uncertainty; never MC as n."""
    grouped = {}
    for root in roots:
        for branch in root["diagnostics"]["branches"]:
            key = (root["horizon"], root["rounds"], root["history"], branch["data_lane"], branch["branch"])
            grouped.setdefault(key, []).append(dict(episode=root["episode_seed"], root=root["decision"],
                value=branch["final"]["model_actor_objective"]))
    rows = []
    for (horizon, rounds, history, lane, branch), records in grouped.items():
        rows.append(dict(horizon=horizon, rounds=rounds, history=history, data_lane=lane,
            branch=branch, metric="final_model_actor_objective",
            **aggregate_paired_root_metrics(records, value_key="value")))
    # Pair effects at a root before averaging. This does not label a model score
    # as real episode return, and source histories are never pooled together.
    effects = {}
    for root in roots:
        for lane in {b["data_lane"] for b in root["diagnostics"]["branches"]}:
            values = {b["branch"]: b["final"]["model_actor_objective"]
                      for b in root["diagnostics"]["branches"] if b["data_lane"] == lane}
            if set(values) != set(BRANCHES):
                raise ValueError("Incomplete four-way diagnostic comparison.")
            for name, value in dict(actor=values["actor"] - values["fresh"],
                critic=values["critic"] - values["fresh"], joint=values["joint"] - values["fresh"],
                interaction=values["joint"] - values["actor"] - values["critic"] + values["fresh"]).items():
                key = root["horizon"], root["rounds"], root["history"], lane, name
                effects.setdefault(key, []).append(dict(episode=root["episode_seed"], root=root["decision"], value=value))
    for (horizon, rounds, history, lane, name), records in effects.items():
        rows.append(dict(horizon=horizon, rounds=rounds, history=history, data_lane=lane,
            branch=name, metric="paired_model_objective_effect",
            **aggregate_paired_root_metrics(records, value_key="value")))
    diagnostic_groups = {}
    def add(root, lane, branch, metric, value):
        if value is None:
            return
        key = root["horizon"], root["rounds"], root["history"], lane, branch, metric
        diagnostic_groups.setdefault(key, []).append(dict(episode=root["episode_seed"],
            root=root["decision"], value=float(value)))
    for root in roots:
        diagnostics = root["diagnostics"]
        for branch in diagnostics["branches"]:
            for stage_name, stage in (("initial", branch["stages"][0]), ("final", branch["final"])):
                for check in stage["critic_checks"]:
                    if check["critic"] == "online" and check["continuation"] == "current":
                        for metric in ("relative_rmse", "spearman", "top_action_regret"):
                            add(root, branch["data_lane"], branch["branch"], f"critic_{stage_name}_{metric}", check[metric])
                add(root, branch["data_lane"], branch["branch"], f"critic_{stage_name}_directional_slope",
                    stage["directional"][0]["reference_slope"])
        for name, curve in diagnostics["stationary"]["curves"].items():
            for stage, point in (("initial", curve[0]), ("final", curve[-1])):
                add(root, "stationary", name, f"heldout_{stage}_rmse", point["heldout_rmse"])
        shift = diagnostics["portability"]["horizon_shift"]
        if shift["applicable"]:
            add(root, "portability", "carried_critic", "mean_absolute_horizon_target_change",
                np.abs(shift["actionwise_target_change"]).mean())
        for crossing in diagnostics["target_cross"]:
            add(root, "common", f"online_{crossing['online']}_target_{crossing['target']}",
                "target_cross_model_objective", crossing["final"]["model_actor_objective"])
        for real in diagnostics["real"]:
            for metric in ("model_prefix_error", "terminal_prediction_error", "real_mc_return"):
                add(root, "real_prefix", real["branch"], metric, real.get(metric))
        for replan in diagnostics["replanning"]:
            add(root, "replanning", f"{replan['intervention']}_{replan['branch']}",
                "real_return", replan["real_return"])
    for (horizon, rounds, history, lane, branch, metric), records in diagnostic_groups.items():
        rows.append(dict(horizon=horizon, rounds=rounds, history=history, data_lane=lane,
            branch=branch, metric=metric, **aggregate_paired_root_metrics(records, value_key="value")))
    return rows


def run(args):
    validate_options(args)
    if not args.dry_run and args.output_dir.exists():
        raise FileExistsError(f"Diagnostic output already exists: {args.output_dir}")
    matrix = load_preset_matrix(args.matrix)
    context = load_checkpoint_context(args.checkpoint, metadata_path=args.metadata) if matrix["base_alg_config"] == "checkpoint" else None
    base = resolve_preset(args.matrix, args.preset, matrix=matrix, checkpoint_context=context)
    _validate_checkpoint_contract(matrix, args.checkpoint, context, [base])
    settings = [(h, j, resolved_setting(base, h, j)) for h in args.horizons for j in args.rounds]
    options = dict(mc_rollouts=args.mc_rollouts, action_count=args.action_count,
        data_lanes=args.data_lanes, fit_steps=args.fit_steps, fit_states=args.fit_states,
        capture_rounds=args.capture_rounds, target_cross=not args.no_target_cross,
        real_rollouts=args.real_rollouts, real_tail_steps=args.real_tail_steps,
        replan_steps=args.replan_steps, replan_repeats=args.replan_repeats)
    manifest = dict(protocol=PROTOCOL, status="resolved", checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=_file_sha256(args.checkpoint), matrix=str(args.matrix.resolve()),
        matrix_sha256=_file_sha256(args.matrix), source=source_identity(),
        preset=args.preset, horizons=args.horizons, rounds=args.rounds,
        histories=args.histories, episode_seeds=args.seeds, decisions=args.decisions,
        max_steps=args.max_steps, controller_seed=args.controller_seed, device=args.device,
        options=options, runtime=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        settings=[dict(horizon=h, rounds=j, resolved=resolved) for h, j, resolved in settings],
        semantics=dict(scope="successive_decisions_within_episode", production_protocol=False,
            optimization_budget="fixed J,C,A,N; model transitions scale with H; no equal-compute claim",
            cross_horizon="H-specific source histories visit different roots; only within-root forks are state matched",
            replay="fresh each decision; common lane uses fixed prior collection actor; natural lane uses each learner",
            target="four-way target copies selected online; separate online/target cross uses prior/carried online weights",
            mc="held-out frozen-model returns, explicit continuation policy and configured terminal value",
            head_reduction="expectation over random critic-pair selection; eager dropout-free evaluation",
            real="optional fixed-policy prefixes/finite tails and separately labeled replanning interventions",
            uncertainty="MC error within root; episode-level aggregation across roots"))
    if args.dry_run:
        return manifest
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.started.json", manifest)
    roots, episodes = [], []
    for horizon, rounds, resolved in settings:
        env = source = audit = None
        try:
            env = _make_env(resolved)
            source, _ = _initialize_frozen_model(resolved, env, args.checkpoint, args.controller_seed, device=args.device)
            audit, _ = _initialize_frozen_model(resolved, env, args.checkpoint, args.controller_seed, device=args.device)
            validate_controller(source)
            validate_controller(audit)
            outer_source, outer_audit = _outer_state_digest(source), _outer_state_digest(audit)
            for history in args.histories:
                for episode_seed in args.seeds:
                    observation, _ = env.reset(seed=episode_seed)
                    source.agent.inner_engine.reset_for_evaluation(solver_seed(args.controller_seed, "source", episode_seed))
                    donor = previous_observation = previous_action = None
                    total_reward, captured = 0., []
                    for decision in range(args.max_steps):
                        rng = deepcopy(source.agent.inner_engine.rng.training_state_dict())
                        if decision in args.decisions:
                            root_name = f"h{horizon}-j{rounds}/{history}/seed-{episode_seed}/decision-{decision}"
                            snapshot = None
                            if args.real_rollouts or args.replan_steps or args.save_snapshots:
                                from utils.transfer_diagnostic_real import capture_simulator_snapshot
                                snapshot = capture_simulator_snapshot(env)
                            diagnostics = audit_root(audit, observation, rng, donor,
                                previous_observation=previous_observation, previous_action=previous_action,
                                horizon=horizon, seed=solver_seed(args.controller_seed, "audit", episode_seed, decision),
                                options=options, env=env, simulator_snapshot=snapshot)
                            root = dict(horizon=horizon, rounds=rounds, history=history,
                                episode_seed=episode_seed, decision=decision, root_id=root_name,
                                observation=np.asarray(observation).tolist(),
                                previous_observation=np.asarray(previous_observation).tolist(),
                                previous_action=np.asarray(previous_action).tolist(), diagnostics=diagnostics)
                            write_json(output / root_name / "diagnostics.json", root)
                            if args.save_snapshots:
                                with (output / root_name / "donor.pt").open("xb") as stream:
                                    torch.save(dict(donor=donor, solver_rng=rng), stream)
                                write_json(output / root_name / "simulator.json", snapshot.to_dict())
                            roots.append(root)
                            captured.append(decision)
                        action, _, final = solve_fork(source, observation, rng, donor=donor,
                            branch=history if donor is not None else "fresh", capture_rounds=())
                        previous_observation = np.asarray(observation).copy()
                        previous_action = source._scale_action(action)
                        observation, reward, terminated, truncated, _ = env.step(action)
                        total_reward += float(reward)
                        donor = final
                        if terminated or truncated:
                            break
                    episodes.append(dict(horizon=horizon, rounds=rounds, history=history,
                        seed=episode_seed, real_return=total_reward, steps=decision + 1,
                        captured_decisions=captured, unreached_decisions=sorted(set(args.decisions) - set(captured))))
                    if _outer_state_digest(source) != outer_source or _outer_state_digest(audit) != outer_audit:
                        raise RuntimeError("Diagnostic changed the frozen outer learner.")
                    write_json(output / f"h{horizon}-j{rounds}/{history}/seed-{episode_seed}/episode.json", episodes[-1])
        finally:
            _close_resources(source, audit, env)
    summary = summarize(roots)
    write_json(output / "summary.json", summary)
    manifest.update(status="complete", root_count=len(roots), episodes=episodes, outer_state_unchanged=True)
    write_json(output / "manifest.json", manifest)
    write_report(output, summary, manifest)
    return manifest


def write_report(output, summary, manifest):
    lines = ["# Inner SAC transfer diagnostics", "",
             f"Protocol: `{PROTOCOL}`. Captured roots: {manifest['root_count']}.", "",
             "H-specific histories visit different states. Effects below are paired at the same root within each H/J/history.",
             "Model objective effects are not measured real-return improvements.", "",
             "| H | J | Source history | Data | Effect | Episode aggregate |",
             "|---|---|---|---|---|---|"]
    for row in summary:
        if row["metric"] == "paired_model_objective_effect":
            error = "unavailable" if row["episode_se"] is None else f"{row['episode_se']:.4g}"
            stats = f"{row['mean']:.4g}; SE {error}; {row['episodes']} episodes, {row['roots']} roots"
            lines.append(f"| {row['horizon']} | {row['rounds']} | {row['history']} | {row['data_lane']} | {row['branch']} | {stats} |")
    lines.extend(["", "Detailed critic/ranking/gradient checks, target crossings, support/horizon probes, stationary learning curves, and optional real branches are in each root's `diagnostics.json`.",
                  "A partial bundle has only `manifest.started.json`; it must not be treated as a completed run.", ""])
    for title, selected in (
        ("Critic action guidance", {"critic_final_relative_rmse", "critic_final_spearman",
                                    "critic_final_top_action_regret", "critic_final_directional_slope"}),
        ("Stationary learning and horizon portability", {"heldout_initial_rmse", "heldout_final_rmse",
                                                        "mean_absolute_horizon_target_change"}),
    ):
        lines.extend([f"## {title}", "", "| H | J | Source | Data | Component | Metric | Mean | Episode SE |",
                      "|---|---|---|---|---|---|---|---|"])
        for row in summary:
            if row["metric"] in selected:
                error = "—" if row["episode_se"] is None else f"{row['episode_se']:.4g}"
                lines.append(f"| {row['horizon']} | {row['rounds']} | {row['history']} | {row['data_lane']} | {row['branch']} | {row['metric']} | {row['mean']:.4g} | {error} |")
        lines.append("")
    with (output / "report.md").open("x") as stream:
        stream.write("\n".join(lines))


def main(argv=None):
    args = parser().parse_args(argv)
    result = run(args)
    if args.dry_run:
        print(json.dumps(result, indent=2))
    else:
        print(f"Saved {result['root_count']} roots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
