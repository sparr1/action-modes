"""Prepare or inspect the four-task Meta-World campaign; never execute jobs."""

import argparse
import copy
import json
from pathlib import Path
import shlex


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/metaworld"
TASKS = ("assembly", "hand-insert", "push", "stick-pull")
ACTION_REPEAT = 2
EPISODE_DECISIONS = 100
PROTOCOL = "metaworld-tdmpc2-04be337-state-final-success-v1"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)


def manifest(task, names, trials):
    return {
        "study_type": "metaworld_four_task_state",
        "study_note": "Single-task TD-MPC2 Meta-World v2 protocol (04be337). "
        "Final-timestep success, action repeat 2, 100 decisions per episode.",
        "env_params": {"task": task, "obs": "state", "render_mode": None, "protocol": PROTOCOL},
        "trials": trials,
        "configs": names,
        "logs": "none",
        "save_trials": "none",
        "checkpoint_every": 25_000,
        "save_strat": ["all", "best", "latest"],
        "checkpoint_best_window": 100,
        "log_info": False,
        "log_type": "summary",
    }


def validate_algorithm(run):
    """Require a deliberate Meta-World recipe, rather than porting DMC silently."""
    if run.get("env") != "MetaWorld-v0":
        raise ValueError("Set env='MetaWorld-v0' in an explicitly reviewed algorithm recipe first.")
    params = run.get("alg_params", {})
    if params.get("obs") != "state" or params.get("episodic", False):
        raise ValueError("This setup requires state observations and episodic=false.")
    if params.get("episode_length", EPISODE_DECISIONS) != EPISODE_DECISIONS:
        raise ValueError("Meta-World has a 100-decision episode length.")
    if params.get("eval_inner_comparison", False):
        raise ValueError("Paired-controller snapshot evaluation is not supported by this adapter.")
    if params.get("eval_freq") is None or params.get("eval_episodes", 0) < 1:
        raise ValueError("Configure online evaluation to record final-step success.")
    if params.get("eval_csv_path"):
        raise ValueError("Remove fixed eval_csv_path; each task/algorithm/seed gets a separate CSV.")
    if params.get("wandb", False) or params.get("wandb_mode", "disabled") != "disabled":
        raise ValueError("Preparation requires wandb=false and wandb_mode='disabled'.")
    if "AMBI" in run.get("alg", "") and params.get("critic_value_mode", "single") == "single":
        if params.get("critic_geometry_diagnostics") is False:
            raise ValueError("State/single-Q AMBI must retain critic geometry diagnostics.")
    for field, minimum in (("total_steps", 1), ("seed", 0)):
        value = run.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}.")
    if run["total_steps"] % EPISODE_DECISIONS:
        raise ValueError("The training budget must end on a 100-decision episode boundary.")


def plan(config_dir, results_dir, *, files=None):
    """Return unique output paths and shell commands, without importing learners."""
    config_dir, results_dir = Path(config_dir).resolve(), Path(results_dir).resolve()
    get = (lambda path: files[path.relative_to(config_dir).as_posix()]) if files is not None else read_json
    cells = []
    for task in TASKS:
        run_path = config_dir / "experiments" / f"{task}.json"
        experiment = get(run_path)
        if experiment.get("env_params") != {
            "task": task, "obs": "state", "render_mode": None, "protocol": PROTOCOL
        }:
            raise ValueError(f"Unexpected environment protocol in {run_path}")
        if experiment.get("overrides_alg"):
            raise ValueError("Use complete algorithm files; shallow manifest overrides are not supported here.")
        names, trials = experiment["configs"], experiment["trials"]
        if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
            raise ValueError("trials must be a positive integer.")
        if not names or len(set(names)) != len(names):
            raise ValueError("Algorithm names must be unique and nonempty.")
        for index, name in enumerate(names):
            if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
                raise ValueError("Algorithm names must be plain basenames.")
            run = get(config_dir / "algs" / f"{name}.json")
            validate_algorithm(run)
            for trial in range(trials):
                seed = run["seed"] + trial
                cell_dir = results_dir / task / name / f"seed_{seed}"
                csv_path = cell_dir / "evaluation.csv"
                command = [
                    "env", "WANDB_MODE=disabled", f"TDMPC2_EVAL_CSV={csv_path}",
                    str(ROOT / "environments/metaworld/python"), str(ROOT / "main.py"),
                    "--run", str(run_path), "--alg-dir", str(config_dir / "algs"),
                    "--alg-index", str(index), "--trial-index", str(trial),
                    "--num-runs", "1", "--log-dir", str(cell_dir / "artifacts"),
                ]
                cells.append({
                    "task": task, "algorithm": name, "trial_index": trial, "seed": seed,
                    "decisions": run["total_steps"], "raw_training_steps": ACTION_REPEAT * run["total_steps"],
                    "evaluation_csv": str(csv_path), "command": shlex.join(command),
                })
    return {"execution": "none; commands only", "working_directory": str(ROOT),
            "action_repeat": ACTION_REPEAT, "episode_decisions": EPISODE_DECISIONS,
            "cell_count": len(cells), "cells": cells}


def prepare(algorithm_paths, output_dir, *, trials=1, decisions=1_000_000, base_seed=55):
    """Build complete configs in memory; only the CLI's prepare command writes."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer.")
    files, names = {}, []
    for path in map(Path, algorithm_paths):
        name = path.stem
        if name in names:
            raise ValueError(f"Duplicate algorithm basename: {name}")
        run = copy.deepcopy(read_json(path))
        # This is a budget/seed change only: inner schedules, entropy targets,
        # discount, model sizes and training objectives remain the user's choice.
        run.update(total_steps=decisions, seed=base_seed, episodes=None)
        validate_algorithm(run)
        names.append(name)
        files[f"algs/{name}.json"] = run
    if not names:
        raise ValueError("Supply at least one explicitly reviewed algorithm configuration.")
    for task in TASKS:
        files[f"experiments/{task}.json"] = manifest(task, names, trials)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("plan", help="Print commands for existing configs; never run them")
    inspect.add_argument("--config-dir", type=Path, default=CONFIG_ROOT)
    inspect.add_argument("--results-dir", type=Path, required=True)
    create = commands.add_parser("prepare", help="Write a fresh config directory; never execute jobs")
    create.add_argument("--algorithm-config", type=Path, action="append", required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--results-dir", type=Path, required=True)
    create.add_argument("--trials", type=int, default=1)
    create.add_argument("--decisions", type=int, default=1_000_000)
    create.add_argument("--base-seed", type=int, default=55)
    create.add_argument("--dry-run", action="store_true", help="Print the plan without writing files")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan(args.config_dir, args.results_dir)
    else:
        files = prepare(args.algorithm_config, args.output_dir, trials=args.trials,
                        decisions=args.decisions, base_seed=args.base_seed)
        result = plan(args.output_dir, args.results_dir, files=files)
        result["prepared"] = not args.dry_run
        if not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            for relative, data in files.items():
                target = args.output_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("x") as stream:
                    json.dump(data, stream, indent=2, allow_nan=False)
                    stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
