"""Read frozen evaluation artifacts as checkpoint records, without ML or W&B.

The identity describes the executed controller, not the training configuration
stored in a checkpoint. Source files and unmodified legacy artifacts remain in
provenance; publication state and wall-clock measurements never identify a
scientific record.
"""

from __future__ import annotations

import ast
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": "nan" if math.isnan(value) else
                "positive_infinity" if value > 0 else "negative_infinity"}
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_safe(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root, *args):
    # On macOS, forking after Torch initializes OpenMP can abort the child.
    # An absolute executable, -C rather than cwd, and close_fds=False permit
    # Python's posix_spawn path. Python-created descriptors are non-inheritable.
    git = shutil.which("git")
    _require(bool(git), "Git is required to verify scientific source")
    return subprocess.check_output([git, "-C", str(root), *args],
                                   stderr=subprocess.PIPE, close_fds=False)


def _number(value, name):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value), f"Missing or nonfinite {name}")
    return float(value)


def _checkpoint(step, digest):
    _require(isinstance(step, int) and not isinstance(step, bool) and step >= 0,
             "Invalid checkpoint step")
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
             "Invalid checkpoint SHA256")
    return {"step": step, "sha256": digest}


def _ancestors(path):
    # Bounded local ancestry, never an unrelated global search or remote lookup.
    return list(Path(path).resolve().parents)[:6]


def _provenance(path):
    for parent in _ancestors(path):
        candidate = parent / "provenance.json"
        if candidate.is_file():
            return _read(candidate), candidate
    return {}, None


def _source(checkpoint, path, *, checkpoint_inventory=None, source_run=None):
    provenance, provenance_path = _provenance(path)
    pin = provenance.get("checkpoint", {})
    pinned_hash = pin.get("sha256", provenance.get("checkpoint_sha256"))
    if pinned_hash is not None:
        _require(pinned_hash == checkpoint["sha256"], "Provenance checkpoint hash mismatch")
    if pin.get("step") is not None:
        _require(pin["step"] == checkpoint["step"], "Provenance checkpoint step mismatch")
    declared = checkpoint.get("source_run")
    pinned_source = provenance.get("source_run") if pinned_hash else None
    candidates = [Path(checkpoint_inventory)] if checkpoint_inventory else []
    if not checkpoint_inventory:
        for parent in _ancestors(path):
            candidates.extend(sorted(parent.glob("*checkpoint-manifest.json")))
    matching = []
    for candidate in dict.fromkeys(candidates):
        inventory = _read(candidate)
        for row in inventory.get("checkpoints", []):
            if row.get("sha256") == checkpoint["sha256"]:
                _require(row.get("step") == checkpoint["step"], "Inventory checkpoint step mismatch")
                if checkpoint.get("metadata_sha256") and row.get("metadata_sha256"):
                    _require(checkpoint["metadata_sha256"] == row["metadata_sha256"],
                             "Inventory checkpoint sidecar hash mismatch")
                matching.append((inventory.get("source_run"), candidate))
        # Transfer inventories predate checkpoint-manifest.json. Verify the
        # exact weight and its named training directory, not merely a step.
        for row in inventory.get("files", []):
            if row.get("kind") == "weights" and row.get("sha256") == checkpoint["sha256"]:
                _require(row.get("step") == checkpoint["step"], "Transfer checkpoint step mismatch")
                trial = checkpoint.get("metadata", {}).get("trial_run_params", {})
                name = trial.get("alg_params", {}).get("wandb_run_name")
                inventories = [item for item in inventory.get("runs", [])
                               if row.get("path", "").startswith(item.get("models_directory", "") + "/")]
                _require(len(inventories) == 1 and name and inventories[0].get("name") == name,
                         "Transfer inventory does not identify the checkpoint training run")
                matching.append((declared or source_run, candidate))
    sources = {item for item in (declared, pinned_source, source_run,
                                *(source for source, _ in matching)) if item}
    _require(len(sources) == 1, "Missing or conflicting prior source-run provenance")
    source = sources.pop()
    _require(len(source.split("/")) == 3, "Source run must be entity/project/run-id")
    if source_run and not (declared or pinned_source or matching):
        raise ValueError("A source-run argument cannot substitute for checkpoint provenance")
    _require(pinned_source or matching or checkpoint.get("source_run_verified"),
             "Prior source is declared but unverified; provide its checkpoint inventory")
    evidence = {"checkpoint_source_declared": bool(declared),
                "checkpoint_source_verified": bool(pinned_source or matching or
                                                     checkpoint.get("source_run_verified")),
                "launch_provenance": provenance}
    files = {}
    if provenance_path:
        files["provenance.json"] = str(provenance_path)
    for index, (_, inventory_path) in enumerate(matching):
        files[f"provenance/checkpoint-inventory-{index}.json"] = str(inventory_path.resolve())
    return source, evidence, files


class _NoDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        if hasattr(node, "body") and isinstance(node.body, list) and node.body:
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                    and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        return node


class _WithoutControlTiming(ast.NodeTransformer):
    """Exclude the audited, observational prediction timer from compatibility.

    Only these two local variable names and this result dictionary field are
    ignored. Controller calls, their arguments, and every state mutation remain
    part of the scientific fingerprint.
    """
    names = {"control_seconds", "prediction_started"}

    def visit_Assign(self, node):
        if all(isinstance(target, ast.Name) and target.id in self.names for target in node.targets):
            return None
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id in self.names:
            return None
        return self.generic_visit(node)

    def visit_Dict(self, node):
        kept = [(key, value) for key, value in zip(node.keys, node.values)
                if not (isinstance(key, ast.Constant) and key.value == "control_seconds")]
        node.keys = [key for key, _ in kept]
        node.values = [value for _, value in kept]
        return self.generic_visit(node)


@lru_cache(maxsize=128)
def scientific_identity(algorithm, controller, commit, dirty=False, source_sha256=None):
    """Conservative scientific-source identity, excluding publisher/report code.

    Hash executable ASTs of the controller stack and environment. Evaluator
    protocol implementation is also pinned. Unrelated branch changes may
    require an explicit compatibility audit; they are never silently accepted.
    Dirty legacy trees remain pinned to their saved complete source digest.
    """
    if dirty:
        _require(bool(source_sha256), "Dirty results lack saved source fingerprint")
        return {"version": 1, "algorithm": algorithm, "dirty_source_sha256": source_sha256}
    _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
             "Missing pinned scientific code commit")
    root = Path(__file__).resolve().parents[1]
    if "XQC" in algorithm:
        paths = ["RL/AMBIXQC.py", "RL/xqc_core.py", "RL/tdmpc2_core/ambixqc_agent.py",
                 "RL/tdmpc2_core/inner_xqc.py", "RL/tdmpc2_core/xqc_controller.py",
                 "RL/tdmpc2_core/xqc_mppi.py"]
    elif "AMBI" in algorithm:
        paths = ["RL/AMBITDMPC2.py", "RL/tdmpc2_core/ambi_agent.py",
                 "RL/tdmpc2_core/inner_improvement.py"]
    else:
        paths = ["RL/TDMPC2.py", "RL/tdmpc2_core/agent.py", "RL/tdmpc2_core/mppi.py"]
    paths += ["RL/tdmpc2_core/common", "domains", "environments/dmcontrol/uv.lock"]
    evaluator = ("evaluate_tdmpc2_mppi_checkpoint.py" if algorithm == "TDMPC2/TDMPC2Baseline"
                 else "evaluate_ambi_checkpoint.py")
    # Main/CLI orchestrate publication; these functions implement scientific
    # initialization, randomness, action execution, and frozen-state checks.
    selected = ({"_namespaced_seed", "_set_controller", "_run_arm", "_reset_paired_environments",
                 "_capture_global_rng", "_restore_global_rng", "_predicted_action_gain",
                 "evaluate_tdmpc2_mppi_checkpoint"}
                if evaluator.startswith("evaluate_tdmpc2") else
                {"_make_env", "_seed_spaces", "_initialize_frozen_model", "evaluate_preset"})
    try:
        entries = _git_output(root, "ls-tree", "-rz", commit, "--", *paths)
        names = sorted(entry.split(b"\t", 1)[1].decode() for entry in entries.split(b"\0") if entry)
        _require(paths[0] in names, "Pinned algorithm source is incomplete")
        hashes = {}
        helper_selections = ({"utils/ambi_benchmark.py": {"canonical_hash", "solver_seed", "protocol_for"}}
                             if evaluator == "evaluate_ambi_checkpoint.py" else {})
        for name in [*names, evaluator, *helper_selections]:
            if not (name.endswith(".py") or name.endswith("uv.lock")):
                continue
            content = _git_output(root, "show", f"{commit}:{name}")
            if name.endswith(".py"):
                tree = ast.parse(content)
                if name == evaluator or name in helper_selections:
                    names_selected = selected if name == evaluator else helper_selections[name]
                    tree.body = [node for node in tree.body if getattr(node, "name", None) in names_selected]
                    _require(any(getattr(n, "name", None) in names_selected for n in tree.body),
                             "Scientific evaluator functions missing")
                if name == evaluator:
                    tree = _WithoutControlTiming().visit(tree)
                content = ast.dump(_NoDocstrings().visit(tree), include_attributes=False).encode()
            hashes[name] = hashlib.sha256(content).hexdigest()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Pinned scientific source is unavailable in Git") from error
    return {"version": 1, "algorithm": algorithm, "source_sha256": _hash(hashes)}


def planner_identity(config, result, algorithm, action_rule):
    """Canonicalize active resolved settings; learned checkpoint values stay out."""
    evaluation = result.get("evaluation_controller")
    if evaluation:
        _require(evaluation.get("type") == "mppi", "Unsupported evaluation controller")
        settings = copy.deepcopy(evaluation["settings"])
        settings.pop("iterations", None)  # Effective iterations determine executed work.
        semantics = copy.deepcopy(evaluation["protocol"])
        semantics.pop("reward_scale", None)
        semantics["reward_scale_initialization"] = "frozen_checkpoint_real_scale"
        return {"type": "mppi", "backend": semantics.pop("algorithm"),
                "settings": settings, "semantics": semantics, "action_rule": action_rule}
    operator = config.get("inner_operator")
    if result.get("controller") in {"prior", "policy_prior"} or operator == "none":
        return {"type": "prior", "action_rule": action_rule}
    _require(operator in {"sac", "td3", "xqc", "mppi"}, "Unknown resolved inner operator")
    active = {key: value for key, value in config.items() if key.startswith("inner_")}
    if operator == "xqc":
        # Before the outer-terminal ablation existed, every XQC terminal
        # bootstrap used the inner learner. The new resolver makes that
        # existing behavior explicit; absence is not a distinct planner.
        active.setdefault("inner_terminal_bootstrap", "inner")
        # Migrated XQC round-update registry identities predate this setting.
        # Keep their exact identity; step timing is an explicit new planner.
        if active.get("inner_update_timing", "round") == "round":
            active.pop("inner_update_timing", None)
        if active.get("inner_policy_delay") in (None, config.get("xqc_policy_delay", 3)):
            active.pop("inner_policy_delay", None)
    ignored = {"inner_execution_action", "inner_execution_noise_std", "inner_execution_std_scale",
               "inner_diagnostic_rollouts", "inner_diagnostics_every", "inner_horizon_ratio",
               "inner_nominal_critic_utd", "inner_nominal_transitions_per_round",
               "inner_nominal_updates_per_round", "inner_expected_update_slots",
               "inner_total_optimizer_steps_per_action", "inner_primary_optimizer_steps_per_action",
               "inner_adaptation", "inner_grad_clip_norm", "inner_horizon", "inner_iterations",
               "inner_buffer_size", "inner_rollouts", "inner_tau", "inner_target_update_interval",
               "inner_updates_per_iteration", "inner_actor_writeback_coef", "inner_critic_writeback_coef"}
    for key in ignored:
        active.pop(key, None)
    if not config.get("inner_explorer_active", False):
        for key in list(active):
            if key.startswith(("inner_explorer_", "inner_primary_", "inner_mixture_")):
                active.pop(key)
    if not config.get("inner_param_noise_active", False):
        for key in list(active):
            if key.startswith("inner_param_noise_"):
                active.pop(key)
    if operator != "mppi":
        for key in list(active):
            if key.startswith("inner_mppi_"):
                active.pop(key)
    if operator != "td3":
        for key in list(active):
            if key.startswith("inner_td3_") or key.startswith("inner_actor_target_"):
                active.pop(key)
    for component in ("actor", "critic"):
        if config.get(f"inner_{component}_adaptation") != "lora":
            for key in list(active):
                if key.startswith(f"inner_{component}_lora_"):
                    active.pop(key)
    if config.get("inner_execution_policy_source", "primary") == "primary":
        active.pop("inner_execution_handoff_samples", None)
    if all(config.get(f"inner_{component}_scope", "action") == "action"
           for component in ("actor", "critic", "temperature", "replay")):
        active.pop("inner_rebase_persistent", None)
    if config.get("inner_temperature_initialization") == "inherit_outer":
        active.pop("inner_temperature", None)
    if config.get("inner_temperature_mode") != "auto":
        for suffix in ("lr", "grad_clip_norm", "optimizer_scope", "updates_per_action"):
            active.pop(f"inner_temperature_{suffix}", None)
    # Inactive scheduler aliases must not split identical resolved schedules.
    for key in list(active):
        if active[key] is None or ("reward_scale" in key and "initialization" not in key):
            active.pop(key)
    shared = {key: config[key] for key in ("discount", "episodic", "grad_clip_norm",
              "sac_actor_loss_scale_mode", "sac_actor_loss_scale_tau", "dropout",
              "xqc_adam_eps", "xqc_policy_delay", "xqc_target_update_interval", "xqc_tau")
              if key in config}
    if shared.get("sac_actor_loss_scale_mode") == "none":
        shared.pop("sac_actor_loss_scale_tau", None)
    return {"type": operator, "backend": algorithm, "action_rule": action_rule,
            "settings": {**shared, **active}}


def _episode_protocol(protocol, seeds, config=None, action_rule=None):
    keys = {"environment", "env_wrapper", "env_wrappers", "observation", "action_rule",
            "controller_seed", "seed_scheme", "max_steps"}
    normalized = {key: copy.deepcopy(value) for key, value in protocol.items() if key in keys}
    normalized["action_rule"] = action_rule or normalized.get("action_rule")
    normalized["environment_seeds"] = sorted(seeds)
    normalized["max_steps"] = normalized.get("max_steps") or (config or {}).get("episode_length")
    normalized["mode"] = "episodes"
    return normalized


def resolved_checkpoint_config(checkpoint, resolved, *, env=None):
    """Run the wrapper's configuration resolver without creating a learner.

    This deliberately reuses the maintained parser: reproducing its defaults
    and legacy schedule aliases in a publisher would drift. Imports are lazy so
    offline record normalization itself remains standard-library only.
    """
    import importlib
    from types import SimpleNamespace
    import gymnasium as gym
    import numpy as np

    algorithm_config = copy.deepcopy(resolved["algorithm_config"])
    module, name = algorithm_config["alg"].rsplit("/", 1)
    cls = getattr(importlib.import_module("RL." + module.replace("/", ".")), name)
    if env is None:
        metadata = checkpoint["metadata"]
        observation = metadata["trial_run_params"]["resolved_runtime"]["observation"]
        _require(algorithm_config.get("env", resolved.get("environment", {}).get("id")) == "DMControl-v0",
                 "An actual environment is required for non-DMControl configuration preflight")
        shape = tuple(observation["shape"])
        mode = observation["mode"]
        env = SimpleNamespace(
            observation_type=mode,
            observation_space=gym.spaces.Box(0, 255, shape, np.uint8) if mode == "rgb" else
                              gym.spaces.Box(-np.inf, np.inf, shape, np.float32),
            action_space=gym.spaces.Box(-1, 1, (observation["action_dim"],), np.float64),
            spec=SimpleNamespace(max_episode_steps=observation["episode_length"]))
    model = cls.__new__(cls)
    model.env = env
    model.run_params = {**algorithm_config, "device": "cpu"}
    model.custom_params = {**algorithm_config.get("alg_params", {}), "device": "cpu"}
    config = vars(model._build_cfg(model.custom_params))
    return _safe(config)


def identity_for_ambi_checkpoint(checkpoint, resolved, protocol, seeds, code, *, path,
                                 inventory_path=None, source_run=None, env=None,
                                 evaluation_controller=None, action_rule=None):
    """Preflight a shared run before any model load or environment decisions.

    ``path`` identifies the future result manifest (or the checkpoint path when
    using an explicit inventory). ``checkpoint`` uses BenchmarkBundle's shape.
    """
    step = checkpoint["metadata"]["checkpoint"]["step"]
    cp = {**checkpoint, **_checkpoint(step, checkpoint["sha256"])}
    source, _, _ = _source(cp, path, checkpoint_inventory=inventory_path, source_run=source_run)
    config = resolved_checkpoint_config(checkpoint, resolved, env=env)
    algorithm = resolved["algorithm_config"]["alg"]
    if evaluation_controller is None and resolved.get("evaluation_controller"):
        from types import SimpleNamespace
        from RL.tdmpc2_core.ambixqc_agent import AMBIXQCAgent
        from RL.tdmpc2_core.xqc_mppi import FrozenXQCMPPIController, resolve_mppi_settings
        configured = resolved["evaluation_controller"]
        _require(configured.get("type") == "mppi", "Unknown preflight evaluation controller")
        agent = AMBIXQCAgent.__new__(AMBIXQCAgent)
        agent.cfg = SimpleNamespace(**config)
        description = FrozenXQCMPPIController.__new__(FrozenXQCMPPIController)
        description.reward_scale = 1.0  # Only the initialization rule enters identity.
        description.discount = agent._get_discount(config["episode_length"])
        evaluation_controller = {"type": "mppi", "protocol": description.protocol,
                                 "settings": resolve_mppi_settings(configured.get("params"),
                                                                   action_dim=config["action_dim"])}
        action_rule = description.protocol["action_rule"]
    result = {"evaluation_controller": evaluation_controller} if evaluation_controller else {}
    action = action_rule or protocol["action_rule"]
    planner = planner_identity(config, result, algorithm, action)
    controller = "prior" if planner["type"] == "prior" else (
        "inner_xqc" if planner["type"] == "xqc" else planner["type"])
    return {"backbone": source, "planner": planner,
            "protocol": _episode_protocol(protocol, seeds, config, action),
            "science": scientific_identity(algorithm, controller, code.get("commit"),
                                             code.get("dirty"), code.get("source_sha256"))}


def _native_planner(settings, planner_result, controller):
    action = "tanh_mean" if controller == "policy_prior" else "weighted_elite_no_execution_noise"
    if controller == "policy_prior":
        return {"type": "prior", "action_rule": action}
    return {"type": "mppi", "backend": "native_tdmpc2", "action_rule": action,
            "settings": {**{key: value for key, value in planner_result.items()
                            if key not in {"configured_iterations", "model_transitions_per_action"}},
                         **{key: settings[key] for key in ("min_std", "max_std", "temperature",
                            "discount", "discount_denom", "discount_min", "discount_max", "episodic")
                            if key in settings}},
            "semantics": {"terminal_value_source": "online_q_mean_pair", "terminal_action": "prior_sample",
                          "warm_start": "shift_previous_mean_within_episode", "std_reset": "max_std"}}


def descriptive_label(identity, selector=None):
    """Readable curve name with no checkpoint, cluster, or publication identity."""
    algorithm = identity.get("science", {}).get("algorithm", "")
    family = "XQC" if "XQC" in algorithm else "AMBI" if "AMBI" in algorithm else "TD-MPC2"
    source = identity["backbone"].rsplit("/", 1)[-1]
    if source.startswith("axqc-prior-"):
        source = source.removeprefix("axqc-prior-")
    planner = identity["planner"]
    prefix = f"{family} {source} | "
    if planner["type"] == "prior":
        return prefix + "Policy prior"
    settings = planner.get("settings", {})
    if planner["type"] == "mppi":
        horizon = settings.get("horizon", settings.get("planning_horizon", settings.get("inner_rollout_horizon")))
        title = (f"MPPI H{horizon} N{settings.get('num_samples', settings.get('inner_mppi_num_samples'))} "
                 f"E{settings.get('num_elites', settings.get('inner_mppi_num_elites'))} "
                 f"pi{settings.get('num_pi_trajs', settings.get('inner_mppi_num_pi_trajs'))} "
                 f"J{settings.get('effective_iterations', settings.get('inner_mppi_iterations'))}")
        if planner.get("backend") == "tdmpc2_mppi_over_frozen_xqc":
            title += " | online XQC Q × frozen scale"
        elif planner.get("backend") == "native_tdmpc2":
            title += " | online TD-MPC2 Q"
        return prefix + title
    rounds = settings.get("inner_rounds")
    budgets = []
    for component, tag in (("critic", "C"), ("actor", "A"), ("temperature", "T")):
        amount = settings.get(f"inner_{component}_updates_per_action")
        if isinstance(amount, (int, float)) and rounds and amount % rounds == 0:
            amount = int(amount / rounds)
        else:
            amount = settings.get(f"inner_{component}_updates_per_round", amount)
        budgets.append(f"{tag}{amount if amount is not None else '?'}")
    if settings.get("inner_terminal_bootstrap") == "outer":
        bootstrap = "outer terminal Q"
    elif str(settings.get("inner_bootstrap_source", "inner")).startswith("outer"):
        bootstrap = "outer Q throughout"
    else:
        bootstrap = "inner Q"
    title = f"{planner['type'].upper()} {'/'.join(budgets)} {bootstrap}"
    title += f" J{rounds}/N{settings.get('inner_rollouts_per_round')}/H{settings.get('inner_rollout_horizon')}"
    if settings.get("inner_steps_per_update") is not None:
        title += f" interval{settings['inner_steps_per_update']}"
    if settings.get("inner_actor_lr") is not None:
        title += f" actorLR{settings['inner_actor_lr']:g}"
    return prefix + title


def identity_for_tdmpc2_checkpoint(checkpoint, metadata, controller, protocol, code, *, path,
                                   inventory_path=None, source_run=None):
    """Native TD-MPC2 preflight using its sidecar and requested episode protocol.

    ``protocol`` supplies environment_seeds, max_steps, and controller_seed;
    controller is policy_prior or native_mppi. No learner is constructed.
    """
    _require(controller in {"policy_prior", "native_mppi"}, "Unknown native controller")
    cp = {**checkpoint, "metadata": metadata,
          **_checkpoint(metadata["checkpoint"]["step"], checkpoint["sha256"])}
    source, _, _ = _source(cp, path, checkpoint_inventory=inventory_path, source_run=source_run)
    settings = metadata["trial_run_params"]["alg_params"]
    observation = metadata["trial_run_params"]["resolved_runtime"]["observation"]
    planner_result = {"effective_iterations": settings["iterations"] + (2 if observation["action_dim"] >= 20 else 0),
                      "num_samples": settings["num_samples"], "num_elites": settings["num_elites"],
                      "num_pi_trajs": settings["num_pi_trajs"], "planning_horizon": settings["outer_planning_horizon"]}
    planner = _native_planner(settings, planner_result, controller)
    normalized_protocol = {"mode": "episodes", "environment": {
        "id": metadata["trial_run_params"]["env"], "params": metadata["experiment_params"]["env_params"]},
        "environment_seeds": sorted(protocol["environment_seeds"]), "max_steps": protocol["max_steps"],
        "controller_seed": protocol["controller_seed"],
        "seed_scheme": "independent fixed namespaced stream per environment seed",
        "action_rule": planner["action_rule"]}
    return {"backbone": source, "planner": planner, "protocol": normalized_protocol,
            "science": scientific_identity("TDMPC2/TDMPC2Baseline", controller,
                                            code.get("commit", code.get("code_sha")))}


def _episodes(episodes, seeds, max_steps):
    _require(bool(episodes), "No completed episode measurements")
    _require(len(set(seeds)) == len(seeds), "Duplicate environment seeds")
    _require(sorted(ep.get("seed") for ep in episodes) == sorted(seeds),
             "Missing, duplicate, or unexpected episode seeds")
    normalized = []
    for episode in episodes:
        _require(episode.get("status", "complete") == "complete", "Incomplete episode")
        length = episode.get("length")
        _require(isinstance(length, int) and 0 < length <= max_steps, "Invalid episode length")
        _require(episode.get("terminated") or episode.get("truncated") or
                 episode.get("truncated_by_evaluator") or episode.get("capped"),
                 "Episode has no completion flag")
        value = copy.deepcopy(episode)
        value["return"] = _number(value.get("return"), "episode return")
        normalized.append(value)
    return sorted(normalized, key=lambda ep: ep["seed"])


def _metrics(episodes):
    returns = [episode["return"] for episode in episodes]
    metrics = {"eval/return_mean": statistics.mean(returns),
               "eval/return_sample_std": statistics.stdev(returns) if len(returns) > 1 else None,
               "eval/episodes": len(returns), "eval/frozen_state_unchanged": True,
               "work/environment_decisions": sum(ep["length"] for ep in episodes)}
    times = [ep.get("control_seconds") for ep in episodes]
    if all(value is not None for value in times):
        metrics["runtime/control_seconds"] = sum(_number(value, "control seconds") for value in times)
        metrics["runtime/control_seconds_per_decision"] = (metrics["runtime/control_seconds"] /
                                                           metrics["work/environment_decisions"])
    if all(ep.get("evaluation_seconds") is not None for ep in episodes):
        metrics["runtime/evaluation_seconds"] = sum(ep["evaluation_seconds"] for ep in episodes)
    gains = [ep.get("paired_return_delta") for ep in episodes]
    if all(value is not None for value in gains):
        gains = [_number(value, "paired gain") for value in gains]
        metrics["eval/paired_gain_mean"] = statistics.mean(gains)
        metrics["eval/paired_gain_sample_std"] = statistics.stdev(gains) if len(gains) > 1 else None
        metrics["eval/paired_episodes"] = len(gains)
    keys = set().union(*(ep.get("model_metrics", ep.get("inner_metrics_mean", {})) for ep in episodes))
    for key in sorted(keys):
        weighted = [(ep.get("model_metrics", ep.get("inner_metrics_mean", {})).get(key), ep["length"])
                    for ep in episodes]
        available = [(value, length) for value, length in weighted
                     if isinstance(value, (float, int)) and math.isfinite(value)]
        if available:
            metrics[f"diagnostics/{key}_mean"] = (sum(value * length for value, length in available) /
                                                   sum(length for _, length in available))
        if any(value is not None and isinstance(value, (float, int)) and not math.isfinite(value)
               for value, _ in weighted):
            metrics[f"diagnostics/{key}_nonfinite"] = {"nonfinite": "nan"}
    for component in ("actor", "critic", "temperature"):
        counts = [ep.get("actual_optimizer_steps", {}).get(component) for ep in episodes]
        if all(value is not None for value in counts):
            metrics[f"work/{component}_updates"] = sum(counts)
        else:
            key = f"inner_{component}_optimizer_steps"
            values = [(ep.get("model_metrics", {}).get(key), ep["length"]) for ep in episodes]
            if all(value is not None for value, _ in values):
                metrics[f"work/{component}_updates"] = sum(value * length for value, length in values)
    for metric, candidates in {"model_steps": ("inner_realized_model_steps", "inner_model_steps"),
                               "policy_evaluations": ("inner_policy_evaluations",),
                               "q_evaluations": ("inner_q_evaluations",)}.items():
        for key in candidates:
            values = [(ep.get("model_metrics", {}).get(key), ep["length"]) for ep in episodes]
            if all(value is not None for value, _ in values):
                metrics[f"work/{metric}"] = sum(value * length for value, length in values)
                break
    return metrics


def _record(identity, checkpoint, episodes, artifacts, provenance, path, label, selector, controller):
    science_episodes = [{key: ep[key] for key in ("seed", "solver_seed", "return", "length",
                        "terminated", "truncated", "truncated_by_evaluator", "capped",
                        "paired_return_delta") if key in ep} for ep in episodes]
    return _safe({"identity": identity, "checkpoint": checkpoint, "metrics": _metrics(episodes),
                  "episodes": episodes, "artifact_files": artifacts, "provenance": provenance,
                  "source_result_path": str(Path(path).resolve()), "label": label,
                  "selector": selector, "controller": controller,
                  "record_id": _hash({"identity": identity, "checkpoint": checkpoint,
                                      "episodes": science_episodes})})


def normalize_bundle(path, *, checkpoint_inventory=None, source_run=None):
    """Normalize completed AMBI/SAC or XQC episode runs in one bundle."""
    path = Path(path).resolve()
    if path.is_dir():
        path /= "manifest.json"
    manifest = _read(path)
    _require(manifest.get("schema_version") == 1 and isinstance(manifest.get("runs"), list),
             "Unsupported benchmark bundle")
    cp = manifest.get("checkpoint", {})
    step = cp.get("metadata", {}).get("checkpoint", {}).get("step")
    checkpoint = _checkpoint(step, cp.get("sha256"))
    source, evidence, extra_files = _source({**cp, **checkpoint}, path,
                                           checkpoint_inventory=checkpoint_inventory,
                                           source_run=source_run)
    code = manifest.get("code", {})
    output = []
    for run in manifest["runs"]:
        result = run.get("result", {})
        if run.get("kind") == "bank" or (run.get("roots") and not run.get("episodes")):
            continue
        if not result:
            continue  # Running/failed scientific solves cannot become curve points.
        _require(result.get("outer_state_unchanged") is True and
                 isinstance(result.get("outer_updates_before"), int) and
                 result.get("outer_updates_before") == result.get("outer_updates_after"),
                 "Frozen outer-state checks failed or missing")
        config = run.get("resolved_config") or result.get("resolved_config")
        _require(isinstance(config, dict) and config, "Missing actual resolved evaluation configuration")
        algorithm = run.get("config", {}).get("alg") or cp.get("metadata", {}).get(
            "trial_run_params", {}).get("alg")
        protocol = copy.deepcopy(manifest.get("protocol", {}))
        action = result.get("action_rule", protocol.get("action_rule"))
        _require(bool(action), "Missing executed action rule")
        seeds = result.get("environment_seeds")
        _require(isinstance(seeds, list) and seeds, "Missing requested environment seeds")
        protocol = _episode_protocol(protocol, seeds, config, action)
        maximum = protocol["max_steps"]
        _require(isinstance(maximum, int) and maximum > 0, "Missing episode decision limit")
        episodes = _episodes(run.get("episodes") or result.get("episodes", []), seeds, maximum)
        if result.get("episodes"):
            returned = {ep["seed"]: ep for ep in result["episodes"]}
            _require(set(returned) == set(seeds), "Returned episode seeds differ from bundle")
            for episode in episodes:
                _require(all(episode.get(key) == returned[episode["seed"]].get(key)
                             for key in ("return", "length", "terminated", "truncated")),
                         "Returned scientific episodes differ from bundle")
        planner = planner_identity(config, result, algorithm, action)
        controller = "prior" if planner["type"] == "prior" else (
            "inner_xqc" if planner["type"] == "xqc" else planner["type"])
        identity = {"backbone": source, "planner": planner, "protocol": protocol,
                    "science": scientific_identity(algorithm, controller, code.get("commit"),
                                                     code.get("dirty"), code.get("source_sha256"))}
        artifacts = {"manifest.json": str(path), **extra_files}
        missing = []
        for name in run.get("trace_files", []):
            trace = (path.parent / name).resolve()
            _require(trace.is_relative_to(path.parent), "Trace path escapes bundle")
            if trace.is_file():
                artifacts[name] = str(trace)
            else:
                missing.append(name)
        provenance = {**evidence, "code": code, "legacy_evaluation_id": manifest.get("evaluation_id"),
                      "legacy_wandb_path": run.get("wandb_path"), "selector": run.get("selector"),
                      "scientific_status": "complete", "legacy_bundle_status": manifest.get("status"),
                      "legacy_run_status": run.get("status"), "missing_artifact_files": missing,
                      "resolved_config": config, "checkpoint_metadata": cp.get("metadata"),
                      "reference": manifest.get("reference"), "source_result_sha256": _file_hash(path)}
        label = descriptive_label(identity, run.get("selector"))
        record = _record(identity, checkpoint, episodes, artifacts, provenance, path, label,
                         run.get("selector", controller), controller)
        for timing in ("initialization_seconds", "warmup_including_compile_seconds", "serialization_seconds"):
            if timing in run:
                record["metrics"][f"runtime/{timing}"] = _safe(run[timing])
        output.append(record)
    return output


def normalize_tdmpc2(path, *, checkpoint_inventory=None, source_run=None):
    """Split a paired native TD-MPC2 result into prior and MPPI records."""
    path = Path(path).resolve()
    result = _read(path)
    _require(result.get("schema_version") == 1 and result.get("algorithm") == "TDMPC2/TDMPC2Baseline",
             "Unsupported TD-MPC2 evaluation")
    checkpoint = _checkpoint(result.get("checkpoint_metadata", {}).get("step"),
                             result.get("checkpoint_sha256"))
    source, evidence, artifacts = _source(checkpoint, path, checkpoint_inventory=checkpoint_inventory,
                                          source_run=source_run)
    provenance = evidence["launch_provenance"]
    metadata_path = path.parent / "checkpoint.metadata.json"
    if not metadata_path.is_file():
        metadata_path = Path(result.get("configuration_source", ""))
    _require(metadata_path.is_file(), "Checkpoint settings sidecar is unavailable")
    metadata = _read(metadata_path)
    _require(metadata.get("checkpoint", {}).get("step") == checkpoint["step"], "Sidecar step mismatch")
    if provenance.get("metadata_sha256"):
        _require(_file_hash(metadata_path) == provenance["metadata_sha256"], "Sidecar hash mismatch")
    frozen = result.get("frozen_state", {})
    _require(frozen.get("unchanged") is True and bool(frozen.get("model_digest_before")) and
             isinstance(frozen.get("num_updates_before"), int) and frozen.get("model_digest_before") ==
             frozen.get("model_digest_after") and frozen.get("num_updates_before") ==
             frozen.get("num_updates_after"), "Frozen TD-MPC2 state checks failed or missing")
    settings = metadata["trial_run_params"]["alg_params"]
    planner_result = result.get("planner", {})
    for result_key, setting_key in (("configured_iterations", "iterations"), ("num_samples", "num_samples"),
                                   ("num_elites", "num_elites"), ("num_pi_trajs", "num_pi_trajs"),
                                   ("planning_horizon", "outer_planning_horizon")):
        _require(planner_result.get(result_key) == settings.get(setting_key),
                 f"Actual MPPI {result_key} differs from checkpoint settings")
    action_dim = result.get("resolved_runtime", {}).get("observation", {}).get("action_dim")
    _require(isinstance(action_dim, int), "Missing native MPPI action dimension")
    effective = settings["iterations"] + (2 if action_dim >= 20 else 0)
    _require(planner_result.get("effective_iterations") == effective, "Incorrect effective MPPI iterations")
    transitions = (effective * settings["num_samples"] * settings["outer_planning_horizon"] +
                   settings["num_pi_trajs"] * (settings["outer_planning_horizon"] - 1))
    _require(planner_result.get("model_transitions_per_action") == transitions, "Incorrect MPPI model-step count")
    protocol = result.get("protocol", {})
    pairs = result.get("episodes", [])
    seeds = list(range(protocol["environment_seed_first"], protocol["environment_seed_last"] + 1))
    _require(len(pairs) == len(seeds), "Missing paired episodes")
    artifacts.update({"paired.json": str(path), "checkpoint.metadata.json": str(metadata_path.resolve())})
    base_protocol = {"mode": "episodes", "environment": {"id": result["environment"],
                     "params": metadata["experiment_params"]["env_params"]},
                     "environment_seeds": seeds, "max_steps": protocol["max_steps"],
                     "controller_seed": protocol["controller_seed_base"], "seed_scheme": protocol["planner_rng"]}
    output = []
    for name, controller in (("policy_prior_mean", "policy_prior"), ("native_mppi", "native_mppi")):
        episodes = []
        for pair in pairs:
            arm = pair[name]
            _require(arm.get("controller") == name, "Mislabeled TD-MPC2 controller")
            rows = arm.get("steps", [])
            _require(len(rows) == arm.get("length"), "Incomplete TD-MPC2 decision records")
            _require(math.isclose(sum(_number(row["reward"], "reward") for row in rows),
                                  arm["return"], rel_tol=1e-10, abs_tol=1e-8), "Return differs from rewards")
            episode = {"seed": pair["environment_seed"], "solver_seed": arm["controller_seed"],
                       "return": arm["return"], "length": arm["length"], "terminated": arm.get("terminated"),
                       "truncated": arm.get("truncated"), "capped": arm.get("capped"),
                       "control_seconds": arm.get("control_seconds"),
                       "evaluation_seconds": arm["seconds"], "model_metrics": {}}
            if name == "native_mppi":
                episode["paired_return_delta"] = arm["return"] - pair["policy_prior_mean"]["return"]
                episode["model_metrics"]["inner_model_steps"] = result["planner"]["model_transitions_per_action"]
            episode["actual_optimizer_steps"] = dict.fromkeys(("actor", "critic", "temperature"), 0)
            for namespace in ("planner", "predicted_action_gain"):
                keys = set().union(*(row.get(namespace, {}) for row in rows))
                for key in keys:
                    values = [row.get(namespace, {}).get(key) for row in rows]
                    finite = [value for value in values if isinstance(value, (float, int)) and math.isfinite(value)]
                    if finite:
                        episode["model_metrics"][key] = statistics.mean(finite)
            episodes.append(episode)
        episodes = _episodes(episodes, seeds, protocol["max_steps"])
        planner = _native_planner(settings, result["planner"], controller)
        identity = {"backbone": source, "planner": planner,
                    "protocol": {**base_protocol, "action_rule": planner["action_rule"]},
                    "science": scientific_identity(result["algorithm"], controller, provenance.get("code_sha"))}
        record = _record(identity, checkpoint, episodes, artifacts,
                         {**evidence, "code": provenance, "checkpoint_metadata": metadata,
                          "scientific_status": "complete", "source_result_sha256": _file_hash(path),
                          "metric_definitions": {
                              "runtime/control_seconds": "Sum of timed model.predict calls; absent in legacy TD-MPC2 artifacts.",
                              "runtime/evaluation_seconds": "Episode wall time including control, environment stepping, and diagnostics."}},
                         path, descriptive_label(identity),
                         controller, controller)
        output.append(record)
    return output


def load_records(path, *, checkpoint_inventory=None, source_run=None, inventory_path=None):
    """Detect a portable bundle or paired TD-MPC2 JSON and normalize it."""
    if checkpoint_inventory is not None and inventory_path is not None:
        raise ValueError("Specify checkpoint_inventory only once")
    checkpoint_inventory = checkpoint_inventory or inventory_path
    path = Path(path)
    if path.is_dir():
        path = path / ("manifest.json" if (path / "manifest.json").is_file() else "paired.json")
    data = _read(path)
    adapter = normalize_bundle if "runs" in data else normalize_tdmpc2
    return adapter(path, checkpoint_inventory=checkpoint_inventory, source_run=source_run)
