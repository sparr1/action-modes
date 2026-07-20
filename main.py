import argparse
import copy
import json
import math
import os
import random
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # Allows non-torch utility usage to keep importing this file.
    torch = None

import domains  # noqa: F401  # registers custom environments through import side effects
# from RL.alg import *
#from RL.baselines import Baseline, TrajectoryLoggerCallback
from utils.core import (
    SUPPORTED_LOG_SETTINGS,
    SUPPORTED_LOG_TYPES,
    build_env,
    initialize_alg,
)
from utils.checkpointing import (
    resolve_checkpoint_config,
    supports_composable_checkpointing,
)
from utils.stats import handle_trial
from utils.utils import datetime_stamp
from log import TrainingLogger, AMBITrainingLogger
#from modes.tasks import *
# from domains.AntPlane import *
# from domains.mpqdn_goal_domain import *
# from domains.mpqdn_platform_domain import *
# from domains.mpqdn_wrappers import *
# import gymnasium_goal
# import gymnasium_platform

# from utils.utils import *


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def seed_env_spaces(env, seed):
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


_SEEDED_LEARN_RESET_ALGORITHMS = {
    "TDMPC2/TDMPC2Baseline",
    "AMBITDMPC2/AMBITDMPC2",
}


def _learn_resets_env_with_seed(alg_path):
    """Whether learn() performs the authoritative first reset with its seed.

    Only algorithms whose training loop explicitly calls ``reset(seed=seed)``
    are listed. Legacy algorithms retain the historical pre-initialization
    reset because some of them subsequently reset without a seed.
    """
    return alg_path in _SEEDED_LEARN_RESET_ALGORITHMS


def _remove_saved_checkpoint(checkpoint_path):
    """Remove one superseded cross-trial save and its matching sidecar only."""

    if not checkpoint_path:
        return
    checkpoint = Path(checkpoint_path)
    for artifact in (checkpoint, Path(f"{checkpoint}.metadata.json")):
        try:
            artifact.unlink()
        except FileNotFoundError:
            pass


_RUNTIME_CONFIG_FIELDS = (
    "train_unroll_horizon",
    "outer_planning_horizon",
    "inner_rollout_horizon",
    "temporal_loss_normalization",
    "temporal_loss_reference_horizon",
    "rho",
    "model_size",
    "num_q",
    "num_bins",
    "vmin",
    "vmax",
    "q_pair_size",
    "q_target_reduction",
    "q_actor_reduction",
    "outer_q_target_reduction",
    "outer_q_actor_reduction",
    "inner_q_target_reduction",
    "inner_q_actor_reduction",
    "compile",
    "compile_strict",
    "inner_operator",
    "inner_schedule_mode",
    "inner_rounds",
    "inner_rollouts_per_round",
    "inner_updates_per_round",
    "inner_nominal_updates_per_round",
    "inner_batch_size",
    "inner_replay_capacity",
    "inner_replay_sampling",
    "inner_replay_scope",
    "inner_model_step_budget",
    "inner_expected_update_slots",
)


def _json_safe_metadata(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_metadata(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    return str(value)


def _resolved_runtime_metadata(model, *, trial_run_params):
    """Return the resolved, portable settings needed to interpret one run."""

    cfg = getattr(model, "cfg", None)
    resolved = {
        key: getattr(cfg, key)
        for key in _RUNTIME_CONFIG_FIELDS
        if cfg is not None and hasattr(cfg, key)
    }

    critic_signature = None
    agent = getattr(model, "agent", None)
    for candidate in (agent, getattr(agent, "model", None)):
        signature = getattr(candidate, "critic_signature", None)
        if isinstance(signature, Mapping):
            critic_signature = dict(signature)
            break
    if critic_signature is None:
        critic_keys = (
            "num_q",
            "num_bins",
            "vmin",
            "vmax",
            "q_pair_size",
            "q_target_reduction",
            "q_actor_reduction",
        )
        inferred = {key: resolved[key] for key in critic_keys if key in resolved}
        critic_signature = inferred or None
    elif critic_signature is not None:
        for key in (
            "q_pair_size",
            "q_target_reduction",
            "q_actor_reduction",
            "outer_q_target_reduction",
            "outer_q_actor_reduction",
            "inner_q_target_reduction",
            "inner_q_actor_reduction",
        ):
            if key in resolved:
                critic_signature.setdefault(key, resolved[key])

    if (
        trial_run_params.get("alg") == "TDMPC2/TDMPC2Baseline"
        and critic_signature is not None
    ):
        critic_signature.setdefault("q_pair_size", 2)
        critic_signature.setdefault("bellman_target_reduction", "min_pair")
        critic_signature.setdefault("actor_reduction", "mean_pair")
        critic_signature.setdefault("planner_terminal_reduction", "mean_pair")

    horizons = {
        key: int(resolved[key])
        for key in (
            "train_unroll_horizon",
            "outer_planning_horizon",
            "inner_rollout_horizon",
        )
        if key in resolved
    }
    temporal = {
        key: resolved[key]
        for key in (
            "temporal_loss_normalization",
            "temporal_loss_reference_horizon",
            "rho",
        )
        if key in resolved
    }
    compile_metadata = {
        "enabled": bool(resolved.get("compile", False)),
        "strict": bool(resolved.get("compile_strict", False)),
    }

    inner_keys = (
        "inner_operator",
        "inner_schedule_mode",
        "inner_rounds",
        "inner_rollouts_per_round",
        "inner_updates_per_round",
        "inner_nominal_updates_per_round",
        "inner_batch_size",
        "inner_replay_capacity",
        "inner_replay_sampling",
        "inner_replay_scope",
        "inner_model_step_budget",
        "inner_expected_update_slots",
    )
    inner = {key: resolved[key] for key in inner_keys if key in resolved}
    if {
        "inner_rounds",
        "inner_rollouts_per_round",
        "inner_rollout_horizon",
    }.issubset(resolved):
        rounds = int(resolved["inner_rounds"])
        rollouts = int(resolved["inner_rollouts_per_round"])
        horizon = int(resolved["inner_rollout_horizon"])
        inner.update(
            branches_per_action=rounds * rollouts,
            transitions_per_round=rollouts * horizon,
            transitions_per_action=rounds * rollouts * horizon,
        )
    if "inner_expected_update_slots" in resolved and "inner_batch_size" in resolved:
        inner["replay_rows_drawn_per_action"] = int(
            resolved["inner_expected_update_slots"] * resolved["inner_batch_size"]
        )

    metadata = {
        "schema_version": 1,
        "algorithm": trial_run_params.get("alg"),
        "seed": int(trial_run_params["seed"]),
        "horizons": horizons,
        "temporal_loss": temporal,
        "critic": critic_signature,
        "compilation": compile_metadata,
        "inner_budget": inner,
    }
    return _json_safe_metadata(metadata)


def main():
    parser = argparse.ArgumentParser(description="action mode learning experiments")
    parser.add_argument('-r', '--run', help='config file for a run', required=True,)
    parser.add_argument('--alg-dir', help='location of alg configs', default = os.path.join("configs","algs", ""))
    parser.add_argument('--log-dir', help='desired location for logging', default = os.path.join(".","logs", ""))
    parser.add_argument('--num-runs', help='number of consecutive trials to run', default = -1, type = int)
    parser.add_argument('--alg-index', help='which algorithm to start running first',default = 0, type=int)
    parser.add_argument('--trial-index', help='which trial index to start from', default = 0, type = int)
    args = vars(parser.parse_args())
    print(args)
    config, alg_dir, log_dir, num_runs, alg_ind, trial_ind = args['run'], args['alg_dir'], args['log_dir'], args['num_runs'], args['alg_index'], args['trial_index']


    with open(config) as f:
        experiment_params = json.load(f)
        experiment_name = config.split('/')[-1][:-5] #cut out everything before the / and the extension .json
    # print(experiment_params)

    #TODO: make a default configuration so I can cut all this code!
    if "logs" in experiment_params:
        log_setting = experiment_params["logs"]
    else:
        log_setting = "warn" #just manually enforcing an annoying default because I love my users and I don't want them deleting their logs or causing clutter by default :)

    if log_setting not in SUPPORTED_LOG_SETTINGS:
        raise Exception("unsupported logging setting. Try none, overwrite, warn, overwrite-safe, or timestamp.")

    if "save_trials" in experiment_params:
        save_trials_setting = experiment_params["save_trials"]
    else:
        save_trials_setting = "first"
    if log_setting == "none" and save_trials_setting == "best":
        raise ValueError(
            "save_trials='best' requires trajectory logging so trials can be "
            "scored; choose logs other than 'none'."
        )

    if "log_info" in experiment_params:
        log_info_setting = experiment_params["log_info"]
    else:
        log_info_setting = True #backwards compatibility

    if "log_type" in experiment_params:
        log_type_setting = experiment_params["log_type"]
    else:
        log_type_setting = "detailed" #backwards compatibility
    if log_type_setting not in SUPPORTED_LOG_TYPES:
        raise ValueError(
            f"unsupported log_type {log_type_setting!r}; expected one of {SUPPORTED_LOG_TYPES}."
        )

    num_algs = len(experiment_params["configs"])
    runtime_params = [dict() for _ in range(num_algs)]
    print("Experiment testing")
    for i, alg_config in enumerate(experiment_params["configs"]):
        print("---------")
        print(alg_config)
        runtime_params[i]["name"] = alg_config
        print("verifying config exists and is proper:")
        with open(os.path.join(alg_dir,alg_config+".json")) as f:
            run_default_params = json.load(f)
        runtime_params[i].update(run_default_params.copy())
        print("config found. Replacing settings based on experiment configs.")
        for override_key, override_value in experiment_params.get("overrides_alg", {}).items():
            if override_key in run_default_params:
                print("setting of",override_key,"currently at",run_default_params[override_key], "overriden to", override_value,".")
            else:
                print("override key of", override_key, "not found in run params, setting it to value", override_value, "anyway.")
            runtime_params[i][override_key] = override_value
        print("full runtime alg configuration settings:")
        print(runtime_params[i])
        print("----------")

    # Resolve checkpoint settings after all per-algorithm overrides have been
    # applied. A key in the algorithm config wins over the experiment value,
    # including an explicit null cadence.
    checkpoint_configs = [
        resolve_checkpoint_config(run_params, experiment_params)
        for run_params in runtime_params
    ]
    checkpointing_requested = any(config.enabled for config in checkpoint_configs)

    # seed = experiment_params["seed"]

    experiment_log_dir = None
    model_save_dir = None
    if log_setting != "none":
        experiment_log_dir = os.path.join(log_dir, f'{experiment_name}', '')
        skip = False
        if log_setting == "warn":
            if os.path.exists(experiment_log_dir):
                print("WARNING: experiment has been run before. Check the files, and delete or change setting from \'warn\'.")
                quit()
        elif log_setting == "overwrite":
            if(os.path.exists(experiment_log_dir)):
                shutil.rmtree(experiment_log_dir) #delete the old experiment! be careful with this setting.
        elif log_setting =="timestamp":
            experiment_log_dir=os.path.join(experiment_log_dir[:-1]+'_'+datetime_stamp(),"") #is there a way to do this with os path?
        elif log_setting == "overwrite-safe":
            print("this run started at", datetime_stamp())
            if os.path.exists(experiment_log_dir):
                print("Experiment folder already exists. Trials will only proceed if not run before.")
                skip = True

        if not skip:
            os.makedirs(experiment_log_dir, exist_ok=True)
            with open(experiment_log_dir+"settings.json", "w") as f:
                json.dump(experiment_params, f, indent=2) #put the experiment json params next to the data which resulted from a run with those parameters
        else:
            try:
                with open(experiment_log_dir+"settings.json", "r") as f:
                    existing_settings = json.load(f)
                    print("Found existing settings:", existing_settings)
            except (OSError, json.JSONDecodeError):
                print("WARNING: experiment folder existed, but there was an issue reading the settings file. Proceeding with caution under the current settings.")

        if save_trials_setting not in (None, "none") or checkpointing_requested:
            model_save_dir = os.path.join(experiment_log_dir, "models", "")
            os.makedirs(model_save_dir, exist_ok=True)
    elif save_trials_setting not in (None, "none") or checkpointing_requested:
        # ``logs: none`` disables trajectory logging, not requested model
        # artifacts. Use a collision-safe run folder so numbered checkpoints
        # and fixed aliases from earlier runs cannot be overwritten silently.
        experiment_log_dir = os.path.join(
            log_dir,
            f"{experiment_name}_{datetime_stamp()}",
            "",
        )
        suffix = 1
        base_experiment_log_dir = experiment_log_dir
        while os.path.exists(experiment_log_dir):
            experiment_log_dir = os.path.join(
                f"{base_experiment_log_dir.rstrip(os.sep)}_{suffix}", ""
            )
            suffix += 1
        os.makedirs(experiment_log_dir, exist_ok=False)
        with open(os.path.join(experiment_log_dir, "settings.json"), "w") as f:
            json.dump(experiment_params, f, indent=2)
        model_save_dir = os.path.join(experiment_log_dir, "models", "")
        os.makedirs(model_save_dir, exist_ok=True)
        print("Trajectory logging disabled; model artifacts will be saved to", model_save_dir)

    ran_so_far = 0

    for i, run_params in enumerate(runtime_params[alg_ind:], start = alg_ind):
        alg_config = run_params["name"]
        checkpoint_config = checkpoint_configs[i]
        print(run_params)

        if save_trials_setting == "best": #this won't take into account old saved models if you're running "best".
            best_score = -math.inf
            best_trial_checkpoint = None

        for t in range(trial_ind, experiment_params["trials"]):
            print()
            if ran_so_far == num_runs:
                print("completed running", num_runs, "trials!")
                quit()

            trial_run_params = copy.deepcopy(run_params)
            trial_seed = int(trial_run_params.get("seed", 0)) + t
            trial_run_params["seed"] = trial_seed
            if "baselines/" in trial_run_params["alg"]:
                trial_run_params.setdefault("alg_params", {})["seed"] = trial_seed

            seed_everything(trial_seed)

            domain = build_env(trial_run_params, experiment_params)

            seed_env_spaces(domain, trial_seed)
            # TD-MPC2 performs the same seeded reset at the start of learn().
            # Avoid a discarded environment transition there while retaining
            # the legacy reset semantics for algorithms that reset unseeded.
            if not _learn_resets_env_with_seed(trial_run_params["alg"]):
                domain.reset(seed=trial_seed)

            model, baseline, alg_name = initialize_alg(trial_run_params["alg"], trial_run_params["alg_params"], domain, full_run_params=trial_run_params, experiment_params=experiment_params)
            print(alg_name, "initialized.")
            trial_run_params["resolved_runtime"] = _resolved_runtime_metadata(
                model, trial_run_params=trial_run_params
            )
            print(
                "Resolved runtime metadata:",
                json.dumps(trial_run_params["resolved_runtime"], sort_keys=True),
            )

            if checkpoint_config.enabled:
                if not supports_composable_checkpointing(model):
                    domain.close()
                    raise ValueError(
                        f"Algorithm {trial_run_params['alg']!r} does not support "
                        "checkpoint_every/save_strat. Supported algorithms are SB3 "
                        "baselines, native SAC, TD-MPC2, and AMBI-TD-MPC2."
                    )
                model.set_checkpointing(
                    save_freq=checkpoint_config.every,
                    save_path=model_save_dir,
                    name_prefix=f'model:{alg_config}_{t}',
                    save_strat=checkpoint_config.strategies,
                    checkpoint_best_window=checkpoint_config.best_window,
                    trial_run_params=trial_run_params,
                    experiment_params=experiment_params,
                )

            training_logger = None
            trial_log_dir = None
            try:
                if log_setting != "none":
                    trial_log_dir = experiment_log_dir + f'{alg_config}_{t}'
                    if os.path.exists(trial_log_dir):
                        print("WARNING: trial log dir alredy existed!")
                        if log_setting == "overwrite-safe":
                            print("quitting, as to continue would risk an overwrite.")
                            quit()
                    else:
                        os.makedirs(trial_log_dir, exist_ok=True)
                    with open(os.path.join(trial_log_dir,"alg_settings.json"), "w") as f:
                        json.dump(trial_run_params, f, indent=2) #put the algorithm parameters next to the data which resulted from a trial using those params
                    is_ambi = "AMBI" in alg_config.upper() or (
                        hasattr(model, "agent") and hasattr(model.agent, "last_inner_rollout_lengths")
                    )
                    logger_class = AMBITrainingLogger if is_ambi else TrainingLogger
                    training_logger = logger_class(log_info=log_info_setting, log_type=log_type_setting)
                    if is_ambi:
                        print("Using AMBI training logger")
                    training_logger.reset()
                    training_logger.set_log_dir(trial_log_dir)
                    model.set_logger(training_logger)

                model.learn(total_timesteps=trial_run_params["total_steps"])

                # Cross-trial saves remain separate from the within-trial
                # checkpoint strategy and therefore run even with logs disabled.
                if t == 0 and save_trials_setting == "first":
                    if training_logger is not None:
                        training_logger.flush()
                    model.save(model_save_dir, f'model:{alg_config}_{t}')
                elif save_trials_setting == "all":
                    if training_logger is not None:
                        training_logger.flush()
                    model.save(model_save_dir, f'model:{alg_config}_{t}')
                elif save_trials_setting == "best":
                    training_logger.flush()
                    _ , trial_contents = handle_trial(trial_log_dir)
                    rewards = trial_contents["rewards"]
                    score = np.average(rewards) if rewards.size > 0 else -math.inf #take a simple average over the whole trial!
                    if not np.isfinite(score):
                        score = -math.inf
                    if score > best_score:
                        new_checkpoint = model.save(model_save_dir, f'model:{alg_config}_{t}')
                        _remove_saved_checkpoint(best_trial_checkpoint)
                        best_score = score
                        best_trial_checkpoint = new_checkpoint

                    with open(os.path.join(model_save_dir,"scores.txt"), "a") as f:
                        f.write(f'{alg_config}_{t}'+ ":" + str(score) + '\n')
            finally:
                if training_logger is not None:
                    # Makes buffered CSV rows and queued detailed trajectories
                    # durable on normal completion and training exceptions.
                    training_logger.close()
                domain.close()
            ran_so_far  += 1

            # print("training count", callback.training_count)
            # print("episode count", callback.episode_count)
            # print("rollout count", callback.rollout_count)
            # print("n_calls", callback.n_calls)
            # print("num_timesteps", callback.num_timesteps)

if __name__ == '__main__':
   main()
