"""Dog transfers preserve the completed gSDE8 learner and resolve Dog shapes."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2
from slurm.ambi_transfer_training import recipe

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task", ["trot", "run"])
def test_dog_recipe_and_resolved_config(task):
    folder = ROOT / "configs/dmcontrol"
    source = json.loads((folder / "algs/ambi_gsde_squashed_real_steps.json").read_text())
    path = folder / f"experiments/ambi_gsde_squashed_real_steps_dog_{task}.json"
    _, manifest, run = recipe(0, path)
    target = deepcopy(run)
    for config in (source, target):
        config.pop("name", None)
        config["alg_params"] = {k: v for k, v in config["alg_params"].items()
                                if not k.startswith("wandb")}
    assert target == source
    original = json.loads((folder / "experiments/ambi_gsde_noise_scopes.json").read_text())
    original["env_params"]["task"] = "dog-" + task
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials",
                "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert original[key] == manifest[key]
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = SimpleNamespace(observation_type="state",
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(223,), dtype=np.float32),
        action_space=gym.spaces.Box(-1., 1., shape=(38,), dtype=np.float32),
        spec=SimpleNamespace(max_episode_steps=500))
    params = {**run["alg_params"], "device": "cpu", "wandb": False}
    algorithm.run_params = {**run, "device": "cpu"}
    algorithm.custom_params = params
    cfg = algorithm._build_cfg(params)
    assert cfg.action_dim == 38 and cfg.obs_shape["state"] == (223,)
    assert cfg.seed == 55 and cfg.steps == 1000000
    assert cfg.actor_distribution == "gsde" and cfg.inner_gsde_noise_scope == "real_steps"
    assert cfg.inner_gsde_resample_steps == cfg.gsde_execution_resample_steps == 8
    assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.ent_coef == cfg.inner_temperature == .0021
    assert cfg.inner_execution_action == cfg.inner_behavior_action == "policy_sample"
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
