"""The H1 screen preserves the selected update dose and critic pairings."""

import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from RL.AMBIXQC import AMBIXQC
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.checkpoint_context import CheckpointContext


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambixqc_humanoid_h1_actor_bn_screen.json"


@pytest.mark.parametrize("arm", ["aux_shared", "aux_detached"])
def test_h1_matrix_resolves_all_eight_cells_with_frozen_actor_bn(arm):
    base = ROOT / f"configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_1m_{arm}.json"
    trial = json.loads(base.read_text())
    context = CheckpointContext(trial, {"env_params": {}}, base)
    matrix = load_preset_matrix(MATRIX)
    selectors = matrix["evaluation"]["default_presets"]
    assert len(selectors) == len(set(selectors)) == 8
    assert matrix["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    assert matrix["evaluation"]["controller_seed"] == 12345
    assert matrix["evaluation"]["max_steps"] == 500
    for selector in selectors:
        preset = resolve_preset(MATRIX, selector, matrix, checkpoint_context=context)
        algorithm = object.__new__(AMBIXQC)
        algorithm.env = SimpleNamespace(
            observation_space=gym.spaces.Box(-np.inf, np.inf, (67,), dtype=np.float32),
            action_space=gym.spaces.Box(-1.0, 1.0, (21,), dtype=np.float32),
            spec=SimpleNamespace(max_episode_steps=500),
        )
        algorithm.run_params = preset["algorithm_config"]
        algorithm.experiment_params = {}
        params = algorithm.run_params["alg_params"]
        algorithm.custom_params = params
        cfg = algorithm._build_cfg({**params, "device": "cpu"})
        rounds = int(selector.rsplit("j", 1)[1])
        assert rounds in (1, 2, 4, 8) and cfg.inner_rounds == rounds
        for key, expected in {
            "inner_actor_bn_mode": "running", "inner_temperature_mode": "auto",
            "inner_temperature_initialization": "inherit_outer",
            "inner_rollout_horizon": 1, "inner_rollouts_per_round": 256,
            "inner_batch_size": 256, "inner_updates_per_round": 3,
            "inner_policy_delay": 3, "inner_update_timing": "round",
            "inner_replay_capacity": 2048, "inner_actor_lr": 5e-5,
            "inner_critic_lr": 5e-5, "inner_terminal_bootstrap": "outer",
            "inner_reward_normalization": "frozen_real_scale",
        }.items():
            assert getattr(cfg, key) == expected, key
        is_return = "return_return" in selector
        source = "aux_return" if is_return else "xqc"
        assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == source
        assert cfg.inner_critic_target == ("reward_only" if is_return else "entropy_augmented")
