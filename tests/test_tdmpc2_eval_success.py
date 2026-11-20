"""Online success evaluation must match the final-step Meta-World protocol."""

import csv
import os

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline
from tests.test_ambi_inner_decoupling import _params as _ambi_params
from tests.test_tdmpc2_online_eval import FakeWandbRun, TwoStepEnv, _tiny_params


class SuccessEnv(TwoStepEnv):
    evaluation_success_key = "success"

    def __init__(self, final_infos):
        super().__init__()
        self.final_infos = final_infos
        self.episode_index = -1

    def reset(self, *, seed=None, options=None):
        self.episode_index += 1
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, _ = super().step(action)
        # Earlier success must not make a terminal failure count as success.
        info = self.final_infos[self.episode_index] if truncated else {"success": 1.0}
        return obs, reward, terminated, truncated, info


def _model(tmp_path, env, model_class=TDMPC2Baseline):
    params = _tiny_params(tmp_path / "eval.csv")
    if model_class is AMBITDMPC2:
        params = {**_ambi_params(), **params}
    model = model_class(
        model_class.__name__, env, params,
        {"seed": 3, "device": "cpu", "env": "test"}, {},
    )
    model._act_agent = lambda *args, **kwargs: torch.zeros(1)
    model._wandb_run = FakeWandbRun()
    return model


@pytest.mark.parametrize("model_class", [TDMPC2Baseline, AMBITDMPC2])
def test_final_step_success_is_averaged_for_both_learners(tmp_path, model_class):
    env = gym.wrappers.TimeLimit(
        SuccessEnv([{"success": 0.0}, {"success": np.bool_(True)}, {"success": 0}]),
        max_episode_steps=2,
    )
    model = _model(tmp_path, env, model_class)

    assert model._evaluate_policy(100) == 2.0

    with (tmp_path / "eval.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["step", "reward", "seed", "success_rate"]
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0] == {
        "step": "100", "reward": "2.0", "seed": "3",
        "success_rate": str(1.0 / 3.0),
    }
    assert model._wandb_run.history[100]["eval/success_rate"] == pytest.approx(1.0 / 3.0)
    assert model._wandb_run.history[100]["eval/episode_reward"] == 2.0
    assert model._wandb_run.history[100]["eval/episodes"] == 3
    assert env.unwrapped.total_env_steps == 6


def test_success_info_without_environment_opt_in_keeps_existing_schema(tmp_path):
    env = SuccessEnv([{"success": 1.0}] * 3)
    env.evaluation_success_key = None
    model = _model(tmp_path, env)

    model._evaluate_policy(100)

    with (tmp_path / "eval.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"step": "100", "reward": "2.0", "seed": "3"}]
    assert "eval/success_rate" not in model._wandb_run.history[100]


@pytest.mark.parametrize(
    "invalid_info, error_type",
    [
        ({}, ValueError),
        (None, ValueError),
        ({"success": float("nan")}, ValueError),
        ({"success": float("inf")}, ValueError),
        ({"success": -1}, ValueError),
        ({"success": 0.5}, ValueError),
        ({"success": 2}, ValueError),
        ({"success": "1"}, TypeError),
        ({"success": np.array([1])}, TypeError),
    ],
)
def test_invalid_terminal_success_rejects_entire_evaluation_before_publication(
    tmp_path, invalid_info, error_type,
):
    env = SuccessEnv([{"success": 1.0}, invalid_info, {"success": 1.0}])
    model = _model(tmp_path, env)

    with pytest.raises(error_type, match="evaluation success"):
        model._evaluate_policy(100)

    assert env.total_env_steps == 4
    assert not (tmp_path / "eval.csv").exists()
    assert not model._wandb_run.history


@pytest.mark.parametrize("success_key", ["", " ", 1, False])
def test_malformed_environment_opt_in_is_rejected(tmp_path, success_key):
    env = SuccessEnv([])
    env.evaluation_success_key = success_key
    with pytest.raises(ValueError, match="evaluation_success_key"):
        _model(tmp_path, env)


def test_resume_csv_retains_episode_column_and_success_rate(tmp_path):
    model = _model(tmp_path, SuccessEnv([{"success": 1.0}] * 3))
    model._resume_enabled = True
    model._episode_idx = 7

    model._evaluate_policy(100)

    with (tmp_path / "eval.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{
        "step": "100", "episode": "7", "reward": "2.0", "seed": "3",
        "success_rate": "1.0",
    }]
    assert model._wandb_run.history[100]["episode/index"] == 7


def test_existing_csv_cannot_be_mixed_with_success_schema(tmp_path):
    path = tmp_path / "eval.csv"
    original = "step,reward,seed\n0,2.0,3\n"
    path.write_text(original)
    model = _model(tmp_path, SuccessEnv([]))

    with pytest.raises(FileExistsError, match="Refusing to mix"):
        model._record_evaluation(100, 2.0, success_rate=0.5)

    assert path.read_text() == original
    assert not model._wandb_run.history


@pytest.mark.parametrize("extras", [{"eval/success_rate": 1.0}, {"probe": float("nan")}])
def test_invalid_metric_extras_cannot_partially_publish_success(tmp_path, extras):
    model = _model(tmp_path, SuccessEnv([]))

    with pytest.raises(ValueError):
        model._record_evaluation(100, 2.0, success_rate=0.5, extras=extras)

    assert not (tmp_path / "eval.csv").exists()
    assert not model._wandb_run.history


@pytest.mark.skipif(
    os.environ.get("AMBI_RUN_REAL_METAWORLD_TESTS") != "1",
    reason="requires the isolated Meta-World runtime; opt in explicitly",
)
@pytest.mark.parametrize("model_class", [TDMPC2Baseline, AMBITDMPC2])
def test_real_metaworld_infers_single_task_learner_configuration(tmp_path, model_class):
    import domains  # noqa: F401
    from utils.core import build_env

    run_params = {"seed": 3, "device": "cpu", "env": "MetaWorld-v0"}
    experiment_params = {"env_params": {"task": "push", "obs": "state"}}
    env = build_env(run_params, experiment_params, render_mode=None)
    params = _tiny_params(tmp_path / "unused.csv")
    if model_class is AMBITDMPC2:
        params = {**_ambi_params(), **params}
    for key in ("episode_length", "seed_steps", "pretrain_steps", "discount"):
        params.pop(key, None)
    model = None
    try:
        # Construct networks and resolve defaults; deliberately never train.
        model = model_class(
            model_class.__name__, env, params, run_params, experiment_params,
        )
        assert model.cfg.obs_shape == {"state": (39,)}
        assert model.cfg.action_dim == 4
        assert model.cfg.episode_length == 100
        assert model.cfg.seed_steps == model.cfg.pretrain_steps == 1000
        assert model.agent.discount == pytest.approx(0.95)
        assert model._evaluation_success_key == "success"
        assert not (tmp_path / "unused.csv").exists()
    finally:
        if model is not None:
            model.close()
        env.close()
