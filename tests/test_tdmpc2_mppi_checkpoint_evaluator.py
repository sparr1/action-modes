import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import evaluate_tdmpc2_mppi_checkpoint as evaluator
from RL.TDMPC2 import TDMPC2Baseline
from render_checkpoint import RenderContext


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_tdmpc2_mppi_mc_25k_hydra.sbatch"


class _FakeWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("sentinel", torch.tensor([3.0]))


class _FakeModel:
    def __init__(self):
        cfg = SimpleNamespace(
            mpc=True,
            iterations=8,
            outer_planning_horizon=3,
            num_samples=512,
            num_elites=64,
            num_pi_trajs=24,
        )
        self.agent = SimpleNamespace(
            cfg=cfg,
            device=torch.device("cpu"),
            model=_FakeWorldModel(),
            num_updates=17,
            last_plan_metrics={},
        )
        self.cfg = cfg
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self):
        self.reset_calls += 1

    def predict(self, observation, *, deterministic, episode_start):
        assert deterministic is True
        assert np.asarray(observation).shape == (1,)
        if self.cfg.mpc:
            self.agent.last_plan_metrics = {
                "planner_value_mean": 2.0,
                "planner_seconds": 0.01,
            }
            return np.array([1.0], dtype=np.float32), None
        self.agent.last_plan_metrics = {}
        return np.array([0.0], dtype=np.float32), None

    @staticmethod
    def _obs_to_tensor(observation):
        return torch.as_tensor(observation, dtype=torch.float32)

    @staticmethod
    def _scale_action(action):
        return np.asarray(action, dtype=np.float32)

    def close(self):
        self.close_calls += 1


class _FakeEnv:
    def __init__(self):
        self.action_space = SimpleNamespace(seed=lambda seed: None)
        self.observation_space = SimpleNamespace(seed=lambda seed: None)
        self.reset_seeds = []
        self.close_calls = 0
        self.steps = 0

    def reset(self, *, seed):
        self.reset_seeds.append(seed)
        self.steps = 0
        return np.array([float(seed)], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        reward = 1.0 + float(np.asarray(action).reshape(-1)[0])
        return (
            np.array([float(self.steps)], dtype=np.float32),
            reward,
            False,
            self.steps == 2,
            {},
        )

    def close(self):
        self.close_calls += 1


class _TinyTDMPCEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_type = "state"
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.spec = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        del action
        return np.zeros(3, dtype=np.float32), 0.0, False, True, {}


def _predicted_gain(*args, **kwargs):
    del args, kwargs
    return {
        "target_q_mppi_mean_all": 4.0,
        "target_q_policy_prior_mean_all": 3.0,
        "target_q_mppi_minus_policy_prior": 1.0,
        "policy_prior_to_mppi_action_l2": 1.0,
        "policy_prior_action_at_mppi_state": [0.0],
        "diagnostic_seconds": 0.0,
    }


def test_paired_checkpoint_evaluation_writes_raw_trajectories_and_restores_state(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint bytes")
    output = tmp_path / "evaluation.json"
    context = RenderContext(
        trial_run_params={
            "alg": "TDMPC2/TDMPC2Baseline",
            "env": "toy",
            "seed": 1,
            "alg_params": {
                "obs": "state",
                "iterations": 6,
            },
            "resolved_runtime": {
                "horizons": {
                    "train_unroll_horizon": 3,
                    "outer_planning_horizon": 3,
                },
                "observation": {"task": "toy"},
            },
        },
        experiment_params={"env_params": {"task": "toy"}},
        source=tmp_path / "metadata.json",
    )
    model = _FakeModel()
    envs = [_FakeEnv(), _FakeEnv(), _FakeEnv()]

    monkeypatch.setattr(evaluator, "resolve_checkpoint_path", lambda path: Path(path))
    monkeypatch.setattr(
        evaluator,
        "resolve_render_context",
        lambda *args, **kwargs: context,
    )
    monkeypatch.setattr(evaluator, "_backend_for", lambda algorithm: "tdmpc2")
    monkeypatch.setattr(
        evaluator,
        "_prepare_run_params",
        lambda *args, **kwargs: (
            copy.deepcopy(context.trial_run_params),
            copy.deepcopy(context.experiment_params),
        ),
    )
    monkeypatch.setattr(evaluator, "build_env", lambda *args, **kwargs: envs.pop(0))
    monkeypatch.setattr(
        evaluator,
        "_initialize_model",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(evaluator, "_predicted_action_gain", _predicted_gain)

    random.seed(701)
    np.random.seed(702)
    torch.manual_seed(703)
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.random.get_rng_state().clone()
    payload = evaluator.evaluate_tdmpc2_mppi_checkpoint(
        checkpoint,
        output=output,
        episodes=2,
        seed=101,
        controller_seed=12345,
        bootstrap_samples=100,
        device="cpu",
    )

    saved = json.loads(output.read_text())
    assert saved == payload
    assert payload["summary"]["policy_prior_return_mean"] == 2.0
    assert payload["summary"]["native_mppi_return_mean"] == 4.0
    assert payload["summary"]["paired_return_delta_mean"] == 2.0
    assert payload["summary"][
        "paired_return_delta_conditional_bootstrap_95_interval"
    ] == [
        2.0,
        2.0,
    ]
    assert payload["planner"] == {
        "configured_iterations": 6,
        "effective_iterations": 8,
        "num_samples": 512,
        "num_elites": 64,
        "num_pi_trajs": 24,
        "planning_horizon": 3,
        "model_transitions_per_action": 12336,
    }
    assert payload["frozen_state"]["unchanged"] is True
    assert payload["trajectory_summary"][1][
        "trajectory_cumulative_return_difference_mean"
    ] == 2.0
    assert payload["trajectory_summary"][1][
        "trajectory_cumulative_return_difference_conditional_pointwise_bootstrap_95_interval"
    ] == [2.0, 2.0]
    assert payload["resolved_runtime"]["horizons"]["train_unroll_horizon"] == 3
    assert [row["environment_seed"] for row in payload["episodes"]] == [101, 102]
    assert payload["episodes"][0]["native_mppi"]["steps"][0]["planner"] == {
        "planner_seconds": 0.01,
        "planner_value_mean": 2.0,
    }
    assert model.cfg.mpc is True
    assert model.close_calls == 1
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    torch.testing.assert_close(
        torch.random.get_rng_state(), torch_state, rtol=0, atol=0
    )


def test_atomic_json_refuses_to_replace_existing_output(tmp_path):
    path = tmp_path / "result.json"
    evaluator._write_json(path, {"first": 1}, overwrite=False)

    with pytest.raises(evaluator.TDMPC2MPPIEvaluationError, match="already exists"):
        evaluator._write_json(path, {"second": 2}, overwrite=False)

    assert json.loads(path.read_text()) == {"first": 1}


def test_failed_evaluation_restores_global_rng_and_closes_resources(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint bytes")
    context = RenderContext(
        trial_run_params={
            "alg": "TDMPC2/TDMPC2Baseline",
            "env": "toy",
            "seed": 1,
            "alg_params": {"obs": "state", "iterations": 6},
        },
        experiment_params={"env_params": {"task": "toy", "obs": "state"}},
        source=tmp_path / "metadata.json",
    )
    model = _FakeModel()
    created_envs = [_FakeEnv(), _FakeEnv(), _FakeEnv()]
    env_queue = list(created_envs)

    monkeypatch.setattr(evaluator, "resolve_checkpoint_path", lambda path: Path(path))
    monkeypatch.setattr(
        evaluator, "resolve_render_context", lambda *args, **kwargs: context
    )
    monkeypatch.setattr(evaluator, "_backend_for", lambda algorithm: "tdmpc2")
    monkeypatch.setattr(
        evaluator,
        "_prepare_run_params",
        lambda *args, **kwargs: (
            copy.deepcopy(context.trial_run_params),
            copy.deepcopy(context.experiment_params),
        ),
    )
    monkeypatch.setattr(
        evaluator, "build_env", lambda *args, **kwargs: env_queue.pop(0)
    )
    monkeypatch.setattr(
        evaluator, "_initialize_model", lambda *args, **kwargs: model
    )

    def fail_after_reseed(*args, **kwargs):
        del args, kwargs
        evaluator._seed_controller(999)
        raise RuntimeError("synthetic rollout failure")

    monkeypatch.setattr(evaluator, "_run_arm", fail_after_reseed)
    random.seed(801)
    np.random.seed(802)
    torch.manual_seed(803)
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="synthetic rollout failure"):
        evaluator.evaluate_tdmpc2_mppi_checkpoint(
            checkpoint,
            output=tmp_path / "never-written.json",
            episodes=1,
            seed=101,
            controller_seed=12345,
            bootstrap_samples=100,
            device="cpu",
        )

    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    torch.testing.assert_close(
        torch.random.get_rng_state(), torch_state, rtol=0, atol=0
    )
    assert model.cfg.mpc is True
    assert model.close_calls == 1
    assert [env.close_calls for env in created_envs] == [1, 1, 1]


def test_bootstrap_resamples_whole_paired_episode_deltas_deterministically():
    values = [-2.0, 0.0, 4.0, 8.0]

    first = evaluator._bootstrap_mean_interval(values, samples=500, seed=7)
    second = evaluator._bootstrap_mean_interval(values, samples=500, seed=7)

    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_environment_declared_rgb_is_rejected_when_algorithm_obs_is_omitted(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint bytes")
    context = RenderContext(
        trial_run_params={
            "alg": "TDMPC2/TDMPC2Baseline",
            "env": "toy",
            "seed": 1,
            "alg_params": {},
        },
        experiment_params={"env_params": {"task": "toy", "obs": "rgb"}},
        source=tmp_path / "metadata.json",
    )
    monkeypatch.setattr(evaluator, "resolve_checkpoint_path", lambda path: Path(path))
    monkeypatch.setattr(
        evaluator, "resolve_render_context", lambda *args, **kwargs: context
    )
    monkeypatch.setattr(evaluator, "_backend_for", lambda algorithm: "tdmpc2")

    with pytest.raises(
        evaluator.TDMPC2MPPIEvaluationError, match="state observations only"
    ):
        evaluator.evaluate_tdmpc2_mppi_checkpoint(
            checkpoint,
            output=tmp_path / "never-written.json",
            episodes=1,
            seed=101,
            device="cpu",
        )


def test_real_tdmpc2_predicted_gain_is_finite_and_preserves_rng():
    env = _TinyTDMPCEnv()
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 32,
        "mlp_dim": 32,
        "latent_dim": 16,
        "num_enc_layers": 2,
        "num_q": 2,
        "simnorm_dim": 8,
        "num_bins": 11,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "train_unroll_horizon": 1,
        "outer_planning_horizon": 1,
        "buffer_size": 1,
        "episode_length": 1,
        "seed_steps": 1,
        "pretrain_steps": 1,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "iterations": 1,
        "num_samples": 4,
        "num_elites": 2,
        "num_pi_trajs": 1,
        "wandb": False,
        "dropout": 0.0,
    }
    model = TDMPC2Baseline(
        "tiny-mppi-eval",
        env,
        params,
        {"seed": 3, "device": "cpu", "env": "toy", "total_steps": 2},
        {},
    )
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    try:
        result = evaluator._predicted_action_gain(
            model,
            np.zeros(3, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
        assert set(result) == {
            "target_q_mppi_mean_all",
            "target_q_policy_prior_mean_all",
            "target_q_mppi_minus_policy_prior",
            "policy_prior_to_mppi_action_l2",
            "policy_prior_action_at_mppi_state",
            "diagnostic_seconds",
        }
        assert all(
            np.isfinite(value)
            for key, value in result.items()
            if key != "policy_prior_action_at_mppi_state"
        )
        assert random.getstate() == python_state
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        torch.testing.assert_close(
            torch.random.get_rng_state(), torch_state, rtol=0, atol=0
        )
    finally:
        model.flush_checkpoints()
        env.close()


def test_hydra_launcher_runs_both_25k_checkpoints_on_separate_l40_gpus():
    contents = LAUNCHER.read_text()

    assert "#SBATCH --nodelist=gpu2301" in contents
    assert "#SBATCH --constraint=l40" in contents
    assert "#SBATCH --gres=gpu:nvidia_l40:1" in contents
    assert "#SBATCH --array=0-1%2" in contents
    assert "#SBATCH --time=08:00:00" in contents
    assert "model:tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_0_25000" in contents
    assert (
        "model:tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5_0_25000"
        in contents
    )
    assert "--episodes 12" in contents
    assert "--seed 101" in contents
    assert "--controller-seed 12345" in contents
    assert "--bootstrap-samples 20000" in contents
    assert "--device cuda" in contents
    assert "git status --porcelain --untracked-files=normal" in contents


def test_parser_defaults_to_twelve_paired_episodes(tmp_path):
    args = evaluator.build_parser().parse_args(
        [str(tmp_path / "checkpoint"), "--output", str(tmp_path / "out.json")]
    )

    assert args.episodes == 12
    assert args.controller_seed == 12345
    assert args.bootstrap_samples == 20000
