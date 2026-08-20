import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main


class _Space:
    def __init__(self):
        self.seeds = []

    def seed(self, seed):
        self.seeds.append(seed)


class _Env:
    def __init__(self):
        self.action_space = _Space()
        self.observation_space = _Space()
        self.reset_seeds = []
        self.closed = False

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        return 0, {}

    def close(self):
        self.closed = True


class _Model:
    def __init__(self, *, supports=True):
        self.supports_composable_checkpointing = supports
        self.checkpoint_calls = []
        self.learn_calls = []
        self.save_calls = []

    def set_checkpointing(self, **kwargs):
        self.checkpoint_calls.append(kwargs)

    def learn(self, **kwargs):
        self.learn_calls.append(kwargs)
        return self

    def save(self, path, name):
        self.save_calls.append((path, name))
        return str(Path(path) / name)


def _write_configs(
    tmp_path,
    *,
    experiment_checkpoint=5,
    algorithm_checkpoint="missing",
    save_strat=("best", "latest"),
    save_trials="none",
):
    alg_dir = tmp_path / "algs"
    alg_dir.mkdir()
    algorithm = {
        "seed": 11,
        "env": "Unused-v0",
        "alg": "TDMPC2/TDMPC2Baseline",
        "alg_params": {},
        "total_steps": 17,
    }
    if algorithm_checkpoint != "missing":
        algorithm["checkpoint_every"] = algorithm_checkpoint
    (alg_dir / "Config.json").write_text(json.dumps(algorithm), encoding="utf-8")

    experiment = {
        "configs": ["Config"],
        "trials": 1,
        "logs": "none",
        "save_trials": save_trials,
        "checkpoint_every": experiment_checkpoint,
        "save_strat": list(save_strat),
        "checkpoint_best_window": 7,
    }
    experiment_path = tmp_path / "Experiment.json"
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    return experiment_path, alg_dir


def _run(monkeypatch, tmp_path, model, **config_kwargs):
    experiment_path, alg_dir = _write_configs(tmp_path, **config_kwargs)
    output_dir = tmp_path / "output"
    env = _Env()
    monkeypatch.setattr(training_main, "build_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        training_main,
        "initialize_alg",
        lambda *args, **kwargs: (model, False, "TDMPC2Baseline"),
    )
    monkeypatch.setattr(training_main, "datetime_stamp", lambda: "STAMP")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--run",
            str(experiment_path),
            "--alg-dir",
            str(alg_dir),
            "--log-dir",
            str(output_dir),
        ],
    )
    training_main.main()
    return env, output_dir


def test_checkpointing_is_configured_without_trajectory_logging(monkeypatch, tmp_path):
    model = _Model()
    env, output_dir = _run(monkeypatch, tmp_path, model)

    assert model.learn_calls == [{"total_timesteps": 17}]
    assert len(model.checkpoint_calls) == 1
    call = model.checkpoint_calls[0]
    assert call["save_freq"] == 5
    assert call["save_strat"] == ("best", "latest")
    assert call["checkpoint_best_window"] == 7
    assert call["trial_run_params"]["seed"] == 11
    assert call["trial_run_params"]["resolved_runtime"]["algorithm"] == (
        "TDMPC2/TDMPC2Baseline"
    )
    assert "actor_loss_scale" not in call["trial_run_params"]["resolved_runtime"]
    assert call["experiment_params"]["checkpoint_every"] == 5
    assert Path(call["save_path"]) == output_dir / "Experiment_STAMP" / "models"
    assert env.closed


def test_resolved_runtime_metadata_contains_horizons_critic_and_inner_budget():
    cfg = SimpleNamespace(
        obs="rgb",
        obs_shape={"rgb": (9, 64, 64)},
        obs_dtype="uint8",
        num_channels=32,
        latent_dim=512,
        action_dim=6,
        episode_length=500,
        train_unroll_horizon=6,
        outer_planning_horizon=3,
        inner_rollout_horizon=6,
        temporal_loss_normalization="reference_weighted_mean",
        temporal_loss_reference_horizon=3,
        rho=0.7,
        outer_critic_target="reward_only",
        inner_sac_critic_target="entropy_augmented",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        sac_actor_loss_scale_tau=0.01,
        compile=True,
        compile_strict=False,
        inner_operator="sac",
        inner_schedule_mode="canonical",
        inner_rounds=4,
        inner_rollouts_per_round=32,
        inner_updates_per_round=192,
        inner_nominal_updates_per_round=192,
        inner_batch_size=64,
        inner_replay_capacity=768,
        inner_replay_sampling="with_replacement",
        inner_replay_scope="action",
        inner_model_step_budget=768,
        inner_expected_update_slots=768,
    )
    critic_signature = {
        "q_representation": "distributional",
        "num_q": 5,
        "num_bins": 101,
        "vmin": -10.0,
        "vmax": 10.0,
    }
    model = SimpleNamespace(
        cfg=cfg,
        agent=SimpleNamespace(model=SimpleNamespace(critic_signature=critic_signature)),
        env=SimpleNamespace(
            task_name="walker-walk",
            action_repeat=2,
            frame_stack=3,
            image_size=64,
            camera_id=0,
        ),
    )

    metadata = training_main._resolved_runtime_metadata(
        model,
        trial_run_params={
            "alg": "AMBITDMPC2/AMBITDMPC2",
            "seed": 55,
        },
    )

    assert metadata["seed"] == 55
    assert metadata["observation"] == {
        "mode": "rgb",
        "shape": [9, 64, 64],
        "dtype": "uint8",
        "num_channels": 32,
        "latent_dim": 512,
        "task": "walker-walk",
        "action_repeat": 2,
        "frame_stack": 3,
        "image_size": 64,
        "camera_id": 0,
        "action_dim": 6,
        "episode_length": 500,
    }
    assert metadata["horizons"] == {
        "train_unroll_horizon": 6,
        "outer_planning_horizon": 3,
        "inner_rollout_horizon": 6,
    }
    assert metadata["critic"] == {
        **critic_signature,
        "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "entropy_augmented",
    }
    assert metadata["actor_loss_scale"] == {
        "mode": "tdmpc2_percentile_range",
        "tau": 0.01,
    }
    assert metadata["compilation"] == {"enabled": True, "strict": False}
    assert metadata["inner_budget"]["branches_per_action"] == 128
    assert metadata["inner_budget"]["transitions_per_round"] == 192
    assert metadata["inner_budget"]["transitions_per_action"] == 768
    assert metadata["inner_budget"]["replay_rows_drawn_per_action"] == 49_152


def test_per_algorithm_null_cadence_disables_experiment_checkpointing(
    monkeypatch, tmp_path
):
    model = _Model(supports=False)
    env, output_dir = _run(
        monkeypatch,
        tmp_path,
        model,
        algorithm_checkpoint=None,
    )

    assert model.checkpoint_calls == []
    assert model.learn_calls == [{"total_timesteps": 17}]
    assert not output_dir.exists()
    assert env.closed


def test_supported_contract_is_required_when_checkpointing_is_enabled(
    monkeypatch, tmp_path
):
    model = _Model(supports=False)
    with pytest.raises(ValueError, match="does not support"):
        _run(monkeypatch, tmp_path, model)
    assert model.learn_calls == []


def test_save_trials_all_remains_active_when_logs_are_disabled(monkeypatch, tmp_path):
    model = _Model()
    env, output_dir = _run(
        monkeypatch,
        tmp_path,
        model,
        experiment_checkpoint=None,
        save_strat=("all",),
        save_trials="all",
    )

    expected_dir = output_dir / "Experiment_STAMP" / "models"
    assert model.checkpoint_calls == []
    assert model.save_calls == [(str(expected_dir) + "/", "model:Config_0")]
    assert env.closed
