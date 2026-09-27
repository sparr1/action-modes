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
    algorithm_overrides=None,
    experiment_overrides=None,
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
    algorithm.update(algorithm_overrides or {})
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
    experiment.update(experiment_overrides or {})
    experiment_path = tmp_path / "Experiment.json"
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    return experiment_path, alg_dir


def _run(monkeypatch, tmp_path, model, *, extra_args=(), **config_kwargs):
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
            *extra_args,
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
        inner_critic_dropout_enabled=False,
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
    assert metadata["inner_budget"]["inner_critic_dropout_enabled"] is False
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


class _ReplayModel(_Model):
    def __init__(self, *, failure=None):
        super().__init__()
        self.archive_calls = []
        self.archive_closed = False
        self.failure = failure

    def enable_replay_archive(self, path, *, name_prefix):
        self.archive_calls.append((path, name_prefix))

    def close_replay_archive(self):
        self.archive_closed = True

    def set_checkpointing(self, **kwargs):
        if self.failure == "setup":
            raise RuntimeError("checkpoint setup failed")
        super().set_checkpointing(**kwargs)

    def learn(self, **kwargs):
        assert self.archive_calls
        if self.failure == "learn":
            raise RuntimeError("training failed")
        return super().learn(**kwargs)


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("algorithm", ["AMBIXQC/AMBIXQC", "AMBIXQC/AMBIXQCBaseline"])
def test_replay_archive_enabled_before_training_for_periodic_and_final_saves(
    monkeypatch, tmp_path, periodic, algorithm
):
    model = _ReplayModel()
    env, output_dir = _run(
        monkeypatch,
        tmp_path,
        model,
        experiment_checkpoint=5 if periodic else None,
        save_trials="none" if periodic else "all",
        algorithm_overrides={"alg": algorithm, "save_replay_buffer": True},
    )
    expected_dir = output_dir / "Experiment_STAMP" / "models"
    assert model.archive_calls == [(str(expected_dir) + "/", "model:Config_0")]
    assert bool(model.checkpoint_calls) == periodic
    assert bool(model.save_calls) != periodic
    assert model.archive_closed
    assert env.closed


@pytest.mark.parametrize("failure", ["setup", "learn"])
def test_replay_archive_closed_after_initialization_or_training_failure(
    monkeypatch, tmp_path, failure
):
    model = _ReplayModel(failure=failure)
    with pytest.raises(RuntimeError, match="failed"):
        _run(
            monkeypatch,
            tmp_path,
            model,
            algorithm_overrides={"alg": "AMBIXQC/AMBIXQC", "save_replay_buffer": True},
        )
    assert model.archive_closed


@pytest.mark.parametrize(
    "algorithm_overrides,experiment_overrides,extra_args,error",
    [
        ({}, {}, (), "only AMBIXQC/AMBIXQC"),
        ({"alg": "AMBIXQC/AMBIXQC", "alg_params": {"obs": "rgb"}}, {}, (), "state observations"),
        ({"alg": "AMBIXQC/AMBIXQC"}, {"env_params": {"obs": "rgb"}}, (), "state observations"),
        ({"alg": "AMBIXQC/AMBIXQC"}, {}, ("--resume-mode", "new", "--lineage-dir", "unused"), "exact resume"),
    ],
)
def test_replay_archive_unsupported_settings_fail_before_artifact_creation(
    monkeypatch, tmp_path, algorithm_overrides, experiment_overrides, extra_args, error
):
    model = _Model()
    with pytest.raises(ValueError, match=error):
        _run(
            monkeypatch,
            tmp_path,
            model,
            algorithm_overrides={**algorithm_overrides, "save_replay_buffer": True},
            experiment_overrides=experiment_overrides,
            extra_args=extra_args,
        )
    assert not model.learn_calls
    assert not (tmp_path / "output").exists()


def test_replay_archive_disabled_does_not_require_algorithm_support(monkeypatch, tmp_path):
    model = _Model()
    env, _ = _run(
        monkeypatch,
        tmp_path,
        model,
        algorithm_overrides={"save_replay_buffer": False},
        experiment_overrides={"save_replay_buffer": True},
    )
    assert model.learn_calls
    assert env.closed
