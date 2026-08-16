import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import main as training_main
import utils.resume_identity as identity_module
import utils.resume_training as resume_training_module
from utils.resume_runtime import register_test_resume_environment


class ResumeEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.spec = SimpleNamespace(id="ResumeEnv-v0", max_episode_steps=2)
        self.closed = False

    def training_resume_state(self):
        return {"schema_version": 1}

    def validate_training_resume_state(self, state):
        assert state == {"schema_version": 1}

    def load_training_resume_state(self, state):
        self.validate_training_resume_state(state)

    def close(self):
        self.closed = True


register_test_resume_environment(ResumeEnv, episode_steps=2)


class ResumeModel:
    supports_composable_checkpointing = True

    def __init__(self):
        self.cfg = SimpleNamespace(
            obs="state",
            obs_shape={"state": (3,)},
            obs_dtype="float32",
            action_dim=1,
            episode_length=2,
            compile=False,
            compile_strict=False,
        )
        self.agent = SimpleNamespace()
        self.buffer = SimpleNamespace()
        self.enable_calls = []
        self.learn_calls = []
        self.checkpoint_calls = []
        self.logger = None

    def enable_training_resume(self, *, total_timesteps):
        self.enable_calls.append(total_timesteps)

    def learn(self, **kwargs):
        self.learn_calls.append(kwargs)
        return 75

    def set_checkpointing(self, **kwargs):
        self.checkpoint_calls.append(kwargs)

    def set_logger(self, logger):
        self.logger = logger


class FakeSession:
    calls = []

    def __init__(self, root):
        self.root = Path(root)
        self.segment_dir = self.root / "segments" / "test-segment"
        self.segment_log_dir = self.segment_dir / "logs"
        self.eval_csv_path = self.segment_dir / "evaluation.csv"
        self.segment_log_dir.mkdir(parents=True)
        self.closed = False

    @classmethod
    def open(cls, lineage_dir, **kwargs):
        cls.calls.append((Path(lineage_dir), kwargs))
        return cls(lineage_dir)

    def close(self):
        self.closed = True


def _write_manifest(tmp_path, *, configs=("Only",), trials=1, total_steps=4):
    alg_dir = tmp_path / "algs"
    alg_dir.mkdir()
    for name in configs:
        (alg_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "seed": 5,
                    "env": "ResumeEnv-v0",
                    "alg": "TDMPC2/TDMPC2Baseline",
                    "alg_params": {
                        "obs": "state",
                        "wandb": True,
                        "wandb_mode": "online",
                    },
                    "total_steps": total_steps,
                }
            )
        )
    manifest = tmp_path / "experiment.json"
    manifest.write_text(
        json.dumps(
            {
                "configs": list(configs),
                "trials": trials,
                "logs": "none",
                "save_trials": "none",
            }
        )
    )
    return manifest, alg_dir


def test_main_resume_acquires_session_before_model_and_passes_explicit_context(
    monkeypatch, tmp_path
):
    manifest, alg_dir = _write_manifest(tmp_path)
    lineage = tmp_path / "lineage"
    env = ResumeEnv()
    model = ResumeModel()
    events = []
    FakeSession.calls = []
    monkeypatch.setattr(training_main, "_PROCESS_STARTED_MONOTONIC", 100.0)
    monkeypatch.setattr(training_main, "_MONOTONIC", lambda: 100.0)

    monkeypatch.setattr(
        identity_module,
        "lineage_identity",
        lambda **_kwargs: {"schema_version": 1, "fingerprint": "exact"},
    )
    monkeypatch.setattr(
        resume_training_module, "TrainingResumeSession", FakeSession
    )
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: (events.append("environment") or env),
    )

    def initialize(*_args, **_kwargs):
        assert FakeSession.calls
        events.append("model")
        return model, False, "TDMPC2Baseline"

    monkeypatch.setattr(training_main, "initialize_alg", initialize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--run",
            str(manifest),
            "--alg-dir",
            str(alg_dir),
            "--num-runs",
            "1",
            "--lineage-dir",
            str(lineage),
            "--resume-mode",
            "new",
            "--resume-checkpoint-minutes",
            "60",
            "--drain-after-seconds",
            "300",
        ],
    )

    assert training_main.main() == 75
    assert events == ["environment", "model"]
    assert model.enable_calls == [4]
    assert len(model.learn_calls) == 1
    assert model.learn_calls[0]["total_timesteps"] == 4
    assert isinstance(model.learn_calls[0]["resume_session"], FakeSession)
    path, call = FakeSession.calls[0]
    assert path == lineage
    assert call["mode"] == "new"
    assert call["checkpoint_minutes"] == 60
    assert call["drain_after_seconds"] == 300
    assert "drain_timing" not in call
    assert env.closed


def test_main_rejects_non_boundary_target_before_creating_lineage(
    monkeypatch, tmp_path
):
    manifest, alg_dir = _write_manifest(tmp_path, total_steps=3)
    lineage = tmp_path / "lineage"
    env = ResumeEnv()
    FakeSession.calls = []
    monkeypatch.setattr(training_main, "build_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(
        resume_training_module, "TrainingResumeSession", FakeSession
    )
    monkeypatch.setattr(
        training_main,
        "initialize_alg",
        lambda *_args, **_kwargs: pytest.fail("model must not be initialized"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--run",
            str(manifest),
            "--alg-dir",
            str(alg_dir),
            "--num-runs",
            "1",
            "--lineage-dir",
            str(lineage),
            "--resume-mode",
            "new",
            "--resume-checkpoint-minutes",
            "60",
        ],
    )

    with pytest.raises(ValueError, match="episode boundary"):
        training_main.main()
    assert FakeSession.calls == []
    assert not lineage.exists()
    assert env.closed


def test_main_charges_python_pre_session_time_to_drain(
    monkeypatch, tmp_path
):
    manifest, alg_dir = _write_manifest(tmp_path)
    lineage = tmp_path / "lineage"
    env = ResumeEnv()
    model = ResumeModel()
    clock = {"now": 100.0}
    FakeSession.calls = []

    monkeypatch.setattr(training_main, "_PROCESS_STARTED_MONOTONIC", 100.0)
    monkeypatch.setattr(training_main, "_MONOTONIC", lambda: clock["now"])

    def build_env(*_args, **_kwargs):
        clock["now"] += 20.0
        return env

    def identity(**_kwargs):
        clock["now"] += 30.0
        return {"schema_version": 1, "fingerprint": "slow-preflight"}

    monkeypatch.setattr(training_main, "build_env", build_env)
    monkeypatch.setattr(identity_module, "lineage_identity", identity)
    monkeypatch.setattr(
        resume_training_module, "TrainingResumeSession", FakeSession
    )
    monkeypatch.setattr(
        training_main,
        "initialize_alg",
        lambda *_args, **_kwargs: (model, False, "TDMPC2Baseline"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--run",
            str(manifest),
            "--alg-dir",
            str(alg_dir),
            "--num-runs",
            "1",
            "--lineage-dir",
            str(lineage),
            "--resume-mode",
            "new",
            "--drain-after-seconds",
            "300",
        ],
    )

    assert training_main.main() == 75
    _, call = FakeSession.calls[0]
    assert call["drain_after_seconds"] == 250.0
    assert "drain_timing" not in call
    assert env.closed


def test_main_fails_closed_when_python_pre_session_exhausts_drain(
    monkeypatch, tmp_path
):
    manifest, alg_dir = _write_manifest(tmp_path)
    lineage = tmp_path / "lineage"
    env = ResumeEnv()
    model = ResumeModel()
    clock = {"now": 100.0}
    FakeSession.calls = []

    monkeypatch.setattr(training_main, "_PROCESS_STARTED_MONOTONIC", 100.0)
    monkeypatch.setattr(training_main, "_MONOTONIC", lambda: clock["now"])
    monkeypatch.setattr(training_main, "build_env", lambda *_args, **_kwargs: env)

    def identity(**_kwargs):
        clock["now"] = 400.0
        return {"schema_version": 1, "fingerprint": "expired-preflight"}

    monkeypatch.setattr(identity_module, "lineage_identity", identity)
    monkeypatch.setattr(
        resume_training_module, "TrainingResumeSession", FakeSession
    )
    monkeypatch.setattr(
        training_main,
        "initialize_alg",
        lambda *_args, **_kwargs: pytest.fail("model must not initialize"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--run",
            str(manifest),
            "--alg-dir",
            str(alg_dir),
            "--num-runs",
            "1",
            "--lineage-dir",
            str(lineage),
            "--resume-mode",
            "new",
            "--drain-after-seconds",
            "300",
        ],
    )

    with pytest.raises(ValueError, match="deadline expired"):
        training_main.main()
    assert FakeSession.calls == []
    assert not lineage.exists()
    assert env.closed


def test_main_rejects_invalid_resume_wandb_before_creating_lineage(
    monkeypatch, tmp_path
):
    manifest, alg_dir = _write_manifest(tmp_path)
    algorithm_path = alg_dir / "Only.json"
    algorithm = json.loads(algorithm_path.read_text())
    algorithm["alg_params"].pop("wandb_mode")
    algorithm_path.write_text(json.dumps(algorithm))
    lineage = tmp_path / "lineage"
    FakeSession.calls = []

    class ForbiddenWandb:
        def init(self, **_kwargs):
            pytest.fail("W&B must not initialize before configuration preflight")

    monkeypatch.setitem(sys.modules, "wandb", ForbiddenWandb())
    monkeypatch.setattr(
        resume_training_module, "TrainingResumeSession", FakeSession
    )
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail(
            "Environment must not initialize before W&B configuration preflight"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--run",
            str(manifest),
            "--alg-dir",
            str(alg_dir),
            "--num-runs",
            "1",
            "--lineage-dir",
            str(lineage),
            "--resume-mode",
            "new",
        ],
    )

    with pytest.raises(ValueError, match="wandb_mode='online'"):
        training_main.main()
    assert FakeSession.calls == []
    assert not lineage.exists()


def test_oscar_real_anchor_resolves_explicit_online_mode_without_mutating_hydra_config(
    monkeypatch, tmp_path
):
    """Exercise the actual Oscar manifest/config through main's resolver."""

    repo_root = Path(training_main.__file__).resolve().parent
    manifest = repo_root / "configs/ambi/experiments/ambi_anchor.json"
    alg_dir = repo_root / "configs/ambi/algs"
    shared_alg_path = alg_dir / "ambi_anchor.json"
    shared_alg_before = shared_alg_path.read_bytes()
    assert "wandb_mode" not in json.loads(shared_alg_before)["alg_params"]

    lineage = tmp_path / "oscar-lineage"
    env = ResumeEnv()
    model = ResumeModel()
    resolved_alg_params = []
    FakeSession.calls = []

    monkeypatch.setattr(
        identity_module,
        "lineage_identity",
        lambda **kwargs: {
            "schema_version": 1,
            "fingerprint": "oscar-real-config",
            "trial_run_params": kwargs["trial_run_params"],
        },
    )
    monkeypatch.setattr(
        resume_training_module, "TrainingResumeSession", FakeSession
    )
    monkeypatch.setattr(training_main, "build_env", lambda *_args, **_kwargs: env)

    def initialize(_algorithm, alg_params, *_args, **_kwargs):
        resolved_alg_params.append(dict(alg_params))
        return model, False, "AMBITDMPC2"

    monkeypatch.setattr(training_main, "initialize_alg", initialize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--run",
            str(manifest),
            "--alg-dir",
            str(alg_dir),
            "--num-runs",
            "1",
            "--lineage-dir",
            str(lineage),
            "--resume-mode",
            "new",
            "--resume-wandb-mode",
            "online",
        ],
    )

    assert training_main.main() == 75
    assert resolved_alg_params[0]["wandb"] is True
    assert resolved_alg_params[0]["wandb_mode"] == "online"
    resolved_run = json.loads(
        (
            lineage
            / "segments"
            / "test-segment"
            / "resolved_run.json"
        ).read_text()
    )
    assert resolved_run["alg_params"]["wandb_mode"] == "online"
    assert shared_alg_path.read_bytes() == shared_alg_before
    assert env.closed


@pytest.mark.parametrize(
    "extra",
    [
        ("--resume-mode", "new"),
        ("--lineage-dir", "somewhere"),
        ("--resume-generation", "step-1"),
        ("--drain-after-seconds", "10"),
        ("--resume-wandb-mode", "online"),
    ],
)
def test_main_rejects_partial_resume_cli(monkeypatch, tmp_path, extra):
    manifest, alg_dir = _write_manifest(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--run",
            str(manifest),
            "--alg-dir",
            str(alg_dir),
            *extra,
        ],
    )
    with pytest.raises(ValueError, match="resume|Resume"):
        training_main.main()


def test_resume_manifest_must_be_one_cell(monkeypatch, tmp_path):
    manifest, alg_dir = _write_manifest(tmp_path, configs=("One", "Two"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--run",
            str(manifest),
            "--alg-dir",
            str(alg_dir),
            "--num-runs",
            "1",
            "--lineage-dir",
            str(tmp_path / "lineage"),
            "--resume-mode",
            "new",
        ],
    )
    with pytest.raises(ValueError, match="exactly one"):
        training_main.main()
