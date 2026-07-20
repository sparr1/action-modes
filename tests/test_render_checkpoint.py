import json
from pathlib import Path

import numpy as np
import pytest

import render_checkpoint as renderer


def _run_params(algorithm="AMBITDMPC2/AMBITDMPC2", *, seed=7, env="Test-v0"):
    return {
        "name": "Config",
        "seed": seed,
        "env": env,
        "alg": algorithm,
        "device": "cuda",
        "alg_params": {
            "device": "cuda",
            "wandb": True,
            "compile": True,
            "compile_strict": True,
            "inner_diagnostic_rollouts": 4,
        },
        "total_steps": 100,
    }


def _experiment_params():
    return {
        "configs": ["Config"],
        "trials": 1,
        "env_params": {"render_mode": None, "max_episode_steps": 10},
    }


def _metadata(run_params=None, experiment_params=None):
    return {
        "schema_version": 1,
        "checkpoint": {
            "kind": "latest",
            "step": 100,
            "episode": 3,
            "best_score": 12.5,
            "best_window": 100,
        },
        "trial_run_params": run_params or _run_params(),
        "experiment_params": experiment_params or _experiment_params(),
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_adjacent_sidecar_is_authoritative_and_malformed_data_never_falls_back(tmp_path):
    root = tmp_path / "run"
    checkpoint = root / "models" / "model:Config_0_latest"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _write_json(root / "settings.json", _experiment_params())
    _write_json(root / "Config_0" / "alg_settings.json", _run_params(env="Legacy-v0"))

    sidecar = Path(str(checkpoint) + ".metadata.json")
    _write_json(sidecar, _metadata(_run_params(env="Portable-v0")))
    context = renderer.resolve_render_context(checkpoint)
    assert context.source == sidecar
    assert context.trial_run_params["env"] == "Portable-v0"

    malformed = _metadata()
    malformed["checkpoint"]["step"] = "old"
    _write_json(sidecar, malformed)
    with pytest.raises(renderer.RenderCheckpointError, match="checkpoint.step"):
        renderer.resolve_render_context(checkpoint)


def test_copied_checkpoint_and_sidecar_pair_is_self_contained(tmp_path):
    source = tmp_path / "run" / "models" / "model:Config_0_latest.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"portable checkpoint")
    source_sidecar = Path(f"{source}.metadata.json")
    _write_json(source_sidecar, _metadata(_run_params(env="Portable-v0")))

    copied = tmp_path / "copied" / source.name
    copied.parent.mkdir()
    copied.write_bytes(source.read_bytes())
    Path(f"{copied}.metadata.json").write_bytes(source_sidecar.read_bytes())

    context = renderer.resolve_render_context(copied)
    assert context.source == Path(f"{copied}.metadata.json")
    assert context.trial_run_params["env"] == "Portable-v0"


def test_explicit_metadata_or_paired_settings_override_discovery(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    adjacent = Path(str(checkpoint) + ".metadata.json")
    _write_json(adjacent, {"stale": True})

    explicit_metadata = tmp_path / "portable.json"
    _write_json(explicit_metadata, _metadata(_run_params(env="Metadata-v0")))
    context = renderer.resolve_render_context(
        checkpoint, metadata_path=explicit_metadata
    )
    assert context.trial_run_params["env"] == "Metadata-v0"

    trial = tmp_path / "trial.json"
    experiment = tmp_path / "experiment.json"
    _write_json(trial, _run_params(env="Paired-v0"))
    _write_json(experiment, _experiment_params())
    context = renderer.resolve_render_context(
        checkpoint,
        trial_settings=trial,
        experiment_settings=experiment,
    )
    assert context.trial_run_params["env"] == "Paired-v0"

    with pytest.raises(renderer.RenderCheckpointError, match="supplied together"):
        renderer.resolve_render_context(checkpoint, trial_settings=trial)
    with pytest.raises(renderer.RenderCheckpointError, match="either --metadata"):
        renderer.resolve_render_context(
            checkpoint,
            metadata_path=explicit_metadata,
            trial_settings=trial,
            experiment_settings=experiment,
        )


def test_legacy_run_tree_discovers_matching_alg_settings(tmp_path):
    root = tmp_path / "OldExperiment"
    checkpoint = root / "models" / "model:Config_Name_12_500_steps.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    run_params = _run_params(env="Legacy-v0")
    run_params["name"] = "Config_Name"
    _write_json(root / "settings.json", _experiment_params())
    trial_settings = root / "Config_Name_12" / "alg_settings.json"
    _write_json(trial_settings, run_params)

    context = renderer.resolve_render_context(checkpoint)
    assert context.source == trial_settings
    assert context.trial_run_params == run_params


def test_sidecar_schema_and_backend_allowlist_errors_are_actionable(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"checkpoint")
    sidecar = Path(str(checkpoint) + ".metadata.json")

    future = _metadata()
    future["schema_version"] = 2
    _write_json(sidecar, future)
    with pytest.raises(renderer.RenderCheckpointError, match="schema_version=2"):
        renderer.resolve_render_context(checkpoint)

    with pytest.raises(renderer.RenderCheckpointError, match="Legacy AMBI, PAMDP"):
        renderer._backend_for("AMBI/AMBI")
    assert renderer._backend_for("baselines/PPO") == "sb3"
    assert renderer._backend_for("SAC/SAC") == "native_sac"
    assert renderer._backend_for("TDMPC2/TDMPC2Baseline") == "tdmpc2"
    assert renderer._backend_for("AMBITDMPC2/AMBITDMPC2") == "ambi_tdmpc2"


def test_render_overrides_are_isolated_and_disable_training_only_features(tmp_path):
    context = renderer.RenderContext(
        _run_params(), _experiment_params(), tmp_path / "metadata.json"
    )
    run_params, experiment_params = renderer._prepare_run_params(
        context,
        backend="ambi_tdmpc2",
        device="cpu",
        controller_seed=19,
    )

    assert run_params["device"] == "cpu"
    assert run_params["seed"] == 19
    assert run_params["alg_params"]["device"] == "cpu"
    assert run_params["alg_params"]["seed"] == 19
    assert run_params["alg_params"]["wandb"] is False
    assert run_params["alg_params"]["compile"] is False
    assert run_params["alg_params"]["compile_strict"] is False
    assert run_params["alg_params"]["inner_diagnostic_rollouts"] == 0
    assert context.trial_run_params["device"] == "cuda"
    assert context.trial_run_params["alg_params"]["wandb"] is True
    assert experiment_params is not context.experiment_params

    native_context = renderer.RenderContext(
        _run_params("SAC/SAC"), _experiment_params(), tmp_path / "native.json"
    )
    native_params, _ = renderer._prepare_run_params(
        native_context,
        backend="native_sac",
        device="cpu",
        controller_seed=19,
    )
    assert native_params["alg_params"]["buffer_size"] == 1

    sb3_context = renderer.RenderContext(
        _run_params("baselines/SAC"), _experiment_params(), tmp_path / "sb3.json"
    )
    sb3_params, _ = renderer._prepare_run_params(
        sb3_context,
        backend="sb3",
        device="cpu",
        controller_seed=19,
    )
    assert sb3_params["alg_params"]["buffer_size"] == 1


class _Space:
    def __init__(self):
        self.seeds = []

    def seed(self, seed):
        self.seeds.append(seed)


class _RolloutEnv:
    metadata = {"render_fps": 17}

    def __init__(self, *, steps_per_episode=2, fail_step=None, varying_frames=False):
        self.action_space = _Space()
        self.observation_space = _Space()
        self.steps_per_episode = steps_per_episode
        self.fail_step = fail_step
        self.varying_frames = varying_frames
        self.step_index = 0
        self.reset_seeds = []
        self.close_calls = 0
        self.render_calls = 0

    def reset(self, *, seed=None):
        self.step_index = 0
        self.reset_seeds.append(seed)
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.step_index += 1
        if self.fail_step == self.step_index:
            raise RuntimeError("environment failed")
        done = self.step_index >= self.steps_per_episode
        return (
            np.array([self.step_index], dtype=np.float32),
            1.25,
            done,
            False,
            {},
        )

    def render(self):
        self.render_calls += 1
        width = 3 if self.varying_frames and self.render_calls > 1 else 2
        frame = np.zeros((2, width, 3), dtype=np.uint8)
        frame[..., 0] = 1
        frame[..., 1] = 2
        frame[..., 2] = 3
        return frame

    def close(self):
        self.close_calls += 1


class _ModelEnv:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _EvalNode:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1


class _FakeModel:
    def __init__(self):
        self.agent = type("Agent", (), {"model": _EvalNode(), "actor": _EvalNode()})()
        self.loaded = None
        self.predict_calls = []

    def load(self, path):
        self.loaded = path
        return self

    def predict(self, observation, **kwargs):
        self.predict_calls.append(kwargs)
        return np.array([0.0], dtype=np.float32), None


def test_display_rollouts_use_separate_envs_and_reset_ambi_planning(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model:Config_0_latest"
    checkpoint.write_bytes(b"checkpoint")
    _write_json(Path(str(checkpoint) + ".metadata.json"), _metadata())

    model_env = _ModelEnv()
    rollout_env = _RolloutEnv()
    render_modes = []
    captured = {}
    model = _FakeModel()

    def fake_build_env(run_params, experiment_params, *, render_mode):
        render_modes.append(render_mode)
        return model_env if render_mode is None else rollout_env

    def fake_initialize(algorithm, alg_params, env, **kwargs):
        captured.update(
            algorithm=algorithm,
            alg_params=alg_params,
            env=env,
            run_params=kwargs["full_run_params"],
        )
        return model, False, "AMBITDMPC2"

    monkeypatch.setattr(renderer, "build_env", fake_build_env)
    monkeypatch.setattr(renderer, "initialize_alg", fake_initialize)

    results = renderer.render_checkpoint(
        checkpoint,
        display=True,
        episodes=2,
        seed=11,
        device="cpu",
    )

    assert render_modes == [None, "human"]
    assert captured["env"] is model_env
    assert captured["alg_params"]["wandb"] is False
    assert captured["alg_params"]["compile"] is False
    assert captured["run_params"]["device"] == "cpu"
    assert rollout_env.reset_seeds == [11, 12]
    assert [result.episode_return for result in results] == [2.5, 2.5]
    assert [result.length for result in results] == [2, 2]
    assert [call["episode_start"] for call in model.predict_calls] == [
        True,
        False,
        True,
        False,
    ]
    assert all(call["deterministic"] is True for call in model.predict_calls)
    assert all(call["collect_diagnostics"] is False for call in model.predict_calls)
    assert model.agent.model.eval_calls == 1
    assert model_env.close_calls == 1
    assert rollout_env.close_calls == 1


def test_results_json_runs_headlessly_and_records_deterministic_seed_set(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "model:Config_0_best"
    checkpoint.write_bytes(b"checkpoint")
    run_params = _run_params()
    run_params["resolved_runtime"] = {
        "horizons": {
            "train_unroll_horizon": 3,
            "outer_planning_horizon": 3,
            "inner_rollout_horizon": 3,
        }
    }
    _write_json(
        Path(str(checkpoint) + ".metadata.json"), _metadata(run_params)
    )

    model_env = _ModelEnv()
    rollout_env = _RolloutEnv()
    model = _FakeModel()
    render_modes = []
    environments = iter((model_env, rollout_env))

    def fake_build_env(run_params, experiment_params, *, render_mode):
        render_modes.append(render_mode)
        return next(environments)

    monkeypatch.setattr(renderer, "build_env", fake_build_env)
    monkeypatch.setattr(
        renderer,
        "initialize_alg",
        lambda *args, **kwargs: (model, False, "AMBITDMPC2"),
    )

    output = tmp_path / "evaluation" / "best.json"
    results = renderer.render_checkpoint(
        checkpoint,
        results_json=output,
        episodes=5,
        seed=101,
        device="cpu",
    )

    assert render_modes == [None, None]
    assert rollout_env.render_calls == 0
    assert [result.seed for result in results] == [101, 102, 103, 104, 105]
    assert all(call["deterministic"] is True for call in model.predict_calls)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["checkpoint"] == str(checkpoint.resolve())
    assert payload["deterministic"] is True
    assert payload["seeds"] == [101, 102, 103, 104, 105]
    assert payload["summary"] == {
        "capped_episodes": 0,
        "episodes": 5,
        "length_max": 2,
        "length_mean": 2.0,
        "length_min": 2,
        "return_max": 2.5,
        "return_mean": 2.5,
        "return_min": 2.5,
        "return_std": 0.0,
    }
    assert payload["checkpoint_metadata"]["kind"] == "latest"
    assert payload["resolved_runtime"] == run_params["resolved_runtime"]
    assert list(output.parent.glob(".*.tmp")) == []
    assert model_env.close_calls == 1
    assert rollout_env.close_calls == 1

    with pytest.raises(renderer.RenderCheckpointError, match="already exists"):
        renderer.render_checkpoint(checkpoint, results_json=output)


def test_output_modes_include_headless_results_json(tmp_path):
    parser = renderer.build_parser()
    args = parser.parse_args(["checkpoint", "--results-json", str(tmp_path / "r.json")])
    assert args.results_json == tmp_path / "r.json"

    with pytest.raises(renderer.RenderCheckpointError, match="exactly one output mode"):
        renderer.render_checkpoint(
            tmp_path / "checkpoint",
            display=True,
            results_json=tmp_path / "r.json",
        )


class _FakeCV2:
    COLOR_RGB2BGR = 9

    def __init__(self):
        self.instances = []
        self.cvt_codes = []

    def VideoWriter_fourcc(self, *letters):
        assert letters == tuple("mp4v")
        return 1234

    def VideoWriter(self, path, fourcc, fps, size):
        instance = _FakeCV2Writer(path, fourcc, fps, size)
        self.instances.append(instance)
        return instance

    def cvtColor(self, frame, code):
        self.cvt_codes.append(code)
        return frame[..., ::-1].copy()


class _FakeCV2Writer:
    def __init__(self, path, fourcc, fps, size):
        self.path = Path(path)
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames = []
        self.release_calls = 0

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.release_calls += 1
        self.path.write_bytes(b"fake-mp4")


def test_video_streams_rgb_as_bgr_at_env_fps_and_publishes_atomically(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "native.pt"
    checkpoint.write_bytes(b"checkpoint")
    run_params = _run_params("SAC/SAC", seed=21)
    _write_json(Path(str(checkpoint) + ".metadata.json"), _metadata(run_params))

    model_env = _ModelEnv()
    rollout_env = _RolloutEnv()
    model = _FakeModel()
    fake_cv2 = _FakeCV2()

    monkeypatch.setattr(
        renderer,
        "build_env",
        lambda run, experiment, *, render_mode: (
            model_env if render_mode is None else rollout_env
        ),
    )
    monkeypatch.setattr(
        renderer,
        "initialize_alg",
        lambda *args, **kwargs: (model, False, "SAC"),
    )
    monkeypatch.setattr(renderer, "_import_cv2", lambda: fake_cv2)

    output_dir = tmp_path / "videos"
    results = renderer.render_checkpoint(
        checkpoint,
        video_dir=output_dir,
        stochastic=True,
    )

    expected = renderer.video_target_path(
        output_dir.resolve(), checkpoint.resolve(), episode=1, seed=21
    )
    assert results[0].video_path == expected
    assert expected.read_bytes() == b"fake-mp4"
    assert list(output_dir.glob(".*.tmp.mp4")) == []
    assert len(fake_cv2.instances) == 1
    writer = fake_cv2.instances[0]
    assert writer.fps == 17
    assert writer.size == (2, 2)
    assert len(writer.frames) == 3  # Reset frame plus one frame per transition.
    assert writer.frames[0][0, 0].tolist() == [3, 2, 1]
    assert fake_cv2.cvt_codes == [fake_cv2.COLOR_RGB2BGR] * 3
    assert all(call["deterministic"] is False for call in model.predict_calls)
    assert rollout_env.close_calls == 1
    assert model_env.close_calls == 1

    monkeypatch.setattr(
        renderer,
        "build_env",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("collision should be rejected before environment creation")
        ),
    )
    with pytest.raises(renderer.RenderCheckpointError, match="Video already exists"):
        renderer.render_checkpoint(checkpoint, video_dir=output_dir)


def test_video_failure_removes_temporary_output_and_closes_both_envs(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "tdmpc"
    checkpoint.write_bytes(b"checkpoint")
    run_params = _run_params("TDMPC2/TDMPC2Baseline")
    _write_json(Path(str(checkpoint) + ".metadata.json"), _metadata(run_params))

    model_env = _ModelEnv()
    rollout_env = _RolloutEnv(varying_frames=True)
    fake_cv2 = _FakeCV2()
    monkeypatch.setattr(
        renderer,
        "build_env",
        lambda run, experiment, *, render_mode: (
            model_env if render_mode is None else rollout_env
        ),
    )
    monkeypatch.setattr(
        renderer,
        "initialize_alg",
        lambda *args, **kwargs: (_FakeModel(), False, "TDMPC2Baseline"),
    )
    monkeypatch.setattr(renderer, "_import_cv2", lambda: fake_cv2)

    output_dir = tmp_path / "videos"
    with pytest.raises(renderer.RenderCheckpointError, match="frame size changed"):
        renderer.render_checkpoint(checkpoint, video_dir=output_dir)

    assert list(output_dir.iterdir()) == []
    assert fake_cv2.instances[0].release_calls == 1
    assert rollout_env.close_calls == 1
    assert model_env.close_calls == 1


def test_max_steps_caps_an_episode_and_sb3_dispatch_uses_underlying_model():
    class Underlying:
        def __init__(self):
            self.calls = []

        def predict(self, observation, *, deterministic):
            self.calls.append(deterministic)
            return "action", None

    underlying = Underlying()
    wrapper = type("Wrapper", (), {"get_model": lambda self: underlying})()
    action = renderer._predict_action(
        wrapper,
        "observation",
        backend="sb3",
        deterministic=True,
        episode_start=True,
    )
    assert action == "action"
    assert underlying.calls == [True]

    env = _RolloutEnv(steps_per_episode=100)
    model = _FakeModel()
    results = renderer._rollout(
        model,
        env,
        checkpoint=Path("checkpoint.pt"),
        backend="native_sac",
        episodes=1,
        first_seed=3,
        deterministic=True,
        max_steps=3,
        video_dir=None,
        overwrite=False,
    )
    assert results[0].length == 3
    assert results[0].capped is True


def test_sb3_restore_forwards_nonrendering_env_and_device(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sb3.zip"
    checkpoint.write_bytes(b"checkpoint")
    model_env = object()

    class Policy:
        def __init__(self):
            self.training_modes = []

        def set_training_mode(self, mode):
            self.training_modes.append(mode)

    class Underlying:
        def __init__(self):
            self.load_calls = []
            self.policy = Policy()

        def load(self, path, *, env, device, custom_objects):
            self.load_calls.append((path, env, device, custom_objects))
            return self

    class Wrapper:
        def __init__(self):
            self.model = Underlying()

        def get_model(self):
            return self.model

    wrapper = Wrapper()
    monkeypatch.setattr(
        renderer,
        "initialize_alg",
        lambda *args, **kwargs: (wrapper, True, "SAC"),
    )
    run_params = _run_params("baselines/SAC")
    run_params["device"] = "cpu"

    restored = renderer._initialize_model(
        checkpoint,
        run_params,
        _experiment_params(),
        model_env,
        "sb3",
    )

    assert restored is wrapper
    assert wrapper.model.load_calls == [
        (str(checkpoint), model_env, "cpu", {"buffer_size": 1})
    ]
    assert wrapper.model.policy.training_modes == [False]
