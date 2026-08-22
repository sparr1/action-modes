import json
import sys
from pathlib import Path

import pytest

import main as training_main


class _Space:
    def seed(self, _seed):
        return None


class _Env:
    def __init__(self, *, close_error=None):
        self.action_space = _Space()
        self.observation_space = _Space()
        self.closed = False
        self.close_error = close_error

    def reset(self, *, seed=None):
        return 0, {}

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _Model:
    cfg = None

    def __init__(
        self,
        *,
        checkpoint_error=None,
        writer=None,
        learn_error=None,
        close_error=None,
    ):
        self.supports_composable_checkpointing = True
        self.checkpoint_error = checkpoint_error
        self._checkpoint_writer = writer
        self.learn_error = learn_error
        self.close_error = close_error
        self.closed = False
        self.learn_calls = []

    def set_checkpointing(self, **_kwargs):
        if self.checkpoint_error is not None:
            raise self.checkpoint_error

    def learn(self, **kwargs):
        self.learn_calls.append(kwargs)
        if self.learn_error is not None:
            raise self.learn_error

    def set_logger(self, logger):
        self.logger = logger

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _write_configs(tmp_path, *, experiment_updates=None):
    algorithm_dir = tmp_path / "algs"
    algorithm_dir.mkdir()
    (algorithm_dir / "Config.json").write_text(
        json.dumps(
            {
                "seed": 11,
                "env": "Dummy-v0",
                "alg": "Dummy/Dummy",
                "alg_params": {},
                "total_steps": 17,
            }
        ),
        encoding="utf-8",
    )
    experiment = {
        "configs": ["Config"],
        "trials": 1,
        "logs": "none",
        "save_trials": "none",
    }
    experiment.update(experiment_updates or {})
    experiment_path = tmp_path / "Experiment.json"
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    return experiment_path, algorithm_dir, experiment


def _argv(experiment_path, algorithm_dir, output_dir, *extra):
    return [
        "main.py",
        "--run",
        str(experiment_path),
        "--alg-dir",
        str(algorithm_dir),
        "--log-dir",
        str(output_dir),
        *extra,
    ]


@pytest.mark.parametrize("collection_type", [list, tuple])
def test_remove_saved_checkpoint_accepts_explicit_artifact_collections(
    tmp_path, collection_type
):
    first = tmp_path / "policy.pt"
    second = tmp_path / "critic.pt"
    unrelated = tmp_path / "keep.pt"
    for path in (first, second, unrelated):
        path.write_bytes(b"checkpoint")
        Path(f"{path}.metadata.json").write_text("{}", encoding="utf-8")

    training_main._remove_saved_checkpoint(collection_type((first, second)))

    assert not first.exists()
    assert not Path(f"{first}.metadata.json").exists()
    assert not second.exists()
    assert not Path(f"{second}.metadata.json").exists()
    assert unrelated.exists()
    assert Path(f"{unrelated}.metadata.json").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--num-runs", "-2"), "--num-runs"),
        (("--alg-index", "-1"), "--alg-index must be non-negative"),
        (("--trial-index", "-1"), "--trial-index must be non-negative"),
        (("--alg-index", "1"), "--alg-index 1 is out of range"),
        (("--trial-index", "1"), "--trial-index 1 is out of range"),
    ],
)
def test_invalid_run_selection_fails_before_environment_creation(
    monkeypatch, tmp_path, extra, message
):
    experiment_path, algorithm_dir, _ = _write_configs(tmp_path)
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("invalid selection built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output", *extra),
    )

    with pytest.raises(ValueError, match=message):
        training_main.main()


def test_zero_num_runs_has_no_environment_or_artifact_side_effects(
    monkeypatch, tmp_path
):
    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path, experiment_updates={"logs": "warn", "save_trials": "first"}
    )
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("zero runs built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            experiment_path,
            algorithm_dir,
            output_dir,
            "--num-runs",
            "0",
        ),
    )

    training_main.main()

    assert not output_dir.exists()


@pytest.mark.parametrize("trials", [True, 0, -1, 1.5, "1"])
def test_invalid_trial_counts_fail_before_environment_creation(
    monkeypatch, tmp_path, trials
):
    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path, experiment_updates={"trials": trials}
    )
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("invalid trials built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="'trials' must be a positive integer"):
        training_main.main()


def test_invalid_save_trials_policy_fails_instead_of_silently_skipping_save(
    monkeypatch, tmp_path
):
    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path, experiment_updates={"save_trials": "frist"}
    )
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("invalid save policy built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="unsupported save_trials"):
        training_main.main()


@pytest.mark.parametrize("log_info", [None, 0, 1, "true", [], {}])
def test_log_info_must_be_boolean(monkeypatch, tmp_path, log_info):
    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path, experiment_updates={"log_info": log_info}
    )
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("invalid log_info built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="'log_info' must be a boolean"):
        training_main.main()


def test_overrides_alg_must_be_an_object(monkeypatch, tmp_path):
    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path, experiment_updates={"overrides_alg": []}
    )
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("invalid overrides built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="'overrides_alg' must be a JSON object"):
        training_main.main()


def test_algorithm_configuration_must_be_an_object(monkeypatch, tmp_path):
    experiment_path, algorithm_dir, _ = _write_configs(tmp_path)
    (algorithm_dir / "Config.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("invalid alg config built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="must be a JSON object"):
        training_main.main()


@pytest.mark.parametrize(
    ("target", "payload", "message"),
    [
        (
            "experiment",
            '{"configs":["Config"],"trials":1,"trials":2}',
            "duplicate JSON key 'trials'",
        ),
        (
            "experiment",
            '{"configs":["Config"],"trials":NaN}',
            "non-finite JSON number 'NaN'",
        ),
        (
            "algorithm",
            '{"alg":"Dummy/Dummy","alg":"Other/Other"}',
            "duplicate JSON key 'alg'",
        ),
        (
            "algorithm",
            '{"alg":"Dummy/Dummy","total_steps":Infinity}',
            "non-finite JSON number 'Infinity'",
        ),
        (
            "algorithm",
            '{"alg":"Dummy/Dummy","total_steps":-Infinity}',
            "non-finite JSON number '-Infinity'",
        ),
        (
            "algorithm",
            '{"alg":"Dummy/Dummy","total_steps":1e999}',
            "non-finite JSON number '1e999'",
        ),
    ],
)
def test_runtime_config_loader_rejects_ambiguous_or_nonfinite_json(
    monkeypatch, tmp_path, target, payload, message
):
    experiment_path, algorithm_dir, _ = _write_configs(tmp_path)
    path = (
        experiment_path
        if target == "experiment"
        else algorithm_dir / "Config.json"
    )
    path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("invalid JSON built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(ValueError, match=message) as captured:
        training_main.main()

    assert str(path) in str(captured.value)


def test_duplicate_algorithm_config_names_are_rejected(monkeypatch, tmp_path):
    experiment_path, algorithm_dir, experiment = _write_configs(tmp_path)
    experiment["configs"] = ["Config", "Config"]
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("duplicate configs built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="must not contain duplicate"):
        training_main.main()


@pytest.mark.parametrize(
    "config_name",
    [".", "..", "../Config", "nested/Config", "nested\\Config", "Config.json"],
)
def test_algorithm_config_names_must_be_safe_extension_free_basenames(
    monkeypatch, tmp_path, config_name
):
    experiment_path, algorithm_dir, experiment = _write_configs(tmp_path)
    experiment["configs"] = [config_name]
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("unsafe config name built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="plain basenames"):
        training_main.main()


def test_dotted_algorithm_config_basename_remains_supported(monkeypatch, tmp_path):
    experiment_path, algorithm_dir, experiment = _write_configs(tmp_path)
    dotted_name = "AntPlaneMoveNew5.0"
    (algorithm_dir / "Config.json").rename(algorithm_dir / f"{dotted_name}.json")
    experiment["configs"] = [dotted_name]
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    env = _Env()
    model = _Model()
    monkeypatch.setattr(training_main, "build_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(
        training_main,
        "initialize_alg",
        lambda *_args, **_kwargs: (model, False, "Dummy"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    training_main.main()

    assert model.learn_calls == [{"total_timesteps": 17}]
    assert env.closed


@pytest.mark.parametrize("total_steps", [True, -1, 1.5, "17", None])
def test_merged_total_steps_must_be_a_nonnegative_integer_value(
    monkeypatch, tmp_path, total_steps
):
    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path,
        experiment_updates={"overrides_alg": {"total_steps": total_steps}},
    )
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid total_steps built an environment"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(ValueError, match="'total_steps'.*integer-valued"):
        training_main.main()


def test_integer_valued_scientific_total_steps_is_normalized_to_int(
    monkeypatch, tmp_path
):
    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path,
        experiment_updates={"overrides_alg": {"total_steps": 4.5e6}},
    )
    env = _Env()
    model = _Model()
    monkeypatch.setattr(training_main, "build_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(
        training_main,
        "initialize_alg",
        lambda *_args, **_kwargs: (model, False, "Dummy"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    training_main.main()

    assert model.learn_calls == [{"total_timesteps": 4_500_000}]
    assert isinstance(model.learn_calls[0]["total_timesteps"], int)
    assert env.closed


def test_trial_index_only_offsets_first_algorithm_and_num_runs_counts_cells(
    monkeypatch, tmp_path
):
    experiment_path, algorithm_dir, experiment = _write_configs(
        tmp_path, experiment_updates={"trials": 3}
    )
    second_config = json.loads(
        (algorithm_dir / "Config.json").read_text(encoding="utf-8")
    )
    second_config["seed"] = 101
    (algorithm_dir / "Second.json").write_text(
        json.dumps(second_config), encoding="utf-8"
    )
    experiment["configs"] = ["Config", "Second"]
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")

    environments = []
    executed_cells = []

    def fake_build_env(*_args, **_kwargs):
        env = _Env()
        environments.append(env)
        return env

    def fake_initialize(_algorithm, _params, _env, **kwargs):
        run_params = kwargs["full_run_params"]
        executed_cells.append((run_params["name"], run_params["seed"]))
        return _Model(), False, "Dummy"

    monkeypatch.setattr(training_main, "build_env", fake_build_env)
    monkeypatch.setattr(training_main, "initialize_alg", fake_initialize)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            experiment_path,
            algorithm_dir,
            tmp_path / "output",
            "--trial-index",
            "2",
            "--num-runs",
            "3",
        ),
    )

    training_main.main()

    assert executed_cells == [("Config", 13), ("Second", 101), ("Second", 102)]
    assert len(environments) == 3
    assert all(env.closed for env in environments)


def test_empty_experiment_name_cannot_overwrite_log_root(monkeypatch, tmp_path):
    experiment_path, algorithm_dir, experiment = _write_configs(
        tmp_path, experiment_updates={"logs": "overwrite"}
    )
    empty_name_path = tmp_path / ".json"
    empty_name_path.write_text(json.dumps(experiment), encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(empty_name_path, algorithm_dir, output_dir),
    )

    with pytest.raises(ValueError, match="non-empty, plain"):
        training_main.main()

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_overwrite_refuses_symlink_resolving_to_log_root(monkeypatch, tmp_path):
    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path, experiment_updates={"logs": "overwrite"}
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    (output_dir / "Experiment").symlink_to(output_dir, target_is_directory=True)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, output_dir),
    )

    with pytest.raises(ValueError, match="unsafe experiment directory"):
        training_main.main()

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_overwrite_safe_rejects_mismatched_existing_provenance(
    monkeypatch, tmp_path
):
    current_env = {"version": "new"}
    experiment_path, algorithm_dir, experiment = _write_configs(
        tmp_path,
        experiment_updates={
            "trials": 2,
            "logs": "overwrite-safe",
            "env_params": current_env,
        },
    )
    output_dir = tmp_path / "output"
    experiment_dir = output_dir / "Experiment"
    experiment_dir.mkdir(parents=True)
    stale = {**experiment, "env_params": {"version": "old"}}
    (experiment_dir / "settings.json").write_text(
        json.dumps(stale), encoding="utf-8"
    )
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail("mismatched provenance built an environment"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            experiment_path,
            algorithm_dir,
            output_dir,
            "--trial-index",
            "1",
        ),
    )

    with pytest.raises(ValueError, match="differs from the current"):
        training_main.main()

    assert json.loads((experiment_dir / "settings.json").read_text()) == stale
    assert not (experiment_dir / "Config_1").exists()


def test_overwrite_safe_rejects_duplicate_keys_in_existing_provenance(
    monkeypatch, tmp_path
):
    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path, experiment_updates={"logs": "overwrite-safe"}
    )
    output_dir = tmp_path / "output"
    experiment_dir = output_dir / "Experiment"
    experiment_dir.mkdir(parents=True)
    settings = experiment_dir / "settings.json"
    settings.write_text(
        '{"configs":["Config"],"trials":1,"trials":2,"logs":"overwrite-safe"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        training_main,
        "build_env",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid existing provenance built an environment"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, output_dir),
    )

    with pytest.raises(ValueError, match="could not verify its settings.json"):
        training_main.main()

    assert '"trials":1,"trials":2' in settings.read_text(encoding="utf-8")


def test_timestamp_logging_allocates_a_numbered_directory_on_collision(
    monkeypatch, tmp_path
):
    experiment_path, algorithm_dir, experiment = _write_configs(
        tmp_path, experiment_updates={"logs": "timestamp"}
    )
    output_dir = tmp_path / "output"
    colliding = output_dir / "Experiment_STAMP"
    colliding.mkdir(parents=True)
    marker = colliding / "existing.txt"
    marker.write_text("keep", encoding="utf-8")
    env = _Env()
    model = _Model()
    monkeypatch.setattr(training_main, "build_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(
        training_main,
        "initialize_alg",
        lambda *_args, **_kwargs: (model, False, "Dummy"),
    )
    monkeypatch.setattr(training_main, "datetime_stamp", lambda: "STAMP")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, output_dir),
    )

    training_main.main()

    allocated = output_dir / "Experiment_STAMP_1"
    assert marker.read_text(encoding="utf-8") == "keep"
    assert json.loads((allocated / "settings.json").read_text()) == experiment
    assert (allocated / "Config_0" / "alg_settings.json").is_file()
    assert model.learn_calls == [{"total_timesteps": 17}]
    assert env.closed


@pytest.mark.parametrize("failure_point", ["initialize", "metadata", "checkpoint"])
def test_environment_closes_when_pretraining_setup_fails(
    monkeypatch, tmp_path, failure_point
):
    updates = {}
    if failure_point == "checkpoint":
        updates.update(checkpoint_every=1, save_strat="latest")
    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path, experiment_updates=updates
    )
    env = _Env()
    checkpoint_error = (
        RuntimeError("checkpoint failed") if failure_point == "checkpoint" else None
    )
    model = _Model(checkpoint_error=checkpoint_error)
    monkeypatch.setattr(training_main, "build_env", lambda *_args, **_kwargs: env)
    if failure_point == "initialize":
        monkeypatch.setattr(
            training_main,
            "initialize_alg",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("initialize failed")
            ),
        )
    else:
        monkeypatch.setattr(
            training_main,
            "initialize_alg",
            lambda *_args, **_kwargs: (model, False, "Dummy"),
        )
    if failure_point == "metadata":
        monkeypatch.setattr(
            training_main,
            "_resolved_runtime_metadata",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("metadata failed")
            ),
        )
    monkeypatch.setattr(training_main, "datetime_stamp", lambda: "STAMP")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(RuntimeError, match=failure_point):
        training_main.main()

    assert env.closed
    if failure_point != "initialize":
        assert model.closed


def test_environment_cleanup_failure_does_not_mask_initialization_failure(
    monkeypatch, tmp_path
):
    experiment_path, algorithm_dir, _ = _write_configs(tmp_path)
    env = _Env(close_error=OSError("close failed"))
    monkeypatch.setattr(training_main, "build_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(
        training_main,
        "initialize_alg",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("initialize failed")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(RuntimeError, match="initialize failed") as captured:
        training_main.main()

    assert env.closed
    assert str(captured.value) == "initialize failed"


def test_normal_finalizer_attempts_every_cleanup_and_preserves_training_error(
    monkeypatch, tmp_path
):
    class FailingWriter:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1
            raise OSError("writer close failed")

    class FailingLogger:
        def __init__(self, **_kwargs):
            self.close_calls = 0

        def reset(self):
            return None

        def set_log_dir(self, _path):
            return None

        def close(self):
            self.close_calls += 1
            raise OSError("logger close failed")

    experiment_path, algorithm_dir, _ = _write_configs(
        tmp_path, experiment_updates={"logs": "timestamp"}
    )
    env = _Env(close_error=OSError("environment close failed"))
    writer = FailingWriter()
    logger = FailingLogger()
    model = _Model(
        writer=writer,
        learn_error=RuntimeError("training failed"),
        close_error=OSError("model close failed"),
    )
    monkeypatch.setattr(training_main, "build_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(
        training_main,
        "initialize_alg",
        lambda *_args, **_kwargs: (model, False, "Dummy"),
    )
    monkeypatch.setattr(training_main, "TrainingLogger", lambda **_kwargs: logger)
    monkeypatch.setattr(training_main, "datetime_stamp", lambda: "STAMP")
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(experiment_path, algorithm_dir, tmp_path / "output"),
    )

    with pytest.raises(RuntimeError, match="training failed") as captured:
        training_main.main()

    assert str(captured.value) == "training failed"
    assert logger.close_calls == 1
    assert writer.shutdown_calls == 1
    assert model.closed
    assert env.closed
    assert getattr(captured.value, "__notes__", ()) == [
        "Additional cleanup failure: logger close failed",
        "Additional cleanup failure: writer close failed",
        "Additional cleanup failure: model close failed",
        "Additional cleanup failure: environment close failed",
    ]
