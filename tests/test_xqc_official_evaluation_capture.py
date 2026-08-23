import argparse
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ORACLE_PATH = (
    Path(__file__).resolve().parent / "oracles" / "run_official_xqc_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "xqc_official_evaluation_capture_oracle",
    ORACLE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def _capture_args(**overrides):
    values = {
        "evaluation_csv": None,
        "base_seed": None,
        "num_seeds": None,
        "action_repeat": None,
        "expected_evaluation_rows": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _canonical_args(**overrides):
    values = {
        "canonical_wandb": False,
        "task": None,
        "implementation": None,
        "source_sha": None,
        "base_seed": None,
        "num_seeds": None,
        "action_repeat": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_evaluation_capture_arguments_are_all_or_none(tmp_path):
    assert ORACLE._validate_evaluation_capture_args(_capture_args()) is False

    with pytest.raises(SystemExit, match="require --evaluation-csv"):
        ORACLE._validate_evaluation_capture_args(_capture_args(base_seed=0))
    with pytest.raises(SystemExit, match="requires --base-seed"):
        ORACLE._validate_evaluation_capture_args(
            _capture_args(
                evaluation_csv=tmp_path / "eval.csv",
                num_seeds=2,
                action_repeat=2,
                expected_evaluation_rows=22,
            )
        )
    with pytest.raises(SystemExit, match="action-repeat must be positive"):
        ORACLE._validate_evaluation_capture_args(
            _capture_args(
                evaluation_csv=tmp_path / "eval.csv",
                base_seed=0,
                num_seeds=2,
                action_repeat=0,
                expected_evaluation_rows=22,
            )
        )
    with pytest.raises(SystemExit, match="num-seeds must be positive"):
        ORACLE._validate_evaluation_capture_args(
            _capture_args(
                evaluation_csv=tmp_path / "eval.csv",
                base_seed=0,
                num_seeds=0,
                action_repeat=2,
                expected_evaluation_rows=22,
            )
        )
    with pytest.raises(SystemExit, match="expected-evaluation-rows must be positive"):
        ORACLE._validate_evaluation_capture_args(
            _capture_args(
                evaluation_csv=tmp_path / "eval.csv",
                base_seed=0,
                num_seeds=2,
                action_repeat=2,
                expected_evaluation_rows=0,
            )
        )
    with pytest.raises(SystemExit, match="must be divisible by --num-seeds"):
        ORACLE._validate_evaluation_capture_args(
            _capture_args(
                evaluation_csv=tmp_path / "eval.csv",
                base_seed=0,
                num_seeds=2,
                action_repeat=2,
                expected_evaluation_rows=21,
            )
        )

    assert ORACLE._validate_evaluation_capture_args(
        _capture_args(
            evaluation_csv=tmp_path / "eval.csv",
            base_seed=0,
            num_seeds=2,
            action_repeat=2,
            expected_evaluation_rows=22,
        )
    ) is True


def test_evaluation_capture_writes_two_seed_paper_schedule_durably(
    tmp_path,
    monkeypatch,
):
    fsynced_descriptors = []
    monkeypatch.setattr(
        ORACLE.os,
        "fsync",
        lambda descriptor: fsynced_descriptors.append(descriptor),
    )
    path = tmp_path / "official-evaluations.csv"
    capture = ORACLE._EvaluationCsvCapture(
        path,
        base_seed=0,
        num_seeds=2,
        action_repeat=2,
        expected_rows=22,
    )
    try:
        capture.record(20_000, {"critic_loss": np.array([1.0, 2.0])})
        assert len(fsynced_descriptors) == 1

        raw_frames = [2, *range(100_000, 1_000_001, 100_000)]
        for evaluation_index, raw_frame in enumerate(raw_frames):
            capture.record(
                raw_frame,
                {
                    "return": np.array(
                        [evaluation_index + 0.25, evaluation_index + 0.75],
                        dtype=np.float64,
                    )
                },
            )

        # Header creation and every evaluation event are independently durable.
        assert len(fsynced_descriptors) == 1 + len(raw_frames)
        capture.assert_expected_rows()
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    finally:
        capture.close()

    assert len(rows) == 22
    assert tuple(rows[0]) == ORACLE.EVALUATION_CSV_FIELDS
    assert rows[0] == {
        "implementation": "official-jax",
        "source_commit": ORACLE.OFFICIAL_COMMIT,
        "base_seed": "0",
        "num_seeds": "2",
        "seed_index": "0",
        "seed": "0",
        "evaluation_index": "0",
        "decision_step": "1",
        "raw_frame": "2",
        "paper_raw_frame": "0",
        "action_repeat": "2",
        "return": "0.25",
    }
    assert rows[1]["seed_index"] == "1"
    assert rows[1]["seed"] == "1"
    assert rows[-1]["evaluation_index"] == "10"
    assert rows[-1]["decision_step"] == "500000"
    assert rows[-1]["raw_frame"] == "1000000"
    assert rows[-1]["paper_raw_frame"] == "1000000"
    assert rows[-1]["seed"] == "1"
    assert rows[-1]["return"] == "10.75"


def test_evaluation_capture_exclusively_creates_and_checks_row_count(tmp_path):
    path = tmp_path / "existing.csv"
    path.write_text("user data\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ORACLE._EvaluationCsvCapture(
            path,
            base_seed=0,
            num_seeds=2,
            action_repeat=2,
            expected_rows=2,
        )
    assert path.read_text(encoding="utf-8") == "user data\n"

    incomplete = ORACLE._EvaluationCsvCapture(
        tmp_path / "incomplete.csv",
        base_seed=4,
        num_seeds=2,
        action_repeat=2,
        expected_rows=3,
    )
    try:
        incomplete.record(2, {"return": [1.0, 2.0]})
        with pytest.raises(SystemExit, match="wrote 2 rows; expected 3"):
            incomplete.assert_expected_rows()
    finally:
        incomplete.close()


@pytest.mark.parametrize(
    ("step", "returns", "message"),
    [
        (3, [1.0], "not divisible"),
        (float("nan"), [1.0], "finite, non-negative integer"),
        (2, [], "non-empty vector"),
        (2, [[1.0]], "non-empty vector"),
        (2, [1.0, 2.0], "return count"),
        (2, [float("inf")], "non-finite"),
        (2, ["bad"], "real numeric"),
        (2, [1.0 + 2.0j], "real numeric"),
    ],
)
def test_evaluation_capture_rejects_ambiguous_or_invalid_rows(
    tmp_path,
    step,
    returns,
    message,
):
    capture = ORACLE._EvaluationCsvCapture(
        tmp_path / "invalid.csv",
        base_seed=0,
        num_seeds=1,
        action_repeat=2,
        expected_rows=1,
    )
    try:
        with pytest.raises(SystemExit, match=message):
            capture.record(step, {"return": returns})
        assert capture.row_count == 0
        assert capture.evaluation_count == 0
    finally:
        capture.close()


def test_evaluation_logging_patch_forwards_and_restores_on_failure():
    forwarded = []
    captured = []

    def original(step, infos, fps=30):
        forwarded.append((step, infos, fps))
        return "logged"

    module = SimpleNamespace(log_multiple_seeds_to_wandb=original)
    capture = SimpleNamespace(
        record=lambda step, infos: captured.append((step, infos))
    )

    with pytest.raises(RuntimeError, match="official failure"):
        with ORACLE._patched_evaluation_logging(module, capture):
            assert module.log_multiple_seeds_to_wandb(
                2,
                {"return": [1.0, 2.0]},
                fps=17,
            ) == "logged"
            assert module.log_multiple_seeds_to_wandb is not original
            raise RuntimeError("official failure")

    assert module.log_multiple_seeds_to_wandb is original
    assert captured == [(2, {"return": [1.0, 2.0]})]
    assert forwarded == [(2, {"return": [1.0, 2.0]}, 17)]


def test_official_main_runs_directly_when_capture_is_absent():
    calls = []

    def official_main():
        calls.append("official")

    ORACLE._run_official_main(official_main)

    assert calls == ["official"]


def test_canonical_wandb_metadata_requires_one_actual_seed_and_exact_source(
    monkeypatch,
):
    assert ORACLE._canonical_wandb_metadata(_canonical_args()) is None
    monkeypatch.setenv("XQC_COMPARISON_ID", "humanoid-parity")
    valid = _canonical_args(
        canonical_wandb=True,
        task="humanoid-walk",
        implementation="official-jax",
        source_sha=ORACLE.OFFICIAL_COMMIT,
        base_seed=1,
        num_seeds=1,
        action_repeat=2,
    )
    assert ORACLE._canonical_wandb_metadata(valid) == {
        "implementation": "official-jax",
        "seed": 1,
        "task": "humanoid-walk",
        "source_sha": ORACLE.OFFICIAL_COMMIT,
        "comparison_id": "humanoid-parity",
        "action_repeat": 2,
    }

    with pytest.raises(SystemExit, match="each actual seed"):
        ORACLE._canonical_wandb_metadata(
            _canonical_args(**{**vars(valid), "num_seeds": 2})
        )
    with pytest.raises(SystemExit, match="requires --source-sha"):
        ORACLE._canonical_wandb_metadata(
            _canonical_args(**{**vars(valid), "source_sha": None})
        )
    with pytest.raises(SystemExit, match="requires --implementation official-jax"):
        ORACLE._canonical_wandb_metadata(
            _canonical_args(**{**vars(valid), "implementation": "action-pytorch"})
        )


def test_canonical_wandb_initialization_labels_run_and_defines_raw_frame_axis(
    monkeypatch,
):
    init_calls = []

    class FakeRun:
        def __init__(self):
            self.definitions = []

        def define_metric(self, name, **kwargs):
            self.definitions.append((name, kwargs))

    run = FakeRun()

    def original_init(*args, **kwargs):
        init_calls.append((args, kwargs))
        return run

    module = SimpleNamespace(init=original_init)
    metadata = {
        "implementation": "official-jax",
        "seed": 1,
        "task": "humanoid-walk",
        "source_sha": ORACLE.OFFICIAL_COMMIT,
        "comparison_id": "humanoid-parity",
        "action_repeat": 2,
    }
    monkeypatch.setenv("WANDB_RUN_GROUP", "humanoid-parity-official-jax")
    with ORACLE._patched_wandb_initialization(module, metadata):
        assert module.init(
            config={
                "upstream": 7,
                "seed": 1,
                "num_seeds": 1,
                "env": {"name": "humanoid-walk", "action_repeat": 2},
            }
        ) is run
        assert module.init is not original_init
    assert module.init is original_init

    assert init_calls == [
        (
            (),
            {
                "config": {
                    "upstream": 7,
                    "seed": 1,
                    "num_seeds": 1,
                    "env": {"name": "humanoid-walk", "action_repeat": 2},
                    **metadata,
                },
                "name": "xqc-official-jax-humanoid-walk-seed1",
                "job_type": "official-jax",
                "group": "humanoid-parity-official-jax",
            },
        )
    ]
    assert run.definitions == [
        ("comparison/raw_frame", {}),
        ("comparison/decision_step", {"step_metric": "comparison/raw_frame"}),
        ("comparison/train_return", {"step_metric": "comparison/raw_frame"}),
        ("comparison/eval_return", {"step_metric": "comparison/raw_frame"}),
    ]


def test_canonical_training_capture_tracks_termination_and_truncation_exactly():
    class FakeWandb:
        def __init__(self):
            self.logs = []

        def log(self, payload, step=None):
            self.logs.append((step, dict(payload)))

    class FakeParallelEnv:
        queued_results = []

        def __init__(self, label):
            self.label = label

        def step(self, _actions):
            return self.queued_results.pop(0)

    def result(reward, terminated=False, truncated=False):
        return (
            np.zeros((1, 2), dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.array([terminated]),
            np.array([truncated]),
            np.zeros(1, dtype=np.float32),
        )

    fake_wandb = FakeWandb()
    logger = ORACLE._CanonicalWandbLogger(
        fake_wandb,
        action_repeat=2,
    )
    original_init = FakeParallelEnv.__init__
    original_step = FakeParallelEnv.step
    FakeParallelEnv.queued_results = [
        result(1.0),
        result(2.0, terminated=True),
        result(100.0, truncated=True),
        result(5.0, truncated=True),
    ]
    with ORACLE._patched_training_returns(
        FakeParallelEnv,
        logger,
        num_seeds=1,
    ):
        train_env = FakeParallelEnv("train")
        eval_env = FakeParallelEnv("eval")
        train_env.step(None)
        train_env.step(None)
        eval_env.step(None)
        train_env.step(None)

    assert FakeParallelEnv.__init__ is original_init
    assert FakeParallelEnv.step is original_step
    assert fake_wandb.logs == [
        (
            None,
            {
                "comparison/raw_frame": 4,
                "comparison/decision_step": 2,
                "comparison/train_return": 3.0,
                "episode/return": 3.0,
                "episode/len": 2,
                "episode/terminated": 1,
                "episode/truncated": 0,
            },
        ),
        (
            None,
            {
                "comparison/raw_frame": 6,
                "comparison/decision_step": 3,
                "comparison/train_return": 5.0,
                "episode/return": 5.0,
                "episode/len": 1,
                "episode/terminated": 0,
                "episode/truncated": 1,
            },
        ),
    ]


def test_canonical_evaluation_logging_uses_same_labels_and_raw_frame_step():
    class FakeWandb:
        def __init__(self):
            self.logs = []

        def log(self, payload, step=None):
            self.logs.append((step, dict(payload)))

    forwarded = []
    fake_wandb = FakeWandb()

    def original_logging(step, infos, fps=30):
        forwarded.append((step, infos, fps))
        fake_wandb.log({"seed0/return": float(infos["return"][0])}, step=step)

    module = SimpleNamespace(
        log_multiple_seeds_to_wandb=original_logging
    )
    logger = ORACLE._CanonicalWandbLogger(
        fake_wandb,
        action_repeat=2,
    )
    with ORACLE._patched_evaluation_logging(
        module,
        canonical_logger=logger,
    ):
        module.log_multiple_seeds_to_wandb(
            100_000,
            {"return": np.array([321.5])},
            fps=19,
        )

    assert len(forwarded) == 1
    assert forwarded[0][0] == 100_000
    np.testing.assert_array_equal(forwarded[0][1]["return"], [321.5])
    assert forwarded[0][2] == 19
    assert fake_wandb.logs == [
        (
            None,
            {
                "comparison/raw_frame": 100_000,
                "comparison/decision_step": 50_000,
                "comparison/eval_return": 321.5,
                "eval/episode_reward": 321.5,
            },
        ),
        (100_000, {"seed0/return": 321.5}),
    ]
