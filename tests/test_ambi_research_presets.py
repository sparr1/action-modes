import json
import warnings
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import pytest

import evaluate_ambi_checkpoint as evaluator
from evaluate_ambi_checkpoint import (
    _attach_paired_return_deltas,
    _validate_frozen_selection,
    build_parser,
)
from RL.AMBITDMPC2 import AMBITDMPC2
from utils.ambi_research import (
    PresetMatrixError,
    list_preset_selectors,
    load_preset_matrix,
    materialize_presets,
    normalize_selectors,
    resolve_preset,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_inner_decoupling.json"


def _build_cfg(resolved):
    run_config = resolved["algorithm_config"]
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        **run_config,
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 12,
    }
    algorithm.custom_params = run_config["alg_params"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return algorithm._build_cfg(
                {"device": "cpu", **run_config["alg_params"]}
            )
    finally:
        algorithm.env.close()


def test_matrix_covers_each_decoupling_axis_and_has_paired_operator_defaults():
    matrix = load_preset_matrix(MATRIX)
    assert {
        "inner_operator",
        "q_representation",
        "adapted_components",
        "temperature",
        "behavior_exploration",
        "execution_noise",
        "model_compute",
        "optimizer_compute",
        "lifecycle",
    } <= set(matrix["comparisons"])
    assert normalize_selectors(matrix) == [
        "inner_operator/none",
        "inner_operator/sac",
        "inner_operator/td3",
        "inner_operator/mppi",
    ]


def test_every_checked_in_preset_materializes_to_a_valid_ambi_config():
    matrix = load_preset_matrix(MATRIX)
    for selector in list_preset_selectors(matrix):
        resolved = resolve_preset(MATRIX, selector, matrix=matrix)
        cfg = _build_cfg(resolved)
        assert cfg.mpc is False
        assert cfg.inner_operator in {"none", "sac", "td3", "mppi"}


def test_resolution_is_complete_and_does_not_mutate_base_or_matrix():
    matrix = load_preset_matrix(MATRIX)
    before = json.dumps(matrix, sort_keys=True)
    scalar = resolve_preset(MATRIX, "q_representation/scalar_twin", matrix=matrix)
    distributional = resolve_preset(
        MATRIX, "q_representation/distributional_five", matrix=matrix
    )

    assert scalar["algorithm_config"]["alg"] == "AMBITDMPC2/AMBITDMPC2"
    assert scalar["algorithm_config"]["alg_params"]["num_q"] == 2
    assert distributional["algorithm_config"]["alg_params"]["num_q"] == 5
    assert distributional["algorithm_config"]["alg_params"]["q_representation"] == "distributional"
    assert json.dumps(matrix, sort_keys=True) == before


def test_materialized_presets_follow_existing_algorithm_json_shape(tmp_path):
    paths = materialize_presets(
        MATRIX,
        tmp_path,
        comparisons=["inner_operator"],
    )
    assert [path.name for path in paths] == [
        "inner_operator__none.json",
        "inner_operator__sac.json",
        "inner_operator__td3.json",
        "inner_operator__mppi.json",
    ]
    for path in paths:
        payload = json.loads(path.read_text())
        assert payload["alg"] == "AMBITDMPC2/AMBITDMPC2"
        assert isinstance(payload["alg_params"], dict)
        assert "inner_operator" in payload["alg_params"]
    experiment = json.loads((tmp_path / "AMBIResearchExperiment.json").read_text())
    assert experiment["env_params"]["terminate_when_unhealthy"] is False
    assert experiment["configs"] == [path.stem for path in paths]


def test_unknown_or_duplicate_preset_input_fails_clearly(tmp_path):
    matrix = load_preset_matrix(MATRIX)
    with pytest.raises(PresetMatrixError, match="Unknown preset"):
        normalize_selectors(matrix, ["inner_operator/not_real"])

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(PresetMatrixError, match="duplicate JSON key"):
        load_preset_matrix(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version": 1, "value": NaN}')
    with pytest.raises(PresetMatrixError, match="non-finite JSON number"):
        load_preset_matrix(nonfinite)


def test_frozen_evaluator_cli_parses_selection_and_reproducibility_controls():
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--comparison",
            "inner_operator",
            "--seeds",
            "7",
            "11",
            "--controller-seed",
            "13",
            "--max-steps",
            "25",
            "--device",
            "cpu",
        ]
    )
    assert args.checkpoint == Path("checkpoint.pt")
    assert args.comparisons == ["inner_operator"]
    assert args.seeds == [7, 11]
    assert args.controller_seed == 13
    assert args.max_steps == 25
    assert args.device == "cpu"


def test_frozen_evaluator_output_is_no_clobber_by_default_and_atomic_on_overwrite(
    monkeypatch, tmp_path
):
    output = tmp_path / "results" / "evaluation.json"
    output.parent.mkdir()
    output.write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        evaluator._write_output_atomic(output, "new\n", overwrite=False)
    assert output.read_text(encoding="utf-8") == "old\n"

    replacements = []
    real_replace = evaluator.os.replace

    def tracking_replace(source, target):
        replacements.append((Path(source), Path(target)))
        return real_replace(source, target)

    monkeypatch.setattr(evaluator.os, "replace", tracking_replace)
    evaluator._write_output_atomic(output, "new\n", overwrite=True)

    assert output.read_text(encoding="utf-8") == "new\n"
    assert replacements and replacements[-1][1] == output
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_frozen_evaluator_preflights_existing_output_before_evaluation(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "evaluation.json"
    output.write_text("keep\n", encoding="utf-8")
    monkeypatch.setattr(
        evaluator,
        "evaluate_matrix",
        lambda *_args, **_kwargs: pytest.fail("evaluation ran before output preflight"),
    )

    with pytest.raises(SystemExit):
        evaluator.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
            ]
        )

    assert output.read_text(encoding="utf-8") == "keep\n"


def test_frozen_evaluator_reports_seed_paired_reference_deltas():
    results = [
        {
            "comparison": "operator",
            "variant": "sac",
            "reference_variant": "sac",
            "episodes": [
                {"seed": 2, "return": 10.0},
                {"seed": 1, "return": 5.0},
            ],
        },
        {
            "comparison": "operator",
            "variant": "mppi",
            "reference_variant": "sac",
            "episodes": [
                {"seed": 1, "return": 8.0},
                {"seed": 2, "return": 9.0},
            ],
        },
    ]
    _attach_paired_return_deltas(results)
    assert results[0]["paired_return_delta_vs_reference"]["mean"] == 0.0
    assert results[1]["paired_return_delta_vs_reference"]["mean"] == 1.0


def test_frozen_selection_rejects_train_only_and_mixed_architecture_axes():
    matrix = load_preset_matrix(MATRIX)
    execution = resolve_preset(MATRIX, "execution_noise/mean", matrix=matrix)
    with pytest.raises(ValueError, match="cannot be used in frozen evaluation"):
        _validate_frozen_selection(matrix, [execution])

    scalar = resolve_preset(MATRIX, "q_representation/scalar_twin", matrix=matrix)
    distributional = resolve_preset(
        MATRIX, "q_representation/distributional_five", matrix=matrix
    )
    with pytest.raises(ValueError, match="different critic architectures"):
        _validate_frozen_selection(matrix, [scalar, distributional])

    state = resolve_preset(MATRIX, "inner_operator/sac", matrix=matrix)
    rgb = deepcopy(state)
    rgb["selector"] = "synthetic/rgb"
    rgb["environment"]["params"]["obs"] = "rgb"
    rgb["algorithm_config"]["alg_params"]["obs"] = "rgb"
    with pytest.raises(ValueError, match="different observation contracts"):
        _validate_frozen_selection(matrix, [state, rgb])


def test_frozen_evaluation_closes_model_and_env_without_masking_primary_error(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    class Space:
        def seed(self, _seed):
            return None

    class Env:
        def __init__(self):
            self.action_space = Space()
            self.observation_space = Space()
            self.close_calls = 0

        def reset(self, *, seed=None):
            return 0, {}

        def close(self):
            self.close_calls += 1
            raise OSError("environment close failed")

    class Model:
        def __init__(self):
            self.agent = type("Agent", (), {"num_updates": 0})()
            self.close_calls = 0

        def predict(self, *_args, **_kwargs):
            raise RuntimeError("evaluation failed")

        def close(self):
            self.close_calls += 1
            raise OSError("model close failed")

    env = Env()
    model = Model()
    monkeypatch.setattr(evaluator, "_make_env", lambda _resolved: env)
    monkeypatch.setattr(
        evaluator,
        "_initialize_frozen_model",
        lambda *_args, **_kwargs: (model, {"alg_params": {}}),
    )
    monkeypatch.setattr(evaluator, "_outer_state_digest", lambda _model: "digest")

    with pytest.raises(RuntimeError, match="evaluation failed") as captured:
        evaluator.evaluate_preset(
            {"selector": "inner_operator/sac"},
            checkpoint,
            [7],
            controller_seed=11,
        )

    assert str(captured.value) == "evaluation failed"
    assert model.close_calls == 1
    assert env.close_calls == 1
    assert getattr(captured.value, "__notes__", ()) == [
        "Additional cleanup failure: model close failed",
        "Additional cleanup failure: environment close failed",
    ]


def test_frozen_model_is_closed_if_checkpoint_load_is_interrupted(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    instances = []

    class InterruptedModel:
        def __init__(self, *_args, **_kwargs):
            self.close_calls = 0
            instances.append(self)

        def load(self, _path):
            raise KeyboardInterrupt()

        def close(self):
            self.close_calls += 1

    module = type("Module", (), {"InterruptedModel": InterruptedModel})()
    monkeypatch.setattr(evaluator.importlib, "import_module", lambda _name: module)
    resolved = {
        "selector": "synthetic/interrupted",
        "algorithm_config": {
            "alg": "Synthetic/InterruptedModel",
            "alg_params": {},
        },
        "environment": {"id": "Synthetic-v0"},
    }

    with pytest.raises(KeyboardInterrupt):
        evaluator._initialize_frozen_model(
            resolved,
            object(),
            checkpoint,
            controller_seed=3,
        )

    assert instances[0].close_calls == 1


def test_frozen_evaluator_closes_base_env_if_wrapper_setup_is_interrupted(
    monkeypatch,
):
    class Env:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            raise OSError("base env close failed")

    env = Env()
    monkeypatch.setattr(evaluator.gym, "make", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(
        "utils.core.setup_wrapper",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    resolved = {
        "algorithm_config": {
            "env_wrapper": {"name": "synthetic:Wrapper"},
        },
        "environment": {"id": "Synthetic-v0"},
    }

    with pytest.raises(KeyboardInterrupt) as captured:
        evaluator._make_env(resolved)

    assert env.close_calls == 1
    assert getattr(captured.value, "__notes__", ()) == [
        "Additional cleanup failure: base env close failed"
    ]
