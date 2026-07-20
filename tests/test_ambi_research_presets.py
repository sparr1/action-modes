import json
import warnings
from pathlib import Path

import gymnasium as gym
import pytest

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
MATRIX = ROOT / "configs/ambi/legacy/ambi_inner_decoupling.json"


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
