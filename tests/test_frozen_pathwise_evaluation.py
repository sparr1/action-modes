import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluate_ambi_checkpoint import evaluate_matrix
from tests.test_ambi_root_local_sac import _tiny_component_model
from utils.ambi_research import (
    PresetMatrixError, load_preset_matrix, normalize_selectors,
    resolve_preset, validate_preset_matrix,
)
from utils.eval_series_data import planner_identity


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "configs" / "research"
MATRIX = RESEARCH / "ambi_checkpoint_pathwise_eval_consecutive.json"
PATHWISE = "frozen_pathwise_v1"
SELECTORS = ["controller/pathwise", "controller/detached"]


def test_matrix_preserves_checkpoint_settings_and_pairs_only_the_new_controllers():
    saved = json.loads(
        (ROOT / "configs/dmcontrol/algs/ambi_inner_entropy_posttanh_outer_adapt.json").read_text()
    )
    context = SimpleNamespace(
        source=Path("metadata.json"),
        trial_run_params={**saved, "env": "DMControl-v0"},
        experiment_params={"env_params": {"task": "humanoid-walk", "obs": "state"}},
    )
    matrix = load_preset_matrix(MATRIX)
    assert normalize_selectors(matrix) == SELECTORS
    assert matrix["checkpoint_contract"]["step"] == 1_000_000
    for selector, detached in zip(SELECTORS, (False, True)):
        resolved = resolve_preset(MATRIX, selector, checkpoint_context=context)
        evaluation = resolved["evaluation"]
        assert evaluation["execution_backend"] == PATHWISE
        assert evaluation["pathwise"] == {
            "rollouts_per_update": 32, "detach_dynamics": detached,
        }
        assert evaluation["environment_seed"] == evaluation["controller_seed"] == 55
        assert evaluation["episode_sampling"] == "consecutive"
        assert evaluation["num_episodes"] == 10
        assert evaluation["max_steps"] == 500
        assert evaluation["execution_action"] == "mean"
        assert evaluation["recording"] == "returns"

        params = resolved["algorithm_config"]["alg_params"]
        assert params == {**saved["alg_params"], **matrix["shared_alg_params"]}
        assert params["inner_critic_updates_per_round"] == 0
        actor_updates = params["inner_rounds"] * params["inner_actor_updates_per_round"]
        assert actor_updates == 32
        assert actor_updates * evaluation["pathwise"]["rollouts_per_update"] * params["inner_rollout_horizon"] == 3072
        assert not params["compile"] and not params["inner_cuda_graph"]

    legacy = load_preset_matrix(RESEARCH / "ambi_checkpoint_graph_eval_consecutive.json")
    assert normalize_selectors(legacy) == ["controller/prior", "controller/graph"]


@pytest.mark.parametrize("override", [
    {"rollouts_per_update": 0, "detach_dynamics": False},
    {"rollouts_per_update": True, "detach_dynamics": False},
    {"rollouts_per_update": 32, "detach_dynamics": "false"},
    {"rollouts_per_update": 32},
])
def test_matrix_rejects_invalid_pathwise_options(override):
    matrix = json.loads(MATRIX.read_text())
    matrix["comparisons"]["controller"]["variants"]["pathwise"]["evaluation"]["pathwise"] = override
    with pytest.raises(PresetMatrixError):
        validate_preset_matrix(matrix)


def test_closed_loop_pair_on_tiny_checkpoint(tmp_path):
    model = _tiny_component_model(
        inner_rounds=1, inner_actor_updates_per_round=1, inner_critic_updates_per_round=0,
        inner_finite_horizon=True, inner_temperature_mode="inherit_outer",
        ent_coef=0.0021,
        outer_actor_entropy_mode="squashed", inner_actor_entropy_mode="squashed",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        inner_actor_loss_scale_scope="action", inner_actor_loss_scale_update="per_update",
        outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
    )
    checkpoint = tmp_path / "checkpoint"
    metadata = {
        "schema_version": 1,
        "checkpoint": {"kind": "periodic", "step": 1, "episode": 1, "best_window": 10},
        "trial_run_params": {
            **model.run_params,
            "alg": "AMBITDMPC2/AMBITDMPC2",
            "env": "Pendulum-v1",
            "alg_params": model.custom_params,
        },
        "experiment_params": {"env_params": {"max_episode_steps": 2}},
    }
    try:
        model.agent.save(checkpoint)
    finally:
        model.close()
    checkpoint.with_suffix(".metadata.json").write_text(json.dumps(metadata))

    matrix = json.loads(MATRIX.read_text())
    matrix["checkpoint_contract"] = {
        "step": 1, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    matrix.pop("source_run")
    for variant in matrix["comparisons"]["controller"]["variants"].values():
        variant["evaluation"]["pathwise"]["rollouts_per_update"] = 2
    smoke_matrix = tmp_path / "matrix.json"
    smoke_matrix.write_text(json.dumps(matrix))
    output = tmp_path / "bundle"
    payload = evaluate_matrix(
        smoke_matrix, checkpoint, num_episodes=2, max_steps=2,
        device="cpu", bundle_dir=output, stage_results=False,
    )

    pathwise, detached = payload["results"]
    assert [result["selector"] for result in payload["results"]] == SELECTORS
    for result, detach_dynamics in zip((pathwise, detached), (False, True)):
        assert result["execution_backend"] == PATHWISE
        assert result["pathwise"]["detach_dynamics"] is detach_dynamics
        assert result["pathwise_solves"] == 4
        assert result["outer_state_unchanged"]
        assert result["outer_updates_before"] == result["outer_updates_after"]
        assert [row["episode_index"] for row in result["episodes"]] == [0, 1]
        assert all(row["length"] == 2 and not row["truncated_by_evaluator"] for row in result["episodes"])
        assert result["paired_return_delta_vs_reference"]["count"] == 2
        assert result["model_metrics"]["inner_actor_optimizer_steps"]["mean"] == 1
        assert result["model_metrics"]["inner_model_steps"]["mean"] == 4
        assert result["model_metrics"]["inner_critic_optimizer_steps"]["mean"] == 0

    assert [row["initial_observation_sha256"] for row in pathwise["episodes"]] == [
        row["initial_observation_sha256"] for row in detached["episodes"]
    ]
    identities = [
        planner_identity(result["resolved_config"], result, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
        for result in payload["results"]
    ]
    assert all(identity["type"] == "pathwise" for identity in identities)
    assert identities[0] != identities[1]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
