import json

import pytest

from slurm.humanoid_walk_ablation import (
    GRAPH, MATRIX, expected_workload, validate_checkpoint, validate_result,
)
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.eval_series import _file_digest
from utils.evaluation_episodes import consecutive_protocol


@pytest.fixture
def checkpoint(tmp_path):
    weight = tmp_path / "walk.pth"
    weight.write_bytes(b"test weights")
    metadata = tmp_path / "walk.pth.metadata.json"
    metadata.write_text(json.dumps({
        "schema_version": 1,
        "checkpoint": {"kind": "step", "step": 1_000_000, "episode": 2000, "best_window": 20},
        "trial_run_params": {
            "name": "ambi_inner_entropy_posttanh_outer_adapt",
            "seed": 55, "env": "DMControl-v0", "alg": "AMBITDMPC2/AMBITDMPC2",
            "alg_params": {
                "inner_operator": "sac", "inner_solve_interval": 1,
                "inner_replay_sampling": "with_replacement", "inner_replay_capacity": 3072,
                "inner_critic_initialization": "prior", "inner_temperature_mode": "inherit_outer",
                "inner_q_actor_reduction": "mean_pair", "inner_q_target_reduction": "min_pair",
                "train_unroll_horizon": 3,
            },
        },
        "experiment_params": {"env_params": {"task": "humanoid-walk", "obs": "state"}},
    }))
    return weight, _file_digest(weight), _file_digest(metadata)


def test_matrix_resolves_18_unique_workloads(checkpoint):
    context = validate_checkpoint(*checkpoint)
    matrix = load_preset_matrix(MATRIX)
    ablation = matrix["comparisons"]["ablation"]
    assert ablation["reference"] == "j8"
    assert list(ablation["variants"]) == [
        "j0", "j1", "j2", "j4", "j6", "j8", "j10", "j12",
        "c0", "c4", "c8", "c12", "a1", "a2", "a6", "a8", "h1", "h5",
    ]
    fields = {
        "j": "inner_rounds", "c": "inner_critic_updates_per_round",
        "a": "inner_actor_updates_per_round", "h": "inner_rollout_horizon",
    }
    outcomes = set()
    for name, variant in ablation["variants"].items():
        resolved = resolve_preset(MATRIX, f"ablation/{name}", matrix=matrix, checkpoint_context=context)
        evaluation = resolved["evaluation"]
        assert evaluation["num_episodes"] == 10
        assert evaluation["max_steps"] == 500
        assert evaluation["controller_seed"] == evaluation["environment_seed"] == 55
        assert evaluation["execution_action"] == "mean"
        assert evaluation["recording"] == "returns"
        parameters = resolved["algorithm_config"]["alg_params"]
        workload = expected_workload(parameters)
        outcomes.add(tuple(workload.values()))
        assert parameters["inner_replay_capacity"] == 5120
        assert parameters["train_unroll_horizon"] == 3
        if name == "j0":
            assert evaluation["execution_backend"] == "native"
            assert parameters["inner_operator"] == "none"
            assert workload["inner_model_steps"] == 0
            continue
        assert evaluation["execution_backend"] == GRAPH
        assert parameters["compile"] and parameters["compile_strict"]
        assert parameters["critic_ensemble_backend"] == "vmap"
        assert parameters["inner_rollouts_per_round"] == 128
        assert parameters["inner_replay_sampling"] == "with_replacement"
        assert parameters["inner_temperature_mode"] == "inherit_outer"
        assert parameters["inner_q_actor_reduction"] == "mean_pair"
        assert parameters["inner_q_target_reduction"] == "min_pair"
        assert workload["inner_model_steps"] <= parameters["inner_replay_capacity"]
        assert parameters[fields[name[0]]] == int(name[1:])
        if name != "j8":
            assert variant["alg_params"] == {fields[name[0]]: int(name[1:])}
    assert len(outcomes) == 18


def test_checkpoint_hash_must_match(checkpoint):
    weight, _, metadata_hash = checkpoint
    with pytest.raises(ValueError, match="SHA-256"):
        validate_checkpoint(weight, "0" * 64, metadata_hash)


@pytest.mark.parametrize("setting", ["j0", "j12", "c0", "h1", "h5"])
def test_validation_requires_requested_workload_and_complete_stream(setting, checkpoint):
    context = validate_checkpoint(*checkpoint)
    resolved = resolve_preset(MATRIX, f"ablation/{setting}", checkpoint_context=context)
    parameters = resolved["algorithm_config"]["alg_params"]
    sampling = consecutive_protocol(resolved["evaluation"])
    episodes = [{
        "episode_index": ep, "length": 500, "return": 123.0, "environment_seed": 55,
        "controller_stream_seed": 55, "initial_observation_sha256": "1" * 64,
        "model_metrics": expected_workload(parameters),
    } for ep in range(10)]
    payload = {
        "checkpoint_sha256": checkpoint[1], "matrix_sha256": _file_digest(MATRIX),
        "deterministic_execution": True,
        "results": [{
            "selector": f"ablation/{setting}", "execution_action": "mean", "outer_state_unchanged": True,
            "outer_updates_before": 100, "outer_updates_after": 100, "controller_seed": 55,
            "episode_sampling": sampling, "episodes": episodes,
            "resolved_config": parameters, **(
                {"execution_backend": GRAPH, "graph_replays": 5000} if setting != "j0" else {}
            ),
        }],
    }
    validate_result(payload, resolved, checkpoint[1], sampling, 500)
    episodes[-1]["model_metrics"]["inner_compile_fallback"] = 1
    with pytest.raises(ValueError, match="inner_compile_fallback"):
        validate_result(payload, resolved, checkpoint[1], sampling, 500)
