"""Sparse solves retain full episode accounting and distinct planner identity."""
import json
import math
from pathlib import Path

import pytest

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_actor_transfer_evaluation import transfer_matrix
from tests.test_ambi_benchmark_evaluation import events
from utils.eval_series_data import _metrics, planner_identity


def hold_matrix(checkpoint, path, horizon):
    matrix = json.loads(path.read_text())
    matrix["study_protocol"] = "actor-transfer-hold-h-v1"
    matrix["evaluation"]["max_steps"] = 7
    matrix["shared_alg_params"] = dict(
        inner_rounds=2, inner_first_action_rounds=None,
        inner_solve_interval=horizon, inner_rollout_horizon=horizon,
        inner_replay_capacity=max(12, 4 * horizon),
    )
    path.write_text(json.dumps(matrix))
    sidecar = Path(str(checkpoint) + ".metadata.json")
    metadata = json.loads(sidecar.read_text())
    metadata["experiment_params"]["env_params"]["max_episode_steps"] = 7
    sidecar.write_text(json.dumps(metadata))
    return matrix


@pytest.mark.parametrize("horizon", [2, 3])
@pytest.mark.parametrize("mode", ["cold", "warm"])
def test_hold_episode_records_actual_solves_and_feedback_actions(transfer_matrix, tmp_path, horizon, mode):
    checkpoint, path = transfer_matrix
    hold_matrix(checkpoint, path, horizon)
    bundle = tmp_path / "held"
    result = evaluator.evaluate_matrix(
        path, checkpoint, selectors=[f"transfer/{mode}"], bundle_dir=bundle,
    )["results"][0]
    assert result["outer_state_unchanged"]
    assert result["study_protocol"] == "actor-transfer-hold-h-v1"
    trace = list(events(bundle))
    decisions = [row for row in trace if row["phase"] == "decision"]
    assert len(decisions) == 14
    for row in decisions:
        decision, values = row["decision_index"], row["metrics"]
        solved = decision % horizon == 0
        assert values["decision/inner_solve_performed"] == solved
        assert values["decision/inner_policy_held"] == (not solved)
        assert values["decision/inner_action_age"] == decision % horizon
        assert values["decision/inner_rounds"] == (2 if solved else 0)
        assert values["decision/inner_critic_optimizer_steps"] == (4 if solved else 0)
        assert values["decision/inner_actor_optimizer_steps"] == (2 if solved else 0)
        same = [event for event in trace if event["episode_id"] == row["episode_id"]
                and event["decision_index"] == decision]
        if not solved:
            assert [event["phase"] for event in same] == ["decision"]
            assert values["decision/diagnostic_seconds"] == 0
            assert not any(key.startswith("decision/inner_togo_") for key in values)
            assert row["actor_updates"] == row["critic_updates"] == row["round_index"] == 0
        else:
            assert values["decision/inner_actor_transferred"] == (mode == "warm" and decision > 0)
            assert len([event for event in same if event["phase"] == "probe"]) == 5
    per_episode = math.ceil(7 / horizon)
    for episode in result["episodes"]:
        assert episode["length"] == 7
        assert episode["solve_count"] == per_episode
        assert episode["held_decision_count"] == 7 - per_episode
        assert episode["solve_control_seconds"] + episode["held_control_seconds"] == pytest.approx(episode["control_seconds"])
        assert episode["control_seconds_per_decision"] == pytest.approx(episode["control_seconds"] / 7)
        timing = episode["transfer_latency"]
        assert timing["solve"]["control_seconds"]["count"] == per_episode
        assert timing["held"]["control_seconds"]["count"] == 7 - per_episode
        assert timing["held"]["diagnostic_seconds"]["total"] == 0
        assert [row["solve_index"] for row in timing["samples"]] == [d // horizon for d in range(7)]
    metrics = _metrics(result["episodes"])
    assert metrics["work/environment_decisions"] == 14
    assert metrics["work/solves"] == metrics["runtime/solve_decisions"] == 2 * per_episode
    assert metrics["work/held_decisions"] == metrics["runtime/held_decisions"] == 14 - 2 * per_episode
    assert metrics["work/critic_updates"] == 2 * per_episode * 4
    assert metrics["work/actor_updates"] == 2 * per_episode * 2
    assert metrics["work/model_steps"] == 2 * per_episode * 4 * horizon
    assert metrics["runtime/solve_control_seconds_total"] + metrics["runtime/held_control_seconds_total"] == pytest.approx(metrics["runtime/control_seconds"])
    assert metrics["runtime/control_seconds_per_decision"] == pytest.approx(metrics["runtime/control_seconds"] / 14)


def test_solve_interval_identity_preserves_historical_one_and_separates_holds():
    config = dict(inner_operator="sac", inner_rounds=2, inner_actor_scope="episode")
    identity = lambda c: planner_identity(c, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    original = identity(config)
    assert identity({**config, "inner_solve_interval": 1}) == original
    held = identity({**config, "inner_solve_interval": 2})
    assert held != original
    assert held["settings"]["inner_solve_interval"] == 2
    assert held["semantics"]["evaluation_protocol"] == "actor-transfer-hold-h-v1"
    prior = {**config, "inner_operator": "none"}
    assert identity(prior) == identity({**prior, "inner_solve_interval": 1})


def test_legacy_timing_records_count_every_decision_as_a_solve():
    samples = [dict(decision_index=d, control_seconds=2., prediction_seconds=3., diagnostic_seconds=1.)
               for d in range(3)]
    episode = dict(return_=1, length=3, control_seconds=6.,
                   transfer_latency={"samples": samples})
    episode["return"] = episode.pop("return_")
    metrics = _metrics([episode])
    assert metrics["work/solves"] == metrics["runtime/solve_decisions"] == 3
    assert metrics["work/held_decisions"] == metrics["runtime/held_decisions"] == 0
    assert metrics["runtime/solve_control_seconds_total"] == 6
    assert metrics["runtime/held_control_seconds_total"] == 0


def test_hold_protocol_rejects_wrong_interval_before_model_creation(transfer_matrix, tmp_path, monkeypatch):
    checkpoint, path = transfer_matrix
    matrix = hold_matrix(checkpoint, path, 2)
    matrix["shared_alg_params"]["inner_solve_interval"] = 3
    path.write_text(json.dumps(matrix))
    monkeypatch.setattr(evaluator, "_make_env", lambda *args: pytest.fail("invalid cadence created an environment"))
    with pytest.raises(ValueError, match="equal the imagined rollout horizon"):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "invalid")


def test_old_protocol_cannot_silently_use_held_execution(transfer_matrix, tmp_path, monkeypatch):
    checkpoint, path = transfer_matrix
    matrix = hold_matrix(checkpoint, path, 2)
    matrix["study_protocol"] = "actor-transfer-v2"
    path.write_text(json.dumps(matrix))
    monkeypatch.setattr(evaluator, "_make_env", lambda *args: pytest.fail("invalid protocol created an environment"))
    with pytest.raises(ValueError, match="Held-policy evaluation requires"):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "invalid")
