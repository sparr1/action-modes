"""Critic-only transfer protocols, full-episode traces, and stable identities."""
import json
import math
from pathlib import Path

import pytest

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_actor_transfer_evaluation import transfer_matrix
from tests.test_ambi_benchmark_evaluation import events
from utils.eval_series_data import _metrics, descriptive_label, planner_identity


def critic_matrix(checkpoint, path, *, source="aux_return", horizon=2, held=False):
    matrix = json.loads(path.read_text())
    matrix["study_protocol"] = "critic-transfer-hold-h-v1" if held else "critic-transfer-v1"
    matrix["evaluation"].pop("actor_transfer_diagnostics")
    matrix["evaluation"].update(transfer_diagnostics=True, max_steps=5)
    matrix["shared_alg_params"] = dict(
        inner_first_action_rounds=None, inner_actor_scope="action",
        inner_critic_source=source, inner_horizon_critic_source=source,
        inner_sac_critic_target="entropy_augmented" if source == "sac" else "reward_only",
        inner_terminal_entropy="outer" if source == "sac" else "none",
        inner_critic_target_initialization="online",
        inner_rollout_horizon=horizon, inner_solve_interval=horizon if held else 1,
        inner_replay_capacity=12,
    )
    matrix["comparisons"]["transfer"]["variants"] = {
        "cold": {"description": "Fresh actor and critic", "alg_params": {"inner_critic_scope": "action"}},
        "warm": {"description": "Critic-only transfer", "alg_params": {"inner_critic_scope": "episode"}},
    }
    path.write_text(json.dumps(matrix))
    sidecar = Path(str(checkpoint) + ".metadata.json")
    metadata = json.loads(sidecar.read_text())
    metadata["experiment_params"]["env_params"]["max_episode_steps"] = 5
    sidecar.write_text(json.dumps(metadata))
    return matrix


@pytest.mark.parametrize("source", ["sac", "aux_return"])
@pytest.mark.parametrize("mode", ["cold", "warm"])
@pytest.mark.parametrize("horizon,held", [(2, False), (1, True), (2, True), (3, True)])
def test_critic_protocol_records_only_actual_solves(transfer_matrix, tmp_path, source, mode, horizon, held):
    checkpoint, path = transfer_matrix
    matrix = critic_matrix(checkpoint, path, source=source, horizon=horizon, held=held)
    bundle = tmp_path / "critic"
    result = evaluator.evaluate_matrix(path, checkpoint, selectors=[f"transfer/{mode}"],
                                       bundle_dir=bundle)["results"][0]
    assert result["outer_state_unchanged"]
    assert result["study_protocol"] == matrix["study_protocol"]
    assert result["transfer"]["transfer_mode"] == ("critic_only" if mode == "warm" else "fresh")
    assert result["transfer"]["target_initialization"] == "starting_online_critic_each_solve"
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["runs"][0]["transfer"] == result["transfer"]
    assert manifest["runs"][0]["study_protocol"] == matrix["study_protocol"]
    interval = horizon if held else 1
    trace = list(events(bundle))
    decisions = [row for row in trace if row["phase"] == "decision"]
    assert len(decisions) == 10
    for row in decisions:
        decision, values = row["decision_index"], row["metrics"]
        solved = decision % interval == 0
        assert values["decision/inner_actor_transferred"] == 0
        if mode == "warm":
            assert values["decision/inner_critic_transferred"] == (solved and decision > 0)
            assert values["decision/inner_critic_target_reinitialized"] == solved
            assert values["decision/inner_critic_updates_initial"] == (2 * (decision // interval) if solved else 0)
        assert values["decision/inner_rounds"] == (1 if solved else 0)
        assert values["decision/inner_critic_optimizer_steps"] == (2 if solved else 0)
        assert values["decision/inner_actor_optimizer_steps"] == (1 if solved else 0)
        same = [event for event in trace if event["episode_id"] == row["episode_id"]
                and event["decision_index"] == decision]
        if solved:
            assert same[0]["phase"] == "initial"
            assert same[-1]["phase"] == "decision"
            assert same[0]["replay_size"] == 0
            if mode == "warm":
                assert same[0]["metrics"]["inner_critic_updates_initial"] == 2 * (decision // interval)
            for component in ("actor", "critic", "temperature"):
                assert same[0]["metrics"][f"{component}_optimizer_steps_initial"] == 0
            assert len([event for event in same if event["phase"] == "probe"]) == 4
        else:
            assert [event["phase"] for event in same] == ["decision"]
            assert values["decision/diagnostic_seconds"] == 0
    solves = math.ceil(5 / interval)
    for episode in result["episodes"]:
        assert episode["solve_count"] == solves
        assert episode["held_decision_count"] == 5 - solves
        assert episode["solve_control_seconds"] + episode["held_control_seconds"] == pytest.approx(episode["control_seconds"])
    metrics = _metrics(result["episodes"])
    assert metrics["work/solves"] == 2 * solves
    assert metrics["work/critic_updates"] == 4 * solves
    assert metrics["work/actor_updates"] == 2 * solves


def test_critic_episode_order_and_neutral_diagnostics_alias(transfer_matrix, tmp_path):
    checkpoint, path = transfer_matrix
    matrix = critic_matrix(checkpoint, path)
    neutral = evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "neutral")["results"][0]
    matrix["evaluation"].pop("transfer_diagnostics")
    matrix["evaluation"]["actor_transfer_diagnostics"] = True
    path.write_text(json.dumps(matrix))
    legacy = evaluator.evaluate_matrix(path, checkpoint, seeds=[102, 101], bundle_dir=tmp_path / "alias")["results"][0]
    assert {ep["seed"]: ep["return"] for ep in neutral["episodes"]} == {
        ep["seed"]: ep["return"] for ep in legacy["episodes"]}
    assert legacy["study_protocol"] == "critic-transfer-v1"


@pytest.mark.parametrize("overrides,match", [
    ({"inner_first_action_rounds": 3}, "selected J"),
    ({"inner_solve_interval": 2}, "Held-policy evaluation"),
    ({"inner_actor_scope": "episode"}, "fresh actors"),
    ({"inner_critic_scope": "run"}, "episode-scoped critics"),
    ({"inner_replay_scope": "episode"}, "fresh action-local"),
    ({"inner_critic_optimizer_scope": "episode"}, "fresh action-local"),
    ({"inner_critic_target_initialization": "outer_target"}, "inner_critic_target_initialization"),
    ({"inner_rebase_persistent": True}, "inner_rebase_persistent"),
    ({"inner_critic_adaptation": "lora_rl"}, "inner_critic_adaptation"),
    ({"inner_actor_writeback_coef": 0.1}, "disable prior writeback"),
])
def test_critic_protocol_rejects_incompatible_configs_before_environment(transfer_matrix, tmp_path, monkeypatch, overrides, match):
    checkpoint, path = transfer_matrix
    matrix = critic_matrix(checkpoint, path)
    matrix["comparisons"]["transfer"]["variants"]["warm"]["alg_params"].update(overrides)
    path.write_text(json.dumps(matrix))
    monkeypatch.setattr(evaluator, "_make_env", lambda *_: pytest.fail("invalid protocol constructed an environment"))
    with pytest.raises(ValueError, match=match):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("option,value", [("transfer_diagnostics", False), ("transfer_diagnostics", "yes"),
                                           ("actor_transfer_diagnostics", 1)])
def test_critic_protocol_requires_boolean_bundled_diagnostics(transfer_matrix, tmp_path, option, value):
    checkpoint, path = transfer_matrix
    matrix = critic_matrix(checkpoint, path)
    matrix["evaluation"][option] = value
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="diagnostics"):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "invalid")


def test_historical_actor_protocol_cannot_mislabel_critic_transfer(transfer_matrix, tmp_path):
    checkpoint, path = transfer_matrix
    matrix = critic_matrix(checkpoint, path)
    matrix["study_protocol"] = "actor-transfer-v2"
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="fresh action-local"):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "invalid")


def test_critic_protocol_hold_requires_horizon_cadence(transfer_matrix, tmp_path):
    checkpoint, path = transfer_matrix
    matrix = critic_matrix(checkpoint, path, horizon=3, held=True)
    matrix["shared_alg_params"]["inner_solve_interval"] = 2
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="equal the imagined rollout horizon"):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "invalid")


def test_critic_specification_preflight_matches_result_identity(transfer_matrix, tmp_path, monkeypatch):
    from utils import ambi_benchmark as storage
    from utils import eval_series_data as data
    checkpoint, path = transfer_matrix
    critic_matrix(checkpoint, path)
    resolved = []
    def identity(checkpoint, resolution, *args, **kwargs):
        params = resolution["algorithm_config"]["alg_params"]
        planner = planner_identity(params, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
        resolved.append(planner)
        return {"backbone": "entity/train/prior", "planner": planner,
                "protocol": {"max_steps": 5, "seeds": [101, 102]}, "science": {"evaluator": "fixture"}}
    monkeypatch.setattr(data, "identity_for_ambi_checkpoint", identity)
    monkeypatch.setattr(storage, "code_identity", lambda: {"commit": "fixture", "dirty": False})
    monkeypatch.setattr(evaluator, "_make_env", lambda *_: pytest.fail("specification created an environment"))
    result = evaluator.evaluate_matrix(path, checkpoint, selectors=["transfer/cold", "transfer/warm"],
        bundle_dir=tmp_path / "unused", eval_series_spec_dir=tmp_path / "specs")
    assert result["mode"] == "evaluation_series_specifications"
    assert len(resolved) == 2
    assert "semantics" not in resolved[0]
    assert resolved[1]["semantics"]["evaluation_protocol"] == "critic-transfer-v1"
    assert not (tmp_path / "unused").exists()


def test_new_critic_identity_preserves_old_fresh_actor_and_generic_structures():
    identity = lambda c, r=None: planner_identity(c, r or {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    base = dict(inner_operator="sac", inner_rounds=1, inner_actor_scope="action", inner_critic_scope="action")
    historical = identity(base)
    for result in ({}, {"study_protocol": "critic-transfer-v1", "transfer": {"transfer_mode": "fresh"}}):
        assert identity(base, result) == historical
    assert "semantics" not in identity({**base, "inner_actor_scope": "episode"})
    assert "semantics" not in identity({**base, "inner_critic_scope": "episode"})
    cold = {**base, "aux_return_mode": "sac"}
    warm = {**cold, "inner_critic_scope": "episode"}
    assert "semantics" not in identity(cold)
    semantics = identity(warm)["semantics"]
    assert semantics["evaluation_protocol"] == "critic-transfer-v1"
    assert semantics["transfer_component"] == "online_inner_critic"
    assert semantics["target_initialization"] == "starting_online_critic_each_solve"
    assert identity(warm) == identity({**warm, "inner_solve_interval": 1})
    held = identity({**warm, "inner_solve_interval": 3})
    assert held["semantics"]["evaluation_protocol"] == "critic-transfer-hold-h-v1"
    assert held["semantics"]["held_action"] == "cached_feedback_actor_at_current_observation"
    assert identity({**cold, "inner_solve_interval": 3})["semantics"]["evaluation_protocol"] == "actor-transfer-hold-h-v1"


def test_critic_failure_keeps_critic_protocol_label(transfer_matrix, tmp_path, monkeypatch):
    checkpoint, path = transfer_matrix
    critic_matrix(checkpoint, path)
    def fail(*args, **kwargs):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(evaluator, "evaluate_preset", fail)
    bundle = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated failure"):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["runs"][0]["study_protocol"] == "critic-transfer-v1"


@pytest.mark.parametrize("source", ["sac", "aux_return"])
def test_critic_labels_name_transfer_value_semantics_and_cadence(source):
    config = dict(inner_operator="sac", aux_return_mode="sac", inner_actor_scope="action",
                  inner_critic_scope="episode", inner_critic_source=source,
                  inner_horizon_critic_source=source, inner_solve_interval=3)
    identity = {"backbone": "entity/project/source", "science": {"algorithm": "AMBITDMPC2/AMBITDMPC2"},
                "planner": planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")}
    label = descriptive_label(identity)
    assert "critic-only transfer" in label and "hold3" in label
    assert ("return/return" if source == "aux_return" else "soft/soft") in label
    assert "actor-warm" not in label


@pytest.mark.parametrize("held", [False, True])
def test_direct_critic_evaluation_infers_truthful_protocol(transfer_matrix, tmp_path, monkeypatch, held):
    checkpoint, path = transfer_matrix
    matrix = critic_matrix(checkpoint, path, held=held)
    original = evaluator.evaluate_preset
    seen = []
    def direct(*args, **kwargs):
        kwargs.pop("study_protocol")
        result = original(*args, **kwargs)
        seen.append(result["study_protocol"])
        return result
    monkeypatch.setattr(evaluator, "evaluate_preset", direct)
    evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "direct")
    assert seen == [matrix["study_protocol"]]
