"""New-head transfer must have its own protocol, chronology, and identity."""

import ast
import json
import math

import pytest

import evaluate_ambi_checkpoint as evaluator
from tests.test_ambi_actor_transfer_evaluation import transfer_matrix
from tests.test_ambi_benchmark_evaluation import events
from tests.test_ambi_critic_transfer_evaluation import critic_matrix
from utils.eval_series_data import descriptive_label, planner_identity
from utils.ambi_research import resolve_preset
from utils.checkpoint_context import load_checkpoint_context


def hidden_matrix(checkpoint, path, *, source="aux_return", horizon=3, held=False):
    matrix = critic_matrix(checkpoint, path, source=source, horizon=horizon, held=held)
    matrix["study_protocol"] = "critic-hidden-transfer-hold-h-v1" if held else "critic-hidden-transfer-v1"
    matrix["shared_alg_params"]["inner_critic_transfer_head"] = "random"
    group = matrix["comparisons"]["transfer"]
    group["variants"].pop("cold")
    group["reference"] = "warm"
    path.write_text(json.dumps(matrix))
    return matrix


@pytest.mark.parametrize("source", ["sac", "aux_return"])
@pytest.mark.parametrize("horizon,held", [(3, False), (1, True), (2, True), (3, True)])
def test_hidden_protocol_fixture_records_resets_only_at_solves(transfer_matrix, tmp_path, source, horizon, held):
    checkpoint, path = transfer_matrix
    matrix = hidden_matrix(checkpoint, path, source=source, horizon=horizon, held=held)
    bundle = tmp_path / "hidden"
    result = evaluator.evaluate_matrix(path, checkpoint, bundle_dir=bundle)["results"][0]
    assert result["outer_state_unchanged"]
    assert result["study_protocol"] == matrix["study_protocol"]
    metadata = result["transfer"]
    assert metadata["transfer_mode"] == "critic_hidden_only"
    assert metadata["critic_head_initialization"] == "xavier_uniform_zero_bias_each_solve"
    assert metadata["critic_head_reset_at_first_solve"] is True
    assert metadata["critic_hidden_normalization"] == "retained_and_trainable"
    assert metadata["target_initialization"] == "starting_online_critic_after_head_reset_each_solve"
    run = json.loads((bundle / "manifest.json").read_text())["runs"][0]
    assert run["transfer"] == metadata
    assert run["study_protocol"] == matrix["study_protocol"]
    interval = horizon if held else 1
    trace = list(events(bundle))
    decisions = [event for event in trace if event["phase"] == "decision"]
    assert len(decisions) == 10
    for event in decisions:
        decision = event["decision_index"]
        solved = decision % interval == 0
        metrics = event["metrics"]
        assert metrics["decision/inner_critic_head_reinitialized"] == solved
        assert metrics["decision/inner_critic_target_reinitialized"] == solved
        assert metrics["decision/inner_critic_transferred"] == (solved and decision > 0)
        assert metrics["decision/inner_actor_transferred"] == 0
        assert metrics["decision/inner_critic_updates_initial"] == (2 * (decision // interval) if solved else 0)
        same = [row for row in trace if row["episode_id"] == event["episode_id"]
                and row["decision_index"] == decision]
        if solved:
            assert same[0]["phase"] == "initial"
            assert same[0]["replay_size"] == 0
            assert same[0]["metrics"]["inner_critic_head_reinitialized"] == 1
            assert same[-1]["phase"] == "decision"
        else:
            assert [row["phase"] for row in same] == ["decision"]
    for episode in result["episodes"]:
        assert episode["length"] == 5
        assert episode["solve_count"] == math.ceil(5 / interval)
        assert episode["held_decision_count"] == 5 - math.ceil(5 / interval)


@pytest.mark.parametrize("overrides,protocol,match", [
    ({"inner_critic_transfer_head": "retain"}, None, "requires inner_critic_transfer_head='random'"),
    ({"inner_critic_scope": "action"}, None, "episode-scoped critic"),
    ({}, "critic-transfer-v1", "critic-hidden-transfer protocol"),
    ({}, "actor-transfer-v2", "critic-hidden-transfer protocol"),
    ({}, "unrecognized", "named transfer full-episode protocol"),
])
def test_hidden_protocol_preflight_rejects_mislabeling(transfer_matrix, tmp_path, monkeypatch, overrides, protocol, match):
    checkpoint, path = transfer_matrix
    matrix = hidden_matrix(checkpoint, path)
    matrix["comparisons"]["transfer"]["variants"]["warm"]["alg_params"].update(overrides)
    if protocol:
        matrix["study_protocol"] = protocol
    path.write_text(json.dumps(matrix))
    monkeypatch.setattr(evaluator, "_make_env", lambda *_: pytest.fail("invalid protocol constructed an environment"))
    with pytest.raises(ValueError, match=match):
        evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("held", [False, True])
def test_direct_hidden_evaluation_infers_new_protocol(transfer_matrix, tmp_path, monkeypatch, held):
    checkpoint, path = transfer_matrix
    matrix = hidden_matrix(checkpoint, path, held=held)
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


@pytest.mark.parametrize("held,overrides,diagnostics,match", [
    (False, {"inner_solve_interval": 3}, True, "requires inner_solve_interval=1"),
    (True, {"inner_solve_interval": 2}, True, "equal the imagined rollout horizon"),
    (False, {"inner_first_action_rounds": 2}, True, "selected J"),
    (False, {"inner_actor_scope": "episode"}, True, "inner_actor_scope='action'"),
    (False, {"inner_critic_target_initialization": "outer_target"}, True, "inner_critic_target_initialization"),
    (False, {"inner_critic_optimizer_scope": "episode"}, True, "inner_critic_optimizer_scope"),
    (False, {}, False, "require transfer diagnostics"),
])
def test_direct_hidden_protocol_rejects_invalid_contract_before_environment(
    transfer_matrix, monkeypatch, held, overrides, diagnostics, match,
):
    checkpoint, path = transfer_matrix
    matrix = hidden_matrix(checkpoint, path, held=held)
    resolved = resolve_preset(path, "transfer/warm", checkpoint_context=load_checkpoint_context(checkpoint))
    resolved["algorithm_config"]["alg_params"].update(overrides)
    monkeypatch.setattr(evaluator, "_make_env", lambda *_: pytest.fail("invalid direct protocol constructed an environment"))
    with pytest.raises(ValueError, match=match):
        evaluator.evaluate_preset(resolved, checkpoint, [101], controller_seed=55,
                                  study_protocol=matrix["study_protocol"], transfer_diagnostics=diagnostics)


@pytest.mark.parametrize("source", ["sac", "aux_return"])
@pytest.mark.parametrize("interval", [1, 3])
def test_hidden_identity_and_labels_are_distinct_without_changing_historical_defaults(source, interval):
    identity = lambda config: planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    config = dict(inner_operator="sac", aux_return_mode="sac", inner_actor_scope="action",
                  inner_critic_scope="episode", inner_critic_source=source,
                  inner_horizon_critic_source=source, inner_solve_interval=interval)
    old = identity(config)
    assert old == identity({**config, "inner_critic_transfer_head": "retain"})
    hidden = identity({**config, "inner_critic_transfer_head": "random"})
    assert hidden != old
    semantics = hidden["semantics"]
    assert semantics["evaluation_protocol"] == ("critic-hidden-transfer-hold-h-v1" if interval > 1
                                               else "critic-hidden-transfer-v1")
    assert semantics["transfer_component"] == "online_inner_critic_hidden_layers"
    assert semantics["critic_head_initialization"] == "xavier_uniform_zero_bias_each_solve"
    assert semantics["critic_head_reset_at_first_solve"] is True
    assert semantics["target_initialization"] == "starting_online_critic_after_head_reset_each_solve"
    label = descriptive_label({"backbone": "entity/project/source",
        "science": {"algorithm": "AMBITDMPC2/AMBITDMPC2"}, "planner": hidden})
    assert "critic hidden transfer + fresh head" in label
    assert ("return/return" if source == "aux_return" else "soft/soft") in label
    assert ("hold3" in label) == (interval > 1)
    for scopes in ({"inner_actor_scope": "action", "inner_critic_scope": "action"},
                   {"inner_actor_scope": "episode", "inner_critic_scope": "action"}):
        historical = {**config, **scopes}
        assert identity(historical) == identity({**historical, "inner_critic_transfer_head": "retain"})


def test_hidden_specification_preflight_uses_same_new_identity(transfer_matrix, tmp_path, monkeypatch):
    from utils import ambi_benchmark as storage
    from utils import eval_series_data as data
    checkpoint, path = transfer_matrix
    hidden_matrix(checkpoint, path)
    recorded = []
    def identity(checkpoint, resolution, *args, **kwargs):
        planner = planner_identity(resolution["algorithm_config"]["alg_params"], {},
                                   "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
        recorded.append(planner)
        return {"backbone": "entity/train/prior", "planner": planner,
                "protocol": {"max_steps": 5, "seeds": [101, 102]}, "science": {"evaluator": "fixture"}}
    monkeypatch.setattr(data, "identity_for_ambi_checkpoint", identity)
    monkeypatch.setattr(storage, "code_identity", lambda: {"commit": "fixture", "dirty": False})
    monkeypatch.setattr(evaluator, "_make_env", lambda *_: pytest.fail("specification created an environment"))
    result = evaluator.evaluate_matrix(path, checkpoint, bundle_dir=tmp_path / "unused",
                                       eval_series_spec_dir=tmp_path / "specs")
    assert result["mode"] == "evaluation_series_specifications"
    assert recorded[0]["semantics"]["evaluation_protocol"] == "critic-hidden-transfer-v1"
    assert not (tmp_path / "unused").exists()


def test_new_protocol_constants_change_scientific_fingerprint_without_rekeying_old_revisions():
    from utils.eval_series_data import _scientific_evaluator_nodes
    historical = '''
_ACTOR_TRANSFER_PROTOCOLS = {"actor-transfer-v2"}
_CRITIC_TRANSFER_PROTOCOLS = {"critic-transfer-v1"}
_HOLD_TRANSFER_PROTOCOLS = {"critic-transfer-hold-h-v1"}
def evaluate_preset():
    return 1
'''
    new = '_CRITIC_HIDDEN_TRANSFER_PROTOCOLS = {"critic-hidden-transfer-v1"}\n' + historical
    def fingerprint(source):
        tree = ast.parse(source)
        tree.body = _scientific_evaluator_nodes(tree, {"evaluate_preset"})
        return ast.dump(tree, include_attributes=False)
    old_function_only = ast.parse(historical)
    old_function_only.body = [node for node in old_function_only.body if isinstance(node, ast.FunctionDef)]
    assert fingerprint(historical) == ast.dump(old_function_only, include_attributes=False)
    assert fingerprint(new) != fingerprint(historical)
    for value in ("actor-transfer-v2", "critic-transfer-v1", "critic-transfer-hold-h-v1", "critic-hidden-transfer-v1"):
        assert fingerprint(new) != fingerprint(new.replace(f'"{value}"', f'"{value}-changed"'))
