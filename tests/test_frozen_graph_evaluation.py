from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from RL.tdmpc2_core.frozen_graph import FrozenGraphController, MeanActionGraph, PROTOCOL
from RL.tdmpc2_core.whole_inner_graph import validate_training_graph
from tests.test_inner_entropy_graph import make_agent, inner_snapshot, outer_snapshot, assert_tree_close
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.eval_series_data import planner_identity


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_checkpoint_graph_eval.json"


def test_matrix_preserves_saved_controller_and_selects_paired_prior():
    import json
    saved = json.loads((ROOT / "configs/dmcontrol/algs/ambi_inner_entropy_posttanh_outer_adapt.json").read_text())
    context = SimpleNamespace(source=Path("metadata.json"), trial_run_params={**saved, "env": "DMControl-v0"},
                              experiment_params={"env_params": {"task": "humanoid-walk", "obs": "state", "render_mode": None}})
    matrix = load_preset_matrix(MATRIX)
    graph = resolve_preset(MATRIX, "controller/graph", matrix=matrix, checkpoint_context=context)
    prior = resolve_preset(MATRIX, "controller/prior", matrix=matrix, checkpoint_context=context)
    for key, value in saved["alg_params"].items():
        if key not in matrix["shared_alg_params"]:
            assert graph["algorithm_config"]["alg_params"][key] == value
    assert graph["evaluation"]["execution_backend"] == PROTOCOL
    assert prior["algorithm_config"]["alg_params"]["inner_operator"] == "none"
    assert prior["algorithm_config"]["alg_params"]["inner_cuda_graph"] is False
    assert graph["evaluation"]["seeds"] == prior["evaluation"]["seeds"] == [101, 102, 103, 104, 105]


def test_graph_controller_rejects_any_protocol_fallback():
    controller = object.__new__(FrozenGraphController)
    for kwargs in ({}, {"eval_mode": True}, {"eval_mode": True, "collect_diagnostics": False, "trace": object()},
                   {"eval_mode": True, "collect_diagnostics": False, "apply_inner_writeback": True},
                   {"eval_mode": True, "collect_diagnostics": False, "return_behavior_policy": True}):
        with pytest.raises(ValueError, match="Frozen graph requires"):
            controller.act(None, **kwargs)


def test_graph_rng_has_distinct_curve_identity():
    config = {"inner_operator": "sac", "inner_rounds": 8}
    native = planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    graph = planner_identity(config, {"execution_backend": PROTOCOL}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    assert native != graph
    assert graph["settings"]["evaluation_execution_backend"] == PROTOCOL


@pytest.mark.parametrize("saved_execution", ["policy_sample", "mean"])
def test_frozen_graph_accepts_saved_execution_without_changing_training_gate(saved_execution):
    engine = SimpleNamespace(
        cfg=SimpleNamespace(inner_rounds=8, compile=True, compile_strict=True,
                            inner_execution_action=saved_execution),
        device=torch.device("cuda"), _retrace_enabled=False, _active_trace=None,
    )
    validate_training_graph(engine, frozen_mean_evaluation=True)
    assert engine.cfg.inner_execution_action == saved_execution
    if saved_execution == "mean":
        with pytest.raises(ValueError, match="inner_execution_action"):
            validate_training_graph(engine)
    else:
        validate_training_graph(engine)
    engine.cfg.inner_execution_action = "mean_plus_gaussian"
    with pytest.raises(ValueError, match="inner_execution_action"):
        validate_training_graph(engine, frozen_mean_evaluation=True)
    engine.cfg.inner_execution_action = saved_execution
    engine.cfg.inner_actor_scope = "run"
    with pytest.raises(ValueError, match="inner_actor_scope"):
        validate_training_graph(engine, frozen_mean_evaluation=True)


def test_mean_checkpoint_native_evaluation_warmup_uses_evaluation_validation():
    from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
    engine = SimpleNamespace(
        cfg=SimpleNamespace(inner_rounds=8, compile=True, compile_strict=True,
                            inner_execution_action="mean", inner_cuda_graph=True),
        device=torch.device("cuda"), _retrace_enabled=False, _active_trace=None,
        _whole_graph_worker=False, prepare_control=lambda **kwargs: None,
        _act_with_trace=lambda *args, **kwargs: (None, {}, []),
    )
    result = InnerImprovementEngine.act(engine, None, eval_mode=True)
    assert result[1]["inner_cuda_graph_used"] == 0.0
    with pytest.raises(ValueError, match="inner_execution_action"):
        InnerImprovementEngine.act(engine, None, eval_mode=False)


@pytest.mark.parametrize("bad_options", [
    {"execution_action": "sample", "recording": "returns"},
    {"execution_action": "mean", "recording": "full"},
])
def test_evaluator_rejects_graph_protocol_changes_before_model_load(tmp_path, bad_options):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.touch()
    resolved = {"algorithm_config": {"alg_params": {"inner_operator": "sac"}},
                "evaluation": {"execution_backend": PROTOCOL, **bad_options}}
    with pytest.raises(ValueError):
        evaluator.evaluate_preset(resolved, checkpoint, [101], controller_seed=55)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires scheduled CUDA GPU")
@pytest.mark.parametrize("saved_execution", ["policy_sample", "mean"])
def test_cuda_mean_graph_matches_control_and_resets_episode_rng(saved_execution):
    torch.set_float32_matmul_precision("high")
    agents = [make_agent("squashed", "action", "per_update", device="cuda", compiled=True) for _ in range(2)]
    agents[1].load(deepcopy(agents[0].checkpoint_state()))
    for agent in agents:
        agent.cfg.inner_execution_action = saved_execution
        validate_training_graph(agent.inner_engine, frozen_mean_evaluation=True)
    before = [outer_snapshot(agent) for agent in agents]
    root = torch.randn(1, agents[0].cfg.latent_dim, device="cuda")
    runners = [MeanActionGraph(agent.inner_engine, root, capture=capture, seed=12345, warmup=4)
               for agent, capture in zip(agents, (False, True))]
    first = None
    for index in range(3):
        outputs = []
        for agent, runner in zip(agents, runners):
            outside = torch.cuda.get_rng_state().clone()
            action, metrics, lengths = runner(root + index / 10)
            outputs.append((action.detach().clone(), {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in metrics.items()}))
            with torch.no_grad():
                expected, _ = agent.model.pi(root + index / 10, policy=runner.engine._action_pool.actor, deterministic=True)
            torch.testing.assert_close(action, expected[0], rtol=0, atol=0)
            torch.testing.assert_close(outside, torch.cuda.get_rng_state(), rtol=0, atol=0)
        assert_tree_close(outputs[0], outputs[1])
        assert_tree_close(inner_snapshot(agents[0].inner_engine), inner_snapshot(agents[1].inner_engine))
        if index == 0:
            first = outputs[1][0]
    runners[1].reset_rng()
    action, _, _ = runners[1](root)
    torch.testing.assert_close(action, first, rtol=0, atol=0)
    for agent, snapshot in zip(agents, before):
        assert_tree_close(outer_snapshot(agent), snapshot)
