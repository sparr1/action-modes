from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from RL.tdmpc2_core.frozen_graph import FrozenGraphController, MeanActionGraph, PROTOCOL
from tests.test_inner_entropy_graph import make_agent, inner_snapshot, outer_snapshot, assert_tree_close
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.eval_series_data import planner_identity


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_checkpoint_graph_eval.json"


@pytest.mark.parametrize("mode", ["outer_frozen", "outer_adapt", "carry_adapt"])
def test_matrix_preserves_saved_controller_and_selects_paired_prior(mode):
    import json
    saved = json.loads((ROOT / f"configs/dmcontrol/algs/ambi_inner_entropy_posttanh_{mode}.json").read_text())
    context = SimpleNamespace(source=Path("metadata.json"), trial_run_params={**saved, "env": "DMControl-v0"},
                              experiment_params={"env_params": {"task": "humanoid-walk", "obs": "state", "render_mode": None}})
    matrix = load_preset_matrix(MATRIX)
    graph = resolve_preset(MATRIX, "controller/graph", matrix=matrix, checkpoint_context=context)
    prior = resolve_preset(MATRIX, "controller/prior", matrix=matrix, checkpoint_context=context)
    for key, value in saved["alg_params"].items():
        if key not in matrix["shared_alg_params"]:
            assert graph["algorithm_config"]["alg_params"][key] == value
    assert graph["evaluation"]["execution_backend"] == PROTOCOL
    evaluator._validate_benchmark_scopes(graph["algorithm_config"]["alg_params"],
                                         episode_evaluation=True, execution_backend=PROTOCOL)
    assert prior["algorithm_config"]["alg_params"]["inner_operator"] == "none"
    assert prior["algorithm_config"]["alg_params"]["inner_cuda_graph"] is False
    assert graph["evaluation"]["seeds"] == prior["evaluation"]["seeds"] == [101, 102, 103, 104, 105]
    from tests.test_ambi_research_presets import _build_cfg
    prior_cfg = _build_cfg(prior)
    graph_cfg = _build_cfg(graph)
    assert prior_cfg.inner_operator == "none"
    assert graph_cfg.inner_critic_updates_per_action == 128
    assert graph_cfg.inner_actor_updates_per_action == 32


def test_q_scale_carry_is_only_permitted_in_full_graph_episodes():
    params = {"inner_operator": "sac", "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
              "inner_actor_loss_scale_scope": "run", "inner_actor_loss_scale_update": "per_update"}
    for options, override in [({"episode_evaluation": False, "execution_backend": PROTOCOL}, {}),
                              ({"episode_evaluation": True}, {}),
                              ({"episode_evaluation": True, "execution_backend": PROTOCOL}, {"inner_actor_scope": "run"}),
                              ({"episode_evaluation": True, "execution_backend": PROTOCOL}, {"inner_actor_loss_scale_update": "per_action"})]:
        with pytest.raises(ValueError, match="Benchmark presets require"):
            evaluator._validate_benchmark_scopes({**params, **override}, **options)


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
def test_cuda_mean_graph_matches_control_and_resets_episode_rng():
    torch.set_float32_matmul_precision("high")
    agents = [make_agent("squashed", "action", "per_update", device="cuda", compiled=True) for _ in range(2)]
    agents[1].load(deepcopy(agents[0].checkpoint_state()))
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires scheduled CUDA GPU")
def test_cuda_frozen_controller_carries_scale_and_resets_independent_episodes():
    torch.set_float32_matmul_precision("high")
    agents = [make_agent("squashed", "run", "per_update", device="cuda", compiled=True)
              for _ in range(2)]
    agents[1].load(deepcopy(agents[0].checkpoint_state()))
    before = [outer_snapshot(agent) for agent in agents]
    observation = torch.zeros(agents[0].cfg.obs_shape["state"], device="cuda")
    with torch.no_grad():
        root = agents[0].model.encode(observation.unsqueeze(0)).detach()
    control = MeanActionGraph(agents[0].inner_engine, root, capture=False, seed=12345, warmup=4)
    model = SimpleNamespace(agent=agents[1], cfg=agents[1].cfg, _obs_to_tensor=lambda obs: obs)
    controller = FrozenGraphController(model, observation, 12345)
    engine = agents[1].inner_engine
    engine.reset_for_evaluation(12345)
    controller.reset_episode(12345)
    carried = agents[0].actor_loss_scale.detach().clone()
    first = None
    for index in range(3):
        expected, expected_metrics, _ = control(root + index / 10, initial_actor_loss_scale=carried)
        carried = expected_metrics["inner_actor_loss_scale_final"].detach().reshape_as(carried).clone()
        outside = torch.cuda.get_rng_state().clone()
        actual, metrics, _ = controller.act(root + index / 10, eval_mode=True, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert_tree_close({k: v for k, v in metrics.items() if k != "inner_cuda_graph_used"}, expected_metrics)
        assert_tree_close(inner_snapshot(controller.runner.engine), inner_snapshot(control.engine))
        torch.testing.assert_close(engine._carried_actor_loss_scale, carried, rtol=0, atol=0)
        assert engine._carried_actor_loss_scale.data_ptr() != metrics["inner_actor_loss_scale_final"].data_ptr()
        torch.testing.assert_close(outside, torch.cuda.get_rng_state(), rtol=0, atol=0)
        if index == 0:
            first = actual.detach().clone()
    engine.reset_for_evaluation(12345)
    controller.reset_episode(12345)
    actual, metrics, _ = controller.act(root, eval_mode=True, collect_diagnostics=False)
    torch.testing.assert_close(actual, first, rtol=0, atol=0)
    torch.testing.assert_close(metrics["inner_actor_loss_scale_initial"], agents[1].actor_loss_scale.reshape(()), rtol=0, atol=0)
    for agent, snapshot in zip(agents, before):
        assert_tree_close(outer_snapshot(agent), snapshot)
    controller.close()
    assert engine.act == controller.original_act
