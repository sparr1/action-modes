from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import evaluate_ambi_checkpoint as evaluator
from RL.tdmpc2_core.frozen_graph import FrozenGraphController, MeanActionGraph, PROTOCOL
from RL.tdmpc2_core.whole_inner_graph import validate_training_graph
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.whole_inner_graph import _gsde_inputs_for_action, _GsdeGraphInputs
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_params
from tests.test_ambi_distillation_collection_graph import _inner_snapshot as inner_snapshot
from tests.test_ambi_distillation_collection_graph import _assert_close as assert_tree_close
from tests.test_ambi_inner_decoupling import _clone_tree
from tests.test_ambi_gsde_lifecycle import _model


def make_agent(entropy="squashed", scope="action", update="per_update", *, device="cpu", compiled=False, distribution="gaussian"):
    params = _tiny_params(
        actor_distribution=distribution, compile=compiled, compile_strict=compiled,
        critic_ensemble_backend="vmap", q_representation="distributional", num_q=5,
        q_num_bins=101, q_vmin=-10, q_vmax=10, q_pair_size=2, num_bins=101,
        outer_q_target_reduction="min_pair", outer_q_actor_reduction="mean_pair",
        inner_q_target_reduction="min_pair", inner_q_actor_reduction="mean_pair",
        dropout=.01, inner_critic_dropout_enabled=True,
        outer_actor_entropy_mode=entropy, inner_actor_entropy_mode=entropy,
        sac_actor_loss_scale_mode="tdmpc2_percentile_range", sac_actor_loss_scale_tau=.25,
        inner_actor_loss_scale_scope=scope, inner_actor_loss_scale_update=update,
        outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
        ent_coef=.0021, inner_temperature_mode="inherit_outer", aux_return_mode="off",
        inner_rounds=2, inner_rollouts_per_round=4, inner_batch_size=4,
        inner_critic_updates_per_round=2, inner_actor_updates_per_round=2,
        inner_replay_capacity=16, inner_finite_horizon=True, inner_solve_interval=1,
        inner_root_entropy_samples=0, inner_gsde_noise_scope="real_steps",
        inner_gsde_resample_steps=2, critic_geometry_diagnostics=False,
    )
    params.pop("inner_updates_per_round", None)
    cfg = _build_cfg(**params)
    cfg.device = device
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    agent.actor_loss_scale.fill_(4.)
    return agent


def outer_snapshot(agent):
    return {"model": deepcopy(agent.model.state_dict()), "scale": agent.actor_loss_scale.clone(),
            "alpha": agent.alpha.clone(), "optimizers": {
                name: None if getattr(agent, name) is None else deepcopy(getattr(agent, name).state_dict())
                for name in ("optim", "pi_optim", "ent_coef_optim")}}
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.eval_series_data import planner_identity


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_checkpoint_graph_eval.json"


def test_matrix_preserves_saved_controller_and_selects_paired_prior():
    import json
    saved = json.loads((ROOT / "configs/dmcontrol/algs/ambi_gsde_squashed_real_steps.json").read_text())
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
        cfg=SimpleNamespace(inner_rounds=8, inner_actor_updates_per_action=32, compile=True, compile_strict=True,
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
        cfg=SimpleNamespace(inner_rounds=8, inner_actor_updates_per_action=32, compile=True, compile_strict=True,
                            inner_execution_action="mean", inner_cuda_graph=True),
        device=torch.device("cuda"), _retrace_enabled=False, _active_trace=None,
        _whole_graph_worker=False, _teacher_only=False,
        _act=lambda *args, **kwargs: (None, {}, []),
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
@pytest.mark.parametrize("distribution", ["gaussian", "gsde"])
def test_cuda_mean_graph_matches_control_and_resets_episode_rng(distribution):
    torch.set_float32_matmul_precision("high")
    agents = [make_agent(device="cuda", compiled=True, distribution=distribution) for _ in range(2)]
    agents[1].load(deepcopy(agents[0].checkpoint_state()))
    before = [outer_snapshot(agent) for agent in agents]
    root = torch.randn(1, agents[0].cfg.latent_dim, device="cuda")
    workers = [InnerImprovementEngine(agent) for agent in agents]
    runners = [MeanActionGraph(worker, root, capture=capture, seed=12345)
               for worker, capture in zip(workers, (False, True))]
    first = first_noise = None
    for index in range(3):
        outputs = []
        for agent, runner in zip(agents, runners):
            engine = agent.inner_engine
            outside = torch.cuda.get_rng_state().clone()
            inputs = {}
            if distribution == "gsde":
                engine._begin_gsde_action(t0=index == 0, eval_mode=True)
                collection, execution = _gsde_inputs_for_action(engine, root, eval_mode=True)
                assert execution is None and runner.gsde_inputs.execution is None
                inputs = dict(gsde_collection=collection, gsde_execution=execution)
            action, metrics, lengths = runner(root + index / 10, **inputs)
            outputs.append(_clone_tree((action, metrics)))
            with torch.no_grad():
                expected, _ = agent.model.pi(root + index / 10, policy=runner.engine._action_pool.actor, deterministic=True)
            torch.testing.assert_close(action, expected[0], rtol=0, atol=0)
            torch.testing.assert_close(outside, torch.cuda.get_rng_state(), rtol=0, atol=0)
            if distribution == "gsde":
                engine._finish_gsde_action()
        assert_tree_close(outputs[0], outputs[1])
        assert_tree_close(inner_snapshot(workers[0]), inner_snapshot(workers[1]))
        if index == 0:
            first, first_noise = outputs[1][0], _clone_tree(inputs)
    runners[1].reset_rng()
    action, _, _ = runners[1](root, **first_noise)
    torch.testing.assert_close(action, first, rtol=0, atol=0)
    for agent, snapshot in zip(agents, before):
        assert_tree_close(outer_snapshot(agent), snapshot, exact=True)


@pytest.mark.parametrize("scope", ["real_steps", "solve", "round"])
def test_frozen_gsde_adapter_matches_native_noise_lifecycle(scope):
    models = [_model(inner_gsde_noise_scope=scope, inner_gsde_resample_steps=2) for _ in range(2)]
    native, live = [model.agent.inner_engine for model in models]
    root = native.model.encode(torch.zeros(1, 3)).detach()
    controller = object.__new__(FrozenGraphController)
    controller.engine = live
    received = []
    inputs = _GsdeGraphInputs(live, root, eval_mode=True)
    def runner(root, *, gsde_collection, gsde_execution):
        inputs.copy_(gsde_collection, gsde_execution)
        received.append(_clone_tree(gsde_collection))
        assert gsde_execution is None
        return root.new_zeros(1), {}, []
    controller.runner = runner
    try:
        execution_before = live.rng.generator("execution").get_state().clone()
        for index in range(5):
            t0 = index in {0, 3}
            expected = []
            original = native._gsde_noise_for_round
            def capture(*args):
                noise = original(*args)
                expected.append(noise.clone())
                return noise
            native._gsde_noise_for_round = capture
            native._act(root, t0=t0, eval_mode=True, collect_diagnostics=False)
            native._gsde_noise_for_round = original
            controller.act(root, t0=t0, eval_mode=True, collect_diagnostics=False)
            distinct = expected if scope == "round" else expected[:1]
            assert_tree_close(received[-1], distinct, exact=True)
            assert native._gsde_decision_index == live._gsde_decision_index
            torch.testing.assert_close(native.rng.generator("collection").get_state(), live.rng.generator("collection").get_state())
        torch.testing.assert_close(live.rng.generator("execution").get_state(), execution_before)
    finally:
        for model in models:model.env.close()


def test_gsde_distribution_has_explicit_publication_identity():
    config = dict(inner_operator="sac", inner_rounds=8, actor_distribution="gsde", gsde_feature_source="actor", inner_gsde_noise_scope="real_steps")
    graph = planner_identity(config, {"execution_backend": PROTOCOL}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    assert graph["settings"]["actor_distribution"] == "gsde"
    assert graph["settings"]["inner_gsde_noise_scope"] == "real_steps"
    config["inner_operator"] = "none"
    prior = planner_identity(config, {}, "AMBITDMPC2/AMBITDMPC2", "tanh_mean")
    assert prior["settings"]["actor_distribution"] == "gsde"
