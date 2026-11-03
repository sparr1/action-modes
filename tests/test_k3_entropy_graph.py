"""K3 keeps the solved feedback policy frozen between captured solves."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from RL.tdmpc2_core.whole_inner_graph import WholeInnerGraph, validate_training_graph
from tests.test_ambi_root_local_sac import _build_cfg
from tests.test_inner_entropy_graph import assert_tree_close, inner_snapshot, outer_snapshot

ROOT = Path(__file__).resolve().parents[1]
NAME = "ambi_inner_entropy_posttanh_outer_adapt_k3"
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires scheduled CUDA GPU.")


def params():
    return json.loads((ROOT / f"configs/dmcontrol/algs/{NAME}.json").read_text())["alg_params"]


def make_agent(*, device="cpu", compiled=False):
    p = params()
    p.update(model_size=None, enc_dim=16, mlp_dim=16, latent_dim=8,
             num_enc_layers=2, simnorm_dim=4, batch_size=4, buffer_size=32,
             inner_rounds=1, inner_rollouts_per_round=4, inner_batch_size=4,
             inner_replay_capacity=12, inner_critic_updates_per_round=2,
             inner_actor_updates_per_round=2, inner_cuda_graph=False,
             critic_geometry_diagnostics=False, wandb=False, wandb_mode="disabled",
             compile=compiled, compile_strict=compiled)
    cfg = _build_cfg(**p)
    cfg.action_dim, cfg.action_dims = 21, [21]
    cfg.obs_shape, cfg.device = {"state": (67,)}, device
    cfg.obs_shapes = [cfg.obs_shape]
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    return agent


def test_k3_only_changes_cadence_and_publication():
    original = json.loads((ROOT / "configs/dmcontrol/algs/ambi_inner_entropy_posttanh_outer_adapt.json").read_text())
    changed = json.loads((ROOT / f"configs/dmcontrol/algs/{NAME}.json").read_text())
    for config in (original, changed):
        config["alg_params"] = {k:v for k,v in config["alg_params"].items() if not k.startswith("wandb")}
    original["alg_params"]["inner_solve_interval"] = 3
    assert original == changed
    manifest = json.loads((ROOT / "configs/dmcontrol/experiments/ambi_posttanh_outer_adapt_k3.json").read_text())
    assert manifest["configs"] == [NAME] and manifest["trials"] == 1
    assert manifest["overrides_alg"]["total_steps"] == 1_000_000
    assert manifest["overrides_alg"]["seed"] == 55
    assert manifest["checkpoint_every"] == 25000 and manifest["save_strat"] == "all"


def test_k3_graph_contract_and_worker_cadence():
    cfg = _build_cfg(**params())
    engine = SimpleNamespace(cfg=cfg, device=torch.device("cuda"), _active_trace=None, _retrace_enabled=False)
    validate_training_graph(engine)
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
    agent = make_agent()
    worker = InnerImprovementEngine(agent)
    assert agent.inner_engine._solve_interval == worker._solve_interval == 3
    worker._whole_graph_worker = True
    assert worker._solve_interval == 1 and agent.inner_engine._solve_interval == 3


def test_explicit_actor_snapshot_owns_its_parameters():
    agent = make_agent()
    agent.act(torch.zeros(67), t0=True, collect_diagnostics=False)
    engine = agent.inner_engine
    source = engine._action_pool.actor
    engine._snapshot_held_actor(source)
    saved = deepcopy(engine._held_actor.state_dict())
    with torch.no_grad():
        next(source.parameters()).add_(1)
    assert_tree_close(engine._held_actor.state_dict(), saved)
    assert all(not p.requires_grad for p in engine._held_actor.parameters())


def assert_k3_graph_resume_continuation(agent):
    """Production-size gate for holds, native diagnostics, resets and recapture."""
    engine = agent.inner_engine
    assert engine._whole_graph_runner is not None and engine._whole_graph_runner.calls > 0
    agent.prepare_training_resume_boundary()
    saved = deepcopy(agent.training_state_dict())
    original_runner = engine._whole_graph_runner
    results = []
    for attempt in range(2):
        if attempt:
            agent.load_training_state_dict(deepcopy(saved))
            assert agent.inner_engine._whole_graph_runner is None
        engine = agent.inner_engine
        agent.reset()
        cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        sequence = []
        for index, age in enumerate((0, 1, 2, 0, 1, 2, 0, 0, 1)):
            if index == 7:
                agent.reset()
            with torch.no_grad():
                next(agent.model._encoder["state"].parameters()).add_(.0001)
                next(agent.model._pi.parameters()).add_(.0001)
                agent.actor_loss_scale.fill_(4. + index)
            held_actor = deepcopy(engine._held_actor.state_dict()) if age else None
            held_encoder = deepcopy(engine._held_encoder.state_dict()) if age else None
            runner = engine._whole_graph_runner
            calls = 0 if runner is None else runner.calls
            action = agent.act(torch.full(agent.cfg.obs_shape["state"], .1 * index),
                               t0=index in (0, 7), collect_diagnostics=index in (3, 4))
            metrics = deepcopy(agent.last_inner_metrics)
            solve = age == 0
            assert all(math.isfinite(float(value)) for value in metrics.values())
            assert metrics["inner_action_age"] == age
            assert metrics["inner_policy_held"] == float(not solve)
            assert metrics["inner_solve_performed"] == float(solve)
            assert metrics["inner_actor_optimizer_steps"] == (agent.cfg.inner_actor_updates_per_action if solve else 0)
            assert metrics["inner_critic_optimizer_steps"] == (agent.cfg.inner_critic_updates_per_action if solve else 0)
            assert metrics["inner_cuda_graph_used"] == float(solve and index != 3)
            if solve:
                assert metrics["inner_actor_loss_scale_initial"] == pytest.approx(4. + index)
                if index == 3:
                    assert metrics["inner_diagnostics_sampled"] == 1
            else:
                assert engine._whole_graph_runner.calls == calls
                assert_tree_close(engine._held_actor.state_dict(), held_actor)
                assert_tree_close(engine._held_encoder.state_dict(), held_encoder)
                assert "inner_actor_loss_scale_initial" not in metrics
            sequence.append((action.clone(), {k:v for k,v in metrics.items() if not k.endswith("seconds")}))
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), gpu_rng, rtol=0, atol=0)
        agent.prepare_training_resume_boundary()
        assert engine._held_actor is engine._held_encoder is None
        results.append((sequence, deepcopy(agent.training_state_dict())))
    assert agent.inner_engine._whole_graph_runner is not original_runner
    assert_tree_close(results[0], results[1])
    agent.load_training_state_dict(deepcopy(saved))


@CUDA
@pytest.mark.parametrize("interval", (1, 3))
def test_cuda_k3_capture_continuation(interval):
    if interval == 1:
        from tests.test_inner_entropy_graph import test_cuda_capture_matches_uncaptured_normalized_solve
        test_cuda_capture_matches_uncaptured_normalized_solve("squashed", "action", "per_update")
        return
    control, graph = [make_agent(device="cuda", compiled=True) for _ in range(2)]
    graph.load(deepcopy(control.checkpoint_state()))
    for agent, capture in ((control, False), (graph, True)):
        agent.cfg.inner_cuda_graph = True
        worker = InnerImprovementEngine(agent)
        worker._whole_graph_worker = True
        agent.inner_engine._whole_graph_runner = WholeInnerGraph(
            worker, torch.zeros(1, agent.cfg.latent_dim, device="cuda"),
            capture=capture, seed=int(agent.cfg.seed) + 8173, warmup=4,
        )
    for index, age in enumerate((0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 2, 0)):
        cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        results = []
        for agent in (control, graph):
            with torch.no_grad():
                next(agent.model._encoder["state"].parameters()).add_(.0001)
                next(agent.model._pi.parameters()).add_(.0001)
                agent.actor_loss_scale.fill_(4. + index)
            before = outer_snapshot(agent)
            action = agent.act(torch.full((67,), .01 * index), t0=index in (0, 8),
                               collect_diagnostics=index in (3, 4))
            assert_tree_close(outer_snapshot(agent), before)
            metrics = {k:v for k,v in agent.last_inner_metrics.items() if not k.endswith("seconds")}
            assert metrics["inner_action_age"] == age
            assert metrics["inner_solve_performed"] == float(age == 0)
            results.append((action, metrics))
        assert_tree_close(results[1], results[0], rtol=5e-5, atol=5e-7)
        assert_tree_close(inner_snapshot(graph.inner_engine._whole_graph_runner.engine),
                          inner_snapshot(control.inner_engine._whole_graph_runner.engine), rtol=5e-5, atol=5e-7)
        assert_tree_close(graph.inner_engine._held_actor.state_dict(), control.inner_engine._held_actor.state_dict(), rtol=5e-5, atol=5e-7)
        assert_tree_close(graph.inner_engine.rng.training_state_dict(), control.inner_engine.rng.training_state_dict())
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), gpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(graph.inner_engine._whole_graph_runner.rng_state,
                                   control.inner_engine._whole_graph_runner.rng_state, rtol=0, atol=0)
    assert graph.inner_engine._whole_graph_runner.calls == 4
    assert_k3_graph_resume_continuation(graph)
