"""K3 graph solves must preserve held feedback, encoder snapshots and resume."""

from copy import deepcopy
import os
from types import SimpleNamespace

import pytest
import torch

from tests.benchmark_whole_inner_graph import make_agent, assert_tree_close
from tests.test_training_inner_graph import RECIPE
from RL.tdmpc2_core.whole_inner_graph import validate_training_graph, training_graph_act


def test_k3_guard_and_solved_actor_snapshot():
    agent = make_agent(device="cpu", tiny=True, compile_enabled=False, config=RECIPE)
    engine = agent.inner_engine
    engine.cfg.inner_solve_interval = 3
    engine.cfg.compile = engine.cfg.compile_strict = True
    engine.device = torch.device("cuda")
    validate_training_graph(engine)
    engine._timer_start = lambda: None
    engine._timer_stop = lambda *args: None
    actor = torch.nn.Linear(8, 1)
    class Runner:
        engine = SimpleNamespace(_action_pool=SimpleNamespace(actor=actor))
        def __call__(self, root):
            return root[0], {}, []
    engine._whole_graph_runner = Runner()
    engine.prepare_control(t0=True)
    root = torch.zeros(1, 8)
    _, metrics, _ = training_graph_act(engine, root)
    assert metrics["inner_solve_performed"] == 1 and metrics["inner_action_age"] == 0
    assert engine._episode_decision_index == engine.action_index == 1
    assert engine._held_actor is not actor
    before = deepcopy(engine._held_actor.state_dict())
    with torch.no_grad():
        actor.weight.add_(1)
    assert_tree_close(engine._held_actor.state_dict(), before, rtol=0, atol=0)
    assert all(not p.requires_grad for p in engine._held_actor.parameters())


def test_graph_dispatch_failure_discards_held_state(monkeypatch):
    from RL.tdmpc2_core import whole_inner_graph
    agent = make_agent(device="cpu", tiny=True, compile_enabled=False, config=RECIPE)
    engine = agent.inner_engine
    agent.cfg.inner_solve_interval = 3
    agent.cfg.inner_cuda_graph = True
    monkeypatch.setattr(whole_inner_graph, "validate_training_graph", lambda engine: None)
    def fail(*args):
        raise RuntimeError("injected capture failure")
    monkeypatch.setattr(whole_inner_graph, "training_graph_act", fail)
    with pytest.raises(RuntimeError, match="injected capture failure"):
        agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    assert engine._held_actor is engine._held_encoder is engine._encoder_pool is None
    assert engine._control_prepared is None and engine._episode_decision_index == 0


@pytest.mark.skipif(os.environ.get("AMBI_RUN_INNER_GRAPH_CUDA") != "1" or not torch.cuda.is_available(),
                    reason="Requires explicit scheduled CUDA validation.")
def test_k3_graph_held_actions_updates_diagnostics_and_exact_resume():
    agent = make_agent(tiny=True, config=RECIPE)
    agent.cfg.inner_solve_interval = 3
    agent.cfg.inner_cuda_graph = True
    engine = agent.inner_engine
    generator = torch.Generator().manual_seed(902)
    h, b = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    data = (torch.randn(h+1, b, 3, generator=generator).cuda(),
            torch.randn(h, b, 1, generator=generator).tanh().cuda(),
            torch.randn(h, b, 1, generator=generator).cuda(), torch.zeros(h, b, 1, device="cuda"))
    agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    initial_runner = engine._whole_graph_runner
    agent.prepare_training_resume_boundary()
    saved = deepcopy(agent.training_state_dict())
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    results = []
    for attempt in range(2):
        if attempt:
            agent.load_training_state_dict(saved)
            assert engine._whole_graph_runner is None
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        agent.reset()
        sequence = []
        for index in range(8):
            agent._update(*data)  # Outer encoder/actor evolve on held decisions too.
            rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            action = agent.act(torch.full((3,), index * .1), collect_diagnostics=index == 3)
            metrics = {k:v for k,v in agent.last_inner_metrics.items() if not k.endswith("seconds")}
            assert_tree_close((torch.get_rng_state(), torch.cuda.get_rng_state()), rng, rtol=0, atol=0)
            solve = index % 3 == 0
            assert metrics["inner_solve_performed"] == float(solve)
            assert metrics["inner_cuda_graph_used"] == float(solve and index != 3)
            assert metrics["inner_critic_optimizer_steps"] == (4 if solve else 0)
            assert metrics["inner_actor_optimizer_steps"] == (2 if solve else 0)
            held = (deepcopy(engine._held_actor.state_dict()), deepcopy(engine._held_encoder.state_dict()))
            if not solve:
                assert_tree_close(held, sequence[-1][2], rtol=0, atol=0)
            sequence.append((action.clone(), metrics, held))
        assert engine._whole_graph_runner.calls == saved["inner"]["whole_graph"]["calls"] + 2
        with pytest.raises(RuntimeError, match="boundary|held|cadence|allocation|transfer"):
            agent.training_state_dict()
        agent.prepare_training_resume_boundary()
        results.append((sequence, deepcopy(agent.training_state_dict())))
    assert engine._whole_graph_runner is not initial_runner
    assert_tree_close(results[0], results[1], rtol=0, atol=0)
