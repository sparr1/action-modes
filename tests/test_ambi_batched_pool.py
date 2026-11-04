"""Ordering, lifetime, failure cleanup and CUDA parity for batched workers."""

from copy import deepcopy
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from RL.tdmpc2_core import batched_teacher_pool as pool
from RL.tdmpc2_core.batched_teacher import BatchedTeacherSolver
from RL.tdmpc2_core.batched_teacher_pool import ConcurrentBatchedTeacherSolver
from tests.test_ambi_batched_teacher import _agent
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from tests.test_ambi_parallel_teacher import _Runtime, _Stream
from tests.test_ambi_reanalyze import _globals, _assert_globals


class _FakeAgent(torch.nn.Module):
    def __init__(self, workers=2, width=3):
        super().__init__()
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(
            latent_dim=4, action_dim=2, reanalyze_parallel_roots=width,
            reanalyze_batched_workers=workers, reanalyze_backend="batched_cuda_graph",
        )
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh())
        self.model[1].eval()
        self.inner_engine = object()


class _FakeWorker:
    def __init__(self, agent, runtime, slot, *, fail_setup=False, fail_seed=None):
        self.agent, self.runtime, self.slot = agent, runtime, slot
        self.stream = None
        self.core = None
        self.fail_setup, self.fail_seed = fail_setup, fail_seed

    def _setup(self):
        self.runtime.events.append(("capture", self.slot))
        self.stream = _Stream(self.runtime, f"worker{self.slot}")
        random.random()
        np.random.rand()
        torch.rand(())
        if self.fail_setup:
            raise RuntimeError("simulated capture failure")
        width = self.agent.cfg.reanalyze_parallel_roots
        self.core = SimpleNamespace(
            root=torch.empty(width, self.agent.cfg.latent_dim),
            counts=dict(inner_total_model_steps=12, inner_actor_optimizer_steps=2,
                        inner_critic_optimizer_steps=8),
            refresh_inputs=self._refresh,
        )
        self.output = (torch.zeros(width, 2), torch.zeros(width, 2))
        self.graph = SimpleNamespace(replay=self._replay)

    def _refresh(self):
        self.runtime.events.append(("refresh", self.slot))
        self.prior = self.agent.model[0].weight.detach()[0, 0].clone()

    def _seed(self, seeds):
        self.seed = BatchedTeacherSolver._seed(self, seeds)
        return self.seed

    def _replay(self):
        self.runtime.events.append(("replay", self.slot, self.seed))
        assert self.runtime.active is self.stream
        if self.seed == self.fail_seed:
            raise RuntimeError("simulated replay failure")
        self.output[0].copy_(self.core.root[:, :2] + torch.randn_like(self.output[0]) + self.prior)
        self.output[1].copy_(torch.randn_like(self.output[1]))


@pytest.fixture
def runtime(monkeypatch):
    value = _Runtime()
    monkeypatch.setattr(pool, "_require_cuda", lambda _: None)
    monkeypatch.setattr(torch.cuda, "current_stream", value.current_stream)
    monkeypatch.setattr(torch.cuda, "stream", value.stream)
    return value


def _install(monkeypatch, solver, runtime, *, fail_setup_at=None, fail_seed=None):
    created = []

    def make():
        worker = _FakeWorker(solver.agent, runtime, len(created),
                             fail_setup=len(created) == fail_setup_at, fail_seed=fail_seed)
        created.append(worker)
        return worker

    monkeypatch.setattr(solver, "_new_worker", make)
    return created


def _expected(agent, roots, seeds):
    values = [[], []]
    width = agent.cfg.reanalyze_parallel_roots
    for start in range(0, len(roots), width):
        seed = BatchedTeacherSolver._seed(None, seeds[start:start + width])
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(width, 2, generator=generator)
        log_std = torch.randn(width, 2, generator=generator)
        count = len(roots[start:start + width])
        values[0].append(roots[start:start + width, :2] + noise[:count] + agent.model[0].weight.detach()[0, 0])
        values[1].append(log_std[:count])
    return tuple(torch.cat(value) for value in values)


def test_chunks_keep_seeds_order_and_private_outputs_across_worker_counts(monkeypatch, runtime):
    roots, seeds = torch.arange(40, dtype=torch.float32).reshape(10, 4), list(range(11, 21))
    agent = _FakeAgent()
    outputs = []
    for count in (1, 2, 4):
        agent.cfg.reanalyze_batched_workers = count
        solver = ConcurrentBatchedTeacherSolver(agent)
        created = _install(monkeypatch, solver, runtime)
        rng = _globals()
        result = solver.solve(roots, seeds)
        _assert_globals(rng)
        assert len(created) == count
        for actual, expected in zip(result[:2], _expected(agent, roots, seeds)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(result[2], torch.tensor([120., 20., 80.]))
        assert result[3]["distillation_reanalyze_chunks"] == 4
        assert result[3]["distillation_reanalyze_padding_roots"] == 2
        assert result[3]["distillation_reanalyze_executed_model_steps"] == 144
        assert result[3]["distillation_reanalyze_batched_workers"] == count
        original = tuple(value.clone() for value in result[:2])
        second = solver.solve(roots + 3, seeds)
        assert second[3]["distillation_reanalyze_setup_seconds"] == 0
        _assert_tree_equal(result[:2], original)
        outputs.append(result[:3])
    for result in outputs[1:]:
        _assert_tree_equal(result, outputs[0])


def test_setup_all_before_launch_and_only_join_after_last_launch(monkeypatch, runtime):
    agent = _FakeAgent(workers=2)
    solver = ConcurrentBatchedTeacherSolver(agent)
    created = _install(monkeypatch, solver, runtime)
    modes = [(m, m.training) for m in agent.modules()]
    parameters = deepcopy(agent.model.state_dict())
    live = agent.inner_engine
    for parameter in agent.parameters():
        parameter.grad = torch.randn_like(parameter)
    gradients = [parameter.grad.clone() for parameter in agent.parameters()]
    rng = _globals()
    clone = torch.Tensor.clone
    copied = []

    def check_clone(value, *args, **kwargs):
        for worker in created:
            if worker.core is not None and any(value.data_ptr() == output.data_ptr() for output in worker.output):
                assert runtime.active is worker.stream
                copied.append(worker.slot)
        return clone(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", check_clone)
    solver.solve(torch.zeros(10, 4), list(range(10)))
    events = runtime.events
    assert events[:2] == [("capture", 0), ("capture", 1)]
    replays = [i for i, event in enumerate(events) if event[0] == "replay"]
    joins = [i for i, event in enumerate(events) if event[:2] == ("wait", "caller")]
    assert min(joins) > max(replays)
    assert not any(event[0] == "wait" and event[1] != "caller" and event[2] != "caller" for event in events)
    assert events[-1] == ("sync", "caller")
    assert sum(event[0] == "sync" for event in events) == 1
    assert copied == [0, 0, 1, 1, 0, 0, 1, 1]
    assert all(m.training == mode for m, mode in modes)
    assert agent.inner_engine is live
    _assert_tree_equal(agent.model.state_dict(), parameters)
    for parameter, expected in zip(agent.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    _assert_globals(rng)


@pytest.mark.parametrize("failure", ["capture", "replay", "nonfinite"])
def test_failure_joins_all_workers_discards_pool_and_restores_state(monkeypatch, runtime, failure):
    agent = _FakeAgent(workers=2)
    solver = ConcurrentBatchedTeacherSolver(agent)
    seeds = list(range(10))
    fail_seed = BatchedTeacherSolver._seed(None, seeds[3:6]) if failure == "replay" else None
    _install(monkeypatch, solver, runtime, fail_setup_at=1 if failure == "capture" else None,
             fail_seed=fail_seed)
    roots = torch.zeros(10, 4)
    if failure == "nonfinite":
        roots[1, 0] = float("nan")
    modes = [(m, m.training) for m in agent.modules()]
    rng = _globals()
    with pytest.raises((RuntimeError, ValueError), match="simulated|finite"):
        solver.solve(roots, seeds)
    assert not solver.workers and not solver._active
    for slot in range(2):
        assert ("sync", f"worker{slot}") in runtime.events
    assert all(m.training == mode for m, mode in modes)
    _assert_globals(rng)


def test_empty_input_does_not_capture_or_synchronize(runtime):
    solver = ConcurrentBatchedTeacherSolver(_FakeAgent())
    mean, std, counts, metrics = solver.solve(torch.empty(0, 4), [])
    assert mean.shape == std.shape == (0, 2)
    assert not counts.any() and not solver.workers and not runtime.events
    assert metrics["distillation_reanalyze_batched_workers"] == 0


def test_reentrant_and_invalid_input_rejected_before_setup(monkeypatch, runtime):
    solver = ConcurrentBatchedTeacherSolver(_FakeAgent())
    created = _install(monkeypatch, solver, runtime)
    with pytest.raises(ValueError, match="float32"):
        solver.solve(torch.zeros(3, 5), [1, 2, 3])
    with pytest.raises(ValueError, match="seed"):
        solver.solve(torch.zeros(3, 4), [1, True, 3])
    solver._active = True
    with pytest.raises(RuntimeError, match="reentrant"):
        solver.solve(torch.zeros(3, 4), [1, 2, 3])
    assert not created


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a scheduled CUDA GPU.")


@CUDA
@pytest.mark.parametrize("compiled,optimized", [(False, False), (True, False), (True, True)])
def test_cuda_worker_counts_match_serial_chunks_and_preserve_live_state(compiled, optimized):
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.reset()
    agent = _agent(device="cuda", compile=compiled, compile_strict=compiled,
                   q_representation="distributional", dropout=.1,
                   inner_critic_dropout_enabled=True, inner_finite_horizon=True,
                   inner_critic_updates_per_round=3, inner_actor_updates_per_round=1,
                   reanalyze_batched_optimizations=optimized,
                   reanalyze_batched_workers=1)
    agent.cfg.reanalyze_backend = "batched_cuda_graph"
    serial = BatchedTeacherSolver(agent)
    pools = []
    for count in (1, 2, 4):
        agent.cfg.reanalyze_batched_workers = count
        pools.append(ConcurrentBatchedTeacherSolver(agent))
    roots = torch.randn(10, agent.cfg.latent_dim, device="cuda")
    seeds = list(range(231, 241))
    agent.model.train()
    agent.model._pi[0].eval()
    modes = [(m, m.training) for m in agent.modules()]
    for p in agent.model.parameters():
        p.grad = torch.randn_like(p)
    gradients = [p.grad.clone() for p in agent.model.parameters()]
    live = deepcopy(agent.inner_engine.training_state_dict())
    for iteration in range(2):
        with torch.no_grad():
            for p in agent.model.parameters():
                p.add_(.002 * (iteration + 1))
            if agent.log_ent_coef is not None:
                agent.log_ent_coef.add_(.1)
            else:
                agent.fixed_ent_coef.mul_(1.1)
            if agent.actor_loss_scale is not None:
                agent.actor_loss_scale.fill_(2. + iteration)
        parameters = deepcopy(agent.model.state_dict())
        rng = _globals()
        cuda_rng = torch.cuda.get_rng_state().clone()
        expected = serial.solve(roots, seeds)
        for solver in pools:
            actual = solver.solve(roots, seeds)
            # Same shape, kernels and chunk seeds: concurrency alone must not
            # alter a single value. This also exercises worker reuse/padding.
            for a, e in zip(actual[:3], expected[:3]):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            repeated = solver.solve(roots, seeds)
            _assert_tree_equal(actual[:3], repeated[:3])
            assert all(m.training == mode for m, mode in modes)
            _assert_tree_equal(agent.model.state_dict(), parameters)
            _assert_tree_equal(agent.inner_engine.training_state_dict(), live)
            for p, grad in zip(agent.model.parameters(), gradients):
                torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
        _assert_globals(rng)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    torch._dynamo.reset()
