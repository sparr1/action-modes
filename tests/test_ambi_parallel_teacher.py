"""Independent root slots, copied graph outputs, and stream ownership."""

from contextlib import contextmanager
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from RL.tdmpc2_core import parallel_teacher
from RL.tdmpc2_core.parallel_teacher import ParallelTeacherSolver
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_reanalyze import _globals, _assert_globals


class _Agent(torch.nn.Module):
    def __init__(self, width=2):
        super().__init__()
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(
            latent_dim=4, action_dim=2, reanalyze_parallel_roots=width,
        )
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh())
        self.model[1].eval()  # Mixed modes must also be restored.
        self.inner_engine = object()


class _Stream:
    def __init__(self, runtime, name):
        self.runtime, self.name = runtime, name

    def wait_stream(self, other):
        self.runtime.events.append(("wait", self.name, other.name))

    def synchronize(self):
        self.runtime.events.append(("sync", self.name))


class _Runtime:
    def __init__(self):
        self.events = []
        self.caller = _Stream(self, "caller")
        self.active = self.caller

    def current_stream(self, *_):
        return self.active

    @contextmanager
    def stream(self, stream):
        before = self.active
        self.active = stream
        try:
            yield
        finally:
            self.active = before


class _Runner:
    def __init__(self, runtime, slot):
        self.runtime = runtime
        self.stream = _Stream(runtime, f"slot{slot}")
        self.warmup_seconds, self.capture_seconds = 0.25, 0.125
        self.teacher = {
            "pre_tanh_mean": torch.zeros(1, 2),
            "log_std": torch.zeros(2),
        }
        self.fail_seed = None

    def __call__(self, root, *, seed):
        self.runtime.events.append(("solve", self.stream.name, seed))
        self.stream.wait_stream(self.runtime.current_stream())
        if seed == self.fail_seed:
            raise RuntimeError("simulated graph replay failure")
        # Static output buffers are deliberately overwritten on every call.
        generator = torch.Generator().manual_seed(seed)
        with self.runtime.stream(self.stream):
            self.teacher["pre_tanh_mean"].copy_(root[:, :2] + torch.randn(1, 2, generator=generator))
            self.teacher["log_std"].fill_(-seed / 100.0)
            torch.rand(())  # Any incidental global draws must be isolated.
        return self.teacher, {
            "inner_total_model_steps": 8,
            "inner_actor_optimizer_steps": 1,
            "inner_critic_optimizer_steps": 2,
        }


@pytest.fixture
def fake_runtime(monkeypatch):
    runtime = _Runtime()
    monkeypatch.setattr(parallel_teacher, "_require_cuda", lambda _: None)
    monkeypatch.setattr(torch.cuda, "current_stream", runtime.current_stream)
    monkeypatch.setattr(torch.cuda, "stream", runtime.stream)
    return runtime


def _install_runners(monkeypatch, solver, runtime, *, fail_factory_at=None, fail_seed=None):
    created = []

    def create(root):
        slot = len(created)
        runtime.events.append(("capture", slot))
        random.random()
        np.random.rand()
        torch.rand(())
        if slot == fail_factory_at:
            raise RuntimeError("simulated graph capture failure")
        runner = _Runner(runtime, slot)
        runner.fail_seed = fail_seed
        created.append(runner)
        return runner

    monkeypatch.setattr(solver, "_make_runner", create)
    return created


def _expected(roots, seeds):
    return torch.cat([
        roots[i:i + 1, :2] + torch.randn(1, 2, generator=torch.Generator().manual_seed(seed))
        for i, seed in enumerate(seeds)
    ])


def test_partial_chunks_copy_static_outputs_and_join_before_return(monkeypatch, fake_runtime):
    agent = _Agent(width=2)
    solver = ParallelTeacherSolver(agent)
    assert not solver.runners
    created = _install_runners(monkeypatch, solver, fake_runtime)
    cloned = []
    original_clone = torch.Tensor.clone

    def clone(value, *args, **kwargs):
        for runner in created:
            if any(value.data_ptr() == output.data_ptr() for output in runner.teacher.values()):
                assert fake_runtime.active is runner.stream
                cloned.append(runner.stream.name)
        return original_clone(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    roots = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    seeds = [11, 12, 13, 14, 15]
    modes = [module.training for module in agent.modules()]
    parameters = _clone_tree(agent.model.state_dict())
    rng = _globals()
    live_engine = agent.inner_engine
    mean, log_std, totals, metrics = solver.solve(roots, seeds)
    torch.testing.assert_close(mean, _expected(roots, seeds), rtol=0, atol=0)
    torch.testing.assert_close(log_std[:, 0], -torch.tensor(seeds) / 100.0)
    torch.testing.assert_close(totals, torch.tensor([40.0, 5.0, 10.0]))
    assert len(created) == 2 and len(cloned) == 10
    assert fake_runtime.events[:2] == [("capture", 0), ("capture", 1)]
    assert fake_runtime.events[-3:] == [
        ("wait", "caller", "slot0"), ("wait", "caller", "slot1"), ("sync", "caller"),
    ]
    assert sum(event[0] == "sync" for event in fake_runtime.events) == 1
    assert metrics["distillation_reanalyze_parallel_roots"] == 2
    assert metrics["distillation_reanalyze_chunks"] == 3
    assert metrics["distillation_reanalyze_warmup_seconds"] == 0.5
    assert metrics["distillation_reanalyze_capture_seconds"] == 0.25
    assert metrics["distillation_reanalyze_setup_seconds"] > 0
    assert [module.training for module in agent.modules()] == modes
    assert agent.inner_engine is live_engine
    _assert_tree_equal(agent.model.state_dict(), parameters)
    _assert_globals(rng)
    # A subsequent sweep reuses slots and cannot overwrite returned targets.
    before = mean.clone()
    _, _, _, second = solver.solve(roots, [21, 22, 23, 24, 25])
    torch.testing.assert_close(mean, before, rtol=0, atol=0)
    assert second["distillation_reanalyze_setup_seconds"] == 0
    assert len(created) == 2


def test_root_seed_results_do_not_depend_on_pool_width(monkeypatch, fake_runtime):
    roots = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    seeds = [17, 18, 19, 20, 21]
    outputs = []
    for width in (1, 2, 4):
        solver = ParallelTeacherSolver(_Agent(width))
        _install_runners(monkeypatch, solver, fake_runtime)
        outputs.append(solver.solve(roots, seeds)[:3])
    for result in outputs[1:]:
        _assert_tree_equal(result, outputs[0])


@pytest.mark.parametrize("failure", ["capture", "replay"])
def test_failure_joins_discards_pool_and_restores_modes_rng(monkeypatch, fake_runtime, failure):
    agent = _Agent()
    solver = ParallelTeacherSolver(agent)
    _install_runners(
        monkeypatch, solver, fake_runtime,
        fail_factory_at=1 if failure == "capture" else None,
        fail_seed=13 if failure == "replay" else None,
    )
    roots = torch.zeros(3, 4)
    modes = [module.training for module in agent.modules()]
    rng = _globals()
    with pytest.raises(RuntimeError, match="simulated graph"):
        solver.solve(roots, [11, 12, 13])
    assert not solver.runners and not solver._active
    assert ("sync", "slot0") in fake_runtime.events
    if failure == "replay":
        assert ("sync", "slot1") in fake_runtime.events
    assert [module.training for module in agent.modules()] == modes
    _assert_globals(rng)


def test_empty_roots_do_not_initialize_or_synchronize(fake_runtime):
    solver = ParallelTeacherSolver(_Agent())
    mean, log_std, totals, metrics = solver.solve(torch.empty(0, 4), [])
    assert mean.shape == log_std.shape == (0, 2)
    assert not totals.any() and not solver.runners and not fake_runtime.events
    assert metrics["distillation_reanalyze_chunks"] == 0


def test_cleanup_join_failure_preserves_original_error_and_joins_other_slots(monkeypatch, fake_runtime):
    solver = ParallelTeacherSolver(_Agent())
    created = _install_runners(monkeypatch, solver, fake_runtime)
    roots = torch.zeros(2, 4)
    solver.solve(roots, [1, 2])
    created[0].fail_seed = 3

    def failed_join():
        fake_runtime.events.append(("failed_sync", "slot0"))
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(created[0].stream, "synchronize", failed_join)
    with pytest.raises(RuntimeError, match="simulated graph replay failure"):
        solver.solve(roots, [3, 4])
    assert ("sync", "slot1") in fake_runtime.events
    assert not solver.runners and not solver._active


@pytest.mark.parametrize("bad", ["width", "shape", "dtype", "seed_count", "seed_bool", "seed_negative"])
def test_invalid_inputs_fail_before_pool_creation(fake_runtime, bad):
    agent = _Agent(width=True if bad == "width" else 2)
    solver = ParallelTeacherSolver(agent)
    roots, seeds = torch.zeros(2, 4), [1, 2]
    if bad == "shape":
        roots = torch.zeros(2, 3)
    elif bad == "dtype":
        roots = roots.double()
    elif bad == "seed_count":
        seeds = [1]
    elif bad == "seed_bool":
        seeds = [True, 2]
    elif bad == "seed_negative":
        seeds = [-1, 2]
    with pytest.raises(ValueError):
        solver.solve(roots, seeds)
    assert not solver.runners and not fake_runtime.events


def test_real_runtime_rejects_cpu_at_solve_not_construction():
    solver = ParallelTeacherSolver(_Agent())
    with pytest.raises(ValueError, match="CUDA"):
        solver.solve(torch.zeros(1, 4), [1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_pool_width_parity_private_state_and_producer_stream():
    from tests.test_ambi_root_local_sac import _tiny_model

    model = _tiny_model(
        device="cuda", inner_rounds=1, inner_updates_per_round=1,
        inner_replay_capacity=8, inner_temperature_mode="inherit_outer",
        inner_replay_scope="action", inner_critic_dropout_enabled=True,
        dropout=0.1,
    )
    try:
        agent = model.agent
        agent.model.train()
        before = _clone_tree(agent.training_state_dict())
        modes = [module.training for module in agent.modules()]
        cpu_rng = _globals()
        cuda_rng = torch.cuda.get_rng_state(agent.device).clone()
        producer = torch.cuda.Stream(device=agent.device)
        results = []
        for width in (1, 2):
            agent.cfg.reanalyze_parallel_roots = width
            solver = ParallelTeacherSolver(agent)
            with torch.cuda.stream(producer):
                roots = torch.arange(3 * agent.cfg.latent_dim, device=agent.device, dtype=torch.float32)
                roots = roots.reshape(3, agent.cfg.latent_dim) / 100
                result = solver.solve(roots, [31, 32, 33])
            assert all(runner.engine is not agent.inner_engine for runner in solver.runners)
            results.append(result[:3])
            assert producer.query()  # solve returns completed caller work.
            _assert_tree_equal(agent.training_state_dict(), before)
            assert [module.training for module in agent.modules()] == modes
            _assert_globals(cpu_rng)
            torch.testing.assert_close(torch.cuda.get_rng_state(agent.device), cuda_rng, rtol=0, atol=0)
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        model.env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_lazy_reanalysis_exact_resume_refreshes_changed_priors():
    from RL.tdmpc2_core.reanalyze import LazyReanalyzer
    from tests.test_ambi_reanalyze import _packet, _ReplayProbe
    from tests.test_ambi_root_local_sac import _tiny_model

    model = _tiny_model(
        device="cuda", batch_size=3, outer_actor_objective="distillation",
        reanalyze_backend="cuda_graph", reanalyze_parallel_roots=2,
        reanalyze_batch_size=2, inner_rounds=1, inner_updates_per_round=1,
        inner_replay_capacity=8, inner_temperature_mode="inherit_outer",
        inner_critic_dropout_enabled=True, dropout=0.1,
    )
    try:
        agent = model.agent
        source = LazyReanalyzer(agent)
        obs, teacher = _packet(agent.cfg.action_dim)
        obs = obs.to(agent.device)
        teacher = {name: value.to(agent.device) for name, value in teacher.items()}
        first, _ = source.refresh(_ReplayProbe(), obs, teacher, agent.outer_version)
        saved_seed_state = _clone_tree(source.state_dict())
        # Change the outer priors through a real learner update after capture.
        # The reused graph pool and a newly captured pool must both read this
        # current snapshot, not the weights from their original warmup.
        before_model = _clone_tree(agent.model.state_dict())
        h, b, a = agent.cfg.train_unroll_horizon, agent.cfg.batch_size, agent.cfg.action_dim
        agent._update(
            obs, torch.zeros(h, b, a, device=agent.device),
            torch.ones(h, b, 1, device=agent.device),
            torch.zeros(h, b, 1, device=agent.device), teacher=first,
        )
        assert any(not torch.equal(value, before_model[name]) for name, value in agent.model.state_dict().items())
        agent.model.train()
        live_snapshot = _clone_tree(agent.training_state_dict())
        modes = [module.training for module in agent.modules()]
        cpu_rng = _globals()
        cuda_rng = torch.cuda.get_rng_state(agent.device).clone()
        expected, expected_metrics = source.refresh(_ReplayProbe(), obs, first, agent.outer_version)
        expected_seed_state = _clone_tree(source.state_dict())
        _assert_tree_equal(agent.training_state_dict(), live_snapshot)
        _assert_globals(cpu_rng)
        torch.testing.assert_close(torch.cuda.get_rng_state(agent.device), cuda_rng, rtol=0, atol=0)
        assert [module.training for module in agent.modules()] == modes

        restored = LazyReanalyzer(agent)
        restored.load_state_dict(saved_seed_state)
        assert restored.parallel_solver is None
        actual, actual_metrics = restored.refresh(_ReplayProbe(), obs, first, agent.outer_version)
        _assert_tree_equal(actual, expected)
        _assert_tree_equal(restored.state_dict(), expected_seed_state)
        for name, value in expected_metrics.items():
            if "seconds" not in name:
                assert actual_metrics[name] == value
        _assert_tree_equal(agent.training_state_dict(), live_snapshot)
        _assert_globals(cpu_rng)
        torch.testing.assert_close(torch.cuda.get_rng_state(agent.device), cuda_rng, rtol=0, atol=0)
        assert [module.training for module in agent.modules()] == modes
    finally:
        model.env.close()
