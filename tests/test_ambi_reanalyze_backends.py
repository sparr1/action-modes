"""Backend identity, transactional resume, and refresh integration contracts."""

import pytest
import torch

from RL.tdmpc2_core.reanalyze import LazyReanalyzer
from tests.test_ambi_distillation_config import _identity
from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_reanalyze import _packet, _ReplayProbe, models  # noqa: F401


def test_native_defaults_preserve_scientific_identity_and_checkpoint(models):
    cfg = _build_cfg()
    assert cfg.reanalyze_backend == "native" and cfg.reanalyze_parallel_roots == 4
    assert cfg.reanalyze_batched_optimizations is True
    assert cfg.reanalyze_batched_workers == 1
    assert _identity({}) == _identity({
        "reanalyze_backend": "cuda_graph", "reanalyze_parallel_roots": 8,
    })  # Both are dormant for the default Q objective.
    base = {"outer_actor_objective": "distillation"}
    assert _identity(base) == _identity({**base, "reanalyze_backend": "NATIVE", "reanalyze_parallel_roots": 8})
    graph = {**base, "reanalyze_backend": "cuda_graph"}
    assert _identity(base) != _identity(graph)
    assert _identity(graph) == _identity({**graph, "reanalyze_parallel_roots": 4})
    assert _identity(graph) != _identity({**graph, "reanalyze_parallel_roots": 8})
    saved = LazyReanalyzer(models().agent).state_dict()
    assert saved.keys() == {"schema", "version", "rng_state", "refreshes", "roots"}
    assert saved["version"] == 1


@pytest.mark.parametrize("key,value", [
    ("reanalyze_backend", "threads"), ("reanalyze_backend", None),
    ("reanalyze_parallel_roots", 0), ("reanalyze_parallel_roots", True),
    ("reanalyze_parallel_roots", 1.5),
    ("reanalyze_batched_optimizations", 1), ("reanalyze_batched_optimizations", "true"),
    ("reanalyze_batched_optimizations", None),
    ("reanalyze_batched_workers", 0), ("reanalyze_batched_workers", True),
    ("reanalyze_batched_workers", 1.5),
])
def test_invalid_backend_controls(key, value):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key: value})


def test_graph_restrictions_do_not_leak_into_native_or_q_objectives():
    common = {"inner_outer_replay_fraction": 0.5}
    assert _build_cfg(outer_actor_objective="distillation", **common).inner_outer_replay_fraction == 0.5
    assert _build_cfg(reanalyze_backend="cuda_graph", **common).outer_actor_objective == "q_maximization"
    with pytest.raises(ValueError, match="inner_outer_replay_fraction"):
        _build_cfg(outer_actor_objective="distillation", reanalyze_backend="cuda_graph", **common)


@pytest.mark.parametrize("backend", ["native", "cuda_graph", "batched", "batched_cuda_graph"])
def test_batched_controls_are_dormant_for_q_maximization(backend):
    explicit = dict(reanalyze_backend=backend, reanalyze_batched_workers=3,
                    reanalyze_batched_optimizations=False)
    cfg = _build_cfg(**explicit)
    assert cfg.outer_actor_objective == "q_maximization"
    assert _identity(explicit) == _identity({})


@pytest.mark.parametrize("backend", ["native", "cuda_graph"])
def test_batched_controls_leave_other_distillation_backends_unchanged(backend, models):
    base = dict(outer_actor_objective="distillation", reanalyze_backend=backend)
    explicit = dict(**base, reanalyze_batched_workers=3,
                    reanalyze_batched_optimizations=False)
    assert _build_cfg(**explicit).reanalyze_batched_workers == 3
    assert _identity(base) == _identity(explicit)
    model = models()
    reference = LazyReanalyzer(model.agent)
    reference.backend = backend  # Validate serialization of CUDA mode on CPU.
    original = reference.state_dict()
    model.cfg.reanalyze_batched_workers = 3
    model.cfg.reanalyze_batched_optimizations = False
    _assert_tree_equal(reference.state_dict(), original)


def test_concurrent_batched_workers_require_cuda_graph_only_when_active():
    options = dict(outer_actor_objective="distillation", reanalyze_batched_workers=2)
    with pytest.raises(ValueError, match="reanalyze_batched_workers.*batched_cuda_graph"):
        _build_cfg(reanalyze_backend="batched", **options)
    assert _build_cfg(reanalyze_backend="batched_cuda_graph", **options).reanalyze_batched_workers == 2


def test_batched_execution_identity_and_reference_projection():
    base = dict(outer_actor_objective="distillation", reanalyze_backend="batched")
    enabled = _identity(base)
    assert enabled == _identity(dict(**base, reanalyze_batched_optimizations=True,
                                    reanalyze_batched_workers=1))
    reference = _identity(dict(**base, reanalyze_batched_optimizations=False,
                              reanalyze_batched_workers=1))
    assert reference != enabled
    assert "reanalyze_batched_optimizations" not in reference["alg_params"]
    assert "reanalyze_batched_workers" not in reference["alg_params"]
    cuda = dict(base, reanalyze_backend="batched_cuda_graph")
    assert _identity(cuda) != _identity(dict(cuda, reanalyze_batched_workers=2))


@pytest.mark.parametrize("optimized,version", [(False, 3), (True, 4)])
def test_batched_checkpoint_execution_specification_and_roundtrip(models, optimized, version):
    options = dict(outer_actor_objective="distillation", reanalyze_backend="batched",
                   reanalyze_batched_optimizations=optimized)
    first = models(**options).agent.distillation.reanalyzer
    first.refreshes, first.roots = 3, 13
    torch.randint(0, 100, (7,), generator=first.generator)
    saved = _clone_tree(first.state_dict())
    assert saved["version"] == version
    if optimized:
        assert saved["reanalyze_execution"] == "root_batched_optimized_v3"
        assert saved["reanalyze_batched_optimizations"] is True
        assert saved["reanalyze_batched_workers"] == 1
    else:
        assert set(saved) == {
            "schema", "version", "rng_state", "refreshes", "roots", "reanalyze_backend",
            "reanalyze_rng_protocol", "reanalyze_parallel_roots", "reanalyze_optimizer",
        }
    restored = models(**options).agent.distillation.reanalyzer
    restored.load_state_dict(saved)
    _assert_tree_equal(restored.state_dict(), saved)
    for key, bad_value in (("reanalyze_batched_optimizations", not optimized),
                           ("reanalyze_batched_workers", 2),
                           ("reanalyze_execution", "unknown_execution"),
                           ("reanalyze_execution", "root_batched_optimized_v2")):
        incompatible = _clone_tree(saved)
        incompatible[key] = bad_value
        before = _clone_tree(restored.state_dict())
        with pytest.raises(ValueError):
            restored.load_state_dict(incompatible)
        _assert_tree_equal(restored.state_dict(), before)


@pytest.mark.parametrize("source_optimized", [False, True])
def test_changed_batched_execution_rejects_full_resume_before_mutation(models, source_optimized):
    options = dict(outer_actor_objective="distillation", reanalyze_backend="batched")
    source = models(**options, reanalyze_batched_optimizations=source_optimized).agent
    saved = _clone_tree(source.training_state_dict())
    target = models(**options, reanalyze_batched_optimizations=not source_optimized).agent
    before = _clone_tree(target.training_state_dict())
    with pytest.raises(ValueError, match="distillation specification"):
        target.load_training_state_dict(saved)
    _assert_tree_equal(target.training_state_dict(), before)


@pytest.mark.parametrize("key,value", [
    ("reanalyze_batched_optimizations", 1),
    ("reanalyze_batched_workers", True),
    ("reanalyze_batched_workers", 1.0),
])
def test_batched_checkpoint_rejects_aliased_control_types(models, key, value):
    reanalyzer = models(outer_actor_objective="distillation", reanalyze_backend="batched").agent.distillation.reanalyzer
    before = _clone_tree(reanalyzer.state_dict())
    saved = _clone_tree(before)
    saved[key] = value
    with pytest.raises(ValueError):
        reanalyzer.load_state_dict(saved)
    _assert_tree_equal(reanalyzer.state_dict(), before)


def test_concurrent_batched_refresh_dispatch_preserves_seed_order(models, monkeypatch):
    import sys
    from types import ModuleType

    calls = []

    class FakeConcurrent:
        def __init__(self, agent):
            assert agent.cfg.reanalyze_batched_workers == 2

        def solve(self, roots, seeds):
            calls.append((roots.clone(), list(seeds)))
            return roots[:, :1], roots[:, :1].new_zeros((len(roots), 1)), roots.new_tensor([9, 3, 6]), {
                "distillation_reanalyze_batched_workers": 2.0,
            }

    module = ModuleType("RL.tdmpc2_core.batched_teacher_pool")
    module.ConcurrentBatchedTeacherSolver = FakeConcurrent
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = models(outer_actor_objective="distillation", reanalyze_backend="batched")
    reanalyzer = model.agent.distillation.reanalyzer
    reanalyzer.backend = "batched_cuda_graph"  # Fake worker exercises CUDA dispatch on CPU.
    model.cfg.reanalyze_batched_workers = 2
    saved = _clone_tree(reanalyzer.state_dict())
    assert saved["version"] == 4 and saved["reanalyze_batched_workers"] == 2
    obs, teacher = _packet()
    first, metrics = reanalyzer.refresh(_ReplayProbe(), obs, teacher, 10)
    reanalyzer.load_state_dict(saved)
    second, _ = reanalyzer.refresh(_ReplayProbe(), obs, teacher, 10)
    _assert_tree_equal(first, second)
    assert calls[0][1] == calls[1][1]
    assert metrics["distillation_reanalyze_roots"] == 3
    assert metrics["distillation_reanalyze_batched_workers"] == 2


def test_graph_backend_resume_rejects_native_and_changed_protocol_before_mutation(models):
    model = models()
    native = LazyReanalyzer(model.agent)
    native_saved = _clone_tree(native.state_dict())
    model.cfg.reanalyze_backend = "cuda_graph"
    with pytest.raises(ValueError, match="CUDA agent"):
        LazyReanalyzer(model.agent)
    model.cfg.reanalyze_backend = "native"
    graph = LazyReanalyzer(model.agent)
    graph.backend = "cuda_graph"  # Validate its serialized contract without a CUDA device.
    graph_saved = _clone_tree(graph.state_dict())
    assert graph_saved["version"] == 2
    for target, invalid, original in ((graph, native_saved, graph_saved),
                                      (native, graph_saved, native_saved)):
        with pytest.raises(ValueError):
            target.load_state_dict(invalid)
        _assert_tree_equal(target.state_dict(), original)
    invalid = _clone_tree(graph_saved)
    invalid["reanalyze_rng_protocol"] = "other"
    with pytest.raises(ValueError, match="protocol"):
        graph.load_state_dict(invalid)
    _assert_tree_equal(graph.state_dict(), graph_saved)


def test_parallel_refresh_uses_deduplicated_private_seeds_and_replays_after_resume(models, monkeypatch):
    import RL.tdmpc2_core.parallel_teacher as parallel

    calls = []

    class FakeParallel:
        def __init__(self, agent):
            self.agent = agent

        def solve(self, roots, seeds):
            calls.append((roots.clone(), list(seeds)))
            means = roots[:, :self.agent.cfg.action_dim] + torch.tensor(seeds)[:, None] / 2**31
            return means, torch.zeros_like(means), roots.new_tensor([12, 3, 3]), {
                "distillation_reanalyze_cuda_graph": 1.0,
                "distillation_reanalyze_parallel_roots": 2.0,
            }

    monkeypatch.setattr(parallel, "ParallelTeacherSolver", FakeParallel)
    model = models()
    reanalyzer = LazyReanalyzer(model.agent)
    reanalyzer.backend = "cuda_graph"  # Fake runner below exercises dispatch on CPU.
    saved = _clone_tree(reanalyzer.state_dict())
    seed_stream = torch.Generator().set_state(saved["rng_state"])
    torch.randint(0, 2**31 - 1, (), generator=seed_stream)
    expected = [int(torch.randint(0, 2**31 - 1, (), generator=seed_stream)) for _ in range(3)]
    obs, teacher = _packet(model.cfg.action_dim)
    first, metrics = reanalyzer.refresh(_ReplayProbe(), obs, teacher, 4)
    assert calls[0][1] == expected
    assert metrics["distillation_reanalyze_model_steps"] == 12
    assert metrics["distillation_reanalyze_cuda_graph"] == 1
    torch.rand(123)
    reanalyzer.load_state_dict(saved)
    second, _ = reanalyzer.refresh(_ReplayProbe(), obs, teacher, 4)
    _assert_tree_equal(first, second)
    assert calls[1][1] == expected


def test_native_refresh_does_not_call_execution_or_metric_aggregation(models, monkeypatch):
    reanalyzer = LazyReanalyzer(models().agent)
    def unused(*args, **kwargs):
        pytest.fail("Teacher refresh invoked unused action/reporting work")
    monkeypatch.setattr(reanalyzer.engine, "_execute_policy", unused)
    monkeypatch.setattr(reanalyzer.engine, "_average_update_metrics", unused)
    obs, teacher = _packet(reanalyzer.cfg.action_dim)
    reanalyzer.refresh(_ReplayProbe(), obs, teacher, 4)
