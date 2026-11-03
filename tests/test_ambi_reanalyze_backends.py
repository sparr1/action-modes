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
