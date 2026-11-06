"""Configuration and strict lineage for BOOM actor compilation."""

import pytest

from tests.test_ambi_boom_config import _boom_cfg, _identity
from tests.test_ambi_config_decoupling import _build_cfg


def test_boom_compilation_preserves_default_and_requires_existing_compile_switch():
    assert _boom_cfg().outer_boom_compile_mode == "default"
    cfg = _boom_cfg(compile=False, outer_boom_compile_mode="REDUCE-OVERHEAD")
    assert cfg.outer_boom_compile_mode == "reduce-overhead"
    assert cfg.compile is False
    assert cfg.store_behavior_policy and not cfg.store_teacher_policy
    cfg = _boom_cfg(compile=True, outer_boom_compile_mode="reduce-overhead")
    assert cfg.compile is True
    assert cfg.outer_boom_compile_mode == "reduce-overhead"
    assert cfg.device == "cpu"


@pytest.mark.parametrize("value", [None, True, 0, "cuda_graph", "max-autotune", ""])
def test_boom_compile_mode_rejects_unsupported_values(value):
    with pytest.raises(ValueError, match="outer_boom_compile_mode"):
        _boom_cfg(outer_boom_compile_mode=value)


def test_boom_compile_mode_does_not_activate_boom():
    cfg = _build_cfg(outer_boom_compile_mode="reduce-overhead")
    assert cfg.outer_actor_objective == "q_maximization"
    assert not cfg.store_behavior_policy


@pytest.mark.parametrize("objective,compiled", [
    ("q_maximization", False), ("q_maximization", True),
    ("distillation", False), ("distillation", True), ("boom", False),
])
def test_inactive_boom_compile_mode_preserves_lineage(objective, compiled):
    base = {"outer_actor_objective": objective, "compile": compiled}
    assert _identity(base) == _identity({
        **base, "outer_boom_compile_mode": "reduce-overhead",
    })


def test_active_boom_compile_mode_pins_execution_but_default_preserves_lineage():
    base = {"outer_actor_objective": "boom", "compile": True}
    assert _identity(base) == _identity({
        **base, "outer_boom_compile_mode": "default",
    })
    accelerated = {**base, "outer_boom_compile_mode": "reduce-overhead"}
    assert _identity(base) != _identity(accelerated)
    assert _identity(accelerated) == _identity({
        **base, "outer_boom_compile_mode": "REDUCE-OVERHEAD",
    })
