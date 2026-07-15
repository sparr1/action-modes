from copy import deepcopy
import gc
import pickle
import weakref

import pytest
import torch

from RL.tdmpc2_core.common.compile_regions import CompileRegion
from RL.tdmpc2_core.common.layers import Ensemble, _DETACHED_PARAMETER_VIEWS


def test_compile_region_is_lazy_fixed_shape_and_cached(monkeypatch):
    compile_calls = []

    def fake_compile(function, **kwargs):
        compile_calls.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    region = CompileRegion("test region", lambda value: value.square(), enabled=True)

    torch.testing.assert_close(region(torch.tensor(3.0)), torch.tensor(9.0))
    torch.testing.assert_close(region(torch.tensor(4.0)), torch.tensor(16.0))

    assert len(compile_calls) == 1
    assert compile_calls[0][1] == {"fullgraph": False, "dynamic": False}
    assert not region.failed


def test_non_strict_compile_failure_warns_once_and_stays_eager(monkeypatch):
    compile_calls = 0

    def fake_compile(function, **kwargs):
        nonlocal compile_calls
        del function, kwargs
        compile_calls += 1

        def fail(*args, **kw):
            del args, kw
            raise RuntimeError("unsupported graph")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    region = CompileRegion("test region", lambda value: value + 1, enabled=True)

    with pytest.warns(RuntimeWarning, match="Falling back to eager test region"):
        assert region(2) == 3
    assert region(3) == 4
    assert compile_calls == 1
    assert region.failed
    assert not region.enabled


def test_non_strict_compile_retry_restores_rng_for_new_guard(monkeypatch):
    eager_draws = []

    def eager(value):
        draw = torch.rand(())
        eager_draws.append(draw)
        return value + draw

    def fake_compile(function, **kwargs):
        del function, kwargs

        def fail(value):
            del value
            torch.rand(())
            raise RuntimeError("failure after random work")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    region = CompileRegion("stochastic region", eager, enabled=True)
    torch.manual_seed(123)
    expected = torch.rand(())
    torch.manual_seed(123)

    with pytest.warns(RuntimeWarning, match="Falling back to eager stochastic region"):
        result = region(torch.tensor(2.0))

    torch.testing.assert_close(eager_draws[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(result, torch.tensor(2.0) + expected, rtol=0, atol=0)


def test_late_compile_region_failure_restores_rng_and_completes_eagerly(monkeypatch):
    eager_calls = 0
    compiled_calls = 0

    def eager(value):
        nonlocal eager_calls
        eager_calls += 1
        return value + torch.rand(())

    def fake_compile(function, **kwargs):
        del function, kwargs

        def compiled(value):
            nonlocal compiled_calls
            compiled_calls += 1
            if compiled_calls == 1:
                return value
            torch.rand(())
            raise RuntimeError("late backend failure")

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    region = CompileRegion("stochastic region", eager, enabled=True)

    assert region(torch.tensor(1.0)) == 1
    rng_before_failure = torch.random.get_rng_state().clone()
    expected_draw = torch.rand(())
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng_before_failure)
    with pytest.warns(RuntimeWarning, match="Falling back to eager stochastic region"):
        result = region(torch.tensor(2.0))

    torch.testing.assert_close(result, torch.tensor(2.0) + expected_draw, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert eager_calls == 1
    assert region(torch.tensor(3.0)) > 3
    assert eager_calls == 2


def test_compile_region_fallback_restores_explicit_generator(monkeypatch):
    def eager(value, *, generator):
        return value + torch.rand((), generator=generator)

    def fake_compile(function, **kwargs):
        del kwargs

        def fail(*args, **call_kwargs):
            function(*args, **call_kwargs)
            raise RuntimeError("explicit generator failure")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    region = CompileRegion("stochastic region", eager, enabled=True)
    generator = torch.Generator().manual_seed(8675309)
    initial_state = generator.get_state().clone()
    expected = eager(torch.tensor(2.0), generator=generator)
    expected_state = generator.get_state().clone()
    generator.set_state(initial_state)

    with pytest.warns(RuntimeWarning, match="Falling back to eager stochastic region"):
        actual = region(torch.tensor(2.0), generator=generator)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(generator.get_state(), expected_state, rtol=0, atol=0)


def test_strict_compile_failure_propagates_without_eager_retry(monkeypatch):
    eager_calls = 0

    def eager(value):
        nonlocal eager_calls
        eager_calls += 1
        return value

    def fake_compile(function, **kwargs):
        del function, kwargs

        def fail(*args, **kw):
            del args, kw
            raise RuntimeError("strict failure")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    region = CompileRegion("strict region", eager, enabled=True, strict=True)

    with pytest.raises(RuntimeError, match="strict failure"):
        region(1)
    assert eager_calls == 0
    assert not region.failed


def test_ensemble_compile_failure_is_sticky(monkeypatch):
    ensemble = Ensemble([torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)])

    def fake_compile(function, **kwargs):
        del function, kwargs

        def fail(*args, **kw):
            del args, kw
            raise RuntimeError("backend failure")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    ensemble.enable_compile(strict=False)
    with pytest.warns(RuntimeWarning, match="eager critic ensemble"):
        result = ensemble(torch.ones(1, 2))
    assert result.shape == (2, 1, 1)
    assert ensemble.compile_failed

    ensemble.enable_compile(strict=False)
    assert not ensemble._compile_enabled


def test_ensemble_compile_mode_change_rebuilds_wrappers(monkeypatch):
    compile_modes = []

    def fake_compile(function, **kwargs):
        compile_modes.append(kwargs["fullgraph"])
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    ensemble = Ensemble([torch.nn.Linear(2, 1)])
    value = torch.ones(1, 2)

    ensemble.enable_compile(strict=False)
    ensemble(value)
    ensemble.enable_compile(strict=True)
    ensemble(value)

    assert compile_modes == [False, True]


def test_strict_mode_change_retries_after_non_strict_ensemble_failure(monkeypatch):
    compile_modes = []

    def fake_compile(function, **kwargs):
        compile_modes.append(kwargs["fullgraph"])
        if not kwargs["fullgraph"]:
            def fail(*args, **call_kwargs):
                del args, call_kwargs
                raise RuntimeError("non-strict backend failure")

            return fail
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    ensemble = Ensemble([torch.nn.Linear(2, 1)])
    value = torch.ones(1, 2)

    ensemble.enable_compile(strict=False)
    with pytest.warns(RuntimeWarning, match="non-strict backend failure"):
        ensemble(value)
    assert ensemble.compile_failed

    ensemble.enable_compile(strict=True)
    result = ensemble(value)

    assert result.shape == (1, 1, 1)
    assert compile_modes == [False, True]
    assert not ensemble.compile_failed


@pytest.mark.parametrize("detached", [False, True])
def test_compiled_ensemble_is_garbage_collectable(monkeypatch, detached):
    monkeypatch.setattr(torch, "compile", lambda function, **_kwargs: function)
    ensemble = Ensemble([torch.nn.Linear(2, 1)])
    ensemble.enable_compile(strict=False)
    call = ensemble.forward_detached if detached else ensemble.forward
    call(torch.ones(1, 2))
    reference = weakref.ref(ensemble)

    del call
    del ensemble
    gc.collect()

    assert reference() is None


@pytest.mark.parametrize("detached", [False, True])
def test_pickle_resets_process_local_compiled_wrapper(monkeypatch, detached):
    compile_calls = 0

    def fake_compile(function, **kwargs):
        nonlocal compile_calls
        del kwargs
        compile_calls += 1
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    ensemble = Ensemble([torch.nn.Linear(2, 1)])
    ensemble.enable_compile(strict=False)
    call = ensemble.forward_detached if detached else ensemble.forward
    call(torch.ones(1, 2))

    restored = pickle.loads(pickle.dumps(ensemble))
    attribute = "_compiled_detached_forward" if detached else "_compiled_forward"
    assert getattr(restored, attribute) is None
    restored_call = restored.forward_detached if detached else restored.forward
    assert restored_call(torch.ones(1, 2)).shape == (1, 1, 1)
    assert compile_calls == 2


def test_ensemble_unpickles_legacy_state_without_compile_runtime_fields():
    ensemble = Ensemble([torch.nn.Linear(2, 1)])
    state = ensemble.__getstate__()
    for name in (
        "_compile_enabled",
        "_compile_strict",
        "_compile_failed",
        "_detached_compile_failed",
        "_compiled_forward",
        "_compiled_detached_forward",
    ):
        state.pop(name, None)

    restored = Ensemble.__new__(Ensemble)
    restored.__setstate__(state)

    assert restored(torch.ones(1, 2)).shape == (1, 1, 1)
    assert not restored._compile_enabled
    assert not restored.compile_failed


@pytest.mark.parametrize("detached", [False, True])
def test_ensemble_synchronous_compile_failure_falls_back(monkeypatch, detached):
    ensemble = Ensemble([torch.nn.Linear(2, 1)])

    def fail_to_compile(function, **kwargs):
        del function, kwargs
        raise RuntimeError("construction failure")

    monkeypatch.setattr(torch, "compile", fail_to_compile)
    ensemble.enable_compile(strict=False)
    call = ensemble.forward_detached if detached else ensemble.forward
    with pytest.warns(RuntimeWarning, match="construction failure"):
        result = call(torch.ones(1, 2))

    assert result.shape == (1, 1, 1)
    assert ensemble.compile_failed
    assert not ensemble._compile_enabled


@pytest.mark.parametrize("detached", [False, True])
def test_first_compiled_failure_restores_rng_before_eager_retry(monkeypatch, detached):
    class RandomModule(torch.nn.Module):
        def forward(self, value):
            return value + torch.rand((), device=value.device)

    ensemble = Ensemble([RandomModule()])

    def compile_then_fail(function, **kwargs):
        del kwargs

        def fail(*args, **call_kwargs):
            function(*args, **call_kwargs)
            raise RuntimeError("post-random failure")

        return fail

    monkeypatch.setattr(torch, "compile", compile_then_fail)
    ensemble.enable_compile(strict=False)
    call = ensemble.forward_detached if detached else ensemble.forward
    value = torch.zeros(1)
    initial_rng = torch.random.get_rng_state().clone()
    expected = ensemble._forward_detached_eager(value) if detached else ensemble._forward_eager(value)
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(initial_rng)

    with pytest.warns(RuntimeWarning, match="post-random failure"):
        actual = call(value)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)


@pytest.mark.parametrize("detached", [False, True])
def test_late_ensemble_failure_restores_rng_and_completes_eagerly(monkeypatch, detached):
    class RandomModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return value + torch.rand((), device=value.device)

    module = RandomModule()
    ensemble = Ensemble([module])
    compiled_calls = 0

    def compile_then_fail_second(function, **kwargs):
        del kwargs

        def compiled(*args, **call_kwargs):
            nonlocal compiled_calls
            compiled_calls += 1
            result = function(*args, **call_kwargs)
            if compiled_calls == 2:
                raise RuntimeError("late stochastic failure")
            return result

        return compiled

    monkeypatch.setattr(torch, "compile", compile_then_fail_second)
    ensemble.enable_compile(strict=False)
    call = ensemble.forward_detached if detached else ensemble.forward
    value = torch.zeros(1)
    call(value)
    rng_before_failure = torch.random.get_rng_state().clone()
    expected_draw = torch.rand(())
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng_before_failure)

    with pytest.warns(RuntimeWarning, match="late stochastic failure"):
        result = call(value)

    torch.testing.assert_close(
        result,
        (value + expected_draw).unsqueeze(0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert module.calls == 3

    call(value)
    assert module.calls == 4
    assert compiled_calls == 2


@pytest.mark.parametrize("detached", [False, True])
def test_ensemble_fallback_restores_explicit_generator_argument(monkeypatch, detached):
    class GeneratorModule(torch.nn.Module):
        def forward(self, value, *, generator):
            return value + torch.rand((), generator=generator, device=value.device)

    ensemble = Ensemble([GeneratorModule()])

    def compile_then_fail(function, **kwargs):
        del kwargs

        def fail(*args, **call_kwargs):
            function(*args, **call_kwargs)
            raise RuntimeError("explicit generator failure")

        return fail

    monkeypatch.setattr(torch, "compile", compile_then_fail)
    ensemble.enable_compile(strict=False)
    call = ensemble.forward_detached if detached else ensemble.forward
    value = torch.zeros(1)
    generator = torch.Generator().manual_seed(8675309)
    initial_state = generator.get_state().clone()
    expected = ensemble._forward_detached_eager(
        value, generator=generator
    ) if detached else ensemble._forward_eager(value, generator=generator)
    expected_state = generator.get_state().clone()
    generator.set_state(initial_state)

    with pytest.warns(RuntimeWarning, match="explicit generator failure"):
        actual = call(value, generator=generator)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(generator.get_state(), expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("detached", [False, True])
def test_deepcopied_ensemble_first_call_can_fall_back_rng_safely(monkeypatch, detached):
    class RandomModule(torch.nn.Module):
        def forward(self, value):
            return value + torch.rand((), device=value.device)

    compile_calls = 0

    def compile_original_but_fail_clone(function, **kwargs):
        nonlocal compile_calls
        del kwargs
        compile_calls += 1
        if compile_calls == 1:
            return function

        def fail(*args, **call_kwargs):
            function(*args, **call_kwargs)
            raise RuntimeError("deepcopy guard failure")

        return fail

    monkeypatch.setattr(torch, "compile", compile_original_but_fail_clone)
    ensemble = Ensemble([RandomModule()])
    ensemble.enable_compile(strict=False)
    call = ensemble.forward_detached if detached else ensemble.forward
    value = torch.zeros(1)
    call(value)

    clone = deepcopy(ensemble)
    attribute = "_compiled_detached_forward" if detached else "_compiled_forward"
    assert getattr(clone, attribute) is None
    clone_call = clone.forward_detached if detached else clone.forward
    initial_rng = torch.random.get_rng_state().clone()
    expected = clone._forward_detached_eager(value) if detached else clone._forward_eager(value)
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(initial_rng)

    with pytest.warns(RuntimeWarning, match="deepcopy guard failure"):
        actual = clone_call(value)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert compile_calls == 2
    assert clone.compile_failed


def test_ensemble_apply_invalidates_cached_detached_parameter_views():
    module = torch.nn.Linear(2, 1)
    ensemble = Ensemble([module])
    ensemble.forward_detached(torch.ones(1, 2))
    assert module in _DETACHED_PARAMETER_VIEWS
    assert _DETACHED_PARAMETER_VIEWS[module]["weight"].dtype == torch.float32

    ensemble.to(dtype=torch.float64)
    assert module not in _DETACHED_PARAMETER_VIEWS
    result = ensemble.forward_detached(torch.ones(1, 2, dtype=torch.float64))

    assert result.dtype == torch.float64
    assert _DETACHED_PARAMETER_VIEWS[module]["weight"].dtype == torch.float64
