"""Small, state-free wrappers for optional fixed-shape compilation.

The wrapped callables must be free of externally visible mutation. Tensor
regions may still consume the default RNG (for example through dropout), so
non-strict compiled calls retain enough RNG state to retry eagerly exactly once
after any lazy compiler/backend failure.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
import torch


def _rng_dependencies(value, devices, generators):
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            devices.add(
                value.device.index
                if value.device.index is not None
                else torch.cuda.current_device()
            )
        return
    if isinstance(value, torch.Generator):
        generators[id(value)] = (value, value.get_state())
        device = torch.device(value.device)
        if device.type == "cuda":
            devices.add(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _rng_dependencies(item, devices, generators)
    elif isinstance(value, dict):
        for item in value.values():
            _rng_dependencies(item, devices, generators)


def _capture_rng_state(args, kwargs, *, include_host_globals=False):
    devices = set()
    generators = {}
    _rng_dependencies(args, devices, generators)
    _rng_dependencies(kwargs, devices, generators)
    return (
        random.getstate() if include_host_globals else None,
        np.random.get_state() if include_host_globals else None,
        torch.random.get_rng_state(),
        {device: torch.cuda.get_rng_state(device) for device in sorted(devices)},
        generators,
    )


def _restore_rng_state(state):
    python_state, numpy_state, cpu_state, cuda_states, generators = state
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch.random.set_rng_state(cpu_state)
    for device, cuda_state in cuda_states.items():
        torch.cuda.set_rng_state(cuda_state, device)
    for generator, generator_state in generators.values():
        generator.set_state(generator_state)


class CompileRegion:
    """Lazily compile one tensor region with a warning/eager fallback."""

    def __init__(self, name, eager, *, enabled=False, strict=False):
        self.name = str(name)
        self.eager = eager
        self.strict = bool(strict)
        self.enabled = bool(enabled)
        self.failed = False
        self._compiled = None
        self._warned = False

        if self.enabled and not hasattr(torch, "compile"):
            error = RuntimeError("torch.compile is unavailable in this PyTorch build.")
            if self.strict:
                raise error
            self._fallback(error)

    def _fallback(self, error):
        self.failed = True
        self.enabled = False
        self._compiled = None
        if not self._warned:
            warnings.warn(
                f"Falling back to eager {self.name} after compile failure: {error}",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned = True

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            return self.eager(*args, **kwargs)
        constructing = self._compiled is None
        rng_state = (
            _capture_rng_state(
                args,
                kwargs,
                include_host_globals=constructing,
            )
            if not self.strict
            else None
        )
        if self._compiled is None:
            construction_rng_state = (
                rng_state
                if rng_state is not None
                else _capture_rng_state(
                    args, kwargs, include_host_globals=True
                )
            )
            try:
                compiled = torch.compile(
                    self.eager,
                    fullgraph=self.strict,
                    dynamic=False,
                )
            except Exception as error:
                _restore_rng_state(construction_rng_state)
                if self.strict:
                    raise
                self._fallback(error)
                return self.eager(*args, **kwargs)
            # Compiler/backend construction is operational work. It may happen
            # at a different point after exact resume, so it must not advance
            # any scientific RNG before the compiled callable actually runs.
            _restore_rng_state(construction_rng_state)
            self._compiled = compiled
        try:
            return self._compiled(*args, **kwargs)
        except Exception as error:
            if self.strict:
                raise
            _restore_rng_state(rng_state)
            self._fallback(error)
            return self.eager(*args, **kwargs)
