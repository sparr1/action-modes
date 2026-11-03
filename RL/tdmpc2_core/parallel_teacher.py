"""Bounded independent CUDA streams for fresh graph-backed teacher solves.

Every slot owns its actor, critic, Adam state, latent replay, and CUDA graph.
Only frozen outer model evaluations are shared. Slot reuse never changes the
root seed or returns a view that a later solve can overwrite.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from numbers import Real
import time

import torch

from .common.compile_regions import _preserve_host_rng_state
from .inner_improvement import InnerImprovementEngine


_COUNTERS = (
    "inner_total_model_steps",
    "inner_actor_optimizer_steps",
    "inner_critic_optimizer_steps",
)


def _require_cuda(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Parallel graph teachers require an available CUDA device.")


@contextmanager
def _preserve_rng(device):
    devices = []
    if device.type == "cuda":
        devices = [torch.cuda.current_device() if device.index is None else device.index]
    with _preserve_host_rng_state(), torch.random.fork_rng(devices=devices):
        yield


def _record_stream(tensor, stream):
    if tensor.device.type == "cuda":
        tensor.record_stream(stream)


class ParallelTeacherSolver:
    """A lazy, bounded slot pool with one final synchronization per sweep.

    The pool is runtime-only: every runner starts each root from current outer
    priors with the supplied seed. Exact continuation therefore needs the
    caller's root-seed stream, not captured graph or optimizer allocations.
    """

    def __init__(self, agent):
        self.agent = agent
        self.cfg = agent.cfg
        self.device = torch.device(agent.device)
        self.parallel_roots = getattr(self.cfg, "reanalyze_parallel_roots", 4)
        self.runners = []
        self.setup_seconds_total = 0.0
        self.warmup_seconds_total = 0.0
        self.capture_seconds_total = 0.0
        self._active = False

    def _make_runner(self, root):
        from .teacher_graph import TeacherGraph

        # A separate engine is essential: graph replay mutates its private
        # actor/critic/optimizer/replay tensors, never the live controller.
        engine = InnerImprovementEngine(self.agent)
        return TeacherGraph(engine, root, capture=True)

    def _validate_inputs(self, roots, seeds):
        _require_cuda(self.device)
        width = self.parallel_roots
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("reanalyze_parallel_roots must be a positive integer.")
        if (
            not torch.is_tensor(roots)
            or roots.ndim != 2
            or roots.shape[1] != int(self.cfg.latent_dim)
            or roots.dtype != torch.float32
        ):
            raise ValueError("Teacher roots must be a float32 tensor with shape [N, latent_dim].")
        expected_index = self.device.index
        if self.device.type == "cuda" and expected_index is None:
            expected_index = torch.cuda.current_device()
        if roots.device.type != self.device.type or roots.device.index != expected_index:
            raise ValueError("Teacher roots must be on the agent's device.")
        if not isinstance(seeds, (list, tuple)) or len(seeds) != len(roots):
            raise ValueError("Teacher seeds must contain one integer for each root.")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
            for seed in seeds
        ):
            raise ValueError("Teacher root seeds must be non-negative int64 integers.")

    def _ensure_pool(self, roots, width):
        setup_seconds = warmup_seconds = capture_seconds = 0.0
        # Complete all warmup/capture before launching any root. Capturing a
        # new slot while another slot has in-flight work is not safe.
        for slot in range(len(self.runners), width):
            started = time.perf_counter()
            runner = self._make_runner(roots[slot:slot + 1])
            setup_seconds += time.perf_counter() - started
            warmup_seconds += float(getattr(runner, "warmup_seconds", 0.0))
            capture_seconds += float(getattr(runner, "capture_seconds", 0.0))
            self.runners.append(runner)
        self.setup_seconds_total += setup_seconds
        self.warmup_seconds_total += warmup_seconds
        self.capture_seconds_total += capture_seconds
        return setup_seconds, warmup_seconds, capture_seconds

    def _copy_outputs(self, runner, teacher, counters):
        action_dim = int(self.cfg.action_dim)
        with torch.cuda.stream(runner.stream):
            values = []
            for name in ("pre_tanh_mean", "log_std"):
                value = teacher[name]
                if not torch.is_tensor(value) or value.numel() != action_dim:
                    raise ValueError(f"Graph teacher {name} must contain action_dim values.")
                values.append(value.detach().reshape(action_dim).clone())
            if isinstance(counters, Mapping):
                # The graph runner's work counts are fixed Python numbers.
                # Avoid three synchronizing scalar CPU->GPU copies per root
                # in the asynchronous launch loop.
                if any(not isinstance(counters[name], Real) for name in _COUNTERS):
                    raise TypeError("Graph counter mappings must contain Python numbers; use a tensor[3] for device counters.")
                counter = tuple(float(counters[name]) for name in _COUNTERS)
            elif torch.is_tensor(counters) and counters.shape == (3,):
                counter = counters.detach().to(device=values[0].device, dtype=torch.float32).clone()
            else:
                raise ValueError("Graph teacher counters must be a mapping or a tensor of shape [3].")
        return values[0], values[1], counter

    def solve(self, roots, seeds):
        """Return completed teachers in input order, independent of pool width.

        TeacherGraph establishes the producer-stream dependency for its input
        root and enqueues work on its own stream. Copies occur on that same
        stream before slot reuse. The calling stream joins all slots before
        stacking results; a single final sync also makes exception cleanup and
        subsequent replay writeback safe.
        """
        self._validate_inputs(roots, seeds)
        if self._active:
            raise RuntimeError("Parallel teacher solves are not reentrant.")
        width = min(int(self.parallel_roots), len(roots))
        metrics = {
            "distillation_reanalyze_parallel_roots": float(width),
            "distillation_reanalyze_chunks": float((len(roots) + width - 1) // width) if width else 0.0,
            "distillation_reanalyze_cuda_graph": 1.0,
            "distillation_reanalyze_setup_seconds": 0.0,
            "distillation_reanalyze_warmup_seconds": 0.0,
            "distillation_reanalyze_capture_seconds": 0.0,
        }
        if not width:
            empty = roots.new_empty((0, int(self.cfg.action_dim)))
            return empty, empty.clone(), roots.new_zeros(3), metrics
        modes = tuple((module, bool(module.training)) for module in self.agent.modules())
        caller = torch.cuda.current_stream(self.device)
        self._active = True
        copies = []
        try:
            with _preserve_rng(self.device):
                self.agent.model.eval()
                setup, warmup, capture = self._ensure_pool(roots, width)
                metrics.update({
                    "distillation_reanalyze_setup_seconds": setup,
                    "distillation_reanalyze_warmup_seconds": warmup,
                    "distillation_reanalyze_capture_seconds": capture,
                })
                for start in range(0, len(roots), width):
                    for slot, index in enumerate(range(start, min(start + width, len(roots)))):
                        runner = self.runners[slot]
                        root = roots[index:index + 1]
                        _record_stream(root, runner.stream)
                        teacher, counters = runner(root, seed=seeds[index])
                        copies.append(self._copy_outputs(runner, teacher, counters))
                for runner in self.runners[:width]:
                    caller.wait_stream(runner.stream)
                # These copies were allocated on side streams and are read on
                # the caller stream. Record that use before references expire.
                host_totals = [0.0, 0.0, 0.0]
                tensor_counters = []
                for group in copies:
                    for tensor in group[:2]:
                        _record_stream(tensor, caller)
                    counter = group[2]
                    if torch.is_tensor(counter):
                        _record_stream(counter, caller)
                        tensor_counters.append(counter)
                    else:
                        for index, value in enumerate(counter):
                            host_totals[index] += value
                means = torch.stack([group[0] for group in copies])
                log_stds = torch.stack([group[1] for group in copies])
                totals = roots.new_tensor(host_totals)
                if tensor_counters:
                    totals = totals + torch.stack(tensor_counters).sum(0)
                caller.synchronize()
                return means, log_stds, totals, metrics
        except BaseException as error:
            # Outstanding graph work must finish before slots can be discarded
            # or the caller restores/mutates its shared frozen model snapshot.
            for runner in self.runners:
                try:
                    runner.stream.synchronize()
                except BaseException as cleanup_error:
                    if hasattr(error, "add_note"):
                        error.add_note(f"Joining a failed teacher slot also failed: {cleanup_error}")
            self.runners.clear()
            raise
        finally:
            for module, was_training in modes:
                module.training = was_training
            self._active = False


__all__ = ["ParallelTeacherSolver"]
