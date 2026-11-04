"""Concurrent CUDA graphs for independent, fixed-width batches of SAC roots.

Chunk boundaries and seeds match the single-worker batched solver. Workers own
separate graph pools, parameters, optimizer moments and replay storage. Only the
frozen world model and outer priors are shared, and all workers finish before
the caller can update those priors or write teachers back to real replay.
"""

from contextlib import contextmanager
import math
import time

import torch

from .batched_teacher import BatchedTeacherSolver
from .common.compile_regions import _preserve_host_rng_state
from .teacher_graph import _RNG_LOCK


_COUNTERS = (
    "inner_total_model_steps", "inner_actor_optimizer_steps", "inner_critic_optimizer_steps",
)


def _require_cuda(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Concurrent batched teachers require an available CUDA device.")


def _record_stream(tensor, stream):
    if tensor.device.type == "cuda":
        tensor.record_stream(stream)


class ConcurrentBatchedTeacherSolver:
    """Enqueue independent graph workers, joining once after the full refresh.

    There are no Python worker threads. The host enqueues root chunks in input
    order; CUDA streams provide concurrency. A worker's output is cloned on its
    own stream before its graph buffers are reused. The worker count does not
    affect the root chunks, seed derivation, padding or adaptation budget.
    """

    def __init__(self, agent):
        self.agent, self.cfg = agent, agent.cfg
        self.device = torch.device(agent.device)
        self.width = self.cfg.reanalyze_parallel_roots
        self.worker_count = self.cfg.reanalyze_batched_workers
        if self.cfg.reanalyze_backend != "batched_cuda_graph":
            raise ValueError("Concurrent batched teachers require reanalyze_backend='batched_cuda_graph'.")
        for name, value in (("reanalyze_parallel_roots", self.width),
                            ("reanalyze_batched_workers", self.worker_count)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.workers = []
        self._active = False

    @property
    def device_index(self):
        return self.device.index if self.device.index is not None else torch.cuda.current_device()

    def _validate_inputs(self, roots, seeds):
        _require_cuda(self.device)
        expected_device = (torch.device("cuda", self.device_index)
                           if self.device.type == "cuda" else self.device)
        if (not torch.is_tensor(roots) or roots.ndim != 2
                or roots.shape[1] != self.cfg.latent_dim
                or roots.dtype != torch.float32 or roots.device != expected_device):
            raise ValueError("Batched teacher roots must be float32 [roots, latent] on the agent device.")
        if not isinstance(seeds, (list, tuple)) or len(seeds) != len(roots) or any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
            for seed in seeds
        ):
            raise ValueError("Batched teachers require one nonnegative int64 seed per root.")

    @contextmanager
    def _seeded_rng(self, seed):
        # In PyTorch 2.3 each graph owns device seed/offset tensors. The host
        # seed swap and graph enqueue are serialized; execution remains on the
        # worker stream. Do not share graph pools across concurrent workers.
        devices = [self.device_index] if self.device.type == "cuda" else []
        with _RNG_LOCK, torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
            yield

    def _new_worker(self):
        return BatchedTeacherSolver(self.agent)

    def _ensure_workers(self, count):
        started = time.perf_counter()
        created = False
        # All captures must finish before any asynchronous teacher solve starts.
        # Register the worker before setup so failures can join a partially
        # initialized stream as well as all previous workers.
        for _ in range(len(self.workers), count):
            worker = self._new_worker()
            self.workers.append(worker)
            with self._seeded_rng(0):
                worker._setup()
            created = True
        return time.perf_counter() - started if created else 0.

    def _metrics(self, roots, active_workers, setup_seconds=0.):
        chunks = math.ceil(roots / self.width)
        model_steps = (self.workers[0].core.counts["inner_total_model_steps"]
                       if self.workers and roots else 0.)
        features = (getattr(self.workers[0].core, "optimization_features", ())
                    if self.workers and roots else ())
        return {
            "distillation_reanalyze_batched": 1.,
            "distillation_reanalyze_cuda_graph": 1.,
            "distillation_reanalyze_fused_adam": 1.,
            "distillation_reanalyze_parallel_roots": float(self.width),
            "distillation_reanalyze_batched_workers": float(active_workers),
            "distillation_reanalyze_selective_q": float("selective_q" in features),
            "distillation_reanalyze_actor_cache": float("actor_cache" in features),
            "distillation_reanalyze_fused_updates": float("fused_updates" in features),
            "distillation_reanalyze_first_layer_fusion": float("first_layer_fusion" in features),
            "distillation_reanalyze_chunks": float(chunks),
            "distillation_reanalyze_padding_roots": float((-roots) % self.width),
            "distillation_reanalyze_executed_model_steps": float(chunks * self.width * model_steps),
            "distillation_reanalyze_setup_seconds": setup_seconds,
        }

    def solve(self, roots, seeds):
        self._validate_inputs(roots, seeds)
        if self._active:
            raise RuntimeError("Concurrent batched teacher solves are not reentrant.")
        active_workers = min(self.worker_count, math.ceil(len(roots) / self.width))
        if not active_workers:
            empty = roots.new_empty((0, self.cfg.action_dim))
            return empty, empty.clone(), roots.new_zeros(3), self._metrics(0, 0)
        caller = torch.cuda.current_stream(self.device)
        modes = tuple((module, module.training) for module in self.agent.modules())
        devices = [self.device_index] if self.device.type == "cuda" else []
        self._active = True
        try:
            # Keep host and tensor RNG/modes isolated through cleanup, including
            # an error after some workers have already been enqueued.
            with _preserve_host_rng_state(), torch.random.fork_rng(devices=devices), torch.enable_grad():
                try:
                    self.agent.model.eval()
                    setup_seconds = self._ensure_workers(active_workers)
                    workers = self.workers[:active_workers]
                    for worker in workers:
                        worker.core.refresh_inputs()
                    for worker in workers:
                        worker.stream.wait_stream(caller)
                        _record_stream(roots, worker.stream)
                    copies = []
                    for chunk_index, start in enumerate(range(0, len(roots), self.width)):
                        worker = workers[chunk_index % active_workers]
                        chunk = roots[start:start + self.width]
                        seed = worker._seed(seeds[start:start + self.width])
                        with torch.cuda.stream(worker.stream), self._seeded_rng(seed), torch.no_grad():
                            worker.core.root.zero_()
                            worker.core.root[:len(chunk)].copy_(chunk)
                            worker.graph.replay()
                            copies.append(tuple(output[:len(chunk)].clone() for output in worker.output))
                    # No worker waits for another worker. Join before reading
                    # results or restoring modes/updating shared outer tensors.
                    for worker in workers:
                        caller.wait_stream(worker.stream)
                    for outputs in copies:
                        for output in outputs:
                            _record_stream(output, caller)
                    mean, log_std = (torch.cat([outputs[index] for outputs in copies]) for index in (0, 1))
                    counts = roots.new_tensor([
                        workers[0].core.counts[key] * len(roots) for key in _COUNTERS
                    ])
                    finite = torch.isfinite(mean).all() & torch.isfinite(log_std).all()
                    caller.synchronize()
                    if not bool(finite):
                        raise ValueError("Batched teacher parameters must be finite.")
                    return mean, log_std, counts, self._metrics(len(roots), active_workers, setup_seconds)
                except BaseException as error:
                    for worker in self.workers:
                        if worker.stream is not None:
                            try:
                                worker.stream.synchronize()
                            except BaseException as cleanup_error:
                                if hasattr(error, "add_note"):
                                    error.add_note(f"Joining a failed batched teacher also failed: {cleanup_error}")
                    self.workers.clear()
                    raise
        finally:
            for module, training in modes:
                module.training = training
            self._active = False


__all__ = ["ConcurrentBatchedTeacherSolver"]
