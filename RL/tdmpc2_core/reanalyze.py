"""Fresh AMBI teacher solves for a bounded subset of sampled real replay."""

from contextlib import contextmanager
import time

import torch

from .common.compile_regions import _preserve_host_rng_state
from .common.training_state import require_exact_keys, require_tensor
from .inner_improvement import InnerImprovementEngine


@contextmanager
def _preserve_global_rng():
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with _preserve_host_rng_state(), torch.random.fork_rng(devices=devices):
        yield


class LazyReanalyzer:
    """Own a disposable inner learner and a checkpointed stream of root seeds.

    Cadence belongs to the outer learner. Each call refreshes all actor states
    in the first configured number of sampled sequences, in sequence-first
    order, solving a repeated physical replay transition only once.
    """

    def __init__(self, agent):
        self.agent = agent
        self.cfg = agent.cfg
        self.backend = getattr(self.cfg, "reanalyze_backend", "native")
        if self.backend not in {"native", "cuda_graph"}:
            raise ValueError("Unknown reanalyze_backend.")
        if self.backend == "cuda_graph" and torch.device(agent.device).type != "cuda":
            raise ValueError("reanalyze_backend='cuda_graph' requires a CUDA agent.")
        self.parallel_solver = None
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed((int(self.cfg.seed) + 982451653) % (2**63 - 1))
        self.refreshes = 0
        self.roots = 0
        with _preserve_global_rng():
            self.engine = InnerImprovementEngine(agent)

    def state_dict(self):
        state = {
            "schema": "ambi-lazy-reanalyze",
            "version": 1,
            "rng_state": self.generator.get_state().clone(),
            "refreshes": self.refreshes,
            "roots": self.roots,
        }
        if self.backend == "cuda_graph":
            state.update(version=2, **self.backend_specification())
        return state

    def backend_specification(self):
        from .teacher_graph import TEACHER_GRAPH_PROTOCOL
        return {
            "reanalyze_backend": self.backend,
            "reanalyze_rng_protocol": TEACHER_GRAPH_PROTOCOL,
            "reanalyze_parallel_roots": int(self.cfg.reanalyze_parallel_roots),
        }

    def preflight_state_dict(self, state):
        expected = {"schema", "version", "rng_state", "refreshes", "roots"}
        graph_spec = self.backend_specification() if self.backend == "cuda_graph" else {}
        state = require_exact_keys(
            state, expected | graph_spec.keys(),
            "lazy reanalyze state",
        )
        version = 2 if graph_spec else 1
        if state["schema"] != "ambi-lazy-reanalyze" or state["version"] != version:
            raise ValueError("Unsupported lazy reanalyze state version.")
        if any(state[key] != value for key, value in graph_spec.items()):
            raise ValueError("Incompatible reanalysis backend or RNG protocol.")
        for name in ("refreshes", "roots"):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Lazy reanalyze {name} must be a non-negative integer.")
        rng_state = require_tensor(state["rng_state"], "lazy reanalyze RNG", dtype=torch.uint8)
        if rng_state.ndim != 1:
            raise ValueError("Lazy reanalyze RNG state must be one-dimensional.")
        rng_state = rng_state.detach().cpu().clone()
        torch.Generator(device="cpu").set_state(rng_state)
        return {**state, "rng_state": rng_state}

    def load_state_dict(self, state):
        candidate = self.preflight_state_dict(state)
        # Reset before committing the seed stream, so a busy engine rejects a
        # load without partially installing its resume state.
        with _preserve_global_rng():
            self.engine.reset_for_evaluation(int(self.cfg.seed), reuse_action_pool=True)
        self.generator.set_state(candidate["rng_state"])
        self.refreshes = candidate["refreshes"]
        self.roots = candidate["roots"]
        return self

    def _positions(self, obs, teacher):
        indices, episodes = teacher["index"], teacher["episode"]
        if indices.ndim != 2 or episodes.shape != indices.shape:
            raise ValueError("Teacher replay identities must have shape [H, B].")
        horizon, batch = indices.shape
        if obs.shape[:2] != (horizon + 1, batch):
            raise ValueError("Replay observations must align with teacher states as [H+1, B, ...].")
        sequences = min(int(self.cfg.reanalyze_batch_size), batch)
        selected_indices = indices[:, :sequences].detach().cpu().tolist()
        selected_episodes = episodes[:, :sequences].detach().cpu().tolist()
        seen, positions = set(), []
        for column in range(sequences):
            for step in range(horizon):
                identity = (selected_episodes[step][column], selected_indices[step][column])
                if identity[0] < 0 or identity[1] < 0:
                    raise ValueError("Reanalysis requires resident replay identities.")
                if identity not in seen:
                    seen.add(identity)
                    positions.append((step, column))
        return torch.tensor(positions, dtype=torch.long).reshape(-1, 2), sequences, horizon

    def refresh(self, buffer, obs, teacher, outer_version):
        positions, sequences, horizon = self._positions(obs, teacher)
        if isinstance(outer_version, bool) or not isinstance(outer_version, int) or outer_version < 0:
            raise ValueError("Reanalysis outer_version must be a non-negative integer.")
        count = len(positions)
        metrics = {
            "distillation_reanalyze_ran": 1.0,
            "distillation_reanalyze_sequences": float(sequences),
            "distillation_reanalyze_requested_roots": float(sequences * horizon),
            "distillation_reanalyze_roots": float(count),
            "distillation_reanalyze_duplicates": float(sequences * horizon - count),
            "distillation_reanalyze_model_steps": 0.0,
            "distillation_reanalyze_actor_updates": 0.0,
            "distillation_reanalyze_critic_updates": 0.0,
            "distillation_reanalyze_seconds": 0.0,
        }
        if count == 0:
            return teacher, metrics
        started = time.perf_counter()
        modes = tuple((module, bool(module.training)) for module in self.agent.modules())
        rng_before = self.generator.get_state().clone()
        self.engine.outer_replay_buffer = buffer
        try:
            with _preserve_global_rng():
                self.agent.model.eval()
                at = positions.to(obs.device)
                selected_obs = obs[at[:, 0], at[:, 1]].to(self.agent.device, non_blocking=True)
                # Pixel augmentation may use the default Torch generator.
                # Its seed belongs to reanalysis too, independently of the
                # outer learner's current RNG position.
                observation_seed = int(torch.randint(0, 2**31 - 1, (), generator=self.generator))
                torch.random.default_generator.manual_seed(observation_seed)
                if torch.device(self.agent.device).type == "cuda":
                    with torch.cuda.device(self.agent.device):
                        torch.cuda.manual_seed(observation_seed)
                with torch.no_grad():
                    roots = self.agent.model.encode(selected_obs).detach()
                seeds = [int(torch.randint(0, 2**31 - 1, (), generator=self.generator))
                         for _ in range(count)]
                with torch.enable_grad():
                    means, log_stds, totals, solve_metrics = self._solve(roots, seeds)
                metrics.update(solve_metrics)
                refreshed = buffer.write_teacher_targets(
                    teacher, positions, means, log_stds, outer_version,
                )
                values = totals.detach().cpu().tolist()
                for name, value in zip(("model_steps", "actor_updates", "critic_updates"), values):
                    metrics[f"distillation_reanalyze_{name}"] = float(value)
        except BaseException:
            self.generator.set_state(rng_before)
            raise
        finally:
            for module, was_training in modes:
                module.training = was_training
        self.refreshes += 1
        self.roots += count
        metrics["distillation_reanalyze_seconds"] = max(0.0, time.perf_counter() - started)
        metrics["distillation_reanalyze_seconds_per_root"] = metrics["distillation_reanalyze_seconds"] / count
        return refreshed, metrics

    def _solve(self, roots, seeds):
        if self.backend == "cuda_graph":
            if self.parallel_solver is None:
                from .parallel_teacher import ParallelTeacherSolver
                self.parallel_solver = ParallelTeacherSolver(self.agent)
            return self.parallel_solver.solve(roots, seeds)
        means, log_stds = [], []
        totals = [0.0, 0.0, 0.0]
        for root, seed in zip(roots.split(1), seeds):
            self.engine.reset_for_evaluation(seed, reuse_action_pool=True)
            target, counters = self.engine.solve_teacher(root, t0=True)
            means.append(target["pre_tanh_mean"])
            log_stds.append(target["log_std"])
            for index, name in enumerate((
                "inner_total_model_steps", "inner_actor_optimizer_steps",
                "inner_critic_optimizer_steps",
            )):
                totals[index] += counters[name]
        return torch.stack(means), torch.stack(log_stds), roots.new_tensor(totals), {
            "distillation_reanalyze_cuda_graph": 0.0,
            "distillation_reanalyze_parallel_roots": 1.0,
            "distillation_reanalyze_setup_seconds": 0.0,
        }


__all__ = ["LazyReanalyzer"]
