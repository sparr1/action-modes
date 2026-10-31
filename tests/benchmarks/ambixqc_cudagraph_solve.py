"""Experimental single-CUDA-Graph XQC solve for the J8 timing harness.

The graph includes fresh prior/target copies, optimizer-moment reset, all
imagined rounds, replay gathers, critic/actor/temperature steps, projections,
target updates, action execution and the usual non-diagnostic metrics.
Root encoding, scalar input refresh, private-stream random draws, CPU action
materialization, held-policy snapshots and outer learning remain outside.

PyTorch 2.3 captures only its default CUDA generator. Inputs preserve every
original private-stream draw (shape, order, randint bound) instead of changing
scientific RNG streams or installing a different runtime. capture=False runs
the identical staged body eagerly for CPU/CUDA parity testing.
"""

from contextlib import nullcontext
import time

import torch

from RL.tdmpc2_core.inner_xqc import InnerXQCEngine, InnerXQCState


class SolveInputs:
    def __init__(self, cfg, reference):
        j, h, n, b, a, g = (int(getattr(cfg, name)) for name in (
            "inner_rounds", "inner_rollout_horizon", "inner_rollouts_per_round",
            "inner_batch_size", "action_dim", "inner_updates_per_round"))
        self.noise = {name: [reference.new_empty(shape) for _ in range(count)]
                      for name, count, shape in (
                          ("collection", j * h, (n, a)),
                          ("bootstrap", j * g, (b, a)),
                          ("gradient_policy", j * g, (b, a)),
                          ("execution", 1, (1, a)))}
        self.indices = [torch.empty(b, dtype=torch.long, device=reference.device)
                        for _ in range(j * g)]
        self.bounds = [(round_index + 1) * h * n for round_index in range(j)
                       for _ in range(g)]
        self.reset_cursors()

    def reset_cursors(self):
        self.cursors = dict.fromkeys(self.noise, 0)
        self.replay_cursor = 0

    def fill(self, rng):
        # Streams are independent. Preserve draw order and shape within each.
        for name, buffers in self.noise.items():
            for buffer in buffers:
                torch.randn(buffer.shape, dtype=buffer.dtype, device=buffer.device,
                            generator=rng.generator(name), out=buffer)
        for bound, buffer in zip(self.bounds, self.indices):
            torch.randint(bound, buffer.shape, device=buffer.device, dtype=buffer.dtype,
                          generator=rng.generator("replay"), out=buffer)

    def normal(self, shape, *, dtype, device, stream):
        index = self.cursors[stream]
        result = self.noise[stream][index]
        self.cursors[stream] += 1
        if tuple(shape) != tuple(result.shape) or dtype != result.dtype or torch.device(device) != result.device:
            raise RuntimeError("Unexpected random input shape/device/dtype")
        return result

    def replay_indices(self):
        result = self.indices[self.replay_cursor]
        self.replay_cursor += 1
        return result

    def assert_consumed(self):
        if any(self.cursors[name] != len(buffers) for name, buffers in self.noise.items()):
            raise RuntimeError("Incomplete solve noise consumption")
        if self.replay_cursor != len(self.indices):
            raise RuntimeError("Incomplete solve replay consumption")


class _PreparedRNG:
    """The original solver's RNG contexts were handled before graph launch."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def action_fork(self):
        return nullcontext()

    def fork(self, name):
        return nullcontext(self.original.generator(name))


class CapturedXQCSolve(InnerXQCEngine):
    def __init__(self, agent, *, capture=True):
        super().__init__(agent)
        cfg = self.cfg
        required = {"inner_operator": "xqc", "inner_update_timing": "round",
                    "inner_reward_normalization": "frozen_real_scale",
                    "inner_replay_sampling": "with_replacement", "aux_return_mode": "off",
                    "inner_terminal_bootstrap": "inner", "inner_critic_source": "xqc",
                    "inner_horizon_critic_source": "xqc", "inner_critic_target": "entropy_augmented"}
        if cfg.episodic or cfg.obs != "state" or any(getattr(cfg, k) != v for k, v in required.items()):
            raise ValueError("Whole-solve benchmark requires dense state XQC, fresh soft critics, round updates and frozen reward scale")
        if cfg.inner_replay_capacity < cfg.inner_model_step_budget:
            raise ValueError("Whole-solve benchmark requires replay without eviction")
        if capture and self.device.type != "cuda":
            raise ValueError("CUDA Graph capture requires a CUDA device")
        if capture and cfg.xqc_optimizer_backend not in {"auto", "fused"}:
            raise ValueError("CUDA solve capture requires the production fused optimizer backend")
        self.capture_enabled = capture
        self._scientific_rng = self.rng
        self.rng = _PreparedRNG(self._scientific_rng)
        self.graph = None
        self.inputs = None
        self.capture_count = self.replay_count = 0
        self.initialization_seconds = 0.0

    def _timer_start(self):
        return None

    def _timer_stop(self, key, start):
        pass

    def _normal(self, shape, **kwargs):
        return self.inputs.normal(shape, **kwargs)

    def _sample_batch(self):
        batch = self.state.replay.sample(self.cfg.inner_batch_size, include_ids=False,
                                         indices=self.inputs.replay_indices())
        self.state.replay_draws += int(batch["z"].shape[0])
        return batch

    def _prepare_action(self):
        workspace = self._static_workspace
        workspace.controller.reset_prior_from_(self.outer_controller)
        for optimizer in self._optimizers:
            for state in optimizer.state.values():
                for value in state.values():
                    if not torch.is_tensor(value):
                        raise RuntimeError("Optimizer state must be tensor-only")
                    value.zero_()
        workspace.update_step = workspace.actor_optimizer_steps = workspace.temperature_optimizer_steps = 0
        self._static_replay.clear()
        self.inputs.reset_cursors()
        self.state = InnerXQCState(workspace=workspace, replay=self._static_replay,
                                   reward_scale=self._scale, reward_scale_initial=self._scale,
                                   reward_normalizer_count_initial=self._normalizer_count)

    def _body(self, eval_mode):
        result = super().act(self._root, eval_mode=eval_mode, collect_diagnostics=False)
        self.inputs.assert_consumed()
        return result

    def _initialize(self, root_z, eval_mode):
        devices = [root_z.device.index] if root_z.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            self._initialize_preserving_private_rng(root_z, eval_mode)

    def _initialize_preserving_private_rng(self, root_z, eval_mode):
        started = time.perf_counter()
        rng_state = self._scientific_rng.training_state_dict()
        action_index = self.action_index
        try:
            with self._scientific_rng.action_fork(), self._scientific_rng.fork("initialization"):
                InnerXQCEngine._prepare_action(self)
            self._static_workspace, self._static_replay = self.state.workspace, self.state.replay
            self._release_action()
            self._optimizers = (self._static_workspace.actor_optimizer,
                                self._static_workspace.critic_optimizer,
                                self._static_workspace.temperature_optimizer)
            for optimizer in self._optimizers:
                for group in optimizer.param_groups:
                    group["capturable"] = self.device.type == "cuda"
                    for parameter in group["params"]:
                        optimizer.state[parameter] = {
                            "step": torch.zeros((), dtype=torch.float32, device=parameter.device),
                            "exp_avg": torch.zeros_like(parameter),
                            "exp_avg_sq": torch.zeros_like(parameter),
                        }
            self._root = root_z.detach().clone()
            self._scale = root_z.new_tensor(self._real_reward_scale())
            self._normalizer_count = root_z.new_tensor(self._real_reward_normalizer_count())
            self.inputs = SolveInputs(self.cfg, root_z)
            self.inputs.fill(self._scientific_rng)
            self._eval_mode = bool(eval_mode)
            if self.capture_enabled:
                current = torch.cuda.current_stream(self.device)
                side = torch.cuda.Stream(device=self.device)
                side.wait_stream(current)
                with torch.cuda.stream(side), torch.enable_grad():
                    for _ in range(3):
                        self._body(eval_mode)
                current.wait_stream(side)
                torch.cuda.synchronize(self.device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=side):
                    self._result = self._body(eval_mode)
                current.wait_stream(side)
                torch.cuda.synchronize(self.device)
                self.capture_count += 1
        finally:
            self._scientific_rng.load_training_state_dict(rng_state)
            self.action_index = action_index
        self.initialization_seconds += time.perf_counter() - started

    def act(self, root_z, *, t0=False, eval_mode=False, collect_diagnostics=True):
        del t0
        if collect_diagnostics:
            raise ValueError("Whole-solve benchmark supports non-diagnostic actions only")
        if self.inputs is None:
            self._initialize(root_z, eval_mode)
        if bool(eval_mode) != self._eval_mode or root_z.shape != self._root.shape:
            raise ValueError("Captured solve shape/evaluation mode changed")
        self._root.copy_(root_z)
        self._scale.fill_(self._real_reward_scale())
        self._normalizer_count.fill_(self._real_reward_normalizer_count())
        # Advance the original initialization phase once even on reused pools.
        with self._scientific_rng.action_fork(), self._scientific_rng.fork("initialization"):
            pass
        self.inputs.fill(self._scientific_rng)
        if self.capture_enabled:
            self.graph.replay()
            self.action_index += 1
            self.replay_count += 1
            return self._result
        return self._body(eval_mode)

    @property
    def capture_status(self):
        return {"enabled": self.capture_enabled, "captures": self.capture_count,
                "replays": self.replay_count, "initialization_seconds": self.initialization_seconds,
                "random_inputs": "fresh private-stream draws outside graph",
                "scope": "fresh prior/reset, all J rounds, replay gathers, all updates, target/projection, execution and non-diagnostic metrics"}

    def clear_all(self):
        self.graph = None
        self.inputs = None
        super().clear_all()
        self._scientific_rng = self.rng
        self.rng = _PreparedRNG(self._scientific_rng)
