"""Utilities shared by AMBI inner-improvement strategies."""

from contextlib import contextmanager
import math

import torch
import torch.nn as nn

from .lora import LoRALinear, LoRANormedLinear
from .training_state import require_exact_keys, require_tensor


_SCOPES = {"action": 0, "episode": 1, "run": 2}


def allocate_across_rounds(total, rounds):
    """Distribute an integer budget deterministically, front-loading remainders."""
    total, rounds = int(total), int(rounds)
    if total < 0:
        raise ValueError("Update totals must be non-negative.")
    if rounds <= 0:
        if total:
            raise ValueError("A positive update total requires at least one inner round.")
        return []
    quotient, remainder = divmod(total, rounds)
    return [quotient + int(index < remainder) for index in range(rounds)]


def scope_rank(scope):
    try:
        return _SCOPES[str(scope).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown inner lifecycle scope: {scope!r}") from exc


def scope_expired(scope, *, t0, action_start):
    scope = str(scope).lower()
    scope_rank(scope)
    return bool(
        (scope == "action" and action_start)
        or (scope == "episode" and t0)
    )


def trainable_parameter_count(module):
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


@torch.no_grad()
def rebase_clone_(adapted, anchor, new_base):
    """Move a full-copy adapter to a new outer anchor while preserving its delta."""
    adapted_state = adapted.state_dict()
    anchor_state = anchor.state_dict()
    new_state = new_base.state_dict()
    if adapted_state.keys() != anchor_state.keys() or adapted_state.keys() != new_state.keys():
        raise ValueError("Cannot rebase modules with different state layouts.")
    _rebase_clone_state_(adapted_state, None, anchor_state, new_state)


def _rebase_clone_state_(adapted_state, target_state, anchor_state, new_state):
    """Batch a persistent clone rebase across all floating tensors."""
    adapted_values = []
    target_values = []
    anchor_values = []
    new_values = []
    for key, adapted_value in adapted_state.items():
        anchor_value = anchor_state[key]
        new_value = new_state[key].to(
            device=adapted_value.device,
            dtype=adapted_value.dtype,
        )
        if adapted_value.is_floating_point():
            adapted_values.append(adapted_value)
            anchor_values.append(anchor_value)
            new_values.append(new_value)
            if target_state is not None:
                target_values.append(target_state[key])
        else:
            adapted_value.copy_(new_value)
            if target_state is not None:
                target_state[key].copy_(new_value)
            anchor_value.copy_(new_value)

    if adapted_values:
        deltas = torch._foreach_sub(new_values, anchor_values)
        torch._foreach_add_(adapted_values, deltas)
        if target_state is not None:
            torch._foreach_add_(target_values, deltas)
        torch._foreach_copy_(anchor_values, new_values)


@torch.no_grad()
def rebase_clone_with_target_(adapted, target, anchor, new_base):
    """Rebase online and target clones without allocating an anchor copy."""
    adapted_state = adapted.state_dict()
    target_state = target.state_dict() if target is not None else None
    anchor_state = anchor.state_dict()
    new_state = new_base.state_dict()
    layouts = (adapted_state.keys(), anchor_state.keys(), new_state.keys())
    if layouts[0] != layouts[1] or layouts[0] != layouts[2]:
        raise ValueError("Cannot rebase modules with different state layouts.")
    if target_state is not None and target_state.keys() != layouts[0]:
        raise ValueError("Cannot rebase a target with a different state layout.")

    _rebase_clone_state_(adapted_state, target_state, anchor_state, new_state)


@torch.no_grad()
def rebase_lora_base_(adapted, new_base):
    """Refresh owned bases or rebind zero-copy shared bases in-place."""
    found = False
    for path, module in adapted.named_modules():
        if not isinstance(module, (LoRALinear, LoRANormedLinear)):
            continue
        source = new_base.get_submodule(path)
        if module.shares_base:
            module.share_base_(source)
        else:
            module.base.load_state_dict(source.state_dict())
        found = True
    if not found:
        raise ValueError("No LoRA adapters were found to rebase.")


def lora_uses_shared_bases(module):
    """Return whether every adapter references an unregistered outer base."""
    adapters = [
        child
        for child in module.modules()
        if isinstance(child, (LoRALinear, LoRANormedLinear))
    ]
    return bool(adapters) and all(child.shares_base for child in adapters)


@torch.no_grad()
def reset_lora_adapters_(module):
    """Reset LoRA tensors exactly like constructing fresh adapters."""
    found = False
    for child in module.modules():
        if not isinstance(child, (LoRALinear, LoRANormedLinear)):
            continue
        nn.init.kaiming_uniform_(child.lora_A, a=math.sqrt(5))
        child.lora_B.zero_()
        found = True
    if not found:
        raise ValueError("No LoRA adapters were found to reset.")


@torch.no_grad()
def copy_lora_adapters_(source, target, tau=1.0):
    """Synchronize only trainable LoRA tensors between compatible modules."""
    source_values = {
        name: value
        for name, value in source.named_parameters()
        if name.endswith(("lora_A", "lora_B"))
    }
    target_values = {
        name: value
        for name, value in target.named_parameters()
        if name.endswith(("lora_A", "lora_B"))
    }
    if source_values.keys() != target_values.keys() or not source_values:
        raise ValueError("LoRA source and target adapter layouts must match.")
    source_tensors = [source_values[name].detach() for name in source_values]
    target_tensors = [target_values[name] for name in source_values]
    tau = float(tau)
    if tau == 1.0:
        torch._foreach_copy_(target_tensors, source_tensors)
    else:
        torch._foreach_lerp_(target_tensors, source_tensors, tau)


class InnerRNG:
    """Independent RNG streams that never advance PyTorch's outer global state."""

    STREAMS = (
        "initialization",
        "collection",
        "replay",
        "bootstrap",
        "gradient_policy",
        "execution",
        "mppi",
        "diagnostics",
        "observation",
    )

    def __init__(self, seed, device):
        self.device = torch.device(device)
        generator_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.generators = {}
        self.phase_generators = {}
        self._action_fork_depth = 0
        for index, name in enumerate(self.STREAMS):
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(int(seed) + 104729 * (index + 1))
            self.generators[name] = generator
            # Implicit-random operations (notably dropout) still need a
            # temporary default-generator seed. Keep that seed source on the
            # host so entering a phase never synchronizes CUDA.
            phase_generator = torch.Generator(device="cpu")
            phase_generator.manual_seed(int(seed) + 130363 * (index + 1))
            self.phase_generators[name] = phase_generator

    def generator(self, name):
        return self.generators[name]

    def training_state_dict(self):
        """Return every private generator at a quiescent action boundary."""
        if self._action_fork_depth != 0:
            raise RuntimeError(
                "Inner RNG state can only be captured outside an active RNG fork."
            )
        return {
            "schema": "ambi-inner-rng-training-state",
            "version": 1,
            "device_type": self.device.type,
            "streams": {
                name: generator.get_state().clone()
                for name, generator in self.generators.items()
            },
            "phase_streams": {
                name: generator.get_state().clone()
                for name, generator in self.phase_generators.items()
            },
            "action_fork_depth": 0,
        }

    def _preflight_training_state_dict(self, state):
        state = require_exact_keys(
            state,
            {
                "schema",
                "version",
                "device_type",
                "streams",
                "phase_streams",
                "action_fork_depth",
            },
            "inner RNG training state",
        )
        if (
            state["schema"] != "ambi-inner-rng-training-state"
            or state["version"] != 1
        ):
            raise ValueError("Unsupported AMBI inner RNG training-state version.")
        if state["device_type"] != self.device.type:
            raise ValueError(
                "AMBI inner RNG device type is incompatible: "
                f"checkpoint={state['device_type']!r}, configured={self.device.type!r}."
            )
        if state["action_fork_depth"] != 0:
            raise ValueError("An active AMBI inner RNG fork cannot be restored.")
        streams = require_exact_keys(
            state["streams"], self.STREAMS, "inner RNG streams"
        )
        phase_streams = require_exact_keys(
            state["phase_streams"], self.STREAMS, "inner RNG phase streams"
        )
        generator_device = (
            self.device if self.device.type == "cuda" else torch.device("cpu")
        )
        # ``set_state`` performs the backend-specific structural validation.
        # Use temporary generators so a bad later stream cannot partially
        # advance or replace any live generator.
        for name in self.STREAMS:
            stream = require_tensor(
                streams[name], f"inner RNG stream {name!r}", dtype=torch.uint8
            )
            phase = require_tensor(
                phase_streams[name],
                f"inner RNG phase stream {name!r}",
                dtype=torch.uint8,
            )
            if stream.ndim != 1 or phase.ndim != 1:
                raise ValueError("Serialized generator states must be one-dimensional.")
            probe = torch.Generator(device=generator_device)
            probe.set_state(stream.detach().cpu())
            phase_probe = torch.Generator(device="cpu")
            phase_probe.set_state(phase.detach().cpu())
        return state

    def load_training_state_dict(self, state):
        """Restore every named and implicit-randomness seed stream exactly."""
        state = self._preflight_training_state_dict(state)
        for name in self.STREAMS:
            self.generators[name].set_state(state["streams"][name].detach().cpu())
            self.phase_generators[name].set_state(
                state["phase_streams"][name].detach().cpu()
            )
        self._action_fork_depth = 0
        return self

    def _fork_devices(self):
        if self.device.type != "cuda":
            return []
        return [
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        ]

    @contextmanager
    def action_fork(self):
        """Save outer default RNG state once for an entire inner action.

        Individual phases are reseeded from private CPU generators below, but
        no longer fetch and restore CUDA RNG state for every phase. This keeps
        implicit module randomness (notably dropout) isolated with one
        action-level save/restore instead of many device synchronizations.
        """
        if self._action_fork_depth:
            self._action_fork_depth += 1
            try:
                yield
            finally:
                self._action_fork_depth -= 1
            return
        devices = self._fork_devices()
        with torch.random.fork_rng(devices=devices, enabled=True):
            self._action_fork_depth = 1
            try:
                yield
            finally:
                self._action_fork_depth = 0

    @contextmanager
    def fork(self, name):
        """Run implicit-random ops (dropout/randperm) on a private phase seed."""
        generator = self.generator(name)
        phase_seed = int(torch.randint(
            0,
            2**31 - 1,
            (1,),
            device="cpu",
            generator=self.phase_generators[name],
        ).item())
        devices = self._fork_devices()

        def seed_phase():
            # Seed the CPU default generator directly. ``torch.manual_seed``
            # also touches every CUDA device, which a single-device fork would
            # not restore in multi-GPU training.
            torch.random.default_generator.manual_seed(phase_seed)
            if self.device.type == "cuda":
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(phase_seed)

        if self._action_fork_depth:
            seed_phase()
            yield generator
            return
        with torch.random.fork_rng(devices=devices, enabled=True):
            seed_phase()
            yield generator
