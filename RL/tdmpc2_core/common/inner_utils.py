"""Utilities shared by AMBI inner-improvement strategies."""

from contextlib import contextmanager

import torch

from .lora import LoRALinear, LoRANormedLinear


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
    for key, adapted_value in adapted_state.items():
        anchor_value = anchor_state[key]
        new_value = new_state[key].to(device=adapted_value.device, dtype=adapted_value.dtype)
        if adapted_value.is_floating_point():
            adapted_value.copy_(new_value + (adapted_value - anchor_value))
        else:
            adapted_value.copy_(new_value)
        anchor_value.copy_(new_value)


@torch.no_grad()
def rebase_lora_base_(adapted, new_base):
    """Refresh frozen LoRA base layers in-place, preserving adapters and optimizers."""
    for path, module in adapted.named_modules():
        if not isinstance(module, (LoRALinear, LoRANormedLinear)):
            continue
        source = new_base.get_submodule(path)
        module.base.load_state_dict(source.state_dict())


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
    )

    def __init__(self, seed, device):
        self.device = torch.device(device)
        generator_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.generators = {}
        for index, name in enumerate(self.STREAMS):
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(int(seed) + 104729 * (index + 1))
            self.generators[name] = generator

    def generator(self, name):
        return self.generators[name]

    @contextmanager
    def fork(self, name):
        """Run implicit-random ops (dropout/randperm) on a private phase seed."""
        generator = self.generator(name)
        seed_tensor = torch.randint(
            0,
            2**31 - 1,
            (1,),
            device=generator.device,
            generator=generator,
        )
        phase_seed = int(seed_tensor.cpu().item())
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices, enabled=True):
            # Seed the CPU default generator directly. ``torch.manual_seed``
            # also touches every CUDA device, which a single-device fork would
            # not restore in multi-GPU training.
            torch.random.default_generator.manual_seed(phase_seed)
            if self.device.type == "cuda":
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(phase_seed)
            yield generator
