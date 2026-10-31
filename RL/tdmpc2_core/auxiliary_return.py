"""Reward-only auxiliary critic for the historical no-inner SAC prior.

This intentionally small backport adds no return actor, planner, or alternate
critic backend. Model parameters and their Adam group belong to the main agent;
this helper owns the auxiliary sample stream and checkpoint provenance.
"""

from contextlib import contextmanager

import torch

from .common import math as td_math
from .common.training_state import require_exact_keys, require_tensor


class AuxiliaryReturnLearner:
    def __init__(self, owner):
        self.cfg, self.model, self.device = owner.cfg, owner.model, owner.device
        self.discount = owner.discount
        self._transition_temporal_weights = owner._transition_temporal_weights
        self.num_updates = 0
        self.has_actor = False
        self.pi_optim = self.ent_coef_optim = self.log_ent_coef = None
        seed = int(self.cfg.seed) + 130363
        self._cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
        self._device_generator = (
            torch.Generator(device=self.device).manual_seed(seed + 1)
            if self.device.type == "cuda" else None
        )

    @contextmanager
    def isolated_rng(self):
        devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                   if self.device.type == "cuda" else [])
        with torch.random.fork_rng(devices=devices):
            torch.random.set_rng_state(self._cpu_generator.get_state())
            if self._device_generator is not None:
                torch.cuda.set_rng_state(self._device_generator.get_state(), self.device)
            try:
                yield
            finally:
                self._cpu_generator.set_state(torch.random.get_rng_state())
                if self._device_generator is not None:
                    self._device_generator.set_state(torch.cuda.get_rng_state(self.device))

    @torch.no_grad()
    def td_target(self, next_z, reward, terminated):
        with self.isolated_rng():
            action = self.model.pi_action(next_z)
            value = self.model.aux_return_Q(
                next_z, action, target=True, reduction=self.cfg.outer_q_target_reduction,
            )
            # There is deliberately no entropy term or extra temperature.
            value = torch.where(terminated == 1, 0.0, value)
            return reward + self.discount * (1.0 - terminated) * value

    def critic_loss(self, latent_states, action, targets):
        latents = latent_states[:-1]
        if self.cfg.aux_return_detach_representation:
            latents = latents.detach()
        with self.isolated_rng():
            predictions = self.model.aux_return_q_predictions(latents, action)
        per_sample = self.model.q_backend.loss(predictions, targets, reduction="none")
        per_time = per_sample.mean(dim=(0,) + tuple(range(2, per_sample.ndim)))
        loss = td_math.reduce_temporal_loss(
            per_time, self.cfg.rho,
            normalization=self.cfg.temporal_loss_normalization,
            reference_horizon=self.cfg.temporal_loss_reference_horizon,
            legacy_order="vector_sum_divide", weights=self._transition_temporal_weights,
        )
        return loss, predictions

    def specification(self):
        return {
            "version": 1, "mode": "sac", "target": "reward_only",
            "critic_coef": float(self.cfg.aux_return_critic_coef),
            "detach_representation": bool(self.cfg.aux_return_detach_representation),
            "critic_lr": float(self.cfg.aux_return_critic_lr),
            "critic": self.model.critic_signature,
            "target_reduction": self.cfg.outer_q_target_reduction,
            "initialization": "historical-trunc-normal-0.02-zero-bias-zero-output-weight",
            "gradient_clipping": "shared-world-primary-auxiliary",
            "temporal_loss_normalization": self.cfg.temporal_loss_normalization,
            "temporal_loss_reference_horizon": int(self.cfg.temporal_loss_reference_horizon),
            "train_unroll_horizon": int(self.cfg.train_unroll_horizon),
            "rho": float(self.cfg.rho),
        }

    def checkpoint_state(self):
        state = {"num_updates": self.num_updates, "rng_cpu": self._cpu_generator.get_state()}
        if self._device_generator is not None:
            state["rng_device"] = self._device_generator.get_state()
        return state

    def preflight_state(self, payload, *, exact=False, expected_updates=None):
        keys = set(self.checkpoint_state())
        if not exact:
            keys.discard("rng_device")
            if isinstance(payload, dict) and "rng_device" in payload:
                keys.add("rng_device")
        state = require_exact_keys(payload, keys, "Auxiliary return state")
        updates = state["num_updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("Auxiliary return num_updates must be a nonnegative integer.")
        if expected_updates is not None and updates != expected_updates:
            raise ValueError("Auxiliary return update count must match the outer learner.")
        for key, generator in (("rng_cpu", self._cpu_generator), ("rng_device", self._device_generator)):
            if key not in state:
                continue
            value = require_tensor(state[key], f"Auxiliary return {key}", dtype=torch.uint8)
            if value.ndim != 1 or value.numel() == 0:
                raise ValueError(f"Auxiliary return {key} must be a nonempty byte vector.")
            if generator is not None:
                if value.shape != generator.get_state().shape:
                    raise ValueError(f"Auxiliary return {key} has incompatible shape.")
                probe = torch.Generator(device=generator.device)
                try:
                    probe.set_state(value.cpu())
                except (RuntimeError, TypeError) as exc:
                    raise ValueError(f"Auxiliary return {key} is invalid.") from exc
        return state

    def load_state(self, state):
        self.num_updates = state["num_updates"]
        self._cpu_generator.set_state(state["rng_cpu"].cpu())
        if self._device_generator is not None and "rng_device" in state:
            self._device_generator.set_state(state["rng_device"].cpu())
