"""Independent return evaluation and optional return-policy learning for AMBI.

The actor uses the ordinary SAC trainer through an explicit model view. This
shares the training equations, not parameters, optimizers, temperature, scale,
configuration, or random-number streams with the original SAC learner.
"""

from contextlib import contextmanager
import math
from types import SimpleNamespace

import torch

from .common import math as td_math
from .common.training_state import (
    load_optimizer_state_preserving_hyperparameters,
    preflight_adam_state,
    preflight_optimizer_state,
    require_exact_keys,
    require_tensor,
)


_SCALE_MODE = "tdmpc2_percentile_range"
_SHARED_METHODS = frozenset({
    "_update_actor", "_clip_actor_grad_norm_", "_update_actor_loss_scale",
    "_behavior_policy_inputs", "_behavior_policy_row_statistics",
    "_behavior_policy_kl_loss", "_behavior_policy_action_ce_loss",
    "_behavior_policy_regularizer_loss", "_actor_loss_scale_spec",
    "_actor_q_range_spec", "_actor_loss_scale_state", "_actor_q_range_state",
    "_preflight_actor_loss_scale", "_preflight_actor_q_range_state",
    "_behavior_policy_kl_spec", "_behavior_policy_kl_state",
    "_preflight_behavior_policy_kl", "_preflight_optimizer",
    "_optimizer_layout",
})
_ACTOR_SETTINGS = (
    "actor_lr", "actor_adam_eps", "adam_eps", "grad_clip_norm",
    "log_std_mapping", "log_std_min", "log_std_max", "ent_coef",
    "ent_coef_lr", "target_entropy", "outer_actor_entropy_mode",
    "sac_actor_loss_scale_mode", "sac_actor_loss_scale_tau",
    "outer_q_actor_reduction", "outer_behavior_policy_objective",
    "outer_behavior_policy_kl_schedule", "outer_behavior_policy_kl_coef",
    "outer_behavior_policy_kl_min_valid_count",
    "outer_behavior_policy_kl_ramp_updates",
    "outer_behavior_policy_kl_q_threshold", "outer_behavior_policy_kl_target",
    "outer_behavior_policy_kl_dual_init", "outer_behavior_policy_kl_dual_lr",
    "outer_behavior_policy_kl_dual_max",
)


class _ReturnModelView:
    """Read-only routing view used by the common actor-training equations."""

    def __init__(self, model, cfg):
        self._model = model
        self.cfg = cfg
        self.value_spec = model.value_spec
        self.q_backend = model.q_backend
        self._pi = getattr(model, "_return_pi", None)

    def pi(self, z, **kwargs):
        return self._model.pi(
            z, policy=self._pi, log_std_mapping=self.cfg.log_std_mapping,
            log_std_min=self.cfg.log_std_min, log_std_max=self.cfg.log_std_max,
            **kwargs,
        )

    def Q(self, z, action, **kwargs):
        return self._model.aux_return_Q(z, action, **kwargs)


class AuxiliaryReturnLearner(torch.nn.Module):
    """Own auxiliary optimizer/scalar/RNG state; the world model owns networks."""

    def __init__(self, owner):
        super().__init__()
        self._trainer_type = type(owner)
        self.device = owner.device
        config = dict(vars(owner.cfg))
        config.update(getattr(owner.cfg, "aux_return_actor_cfg", {}))
        config["outer_policy_diagnostics"] = False
        self.cfg = SimpleNamespace(**config)
        self.mode = str(owner.cfg.aux_return_mode)
        self.has_actor = self.mode == "return_actor"
        self.model = _ReturnModelView(owner.model, self.cfg)
        self.discount = owner.discount
        self.num_updates = 0
        self._outer_policy_diagnostics_force = False
        self._outer_policy_diagnostics_packet = None
        self._actor_temporal_weights = owner._actor_temporal_weights
        self._transition_temporal_weights = owner._transition_temporal_weights
        self._actor_entropy_mode = self.cfg.outer_actor_entropy_mode
        self.target_entropy = (
            -float(self.cfg.action_dim) if self.cfg.target_entropy == "auto"
            else float(self.cfg.target_entropy)
        )
        self._actor_loss_scale_mode = self.cfg.sac_actor_loss_scale_mode
        self._behavior_policy_kl_schedule = (
            self.cfg.outer_behavior_policy_kl_schedule if self.has_actor else "none"
        )
        self._behavior_policy_objective = self.cfg.outer_behavior_policy_objective
        self._actor_q_range_enabled = (
            self.actor_loss_scale_enabled
            or self._behavior_policy_kl_schedule == "quantile_gate"
        )
        self._actor_loss_scale_tau = float(self.cfg.sac_actor_loss_scale_tau)
        self.register_buffer("_actor_loss_scale_value", torch.ones(1, device=self.device))
        self.register_buffer(
            "_actor_loss_scale_percentiles", torch.tensor([5., 95.], device=self.device),
        )
        self.pi_optim = self.ent_coef_optim = self.behavior_policy_kl_optim = None
        self.log_ent_coef = self.log_behavior_policy_kl_coef = None
        self.behavior_policy_kl_eligible_updates = 0
        self.behavior_policy_kl_dual_updates = 0
        coefficient = self.cfg.ent_coef if self.has_actor else 0.0
        automatic = isinstance(coefficient, str) and coefficient.startswith("auto")
        self.register_buffer(
            "fixed_ent_coef", torch.tensor(
                float("nan") if automatic else float(coefficient), device=self.device,
            ),
        )
        optim_options = {"capturable": self.device.type == "cuda", "foreach": self.device.type == "cuda"}
        if self.has_actor:
            self.pi_optim = torch.optim.Adam(
                self.model._pi.parameters(), lr=float(self.cfg.actor_lr),
                eps=float(self.cfg.actor_adam_eps), **optim_options,
            )
            if automatic:
                initial = float(coefficient.split("_", 1)[1]) if "_" in coefficient else 1.0
                if initial <= 0:
                    raise ValueError("Auxiliary automatic entropy must start positive.")
                self.log_ent_coef = torch.nn.Parameter(
                    torch.tensor([max(initial, 1e-8)], device=self.device).log(),
                )
                self.ent_coef_optim = torch.optim.Adam(
                    [self.log_ent_coef], lr=float(self.cfg.ent_coef_lr),
                    eps=float(self.cfg.adam_eps), **optim_options,
                )
            elif float(coefficient) < 0:
                raise ValueError("Auxiliary fixed entropy coefficient must be nonnegative.")
            if self._behavior_policy_kl_schedule == "dual":
                self.log_behavior_policy_kl_coef = torch.nn.Parameter(
                    torch.tensor(
                        [float(self.cfg.outer_behavior_policy_kl_dual_init)],
                        dtype=torch.float64, device=self.device,
                    ).log(),
                )
                self.behavior_policy_kl_optim = torch.optim.Adam(
                    [self.log_behavior_policy_kl_coef],
                    lr=float(self.cfg.outer_behavior_policy_kl_dual_lr),
                    eps=float(self.cfg.adam_eps), **optim_options,
                )
        seed = int(getattr(owner.cfg, "seed", 0)) + 130363
        self._cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
        self._device_generator = (
            torch.Generator(device=self.device).manual_seed(seed + 1)
            if self.device.type == "cuda" else None
        )

    def __getattr__(self, name):
        # Share only these established actor primitives. Never fall back to the
        # primary agent's state, which would silently couple the two learners.
        if name in _SHARED_METHODS:
            descriptor = self._trainer_type.__dict__.get(name)
            if descriptor is None:
                descriptor = getattr(self._trainer_type, name)
            return descriptor.__get__(self, type(self))
        return super().__getattr__(name)

    @property
    def alpha(self):
        return self.fixed_ent_coef if self.log_ent_coef is None else self.log_ent_coef.exp().clamp_min(1e-8)

    @property
    def actor_loss_scale_enabled(self):
        return self._actor_loss_scale_mode == _SCALE_MODE

    @property
    def actor_loss_scale(self):
        return self._actor_loss_scale_value if self.actor_loss_scale_enabled else None

    @property
    def behavior_policy_kl_enabled(self):
        return self._behavior_policy_kl_schedule != "none"

    @contextmanager
    def isolated_rng(self):
        """Isolate samples, pair selection and critic dropout from primary SAC."""
        devices = (
            [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            if self.device.type == "cuda" else []
        )
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
            if self.has_actor:
                action, _ = self.model.pi(next_z)
            else:
                action = self.model._model.pi_action(next_z)
            value = self.model.Q(
                next_z, action, target=True,
                reduction=self.cfg.outer_q_target_reduction,
            )
            value = torch.where(terminated == 1, 0.0, value)
            return reward + self.discount * (1.0 - terminated) * value

    def critic_loss(self, latent_states, action, targets):
        latents = latent_states[:-1]
        if self.cfg.aux_return_detach_representation:
            latents = latents.detach()
        with self.isolated_rng():
            predictions = self.model._model.aux_return_q_predictions(latents, action)
        per_sample = self.model.q_backend.loss(predictions, targets, reduction="none")
        per_time = per_sample.mean(dim=(0,) + tuple(range(2, per_sample.ndim)))
        loss = td_math.reduce_temporal_loss(
            per_time, self.cfg.rho, legacy_order="vector_sum_divide",
        )
        return loss, predictions

    def update_actor(self, latent_states, **behavior):
        with self.isolated_rng():
            if self.has_actor:
                metrics = self._update_actor(latent_states.detach(), **behavior)
            else:
                metrics = {}
                if self._actor_q_range_enabled:
                    with torch.no_grad():
                        action = self.model._model.pi_action(latent_states[0].detach())
                        values = self.model.Q(
                            latent_states[0].detach(), action,
                            reduction=self.cfg.outer_q_actor_reduction,
                        )
                        self._update_actor_loss_scale(values)
                        metrics["actor_loss_scale"] = self.actor_loss_scale.detach()
        self.num_updates += 1
        return {f"aux_return_{key}": value for key, value in metrics.items()}

    def specification(self):
        return {
            "version": 1, "mode": self.mode, "target": "reward_only",
            "critic_coef": float(self.cfg.aux_return_critic_coef),
            "detach_representation": bool(self.cfg.aux_return_detach_representation),
            "critic_lr": float(self.cfg.aux_return_critic_lr),
            "critic": self.model.q_backend.signature.as_dict(),
            "codec": self.model.q_backend.codec,
            "target_reduction": self.cfg.outer_q_target_reduction,
            "actor": {key: getattr(self.cfg, key) for key in _ACTOR_SETTINGS},
        }

    def checkpoint_state(self):
        state = {"num_updates": self.num_updates, "rng_cpu": self._cpu_generator.get_state()}
        if self._device_generator is not None:
            state["rng_device"] = self._device_generator.get_state()
        if self.actor_loss_scale_enabled:
            state["q_range"] = self._actor_q_range_state()
        if self.has_actor:
            state["pi_optim"] = self.pi_optim.state_dict()
            if self.log_ent_coef is None:
                state["fixed_ent_coef"] = self.fixed_ent_coef.detach()
            else:
                state["log_ent_coef"] = self.log_ent_coef.detach()
                state["ent_coef_optim"] = self.ent_coef_optim.state_dict()
            if self.behavior_policy_kl_enabled:
                state["behavior"] = self._behavior_policy_kl_state()
        return state

    def preflight_state(self, payload, *, exact=False, expected_updates=None):
        keys = set(self.checkpoint_state())
        if not exact:
            # Portable checkpoints can move between CPU and CUDA. Only exact
            # resumes require the same device RNG inventory.
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
            if generator is None:
                value = require_tensor(state[key], f"Auxiliary return {key}", dtype=torch.uint8)
                if value.ndim != 1 or value.numel() == 0:
                    raise ValueError("Auxiliary return device RNG state must be a byte vector.")
                continue
            value = require_tensor(state[key], f"Auxiliary return {key}", shape=generator.get_state().shape, dtype=torch.uint8)
            probe = torch.Generator(device=generator.device)
            try:
                probe.set_state(value.cpu())
            except (RuntimeError, TypeError) as exc:
                raise ValueError(f"Auxiliary return {key} is invalid.") from exc
        if self.actor_loss_scale_enabled:
            self._preflight_actor_q_range_state(state["q_range"], "Auxiliary return Q scale")
        if self.has_actor:
            def optimizer(name, optimizer):
                if exact:
                    preflight_adam_state(optimizer, state[name], f"Auxiliary return {name}", expected_steps=updates)
                else:
                    preflight_optimizer_state(optimizer, state[name], f"Auxiliary return {name}")
                parameters = {
                    index: parameter
                    for saved, live in zip(state[name]["param_groups"], optimizer.param_groups)
                    for index, parameter in zip(saved["params"], live["params"])
                }
                for index, moments in state[name]["state"].items():
                    for field, value in moments.items():
                        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all().item()):
                            raise ValueError(f"Auxiliary return {name} {field} must be a finite tensor.")
                        if field != "step" and value.dtype != parameters[index].dtype:
                            raise ValueError(f"Auxiliary return {name} {field} dtype is incompatible.")
                        if field in {"exp_avg_sq", "max_exp_avg_sq"} and bool((value < 0).any().item()):
                            raise ValueError(f"Auxiliary return {name} second moment must be nonnegative.")
            optimizer("pi_optim", self.pi_optim)
            name = "log_ent_coef" if self.log_ent_coef is not None else "fixed_ent_coef"
            configured = getattr(self, name)
            value = require_tensor(state[name], f"Auxiliary return {name}", shape=configured.shape, dtype=configured.dtype)
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("Auxiliary return entropy coefficient must be finite.")
            if self.log_ent_coef is None:
                if bool((value < 0).any().item()) or (exact and not torch.equal(value.cpu(), configured.detach().cpu())):
                    raise ValueError("Auxiliary fixed entropy coefficient is incompatible.")
            else:
                if bool((value < math.log(1e-8)).any().item()):
                    raise ValueError("Auxiliary learned entropy coefficient is below its floor.")
                if not bool(torch.isfinite(value.exp()).all().item()):
                    raise ValueError("Auxiliary learned entropy coefficient must exponentiate to a finite value.")
                optimizer("ent_coef_optim", self.ent_coef_optim)
            if self.behavior_policy_kl_enabled:
                self._preflight_behavior_policy_kl(self._behavior_policy_kl_spec(), state["behavior"], exact=exact)
                counter = state["behavior"].get("eligible_updates", state["behavior"].get("dual_updates", 0))
                if counter > updates:
                    raise ValueError("Auxiliary behavior regularizer updates exceed actor updates.")
        return state

    @torch.no_grad()
    def load_state(self, state, *, exact=False):
        self.num_updates = state["num_updates"]
        self._cpu_generator.set_state(state["rng_cpu"].cpu())
        if self._device_generator is not None and "rng_device" in state:
            self._device_generator.set_state(state["rng_device"].cpu())
        if self.actor_loss_scale_enabled:
            self._actor_loss_scale_value.copy_(state["q_range"]["value"])
        if self.has_actor:
            load_optimizer_state_preserving_hyperparameters(self.pi_optim, state["pi_optim"])
            if self.log_ent_coef is not None:
                self.log_ent_coef.copy_(state["log_ent_coef"])
                load_optimizer_state_preserving_hyperparameters(self.ent_coef_optim, state["ent_coef_optim"])
            else:
                self.fixed_ent_coef.copy_(state["fixed_ent_coef"])
            if self._behavior_policy_kl_schedule == "smooth":
                self.behavior_policy_kl_eligible_updates = state["behavior"]["eligible_updates"]
            elif self._behavior_policy_kl_schedule == "quantile_gate" and not self.actor_loss_scale_enabled:
                self._actor_loss_scale_value.copy_(state["behavior"]["q_range"]["value"])
            elif self._behavior_policy_kl_schedule == "dual":
                behavior = state["behavior"]
                self.log_behavior_policy_kl_coef.copy_(behavior["log_coef"])
                load_optimizer_state_preserving_hyperparameters(self.behavior_policy_kl_optim, behavior["optim"])
                self.behavior_policy_kl_dual_updates = behavior["dual_updates"]
