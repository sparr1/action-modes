"""AMBI agent with TOLD priors and fresh per-root actor-critic adaptation."""

import math

import torch
import torch.nn.functional as F

from .common import math as td_math
from .common.compile_regions import CompileRegion
from .common.checkpoint import save_checkpoint
from .common.device import resolve_device
from .common.layers import api_model_conversion
from .common.scale import percentile_range
from .common.soft_world_model import SoftWorldModel
from .inner_improvement import InnerImprovementEngine, polyak_update
from .common.training_state import (
    load_optimizer_state_preserving_hyperparameters,
    preflight_adam_state,
    preflight_module_state,
    preflight_optimizer_state,
    require_exact_keys,
    require_tensor,
)


# Backward-compatible public test/debug hook.
_polyak_update = polyak_update

_ACTOR_LOSS_SCALE_MODE = "tdmpc2_percentile_range"
_ACTOR_LOSS_SCALE_PERCENTILES = (5.0, 95.0)
_ACTOR_LOSS_SCALE_FLOOR = 1.0
_BEHAVIOR_POLICY_KL_SCHEDULES = {
    "none",
    "smooth",
    "quantile_gate",
    "dual",
}
_BEHAVIOR_POLICY_OBJECTIVES = {
    "reverse_kl",
    "action_space_cross_entropy",
}
_BEHAVIOR_POLICY_KL_DUAL_MIN = 1e-8
_ENTROPY_COEF_MIN = 1e-8


class AMBITDMPC2Agent(torch.nn.Module):
    """Online TOLD/prior learner with canonical cloned inner SAC.

    Other inner operators and adaptation lifecycles are auxiliary ablations or
    comparison methods.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = resolve_device(getattr(cfg, "device", None))
        self.cfg.device = str(self.device)
        if self.device.type not in {"cpu", "cuda"}:
            raise NotImplementedError(
                "AMBI inner-loop RNG isolation currently supports CPU and CUDA devices only."
            )
        self.model = SoftWorldModel(cfg).to(self.device)
        if bool(getattr(cfg, "compile", False)):
            strict = bool(getattr(cfg, "compile_strict", False))
            self.model._Qs.enable_compile(strict=strict)
            self.model._target_Qs.enable_compile(strict=strict)
        self._outer_update_region = CompileRegion(
            "outer update",
            self._outer_update_kernel,
            enabled=bool(getattr(cfg, "compile", False)),
            strict=bool(getattr(cfg, "compile_strict", False)),
        )
        self.register_buffer(
            "_transition_temporal_weights",
            td_math.temporal_loss_weights(
                cfg.train_unroll_horizon,
                cfg.rho,
                normalization=cfg.temporal_loss_normalization,
                reference_horizon=cfg.temporal_loss_reference_horizon,
                device=self.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_actor_temporal_weights",
            td_math.temporal_loss_weights(
                cfg.train_unroll_horizon,
                cfg.rho,
                normalization=cfg.temporal_loss_normalization,
                reference_horizon=cfg.temporal_loss_reference_horizon,
                include_terminal=True,
                device=self.device,
            ),
            persistent=False,
        )
        self._actor_loss_scale_mode = str(
            getattr(cfg, "sac_actor_loss_scale_mode", "none")
        ).lower()
        self._behavior_policy_kl_schedule = str(
            getattr(cfg, "outer_behavior_policy_kl_schedule", "none")
        ).lower()
        self._behavior_policy_objective = str(
            getattr(cfg, "outer_behavior_policy_objective", "reverse_kl")
        ).lower()
        if self._behavior_policy_kl_schedule not in _BEHAVIOR_POLICY_KL_SCHEDULES:
            raise ValueError(
                "outer_behavior_policy_kl_schedule must be one of "
                f"{sorted(_BEHAVIOR_POLICY_KL_SCHEDULES)}."
            )
        if self._behavior_policy_objective not in _BEHAVIOR_POLICY_OBJECTIVES:
            raise ValueError(
                "outer_behavior_policy_objective must be one of "
                f"{sorted(_BEHAVIOR_POLICY_OBJECTIVES)}."
            )
        if (
            self._behavior_policy_objective == "action_space_cross_entropy"
            and self._behavior_policy_kl_schedule == "dual"
        ):
            raise ValueError(
                "outer_behavior_policy_objective='action_space_cross_entropy' "
                "does not support outer_behavior_policy_kl_schedule='dual'."
            )
        if self._actor_loss_scale_mode not in {
            "none",
            _ACTOR_LOSS_SCALE_MODE,
        }:
            raise ValueError(
                "sac_actor_loss_scale_mode must be 'none' or "
                f"{_ACTOR_LOSS_SCALE_MODE!r}."
            )
        self._actor_q_range_enabled = (
            self._actor_loss_scale_mode == _ACTOR_LOSS_SCALE_MODE
            or self._behavior_policy_kl_schedule == "quantile_gate"
        )
        if self._actor_q_range_enabled:
            scale_tau = float(getattr(cfg, "sac_actor_loss_scale_tau", 0.01))
            if not math.isfinite(scale_tau) or not 0.0 < scale_tau <= 1.0:
                raise ValueError("sac_actor_loss_scale_tau must be in (0, 1].")
            self._actor_loss_scale_tau = scale_tau
            self.register_buffer(
                "_actor_loss_scale_value",
                torch.ones(1, dtype=torch.float32, device=self.device),
            )
            self.register_buffer(
                "_actor_loss_scale_percentiles",
                torch.tensor(
                    _ACTOR_LOSS_SCALE_PERCENTILES,
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        else:
            self._actor_loss_scale_tau = None
            self._actor_loss_scale_value = None
            self._actor_loss_scale_percentiles = None
        self._world_critic_params = (
            list(self.model._encoder.parameters())
            + list(self.model._dynamics.parameters())
            + list(self.model._reward.parameters())
            + (list(self.model._termination.parameters()) if cfg.episodic else [])
            + list(self.model._Qs.parameters())
        )
        optimizer_groups = [
            {
                "params": self.model._encoder.parameters(),
                "lr": float(cfg.lr) * float(cfg.enc_lr_scale),
            },
            {"params": self.model._dynamics.parameters()},
            {"params": self.model._reward.parameters()},
            {
                "params": self.model._Qs.parameters(),
                "lr": float(getattr(cfg, "critic_lr", cfg.lr)),
            },
        ]
        if cfg.episodic:
            optimizer_groups.append({"params": self.model._termination.parameters()})
        self.optim = torch.optim.Adam(
            optimizer_groups,
            lr=float(cfg.lr),
            eps=float(getattr(cfg, "adam_eps", 1e-8)),
            capturable=self.device.type == "cuda",
            foreach=self.device.type == "cuda",
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(),
            lr=float(getattr(cfg, "actor_lr", cfg.lr)),
            eps=float(cfg.actor_adam_eps),
            capturable=self.device.type == "cuda",
            foreach=self.device.type == "cuda",
        )

        self.target_entropy = self._target_entropy(
            getattr(cfg, "target_entropy", "auto")
        )
        self.log_ent_coef = None
        self.ent_coef_optim = None
        ent_coef = getattr(cfg, "ent_coef", "auto")
        if isinstance(ent_coef, str) and ent_coef.startswith("auto"):
            initial = 1.0
            if "_" in ent_coef:
                initial = float(ent_coef.split("_", 1)[1])
            if initial <= 0:
                raise ValueError("Initial automatic entropy coefficient must be positive.")
            initial = max(initial, _ENTROPY_COEF_MIN)
            self.log_ent_coef = torch.nn.Parameter(
                torch.log(torch.tensor([initial], dtype=torch.float32, device=self.device))
            )
            self.ent_coef_optim = torch.optim.Adam(
                [self.log_ent_coef],
                lr=float(getattr(cfg, "ent_coef_lr", getattr(cfg, "actor_lr", cfg.lr))),
                eps=float(getattr(cfg, "adam_eps", 1e-8)),
                capturable=self.device.type == "cuda",
                foreach=self.device.type == "cuda",
            )
            self.register_buffer(
                "fixed_ent_coef", torch.tensor(float("nan"), device=self.device)
            )
        else:
            fixed = float(ent_coef)
            if fixed <= 0:
                raise ValueError("Entropy coefficient must be positive.")
            self.register_buffer(
                "fixed_ent_coef", torch.tensor(fixed, device=self.device)
            )

        self.behavior_policy_kl_eligible_updates = 0
        self.behavior_policy_kl_dual_updates = 0
        self.log_behavior_policy_kl_coef = None
        self.behavior_policy_kl_optim = None
        if self._behavior_policy_kl_schedule == "dual":
            initial_kl_coef = float(
                getattr(cfg, "outer_behavior_policy_kl_dual_init", 0.1)
            )
            self.log_behavior_policy_kl_coef = torch.nn.Parameter(
                torch.log(
                    torch.tensor(
                        [initial_kl_coef],
                        # Only this scalar and its Adam moments need the wider
                        # range: squaring a large finite KL violation can
                        # overflow float32 and permanently poison Adam state.
                        dtype=torch.float64,
                        device=self.device,
                    )
                )
            )
            self.behavior_policy_kl_optim = torch.optim.Adam(
                [self.log_behavior_policy_kl_coef],
                lr=float(getattr(cfg, "outer_behavior_policy_kl_dual_lr", 3e-4)),
                eps=float(getattr(cfg, "adam_eps", 1e-8)),
                capturable=self.device.type == "cuda",
                foreach=self.device.type == "cuda",
            )

        self.discount = self._get_discount(cfg.episode_length)
        self.num_updates = 0
        self.outer_version = 0
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        self.inner_engine = InnerImprovementEngine(self)
        self._resume_boundary_prepared = False
        self.model.eval()

        print("Episode length:", cfg.episode_length)
        print("Discount factor:", self.discount)
        print("Inner operator:", cfg.inner_operator)
        critic_spec = self.model.critic_signature
        if critic_spec["q_representation"] == "distributional":
            critic_detail = (
                f"heads={critic_spec['num_q']}, bins={critic_spec['q_num_bins']}, "
                f"support=[{critic_spec['q_vmin']:g}, {critic_spec['q_vmax']:g}], "
                f"pair={cfg.q_pair_size}"
            )
        else:
            critic_detail = (
                f"heads={critic_spec['num_q']}, scalar, pair={cfg.q_pair_size}"
            )
        print(
            "Q critic:",
            f"{critic_spec['q_representation']} ({critic_detail})",
        )
        print(
            "Critic targets:",
            f"outer={cfg.outer_critic_target}, "
            f"inner_sac={cfg.inner_sac_critic_target}",
        )
        if cfg.inner_operator in {"sac", "td3"}:
            nominal_steps = (
                int(cfg.inner_rounds)
                * int(cfg.inner_rollouts_per_round)
                * int(cfg.inner_rollout_horizon)
            )
            if cfg.inner_component_update_schedule:
                update_schedule = (
                    f"C={cfg.inner_critic_updates_per_round}, "
                    f"A={cfg.inner_actor_updates_per_round}"
                )
            else:
                update_schedule = f"G={cfg.inner_updates_per_round}"
            print(
                "Inner schedule:",
                f"J={cfg.inner_rounds}, N={cfg.inner_rollouts_per_round}, "
                f"H={cfg.inner_rollout_horizon}, {update_schedule}, "
                f"nominal_transitions={nominal_steps}, "
                f"expected_update_slots={cfg.inner_expected_update_slots}, "
                f"critic_utd={cfg.inner_nominal_critic_utd:g}",
            )
        elif cfg.inner_operator == "mppi":
            print(
                "Inner MPPI:",
                f"iterations={cfg.inner_mppi_iterations}, "
                f"samples={cfg.inner_mppi_num_samples}, "
                f"H={cfg.inner_rollout_horizon}",
            )

    def _get_discount(self, episode_length):
        fraction = float(episode_length) / float(self.cfg.discount_denom)
        return min(
            max((fraction - 1.0) / fraction, self.cfg.discount_min),
            self.cfg.discount_max,
        )

    def _target_entropy(self, target_entropy):
        if target_entropy == "auto":
            return float(-self.cfg.action_dim)
        return float(target_entropy)

    @property
    def alpha(self):
        if self.log_ent_coef is not None:
            return self.log_ent_coef.exp().clamp_min(_ENTROPY_COEF_MIN)
        return self.fixed_ent_coef

    @property
    def actor_loss_scale_enabled(self):
        return self._actor_loss_scale_mode == _ACTOR_LOSS_SCALE_MODE

    @property
    def actor_loss_scale(self):
        if not self.actor_loss_scale_enabled:
            return None
        return self._actor_loss_scale_value

    @property
    def behavior_policy_kl_enabled(self):
        return self._behavior_policy_kl_schedule != "none"

    @property
    def behavior_policy_objective(self):
        return self._behavior_policy_objective

    @property
    def actor_q_range(self):
        if not self._actor_q_range_enabled:
            return None
        return self._actor_loss_scale_value

    def _checkpoint_version(self):
        if self.behavior_policy_kl_enabled:
            return 6 if self.actor_loss_scale_enabled else 5
        return 4 if self.actor_loss_scale_enabled else 3

    def _actor_loss_scale_spec(self):
        if not self.actor_loss_scale_enabled:
            return None
        return {
            "mode": _ACTOR_LOSS_SCALE_MODE,
            "application": "full_sac_actor_objective",
            "source": "decoded_outer_actor_q_depth0",
            "reduction": str(self.cfg.outer_q_actor_reduction),
            "percentiles": list(_ACTOR_LOSS_SCALE_PERCENTILES),
            "tau": self._actor_loss_scale_tau,
            "floor": _ACTOR_LOSS_SCALE_FLOOR,
        }

    def _actor_q_range_spec(self):
        if not self._actor_q_range_enabled:
            return None
        return {
            "source": "decoded_outer_actor_q_depth0",
            "reduction": str(self.cfg.outer_q_actor_reduction),
            "percentiles": list(_ACTOR_LOSS_SCALE_PERCENTILES),
            "tau": self._actor_loss_scale_tau,
            "floor": _ACTOR_LOSS_SCALE_FLOOR,
        }

    def _actor_loss_scale_state(self):
        if not self.actor_loss_scale_enabled:
            return None
        return self._actor_q_range_state()

    def _actor_q_range_state(self):
        if not self._actor_q_range_enabled:
            return None
        return {
            "value": self._actor_loss_scale_value.detach(),
            "percentiles": self._actor_loss_scale_percentiles.detach(),
        }

    def _preflight_actor_loss_scale(self, spec, state):
        if not self.actor_loss_scale_enabled:
            raise ValueError(
                "Checkpoint actor-loss scaling is enabled but the configured agent "
                "has sac_actor_loss_scale_mode='none'."
            )
        expected_spec = self._actor_loss_scale_spec()
        spec = require_exact_keys(
            spec, expected_spec.keys(), "AMBI actor-loss scale specification"
        )
        if dict(spec) != expected_spec:
            raise ValueError(
                "Checkpoint actor-loss scale specification does not match this agent: "
                f"checkpoint={dict(spec)}, configured={expected_spec}."
            )
        return self._preflight_actor_q_range_state(
            state, "AMBI actor-loss scale state"
        )

    def _preflight_actor_q_range_state(self, state, label):
        if not self._actor_q_range_enabled:
            raise ValueError(f"{label} is present but Q-range tracking is disabled.")
        state = require_exact_keys(state, {"value", "percentiles"}, label)
        value = require_tensor(
            state["value"],
            f"{label} value",
            shape=self._actor_loss_scale_value.shape,
            dtype=self._actor_loss_scale_value.dtype,
        )
        percentiles = require_tensor(
            state["percentiles"],
            f"{label} percentiles",
            shape=self._actor_loss_scale_percentiles.shape,
            dtype=self._actor_loss_scale_percentiles.dtype,
        )
        if not bool(torch.isfinite(value).all().item()) or bool(
            (value < _ACTOR_LOSS_SCALE_FLOOR).any().item()
        ):
            raise ValueError(
                "AMBI actor-loss scale value must be finite and at least its floor."
            )
        if not torch.equal(
            percentiles.detach().cpu(),
            self._actor_loss_scale_percentiles.detach().cpu(),
        ):
            raise ValueError(
                "AMBI actor-loss scale percentiles differ from the configured policy."
            )
        return state

    def _behavior_policy_kl_spec(self):
        if not self.behavior_policy_kl_enabled:
            return None
        spec = {
            "schedule": self._behavior_policy_kl_schedule,
            "direction": "current_outer_to_replayed_behavior",
            "estimator": "analytic_diagonal_gaussian_jensen_component",
            "action_transform": "shared_tanh",
            "reduction": "per_action_dim_valid_weighted_temporal",
            "min_valid_count": int(
                self.cfg.outer_behavior_policy_kl_min_valid_count
            ),
            "coef": float(self.cfg.outer_behavior_policy_kl_coef),
            "ramp_updates": int(
                self.cfg.outer_behavior_policy_kl_ramp_updates
            ),
            "q_threshold": float(
                self.cfg.outer_behavior_policy_kl_q_threshold
            ),
            "q_range": (
                self._actor_q_range_spec()
                if self._behavior_policy_kl_schedule == "quantile_gate"
                else None
            ),
            "target": float(self.cfg.outer_behavior_policy_kl_target),
            "dual_init": float(
                self.cfg.outer_behavior_policy_kl_dual_init
            ),
            "dual_lr": float(self.cfg.outer_behavior_policy_kl_dual_lr),
            "dual_min": _BEHAVIOR_POLICY_KL_DUAL_MIN,
            "dual_max": float(self.cfg.outer_behavior_policy_kl_dual_max),
            "actor_loss_scaling": (
                "full_objective" if self.actor_loss_scale_enabled else "none"
            ),
        }
        if self._behavior_policy_objective == "reverse_kl":
            # Preserve the exact version-5/6 reverse-KL specification so
            # existing structured checkpoints remain loadable.
            return spec
        return {
            **spec,
            "objective": "action_space_cross_entropy",
            "direction": "current_outer_samples_under_replayed_behavior",
            "estimator": (
                "partially_analytic_single_sample_exact_action_space_"
                "cross_entropy_jensen_component"
            ),
            "action_transform": "normalized_tanh_exact_log_abs_det_jacobian",
        }

    def _behavior_policy_kl_state(self):
        schedule = self._behavior_policy_kl_schedule
        if schedule == "smooth":
            return {
                "eligible_updates": int(
                    self.behavior_policy_kl_eligible_updates
                )
            }
        if schedule == "quantile_gate":
            if self.actor_loss_scale_enabled:
                return {}
            return {"q_range": self._actor_q_range_state()}
        if schedule == "dual":
            return {
                "log_coef": self.log_behavior_policy_kl_coef.detach(),
                "optim": self.behavior_policy_kl_optim.state_dict(),
                "dual_updates": int(self.behavior_policy_kl_dual_updates),
            }
        return None

    def _preflight_behavior_policy_kl(self, spec, state, *, exact=False):
        if not self.behavior_policy_kl_enabled:
            raise ValueError(
                "Checkpoint behavior-policy KL is enabled but the configured "
                "schedule is 'none'."
            )
        expected_spec = self._behavior_policy_kl_spec()
        spec = require_exact_keys(
            spec,
            expected_spec.keys(),
            "AMBI behavior-policy KL specification",
        )
        if dict(spec) != expected_spec:
            raise ValueError(
                "Checkpoint behavior-policy KL specification does not match "
                f"this agent: checkpoint={dict(spec)}, configured={expected_spec}."
            )
        schedule = self._behavior_policy_kl_schedule
        if schedule == "smooth":
            state = require_exact_keys(
                state,
                {"eligible_updates"},
                "AMBI behavior-policy KL smooth state",
            )
            value = state["eligible_updates"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "AMBI behavior-policy KL eligible_updates must be a "
                    "non-negative integer."
                )
        elif schedule == "quantile_gate":
            expected_keys = set() if self.actor_loss_scale_enabled else {"q_range"}
            state = require_exact_keys(
                state,
                expected_keys,
                "AMBI behavior-policy KL quantile state",
            )
            if not self.actor_loss_scale_enabled:
                self._preflight_actor_q_range_state(
                    state["q_range"], "AMBI behavior-policy KL Q-range state"
                )
        elif schedule == "dual":
            state = require_exact_keys(
                state,
                {"log_coef", "optim", "dual_updates"},
                "AMBI behavior-policy KL dual state",
            )
            dual_updates = state["dual_updates"]
            if (
                isinstance(dual_updates, bool)
                or not isinstance(dual_updates, int)
                or dual_updates < 0
            ):
                raise ValueError(
                    "AMBI behavior-policy KL dual_updates must be a "
                    "non-negative integer."
                )
            log_coef = require_tensor(
                state["log_coef"],
                "AMBI behavior-policy KL log coefficient",
                shape=self.log_behavior_policy_kl_coef.shape,
                dtype=self.log_behavior_policy_kl_coef.dtype if exact else None,
            )
            if log_coef.dtype not in {torch.float32, torch.float64}:
                raise ValueError(
                    "AMBI behavior-policy KL log coefficient must have "
                    "float32 (legacy portable) or float64 dtype."
                )
            if not bool(torch.isfinite(log_coef).all().item()):
                raise ValueError(
                    "AMBI behavior-policy KL log coefficient must be finite."
                )
            minimum = math.log(_BEHAVIOR_POLICY_KL_DUAL_MIN)
            maximum = math.log(float(self.cfg.outer_behavior_policy_kl_dual_max))
            if bool(((log_coef < minimum) | (log_coef > maximum)).any().item()):
                raise ValueError(
                    "AMBI behavior-policy KL log coefficient is outside its "
                    "configured bounds."
                )
            if exact:
                preflight_adam_state(
                    self.behavior_policy_kl_optim,
                    state["optim"],
                    "AMBI behavior-policy KL optimizer",
                    expected_steps=dual_updates,
                )
            else:
                self._preflight_optimizer(
                    "behavior_policy_kl_optim",
                    state["optim"],
                    self.behavior_policy_kl_optim,
                )
            # Portable loads may promote legacy float32 scalar/moment state.
            # Widening an already overflowed moment cannot recover its value;
            # reject it before changing any live model or optimizer state.
            for parameter_state in state["optim"]["state"].values():
                for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if field not in parameter_state:
                        continue
                    moment = parameter_state[field]
                    if moment.dtype != log_coef.dtype:
                        raise ValueError(
                            f"AMBI behavior-policy KL optimizer {field} dtype "
                            "must match the saved log coefficient."
                        )
                    if not bool(torch.isfinite(moment).all().item()):
                        raise ValueError(
                            f"AMBI behavior-policy KL optimizer {field} must "
                            "be finite; overflowed moments cannot be recovered "
                            "by precision promotion."
                        )
        return state

    @torch.no_grad()
    def _reset_behavior_policy_kl_state(self):
        self.behavior_policy_kl_eligible_updates = 0
        self.behavior_policy_kl_dual_updates = 0
        if self.log_behavior_policy_kl_coef is not None:
            initial = float(self.cfg.outer_behavior_policy_kl_dual_init)
            self.log_behavior_policy_kl_coef.fill_(math.log(initial))
            self.behavior_policy_kl_optim.state.clear()
        if (
            self._behavior_policy_kl_schedule == "quantile_gate"
            and not self.actor_loss_scale_enabled
        ):
            self._actor_loss_scale_value.fill_(_ACTOR_LOSS_SCALE_FLOOR)

    @torch.no_grad()
    def _update_actor_loss_scale(self, q_values):
        if not self._actor_q_range_enabled:
            return None
        value = percentile_range(
            q_values.detach(),
            self._actor_loss_scale_percentiles,
            minimum=_ACTOR_LOSS_SCALE_FLOOR,
        )
        self._actor_loss_scale_value.lerp_(value, self._actor_loss_scale_tau)
        return value

    def set_outer_replay_buffer(self, buffer):
        """Expose current real replay to optional critic-only inner sampling."""
        self.inner_engine.outer_replay_buffer = buffer

    def reset(self):
        if self._resume_boundary_prepared:
            # A full checkpoint already advanced the inner episode lifecycle.
            # The first environment reset after that checkpoint must not
            # expire/increment it a second time, in either the source process
            # or a resumed process.
            self._resume_boundary_prepared = False
            return
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        self.inner_engine.reset_episode()

    def prepare_training_resume_boundary(self):
        """Expire episode state once, deferring the actual environment reset."""
        if not self._resume_boundary_prepared:
            self.last_inner_metrics = {}
            self.last_inner_rollout_lengths = []
            self.inner_engine.prepare_training_resume_boundary()
            self._resume_boundary_prepared = True
        return self

    def observation_signature(self):
        """Return the portable observation contract used by this checkpoint."""
        mode = str(self.cfg.obs)
        shape = tuple(int(value) for value in self.cfg.obs_shape[mode])
        dtype = getattr(
            self.cfg,
            "obs_dtype",
            "uint8" if mode == "rgb" else "float32",
        )
        return {
            "mode": mode,
            "shape": list(shape),
            "dtype": str(dtype),
        }

    def _policy_spec(self):
        return {
            "log_std_mapping": str(self.cfg.log_std_mapping),
            "log_std_min": float(self.cfg.log_std_min),
            "log_std_max": float(self.cfg.log_std_max),
        }

    @staticmethod
    def _normalize_saved_policy_spec(policy_spec):
        """Normalize policy metadata written before log-std mappings existed."""
        if not isinstance(policy_spec, dict):
            return policy_spec
        legacy_keys = {"log_std_min", "log_std_max"}
        current_keys = legacy_keys | {"log_std_mapping"}
        keys = set(policy_spec)
        if keys == legacy_keys:
            # Direct clamping was the sole policy mapping before this field was
            # added to version-3/4 checkpoints.
            policy_spec = {**policy_spec, "log_std_mapping": "direct_clamp"}
        elif keys != current_keys:
            return policy_spec
        else:
            policy_spec = dict(policy_spec)
        mapping = policy_spec["log_std_mapping"]
        if isinstance(mapping, str):
            policy_spec["log_std_mapping"] = mapping.lower()
        return policy_spec

    def checkpoint_state(self):
        """Return the portable checkpoint structure over live outer state."""
        state = {
            "checkpoint_version": self._checkpoint_version(),
            "observation_spec": self.observation_signature(),
            "critic_spec": self.model.critic_signature,
            "policy_spec": self._policy_spec(),
            "entropy_spec": {
                "mode": "auto" if self.log_ent_coef is not None else "fixed",
                "target_entropy": float(self.target_entropy),
            },
            "critic_target_spec": self._critic_target_spec(),
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "pi_optim": self.pi_optim.state_dict(),
            "num_updates": self.num_updates,
            "outer_version": self.outer_version,
        }
        if self.log_ent_coef is not None:
            state["log_ent_coef"] = self.log_ent_coef.detach()
            state["ent_coef_optim"] = self.ent_coef_optim.state_dict()
        else:
            state["fixed_ent_coef"] = self.fixed_ent_coef.detach()
        if self.actor_loss_scale_enabled:
            state["actor_loss_scale_spec"] = self._actor_loss_scale_spec()
            state["actor_loss_scale_state"] = self._actor_loss_scale_state()
        if self.behavior_policy_kl_enabled:
            state["behavior_policy_kl_spec"] = self._behavior_policy_kl_spec()
            state["behavior_policy_kl_state"] = self._behavior_policy_kl_state()
        return state

    def _critic_target_spec(self):
        """Return Bellman-target semantics recorded for reproducibility."""
        spec = {
            "outer_critic_target": str(self.cfg.outer_critic_target),
            "inner_sac_critic_target": str(self.cfg.inner_sac_critic_target),
        }
        # Preserve the feature-off checkpoint contract byte-for-byte. Active
        # populations add their complete resolved Bellman/execution identity;
        # their absence itself denotes the legacy single-policy controller.
        if str(getattr(self.cfg, "inner_explorer_mode", "none")) != "none":
            spec["inner_population"] = self._inner_population_spec()
        options = {
            "finite_horizon": bool(getattr(self.cfg, "inner_finite_horizon", False)),
            "steps_per_update": getattr(self.cfg, "inner_steps_per_update", None),
            "outer_replay_fraction": float(getattr(self.cfg, "inner_outer_replay_fraction", 0.0)),
        }
        if any(options.values()):
            if options["finite_horizon"]:
                options["horizon"] = int(self.cfg.inner_rollout_horizon)
                options["terminal_q_reduction"] = self.cfg.mppi_terminal_q_reduction
            spec["inner_solve"] = options
        return spec

    def _inner_population_spec(self):
        """Return resolved two-policy inner-control semantics.

        The auxiliary policy is action-local and therefore is not part of a
        portable model checkpoint.  Its algorithm and resolved compute dose
        are nevertheless scientific state: exact continuation must reject a
        checkpoint produced under different Bellman or execution semantics.
        """

        cfg = self.cfg
        mode = str(getattr(cfg, "inner_explorer_mode", "none"))
        spec = {
            "mode": mode,
            "prior_rollout_weight": float(
                getattr(cfg, "inner_prior_rollout_weight", 0.5)
            ),
            "mixture_target_estimator": str(
                getattr(cfg, "inner_mixture_target_estimator", "stratified")
            ),
            "primary_rollouts_per_round": int(
                getattr(cfg, "inner_primary_rollouts_per_round", 0)
            ),
            "explorer_rollouts_per_round": int(
                getattr(cfg, "inner_explorer_rollouts_per_round", 0)
            ),
            "primary_target_rows_per_batch": getattr(
                cfg, "inner_primary_target_rows_per_batch", None
            ),
            "explorer_target_rows_per_batch": getattr(
                cfg, "inner_explorer_target_rows_per_batch", None
            ),
            "rounds": int(getattr(cfg, "inner_rounds", 0)),
            "batch_size": int(getattr(cfg, "inner_batch_size", 0)),
            "component_update_schedule": bool(
                getattr(cfg, "inner_component_update_schedule", False)
            ),
            "nominal_updates_per_round": int(
                getattr(cfg, "inner_nominal_updates_per_round", 0)
            ),
            "expected_update_slots": int(
                getattr(cfg, "inner_expected_update_slots", 0)
            ),
            "primary_actor_updates_per_round": int(
                getattr(cfg, "inner_primary_actor_updates_per_round", 0) or 0
            ),
            "primary_critic_updates_per_round": int(
                getattr(cfg, "inner_primary_critic_updates_per_round", 0) or 0
            ),
            "primary_temperature_updates_per_round": int(
                getattr(cfg, "inner_primary_temperature_updates_per_round", 0)
                or 0
            ),
            "explorer_actor_updates_per_round": int(
                getattr(cfg, "inner_explorer_actor_updates_per_round", 0) or 0
            ),
            "explorer_critic_updates_per_round": int(
                getattr(cfg, "inner_explorer_critic_updates_per_round", 0) or 0
            ),
            "explorer_temperature_updates_per_round": int(
                getattr(cfg, "inner_explorer_temperature_updates_per_round", 0)
                or 0
            ),
            "primary_optimizer_steps_per_action": int(
                getattr(cfg, "inner_primary_optimizer_steps_per_action", 0)
            ),
            "explorer_optimizer_steps_per_action": int(
                getattr(cfg, "inner_explorer_optimizer_steps_per_action", 0)
            ),
            "total_optimizer_steps_per_action": int(
                getattr(cfg, "inner_total_optimizer_steps_per_action", 0)
            ),
            "execution_policy_source": str(
                getattr(cfg, "inner_execution_policy_source", "primary")
            ),
            "execution_handoff_samples": int(
                getattr(cfg, "inner_execution_handoff_samples", 8)
            ),
            "primary_updates_per_round_is_auto": bool(
                getattr(cfg, "inner_primary_updates_per_round_is_auto", False)
            ),
            "explorer_actor_updates_inherit_primary": bool(
                getattr(
                    cfg,
                    "inner_explorer_actor_updates_inherit_primary",
                    False,
                )
            ),
            "explorer_critic_updates_inherit_primary": bool(
                getattr(
                    cfg,
                    "inner_explorer_critic_updates_inherit_primary",
                    False,
                )
            ),
            "explorer_temperature_updates_inherit_primary": bool(
                getattr(
                    cfg,
                    "inner_explorer_temperature_updates_inherit_primary",
                    False,
                )
            ),
        }
        # Preserve every existing random-explorer specification exactly.
        # Adaptive parameter noise adds its action-local population and fixed
        # calibration semantics only when that mode is active.
        if mode == "adaptive_param_noise":
            spec["parameter_noise"] = {
                "actor_count": int(cfg.inner_param_noise_actor_count),
                "rollouts_per_actor": int(
                    cfg.inner_param_noise_rollouts_per_actor
                ),
                "target_action_rms": float(
                    cfg.inner_param_noise_target_action_rms
                ),
                "sigma_init": float(cfg.inner_param_noise_sigma_init),
                "sigma_min": float(cfg.inner_param_noise_sigma_min),
                "sigma_max": float(cfg.inner_param_noise_sigma_max),
                "calibration_directions": int(
                    cfg.inner_param_noise_calibration_directions
                ),
                "calibration_batch_size": int(
                    cfg.inner_param_noise_calibration_batch_size
                ),
                "calibration_max_probes": int(
                    cfg.inner_param_noise_calibration_max_probes
                ),
                "behavior_action": str(cfg.inner_behavior_action),
                "behavior_std_scale": float(cfg.inner_behavior_std_scale),
                "perturbed_policy_output": "mean_only",
                "behavior_log_std_source": "clean_actor",
                "clean_log_std_mapping": str(cfg.inner_log_std_mapping),
                "clean_log_std_min": float(cfg.inner_log_std_min),
                "clean_log_std_max": float(cfg.inner_log_std_max),
                "reset_per_action": True,
                "recalibrate_per_round": True,
                "calibration_relative_tolerance": 0.10,
                "calibration_log_error_exponent": 0.5,
                "calibration_update_ratio_min": 0.5,
                "calibration_update_ratio_max": 2.0,
            }
        return spec

    def _normalize_saved_critic_target_spec(self, spec):
        """Upgrade exact states written before inner populations existed.

        Legacy states are valid only for the disabled population mode.  In
        that case the new settings are inert, so canonicalize them to this
        agent's resolved disabled specification.  Active modes deliberately
        receive no migration and fail the ordinary exact-spec comparison.
        """

        if not isinstance(spec, dict):
            return spec
        legacy_keys = {"outer_critic_target", "inner_sac_critic_target"}
        if (
            set(spec) == legacy_keys
            and str(getattr(self.cfg, "inner_explorer_mode", "none")) == "none"
        ):
            return spec
        return spec

    def training_state_dict(self):
        """Return exact outer and persistent inner state for run continuation."""
        if self.outer_version != self.num_updates:
            raise RuntimeError(
                "AMBI live outer state is inconsistent: "
                "outer_version must equal num_updates."
            )
        inner = self.inner_engine.training_state_dict()
        inner_outer_version = inner["workspace"]["counters"]["outer_version"]
        if inner_outer_version > self.outer_version:
            raise RuntimeError(
                "AMBI live persistent inner workspace is newer than the outer learner."
            )
        # Versions 3/4 preserve the feature-disabled checkpoint contract.
        # Versions 5/6 add behavior-policy KL state, with even versions also
        # carrying actor-loss scaling state.
        return {
            "schema": "ambi-tdmpc2-agent-training-state",
            "version": self._checkpoint_version(),
            "outer": self.checkpoint_state(),
            "inner": inner,
            "model_training": bool(self.model.training),
            "boundary_prepared": bool(self._resume_boundary_prepared),
        }

    def _preflight_outer_training_state(self, state):
        expected_keys = {
            "checkpoint_version",
            "observation_spec",
            "critic_spec",
            "policy_spec",
            "entropy_spec",
            "critic_target_spec",
            "model",
            "optim",
            "pi_optim",
            "num_updates",
            "outer_version",
        }
        if self.log_ent_coef is not None:
            expected_keys.update({"log_ent_coef", "ent_coef_optim"})
        else:
            expected_keys.add("fixed_ent_coef")
        if self.actor_loss_scale_enabled:
            expected_keys.update(
                {"actor_loss_scale_spec", "actor_loss_scale_state"}
            )
        if self.behavior_policy_kl_enabled:
            expected_keys.update(
                {"behavior_policy_kl_spec", "behavior_policy_kl_state"}
            )
        state = require_exact_keys(state, expected_keys, "AMBI outer training state")
        expected_checkpoint_version = self._checkpoint_version()
        if state["checkpoint_version"] != expected_checkpoint_version:
            raise ValueError(
                "AMBI exact training state requires outer checkpoint version "
                f"{expected_checkpoint_version}."
            )
        saved_target_spec = self._normalize_saved_critic_target_spec(
            state["critic_target_spec"]
        )
        if saved_target_spec != self._critic_target_spec():
            raise ValueError(
                "AMBI exact training-state critic-target specification does not "
                "match this agent: "
                f"checkpoint={saved_target_spec}, "
                f"configured={self._critic_target_spec()}."
            )
        for key in ("num_updates", "outer_version"):
            value = state[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"AMBI outer {key} must be a non-negative integer.")
        if state["outer_version"] != state["num_updates"]:
            raise ValueError(
                "AMBI outer exact training state requires "
                "outer_version == num_updates."
            )
        preflight_module_state(self.model, state["model"], "AMBI outer model")
        preflight_adam_state(
            self.optim,
            state["optim"],
            "AMBI outer optimizer",
            expected_steps=state["num_updates"],
        )
        preflight_adam_state(
            self.pi_optim,
            state["pi_optim"],
            "AMBI outer policy optimizer",
            expected_steps=state["num_updates"],
        )
        if self.ent_coef_optim is not None:
            preflight_adam_state(
                self.ent_coef_optim,
                state["ent_coef_optim"],
                "AMBI outer entropy optimizer",
                expected_steps=state["num_updates"],
            )
            require_tensor(
                state["log_ent_coef"],
                "AMBI outer log_ent_coef",
                shape=self.log_ent_coef.shape,
                dtype=self.log_ent_coef.dtype,
            )
        else:
            fixed = require_tensor(
                state["fixed_ent_coef"],
                "AMBI outer fixed_ent_coef",
                shape=self.fixed_ent_coef.shape,
                dtype=self.fixed_ent_coef.dtype,
            )
            if not torch.equal(
                fixed.detach().cpu(), self.fixed_ent_coef.detach().cpu()
            ):
                raise ValueError(
                    "AMBI outer fixed entropy coefficient differs from configuration."
                )
        if self.actor_loss_scale_enabled:
            self._preflight_actor_loss_scale(
                state["actor_loss_scale_spec"], state["actor_loss_scale_state"]
            )
        if self.behavior_policy_kl_enabled:
            self._preflight_behavior_policy_kl(
                state["behavior_policy_kl_spec"],
                state["behavior_policy_kl_state"],
                exact=True,
            )
        # The lineage fingerprint checks every resolved scientific setting.
        # The established model-checkpoint loader below remains the single
        # authority for signature, entropy, and optimizer-layout validation.
        return state

    def _preflight_training_state_dict(self, state):
        state = require_exact_keys(
            state,
            {
                "schema",
                "version",
                "outer",
                "inner",
                "model_training",
                "boundary_prepared",
            },
            "AMBI agent training state",
        )
        expected_version = self._checkpoint_version()
        if (
            state["schema"] != "ambi-tdmpc2-agent-training-state"
            or state["version"] != expected_version
        ):
            raise ValueError("Unsupported AMBI agent training-state version.")
        outer = self._preflight_outer_training_state(state["outer"])
        inner_candidate = self.inner_engine._preflight_training_state_dict(
            state["inner"]
        )
        inner_outer_version = inner_candidate["state"].outer_version
        if inner_outer_version > outer["outer_version"]:
            raise ValueError(
                "AMBI persistent inner workspace cannot be newer than the saved "
                "outer learner: "
                f"inner outer_version={inner_outer_version}, "
                f"outer_version={outer['outer_version']}."
            )
        if not isinstance(state["model_training"], bool):
            raise TypeError("AMBI model_training must be bool.")
        if not isinstance(state["boundary_prepared"], bool):
            raise TypeError("AMBI boundary_prepared must be bool.")
        return state, outer, inner_candidate

    def load_training_state_dict(self, state):
        """Strictly restore outer state and every run-persistent inner component."""
        candidate = self._preflight_training_state_dict(state)
        return self._commit_training_state_candidate(candidate)

    def _commit_training_state_candidate(self, candidate):
        """Install a candidate returned by strict exact-state preflight."""
        state, outer, inner_candidate = candidate
        # ``load`` retains the established model-checkpoint implementation and
        # clears old inner allocations. Exact preflight above makes its input a
        # complete versioned state rather than a permissive transfer checkpoint.
        self.load(outer)
        self.inner_engine._commit_training_state_candidate(inner_candidate)
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        self.model.train(state["model_training"])
        self._resume_boundary_prepared = state["boundary_prepared"]
        return self

    def save(self, fp):
        save_checkpoint(self.checkpoint_state(), fp)

    @staticmethod
    def _optimizer_layout(optimizer_state):
        return [len(group.get("params", ())) for group in optimizer_state["param_groups"]]

    def _preflight_optimizer(self, name, incoming, optimizer):
        if incoming is None:
            return
        preflight_optimizer_state(optimizer, incoming, f"Checkpoint {name}")

    def load(self, fp):
        state = (
            fp
            if isinstance(fp, dict)
            else torch.load(fp, map_location=self.device, weights_only=False)
        )
        checkpoint_version = state.get("checkpoint_version") if isinstance(state, dict) else None
        if checkpoint_version is not None and int(checkpoint_version) not in {
            1,
            2,
            3,
            4,
            5,
            6,
        }:
            raise ValueError(
                f"Unsupported AMBI checkpoint_version={checkpoint_version!r}; "
                "supported versions are 1 through 6."
            )
        structured_checkpoint = isinstance(state, dict) and "model" in state
        actor_loss_scale_candidate = None
        behavior_policy_kl_candidate = None
        if structured_checkpoint:
            if self.behavior_policy_kl_enabled:
                expected = self._checkpoint_version()
                if checkpoint_version != expected:
                    raise ValueError(
                        "An active behavior-policy KL schedule requires a "
                        f"version-{expected} AMBI checkpoint with matching "
                        "scheduler state. For intentional weight transfer, "
                        "load the raw model state into a fresh agent instead."
                    )
                behavior_policy_kl_candidate = (
                    self._preflight_behavior_policy_kl(
                        state.get("behavior_policy_kl_spec"),
                        state.get("behavior_policy_kl_state"),
                        exact=False,
                    )
                )
            elif checkpoint_version in {5, 6}:
                raise ValueError(
                    "Checkpoint behavior-policy KL is enabled but the "
                    "configured schedule is 'none'."
                )
            if self.actor_loss_scale_enabled:
                expected_scale_version = 6 if self.behavior_policy_kl_enabled else 4
                if checkpoint_version != expected_scale_version:
                    raise ValueError(
                        "An enabled actor-loss scaler requires a "
                        f"version-{expected_scale_version} AMBI "
                        "checkpoint with persistent scale state. For intentional "
                        "weight transfer from a legacy checkpoint, load its model "
                        "state into a fresh agent instead."
                    )
                actor_loss_scale_candidate = self._preflight_actor_loss_scale(
                    state.get("actor_loss_scale_spec"),
                    state.get("actor_loss_scale_state"),
                )
            elif checkpoint_version in {4, 6}:
                raise ValueError(
                    "Checkpoint actor-loss scaling is enabled but the configured "
                    "agent has sac_actor_loss_scale_mode='none'."
                )
        saved_observation = (
            state.get("observation_spec") if isinstance(state, dict) else None
        )
        configured_observation = self.observation_signature()
        if (
            saved_observation is not None
            and saved_observation != configured_observation
        ):
            raise ValueError(
                "Checkpoint observation specification does not match this agent: "
                f"checkpoint={saved_observation}, "
                f"configured={configured_observation}."
            )
        saved_spec = state.get("critic_spec") if isinstance(state, dict) else None
        if saved_spec is not None and saved_spec != self.model.critic_signature:
            raise ValueError(
                "Checkpoint critic specification does not match this agent: "
                f"checkpoint={saved_spec}, configured={self.model.critic_signature}."
            )
        # ``critic_target_spec`` is provenance rather than an architecture
        # constraint for portable saves. Frozen-checkpoint studies may
        # intentionally transfer these weights across target objectives; exact
        # training-state restoration validates the specification separately.
        saved_policy_spec = state.get("policy_spec") if isinstance(state, dict) else None
        saved_policy_spec = self._normalize_saved_policy_spec(saved_policy_spec)
        configured_policy_spec = self._policy_spec()
        if saved_policy_spec is not None and saved_policy_spec != configured_policy_spec:
            raise ValueError(
                "Checkpoint policy specification does not match this agent: "
                f"checkpoint={saved_policy_spec}, configured={configured_policy_spec}."
            )
        saved_entropy_spec = state.get("entropy_spec") if isinstance(state, dict) else None
        configured_entropy_spec = {
            "mode": "auto" if self.log_ent_coef is not None else "fixed",
            "target_entropy": float(self.target_entropy),
        }
        if saved_entropy_spec is not None and saved_entropy_spec != configured_entropy_spec:
            raise ValueError(
                "Checkpoint entropy specification does not match this agent: "
                f"checkpoint={saved_entropy_spec}, configured={configured_entropy_spec}."
            )
        if structured_checkpoint:
            has_log_alpha = "log_ent_coef" in state
            has_alpha_optimizer = "ent_coef_optim" in state
            if has_log_alpha != has_alpha_optimizer:
                raise ValueError(
                    "Automatic-entropy checkpoint state is incomplete before load."
                )
            saved_auto_alpha = has_log_alpha
            configured_auto_alpha = self.log_ent_coef is not None
            if saved_auto_alpha != configured_auto_alpha:
                raise ValueError(
                    "Checkpoint entropy mode is incompatible before load: "
                    f"checkpoint={'auto' if saved_auto_alpha else 'fixed'}, "
                    f"configured={'auto' if configured_auto_alpha else 'fixed'}."
                )
            if saved_auto_alpha:
                saved_log_alpha = require_tensor(
                    state["log_ent_coef"],
                    "Checkpoint log_ent_coef",
                    shape=self.log_ent_coef.shape,
                    dtype=self.log_ent_coef.dtype,
                )
                if not bool(torch.isfinite(saved_log_alpha).all().item()):
                    raise ValueError("Checkpoint log_ent_coef must be finite.")
            else:
                saved_fixed_alpha = require_tensor(
                    state.get("fixed_ent_coef"),
                    "Checkpoint fixed_ent_coef",
                    shape=self.fixed_ent_coef.shape,
                    dtype=self.fixed_ent_coef.dtype,
                )
                if not bool(torch.isfinite(saved_fixed_alpha).all().item()):
                    raise ValueError("Checkpoint fixed_ent_coef must be finite.")

            for key in ("num_updates", "outer_version"):
                value = state.get(key, 0)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"Checkpoint {key} must be a non-negative integer."
                    )

        incoming = state["model"] if "model" in state else state
        incoming = api_model_conversion(self.model.state_dict(), incoming)
        expected = self.model.state_dict()
        preflight_module_state(self.model, incoming, "Checkpoint model architecture")
        for key in ("log_std_min", "log_std_dif"):
            if key in incoming and key in expected and not torch.equal(
                incoming[key].to(device=expected[key].device, dtype=expected[key].dtype),
                expected[key],
            ):
                raise ValueError(
                    f"Checkpoint policy bound buffer {key!r} is incompatible before load."
                )
        if "optim" in state:
            self._preflight_optimizer("optim", state["optim"], self.optim)
        if "pi_optim" in state:
            self._preflight_optimizer("pi_optim", state["pi_optim"], self.pi_optim)
        if self.ent_coef_optim is not None and "ent_coef_optim" in state:
            self._preflight_optimizer(
                "ent_coef_optim", state["ent_coef_optim"], self.ent_coef_optim
            )

        # All compatibility checks finish before any live state is mutated.
        self.model.load_state_dict(incoming)
        if "optim" in state:
            load_optimizer_state_preserving_hyperparameters(
                self.optim, state["optim"]
            )
        if "pi_optim" in state:
            load_optimizer_state_preserving_hyperparameters(
                self.pi_optim, state["pi_optim"]
            )
        self.num_updates = int(state.get("num_updates", 0))
        self.outer_version = int(state.get("outer_version", self.num_updates))
        if self.log_ent_coef is not None and "log_ent_coef" in state:
            self.log_ent_coef.data.copy_(state["log_ent_coef"].to(self.device))
            self.log_ent_coef.data.clamp_(min=math.log(_ENTROPY_COEF_MIN))
            if "ent_coef_optim" in state:
                load_optimizer_state_preserving_hyperparameters(
                    self.ent_coef_optim, state["ent_coef_optim"]
                )
        # A fixed entropy coefficient is configuration, not learned state.
        # Portable checkpoints are also used for weight transfer, so keep the
        # receiving value just as we keep receiving optimizer hyperparameters.
        # Exact resume preflight already requires the saved/configured values
        # to match, while automatic entropy still restores its learned state.
        if self.actor_loss_scale_enabled:
            if actor_loss_scale_candidate is None:
                # Raw model-only imports are explicit weight transfers. Their
                # missing training statistic starts from the documented neutral
                # value rather than inheriting live state from unrelated weights.
                self._actor_loss_scale_value.fill_(_ACTOR_LOSS_SCALE_FLOOR)
            else:
                self._actor_loss_scale_value.copy_(
                    actor_loss_scale_candidate["value"].to(self.device)
                )
                self._actor_loss_scale_percentiles.copy_(
                    actor_loss_scale_candidate["percentiles"].to(self.device)
                )
        if self.behavior_policy_kl_enabled:
            if behavior_policy_kl_candidate is None:
                self._reset_behavior_policy_kl_state()
            elif self._behavior_policy_kl_schedule == "smooth":
                self.behavior_policy_kl_eligible_updates = int(
                    behavior_policy_kl_candidate["eligible_updates"]
                )
            elif self._behavior_policy_kl_schedule == "quantile_gate":
                if not self.actor_loss_scale_enabled:
                    q_range = behavior_policy_kl_candidate["q_range"]
                    self._actor_loss_scale_value.copy_(
                        q_range["value"].to(self.device)
                    )
                    self._actor_loss_scale_percentiles.copy_(
                        q_range["percentiles"].to(self.device)
                    )
            elif self._behavior_policy_kl_schedule == "dual":
                self.log_behavior_policy_kl_coef.data.copy_(
                    behavior_policy_kl_candidate["log_coef"].to(self.device)
                )
                # A legacy float32 endpoint can round just outside the exact
                # bounds when promoted to float64.
                self.log_behavior_policy_kl_coef.data.clamp_(
                    min=math.log(_BEHAVIOR_POLICY_KL_DUAL_MIN),
                    max=math.log(float(self.cfg.outer_behavior_policy_kl_dual_max)),
                )
                load_optimizer_state_preserving_hyperparameters(
                    self.behavior_policy_kl_optim,
                    behavior_policy_kl_candidate["optim"],
                )
                self.behavior_policy_kl_dual_updates = int(
                    behavior_policy_kl_candidate["dual_updates"]
                )

        # Inner state is deliberately not checkpointed or resumed.
        self.inner_engine.clear_all()
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        self._resume_boundary_prepared = False
        self.model.eval()
        return self

    @staticmethod
    def _materialize_action_metrics(
        action,
        metrics,
        rollout_lengths,
        behavior_policy=None,
        trace=None,
    ):
        """Copy action, metrics, lengths, and optional policy data together."""
        tensor_items = [
            (key, value)
            for key, value in metrics.items()
            if torch.is_tensor(value)
        ]
        pieces = [action.reshape(-1)]
        pieces.extend(
            value.detach().to(device=action.device, dtype=action.dtype).reshape(1)
            for _, value in tensor_items
        )
        tensor_lengths = (
            rollout_lengths.detach().reshape(-1)
            if torch.is_tensor(rollout_lengths)
            else None
        )
        if tensor_lengths is not None:
            pieces.append(
                tensor_lengths.to(device=action.device, dtype=action.dtype)
            )
        behavior_shapes = None
        if behavior_policy is not None:
            behavior_mean = behavior_policy["pre_tanh_mean"].detach().to(
                device=action.device, dtype=action.dtype
            )
            behavior_log_std = behavior_policy["log_std"].detach().to(
                device=action.device, dtype=action.dtype
            )
            behavior_shapes = (behavior_mean.shape, behavior_log_std.shape)
            pieces.extend((behavior_mean.reshape(-1), behavior_log_std.reshape(-1)))
        trace_start = sum(piece.numel() for piece in pieces) if trace is not None else 0
        trace_items = trace.tensor_items() if trace is not None else []
        pieces.extend(
            value.to(device=action.device, dtype=action.dtype).reshape(1)
            for _, _, value in trace_items
        )
        packed = torch.cat(pieces).detach().cpu()
        if trace is not None:
            trace.materialize(trace_items, packed[trace_start:])
        action_size = int(action.numel())
        cpu_action = packed[:action_size].reshape(action.shape)
        materialized = dict(metrics)
        for offset, (key, _) in enumerate(tensor_items, start=action_size):
            materialized[key] = float(packed[offset])
        if tensor_lengths is not None:
            lengths_start = action_size + len(tensor_items)
            rollout_lengths = [
                int(value)
                for value in packed[
                    lengths_start : lengths_start + tensor_lengths.numel()
                ].tolist()
            ]
        if behavior_shapes is None:
            return cpu_action, materialized, rollout_lengths
        behavior_start = action_size + len(tensor_items)
        if tensor_lengths is not None:
            behavior_start += int(tensor_lengths.numel())
        mean_shape, log_std_shape = behavior_shapes
        mean_size = math.prod(mean_shape)
        log_std_size = math.prod(log_std_shape)
        materialized_behavior = {
            "pre_tanh_mean": packed[
                behavior_start : behavior_start + mean_size
            ].reshape(mean_shape),
            "log_std": packed[
                behavior_start + mean_size : behavior_start + mean_size + log_std_size
            ].reshape(log_std_shape),
        }
        return cpu_action, materialized, rollout_lengths, materialized_behavior

    @torch.no_grad()
    def act_outer_policy(
        self,
        obs,
        *,
        generator=None,
        deterministic=False,
        return_behavior_policy=False,
    ):
        """Act with the online outer actor without entering the inner engine.

        Training interventions retain their historical stochastic behavior and
        require an episode-private generator.  Observational evaluation can
        instead request the deterministic policy mean without supplying or
        advancing a generator.
        """
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be bool.")
        if deterministic:
            if generator is not None:
                raise ValueError(
                    "Deterministic outer-policy actions do not accept a generator."
                )
        else:
            if not isinstance(generator, torch.Generator):
                raise TypeError(
                    "Stochastic outer-policy actions require a torch.Generator."
                )
            generator_device = torch.device(generator.device)
            expected_device = (
                self.device
                if self.device.type == "cuda"
                else torch.device("cpu")
            )
            if generator_device.type != expected_device.type or (
                generator_device.type == "cuda"
                and generator_device.index not in {None, expected_device.index}
            ):
                raise ValueError(
                    "The outer-policy generator must be on the AMBI agent device."
                )

        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        training_modes = tuple(
            (module, bool(module.training)) for module in self.model.modules()
        )
        fork_devices = []
        if self.device.type == "cuda":
            fork_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        try:
            self.model.eval()
            # Pixel augmentation may use the default generator. Forking keeps
            # that incidental randomness from advancing the outer learner's
            # global stream; policy noise comes only from the episode-private
            # generator supplied above.
            with torch.random.fork_rng(devices=fork_devices, enabled=True):
                root_z = self.model.encode(obs).detach()
                if return_behavior_policy:
                    action, policy_info = self.model.pi(
                        root_z,
                        deterministic=deterministic,
                        generator=generator,
                    )
                else:
                    action = self.model.pi_action(
                        root_z,
                        deterministic=deterministic,
                        generator=generator,
                    )
        finally:
            for module, was_training in training_modes:
                module.training = was_training

        if action.ndim < 1 or int(action.shape[0]) != 1:
            raise ValueError(
                "The outer policy must return one batched action per observation."
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError("The outer policy returned a non-finite action.")
        if not return_behavior_policy:
            return action[0].detach().cpu()
        packed = torch.cat(
            (
                action[0].reshape(-1),
                policy_info["pre_tanh_mean"][0].reshape(-1),
                policy_info["log_std"][0].reshape(-1),
            )
        ).detach().cpu()
        action_size = int(action[0].numel())
        return packed[:action_size].reshape(action[0].shape), {
            "pre_tanh_mean": packed[action_size : 2 * action_size].reshape(
                action[0].shape
            ),
            "log_std": packed[2 * action_size : 3 * action_size].reshape(
                action[0].shape
            ),
        }

    def act(
        self,
        obs,
        t0=False,
        eval_mode=False,
        task=None,
        *,
        collect_diagnostics=True,
        return_behavior_policy=False,
        apply_inner_writeback=False,
        trace=None,
    ):
        if task is not None:
            raise ValueError("AMBI-TD-MPC2 currently supports single-task training only.")
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        was_training = self.model.training
        self.model.eval()
        try:
            if self.cfg.obs == "rgb":
                # ShiftAug uses the default generator. Enclose the root pixel
                # crop and the nested inner action in one saved RNG scope so
                # action selection cannot advance the outer learner's global
                # CPU/CUDA RNG state.
                with self.inner_engine.rng.action_fork():
                    with self.inner_engine.rng.fork("observation"):
                        with torch.no_grad():
                            root_z = self.model.encode(obs).detach()
                    # The shared TD-MPC2 evaluator is inference-only, but an
                    # AMBI action still performs its configured root-local
                    # optimizer steps before returning a deterministic action.
                    # Re-enable gradients locally without reconnecting the
                    # detached root latent to the outer encoder.
                    with torch.enable_grad():
                        result = self.inner_engine.act(
                            root_z,
                            t0=t0,
                            eval_mode=eval_mode,
                            collect_diagnostics=collect_diagnostics,
                            return_behavior_policy=return_behavior_policy,
                            apply_inner_writeback=apply_inner_writeback,
                            **({"trace": trace} if trace is not None else {}),
                        )
            else:
                with torch.no_grad():
                    root_z = self.model.encode(obs).detach()
                # See the pixel-observation branch above. Evaluation controls
                # the executed action, not whether AMBI performs inner learning.
                with torch.enable_grad():
                    result = self.inner_engine.act(
                        root_z,
                        t0=t0,
                        eval_mode=eval_mode,
                        collect_diagnostics=collect_diagnostics,
                        return_behavior_policy=return_behavior_policy,
                        apply_inner_writeback=apply_inner_writeback,
                        **({"trace": trace} if trace is not None else {}),
                    )
        finally:
            self.model.train(was_training)
        if return_behavior_policy:
            action, metrics, lengths, behavior_policy = result
            if behavior_policy is None:
                action, metrics, lengths = self._materialize_action_metrics(
                    action,
                    metrics,
                    lengths,
                    trace=trace,
                )
            else:
                action, metrics, lengths, behavior_policy = (
                    self._materialize_action_metrics(
                        action,
                        metrics,
                        lengths,
                        behavior_policy=behavior_policy,
                        trace=trace,
                    )
                )
        else:
            action, metrics, lengths = result
            action, metrics, lengths = self._materialize_action_metrics(
                action, metrics, lengths, trace=trace
            )
        metrics = self.inner_engine.finalize_timing_metrics(metrics)
        self.last_inner_metrics = metrics
        self.last_inner_rollout_lengths = lengths
        if return_behavior_policy:
            return action, behavior_policy
        return action

    def _make_inner_modules(self):
        """Compatibility hook used by existing target-sync/isolation tests."""
        return self.inner_engine.make_modules_for_compatibility()

    @torch.no_grad()
    def _soft_td_target(self, next_z, reward, terminated):
        """Build the outer critic target; retain the historical helper name."""
        next_action, next_info = self.model.pi(next_z)
        next_q = self.model.Q(
            next_z,
            next_action,
            target=True,
            reduction=self.cfg.outer_q_target_reduction,
        )
        bootstrap = next_q
        if self.cfg.outer_critic_target == "entropy_augmented":
            bootstrap = bootstrap - self.alpha.detach() * next_info["log_prob"]
        return reward + self.discount * (1.0 - terminated) * bootstrap

    def _should_run_value_equivalence_diagnostics(self):
        """Whether the upcoming completed outer update is a sampled event."""
        if not bool(getattr(self.cfg, "value_equivalence_diagnostics", False)):
            return False
        cadence = int(getattr(self.cfg, "value_equivalence_every_updates", 1000))
        return (self.num_updates + 1) % cadence == 0

    def _initial_inner_diagnostic_alpha(self):
        """Return alpha at the beginning of a fresh inner SAC solve."""
        mode = str(self.cfg.inner_temperature_mode)
        if mode == "inherit_outer":
            return self.alpha.detach()
        if mode == "fixed":
            return self.alpha.detach().new_tensor(float(self.cfg.inner_temperature))
        if mode != "auto":
            raise ValueError(f"Unknown inner temperature mode: {mode!r}")

        initialization = str(self.cfg.inner_temperature_initialization)
        if initialization == "inherit_outer":
            return self.alpha.detach()
        if initialization == "fixed":
            return self.alpha.detach().new_tensor(float(self.cfg.inner_temperature))
        raise ValueError(
            "Unknown inner temperature initialization: "
            f"{initialization!r}"
        )

    def _value_equivalence_reference_critic(self):
        """Resolve the critic used by the fresh inner Bellman target."""
        source = str(self.cfg.inner_bootstrap_source)
        if source == "outer_target":
            return self.model._target_Qs
        if source in {"inner_target", "outer_online"}:
            # A fresh action-local inner target is an eval-mode hard copy of
            # the online critic, so evaluating the online module is equivalent
            # without allocating or mutating an inner workspace.
            return self.model._Qs
        raise ValueError(f"Unknown inner bootstrap source: {source!r}")

    @staticmethod
    def _value_equivalence_masked_mean(value, mask=None):
        if mask is None:
            return value.mean()
        weights = torch.broadcast_to(mask.to(dtype=value.dtype), value.shape)
        return (value * weights).sum() / weights.sum()

    @classmethod
    def _value_equivalence_rmse(cls, error, mask=None):
        return cls._value_equivalence_masked_mean(error.square(), mask).sqrt()

    def _value_equivalence_q(
        self,
        critic,
        z,
        action,
        reduction,
        pair_indices,
    ):
        """Evaluate a paired diagnostic Q without touching lazy compile state."""
        q_input = self.model.joint_input(z, action)
        q_predictions = critic._forward_eager(q_input)
        q_values = self.model.q_backend.decode(q_predictions)
        return self.model.q_backend.reduce(
            q_values,
            reduction,
            pair_indices=pair_indices,
            trusted_pair_indices=pair_indices is not None,
        )

    def _value_equivalence_q_with_input_grad(
        self,
        critic,
        z,
        action,
        reduction,
        pair_indices,
    ):
        """Evaluate a frozen Q probe while preserving latent/action gradients."""
        q_input = self.model.joint_input(z, action)
        q_predictions = critic._forward_detached_eager(q_input)
        q_values = self.model.q_backend.decode(q_predictions)
        return self.model.q_backend.reduce(
            q_values,
            reduction,
            pair_indices=pair_indices,
            trusted_pair_indices=pair_indices is not None,
        )

    def _value_equivalence_loss(
        self,
        latent_states,
        next_z_targets,
        loss_update,
    ):
        """Match fresh-inner successor values without training the probe.

        This is the value component of Bellman equivalence. TOLD's existing
        reward objective continues to supervise rewards independently. The
        maintained loss is continuing-task only, so it contains neither a
        termination model nor a continuation mask.
        """
        model_next_z = latent_states[1:]
        real_next_z = next_z_targets.detach()
        paired_next_z = torch.stack((real_next_z, model_next_z), dim=0)
        critic = self._value_equivalence_reference_critic()
        actor_training_modes = tuple(
            (module, bool(module.training)) for module in self.model._pi.modules()
        )
        critic_training_modes = tuple(
            (module, bool(module.training)) for module in critic.modules()
        )

        generator_device = self.device if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        seed = (
            int(self.cfg.seed)
            + 1_000_003 * int(loss_update)
            + 0x56454C4F5353
        ) % (2**63 - 1)
        generator.manual_seed(seed)

        alpha = self._initial_inner_diagnostic_alpha()
        reduction = str(self.cfg.inner_q_target_reduction)
        mc_samples = int(self.cfg.value_equivalence_loss_mc_samples)
        sampled_values = []
        try:
            self.model._pi.eval()
            critic.eval()
            for _ in range(mc_samples):
                policy_noise = torch.randn(
                    real_next_z.shape[:-1] + (int(self.cfg.action_dim),),
                    dtype=real_next_z.dtype,
                    device=real_next_z.device,
                    generator=generator,
                )
                paired_noise = policy_noise.unsqueeze(0).expand(
                    (2,) + policy_noise.shape
                )
                next_action, next_info = self.model.pi(
                    paired_next_z,
                    policy=self.model._pi,
                    noise=paired_noise,
                    log_std_mapping=self.cfg.inner_log_std_mapping,
                    log_std_min=self.cfg.inner_log_std_min,
                    log_std_max=self.cfg.inner_log_std_max,
                    detach_policy=True,
                )
                pair_indices = (
                    self.model.q_backend.sample_pair_indices(
                        self.device, generator=generator
                    )
                    if reduction.endswith("_pair")
                    else None
                )
                next_q = self._value_equivalence_q_with_input_grad(
                    critic,
                    paired_next_z,
                    next_action,
                    reduction,
                    pair_indices,
                )
                bootstrap_value = next_q
                if self.cfg.inner_sac_critic_target == "entropy_augmented":
                    bootstrap_value = (
                        bootstrap_value - alpha * next_info["log_prob"]
                    )
                sampled_values.append(bootstrap_value)
        finally:
            for module, was_training in actor_training_modes:
                module.training = was_training
            for module, was_training in critic_training_modes:
                module.training = was_training

        mean_value = torch.stack(sampled_values, dim=0).mean(dim=0)
        real_value, model_value = mean_value.unbind(0)
        value_error = float(self.discount) * (
            model_value - real_value.detach()
        )
        per_depth_losses = value_error.square().mean(
            dim=tuple(range(1, value_error.ndim))
        )
        loss = td_math.reduce_temporal_loss(
            per_depth_losses,
            self.cfg.rho,
            normalization=self.cfg.temporal_loss_normalization,
            reference_horizon=self.cfg.temporal_loss_reference_horizon,
            legacy_order="vector_sum_divide",
            weights=self._transition_temporal_weights,
        )
        return loss, per_depth_losses

    @torch.no_grad()
    def _value_equivalence_diagnostics(
        self,
        latent_states,
        reward_predictions,
        termination_prediction,
        next_z_targets,
        reward,
        terminated,
        diagnostic_update,
    ):
        """Compare fresh-inner soft Bellman targets on model and replay paths.

        Depth one begins at the encoded replay root. Later depths use the
        recurrent TOLD rollout under the recorded replay actions, so their
        errors include accumulated latent-model drift. The evaluator remains
        the fresh outer prior even when a non-canonical configuration persists
        adapted inner modules across roots.
        """
        model_next_z = latent_states[1:].detach()
        real_next_z = next_z_targets.detach()
        real_reward = reward.detach()
        real_terminated = terminated.detach().to(dtype=real_reward.dtype)
        model_reward = td_math.two_hot_inv(
            reward_predictions.detach(), self.cfg
        )
        if bool(self.cfg.episodic):
            if termination_prediction is None:
                raise RuntimeError(
                    "Episodic value-equivalence diagnostics require termination logits."
                )
            model_terminated = (
                torch.sigmoid(termination_prediction.detach())
                > float(self.cfg.inner_termination_threshold)
            ).to(dtype=real_reward.dtype)
        else:
            model_terminated = torch.zeros_like(real_terminated)

        alive_before_depth = torch.ones_like(real_terminated)
        if int(real_terminated.shape[0]) > 1:
            matched_continuation = (
                (1.0 - real_terminated[:-1])
                * (1.0 - model_terminated[:-1])
            )
            alive_before_depth[1:] = torch.cumprod(
                matched_continuation, dim=0
            )

        paired_next_z = torch.stack((real_next_z, model_next_z), dim=0)
        critic = self._value_equivalence_reference_critic()
        actor_training_modes = tuple(
            (module, bool(module.training)) for module in self.model._pi.modules()
        )
        critic_training_modes = tuple(
            (module, bool(module.training)) for module in critic.modules()
        )

        generator_device = self.device if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        seed = (
            int(self.cfg.seed)
            + 1_000_003 * int(diagnostic_update)
            + 0x5EED5EED
        ) % (2**63 - 1)
        generator.manual_seed(seed)

        value_sum = torch.zeros(
            paired_next_z.shape[:-1] + (1,),
            dtype=paired_next_z.dtype,
            device=paired_next_z.device,
        )
        alpha = self._initial_inner_diagnostic_alpha()
        reduction = str(self.cfg.inner_q_target_reduction)
        mc_samples = int(self.cfg.value_equivalence_mc_samples)
        try:
            self.model._pi.eval()
            critic.eval()
            for _ in range(mc_samples):
                policy_noise = torch.randn(
                    real_next_z.shape[:-1] + (int(self.cfg.action_dim),),
                    dtype=real_next_z.dtype,
                    device=real_next_z.device,
                    generator=generator,
                )
                paired_noise = policy_noise.unsqueeze(0).expand(
                    (2,) + policy_noise.shape
                )
                next_action, next_info = self.model.pi(
                    paired_next_z,
                    policy=self.model._pi,
                    noise=paired_noise,
                    log_std_mapping=self.cfg.inner_log_std_mapping,
                    log_std_min=self.cfg.inner_log_std_min,
                    log_std_max=self.cfg.inner_log_std_max,
                )
                pair_indices = (
                    self.model.q_backend.sample_pair_indices(
                        self.device, generator=generator
                    )
                    if reduction.endswith("_pair")
                    else None
                )
                next_q = self._value_equivalence_q(
                    critic,
                    paired_next_z,
                    next_action,
                    reduction,
                    pair_indices,
                )
                bootstrap_value = next_q
                if self.cfg.inner_sac_critic_target == "entropy_augmented":
                    bootstrap_value = (
                        bootstrap_value - alpha * next_info["log_prob"]
                    )
                value_sum.add_(bootstrap_value)
        finally:
            for module, was_training in actor_training_modes:
                module.training = was_training
            for module, was_training in critic_training_modes:
                module.training = was_training

        mean_value = value_sum / float(mc_samples)
        real_value, model_value = mean_value.unbind(0)
        real_bootstrap = (
            float(self.discount) * (1.0 - real_terminated) * real_value
        )
        model_bootstrap = (
            float(self.discount) * (1.0 - model_terminated) * model_value
        )
        reward_error = model_reward - real_reward
        bootstrap_error = model_bootstrap - real_bootstrap
        target_error = reward_error + bootstrap_error
        reference_target = real_reward + real_bootstrap

        target_rmse = self._value_equivalence_rmse(
            target_error, alive_before_depth
        )
        eps = torch.finfo(target_error.dtype).eps
        reference_rms = self._value_equivalence_rmse(
            reference_target, alive_before_depth
        )
        reward_mse = self._value_equivalence_masked_mean(
            reward_error.square(), alive_before_depth
        )
        bootstrap_mse = self._value_equivalence_masked_mean(
            bootstrap_error.square(), alive_before_depth
        )
        cancellation_fraction = (
            -2.0
            * self._value_equivalence_masked_mean(
                reward_error * bootstrap_error, alive_before_depth
            )
            / (reward_mse + bootstrap_mse + eps)
        ).clamp(min=0.0, max=1.0)
        info = {
            "ve_prior_target_mae": self._value_equivalence_masked_mean(
                target_error.abs(), alive_before_depth
            ),
            "ve_prior_target_rmse": target_rmse,
            "ve_prior_target_bias": self._value_equivalence_masked_mean(
                target_error, alive_before_depth
            ),
            "ve_prior_target_nrmse": target_rmse
            / reference_rms.clamp_min(eps),
            "ve_prior_target_abs_p95": torch.quantile(
                target_error.abs()[alive_before_depth.bool()], 0.95
            ),
            "ve_prior_reference_target_rms": reference_rms,
            "ve_prior_reward_rmse": reward_mse.sqrt(),
            "ve_prior_bootstrap_rmse": bootstrap_mse.sqrt(),
            "ve_prior_cancellation_fraction": cancellation_fraction,
        }
        for depth in range(int(self.cfg.train_unroll_horizon)):
            depth_index = depth + 1
            depth_target_error = target_error[depth]
            depth_reward_error = reward_error[depth]
            depth_bootstrap_error = bootstrap_error[depth]
            depth_mask = alive_before_depth[depth]
            if not bool(depth_mask.any().item()):
                continue
            info[f"ve_prior_target_mae_depth_{depth_index}"] = (
                self._value_equivalence_masked_mean(
                    depth_target_error.abs(), depth_mask
                )
            )
            info[f"ve_prior_target_rmse_depth_{depth_index}"] = (
                self._value_equivalence_rmse(depth_target_error, depth_mask)
            )
            info[f"ve_prior_target_bias_depth_{depth_index}"] = (
                self._value_equivalence_masked_mean(
                    depth_target_error, depth_mask
                )
            )
            info[f"ve_prior_reward_rmse_depth_{depth_index}"] = (
                self._value_equivalence_rmse(depth_reward_error, depth_mask)
            )
            info[f"ve_prior_bootstrap_rmse_depth_{depth_index}"] = (
                self._value_equivalence_rmse(
                    depth_bootstrap_error, depth_mask
                )
            )
        if bool(self.cfg.episodic):
            disagreement = (model_terminated != real_terminated).to(
                dtype=real_reward.dtype
            )
            info["ve_prior_termination_disagreement"] = (
                self._value_equivalence_masked_mean(
                    disagreement, alive_before_depth
                )
            )
            for depth in range(int(self.cfg.train_unroll_horizon)):
                depth_mask = alive_before_depth[depth]
                if not bool(depth_mask.any().item()):
                    continue
                info[f"ve_prior_termination_disagreement_depth_{depth + 1}"] = (
                    self._value_equivalence_masked_mean(
                        disagreement[depth], depth_mask
                    )
                )
        return info

    def _behavior_policy_inputs(
        self,
        policy_info,
        behavior_pre_tanh_mean,
        behavior_log_std,
        behavior_policy_valid,
        *,
        require_pre_tanh_action=False,
    ):
        """Validate and align current-policy and replayed behavior tensors."""
        if any(
            value is None
            for value in (
                behavior_pre_tanh_mean,
                behavior_log_std,
                behavior_policy_valid,
            )
        ):
            raise ValueError(
                "An active behavior-policy regularizer requires replayed "
                "behavior mean, log-std, and validity tensors."
            )
        current_mean = policy_info["pre_tanh_mean"][:-1]
        current_log_std = policy_info["log_std"][:-1]
        current_pre_tanh_action = None
        if require_pre_tanh_action:
            if "pre_tanh_action" not in policy_info:
                raise ValueError(
                    "Action-space behavior cross-entropy requires the current "
                    "policy's pre-tanh action sample."
                )
            current_pre_tanh_action = policy_info["pre_tanh_action"][:-1]
            if current_pre_tanh_action.shape != current_mean.shape:
                raise ValueError(
                    "Current pre-tanh action shape does not match the "
                    "nonterminal actor states: "
                    f"{tuple(current_pre_tanh_action.shape)} != "
                    f"{tuple(current_mean.shape)}."
                )
        if behavior_pre_tanh_mean.shape != current_mean.shape:
            raise ValueError(
                "Behavior-policy mean shape does not match the nonterminal "
                f"actor states: {tuple(behavior_pre_tanh_mean.shape)} != "
                f"{tuple(current_mean.shape)}."
            )
        if behavior_log_std.shape != current_log_std.shape:
            raise ValueError(
                "Behavior-policy log-std shape does not match the nonterminal "
                f"actor states: {tuple(behavior_log_std.shape)} != "
                f"{tuple(current_log_std.shape)}."
            )
        valid = behavior_policy_valid.to(device=self.device, dtype=torch.bool)
        if valid.ndim == current_mean.ndim and valid.shape[-1] == 1:
            valid = valid.squeeze(-1)
        expected_valid_shape = current_mean.shape[:-1]
        if valid.shape != expected_valid_shape:
            raise ValueError(
                "Behavior-policy validity shape does not match replay rows: "
                f"{tuple(valid.shape)} != {tuple(expected_valid_shape)}."
            )
        return current_mean, current_log_std, current_pre_tanh_action, valid

    def _behavior_policy_row_statistics(self, values, valid, zero_anchor):
        """Reduce per-row behavior losses with the established valid weights."""
        temporal_weights = self._transition_temporal_weights.to(dtype=values.dtype)
        temporal_weights = temporal_weights.reshape(
            (int(temporal_weights.shape[0]),)
            + (1,) * (values.ndim - 1)
        )
        weighted_valid = temporal_weights * valid.to(dtype=values.dtype)
        denominator = weighted_valid.sum()
        valid_count = int(valid.sum().item())
        has_weighted_support = bool(
            valid_count
            and torch.isfinite(denominator).item()
            and denominator.item() > 0.0
        )
        if has_weighted_support:
            loss = (values[valid] * weighted_valid[valid]).sum() / denominator
            valid_values = values[valid]
            p50 = torch.quantile(valid_values.detach(), 0.5)
            p95 = torch.quantile(valid_values.detach(), 0.95)
        else:
            loss = zero_anchor * 0.0
            p50 = loss.detach()
            p95 = loss.detach()
        ready = has_weighted_support and valid_count >= int(
            self.cfg.outer_behavior_policy_kl_min_valid_count
        )
        return {
            "loss": loss,
            "p50": p50,
            "p95": p95,
            "valid_count": valid_count,
            "valid_fraction": float(valid_count) / float(valid.numel()),
            "ready": ready,
            "has_weighted_support": has_weighted_support,
            "weighted_valid": weighted_valid,
            "denominator": denominator,
        }

    def _behavior_policy_kl_loss(
        self,
        policy_info,
        behavior_pre_tanh_mean,
        behavior_log_std,
        behavior_policy_valid,
    ):
        """Return the valid-weighted Jensen component reverse-KL estimate."""
        current_mean, current_log_std, _, valid = self._behavior_policy_inputs(
            policy_info,
            behavior_pre_tanh_mean,
            behavior_log_std,
            behavior_policy_valid,
        )
        kl_per_row = td_math.diagonal_gaussian_reverse_kl(
            current_mean,
            current_log_std,
            behavior_pre_tanh_mean,
            behavior_log_std,
        ).squeeze(-1) / float(self.cfg.action_dim)
        stats = self._behavior_policy_row_statistics(
            kl_per_row,
            valid,
            current_mean.sum() + current_log_std.sum(),
        )
        metrics = {
            "behavior_policy_kl": stats["loss"].detach(),
            "behavior_policy_kl_p50": stats["p50"],
            "behavior_policy_kl_p95": stats["p95"],
            "behavior_policy_kl_valid_count": float(stats["valid_count"]),
            "behavior_policy_kl_valid_fraction": stats["valid_fraction"],
            "behavior_policy_kl_ready": float(stats["ready"]),
        }
        return stats["loss"], stats["ready"], metrics

    def _behavior_policy_action_ce_loss(
        self,
        policy_info,
        behavior_pre_tanh_mean,
        behavior_log_std,
        behavior_policy_valid,
    ):
        """Return exact normalized-action CE with a sampled tanh correction."""
        (
            current_mean,
            current_log_std,
            current_pre_tanh_action,
            valid,
        ) = self._behavior_policy_inputs(
            policy_info,
            behavior_pre_tanh_mean,
            behavior_log_std,
            behavior_policy_valid,
            require_pre_tanh_action=True,
        )
        action_dim = float(self.cfg.action_dim)
        gaussian_per_row = td_math.diagonal_gaussian_cross_entropy(
            current_mean,
            current_log_std,
            behavior_pre_tanh_mean,
            behavior_log_std,
        ).squeeze(-1) / action_dim
        log_jacobian_per_row = td_math.tanh_log_abs_det_jacobian(
            current_pre_tanh_action,
        ).squeeze(-1) / action_dim
        ce_per_row = gaussian_per_row + log_jacobian_per_row
        stats = self._behavior_policy_row_statistics(
            ce_per_row,
            valid,
            (
                current_mean.sum()
                + current_log_std.sum()
                + current_pre_tanh_action.sum()
            ),
        )
        if stats["has_weighted_support"]:
            weighted_valid = stats["weighted_valid"]
            denominator = stats["denominator"]
            gaussian_component = (
                gaussian_per_row[valid] * weighted_valid[valid]
            ).sum() / denominator
            jacobian_component = (
                log_jacobian_per_row[valid] * weighted_valid[valid]
            ).sum() / denominator
        else:
            gaussian_component = stats["loss"]
            jacobian_component = stats["loss"]
        metrics = {
            "behavior_policy_action_ce": stats["loss"].detach(),
            "behavior_policy_action_ce_gaussian": gaussian_component.detach(),
            "behavior_policy_action_ce_log_abs_det_jacobian": (
                jacobian_component.detach()
            ),
            "behavior_policy_action_ce_p50": stats["p50"],
            "behavior_policy_action_ce_p95": stats["p95"],
            "behavior_policy_action_ce_valid_count": float(
                stats["valid_count"]
            ),
            "behavior_policy_action_ce_valid_fraction": stats["valid_fraction"],
            "behavior_policy_action_ce_ready": float(stats["ready"]),
        }
        return stats["loss"], stats["ready"], metrics

    def _behavior_policy_regularizer_loss(
        self,
        policy_info,
        behavior_pre_tanh_mean,
        behavior_log_std,
        behavior_policy_valid,
    ):
        if self._behavior_policy_objective == "reverse_kl":
            return self._behavior_policy_kl_loss(
                policy_info,
                behavior_pre_tanh_mean,
                behavior_log_std,
                behavior_policy_valid,
            )
        return self._behavior_policy_action_ce_loss(
            policy_info,
            behavior_pre_tanh_mean,
            behavior_log_std,
            behavior_policy_valid,
        )

    @torch.no_grad()
    def _clip_actor_grad_norm_(self):
        """Keep ordinary L2 clipping safe for large finite behavior gradients."""
        parameters = tuple(self.model._pi.parameters())
        maximum = float(self.cfg.grad_clip_norm)
        if not self.behavior_policy_kl_enabled:
            return torch.nn.utils.clip_grad_norm_(parameters, maximum)

        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            return torch.zeros((), device=self.device, dtype=torch.float64)
        # Widen each reduction, not just the final scalar: a float32 per-tensor
        # sum of squares can overflow even when every gradient is finite.
        norms = torch.stack([
            torch.linalg.vector_norm(g, dtype=torch.float64) for g in gradients
        ])
        total_norm = torch.linalg.vector_norm(norms)
        coefficient = (maximum / (total_norm + 1e-6)).clamp(max=1.0)
        # All actor gradients share the model's dtype/device. Keep the network
        # and its Adam state in float32; only norm accumulation is widened.
        torch._foreach_mul_(gradients, coefficient.to(gradients[0]))
        return total_norm

    def _update_actor(
        self,
        zs,
        behavior_pre_tanh_mean=None,
        behavior_log_std=None,
        behavior_policy_valid=None,
    ):
        action, policy_info = self.model.pi(zs)
        actor_saturation_metrics = {}
        # Real SoftWorldModel policy samples always expose their pre-tanh action.
        # Keep the established lightweight policy-stub seam usable in tests and
        # downstream diagnostics that only provide the fields consumed by SAC.
        if "pre_tanh_action" in policy_info:
            (
                actor_pre_tanh_abs_mean,
                actor_pre_tanh_abs_max,
                actor_pre_tanh_abs_ge_7p6_fraction,
                actor_action_exact_saturation_fraction,
            ) = td_math.tanh_saturation_statistics(
                policy_info["pre_tanh_action"].detach(),
                action.detach(),
            )
            actor_saturation_metrics = {
                "actor_pre_tanh_abs_mean": actor_pre_tanh_abs_mean,
                "actor_pre_tanh_abs_max": actor_pre_tanh_abs_max,
                "actor_pre_tanh_abs_ge_7p6_fraction": (
                    actor_pre_tanh_abs_ge_7p6_fraction
                ),
                "actor_action_exact_saturation_fraction": (
                    actor_action_exact_saturation_fraction
                ),
            }
        behavior_regularizer = torch.zeros((), device=self.device)
        behavior_regularizer_ready = False
        behavior_regularizer_metrics = {}
        if self.behavior_policy_kl_enabled:
            (
                behavior_regularizer,
                behavior_regularizer_ready,
                behavior_regularizer_metrics,
            ) = (
                self._behavior_policy_regularizer_loss(
                    policy_info,
                    behavior_pre_tanh_mean,
                    behavior_log_std,
                    behavior_policy_valid,
                )
            )
        # Preserve the established SAC ordering: this slot's actor uses alpha
        # from the beginning of the slot; an automatic-alpha step affects the
        # next outer update.
        alpha = self.alpha.detach()
        entropy_coefficient_loss = torch.zeros((), device=self.device)
        if self.ent_coef_optim is not None:
            entropy_residual_per_time = (
                policy_info["log_prob"] + self.target_entropy
            ).detach().mean(dim=(1, 2))
            # Match the actor's relative depth mixture without making the
            # temperature step size depend on horizon-level loss scaling.
            entropy_temporal_weights = self._actor_temporal_weights.to(
                dtype=entropy_residual_per_time.dtype
            )
            entropy_temporal_weights = (
                entropy_temporal_weights / entropy_temporal_weights.sum()
            )
            weighted_entropy_residual = (
                entropy_residual_per_time * entropy_temporal_weights
            ).sum()
            entropy_coefficient_loss = -(
                self.log_ent_coef * weighted_entropy_residual
            ).mean()
            self.ent_coef_optim.zero_grad(set_to_none=True)
            entropy_coefficient_loss.backward()
            self.ent_coef_optim.step()
            with torch.no_grad():
                self.log_ent_coef.clamp_(min=math.log(_ENTROPY_COEF_MIN))

        behavior_regularizer_coefficient = torch.zeros((), device=self.device)
        behavior_kl_dual_loss = torch.zeros((), device=self.device)
        behavior_kl_dual_violation = torch.zeros((), device=self.device)
        behavior_kl_dual_cap_hit = torch.zeros((), device=self.device)
        behavior_kl_dual_updated = torch.zeros((), device=self.device)
        dual_coefficient_after = None
        if self._behavior_policy_kl_schedule == "dual":
            slot_coefficient = self.log_behavior_policy_kl_coef.exp().detach()
            if behavior_regularizer_ready:
                behavior_regularizer_coefficient = slot_coefficient.to(
                    behavior_regularizer
                ).reshape(())
                behavior_kl_dual_violation = (
                    behavior_regularizer.detach().to(slot_coefficient.dtype)
                    - float(self.cfg.outer_behavior_policy_kl_target)
                )
                behavior_kl_dual_loss = -(
                    self.log_behavior_policy_kl_coef
                    * behavior_kl_dual_violation
                ).mean()
                self.behavior_policy_kl_optim.zero_grad(set_to_none=True)
                behavior_kl_dual_loss.backward()
                self.behavior_policy_kl_optim.step()
                with torch.no_grad():
                    self.log_behavior_policy_kl_coef.clamp_(
                        min=math.log(_BEHAVIOR_POLICY_KL_DUAL_MIN),
                        max=math.log(
                            float(self.cfg.outer_behavior_policy_kl_dual_max)
                        ),
                    )
                self.behavior_policy_kl_dual_updates += 1
                behavior_kl_dual_updated.fill_(1.0)
            dual_coefficient_after = (
                self.log_behavior_policy_kl_coef.exp().detach().reshape(())
            )
            behavior_kl_dual_cap_hit = (
                self.log_behavior_policy_kl_coef.detach().reshape(())
                >= math.log(float(self.cfg.outer_behavior_policy_kl_dual_max))
            ).to(dtype=torch.float32)

        # Materialize the decoded ensemble once so the configured actor
        # reduction and the observational head-gap diagnostics see exactly the
        # same critic/dropout sample. The configured reduction below remains
        # the only Q signal used by the actor objective.
        q_policy_all = self.model.Q(
            zs,
            action,
            reduction="all",
            detach=True,
        )
        q_policy = self.model.q_backend.reduce(
            q_policy_all,
            self.cfg.outer_q_actor_reduction,
        )
        q_policy_all_detached = q_policy_all.detach()
        q_policy_mean_all = self.model.q_backend.reduce(
            q_policy_all_detached,
            "mean_all",
        )
        q_policy_min_all = self.model.q_backend.reduce(
            q_policy_all_detached,
            "min_all",
        )
        q_range_instant = self._update_actor_loss_scale(q_policy[0])

        smooth_progress = 0.0
        quantile_gate_active = False
        if self._behavior_policy_kl_schedule == "smooth":
            ramp_updates = int(self.cfg.outer_behavior_policy_kl_ramp_updates)
            completed = int(self.behavior_policy_kl_eligible_updates)
            if behavior_regularizer_ready:
                smooth_progress = min(
                    float(completed + 1) / float(ramp_updates), 1.0
                )
                smooth_weight = smooth_progress * smooth_progress * (
                    3.0 - 2.0 * smooth_progress
                )
                behavior_regularizer_coefficient = behavior_regularizer.new_tensor(
                    float(self.cfg.outer_behavior_policy_kl_coef)
                    * smooth_weight
                )
            else:
                smooth_progress = min(
                    float(completed) / float(ramp_updates), 1.0
                )
        elif self._behavior_policy_kl_schedule == "quantile_gate":
            quantile_gate_active = bool(
                self._actor_loss_scale_value.item()
                > float(self.cfg.outer_behavior_policy_kl_q_threshold)
            )
            if behavior_regularizer_ready and quantile_gate_active:
                behavior_regularizer_coefficient = behavior_regularizer.new_tensor(
                    float(self.cfg.outer_behavior_policy_kl_coef)
                )

        actor_objective = alpha * policy_info["log_prob"] - q_policy
        actor_per_time = actor_objective.mean(dim=(1, 2))
        sac_actor_loss = td_math.reduce_temporal_loss(
            actor_per_time,
            self.cfg.rho,
            normalization=self.cfg.temporal_loss_normalization,
            reference_horizon=self.cfg.temporal_loss_reference_horizon,
            include_terminal=True,
            legacy_order="vector_mean",
            weights=self._actor_temporal_weights,
        )
        if self.behavior_policy_kl_enabled:
            actor_loss = (
                sac_actor_loss
                + behavior_regularizer_coefficient * behavior_regularizer
            )
            if self.actor_loss_scale_enabled:
                actor_loss = actor_loss / self._actor_loss_scale_value.reshape(())
        else:
            # Preserve the historical feature-disabled ordering, including
            # scaling before temporal reduction.
            if self.actor_loss_scale_enabled:
                actor_objective = actor_objective / self._actor_loss_scale_value
                actor_per_time = actor_objective.mean(dim=(1, 2))
                actor_loss = td_math.reduce_temporal_loss(
                    actor_per_time,
                    self.cfg.rho,
                    normalization=self.cfg.temporal_loss_normalization,
                    reference_horizon=self.cfg.temporal_loss_reference_horizon,
                    include_terminal=True,
                    legacy_order="vector_mean",
                    weights=self._actor_temporal_weights,
                )
            else:
                actor_loss = sac_actor_loss

        self.pi_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = self._clip_actor_grad_norm_()
        self.pi_optim.step()
        if (
            self._behavior_policy_kl_schedule == "smooth"
            and behavior_regularizer_ready
        ):
            self.behavior_policy_kl_eligible_updates += 1
        metrics = {
            "actor_loss": actor_loss.detach(),
            "actor_grad_norm": torch.as_tensor(actor_grad_norm).detach(),
            "actor_entropy": policy_info["entropy"].detach().mean(),
            "actor_q_mean": q_policy.detach().mean(),
            "actor_q_mean_all": q_policy_mean_all.mean(),
            "actor_q_min_all": q_policy_min_all.mean(),
            "actor_q_mean_all_minus_min_all": (
                q_policy_mean_all - q_policy_min_all
            ).mean(),
            **actor_saturation_metrics,
            "ent_coef": alpha.detach(),
            "ent_coef_loss": entropy_coefficient_loss.detach(),
        }
        if self.log_ent_coef is not None:
            metrics["ent_coef_floor_hit"] = (
                self.log_ent_coef.detach().reshape(())
                <= math.log(_ENTROPY_COEF_MIN)
            ).to(dtype=torch.float32)
        if self.actor_loss_scale_enabled:
            metrics["actor_loss_scale"] = self._actor_loss_scale_value.detach()
            metrics["actor_effective_ent_coef"] = (
                alpha / self._actor_loss_scale_value
            ).detach()
        if self.behavior_policy_kl_enabled:
            behavior_metric_prefix = (
                "behavior_policy_kl"
                if self._behavior_policy_objective == "reverse_kl"
                else "behavior_policy_action_ce"
            )
            actor_scale = (
                self._actor_loss_scale_value.reshape(())
                if self.actor_loss_scale_enabled
                else behavior_regularizer.new_ones(())
            )
            metrics.update(behavior_regularizer_metrics)
            metrics.update(
                {
                    f"{behavior_metric_prefix}_coefficient": (
                        behavior_regularizer_coefficient.detach()
                    ),
                    f"{behavior_metric_prefix}_effective_coefficient": (
                        behavior_regularizer_coefficient / actor_scale
                    ).detach(),
                    f"{behavior_metric_prefix}_weighted_loss": (
                        behavior_regularizer_coefficient
                        * behavior_regularizer
                        / actor_scale
                    ).detach(),
                }
            )
            if self._actor_q_range_enabled:
                metrics["actor_q_range_instant"] = q_range_instant.detach()
                metrics["actor_q_range_ema"] = (
                    self._actor_loss_scale_value.detach()
                )
            if self._behavior_policy_kl_schedule == "smooth":
                metrics[f"{behavior_metric_prefix}_ramp_progress"] = float(
                    smooth_progress
                )
                metrics[f"{behavior_metric_prefix}_eligible_updates"] = float(
                    self.behavior_policy_kl_eligible_updates
                )
            elif self._behavior_policy_kl_schedule == "quantile_gate":
                metrics[f"{behavior_metric_prefix}_gate_active"] = float(
                    quantile_gate_active
                )
                metrics[f"{behavior_metric_prefix}_q_threshold"] = float(
                    self.cfg.outer_behavior_policy_kl_q_threshold
                )
            elif self._behavior_policy_kl_schedule == "dual":
                metrics.update(
                    {
                        "behavior_policy_kl_dual_coefficient": (
                            dual_coefficient_after
                        ),
                        "behavior_policy_kl_dual_log_coefficient": (
                            self.log_behavior_policy_kl_coef.detach().reshape(())
                        ),
                        "behavior_policy_kl_dual_violation": (
                            behavior_kl_dual_violation.detach()
                        ),
                        "behavior_policy_kl_dual_loss": (
                            behavior_kl_dual_loss.detach()
                        ),
                        "behavior_policy_kl_dual_updated": (
                            behavior_kl_dual_updated.detach()
                        ),
                        "behavior_policy_kl_dual_updates": float(
                            self.behavior_policy_kl_dual_updates
                        ),
                        "behavior_policy_kl_dual_cap_hit": (
                            behavior_kl_dual_cap_hit.detach()
                        ),
                    }
                )
        return metrics

    def _outer_update_kernel(
        self,
        initial_obs,
        action,
        reward,
        terminated,
        next_z_targets,
        td_targets,
    ):
        """Pure fixed-horizon world/critic loss region."""
        z = self.model.encode(initial_obs)
        latent_states = [z]
        consistency_errors = []
        for recorded_action, next_z_target in zip(
            action.unbind(0), next_z_targets.unbind(0)
        ):
            z = self.model.next(z, recorded_action)
            consistency_error = F.mse_loss(z, next_z_target)
            consistency_errors.append(consistency_error)
            latent_states.append(z)
        latent_states = torch.stack(latent_states, dim=0)
        consistency_per_time = torch.stack(consistency_errors, dim=0)
        consistency_loss = td_math.reduce_temporal_loss(
            consistency_errors,
            self.cfg.rho,
            normalization=self.cfg.temporal_loss_normalization,
            reference_horizon=self.cfg.temporal_loss_reference_horizon,
            legacy_order="sequential",
            weights=self._transition_temporal_weights,
        )

        rollout_states = latent_states[:-1]
        rollout_joint = self.model.joint_input(rollout_states, action)
        reward_predictions = self.model.reward_from_joint(rollout_joint)
        q_predictions = self.model.q_predictions_from_joint(rollout_joint)
        termination_prediction = (
            self.model.termination(latent_states[1:], unnormalized=True)
            if self.cfg.episodic
            else None
        )

        reward_per_sample = td_math.soft_ce(
            reward_predictions, reward, self.cfg
        )
        reward_per_time = reward_per_sample.mean(
            dim=tuple(range(1, reward_per_sample.ndim))
        )
        reward_loss = td_math.reduce_temporal_loss(
            reward_per_time,
            self.cfg.rho,
            normalization=self.cfg.temporal_loss_normalization,
            reference_horizon=self.cfg.temporal_loss_reference_horizon,
            legacy_order="vector_sum_divide",
            weights=self._transition_temporal_weights,
        )
        critic_per_sample = self.model.critic_loss(
            q_predictions, td_targets, reduction="none"
        )
        critic_per_time = critic_per_sample.mean(
            dim=(0,) + tuple(range(2, critic_per_sample.ndim))
        )
        critic_loss = td_math.reduce_temporal_loss(
            critic_per_time,
            self.cfg.rho,
            normalization=self.cfg.temporal_loss_normalization,
            reference_horizon=self.cfg.temporal_loss_reference_horizon,
            legacy_order="vector_sum_divide",
            weights=self._transition_temporal_weights,
        )

        if self.cfg.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(
                termination_prediction, terminated
            )
        else:
            termination_loss = torch.zeros((), device=self.device)
        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.termination_coef * termination_loss
            + self.cfg.critic_coef * critic_loss
        )
        return (
            latent_states,
            reward_predictions,
            q_predictions,
            termination_prediction,
            consistency_per_time,
            consistency_loss,
            reward_loss,
            critic_loss,
            termination_loss,
            total_loss,
        )

    def _update(
        self,
        obs,
        action,
        reward,
        terminated,
        behavior_pre_tanh_mean=None,
        behavior_log_std=None,
        behavior_policy_valid=None,
    ):
        """One TD-MPC2-style model update with configurable Q regression."""
        with torch.no_grad():
            next_z_targets = self.model.encode(obs[1:])
            td_targets = self._soft_td_target(next_z_targets, reward, terminated)

        self.model.train()
        (
            latent_states,
            reward_predictions,
            q_predictions,
            termination_prediction,
            consistency_per_time,
            consistency_loss,
            reward_loss,
            critic_loss,
            termination_loss,
            total_loss,
        ) = self._outer_update_region(
            obs[0],
            action,
            reward,
            terminated,
            next_z_targets,
            td_targets,
        )

        value_equivalence_info = {}
        value_equivalence_loss_coef = float(
            self.cfg.value_equivalence_loss_coef
        )
        if value_equivalence_loss_coef > 0.0:
            value_equivalence_loss, value_equivalence_depth_losses = (
                self._value_equivalence_loss(
                    latent_states,
                    next_z_targets,
                    loss_update=self.num_updates + 1,
                )
            )
            weighted_value_equivalence_loss = (
                value_equivalence_loss_coef * value_equivalence_loss
            )
            total_loss = total_loss + weighted_value_equivalence_loss
            value_equivalence_info.update(
                {
                    "value_equivalence_loss": value_equivalence_loss.detach(),
                    "value_equivalence_weighted_loss": (
                        weighted_value_equivalence_loss.detach()
                    ),
                }
            )
            for depth, depth_loss in enumerate(
                value_equivalence_depth_losses.unbind(0), start=1
            ):
                value_equivalence_info[
                    f"value_equivalence_loss_depth_{depth}"
                ] = depth_loss.detach()
        if self._should_run_value_equivalence_diagnostics():
            value_equivalence_info.update(
                self._value_equivalence_diagnostics(
                    latent_states,
                    reward_predictions,
                    termination_prediction,
                    next_z_targets,
                    reward,
                    terminated,
                    diagnostic_update=self.num_updates + 1,
                )
            )

        self.optim.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._world_critic_params, float(self.cfg.grad_clip_norm)
        )
        self.optim.step()

        actor_info = self._update_actor(
            latent_states.detach(),
            behavior_pre_tanh_mean=behavior_pre_tanh_mean,
            behavior_log_std=behavior_log_std,
            behavior_policy_valid=behavior_policy_valid,
        )
        if self.num_updates % int(self.cfg.target_update_interval) == 0:
            self.model.soft_update_target_Q()
        self.num_updates += 1
        self.outer_version += 1
        self.inner_engine.mark_outer_update(self.outer_version)

        self.model.eval()
        reward_values = td_math.two_hot_inv(reward_predictions.detach(), self.cfg)
        q_values = self.model.q_backend.decode(q_predictions.detach())
        q_target_clip_fraction = torch.zeros((), device=self.device)
        if self.cfg.q_representation == "distributional":
            symlog_target = td_math.symlog(td_targets.detach())
            q_target_clip_fraction = (
                (symlog_target <= float(self.cfg.q_vmin))
                | (symlog_target >= float(self.cfg.q_vmax))
            ).float().mean()
        info = {
            "consistency_loss": consistency_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "termination_loss": termination_loss.detach(),
            "total_loss": total_loss.detach(),
            "grad_norm": torch.as_tensor(grad_norm).detach(),
            "q_target_mean": td_targets.detach().mean(),
            "q_mean": q_values.mean(),
            "q_abs_mean": q_values.abs().mean(),
            "alpha_to_abs_q": self.alpha.detach()
            / q_values.abs().mean().clamp_min(1e-8),
            "q_target_clip_fraction": q_target_clip_fraction,
            "td_error_abs_mean": (
                q_values - td_targets.detach().unsqueeze(0)
            ).abs().mean(),
            "reward_pred_mean": reward_values.mean(),
            "reward_target_mean": reward.detach().mean(),
            "reward_abs_mean": reward.detach().abs().mean(),
            "num_updates": float(self.num_updates),
            "compile_fallback": float(
                self.model._Qs.compile_failed
                or self.model._target_Qs.compile_failed
                or self._outer_update_region.failed
            ),
            "compile_outer_update_fallback": float(
                self._outer_update_region.failed
            ),
        }
        for depth in range(int(self.cfg.train_unroll_horizon)):
            q_at_depth = q_values[:, depth]
            info[f"consistency_error_depth_{depth + 1}"] = (
                consistency_per_time[depth].detach()
            )
            info[f"reward_error_depth_{depth + 1}"] = (
                reward_values[depth] - reward[depth]
            ).abs().mean()
            info[f"q_error_depth_{depth + 1}"] = (
                q_at_depth - td_targets[depth].detach().unsqueeze(0)
            ).abs().mean()
            info[f"q_head_disagreement_depth_{depth + 1}"] = q_at_depth.std(
                dim=0, unbiased=False
            ).mean()
        if self.cfg.episodic:
            info.update(
                td_math.termination_statistics(
                    torch.sigmoid(termination_prediction[-1]).detach(),
                    terminated[-1].detach(),
                )
            )
        info.update(actor_info)
        info.update(value_equivalence_info)
        return info

    def update(self, buffer):
        if self.behavior_policy_kl_enabled:
            (
                obs,
                action,
                reward,
                terminated,
                task,
                behavior_pre_tanh_mean,
                behavior_log_std,
                behavior_policy_valid,
            ) = buffer.sample(include_behavior_policy=True)
        else:
            obs, action, reward, terminated, task = buffer.sample()
            behavior_pre_tanh_mean = None
            behavior_log_std = None
            behavior_policy_valid = None
        if task is not None:
            raise NotImplementedError(
                "AMBI-TD-MPC2 currently supports single-task training only."
            )
        if (
            bool(getattr(self.cfg, "compile", False))
            and self.device.type == "cuda"
            and hasattr(torch, "compiler")
            and hasattr(torch.compiler, "cudagraph_mark_step_begin")
        ):
            torch.compiler.cudagraph_mark_step_begin()
        return self._update(
            obs,
            action,
            reward,
            terminated,
            behavior_pre_tanh_mean=behavior_pre_tanh_mean,
            behavior_log_std=behavior_log_std,
            behavior_policy_valid=behavior_policy_valid,
        )
