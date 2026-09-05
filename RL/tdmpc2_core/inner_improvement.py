"""Componentized AMBI inner-loop improvement strategies.

Inner learner state in this module is private to action selection. The optional
training-only prior-writeback ablation can mutate the outer actor and online
critic in place after action selection; outer optimizer state, target critics,
the rest of the world model, and the entropy coefficient remain untouched.
"""

from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields
import math
import time
import warnings

import torch

from .common import init as td_init
from .common import math as td_math
from .common.compile_regions import CompileRegion
from .common.inner_utils import (
    InnerRNG,
    allocate_across_rounds,
    updates_for_transitions,
    copy_lora_adapters_,
    lora_uses_shared_bases,
    rebase_clone_with_target_,
    rebase_lora_base_,
    reset_lora_adapters_,
    trainable_parameter_count,
)
from .common.latent_buffer import LatentReplayBuffer
from .common.lora import lorafy_copy, lorafy_shared, trainable_parameters
from .common.parameter_noise import (
    adapt_parameter_noise_stddev,
    classify_parameter_noise_actor,
    make_perturbed_actor_parameters,
    parameter_noise_action_rms,
    population_actor_mean_raw,
    sample_parameter_deltas,
)
from .common.training_state import (
    preflight_adam_state,
    preflight_module_state,
    require_exact_keys,
    require_mapping,
    require_tensor,
)


_INNER_ALPHA_FLOOR = 1e-8
_INNER_LOG_ALPHA_FLOOR = math.log(_INNER_ALPHA_FLOOR)
_PARAMETER_NOISE_FUNCTIONAL_CHUNK_SIZE = 8


@torch.no_grad()
def polyak_update(source, target, tau, *, adapters_only=False):
    tau = float(tau)
    if adapters_only:
        copy_lora_adapters_(source, target, tau=tau)
        return
    source_parameters = [parameter.detach() for parameter in source.parameters()]
    target_parameters = list(target.parameters())
    if tau == 1.0:
        torch._foreach_copy_(target_parameters, source_parameters)
    else:
        torch._foreach_lerp_(target_parameters, source_parameters, tau)
    for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
        target_buffer.copy_(source_buffer)


@dataclass
class InnerWorkspace:
    actor: torch.nn.Module | None = None
    actor_anchor: torch.nn.Module | None = None
    critic: torch.nn.Module | None = None
    critic_anchor: torch.nn.Module | None = None
    critic_target: torch.nn.Module | None = None
    actor_target: torch.nn.Module | None = None
    actor_optim: torch.optim.Optimizer | None = None
    critic_optim: torch.optim.Optimizer | None = None
    log_alpha: torch.nn.Parameter | None = None
    alpha_fixed: torch.Tensor | None = None
    temperature_optim: torch.optim.Optimizer | None = None
    replay: LatentReplayBuffer | None = None
    outer_version: int = -1
    critic_steps: int = 0
    critic_lifetime_steps: int = 0
    actor_steps: int = 0
    actor_lifetime_steps: int = 0
    temperature_steps: int = 0
    temperature_lifetime_steps: int = 0
    target_steps: int = 0
    critic_target_steps: int = 0
    actor_target_steps: int = 0
    replay_draws: int = 0
    policy_evaluations: int = 0
    q_evaluations: int = 0
    sampled_ids: list[torch.Tensor] = field(default_factory=list)
    actor_params: list[torch.nn.Parameter] = field(default_factory=list)
    critic_params: list[torch.nn.Parameter] = field(default_factory=list)
    actor_trainable_count: int = 0
    critic_trainable_count: int = 0
    # Random-explorer components are deliberately action-local.  Keeping them
    # in the same workspace/pool lets the hot allocations be reused without
    # ever turning the freshly initialized explorer into persistent state.
    explorer_actor: torch.nn.Module | None = None
    explorer_actor_optim: torch.optim.Optimizer | None = None
    explorer_actor_params: list[torch.nn.Parameter] = field(default_factory=list)
    explorer_actor_trainable_count: int = 0
    explorer_actor_steps: int = 0
    explorer_actor_lifetime_steps: int = 0
    explorer_critic: torch.nn.Module | None = None
    explorer_critic_target: torch.nn.Module | None = None
    explorer_critic_optim: torch.optim.Optimizer | None = None
    explorer_critic_params: list[torch.nn.Parameter] = field(default_factory=list)
    explorer_critic_trainable_count: int = 0
    explorer_critic_steps: int = 0
    explorer_critic_lifetime_steps: int = 0
    explorer_critic_target_steps: int = 0
    explorer_log_alpha: torch.nn.Parameter | None = None
    explorer_alpha_fixed: torch.Tensor | None = None
    explorer_temperature_optim: torch.optim.Optimizer | None = None
    explorer_temperature_steps: int = 0
    explorer_temperature_lifetime_steps: int = 0
    sampled_sources: list[torch.Tensor] = field(default_factory=list)
    primary_rollouts: int = 0
    explorer_rollouts: int = 0
    primary_transitions: int = 0
    explorer_transitions: int = 0


class InnerImprovementEngine:
    """Run none/SAC/TD3 inner improvement behind ``AMBI.agent.act``."""

    def __init__(self, agent):
        self.agent = agent
        self.cfg = agent.cfg
        self.model = agent.model
        self.device = agent.device
        self.rng = InnerRNG(self.cfg.seed, self.device)
        self.state = InnerWorkspace()
        # Action-scoped state is logically expired after every action, but its
        # allocations can be reset and reused without becoming observable.
        self._action_pool = InnerWorkspace()
        self.action_index = 0
        self.episode_index = 0
        self._mppi_prev_mean = None
        self._collect_diagnostics = True
        self._active_trace = None
        self._pending_timers = {}
        self._parameter_noise_spec = None
        self._clear_parameter_noise_action_state()
        self._initialize_compile_regions()

    def _initialize_compile_regions(self):
        enabled = bool(getattr(self.cfg, "compile", False))
        strict = bool(getattr(self.cfg, "compile_strict", False))
        operator = str(self.cfg.inner_operator)
        critic_kernel = (
            self._td3_critic_kernel if operator == "td3" else self._sac_critic_kernel
        )
        if operator == "td3":
            actor_kernel = self._td3_actor_kernel
        elif self._sac_actor_loss_scale_enabled:
            actor_kernel = self._scaled_sac_actor_kernel
        else:
            # Keep the feature-off compiled callable's input contract free of
            # an actor-loss-scale argument.
            actor_kernel = self._sac_actor_kernel
        self._compile_regions = {
            "rollout": CompileRegion(
                "fixed-shape inner rollout",
                self._dense_rollout_kernel,
                enabled=enabled,
                strict=strict,
            ),
            "critic": CompileRegion(
                "inner critic",
                critic_kernel,
                enabled=enabled,
                strict=strict,
            ),
            "actor": CompileRegion(
                "inner actor",
                actor_kernel,
                enabled=enabled,
                strict=strict,
            ),
        }
        if getattr(self.cfg, "inner_finite_horizon", False):
            self._compile_regions["prior_value"] = CompileRegion(
                "inner horizon prior value", self._prior_bootstrap,
                enabled=enabled, strict=strict,
            )

    @property
    def alpha(self):
        if self.state.log_alpha is not None:
            return self.state.log_alpha.exp().clamp_min(_INNER_ALPHA_FLOOR)
        if self.state.alpha_fixed is not None:
            if str(self.cfg.inner_temperature_mode) == "inherit_outer":
                alpha = self.state.alpha_fixed
                if getattr(self.agent, "log_ent_coef", None) is not None:
                    return alpha.clamp_min(_INNER_ALPHA_FLOOR)
                return alpha
            return self.state.alpha_fixed
        alpha = self.agent.alpha.detach()
        if (
            str(self.cfg.inner_operator) == "sac"
            and (
                str(self.cfg.inner_temperature_mode) == "auto"
                or (
                    str(self.cfg.inner_temperature_mode) == "inherit_outer"
                    and getattr(self.agent, "log_ent_coef", None) is not None
                )
            )
        ):
            return alpha.clamp_min(_INNER_ALPHA_FLOOR)
        return alpha

    @property
    def _explorer_mode(self):
        return str(getattr(self.cfg, "inner_explorer_mode", "none"))

    @property
    def _explorer_active(self):
        """Whether replay contains a labelled auxiliary population."""
        return self._explorer_mode != "none"

    @property
    def _materialized_explorer_active(self):
        """Whether the auxiliary population owns a concrete actor module."""
        return self._explorer_mode in {
            "frozen_random",
            "shared_mixture",
            "separate_critics",
        }

    @property
    def _parameter_noise_active(self):
        return self._explorer_mode == "adaptive_param_noise"

    @property
    def explorer_alpha(self):
        """Entropy coefficient owned by R in the separate-critic variant."""
        state = self.state
        if state.explorer_log_alpha is not None:
            return state.explorer_log_alpha.exp().clamp_min(_INNER_ALPHA_FLOOR)
        if state.explorer_alpha_fixed is not None:
            return state.explorer_alpha_fixed
        # Shared-mixture and frozen modes intentionally have no R alpha.
        return self.alpha

    @property
    def _sac_actor_loss_scale_enabled(self):
        return (
            str(self.cfg.inner_operator) == "sac"
            and str(
                getattr(self.cfg, "sac_actor_loss_scale_mode", "none")
            )
            == "tdmpc2_percentile_range"
        )

    @property
    def _uses_canonical_schedule(self):
        return (
            str(getattr(self.cfg, "inner_schedule_mode", "legacy"))
            == "canonical"
        )

    @property
    def _uses_component_update_schedule(self):
        return self._uses_canonical_schedule and bool(
            getattr(self.cfg, "inner_component_update_schedule", False)
        )

    @property
    def _uses_steps_per_update(self):
        return self._uses_canonical_schedule and (
            getattr(self.cfg, "inner_steps_per_update", None) is not None
        )

    @property
    def _mppi_iterations(self):
        return int(
            getattr(self.cfg, "inner_mppi_iterations", self.cfg.inner_rounds)
        )

    def _canonical_schedule_has_updates(self):
        if not self._uses_canonical_schedule:
            return False
        if self._uses_steps_per_update:
            return int(self.cfg.inner_critic_updates_per_action) > 0
        if self._uses_component_update_schedule:
            return any(
                int(getattr(self.cfg, key, 0)) > 0
                for key in (
                    "inner_critic_updates_per_round",
                    "inner_actor_updates_per_round",
                )
            )
        updates = getattr(self.cfg, "inner_updates_per_round", 0)
        return updates == "auto" or int(updates) > 0

    def _component_has_updates(self, component):
        """Whether an optimizer is needed for this action's resolved schedule."""
        cfg = self.cfg
        if self._uses_canonical_schedule:
            if self._uses_component_update_schedule:
                if component == "temperature":
                    return (
                        int(cfg.inner_actor_updates_per_round) > 0
                        and cfg.inner_operator == "sac"
                        and str(cfg.inner_temperature_mode) == "auto"
                    )
                return int(
                    getattr(cfg, f"inner_{component}_updates_per_round")
                ) > 0
            if not self._canonical_schedule_has_updates():
                return False
            if component == "temperature":
                return (
                    cfg.inner_operator == "sac"
                    and str(cfg.inner_temperature_mode) == "auto"
                )
            return str(getattr(cfg, f"inner_{component}_adaptation")) != "frozen"
        return int(getattr(cfg, f"inner_{component}_updates_per_action")) > 0

    def _resolved_inner_target_entropy(self):
        value = getattr(self.cfg, "inner_target_entropy", "auto")
        if value == "inherit_outer":
            if hasattr(self.agent, "target_entropy"):
                return float(self.agent.target_entropy)
            value = getattr(self.cfg, "target_entropy", "auto")
        if value == "auto":
            return -float(self.cfg.action_dim)
        return float(value)

    def _initial_inner_alpha(self):
        initialization = str(
            getattr(self.cfg, "inner_temperature_initialization", "fixed")
        )
        if initialization == "inherit_outer":
            initial_alpha = self.agent.alpha.detach().reshape(-1)[0]
        elif initialization == "fixed":
            initial_alpha = torch.as_tensor(
                float(self.cfg.inner_temperature), device=self.device
            )
        else:
            raise ValueError(
                "inner_temperature_initialization must be 'inherit_outer' or "
                f"'fixed', got {initialization!r}."
            )
        valid = torch.isfinite(initial_alpha) & (initial_alpha >= 0)
        if not bool(valid.item()):
            raise ValueError(
                "The initial inner entropy coefficient must be finite and "
                "non-negative before flooring, "
                f"got {float(initial_alpha.detach().item())}."
            )
        return initial_alpha.clamp_min(_INNER_ALPHA_FLOOR)

    def clear_all(self):
        self.state = InnerWorkspace()
        self._action_pool = InnerWorkspace()
        self.rng = InnerRNG(self.cfg.seed, self.device)
        self.action_index = 0
        self.episode_index = 0
        self._mppi_prev_mean = None
        self._pending_timers = {}
        self._parameter_noise_spec = None
        self._clear_parameter_noise_action_state()
        # A checkpoint load invalidates action-local module identities. Rebuild
        # only the non-serialized compile callables; model/checkpoint keys stay
        # untouched and ordinary action/episode resets retain their cache.
        self._initialize_compile_regions()

    def reset_for_evaluation(self, seed, *, reuse_action_pool=False):
        """Reset evaluation state and RNG, optionally retaining safe allocations.

        Paired controller evaluation reuses one engine that is never the live
        training engine.  Each evaluation episode must nevertheless begin from
        a fresh root-local workspace and an episode-private RNG stream. The
        default discards allocations. Opt-in reuse retains only a single-policy,
        fully action-scoped allocation pool; ordinary workspace preparation
        restores priors, target networks, optimizer moments, replay and alpha
        before use. Keeping module identities also avoids recompiling Dynamo
        guards after every root. Other lifecycles still discard their pools.
        """

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Evaluation inner-engine seed must be non-negative.")
        if not isinstance(reuse_action_pool, bool):
            raise TypeError("reuse_action_pool must be bool.")
        if self.rng._action_fork_depth != 0:
            raise RuntimeError(
                "Cannot reset the evaluation inner engine during an active action."
            )
        retain_pool = (
            reuse_action_pool
            and str(self.cfg.inner_operator) in {"sac", "td3"}
            and not self._explorer_active
            and all(
                str(getattr(self.cfg, f"inner_{component}_scope")) == "action"
                for component in (
                    "actor", "critic", "temperature", "replay",
                    "actor_optimizer", "critic_optimizer", "temperature_optimizer",
                )
            )
        )
        self.state = InnerWorkspace()
        if not retain_pool:
            self._action_pool = InnerWorkspace()
        self.rng = InnerRNG(seed, self.device)
        self.action_index = 0
        self.episode_index = 0
        self._mppi_prev_mean = None
        self._collect_diagnostics = True
        self._pending_timers = {}
        self._clear_parameter_noise_action_state()
        return self

    def _clear_parameter_noise_action_state(self):
        """Drop action-local adaptive-noise state and metric accumulators."""
        self._parameter_noise_stddev = None
        self._parameter_noise_sigma_values = []
        self._parameter_noise_calibration_rms = []
        self._parameter_noise_calibration_hits = []
        self._parameter_noise_sigma_bound_hits = []
        self._parameter_noise_behavior_action_rms = []
        self._parameter_noise_saturation_sum = None
        self._parameter_noise_saturation_count = 0
        self._parameter_noise_calibration_probes = 0
        self._parameter_noise_calibration_policy_evaluations = 0

    def _reset_parameter_noise_action_state(self):
        self._clear_parameter_noise_action_state()
        if self._parameter_noise_active:
            initial = float(self.cfg.inner_param_noise_sigma_init)
            self._parameter_noise_stddev = initial

    def _timer_start(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def _timer_stop(self, key, start):
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            value = (start, end)
        else:
            value = time.perf_counter() - start
        self._pending_timers.setdefault(key, []).append(value)

    def finalize_timing_metrics(self, metrics):
        """Resolve phase timers after the caller's unavoidable action sync."""
        for key, measurements in self._pending_timers.items():
            if self.device.type == "cuda":
                metrics[key] = sum(
                    start.elapsed_time(end) / 1000.0
                    for start, end in measurements
                )
            else:
                metrics[key] = sum(measurements)
        self._pending_timers = {}
        return metrics

    def reset_episode(self):
        """Clear action/episode state while preserving explicitly run-scoped state."""
        self.episode_index += 1
        self._clear_expired(t0=True, include_action=True)
        if str(self.cfg.inner_mppi_warm_start_scope) != "run":
            self._mppi_prev_mean = None

    def prepare_training_resume_boundary(self):
        """Canonicalize an episode boundary for exact process continuation.

        Ordinary episode resets retain action-scoped allocations as an
        unobservable performance cache. A restarted process cannot retain that
        cache, however, and constructing a replacement can consume private
        initialization RNG. Resume capture therefore drops it in the source
        process as well and clears every action-local diagnostic/counter field
        before serializing.
        """
        if self._pending_timers:
            raise RuntimeError(
                "AMBI inner timing must be finalized before preparing a resume boundary."
            )
        self.reset_episode()
        self._action_pool = InnerWorkspace()
        self._clear_action_transients()
        self._require_resume_boundary()
        return self

    _PERSISTENT_COUNTER_FIELDS = (
        "outer_version",
        "critic_lifetime_steps",
        "actor_lifetime_steps",
        "temperature_lifetime_steps",
    )

    _ACTION_TRANSIENT_COUNTER_FIELDS = (
        "critic_steps",
        "actor_steps",
        "temperature_steps",
        "target_steps",
        "critic_target_steps",
        "actor_target_steps",
        "replay_draws",
        "policy_evaluations",
        "q_evaluations",
        "explorer_actor_steps",
        "explorer_critic_steps",
        "explorer_critic_target_steps",
        "explorer_temperature_steps",
        "primary_rollouts",
        "explorer_rollouts",
        "primary_transitions",
        "explorer_transitions",
    )

    @staticmethod
    def _nondefault_workspace_fields(workspace):
        """Return populated fields without comparing tensors or modules."""
        empty = InnerWorkspace()
        populated = []
        for descriptor in dataclass_fields(InnerWorkspace):
            value = getattr(workspace, descriptor.name)
            default = getattr(empty, descriptor.name)
            if default is None:
                nondefault = value is not None
            elif isinstance(default, list):
                nondefault = bool(value)
            else:
                nondefault = value != default
            if nondefault:
                populated.append(descriptor.name)
        return populated

    def _clear_action_transients(self):
        for name in self._ACTION_TRANSIENT_COUNTER_FIELDS:
            setattr(self.state, name, 0)
        self.state.sampled_ids.clear()
        self.state.sampled_sources.clear()
        self._collect_diagnostics = True

    def _active_rl_workspace(self):
        return (
            str(self.cfg.inner_operator) in {"sac", "td3"}
            and int(self.cfg.inner_rounds) > 0
        )

    def _expected_persistent_workspace_fields(self, *, initialized):
        """Return the exact run-scoped component inventory after a boundary."""
        fields = {
            name: False
            for name in (
                "actor",
                "actor_anchor",
                "actor_target",
                "critic",
                "critic_anchor",
                "critic_target",
                "actor_optim",
                "critic_optim",
                "log_alpha",
                "alpha_fixed",
                "temperature_optim",
                "replay",
            )
        }
        if not initialized:
            return fields

        cfg = self.cfg
        actor_run = str(cfg.inner_actor_scope) == "run"
        critic_run = str(cfg.inner_critic_scope) == "run"
        temperature_run = str(cfg.inner_temperature_scope) == "run"
        if actor_run:
            fields["actor"] = True
            shared_lora = (
                str(cfg.inner_actor_adaptation) == "lora"
                and bool(cfg.inner_rebase_persistent)
            )
            needs_anchor = (
                float(cfg.inner_outer_policy_kl_coef) > 0.0
                or float(cfg.inner_outer_action_l2_coef) > 0.0
            )
            fields["actor_anchor"] = not shared_lora or needs_anchor
            fields["actor_target"] = str(cfg.inner_operator) == "td3"
        if critic_run:
            fields["critic"] = True
            fields["critic_anchor"] = not (
                str(cfg.inner_critic_adaptation) == "lora"
                and bool(cfg.inner_rebase_persistent)
            )
            fields["critic_target"] = (
                str(cfg.inner_bootstrap_source) == "inner_target"
            )
        fields["actor_optim"] = (
            actor_run
            and str(cfg.inner_actor_optimizer_scope) == "run"
            and str(cfg.inner_actor_adaptation) != "frozen"
            and self._component_has_updates("actor")
        )
        fields["critic_optim"] = (
            critic_run
            and str(cfg.inner_critic_optimizer_scope) == "run"
            and str(cfg.inner_critic_adaptation) != "frozen"
            and self._component_has_updates("critic")
        )
        if temperature_run:
            if str(cfg.inner_temperature_mode) == "auto":
                fields["log_alpha"] = True
            else:
                fields["alpha_fixed"] = True
        fields["temperature_optim"] = (
            temperature_run
            and str(cfg.inner_temperature_optimizer_scope) == "run"
            and self._component_has_updates("temperature")
        )
        fields["replay"] = str(cfg.inner_replay_scope) == "run"
        return fields

    def _inventory_mismatch(self, workspace, *, initialized, serialized=False):
        expected = self._expected_persistent_workspace_fields(
            initialized=initialized
        )
        return {
            name: {"expected": present, "actual": actual}
            for name, present in expected.items()
            if (actual := (
                workspace[name] is not None
                if serialized
                else getattr(workspace, name) is not None
            ))
            is not present
        }

    @staticmethod
    def _module_training_state(module, *, outer=None):
        if module is None:
            return None
        if outer is not None and module is outer:
            return {"kind": "outer-reference"}
        return {
            "kind": "module",
            "state": module.state_dict(),
            "training": bool(module.training),
        }

    def _require_resume_boundary(self):
        """Reject snapshots that would silently discard transient inner state."""
        if self._pending_timers:
            raise RuntimeError(
                "AMBI inner timing must be finalized before capturing training state."
            )
        pooled = self._nondefault_workspace_fields(self._action_pool)
        if pooled:
            raise RuntimeError(
                "AMBI action allocation state must be cleared at the exact resume "
                f"boundary; pooled fields are still live: {sorted(pooled)}."
            )
        live_action_counters = {
            name: int(getattr(self.state, name))
            for name in self._ACTION_TRANSIENT_COUNTER_FIELDS
            if int(getattr(self.state, name)) != 0
        }
        if live_action_counters or self.state.sampled_ids:
            raise RuntimeError(
                "AMBI action-local counters and sampled IDs must be cleared at "
                "the exact resume boundary."
            )
        if self._collect_diagnostics is not True:
            raise RuntimeError(
                "AMBI action-local diagnostic mode must be reset at the exact "
                "resume boundary."
            )
        explorer_live = {
            name: value
            for name in (
                "explorer_actor",
                "explorer_actor_optim",
                "explorer_critic",
                "explorer_critic_target",
                "explorer_critic_optim",
                "explorer_log_alpha",
                "explorer_alpha_fixed",
                "explorer_temperature_optim",
            )
            if (value := getattr(self.state, name)) is not None
        }
        explorer_counts = {
            name: int(getattr(self.state, name))
            for name in (
                "explorer_actor_trainable_count",
                "explorer_actor_lifetime_steps",
                "explorer_critic_trainable_count",
                "explorer_critic_lifetime_steps",
                "explorer_temperature_lifetime_steps",
            )
            if int(getattr(self.state, name)) != 0
        }
        if (
            explorer_live
            or explorer_counts
            or self.state.explorer_actor_params
            or self.state.explorer_critic_params
            or self.state.sampled_sources
        ):
            raise RuntimeError(
                "Random-explorer state must be empty at the exact action/episode "
                "resume boundary."
            )
        initialized_rl = self._active_rl_workspace() and self.action_index > 0
        if initialized_rl:
            if self.state.outer_version < 0:
                raise RuntimeError(
                    "Initialized AMBI inner state lacks an outer-version anchor."
                )
        elif self.state.outer_version != -1:
            raise RuntimeError(
                "Uninitialized AMBI inner state has an unexpected outer version."
            )
        mismatched = self._inventory_mismatch(
            self.state, initialized=initialized_rl
        )
        if mismatched:
            raise RuntimeError(
                "AMBI training state requires a canonical episode boundary; "
                "persistent workspace inventory is incomplete or unexpected: "
                f"{mismatched}."
            )
        expected_mppi = (
            str(self.cfg.inner_operator) == "mppi"
            and self.action_index > 0
            and str(self.cfg.inner_mppi_warm_start_scope) == "run"
        )
        if (self._mppi_prev_mean is not None) is not expected_mppi:
            raise RuntimeError(
                "AMBI MPPI warm-start inventory does not match its configured "
                "lifetime and action history."
            )

    def training_state_dict(self):
        """Return persistent inner scientific state at an episode boundary."""
        self._require_resume_boundary()
        state = self.state
        payload = {
            "schema": "ambi-inner-engine-training-state",
            # Version 1 remains byte-for-byte compatible for feature-off runs.
            # Version 2 records the active population identity; all R modules
            # are action-local and are therefore intentionally absent here.
            "version": 2 if self._explorer_active else 1,
            "action_index": int(self.action_index),
            "episode_index": int(self.episode_index),
            "rng": self.rng.training_state_dict(),
            "workspace": {
                "actor": self._module_training_state(state.actor),
                "actor_anchor": self._module_training_state(
                    state.actor_anchor, outer=self.model._pi
                ),
                "actor_target": self._module_training_state(state.actor_target),
                "critic": self._module_training_state(state.critic),
                "critic_anchor": self._module_training_state(
                    state.critic_anchor, outer=self.model._Qs
                ),
                "critic_target": self._module_training_state(state.critic_target),
                "actor_optim": (
                    None if state.actor_optim is None else state.actor_optim.state_dict()
                ),
                "critic_optim": (
                    None if state.critic_optim is None else state.critic_optim.state_dict()
                ),
                "log_alpha": (
                    None if state.log_alpha is None else state.log_alpha.detach()
                ),
                "alpha_fixed": (
                    None if state.alpha_fixed is None else state.alpha_fixed.detach()
                ),
                "temperature_optim": (
                    None
                    if state.temperature_optim is None
                    else state.temperature_optim.state_dict()
                ),
                "replay": (
                    None if state.replay is None else state.replay.training_state_dict()
                ),
                "counters": {
                    field: int(getattr(state, field))
                    for field in self._PERSISTENT_COUNTER_FIELDS
                },
            },
            "mppi_prev_mean": self._mppi_prev_mean,
        }
        if self._explorer_active:
            payload["explorer_mode"] = self._explorer_mode
        return payload

    def _load_module_candidate(
        self,
        payload,
        *,
        name,
        factory,
        outer_reference=None,
    ):
        if payload is None:
            return None
        payload = require_mapping(payload, name)
        kind = payload.get("kind")
        if kind == "outer-reference":
            require_exact_keys(payload, {"kind"}, name)
            if outer_reference is None:
                raise ValueError(f"{name} cannot reference an outer module.")
            return outer_reference
        if kind != "module":
            raise ValueError(f"{name} has unknown module payload kind {kind!r}.")
        payload = require_exact_keys(
            payload, {"kind", "state", "training"}, name
        )
        if not isinstance(payload["training"], bool):
            raise TypeError(f"{name}.training must be bool.")
        module = factory()
        preflight_module_state(module, payload["state"], name)
        try:
            module.load_state_dict(payload["state"], strict=True)
        except (KeyError, TypeError, RuntimeError, ValueError) as exc:
            raise ValueError(f"{name} is incompatible: {exc}") from exc
        module.train(payload["training"])
        return module

    @staticmethod
    def _load_optimizer_candidate(optimizer, payload, name, *, expected_steps):
        if optimizer is None:
            raise ValueError(f"{name} has no trainable parameters.")
        preflight_adam_state(
            optimizer, payload, name, expected_steps=expected_steps
        )
        try:
            optimizer.load_state_dict(payload)
        except (KeyError, TypeError, RuntimeError, ValueError) as exc:
            raise ValueError(f"{name} is incompatible: {exc}") from exc
        return optimizer

    @staticmethod
    def _validate_index(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return int(value)

    def _preflight_training_state_dict(self, state):
        state = require_mapping(state, "AMBI inner-engine training state")
        version = state.get("version")
        common_keys = {
                "schema",
                "version",
                "action_index",
                "episode_index",
                "rng",
                "workspace",
                "mppi_prev_mean",
        }
        if version == 1:
            if self._explorer_active:
                raise ValueError(
                    "Legacy AMBI inner-engine state can only be loaded with "
                    "inner_explorer_mode='none'."
                )
            expected_keys = common_keys
        elif version == 2:
            expected_keys = common_keys | {"explorer_mode"}
            if not self._explorer_active:
                raise ValueError(
                    "Random-explorer AMBI state cannot be loaded with "
                    "inner_explorer_mode='none'."
                )
            if state.get("explorer_mode") != self._explorer_mode:
                raise ValueError(
                    "AMBI random-explorer mode is incompatible: "
                    f"checkpoint={state.get('explorer_mode')!r}, "
                    f"configured={self._explorer_mode!r}."
                )
        else:
            raise ValueError("Unsupported AMBI inner-engine training-state version.")
        state = require_exact_keys(
            state,
            expected_keys,
            "AMBI inner-engine training state",
        )
        if (
            state["schema"] != "ambi-inner-engine-training-state"
            or state["version"] not in {1, 2}
        ):
            raise ValueError("Unsupported AMBI inner-engine training-state version.")
        action_index = self._validate_index(state["action_index"], "action_index")
        episode_index = self._validate_index(state["episode_index"], "episode_index")
        workspace = require_exact_keys(
            state["workspace"],
            {
                "actor",
                "actor_anchor",
                "actor_target",
                "critic",
                "critic_anchor",
                "critic_target",
                "actor_optim",
                "critic_optim",
                "log_alpha",
                "alpha_fixed",
                "temperature_optim",
                "replay",
                "counters",
            },
            "AMBI inner workspace training state",
        )
        counters = require_exact_keys(
            workspace["counters"],
            self._PERSISTENT_COUNTER_FIELDS,
            "inner workspace counters",
        )
        normalized_counters = {}
        for field in self._PERSISTENT_COUNTER_FIELDS:
            value = counters[field]
            if field == "outer_version":
                if isinstance(value, bool) or not isinstance(value, int) or value < -1:
                    raise ValueError("outer_version must be an integer greater than -2.")
                normalized_counters[field] = int(value)
            else:
                normalized_counters[field] = self._validate_index(value, field)
        initialized_rl = self._active_rl_workspace() and action_index > 0
        outer_version = normalized_counters["outer_version"]
        if initialized_rl:
            if outer_version < 0:
                raise ValueError(
                    "Initialized AMBI inner checkpoint lacks an outer-version anchor."
                )
        elif outer_version != -1:
            raise ValueError(
                "Uninitialized AMBI inner checkpoint has an unexpected outer version."
            )

        mismatched_inventory = self._inventory_mismatch(
            workspace, initialized=initialized_rl, serialized=True
        )
        if mismatched_inventory:
            raise ValueError(
                "AMBI checkpoint persistent workspace inventory is incomplete or "
                f"unexpected: {mismatched_inventory}."
            )
        # Build the complete destination off to the side. Failures below do
        # not mutate the current engine or its RNG streams.
        probe_rng = InnerRNG(self.cfg.seed, self.device)
        with probe_rng.action_fork():
            with probe_rng.fork("initialization"):
                candidate = InnerWorkspace()
                module_specs = (
                    (
                        "actor",
                        lambda: self._adapt_module(self.model._pi, "actor"),
                        None,
                    ),
                    (
                        "critic",
                        lambda: self._adapt_module(self.model._Qs, "critic"),
                        None,
                    ),
                    (
                        "actor_anchor",
                        lambda: deepcopy(self.model._pi)
                        .to(self.device)
                        .requires_grad_(False),
                        self.model._pi,
                    ),
                    (
                        "critic_anchor",
                        lambda: deepcopy(self.model._Qs)
                        .to(self.device)
                        .requires_grad_(False),
                        self.model._Qs,
                    ),
                    (
                        "actor_target",
                        lambda: deepcopy(candidate.actor)
                        .to(self.device)
                        .requires_grad_(False),
                        None,
                    ),
                    (
                        "critic_target",
                        lambda: deepcopy(candidate.critic)
                        .to(self.device)
                        .requires_grad_(False),
                        None,
                    ),
                )
                for name, factory, outer_reference in module_specs:
                    setattr(
                        candidate,
                        name,
                        self._load_module_candidate(
                            workspace[name],
                            name=f"inner {name.replace('_', ' ')}",
                            factory=factory,
                            outer_reference=outer_reference,
                        ),
                    )

        candidate.actor_params = (
            [] if candidate.actor is None else trainable_parameters(candidate.actor)
        )
        candidate.critic_params = (
            [] if candidate.critic is None else trainable_parameters(candidate.critic)
        )
        candidate.actor_trainable_count = sum(
            parameter.numel() for parameter in candidate.actor_params
        )
        candidate.critic_trainable_count = sum(
            parameter.numel() for parameter in candidate.critic_params
        )
        if workspace["actor_optim"] is not None:
            if candidate.actor is None:
                raise ValueError("Inner actor optimizer exists without an actor.")
            candidate.actor_optim = self._load_optimizer_candidate(
                self._new_optimizer(candidate.actor, "actor"),
                workspace["actor_optim"],
                "Inner actor optimizer",
                expected_steps=normalized_counters["actor_lifetime_steps"],
            )
        if workspace["critic_optim"] is not None:
            if candidate.critic is None:
                raise ValueError("Inner critic optimizer exists without a critic.")
            candidate.critic_optim = self._load_optimizer_candidate(
                self._new_optimizer(candidate.critic, "critic"),
                workspace["critic_optim"],
                "Inner critic optimizer",
                expected_steps=normalized_counters["critic_lifetime_steps"],
            )

        log_alpha, alpha_fixed = workspace["log_alpha"], workspace["alpha_fixed"]
        if log_alpha is not None and alpha_fixed is not None:
            raise ValueError("Inner temperature cannot be both learned and fixed.")
        if log_alpha is not None:
            log_alpha = require_tensor(
                log_alpha,
                "inner log_alpha",
                shape=(),
                dtype=self.agent.alpha.dtype,
            )
            if not bool(torch.isfinite(log_alpha).all().item()):
                raise ValueError("inner log_alpha must be finite.")
            candidate.log_alpha = torch.nn.Parameter(
                log_alpha.detach()
                .to(self.device)
                .clamp_min(_INNER_LOG_ALPHA_FLOOR)
                .clone()
            )
        if alpha_fixed is not None:
            temperature_mode = str(self.cfg.inner_temperature_mode)
            expected_shape = (
                self.agent.alpha.shape
                if temperature_mode == "inherit_outer"
                else torch.Size([])
            )
            alpha_fixed = require_tensor(
                alpha_fixed,
                "inner alpha_fixed",
                shape=expected_shape,
                dtype=self.agent.alpha.dtype,
            )
            if not bool(torch.isfinite(alpha_fixed).all().item()):
                raise ValueError("inner alpha_fixed must be finite.")
            if temperature_mode == "inherit_outer" and bool(
                (alpha_fixed < 0).any().item()
            ):
                raise ValueError("inherited inner alpha_fixed must be non-negative.")
            candidate.alpha_fixed = alpha_fixed.detach().to(self.device).clone()
            if temperature_mode == "inherit_outer" and (
                getattr(self.agent, "log_ent_coef", None) is not None
                or bool((candidate.alpha_fixed <= 0).any().item())
            ):
                candidate.alpha_fixed.clamp_min_(_INNER_ALPHA_FLOOR)
        if workspace["temperature_optim"] is not None:
            if candidate.log_alpha is None:
                raise ValueError(
                    "Inner temperature optimizer exists without learned log_alpha."
                )
            candidate.temperature_optim = torch.optim.Adam(
                [candidate.log_alpha],
                lr=float(self.cfg.inner_temperature_lr),
                eps=float(self.cfg.inner_adam_eps),
                capturable=False,
                foreach=self.device.type == "cuda",
            )
            self._load_optimizer_candidate(
                candidate.temperature_optim,
                workspace["temperature_optim"],
                "Inner temperature optimizer",
                expected_steps=normalized_counters["temperature_lifetime_steps"],
            )

        if workspace["replay"] is not None:
            candidate.replay = LatentReplayBuffer(
                capacity=self.cfg.inner_replay_capacity,
                latent_dim=self.cfg.latent_dim,
                action_dim=self.cfg.action_dim,
                device=self.device,
                store_horizon=bool(getattr(self.cfg, "inner_finite_horizon", False)),
            )
            candidate.replay.load_training_state_dict(workspace["replay"])

        for field, value in normalized_counters.items():
            setattr(candidate, field, value)
        candidate.sampled_ids = []

        previous_mean = state["mppi_prev_mean"]
        expected_mppi = (
            str(self.cfg.inner_operator) == "mppi"
            and action_index > 0
            and str(self.cfg.inner_mppi_warm_start_scope) == "run"
        )
        if (previous_mean is not None) is not expected_mppi:
            raise ValueError(
                "AMBI checkpoint MPPI warm-start inventory does not match its "
                "configured lifetime and action history."
            )
        if previous_mean is not None:
            previous_mean = require_tensor(
                previous_mean,
                "inner MPPI previous mean",
                shape=(self.cfg.inner_rollout_horizon, self.cfg.action_dim),
                dtype=next(self.model.parameters()).dtype,
            )
            if not previous_mean.is_floating_point():
                raise ValueError("Inner MPPI previous mean must be floating point.")
            previous_mean = previous_mean.detach().to(self.device).clone()

        candidate_rng = InnerRNG(self.cfg.seed, self.device)
        candidate_rng.load_training_state_dict(state["rng"])
        return {
            "state": candidate,
            "rng": candidate_rng,
            "action_index": action_index,
            "episode_index": episode_index,
            "mppi_prev_mean": previous_mean,
        }

    def load_training_state_dict(self, state):
        """Transactionally restore validated persistent inner state."""
        candidate = self._preflight_training_state_dict(state)
        return self._commit_training_state_candidate(candidate)

    def _commit_training_state_candidate(self, candidate):
        """Install a candidate returned by strict preflight."""
        self.state = candidate["state"]
        self._action_pool = InnerWorkspace()
        self.rng = candidate["rng"]
        self.action_index = candidate["action_index"]
        self.episode_index = candidate["episode_index"]
        self._mppi_prev_mean = candidate["mppi_prev_mean"]
        self._collect_diagnostics = True
        self._pending_timers = {}
        self._parameter_noise_spec = None
        self._clear_parameter_noise_action_state()
        self._initialize_compile_regions()
        return self

    def mark_outer_update(self, version):
        # Workspaces are refreshed lazily at the next action, after all outer
        # updates in the current real environment step have completed.
        del version

    def _scope_expires(self, scope, *, t0, include_action=True):
        scope = str(scope)
        return (include_action and scope == "action") or (t0 and scope == "episode")

    def _clear_expired(self, *, t0, include_action=True):
        state, cfg = self.state, self.cfg
        if self._scope_expires(cfg.inner_actor_scope, t0=t0, include_action=include_action):
            if str(cfg.inner_actor_scope) == "action" and state.actor is not None:
                self._action_pool.actor = state.actor
                self._action_pool.actor_anchor = state.actor_anchor
                self._action_pool.actor_target = state.actor_target
                self._action_pool.actor_optim = state.actor_optim
                self._action_pool.actor_params = state.actor_params
                self._action_pool.actor_trainable_count = (
                    state.actor_trainable_count
                )
            elif str(cfg.inner_actor_scope) != "action":
                # An optimizer may have a shorter lifetime than its owning
                # module (for example, episode parameters with action Adam).
                # Once the module expires, a pooled optimizer still references
                # those dead parameters and must never be attached to its
                # replacement.
                self._action_pool.actor_optim = None
            state.actor = state.actor_anchor = state.actor_target = None
            state.actor_optim = None
            state.actor_params = []
            state.actor_trainable_count = 0
            state.actor_lifetime_steps = 0
        elif self._scope_expires(
            cfg.inner_actor_optimizer_scope, t0=t0, include_action=include_action
        ):
            if (
                str(cfg.inner_actor_optimizer_scope) == "action"
                and state.actor_optim is not None
            ):
                self._action_pool.actor_optim = state.actor_optim
            state.actor_optim = None

        if self._scope_expires(cfg.inner_critic_scope, t0=t0, include_action=include_action):
            if str(cfg.inner_critic_scope) == "action" and state.critic is not None:
                self._action_pool.critic = state.critic
                self._action_pool.critic_anchor = state.critic_anchor
                self._action_pool.critic_target = state.critic_target
                self._action_pool.critic_optim = state.critic_optim
                self._action_pool.critic_params = state.critic_params
                self._action_pool.critic_trainable_count = (
                    state.critic_trainable_count
                )
            elif str(cfg.inner_critic_scope) != "action":
                self._action_pool.critic_optim = None
            state.critic = state.critic_anchor = state.critic_target = None
            state.critic_optim = None
            state.critic_params = []
            state.critic_trainable_count = 0
            state.critic_lifetime_steps = 0
        elif self._scope_expires(
            cfg.inner_critic_optimizer_scope, t0=t0, include_action=include_action
        ):
            if (
                str(cfg.inner_critic_optimizer_scope) == "action"
                and state.critic_optim is not None
            ):
                self._action_pool.critic_optim = state.critic_optim
            state.critic_optim = None

        if self._scope_expires(
            cfg.inner_temperature_scope, t0=t0, include_action=include_action
        ):
            if (
                str(cfg.inner_temperature_scope) == "action"
                and (state.log_alpha is not None or state.alpha_fixed is not None)
            ):
                self._action_pool.log_alpha = state.log_alpha
                self._action_pool.alpha_fixed = state.alpha_fixed
                self._action_pool.temperature_optim = state.temperature_optim
            else:
                self._action_pool.temperature_optim = None
            state.log_alpha = state.alpha_fixed = state.temperature_optim = None
            state.temperature_lifetime_steps = 0
        elif self._scope_expires(
            cfg.inner_temperature_optimizer_scope,
            t0=t0,
            include_action=include_action,
        ):
            if (
                str(cfg.inner_temperature_optimizer_scope) == "action"
                and state.temperature_optim is not None
            ):
                self._action_pool.temperature_optim = state.temperature_optim
            state.temperature_optim = None
            state.temperature_lifetime_steps = 0

        if self._scope_expires(cfg.inner_replay_scope, t0=t0, include_action=include_action):
            if str(cfg.inner_replay_scope) == "action" and state.replay is not None:
                self._action_pool.replay = state.replay
            state.replay = None

        # Active explorer configurations are validated as action-local.  Pool
        # the allocations, but reset every scientific value at the next root.
        if self._materialized_explorer_active and include_action:
            pool = self._action_pool
            if state.explorer_actor is not None:
                pool.explorer_actor = state.explorer_actor
                pool.explorer_actor_optim = state.explorer_actor_optim
                pool.explorer_actor_params = state.explorer_actor_params
                pool.explorer_actor_trainable_count = (
                    state.explorer_actor_trainable_count
                )
            if state.explorer_critic is not None:
                pool.explorer_critic = state.explorer_critic
                pool.explorer_critic_target = state.explorer_critic_target
                pool.explorer_critic_optim = state.explorer_critic_optim
                pool.explorer_critic_params = state.explorer_critic_params
                pool.explorer_critic_trainable_count = (
                    state.explorer_critic_trainable_count
                )
            if (
                state.explorer_log_alpha is not None
                or state.explorer_alpha_fixed is not None
            ):
                pool.explorer_log_alpha = state.explorer_log_alpha
                pool.explorer_alpha_fixed = state.explorer_alpha_fixed
                pool.explorer_temperature_optim = state.explorer_temperature_optim

            state.explorer_actor = state.explorer_actor_optim = None
            state.explorer_actor_params = []
            state.explorer_actor_trainable_count = 0
            state.explorer_actor_lifetime_steps = 0
            state.explorer_critic = state.explorer_critic_target = None
            state.explorer_critic_optim = None
            state.explorer_critic_params = []
            state.explorer_critic_trainable_count = 0
            state.explorer_critic_lifetime_steps = 0
            state.explorer_log_alpha = state.explorer_alpha_fixed = None
            state.explorer_temperature_optim = None
            state.explorer_temperature_lifetime_steps = 0

        if include_action:
            self._clear_parameter_noise_action_state()

    @staticmethod
    def _reset_optimizer(optimizer):
        """Reset Adam state in-place while preserving allocated moment tensors."""
        if optimizer is None:
            return
        optimizer.zero_grad(set_to_none=True)
        for parameter_state in optimizer.state.values():
            for key, value in parameter_state.items():
                if torch.is_tensor(value):
                    value.zero_()
                elif isinstance(value, (int, float)):
                    parameter_state[key] = type(value)(0)

    def _reset_action_component(self, component, module, outer):
        mode = str(getattr(self.cfg, f"inner_{component}_adaptation"))
        if mode == "lora":
            rebase_lora_base_(module, outer)
            reset_lora_adapters_(module)
        else:
            module.load_state_dict(outer.state_dict())
            module.requires_grad_(mode != "frozen")

    def _restore_action_component(self, component, outer):
        state, pool = self.state, self._action_pool
        module = getattr(pool, component)
        if module is None:
            return False
        setattr(state, component, module)
        setattr(pool, component, None)
        setattr(pool, f"{component}_anchor", None)
        params_name = f"{component}_params"
        count_name = f"{component}_trainable_count"
        setattr(state, params_name, getattr(pool, params_name))
        setattr(pool, params_name, [])
        setattr(state, count_name, getattr(pool, count_name))
        setattr(pool, count_name, 0)
        self._reset_action_component(component, module, outer)

        optimizer_name = f"{component}_optim"
        optimizer = getattr(pool, optimizer_name)
        setattr(state, optimizer_name, optimizer)
        setattr(pool, optimizer_name, None)
        self._reset_optimizer(optimizer)

        target_name = f"{component}_target"
        target = getattr(pool, target_name)
        setattr(state, target_name, target)
        setattr(pool, target_name, None)
        if target is not None:
            mode = str(getattr(self.cfg, f"inner_{component}_adaptation"))
            if mode == "lora":
                rebase_lora_base_(target, outer)
                copy_lora_adapters_(module, target)
            else:
                target.load_state_dict(module.state_dict())
            target.requires_grad_(False)
        return True

    def _adapt_module(self, base, component):
        mode = str(getattr(self.cfg, f"inner_{component}_adaptation"))
        if mode == "frozen":
            module = deepcopy(base).to(self.device)
            module.requires_grad_(False)
        elif mode == "clone":
            module = deepcopy(base).to(self.device)
            module.requires_grad_(True)
        elif mode == "lora":
            scope = str(getattr(self.cfg, f"inner_{component}_scope"))
            # Action-local LoRA always follows the current outer head. A
            # persistent adapter may share as well when rebasing is enabled;
            # the opt-out retains the historical frozen-base snapshot.
            factory = (
                lorafy_shared
                if scope == "action" or bool(self.cfg.inner_rebase_persistent)
                else lorafy_copy
            )
            module = factory(
                base,
                rank=getattr(self.cfg, f"inner_{component}_lora_rank"),
                scale=getattr(self.cfg, f"inner_{component}_lora_scale"),
                dropout=getattr(self.cfg, f"inner_{component}_lora_dropout"),
            ).to(self.device)
        else:
            raise ValueError(f"Unknown {component} adaptation mode: {mode!r}")
        return module

    def _new_optimizer(self, module, component):
        params = trainable_parameters(module)
        if not params:
            return None
        return torch.optim.Adam(
            params,
            lr=float(getattr(self.cfg, f"inner_{component}_lr")),
            eps=float(self.cfg.inner_adam_eps),
            # Inner lifecycle orchestration remains eager; capturable Adam is
            # measurably slower unless an optimizer is actually graphed.
            capturable=False,
            foreach=self.device.type == "cuda",
        )

    def _refresh_persistent_component(self, component, outer):
        state = self.state
        adapted = getattr(state, component)
        anchor = getattr(state, f"{component}_anchor")
        target = getattr(state, f"{component}_target", None)
        mode = str(getattr(self.cfg, f"inner_{component}_adaptation"))
        if adapted is None:
            return
        if mode == "lora":
            rebase_lora_base_(adapted, outer)
            if target is not None:
                rebase_lora_base_(target, outer)
            if anchor is not None and anchor is not outer:
                anchor.load_state_dict(outer.state_dict())
            return
        if anchor is None:
            return
        if mode == "clone":
            rebase_clone_with_target_(adapted, target, anchor, outer)
        else:
            adapted.load_state_dict(outer.state_dict())
            anchor.load_state_dict(outer.state_dict())
            if target is not None:
                target.load_state_dict(outer.state_dict())

    def _prepare_explorer_workspace(self):
        """Create/reset the action-local randomly initialized policy R."""
        if not self._materialized_explorer_active:
            return
        state, pool, cfg = self.state, self._action_pool, self.cfg
        mode = self._explorer_mode
        train_actor = mode in {"shared_mixture", "separate_critics"}

        explorer_actor = pool.explorer_actor
        if explorer_actor is None:
            # Deepcopy supplies only the architecture and device placement; the
            # next line overwrites every initialized TD-MPC parameter.
            explorer_actor = deepcopy(self.model._pi).to(self.device)
        else:
            pool.explorer_actor = None
        explorer_actor.apply(td_init.weight_init)
        # ``weight_init`` mirrors model construction for linear layers; the
        # LayerNorm defaults are supplied by their constructors and therefore
        # need an explicit reset when reusing/deep-copying an actor.
        for module in explorer_actor.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.reset_parameters()
        explorer_actor.requires_grad_(train_actor)
        explorer_actor.train(train_actor)
        state.explorer_actor = explorer_actor
        state.explorer_actor_params = (
            trainable_parameters(explorer_actor) if train_actor else []
        )
        state.explorer_actor_trainable_count = sum(
            parameter.numel() for parameter in state.explorer_actor_params
        )
        state.explorer_actor_lifetime_steps = 0

        explorer_actor_optim = pool.explorer_actor_optim
        pool.explorer_actor_optim = None
        if train_actor and int(
            getattr(cfg, "inner_explorer_actor_updates_per_action", 0)
        ) > 0:
            if explorer_actor_optim is None:
                explorer_actor_optim = self._new_optimizer(explorer_actor, "actor")
            else:
                self._reset_optimizer(explorer_actor_optim)
        else:
            explorer_actor_optim = None
        state.explorer_actor_optim = explorer_actor_optim

        if mode == "separate_critics":
            explorer_critic = pool.explorer_critic
            pool.explorer_critic = None
            if explorer_critic is None:
                explorer_critic = deepcopy(self.model._Qs).to(self.device)
            else:
                explorer_critic.load_state_dict(self.model._Qs.state_dict())
            explorer_critic.requires_grad_(True)
            explorer_critic.train(bool(cfg.inner_critic_dropout_enabled))
            state.explorer_critic = explorer_critic
            state.explorer_critic_params = trainable_parameters(explorer_critic)
            state.explorer_critic_trainable_count = sum(
                parameter.numel() for parameter in state.explorer_critic_params
            )
            state.explorer_critic_lifetime_steps = 0

            explorer_target = pool.explorer_critic_target
            pool.explorer_critic_target = None
            if explorer_target is None:
                explorer_target = deepcopy(explorer_critic).to(self.device)
            else:
                explorer_target.load_state_dict(explorer_critic.state_dict())
            explorer_target.requires_grad_(False).eval()
            state.explorer_critic_target = explorer_target
            if bool(getattr(cfg, "compile", False)):
                for critic in (explorer_critic, explorer_target):
                    if hasattr(critic, "enable_compile"):
                        critic.enable_compile(
                            strict=bool(getattr(cfg, "compile_strict", False))
                        )

            explorer_critic_optim = pool.explorer_critic_optim
            pool.explorer_critic_optim = None
            if int(getattr(cfg, "inner_explorer_critic_updates_per_action", 0)) > 0:
                if explorer_critic_optim is None:
                    explorer_critic_optim = self._new_optimizer(
                        explorer_critic, "critic"
                    )
                else:
                    self._reset_optimizer(explorer_critic_optim)
            else:
                explorer_critic_optim = None
            state.explorer_critic_optim = explorer_critic_optim

            temperature_mode = str(cfg.inner_temperature_mode)
            explorer_log_alpha = pool.explorer_log_alpha
            explorer_alpha_fixed = pool.explorer_alpha_fixed
            explorer_temperature_optim = pool.explorer_temperature_optim
            pool.explorer_log_alpha = pool.explorer_alpha_fixed = None
            pool.explorer_temperature_optim = None
            if temperature_mode == "auto":
                initial_log_alpha = self._initial_inner_alpha().log()
                if explorer_log_alpha is None:
                    explorer_log_alpha = torch.nn.Parameter(
                        initial_log_alpha.clone()
                    )
                else:
                    with torch.no_grad():
                        explorer_log_alpha.copy_(initial_log_alpha)
                explorer_alpha_fixed = None
                if int(
                    getattr(cfg, "inner_explorer_temperature_updates_per_action", 0)
                ) > 0:
                    if explorer_temperature_optim is None:
                        explorer_temperature_optim = torch.optim.Adam(
                            [explorer_log_alpha],
                            lr=float(cfg.inner_temperature_lr),
                            eps=float(cfg.inner_adam_eps),
                            capturable=False,
                            foreach=self.device.type == "cuda",
                        )
                    else:
                        self._reset_optimizer(explorer_temperature_optim)
                else:
                    explorer_temperature_optim = None
            else:
                explorer_log_alpha = explorer_temperature_optim = None
                inherited = (
                    self.agent.alpha.detach().clone()
                    if temperature_mode == "inherit_outer"
                    else torch.tensor(float(cfg.inner_temperature), device=self.device)
                )
                if explorer_alpha_fixed is None:
                    explorer_alpha_fixed = inherited
                else:
                    explorer_alpha_fixed.copy_(inherited)
            state.explorer_log_alpha = explorer_log_alpha
            state.explorer_alpha_fixed = explorer_alpha_fixed
            state.explorer_temperature_optim = explorer_temperature_optim
            state.explorer_temperature_lifetime_steps = 0
        else:
            state.explorer_critic = state.explorer_critic_target = None
            state.explorer_critic_optim = None
            state.explorer_critic_params = []
            state.explorer_critic_trainable_count = 0
            state.explorer_log_alpha = state.explorer_alpha_fixed = None
            state.explorer_temperature_optim = None

    def _prepare_workspace(self, *, t0):
        state, cfg = self.state, self.cfg
        self._clear_expired(t0=t0, include_action=True)

        actor_was_missing = state.actor is None
        critic_was_missing = state.critic is None
        actor_restored = False
        critic_restored = False
        if actor_was_missing:
            actor_restored = (
                str(cfg.inner_actor_scope) == "action"
                and self._restore_action_component("actor", self.model._pi)
            )
            if not actor_restored:
                state.actor = self._adapt_module(self.model._pi, "actor")
            needs_actor_anchor = (
                float(cfg.inner_outer_policy_kl_coef) > 0.0
                or float(cfg.inner_outer_action_l2_coef) > 0.0
            )
            if str(cfg.inner_actor_scope) == "action":
                # The immutable outer policy is the exact action-local anchor.
                state.actor_anchor = self.model._pi if needs_actor_anchor else None
            elif (
                str(cfg.inner_actor_adaptation) == "lora"
                and lora_uses_shared_bases(state.actor)
            ):
                # Shared LoRA reads the live outer policy directly. Keeping a
                # second full anchor would defeat the memory/transfer saving;
                # only regularizers need a reference at all.
                state.actor_anchor = self.model._pi if needs_actor_anchor else None
            elif not actor_restored or state.actor_anchor is None:
                state.actor_anchor = (
                    deepcopy(self.model._pi).to(self.device).requires_grad_(False)
                )
            state.actor_lifetime_steps = 0
        if critic_was_missing:
            critic_restored = (
                str(cfg.inner_critic_scope) == "action"
                and self._restore_action_component("critic", self.model._Qs)
            )
            if not critic_restored:
                state.critic = self._adapt_module(self.model._Qs, "critic")
            if str(cfg.inner_critic_scope) == "action":
                state.critic_anchor = None
            elif (
                str(cfg.inner_critic_adaptation) == "lora"
                and lora_uses_shared_bases(state.critic)
            ):
                state.critic_anchor = None
            elif not critic_restored or state.critic_anchor is None:
                state.critic_anchor = (
                    deepcopy(self.model._Qs).to(self.device).requires_grad_(False)
                )
            state.critic_lifetime_steps = 0

        if bool(getattr(cfg, "compile", False)) and hasattr(
            state.critic, "enable_compile"
        ):
            state.critic.enable_compile(
                strict=bool(getattr(cfg, "compile_strict", False))
            )

        outer_changed = state.outer_version >= 0 and state.outer_version != self.agent.outer_version
        if outer_changed and bool(cfg.inner_rebase_persistent):
            if not actor_was_missing:
                self._refresh_persistent_component("actor", self.model._pi)
            if not critic_was_missing:
                self._refresh_persistent_component("critic", self.model._Qs)

        if (
            cfg.inner_bootstrap_source == "inner_target"
            and state.critic_target is None
        ):
            state.critic_target = deepcopy(state.critic).to(self.device).requires_grad_(False)
        if (
            state.critic_target is not None
            and bool(getattr(cfg, "compile", False))
            and hasattr(state.critic_target, "enable_compile")
        ):
            state.critic_target.enable_compile(
                strict=bool(getattr(cfg, "compile_strict", False))
            )
        if cfg.inner_operator == "td3" and state.actor_target is None:
            state.actor_target = deepcopy(state.actor).to(self.device).requires_grad_(False)

        if (
            state.actor_optim is None
            and str(cfg.inner_actor_optimizer_scope) == "action"
            and self._action_pool.actor_optim is not None
        ):
            state.actor_optim = self._action_pool.actor_optim
            self._action_pool.actor_optim = None
            self._reset_optimizer(state.actor_optim)
        if (
            state.critic_optim is None
            and str(cfg.inner_critic_optimizer_scope) == "action"
            and self._action_pool.critic_optim is not None
        ):
            state.critic_optim = self._action_pool.critic_optim
            self._action_pool.critic_optim = None
            self._reset_optimizer(state.critic_optim)

        if (
            state.actor_optim is None
            and cfg.inner_actor_adaptation != "frozen"
            and self._component_has_updates("actor")
        ):
            state.actor_optim = self._new_optimizer(state.actor, "actor")
        if (
            state.critic_optim is None
            and cfg.inner_critic_adaptation != "frozen"
            and self._component_has_updates("critic")
        ):
            state.critic_optim = self._new_optimizer(state.critic, "critic")

        mode = str(cfg.inner_temperature_mode)
        if mode == "inherit_outer":
            state.log_alpha = state.temperature_optim = None
            inherited_alpha = self.agent.alpha.detach()
            if getattr(self.agent, "log_ent_coef", None) is not None:
                inherited_alpha = inherited_alpha.clamp_min(_INNER_ALPHA_FLOOR)
            if (
                str(cfg.inner_temperature_scope) == "action"
                and self._action_pool.alpha_fixed is not None
            ):
                state.alpha_fixed = self._action_pool.alpha_fixed
                self._action_pool.alpha_fixed = None
                state.alpha_fixed.copy_(inherited_alpha)
            else:
                state.alpha_fixed = inherited_alpha.clone()
        elif mode == "fixed":
            state.log_alpha = state.temperature_optim = None
            if (
                str(cfg.inner_temperature_scope) == "action"
                and self._action_pool.alpha_fixed is not None
            ):
                state.alpha_fixed = self._action_pool.alpha_fixed
                self._action_pool.alpha_fixed = None
                state.alpha_fixed.fill_(float(cfg.inner_temperature))
            else:
                state.alpha_fixed = torch.tensor(
                    float(cfg.inner_temperature), device=self.device
                )
        elif cfg.inner_operator == "sac":
            if state.log_alpha is None:
                initial_alpha = self._initial_inner_alpha()
                if (
                    str(cfg.inner_temperature_scope) == "action"
                    and self._action_pool.log_alpha is not None
                ):
                    state.log_alpha = self._action_pool.log_alpha
                    self._action_pool.log_alpha = None
                    with torch.no_grad():
                        state.log_alpha.copy_(initial_alpha.log())
                    state.temperature_optim = self._action_pool.temperature_optim
                    self._action_pool.temperature_optim = None
                    self._reset_optimizer(state.temperature_optim)
                else:
                    state.log_alpha = torch.nn.Parameter(
                        initial_alpha.log().clone()
                    )
            state.alpha_fixed = None
            if (
                state.temperature_optim is None
                and self._component_has_updates("temperature")
            ):
                state.temperature_optim = torch.optim.Adam(
                    [state.log_alpha],
                    lr=float(cfg.inner_temperature_lr),
                    eps=float(cfg.inner_adam_eps),
                    capturable=False,
                    foreach=self.device.type == "cuda",
                )

        if state.replay is None:
            if (
                str(cfg.inner_replay_scope) == "action"
                and self._action_pool.replay is not None
            ):
                state.replay = self._action_pool.replay
                self._action_pool.replay = None
                state.replay.clear()
            else:
                state.replay = LatentReplayBuffer(
                    capacity=cfg.inner_replay_capacity,
                    latent_dim=cfg.latent_dim,
                    action_dim=cfg.action_dim,
                    device=self.device,
                    store_source=self._explorer_active,
                    store_horizon=bool(getattr(cfg, "inner_finite_horizon", False)),
                )
        elif outer_changed:
            # Latents are coordinates of the current encoder/dynamics and may
            # never survive a representation update.
            state.replay.clear()

        state.outer_version = self.agent.outer_version
        state.critic_steps = state.actor_steps = state.temperature_steps = 0
        state.target_steps = state.critic_target_steps = state.actor_target_steps = 0
        state.replay_draws = 0
        state.policy_evaluations = state.q_evaluations = 0
        state.sampled_ids.clear()
        state.sampled_sources.clear()
        state.explorer_actor_steps = state.explorer_critic_steps = 0
        state.explorer_critic_target_steps = 0
        state.explorer_temperature_steps = 0
        state.primary_rollouts = state.explorer_rollouts = 0
        state.primary_transitions = state.explorer_transitions = 0
        if actor_was_missing and not actor_restored:
            state.actor_params = trainable_parameters(state.actor)
            state.actor_trainable_count = sum(
                parameter.numel() for parameter in state.actor_params
            )
        if critic_was_missing and not critic_restored:
            state.critic_params = trainable_parameters(state.critic)
            state.critic_trainable_count = sum(
                parameter.numel() for parameter in state.critic_params
            )
        state.actor.train(cfg.inner_actor_adaptation != "frozen")
        # The critic uses LayerNorm rather than BatchNorm, so eval mode only
        # suppresses its inherited/adapter dropout; parameters and gradients
        # remain active for inner critic and actor updates.
        state.critic.train(
            cfg.inner_critic_adaptation != "frozen"
            and cfg.inner_critic_dropout_enabled
        )
        if state.critic_target is not None:
            state.critic_target.eval()
        if state.actor_target is not None:
            state.actor_target.eval()
        self._prepare_explorer_workspace()
        self._reset_parameter_noise_action_state()

    def make_modules_for_compatibility(self):
        """Legacy test/debug hook returning freshly created inner modules."""
        actor = self._adapt_module(self.model._pi, "actor")
        critic = self._adapt_module(self.model._Qs, "critic")
        target = deepcopy(critic).to(self.device).requires_grad_(False)
        actor_optim = self._new_optimizer(actor, "actor")
        critic_optim = self._new_optimizer(critic, "critic")
        return (
            actor,
            critic,
            target,
            actor_optim,
            critic_optim,
            trainable_parameters(actor),
            trainable_parameters(critic),
        )

    def _policy_action(
        self,
        z,
        policy,
        *,
        mode,
        generator,
        noise=None,
        std_scale=1.0,
        noise_std=0.0,
        inner_bounds=True,
        return_info=False,
    ):
        kwargs = {}
        if inner_bounds:
            kwargs = {
                "log_std_mapping": self.cfg.inner_log_std_mapping,
                "log_std_min": self.cfg.inner_log_std_min,
                "log_std_max": self.cfg.inner_log_std_max,
            }
        if mode == "policy_sample":
            if hasattr(self.model, "pi_action") and not return_info:
                return self.model.pi_action(
                    z,
                    policy=policy,
                    generator=generator,
                    noise=noise,
                    std_scale=std_scale,
                    **kwargs,
                ), None
            return self.model.pi(
                z,
                policy=policy,
                generator=generator,
                noise=noise,
                std_scale=std_scale,
                **kwargs,
            )
        if hasattr(self.model, "pi_action") and not return_info:
            mean = self.model.pi_action(
                z,
                policy=policy,
                deterministic=True,
                **kwargs,
            )
            info = None
        else:
            mean, info = self.model.pi(z, policy=policy, deterministic=True, **kwargs)
        if mode == "mean":
            return mean, info
        if mode != "mean_plus_gaussian":
            raise ValueError(f"Unknown action sampling mode: {mode!r}")
        if noise is None:
            noise = torch.randn(
                mean.shape,
                device=mean.device,
                dtype=mean.dtype,
                generator=generator,
            )
        return (mean + noise * float(noise_std)).clamp(-1.0, 1.0), info

    @torch.no_grad()
    def _collect_policy_population(
        self,
        root_z,
        policy,
        *,
        count,
        horizon,
        source,
        generator,
    ):
        """Collect one rollout population while preserving actor identity."""
        cfg, state = self.cfg, self.state
        z = root_z.expand(count, -1).clone()
        alive = torch.ones(count, dtype=torch.bool, device=self.device)
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        reward_sums = torch.zeros(count, device=self.device)
        discounted_rewards = torch.zeros(count, device=self.device)
        terminated_rollout = torch.zeros(count, dtype=torch.bool, device=self.device)
        discounts = torch.ones(count, device=self.device)
        transition_fields = ([], [], [], [], [])
        horizon_flags = []
        transition_count = 0
        std_scale = float(cfg.inner_behavior_std_scale)
        behavior_mode = str(cfg.inner_behavior_action)
        if behavior_mode == "policy_sample" and std_scale == 0.0:
            behavior_mode = "mean"

        for step in range(horizon):
            active = (
                torch.nonzero(alive, as_tuple=False).squeeze(-1)
                if cfg.episodic
                else torch.arange(count, device=self.device)
            )
            if active.numel() == 0:
                break
            active_z = z.index_select(0, active)
            action, _ = self._policy_action(
                active_z,
                policy,
                mode=behavior_mode,
                generator=generator,
                std_scale=max(std_scale, 1e-12),
                noise_std=cfg.inner_behavior_noise_std,
            )
            active_count = int(active.numel())
            state.policy_evaluations += active_count
            transition_count += active_count
            joint = self.model.joint_input(active_z, action)
            reward = td_math.two_hot_inv(self.model.reward_from_joint(joint), cfg)
            next_z = self.model.next_from_joint(joint)
            terminated = (
                (
                    self.model.termination(next_z)
                    > float(cfg.inner_termination_threshold)
                ).float()
                if cfg.episodic
                else reward.new_zeros(active_count, 1)
            )
            for field, value in zip(
                transition_fields,
                (active_z, action, reward, next_z, terminated),
            ):
                field.append(value)
            if getattr(cfg, "inner_finite_horizon", False):
                horizon_flags.append(torch.full_like(terminated, float(step == horizon - 1)))
            reward_vector = reward.squeeze(-1)
            lengths[active] += 1
            reward_sums[active] += reward_vector
            discounted_rewards[active] += discounts[active] * reward_vector
            discounts[active] *= float(self.agent.discount)
            z[active] = next_z
            if cfg.episodic:
                just_terminated = terminated.squeeze(-1) >= 0.5
                terminated_rollout[active] |= just_terminated
                alive[active] = ~just_terminated

        if transition_fields[0]:
            flattened_fields = tuple(
                torch.cat(values, dim=0) for values in transition_fields
            )
            state.replay.add_batch(
                *flattened_fields,
                source=int(source),
                **({"horizon_end": torch.cat(horizon_flags, dim=0)} if horizon_flags else {}),
            )
            transition_rewards = flattened_fields[2].reshape(-1)
            transition_terminated = flattened_fields[4].reshape(-1).to(
                dtype=torch.bool
            )
        else:
            transition_rewards = root_z.new_empty(0)
            transition_terminated = torch.empty(
                0, dtype=torch.bool, device=self.device
            )
        return {
            "lengths": lengths,
            "reward_sums": reward_sums,
            "discounted_rewards": discounted_rewards,
            "terminated": terminated_rollout,
            "transition_count": transition_count,
            "sources": torch.full(
                (count,), int(source), dtype=torch.uint8, device=self.device
            ),
            "transition_rewards": transition_rewards,
            "transition_terminated": transition_terminated,
            "transition_sources": torch.full(
                (transition_count,),
                int(source),
                dtype=torch.uint8,
                device=self.device,
            ),
        }

    def _parameter_noise_actor_spec(self):
        if self._parameter_noise_spec is None:
            self._parameter_noise_spec = classify_parameter_noise_actor(
                self.state.actor,
                action_dim=int(self.cfg.action_dim),
            )
        return self._parameter_noise_spec

    @torch.no_grad()
    def _calibrate_parameter_noise(self, root_z, generator):
        """Fit sigma at a round boundary using only previously known states."""
        cfg, state = self.cfg, self.state
        spec = self._parameter_noise_actor_spec()
        directions = int(cfg.inner_param_noise_calibration_directions)
        batch_size = int(cfg.inner_param_noise_calibration_batch_size)
        if int(root_z.shape[0]) != 1:
            raise ValueError(
                "Adaptive parameter-noise calibration expects one root latent."
            )
        calibration_rows = [root_z.detach()]
        replay_rows = min(max(0, batch_size - 1), int(state.replay.size))
        if replay_rows:
            # Uniform without replacement. Round one has an empty replay and
            # therefore calibrates on the root alone; later rounds can use at
            # most B-1 states that existed before the current collection.
            indices = torch.randperm(
                int(state.replay.size),
                device=self.device,
                generator=generator,
            )[:replay_rows]
            calibration_rows.append(
                state.replay.z[: state.replay.size]
                .index_select(0, indices)
                .detach()
            )
        sampled = torch.cat(calibration_rows, dim=0)
        calibration_latents = sampled.unsqueeze(0).expand(directions, -1, -1)
        deltas = sample_parameter_deltas(
            state.actor,
            spec,
            directions,
            generator=generator,
        )

        sigma = float(self._parameter_noise_stddev)
        target = float(cfg.inner_param_noise_target_action_rms)
        target_hit = False
        evaluations_per_probe = 2 * directions * int(sampled.shape[0])
        for _ in range(int(cfg.inner_param_noise_calibration_max_probes)):
            parameters = make_perturbed_actor_parameters(
                state.actor,
                spec,
                deltas,
                sigma,
            )
            measured = parameter_noise_action_rms(
                state.actor,
                spec,
                parameters,
                calibration_latents,
                chunk_size=_PARAMETER_NOISE_FUNCTIONAL_CHUNK_SIZE,
            )
            if not bool(torch.isfinite(measured).item()):
                raise RuntimeError(
                    "Adaptive parameter-noise calibration produced a non-finite "
                    "post-tanh action RMS."
                )
            self._parameter_noise_calibration_rms.append(measured.detach())
            self._parameter_noise_calibration_probes += 1
            self._parameter_noise_calibration_policy_evaluations += (
                evaluations_per_probe
            )
            state.policy_evaluations += evaluations_per_probe

            measured_value = float(measured.item())
            target_hit = abs(measured_value - target) <= 0.10 * target
            if target_hit:
                self._parameter_noise_sigma_bound_hits.append(0.0)
                break
            sigma = adapt_parameter_noise_stddev(
                sigma,
                measured_value,
                target,
                adaptation_rate=0.5,
                min_stddev=float(cfg.inner_param_noise_sigma_min),
                max_stddev=float(cfg.inner_param_noise_sigma_max),
                max_update_ratio=2.0,
            )
            at_bound = sigma in {
                float(cfg.inner_param_noise_sigma_min),
                float(cfg.inner_param_noise_sigma_max),
            }
            self._parameter_noise_sigma_bound_hits.append(float(at_bound))
            if at_bound:
                break

        self._parameter_noise_stddev = float(sigma)
        self._parameter_noise_sigma_values.append(float(sigma))
        self._parameter_noise_calibration_hits.append(float(target_hit))
        return spec

    @torch.no_grad()
    def _collect_parameter_noise_population(
        self,
        root_z,
        *,
        spec,
        batched_parameters,
        actor_count,
        rollouts_per_actor,
        horizon,
        generator,
    ):
        """Roll out K fixed perturbed actors with equal actor-major groups."""
        cfg, state = self.cfg, self.state
        count = actor_count * rollouts_per_actor
        z = (
            root_z.expand(actor_count, rollouts_per_actor, -1)
            .clone()
            .contiguous()
        )
        alive = torch.ones(count, dtype=torch.bool, device=self.device)
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        reward_sums = torch.zeros(count, device=self.device)
        discounted_rewards = torch.zeros(count, device=self.device)
        terminated_rollout = torch.zeros(
            count, dtype=torch.bool, device=self.device
        )
        discounts = torch.ones(count, device=self.device)
        transition_fields = ([], [], [], [], [])
        horizon_flags = []
        transition_count = 0
        action_dim = int(cfg.action_dim)
        std_scale = max(float(cfg.inner_behavior_std_scale), 1e-12)

        for step in range(horizon):
            active = (
                torch.nonzero(alive, as_tuple=False).squeeze(-1)
                if cfg.episodic
                else torch.arange(count, device=self.device)
            )
            if active.numel() == 0:
                break
            flat_z = z.reshape(count, -1)
            noisy_mean_raw = population_actor_mean_raw(
                state.actor,
                spec,
                batched_parameters,
                z,
                chunk_size=_PARAMETER_NOISE_FUNCTIONAL_CHUNK_SIZE,
            )
            clean_stats = self.model.policy_stats(
                flat_z,
                policy=state.actor,
                std_scale=std_scale,
                **self._inner_policy_kwargs(),
            )
            if noisy_mean_raw.shape != (
                actor_count,
                rollouts_per_actor,
                action_dim,
            ):
                raise RuntimeError(
                    "Perturbed actor population returned an unexpected action shape: "
                    f"{tuple(noisy_mean_raw.shape)}."
                )
            torch._assert_async(
                torch.isfinite(noisy_mean_raw).all(),
                "Adaptive parameter-noise behavior produced non-finite means.",
            )
            noise = torch.randn(
                noisy_mean_raw.shape,
                dtype=noisy_mean_raw.dtype,
                device=noisy_mean_raw.device,
                generator=generator,
            )
            clean_log_std = clean_stats["log_std"].reshape(
                actor_count, rollouts_per_actor, action_dim
            )
            action_population = torch.tanh(
                noisy_mean_raw + noise * clean_log_std.exp()
            )
            noisy_mean = torch.tanh(noisy_mean_raw).reshape(count, action_dim)
            clean_mean = clean_stats["mean"].reshape(count, action_dim)

            active_count = int(active.numel())
            active_z = flat_z.index_select(0, active)
            action = action_population.reshape(count, action_dim).index_select(
                0, active
            )
            torch._assert_async(
                torch.isfinite(action).all(),
                "Adaptive parameter-noise behavior produced non-finite actions.",
            )

            row_rms = (
                (noisy_mean - clean_mean).square().mean(dim=-1).sqrt()
            ).index_select(0, active)
            self._parameter_noise_behavior_action_rms.append(row_rms.detach())
            active_noisy_mean = noisy_mean.index_select(0, active)
            saturation = (active_noisy_mean.abs() >= 0.99).sum()
            if self._parameter_noise_saturation_sum is None:
                self._parameter_noise_saturation_sum = saturation
            else:
                self._parameter_noise_saturation_sum = (
                    self._parameter_noise_saturation_sum + saturation
                )
            self._parameter_noise_saturation_count += active_count * action_dim

            # Both the functional noisy actor and the clean actor statistics
            # execute for every fixed-shape group row at each horizon step.
            state.policy_evaluations += 2 * count
            transition_count += active_count
            joint = self.model.joint_input(active_z, action)
            reward = td_math.two_hot_inv(self.model.reward_from_joint(joint), cfg)
            next_z = self.model.next_from_joint(joint)
            terminated = (
                (
                    self.model.termination(next_z)
                    > float(cfg.inner_termination_threshold)
                ).float()
                if cfg.episodic
                else reward.new_zeros(active_count, 1)
            )
            for field, value in zip(
                transition_fields,
                (active_z, action, reward, next_z, terminated),
            ):
                field.append(value)
            if getattr(cfg, "inner_finite_horizon", False):
                horizon_flags.append(torch.full_like(terminated, float(step == horizon - 1)))
            reward_vector = reward.squeeze(-1)
            lengths[active] += 1
            reward_sums[active] += reward_vector
            discounted_rewards[active] += discounts[active] * reward_vector
            discounts[active] *= float(self.agent.discount)
            flat_z[active] = next_z
            if cfg.episodic:
                just_terminated = terminated.squeeze(-1) >= 0.5
                terminated_rollout[active] |= just_terminated
                alive[active] = ~just_terminated

        if transition_fields[0]:
            flattened_fields = tuple(
                torch.cat(values, dim=0) for values in transition_fields
            )
            state.replay.add_batch(
                *flattened_fields, source=1,
                **({"horizon_end": torch.cat(horizon_flags, dim=0)} if horizon_flags else {}),
            )
            transition_rewards = flattened_fields[2].reshape(-1)
            transition_terminated = flattened_fields[4].reshape(-1).to(
                dtype=torch.bool
            )
        else:
            transition_rewards = root_z.new_empty(0)
            transition_terminated = torch.empty(
                0, dtype=torch.bool, device=self.device
            )
        return {
            "lengths": lengths,
            "reward_sums": reward_sums,
            "discounted_rewards": discounted_rewards,
            "terminated": terminated_rollout,
            "transition_count": transition_count,
            "sources": torch.ones(count, dtype=torch.uint8, device=self.device),
            "transition_rewards": transition_rewards,
            "transition_terminated": transition_terminated,
            "transition_sources": torch.ones(
                transition_count, dtype=torch.uint8, device=self.device
            ),
        }

    @torch.no_grad()
    def _collect_parameter_noise_round(self, root_z):
        """Collect clean P and grouped parameter-noise populations."""
        cfg, state = self.cfg, self.state
        primary_count = int(cfg.inner_primary_rollouts_per_round)
        explorer_count = int(cfg.inner_explorer_rollouts_per_round)
        actor_count = int(cfg.inner_param_noise_actor_count)
        rollouts_per_actor = int(cfg.inner_param_noise_rollouts_per_actor)
        if explorer_count != actor_count * rollouts_per_actor:
            raise RuntimeError(
                "Adaptive parameter-noise rollout allocation is not exactly grouped."
            )
        if primary_count <= 0 or explorer_count <= 0:
            raise RuntimeError(
                "Adaptive parameter-noise rounds require non-empty populations."
            )

        training_modes = tuple(
            (module, bool(module.training))
            for root in (state.actor, self.model)
            for module in root.modules()
        )
        state.actor.eval()
        self.model.eval()
        try:
            # Calibration is logically part of round initialization. It sees
            # the root plus replay states from completed rounds only, and its
            # random draws cannot perturb behavior collection's RNG sequence.
            calibration_start = self._timer_start()
            with self.rng.fork("initialization") as calibration_generator:
                spec = self._calibrate_parameter_noise(
                    root_z,
                    calibration_generator,
                )
            self._timer_stop(
                "inner_param_noise_calibration_seconds", calibration_start
            )
            with self.rng.fork("collection") as generator:
                primary = self._collect_policy_population(
                    root_z,
                    state.actor,
                    count=primary_count,
                    horizon=int(cfg.inner_rollout_horizon),
                    source=0,
                    generator=generator,
                )
                deltas = sample_parameter_deltas(
                    state.actor,
                    spec,
                    actor_count,
                    generator=generator,
                )
                parameters = make_perturbed_actor_parameters(
                    state.actor,
                    spec,
                    deltas,
                    float(self._parameter_noise_stddev),
                )
                explorer = self._collect_parameter_noise_population(
                    root_z,
                    spec=spec,
                    batched_parameters=parameters,
                    actor_count=actor_count,
                    rollouts_per_actor=rollouts_per_actor,
                    horizon=int(cfg.inner_rollout_horizon),
                    generator=generator,
                )
        finally:
            for module, was_training in training_modes:
                module.training = was_training

        state.primary_rollouts += primary_count
        state.explorer_rollouts += explorer_count
        state.primary_transitions += int(primary["transition_count"])
        state.explorer_transitions += int(explorer["transition_count"])
        return {
            key: torch.cat((primary[key], explorer[key]), dim=0)
            for key in (
                "lengths",
                "reward_sums",
                "discounted_rewards",
                "terminated",
                "sources",
                "transition_rewards",
                "transition_terminated",
                "transition_sources",
            )
        } | {
            "transition_count": int(primary["transition_count"])
            + int(explorer["transition_count"])
        }

    @torch.no_grad()
    def _collect_explorer_round(self, root_z):
        """Collect exact P/R populations into one source-labelled replay."""
        cfg, state = self.cfg, self.state
        total = int(cfg.inner_rollouts_per_round)
        horizon = int(cfg.inner_rollout_horizon)
        primary_exact = total * float(cfg.inner_prior_rollout_weight)
        primary_count = int(round(primary_exact))
        if not math.isclose(primary_exact, primary_count, abs_tol=1e-9):
            raise RuntimeError(
                "inner_rollouts_per_round * inner_prior_rollout_weight must "
                "resolve to an integer population."
            )
        explorer_count = total - primary_count
        if primary_count <= 0 or explorer_count <= 0:
            raise RuntimeError("Active explorer rounds require non-empty P and R populations.")

        training_modes = tuple(
            (module, bool(module.training))
            for root in (state.actor, state.explorer_actor, self.model)
            for module in root.modules()
        )
        state.actor.eval()
        state.explorer_actor.eval()
        self.model.eval()
        try:
            with self.rng.fork("collection") as generator:
                # Fixed order is part of the reproducibility contract.
                primary = self._collect_policy_population(
                    root_z,
                    state.actor,
                    count=primary_count,
                    horizon=horizon,
                    source=0,
                    generator=generator,
                )
                explorer = self._collect_policy_population(
                    root_z,
                    state.explorer_actor,
                    count=explorer_count,
                    horizon=horizon,
                    source=1,
                    generator=generator,
                )
        finally:
            # Restore exact per-module modes; calling model.train(...) here
            # would incorrectly switch frozen target critics back to training.
            for module, was_training in training_modes:
                module.training = was_training

        state.primary_rollouts += primary_count
        state.explorer_rollouts += explorer_count
        state.primary_transitions += int(primary["transition_count"])
        state.explorer_transitions += int(explorer["transition_count"])
        return {
            key: torch.cat((primary[key], explorer[key]), dim=0)
            for key in (
                "lengths",
                "reward_sums",
                "discounted_rewards",
                "terminated",
                "sources",
                "transition_rewards",
                "transition_terminated",
                "transition_sources",
            )
        } | {
            "transition_count": int(primary["transition_count"])
            + int(explorer["transition_count"])
        }

    @torch.no_grad()
    def _collect_round(self, root_z):
        cfg, state = self.cfg, self.state
        if self._parameter_noise_active:
            return self._collect_parameter_noise_round(root_z)
        if self._materialized_explorer_active:
            return self._collect_explorer_round(root_z)
        count = int(cfg.inner_rollouts_per_round)
        horizon = int(cfg.inner_rollout_horizon)
        if count == 0:
            empty = root_z.new_empty((0,))
            return {
                "lengths": empty.to(dtype=torch.long),
                "reward_sums": empty,
                "discounted_rewards": empty,
                "terminated": empty.to(dtype=torch.bool),
                "transition_count": 0,
            }

        if not cfg.episodic:
            return self._collect_dense_round(root_z, count=count, horizon=horizon)

        z = root_z.expand(count, -1).clone()
        alive = torch.ones(count, dtype=torch.bool, device=self.device)
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        reward_sums = torch.zeros(count, device=self.device)
        discounted_rewards = torch.zeros(count, device=self.device)
        terminated_rollout = torch.zeros(count, dtype=torch.bool, device=self.device)
        discounts = torch.ones(count, device=self.device)

        state.actor.eval()
        self.model.eval()
        transition_fields = ([], [], [], [], [])
        horizon_flags = []
        transition_count = 0
        with self.rng.fork("collection") as generator:
            for step in range(horizon):
                active = torch.nonzero(alive, as_tuple=False).squeeze(-1)
                if active.numel() == 0:
                    break
                transition_count += int(active.numel())
                active_z = z[active]
                std_scale = float(cfg.inner_behavior_std_scale)
                behavior_mode = str(cfg.inner_behavior_action)
                if behavior_mode == "policy_sample" and std_scale == 0.0:
                    behavior_mode = "mean"
                action, _ = self._policy_action(
                    active_z,
                    state.actor,
                    mode=behavior_mode,
                    generator=generator,
                    std_scale=max(std_scale, 1e-12),
                    noise_std=cfg.inner_behavior_noise_std,
                )
                state.policy_evaluations += int(active.numel())
                joint = self.model.joint_input(active_z, action)
                reward = td_math.two_hot_inv(
                    self.model.reward_from_joint(joint), cfg
                )
                next_z = self.model.next_from_joint(joint)
                if cfg.episodic:
                    terminated = (
                        self.model.termination(next_z)
                        > float(cfg.inner_termination_threshold)
                    ).float()
                transition_fields[0].append(active_z)
                transition_fields[1].append(action)
                transition_fields[2].append(reward)
                transition_fields[3].append(next_z)
                transition_fields[4].append(terminated)
                if getattr(cfg, "inner_finite_horizon", False):
                    horizon_flags.append(torch.full_like(terminated, float(step == horizon - 1)))
                reward_vector = reward.squeeze(-1)
                lengths[active] += 1
                reward_sums[active] += reward_vector
                discounted_rewards[active] += discounts[active] * reward_vector
                discounts[active] *= float(self.agent.discount)
                z[active] = next_z
                just_terminated = terminated.squeeze(-1) >= 0.5
                terminated_rollout[active] |= just_terminated
                alive[active] = ~just_terminated

        if transition_fields[0]:
            state.replay.add_batch(
                *(torch.cat(values, dim=0) for values in transition_fields),
                **({"horizon_end": torch.cat(horizon_flags, dim=0)} if horizon_flags else {}),
            )

        if cfg.inner_actor_adaptation != "frozen":
            state.actor.train()
        return {
            "lengths": lengths,
            "reward_sums": reward_sums,
            "discounted_rewards": discounted_rewards,
            "terminated": terminated_rollout,
            "transition_count": transition_count,
        }

    def _dense_rollout_kernel(self, root_z, policy_noise, reward_support):
        """Pure rollout; the collector owns no_grad, replay and counters.

        Keep this compile entrypoint undecorated: the locked PyTorch version
        cannot capture a no_grad-decorated bound method with fullgraph=True.
        """
        cfg, state = self.cfg, self.state
        count = int(cfg.inner_rollouts_per_round)
        horizon = int(cfg.inner_rollout_horizon)
        z = root_z.expand(count, -1)
        reward_sums = root_z.new_zeros(count)
        discounted_rewards = root_z.new_zeros(count)
        transitions = []
        std_scale = float(cfg.inner_behavior_std_scale)
        behavior_mode = str(cfg.inner_behavior_action)
        if behavior_mode == "policy_sample" and std_scale == 0.0:
            behavior_mode = "mean"

        for step in range(horizon):
            action, _ = self._policy_action(
                z,
                state.actor,
                mode=behavior_mode,
                generator=None,
                noise=(policy_noise[step] if policy_noise.numel() else None),
                std_scale=max(std_scale, 1e-12),
                noise_std=cfg.inner_behavior_noise_std,
            )
            joint = self.model.joint_input(z, action)
            reward_prediction = self.model.reward_from_joint(joint)
            if bool(getattr(cfg, "compile", False)):
                reward = td_math.two_hot_inv(
                    reward_prediction,
                    cfg,
                    support=reward_support,
                )
            else:
                reward = td_math.two_hot_inv(reward_prediction, cfg)
            next_z = self.model.next_from_joint(joint)
            terminated = reward.new_zeros(count, 1)
            fields = (z, action, reward, next_z, terminated)
            if getattr(cfg, "inner_finite_horizon", False):
                fields += (torch.full_like(terminated, float(step == horizon - 1)),)
            transitions.append(torch.cat(fields, dim=-1))
            reward_vector = reward.squeeze(-1)
            reward_sums = reward_sums + reward_vector
            discounted_rewards = discounted_rewards + (
                reward_vector * (float(self.agent.discount) ** step)
            )
            z = next_z

        return (
            torch.cat(transitions, dim=0),
            reward_sums,
            discounted_rewards,
        )

    @torch.no_grad()
    def _collect_dense_round(self, root_z, *, count, horizon):
        """Fixed-shape rollout for non-episodic world models."""
        cfg, state = self.cfg, self.state
        std_scale = float(cfg.inner_behavior_std_scale)
        behavior_mode = str(cfg.inner_behavior_action)
        if behavior_mode == "policy_sample" and std_scale == 0.0:
            behavior_mode = "mean"

        state.actor.eval()
        self.model.eval()
        needs_noise = behavior_mode in {"policy_sample", "mean_plus_gaussian"}
        with self.rng.fork("collection") as generator:
            if needs_noise:
                # Draw once per horizon step, exactly matching eager RNG
                # progression, then pass tensors (not Generators) into Dynamo.
                policy_noise = torch.stack(
                    [
                        torch.randn(
                            (count, int(cfg.action_dim)),
                            device=self.device,
                            dtype=root_z.dtype,
                            generator=generator,
                        )
                        for _ in range(horizon)
                    ],
                    dim=0,
                )
            else:
                policy_noise = root_z.new_empty((0, count, int(cfg.action_dim)))
            reward_support = td_math.categorical_support(root_z, cfg)
            rollout = self._compile_regions["rollout"](
                root_z,
                policy_noise,
                reward_support,
            )

        state.replay.add_packed(rollout[0])
        state.policy_evaluations += count * horizon
        if cfg.inner_actor_adaptation != "frozen":
            state.actor.train()
        return {
            "lengths": torch.full(
                (count,), horizon, dtype=torch.long, device=self.device
            ),
            "reward_sums": rollout[1],
            "discounted_rewards": rollout[2],
            "terminated": torch.zeros(count, dtype=torch.bool, device=self.device),
            "transition_count": count * horizon,
        }

    def _sample_batch(self, indices=None):
        replacement = self.cfg.inner_replay_sampling == "with_replacement"
        kwargs = {
            "replacement": replacement,
            "generator": self.rng.generator("replay"),
        }
        if indices is not None:
            kwargs["indices"] = indices
        try:
            batch = self.state.replay.sample(
                self.cfg.inner_batch_size,
                include_ids=self._collect_diagnostics,
                **kwargs,
            )
        except TypeError:
            # Backward-compatible bridge for custom replay implementations.
            batch = self.state.replay.sample(self.cfg.inner_batch_size, **kwargs)
        self.state.replay_draws += int(batch["z"].shape[0])
        if self._collect_diagnostics and "sample_ids" in batch:
            self.state.sampled_ids.append(batch["sample_ids"].detach())
        if "source" in batch:
            self.state.sampled_sources.append(batch["source"].detach())
        return batch

    @torch.no_grad()
    def _mix_outer_critic_batch(self, batch):
        """Replace critic rows with real transitions, retaining actor replay.

        The original imagined minibatch is never mutated. Real observations
        are encoded anew with the frozen current representation, and real
        transitions always use the ordinary inner Bellman continuation.
        """
        fraction = float(getattr(self.cfg, "inner_outer_replay_fraction", 0.0))
        if fraction == 0.0:
            return batch, {}
        batch_size = int(batch["z"].shape[0])
        count = int(math.floor(fraction * batch_size + 0.5))
        replay = getattr(self, "outer_replay_buffer", None)
        available = (
            replay is not None and replay.num_sampleable_transitions > 0
        )
        metrics = {
            "outer_replay_available": float(available),
            "outer_replay_samples": 0.0,
            "outer_replay_fraction": 0.0,
        }
        if not available or count == 0:
            if not available and not getattr(self, "_warned_empty_outer_replay", False):
                warnings.warn(
                    "inner_outer_replay_fraction requested real critic data, but no "
                    "usable outer replay is available; using imagined data until "
                    "replay is populated (realized fraction=0).",
                    RuntimeWarning, stacklevel=2,
                )
                self._warned_empty_outer_replay = True
            return batch, metrics
        obs, action, reward, next_obs, terminated, task = replay.sample_transitions(
            count, generator=self.rng.generator("replay")
        )
        with self.rng.fork("observation"):
            encoded = self.model.encode(
                torch.cat((obs, next_obs), dim=0),
                None if task is None else torch.cat((task, task), dim=0),
            )
            real_z, real_next_z = encoded.split(count, dim=0)
        real = {
            "z": real_z,
            "action": action,
            "reward": reward,
            "next_z": real_next_z,
            "terminated": terminated,
        }
        mixed = {}
        imagined_count = batch_size - count
        for name, value in batch.items():
            if name in real:
                rows = real[name].to(device=value.device, dtype=value.dtype)
            elif name == "horizon_end":
                rows = value.new_zeros((count, *value.shape[1:]))
            elif name == "source":
                # Primary/explorer source summaries describe imagined data;
                # label real rows separately so neither inherits their errors.
                rows = value.new_full((count, *value.shape[1:]), 2)
            else:
                # Sample IDs refer to imagined replay and cannot describe real
                # rows. Critic kernels need neither index metadata field.
                continue
            mixed[name] = torch.cat((value[:imagined_count], rows), dim=0)
        metrics["outer_replay_samples"] = float(count)
        metrics["outer_replay_fraction"] = float(count) / float(batch_size)
        return mixed, metrics

    def _bootstrap_q(
        self,
        z,
        action,
        *,
        pair_indices=None,
        trusted_pair_indices=False,
    ):
        source = str(self.cfg.inner_bootstrap_source)
        kwargs = {"reduction": self.cfg.inner_q_target_reduction}
        if source == "inner_target":
            kwargs["qs"] = self.state.critic_target
        elif source == "outer_target":
            kwargs["target"] = True
        elif source == "outer_online":
            pass
        else:
            raise ValueError(f"Unknown inner bootstrap source: {source!r}")
        return self.model.Q(
            z,
            action,
            pair_indices=pair_indices,
            trusted_pair_indices=trusted_pair_indices,
            **kwargs,
        )

    def _sample_pair_indices(self, generator):
        return self.model.q_backend.sample_pair_indices(
            self.device,
            generator=generator,
        )

    def _prior_noise(self, next_z):
        count = int(next_z.shape[0])
        self.state.policy_evaluations += count
        self.state.q_evaluations += count
        return torch.randn(
            (*next_z.shape[:-1], int(self.cfg.action_dim)),
            device=next_z.device, dtype=next_z.dtype,
            generator=self.rng.generator("bootstrap"),
        )

    def _prior_bootstrap(self, next_z, noise):
        """The MPPI tail: outer distribution, online Q, no extra entropy."""
        with torch.no_grad():
            action, _ = self.model.pi(next_z, noise=noise)
            return self.model.Q(
                next_z, action, reduction=self.cfg.mppi_terminal_q_reduction
            )

    def _sac_target(self, reward, terminated, bootstrap):
        if (getattr(self.cfg, "inner_finite_horizon", False)
                or getattr(self.cfg, "inner_outer_replay_fraction", 0.0) > 0):
            # Multiplication by zero does not mask NaN/Inf. Select first so
            # a truly terminal row cannot inherit an unused continuation.
            bootstrap = torch.where(terminated == 1, 0.0, bootstrap)
        return reward + float(self.agent.discount) * (1.0 - terminated) * bootstrap

    def _sac_critic_kernel(
        self,
        z,
        action,
        reward,
        next_z,
        terminated,
        alpha,
        policy_noise,
        pair_indices,
        horizon_end=None,
        prior_noise=None,
    ):
        """Pure SAC critic loss region; the optimizer step stays eager."""
        state, cfg = self.state, self.cfg
        with torch.no_grad():
            next_action, next_info = self.model.pi(
                next_z,
                policy=state.actor,
                noise=policy_noise,
                log_std_mapping=cfg.inner_log_std_mapping,
                log_std_min=cfg.inner_log_std_min,
                log_std_max=cfg.inner_log_std_max,
            )
            if pair_indices is None:
                next_q = self._bootstrap_q(next_z, next_action)
            else:
                next_q = self._bootstrap_q(
                    next_z,
                    next_action,
                    pair_indices=pair_indices,
                    trusted_pair_indices=True,
                )
            bootstrap = next_q
            if cfg.inner_sac_critic_target == "entropy_augmented":
                bootstrap = bootstrap - alpha * next_info["log_prob"]
            if horizon_end is not None:
                prior_value = self._prior_bootstrap(next_z, prior_noise)
                bootstrap = torch.where(horizon_end.bool(), prior_value, bootstrap)
            target_q = self._sac_target(reward, terminated, bootstrap)

        predictions = self.model.q_predictions(z, action, qs=state.critic)
        critic_loss = self.model.critic_loss(predictions, target_q)
        values = self.model.q_backend.decode(predictions.detach())
        clip_fraction = values.new_zeros(())
        if cfg.q_representation == "distributional":
            symlog_target = td_math.symlog(target_q.detach())
            clip_fraction = (
                (symlog_target <= float(cfg.q_vmin))
                | (symlog_target >= float(cfg.q_vmax))
            ).float().mean()
        return critic_loss, values, target_q, clip_fraction

    def _sac_critic_step(self, batch, alpha):
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        generator = self.rng.generator("bootstrap")
        policy_noise = torch.randn(
            (*batch["next_z"].shape[:-1], int(cfg.action_dim)),
            device=self.device,
            dtype=batch["next_z"].dtype,
            generator=generator,
        )
        # Keep pair selection inside Q with generator=None. It must follow any
        # forked-default dropout draws exactly as it did in eager execution.
        pair_indices = None
        horizon_args = ()
        if getattr(cfg, "inner_finite_horizon", False):
            horizon_args = (batch["horizon_end"], self._prior_noise(batch["next_z"]))
        critic_loss, values, target_q, clip_fraction = self._compile_regions[
            "critic"
        ](
            batch["z"],
            batch["action"],
            batch["reward"],
            batch["next_z"],
            batch["terminated"],
            alpha,
            policy_noise,
            pair_indices,
            *horizon_args,
        )
        state.policy_evaluations += batch_size
        state.q_evaluations += batch_size

        state.q_evaluations += batch_size
        state.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            state.critic_params, float(cfg.inner_critic_grad_clip_norm)
        )
        state.critic_optim.step()
        state.critic_steps += 1
        state.critic_lifetime_steps += 1

        metrics = {
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": torch.as_tensor(grad_norm).detach(),
            "q_mean": values.mean(),
            "q_abs_mean": values.abs().mean(),
            "q_target_mean": target_q.mean(),
            "q_target_clip_fraction": clip_fraction.detach(),
            "td_error_abs_mean": (values - target_q.unsqueeze(0)).abs().mean(),
        }
        metrics.update(self._source_td_metrics(batch, values, target_q))
        return metrics

    @staticmethod
    def _source_td_metrics(batch, values, target_q, *, prefix=""):
        if "source" not in batch:
            return {}
        row_error = (values - target_q.unsqueeze(0)).abs().mean(dim=0).reshape(-1)
        source = batch["source"].reshape(-1).to(dtype=torch.long)
        result = {}
        for value, label in ((0, "primary"), (1, "explorer")):
            selected = row_error[source == value]
            if selected.numel():
                stem = f"{prefix}{label}_td_error_abs"
                result[f"{stem}_mean"] = selected.mean()
                result[f"{stem}_sum"] = selected.sum()
                result[f"{stem}_count"] = selected.new_tensor(selected.numel())
        return result

    def _inner_policy_kwargs(self):
        return {
            "log_std_mapping": self.cfg.inner_log_std_mapping,
            "log_std_min": self.cfg.inner_log_std_min,
            "log_std_max": self.cfg.inner_log_std_max,
        }

    def _q_with(self, z, action, critic, *, reduction=None):
        return self.model.Q(
            z,
            action,
            qs=critic,
            reduction=(reduction or self.cfg.inner_q_target_reduction),
        )

    def _q_target_clip_fraction(self, target_q):
        if self.cfg.q_representation != "distributional":
            return target_q.new_zeros(())
        symlog_target = td_math.symlog(target_q.detach())
        return (
            (symlog_target <= float(self.cfg.q_vmin))
            | (symlog_target >= float(self.cfg.q_vmax))
        ).float().mean()

    def _shared_mixture_critic_step(self, batch, alpha):
        """One Q^mu update with a source-independent continuation mixture."""
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        generator = self.rng.generator("bootstrap")
        shape = (*batch["next_z"].shape[:-1], int(cfg.action_dim))
        primary_noise = torch.randn(
            shape,
            device=self.device,
            dtype=batch["next_z"].dtype,
            generator=generator,
        )
        explorer_noise = torch.randn(
            shape,
            device=self.device,
            dtype=batch["next_z"].dtype,
            generator=generator,
        )
        weight = float(cfg.inner_prior_rollout_weight)
        with torch.no_grad():
            primary_action, primary_info = self.model.pi(
                batch["next_z"],
                policy=state.actor,
                noise=primary_noise,
                **self._inner_policy_kwargs(),
            )
            explorer_action, explorer_info = self.model.pi(
                batch["next_z"],
                policy=state.explorer_actor,
                noise=explorer_noise,
                **self._inner_policy_kwargs(),
            )
            estimator = str(cfg.inner_mixture_target_estimator)
            if estimator == "stratified":
                primary_exact = batch_size * weight
                primary_count = int(round(primary_exact))
                if not math.isclose(primary_exact, primary_count, abs_tol=1e-9):
                    raise RuntimeError(
                        "inner_batch_size * inner_prior_rollout_weight must be "
                        "integral for stratified mixture targets."
                    )
                permutation = torch.randperm(
                    batch_size, device=self.device, generator=generator
                )
                choose_primary = torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=self.device
                )
                choose_primary[permutation[:primary_count]] = True
                next_action = torch.where(
                    choose_primary, primary_action, explorer_action
                )
                pre_tanh_action = torch.where(
                    choose_primary,
                    primary_info["pre_tanh_action"],
                    explorer_info["pre_tanh_action"],
                )
                log_mu = self.model.mixture_log_prob(
                    pre_tanh_action,
                    primary_info,
                    explorer_info,
                    weight,
                )
                bootstrap = self._bootstrap_q(batch["next_z"], next_action)
                state.q_evaluations += batch_size
                if cfg.inner_sac_critic_target == "entropy_augmented":
                    bootstrap = bootstrap - alpha * log_mu
            elif estimator == "weighted":
                primary_log_mu = self.model.mixture_log_prob(
                    primary_info["pre_tanh_action"],
                    primary_info,
                    explorer_info,
                    weight,
                )
                explorer_log_mu = self.model.mixture_log_prob(
                    explorer_info["pre_tanh_action"],
                    primary_info,
                    explorer_info,
                    weight,
                )
                # One critic call makes the randomly selected Q pair common to
                # both mixture components instead of injecting component-wise
                # evaluator noise into the weighted expectation.
                both_value = self._bootstrap_q(
                    torch.cat((batch["next_z"], batch["next_z"]), dim=0),
                    torch.cat((primary_action, explorer_action), dim=0),
                )
                primary_value, explorer_value = both_value.split(batch_size, dim=0)
                state.q_evaluations += 2 * batch_size
                if cfg.inner_sac_critic_target == "entropy_augmented":
                    primary_value = primary_value - alpha * primary_log_mu
                    explorer_value = explorer_value - alpha * explorer_log_mu
                bootstrap = weight * primary_value + (1.0 - weight) * explorer_value
            else:
                raise ValueError(
                    f"Unknown inner mixture target estimator: {estimator!r}"
                )
            if getattr(cfg, "inner_finite_horizon", False):
                prior_value = self._compile_regions["prior_value"](
                    batch["next_z"], self._prior_noise(batch["next_z"])
                )
                bootstrap = torch.where(batch["horizon_end"].bool(), prior_value, bootstrap)
            target_q = self._sac_target(batch["reward"], batch["terminated"], bootstrap)

        state.policy_evaluations += 2 * batch_size
        predictions = self.model.q_predictions(
            batch["z"], batch["action"], qs=state.critic
        )
        critic_loss = self.model.critic_loss(predictions, target_q)
        values = self.model.q_backend.decode(predictions.detach())
        state.q_evaluations += batch_size
        state.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            state.critic_params, float(cfg.inner_critic_grad_clip_norm)
        )
        state.critic_optim.step()
        state.critic_steps += 1
        state.critic_lifetime_steps += 1
        metrics = {
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": torch.as_tensor(grad_norm).detach(),
            "q_mean": values.mean(),
            "q_abs_mean": values.abs().mean(),
            "q_target_mean": target_q.mean(),
            "q_target_clip_fraction": self._q_target_clip_fraction(target_q),
            "td_error_abs_mean": (values - target_q.unsqueeze(0)).abs().mean(),
        }
        metrics.update(self._source_td_metrics(batch, values, target_q))
        return metrics

    def _separate_critics_step(
        self,
        batch,
        *,
        update_primary,
        update_explorer,
    ):
        """Paired Q_P/Q_R updates from the same replay minibatch."""
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        generator = self.rng.generator("bootstrap")
        shape = (*batch["next_z"].shape[:-1], int(cfg.action_dim))
        primary_noise = torch.randn(
            shape,
            device=self.device,
            dtype=batch["next_z"].dtype,
            generator=generator,
        )
        explorer_noise = torch.randn(
            shape,
            device=self.device,
            dtype=batch["next_z"].dtype,
            generator=generator,
        )
        with torch.no_grad():
            primary_action, primary_info = self.model.pi(
                batch["next_z"],
                policy=state.actor,
                noise=primary_noise,
                **self._inner_policy_kwargs(),
            )
            explorer_action, explorer_info = self.model.pi(
                batch["next_z"],
                policy=state.explorer_actor,
                noise=explorer_noise,
                **self._inner_policy_kwargs(),
            )
            primary_bootstrap = self._q_with(
                batch["next_z"], primary_action, state.critic_target
            )
            explorer_bootstrap = self._q_with(
                batch["next_z"], explorer_action, state.explorer_critic_target
            )
            if cfg.inner_sac_critic_target == "entropy_augmented":
                primary_bootstrap = (
                    primary_bootstrap - self.alpha.detach() * primary_info["log_prob"]
                )
                explorer_bootstrap = (
                    explorer_bootstrap
                    - self.explorer_alpha.detach() * explorer_info["log_prob"]
                )
            if getattr(cfg, "inner_finite_horizon", False):
                prior_value = self._compile_regions["prior_value"](
                    batch["next_z"], self._prior_noise(batch["next_z"])
                )
                boundary = batch["horizon_end"].bool()
                primary_bootstrap = torch.where(boundary, prior_value, primary_bootstrap)
                explorer_bootstrap = torch.where(boundary, prior_value, explorer_bootstrap)
            primary_target = self._sac_target(batch["reward"], batch["terminated"], primary_bootstrap)
            explorer_target = self._sac_target(batch["reward"], batch["terminated"], explorer_bootstrap)
        state.policy_evaluations += 2 * batch_size
        state.q_evaluations += 2 * batch_size
        metrics = {}

        # Build both disjoint online-critic graphs before mutating either
        # critic.  A single autograd traversal can then schedule the paired
        # branches together while the two optimizers retain fully independent
        # parameters and Adam state.  Keeping the forward order P then R also
        # preserves the private dropout/RNG contract of the former sequential
        # implementation.
        primary_update = None
        if update_primary:
            primary_predictions = self.model.q_predictions(
                batch["z"], batch["action"], qs=state.critic
            )
            primary_loss = self.model.critic_loss(
                primary_predictions, primary_target
            )
            primary_update = (
                primary_loss,
                self.model.q_backend.decode(primary_predictions.detach()),
            )

        explorer_update = None
        if update_explorer:
            explorer_predictions = self.model.q_predictions(
                batch["z"], batch["action"], qs=state.explorer_critic
            )
            explorer_loss = self.model.critic_loss(
                explorer_predictions, explorer_target
            )
            explorer_update = (
                explorer_loss,
                self.model.q_backend.decode(explorer_predictions.detach()),
            )

        # Clear both extant optimizers even in an asymmetric update slot so a
        # component whose dose is already exhausted never retains stale grads
        # from the preceding paired slot.
        if state.critic_optim is not None:
            state.critic_optim.zero_grad(set_to_none=True)
        if state.explorer_critic_optim is not None:
            state.explorer_critic_optim.zero_grad(set_to_none=True)
        losses = tuple(
            update[0]
            for update in (primary_update, explorer_update)
            if update is not None
        )
        if losses:
            torch.autograd.backward(losses)

        if primary_update is not None:
            primary_loss, primary_values = primary_update
            primary_grad_norm = torch.nn.utils.clip_grad_norm_(
                state.critic_params, float(cfg.inner_critic_grad_clip_norm)
            )
            state.critic_optim.step()
            state.critic_steps += 1
            state.critic_lifetime_steps += 1
            state.q_evaluations += batch_size
            metrics.update(
                critic_loss=primary_loss.detach(),
                critic_grad_norm=torch.as_tensor(primary_grad_norm).detach(),
                q_mean=primary_values.mean(),
                q_abs_mean=primary_values.abs().mean(),
                q_target_mean=primary_target.mean(),
                q_target_clip_fraction=self._q_target_clip_fraction(
                    primary_target
                ),
                td_error_abs_mean=(
                    primary_values - primary_target.unsqueeze(0)
                ).abs().mean(),
            )
            metrics.update(
                self._source_td_metrics(
                    batch, primary_values, primary_target
                )
            )
        if explorer_update is not None:
            explorer_loss, explorer_values = explorer_update
            explorer_grad_norm = torch.nn.utils.clip_grad_norm_(
                state.explorer_critic_params,
                float(cfg.inner_critic_grad_clip_norm),
            )
            state.explorer_critic_optim.step()
            state.explorer_critic_steps += 1
            state.explorer_critic_lifetime_steps += 1
            state.q_evaluations += batch_size
            metrics.update(
                explorer_critic_loss=explorer_loss.detach(),
                explorer_critic_grad_norm=torch.as_tensor(
                    explorer_grad_norm
                ).detach(),
                explorer_q_mean=explorer_values.mean(),
                explorer_q_abs_mean=explorer_values.abs().mean(),
                explorer_q_target_mean=explorer_target.mean(),
                explorer_q_target_clip_fraction=self._q_target_clip_fraction(
                    explorer_target
                ),
                explorer_critic_td_error_abs_mean=(
                    explorer_values - explorer_target.unsqueeze(0)
                ).abs().mean(),
            )
            metrics.update(
                self._source_td_metrics(
                    batch,
                    explorer_values,
                    explorer_target,
                    prefix="explorer_critic_",
                )
            )
        return metrics

    @staticmethod
    def _gaussian_kl(inner_info, outer_info):
        inner_log_std = inner_info["log_std"]
        outer_log_std = outer_info["log_std"]
        mean_delta = inner_info["pre_tanh_mean"] - outer_info["pre_tanh_mean"]
        ratio = (
            inner_log_std.mul(2).exp() + mean_delta.square()
        ) / outer_log_std.mul(2).exp().clamp_min(1e-12)
        return (outer_log_std - inner_log_std + 0.5 * ratio - 0.5).sum(
            dim=-1, keepdim=True
        )

    @staticmethod
    def _tanh_saturation_metrics(pre_tanh_action, action, *, prefix):
        (
            pre_tanh_abs_mean,
            pre_tanh_abs_max,
            pre_tanh_abs_ge_7p6_fraction,
            action_exact_saturation_fraction,
        ) = td_math.tanh_saturation_statistics(pre_tanh_action, action)
        return {
            f"{prefix}pre_tanh_abs_mean": pre_tanh_abs_mean,
            f"{prefix}pre_tanh_abs_max": pre_tanh_abs_max,
            f"{prefix}pre_tanh_abs_ge_7p6_fraction": (
                pre_tanh_abs_ge_7p6_fraction
            ),
            f"{prefix}action_exact_saturation_fraction": (
                action_exact_saturation_fraction
            ),
        }

    def _sac_actor_kernel(
        self,
        z,
        alpha,
        policy_noise,
        pair_indices,
        update_actor,
    ):
        """Pure SAC policy/loss region; optimizer mutation stays eager."""
        state, cfg = self.state, self.cfg
        action, info = self.model.pi(
            z,
            policy=state.actor,
            noise=policy_noise,
            log_std_mapping=cfg.inner_log_std_mapping,
            log_std_min=cfg.inner_log_std_min,
            log_std_max=cfg.inner_log_std_max,
        )
        # Surface the exact sample already used by the compiled actor update so
        # optional diagnostics can run eagerly without another policy forward
        # or any additional RNG consumption. Lightweight test/downstream policy
        # stubs that omit pre_tanh_action retain the historical 8-tuple seam.
        actor_sample = (
            (info["pre_tanh_action"].detach(), action.detach())
            if "pre_tanh_action" in info
            else ()
        )
        zero = info["log_prob"].new_zeros(())
        if not update_actor:
            return (
                info["log_prob"],
                info["entropy"],
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                *actor_sample,
            )

        # Reuse one decoded all-head evaluation for both the configured actor
        # objective and diagnostics. This keeps dropout and pair-sampling RNG
        # identical to the pre-diagnostic actor update.
        q_pi_all = self.model.Q(
            z,
            action,
            qs=state.critic,
            detach=True,
            reduction="all",
        )
        q_pi = self.model.q_backend.reduce(
            q_pi_all,
            cfg.inner_q_actor_reduction,
            pair_indices=pair_indices,
            trusted_pair_indices=pair_indices is not None,
        )
        q_pi_all_detached = q_pi_all.detach()
        q_pi_mean_all = self.model.q_backend.reduce(
            q_pi_all_detached,
            "mean_all",
        )
        q_pi_min_all = self.model.q_backend.reduce(
            q_pi_all_detached,
            "min_all",
        )
        actor_loss_values = alpha * info["log_prob"] - q_pi
        kl = torch.zeros_like(actor_loss_values)
        if float(cfg.inner_outer_policy_kl_coef) > 0.0:
            with torch.no_grad():
                _, outer_info = self.model.pi(
                    z,
                    policy=state.actor_anchor,
                    deterministic=True,
                )
            kl = self._gaussian_kl(info, outer_info)
            actor_loss_values = actor_loss_values + float(
                cfg.inner_outer_policy_kl_coef
            ) * kl
        return (
            info["log_prob"],
            info["entropy"],
            actor_loss_values.mean(),
            q_pi.mean(),
            kl.mean(),
            q_pi_mean_all.mean(),
            q_pi_min_all.mean(),
            (q_pi_mean_all - q_pi_min_all).mean(),
            *actor_sample,
        )

    def _scaled_sac_actor_kernel(
        self,
        z,
        alpha,
        actor_loss_scale,
        policy_noise,
        pair_indices,
        update_actor,
    ):
        """SAC policy region with one explicit, action-frozen loss scale."""
        actor_outputs = self._sac_actor_kernel(
            z,
            alpha,
            policy_noise,
            pair_indices,
            update_actor,
        )
        (
            log_prob,
            entropy,
            actor_loss,
            q_mean,
            kl_mean,
            q_mean_all,
            q_min_all,
            q_mean_all_minus_min_all,
        ) = actor_outputs[:8]
        return (
            log_prob,
            entropy,
            actor_loss / actor_loss_scale.reshape(()),
            q_mean,
            kl_mean,
            q_mean_all,
            q_min_all,
            q_mean_all_minus_min_all,
            *actor_outputs[8:],
        )

    def _sac_policy_step(
        self,
        batch,
        *,
        update_temperature,
        update_actor,
        alpha,
        actor_loss_scale=None,
    ):
        state, cfg = self.state, self.cfg
        scale_enabled = self._sac_actor_loss_scale_enabled
        if scale_enabled and actor_loss_scale is None:
            raise RuntimeError(
                "Scaled inner SAC actor update requires an action-frozen "
                "actor_loss_scale tensor."
            )
        if not scale_enabled and actor_loss_scale is not None:
            raise RuntimeError(
                "Inner SAC actor_loss_scale was supplied while "
                "sac_actor_loss_scale_mode='none'."
            )
        batch_size = int(batch["z"].shape[0])
        with self.rng.fork("gradient_policy") as generator:
            policy_noise = torch.randn(
                (*batch["z"].shape[:-1], int(cfg.action_dim)),
                device=self.device,
                dtype=batch["z"].dtype,
                generator=generator,
            )
            # SAC pair selection remains inside Q on the forked default RNG so
            # it follows policy/critic dropout in the legacy order.
            pair_indices = None
            if not scale_enabled:
                actor_outputs = self._compile_regions["actor"](
                    batch["z"],
                    alpha,
                    policy_noise,
                    pair_indices,
                    update_actor,
                )
            else:
                actor_outputs = self._compile_regions["actor"](
                    batch["z"],
                    alpha,
                    actor_loss_scale,
                    policy_noise,
                    pair_indices,
                    update_actor,
                )
            (
                log_prob,
                entropy,
                actor_loss,
                q_mean,
                kl_mean,
                q_mean_all,
                q_min_all,
                q_mean_all_minus_min_all,
            ) = actor_outputs[:8]
            actor_sample = actor_outputs[8:]
            if len(actor_sample) not in {0, 2}:
                raise RuntimeError(
                    "Inner SAC actor kernel returned an invalid diagnostic "
                    "sample payload."
                )
            state.policy_evaluations += batch_size

            metrics = {}
            if update_actor:
                state.q_evaluations += batch_size
                if float(cfg.inner_outer_policy_kl_coef) > 0.0:
                    state.policy_evaluations += batch_size

                state.actor_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    state.actor_params, float(cfg.inner_actor_grad_clip_norm)
                )
                state.actor_optim.step()
                state.actor_steps += 1
                state.actor_lifetime_steps += 1
                metrics.update(
                    actor_loss=actor_loss.detach(),
                    actor_grad_norm=torch.as_tensor(actor_grad_norm).detach(),
                    actor_q_mean=q_mean.detach(),
                    actor_q_mean_all=q_mean_all.detach(),
                    actor_q_min_all=q_min_all.detach(),
                    actor_q_mean_all_minus_min_all=(
                        q_mean_all_minus_min_all.detach()
                    ),
                    actor_entropy=entropy.detach().mean(),
                )
                if self._collect_diagnostics and actor_sample:
                    metrics.update(
                        self._tanh_saturation_metrics(
                            actor_sample[0],
                            actor_sample[1],
                            prefix="actor_",
                        )
                    )
                if float(cfg.inner_outer_policy_kl_coef) > 0.0:
                    # This is the update-time regularizer statistic. When the
                    # regularizer is disabled, omit it instead of publishing a
                    # zero that could be mistaken for an evaluated KL.
                    metrics["outer_policy_kl"] = kl_mean.detach()
            if update_temperature:
                target_entropy = self._resolved_inner_target_entropy()
                temperature_loss = -(
                    state.log_alpha * (log_prob + target_entropy).detach()
                ).mean()
                state.temperature_optim.zero_grad(set_to_none=True)
                temperature_loss.backward()
                temperature_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [state.log_alpha], float(cfg.inner_temperature_grad_clip_norm)
                )
                state.temperature_optim.step()
                with torch.no_grad():
                    state.log_alpha.clamp_(min=_INNER_LOG_ALPHA_FLOOR)
                state.temperature_steps += 1
                state.temperature_lifetime_steps += 1
                metrics.update(
                    temperature_loss=temperature_loss.detach(),
                    temperature_grad_norm=torch.as_tensor(
                        temperature_grad_norm
                    ).detach(),
                )
            return metrics

    def _shared_mixture_policy_step(
        self,
        batch,
        *,
        update_actor,
        update_temperature,
        actor_loss_scale=None,
    ):
        """Joint exact-mixture actor/temperature update.

        Both component optimizers step from the same backward pass.  The
        cross-density terms in log(mu) intentionally connect P and R; critic
        parameters are detached while action gradients remain live.
        """
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        weight = float(cfg.inner_prior_rollout_weight)
        with self.rng.fork("gradient_policy") as generator:
            shape = (*batch["z"].shape[:-1], int(cfg.action_dim))
            primary_noise = torch.randn(
                shape,
                device=self.device,
                dtype=batch["z"].dtype,
                generator=generator,
            )
            explorer_noise = torch.randn(
                shape,
                device=self.device,
                dtype=batch["z"].dtype,
                generator=generator,
            )
            primary_action, primary_info = self.model.pi(
                batch["z"],
                policy=state.actor,
                noise=primary_noise,
                **self._inner_policy_kwargs(),
            )
            explorer_action, explorer_info = self.model.pi(
                batch["z"],
                policy=state.explorer_actor,
                noise=explorer_noise,
                **self._inner_policy_kwargs(),
            )
            primary_log_mu = self.model.mixture_log_prob(
                primary_info["pre_tanh_action"],
                primary_info,
                explorer_info,
                weight,
            )
            explorer_log_mu = self.model.mixture_log_prob(
                explorer_info["pre_tanh_action"],
                primary_info,
                explorer_info,
                weight,
            )
            state.policy_evaluations += 2 * batch_size
            metrics = {}
            if update_actor:
                both_q_all = self.model.Q(
                    torch.cat((batch["z"], batch["z"]), dim=0),
                    torch.cat((primary_action, explorer_action), dim=0),
                    qs=state.critic,
                    detach=True,
                    reduction="all",
                )
                both_q = self.model.q_backend.reduce(
                    both_q_all, cfg.inner_q_actor_reduction
                )
                primary_q, explorer_q = both_q.split(batch_size, dim=0)
                alpha = self.alpha.detach()
                primary_objective = alpha * primary_log_mu - primary_q
                explorer_objective = alpha * explorer_log_mu - explorer_q
                actor_loss = (
                    weight * primary_objective.mean()
                    + (1.0 - weight) * explorer_objective.mean()
                )
                if actor_loss_scale is not None:
                    actor_loss = actor_loss / actor_loss_scale.reshape(())
                state.actor_optim.zero_grad(set_to_none=True)
                state.explorer_actor_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                primary_grad_norm = torch.nn.utils.clip_grad_norm_(
                    state.actor_params, float(cfg.inner_actor_grad_clip_norm)
                )
                explorer_grad_norm = torch.nn.utils.clip_grad_norm_(
                    state.explorer_actor_params,
                    float(cfg.inner_actor_grad_clip_norm),
                )
                state.actor_optim.step()
                state.explorer_actor_optim.step()
                state.actor_steps += 1
                state.actor_lifetime_steps += 1
                state.explorer_actor_steps += 1
                state.explorer_actor_lifetime_steps += 1
                state.q_evaluations += 2 * batch_size
                metrics.update(
                    actor_loss=actor_loss.detach(),
                    actor_grad_norm=torch.as_tensor(primary_grad_norm).detach(),
                    explorer_actor_grad_norm=torch.as_tensor(
                        explorer_grad_norm
                    ).detach(),
                    actor_q_mean=primary_q.detach().mean(),
                    explorer_actor_q_mean=explorer_q.detach().mean(),
                    mixture_log_prob=(
                        weight * primary_log_mu.detach().mean()
                        + (1.0 - weight) * explorer_log_mu.detach().mean()
                    ),
                    actor_entropy=-primary_log_mu.detach().mean(),
                    explorer_actor_entropy=-explorer_log_mu.detach().mean(),
                )
                if self._collect_diagnostics:
                    if "pre_tanh_action" in primary_info:
                        metrics.update(
                            self._tanh_saturation_metrics(
                                primary_info["pre_tanh_action"],
                                primary_action,
                                prefix="actor_",
                            )
                        )
                    if "pre_tanh_action" in explorer_info:
                        metrics.update(
                            self._tanh_saturation_metrics(
                                explorer_info["pre_tanh_action"],
                                explorer_action,
                                prefix="explorer_actor_",
                            )
                        )
            if update_temperature:
                expected_log_mu = (
                    weight * primary_log_mu
                    + (1.0 - weight) * explorer_log_mu
                )
                temperature_loss = -(
                    state.log_alpha
                    * (
                        expected_log_mu
                        + self._resolved_inner_target_entropy()
                    ).detach()
                ).mean()
                state.temperature_optim.zero_grad(set_to_none=True)
                temperature_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [state.log_alpha], float(cfg.inner_temperature_grad_clip_norm)
                )
                state.temperature_optim.step()
                with torch.no_grad():
                    state.log_alpha.clamp_(min=_INNER_LOG_ALPHA_FLOOR)
                state.temperature_steps += 1
                state.temperature_lifetime_steps += 1
                metrics.update(
                    temperature_loss=temperature_loss.detach(),
                    temperature_grad_norm=torch.as_tensor(grad_norm).detach(),
                )
            return metrics

    def _separate_policy_step(
        self,
        batch,
        *,
        update_primary_actor,
        update_explorer_actor,
        update_primary_temperature,
        update_explorer_temperature,
        actor_loss_scale=None,
    ):
        """Disjoint P/R actor and alpha updates on a paired minibatch."""
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        with self.rng.fork("gradient_policy") as generator:
            shape = (*batch["z"].shape[:-1], int(cfg.action_dim))
            primary_noise = torch.randn(
                shape,
                device=self.device,
                dtype=batch["z"].dtype,
                generator=generator,
            )
            explorer_noise = torch.randn(
                shape,
                device=self.device,
                dtype=batch["z"].dtype,
                generator=generator,
            )
            primary_action, primary_info = self.model.pi(
                batch["z"],
                policy=state.actor,
                noise=primary_noise,
                **self._inner_policy_kwargs(),
            )
            explorer_action, explorer_info = self.model.pi(
                batch["z"],
                policy=state.explorer_actor,
                noise=explorer_noise,
                **self._inner_policy_kwargs(),
            )
            state.policy_evaluations += 2 * batch_size
            metrics = {}

            def actor_update(
                *,
                enabled,
                action,
                info,
                critic,
                alpha,
                optimizer,
                params,
                explorer,
            ):
                if not enabled:
                    return
                q_all = self.model.Q(
                    batch["z"],
                    action,
                    qs=critic,
                    detach=True,
                    reduction="all",
                )
                q_value = self.model.q_backend.reduce(
                    q_all, cfg.inner_q_actor_reduction
                )
                loss = (alpha.detach() * info["log_prob"] - q_value).mean()
                if actor_loss_scale is not None:
                    loss = loss / actor_loss_scale.reshape(())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, float(cfg.inner_actor_grad_clip_norm)
                )
                optimizer.step()
                state.q_evaluations += batch_size
                prefix = "explorer_" if explorer else ""
                metrics.update(
                    {
                        f"{prefix}actor_loss": loss.detach(),
                        f"{prefix}actor_grad_norm": torch.as_tensor(
                            grad_norm
                        ).detach(),
                        f"{prefix}actor_q_mean": q_value.detach().mean(),
                        f"{prefix}actor_entropy": info["entropy"].detach().mean(),
                    }
                )
                if self._collect_diagnostics and "pre_tanh_action" in info:
                    metrics.update(
                        self._tanh_saturation_metrics(
                            info["pre_tanh_action"],
                            action,
                            prefix=f"{prefix}actor_",
                        )
                    )
                if explorer:
                    state.explorer_actor_steps += 1
                    state.explorer_actor_lifetime_steps += 1
                else:
                    state.actor_steps += 1
                    state.actor_lifetime_steps += 1

            actor_update(
                enabled=update_primary_actor,
                action=primary_action,
                info=primary_info,
                critic=state.critic,
                alpha=self.alpha,
                optimizer=state.actor_optim,
                params=state.actor_params,
                explorer=False,
            )
            actor_update(
                enabled=update_explorer_actor,
                action=explorer_action,
                info=explorer_info,
                critic=state.explorer_critic,
                alpha=self.explorer_alpha,
                optimizer=state.explorer_actor_optim,
                params=state.explorer_actor_params,
                explorer=True,
            )

            def temperature_update(*, enabled, info, parameter, optimizer, explorer):
                if not enabled:
                    return
                loss = -(
                    parameter
                    * (
                        info["log_prob"]
                        + self._resolved_inner_target_entropy()
                    ).detach()
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter], float(cfg.inner_temperature_grad_clip_norm)
                )
                optimizer.step()
                with torch.no_grad():
                    parameter.clamp_(min=_INNER_LOG_ALPHA_FLOOR)
                prefix = "explorer_" if explorer else ""
                metrics[f"{prefix}temperature_loss"] = loss.detach()
                metrics[f"{prefix}temperature_grad_norm"] = torch.as_tensor(
                    grad_norm
                ).detach()
                if explorer:
                    state.explorer_temperature_steps += 1
                    state.explorer_temperature_lifetime_steps += 1
                else:
                    state.temperature_steps += 1
                    state.temperature_lifetime_steps += 1

            temperature_update(
                enabled=update_primary_temperature,
                info=primary_info,
                parameter=state.log_alpha,
                optimizer=state.temperature_optim,
                explorer=False,
            )
            temperature_update(
                enabled=update_explorer_temperature,
                info=explorer_info,
                parameter=state.explorer_log_alpha,
                optimizer=state.explorer_temperature_optim,
                explorer=True,
            )
            return metrics

    def _td3_critic_kernel(
        self,
        z,
        action,
        reward,
        next_z,
        terminated,
        alpha,
        policy_noise,
        pair_indices,
    ):
        """Pure TD3 critic loss region; ``alpha`` only normalizes the API."""
        del alpha
        state, cfg = self.state, self.cfg
        with torch.no_grad():
            next_action, _ = self.model.pi(
                next_z, policy=state.actor_target, deterministic=True
            )
            noise = policy_noise * float(cfg.inner_td3_target_noise_std)
            noise = noise.clamp(
                -float(cfg.inner_td3_target_noise_clip),
                float(cfg.inner_td3_target_noise_clip),
            )
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            if pair_indices is None:
                next_q = self._bootstrap_q(next_z, next_action)
            else:
                next_q = self._bootstrap_q(
                    next_z,
                    next_action,
                    pair_indices=pair_indices,
                    trusted_pair_indices=True,
                )
            target_q = reward + float(self.agent.discount) * (
                1.0 - terminated
            ) * next_q

        predictions = self.model.q_predictions(z, action, qs=state.critic)
        critic_loss = self.model.critic_loss(predictions, target_q)
        values = self.model.q_backend.decode(predictions.detach())
        clip_fraction = values.new_zeros(())
        if cfg.q_representation == "distributional":
            symlog_target = td_math.symlog(target_q.detach())
            clip_fraction = (
                (symlog_target <= float(cfg.q_vmin))
                | (symlog_target >= float(cfg.q_vmax))
            ).float().mean()
        return critic_loss, values, target_q, clip_fraction

    def _td3_critic_step(self, batch):
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        generator = self.rng.generator("bootstrap")
        policy_noise = torch.randn(
            (*batch["next_z"].shape[:-1], int(cfg.action_dim)),
            device=self.device,
            dtype=batch["next_z"].dtype,
            generator=generator,
        )
        # Legacy TD3 critic pairs are selected inside Q from the forked default
        # RNG. Target noise alone advances the explicit bootstrap generator.
        pair_indices = None
        critic_loss, values, target_q, clip_fraction = self._compile_regions[
            "critic"
        ](
            batch["z"],
            batch["action"],
            batch["reward"],
            batch["next_z"],
            batch["terminated"],
            self.alpha.detach(),
            policy_noise,
            pair_indices,
        )
        state.policy_evaluations += batch_size
        state.q_evaluations += batch_size

        state.q_evaluations += batch_size
        state.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            state.critic_params, float(cfg.inner_critic_grad_clip_norm)
        )
        state.critic_optim.step()
        state.critic_steps += 1
        state.critic_lifetime_steps += 1
        return {
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": torch.as_tensor(grad_norm).detach(),
            "q_mean": values.mean(),
            "q_abs_mean": values.abs().mean(),
            "q_target_mean": target_q.mean(),
            "q_target_clip_fraction": clip_fraction.detach(),
            "td_error_abs_mean": (values - target_q.unsqueeze(0)).abs().mean(),
        }

    def _td3_actor_kernel(self, z, pair_indices):
        """Pure TD3 actor loss region; the optimizer step stays eager."""
        state, cfg = self.state, self.cfg
        action, _ = self.model.pi(z, policy=state.actor, deterministic=True)
        q_pi_all = self.model.Q(
            z,
            action,
            qs=state.critic,
            detach=True,
            reduction="all",
        )
        q_pi = self.model.q_backend.reduce(
            q_pi_all,
            cfg.inner_q_actor_reduction,
            pair_indices=pair_indices,
            trusted_pair_indices=pair_indices is not None,
        )
        q_pi_all_detached = q_pi_all.detach()
        q_pi_mean_all = self.model.q_backend.reduce(
            q_pi_all_detached,
            "mean_all",
        )
        q_pi_min_all = self.model.q_backend.reduce(
            q_pi_all_detached,
            "min_all",
        )
        anchor_l2 = torch.zeros_like(q_pi)
        if float(cfg.inner_outer_action_l2_coef) > 0.0:
            with torch.no_grad():
                outer_action, _ = self.model.pi(
                    z, policy=state.actor_anchor, deterministic=True
                )
            anchor_l2 = (action - outer_action).square().sum(dim=-1, keepdim=True)
        actor_loss = (
            -q_pi + float(cfg.inner_outer_action_l2_coef) * anchor_l2
        ).mean()
        return (
            actor_loss,
            q_pi.mean(),
            anchor_l2.mean(),
            q_pi_mean_all.mean(),
            q_pi_min_all.mean(),
            (q_pi_mean_all - q_pi_min_all).mean(),
        )

    def _td3_actor_step(self, batch):
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        pair_indices = self._sample_pair_indices(
            self.rng.generator("gradient_policy")
        )
        (
            actor_loss,
            q_mean,
            anchor_l2_mean,
            q_mean_all,
            q_min_all,
            q_mean_all_minus_min_all,
        ) = self._compile_regions["actor"](batch["z"], pair_indices)
        state.policy_evaluations += batch_size
        state.q_evaluations += batch_size
        if float(cfg.inner_outer_action_l2_coef) > 0.0:
            state.policy_evaluations += batch_size
        state.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            state.actor_params, float(cfg.inner_actor_grad_clip_norm)
        )
        state.actor_optim.step()
        state.actor_steps += 1
        state.actor_lifetime_steps += 1
        return {
            "actor_loss": actor_loss.detach(),
            "actor_grad_norm": torch.as_tensor(grad_norm).detach(),
            "actor_q_mean": q_mean.detach(),
            "actor_q_mean_all": q_mean_all.detach(),
            "actor_q_min_all": q_min_all.detach(),
            "actor_q_mean_all_minus_min_all": (
                q_mean_all_minus_min_all.detach()
            ),
            "outer_action_l2": anchor_l2_mean.detach(),
        }

    def _maybe_update_targets(self, *, critic_updated, actor_updated):
        state, cfg = self.state, self.cfg
        if (
            critic_updated
            and state.critic_lifetime_steps > 0
            and state.critic_lifetime_steps
            % int(cfg.inner_critic_target_update_interval)
            == 0
            and cfg.inner_bootstrap_source == "inner_target"
        ):
            polyak_update(
                state.critic,
                state.critic_target,
                cfg.inner_critic_target_tau,
                adapters_only=cfg.inner_critic_adaptation == "lora",
            )
            state.target_steps += 1
            state.critic_target_steps += 1
        if (
            actor_updated
            and cfg.inner_operator == "td3"
            and state.actor_target is not None
            and state.actor_lifetime_steps > 0
            and state.actor_lifetime_steps
            % int(cfg.inner_actor_target_update_interval)
            == 0
        ):
            polyak_update(
                state.actor,
                state.actor_target,
                cfg.inner_actor_target_tau,
                adapters_only=cfg.inner_actor_adaptation == "lora",
            )
            state.target_steps += 1
            state.actor_target_steps += 1

    def _run_updates(
        self,
        round_index,
        allocations,
        *,
        actor_loss_scale=None,
    ):
        critic_count = allocations["critic"][round_index]
        actor_count = allocations["actor"][round_index]
        temperature_count = allocations["temperature"][round_index]
        return self._run_update_counts(
            critic_count=critic_count,
            actor_count=actor_count,
            temperature_count=temperature_count,
            actor_loss_scale=actor_loss_scale,
        )

    def _run_update_counts(
        self,
        *,
        critic_count,
        actor_count,
        temperature_count,
        actor_loss_scale=None,
    ):
        slots = max(critic_count, actor_count, temperature_count)
        metrics = []
        replay_indices = None
        if slots:
            batch_size = int(self.cfg.inner_batch_size)
            replay_generator = self.rng.generator("replay")
            if self.cfg.inner_replay_sampling == "with_replacement":
                replay_indices = torch.randint(
                    self.state.replay.size,
                    (slots, batch_size),
                    device=self.device,
                    generator=replay_generator,
                )
            else:
                if batch_size > self.state.replay.size:
                    raise ValueError(
                        "Cannot sample latent replay without replacement: "
                        f"batch_size={batch_size} exceeds replay "
                        f"size={self.state.replay.size}."
                    )
                # Preserve the historical sequence of one randperm per update,
                # but generate all update indices before entering hot kernels.
                replay_indices = torch.stack(
                    [
                        torch.randperm(
                            self.state.replay.size,
                            device=self.device,
                            generator=replay_generator,
                        )[:batch_size]
                        for _ in range(slots)
                    ]
                )
        for slot in range(slots):
            do_critic = slot < critic_count
            do_actor = slot < actor_count
            do_temperature = slot < temperature_count
            batch = self._sample_batch(
                None if replay_indices is None else replay_indices[slot]
            )
            alpha = self.alpha.detach()
            slot_metrics = {}
            if self.cfg.inner_operator == "sac":
                if do_critic:
                    critic_batch, mix_metrics = self._mix_outer_critic_batch(batch)
                    slot_metrics.update(mix_metrics)
                    with self.rng.fork("bootstrap"):
                        slot_metrics.update(self._sac_critic_step(critic_batch, alpha))
                if do_temperature or do_actor:
                    slot_metrics.update(
                        self._sac_policy_step(
                            batch,
                            update_temperature=do_temperature,
                            update_actor=do_actor,
                            alpha=alpha,
                            actor_loss_scale=actor_loss_scale,
                        )
                    )
            else:
                if do_critic:
                    with self.rng.fork("bootstrap"):
                        slot_metrics.update(self._td3_critic_step(batch))
                if do_actor:
                    with self.rng.fork("gradient_policy"):
                        slot_metrics.update(self._td3_actor_step(batch))
            self._maybe_update_targets(
                critic_updated=do_critic,
                actor_updated=do_actor,
            )
            if self._active_trace is not None:
                self._active_trace.record(
                    "update", self.state, {**slot_metrics, "alpha_used": alpha},
                    updated_critic=do_critic, updated_actor=do_actor,
                    updated_temperature=do_temperature,
                    measurement="pre_update_minibatch",
                )
            metrics.append(slot_metrics)
        return metrics

    def _run_component_update_counts(
        self,
        *,
        critic_count,
        actor_count,
        actor_loss_scale=None,
    ):
        """Run canonical component counts in critic-first phases.

        Each component optimizer step samples its own batch. Automatic SAC
        temperature updates share the corresponding actor-phase batch and do
        not create additional update slots.
        """
        metrics = self._run_update_counts(
            critic_count=critic_count,
            actor_count=0,
            temperature_count=0,
            actor_loss_scale=actor_loss_scale,
        )
        metrics.extend(
            self._run_update_counts(
                critic_count=0,
                actor_count=actor_count,
                temperature_count=(
                    actor_count
                    if self.cfg.inner_operator == "sac"
                    and self.cfg.inner_temperature_mode == "auto"
                    else 0
                ),
                actor_loss_scale=actor_loss_scale,
            )
        )
        return metrics

    def _maybe_update_explorer_critic_target(self, *, critic_updated):
        state, cfg = self.state, self.cfg
        if (
            critic_updated
            and state.explorer_critic_lifetime_steps > 0
            and state.explorer_critic_lifetime_steps
            % int(cfg.inner_critic_target_update_interval)
            == 0
        ):
            polyak_update(
                state.explorer_critic,
                state.explorer_critic_target,
                cfg.inner_critic_target_tau,
            )
            state.target_steps += 1
            state.explorer_critic_target_steps += 1

    def _resolved_primary_round_count(self, component):
        return int(
            getattr(
                self.cfg,
                f"inner_primary_{component}_updates_per_round",
                getattr(self.cfg, f"inner_{component}_updates_per_round", 0),
            )
        )

    def _run_two_policy_component_updates(
        self,
        *,
        realized_transition_count,
        scheduled_update_count=None,
        actor_loss_scale=None,
    ):
        """Run critic-first component doses for an active explorer round."""
        cfg = self.cfg
        mode = self._explorer_mode
        primary_critic = self._resolved_primary_round_count("critic")
        primary_actor = self._resolved_primary_round_count("actor")
        primary_temperature = self._resolved_primary_round_count("temperature")
        explorer_critic = int(cfg.inner_explorer_critic_updates_per_round)
        explorer_actor = int(cfg.inner_explorer_actor_updates_per_round)
        explorer_temperature = int(
            cfg.inner_explorer_temperature_updates_per_round
        )
        canonical_auto = (
            not self._uses_component_update_schedule
            and bool(
                getattr(
                    cfg,
                    "inner_primary_updates_per_round_is_auto",
                    getattr(cfg, "inner_updates_per_round", None) == "auto",
                )
            )
        )
        if canonical_auto or scheduled_update_count is not None:
            realized = int(
                realized_transition_count
                if scheduled_update_count is None else scheduled_update_count
            )
            primary_critic = realized
            primary_actor = realized
            primary_temperature = (
                realized if str(cfg.inner_temperature_mode) == "auto" else 0
            )
            if mode == "shared_mixture":
                explorer_actor = primary_actor
            elif mode == "separate_critics":
                for component, primary_value in (
                    ("critic", primary_critic),
                    ("actor", primary_actor),
                    ("temperature", primary_temperature),
                ):
                    if bool(
                        getattr(
                            cfg,
                            f"inner_explorer_{component}_updates_inherit_primary",
                            True,
                        )
                    ):
                        if component == "critic":
                            explorer_critic = primary_value
                        elif component == "actor":
                            explorer_actor = primary_value
                        else:
                            explorer_temperature = primary_value

        if not self._uses_component_update_schedule:
            # Canonical G scheduling is joint: each slot draws exactly one
            # minibatch, runs critic work first, then actor/temperature work
            # on that same batch.  This preserves the historical replay-draw
            # and update-slot contract while allowing paired R updates.
            if mode in {"frozen_random", "adaptive_param_noise"}:
                return self._run_update_counts(
                    critic_count=primary_critic,
                    actor_count=primary_actor,
                    temperature_count=primary_temperature,
                    actor_loss_scale=actor_loss_scale,
                )
            history = []
            slots = max(
                primary_critic,
                primary_actor,
                primary_temperature,
                explorer_critic,
                explorer_actor,
                explorer_temperature,
            )
            for slot in range(slots):
                batch = self._sample_batch()
                update_primary_critic = slot < primary_critic
                update_explorer_critic = slot < explorer_critic
                update_primary_actor = slot < primary_actor
                update_explorer_actor = slot < explorer_actor
                update_primary_temperature = slot < primary_temperature
                update_explorer_temperature = slot < explorer_temperature
                slot_metrics = {}
                critic_batch = batch
                if update_primary_critic or update_explorer_critic:
                    critic_batch, mix_metrics = self._mix_outer_critic_batch(batch)
                    slot_metrics.update(mix_metrics)
                if mode == "shared_mixture":
                    if update_primary_critic:
                        with self.rng.fork("bootstrap"):
                            slot_metrics.update(
                                self._shared_mixture_critic_step(
                                    critic_batch, self.alpha.detach()
                                )
                            )
                    if update_primary_actor or update_primary_temperature:
                        slot_metrics.update(
                            self._shared_mixture_policy_step(
                                batch,
                                update_actor=update_primary_actor,
                                update_temperature=update_primary_temperature,
                                actor_loss_scale=actor_loss_scale,
                            )
                        )
                    self._maybe_update_targets(
                        critic_updated=update_primary_critic,
                        actor_updated=update_primary_actor,
                    )
                elif mode == "separate_critics":
                    if update_primary_critic or update_explorer_critic:
                        with self.rng.fork("bootstrap"):
                            slot_metrics.update(
                                self._separate_critics_step(
                                    critic_batch,
                                    update_primary=update_primary_critic,
                                    update_explorer=update_explorer_critic,
                                )
                            )
                    if (
                        update_primary_actor
                        or update_explorer_actor
                        or update_primary_temperature
                        or update_explorer_temperature
                    ):
                        slot_metrics.update(
                            self._separate_policy_step(
                                batch,
                                update_primary_actor=update_primary_actor,
                                update_explorer_actor=update_explorer_actor,
                                update_primary_temperature=(
                                    update_primary_temperature
                                ),
                                update_explorer_temperature=(
                                    update_explorer_temperature
                                ),
                                actor_loss_scale=actor_loss_scale,
                            )
                        )
                    self._maybe_update_targets(
                        critic_updated=update_primary_critic,
                        actor_updated=update_primary_actor,
                    )
                    self._maybe_update_explorer_critic_target(
                        critic_updated=update_explorer_critic
                    )
                else:
                    raise ValueError(f"Unknown explorer mode: {mode!r}")
                history.append(slot_metrics)
            return history

        if mode in {"frozen_random", "adaptive_param_noise"}:
            return self._run_component_update_counts(
                critic_count=primary_critic,
                actor_count=primary_actor,
                actor_loss_scale=actor_loss_scale,
            )

        history = []
        critic_slots = max(primary_critic, explorer_critic)
        for slot in range(critic_slots):
            batch = self._sample_batch()
            batch, mix_metrics = self._mix_outer_critic_batch(batch)
            update_primary = slot < primary_critic
            update_explorer = slot < explorer_critic
            if mode == "shared_mixture":
                if not update_primary:
                    continue
                with self.rng.fork("bootstrap"):
                    slot_metrics = self._shared_mixture_critic_step(
                        batch, self.alpha.detach()
                    )
                self._maybe_update_targets(
                    critic_updated=True, actor_updated=False
                )
            elif mode == "separate_critics":
                with self.rng.fork("bootstrap"):
                    slot_metrics = self._separate_critics_step(
                        batch,
                        update_primary=update_primary,
                        update_explorer=update_explorer,
                    )
                self._maybe_update_targets(
                    critic_updated=update_primary, actor_updated=False
                )
                self._maybe_update_explorer_critic_target(
                    critic_updated=update_explorer
                )
            else:
                raise ValueError(f"Unknown explorer mode: {mode!r}")
            slot_metrics.update(mix_metrics)
            history.append(slot_metrics)

        actor_slots = max(
            primary_actor,
            explorer_actor,
            primary_temperature,
            explorer_temperature,
        )
        for slot in range(actor_slots):
            batch = self._sample_batch()
            if mode == "shared_mixture":
                update_actor = slot < primary_actor
                update_temperature = slot < primary_temperature
                slot_metrics = self._shared_mixture_policy_step(
                    batch,
                    update_actor=update_actor,
                    update_temperature=update_temperature,
                    actor_loss_scale=actor_loss_scale,
                )
                self._maybe_update_targets(
                    critic_updated=False, actor_updated=update_actor
                )
            else:
                update_primary_actor = slot < primary_actor
                update_explorer_actor = slot < explorer_actor
                slot_metrics = self._separate_policy_step(
                    batch,
                    update_primary_actor=update_primary_actor,
                    update_explorer_actor=update_explorer_actor,
                    update_primary_temperature=slot < primary_temperature,
                    update_explorer_temperature=slot < explorer_temperature,
                    actor_loss_scale=actor_loss_scale,
                )
                self._maybe_update_targets(
                    critic_updated=False, actor_updated=update_primary_actor
                )
            history.append(slot_metrics)
        return history

    @torch.no_grad()
    def _execute_policy(
        self,
        root_z,
        policy,
        *,
        eval_mode,
        inner_bounds=True,
        return_info=False,
    ):
        mode = "mean" if eval_mode else str(self.cfg.inner_execution_action)
        std_scale = float(self.cfg.inner_execution_std_scale)
        if mode == "policy_sample" and std_scale == 0.0:
            mode = "mean"
        training_modes = tuple(
            (module, bool(module.training)) for module in policy.modules()
        )
        try:
            policy.eval()
            with self.rng.fork("execution") as generator:
                action, info = self._policy_action(
                    root_z,
                    policy,
                    mode=mode,
                    generator=generator,
                    std_scale=max(std_scale, 1e-12),
                    noise_std=self.cfg.inner_execution_noise_std,
                    inner_bounds=inner_bounds,
                    return_info=return_info,
                )
        finally:
            for module, was_training in training_modes:
                module.training = was_training
        self.state.policy_evaluations += int(root_z.shape[0])
        if return_info:
            return action, info
        return action

    @torch.no_grad()
    def _outer_soft_handoff_scores(self, root_z, policy, generator):
        """Score one actor with a soft H-step prefix and one outer tail."""
        cfg = self.cfg
        samples = int(cfg.inner_execution_handoff_samples)
        horizon = int(cfg.inner_rollout_horizon)
        z = root_z.expand(samples, -1).clone()
        score = z.new_zeros(samples, 1)
        continuation = z.new_ones(samples, 1)
        discount = z.new_ones(samples, 1)
        outer_alpha = self.agent.alpha.detach()
        for _ in range(horizon):
            action, info = self.model.pi(
                z,
                policy=policy,
                generator=generator,
                **self._inner_policy_kwargs(),
            )
            self.state.policy_evaluations += samples
            joint = self.model.joint_input(z, action)
            reward = td_math.two_hot_inv(
                self.model.reward_from_joint(joint), cfg
            )
            score += discount * continuation * (
                reward - outer_alpha * info["log_prob"]
            )
            z = self.model.next_from_joint(joint)
            if cfg.episodic:
                terminated = (
                    self.model.termination(z)
                    > float(cfg.inner_termination_threshold)
                ).float()
                continuation *= 1.0 - terminated
            discount *= float(self.agent.discount)

        # The imagined collection horizon is not terminal.  Only this explicit
        # selector appends one outer-prior action/value tail.
        terminal_action, terminal_info = self.model.pi(
            z,
            policy=self.model._pi,
            generator=generator,
        )
        terminal_q = self.model.Q(
            z,
            terminal_action,
            target=True,
            reduction="min_all",
        )
        self.state.policy_evaluations += samples
        self.state.q_evaluations += samples
        score += discount * continuation * (
            terminal_q - outer_alpha * terminal_info["log_prob"]
        )
        return score, samples * horizon

    @torch.no_grad()
    def _fixed_q_counterfactual(self, root_z):
        """Evaluate the deterministic P/R means under the fixed outer target Q."""
        state = self.state
        training_modes = tuple(
            (module, bool(module.training))
            for root in (state.actor, state.explorer_actor, self.model._target_Qs)
            for module in root.modules()
        )
        try:
            state.actor.eval()
            state.explorer_actor.eval()
            self.model._target_Qs.eval()
            primary_stats = self.model.policy_stats(
                root_z,
                policy=state.actor,
                **self._inner_policy_kwargs(),
            )
            explorer_stats = self.model.policy_stats(
                root_z,
                policy=state.explorer_actor,
                **self._inner_policy_kwargs(),
            )
            primary_q = self.model.Q(
                root_z,
                primary_stats["mean"],
                target=True,
                reduction="min_all",
            )
            explorer_q = self.model.Q(
                root_z,
                explorer_stats["mean"],
                target=True,
                reduction="min_all",
            )
        finally:
            for module, was_training in training_modes:
                module.training = was_training

        batch_size = int(root_z.shape[0])
        state.policy_evaluations += 2 * batch_size
        state.q_evaluations += 2 * batch_size
        primary_q = primary_q.reshape(())
        explorer_q = explorer_q.reshape(())
        selected_primary = bool((primary_q >= explorer_q).item())
        selected_action = (
            primary_stats["mean"]
            if selected_primary
            else explorer_stats["mean"]
        )
        metrics = {
            "inner_fixed_q_counterfactual_primary_q": primary_q,
            "inner_fixed_q_counterfactual_explorer_q": explorer_q,
            "inner_fixed_q_counterfactual_margin": explorer_q - primary_q,
            "inner_fixed_q_counterfactual_primary_wins": float(selected_primary),
            "inner_fixed_q_counterfactual_explorer_wins": float(
                not selected_primary
            ),
            "inner_fixed_q_counterfactual_explorer_rate": float(
                not selected_primary
            ),
            "inner_fixed_q_counterfactual_policy_evaluations": float(
                2 * batch_size
            ),
            "inner_fixed_q_counterfactual_q_evaluations": float(2 * batch_size),
            "inner_primary_explorer_action_l2": torch.linalg.vector_norm(
                primary_stats["mean"] - explorer_stats["mean"], dim=-1
            ).mean(),
        }
        for index, value in enumerate(selected_action[0].reshape(-1)):
            metrics[f"inner_fixed_q_counterfactual_action_{index}"] = value
        return {
            "metrics": metrics,
            "selected_primary": selected_primary,
            "selected_action": selected_action,
        }

    @torch.no_grad()
    def _execute_two_policy(
        self,
        root_z,
        *,
        eval_mode,
        return_info,
    ):
        """Select P/R according to the configured execution controller."""
        selector_start = self._timer_start()
        if int(root_z.shape[0]) != 1:
            raise ValueError("Random-explorer execution expects one root latent.")
        state, cfg = self.state, self.cfg
        source = str(cfg.inner_execution_policy_source)
        selected = state.actor
        selected_primary = True
        metrics = {
            "inner_selector_model_steps": 0.0,
            "inner_selector_primary_wins": 0.0,
            "inner_selector_explorer_wins": 0.0,
            "inner_selector_score_margin": 0.0,
            "inner_selector_score_variance": 0.0,
        }
        counterfactual = self._fixed_q_counterfactual(root_z)
        metrics.update(counterfactual["metrics"])

        if source == "primary":
            pass
        elif source == "explorer":
            selected = state.explorer_actor
            selected_primary = False
        elif source == "mixture_sample":
            with self.rng.fork("execution") as generator:
                selected_primary = bool(
                    (
                        torch.rand((), device=self.device, generator=generator)
                        < float(cfg.inner_prior_rollout_weight)
                    ).item()
                )
            selected = state.actor if selected_primary else state.explorer_actor
        elif source == "outer_q_gate":
            selected_primary = counterfactual["selected_primary"]
            selected = state.actor if selected_primary else state.explorer_actor
            metrics["inner_selector_score_margin"] = (
                metrics["inner_fixed_q_counterfactual_margin"].abs()
            )
        elif source == "outer_soft_handoff":
            training_modes = tuple(
                (module, bool(module.training))
                for root in (state.actor, state.explorer_actor, self.model)
                for module in root.modules()
            )
            try:
                state.actor.eval()
                state.explorer_actor.eval()
                self.model.eval()
                with self.rng.fork("execution") as generator:
                    common_state = generator.get_state()
                    primary_scores, primary_steps = self._outer_soft_handoff_scores(
                        root_z, state.actor, generator
                    )
                    generator.set_state(common_state)
                    explorer_scores, explorer_steps = self._outer_soft_handoff_scores(
                        root_z, state.explorer_actor, generator
                    )
            finally:
                for module, was_training in training_modes:
                    module.training = was_training
            primary_score = primary_scores.mean()
            explorer_score = explorer_scores.mean()
            selected_primary = bool((primary_score >= explorer_score).item())
            selected = state.actor if selected_primary else state.explorer_actor
            metrics.update(
                inner_selector_model_steps=float(primary_steps + explorer_steps),
                inner_selector_score_margin=(
                    primary_score - explorer_score
                ).abs(),
                inner_selector_score_variance=torch.stack(
                    (
                        primary_scores.var(unbiased=False),
                        explorer_scores.var(unbiased=False),
                    )
                ).mean(),
                inner_selector_primary_score=primary_score,
                inner_selector_explorer_score=explorer_score,
            )
        else:
            raise ValueError(f"Unknown inner execution policy source: {source!r}")

        metrics["inner_selector_primary_wins"] = float(selected_primary)
        metrics["inner_selector_explorer_wins"] = float(not selected_primary)
        self._timer_stop("inner_selector_seconds", selector_start)
        executed = self._execute_policy(
            root_z,
            selected,
            eval_mode=eval_mode,
            return_info=return_info,
        )
        executed_action = executed[0] if return_info else executed
        metrics["inner_fixed_q_counterfactual_execution_agreement"] = float(
            selected_primary == counterfactual["selected_primary"]
        )
        metrics["inner_fixed_q_counterfactual_action_l2_to_executed"] = (
            torch.linalg.vector_norm(
                counterfactual["selected_action"] - executed_action,
                dim=-1,
            ).mean()
        )
        return executed, metrics, selected

    @torch.no_grad()
    def _evaluate_policy_trajectory(
        self,
        root_z,
        policy,
        generator,
        *,
        stochastic,
        log_std_mapping=None,
        log_std_min=None,
        log_std_max=None,
    ):
        count = int(getattr(self.cfg, "inner_diagnostic_rollouts", 0))
        if count <= 0:
            return None
        z = root_z.expand(count, -1).clone()
        continuation = torch.ones(count, 1, device=self.device)
        discount = torch.ones(count, 1, device=self.device)
        score = torch.zeros(count, 1, device=self.device)
        soft_score = torch.zeros(count, 1, device=self.device)
        alpha = self.agent.alpha.detach()
        for _ in range(int(self.cfg.inner_rollout_horizon)):
            action, info = self.model.pi(
                z,
                policy=policy,
                deterministic=not stochastic,
                generator=generator,
                log_std_mapping=log_std_mapping,
                log_std_min=log_std_min,
                log_std_max=log_std_max,
            )
            self.state.policy_evaluations += count
            joint = self.model.joint_input(z, action)
            reward = td_math.two_hot_inv(
                self.model.reward_from_joint(joint), self.cfg
            )
            score += discount * continuation * reward
            soft_score += discount * continuation * (
                reward - alpha * info["log_prob"]
            )
            z = self.model.next_from_joint(joint)
            if self.cfg.episodic:
                terminated = (
                    self.model.termination(z)
                    > float(self.cfg.inner_termination_threshold)
                ).float()
                continuation *= 1.0 - terminated
            discount *= float(self.agent.discount)

        terminal_action, terminal_info = self.model.pi(
            z,
            policy=policy,
            deterministic=not stochastic,
            generator=generator,
            log_std_mapping=log_std_mapping,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.state.policy_evaluations += count
        terminal_q = self.model.Q(
            z,
            terminal_action,
            target=True,
            reduction="mean_all",
        )
        self.state.q_evaluations += count
        score += discount * continuation * terminal_q
        soft_score += discount * continuation * (
            terminal_q - alpha * terminal_info["log_prob"]
        )
        return {
            "score": score.mean(),
            "soft_score": soft_score.mean(),
            "model_steps": count * int(self.cfg.inner_rollout_horizon),
        }

    @torch.no_grad()
    def _diagnostics(self, root_z, improved_policy):
        with self.rng.fork("diagnostics") as generator:
            final_outer_policy_kl = None
            if self.cfg.inner_operator == "sac":
                policy_training_modes = tuple(
                    (module, bool(module.training))
                    for policy in (self.model._pi, improved_policy)
                    for module in policy.modules()
                )
                try:
                    self.model._pi.eval()
                    improved_policy.eval()
                    outer_action, outer_info = self.model.pi(
                        root_z,
                        policy=self.model._pi,
                        deterministic=True,
                    )
                    improved_action, improved_info = self.model.pi(
                        root_z,
                        policy=improved_policy,
                        deterministic=True,
                        log_std_mapping=self.cfg.inner_log_std_mapping,
                        log_std_min=self.cfg.inner_log_std_min,
                        log_std_max=self.cfg.inner_log_std_max,
                    )
                finally:
                    for module, was_training in policy_training_modes:
                        module.training = was_training
                final_outer_policy_kl = self._gaussian_kl(
                    improved_info, outer_info
                ).mean()
            elif hasattr(self.model, "pi_action"):
                outer_action = self.model.pi_action(root_z, deterministic=True)
                improved_action = self.model.pi_action(
                    root_z, policy=improved_policy, deterministic=True
                )
            else:
                outer_action, _ = self.model.pi(root_z, deterministic=True)
                improved_action, _ = self.model.pi(
                    root_z, policy=improved_policy, deterministic=True
                )
            self.state.policy_evaluations += 2 * int(root_z.shape[0])
            outer_predictions = self.model.q_predictions(
                root_z, outer_action, target=True
            )
            improved_predictions = self.model.q_predictions(
                root_z, improved_action, target=True
            )
            outer_q = self.model.q_backend.reduce(
                self.model.q_backend.decode(outer_predictions), "mean_all"
            )
            improved_q = self.model.q_backend.reduce(
                self.model.q_backend.decode(improved_predictions), "mean_all"
            )
            self.state.q_evaluations += 2 * int(root_z.shape[0])
            action_delta = torch.linalg.vector_norm(
                improved_action - outer_action, dim=-1
            ).mean()

            if int(self.cfg.inner_diagnostic_rollouts) > 0:
                state_before = generator.get_state()
                outer_eval = self._evaluate_policy_trajectory(
                    root_z,
                    self.model._pi,
                    generator,
                    stochastic=self.cfg.inner_operator == "sac",
                )
                generator.set_state(state_before)
                improved_eval = self._evaluate_policy_trajectory(
                    root_z,
                    improved_policy,
                    generator,
                    stochastic=self.cfg.inner_operator == "sac",
                    log_std_mapping=self.cfg.inner_log_std_mapping,
                    log_std_min=self.cfg.inner_log_std_min,
                    log_std_max=self.cfg.inner_log_std_max,
                )
            else:
                outer_eval = improved_eval = None

        q_gain = (improved_q - outer_q).mean()
        metrics = {
            "inner_policy_mean_delta_l2": action_delta,
            "inner_fixed_target_q_action_gain": q_gain,
            # One-release compatibility alias. The evaluator is now fixed and
            # cannot be changed by the adapted critic.
            "inner_outer_q_gain": q_gain,
            "inner_fixed_target_q_outer": outer_q.mean(),
            "inner_fixed_target_q_improved": improved_q.mean(),
            "inner_fixed_target_q_abs_mean": torch.stack(
                (outer_q.abs().mean(), improved_q.abs().mean())
            ).mean(),
            "inner_fixed_evaluator_alpha": self.agent.alpha.detach().mean(),
        }
        if final_outer_policy_kl is not None:
            metrics["inner_final_outer_policy_kl"] = final_outer_policy_kl
        if self.cfg.q_representation == "distributional":
            probabilities = torch.softmax(
                torch.cat((outer_predictions, improved_predictions), dim=1), dim=-1
            )
            metrics["inner_distributional_q_entropy"] = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(dim=-1).mean()
            metrics["inner_distributional_q_edge_mass"] = (
                probabilities[..., 0] + probabilities[..., -1]
            ).mean()
        if outer_eval is not None and improved_eval is not None:
            metrics.update(
                inner_predicted_j_outer=outer_eval["score"],
                inner_predicted_j_improved=improved_eval["score"],
                inner_predicted_j_gain=improved_eval["score"] - outer_eval["score"],
                inner_predicted_soft_j_outer=outer_eval["soft_score"],
                inner_predicted_soft_j_improved=improved_eval["soft_score"],
                inner_predicted_soft_j_gain=(
                    improved_eval["soft_score"] - outer_eval["soft_score"]
                ),
                inner_fixed_alpha_soft_j_outer=outer_eval["soft_score"],
                inner_fixed_alpha_soft_j_improved=improved_eval["soft_score"],
                inner_fixed_alpha_soft_j_gain=(
                    improved_eval["soft_score"] - outer_eval["soft_score"]
                ),
                inner_diagnostic_model_steps=(
                    outer_eval["model_steps"] + improved_eval["model_steps"]
                ),
            )
        else:
            metrics["inner_diagnostic_model_steps"] = 0.0
        return metrics

    @staticmethod
    def _stats(values):
        if isinstance(values, (list, tuple)):
            if not values:
                return 0.0, 0.0, 0.0, 0.0
            if torch.is_tensor(values[0]):
                values = torch.cat(values, dim=0)
            else:
                values = torch.as_tensor(values, dtype=torch.float64)
        if not torch.is_tensor(values):
            values = torch.as_tensor(values, dtype=torch.float64)
        if values.numel() == 0:
            return 0.0, 0.0, 0.0, 0.0
        if not values.is_floating_point():
            values = values.float()
        return (
            values.mean(),
            values.std(unbiased=False),
            values.min(),
            values.max(),
        )

    @staticmethod
    def _average_update_metrics(history):
        grouped = {}
        for item in history:
            for key, value in item.items():
                grouped.setdefault(key, []).append(torch.as_tensor(value).reshape(()))
        metrics = {}
        auxiliary = {
            key
            for key in grouped
            if key.endswith(("_sum", "_count"))
        }
        for key, values in grouped.items():
            if not values or key in auxiliary:
                continue
            stacked = torch.stack(values)
            prefix = f"inner_{key}"
            metrics[prefix] = stacked.mean()
            metrics[f"{prefix}_std"] = stacked.std(unbiased=False)
            metrics[f"{prefix}_min"] = stacked.min()
            metrics[f"{prefix}_max"] = stacked.max()
        # Per-source minibatch populations vary. Aggregate TD errors by their
        # exact row count instead of averaging per-slot means equally.
        for sum_key in sorted(key for key in grouped if key.endswith("_sum")):
            stem = sum_key[: -len("_sum")]
            count_key = f"{stem}_count"
            if count_key not in grouped:
                continue
            total_sum = torch.stack(grouped[sum_key]).sum()
            total_count = torch.stack(grouped[count_key]).sum()
            metrics[f"inner_{stem}_mean"] = total_sum / total_count.clamp_min(1)
            metrics[f"inner_{stem}_count"] = total_count
        return metrics

    def _compile_fallback_metrics(self):
        """Report sticky compile failures after every region used by the action.

        This is deliberately evaluated at the action return boundary.  MPPI
        diagnostics can invoke the outer critic ensembles after the base
        metric dictionary is created, so sampling these flags earlier would
        miss a fallback that occurred during the current action.
        """
        rollout_fallback = self._compile_regions["rollout"].failed
        critic_region_fallback = self._compile_regions["critic"].failed
        if "prior_value" in self._compile_regions:
            critic_region_fallback |= self._compile_regions["prior_value"].failed
        actor_fallback = self._compile_regions["actor"].failed
        critic_module_fallback = bool(
            self.state.critic is not None
            and getattr(self.state.critic, "compile_failed", False)
        )
        target_module_fallback = bool(
            self.state.critic_target is not None
            and getattr(self.state.critic_target, "compile_failed", False)
        )
        explorer_critic_fallback = bool(
            self.state.explorer_critic is not None
            and getattr(self.state.explorer_critic, "compile_failed", False)
        )
        explorer_target_fallback = bool(
            self.state.explorer_critic_target is not None
            and getattr(
                self.state.explorer_critic_target, "compile_failed", False
            )
        )
        outer_module_fallback = bool(
            getattr(getattr(self.model, "_Qs", None), "compile_failed", False)
        )
        outer_target_fallback = bool(
            getattr(getattr(self.model, "_target_Qs", None), "compile_failed", False)
        )
        critic_fallback = bool(
            critic_region_fallback
            or critic_module_fallback
            or target_module_fallback
            or explorer_critic_fallback
            or explorer_target_fallback
            or outer_module_fallback
            or outer_target_fallback
        )
        return {
            "inner_compile_rollout_fallback": float(rollout_fallback),
            "inner_compile_critic_fallback": float(critic_fallback),
            "inner_compile_actor_fallback": float(actor_fallback),
            "inner_compile_fallback": float(
                rollout_fallback or critic_fallback or actor_fallback
            ),
        }

    def _parameter_noise_metrics(self):
        """Summarize the action-local calibration and realized population."""
        if not self._parameter_noise_active:
            return {}
        sigma_stats = self._stats(self._parameter_noise_sigma_values)
        behavior_stats = self._stats(self._parameter_noise_behavior_action_rms)
        behavior_count = sum(
            int(values.numel())
            for values in self._parameter_noise_behavior_action_rms
        )
        if self._parameter_noise_calibration_rms:
            calibration_values = torch.stack(
                self._parameter_noise_calibration_rms
            )
            calibration_stats = self._stats(calibration_values)
        else:
            calibration_stats = (0.0, 0.0, 0.0, 0.0)
        target_hit_fraction = (
            sum(self._parameter_noise_calibration_hits)
            / len(self._parameter_noise_calibration_hits)
            if self._parameter_noise_calibration_hits
            else 0.0
        )
        bound_hit_fraction = (
            sum(self._parameter_noise_sigma_bound_hits)
            / len(self._parameter_noise_sigma_bound_hits)
            if self._parameter_noise_sigma_bound_hits
            else 0.0
        )
        saturation_fraction = (
            self._parameter_noise_saturation_sum.float()
            / float(self._parameter_noise_saturation_count)
            if self._parameter_noise_saturation_sum is not None
            and self._parameter_noise_saturation_count > 0
            else 0.0
        )
        return {
            "inner_param_noise_actor_count": float(
                self.cfg.inner_param_noise_actor_count
            ),
            "inner_param_noise_rollouts_per_actor": float(
                self.cfg.inner_param_noise_rollouts_per_actor
            ),
            "inner_param_noise_target_action_rms": float(
                self.cfg.inner_param_noise_target_action_rms
            ),
            "inner_param_noise_sigma_initial": float(
                self.cfg.inner_param_noise_sigma_init
            ),
            "inner_param_noise_sigma_final": float(
                self._parameter_noise_stddev
            ),
            "inner_param_noise_sigma_mean": sigma_stats[0],
            "inner_param_noise_sigma_min": sigma_stats[2],
            "inner_param_noise_sigma_max": sigma_stats[3],
            "inner_param_noise_calibration_probes": float(
                self._parameter_noise_calibration_probes
            ),
            "inner_param_noise_calibration_policy_evaluations": float(
                self._parameter_noise_calibration_policy_evaluations
            ),
            "inner_param_noise_calibration_rounds": float(
                len(self._parameter_noise_calibration_hits)
            ),
            "inner_param_noise_calibration_action_rms_count": float(
                len(self._parameter_noise_calibration_rms)
            ),
            "inner_param_noise_calibration_action_rms_mean": calibration_stats[0],
            "inner_param_noise_calibration_action_rms_std": calibration_stats[1],
            "inner_param_noise_calibration_action_rms_min": calibration_stats[2],
            "inner_param_noise_calibration_action_rms_max": calibration_stats[3],
            "inner_param_noise_calibration_target_hit_fraction": float(
                target_hit_fraction
            ),
            "inner_param_noise_sigma_bound_hit_fraction": float(
                bound_hit_fraction
            ),
            "inner_param_noise_behavior_action_rms_count": float(behavior_count),
            "inner_param_noise_behavior_action_rms_mean": behavior_stats[0],
            "inner_param_noise_behavior_action_rms_std": behavior_stats[1],
            "inner_param_noise_behavior_action_rms_min": behavior_stats[2],
            "inner_param_noise_behavior_action_rms_max": behavior_stats[3],
            "inner_param_noise_mean_action_saturation_count": float(
                self._parameter_noise_saturation_count
            ),
            "inner_param_noise_mean_action_saturation_fraction": (
                saturation_fraction
            ),
        }

    def _base_metrics(self, *, active, action_seconds=0.0):
        metrics = {
            "inner_active": float(active),
            "inner_actions": 1.0,
            "inner_model_steps_budget": float(self.cfg.inner_model_step_budget),
            "inner_steps": 0.0,
            "inner_model_steps": 0.0,
            "inner_total_model_steps": 0.0,
            "inner_nominal_model_steps": 0.0,
            "inner_realized_model_steps": 0.0,
            "inner_rounds": 0.0,
            "inner_iterations": 0.0,
            "inner_mppi_iterations": 0.0,
            "inner_rollout_horizon": float(self.cfg.inner_rollout_horizon),
            "inner_rollouts_per_round": float(
                getattr(self.cfg, "inner_rollouts_per_round", 0)
            ),
            "inner_nominal_transitions_per_round": float(
                getattr(self.cfg, "inner_nominal_transitions_per_round", 0)
            ),
            "inner_nominal_updates_per_round": float(
                getattr(self.cfg, "inner_nominal_updates_per_round", 0)
            ),
            "inner_nominal_critic_utd": float(
                getattr(self.cfg, "inner_nominal_critic_utd", 0.0)
            ),
            "inner_steps_per_update": float(
                getattr(self.cfg, "inner_steps_per_update", None) or 0.0
            ),
            "inner_finite_horizon": float(
                getattr(self.cfg, "inner_finite_horizon", False)
            ),
            "inner_outer_replay_fraction_requested": float(
                getattr(self.cfg, "inner_outer_replay_fraction", 0.0)
            ),
            "inner_outer_replay_samples": 0.0,
            "inner_horizon_ratio": float(self.cfg.inner_horizon_ratio),
            "inner_requested_rollouts": 0.0,
            "inner_rollouts": 0.0,
            "inner_updates": 0.0,
            "inner_update_slots": 0.0,
            "inner_requested_update_slots": 0.0,
            "inner_updates_per_round_realized": 0.0,
            "inner_critic_utd": 0.0,
            "inner_actor_utd": 0.0,
            "inner_temperature_utd": 0.0,
            "inner_critic_optimizer_steps": 0.0,
            "inner_actor_optimizer_steps": 0.0,
            "inner_temperature_optimizer_steps": 0.0,
            "inner_target_updates": 0.0,
            "inner_critic_target_updates": 0.0,
            "inner_actor_target_updates": 0.0,
            "inner_policy_evaluations": 0.0,
            "inner_q_evaluations": 0.0,
            "inner_replay_draws": 0.0,
            "inner_buffer_size": 0.0,
            "inner_buffer_capacity": 0.0,
            "inner_buffer_fill_ratio": 0.0,
            "inner_return_mean": 0.0,
            "inner_return_std": 0.0,
            "inner_return_min": 0.0,
            "inner_return_max": 0.0,
            "inner_behavior_reward_sum_mean": 0.0,
            "inner_behavior_discounted_reward_mean": 0.0,
            "inner_rollout_len_mean": 0.0,
            "inner_rollout_len_std": 0.0,
            "inner_rollout_len_min": 0.0,
            "inner_rollout_len_max": 0.0,
            "inner_termination_rate": 0.0,
            "inner_alpha": self.agent.alpha.detach().mean(),
            "inner_alpha_initial": self.agent.alpha.detach().mean(),
            "inner_alpha_final": self.agent.alpha.detach().mean(),
            "inner_alpha_delta": 0.0,
            "inner_target_entropy": (
                float(self._resolved_inner_target_entropy())
                if self.cfg.inner_operator == "sac"
                else 0.0
            ),
            "inner_action_seconds": float(action_seconds),
            "inner_diagnostics_sampled": 0.0,
            "inner_diagnostics_sample_count": 0.0,
            "inner_primary_rollouts": 0.0,
            "inner_explorer_rollouts": 0.0,
            "inner_primary_transitions": 0.0,
            "inner_explorer_transitions": 0.0,
            "inner_primary_replay_fraction": 0.0,
            "inner_explorer_replay_fraction": 0.0,
            "inner_primary_replay_samples": 0.0,
            "inner_explorer_replay_samples": 0.0,
            "inner_primary_replay_sample_fraction": 0.0,
            "inner_explorer_replay_sample_fraction": 0.0,
            "inner_optimization_model_steps": 0.0,
            "inner_selector_model_steps": 0.0,
            "inner_selector_primary_wins": 0.0,
            "inner_selector_explorer_wins": 0.0,
            "inner_selector_score_margin": 0.0,
            "inner_selector_score_variance": 0.0,
            "inner_explorer_actor_optimizer_steps": 0.0,
            "inner_explorer_critic_optimizer_steps": 0.0,
            "inner_explorer_temperature_optimizer_steps": 0.0,
            "inner_explorer_critic_target_updates": 0.0,
            "inner_explorer_target_updates": 0.0,
            "inner_explorer_actor_trainable_params": 0.0,
            "inner_explorer_critic_trainable_params": 0.0,
            "inner_explorer_temperature_trainable_params": 0.0,
            "inner_explorer_alpha": 0.0,
            "inner_explorer_alpha_initial": 0.0,
            "inner_explorer_alpha_final": 0.0,
            "inner_explorer_alpha_delta": 0.0,
            "inner_primary_explorer_action_l2": 0.0,
            "inner_explorer_actor_utd": 0.0,
            "inner_explorer_critic_utd": 0.0,
            "inner_explorer_temperature_utd": 0.0,
            "inner_primary_rollout_fraction": 0.0,
            "inner_explorer_rollout_fraction": 0.0,
            "inner_primary_td_error_abs_count": 0.0,
            "inner_explorer_td_error_abs_count": 0.0,
            "inner_explorer_critic_primary_td_error_abs_count": 0.0,
            "inner_explorer_critic_explorer_td_error_abs_count": 0.0,
        }
        if getattr(self.cfg, "inner_outer_replay_fraction", 0.0) > 0:
            replay = getattr(self, "outer_replay_buffer", None)
            metrics.update(
                inner_outer_replay_fraction=0.0,
                inner_outer_replay_available=float(
                    replay is not None and replay.num_sampleable_transitions > 0
                ),
            )
        metrics.update(self._compile_fallback_metrics())
        return metrics

    def _act_none(
        self,
        root_z,
        *,
        eval_mode,
        start,
        return_behavior_policy=False,
    ):
        capture_behavior = bool(return_behavior_policy and not eval_mode)
        executed = self._execute_policy(
            root_z,
            self.model._pi,
            eval_mode=eval_mode,
            inner_bounds=False,
            return_info=capture_behavior,
        )
        if capture_behavior:
            action, execution_info = executed
            behavior_policy = {
                "pre_tanh_mean": execution_info["pre_tanh_mean"][0].detach(),
                "log_std": execution_info["log_std"][0].detach(),
            }
        else:
            action = executed
            behavior_policy = None
        metrics = self._base_metrics(active=False)
        metrics["inner_policy_evaluations"] = 1.0
        if return_behavior_policy:
            return action[0], metrics, [], behavior_policy
        return action[0], metrics, []

    def _act_rl(
        self,
        root_z,
        *,
        t0,
        eval_mode,
        start,
        return_behavior_policy=False,
    ):
        cfg, state = self.cfg, self.state
        setup_start = self._timer_start()
        # LoRA adapter initialization uses ordinary PyTorch initializers; fork
        # it onto the private optimization stream so act() cannot advance the
        # outer learner's global RNG.
        with self.rng.fork("initialization"):
            self._prepare_workspace(t0=t0)
        self._timer_stop("inner_setup_seconds", setup_start)
        actor_loss_scale = None
        if self._sac_actor_loss_scale_enabled:
            # Outer training owns the running estimator. Each real action uses
            # one immutable snapshot throughout all root-local update slots.
            actor_loss_scale = self.agent.actor_loss_scale.detach().clone()
        alpha_initial = self.alpha.detach().clone()
        trace = self._active_trace
        if trace is not None:
            trace.record("initial", state, {"alpha": alpha_initial})
            if trace.probes:
                trace.probe(self, root_z, state.actor)
        explorer_alpha_initial = (
            self.explorer_alpha.detach().clone()
            if self._explorer_mode == "separate_critics"
            else root_z.new_zeros(())
        )
        allocations = None
        if not self._uses_canonical_schedule:
            allocations = {
                "critic": allocate_across_rounds(
                    cfg.inner_critic_updates_per_action, cfg.inner_rounds
                ),
                "actor": allocate_across_rounds(
                    cfg.inner_actor_updates_per_action, cfg.inner_rounds
                ),
                "temperature": allocate_across_rounds(
                    cfg.inner_temperature_updates_per_action, cfg.inner_rounds
                ),
            }
        all_lengths, reward_sums, discounted_rewards, terminated = [], [], [], []
        transition_rewards_by_source = []
        transition_terminated_by_source = []
        transition_sources = []
        update_history = []
        update_slots = 0
        requested_update_slots = 0
        collected_transition_count = 0
        interval_updates_requested = 0
        for round_index in range(int(cfg.inner_rounds)):
            if trace is not None:
                trace.round_index = round_index + 1
            rollout_start = self._timer_start()
            rollout = self._collect_round(root_z)
            self._timer_stop("inner_rollout_seconds", rollout_start)
            if trace is not None:
                trace.record("collection", state, {
                    "collection_transitions": rollout["transition_count"],
                    "collection_reward_sum_mean": rollout["reward_sums"].float().mean(),
                    "collection_discounted_reward_mean": rollout["discounted_rewards"].float().mean(),
                })
            all_lengths.append(rollout["lengths"])
            reward_sums.append(rollout["reward_sums"])
            discounted_rewards.append(rollout["discounted_rewards"])
            terminated.append(rollout["terminated"])
            if "sources" in rollout:
                transition_rewards_by_source.append(rollout["transition_rewards"])
                transition_terminated_by_source.append(
                    rollout["transition_terminated"]
                )
                transition_sources.append(rollout["transition_sources"])

            update_start = self._timer_start()
            scheduled_update_count = None
            if self._uses_steps_per_update:
                collected_transition_count += int(rollout["transition_count"])
                cumulative_updates = updates_for_transitions(
                    collected_transition_count, cfg.inner_steps_per_update
                )
                scheduled_update_count = cumulative_updates - interval_updates_requested
                interval_updates_requested = cumulative_updates
            if self._uses_canonical_schedule:
                if self._explorer_active or self._uses_component_update_schedule:
                    if self._explorer_active:
                        round_metrics = self._run_two_policy_component_updates(
                            realized_transition_count=rollout["transition_count"],
                            scheduled_update_count=scheduled_update_count,
                            actor_loss_scale=actor_loss_scale
                        )
                        # The realized history length is the exact number of
                        # critic-first + actor-phase slots actually requested,
                        # including episodic canonical-auto compaction.
                        requested_update_slots += len(round_metrics)
                    else:
                        critic_count = int(cfg.inner_critic_updates_per_round)
                        actor_count = int(cfg.inner_actor_updates_per_round)
                        round_metrics = self._run_component_update_counts(
                            critic_count=critic_count,
                            actor_count=actor_count,
                            actor_loss_scale=actor_loss_scale,
                        )
                        requested_update_slots += critic_count + actor_count
                else:
                    configured_updates = cfg.inner_updates_per_round
                    if scheduled_update_count is not None:
                        round_updates = scheduled_update_count
                    elif configured_updates == "auto":
                        # Episodic branches may terminate before H; UTD=1 tracks
                        # transitions actually appended during this collection.
                        round_updates = int(rollout["transition_count"])
                    else:
                        round_updates = int(configured_updates)
                    round_metrics = self._run_update_counts(
                        critic_count=(
                            round_updates
                            if cfg.inner_critic_adaptation != "frozen"
                            else 0
                        ),
                        actor_count=(
                            round_updates
                            if cfg.inner_actor_adaptation != "frozen"
                            else 0
                        ),
                        temperature_count=(
                            round_updates
                            if cfg.inner_operator == "sac"
                            and cfg.inner_temperature_mode == "auto"
                            else 0
                        ),
                        actor_loss_scale=actor_loss_scale,
                    )
                    requested_update_slots += round_updates
            else:
                round_metrics = self._run_updates(
                    round_index,
                    allocations,
                    actor_loss_scale=actor_loss_scale,
                )
                requested_update_slots += max(
                    allocations["critic"][round_index],
                    allocations["actor"][round_index],
                    allocations["temperature"][round_index],
                )
            self._timer_stop("inner_update_seconds", update_start)
            update_history.extend(round_metrics)
            update_slots += len(round_metrics)
            if trace is not None and trace.probes:
                trace.probe(self, root_z, state.actor)

        execution_start = self._timer_start()
        capture_behavior = bool(return_behavior_policy and not eval_mode)
        selector_metrics = {}
        selected_policy = state.actor
        if self._materialized_explorer_active:
            executed, selector_metrics, selected_policy = self._execute_two_policy(
                root_z,
                eval_mode=eval_mode,
                return_info=capture_behavior,
            )
        else:
            if self._parameter_noise_active:
                selector_metrics["inner_selector_primary_wins"] = 1.0
            executed = self._execute_policy(
                root_z,
                state.actor,
                eval_mode=eval_mode,
                return_info=capture_behavior,
            )
        if capture_behavior:
            action, execution_info = executed
            behavior_policy = {
                "pre_tanh_mean": execution_info["pre_tanh_mean"][0].detach(),
                "log_std": execution_info["log_std"][0].detach(),
            }
        else:
            action = executed
            behavior_policy = None
        self._timer_stop("inner_execution_seconds", execution_start)
        if self._collect_diagnostics:
            diagnostic_start = self._timer_start()
            diagnostic_metrics = self._diagnostics(root_z, selected_policy)
            self._timer_stop("inner_diagnostic_seconds", diagnostic_start)
        else:
            diagnostic_metrics = {}

        reward_stats = self._stats(reward_sums)
        discounted_stats = self._stats(discounted_rewards)
        length_stats = self._stats(all_lengths)
        length_values = torch.cat(all_lengths, dim=0) if all_lengths else root_z.new_empty(0)
        terminated_values = (
            torch.cat(terminated, dim=0) if terminated else root_z.new_empty(0)
        )
        source_transition_reward_values = (
            torch.cat(transition_rewards_by_source, dim=0)
            if transition_rewards_by_source
            else root_z.new_empty(0)
        )
        source_transition_terminated_values = (
            torch.cat(transition_terminated_by_source, dim=0)
            if transition_terminated_by_source
            else torch.empty(0, dtype=torch.bool, device=self.device)
        )
        transition_source_values = (
            torch.cat(transition_sources, dim=0)
            if transition_sources
            else torch.empty(0, dtype=torch.uint8, device=self.device)
        )
        termination_stats = self._stats(terminated_values.float())
        rollout_count = int(length_values.numel())
        realized_model_steps = length_values.sum()
        nominal_model_steps = float(
            cfg.inner_rounds
            * cfg.inner_rollouts_per_round
            * cfg.inner_rollout_horizon
        )
        utd_denominator = realized_model_steps.clamp_min(1)
        alpha_final = self.alpha.detach().clone()
        explorer_alpha_final = (
            self.explorer_alpha.detach().clone()
            if self._explorer_mode == "separate_critics"
            else root_z.new_zeros(())
        )
        metrics = self._base_metrics(active=True)
        metrics.update(
            inner_rounds=float(cfg.inner_rounds),
            inner_iterations=float(cfg.inner_rounds),
            inner_rollouts=float(rollout_count),
            inner_requested_rollouts=float(
                cfg.inner_rounds * cfg.inner_rollouts_per_round
            ),
            inner_rollout_count=float(rollout_count),
            inner_steps=realized_model_steps,
            inner_model_steps=realized_model_steps,
            inner_nominal_model_steps=nominal_model_steps,
            inner_realized_model_steps=realized_model_steps,
            inner_updates=float(update_slots),
            inner_update_slots=float(update_slots),
            inner_requested_update_slots=float(requested_update_slots),
            inner_updates_per_round_realized=(
                float(update_slots) / float(cfg.inner_rounds)
                if cfg.inner_rounds
                else 0.0
            ),
            inner_critic_utd=(
                torch.as_tensor(float(state.critic_steps), device=self.device)
                / utd_denominator
            ),
            inner_actor_utd=(
                torch.as_tensor(float(state.actor_steps), device=self.device)
                / utd_denominator
            ),
            inner_temperature_utd=(
                torch.as_tensor(float(state.temperature_steps), device=self.device)
                / utd_denominator
            ),
            inner_critic_optimizer_steps=float(state.critic_steps),
            inner_actor_optimizer_steps=float(state.actor_steps),
            inner_temperature_optimizer_steps=float(state.temperature_steps),
            inner_target_updates=float(state.target_steps),
            inner_critic_target_updates=float(state.critic_target_steps),
            inner_actor_target_updates=float(state.actor_target_steps),
            inner_policy_evaluations=float(state.policy_evaluations),
            inner_q_evaluations=float(state.q_evaluations),
            inner_replay_draws=float(state.replay_draws),
            inner_buffer_size=float(state.replay.size),
            inner_buffer_capacity=float(state.replay.capacity),
            inner_buffer_fill_ratio=float(state.replay.size / state.replay.capacity),
            inner_return_mean=reward_stats[0],
            inner_return_std=reward_stats[1],
            inner_return_min=reward_stats[2],
            inner_return_max=reward_stats[3],
            inner_behavior_reward_sum_mean=reward_stats[0],
            inner_behavior_reward_sum_std=reward_stats[1],
            inner_behavior_reward_sum_min=reward_stats[2],
            inner_behavior_reward_sum_max=reward_stats[3],
            inner_behavior_discounted_reward_mean=discounted_stats[0],
            inner_behavior_discounted_reward_std=discounted_stats[1],
            inner_behavior_discounted_reward_min=discounted_stats[2],
            inner_behavior_discounted_reward_max=discounted_stats[3],
            inner_rollout_len_mean=length_stats[0],
            inner_rollout_len_std=length_stats[1],
            inner_rollout_len_min=length_stats[2],
            inner_rollout_len_max=length_stats[3],
            inner_termination_rate=(
                termination_stats[0]
            ),
            inner_termination_rate_std=termination_stats[1],
            inner_termination_rate_min=termination_stats[2],
            inner_termination_rate_max=termination_stats[3],
            inner_alpha=alpha_final.mean(),
            inner_alpha_initial=alpha_initial.mean(),
            inner_alpha_final=alpha_final.mean(),
            inner_alpha_delta=(alpha_final - alpha_initial).mean(),
            inner_actor_trainable_params=float(state.actor_trainable_count),
            inner_critic_trainable_params=float(state.critic_trainable_count),
            inner_temperature_trainable_params=float(
                state.log_alpha.numel() if state.log_alpha is not None else 0
            ),
            inner_primary_rollouts=float(state.primary_rollouts),
            inner_explorer_rollouts=float(state.explorer_rollouts),
            inner_primary_transitions=float(state.primary_transitions),
            inner_explorer_transitions=float(state.explorer_transitions),
            inner_optimization_model_steps=realized_model_steps,
            inner_explorer_actor_optimizer_steps=float(
                state.explorer_actor_steps
            ),
            inner_explorer_critic_optimizer_steps=float(
                state.explorer_critic_steps
            ),
            inner_explorer_temperature_optimizer_steps=float(
                state.explorer_temperature_steps
            ),
            inner_explorer_critic_target_updates=float(
                state.explorer_critic_target_steps
            ),
            inner_explorer_target_updates=float(
                state.explorer_critic_target_steps
            ),
            inner_explorer_actor_trainable_params=float(
                state.explorer_actor_trainable_count
            ),
            inner_explorer_critic_trainable_params=float(
                state.explorer_critic_trainable_count
            ),
            inner_explorer_temperature_trainable_params=float(
                state.explorer_log_alpha.numel()
                if state.explorer_log_alpha is not None
                else 0
            ),
            inner_explorer_alpha=explorer_alpha_final.mean(),
            inner_explorer_alpha_initial=explorer_alpha_initial.mean(),
            inner_explorer_alpha_final=explorer_alpha_final.mean(),
            inner_explorer_alpha_delta=(
                explorer_alpha_final - explorer_alpha_initial
            ).mean(),
            inner_explorer_actor_utd=(
                torch.as_tensor(
                    float(state.explorer_actor_steps), device=self.device
                )
                / utd_denominator
            ),
            inner_explorer_critic_utd=(
                torch.as_tensor(
                    float(state.explorer_critic_steps), device=self.device
                )
                / utd_denominator
            ),
            inner_explorer_temperature_utd=(
                torch.as_tensor(
                    float(state.explorer_temperature_steps), device=self.device
                )
                / utd_denominator
            ),
            inner_primary_rollout_fraction=(
                float(state.primary_rollouts) / float(max(1, rollout_count))
            ),
            inner_explorer_rollout_fraction=(
                float(state.explorer_rollouts) / float(max(1, rollout_count))
            ),
        )
        metrics.update(selector_metrics)
        if self._explorer_active:
            replay_sources = state.replay.source[: state.replay.size].float()
            explorer_replay_fraction = (
                replay_sources.mean() if replay_sources.numel() else root_z.new_zeros(())
            )
            metrics["inner_explorer_replay_fraction"] = explorer_replay_fraction
            metrics["inner_primary_replay_fraction"] = 1.0 - explorer_replay_fraction
            for source_value, source_name in ((0, "primary"), (1, "explorer")):
                selected_rewards = source_transition_reward_values[
                    transition_source_values == source_value
                ]
                selected_terminated = source_transition_terminated_values[
                    transition_source_values == source_value
                ]
                reward_source_stats = self._stats(selected_rewards)
                termination_source_stats = self._stats(selected_terminated.float())
                for suffix, value in zip(
                    ("mean", "std", "min", "max"), reward_source_stats
                ):
                    metrics[f"inner_{source_name}_reward_{suffix}"] = value
                for suffix, value in zip(
                    ("mean", "std", "min", "max"), termination_source_stats
                ):
                    metrics[
                        f"inner_{source_name}_termination_rate_{suffix}"
                    ] = value
                metrics[f"inner_{source_name}_termination_rate"] = (
                    termination_source_stats[0]
                )

        metrics.update(self._parameter_noise_metrics())
        if actor_loss_scale is not None:
            metrics["inner_actor_loss_scale"] = actor_loss_scale.reshape(())
            metrics["inner_effective_alpha"] = (
                alpha_final / actor_loss_scale
            ).mean()
        if self._collect_diagnostics and state.sampled_ids:
            sampled = torch.cat(state.sampled_ids)
            metrics["inner_replay_unique_fraction"] = (
                sampled.unique().numel() / sampled.numel()
            )
        if state.sampled_sources:
            sampled_sources = torch.cat(state.sampled_sources).reshape(-1)
            total_samples = int(sampled_sources.numel())
            primary_samples = (sampled_sources == 0).sum()
            explorer_samples = (sampled_sources == 1).sum()
            metrics["inner_primary_replay_samples"] = primary_samples
            metrics["inner_explorer_replay_samples"] = explorer_samples
            metrics["inner_primary_replay_sample_fraction"] = (
                primary_samples.float() / float(max(1, total_samples))
            )
            metrics["inner_explorer_replay_sample_fraction"] = (
                explorer_samples.float() / float(max(1, total_samples))
            )
        metrics.update(self._average_update_metrics(update_history))
        metrics["inner_outer_replay_samples"] = sum(
            (item.get("outer_replay_samples", 0.0) for item in update_history), 0.0
        )
        metrics.update(diagnostic_metrics)
        metrics["inner_total_model_steps"] = (
            metrics["inner_model_steps"]
            + metrics.get("inner_diagnostic_model_steps", 0.0)
            + metrics.get("inner_selector_model_steps", 0.0)
        )
        if self._collect_diagnostics:
            metrics["inner_diagnostics_sampled"] = 1.0
            metrics["inner_diagnostics_sample_count"] = 1.0
            metrics["inner_diagnostics_step"] = float(self.action_index)
        # Counters include the fixed-evaluator calls performed above.
        metrics["inner_policy_evaluations"] = float(state.policy_evaluations)
        metrics["inner_q_evaluations"] = float(state.q_evaluations)
        q_scale = (
            metrics["inner_q_abs_mean"]
            if "inner_q_abs_mean" in metrics
            else metrics.get("inner_fixed_target_q_abs_mean")
        )
        if q_scale is not None:
            q_scale = torch.as_tensor(q_scale, device=self.device)
            metrics["inner_q_abs_mean"] = q_scale
            metrics["inner_alpha_to_abs_q"] = (
                self.alpha.detach().mean() / q_scale.clamp_min(1e-8)
            )
        if cfg.episodic:
            # The agent packs this with the action and scalar metrics in the
            # single unavoidable host transfer at the action boundary.
            lengths_host = length_values.detach()
        else:
            lengths_host = [int(cfg.inner_rollout_horizon)] * rollout_count
        if return_behavior_policy:
            return action[0], metrics, lengths_host, behavior_policy
        return action[0], metrics, lengths_host

    @torch.no_grad()
    def _act_mppi(self, root_z, *, t0, eval_mode, start):
        from .mppi import mppi_plan

        iterations = self._mppi_iterations
        scope = str(self.cfg.inner_mppi_warm_start_scope)
        if scope == "action" or (scope == "episode" and t0):
            previous_mean = None
        else:
            previous_mean = self._mppi_prev_mean
        planner_start = self._timer_start()
        with self.rng.fork("mppi") as generator:
            result = mppi_plan(
                model=self.model,
                root_z=root_z,
                horizon=self.cfg.inner_rollout_horizon,
                iterations=iterations,
                num_samples=self.cfg.inner_mppi_num_samples,
                num_elites=self.cfg.inner_mppi_num_elites,
                num_pi_trajs=self.cfg.inner_mppi_num_pi_trajs,
                temperature=self.cfg.inner_mppi_temperature,
                min_std=self.cfg.inner_mppi_min_std,
                max_std=self.cfg.inner_mppi_max_std,
                discount=float(self.agent.discount),
                q_reduction=self.cfg.mppi_terminal_q_reduction,
                termination_threshold=self.cfg.inner_termination_threshold,
                generator=generator,
                previous_mean=previous_mean,
                t0=t0,
                eval_mode=eval_mode,
                materialize_metrics=False,
            )
        self._timer_stop("inner_mppi_seconds", planner_start)
        self._mppi_prev_mean = None if scope == "action" else result.next_mean.detach()
        metrics = self._base_metrics(active=True)
        metrics.update(result.metrics)
        metrics.update(
            # ``inner_rounds`` remains a metric alias for one compatibility
            # release; MPPI execution is controlled only by the dedicated key.
            inner_rounds=float(iterations),
            inner_iterations=float(iterations),
            inner_mppi_iterations=float(iterations),
            inner_rollouts=float(
                iterations * self.cfg.inner_mppi_num_samples
                + self.cfg.inner_mppi_num_pi_trajs
            ),
            inner_requested_rollouts=float(
                iterations * self.cfg.inner_mppi_num_samples
                + self.cfg.inner_mppi_num_pi_trajs
            ),
            inner_steps=float(result.model_steps),
            inner_model_steps=float(result.model_steps),
            inner_total_model_steps=float(result.model_steps),
            inner_policy_evaluations=float(
                result.metrics["planner_policy_evaluations"]
            ),
            inner_q_evaluations=float(result.metrics["planner_q_evaluations"]),
        )
        if self._collect_diagnostics:
            diagnostic_start = self._timer_start()
            with torch.no_grad():
                if hasattr(self.model, "pi_action"):
                    outer_action = self.model.pi_action(root_z, deterministic=True)
                else:
                    outer_action, _ = self.model.pi(root_z, deterministic=True)
                outer_predictions = self.model.q_predictions(
                    root_z, outer_action, target=True
                )
                improved_predictions = self.model.q_predictions(
                    root_z, result.action.unsqueeze(0), target=True
                )
                outer_q = self.model.q_backend.reduce(
                    self.model.q_backend.decode(outer_predictions), "mean_all"
                )
                improved_q = self.model.q_backend.reduce(
                    self.model.q_backend.decode(improved_predictions), "mean_all"
                )
            gain = (improved_q - outer_q).mean()
            metrics["inner_fixed_target_q_action_gain"] = gain
            metrics["inner_outer_q_gain"] = gain
            metrics["inner_fixed_target_q_outer"] = outer_q.mean()
            metrics["inner_fixed_target_q_improved"] = improved_q.mean()
            q_scale = torch.stack(
                (outer_q.abs().mean(), improved_q.abs().mean())
            ).mean()
            metrics["inner_fixed_target_q_abs_mean"] = q_scale
            metrics["inner_fixed_evaluator_alpha"] = self.agent.alpha.detach().mean()
            metrics["inner_q_abs_mean"] = q_scale
            metrics["inner_alpha_to_abs_q"] = (
                self.agent.alpha.detach().mean() / q_scale.clamp_min(1e-8)
            )
            proposal_action = result.next_mean[0].clamp(-1.0, 1.0).unsqueeze(0)
            proposal_delta = torch.linalg.vector_norm(
                proposal_action - outer_action, dim=-1
            ).mean()
            metrics["inner_proposal_mean_delta_l2"] = proposal_delta
            metrics["inner_policy_mean_delta_l2"] = proposal_delta
            if self.cfg.q_representation == "distributional":
                probabilities = torch.softmax(
                    torch.cat((outer_predictions, improved_predictions), dim=1),
                    dim=-1,
                )
                metrics["inner_distributional_q_entropy"] = -(
                    probabilities * probabilities.clamp_min(1e-12).log()
                ).sum(dim=-1).mean()
                metrics["inner_distributional_q_edge_mass"] = (
                    probabilities[..., 0] + probabilities[..., -1]
                ).mean()
            metrics["inner_policy_evaluations"] += 1.0
            metrics["inner_q_evaluations"] += 2.0
            metrics["inner_diagnostics_sampled"] = 1.0
            metrics["inner_diagnostics_sample_count"] = 1.0
            metrics["inner_diagnostics_step"] = float(self.action_index)
            self._timer_stop("inner_diagnostic_seconds", diagnostic_start)
        candidate_lengths = [int(self.cfg.inner_rollout_horizon)] * int(
            iterations * self.cfg.inner_mppi_num_samples
        )
        policy_lengths = [max(0, int(self.cfg.inner_rollout_horizon) - 1)] * int(
            self.cfg.inner_mppi_num_pi_trajs
        )
        return result.action, metrics, candidate_lengths + policy_lengths

    @torch.no_grad()
    def _apply_control_prior_writeback(self):
        """Assimilate final action-local SAC weights into online outer priors."""

        state, cfg = self.state, self.cfg
        actor_coef = float(cfg.inner_actor_writeback_coef)
        critic_coef = float(cfg.inner_critic_writeback_coef)
        applied = {
            "inner_actor_writeback_coef": actor_coef,
            "inner_critic_writeback_coef": critic_coef,
            "inner_actor_writeback_applied": 0.0,
            "inner_critic_writeback_applied": 0.0,
        }
        if actor_coef > 0.0 and state.actor is None:
            raise RuntimeError(
                "Actor prior write-back requires a live adapted inner actor."
            )
        if critic_coef > 0.0 and state.critic is None:
            raise RuntimeError(
                "Critic prior write-back requires a live adapted inner critic."
            )

        # Preflight both components before either mutation so an invalid joint
        # request cannot leave only one persistent prior written back.
        if actor_coef > 0.0:
            polyak_update(state.actor, self.model._pi, actor_coef)
            applied["inner_actor_writeback_applied"] = 1.0
        if critic_coef > 0.0:
            # Deliberately update only the online value prior. The persistent
            # target critic retains its ordinary real-update EMA cadence.
            polyak_update(state.critic, self.model._Qs, critic_coef)
            applied["inner_critic_writeback_applied"] = 1.0
        return applied

    def act(
        self,
        root_z,
        *,
        t0=False,
        eval_mode=False,
        collect_diagnostics=True,
        return_behavior_policy=False,
        apply_inner_writeback=False,
        trace=None,
    ):
        """Run an action with an optional single-use observational recorder."""
        if trace is not None:
            from .inner_trace import InnerActionTrace

            if not isinstance(trace, InnerActionTrace):
                raise TypeError("trace must be an InnerActionTrace.")
            if self._active_trace is not None:
                raise RuntimeError("Inner action tracing is not reentrant.")
            operator = str(self.cfg.inner_operator)
            if operator not in {"none", "sac", "td3"} or self._explorer_active:
                raise ValueError(
                    "Inner tracing supports single-policy none/SAC/TD3 configurations."
                )
            if trace.probes and operator == "td3":
                raise ValueError(
                    "Fixed-noise trace probes require SAC or no inner optimization."
                )
            trace.begin()
        self._active_trace = trace
        try:
            return self._act(
                root_z, t0=t0, eval_mode=eval_mode,
                collect_diagnostics=collect_diagnostics,
                return_behavior_policy=return_behavior_policy,
                apply_inner_writeback=apply_inner_writeback,
            )
        finally:
            self._active_trace = None

    def _act(
        self,
        root_z,
        *,
        t0=False,
        eval_mode=False,
        collect_diagnostics=True,
        return_behavior_policy=False,
        apply_inner_writeback=False,
    ):
        self._pending_timers = {}
        start = self._timer_start()
        self.action_index += 1
        self._collect_diagnostics = bool(collect_diagnostics)
        with self.rng.action_fork():
            operator = str(self.cfg.inner_operator)
            inactive = operator == "none" or (
                operator == "mppi"
                and self._mppi_iterations == 0
            ) or (
                operator in {"sac", "td3"}
                and int(self.cfg.inner_rounds) == 0
            )
            if inactive:
                if self._active_trace is not None:
                    self._active_trace.record(
                        "initial", self.state, {"alpha": self.agent.alpha.detach()}
                    )
                    if self._active_trace.probes:
                        self._active_trace.probe(
                            self, root_z, self.model._pi, inner=False
                        )
                result = self._act_none(
                    root_z,
                    eval_mode=eval_mode,
                    start=start,
                    return_behavior_policy=return_behavior_policy,
                )
                if return_behavior_policy:
                    action, metrics, lengths, behavior_policy = result
                else:
                    action, metrics, lengths = result
            elif operator == "mppi":
                action, metrics, lengths = self._act_mppi(
                    root_z, t0=t0, eval_mode=eval_mode, start=start
                )
            else:
                result = self._act_rl(
                    root_z,
                    t0=t0,
                    eval_mode=eval_mode,
                    start=start,
                    return_behavior_policy=return_behavior_policy,
                )
                if return_behavior_policy:
                    action, metrics, lengths, behavior_policy = result
                else:
                    action, metrics, lengths = result

                writeback_active = bool(
                    self.cfg.inner_actor_writeback_coef > 0.0
                    or self.cfg.inner_critic_writeback_coef > 0.0
                )
                if writeback_active:
                    metrics.update(
                        {
                            "inner_actor_writeback_coef": float(
                                self.cfg.inner_actor_writeback_coef
                            ),
                            "inner_critic_writeback_coef": float(
                                self.cfg.inner_critic_writeback_coef
                            ),
                            "inner_actor_writeback_applied": 0.0,
                            "inner_critic_writeback_applied": 0.0,
                        }
                    )
                    if bool(apply_inner_writeback) and not bool(eval_mode):
                        metrics.update(self._apply_control_prior_writeback())

            self._timer_stop("inner_action_seconds", start)

            # Refresh after diagnostics/planning, since those calls may be the
            # first invocation that discovers an unsupported compiled critic.
            metrics.update(self._compile_fallback_metrics())

            # Action-scoped tensors are explicitly released after producing
            # the action; episode/run scopes survive by configuration.
            self._clear_expired(t0=False, include_action=True)
        if return_behavior_policy:
            if operator != "sac":
                behavior_policy = None
            return action, metrics, lengths, behavior_policy
        return action, metrics, lengths
