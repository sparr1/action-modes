"""Componentized AMBI inner-loop improvement strategies.

All state in this module is private to action selection. The outer world model,
optimizers, target critics, and entropy coefficient are never mutated here.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import time

import torch

from .common import math as td_math
from .common.compile_regions import CompileRegion
from .common.inner_utils import (
    InnerRNG,
    allocate_across_rounds,
    copy_lora_adapters_,
    lora_uses_shared_bases,
    rebase_clone_with_target_,
    rebase_lora_base_,
    reset_lora_adapters_,
    trainable_parameter_count,
)
from .common.latent_buffer import LatentReplayBuffer
from .common.lora import lorafy_copy, lorafy_shared, trainable_parameters


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
        self._pending_timers = {}
        self._initialize_compile_regions()

    def _initialize_compile_regions(self):
        enabled = bool(getattr(self.cfg, "compile", False))
        strict = bool(getattr(self.cfg, "compile_strict", False))
        operator = str(self.cfg.inner_operator)
        critic_kernel = (
            self._td3_critic_kernel if operator == "td3" else self._sac_critic_kernel
        )
        actor_kernel = (
            self._td3_actor_kernel if operator == "td3" else self._sac_actor_kernel
        )
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

    @property
    def alpha(self):
        if self.state.log_alpha is not None:
            return self.state.log_alpha.exp()
        if self.state.alpha_fixed is not None:
            return self.state.alpha_fixed
        return self.agent.alpha.detach()

    def clear_all(self):
        self.state = InnerWorkspace()
        self._action_pool = InnerWorkspace()
        self.rng = InnerRNG(self.cfg.seed, self.device)
        self.action_index = 0
        self.episode_index = 0
        self._mppi_prev_mean = None
        self._pending_timers = {}
        # A checkpoint load invalidates action-local module identities. Rebuild
        # only the non-serialized compile callables; model/checkpoint keys stay
        # untouched and ordinary action/episode resets retain their cache.
        self._initialize_compile_regions()

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

        if self._scope_expires(cfg.inner_replay_scope, t0=t0, include_action=include_action):
            if str(cfg.inner_replay_scope) == "action" and state.replay is not None:
                self._action_pool.replay = state.replay
            state.replay = None

    @staticmethod
    def _reset_optimizer(optimizer):
        """Reset Adam state in-place while preserving allocated moment tensors."""
        if optimizer is None:
            return
        optimizer.zero_grad(set_to_none=True)
        for parameter_state in optimizer.state.values():
            for value in parameter_state.values():
                if torch.is_tensor(value):
                    value.zero_()

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
            and cfg.inner_actor_updates_per_action > 0
        ):
            state.actor_optim = self._new_optimizer(state.actor, "actor")
        if (
            state.critic_optim is None
            and cfg.inner_critic_adaptation != "frozen"
            and cfg.inner_critic_updates_per_action > 0
        ):
            state.critic_optim = self._new_optimizer(state.critic, "critic")

        mode = str(cfg.inner_temperature_mode)
        if mode == "inherit_outer":
            state.log_alpha = state.temperature_optim = None
            if (
                str(cfg.inner_temperature_scope) == "action"
                and self._action_pool.alpha_fixed is not None
            ):
                state.alpha_fixed = self._action_pool.alpha_fixed
                self._action_pool.alpha_fixed = None
                state.alpha_fixed.copy_(self.agent.alpha.detach())
            else:
                state.alpha_fixed = self.agent.alpha.detach().clone()
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
                if (
                    str(cfg.inner_temperature_scope) == "action"
                    and self._action_pool.log_alpha is not None
                ):
                    state.log_alpha = self._action_pool.log_alpha
                    self._action_pool.log_alpha = None
                    state.log_alpha.data.fill_(float(cfg.inner_temperature)).log_()
                    state.temperature_optim = self._action_pool.temperature_optim
                    self._action_pool.temperature_optim = None
                    self._reset_optimizer(state.temperature_optim)
                else:
                    state.log_alpha = torch.nn.Parameter(
                        torch.log(
                            torch.tensor(float(cfg.inner_temperature), device=self.device)
                        )
                    )
            state.alpha_fixed = None
            if (
                state.temperature_optim is None
                and cfg.inner_temperature_updates_per_action > 0
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
        state.critic.train(cfg.inner_critic_adaptation != "frozen")
        if state.critic_target is not None:
            state.critic_target.eval()
        if state.actor_target is not None:
            state.actor_target.eval()

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
    ):
        kwargs = {}
        if inner_bounds:
            kwargs.update(
                log_std_min=self.cfg.inner_log_std_min,
                log_std_max=self.cfg.inner_log_std_max,
            )
        if mode == "policy_sample":
            if hasattr(self.model, "pi_action"):
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
        if hasattr(self.model, "pi_action"):
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
    def _collect_round(self, root_z):
        cfg, state = self.cfg, self.state
        count = int(cfg.inner_rollouts_per_round)
        horizon = int(cfg.inner_rollout_horizon)
        if count == 0:
            empty = root_z.new_empty((0,))
            return {
                "lengths": empty.to(dtype=torch.long),
                "reward_sums": empty,
                "discounted_rewards": empty,
                "terminated": empty.to(dtype=torch.bool),
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
        with self.rng.fork("collection") as generator:
            for _ in range(horizon):
                active = torch.nonzero(alive, as_tuple=False).squeeze(-1)
                if active.numel() == 0:
                    break
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
            state.replay.add_batch(*(torch.cat(values, dim=0) for values in transition_fields))

        if cfg.inner_actor_adaptation != "frozen":
            state.actor.train()
        return {
            "lengths": lengths,
            "reward_sums": reward_sums,
            "discounted_rewards": discounted_rewards,
            "terminated": terminated_rollout,
        }

    @torch.no_grad()
    def _dense_rollout_kernel(self, root_z, policy_noise, reward_support):
        """Pure fixed-shape rollout; replay/counters are updated by the caller."""
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
            transitions.append(
                torch.cat((z, action, reward, next_z, terminated), dim=-1)
            )
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
        return batch

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
    ):
        """Pure SAC critic loss region; the optimizer step stays eager."""
        state, cfg = self.state, self.cfg
        with torch.no_grad():
            next_action, next_info = self.model.pi(
                next_z,
                policy=state.actor,
                noise=policy_noise,
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
            target_q = reward + float(self.agent.discount) * (
                1.0 - terminated
            ) * (next_q - alpha * next_info["log_prob"])

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
            log_std_min=cfg.inner_log_std_min,
            log_std_max=cfg.inner_log_std_max,
        )
        zero = info["log_prob"].new_zeros(())
        if not update_actor:
            return info["log_prob"], info["entropy"], zero, zero, zero

        q_pi = self.model.Q(
            z,
            action,
            qs=state.critic,
            detach=True,
            reduction=cfg.inner_q_actor_reduction,
            pair_indices=pair_indices,
            trusted_pair_indices=pair_indices is not None,
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
        )

    def _sac_policy_step(self, batch, *, update_temperature, update_actor, alpha):
        state, cfg = self.state, self.cfg
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
            log_prob, entropy, actor_loss, q_mean, kl_mean = (
                self._compile_regions["actor"](
                    batch["z"],
                    alpha,
                    policy_noise,
                    pair_indices,
                    update_actor,
                )
            )
            state.policy_evaluations += batch_size

            metrics = {}
            if update_temperature:
                target_entropy = (
                    -float(cfg.action_dim)
                    if cfg.inner_target_entropy == "auto"
                    else float(cfg.inner_target_entropy)
                )
                temperature_loss = -(
                    state.log_alpha * (log_prob + target_entropy).detach()
                ).mean()
                state.temperature_optim.zero_grad(set_to_none=True)
                temperature_loss.backward()
                temperature_grad_norm = torch.nn.utils.clip_grad_norm_(
                    [state.log_alpha], float(cfg.inner_temperature_grad_clip_norm)
                )
                state.temperature_optim.step()
                state.temperature_steps += 1
                metrics.update(
                    temperature_loss=temperature_loss.detach(),
                    temperature_grad_norm=torch.as_tensor(
                        temperature_grad_norm
                    ).detach(),
                )

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
                    actor_entropy=entropy.detach().mean(),
                    outer_policy_kl=kl_mean.detach(),
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
        q_pi = self.model.Q(
            z,
            action,
            qs=state.critic,
            detach=True,
            reduction=cfg.inner_q_actor_reduction,
            pair_indices=pair_indices,
            trusted_pair_indices=pair_indices is not None,
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
        return actor_loss, q_pi.mean(), anchor_l2.mean()

    def _td3_actor_step(self, batch):
        state, cfg = self.state, self.cfg
        batch_size = int(batch["z"].shape[0])
        pair_indices = self._sample_pair_indices(
            self.rng.generator("gradient_policy")
        )
        actor_loss, q_mean, anchor_l2_mean = self._compile_regions["actor"](
            batch["z"],
            pair_indices,
        )
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

    def _run_updates(self, round_index, allocations):
        critic_count = allocations["critic"][round_index]
        actor_count = allocations["actor"][round_index]
        temperature_count = allocations["temperature"][round_index]
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
                    with self.rng.fork("bootstrap"):
                        slot_metrics.update(self._sac_critic_step(batch, alpha))
                if do_temperature or do_actor:
                    slot_metrics.update(
                        self._sac_policy_step(
                            batch,
                            update_temperature=do_temperature,
                            update_actor=do_actor,
                            alpha=alpha,
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
            metrics.append(slot_metrics)
        return metrics

    @torch.no_grad()
    def _execute_policy(self, root_z, policy, *, eval_mode):
        mode = "mean" if eval_mode else str(self.cfg.inner_execution_action)
        std_scale = float(self.cfg.inner_execution_std_scale)
        if mode == "policy_sample" and std_scale == 0.0:
            mode = "mean"
        was_training = policy.training
        policy.eval()
        with self.rng.fork("execution") as generator:
            action, _ = self._policy_action(
                root_z,
                policy,
                mode=mode,
                generator=generator,
                std_scale=max(std_scale, 1e-12),
                noise_std=self.cfg.inner_execution_noise_std,
            )
        self.state.policy_evaluations += int(root_z.shape[0])
        policy.train(was_training)
        return action

    @torch.no_grad()
    def _evaluate_policy_trajectory(self, root_z, policy, generator, *, stochastic):
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
                log_std_min=self.cfg.inner_log_std_min,
                log_std_max=self.cfg.inner_log_std_max,
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
            log_std_min=self.cfg.inner_log_std_min,
            log_std_max=self.cfg.inner_log_std_max,
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
            if hasattr(self.model, "pi_action"):
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
        for key, values in grouped.items():
            if not values:
                continue
            stacked = torch.stack(values)
            prefix = f"inner_{key}"
            metrics[prefix] = stacked.mean()
            metrics[f"{prefix}_std"] = stacked.std(unbiased=False)
            metrics[f"{prefix}_min"] = stacked.min()
            metrics[f"{prefix}_max"] = stacked.max()
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
        actor_fallback = self._compile_regions["actor"].failed
        critic_module_fallback = bool(
            self.state.critic is not None
            and getattr(self.state.critic, "compile_failed", False)
        )
        target_module_fallback = bool(
            self.state.critic_target is not None
            and getattr(self.state.critic_target, "compile_failed", False)
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

    def _base_metrics(self, *, active, action_seconds=0.0):
        metrics = {
            "inner_active": float(active),
            "inner_actions": 1.0,
            "inner_model_steps_budget": float(self.cfg.inner_model_step_budget),
            "inner_steps": 0.0,
            "inner_model_steps": 0.0,
            "inner_total_model_steps": 0.0,
            "inner_rounds": 0.0,
            "inner_iterations": 0.0,
            "inner_rollout_horizon": float(self.cfg.inner_rollout_horizon),
            "inner_horizon_ratio": float(self.cfg.inner_horizon_ratio),
            "inner_requested_rollouts": 0.0,
            "inner_rollouts": 0.0,
            "inner_updates": 0.0,
            "inner_update_slots": 0.0,
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
            "inner_action_seconds": float(action_seconds),
            "inner_diagnostics_sampled": 0.0,
            "inner_diagnostics_sample_count": 0.0,
        }
        metrics.update(self._compile_fallback_metrics())
        return metrics

    def _act_none(self, root_z, *, eval_mode, start):
        action = self._execute_policy(root_z, self.model._pi, eval_mode=eval_mode)
        metrics = self._base_metrics(active=False)
        metrics["inner_policy_evaluations"] = 1.0
        return action[0], metrics, []

    def _act_rl(self, root_z, *, t0, eval_mode, start):
        cfg, state = self.cfg, self.state
        setup_start = self._timer_start()
        # LoRA adapter initialization uses ordinary PyTorch initializers; fork
        # it onto the private optimization stream so act() cannot advance the
        # outer learner's global RNG.
        with self.rng.fork("initialization"):
            self._prepare_workspace(t0=t0)
        self._timer_stop("inner_setup_seconds", setup_start)
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
        update_history = []
        update_slots = 0
        for round_index in range(int(cfg.inner_rounds)):
            rollout_start = self._timer_start()
            rollout = self._collect_round(root_z)
            self._timer_stop("inner_rollout_seconds", rollout_start)
            all_lengths.append(rollout["lengths"])
            reward_sums.append(rollout["reward_sums"])
            discounted_rewards.append(rollout["discounted_rewards"])
            terminated.append(rollout["terminated"])

            update_start = self._timer_start()
            round_metrics = self._run_updates(round_index, allocations)
            self._timer_stop("inner_update_seconds", update_start)
            update_history.extend(round_metrics)
            update_slots += len(round_metrics)

        execution_start = self._timer_start()
        action = self._execute_policy(root_z, state.actor, eval_mode=eval_mode)
        self._timer_stop("inner_execution_seconds", execution_start)
        if self._collect_diagnostics:
            diagnostic_start = self._timer_start()
            diagnostic_metrics = self._diagnostics(root_z, state.actor)
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
        termination_stats = self._stats(terminated_values.float())
        rollout_count = int(length_values.numel())
        metrics = self._base_metrics(active=True)
        metrics.update(
            inner_rounds=float(cfg.inner_rounds),
            inner_iterations=float(cfg.inner_rounds),
            inner_rollouts=float(rollout_count),
            inner_requested_rollouts=float(
                cfg.inner_rounds * cfg.inner_rollouts_per_round
            ),
            inner_rollout_count=float(rollout_count),
            inner_steps=length_values.sum(),
            inner_model_steps=length_values.sum(),
            inner_updates=float(update_slots),
            inner_update_slots=float(update_slots),
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
            inner_alpha=self.alpha.detach().mean(),
            inner_actor_trainable_params=float(state.actor_trainable_count),
            inner_critic_trainable_params=float(state.critic_trainable_count),
            inner_temperature_trainable_params=float(
                state.log_alpha.numel() if state.log_alpha is not None else 0
            ),
        )
        if self._collect_diagnostics and state.sampled_ids:
            sampled = torch.cat(state.sampled_ids)
            metrics["inner_replay_unique_fraction"] = (
                sampled.unique().numel() / sampled.numel()
            )
        metrics.update(self._average_update_metrics(update_history))
        metrics.update(diagnostic_metrics)
        metrics["inner_total_model_steps"] = (
            metrics["inner_model_steps"]
            + metrics.get("inner_diagnostic_model_steps", 0.0)
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
        return action[0], metrics, lengths_host

    @torch.no_grad()
    def _act_mppi(self, root_z, *, t0, eval_mode, start):
        from .mppi import mppi_plan

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
                iterations=self.cfg.inner_rounds,
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
            inner_rounds=float(self.cfg.inner_rounds),
            inner_iterations=float(self.cfg.inner_rounds),
            inner_rollouts=float(
                self.cfg.inner_rounds * self.cfg.inner_mppi_num_samples
                + self.cfg.inner_mppi_num_pi_trajs
            ),
            inner_requested_rollouts=float(
                self.cfg.inner_rounds * self.cfg.inner_mppi_num_samples
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
            self.cfg.inner_rounds * self.cfg.inner_mppi_num_samples
        )
        policy_lengths = [max(0, int(self.cfg.inner_rollout_horizon) - 1)] * int(
            self.cfg.inner_mppi_num_pi_trajs
        )
        return result.action, metrics, candidate_lengths + policy_lengths

    def act(self, root_z, *, t0=False, eval_mode=False, collect_diagnostics=True):
        self._pending_timers = {}
        start = self._timer_start()
        self.action_index += 1
        self._collect_diagnostics = bool(collect_diagnostics)
        with self.rng.action_fork():
            operator = str(self.cfg.inner_operator)
            if operator == "none" or self.cfg.inner_rounds == 0:
                action, metrics, lengths = self._act_none(
                    root_z, eval_mode=eval_mode, start=start
                )
            elif operator == "mppi":
                action, metrics, lengths = self._act_mppi(
                    root_z, t0=t0, eval_mode=eval_mode, start=start
                )
            else:
                action, metrics, lengths = self._act_rl(
                    root_z, t0=t0, eval_mode=eval_mode, start=start
                )

            self._timer_stop("inner_action_seconds", start)

            # Refresh after diagnostics/planning, since those calls may be the
            # first invocation that discovers an unsupported compiled critic.
            metrics.update(self._compile_fallback_metrics())

            # Action-scoped tensors are explicitly released after producing
            # the action; episode/run scopes survive by configuration.
            self._clear_expired(t0=False, include_action=True)
        return action, metrics, lengths
