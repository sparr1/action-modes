"""Root-batched, fresh SAC teachers with independent parameters and Adam moments.

The leading parameter/replay axis is a root, never a shared policy minibatch.
Losses average within each root and sum across roots; clipping is root-local.
The fixed schedule has one shared Adam step counter because every root executes
the same updates. All moment tensors and target parameters retain the root axis.
"""

import hashlib
from copy import copy
import math
import time

import torch

from .common import math as td_math
from .common.compile_regions import CompileRegion, _preserve_host_rng_state
from .common.ensemble_vmap import ensemble_vmap_spec, _tensor_mlp_vmap
from .common.inner_utils import allocate_across_rounds
from .common.layers import NormedLinear
from .common.scale import percentile_range
from .inner_improvement import InnerImprovementEngine
from .teacher_graph import validate_teacher_graph_config


BATCHED_TEACHER_PROTOCOL = "root_batched_sac_chunk_seed_v1"
BATCHED_TEACHER_OPTIMIZER = "adam_fused_cuda_eager_cpu_v1"


def validate_batched_teacher_config(cfg):
    """The static fresh-SAC subset; native reanalysis keeps its broader support."""
    validate_teacher_graph_config(cfg)
    if getattr(cfg, "inner_actor_loss_scale_scope", "action") != "action":
        raise ValueError("Batched teachers require inner_actor_loss_scale_scope='action'.")


def _dropout_spec(spec, training):
    return tuple((*row[:3], None if row[3] is None else (*row[3][:-1], training)) for row in spec)


def clip_root_grad_norm_(parameters, maximum):
    """The ordinary global parameter norm, independently for each root."""
    parameters = tuple(parameters)
    norms = torch.stack([torch.linalg.vector_norm(p.grad.reshape(p.shape[0], -1), dim=1)
                         for p in parameters if p.grad is not None])
    total = torch.linalg.vector_norm(norms, dim=0)
    coefficient = (float(maximum) / (total + 1e-6)).clamp(max=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient.reshape(-1, *([1] * (parameter.ndim - 1))))
    return total


def _clipped_root_gradients(gradients, maximum):
    """Pure clipping region; keep every root's norm and scale independent."""
    norms = torch.stack([torch.linalg.vector_norm(g.reshape(g.shape[0], -1), dim=1)
                         for g in gradients])
    total = torch.linalg.vector_norm(norms, dim=0)
    coefficient = (float(maximum) / (total + 1e-6)).clamp(max=1.0)
    return tuple(g * coefficient.reshape(-1, *([1] * (g.ndim - 1))) for g in gradients)


def _lerp_targets_(targets, online, tau):
    """Explicit tensor mutation; target allocations remain stable for capture."""
    torch._foreach_lerp_(targets, online, tau)


class BatchedTeacherCore:
    """One reusable root-batched workspace; orchestration never loops over roots.

    Tensor-only regions use the existing functional MLP vmap and Q/policy
    codecs. ``draws`` is an optional explicit-randomness seam for independent
    eager reference tests; production uses isolated default Torch draws.
    """

    def __init__(self, agent, width, *, compiled=None, capturable=False, optimization_features=None):
        validate_batched_teacher_config(agent.cfg)
        self.agent, self.model, self.cfg = agent, agent.model, agent.cfg
        self.device, self.width = torch.device(agent.device), int(width)
        available = frozenset({"selective_q", "actor_cache", "fused_updates"})
        self.optimization_features = (available if getattr(self.cfg, "reanalyze_batched_optimizations", True)
                                      else frozenset()) if optimization_features is None else frozenset(optimization_features)
        if not self.optimization_features <= available:
            raise ValueError("Unknown batched teacher optimization feature.")
        if self.width < 1:
            raise ValueError("Batched teacher width must be positive.")
        self.helper = InnerImprovementEngine(agent)
        self.actor_base = self.helper._actor_base
        self.critic_base = self.helper._critic_base
        self.outer_target = self.helper._critic_outer_target
        actor_spec = ensemble_vmap_spec((self.actor_base,), NormedLinear)
        critic_spec = ensemble_vmap_spec(tuple(self.critic_base), NormedLinear)
        if actor_spec is None or critic_spec is None:
            raise ValueError("Batched teachers require supported dense, hook-free actor and critic MLPs.")
        self.actor_spec = _dropout_spec(actor_spec, False)
        self.critic_eval_spec = _dropout_spec(critic_spec, False)
        self.critic_train_spec = _dropout_spec(critic_spec, bool(self.cfg.inner_critic_dropout_enabled))
        self.heads = int(self.cfg.num_q)
        # Decode with the existing representation/codec, validating the selected
        # head axis rather than changing the shared model's ensemble signature.
        self.selected_q_backend = copy(self.model.q_backend)
        self.selected_q_backend.num_q = int(self.cfg.q_pair_size)
        self.actor_sources = dict(self.actor_base.named_parameters())
        self.critic_sources = tuple(dict(head.named_parameters()) for head in self.critic_base)
        self.target_sources = tuple(dict(head.named_parameters()) for head in self.outer_target)
        self.actor = {k: torch.nn.Parameter(v.detach().expand(self.width, *v.shape).clone())
                      for k, v in self.actor_sources.items()}
        self.critic = {k: torch.nn.Parameter(torch.stack([p[k].detach() for p in self.critic_sources])
                                           .expand(self.width, self.heads, *self.critic_sources[0][k].shape).clone())
                       for k in self.critic_sources[0]}
        self.target = {k: torch.empty_like(v) for k, v in self.critic.items()}
        self.prior_actor = {k: torch.empty_like(v) for k, v in self.actor.items()}
        self.prior_critic = {k: torch.empty_like(v) for k, v in self.critic.items()}
        self.root = torch.zeros(self.width, self.cfg.latent_dim, device=self.device)
        self.alpha_input = self.root.new_ones(())
        self.scale_input = self.root.new_ones(1)
        self.scale = self.root.new_ones(self.width, 1, 1)
        self.scaled_q = self.helper._sac_actor_loss_scale_enabled
        self.entropy_enabled = bool(self.cfg.inner_entropy_enabled)
        self.log_alpha = torch.nn.Parameter(self.root.new_zeros(self.width, 1, 1))
        self.alpha_fixed = torch.empty_like(self.log_alpha)
        self.actor_optim = self._optimizer(self.actor.values(), "actor", capturable)
        self.critic_optim = self._optimizer(self.critic.values(), "critic", capturable)
        self.temperature_optim = (
            self._optimizer((self.log_alpha,), "temperature", capturable)
            if self.cfg.inner_temperature_mode == "auto" and self.entropy_enabled else None
        )
        # Allocate optimizer state outside capture, with no dummy learning step.
        for optimizer in self.optimizers:
            for parameter in optimizer.param_groups[0]["params"]:
                optimizer.state[parameter] = {
                    "step": torch.zeros((), device=self.device if capturable or self.device.type == "cuda" else "cpu"),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
        self.capacity = int(self.cfg.inner_replay_capacity)
        widths = (self.cfg.latent_dim, self.cfg.action_dim, 1, self.cfg.latent_dim, 1)
        if self.cfg.inner_finite_horizon:
            widths += (1,)
        self.field_widths = tuple(int(x) for x in widths)
        self.replay = self.root.new_empty(self.width, self.capacity, sum(widths))
        self.schedule = self._schedule()
        self.counts = {
            "inner_total_model_steps": float(self.cfg.inner_rounds * self.cfg.inner_rollouts_per_round * self.cfg.inner_rollout_horizon),
            "inner_actor_optimizer_steps": float(sum(a for group in self.schedule for c, a, t in group)),
            "inner_critic_optimizer_steps": float(sum(c for group in self.schedule for c, a, t in group)),
            "inner_temperature_optimizer_steps": float(sum(t for group in self.schedule for c, a, t in group)),
        }
        enabled = bool(self.cfg.compile) if compiled is None else bool(compiled)
        self.regions = {name: CompileRegion("batched_teacher_" + name, method, enabled=enabled, strict=True)
                        for name, method in (("rollout", self._rollout), ("critic", self._critic_loss),
                                             ("actor", self._actor_loss))}
        if "actor_cache" in self.optimization_features:
            self.regions["actor_cache"] = CompileRegion(
                "batched_teacher_actor_cache", self._cache_actor_outputs, enabled=enabled, strict=True,
            )
        if "fused_updates" in self.optimization_features:
            self.regions["clip"] = CompileRegion(
                "batched_teacher_clip", _clipped_root_gradients, enabled=enabled, strict=True,
            )
            self.regions["target_update"] = CompileRegion(
                "batched_teacher_target_update", _lerp_targets_, enabled=enabled, strict=True,
            )
        self.refresh_inputs()

    @property
    def optimizers(self):
        return tuple(o for o in (self.actor_optim, self.critic_optim, self.temperature_optim) if o is not None)

    def _optimizer(self, params, component, capturable):
        epsilon = getattr(self.cfg, "inner_actor_adam_eps", None) if component == "actor" else None
        if epsilon is None:
            epsilon = self.cfg.inner_adam_eps
        return torch.optim.Adam(list(params), lr=float(getattr(self.cfg, f"inner_{component}_lr")),
                                eps=float(epsilon), foreach=False, fused=self.device.type == "cuda",
                                capturable=capturable)

    def _schedule(self):
        cfg, helper = self.cfg, self.helper
        result = []
        for round_index in range(int(cfg.inner_rounds)):
            if helper._uses_component_update_schedule:
                c, a = int(cfg.inner_critic_updates_per_round), int(cfg.inner_actor_updates_per_round)
                result.append([(True, False, False)] * c +
                              [(False, True, cfg.inner_temperature_mode == "auto" and self.entropy_enabled)] * a)
            else:
                if helper._uses_canonical_schedule:
                    count = cfg.inner_updates_per_round
                    count = cfg.inner_rollouts_per_round * cfg.inner_rollout_horizon if count == "auto" else int(count)
                    c = a = count
                    t = count if cfg.inner_temperature_mode == "auto" and self.entropy_enabled else 0
                else:
                    c, a, t = (allocate_across_rounds(getattr(cfg, f"inner_{key}_updates_per_action"), cfg.inner_rounds)[round_index]
                               for key in ("critic", "actor", "temperature"))
                    t = t if self.entropy_enabled else 0
                result.append([(i < c, i < a, i < t) for i in range(max(c, a, t))])
        return result

    @torch.no_grad()
    def refresh_inputs(self):
        mode = self.cfg.inner_temperature_mode
        if mode == "inherit_outer":
            value = self.agent.alpha.detach()
            if self.agent.log_ent_coef is not None:
                value = value.clamp_min(1e-8)
        elif mode == "fixed":
            value = self.alpha_input.new_full((), float(self.cfg.inner_temperature))
        else:
            value = self.helper._initial_inner_alpha()
        if not bool((torch.isfinite(value) & (value >= 0)).all()):
            raise ValueError("Batched teacher temperature must be finite and nonnegative.")
        self.alpha_input.copy_(value.reshape(()))
        if self.scaled_q:
            self.scale_input.copy_(self.agent.actor_loss_scale.detach())

    @torch.no_grad()
    def reset(self):
        for key, source in self.actor_sources.items():
            self.actor[key].copy_(source.detach().unsqueeze(0))
            self.prior_actor[key].copy_(self.actor[key])
        target_outer = self.cfg.inner_bootstrap_source == "outer_target" or self.helper._initialize_critic_from_outer_target
        for key in self.critic:
            online = torch.stack([head[key].detach() for head in self.critic_sources])
            self.critic[key].copy_(online.unsqueeze(0))
            self.prior_critic[key].copy_(self.critic[key])
            initial = torch.stack([head[key].detach() for head in self.target_sources]) if target_outer else online
            self.target[key].copy_(initial.unsqueeze(0))
        self.alpha_fixed.copy_(self.alpha_input)
        self.log_alpha.copy_(self.alpha_input.clamp_min(1e-8).log())
        self.scale.copy_(self.scale_input)
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
            for state in optimizer.state.values():
                for value in state.values():
                    value.zero_()
        self.position = self.size = self.critic_steps = 0

    def _actor(self, parameters, z):
        return _tensor_mlp_vmap(parameters, z, self.actor_spec, True)

    def _pi(self, parameters, z, noise, *, prior=False, std_scale=1.0, raw_output=None):
        options = dict(self.helper._actor_options) if prior else {
            "log_std_mapping": self.cfg.inner_log_std_mapping,
            "log_std_min": self.cfg.inner_log_std_min, "log_std_max": self.cfg.inner_log_std_max,
        }
        policy = (lambda x: self._actor(parameters, x)) if raw_output is None else (lambda x: raw_output)
        return self.model.pi(z, policy=policy, noise=noise,
                             std_scale=std_scale, include_scaled_entropy=self.cfg.inner_actor_entropy_mode == "tdmpc2_scaled",
                             **options)

    def _predictions(self, parameters, z, action, masks=None, *, training=False, pairs=None):
        heads = self.heads
        if pairs is not None:
            heads = pairs.shape[1]
            parameters = {
                key: value.gather(1, pairs.reshape(self.width, heads, *([1] * (value.ndim - 2)))
                                  .expand(self.width, heads, *value.shape[2:]))
                for key, value in parameters.items()
            }
            if masks is not None:
                masks = {
                    key: value.reshape(self.width, self.heads, *value.shape[1:]).gather(
                        1, pairs[:, :, None, None].expand(-1, -1, value.shape[1], value.shape[2]),
                    ).flatten(0, 1)
                    for key, value in masks.items()
                }
        joined = torch.cat((z, action), dim=-1)
        inputs = joined[:, None].expand(-1, heads, -1, -1).reshape(self.width * heads, joined.shape[1], -1)
        parameters = {k: v.flatten(0, 1) for k, v in parameters.items()}
        spec = self.critic_train_spec if training else self.critic_eval_spec
        predictions = _tensor_mlp_vmap(parameters, inputs, spec, True, masks)
        return predictions.reshape(self.width, heads, z.shape[1], -1).movedim(1, 0)

    def _values(self, parameters, z, action, masks=None, *, training=False):
        return self.model.q_backend.decode(self._predictions(parameters, z, action, masks, training=training)).movedim(0, 1)

    def _reduced_values(self, parameters, z, action, reduction, pairs, masks=None, *, training=False):
        if "selective_q" in self.optimization_features and reduction.endswith("_pair") and pairs is not None:
            predictions = self._predictions(parameters, z, action, masks, training=training, pairs=pairs)
            values = self.selected_q_backend.decode(predictions).movedim(0, 1)
            return self._reduce(values, reduction, None)
        return self._reduce(self._values(parameters, z, action, masks, training=training), reduction, pairs)

    def _reduce(self, values, reduction, pairs):
        if reduction.endswith("_pair") and pairs is not None:
            values = values.gather(1, pairs[:, :, None, None].expand(-1, -1, values.shape[2], 1))
        return values.min(dim=1).values if reduction.startswith("min_") else values.mean(dim=1)

    def _entropy(self, info):
        return info["scaled_entropy"] if self.cfg.inner_actor_entropy_mode == "tdmpc2_scaled" else -info["log_prob"]

    def _rollout(self, noise):
        cfg = self.cfg
        z = self.root[:, None].expand(-1, cfg.inner_rollouts_per_round, -1)
        rows = []
        for depth in range(int(cfg.inner_rollout_horizon)):
            mode, std = cfg.inner_behavior_action, float(cfg.inner_behavior_std_scale)
            if mode == "policy_sample" and std > 0:
                action, _ = self._pi(self.actor, z, noise[:, depth], std_scale=std)
            else:
                action, _ = self._pi(self.actor, z, torch.zeros_like(noise[:, depth]))
                if mode == "mean_plus_gaussian":
                    action = (action + noise[:, depth] * float(cfg.inner_behavior_noise_std)).clamp(-1, 1)
            flat_z, flat_a = z.flatten(0, 1), action.flatten(0, 1)
            joint = self.model.joint_input(flat_z, flat_a)
            reward = self.model.decode_reward(self.model.reward_from_joint(joint)).reshape(self.width, -1, 1)
            next_z = self.model.next_from_joint(joint).reshape_as(z)
            terminal = torch.zeros_like(reward)
            fields = (z, action, reward, next_z, terminal)
            if cfg.inner_finite_horizon:
                fields += (torch.full_like(terminal, float(depth == cfg.inner_rollout_horizon - 1)),)
            rows.append(torch.cat(fields, dim=-1))
            z = next_z
        return torch.cat(rows, dim=1)

    @torch.no_grad()
    def append(self, packed):
        count, old = packed.shape[1], self.position
        write = old
        if count >= self.capacity:
            packed = packed[:, -self.capacity:]
            write = (old + count) % self.capacity
        first = min(packed.shape[1], self.capacity - write)
        self.replay[:, write:write + first].copy_(packed[:, :first])
        if first < packed.shape[1]:
            self.replay[:, :packed.shape[1] - first].copy_(packed[:, first:])
        self.position = (old + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def _cache_actor_outputs(self, next_z):
        # Store untransformed network outputs. The ordinary policy codec still
        # applies bounds, fresh noise, tanh and the configured entropy per update.
        return self._actor(self.actor, next_z)

    def _cache_replay_actor_outputs(self):
        # Match the critic minibatch's matrix shapes AND packed-field layout.
        # Evaluating the whole replay at once can select different TF32 kernels
        # from the ordinary B-row actor forward. The scratch batch also keeps
        # input strides/offsets identical for full and padded final tiles.
        batch_size = int(self.cfg.inner_batch_size)
        packed = self.replay.new_empty(self.width, batch_size, self.replay.shape[-1])
        offset = sum(self.field_widths[:3])
        next_z = packed[..., offset:offset + self.cfg.latent_dim]
        outputs = []
        for start in range(0, self.size, batch_size):
            valid = min(batch_size, self.size - start)
            packed[:, :valid].copy_(self.replay[:, start:start + valid])
            if valid < batch_size:
                packed[:, valid:].zero_()
            outputs.append(self.regions["actor_cache"](next_z)[:, :valid])
        return torch.cat(outputs, dim=1)

    def _clip(self, parameters, maximum):
        parameters = tuple(p for p in parameters if p.grad is not None)
        if "fused_updates" not in self.optimization_features:
            clip_root_grad_norm_(parameters, maximum)
            return
        gradients = self.regions["clip"](tuple(p.grad for p in parameters), maximum)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient

    def _update_targets(self):
        with torch.no_grad():
            if "fused_updates" in self.optimization_features:
                # Multi-tensor EMA keeps target storage stable for CUDA replay.
                self.regions["target_update"](tuple(self.target.values()), tuple(self.critic.values()),
                                              float(self.cfg.inner_critic_target_tau))
            else:
                for key in self.target:
                    self.target[key].lerp_(self.critic[key], float(self.cfg.inner_critic_target_tau))

    def _critic_loss(self, batch, alpha, noise, pairs, masks, prior_noise, prior_pairs, next_policy_output=None):
        fields = batch.split(self.field_widths, dim=-1)
        z, action, reward, next_z, terminated = fields[:5]
        with torch.no_grad():
            next_action, info = self._pi(self.actor, next_z, noise, raw_output=next_policy_output)
            source = self.prior_critic if self.cfg.inner_bootstrap_source == "outer_online" else self.target
            bootstrap = self._reduced_values(source, next_z, next_action, self.cfg.inner_q_target_reduction, pairs)
            if self.cfg.inner_sac_critic_target == "entropy_augmented":
                coefficient = alpha * self.scale if self.scaled_q else alpha
                bootstrap = bootstrap + coefficient * self._entropy(info)
            if self.cfg.inner_finite_horizon:
                prior_action, _ = self._pi(self.prior_actor, next_z, prior_noise, prior=True)
                tail = self._reduced_values(self.prior_critic, next_z, prior_action, self.cfg.mppi_terminal_q_reduction, prior_pairs)
                bootstrap = torch.where(fields[5].bool(), tail, bootstrap)
                bootstrap = torch.where(terminated == 1, 0., bootstrap)
            target = reward + float(self.agent.discount) * (1 - terminated) * bootstrap
        predictions = self._predictions(self.critic, z, action, masks, training=True)
        losses = self.model.critic_loss(predictions, target, reduction="none")
        return float(self.cfg.inner_critic_loss_coef) * losses.mean(dim=(0, 2, 3))

    def _actor_loss(self, z, alpha, noise, pairs, masks, update_actor):
        action, info = self._pi(self.actor, z, noise)
        entropy = self._entropy(info)
        scale = self.scale.detach()
        loss = entropy.new_zeros(self.width)
        if update_actor:
            critic = {k: v.detach() for k, v in self.critic.items()}
            q = self._reduced_values(critic, z, action, self.cfg.inner_q_actor_reduction, pairs, masks, training=True)
            if self.scaled_q and self.cfg.inner_actor_loss_scale_update == "per_update":
                with torch.no_grad():
                    robust = percentile_range(q.detach().movedim(0, 1), q.new_tensor([5., 95.])).unsqueeze(-1)
                    scale = scale.lerp(robust, float(self.cfg.sac_actor_loss_scale_tau))
            actor_q = q / scale if self.scaled_q else q
            values = -actor_q - alpha * entropy if self.entropy_enabled else -actor_q
            if float(self.cfg.inner_outer_policy_kl_coef) > 0:
                with torch.no_grad():
                    _, prior = self._pi(self.prior_actor, z, torch.zeros_like(noise), prior=True)
                values = values + float(self.cfg.inner_outer_policy_kl_coef) * td_math.diagonal_gaussian_reverse_kl(
                    info["pre_tanh_mean"], info["log_std"], prior["pre_tanh_mean"], prior["log_std"],
                )
            loss = values.mean(dim=(1, 2))
        return loss, entropy.detach(), scale.detach()

    def _draw(self, kind, shape, upper=None):
        if kind == "normal":
            return torch.randn(shape, device=self.device)
        if kind == "uniform":
            return torch.rand(shape, device=self.device)
        return torch.randint(upper, shape, device=self.device)

    def _pairs(self, reduction, draw):
        if not reduction.endswith("_pair") or self.cfg.q_pair_size == self.heads:
            return None
        return draw("uniform", (self.width, self.heads)).argsort(dim=-1)[:, :self.cfg.q_pair_size]

    def _masks(self, batch_size, draw):
        masks = {}
        for index, row in enumerate(self.critic_train_spec):
            norm = row[3]
            if norm is not None and norm[-2] and norm[-1]:
                p = norm[-2]
                uniform = draw("uniform", (self.width, self.heads, batch_size, row[1]))
                masks[index] = ((uniform >= p).to(self.root.dtype) / (1 - p)).flatten(0, 1)
        return masks if masks else None

    def run(self, draws=None):
        draw = self._draw if draws is None else draws
        self.reset()
        cfg = self.cfg
        for slots in self.schedule:
            noise = draw("normal", (self.width, cfg.inner_rollout_horizon, cfg.inner_rollouts_per_round, cfg.action_dim))
            with torch.no_grad():
                self.append(self.regions["rollout"](noise))
            next_policy_outputs = None
            batch_size = int(cfg.inner_batch_size)
            padded_cache_rows = ((self.size + batch_size - 1) // batch_size) * batch_size
            if ("actor_cache" in self.optimization_features and self.helper._uses_component_update_schedule
                    and sum(c for c, a, t in slots) > 1
                    and padded_cache_rows <= batch_size * sum(c for c, a, t in slots)):
                # Component schedules place all critic-only updates before any
                # actor update. Rebuild after every append/actor block; never
                # retain these values across rounds, roots, or solve resets.
                with torch.no_grad():
                    next_policy_outputs = self._cache_replay_actor_outputs()
            for do_critic, do_actor, do_temperature in slots:
                if cfg.inner_replay_sampling == "with_replacement":
                    indices = draw("integer", (self.width, cfg.inner_batch_size), self.size)
                else:
                    if cfg.inner_batch_size > self.size:
                        raise ValueError("Cannot sample batched latent replay without replacement beyond its size.")
                    indices = draw("uniform", (self.width, self.size)).argsort(dim=-1)[:, :cfg.inner_batch_size]
                batch = self.replay.gather(1, indices[:, :, None].expand(-1, -1, self.replay.shape[-1]))
                alpha = (self.log_alpha.exp().clamp_min(1e-8).detach() if cfg.inner_temperature_mode == "auto"
                         else self.alpha_fixed)
                if not self.entropy_enabled:
                    alpha = torch.zeros_like(alpha)
                if do_critic:
                    noise = draw("normal", (self.width, cfg.inner_batch_size, cfg.action_dim))
                    pairs = self._pairs(cfg.inner_q_target_reduction, draw)
                    prior_noise = draw("normal", noise.shape) if cfg.inner_finite_horizon else None
                    prior_pairs = self._pairs(cfg.mppi_terminal_q_reduction, draw) if cfg.inner_finite_horizon else None
                    masks = self._masks(cfg.inner_batch_size, draw)
                    next_output = (next_policy_outputs.gather(
                        1, indices[:, :, None].expand(-1, -1, 2 * cfg.action_dim),
                    ) if next_policy_outputs is not None else None)
                    loss = self.regions["critic"](batch, alpha, noise, pairs, masks, prior_noise, prior_pairs, next_output)
                    self.critic_optim.zero_grad(set_to_none=True)
                    loss.sum().backward()
                    self._clip(self.critic.values(), cfg.inner_critic_grad_clip_norm)
                    self.critic_optim.step()
                    self.critic_steps += 1
                if do_actor or do_temperature:
                    noise = draw("normal", (self.width, cfg.inner_batch_size, cfg.action_dim))
                    pairs = self._pairs(cfg.inner_q_actor_reduction, draw) if do_actor else None
                    masks = self._masks(cfg.inner_batch_size, draw) if do_actor else None
                    z = batch[..., :cfg.latent_dim]
                    loss, entropy, scale = self.regions["actor"](z, alpha, noise, pairs, masks, do_actor)
                    if do_actor:
                        self.actor_optim.zero_grad(set_to_none=True)
                        loss.sum().backward()
                        self._clip(self.actor.values(), cfg.inner_actor_grad_clip_norm)
                        self.actor_optim.step()
                        self.scale.copy_(scale)
                        next_policy_outputs = None
                    if do_temperature and self.temperature_optim is not None:
                        target = self.helper._resolved_inner_target_entropy()
                        loss = (self.log_alpha * (entropy - target)).mean(dim=(1, 2)).sum()
                        self.temperature_optim.zero_grad(set_to_none=True)
                        loss.backward()
                        self._clip((self.log_alpha,), self.helper._temperature_grad_clip_maximum)
                        self.temperature_optim.step()
                        with torch.no_grad():
                            self.log_alpha.clamp_(min=math.log(1e-8))
                if do_critic and cfg.inner_bootstrap_source == "inner_target" and self.critic_steps % cfg.inner_critic_target_update_interval == 0:
                    self._update_targets()
        with torch.no_grad():
            stats = self.model.policy_stats(self.root[:, None], policy=lambda x: self._actor(self.actor, x),
                                           log_std_mapping=cfg.inner_log_std_mapping,
                                           log_std_min=cfg.inner_log_std_min, log_std_max=cfg.inner_log_std_max)
            return stats["pre_tanh_mean"][:, 0].clone(), stats["log_std"][:, 0].clone()


class BatchedTeacherSolver:
    """Synchronous root batches on one stream, with optional whole-batch capture.

    The ordered root seeds deterministically seed each fixed-width chunk.
    Chunk width is therefore part of exact-resume identity. This protocol is
    distinct from the native and per-root-stream protocols; it never mutates
    the live controller's state or global random streams.
    """

    def __init__(self, agent):
        validate_batched_teacher_config(agent.cfg)
        self.agent, self.cfg = agent, agent.cfg
        self.device = torch.device(agent.device)
        self.width = int(self.cfg.reanalyze_parallel_roots)
        self.capture = self.cfg.reanalyze_backend == "batched_cuda_graph"
        if self.capture and self.device.type != "cuda":
            raise ValueError("batched_cuda_graph requires a CUDA agent.")
        self.core = self.graph = self.output = self.stream = None
        self._active = False

    def _seed(self, seeds):
        message = BATCHED_TEACHER_PROTOCOL.encode() + b"".join(int(seed).to_bytes(8, "little") for seed in seeds)
        return int.from_bytes(hashlib.sha256(message).digest()[:8], "little") % (2**63 - 1)

    def _setup(self):
        self.core = BatchedTeacherCore(self.agent, self.width, capturable=self.capture)
        if not self.capture:
            return
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            from torch._dynamo.utils import counters
            for _ in range(3):
                self.output = self.core.run()
            compiled = int(counters["stats"]["unique_graphs"])
            self.output = self.core.run()
            if int(counters["stats"]["unique_graphs"]) != compiled:
                raise RuntimeError("Batched teacher compilation did not stabilize before capture.")
            self.stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.output = self.core.run()
        self.stream.synchronize()

    def solve(self, roots, seeds):
        if (not torch.is_tensor(roots) or roots.ndim != 2 or roots.shape[1] != self.cfg.latent_dim
                or roots.dtype != torch.float32 or roots.device != self.core_device):
            raise ValueError("Batched teacher roots must be float32 [roots, latent] on the agent device.")
        if not isinstance(seeds, (list, tuple)) or len(seeds) != len(roots) or any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63 for seed in seeds
        ):
            raise ValueError("Batched teachers require one nonnegative int64 seed per root.")
        if self._active:
            raise RuntimeError("Batched teacher solves are not reentrant.")
        modes = tuple((module, module.training) for module in self.agent.modules())
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        setup_seconds = 0.
        means, log_stds = [], []
        self._active = True
        failed = True
        try:
            with _preserve_host_rng_state(), torch.random.fork_rng(devices=devices), torch.enable_grad():
                self.agent.model.eval()
                if self.core is None:
                    started = time.perf_counter()
                    torch.random.default_generator.manual_seed(0)
                    if devices:
                        with torch.cuda.device(self.device):
                            torch.cuda.manual_seed(0)
                    self._setup()
                    setup_seconds = time.perf_counter() - started
                self.core.refresh_inputs()
                for start in range(0, len(roots), self.width):
                    chunk = roots[start:start + self.width]
                    seed = self._seed(seeds[start:start + self.width])
                    torch.random.default_generator.manual_seed(seed)
                    if devices:
                        with torch.cuda.device(self.device):
                            torch.cuda.manual_seed(seed)
                    # Padding is independent work and never contributes labels,
                    # replay accounting, or the work counters returned below.
                    with torch.no_grad():
                        self.core.root.zero_()
                        self.core.root[:len(chunk)].copy_(chunk)
                    if self.capture:
                        self.stream.wait_stream(torch.cuda.current_stream(self.device))
                        with torch.cuda.stream(self.stream):
                            self.graph.replay()
                            means.append(self.output[0][:len(chunk)].clone())
                            log_stds.append(self.output[1][:len(chunk)].clone())
                        torch.cuda.current_stream(self.device).wait_stream(self.stream)
                    else:
                        output = self.core.run()
                        means.append(output[0][:len(chunk)])
                        log_stds.append(output[1][:len(chunk)])
                if devices:
                    torch.cuda.current_stream(self.device).synchronize()
                mean = torch.cat(means) if means else roots.new_empty((0, self.cfg.action_dim))
                log_std = torch.cat(log_stds) if log_stds else mean.clone()
                if not bool(torch.isfinite(mean).all() & torch.isfinite(log_std).all()):
                    raise ValueError("Batched teacher parameters must be finite.")
                counts = roots.new_tensor([self.core.counts[key] * len(roots) for key in (
                    "inner_total_model_steps", "inner_actor_optimizer_steps", "inner_critic_optimizer_steps")])
                failed = False
                return mean, log_std, counts, {
                    "distillation_reanalyze_batched": 1.,
                    "distillation_reanalyze_cuda_graph": float(self.capture),
                    "distillation_reanalyze_fused_adam": float(self.device.type == "cuda"),
                    "distillation_reanalyze_parallel_roots": float(self.width),
                    "distillation_reanalyze_batched_workers": 1.,
                    "distillation_reanalyze_selective_q": float("selective_q" in self.core.optimization_features),
                    "distillation_reanalyze_actor_cache": float("actor_cache" in self.core.optimization_features),
                    "distillation_reanalyze_fused_updates": float("fused_updates" in self.core.optimization_features),
                    "distillation_reanalyze_chunks": float(math.ceil(len(roots) / self.width)),
                    "distillation_reanalyze_padding_roots": float((-len(roots)) % self.width),
                    "distillation_reanalyze_executed_model_steps": float(math.ceil(len(roots) / self.width) * self.width * self.core.counts["inner_total_model_steps"]),
                    "distillation_reanalyze_setup_seconds": setup_seconds,
                }
        finally:
            try:
                if self.stream is not None:
                    try:
                        self.stream.synchronize()
                    except Exception:
                        if not failed:
                            raise
            finally:
                for module, training in modes:
                    module.training = training
                self._active = False
                if failed:
                    # A failed lazy setup must not become a half-built runner.
                    self.core = self.graph = self.output = self.stream = None

    @property
    def core_device(self):
        if self.device.type == "cuda" and self.device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return self.device
