"""Cached inner-policy distillation for the persistent AMBI actor."""

import math

import torch

from .common import math as td_math
from .common.entropy import policy_entropy
from .common.policy_diagnostics import policy_sample_diagnostics
from .common.scale import percentile_range
from .common.training_state import require_exact_keys, require_tensor
from .reanalyze import LazyReanalyzer


class PriorDistillation:
    """Own the imitation statistic and isolated refresh state, not the critic."""

    def __init__(self, agent):
        self.agent = agent
        self.scale = torch.ones(1, device=agent.device, dtype=torch.float32)
        self.percentiles = torch.tensor([5.0, 95.0], device=agent.device, dtype=torch.float32)
        self.actor_updates = 0
        self.reanalyzer = LazyReanalyzer(agent)

    def specification(self):
        specification = {
            "objective": "distillation",
            "teacher": "unscaled_adapted_primary_inner_sac",
            "direction": "prior_to_teacher",
            "action_reduction": "mean",
            "temporal_reduction": "valid_weighted_mean_times_mean_rho",
            "scale": {"percentiles": [5, 95], "tau": 0.01, "floor": 1.0},
            "reanalyze_interval": int(self.agent.cfg.reanalyze_interval),
            "reanalyze_batch_size": int(self.agent.cfg.reanalyze_batch_size),
        }
        if self.reanalyzer.backend == "cuda_graph":
            specification.update(self.reanalyzer.backend_specification())
        return specification

    def state_dict(self):
        return {
            "scale": self.scale.detach().clone(),
            "actor_updates": self.actor_updates,
            "reanalyzer": self.reanalyzer.state_dict(),
        }

    def preflight(self, spec, state, *, num_updates):
        if spec != self.specification():
            raise ValueError("Checkpoint distillation specification does not match this agent.")
        require_exact_keys(state, {"scale", "actor_updates", "reanalyzer"}, "distillation state")
        scale = require_tensor(state["scale"], "distillation scale", shape=(1,), dtype=torch.float32)
        if not bool(torch.isfinite(scale).all()) or not bool((scale >= 1).all()):
            raise ValueError("Distillation scale must be finite and at least one.")
        count = state["actor_updates"]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= num_updates:
            raise ValueError("Distillation actor_updates must lie between zero and num_updates.")
        self.reanalyzer.preflight_state_dict(state["reanalyzer"])
        return state

    def load_state_dict(self, state):
        self.scale.copy_(state["scale"].to(self.scale))
        self.actor_updates = state["actor_updates"]
        self.reanalyzer.load_state_dict(state["reanalyzer"])

    def _inputs(self, zs, teacher):
        if teacher is None:
            raise ValueError("Distillation requires cached teacher metadata.")
        # zs contains H+1 states, whereas teachers supervise its first H states.
        shape = (*zs[:-1].shape[:-1], self.agent.cfg.action_dim)
        for name in ("mean", "log_std"):
            require_tensor(teacher[name], f"teacher {name}", shape=shape, dtype=torch.float32)
        for name, dtype in (("valid", torch.bool), ("version", torch.int64)):
            require_tensor(teacher[name], f"teacher {name}", shape=(*shape[:-1], 1), dtype=dtype)
        valid = teacher["valid"].to(zs.device).squeeze(-1)
        mean = teacher["mean"].to(zs.device).detach()
        log_std = teacher["log_std"].to(zs.device).detach()
        version = teacher["version"].to(zs.device).squeeze(-1)
        if not bool(torch.isfinite(mean[valid]).all()) or not bool(torch.isfinite(log_std[valid]).all()):
            raise ValueError("Valid teacher parameters must be finite.")
        if bool(((version[valid] < 0) | (version[valid] > self.agent.outer_version)).any()):
            raise ValueError("Valid teacher version must refer to an existing outer snapshot.")
        return mean, log_std, valid, version

    @torch.no_grad()
    def _update_q_scale(self, zs, action, *, alpha, scale_snapshot):
        """Keep the independent Q statistic needed by existing inner/TD recipes."""
        agent = self.agent
        if not agent._actor_q_range_enabled:
            return {}
        if action is None:
            action, _ = agent.model.pi(zs)
        if agent.model.value_spec.is_split:
            _, scale = agent._outer_value_coefficients(alpha, scale_snapshot)
            components = agent.model.reduce_components(
                agent.model.q_values(zs, action.detach(), detach=True),
                agent.cfg.outer_q_actor_reduction, projection="policy", beta=alpha * scale,
            )
            q = agent.model.project_values(components, projection="return")
        else:
            q = agent.model.Q(zs, action.detach(), reduction=agent.cfg.outer_q_actor_reduction, detach=True)
        instantaneous = agent._update_actor_loss_scale(q[0])
        return {
            "actor_q_range_instant": instantaneous.detach().clone(),
            "actor_q_range_ema": agent._actor_loss_scale_value.detach().clone(),
            "actor_loss_scale": agent._actor_loss_scale_value.detach().clone(),
        }

    def update(self, zs, teacher, *, alpha_snapshot=None, scale_snapshot=None):
        agent = self.agent
        zs = zs.detach()
        mean, log_std, valid, version = self._inputs(zs, teacher)
        weights = agent._transition_temporal_weights[:, None].expand_as(valid)
        supported = valid & (weights > 0)
        count = int(valid.sum().item())
        alpha = agent.alpha.detach().clone() if alpha_snapshot is None else alpha_snapshot.detach()
        zero = zs.new_zeros(())
        metrics = {
            "distillation_valid_count": float(count),
            "distillation_valid_fraction": float(count) / valid.numel(),
            "distillation_actor_skipped": float(not bool(supported.any())),
            "distillation_actor_updates": float(self.actor_updates),
            "distillation_scale": self.scale.detach().clone(),
            "ent_coef": alpha,
        }
        if not bool(supported.any()):
            metrics.update(actor_loss=zero, actor_grad_norm=zero, ent_coef_loss=zero)
            metrics.update(self._update_q_scale(zs[:-1], None, alpha=alpha, scale_snapshot=scale_snapshot))
            return metrics

        kwargs = {"include_scaled_entropy": True} if agent._actor_entropy_mode == "tdmpc2_scaled" else {}
        action, info = agent.model.pi(zs[:-1], **kwargs)
        # Slice before arithmetic: invalid labels may be missing or non-finite,
        # and multiplying their losses by zero would not remove NaN gradients.
        kl = td_math.diagonal_gaussian_reverse_kl(
            info["pre_tanh_mean"][valid], info["log_std"][valid],
            mean[valid], log_std[valid],
        ).squeeze(-1) / float(agent.cfg.action_dim)
        if not bool(torch.isfinite(kl).all()):
            raise ValueError("Distillation KL must be finite.")
        depth_zero = torch.zeros_like(valid)
        depth_zero[0] = valid[0]
        depth_zero_values = kl[depth_zero[valid]]
        if depth_zero_values.numel():
            with torch.no_grad():
                value = percentile_range(depth_zero_values[:, None].detach(), self.percentiles, minimum=1.0)
                self.scale.lerp_(value, 0.01)
        selected_entropy = policy_entropy(info, agent._actor_entropy_mode).squeeze(-1)[valid]
        valid_weights = weights[valid]
        normalized_weights = valid_weights / valid_weights.sum()
        # temporal_loss_weights already divides by H. Its sum is C_H, not
        # its mean; a second division would weaken learning as H grows.
        horizon_factor = agent._transition_temporal_weights.sum()
        loss = horizon_factor * ((kl / self.scale.detach() - alpha * selected_entropy) * normalized_weights).sum()
        temperature_loss = zero
        if agent.ent_coef_optim is not None:
            temperature_loss = (
                agent.log_ent_coef * ((selected_entropy.detach() - agent.target_entropy) * normalized_weights).sum()
            ).mean()

        q_scale_info = self._update_q_scale(zs[:-1], action, alpha=alpha, scale_snapshot=scale_snapshot)
        agent.pi_optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = agent._clip_actor_grad_norm_()
        agent.pi_optim.step()
        if agent.ent_coef_optim is not None:
            agent.ent_coef_optim.zero_grad(set_to_none=True)
            temperature_loss.backward()
            agent.ent_coef_optim.step()
            with torch.no_grad():
                agent.log_ent_coef.clamp_(min=math.log(1e-8))
        self.actor_updates += 1
        age = (agent.outer_version - version[valid]).to(kl)
        metrics.update({
            "actor_loss": loss.detach(),
            "actor_grad_norm": torch.as_tensor(grad_norm).detach(),
            "actor_entropy": (info["entropy"].squeeze(-1)[valid].detach() * normalized_weights).sum(),
            "ent_coef_loss": temperature_loss.detach(),
            "distillation_kl": (kl.detach() * normalized_weights).sum(),
            "distillation_normalized_kl": (kl.detach() / self.scale * normalized_weights).sum(),
            "distillation_scale": self.scale.detach().clone(),
            "distillation_teacher_age_mean": (age * normalized_weights).sum(),
            "distillation_teacher_age_max": age.max(),
            "distillation_actor_updates": float(self.actor_updates),
            **policy_sample_diagnostics(action, info),
            **q_scale_info,
        })
        if agent._actor_entropy_mode == "tdmpc2_scaled":
            metrics["actor_scaled_entropy"] = (selected_entropy.detach() * normalized_weights).sum()
            metrics["actor_entropy_bonus"] = alpha * metrics["actor_scaled_entropy"]
        return metrics
