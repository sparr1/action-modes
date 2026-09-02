"""AMBI-XQC: recurrent TOLD learning with XQC control priors."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .common import math as td_math
from .common.checkpoint import save_checkpoint
from .common.device import resolve_device
from .common.training_state import (
    load_optimizer_state_preserving_hyperparameters,
    preflight_module_state,
    preflight_optimizer_state,
    require_exact_keys,
    require_tensor,
)
from .common.xqc_world_model import XQCTOLDWorldModel
from .inner_xqc import InnerXQCEngine
from .xqc_controller import (
    LatentXQCBatch,
    LatentXQCConfig,
    LatentXQCController,
)
from RL.xqc_core import (
    BN_EPSILON,
    BN_MOMENTUM,
    LOG_STD_MAX,
    LOG_STD_MIN,
    DiscountedReturnNormalizer,
)


class AMBIXQCAgent(nn.Module):
    """Persistent TOLD model and XQC priors with fresh inner XQC per action."""

    _CHECKPOINT_VERSION = 1

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = resolve_device(getattr(cfg, "device", None))
        self.cfg.device = str(self.device)
        if self.device.type not in {"cpu", "cuda"}:
            raise NotImplementedError("AMBI-XQC supports CPU and CUDA only.")
        if str(cfg.obs) != "state":
            raise NotImplementedError("AMBI-XQC v1 supports state observations only.")

        self.model = XQCTOLDWorldModel(cfg).to(self.device)
        controller_cfg = LatentXQCConfig(
            actor_net_arch=tuple(cfg.xqc_actor_net_arch),
            critic_net_arch=tuple(cfg.xqc_critic_net_arch),
            num_atoms=int(cfg.xqc_num_atoms),
            vmin=float(cfg.xqc_vmin),
            vmax=float(cfg.xqc_vmax),
            tau=float(cfg.xqc_tau),
            target_update_interval=int(cfg.xqc_target_update_interval),
            policy_delay=int(cfg.xqc_policy_delay),
            init_temperature=float(cfg.xqc_init_temperature),
            target_entropy=float(cfg.xqc_resolved_target_entropy),
            adam_eps=float(cfg.xqc_adam_eps),
            optimizer_backend=str(cfg.xqc_optimizer_backend),
        )
        self.xqc_controller = LatentXQCController(
            int(cfg.latent_dim), int(cfg.action_dim), controller_cfg
        ).to(self.device)
        self.xqc_controller.configure_compile(
            enabled=bool(getattr(cfg, "compile", False)),
            strict=bool(getattr(cfg, "compile_strict", False)),
        )
        self.xqc_workspace = self.xqc_controller.make_workspace(
            actor_lr=float(cfg.xqc_actor_lr),
            critic_lr=float(cfg.xqc_critic_lr),
            actor_lr_end=float(cfg.xqc_lr_end),
            critic_lr_end=float(cfg.xqc_lr_end),
            transition_steps=int(cfg.xqc_lr_transition_steps),
            optimizer_backend=str(cfg.xqc_optimizer_backend),
        )

        self._world_params = (
            list(self.model._encoder.parameters())
            + list(self.model._dynamics.parameters())
            + list(self.model._reward.parameters())
            + (
                list(self.model._termination.parameters())
                if bool(cfg.episodic)
                else []
            )
        )
        optimizer_groups = [
            {
                "params": self.model._encoder.parameters(),
                "lr": float(cfg.lr) * float(cfg.enc_lr_scale),
            },
            {"params": self.model._dynamics.parameters()},
            {"params": self.model._reward.parameters()},
        ]
        if bool(cfg.episodic):
            optimizer_groups.append({"params": self.model._termination.parameters()})
        self.world_optimizer = torch.optim.Adam(
            optimizer_groups,
            lr=float(cfg.lr),
            eps=float(getattr(cfg, "adam_eps", 1e-8)),
            capturable=self.device.type == "cuda",
            foreach=self.device.type == "cuda",
        )

        self.discount = self._get_discount(cfg.episode_length)
        self.reward_normalizer = DiscountedReturnNormalizer(self.discount)
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
        generator_device = self.device if self.device.type == "cuda" else "cpu"
        self._outer_generator = torch.Generator(device=generator_device)
        self._outer_generator.manual_seed(int(cfg.seed) + 9_973_199)

        self.num_updates = 0
        self.outer_version = 0
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        self._resume_boundary_prepared = False
        self.inner_engine = InnerXQCEngine(self)
        self.model.eval()

        print("Episode length:", cfg.episode_length)
        print("Discount factor:", self.discount)
        print("Control prior: XQC", self.critic_signature)
        print(
            "Inner XQC schedule:",
            f"J={cfg.inner_rounds}, N={cfg.inner_rollouts_per_round}, "
            f"H={cfg.inner_rollout_horizon}, G={cfg.inner_updates_per_round}",
        )

    @property
    def critic_signature(self):
        return self.xqc_controller.critic_signature

    @property
    def alpha(self):
        return self.xqc_controller.temperature

    @property
    def target_entropy(self):
        return self.xqc_controller.target_entropy

    def _get_discount(self, episode_length):
        fraction = float(episode_length) / float(self.cfg.discount_denom)
        return min(
            max((fraction - 1.0) / fraction, self.cfg.discount_min),
            self.cfg.discount_max,
        )

    def observe_reward(self, reward, terminated, truncated):
        """Update one chronological real-return stream; replay stays raw."""

        self.reward_normalizer.update(
            float(reward), bool(terminated) or bool(truncated)
        )

    def reset(self):
        if self._resume_boundary_prepared:
            self._resume_boundary_prepared = False
            return
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        self.inner_engine.reset_episode()

    def prepare_training_resume_boundary(self):
        if not self._resume_boundary_prepared:
            self.last_inner_metrics = {}
            self.last_inner_rollout_lengths = []
            self.inner_engine.prepare_training_resume_boundary()
            self._resume_boundary_prepared = True
        return self

    @staticmethod
    def _materialize_action_metrics(action, metrics, rollout_lengths):
        tensor_items = [
            (key, value) for key, value in metrics.items() if torch.is_tensor(value)
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
            pieces.append(tensor_lengths.to(device=action.device, dtype=action.dtype))
        packed = torch.cat(pieces).detach().cpu()
        action_size = int(action.numel())
        cpu_action = packed[:action_size].reshape(action.shape)
        materialized = dict(metrics)
        for offset, (key, _) in enumerate(tensor_items, start=action_size):
            materialized[key] = float(packed[offset])
        if tensor_lengths is not None:
            start = action_size + len(tensor_items)
            rollout_lengths = [int(value) for value in packed[start:].tolist()]
        return cpu_action, materialized, rollout_lengths

    def act(
        self,
        obs,
        t0=False,
        eval_mode=False,
        task=None,
        *,
        collect_diagnostics=True,
    ):
        if task is not None:
            raise ValueError("AMBI-XQC currently supports single-task training only.")
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                root_z = self.model.encode(obs).detach()
            with torch.enable_grad():
                action, metrics, lengths = self.inner_engine.act(
                    root_z,
                    t0=t0,
                    eval_mode=eval_mode,
                    collect_diagnostics=collect_diagnostics,
                )
        finally:
            self.model.train(was_training)
        action, metrics, lengths = self._materialize_action_metrics(
            action, metrics, lengths
        )
        metrics = self.inner_engine.finalize_timing_metrics(metrics)
        self.last_inner_metrics = metrics
        self.last_inner_rollout_lengths = lengths
        return action

    def _noise(self, leading_shape, dtype):
        return torch.randn(
            tuple(leading_shape) + (int(self.cfg.action_dim),),
            dtype=dtype,
            device=self.device,
            generator=self._outer_generator,
        )

    def _recurrent_world_and_value_losses(
        self, obs, action, reward, terminated, next_z_targets
    ):
        """Build the full unbroken TOLD recurrent graph and XQC value loss."""

        z = self.model.encode(obs[0])
        latent_states = [z]
        consistency_errors = []
        for recorded_action, next_z_target in zip(
            action.unbind(0), next_z_targets.unbind(0)
        ):
            z = self.model.next(z, recorded_action)
            consistency_errors.append(F.mse_loss(z, next_z_target))
            latent_states.append(z)
        latent_states = torch.stack(latent_states, dim=0)
        consistency_per_time = torch.stack(consistency_errors)
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
        reward_per_sample = td_math.soft_ce(reward_predictions, reward, self.cfg)
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

        xqc_batch = LatentXQCBatch(
            latents=rollout_states,
            actions=action,
            rewards=reward,
            next_latents=next_z_targets,
            bootstrap_mask=1.0 - terminated,
            discount=self.discount,
        )
        critic_objective = self.xqc_controller.critic_objective(
            xqc_batch,
            next_noise=self._noise(action.shape[:-1], action.dtype),
            reward_scale=self.reward_normalizer.scale,
        )
        critic_per_time = critic_objective.per_sample_loss.mean(dim=1)
        critic_loss = td_math.reduce_temporal_loss(
            critic_per_time,
            self.cfg.rho,
            normalization=self.cfg.temporal_loss_normalization,
            reference_horizon=self.cfg.temporal_loss_reference_horizon,
            legacy_order="vector_sum_divide",
            weights=self._transition_temporal_weights,
        )

        termination_prediction = (
            self.model.termination(latent_states[1:], unnormalized=True)
            if bool(self.cfg.episodic)
            else None
        )
        if bool(self.cfg.episodic):
            termination_per_sample = F.binary_cross_entropy_with_logits(
                termination_prediction,
                terminated,
                reduction="none",
            )
            termination_per_time = termination_per_sample.mean(
                dim=tuple(range(1, termination_per_sample.ndim))
            )
            termination_loss = td_math.reduce_temporal_loss(
                termination_per_time,
                self.cfg.rho,
                normalization=self.cfg.temporal_loss_normalization,
                reference_horizon=self.cfg.temporal_loss_reference_horizon,
                legacy_order="vector_sum_divide",
                weights=self._transition_temporal_weights,
            )
        else:
            termination_loss = torch.zeros((), device=self.device)
        total_loss = (
            float(self.cfg.consistency_coef) * consistency_loss
            + float(self.cfg.reward_coef) * reward_loss
            + float(self.cfg.termination_coef) * termination_loss
            + float(self.cfg.value_coef) * critic_loss
        )
        return {
            "latent_states": latent_states,
            "consistency_per_time": consistency_per_time,
            "consistency_loss": consistency_loss,
            "reward_predictions": reward_predictions,
            "reward_loss": reward_loss,
            "critic": critic_objective,
            "critic_loss": critic_loss,
            "termination_prediction": termination_prediction,
            "termination_loss": termination_loss,
            "total_loss": total_loss,
        }

    def _update_actor_and_temperature(self, latent_states):
        detached = latent_states.detach()
        objective = self.xqc_controller.actor_objective(
            detached,
            actor_noise=self._noise(detached.shape[:-1], detached.dtype),
        )
        actor_per_time = objective.per_sample_loss.mean(dim=1)
        actor_loss = td_math.reduce_temporal_loss(
            actor_per_time,
            self.cfg.rho,
            normalization=self.cfg.temporal_loss_normalization,
            reference_horizon=self.cfg.temporal_loss_reference_horizon,
            include_terminal=True,
            legacy_order="vector_mean",
            weights=self._actor_temporal_weights,
        )
        step_info = self.xqc_workspace.step_actor_and_temperature(
            actor_loss, objective.entropy.mean()
        )
        q_values = objective.q_values.detach()
        return {
            "actor_loss": actor_loss.detach(),
            "actor_entropy": objective.entropy.detach().mean(),
            "actor_log_prob": objective.log_prob.detach().mean(),
            "actor_q_mean": objective.minimum_q.detach().mean(),
            "actor_q_mean_all": q_values.mean(dim=0).mean(),
            "actor_q_min_all": q_values.min(dim=0).values.mean(),
            "actor_q_mean_all_minus_min_all": (
                q_values.mean(dim=0) - q_values.min(dim=0).values
            ).mean(),
            "ent_coef": step_info["temperature"],
            "ent_coef_loss": step_info["temperature_loss"],
            "actor_grad_norm": step_info["actor_grad_norm"],
            "temperature_grad_norm": step_info["temperature_grad_norm"],
            "actor_update_accepted": step_info["actor_update_accepted"],
            "actor_learning_rate": step_info["actor_learning_rate"],
            "temperature_learning_rate": step_info[
                "temperature_learning_rate"
            ],
        }

    def _update(self, obs, action, reward, terminated):
        with torch.no_grad():
            next_z_targets = self.model.encode(obs[1:])

        self.model.train()
        losses = self._recurrent_world_and_value_losses(
            obs, action, reward, terminated, next_z_targets
        )
        self.world_optimizer.zero_grad(set_to_none=True)
        self.xqc_workspace.zero_critic_grad()
        losses["total_loss"].backward()

        # TOLD's value coefficient controls how strongly value learning shapes
        # the representation.  XQC's own critic Adam still receives the
        # released, unscaled summed-head CE gradient.
        value_coef = float(self.cfg.value_coef)
        if not math.isfinite(value_coef) or value_coef <= 0.0:
            raise ValueError("AMBI-XQC requires a positive finite value_coef.")
        if value_coef != 1.0:
            with torch.no_grad():
                for parameter in self.xqc_controller.critic.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(value_coef)

        world_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._world_params, float(self.cfg.grad_clip_norm)
        )
        critic_grad_norm = torch.stack(
            [
                parameter.grad.detach().norm(2)
                for parameter in self.xqc_controller.critic.parameters()
                if parameter.grad is not None
            ]
        ).norm(2)
        self.world_optimizer.step()
        critic_lr, target_updated = self.xqc_workspace.step_critic()
        actor_info = self._update_actor_and_temperature(losses["latent_states"])

        self.num_updates += 1
        self.outer_version += 1
        self.inner_engine.mark_outer_update(self.outer_version)
        if self.xqc_workspace.update_step != self.num_updates:
            raise RuntimeError("Outer XQC and TOLD update counters diverged.")
        self.model.eval()

        reward_values = td_math.two_hot_inv(
            losses["reward_predictions"].detach(), self.cfg
        )
        critic = losses["critic"]
        q_values = critic.current_values.detach()
        info = {
            "consistency_loss": losses["consistency_loss"].detach(),
            "reward_loss": losses["reward_loss"].detach(),
            "critic_loss": losses["critic_loss"].detach(),
            "termination_loss": losses["termination_loss"].detach(),
            "total_loss": losses["total_loss"].detach(),
            "grad_norm": torch.as_tensor(world_grad_norm).detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "q_target_mean": critic.target_values.detach().mean(),
            "q_mean": q_values.mean(),
            "q_abs_mean": q_values.abs().mean(),
            "q_head_disagreement": (q_values[0] - q_values[1]).abs().mean(),
            "q_target_clip_fraction": critic.clip_fraction.detach().clone(),
            "reward_pred_mean": reward_values.mean(),
            "reward_target_mean": reward.detach().mean(),
            "reward_abs_mean": reward.detach().abs().mean(),
            "reward_scale": torch.tensor(
                self.reward_normalizer.scale, device=self.device
            ),
            "critic_learning_rate": float(critic_lr),
            "target_updated": float(target_updated),
            "num_updates": float(self.num_updates),
            "compile_fallback": float(
                self.xqc_controller.compile_status["fallback"]
            ),
        }
        for depth in range(int(self.cfg.train_unroll_horizon)):
            info[f"consistency_error_depth_{depth + 1}"] = losses[
                "consistency_per_time"
            ][depth].detach()
            info[f"reward_error_depth_{depth + 1}"] = (
                reward_values[depth] - reward[depth]
            ).abs().mean()
            values_at_depth = q_values[:, depth]
            info[f"q_error_depth_{depth + 1}"] = (
                values_at_depth - critic.target_values[depth].detach()
            ).abs().mean()
            info[f"q_head_disagreement_depth_{depth + 1}"] = (
                values_at_depth[0] - values_at_depth[1]
            ).abs().mean()
        if bool(self.cfg.episodic):
            info.update(
                td_math.termination_statistics(
                    torch.sigmoid(losses["termination_prediction"][-1]).detach(),
                    terminated[-1].detach(),
                )
            )
        info.update(actor_info)
        return info

    def update(self, buffer):
        obs, action, reward, terminated, task = buffer.sample()
        if task is not None:
            raise NotImplementedError("AMBI-XQC supports single-task training only.")
        return self._update(obs, action, reward, terminated)

    def observation_signature(self):
        mode = str(self.cfg.obs)
        return {
            "mode": mode,
            "shape": [int(value) for value in self.cfg.obs_shape[mode]],
            "dtype": str(getattr(self.cfg, "obs_dtype", "float32")),
        }

    def semantic_signature(self):
        return {
            "algorithm": "AMBIXQC",
            "official_xqc_commit": str(self.cfg.xqc_official_commit),
            "observation": self.observation_signature(),
            "action_dim": int(self.cfg.action_dim),
            "told": {
                "enc_dim": int(self.cfg.enc_dim),
                "mlp_dim": int(self.cfg.mlp_dim),
                "latent_dim": int(self.cfg.latent_dim),
                "num_enc_layers": int(self.cfg.num_enc_layers),
                "simnorm_dim": int(self.cfg.simnorm_dim),
                "reward_num_bins": int(self.cfg.num_bins),
                "reward_vmin": float(self.cfg.vmin),
                "reward_vmax": float(self.cfg.vmax),
                "episodic": bool(self.cfg.episodic),
                "train_unroll_horizon": int(self.cfg.train_unroll_horizon),
                "rho": float(self.cfg.rho),
                "temporal_loss_normalization": str(
                    self.cfg.temporal_loss_normalization
                ),
                "temporal_loss_reference_horizon": int(
                    self.cfg.temporal_loss_reference_horizon
                ),
                "consistency_coef": float(self.cfg.consistency_coef),
                "reward_coef": float(self.cfg.reward_coef),
                "termination_coef": float(self.cfg.termination_coef),
                "value_coef": float(self.cfg.value_coef),
                "learning_rate": float(self.cfg.lr),
                "encoder_learning_rate_scale": float(self.cfg.enc_lr_scale),
                "adam_eps": float(getattr(self.cfg, "adam_eps", 1e-8)),
                "gradient_clip_norm": float(self.cfg.grad_clip_norm),
            },
            "critic": self.critic_signature,
            "actor_arch": tuple(self.cfg.xqc_actor_net_arch),
            "critic_arch": tuple(self.cfg.xqc_critic_net_arch),
            "log_std_min": LOG_STD_MIN,
            "log_std_max": LOG_STD_MAX,
            "bn_momentum": BN_MOMENTUM,
            "bn_epsilon": BN_EPSILON,
            "discount": float(self.discount),
            "tau": float(self.cfg.xqc_tau),
            "policy_delay": int(self.cfg.xqc_policy_delay),
            "target_update_interval": int(self.cfg.xqc_target_update_interval),
            "target_entropy": float(self.target_entropy),
            "init_temperature": float(self.cfg.xqc_init_temperature),
            "actor_lr": float(self.cfg.xqc_actor_lr),
            "critic_lr": float(self.cfg.xqc_critic_lr),
            "lr_end": float(self.cfg.xqc_lr_end),
            "lr_transition_steps": int(self.cfg.xqc_lr_transition_steps),
            "adam_eps": float(self.cfg.xqc_adam_eps),
            "optimizer_backend": str(self.cfg.xqc_optimizer_backend),
            "reward_normalization": (
                "real_discounted_return_plus_fresh_action_local_imagined_returns"
                if self.cfg.inner_reward_normalization
                == "action_local_imagined"
                else "real_discounted_return_only"
            ),
            "weight_projection": "all_linear_weight_rows",
            "temperature_lr_source": "actor_lr",
            "training_noise_rng": "outer_device_local_plus_named_inner_streams_v1",
            "inner_lifecycle": "fresh_per_action",
            "inner_schedule": {
                "rounds": int(self.cfg.inner_rounds),
                "rollouts": int(self.cfg.inner_rollouts_per_round),
                "horizon": int(self.cfg.inner_rollout_horizon),
                "updates": int(self.cfg.inner_updates_per_round),
                "batch_size": int(self.cfg.inner_batch_size),
                "replay_capacity": int(self.cfg.inner_replay_capacity),
                "replay_sampling": str(self.cfg.inner_replay_sampling),
                "actor_lr": float(self.cfg.inner_actor_lr),
                "critic_lr": float(self.cfg.inner_critic_lr),
            },
        }

    def checkpoint_state(self):
        return {
            "checkpoint_version": self._CHECKPOINT_VERSION,
            "semantic_signature": self.semantic_signature(),
            "module": self.state_dict(),
            "world_optimizer": self.world_optimizer.state_dict(),
            "xqc_workspace": self.xqc_workspace.state_dict(),
            "reward_normalizer": self.reward_normalizer.state_dict(),
            "outer_generator": self._outer_generator.get_state(),
            "num_updates": int(self.num_updates),
            "outer_version": int(self.outer_version),
            "inner": self.inner_engine.training_state_dict(),
        }

    def save(self, fp):
        return save_checkpoint(self.checkpoint_state(), fp)

    @staticmethod
    def _validate_finite_state_tensors(state, name):
        for key, value in state.items():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} tensor {key!r} must be finite.")

    @staticmethod
    def _validate_optimizer_steps(optimizer, incoming, name, expected_steps):
        optimizer_state = incoming["state"]
        parameter_count = sum(
            len(group["params"]) for group in optimizer.param_groups
        )
        if expected_steps == 0:
            if optimizer_state:
                raise ValueError(f"{name} must be uninitialized at step zero.")
            return
        if len(optimizer_state) != parameter_count:
            raise ValueError(f"{name} optimizer state inventory is incomplete.")
        for parameter_state in optimizer_state.values():
            step = parameter_state["step"]
            numeric_step = float(step.item() if torch.is_tensor(step) else step)
            if numeric_step != expected_steps:
                raise ValueError(f"{name} optimizer step does not match its counter.")
            if any(
                torch.is_tensor(value)
                and not bool(torch.isfinite(value).all().item())
                for value in parameter_state.values()
            ):
                raise ValueError(f"{name} optimizer state must be finite.")

    def _preflight_checkpoint(self, state):
        expected = {
            "checkpoint_version",
            "semantic_signature",
            "module",
            "world_optimizer",
            "xqc_workspace",
            "reward_normalizer",
            "outer_generator",
            "num_updates",
            "outer_version",
            "inner",
        }
        state = require_exact_keys(state, expected, "AMBI-XQC checkpoint")
        if (
            isinstance(state["checkpoint_version"], bool)
            or not isinstance(state["checkpoint_version"], int)
            or state["checkpoint_version"] != self._CHECKPOINT_VERSION
        ):
            raise ValueError("Unsupported AMBI-XQC checkpoint version.")
        if state["semantic_signature"] != self.semantic_signature():
            raise ValueError("AMBI-XQC checkpoint semantics do not match this agent.")
        for key in ("num_updates", "outer_version"):
            value = state[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"AMBI-XQC checkpoint {key} is invalid.")
        if state["num_updates"] != state["outer_version"]:
            raise ValueError("AMBI-XQC outer counters must agree.")
        workspace = state["xqc_workspace"]
        workspace = require_exact_keys(
            workspace,
            {
                "actor_optimizer",
                "critic_optimizer",
                "temperature_optimizer",
                "update_step",
                "actor_optimizer_steps",
                "temperature_optimizer_steps",
            },
            "AMBI-XQC workspace checkpoint",
        )
        for key in (
            "update_step",
            "actor_optimizer_steps",
            "temperature_optimizer_steps",
        ):
            value = workspace[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"AMBI-XQC workspace {key} is invalid.")
        if workspace["update_step"] != state["num_updates"]:
            raise ValueError("AMBI-XQC workspace and outer counters differ.")
        expected_delayed_steps = (
            0
            if workspace["update_step"] == 0
            else (workspace["update_step"] - 1) // int(self.cfg.xqc_policy_delay) + 1
        )
        if (
            workspace["actor_optimizer_steps"] != expected_delayed_steps
            or workspace["temperature_optimizer_steps"] != expected_delayed_steps
        ):
            raise ValueError(
                "AMBI-XQC delayed actor/temperature counters are inconsistent."
            )
        preflight_module_state(self, state["module"], "AMBI-XQC module")
        preflight_optimizer_state(
            self.world_optimizer,
            state["world_optimizer"],
            "AMBI-XQC world optimizer",
        )
        preflight_optimizer_state(
            self.xqc_workspace.actor_optimizer,
            workspace["actor_optimizer"],
            "AMBI-XQC actor optimizer",
        )
        preflight_optimizer_state(
            self.xqc_workspace.critic_optimizer,
            workspace["critic_optimizer"],
            "AMBI-XQC critic optimizer",
        )
        preflight_optimizer_state(
            self.xqc_workspace.temperature_optimizer,
            workspace["temperature_optimizer"],
            "AMBI-XQC temperature optimizer",
        )
        self._validate_optimizer_steps(
            self.world_optimizer,
            state["world_optimizer"],
            "AMBI-XQC world",
            state["num_updates"],
        )
        self._validate_optimizer_steps(
            self.xqc_workspace.actor_optimizer,
            workspace["actor_optimizer"],
            "AMBI-XQC actor",
            workspace["actor_optimizer_steps"],
        )
        self._validate_optimizer_steps(
            self.xqc_workspace.critic_optimizer,
            workspace["critic_optimizer"],
            "AMBI-XQC critic",
            workspace["update_step"],
        )
        self._validate_optimizer_steps(
            self.xqc_workspace.temperature_optimizer,
            workspace["temperature_optimizer"],
            "AMBI-XQC temperature",
            workspace["temperature_optimizer_steps"],
        )
        self._validate_finite_state_tensors(state["module"], "AMBI-XQC module")
        self.reward_normalizer._validated_state(state["reward_normalizer"])
        generator_state = require_tensor(
            state["outer_generator"],
            "AMBI-XQC outer generator",
            dtype=torch.uint8,
        )
        # Validate opaque generator bytes before mutating any live module.
        generator_probe = torch.Generator(
            device=self.device if self.device.type == "cuda" else "cpu"
        )
        try:
            generator_probe.set_state(generator_state.detach().cpu())
        except RuntimeError as exc:
            raise ValueError("AMBI-XQC outer generator state is invalid.") from exc
        inner = self.inner_engine._preflight_training_state_dict(state["inner"])
        return state, workspace, generator_state, inner

    def load(self, fp):
        state = (
            fp
            if isinstance(fp, Mapping)
            else torch.load(fp, map_location=self.device, weights_only=False)
        )
        state, workspace, generator_state, inner = self._preflight_checkpoint(state)
        self.load_state_dict(state["module"])
        load_optimizer_state_preserving_hyperparameters(
            self.world_optimizer, state["world_optimizer"]
        )
        load_optimizer_state_preserving_hyperparameters(
            self.xqc_workspace.actor_optimizer, workspace["actor_optimizer"]
        )
        load_optimizer_state_preserving_hyperparameters(
            self.xqc_workspace.critic_optimizer, workspace["critic_optimizer"]
        )
        load_optimizer_state_preserving_hyperparameters(
            self.xqc_workspace.temperature_optimizer,
            workspace["temperature_optimizer"],
        )
        self.xqc_workspace.update_step = int(workspace["update_step"])
        self.xqc_workspace.actor_optimizer_steps = int(
            workspace["actor_optimizer_steps"]
        )
        self.xqc_workspace.temperature_optimizer_steps = int(
            workspace["temperature_optimizer_steps"]
        )
        self.xqc_workspace.restore_learning_rate_phase_()
        self.reward_normalizer.load_state_dict(state["reward_normalizer"])
        self._outer_generator.set_state(generator_state.to("cpu"))
        self.num_updates = int(state["num_updates"])
        self.outer_version = int(state["outer_version"])
        self.inner_engine._commit_training_state_candidate(inner)
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        self._resume_boundary_prepared = False
        self.model.eval()
        return self

    def training_state_dict(self):
        raise NotImplementedError(
            "Exact AMBI-XQC trainer resume is not supported in v1; use portable "
            "controller/world checkpoints instead."
        )
