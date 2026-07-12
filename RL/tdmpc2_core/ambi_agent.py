"""AMBI agent built from TD-MPC2 world-model updates and latent SAC control."""

from copy import deepcopy

import torch
import torch.nn.functional as F

from .common import math as td_math
from .common.device import resolve_device
from .common.latent_buffer import LatentReplayBuffer
from .common.lora import lorafy_copy, trainable_parameters
from .common.soft_world_model import SoftWorldModel


@torch.no_grad()
def _polyak_update(source, target, tau):
    tau = float(tau)
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.lerp_(source_param.data, tau)
    for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
        target_buffer.data.copy_(source_buffer.data)


class AMBITDMPC2Agent(torch.nn.Module):
    """TD-MPC2-style online learner whose MPPI call is replaced by inner SAC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = resolve_device(getattr(cfg, "device", None))
        self.cfg.device = str(self.device)
        if getattr(cfg, "compile", False):
            raise ValueError("AMBI-TD-MPC2 does not support compile=True because it creates an inner learner per action.")

        self.model = SoftWorldModel(cfg).to(self.device)
        self._world_critic_params = (
            list(self.model._encoder.parameters())
            + list(self.model._dynamics.parameters())
            + list(self.model._reward.parameters())
            + (list(self.model._termination.parameters()) if cfg.episodic else [])
            + list(self.model._Qs.parameters())
        )
        optim_groups = [
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
            optim_groups.append({"params": self.model._termination.parameters()})
        self.optim = torch.optim.Adam(
            optim_groups,
            lr=float(cfg.lr),
            eps=float(getattr(cfg, "adam_eps", 1e-8)),
            capturable=self.device.type == "cuda",
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(),
            lr=float(getattr(cfg, "actor_lr", cfg.lr)),
            eps=float(getattr(cfg, "adam_eps", 1e-8)),
            capturable=self.device.type == "cuda",
        )

        self.target_entropy = self._target_entropy(getattr(cfg, "target_entropy", "auto"))
        self.log_ent_coef = None
        self.ent_coef_optim = None
        ent_coef = getattr(cfg, "ent_coef", "auto")
        if isinstance(ent_coef, str) and ent_coef.startswith("auto"):
            initial = 1.0
            if "_" in ent_coef:
                initial = float(ent_coef.split("_", 1)[1])
            if initial <= 0:
                raise ValueError("Initial automatic entropy coefficient must be positive.")
            self.log_ent_coef = torch.nn.Parameter(
                torch.log(torch.tensor([initial], dtype=torch.float32, device=self.device))
            )
            self.ent_coef_optim = torch.optim.Adam(
                [self.log_ent_coef],
                lr=float(getattr(cfg, "ent_coef_lr", getattr(cfg, "actor_lr", cfg.lr))),
                eps=float(getattr(cfg, "adam_eps", 1e-8)),
                capturable=self.device.type == "cuda",
            )
            self.register_buffer("fixed_ent_coef", torch.tensor(float("nan"), device=self.device))
        else:
            fixed = float(ent_coef)
            if fixed <= 0:
                raise ValueError("Entropy coefficient must be positive.")
            self.register_buffer("fixed_ent_coef", torch.tensor(fixed, device=self.device))

        self.discount = self._get_discount(cfg.episode_length)
        self.num_updates = 0
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        self.model.eval()

        print("Episode length:", cfg.episode_length)
        print("Discount factor:", self.discount)
        print("Inner adaptation:", cfg.inner_adaptation)

    def _get_discount(self, episode_length):
        frac = float(episode_length) / float(self.cfg.discount_denom)
        return min(max((frac - 1.0) / frac, self.cfg.discount_min), self.cfg.discount_max)

    def _target_entropy(self, target_entropy):
        if target_entropy == "auto":
            return float(-self.cfg.action_dim)
        return float(target_entropy)

    @property
    def alpha(self):
        if self.log_ent_coef is not None:
            return self.log_ent_coef.exp()
        return self.fixed_ent_coef

    def reset(self):
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []

    def save(self, fp):
        state = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "pi_optim": self.pi_optim.state_dict(),
            "num_updates": self.num_updates,
        }
        if self.log_ent_coef is not None:
            state["log_ent_coef"] = self.log_ent_coef.detach().cpu()
            state["ent_coef_optim"] = self.ent_coef_optim.state_dict()
        else:
            state["fixed_ent_coef"] = self.fixed_ent_coef.detach().cpu()
        torch.save(state, fp)

    def load(self, fp):
        state = fp if isinstance(fp, dict) else torch.load(fp, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"] if "model" in state else state)
        if "optim" in state:
            self.optim.load_state_dict(state["optim"])
        if "pi_optim" in state:
            self.pi_optim.load_state_dict(state["pi_optim"])
        self.num_updates = int(state.get("num_updates", 0))
        if self.log_ent_coef is not None and "log_ent_coef" in state:
            self.log_ent_coef.data.copy_(state["log_ent_coef"].to(self.device))
            if "ent_coef_optim" in state:
                self.ent_coef_optim.load_state_dict(state["ent_coef_optim"])
        elif "fixed_ent_coef" in state:
            self.fixed_ent_coef.copy_(state["fixed_ent_coef"].to(self.device))
        self.model.eval()
        return self

    def act(self, obs, t0=False, eval_mode=False, task=None):
        del t0
        if task is not None:
            raise ValueError("AMBI-TD-MPC2 currently supports single-task training only.")
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            root_z = self.model.encode(obs).detach()

        if int(self.cfg.inner_iterations) <= 0:
            self.last_inner_metrics = {"inner_steps": 0.0, "inner_updates": 0.0}
            self.last_inner_rollout_lengths = []
            with torch.no_grad():
                action, _ = self.model.pi(root_z, deterministic=eval_mode)
            return action[0].cpu()

        return self._inner_improve(root_z, eval_mode=eval_mode).cpu()

    def _make_inner_modules(self):
        mode = str(self.cfg.inner_adaptation).lower()
        if mode == "clone":
            actor = deepcopy(self.model._pi).to(self.device)
            critic = deepcopy(self.model._Qs).to(self.device)
            actor.requires_grad_(True)
            critic.requires_grad_(True)
        elif mode == "lora":
            actor = lorafy_copy(
                self.model._pi,
                rank=self.cfg.lora_rank,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
            ).to(self.device)
            critic = lorafy_copy(
                self.model._Qs,
                rank=self.cfg.lora_rank,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
            ).to(self.device)
        else:
            raise ValueError("inner_adaptation must be either 'clone' or 'lora'.")

        # A standard SAC target critic starts as an exact copy of the current
        # inner critic. The outer target critic belongs to the outer learner and
        # must not seed a different local Bellman operator.
        critic_target = deepcopy(critic).to(self.device)
        critic_target.requires_grad_(False)
        actor.train()
        critic.train()
        critic_target.eval()

        actor_params = trainable_parameters(actor)
        critic_params = trainable_parameters(critic)
        if not actor_params or not critic_params:
            raise RuntimeError("Inner adaptation produced no trainable actor or critic parameters.")
        actor_optim = torch.optim.Adam(
            actor_params,
            lr=float(self.cfg.inner_actor_lr),
            eps=float(getattr(self.cfg, "inner_adam_eps", 1e-8)),
            capturable=self.device.type == "cuda",
        )
        critic_optim = torch.optim.Adam(
            critic_params,
            lr=float(self.cfg.inner_critic_lr),
            eps=float(getattr(self.cfg, "inner_adam_eps", 1e-8)),
            capturable=self.device.type == "cuda",
        )
        return actor, critic, critic_target, actor_optim, critic_optim, actor_params, critic_params

    @torch.no_grad()
    def _collect_imagined_rollouts(self, root_z, actor, replay):
        num_rollouts = int(self.cfg.inner_rollouts)
        horizon = int(self.cfg.inner_horizon)
        z = root_z.expand(num_rollouts, -1).clone()
        alive = torch.ones(num_rollouts, dtype=torch.bool, device=self.device)
        lengths = torch.zeros(num_rollouts, dtype=torch.long, device=self.device)

        self.model.eval()
        actor.eval()
        for _ in range(horizon):
            active_idx = torch.nonzero(alive, as_tuple=False).squeeze(-1)
            if active_idx.numel() == 0:
                break
            active_z = z[active_idx]
            action, _ = self.model.pi(active_z, policy=actor)
            reward = td_math.two_hot_inv(self.model.reward(active_z, action), self.cfg)
            next_z = self.model.next(active_z, action)
            if self.cfg.episodic:
                terminated = (
                    self.model.termination(next_z) > float(self.cfg.inner_termination_threshold)
                ).float()
            else:
                terminated = torch.zeros(active_idx.numel(), 1, device=self.device)

            replay.add_batch(active_z, action, reward, next_z, terminated)
            lengths[active_idx] += 1
            z[active_idx] = next_z
            alive[active_idx] = terminated.squeeze(-1) < 0.5

        actor.train()
        return lengths.tolist()

    def _inner_sac_update(
        self,
        replay,
        actor,
        critic,
        critic_target,
        actor_optim,
        critic_optim,
        actor_params,
        critic_params,
        update_index,
    ):
        """Run one standard SAC update on detached imagined transitions."""
        batch = replay.sample(self.cfg.inner_batch_size)
        alpha = self.alpha.detach()

        with torch.no_grad():
            next_action, next_info = self.model.pi(batch["next_z"], policy=actor)
            next_q = self.model.Q(
                batch["next_z"],
                next_action,
                qs=critic_target,
                return_type="min",
            )
            target_q = batch["reward"] + self.discount * (1.0 - batch["terminated"]) * (
                next_q - alpha * next_info["log_prob"]
            )

        current_q = self.model.Q(
            batch["z"],
            batch["action"],
            qs=critic,
            return_type="all",
        )
        critic_loss = 0.5 * (
            F.mse_loss(current_q[0], target_q) + F.mse_loss(current_q[1], target_q)
        )
        critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_params, float(self.cfg.inner_grad_clip_norm)
        )
        critic_optim.step()
        critic_optim.zero_grad(set_to_none=True)

        action_pi, pi_info = self.model.pi(batch["z"], policy=actor)
        current_requires_grad = [p.requires_grad for p in critic.parameters()]
        for param in critic.parameters():
            param.requires_grad_(False)
        try:
            q_pi = self.model.Q(batch["z"], action_pi, qs=critic, return_type="min")
            actor_loss = (alpha * pi_info["log_prob"] - q_pi).mean()
        finally:
            for param, requires_grad in zip(critic.parameters(), current_requires_grad):
                param.requires_grad_(requires_grad)

        actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_params, float(self.cfg.inner_grad_clip_norm)
        )
        actor_optim.step()

        if update_index % int(self.cfg.inner_target_update_interval) == 0:
            _polyak_update(critic, critic_target, self.cfg.inner_tau)

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_grad_norm": float(torch.as_tensor(critic_grad_norm).detach().cpu()),
            "actor_grad_norm": float(torch.as_tensor(actor_grad_norm).detach().cpu()),
        }

    def _inner_improve(self, root_z, eval_mode=False):
        (
            actor,
            critic,
            critic_target,
            actor_optim,
            critic_optim,
            actor_params,
            critic_params,
        ) = self._make_inner_modules()

        default_capacity = (
            int(self.cfg.inner_iterations)
            * int(self.cfg.inner_rollouts)
            * int(self.cfg.inner_horizon)
        )
        capacity = int(self.cfg.inner_buffer_size or default_capacity)
        replay = LatentReplayBuffer(
            capacity=capacity,
            latent_dim=self.cfg.latent_dim,
            action_dim=self.cfg.action_dim,
            device=self.device,
        )

        rollout_lengths = []
        metric_history = []
        update_index = 0
        for _ in range(int(self.cfg.inner_iterations)):
            rollout_lengths.extend(self._collect_imagined_rollouts(root_z, actor, replay))
            for _ in range(int(self.cfg.inner_updates_per_iteration)):
                metric_history.append(
                    self._inner_sac_update(
                        replay,
                        actor,
                        critic,
                        critic_target,
                        actor_optim,
                        critic_optim,
                        actor_params,
                        critic_params,
                        update_index,
                    )
                )
                update_index += 1

        actor.eval()
        with torch.no_grad():
            action, _ = self.model.pi(root_z, policy=actor, deterministic=eval_mode)

        self.last_inner_rollout_lengths = rollout_lengths
        self.last_inner_metrics = {
            "inner_steps": float(sum(rollout_lengths)),
            "inner_updates": float(update_index),
            "inner_buffer_size": float(replay.size),
            "inner_alpha": float(self.alpha.detach().cpu()),
        }
        if metric_history:
            for key in metric_history[0]:
                self.last_inner_metrics[f"inner_{key}"] = sum(m[key] for m in metric_history) / len(metric_history)
        return action[0]

    @torch.no_grad()
    def _soft_td_target(self, next_z, reward, terminated):
        next_action, next_info = self.model.pi(next_z)
        next_q = self.model.Q(next_z, next_action, target=True, return_type="min")
        return reward + self.discount * (1.0 - terminated) * (
            next_q - self.alpha.detach() * next_info["log_prob"]
        )

    def _update_actor(self, zs):
        action, pi_info = self.model.pi(zs)
        alpha = self.alpha.detach()

        ent_coef_loss = torch.zeros((), device=self.device)
        if self.ent_coef_optim is not None:
            ent_coef_loss = -(
                self.log_ent_coef * (pi_info["log_prob"] + self.target_entropy).detach()
            ).mean()
            self.ent_coef_optim.zero_grad(set_to_none=True)
            ent_coef_loss.backward()
            self.ent_coef_optim.step()

        q_pi = self.model.Q(zs, action, return_type="min", detach=True)
        rho = torch.pow(
            torch.tensor(float(self.cfg.rho), device=self.device),
            torch.arange(zs.shape[0], device=self.device),
        )
        actor_per_t = (alpha * pi_info["log_prob"] - q_pi).mean(dim=(1, 2))
        actor_loss = (actor_per_t * rho).mean()

        self.pi_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), float(self.cfg.grad_clip_norm)
        )
        self.pi_optim.step()

        return {
            "actor_loss": actor_loss.detach(),
            "actor_grad_norm": torch.as_tensor(actor_grad_norm).detach(),
            "actor_entropy": pi_info["entropy"].detach().mean(),
            "ent_coef": alpha.detach(),
            "ent_coef_loss": ent_coef_loss.detach(),
        }

    def _update(self, obs, action, reward, terminated):
        """One TD-MPC2-style model update with scalar soft-Q regression."""
        # TD targets and latent consistency targets never receive gradients.
        with torch.no_grad():
            next_z_targets = self.model.encode(obs[1:])
            td_targets = self._soft_td_target(next_z_targets, reward, terminated)

        self.model.train()
        z = self.model.encode(obs[0])
        zs = [z]
        consistency_loss = torch.zeros((), device=self.device)
        for t, (recorded_action, next_z_target) in enumerate(
            zip(action.unbind(0), next_z_targets.unbind(0))
        ):
            z = self.model.next(z, recorded_action)
            consistency_loss = consistency_loss + F.mse_loss(z, next_z_target) * self.cfg.rho ** t
            zs.append(z)
        zs = torch.stack(zs, dim=0)

        rollout_zs = zs[:-1]
        reward_preds = self.model.reward(rollout_zs, action)
        qs = self.model.Q(rollout_zs, action, return_type="all")
        termination_pred = (
            self.model.termination(zs[1:], unnormalized=True)
            if self.cfg.episodic
            else None
        )

        reward_loss = torch.zeros((), device=self.device)
        critic_loss = torch.zeros((), device=self.device)
        for t in range(self.cfg.horizon):
            weight = self.cfg.rho ** t
            reward_loss = reward_loss + td_math.soft_ce(
                reward_preds[t], reward[t], self.cfg
            ).mean() * weight
            critic_loss = critic_loss + 0.5 * (
                F.mse_loss(qs[0, t], td_targets[t])
                + F.mse_loss(qs[1, t], td_targets[t])
            ) * weight

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        critic_loss = critic_loss / self.cfg.horizon
        if self.cfg.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
        else:
            termination_loss = torch.zeros((), device=self.device)

        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.termination_coef * termination_loss
            + self.cfg.critic_coef * critic_loss
        )

        self.optim.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._world_critic_params, float(self.cfg.grad_clip_norm)
        )
        self.optim.step()
        # Clear world-model/critic gradients before the actor pass. This is not
        # needed for the optimizer step itself, but makes the stop-gradient
        # boundary explicit and prevents stale critic gradients from lingering.
        self.optim.zero_grad(set_to_none=True)

        # SAC actor/temperature updates do not backpropagate into the world model.
        actor_info = self._update_actor(zs.detach())

        # Match the usual SAC schedule: update the target on the first critic
        # update, then every target_update_interval updates thereafter.
        if self.num_updates % int(self.cfg.target_update_interval) == 0:
            self.model.soft_update_target_Q()
        self.num_updates += 1

        self.model.eval()
        info = {
            "consistency_loss": consistency_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "termination_loss": termination_loss.detach(),
            "total_loss": total_loss.detach(),
            "grad_norm": torch.as_tensor(grad_norm).detach(),
            "q_target_mean": td_targets.detach().mean(),
            "q_mean": qs.detach().mean(),
        }
        info.update(actor_info)
        return info

    def update(self, buffer):
        obs, action, reward, terminated, task = buffer.sample()
        if task is not None:
            raise NotImplementedError("AMBI-TD-MPC2 currently supports single-task training only.")
        if (
            self.device.type == "cuda"
            and hasattr(torch, "compiler")
            and hasattr(torch.compiler, "cudagraph_mark_step_begin")
        ):
            torch.compiler.cudagraph_mark_step_begin()
        return self._update(obs, action, reward, terminated)
