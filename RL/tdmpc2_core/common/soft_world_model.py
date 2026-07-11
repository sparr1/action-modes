"""TD-MPC2 world model with SAC-style scalar twin critics.

The encoder, latent dynamics, reward, termination, policy architecture, SimNorm,
and initialization follow the vendored TD-MPC2 implementation. The only control
head change is replacing TD-MPC2's distributional Q ensemble with two scalar
soft Q-functions suitable for SAC.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from . import init, layers, math


class SoftWorldModel(nn.Module):
    """Single-task TD-MPC2 world model with a squashed-Gaussian SAC actor."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.multitask:
            raise NotImplementedError("AMBI-TD-MPC2 currently supports single-task training only.")
        if int(cfg.num_q) != 2:
            raise ValueError("SAC requires exactly two Q-functions; set num_q=2.")

        self.cfg = cfg
        self._encoder = layers.enc(cfg)
        self._dynamics = layers.mlp(
            cfg.latent_dim + cfg.action_dim,
            2 * [cfg.mlp_dim],
            cfg.latent_dim,
            act=layers.SimNorm(cfg),
        )
        self._reward = layers.mlp(
            cfg.latent_dim + cfg.action_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
        )
        self._termination = (
            layers.mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], 1)
            if cfg.episodic
            else None
        )
        self._pi = layers.mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    cfg.latent_dim + cfg.action_dim,
                    2 * [cfg.mlp_dim],
                    1,
                    dropout=cfg.dropout,
                )
                for _ in range(2)
            ]
        )

        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight] + [q[-1].weight for q in self._Qs])

        self.register_buffer("log_std_min", torch.tensor(float(cfg.log_std_min)))
        self.register_buffer(
            "log_std_dif",
            torch.tensor(float(cfg.log_std_max) - float(cfg.log_std_min)),
        )
        self.init()

    def init(self):
        """Create a frozen EMA target critic."""
        self._target_Qs = deepcopy(self._Qs)
        self._target_Qs.requires_grad_(False)
        self._target_Qs.train(False)

    def __repr__(self):
        result = "AMBI TD-MPC2 Soft World Model\n"
        modules = [
            ("Encoder", self._encoder),
            ("Dynamics", self._dynamics),
            ("Reward", self._reward),
            ("Termination", self._termination),
            ("SAC actor", self._pi),
            ("Twin soft Q-functions", self._Qs),
        ]
        for name, module in modules:
            if module is not None:
                result += f"{name}: {module}\n"
        result += "Learnable parameters: {:,}".format(self.total_params)
        return result

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Rebuild targets from the now-moved online critic, matching WorldModel.
        self.init()
        return self

    def train(self, mode=True):
        super().train(mode)
        self._target_Qs.train(False)
        return self

    @torch.no_grad()
    def soft_update_target_Q(self, tau=None):
        tau = float(self.cfg.tau if tau is None else tau)
        for target_param, param in zip(self._target_Qs.parameters(), self._Qs.parameters()):
            target_param.data.lerp_(param.data, tau)
        for target_buffer, buffer in zip(self._target_Qs.buffers(), self._Qs.buffers()):
            target_buffer.data.copy_(buffer.data)

    def encode(self, obs, task=None):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task=None):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        return self._dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z, a, task=None):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        return self._reward(torch.cat([z, a], dim=-1))

    def termination(self, z, task=None, unnormalized=False):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        if self._termination is None:
            raise RuntimeError("Termination model is disabled because episodic=False.")
        logits = self._termination(z)
        return logits if unnormalized else torch.sigmoid(logits)

    def pi(self, z, task=None, *, policy=None, deterministic=False):
        """Sample a tanh-squashed action and return its corrected log-probability."""
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        policy = self._pi if policy is None else policy

        mean_raw, log_std = policy(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.zeros_like(mean_raw) if deterministic else torch.randn_like(mean_raw)
        log_prob = math.gaussian_logprob(eps, log_std)

        pre_tanh_action = mean_raw + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean_raw, pre_tanh_action, log_prob)
        return action, {
            "mean": mean,
            "log_std": log_std,
            "log_prob": log_prob,
            "entropy": -log_prob,
        }

    def Q(
        self,
        z,
        a,
        task=None,
        *,
        return_type="min",
        target=False,
        detach=False,
        qs=None,
    ):
        """Evaluate online, target, detached, or explicitly supplied twin critics."""
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        if return_type not in {"min", "avg", "all"}:
            raise ValueError(f"Unknown Q return_type: {return_type}")
        if target and (detach or qs is not None):
            raise ValueError("target=True cannot be combined with detach=True or an explicit critic.")

        q_input = torch.cat([z, a], dim=-1)
        if qs is not None:
            out = qs.forward_detached(q_input) if detach else qs(q_input)
        elif target:
            out = self._target_Qs(q_input)
        elif detach:
            out = self._Qs.forward_detached(q_input)
        else:
            out = self._Qs(q_input)

        if return_type == "all":
            return out
        if return_type == "min":
            return out.min(dim=0).values
        return out.mean(dim=0)
