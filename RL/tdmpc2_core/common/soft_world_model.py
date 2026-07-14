"""TD-MPC2 world model with SAC-style scalar or distributional critics.

The encoder, latent dynamics, reward, termination, policy architecture, SimNorm,
and initialization follow the vendored TD-MPC2 implementation. Its control head
supports either scalar twin critics or a TD-MPC2-style categorical Q ensemble
while retaining the same soft SAC Bellman semantics.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from . import init, layers, math
from .q_representation import QRepresentation


class SoftWorldModel(nn.Module):
    """Single-task TD-MPC2 world model with a squashed-Gaussian SAC actor."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.multitask:
            raise NotImplementedError("AMBI-TD-MPC2 currently supports single-task training only.")

        self.cfg = cfg
        self.q_backend = QRepresentation.from_config(cfg)
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
                    self.q_backend.output_dim,
                    dropout=cfg.dropout,
                )
                for _ in range(self.q_backend.num_q)
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
        q_label = (
            "Twin soft Q-functions"
            if self.q_backend.representation == "scalar"
            else "Distributional soft Q ensemble"
        )
        modules = [
            ("Encoder", self._encoder),
            ("Dynamics", self._dynamics),
            ("Reward", self._reward),
            ("Termination", self._termination),
            ("SAC actor", self._pi),
            (q_label, self._Qs),
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

    def pi(
        self,
        z,
        task=None,
        *,
        policy=None,
        deterministic=False,
        generator=None,
        std_scale=1.0,
        log_std_min=None,
        log_std_max=None,
    ):
        """Sample a tanh-squashed action and return its corrected log-probability."""
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        policy = self._pi if policy is None else policy

        std_scale = float(std_scale)
        if not 0.0 < std_scale < float("inf"):
            raise ValueError(f"std_scale must be positive, got {std_scale}.")

        mean_raw, log_std = policy(z).chunk(2, dim=-1)
        # SAC/SB3 clamp the predicted log standard deviation directly. TD-MPC2's
        # policy prior instead interpolates tanh(log_std) across its bounds; with
        # SAC's [-20, 2] bounds that would initialize near log_std=-9 and almost
        # eliminate exploration.
        lower_bound = (
            float(self.log_std_min.item())
            if log_std_min is None
            else float(log_std_min)
        )
        upper_bound = (
            float((self.log_std_min + self.log_std_dif).item())
            if log_std_max is None
            else float(log_std_max)
        )
        if not (
            float("-inf") < lower_bound < upper_bound < float("inf")
        ):
            raise ValueError(
                "log_std_min must be smaller than log_std_max, "
                f"got {lower_bound} >= {upper_bound}."
            )
        log_std = torch.clamp(log_std, min=lower_bound, max=upper_bound)
        if std_scale != 1.0:
            log_std = log_std + torch.log(log_std.new_tensor(std_scale))

        if deterministic:
            eps = torch.zeros_like(mean_raw)
        elif generator is None:
            eps = torch.randn_like(mean_raw)
        else:
            eps = torch.randn(
                mean_raw.shape,
                dtype=mean_raw.dtype,
                device=mean_raw.device,
                generator=generator,
            )
        log_prob = math.gaussian_logprob(eps, log_std)

        pre_tanh_action = mean_raw + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean_raw, pre_tanh_action, log_prob)
        return action, {
            "mean": mean,
            "pre_tanh_mean": mean_raw,
            "log_std": log_std,
            "log_prob": log_prob,
            "entropy": -log_prob,
        }

    @property
    def critic_signature(self):
        """Serializable critic architecture metadata for checkpoint preflight."""
        return self.q_backend.signature.as_dict()

    def q_predictions(
        self,
        z,
        a,
        task=None,
        *,
        target=False,
        detach=False,
        qs=None,
    ):
        """Return raw predictions from every selected critic head.

        Scalar critics return values with shape ``[num_q, ..., 1]``;
        distributional critics return logits with shape
        ``[num_q, ..., q_num_bins]``.
        """
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
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
        return out

    def q_values(self, z, a, task=None, *, target=False, detach=False, qs=None):
        """Return decoded scalar values from every selected critic head."""
        predictions = self.q_predictions(
            z,
            a,
            task,
            target=target,
            detach=detach,
            qs=qs,
        )
        return self.q_backend.decode(predictions)

    def critic_loss(self, predictions, scalar_target, *, reduction="mean"):
        """Compute MSE or two-hot cross entropy against one scalar target."""
        return self.q_backend.loss(predictions, scalar_target, reduction=reduction)

    def Q(
        self,
        z,
        a,
        task=None,
        *,
        return_type=None,
        reduction=None,
        target=False,
        detach=False,
        qs=None,
        pair_indices=None,
        generator=None,
    ):
        """Evaluate critics and return decoded scalar Q-values.

        ``return_type`` retains the legacy ``min``/``avg``/``all`` API. New
        callers should use explicit ``*_pair`` or ``*_all`` reductions.
        """
        if reduction is not None and return_type is not None:
            raise ValueError("Specify either return_type or reduction, not both.")
        if reduction is None:
            reduction = "min_pair" if return_type is None else return_type
        values = self.q_values(
            z,
            a,
            task,
            target=target,
            detach=detach,
            qs=qs,
        )
        return self.q_backend.reduce(
            values,
            reduction,
            pair_indices=pair_indices,
            generator=generator,
        )
