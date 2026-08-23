"""TOLD world model used by AMBI-XQC.

The encoder, recurrent latent dynamics, reward predictor, and optional
termination predictor retain the TD-MPC2/TOLD architecture.  Actor and value
heads deliberately live outside this module: AMBI-XQC supplies them through a
reusable XQC controller, so the TOLD reward support and the XQC value support
cannot be confused.
"""

from __future__ import annotations

import torch
from torch import nn

from . import init, layers


class XQCTOLDWorldModel(nn.Module):
    """The TOLD representation and transition model without control heads."""

    def __init__(self, cfg):
        super().__init__()
        if bool(cfg.multitask):
            raise NotImplementedError(
                "AMBI-XQC currently supports single-task training only."
            )
        if str(cfg.obs) != "state":
            raise NotImplementedError(
                "The first AMBI-XQC implementation supports state observations only."
            )

        self.cfg = cfg
        self._encoder = layers.enc(cfg)
        self._dynamics = layers.mlp(
            cfg.latent_dim + cfg.action_dim,
            2 * [cfg.mlp_dim],
            cfg.latent_dim,
            act=layers.SimNorm(cfg),
        )
        # This is the existing raw-reward TOLD head.  It remains independent
        # from XQC's linear categorical value support.
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

        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight])

    @property
    def total_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def __repr__(self):
        modules = [
            ("Encoder", self._encoder),
            ("Dynamics", self._dynamics),
            ("Raw-reward predictor", self._reward),
            ("Termination predictor", self._termination),
        ]
        lines = ["AMBI-XQC TOLD World Model"]
        lines.extend(
            f"{name}: {module}" for name, module in modules if module is not None
        )
        lines.append(f"Learnable parameters: {self.total_params:,}")
        return "\n".join(lines)

    def encode(self, obs, task=None):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-XQC.")
        return self._encoder[self.cfg.obs](obs)

    @staticmethod
    def joint_input(z, action):
        return torch.cat((z, action), dim=-1)

    def next_from_joint(self, joint):
        return self._dynamics(joint)

    def reward_from_joint(self, joint):
        return self._reward(joint)

    def next(self, z, action, task=None):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-XQC.")
        return self.next_from_joint(self.joint_input(z, action))

    def reward(self, z, action, task=None):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-XQC.")
        return self.reward_from_joint(self.joint_input(z, action))

    def termination(self, z, task=None, *, unnormalized=False):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-XQC.")
        if self._termination is None:
            raise RuntimeError("Termination prediction is disabled when episodic=False.")
        logits = self._termination(z)
        return logits if unnormalized else torch.sigmoid(logits)
