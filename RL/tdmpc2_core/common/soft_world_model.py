"""TD-MPC2 world model with SAC-style scalar or distributional critics.

The encoder, latent dynamics, reward, termination, policy architecture, SimNorm,
and initialization follow the vendored TD-MPC2 implementation. Its control head
supports either scalar twin critics or a TD-MPC2-style categorical Q ensemble.
AMBI configures whether their Bellman bootstrap includes policy entropy.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from . import init, layers, math
from .q_representation import QRepresentation


DEFAULT_LOG_STD_MAPPING = "direct_clamp"
LOG_STD_MAPPINGS = frozenset({DEFAULT_LOG_STD_MAPPING, "tdmpc2_tanh"})


def normalize_log_std_mapping(value, key="log_std_mapping"):
    """Validate and canonicalize a policy log-standard-deviation mapping."""
    if not isinstance(value, str):
        raise ValueError(
            f"{key} must be one of {sorted(LOG_STD_MAPPINGS)}, got {value!r}."
        )
    value = value.lower()
    if value not in LOG_STD_MAPPINGS:
        raise ValueError(
            f"{key} must be one of {sorted(LOG_STD_MAPPINGS)}, got {value!r}."
        )
    return value


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

        self._log_std_min_value = float(cfg.log_std_min)
        self._log_std_max_value = float(cfg.log_std_max)
        self._log_std_mapping = normalize_log_std_mapping(
            getattr(cfg, "log_std_mapping", DEFAULT_LOG_STD_MAPPING)
        )
        self.register_buffer("log_std_min", torch.tensor(self._log_std_min_value))
        self.register_buffer(
            "log_std_dif",
            torch.tensor(self._log_std_max_value - self._log_std_min_value),
        )
        self.init()

    def init(self):
        """Create a frozen EMA target critic."""
        if hasattr(self, "_target_Qs"):
            self.soft_update_target_Q(tau=1.0)
        else:
            self._target_Qs = deepcopy(self._Qs)
        self._target_Qs.requires_grad_(False)
        self._target_Qs.train(False)

    def __repr__(self):
        result = "AMBI TD-MPC2 Soft World Model\n"
        q_label = (
            "Twin Q-functions"
            if self.q_backend.representation == "scalar"
            else "Distributional Q ensemble"
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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Checkpoints retain the legacy buffers. Refresh the Python scalars once
        # at load time so pi()/pi_action() never read a device scalar per call.
        self._log_std_min_value = float(self.log_std_min.item())
        self._log_std_max_value = self._log_std_min_value + float(
            self.log_std_dif.item()
        )

    def train(self, mode=True):
        super().train(mode)
        self._target_Qs.train(False)
        return self

    @torch.no_grad()
    def soft_update_target_Q(self, tau=None):
        tau = float(self.cfg.tau if tau is None else tau)
        target_parameters = tuple(self._target_Qs.parameters())
        online_parameters = tuple(self._Qs.parameters())
        if tau == 1.0:
            torch._foreach_copy_(target_parameters, online_parameters)
        elif tau != 0.0:
            torch._foreach_lerp_(target_parameters, online_parameters, tau)
        for target_buffer, buffer in zip(self._target_Qs.buffers(), self._Qs.buffers()):
            target_buffer.copy_(buffer)

    def encode(self, obs, task=None):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return torch.stack(
                [self._encoder[self.cfg.obs](time_obs) for time_obs in obs]
            )
        return self._encoder[self.cfg.obs](obs)

    @staticmethod
    def joint_input(z, a):
        """Pack a latent/action pair once for all transition and value heads."""
        return torch.cat((z, a), dim=-1)

    def next_from_joint(self, joint):
        """Predict the next latent from a prepacked latent/action tensor."""
        return self._dynamics(joint)

    def reward_from_joint(self, joint):
        """Predict reward logits from a prepacked latent/action tensor."""
        return self._reward(joint)

    def next(self, z, a, task=None):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        return self.next_from_joint(self.joint_input(z, a))

    def reward(self, z, a, task=None):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        return self.reward_from_joint(self.joint_input(z, a))

    def termination(self, z, task=None, unnormalized=False):
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        if self._termination is None:
            raise RuntimeError("Termination model is disabled because episodic=False.")
        logits = self._termination(z)
        return logits if unnormalized else torch.sigmoid(logits)

    def _policy_sample(
        self,
        z,
        task=None,
        *,
        policy=None,
        deterministic=False,
        generator=None,
        noise=None,
        std_scale=1.0,
        log_std_min=None,
        log_std_max=None,
        log_std_mapping=None,
    ):
        """Return policy parameters and one isolated standard-normal sample."""
        if task is not None:
            raise ValueError("Task IDs are not used in single-task AMBI-TD-MPC2.")
        policy = self._pi if policy is None else policy

        std_scale = float(std_scale)
        if not 0.0 < std_scale < float("inf"):
            raise ValueError(f"std_scale must be positive, got {std_scale}.")

        mean_raw, log_std = policy(z).chunk(2, dim=-1)
        lower_bound = (
            self._log_std_min_value
            if log_std_min is None
            else float(log_std_min)
        )
        upper_bound = (
            self._log_std_max_value
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
        mapping = (
            self._log_std_mapping
            if log_std_mapping is None
            else normalize_log_std_mapping(log_std_mapping)
        )
        if mapping == "direct_clamp":
            log_std = torch.clamp(log_std, min=lower_bound, max=upper_bound)
        else:
            # Exact TD-MPC2 policy-prior mapping: smoothly interpolate the
            # tanh-bounded head output across the configured interval.
            log_std = lower_bound + 0.5 * (upper_bound - lower_bound) * (
                torch.tanh(log_std) + 1.0
            )
        if std_scale != 1.0:
            log_std = log_std + torch.log(log_std.new_tensor(std_scale))

        if noise is not None and generator is not None:
            raise ValueError("Specify either policy noise or a generator, not both.")
        if noise is not None and noise.shape != mean_raw.shape:
            raise ValueError(
                "Policy noise must match the action-parameter shape, "
                f"got {tuple(noise.shape)} != {tuple(mean_raw.shape)}."
            )
        if deterministic:
            eps = torch.zeros_like(mean_raw)
        elif noise is not None:
            eps = noise.to(device=mean_raw.device, dtype=mean_raw.dtype)
        elif generator is None:
            eps = torch.randn_like(mean_raw)
        else:
            eps = torch.randn(
                mean_raw.shape,
                dtype=mean_raw.dtype,
                device=mean_raw.device,
                generator=generator,
            )
        return mean_raw, log_std, eps

    def pi_action(
        self,
        z,
        task=None,
        *,
        policy=None,
        deterministic=False,
        generator=None,
        noise=None,
        std_scale=1.0,
        log_std_min=None,
        log_std_max=None,
        log_std_mapping=None,
    ):
        """Sample only an action, omitting policy statistics and log-probability."""
        mean_raw, log_std, eps = self._policy_sample(
            z,
            task,
            policy=policy,
            deterministic=deterministic,
            generator=generator,
            noise=noise,
            std_scale=std_scale,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            log_std_mapping=log_std_mapping,
        )
        return torch.tanh(mean_raw + eps * log_std.exp())

    def pi(
        self,
        z,
        task=None,
        *,
        policy=None,
        deterministic=False,
        generator=None,
        noise=None,
        std_scale=1.0,
        log_std_min=None,
        log_std_max=None,
        log_std_mapping=None,
    ):
        """Sample a tanh-squashed action and return its corrected log-probability."""
        mean_raw, log_std, eps = self._policy_sample(
            z,
            task,
            policy=policy,
            deterministic=deterministic,
            generator=generator,
            noise=noise,
            std_scale=std_scale,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            log_std_mapping=log_std_mapping,
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

        q_input = self.joint_input(z, a)
        return self.q_predictions_from_joint(
            q_input,
            target=target,
            detach=detach,
            qs=qs,
        )

    def q_predictions_from_joint(
        self,
        q_input,
        *,
        target=False,
        detach=False,
        qs=None,
    ):
        """Run critics on an already packed latent/action tensor."""
        if target and (detach or qs is not None):
            raise ValueError("target=True cannot be combined with detach=True or an explicit critic.")
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
        trusted_pair_indices=False,
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
            trusted_pair_indices=trusted_pair_indices,
        )
