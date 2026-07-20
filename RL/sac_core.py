"""AMBI-native Soft Actor-Critic core.

This module intentionally mirrors the core semantics of Stable-Baselines3 SAC
while staying independent from SB3's VecEnv/replay-buffer stack so it can later
be reused with AMBI/TOLD rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from RL.tdmpc2_core.common.q_representation import QRepresentation


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6
_Q_REDUCTIONS = {"min_pair", "mean_pair", "min_all", "mean_all"}


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve SB3-style device strings without crashing on CPU-only jobs."""
    requested = str(device)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested for native SAC, but CUDA is unavailable. Falling back to CPU.")
        requested = "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    return torch.device(requested)


def make_feature_mlp(input_dim: int, hidden_dims: Iterable[int], activation_fn: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    hidden_dims = list(hidden_dims)
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation_fn())
        last_dim = hidden_dim
    return nn.Sequential(*layers)


def make_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int, activation_fn: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    hidden_dims = list(hidden_dims)
    layers = list(make_feature_mlp(input_dim, hidden_dims, activation_fn).children())
    last_dim = input_dim if len(hidden_dims) == 0 else hidden_dims[-1]
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau)
            torch.add(target_param.data, source_param.data, alpha=tau, out=target_param.data)


def _grad_norm(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    """Return the global L2 gradient norm without modifying gradients."""
    norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(norms).norm(2)


@dataclass
class SACConfig:
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str | float = "auto"
    target_entropy: str | float = "auto"
    target_update_interval: int = 1
    net_arch: Tuple[int, ...] = (256, 256)
    actor_net_arch: Optional[Tuple[int, ...]] = None
    critic_net_arch: Optional[Tuple[int, ...]] = None
    q_representation: str = "scalar"
    num_q: Optional[int] = None
    q_pair_size: int = 2
    q_target_reduction: str = "min_pair"
    q_actor_reduction: str = "min_pair"
    q_num_bins: int = 101
    q_vmin: float = -10.0
    q_vmax: float = 10.0
    adam_eps: float = 1e-8
    seed: Optional[int] = None
    device: str = "auto"
    verbose: int = 1

    def __post_init__(self):
        self.q_representation = str(self.q_representation).lower()
        if self.num_q is None:
            self.num_q = 5 if self.q_representation == "distributional" else 2


class ReplayBuffer:
    """Single-env replay buffer that stores SB3-style normalized actions.

    Observations are flattened by the wrapper before insertion. For time limits,
    TD targets use true termination only, matching SB3's timeout handling intent.
    """

    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        if int(capacity) <= 0:
            raise ValueError(f"Replay buffer capacity must be positive, got {capacity}.")
        if int(obs_dim) <= 0 or int(action_dim) <= 0:
            raise ValueError("Replay buffer observation and action dimensions must be positive.")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.terminated = np.zeros((self.capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        obs = self._validated_vector("obs", obs, self.obs_dim)
        action = self._validated_vector("action", action, self.action_dim)
        next_obs = self._validated_vector("next_obs", next_obs, self.obs_dim)
        reward = np.asarray(reward, dtype=np.float32)
        if reward.size != 1 or not np.isfinite(reward).all():
            raise ValueError("Replay reward must be one finite scalar.")

        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = float(reward.reshape(-1)[0])
        self.next_obs[self.pos] = next_obs
        self.terminated[self.pos] = float(terminated)
        self.truncated[self.pos] = float(truncated)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        # Match SB3: sample with replacement, including when size < batch_size.
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_obs": torch.as_tensor(self.next_obs[indices], device=device),
            # SB3 stores done=True at time limits but masks timeouts out when sampling.
            # With Gymnasium, terminated is the true bootstrap mask and truncated is a timeout.
            "dones": torch.as_tensor(self.terminated[indices], device=device),
        }

    @staticmethod
    def _validated_vector(name: str, value, expected_size: int) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.size != expected_size:
            raise ValueError(f"Replay {name} must contain {expected_size} values, got shape {value.shape}.")
        if not np.isfinite(value).all():
            raise ValueError(f"Replay {name} must contain only finite values.")
        return value


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, net_arch: Tuple[int, ...]):
        super().__init__()
        self.latent_pi = make_feature_mlp(obs_dim, net_arch)
        latent_dim = obs_dim if len(net_arch) == 0 else net_arch[-1]
        self.mu = nn.Linear(latent_dim, action_dim)
        self.log_std = nn.Linear(latent_dim, action_dim)

    def get_action_dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_pi = self.latent_pi(obs)
        mean_actions = self.mu(latent_pi)
        log_std = torch.clamp(self.log_std(latent_pi), LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_actions, log_std = self.get_action_dist_params(obs)
        std = torch.exp(log_std)
        dist = Normal(mean_actions, std)
        gaussian_actions = dist.rsample()
        actions = torch.tanh(gaussian_actions)
        log_prob = dist.log_prob(gaussian_actions)
        log_prob = log_prob - torch.log(1.0 - actions.pow(2) + EPS)
        return actions, log_prob.sum(dim=1, keepdim=True)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean_actions, log_std = self.get_action_dist_params(obs)
        if deterministic:
            return torch.tanh(mean_actions)
        std = torch.exp(log_std)
        return torch.tanh(Normal(mean_actions, std).rsample())


class ContinuousCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        net_arch: Tuple[int, ...],
        output_dim: int = 1,
        num_q: int = 2,
    ):
        super().__init__()
        self.num_q = int(num_q)
        if self.num_q < 2:
            raise ValueError("ContinuousCritic requires at least two Q heads.")
        input_dim = obs_dim + action_dim
        # Keep the historical qf1/qf2 module names exactly. Two-head scalar and
        # distributional checkpoints therefore retain byte-for-byte state-dict
        # key compatibility, while larger ensembles extend the same naming
        # scheme with qf3, qf4, ... .
        for index in range(self.num_q):
            self.add_module(
                f"qf{index + 1}",
                make_mlp(input_dim, net_arch, output_dim),
            )

    @property
    def q_networks(self) -> Tuple[nn.Module, ...]:
        return tuple(getattr(self, f"qf{index + 1}") for index in range(self.num_q))

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        q_input = torch.cat([obs, actions], dim=1)
        return tuple(qf(q_input) for qf in self.q_networks)


class SACAgent:
    """Reusable SAC learner with SB3-like update semantics."""

    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = resolve_device(config.device)

        if not 0.0 < float(config.gamma) <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {config.gamma}.")
        if not 0.0 < float(config.tau) <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {config.tau}.")
        if int(config.target_update_interval) <= 0:
            raise ValueError("target_update_interval must be positive.")
        if float(config.adam_eps) <= 0.0:
            raise ValueError("adam_eps must be positive.")

        for key in ("num_q", "q_pair_size"):
            raw_value = getattr(config, key)
            if isinstance(raw_value, bool):
                raise ValueError(f"{key} must be an integer, not a boolean.")
            try:
                value = int(raw_value)
                numeric_value = float(raw_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{key} must be an integer.") from exc
            if not np.isfinite(numeric_value) or numeric_value != value:
                raise ValueError(f"{key} must be an integer, got {raw_value!r}.")
            setattr(config, key, value)

        for key in ("q_target_reduction", "q_actor_reduction"):
            value = str(getattr(config, key)).lower()
            if value not in _Q_REDUCTIONS:
                raise ValueError(
                    f"{key} must be one of {sorted(_Q_REDUCTIONS)}, got {value!r}."
                )
            setattr(config, key, value)

        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        actor_arch = config.actor_net_arch if config.actor_net_arch is not None else config.net_arch
        critic_arch = config.critic_net_arch if config.critic_net_arch is not None else config.net_arch
        config.q_representation = str(config.q_representation).lower()
        self.q_backend = QRepresentation(
            config.q_representation,
            num_q=config.num_q,
            pair_size=config.q_pair_size,
            num_bins=config.q_num_bins,
            vmin=config.q_vmin,
            vmax=config.q_vmax,
        )
        if self.q_backend.representation == "scalar" and config.q_pair_size != 2:
            raise ValueError("Scalar SAC requires q_pair_size=2 for twin-Q semantics.")
        self.actor = SquashedGaussianActor(obs_dim, action_dim, actor_arch).to(self.device)
        self.critic = ContinuousCritic(
            obs_dim,
            action_dim,
            critic_arch,
            output_dim=self.q_backend.output_dim,
            num_q=self.q_backend.num_q,
        ).to(self.device)
        self.critic_target = ContinuousCritic(
            obs_dim,
            action_dim,
            critic_arch,
            output_dim=self.q_backend.output_dim,
            num_q=self.q_backend.num_q,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.critic_target.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate, eps=config.adam_eps
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate, eps=config.adam_eps
        )

        self.target_entropy = self._make_target_entropy(config.target_entropy)
        self.ent_coef_optimizer = None
        self.log_ent_coef = None
        if isinstance(config.ent_coef, str) and config.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in config.ent_coef:
                init_value = float(config.ent_coef.split("_", 1)[1])
                if init_value <= 0:
                    raise ValueError("Initial entropy coefficient must be > 0.")
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef], lr=config.learning_rate, eps=config.adam_eps
            )
            self.ent_coef_tensor = None
        else:
            fixed_ent_coef = float(config.ent_coef)
            if fixed_ent_coef <= 0.0:
                raise ValueError("Fixed entropy coefficient must be positive.")
            self.ent_coef_tensor = torch.tensor(fixed_ent_coef, device=self.device)

        self.num_updates = 0

    def _make_target_entropy(self, target_entropy: str | float) -> float:
        if target_entropy == "auto":
            return float(-self.action_dim)
        return float(target_entropy)

    @property
    def critic_signature(self) -> Dict[str, object]:
        """Return the semantic critic layout required for safe checkpoint loads."""
        return self.q_backend.signature.as_dict()

    @property
    def reduction_signature(self) -> Dict[str, object]:
        """Return non-architectural ensemble semantics required by a checkpoint."""
        return {
            "q_pair_size": int(self.config.q_pair_size),
            "q_target_reduction": self.config.q_target_reduction,
            "q_actor_reduction": self.config.q_actor_reduction,
        }

    def _decode_q_predictions(
        self, predictions: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        values = self.q_backend.decode(torch.stack(tuple(predictions), dim=0))
        return tuple(values.unbind(0))

    def _reduce_q_values(
        self, values: Tuple[torch.Tensor, ...], reduction: str
    ) -> torch.Tensor:
        return self.q_backend.reduce(torch.stack(tuple(values), dim=0), reduction)

    @staticmethod
    def _mean_pairwise_disagreement(values: torch.Tensor) -> torch.Tensor:
        """Mean absolute difference over every unordered pair of Q heads."""
        num_q = int(values.shape[0])
        differences = [
            (values[first] - values[second]).abs()
            for first in range(num_q)
            for second in range(first + 1, num_q)
        ]
        return torch.stack(differences, dim=0).mean()

    @staticmethod
    def _as_batch_tensor(value, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        self.actor.eval()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        action = self.actor(obs_t, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def q_predictions(self, obs, actions) -> Tuple[torch.Tensor, ...]:
        """Return raw scalar predictions or categorical logits from all critics."""
        obs_t = self._as_batch_tensor(obs, self.device)
        act_t = self._as_batch_tensor(actions, self.device)
        return self.critic(obs_t, act_t)

    def q_values(self, obs, actions) -> Tuple[torch.Tensor, ...]:
        """Expose decoded scalar Q-values from every native SAC critic."""
        return self._decode_q_predictions(self.q_predictions(obs, actions))

    def update(self, replay_buffer: ReplayBuffer, gradient_steps: int, batch_size: int) -> Dict[str, float]:
        self.actor.train()
        self.critic.train()
        metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "ent_coef": [],
            "ent_coef_loss": [],
            "policy_log_prob": [],
            "policy_entropy": [],
            "q1_mean": [],
            "q2_mean": [],
            "q_target_mean": [],
            "q_policy_mean": [],
            "q_disagreement_mean": [],
            "td_error_abs_mean": [],
            "actor_grad_norm": [],
            "critic_grad_norm": [],
            "ent_coef_grad_norm": [],
        }

        for _ in range(int(gradient_steps)):
            batch = replay_buffer.sample(batch_size, self.device)

            actions_pi, log_prob = self.actor.action_log_prob(batch["obs"])

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                ent_coef_grad_norm = _grad_norm([self.log_ent_coef])
                self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor
                ent_coef_grad_norm = None

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(batch["next_obs"])
                next_predictions = self.critic_target(batch["next_obs"], next_actions)
                next_values = self._decode_q_predictions(next_predictions)
                next_q = self._reduce_q_values(
                    next_values, self.config.q_target_reduction
                ) - ent_coef * next_log_prob
                target_q = batch["rewards"] + (1.0 - batch["dones"]) * self.config.gamma * next_q

            current_predictions = self.critic(batch["obs"], batch["actions"])
            current_values = self._decode_q_predictions(current_predictions)
            current_q1, current_q2 = current_values[:2]
            if self.q_backend.representation == "scalar":
                # Preserve the established reduction order for exact SB3 parity.
                critic_loss = 0.5 * (
                    F.mse_loss(current_q1, target_q)
                    + F.mse_loss(current_q2, target_q)
                )
            else:
                critic_loss = self.q_backend.loss(
                    torch.stack(current_predictions, dim=0), target_q
                )
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = _grad_norm(self.critic.parameters())
            self.critic_optimizer.step()

            policy_predictions = self.critic(batch["obs"], actions_pi)
            policy_values = self._decode_q_predictions(policy_predictions)
            reduced_policy_q = self._reduce_q_values(
                policy_values, self.config.q_actor_reduction
            )
            actor_loss = (ent_coef * log_prob - reduced_policy_q).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = _grad_norm(self.actor.parameters())
            self.actor_optimizer.step()

            # Keep the target-update phase continuous across separate update()
            # calls. This preserves the existing zero-based schedule (update on
            # steps 0, interval, 2 * interval, ...) without restarting it for
            # every rollout/train chunk.
            if self.num_updates % int(self.config.target_update_interval) == 0:
                polyak_update(self.critic, self.critic_target, self.config.tau)

            self.num_updates += 1
            metrics["actor_loss"].append(float(actor_loss.detach().cpu()))
            metrics["critic_loss"].append(float(critic_loss.detach().cpu()))
            metrics["ent_coef"].append(float(ent_coef.detach().cpu()))
            metrics["policy_log_prob"].append(float(log_prob.detach().mean().cpu()))
            metrics["policy_entropy"].append(float(-log_prob.detach().mean().cpu()))
            metrics["q1_mean"].append(float(current_q1.detach().mean().cpu()))
            metrics["q2_mean"].append(float(current_q2.detach().mean().cpu()))
            metrics["q_target_mean"].append(float(target_q.detach().mean().cpu()))
            metrics["q_policy_mean"].append(
                float(reduced_policy_q.detach().mean().cpu())
            )
            current_value_tensor = torch.stack(current_values, dim=0).detach()
            metrics["q_disagreement_mean"].append(
                float(self._mean_pairwise_disagreement(current_value_tensor).cpu())
            )
            metrics["td_error_abs_mean"].append(
                float((current_value_tensor - target_q.detach()).abs().mean().cpu())
            )
            if self.q_backend.num_q > 2:
                metrics.setdefault("q_ensemble_mean", []).append(
                    float(current_value_tensor.mean().cpu())
                )
                metrics.setdefault("q_ensemble_std", []).append(
                    float(
                        current_value_tensor.std(dim=0, unbiased=False).mean().cpu()
                    )
                )
                metrics.setdefault("q_ensemble_range", []).append(
                    float(
                        (
                            current_value_tensor.max(dim=0).values
                            - current_value_tensor.min(dim=0).values
                        )
                        .mean()
                        .cpu()
                    )
                )
                for head_index, head_values in enumerate(current_values[2:], start=3):
                    metrics.setdefault(f"q{head_index}_mean", []).append(
                        float(head_values.detach().mean().cpu())
                    )
            if self.q_backend.representation == "distributional":
                symlog_target = torch.sign(target_q) * torch.log1p(target_q.abs())
                current_logits = torch.stack(current_predictions, dim=0).detach()
                current_log_probs = F.log_softmax(current_logits, dim=-1)
                current_probs = current_log_probs.exp()
                metrics.setdefault("q_target_clip_fraction", []).append(
                    float(
                        (
                            (symlog_target <= float(self.q_backend.vmin))
                            | (symlog_target >= float(self.q_backend.vmax))
                        )
                        .float()
                        .mean()
                        .cpu()
                    )
                )
                metrics.setdefault("q_distribution_entropy", []).append(
                    float(
                        (-(current_probs * current_log_probs).sum(dim=-1))
                        .mean()
                        .cpu()
                    )
                )
                metrics.setdefault("q_distribution_max_probability", []).append(
                    float(current_probs.max(dim=-1).values.mean().cpu())
                )
            metrics["actor_grad_norm"].append(float(actor_grad_norm.detach().cpu()))
            metrics["critic_grad_norm"].append(float(critic_grad_norm.detach().cpu()))
            if ent_coef_loss is not None:
                metrics["ent_coef_loss"].append(float(ent_coef_loss.detach().cpu()))
            if ent_coef_grad_norm is not None:
                metrics["ent_coef_grad_norm"].append(float(ent_coef_grad_norm.detach().cpu()))

        return {key: float(np.mean(values)) for key, values in metrics.items() if values}

    def state_dict(self) -> Dict[str, object]:
        state = {
            "critic_spec": self.critic_signature,
            "reduction_spec": self.reduction_signature,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "num_updates": self.num_updates,
        }
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            state["log_ent_coef"] = self.log_ent_coef.detach().cpu()
            state["ent_coef_optimizer"] = self.ent_coef_optimizer.state_dict()
        else:
            state["ent_coef_tensor"] = self.ent_coef_tensor.detach().cpu()
        return state

    def load_state_dict(self, state: Dict[str, object]) -> None:
        legacy_scalar_spec = {
            "q_representation": "scalar",
            "num_q": 2,
            "q_num_bins": 1,
            "q_vmin": None,
            "q_vmax": None,
        }
        saved_spec = state.get("critic_spec", legacy_scalar_spec)
        if saved_spec != self.critic_signature:
            raise ValueError(
                "Checkpoint critic specification does not match this SAC agent: "
                f"checkpoint={saved_spec}, configured={self.critic_signature}."
            )
        legacy_reduction_spec = {
            "q_pair_size": 2,
            "q_target_reduction": "min_pair",
            "q_actor_reduction": "min_pair",
        }
        saved_reduction_spec = state.get(
            "reduction_spec", legacy_reduction_spec
        )
        if saved_reduction_spec != self.reduction_signature:
            raise ValueError(
                "Checkpoint Q reduction specification does not match this SAC agent: "
                f"checkpoint={saved_reduction_spec}, "
                f"configured={self.reduction_signature}."
            )
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.num_updates = int(state.get("num_updates", 0))
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            if "log_ent_coef" not in state or "ent_coef_optimizer" not in state:
                raise ValueError("Checkpoint uses a fixed entropy coefficient, but this agent uses automatic entropy tuning.")
            self.log_ent_coef.data.copy_(state["log_ent_coef"].to(self.device))
            self.ent_coef_optimizer.load_state_dict(state["ent_coef_optimizer"])
        else:
            if "ent_coef_tensor" not in state:
                raise ValueError("Checkpoint uses automatic entropy tuning, but this agent uses a fixed coefficient.")
            self.ent_coef_tensor = state["ent_coef_tensor"].to(self.device)
