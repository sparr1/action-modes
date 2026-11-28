"""Fresh, actor-only pathwise optimization at a frozen evaluation root."""

from copy import deepcopy
import math

import torch

from utils.frozen_pathwise_settings import PROTOCOL, pathwise_settings

from .common.layers import detached_module_forward
from .common.scale import percentile_range


def _validate_config(cfg):
    # Disallow saved extensions with semantics this actor-only evaluator does not
    # implement. The saved critic is consulted only as the frozen outer tail.
    required = {
        "inner_operator": "sac", "obs": "state", "episodic": False,
        "critic_value_mode": "single", "aux_return_mode": "off",
        "outer_critic_target": "reward_only", "inner_sac_critic_target": "reward_only",
        "inner_finite_horizon": True, "inner_sac_return_estimator": "one_step",
        "inner_actor_adaptation": "clone", "inner_actor_initialization": "prior",
        "inner_actor_source": "sac", "inner_horizon_actor_source": "sac",
        "inner_horizon_critic_source": "sac", "inner_execution_policy_source": "primary",
        "inner_explorer_mode": "none", "inner_horizon_conditioning": "none",
        "inner_solve_interval": 1, "inner_update_timing": "round",
        "inner_schedule_mode": "canonical", "inner_component_update_schedule": True,
        "inner_critic_updates_per_round": 0, "inner_critic_updates_per_action": 0,
        "inner_actor_scope": "action", "inner_actor_optimizer_scope": "action",
        "inner_actor_loss_scale_scope": "action", "inner_actor_loss_scale_update": "per_update",
        "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
        "inner_actor_entropy_mode": "squashed", "inner_temperature_mode": "inherit_outer",
        "inner_temperature_initialization": "inherit_outer", "inner_temperature_scope": "action",
        "inner_temperature_optimizer_scope": "action", "inner_temperature_updates_per_action": 0,
        "inner_actor_writeback_coef": 0.0, "inner_critic_writeback_coef": 0.0,
        "inner_outer_policy_kl_coef": 0.0, "inner_outer_action_l2_coef": 0.0,
        "inner_critic_bernoulli_p": 0.0, "inner_outer_replay_fraction": 0.0,
        "compile": False, "inner_cuda_graph": False,
    }
    for key, expected in required.items():
        actual = getattr(cfg, key)
        if actual != expected:
            raise ValueError(f"Frozen pathwise requires {key}={expected!r}; got {actual!r}.")


class FrozenPathwiseController:
    """Install a native action-local actor solve behind ordinary AMBI prediction.

    Each update uses a fresh batch of reparameterized imagined trajectories.
    For the control, detaching successor latents changes only the backward path:
    local actor/reward gradients remain at every depth and no gradient travels
    through transitions. Both modes use the same stochastic forward objective.
    """

    def __init__(self, model, *, rollouts_per_update, detach_dynamics):
        cfg = model.cfg
        _validate_config(cfg)
        self._settings = pathwise_settings(
            cfg, rollouts_per_update=rollouts_per_update, detach_dynamics=detach_dynamics,
        )
        self._world = model.agent.model
        self._engine = model.agent.inner_engine
        self._device = model.agent.device
        if self._device.type not in {"cpu", "cuda"}:
            raise ValueError("Frozen pathwise supports CPU and CUDA private generators.")
        self._discount = float(model.agent.discount)
        if not math.isfinite(self._discount) or not 0 < self._discount <= 1:
            raise ValueError("Frozen pathwise requires a finite discount in (0, 1].")
        self._noise_shape = (self._settings["horizon"] + 1, rollouts_per_update, int(cfg.action_dim))
        self._initial_actor = self._engine._actor_base
        self._actor = self._engine._adapt_module(self._initial_actor, "actor").eval()
        self._actor_options = self._engine._inner_policy_kwargs()
        self._rng = torch.Generator(device=self._device)
        self._seeded = False
        self.calls = 0
        self._original_act = self._engine.act
        self._engine.act = self.act

    @property
    def settings(self):
        return deepcopy(self._settings)

    def reset_episode(self, seed):
        self._rng.manual_seed(int(seed))
        self._seeded = True

    def close(self):
        self._engine.act = self._original_act

    def _rollout(self, root, noise, pair):
        world = self._world
        horizon = self._settings["horizon"]
        z = root.detach().expand(noise.shape[1], -1)
        actor_options = {"policy": self._actor, **self._actor_options}
        action, info = world.pi(z, noise=noise[0], **actor_options)
        root_entropy = -info["log_prob"]
        total = z.new_zeros((z.shape[0], 1))
        discount = 1.0
        for depth in range(horizon):
            if depth:
                action = world.pi_action(z, noise=noise[depth], **actor_options)
            joint = world.joint_input(z, action)
            logits = detached_module_forward(world._reward, joint)
            total = total + discount * world.decode_reward(logits)
            z = detached_module_forward(world._dynamics, joint)
            if self._settings["detach_dynamics"]:
                z = z.detach()
            discount *= self._discount
        tail_action = world.pi_action(z, detach_policy=True, noise=noise[horizon])
        tail_q = world.Q(
            z, tail_action, detach=True, reduction=self._settings["terminal_q_reduction"],
            pair_indices=pair, trusted_pair_indices=True,
        )
        return total + discount * tail_q, root_entropy

    def act(self, root, *, t0=False, eval_mode=False, collect_diagnostics=True,
            trace=None, apply_inner_writeback=False, return_behavior_policy=False):
        if (not eval_mode or collect_diagnostics or trace is not None
                or apply_inner_writeback or return_behavior_policy):
            raise ValueError("Frozen pathwise requires mean execution without traces, probes or writeback.")
        if not self._seeded:
            raise ValueError("Seed the frozen pathwise controller before the first decision.")
        if self._world.training or root.ndim != 2 or root.shape[0] != 1 or root.device != self._device:
            raise ValueError("Frozen pathwise requires an eval-mode world model and one root on its device.")

        # The clone's storage may be reused, but its parameters, gradients and
        # Adam moments are always reset to a fresh solve.
        self._actor.load_state_dict(self._initial_actor.state_dict())
        optimizer = self._engine._new_optimizer(self._actor, "actor")
        parameters = tuple(self._actor.parameters())
        initial_scale = self._engine._critic_owner.actor_loss_scale.detach().reshape(())
        alpha = self._engine._actor_owner.alpha.detach().reshape(())
        finite = torch.isfinite(initial_scale) & (initial_scale >= 1) & torch.isfinite(alpha) & (alpha >= 0)
        scale = initial_scale
        percentiles = root.new_tensor([5.0, 95.0])
        pair = None
        if self._settings["terminal_q_reduction"].endswith("_pair"):
            pair = self._world.q_backend.sample_pair_indices(self._device, generator=self._rng)
        with torch.enable_grad():
            for _ in range(self._settings["actor_optimizer_steps"]):
                noise = torch.randn(self._noise_shape, device=self._device, dtype=root.dtype, generator=self._rng)
                returns, entropy = self._rollout(root, noise, pair)
                width = percentile_range(returns.detach(), percentiles).reshape(())
                # Use the proposed scale for this update, matching AMBI's
                # action-local per-update normalization convention.
                scale = scale.lerp(width, self._settings["scale_tau"])
                loss = -(returns / scale + alpha * entropy).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, self._settings["actor_grad_clip_norm"])
                optimizer.step()
                finite = finite & torch.isfinite(loss.detach()) & torch.isfinite(norm)
        with torch.no_grad():
            action = self._world.pi_action(root, policy=self._actor, deterministic=True, **self._actor_options)[0]
            if not bool((finite & torch.isfinite(action).all()).item()):
                raise FloatingPointError("Frozen pathwise produced a non-finite loss, gradient or action.")
        self.calls += 1
        return action, {
            "inner_rounds": float(self._settings["rounds"]),
            "inner_actor_optimizer_steps": float(self._settings["actor_optimizer_steps"]),
            "inner_critic_optimizer_steps": 0.0,
            "inner_temperature_optimizer_steps": 0.0,
            "inner_model_steps": float(self._settings["model_steps"]),
            "inner_cuda_graph_used": 0.0,
            "inner_compile_fallback": 0.0,
            "inner_actor_loss": loss.detach(),
            "inner_actor_loss_scale_initial": initial_scale,
            "inner_actor_loss_scale_final": scale,
            "inner_alpha": alpha,
            "inner_pathwise_discount": self._discount,
        }, []
