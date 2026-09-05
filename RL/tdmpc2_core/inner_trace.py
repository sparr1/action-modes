"""Optional, action-owned inner-learning observations; never experiment I/O.

Update metrics describe the minibatch *before* its optimizer step. Probe
metrics describe the policy *after* the completed updates. Tensor scalars stay
on the device until the agent packs them with its returned action.
"""

from numbers import Integral

import torch

from .common import math as td_math


_DEFINITIONS = {
    "critic_loss": "Critic training loss on the pre-update sampled minibatch.",
    "critic_grad_norm": "Critic gradient norm before gradient clipping.",
    "td_error_abs_mean": "Mean absolute decoded TD error on the pre-update minibatch.",
    "q_mean": "Mean decoded online inner Q on the pre-update critic minibatch.",
    "q_abs_mean": "Mean absolute decoded online inner Q on the critic minibatch.",
    "q_target_mean": "Mean bootstrap target used for this critic update.",
    "q_target_clip_fraction": "Fraction of targets at or outside the distributional support edges.",
    "actor_loss": "Actor training objective evaluated before its optimizer step.",
    "actor_grad_norm": "Actor gradient norm before gradient clipping.",
    "actor_q_mean": "Q reduction used by the actor on its pre-update sampled actions.",
    "actor_q_mean_all": "All-head mean Q on the actor's sampled actions.",
    "actor_q_min_all": "All-head minimum Q on the actor's sampled actions.",
    "actor_q_mean_all_minus_min_all": "All-head mean-minus-minimum Q on actor samples.",
    "actor_entropy": "Mean negative squashed-policy log probability on actor samples.",
    "actor_pre_tanh_abs_mean": "Mean absolute pre-tanh value of sampled actor actions.",
    "actor_pre_tanh_abs_max": "Maximum absolute pre-tanh value of sampled actor actions.",
    "actor_pre_tanh_abs_ge_7p6_fraction": (
        "Fraction of actor samples beyond the former tanh Jacobian floor crossover."
    ),
    "actor_action_exact_saturation_fraction": (
        "Fraction of sampled action coordinates rounded exactly to a tanh bound."
    ),
    "outer_action_l2": "Pre-update sampled-action L2 anchor penalty for the TD3 actor.",
    "temperature_loss": "Automatic-temperature objective evaluated before its update.",
    "temperature_grad_norm": "Temperature gradient norm before clipping.",
    "alpha_used": "Temperature used by this update's losses, before temperature adaptation.",
    "alpha": "Inner temperature at this event boundary.",
    "outer_policy_kl": "Pre-update analytic Gaussian KL used by an enabled actor regularizer.",
    "policy_mean_delta_l2": "Root mean-action L2 displacement from the frozen outer policy.",
    "outer_policy_kl_probe": "Post-update root Gaussian KL(inner || outer); no policy sampling.",
    "fixed_target_q_action_gain": (
        "Frozen outer target mean-all Q(inner mean action) minus Q(outer mean action)."
    ),
    "fixed_evaluator_alpha": "Frozen outer temperature used for every probe's soft score.",
    "probe_model_steps": "Additional model transitions actually computed by this probe event.",
    "probe_seconds": (
        "Probe elapsed time; CUDA events resolved after the action's existing host transfer."
    ),
    "collection_transitions": "Imagined transitions appended during this collection round.",
    "collection_reward_sum_mean": (
        "Mean undiscounted imagined return of this round's collection policy."
    ),
    "collection_discounted_reward_mean": (
        "Mean discounted imagined reward of this round's collection policy."
    ),
}


def metric_definitions():
    """Return portable descriptions; callers may retain unlisted raw metrics."""
    result = dict(_DEFINITIONS)
    components = {
        "discounted_reward": "Discounted predicted rewards over the probe horizon",
        "discounted_terminal_q": "Discounted frozen outer target mean-all Q at the probe horizon",
        "fixed_alpha_entropy_bonus": (
            "Discounted -outer-alpha*log(pi) over the horizon and terminal action"
        ),
        "predicted_score": "Predicted reward sum plus terminal Q; Q may represent a soft return",
        "fixed_alpha_soft_score": "Predicted score plus the fixed-alpha entropy bonus",
    }
    for name, description in components.items():
        for suffix, subject in (
            ("outer", "outer policy"), ("inner", "current inner policy"),
            ("gain", "inner minus outer"),
        ):
            result[f"{name}_{suffix}"] = f"{description}, {subject}; paired fixed probe noise."
    return result


def metric_catalog(metric_names=()):
    """Structured metric identity shared by artifact writers and visualizations.

    Optional additional raw keys remain available with explicit fallback
    descriptions rather than being silently discarded by the reporting layer.
    """
    definitions = metric_definitions()
    probe_names = {
        name for name in definitions
        if name.endswith(("_outer", "_inner", "_gain"))
        or name.startswith("probe_")
    } | {"policy_mean_delta_l2", "outer_policy_kl_probe", "fixed_evaluator_alpha"}
    result = {}
    for name in sorted(set(definitions) | set(metric_names)):
        unit = "scalar"
        if name in probe_names:
            phase, axis = "post_update_fixed_probe", "round_index"
        elif name.startswith("collection_"):
            phase, axis = "post_collection", "round_index"
        elif name == "alpha":
            phase, axis = "initial_or_post_update_probe", "round_index"
        elif name.startswith("temperature_") or name == "alpha_used":
            phase, axis = "pre_update_minibatch", "temperature_updates"
        elif name.startswith("actor_") or name in {"outer_policy_kl", "outer_action_l2"}:
            phase, axis = "pre_update_minibatch", "actor_updates"
        else:
            phase, axis = "pre_update_minibatch", "critic_updates"
        if name.endswith("seconds"):
            unit = "seconds"
        elif name.endswith(("_fraction", "_rate")):
            unit = "fraction"
        elif name.endswith(("_steps", "_transitions", "_count")):
            unit = "count"
        elif "kl" in name or name == "actor_entropy":
            unit = "nats"
        elif "alpha" in name and "entropy_bonus" not in name and "soft_score" not in name:
            unit = "temperature"
        elif "q" in name or "score" in name or "reward" in name or "entropy_bonus" in name:
            unit = "value"
        elif "loss" in name:
            unit = "objective"
        elif "l2" in name:
            unit = "normalized_action"
        result[name] = {
            "definition": definitions.get(name, f"Raw inner optimizer metric {name}."),
            "unit": unit, "sampling_phase": phase, "preferred_axis": axis,
        }
    return result


class InnerActionTrace:
    """Single-use recorder populated by ``predict(..., trace=recorder)``.

    ``events`` contains only host dictionaries after the action returns. One
    recorder owns one action; reuse is rejected to prevent accidental root mixing.
    """

    def __init__(self, *, probes=False, probe_seed=0, probe_rollouts=8, probe_horizon=3):
        if not isinstance(probes, bool):
            raise TypeError("probes must be bool.")
        for name, value, minimum in (
            ("probe_seed", probe_seed, 0),
            ("probe_rollouts", probe_rollouts, 1),
            ("probe_horizon", probe_horizon, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if int(probe_seed) >= 2**63:
            raise ValueError("probe_seed must be smaller than 2**63.")
        self.probes = probes
        self.probe_seed = int(probe_seed)
        self.probe_rollouts = int(probe_rollouts)
        self.probe_horizon = int(probe_horizon)
        self.events = []
        self.round_index = 0
        self._started = False
        self._materialized = False
        self._noise = None
        self._outer_probe = None
        self._outer_stats = None
        self._outer_q = None
        self._alpha = None
        self._probe_timings = []

    def begin(self):
        if self._started:
            raise ValueError("An InnerActionTrace can record only one action.")
        self._started = True

    def record(self, phase, state, metrics=None, **metadata):
        if not self._started or self._materialized:
            raise RuntimeError("Trace recording requires an active action.")
        values = dict(metrics or {})
        for key, value in values.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError(f"Trace metric {key!r} must be scalar.")
                values[key] = value.detach().reshape(())
            else:
                values[key] = float(value)
        self.events.append({
            "event_index": len(self.events),
            "phase": phase,
            "round_index": self.round_index,
            "critic_updates": int(state.critic_steps),
            "actor_updates": int(state.actor_steps),
            "temperature_updates": int(state.temperature_steps),
            "replay_size": int(state.replay.size) if state.replay is not None else 0,
            "metrics": values,
            **metadata,
        })

    def tensor_items(self):
        """Ordered scalar references for the existing action-boundary pack."""
        return [
            (event["metrics"], key, value)
            for event in self.events
            for key, value in event["metrics"].items()
            if torch.is_tensor(value)
        ]

    def materialize(self, items, values):
        """Replace device references using an already-host slice, without copying."""
        if values.device.type != "cpu" or values.numel() != len(items):
            raise ValueError("Trace materialization requires one host value per tensor metric.")
        for (metrics, key, _), value in zip(items, values.tolist()):
            metrics[key] = float(value)
        for event, start, end in self._probe_timings:
            event["metrics"]["probe_seconds"] = (
                start.elapsed_time(end) / 1000.0
                if isinstance(start, torch.cuda.Event) else end - start
            )
        self._probe_timings.clear()
        self._materialized = True
        self._noise = self._outer_probe = self._outer_stats = self._outer_q = self._alpha = None

    @torch.no_grad()
    def _trajectory(self, engine, root_z, policy, *, inner):
        model, cfg = engine.model, engine.cfg
        z = root_z.expand(self.probe_rollouts, -1).clone()
        reward_sum = z.new_zeros(self.probe_rollouts, 1)
        entropy_bonus = torch.zeros_like(reward_sum)
        continuation = torch.ones_like(reward_sum)
        discount = 1.0
        bounds = self._policy_bounds(cfg) if inner else {}
        for step in range(self.probe_horizon):
            action, info = model.pi(z, policy=policy, noise=self._noise[step], **bounds)
            joint = model.joint_input(z, action)
            reward = td_math.two_hot_inv(model.reward_from_joint(joint), cfg)
            reward_sum += discount * continuation * reward
            entropy_bonus -= discount * continuation * self._alpha * info["log_prob"]
            z = model.next_from_joint(joint)
            if cfg.episodic:
                alive = model.termination(z) <= float(cfg.inner_termination_threshold)
                continuation *= alive.to(z.dtype)
            discount *= float(engine.agent.discount)
        action, info = model.pi(z, policy=policy, noise=self._noise[-1], **bounds)
        terminal_q = discount * continuation * model.Q(
            z, action, target=True, reduction="mean_all"
        )
        entropy_bonus -= discount * continuation * self._alpha * info["log_prob"]
        score = reward_sum + terminal_q
        return {
            "discounted_reward": reward_sum.mean(),
            "discounted_terminal_q": terminal_q.mean(),
            "fixed_alpha_entropy_bonus": entropy_bonus.mean(),
            "predicted_score": score.mean(),
            "fixed_alpha_soft_score": (score + entropy_bonus).mean(),
        }

    @staticmethod
    def _policy_bounds(cfg):
        return {
            "log_std_mapping": cfg.inner_log_std_mapping,
            "log_std_min": cfg.inner_log_std_min,
            "log_std_max": cfg.inner_log_std_max,
        }

    @torch.no_grad()
    def probe(self, engine, root_z, policy, *, inner=True):
        """Evaluate with fixed explicit noise and dropout off; consume no learner RNG."""
        model, cfg = engine.model, engine.cfg
        modes = tuple(
            (module, bool(module.training))
            for root in (model, policy) for module in root.modules()
        )
        started = engine._timer_start()
        event = None
        try:
            model.eval()
            policy.eval()
            model_steps = self.probe_rollouts * self.probe_horizon
            if self._noise is None:
                generator = torch.Generator(device=root_z.device).manual_seed(self.probe_seed)
                self._noise = torch.randn(
                    (self.probe_horizon + 1, self.probe_rollouts, int(cfg.action_dim)),
                    device=root_z.device, dtype=root_z.dtype, generator=generator,
                )
                self._alpha = engine.agent.alpha.detach().clone()
                self._outer_stats = model.policy_stats(root_z, policy=model._pi)
                self._outer_q = model.Q(
                    root_z, self._outer_stats["mean"], target=True, reduction="mean_all"
                )
                self._outer_probe = self._trajectory(engine, root_z, model._pi, inner=False)
                model_steps *= 2
            bounds = self._policy_bounds(cfg) if inner else {}
            stats = model.policy_stats(root_z, policy=policy, **bounds)
            q = model.Q(root_z, stats["mean"], target=True, reduction="mean_all")
            scores = self._trajectory(engine, root_z, policy, inner=inner)
            metrics = {
                "policy_mean_delta_l2": torch.linalg.vector_norm(
                    stats["mean"] - self._outer_stats["mean"], dim=-1
                ).mean(),
                "outer_policy_kl_probe": engine._gaussian_kl(stats, self._outer_stats).mean(),
                "fixed_target_q_action_gain": (q - self._outer_q).mean(),
                "fixed_evaluator_alpha": self._alpha,
                "alpha": engine.alpha.detach() if inner else self._alpha,
                "probe_model_steps": model_steps,
            }
            for name, value in scores.items():
                metrics[f"{name}_outer"] = self._outer_probe[name]
                metrics[f"{name}_inner"] = value
                metrics[f"{name}_gain"] = value - self._outer_probe[name]
            self.record("probe", engine.state, metrics, measurement="post_update_fixed_probe")
            event = self.events[-1]
        finally:
            for module, was_training in modes:
                module.training = was_training
            if event is not None:
                self._probe_timings.append((event, started, engine._timer_start()))
