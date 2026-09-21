"""Optional, action-owned inner-learning observations; never experiment I/O.

Update metrics describe the minibatch *before* its optimizer step. Probe
metrics describe the policy *after* the completed updates. Tensor scalars stay
on the device until the agent packs them with its returned action.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
from numbers import Integral
import time

import torch

from .horizon_diagnostics import horizon_metric_description
from .common import math as td_math


@dataclass(frozen=True)
class FrozenActorSnapshot:
    """Immutable, process-local actor export, independent of the learner's storage.

    The byte payload is a trusted local PyTorch module serialization. It is not
    a portable checkpoint format and must not be loaded from untrusted input.
    ``make_policy`` returns a new module; mutating it cannot change the snapshot.
    """

    round_index: int
    actor_updates: int
    critic_updates: int
    temperature_updates: int
    inner: bool
    bounds: tuple
    payload: bytes
    horizon_conditioning_horizon: int | None = None

    @property
    def policy_bounds(self):
        return dict(self.bounds)

    @property
    def sha256(self):
        return hashlib.sha256(self.payload).hexdigest()

    def make_policy(self, device="cpu"):
        policy = torch.load(io.BytesIO(self.payload), map_location=device,
                            weights_only=False)
        if getattr(policy, "horizon_conditioning_horizon", None) != self.horizon_conditioning_horizon:
            raise ValueError("Frozen actor horizon-conditioning metadata is incompatible.")
        return policy.eval().requires_grad_(False)


@torch.no_grad()
def evaluate_frozen_outer_q(model, z, action, *, reduction, pair_indices=None, critic=None):
    """Use the ordinary online-Q decoding/reduction without compiling a probe.

    Diagnostic batch sizes must not create, recompile, or disable the learner's
    cached ensemble kernel. Passing the eager callable through ``Q`` preserves
    its native/scalar/distributional decoding and explicit pair semantics.
    """
    critic = model._Qs if critic is None else critic
    modes = tuple((module, bool(module.training)) for module in critic.modules())
    try:
        critic.eval()
        split = getattr(getattr(model, "cfg", None), "critic_value_mode", "single") == "return_entropy"
        return model.Q(z, action, qs=critic._forward_eager, reduction=reduction,
                       pair_indices=pair_indices, trusted_pair_indices=True,
                       **({"projection": "return"} if split else {}))
    finally:
        for module, was_training in modes:
            module.training = was_training


@torch.no_grad()
def evaluate_outer_tail(engine, root_z, policy, noise, *, pair_indices=None,
                        policy_bounds=None):
    """Return paired per-rollout reward, bootstrap and total in raw reward units.

    ``noise`` has shape ``(H + 1, rollouts, action_dim)``. Its last row samples
    the frozen outer-policy action at the horizon. Pair reductions require an
    explicit critic pair unless the pair is the entire ensemble, so this
    diagnostic never draws from learner RNG.
    The returned tensors have shape ``(rollouts, 1)`` and remain on device.
    """
    model, cfg = engine.model, engine.cfg
    if noise.ndim != 3 or noise.shape[0] < 2 or noise.shape[1] < 1:
        raise ValueError("noise must have shape (H + 1, rollouts, action_dim), H >= 1.")
    if noise.shape[2] != int(cfg.action_dim):
        raise ValueError("noise action dimension does not match the checkpoint.")
    if root_z.ndim != 2 or root_z.shape[0] not in (1, noise.shape[1]):
        raise ValueError("root_z must contain one root or one root per rollout.")
    conditioned_horizon = getattr(policy, "horizon_conditioning_horizon", None)
    if conditioned_horizon is not None:
        if (noise.shape[0] - 1 != conditioned_horizon
                or getattr(cfg, "inner_rollout_horizon", conditioned_horizon) != conditioned_horizon):
            raise ValueError("Conditioned actor probe horizon must match its horizon_conditioning_horizon.")
    reduction = ("min_pair" if getattr(cfg, "critic_value_mode", "single") == "return_entropy"
                 else cfg.mppi_terminal_q_reduction)
    backend = getattr(model, "q_backend", None)
    pair_needs_rng = backend is None or backend.pair_size != backend.num_q
    if reduction.endswith("_pair") and pair_needs_rng and pair_indices is None:
        raise ValueError("A paired outer-Q reduction requires explicit pair_indices.")
    modes = {module: bool(module.training)
             for root in (model, policy) for module in root.modules()}
    try:
        model.eval()
        policy.eval()
        z = root_z.expand(noise.shape[1], -1).clone()
        rewards = z.new_zeros(noise.shape[1], 1)
        alive = torch.ones_like(rewards, dtype=torch.bool)
        discount = 1.0
        for step in range(noise.shape[0] - 1):
            horizon_kwargs = ({"remaining_horizon": torch.full(
                (z.shape[0], 1), conditioned_horizon - step,
                dtype=torch.long, device=z.device,
            )} if conditioned_horizon is not None else {})
            action, _ = model.pi(z, policy=policy, noise=noise[step],
                                 **(policy_bounds or {}), **horizon_kwargs)
            joint = model.joint_input(z, action)
            prediction = model.reward_from_joint(joint)
            # External analytic probe models retain the historical scalar seam.
            reward = (model.decode_reward(prediction) if hasattr(model, "decode_reward")
                      else td_math.two_hot_inv(prediction, cfg))
            rewards += torch.where(alive, discount * reward, 0.0)
            z = model.next_from_joint(joint)
            if cfg.episodic:
                alive &= model.termination(z) <= float(cfg.inner_termination_threshold)
            discount *= float(engine.agent.discount)
        routed = getattr(cfg, "aux_return_mode", "off") != "off"
        action, _ = model.pi(
            z, noise=noise[-1],
            **({"policy": engine._horizon_actor, **engine._horizon_actor_options}
               if routed else {}),
        )
        q = evaluate_frozen_outer_q(
            model, z, action, reduction=reduction, pair_indices=pair_indices,
            **({"critic": engine._horizon_critic} if routed else {}),
        )
        bootstrap = torch.where(alive, discount * q, 0.0)
        return {"reward": rewards, "bootstrap": bootstrap,
                "total": rewards + bootstrap}
    finally:
        for module, was_training in modes.items():
            module.training = was_training


_DEFINITIONS = {
    "critic_loss": "Critic training loss on the pre-update sampled minibatch.",
    "critic_grad_norm": "Critic gradient norm before gradient clipping.",
    "critic_value_loss": "Native TD-MPC2 head-averaged distributional cross entropy before value coefficient.",
    "actor_q_scaled_mean": "Native TD-MPC2 actor mean-pair Q divided by the updated local Q scale.",
    "actor_native_entropy": "Native TD-MPC2 negative squashed-policy log probability; not SAC temperature.",
    "actor_native_scaled_entropy": "Native TD-MPC2 scaled entropy including its action-dimension and entropy-ratio expression.",
    "actor_native_entropy_contribution": "Fixed native entropy coefficient times native scaled entropy in the actor objective.",
    "actor_q_scale_before": "Local Q scale before the actor minibatch percentile-range EMA update.",
    "actor_q_scale_after": "Local Q scale after the actor minibatch percentile-range EMA update; used by this actor loss.",
    "actor_q_percentile_range": "Current actor minibatch P95 minus P5 of mean-pair Q, clamped to at least one.",
    "tdambi_entropy_coef": "Fixed native TD-MPC2 entropy coefficient; not SAC alpha.",
    "tdambi_calibration_scale": "Initial Q scale from frozen-prior actions and frozen online mean-pair Q on first-collection replay.",
    "tdambi_calibration_samples": "Additional replay rows, policy evaluations and Q evaluations used once for local scale initialization.",
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
    "actor_scaled_entropy": (
        "Mean selected TD-MPC2 scaled-entropy statistic on the actor sample, "
        "including the literal action-dimension and entropy-ratio expression; "
        "this is not squashed-action entropy."
    ),
    "actor_entropy_bonus": (
        "Pre-update alpha times selected entropy, averaged over the actor "
        "minibatch and subtracted from its objective; not divided by the Q scale."
    ),
    "explorer_actor_scaled_entropy": (
        "Mean selected TD-MPC2 scaled-entropy statistic on the separate "
        "explorer actor sample; this is not squashed-action entropy."
    ),
    "explorer_actor_entropy_bonus": (
        "Pre-update explorer alpha times selected entropy, averaged over its "
        "actor minibatch and subtracted from its objective; not divided by the Q scale."
    ),
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
    "probe_policy_evaluations": "Additional sampled policy rows computed by this probe event, including the outer handoff.",
    "probe_q_evaluations": "Additional outer-Q ensemble rows computed by this probe event; selected heads reduce each row.",
    "probe_reward_evaluations": "Additional predicted-reward rows computed by this probe event.",
    "actor_snapshot_seconds": "Host elapsed time copying and serializing this immutable actor snapshot, including device transfer.",
    "actor_snapshot_bytes": "Serialized size of the immutable actor snapshot.",
    "probe_seconds": (
        "Probe elapsed time; CUDA events resolved after the action's existing host transfer."
    ),
    "collection_transitions": "Imagined transitions appended in this collection event (a round or parallel step).",
    "collection_rollout_step": "One-based depth of this parallel collection step within its round.",
    "collection_reward_sum_mean": (
        "Mean accumulated undiscounted imagined reward in this round through this collection event."
    ),
    "collection_discounted_reward_mean": (
        "Mean accumulated discounted imagined reward in this round through this collection event."
    ),
    "togo_return_mean": "Mean H-step current-policy model return plus discounted frozen outer-policy/online-Q tail; no entropy bonus.",
    "togo_return_std": "Population standard deviation across to-go probe rollouts, conditional on the fixed sampled critic pair.",
    "togo_reward_mean": "Mean discounted predicted reward over the H current-policy steps of the to-go probe.",
    "togo_bootstrap_mean": "Mean discounted frozen outer online Q at an outer-policy terminal action, using mppi_terminal_q_reduction.",
    "togo_return_initial_mean": "To-go return of this solve's initial actor, evaluated with the same probe noise and critic pair.",
    "togo_return_outer_mean": "To-go return of the frozen outer actor over all H steps and the outer tail, with paired probe noise.",
    "togo_return_gain_vs_initial": "Current minus initial to-go return at the identical root with paired probe noise.",
    "togo_return_gain_vs_outer": "Current minus frozen outer-policy to-go return at the identical root with paired probe noise.",
}


def metric_definitions():
    """Return portable descriptions; callers may retain unlisted raw metrics."""
    result = dict(_DEFINITIONS)
    for component in ("critic", "actor"):
        for suffix, description in {
            "window_rounds": "Number of newest whole collection rounds eligible for this update.",
            "window_transitions": "Number of transitions eligible for this update.",
            "window_fraction": "Eligible transitions divided by live replay transitions.",
            "window_round_fraction": "Eligible collection rounds divided by retained rounds.",
            "round_age_mean": "Mean sampled round age; the newest collection round has age zero.",
            "round_age_min": "Minimum sampled round age, in collection rounds.",
            "round_age_max": "Maximum sampled round age, in collection rounds.",
            "newest_round_fraction": "Fraction of sampled rows generated in the newest collection round.",
            "batch_unique_fraction": "Unique transition IDs divided by minibatch size.",
        }.items():
            result[f"{component}_replay_{suffix}"] = description
    for component in ("reward", "bootstrap"):
        for reference in ("initial", "outer"):
            result[f"togo_{component}_{reference}_mean"] = (
                f"Mean discounted {component} component of the {reference} actor's paired to-go return."
            )
            result[f"togo_{component}_gain_vs_{reference}"] = (
                f"Current minus {reference} discounted {component}, using paired diagnostic noise."
            )
    for reference in ("initial", "outer"):
        result[f"togo_return_gain_vs_{reference}_std"] = (
            f"Population standard deviation of current minus {reference} paired to-go return."
        )
    components = {
        "discounted_reward": "Discounted predicted rewards over the probe horizon",
        "discounted_terminal_q": "Discounted frozen outer Q at the probe horizon, using the recorded controller sources and reduction",
        "fixed_alpha_entropy_bonus": (
            "Discounted fixed-alpha entropy over the probe; split and auxiliary boundaries add no terminal entropy"
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
        ere_round_count = (
            name.startswith(("inner_critic_replay_round_", "inner_actor_replay_round_"))
            and name.endswith("_sample_count")
        )
        if ere_round_count:
            phase, axis = "post_decision", "decision_index"
            definitions[name] = "Total sampled rows from this generation round during the real decision, including repeat draws."
        elif name.startswith("togo_"):
            phase, axis = "post_update_togo_probe", "actor_updates"
            unit = "value"
        elif name in probe_names:
            phase, axis = "post_update_fixed_probe", "round_index"
        elif name.startswith("actor_snapshot_"):
            phase, axis = "initial_or_post_update_snapshot", "actor_updates"
        elif name.startswith("collection_"):
            phase, axis = "post_collection", "round_index"
        elif name.startswith("tdambi_calibration_"):
            phase, axis = "before_first_optimizer_update", "round_index"
        elif name == "tdambi_entropy_coef":
            phase, axis = "initial", "round_index"
        elif name == "alpha":
            phase, axis = "initial_or_post_update_probe", "round_index"
        elif name.startswith("temperature_") or name == "alpha_used":
            phase, axis = "pre_update_minibatch", "temperature_updates"
        elif name.startswith(("actor_", "explorer_actor_")) or name in {"outer_policy_kl", "outer_action_l2"}:
            phase, axis = "pre_update_minibatch", "actor_updates"
        else:
            phase, axis = "pre_update_minibatch", "critic_updates"
        if "_replay_round_age_" in name or name.endswith("_replay_window_rounds"):
            unit = "collection_rounds"
        elif name.startswith("actor_horizon_") and "entropy_" in name:
            unit = "configured_entropy_statistic"
        elif name.endswith("seconds"):
            unit = "seconds"
        elif name.endswith(("_fraction", "_rate")):
            unit = "fraction"
        elif name.endswith(("_steps", "_transitions", "_count", "_evaluations", "_bytes")):
            unit = "count"
        elif name in {"actor_scaled_entropy", "explorer_actor_scaled_entropy"}:
            unit = "scaled_entropy_statistic"
        elif name in {"actor_entropy_bonus", "explorer_actor_entropy_bonus"}:
            unit = "objective"
        elif "kl" in name or name in {"actor_entropy", "explorer_actor_entropy"}:
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
            "definition": (horizon_metric_description(name)
                           or definitions.get(name, f"Raw inner optimizer metric {name}.")),
            "unit": unit, "sampling_phase": phase, "preferred_axis": axis,
        }
    return result


class InnerActionTrace:
    """Single-use recorder populated by ``predict(..., trace=recorder)``.

    ``events`` contains only host dictionaries after the action returns. One
    recorder owns one action; reuse is rejected to prevent accidental root mixing.
    """

    def __init__(self, *, probes=False, probe_seed=0, probe_rollouts=8, probe_horizon=3,
                 probe_mode="legacy", capture_actors=False, actor_rounds=None):
        if not isinstance(probes, bool):
            raise TypeError("probes must be bool.")
        if not isinstance(capture_actors, bool):
            raise TypeError("capture_actors must be bool.")
        if actor_rounds is not None:
            actor_rounds = tuple(actor_rounds)
            if any(isinstance(r, bool) or not isinstance(r, Integral) or r < 0
                   for r in actor_rounds):
                raise ValueError("actor_rounds must contain nonnegative integer round indices.")
            if len(set(actor_rounds)) != len(actor_rounds):
                raise ValueError("actor_rounds must not contain duplicate round indices.")
        if probe_mode not in {"legacy", "outer_tail"}:
            raise ValueError("probe_mode must be 'legacy' or 'outer_tail'.")
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
        self.probe_mode = probe_mode
        self.capture_actors = capture_actors
        self.actor_rounds = None if actor_rounds is None else frozenset(actor_rounds)
        self.actor_snapshots = []
        self.events = []
        self.round_index = 0
        self._started = False
        self._materialized = False
        self._noise = None
        self._outer_probe = None
        self._outer_stats = None
        self._outer_q = None
        self._alpha = None
        self._togo_initial = None
        self._togo_pair_indices = None
        self._probe_timings = []
        self.value_routing = None

    @torch.no_grad()
    def capture_actor(self, engine, policy, *, inner=True):
        """Copy the actual actor at an initialization/completed-round boundary."""
        if not self.capture_actors or (
            self.actor_rounds is not None and self.round_index not in self.actor_rounds
        ):
            return
        if engine.device.type == "cuda":
            # The CPU export already requires a stream barrier. Put earlier
            # learner work outside the diagnostic wall-time interval.
            torch.cuda.current_stream(engine.device).synchronize()
        started = time.perf_counter()
        frozen = deepcopy(policy).to("cpu").eval().requires_grad_(False)
        output = io.BytesIO()
        torch.save(frozen, output)
        state = engine.state
        snapshot = FrozenActorSnapshot(
            round_index=self.round_index, actor_updates=int(state.actor_steps),
            critic_updates=int(state.critic_steps),
            temperature_updates=int(state.temperature_steps), inner=inner,
            bounds=tuple((self._policy_bounds(engine.cfg) if inner else
                          getattr(engine, "_actor_options", {})).items()),
            payload=output.getvalue(),
            horizon_conditioning_horizon=getattr(policy, "horizon_conditioning_horizon", None),
        )
        self.actor_snapshots.append(snapshot)
        metadata = ({"horizon_conditioning_horizon": snapshot.horizon_conditioning_horizon}
                    if snapshot.horizon_conditioning_horizon is not None else {})
        self.record("actor_snapshot", state, {
            "actor_snapshot_seconds": time.perf_counter() - started,
            "actor_snapshot_bytes": len(snapshot.payload),
        }, actor_sha256=snapshot.sha256, measurement="initial_or_post_update_snapshot", **metadata)

    def begin(self, *, value_routing=None):
        if self._started:
            raise ValueError("An InnerActionTrace can record only one action.")
        self._started = True
        self.value_routing = value_routing

    def abort(self):
        """Discard incomplete measurements and release device references on error."""
        self.events.clear()
        self.actor_snapshots.clear()
        self._probe_timings.clear()
        self._noise = self._outer_probe = self._outer_stats = self._outer_q = self._alpha = None
        self._togo_initial = self._togo_pair_indices = None
        self._materialized = True

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
            **({"value_routing": dict(self.value_routing)} if self.value_routing is not None else {}),
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
        self._togo_initial = self._togo_pair_indices = None

    @torch.no_grad()
    def _togo_trajectory(self, engine, root_z, policy, *, inner):
        """H steps of this actor, followed by the same tail as finite-horizon SAC."""
        result = evaluate_outer_tail(
            engine, root_z, policy, self._noise,
            pair_indices=self._togo_pair_indices,
            policy_bounds=self._policy_bounds(engine.cfg) if inner else getattr(engine, "_actor_options", None),
        )
        return result["reward"], result["bootstrap"], result["total"]

    @torch.no_grad()
    def _togo_probe(self, engine, root_z, policy, *, inner):
        model, cfg = engine.model, engine.cfg
        if self.probe_horizon != int(cfg.inner_rollout_horizon):
            raise ValueError("Outer-tail probe_horizon must equal inner_rollout_horizon.")
        modes = tuple((module, bool(module.training))
                      for root in (model, policy) for module in root.modules())
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
                if (getattr(cfg, "critic_value_mode", "single") == "return_entropy"
                        or cfg.mppi_terminal_q_reduction.endswith("_pair")):
                    self._togo_pair_indices = model.q_backend.sample_pair_indices(
                        root_z.device, generator=generator)
                self._outer_probe = self._togo_trajectory(engine, root_z, getattr(engine, "_actor_base", model._pi), inner=False)
                model_steps *= 2
            rewards, tail, scores = self._togo_trajectory(engine, root_z, policy, inner=inner)
            if self._togo_initial is None:
                self._togo_initial = tuple(value.clone() for value in (rewards, tail, scores))
            trajectories = model_steps // self.probe_horizon
            metrics = {
                "togo_return_mean": scores.mean(),
                "togo_return_std": scores.std(unbiased=False),
                "togo_reward_mean": rewards.mean(),
                "togo_bootstrap_mean": tail.mean(),
                "probe_model_steps": model_steps,
                "probe_reward_evaluations": model_steps,
                "probe_policy_evaluations": trajectories * (self.probe_horizon + 1),
                "probe_q_evaluations": trajectories,
            }
            for reference, values in (("initial", self._togo_initial),
                                      ("outer", self._outer_probe)):
                for component, current, fixed in zip(
                    ("reward", "bootstrap", "return"), (rewards, tail, scores), values
                ):
                    metrics[f"togo_{component}_{reference}_mean"] = fixed.mean()
                    metrics[f"togo_{component}_gain_vs_{reference}"] = (current - fixed).mean()
                metrics[f"togo_return_gain_vs_{reference}_std"] = (scores - values[2]).std(unbiased=False)
            self.record("probe", engine.state, metrics, measurement="post_update_togo_probe")
            event = self.events[-1]
        finally:
            for module, was_training in modes:
                module.training = was_training
            if event is not None:
                self._probe_timings.append((event, started, engine._timer_start()))

    @torch.no_grad()
    def _trajectory(self, engine, root_z, policy, *, inner):
        model, cfg = engine.model, engine.cfg
        z = root_z.expand(self.probe_rollouts, -1).clone()
        reward_sum = z.new_zeros(self.probe_rollouts, 1)
        entropy_bonus = torch.zeros_like(reward_sum)
        continuation = torch.ones_like(reward_sum)
        discount = 1.0
        bounds = self._policy_bounds(cfg) if inner else getattr(engine, "_actor_options", {})
        for step in range(self.probe_horizon):
            action, info = model.pi(z, policy=policy, noise=self._noise[step], **bounds)
            joint = model.joint_input(z, action)
            prediction = model.reward_from_joint(joint)
            # External analytic probe models retain the historical scalar seam.
            reward = (model.decode_reward(prediction) if hasattr(model, "decode_reward")
                      else td_math.two_hot_inv(prediction, cfg))
            reward_sum += discount * continuation * reward
            entropy_bonus -= discount * continuation * self._alpha * info["log_prob"]
            z = model.next_from_joint(joint)
            if cfg.episodic:
                alive = model.termination(z) <= float(cfg.inner_termination_threshold)
                continuation *= alive.to(z.dtype)
            discount *= float(engine.agent.discount)
        if getattr(cfg, "aux_return_mode", "off") != "off":
            action, _ = model.pi(
                z, policy=engine._horizon_actor, noise=self._noise[-1],
                **engine._horizon_actor_options,
            )
            tail = evaluate_frozen_outer_q(
                model, z, action, critic=engine._horizon_critic,
                reduction=cfg.mppi_terminal_q_reduction,
                pair_indices=self._togo_pair_indices,
            )
        elif getattr(cfg, "critic_value_mode", "single") == "return_entropy":
            action, _ = model.pi(z, noise=self._noise[-1])
            tail = model.Q(z, action, reduction="min_pair", projection="return",
                           pair_indices=self._togo_pair_indices, trusted_pair_indices=True)
        else:
            action, info = model.pi(z, policy=policy, noise=self._noise[-1], **bounds)
            tail = model.Q(z, action, target=True, reduction="mean_all")
            entropy_bonus -= discount * continuation * self._alpha * info["log_prob"]
        terminal_q = torch.where(continuation.bool(), discount * tail, 0.0) if getattr(cfg, "aux_return_mode", "off") != "off" else discount * continuation * tail
        score = reward_sum + terminal_q
        result = {
            "discounted_reward": reward_sum.mean(),
            "discounted_terminal_q": terminal_q.mean(),
            "predicted_score": score.mean(),
        }
        if cfg.inner_operator != "tdambi":
            result.update(
                fixed_alpha_entropy_bonus=entropy_bonus.mean(),
                fixed_alpha_soft_score=(score + entropy_bonus).mean(),
            )
        return result

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
        conditioned_horizon = getattr(policy, "horizon_conditioning_horizon", None)
        conditioning_active = (
            conditioned_horizon is not None
            or getattr(engine.cfg, "inner_horizon_conditioning", "none") != "none"
        )
        if conditioning_active:
            if self.probe_mode != "outer_tail":
                raise ValueError("Horizon-conditioned actors require probe_mode='outer_tail'; legacy probes are unsupported.")
            expected_horizon = getattr(engine.cfg, "inner_rollout_horizon", conditioned_horizon)
            if (self.probe_horizon != expected_horizon
                    or conditioned_horizon is not None and self.probe_horizon != conditioned_horizon):
                raise ValueError("Conditioned actor probe horizon must match inner_rollout_horizon.")
        if self.probe_mode == "outer_tail":
            return self._togo_probe(engine, root_z, policy, inner=inner)
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
                self._alpha = getattr(engine, "_actor_owner", engine.agent).alpha.detach().clone()
                if getattr(cfg, "critic_value_mode", "single") == "return_entropy":
                    if engine.agent.actor_loss_scale_enabled:
                        self._alpha = self._alpha * engine.agent.actor_loss_scale.detach()
                    self._togo_pair_indices = model.q_backend.sample_pair_indices(
                        root_z.device, generator=generator)
                routed = getattr(cfg, "aux_return_mode", "off") != "off"
                if routed:
                    if engine._critic_owner.actor_loss_scale_enabled:
                        self._alpha = self._alpha * engine._critic_owner.actor_loss_scale.detach()
                    if cfg.mppi_terminal_q_reduction.endswith("_pair"):
                        self._togo_pair_indices = model.q_backend.sample_pair_indices(
                            root_z.device, generator=generator)
                self._outer_stats = model.policy_stats(
                    root_z, policy=getattr(engine, "_actor_base", model._pi),
                    **getattr(engine, "_actor_options", {}),
                )
                self._outer_q = model.Q(
                    root_z, self._outer_stats["mean"], reduction="mean_all",
                    **({"qs": engine._horizon_critic} if routed else {"target": True}),
                    **({"projection": "return"} if getattr(cfg, "critic_value_mode", "single") == "return_entropy" else {}),
                )
                self._outer_probe = self._trajectory(engine, root_z, getattr(engine, "_actor_base", model._pi), inner=False)
                model_steps *= 2
            bounds = self._policy_bounds(cfg) if inner else getattr(engine, "_actor_options", {})
            stats = model.policy_stats(root_z, policy=policy, **bounds)
            q = model.Q(root_z, stats["mean"], reduction="mean_all",
                        **({"qs": engine._horizon_critic} if getattr(cfg, "aux_return_mode", "off") != "off" else {"target": True}),
                        **({"projection": "return"} if getattr(cfg, "critic_value_mode", "single") == "return_entropy" else {}))
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
            if cfg.inner_operator == "tdambi":
                metrics.pop("fixed_evaluator_alpha")
                metrics.pop("alpha")
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
