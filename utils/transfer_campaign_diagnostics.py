"""Sampled, observation-only diagnostics for full-episode transfer campaigns.

The prior continuation and its model-return labels are shared across all stages
at a root. These are model references, not real-return ground truth. Stationary
fits operate on disposable clones, not on the running inner learner.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import time

import numpy as np
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace
from utils.ambi_benchmark import solver_seed
from utils.transfer_diagnostic_metrics import (
    action_value_metrics, fit_stationary_targets, forced_action_mc_returns,
)
from utils.transfer_diagnostics import Reference, evaluating, json_value


FAMILIES = ("fixed_target", "action_gradient", "policy", "features", "stationary")
# Only elapsed-time observations are exempt from exact smoke equality. These
# names are the complete explicit _timer_stop keys in InnerImprovementEngine;
# scalar losses, counts, state and RNG remain strict even with all_scalars.
OBSERVATIONAL_TIMING_METRICS = frozenset({
    "inner_action_seconds", "inner_setup_seconds", "inner_rollout_seconds",
    "inner_update_seconds", "inner_execution_seconds", "inner_diagnostic_seconds",
    "inner_selector_seconds", "inner_mppi_seconds",
    "inner_param_noise_calibration_seconds", "inner_tdambi_calibration_seconds",
})
DEFAULTS = dict(enabled=True, decisions=[0, 1, 25, 100, 250, 499],
    stationary_decisions=[25, 250], mc_rollouts=8, action_count=8,
    state_count=32, fit_steps=4)


def diagnostic_settings(settings):
    if settings is None or settings == {}:
        return None
    if not isinstance(settings, dict) or set(settings) - set(DEFAULTS):
        raise ValueError("Unknown campaign diagnostic settings.")
    values = {**DEFAULTS, **settings}
    if values["enabled"] is not True:
        raise ValueError("Omit diagnostics to disable them; enabled must be true.")
    for key in ("decisions", "stationary_decisions"):
        rows = values[key]
        if not isinstance(rows, list) or not rows or len(set(rows)) != len(rows) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in rows):
            raise ValueError(f"{key} requires unique nonnegative decisions.")
    if not set(values["stationary_decisions"]) <= set(values["decisions"]):
        raise ValueError("Stationary decisions must be sampled diagnostic decisions.")
    for name, minimum in (("mc_rollouts", 2), ("action_count", 4), ("state_count", 4), ("fit_steps", 1)):
        if isinstance(values[name], bool) or not isinstance(values[name], int) or values[name] < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if values["state_count"] % 2:
        raise ValueError("state_count must be even for the fixed train/heldout split.")
    return deepcopy(values)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def relative_feature_metrics(features):
    """Scale-relative Mish-compatible descriptors with the sample-rank ceiling."""
    values = features.detach().flatten(1).cpu().double()
    if len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("Feature probes require >=2 finite samples.")
    centered = values - values.mean(0)
    spectrum = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0)
    energy = spectrum.sum()
    ceiling = min(len(values) - 1, values.shape[1])
    positive = spectrum[spectrum > energy * 1e-12]
    probabilities = positive / positive.sum() if positive.numel() else positive
    effective = float(torch.exp(-(probabilities * probabilities.log()).sum())) if positive.numel() else 0.
    activity = values.abs().mean(0)
    deviation = values.std(0, unbiased=False)
    return dict(effective_rank=effective, rank_ceiling=ceiling,
        effective_rank_fraction=effective / ceiling,
        relative_low_activity_fraction=float((activity <= .1 * activity.mean()).double().mean()),
        relative_low_variance_fraction=float((deviation <= .1 * deviation.mean()).double().mean()),
        samples=len(values), width=values.shape[1], relative_threshold=.1)


class CampaignTrace(InnerActionTrace):
    """Only two weight snapshots; no update trace, replay hashes or serialization."""
    def __init__(self):
        super().__init__(capture_learners=True, learner_rounds=())
        self.states = {}
        self.snapshot_seconds = 0.

    def record(self, phase, state, metrics=None, **metadata):
        # The engine augments the initial event in place. Other optimizer events
        # are deliberately omitted; their training summaries remain available.
        if phase == "initial":
            super().record(phase, state, metrics, **metadata)

    @staticmethod
    def replay_sha256(replay):
        return "not-captured-sampled-campaign"

    @torch.no_grad()
    def capture_learner(self, engine, *, stage, actor_loss_scale=None):
        if stage not in {"initial", "pre_reset"}:
            return
        _synchronize(engine.device)
        started = time.perf_counter()
        self.states["initial" if stage == "initial" else "final"] = {
            name: {key: value.detach().cpu().clone() for key, value in getattr(engine.state, name).state_dict().items()}
            for name in ("actor", "critic")}
        _synchronize(engine.device)
        self.snapshot_seconds += time.perf_counter() - started


def _module(base, state=None):
    result = deepcopy(base).eval().requires_grad_(False)
    if state is not None:
        result.load_state_dict(state, strict=True)
    return result


class CampaignDiagnostics:
    def __init__(self, settings, *, episode_seed, controller_seed, smoke=False):
        self.settings = diagnostic_settings(settings)
        if self.settings is None:
            raise ValueError("CampaignDiagnostics requires enabled settings.")
        self.seed = solver_seed(controller_seed, "campaign-diagnostics", episode_seed)
        self.decisions = set(self.settings["decisions"])
        self.stationary_decisions = set(self.settings["stationary_decisions"])
        if smoke:
            self.decisions.update((0, 1))
            self.stationary_decisions.add(1)
        self.rows = []
        self.decision_seconds = self.in_prediction_seconds = 0.

    def begin(self, wrapped, observation, decision, donor):
        self.decision_seconds = self.in_prediction_seconds = 0.
        if decision not in self.decisions:
            return None
        started = time.perf_counter()
        self.wrapped, self.decision = wrapped, int(decision)
        self.observation = np.array(observation, copy=True)
        # Exported donor owns its storage; the controller replaces, not mutates,
        # this dictionary after prediction. No extra large copy is necessary.
        self.donor = donor
        trace = CampaignTrace()
        self.decision_seconds = time.perf_counter() - started
        return trace

    @torch.no_grad()
    def _returns(self, roots, actions, continuation, seed):
        r, h, count = self.reference, self.horizon, self.settings["mc_rollouts"]
        # All first actions share a continuation-noise stream. Probe tensors use
        # named local generators and never consume learner or global RNG.
        noise = r.noise((h - 1, count, 1, r.cfg.action_dim), seed).expand(-1, -1, len(actions), -1)
        tail = r.noise((h, count, 1, r.cfg.action_dim), solver_seed(seed, "tail")).expand(-1, -1, len(actions), -1)
        with evaluating(r.model, continuation):
            return forced_action_mc_returns(roots, actions, horizons=(h,),
                transition=r.transition, policy=r.policy(continuation), tail=r.tail,
                discount=r.discount, policy_noise=noise, tail_noise=tail,
                critic_target="reward_only", entropy_coefficient=0.)[h]

    @torch.no_grad()
    def _bank(self, prior, donor):
        r, n, k = self.reference, self.settings["state_count"], self.settings["action_count"]
        root = r.encode(self.observation[None])
        with evaluating(r.model, prior, donor):
            actions, _ = r.policy(prior)(root.expand(n - 1, -1), r.noise((n - 1, r.cfg.action_dim),
                solver_seed(self.root_seed, "state-bank")))
            _, successors, _ = r.transition(root.expand(n - 1, -1), actions)
            # Actual root plus independently sampled one-step prior successors.
            states = torch.cat((root, successors))
            prior_stats = r.model.policy_stats(states, policy=prior, **r.bounds)
            donor_stats = r.model.policy_stats(states, policy=donor, **r.bounds)
            samples = []
            for index in range(k - 2):
                actor = prior if index % 2 == 0 else donor
                samples.append(r.policy(actor)(states, r.noise((n, r.cfg.action_dim),
                    solver_seed(self.root_seed, "action", index // 2)))[0])
            actions = torch.stack((prior_stats["mean"], donor_stats["mean"], *samples), 1)
        roots = states[:, None, :].expand(-1, k, -1).reshape(n * k, -1)
        samples = self._returns(roots, actions.flatten(0, 1), prior, self.root_seed).reshape(-1, n, k)
        labels = samples.mean(0)
        return states.detach(), actions.detach(), samples.detach(), labels.detach(), prior_stats

    def _stage(self, actor, critic):
        r, states, actions, labels = self.reference, self.states, self.actions, self.labels
        n, k = labels.shape
        roots = states[:, None, :].expand(-1, k, -1).reshape(n * k, -1)
        with evaluating(r.model, actor, critic):
            with torch.no_grad():
                q = r.q(critic, roots, actions.flatten(0, 1)).reshape(n, k)
                stats = r.model.policy_stats(states, policy=actor, **r.bounds)
                feature_actor = relative_feature_metrics(actor[:-1](states))
                inputs = torch.cat((states, actions[:, 0]), -1)
                feature_critic = [relative_feature_metrics(head[:-1](inputs)) for head in critic.modules_list]
                policy_kl = float(r.engine._gaussian_kl(stats, self.prior_stats).mean())
                mean_action_l2 = float(torch.linalg.vector_norm(stats["mean"] - self.prior_stats["mean"], dim=-1).mean())
                mean_samples = self._returns(states, stats["mean"], self.prior, self.root_seed)
                current_samples = self._returns(states[:1], stats["mean"][:1], actor, self.root_seed)
                root_check = json_value(action_value_metrics(q[0], self.samples[:, 0]))
                error = q - labels
                chosen = q.argmax(1)
                regret = labels.max(1).values - labels.gather(1, chosen[:, None]).flatten()
            # Explicit input gradient, never a backward pass into model weights.
            with torch.enable_grad():
                u = torch.atanh(stats["mean"].detach().clamp(-1 + 1e-6, 1 - 1e-6)).requires_grad_(True)
                gradient, = torch.autograd.grad(r.q(critic, states, torch.tanh(u)).sum(), u)
            norms = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True)
            direction = gradient / norms.clamp_min(1e-12)
            plus = self._returns(states, torch.tanh(u.detach() + .05 * direction), self.prior, self.root_seed)
            minus = self._returns(states, torch.tanh(u.detach() - .05 * direction), self.prior, self.root_seed)
            differences = plus - minus
        return dict(fixed_target=dict(rmse=float(error.square().mean().sqrt()), bias=float(error.mean()),
                centered_rmse=float((error - error.mean(1, keepdim=True)).square().mean().sqrt()),
                action_relative_rmse=float(((q - q[:, :1]) - (labels - labels[:, :1])).square().mean().sqrt()),
                empirical_selection_regret=float(regret.mean()), root=root_check),
            action_gradient=dict(norm=float(norms.mean()), epsilon=.05,
                paired_return_gain=float(differences.mean()),
                conditional_mc_se=float(differences.mean(1).std(unbiased=True) / math.sqrt(len(differences))),
                positive_state_fraction=float((differences.mean(0) > 0).float().mean()),
                zero_gradient_fraction=float((norms <= 1e-12).float().mean())),
            policy=dict(kl_vs_prior=policy_kl, mean_action_l2_vs_prior=mean_action_l2,
                std_mean=float(stats["log_std"].exp().mean()),
                fixed_continuation_first_action_gain=float((mean_samples - self.samples[:, :, 0]).mean()),
                root_current_policy_model_return=float(current_samples.mean())),
            features=dict(actor=feature_actor, critic_heads=feature_critic))

    def _stationary(self, initial):
        r, states, actions, labels = self.reference, self.states, self.actions, self.labels
        n, k = labels.shape
        # Even/odd indices interleave roots/successors. Held-out states are never
        # used by these diagnostic optimizer steps; this is not a replay holdout.
        train, heldout = torch.arange(0, n, 2, device=states.device), torch.arange(1, n, 2, device=states.device)
        target_actions = actions[torch.arange(n, device=states.device), labels.argmax(1)]
        critic_x = torch.cat((states[:, None, :].expand(-1, k, -1), actions), -1)
        curves = {}
        for component in ("actor", "critic"):
            if component == "critic":
                x, y = critic_x, labels[..., None]
                def predict(net, values):
                    latent = r.cfg.latent_dim
                    return r.q_heads(net, values[:, :latent], values[:, latent:])[..., 0].T
                x_train, x_test = x[train].flatten(0, 1), x[heldout].flatten(0, 1)
                y_train, y_test = y[train].flatten(0, 1), y[heldout].flatten(0, 1)
            else:
                def predict(net, values):
                    return r.model.policy_stats(values, policy=net, **r.bounds)["mean"]
                x_train, x_test, y_train, y_test = states[train], states[heldout], target_actions[train], target_actions[heldout]
            generator = torch.Generator().manual_seed(solver_seed(self.root_seed, "batches", component))
            batches = torch.randint(len(x_train), (self.settings["fit_steps"], min(32, len(x_train))), generator=generator)
            for name, state in (("prior", None), ("initial", initial[component])):
                module = _module(getattr(r.engine, f"_{component}_base"), state)
                fitted = fit_stationary_targets(module, predict, x_train, y_train, x_test, y_test,
                    batch_indices=batches, learning_rate=float(getattr(r.cfg, f"inner_{component}_lr")),
                    seed=solver_seed(self.root_seed, "fit", component),
                    trainable_selector=lambda name, parameter: True)
                curves[f"{component}_{name}"] = fitted["curve"]
        return dict(curves=curves, steps=self.settings["fit_steps"], train_states=len(train), heldout_states=len(heldout),
            semantics="Fresh Adam on disposable clones; fixed prior-continuation decoded Q labels; actor mean imitation of bank winner. No covariance-fit claim.")

    def finish(self, trace):
        if trace is None:
            return None
        device = self.wrapped.agent.device
        _synchronize(device)
        started = time.perf_counter()
        self.in_prediction_seconds = trace.snapshot_seconds
        if set(trace.states) != {"initial", "final"}:
            raise RuntimeError("Missing sampled initial/final learner snapshots.")
        r = self.reference = Reference(self.wrapped, rollouts=self.settings["mc_rollouts"])
        if r.cfg.inner_sac_critic_target != "reward_only" or r.cfg.inner_terminal_entropy != "none":
            raise ValueError("Campaign sampled diagnostics currently require reward-only return/return semantics.")
        self.horizon = int(r.cfg.inner_rollout_horizon)
        self.root_seed = solver_seed(self.seed, "decision", self.decision)
        self.prior = _module(r.engine._actor_base)
        donor_actor = _module(r.engine._actor_base, None if self.donor is None else self.donor["modules"]["actor"])
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), evaluating(r.model):
            self.states, self.actions, self.samples, self.labels, self.prior_stats = self._bank(self.prior, donor_actor)
            stages = {"prior": None}
            if self.donor is not None:
                stages["donor"] = self.donor["modules"]
            stages.update(trace.states)
            results = {}
            for name, state in stages.items():
                actor = _module(r.engine._actor_base, None if state is None else state["actor"])
                critic = _module(r.engine._critic_base, None if state is None else state["critic"])
                results[name] = self._stage(actor, critic)
            stationary = self._stationary(trace.states["initial"]) if self.decision in self.stationary_decisions else None
        summary = {}
        for stage, row in results.items():
            summary.update({f"{stage}_critic_rmse": row["fixed_target"]["rmse"],
                f"{stage}_critic_action_relative_rmse": row["fixed_target"]["action_relative_rmse"],
                f"{stage}_critic_selection_regret": row["fixed_target"]["empirical_selection_regret"],
                f"{stage}_policy_kl": row["policy"]["kl_vs_prior"],
                f"{stage}_model_first_action_gain_vs_prior": row["policy"]["fixed_continuation_first_action_gain"],
                f"{stage}_root_model_return": row["policy"]["root_current_policy_model_return"],
                f"{stage}_action_gradient_gain": row["action_gradient"]["paired_return_gain"],
                f"{stage}_actor_feature_effective_rank_fraction": row["features"]["actor"]["effective_rank_fraction"],
                f"{stage}_actor_low_activity_fraction": row["features"]["actor"]["relative_low_activity_fraction"],
                f"{stage}_critic_feature_effective_rank_fraction": float(np.mean([head["effective_rank_fraction"] for head in row["features"]["critic_heads"]]))})
        if stationary is not None:
            for name, curve in stationary["curves"].items():
                summary[f"{name}_stationary_start_heldout_rmse"] = curve[0]["heldout_rmse"]
                summary[f"{name}_stationary_heldout_improvement"] = curve[0]["heldout_rmse"] - curve[-1]["heldout_rmse"]
                summary[f"{name}_stationary_final_grad_norm"] = curve[-1]["gradient_norm"]
        record = json_value(dict(decision=self.decision, donor_available=self.donor is not None,
            stages=results, stationary=stationary, summary=summary,
            target_sha256=hashlib.sha256(self.labels.cpu().numpy().tobytes()).hexdigest(),
            reference=dict(state_count=len(self.states), action_count=self.settings["action_count"],
                mc_rollouts=self.settings["mc_rollouts"], horizon=self.horizon,
                continuation="frozen_prior", tail="frozen_horizon_actor_and_critic", target="reward_only",
                semantics="Identical fixed bank/labels across stages; model reference, not ground truth. Rank limited by state count.")))
        _synchronize(device)
        self.decision_seconds += self.in_prediction_seconds + time.perf_counter() - started
        record["diagnostic_seconds"] = self.decision_seconds
        self.rows.append(record)
        trace.states.clear()
        self.donor = None
        return record

    def coverage(self, steps):
        expected = sorted(decision for decision in self.decisions if decision < steps)
        actual = [row["decision"] for row in self.rows]
        expected_stationary = sorted(set(expected) & self.stationary_decisions)
        completed_stationary = [row["decision"] for row in self.rows if row["stationary"] is not None]
        if expected != actual or expected_stationary != completed_stationary:
            raise RuntimeError("Missing campaign diagnostic decisions or stationary probes.")
        for row in self.rows:
            required = {"prior", "initial", "final"} | ({"donor"} if row["donor_available"] else set())
            if not required <= set(row["stages"]) or not row["summary"] or not all(math.isfinite(v) for v in row["summary"].values()):
                raise RuntimeError("Missing or nonfinite campaign diagnostic stage/family metrics.")
        keys = sorted({key for row in self.rows for key in row["summary"]})
        return dict(enabled=True, complete=True, samples=len(self.rows), expected_decisions=expected,
            completed_decisions=actual, expected_stationary_decisions=expected_stationary,
            completed_stationary_decisions=completed_stationary,
            stages=sorted({stage for row in self.rows for stage in row["stages"]}), families=list(FAMILIES),
            summary={key: float(np.mean([row["summary"][key] for row in self.rows if key in row["summary"]])) for key in keys})


def verify_episode_diagnostics(episode, settings, *, smoke=False):
    """Fail closed on missing finite measurements; suitable for the smoke gate."""
    settings = diagnostic_settings(settings)
    if settings is None:
        return
    decisions = set(settings["decisions"]) | ({0, 1} if smoke else set())
    stationary = set(settings["stationary_decisions"]) | ({1} if smoke else set())
    expected = sorted(value for value in decisions if value < episode["steps"])
    coverage = episode.get("diagnostics", {})
    if not coverage.get("complete") or coverage.get("completed_decisions") != expected or coverage.get("samples") != len(expected):
        raise RuntimeError("Episode diagnostic coverage is missing or incomplete.")
    if coverage.get("completed_stationary_decisions") != sorted(set(expected) & stationary):
        raise RuntimeError("Episode stationary diagnostic coverage is incomplete.")
    if set(coverage.get("families", ())) != set(FAMILIES):
        raise RuntimeError("Episode diagnostic measurement families are incomplete.")
    required = {f"{stage}_{metric}" for stage in ("prior", "initial", "final") for metric in
        ("critic_rmse", "policy_kl", "action_gradient_gain", "actor_feature_effective_rank_fraction")}
    summary = coverage.get("summary", {})
    if not required <= set(summary) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in summary.values()):
        raise RuntimeError("Episode diagnostic numeric summary is missing or nonfinite.")
    if not math.isfinite(episode.get("diagnostic_seconds", float("nan"))) or episode["diagnostic_seconds"] <= 0:
        raise RuntimeError("Episode diagnostic timing is missing.")


def verify_observational_isolation(reference_rows, observed_rows, reference_state, observed_state):
    """Exact smoke gate: diagnostics cannot alter actions, rewards or learner RNG."""
    if len(reference_rows) != len(observed_rows):
        raise RuntimeError("Diagnostic isolation changed the episode length.")
    for before, after in zip(reference_rows, observed_rows):
        for name in ("action", "reward", "terminated", "truncated"):
            if before[name] != after[name]:
                raise RuntimeError(f"Diagnostic isolation changed {name} at decision {before['decision']}.")
        # Timing values necessarily differ between independent smoke forks and
        # include diagnostic snapshot work. Filter views without altering the
        # saved rows, so all observed timings remain available for analysis.
        before_metrics = {key: value for key, value in before["metrics"].items()
                          if key not in OBSERVATIONAL_TIMING_METRICS}
        after_metrics = {key: value for key, value in after["metrics"].items()
                         if key not in OBSERVATIONAL_TIMING_METRICS}
        if before_metrics != after_metrics:
            raise RuntimeError(f"Diagnostic isolation changed metrics at decision {before['decision']}.")
    def compare(left, right, path):
        if torch.is_tensor(left):
            same = torch.is_tensor(right) and left.shape == right.shape and torch.equal(left, right)
        elif isinstance(left, dict):
            same = isinstance(right, dict) and left.keys() == right.keys()
            if same:
                for key in left:
                    compare(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            same = type(left) is type(right) and len(left) == len(right)
            if same:
                for index, (a, b) in enumerate(zip(left, right)):
                    compare(a, b, f"{path}.{index}")
        else:
            same = left == right
        if not same:
            raise RuntimeError(f"Diagnostic isolation changed {path}.")
    compare(reference_state, observed_state, "learner_state")
