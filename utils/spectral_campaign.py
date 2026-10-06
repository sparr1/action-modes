"""Independent held-out spectral measurements composed with campaign probes."""
from copy import deepcopy
import math
import time

import numpy as np

from utils.transfer_campaign_diagnostics import CampaignDiagnostics, CampaignTrace, _synchronize


DEFAULTS = dict(enabled=True, decisions=[0, 1, 25, 100, 250, 499],
    ranks=[1, 2, 4, 8, 16, 32, 64, 128], state_count=32, action_count=8, mc_rollouts=8)


def spectral_settings(settings):
    if settings is None or settings == {}:
        return None
    if not isinstance(settings, dict) or set(settings) - set(DEFAULTS):
        raise ValueError("Unknown spectral diagnostic settings.")
    result = {**DEFAULTS, **settings}
    if result["enabled"] is not True:
        raise ValueError("Omit spectral_diagnostics to disable them.")
    for key, minimum in (("decisions", 0), ("ranks", 1)):
        values = result[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values) or any(
                isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in values):
            raise ValueError(f"{key} must contain unique integers >= {minimum}.")
    validate_probe_settings({key: result[key] for key in ("state_count", "action_count", "mc_rollouts")})
    return deepcopy(result)


def validate_probe_settings(settings):
    values = {key: DEFAULTS[key] for key in ("state_count", "action_count", "mc_rollouts")}
    if settings is not None:
        if not isinstance(settings, dict) or set(settings) - set(values):
            raise ValueError("Unknown spectral selection probe settings.")
        values.update(settings)
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{key} must be an integer >= 2.")
    return values


class SpectralCampaignDiagnostics:
    """One snapshot pair, independent selection/evaluation banks, no learner writes."""
    needs_donor = True

    def __init__(self, settings, *, basic_settings=None, episode_seed, controller_seed, smoke=False):
        self.settings = spectral_settings(settings)
        if self.settings is None:
            raise ValueError("SpectralCampaignDiagnostics requires enabled settings.")
        self.basic = (CampaignDiagnostics(basic_settings, episode_seed=episode_seed,
            controller_seed=controller_seed, smoke=smoke) if basic_settings else None)
        self.episode_seed, self.controller_seed = episode_seed, controller_seed
        self.decisions = set(self.settings["decisions"]) | ({0, 1} if smoke else set())
        self.rows = []
        self.decision_seconds = self.in_prediction_seconds = 0.

    def begin(self, wrapped, observation, decision, donor):
        self.decision_seconds = self.in_prediction_seconds = 0.
        trace = None if self.basic is None else self.basic.begin(wrapped, observation, decision, donor)
        self.basic_requested = trace is not None
        self.spectral_requested = decision in self.decisions
        if self.spectral_requested:
            self.wrapped, self.observation = wrapped, np.array(observation, copy=True)
            self.decision, self.donor = int(decision), donor
            trace = trace or CampaignTrace()
        return trace

    def finish(self, trace):
        if trace is None:
            return None
        spectral = None
        spectral_seconds = 0.
        self.in_prediction_seconds = trace.snapshot_seconds
        if self.spectral_requested:
            from utils.spectral_transfer_probes import build_spectral_context, evaluate_spectral_handoff
            _synchronize(self.wrapped.agent.device)
            started = time.perf_counter()
            if set(trace.states) != {"initial", "final"}:
                raise RuntimeError("Spectral diagnostics require initial and post-J snapshots.")
            context = build_spectral_context(self.wrapped, self.observation,
                controller_seed=self.controller_seed, episode_seed=self.episode_seed,
                decision=self.decision, purpose="heldout", settings={key: self.settings[key]
                    for key in ("state_count", "action_count", "mc_rollouts")})
            spectral = evaluate_spectral_handoff(self.wrapped, donor=self.donor,
                initial_states=trace.states["initial"], final_states=trace.states["final"],
                context=context, ranks=tuple(self.settings["ranks"]))
            spectral.update(decision=self.decision, donor_available=self.donor is not None)
            if not spectral.get("summary") or not all(math.isfinite(v) for v in spectral["summary"].values()):
                raise RuntimeError("Missing or nonfinite spectral diagnostic summary.")
            self.rows.append(spectral)
            _synchronize(self.wrapped.agent.device)
            spectral_seconds = time.perf_counter() - started
        record = deepcopy(self.basic.finish(trace)) if self.basic_requested else dict(
            decision=self.decision, summary={})
        self.decision_seconds = spectral_seconds + (self.basic.decision_seconds if self.basic_requested
            else trace.snapshot_seconds)
        if spectral is not None:
            record["spectral"] = spectral
            record["summary"].update({f"spectral_{key}": value for key, value in spectral["summary"].items()})
        record["diagnostic_seconds"] = self.decision_seconds
        trace.states.clear()
        self.donor = None
        return record

    def coverage(self, steps):
        expected = sorted(decision for decision in self.decisions if decision < steps)
        actual = [row["decision"] for row in self.rows]
        if expected != actual:
            raise RuntimeError("Spectral diagnostic sample coverage is incomplete.")
        keys = sorted({key for row in self.rows for key in row["summary"]})
        summary, summary_counts = {}, {}
        for key in keys:
            # The first solve has meaningful objective losses but no donor
            # geometry. Do not dilute spectral rank/energy means with zeros.
            objective = (key.endswith("_loss") or "loss_gain" in key or "loss_improvement" in key)
            values = [row["summary"][key] for row in self.rows if key in row["summary"]
                      and (objective or row["donor_available"])]
            if values:
                summary[f"spectral_{key}"] = float(np.mean(values))
                summary_counts[f"spectral_{key}"] = len(values)
        result = (self.basic.coverage(steps) if self.basic else
            dict(enabled=True, complete=True, samples=len(self.rows), summary={}))
        result["summary"].update(summary)
        result["summary_counts"] = summary_counts
        result["spectral"] = dict(complete=True, expected_decisions=expected,
            completed_decisions=actual, samples=len(self.rows),
            donor_samples=sum(row["donor_available"] for row in self.rows),
            stages=sorted({stage for row in self.rows for stage in row["objectives"]}),
            reference="independent heldout prior bank; fixed-model proxies, not environment truth")
        return result


def verify_spectral_diagnostics(episode, settings, *, smoke=False):
    settings = spectral_settings(settings)
    if settings is None:
        return
    expected = sorted(v for v in set(settings["decisions"]) | ({0, 1} if smoke else set())
                      if v < episode["steps"])
    coverage = episode.get("diagnostics", {}).get("spectral", {})
    if not coverage.get("complete") or coverage.get("completed_decisions") != expected or coverage.get("samples") != len(expected):
        raise RuntimeError("Spectral diagnostic coverage is missing or incomplete.")
    if coverage.get("donor_samples") != sum(v > 0 for v in expected):
        raise RuntimeError("Spectral diagnostic donor coverage is incomplete.")
    summary = episode.get("diagnostics", {}).get("summary", {})
    required = {f"spectral_{stage}_{component}_loss" for component in ("actor", "critic")
                for stage in ("prior", "initial", "final")}
    if coverage.get("donor_samples", 0):
        for component in ("actor", "critic"):
            required.update(f"spectral_{component}_{metric}" for metric in (
                "donor_squared_norm", "initial_squared_norm", "transferred_energy_ratio",
                "parameter_residual_energy_ratio", "donor_initial_cosine",
                "initial_first_order_benefit", "donor_first_order_benefit", "prior_input_output_mse",
                "mean_stable_rank", "mean_effective_rank", "mean_energy_effective_rank",
                "mean_rank_50", "mean_rank_90", "mean_rank_95", "mean_rank_99",
                "mean_positive_benefit_fraction", "mean_positive_benefit_energy_fraction"))
            required.update(f"spectral_{component}_mean_energy_at_rank_{rank}" for rank in settings["ranks"])
    if not required <= set(summary) or not all(isinstance(summary[k], (int, float))
            and math.isfinite(summary[k]) for k in required):
        raise RuntimeError("Spectral stage objective metrics are missing or nonfinite.")
    for key in ("spectral_probe_seconds", "spectral_filter_seconds", "donor_export_seconds"):
        if not math.isfinite(episode.get(key, float("nan"))) or episode[key] < 0:
            raise RuntimeError(f"Missing or invalid controller cost: {key}.")
