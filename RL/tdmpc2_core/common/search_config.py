"""Resolved semantic predicates for AMBI's opt-in finite search operators.

The legacy continuing-task inner loop predates the finite-horizon search
contract and deliberately does not consult this module.  New code should use
``resolve_inner_search_semantics`` instead of inferring target inventory from
``inner_bootstrap_source``: that field belongs exclusively to the legacy path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


INNER_Q_OBJECTIVES = frozenset({"legacy_continuing", "finite_horizon"})
INNER_CRITIC_HORIZON_MODES = frozenset(
    {"shared", "depth_conditioned", "stage_heads"}
)
INNER_RETURN_ESTIMATORS = frozenset(
    {"td0", "n_step", "lambda_return", "full_suffix", "retrace"}
)
INNER_SEARCH_REPLAY_RETENTIONS = frozenset({"round", "action"})
INNER_OFFPOLICY_MODES = frozenset(
    {"none", "uncorrected", "per_decision_is", "resimulate"}
)
INNER_SEARCH_BOOTSTRAP_CRITICS = frozenset(
    {"target", "frozen_target", "online", "none"}
)
INNER_TARGET_UPDATE_EVENTS = frozenset(
    {"optimizer_step", "round_end", "depth_stage", "none"}
)
INNER_DEPTH_UPDATE_ORDERS = frozenset({"mixed", "backward"})
INNER_LEAF_Q_SOURCES = frozenset({"outer_target", "outer_online"})


@dataclass(frozen=True)
class InnerSearchSemantics:
    """Normalized finite-search choices and their derived resource needs."""

    operator: str
    q_objective: str
    critic_horizon_mode: str
    return_estimator: str
    return_steps: int | None
    return_lambda: float | None
    leaf_q_source: str
    leaf_value_samples: int
    replay_retention: str
    offpolicy_mode: str
    bootstrap_critic: str
    configured_target_update_event: str
    depth_update_order: str

    @property
    def is_legacy(self) -> bool:
        return self.q_objective == "legacy_continuing" and self.operator != "vtrace"

    @property
    def is_vtrace(self) -> bool:
        return self.operator == "vtrace"

    @property
    def is_finite_q(self) -> bool:
        return self.q_objective == "finite_horizon" and self.operator == "sac"

    @property
    def is_search(self) -> bool:
        return self.is_finite_q or self.is_vtrace

    @property
    def uses_outer_leaf(self) -> bool:
        return self.is_search

    @property
    def uses_structured_replay(self) -> bool:
        return self.is_search

    @property
    def needs_behavior_log_prob(self) -> bool:
        return self.is_vtrace or self.return_estimator == "retrace" or (
            self.offpolicy_mode == "per_decision_is"
        )

    @property
    def uses_inner_target(self) -> bool:
        return self.is_search and self.bootstrap_critic in {
            "target",
            "frozen_target",
        }

    @property
    def creates_inner_target(self) -> bool:
        return self.uses_inner_target

    @property
    def updates_inner_target(self) -> bool:
        return self.uses_inner_target and self.bootstrap_critic == "target"

    @property
    def target_update_event(self) -> str:
        return (
            self.configured_target_update_event
            if self.updates_inner_target
            else "none"
        )

    @property
    def requires_multistep_labels(self) -> bool:
        return self.is_vtrace or self.return_estimator != "td0"

    def exact_spec(self, cfg: Any) -> dict[str, Any]:
        """Return all active search semantics used by exact resume checks."""

        if not self.is_search:
            return {}
        spec = {
            "operator": self.operator,
            "q_objective": self.q_objective,
            "rollout_horizon": int(cfg.inner_rollout_horizon),
            "critic_horizon_mode": self.critic_horizon_mode,
            "return_estimator": self.return_estimator,
            "return_steps": self.return_steps,
            "return_lambda": self.return_lambda,
            "leaf_q_source": self.leaf_q_source,
            "leaf_value_samples": self.leaf_value_samples,
            "replay_retention": self.replay_retention,
            "offpolicy_mode": self.offpolicy_mode,
            "bootstrap_critic": self.bootstrap_critic,
            "target_update_event": self.target_update_event,
            "depth_update_order": self.depth_update_order,
            "critic_target_tau": float(cfg.inner_critic_target_tau),
            "critic_target_update_interval": int(
                cfg.inner_critic_target_update_interval
            ),
            "inner_critic_target": str(cfg.inner_sac_critic_target),
            "outer_critic_target": str(cfg.outer_critic_target),
            "target_q_reduction": str(cfg.inner_q_target_reduction),
            "actor_q_reduction": str(cfg.inner_q_actor_reduction),
        }
        if self.is_vtrace:
            spec["vtrace"] = {
                "rho_clip": float(cfg.inner_vtrace_rho_clip),
                "c_clip": float(cfg.inner_vtrace_c_clip),
                "pg_rho_clip": float(cfg.inner_vtrace_pg_rho_clip),
                "distill_updates": int(cfg.inner_vtrace_distill_updates),
                "distill_action_samples": int(
                    cfg.inner_vtrace_distill_action_samples
                ),
                "outer_value_semantics": "reward_only",
            }
        return spec


def resolve_inner_search_semantics(cfg: Any) -> InnerSearchSemantics:
    """Build predicates from an already normalized AMBI configuration."""

    return InnerSearchSemantics(
        operator=str(getattr(cfg, "inner_operator", "sac")).lower(),
        q_objective=str(
            getattr(cfg, "inner_q_objective", "legacy_continuing")
        ).lower(),
        critic_horizon_mode=str(
            getattr(cfg, "inner_critic_horizon_mode", "shared")
        ).lower(),
        return_estimator=str(
            getattr(cfg, "inner_return_estimator", "td0")
        ).lower(),
        return_steps=getattr(cfg, "inner_return_steps", None),
        return_lambda=getattr(cfg, "inner_return_lambda", None),
        leaf_q_source=str(
            getattr(cfg, "inner_leaf_q_source", "outer_target")
        ).lower(),
        leaf_value_samples=int(getattr(cfg, "inner_leaf_value_samples", 1)),
        replay_retention=str(
            getattr(cfg, "inner_search_replay_retention", "action")
        ).lower(),
        offpolicy_mode=str(
            getattr(cfg, "inner_offpolicy_mode", "none")
        ).lower(),
        bootstrap_critic=str(
            getattr(cfg, "inner_search_bootstrap_critic", "target")
        ).lower(),
        configured_target_update_event=str(
            getattr(cfg, "inner_target_update_event", "optimizer_step")
        ).lower(),
        depth_update_order=str(
            getattr(cfg, "inner_depth_update_order", "mixed")
        ).lower(),
    )


def uses_outer_leaf(cfg: Any) -> bool:
    return resolve_inner_search_semantics(cfg).uses_outer_leaf


def uses_inner_target(cfg: Any) -> bool:
    return resolve_inner_search_semantics(cfg).uses_inner_target


def target_update_event(cfg: Any) -> str:
    return resolve_inner_search_semantics(cfg).target_update_event
