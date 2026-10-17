"""Explicit outer actor/critic identities for an AMBI control solve.

These selectors do not change the original SAC learner or split-head semantics.
The independently trained auxiliary critic is available only in single mode.
"""


SOURCE_DEFAULTS = {
    "inner_actor_source": "sac",
    "inner_critic_source": "sac",
    "inner_horizon_actor_source": "sac",
    "inner_horizon_critic_source": "sac",
}


def source_metadata(cfg):
    """Return stable identities, including inactive controller settings."""
    return {name: str(getattr(cfg, name, default))
            for name, default in SOURCE_DEFAULTS.items()}


def actor_owner(agent, source):
    if source == "sac":
        return agent
    if source == "return_actor":
        owner = getattr(agent, "aux_return", None)
        if owner is None or getattr(agent.model, "_return_pi", None) is None:
            raise ValueError("return_actor requires the optional auxiliary return actor.")
        return owner
    raise ValueError(f"Unknown actor source: {source!r}.")


def actor_module(agent, source):
    actor_owner(agent, source)
    return agent.model._pi if source == "sac" else agent.model._return_pi


def actor_options(agent, source):
    """Leave original-SAC calls byte-for-byte compatible with default bounds."""
    if source == "sac":
        return {}
    cfg = actor_owner(agent, source).cfg
    return {name: getattr(cfg, name)
            for name in ("log_std_mapping", "log_std_min", "log_std_max")}


def critic_module(agent, source, *, target=False):
    if source == "sac":
        return agent.model._target_Qs if target else agent.model._Qs
    if source == "aux_return":
        name = "_target_aux_return_Qs" if target else "_aux_return_Qs"
        critic = getattr(agent.model, name, None)
        if critic is None:
            raise ValueError("aux_return requires the independently trained return critic.")
        return critic
    raise ValueError(f"Unknown critic source: {source!r}.")


def critic_owner(agent, source):
    if source == "sac":
        return agent
    if source == "aux_return" and getattr(agent, "aux_return", None) is not None:
        return agent.aux_return
    raise ValueError(f"Unavailable critic source: {source!r}.")
