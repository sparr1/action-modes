"""Portable descriptions of auxiliary return learning and controller routing.

These helpers do not import the learner, so publication and resume preflight
can describe a checkpoint without constructing networks or consuming RNG.
"""

from collections.abc import Mapping


AUX_RETURN_SOURCE_DEFAULTS = {
    "inner_actor_source": "sac",
    "inner_critic_source": "sac",
    "inner_horizon_actor_source": "sac",
    "inner_horizon_critic_source": "sac",
}


def auxiliary_return_mode(config):
    return str(config.get("aux_return_mode", "off")).lower()


def normalize_aux_return_identity(config):
    """Keep old identities unchanged when the feature and routes are unused."""
    result = dict(config)
    mode = auxiliary_return_mode(result)
    # The resolved actor view duplicates the flat scientific configuration.
    result.pop("aux_return_actor_cfg", None)
    if mode == "off":
        for key in tuple(result):
            if key.startswith("aux_return_"):
                result.pop(key)
    else:
        result["aux_return_mode"] = mode
        if str(result.get("inner_actor_source", "sac")).lower() == "return_actor":
            for key, default in (("log_std_mapping", "direct_clamp"),
                                 ("log_std_min", -20.), ("log_std_max", 2.)):
                if result.get(f"inner_{key}") is None:
                    value = result.get(f"aux_return_{key}")
                    result[f"inner_{key}"] = result.get(key, default) if value is None else value
            if (result.get("inner_actor_entropy_mode") is None
                    and str(result.get("inner_operator", "sac")).lower() == "sac"):
                value = result.get("aux_return_outer_actor_entropy_mode")
                result["inner_actor_entropy_mode"] = (
                    result.get("outer_actor_entropy_mode", "squashed") if value is None else value
                )
        for key, value in tuple(result.items()):
            if not key.startswith("aux_return_") or key == "aux_return_mode":
                continue
            inherited_key = key.removeprefix("aux_return_")
            if isinstance(value, str):
                value = value.lower()
            if key == "aux_return_ent_coef" and value == "off":
                value = 0.0
            inherited = result.get(inherited_key)
            if (inherited_key == "outer_behavior_policy_kl_min_valid_count"
                    and inherited == "auto"):
                inherited = result.get("batch_size")
            if key == "aux_return_critic_coef":
                inherited = 0.1
            elif key == "aux_return_detach_representation":
                inherited = False
            if value is None or (inherited is not None and value == inherited):
                result.pop(key)
            else:
                result[key] = value
    for key, default in AUX_RETURN_SOURCE_DEFAULTS.items():
        value = str(result.get(key, default)).lower()
        if value == default:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def auxiliary_return_architecture(config):
    """Additional learned modules, independently of runtime controller routes."""
    mode = auxiliary_return_mode(config)
    if mode == "off":
        return ()
    if mode not in {"sac", "return_actor"}:
        raise ValueError(f"Unknown aux_return_mode {mode!r}.")
    actor = None
    if mode == "return_actor":
        resolved = config.get("aux_return_actor_cfg", {})
        if not isinstance(resolved, Mapping):
            resolved = vars(resolved)

        def setting(key, default):
            value = resolved.get(key, config.get(f"aux_return_{key}"))
            return config.get(key, default) if value is None else value

        actor = (
            "squashed_gaussian",
            str(setting("log_std_mapping", "direct_clamp")).lower(),
            float(setting("log_std_min", -20)),
            float(setting("log_std_max", 2)),
        )
    # Ensemble sizes/supports already belong to the primary architecture key;
    # the independent auxiliary ensemble uses that same numeric architecture.
    return ("independent_reward_critic_v1", mode, actor)


def auxiliary_return_routing(config):
    """Report selected sources without treating them as checkpoint architecture."""
    mode = auxiliary_return_mode(config)
    if mode == "off":
        return None
    return {
        "aux_return_mode": mode,
        "inner_operator": str(config.get("inner_operator", "sac")).lower(),
        **{key: str(config.get(key, default)).lower()
           for key, default in AUX_RETURN_SOURCE_DEFAULTS.items()},
    }
