"""LoRA-RL identity normalization without learner or publication dependencies."""

from numbers import Integral, Real


_LORA_RL_DEFAULTS = {
    "inner_actor_adaptation": "clone",
    "inner_critic_lora_layers": "input_hidden",
    "inner_critic_lora_rank": 96,
    "inner_critic_lora_scale": 1.0,
    "inner_critic_lora_weight_decay": 0.0002,
}


def normalize_lora_rl_identity(config):
    """Resolve new-method defaults while preserving every historical mapping.

    In particular, ``lora`` remains the retired all-layer method. Dense raw
    training configurations retain any inactive fields that older lineage
    fingerprints included. Validation of executable settings belongs to the
    learner; this function does not migrate the retired method.
    """
    result = dict(config)
    if str(result.get("inner_critic_adaptation", "")).lower() != "lora_rl":
        return result
    result["inner_critic_adaptation"] = "lora_rl"
    for field in list(result):
        if field.startswith("inner_actor_lora_") or field in {
            "lora_rank", "lora_alpha", "lora_dropout", "inner_critic_lora_dropout",
        }:
            result.pop(field)
    for field, default in _LORA_RL_DEFAULTS.items():
        value = result.get(field, default)
        if isinstance(default, str) and isinstance(value, str):
            value = value.lower()
        elif isinstance(default, int) and isinstance(value, Integral) and not isinstance(value, bool):
            value = int(value)
        elif isinstance(default, float) and isinstance(value, Real) and not isinstance(value, bool):
            value = float(value)
        result[field] = value
    return result


def publication_lora_identity(config):
    """Exclude inactive adapter knobs from an executed planner's identity."""
    result = dict(config)
    for component in ("actor", "critic"):
        mode = result.get(f"inner_{component}_adaptation")
        if component == "critic" and str(mode).lower() == "lora_rl":
            continue
        if mode != "lora":
            for field in list(result):
                if field.startswith(f"inner_{component}_lora_"):
                    result.pop(field)
    return normalize_lora_rl_identity(result)
