"""Execution backend controls and exact-resume identity for AMBI critics."""


DEFAULT_CRITIC_ENSEMBLE_BACKEND = "vmap"
CRITIC_ENSEMBLE_BACKENDS = frozenset({"loop", "vmap"})


def normalize_critic_ensemble_backend(value):
    """Validate a backend without coercing non-string configuration values."""
    if not isinstance(value, str) or value.lower() not in CRITIC_ENSEMBLE_BACKENDS:
        raise ValueError("critic_ensemble_backend must be 'loop' or 'vmap'.")
    return value.lower()


def normalize_ensemble_backend_identity(config):
    """Distinguish the new default from historical loop execution.

    Scientific identity is constructed before model configuration is resolved.
    Omission therefore means the current vmap default. Explicit loop execution
    retains the pre-feature identity, whose configuration had no backend field.
    Saved checkpoint specifications separately interpret missing keys as loop.
    """
    result = dict(config)
    backend = normalize_critic_ensemble_backend(
        result.get("critic_ensemble_backend", DEFAULT_CRITIC_ENSEMBLE_BACKEND)
    )
    if backend == "loop":
        result.pop("critic_ensemble_backend", None)
    else:
        result["critic_ensemble_backend"] = backend
    return result
