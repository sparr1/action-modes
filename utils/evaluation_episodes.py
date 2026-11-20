"""Episode sampling and pairing shared by evaluation and CPU publication.

Legacy results keep their per-episode seed identity. Consecutive episodes use
indices within an explicitly identified environment/controller stream instead.
This module has no numerical or GPU dependencies.
"""

from numbers import Integral


def _integer(value, name, minimum=0, maximum=2**32 - 1):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    return int(value)


def consecutive_protocol(evaluation):
    mode = evaluation.get("episode_sampling", "independent_seeds")
    if mode not in ("independent_seeds", "consecutive"):
        raise ValueError("episode_sampling must be 'independent_seeds' or 'consecutive'.")
    if mode == "independent_seeds":
        if "environment_seed" in evaluation or "num_episodes" in evaluation:
            raise ValueError("environment_seed and num_episodes require consecutive episode sampling.")
        return None
    if "seeds" in evaluation:
        raise ValueError("Consecutive evaluation uses environment_seed once; do not supply per-episode seeds.")
    return {
        "mode": "consecutive",
        "environment_seed": _integer(evaluation.get("environment_seed"), "environment_seed"),
        "num_episodes": _integer(evaluation.get("num_episodes"), "num_episodes", minimum=1),
        "environment_rng": "seed_once_then_unseeded_resets",
        "controller_rng": "seed_once_then_advance",
        "checkpoint_rng": "restart_same_stream",
    }


def validate_consecutive_protocol(sampling):
    if not isinstance(sampling, dict):
        raise ValueError("Consecutive episode protocol must be a mapping.")
    expected = consecutive_protocol({
        "episode_sampling": "consecutive",
        "environment_seed": sampling.get("environment_seed"),
        "num_episodes": sampling.get("num_episodes"),
    })
    if sampling != expected:
        raise ValueError("Unsupported consecutive episode RNG protocol.")
    return expected


def episode_key(episode):
    return episode["episode_index"] if "episode_index" in episode else episode["seed"]


def episode_id(episode):
    prefix = "episode" if "episode_index" in episode else "seed"
    return f"{prefix}-{episode_key(episode)}"


def validate_consecutive_episodes(episodes, sampling, controller_seed):
    validate_consecutive_protocol(sampling)
    indices = [ep.get("episode_index") for ep in episodes]
    if (any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
            or indices != list(range(sampling["num_episodes"]))):
        raise ValueError("Missing, duplicate, or out-of-order consecutive episode indices.")
    for episode in episodes:
        if ("seed" in episode or "solver_seed" in episode
                or episode.get("environment_seed") != sampling["environment_seed"]
                or episode.get("controller_stream_seed") != controller_seed):
            raise ValueError("Consecutive episode stream seeds disagree with the protocol.")
        digest = episode.get("initial_observation_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Consecutive episodes require a hash of each starting observation.")


def validate_paired_starts(reference, candidate):
    """Never report paired gains when controllers received different starts."""
    def starts(episodes):
        return {episode_key(ep): ep["initial_observation_sha256"] for ep in episodes}
    if (len(reference) != len(candidate) or len(starts(reference)) != len(reference)
            or len(starts(candidate)) != len(candidate) or starts(reference) != starts(candidate)):
        raise ValueError("Paired consecutive controllers received different episode starts.")
