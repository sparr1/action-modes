"""Keep disabled ERE controls out of historical scientific identities."""

from copy import deepcopy


def normalize_ere_identity(config):
    result = deepcopy(config)
    strategy = str(result.get("inner_replay_strategy", "uniform")).lower()
    if strategy == "uniform" or result.get("inner_ere_actor", True):
        result.pop("inner_ere_actor", None)
    if strategy == "uniform":
        for key in ("inner_replay_strategy", "inner_ere_final_fraction", "inner_ere_min_rounds"):
            result.pop(key, None)
    else:
        result["inner_replay_strategy"] = strategy
        result.setdefault("inner_ere_final_fraction", .25)
        result.setdefault("inner_ere_min_rounds", 1)
    return result
