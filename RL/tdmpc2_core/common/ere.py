"""Whole-collection-round ERE windows; independent of sampling RNG and tensors."""

import math


def round_windows(rounds, updates, final_fraction, min_rounds):
    """Return eligible whole-round counts for one complete optimizer phase.

    Inputs are resolved configuration values. No updates means no windows; a
    single update uses all retained data. The last update uses exactly the
    configured endpoint, avoiding exponent-rounding drift there.
    """
    if updates == 0:
        return []
    if rounds < 1:
        raise ValueError("ERE updates require at least one collected replay round.")
    if updates == 1:
        return [rounds]
    windows = []
    for k in range(updates):
        fraction = final_fraction if k == updates - 1 else final_fraction ** (k / (updates - 1))
        windows.append(min(rounds, max(min_rounds, math.ceil(rounds * fraction))))
    return windows
