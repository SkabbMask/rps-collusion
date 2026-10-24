"""Relative average-z growth from SkabbMask/rps-collusion new-decoder.

Adapted from grow_from_top_pair at 1bea281244ad2d756b9b507145b2046fefcf1582.
This selects one candidate group. Its strongest-pair score must independently
pass honest episode calibration before the candidate is named as a detection.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .graph import validate_matrix


@dataclass
class GrowthResult:
    members: frozenset[int]
    seed_pair: tuple[int, int]
    seed_z: float
    added: list[tuple[int, float, float]] = field(default_factory=list)
    rejected: tuple[int, float, float] | None = None


def grow_from_top_pair(z, frac=0.25):
    """Join iff mean z to the group >= frac * current internal mean z.

    Recompute the threshold after each addition. Missing edges contribute zero
    to means; seeds must be actually observed pairs. Ties use player order.
    All-unsupported episodes return None. No true group size is supplied.
    """
    z = validate_matrix(z)
    if not math.isfinite(frac) or not 0 <= frac <= 1:
        raise ValueError("growth fraction must be finite and in [0,1]")
    ii, jj = np.triu_indices(len(z), 1)
    supported = np.isfinite(z[ii, jj])
    if not supported.any():
        return None
    best = int(np.argmax(np.where(supported, z[ii, jj], -np.inf)))
    i, j = int(ii[best]), int(jj[best])
    members = [i, j]
    result = GrowthResult(frozenset(members), (i, j), float(z[i, j]))
    zz = np.nan_to_num(z, nan=0.0)
    while len(members) < len(z):
        a, b = np.triu_indices(len(members), 1)
        threshold = frac * float(zz[np.asarray(members)[a], np.asarray(members)[b]].mean())
        outside = [c for c in range(len(z)) if c not in members]
        best = max(outside, key=lambda c: float(zz[c, members].sum()))
        mean = float(zz[best, members].mean())
        if mean < threshold:
            result.rejected = (best, mean, threshold)
            break
        members.append(best)
        result.added.append((best, mean, threshold))
    result.members = frozenset(members)
    return result
