"""Maximum-likelihood coalition decoder adapted from colav/detect/decoder.py.

Working likelihood: mu * sum(z_ij in S) - mu^2/2 * C(k,2).
Unknown k is chosen by subtracting ln C(N,k), using the source's plug-in mu
(mean of the strongest C(k,2) edges, floored at 1e-6). The Gaussian model
is a scoring heuristic, not a claim that overlapping pair scores are iid.
An independent honest calibration is required before naming a coalition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations

import numpy as np

from .graph import validate_matrix


@dataclass
class DecodeResult:
    k: int
    members: frozenset[int]
    loglik: float
    mu: float
    score: float
    search: str
    k_scores: dict[int, float] = field(default_factory=dict)


@lru_cache(maxsize=32)
def _combinations(n, k):
    array = np.asarray(list(combinations(range(n), k)), dtype=int)
    array.flags.writeable = False
    return array


def _subset_sums(z, k):
    combos = _combinations(len(z), k)
    ii, jj = np.triu_indices(k, 1)
    values = np.nan_to_num(z, nan=0.0)
    return combos, values[combos[:, ii], combos[:, jj]].sum(axis=1)


def _greedy(z, k):
    """Start at an actual off-diagonal edge, even when all scores are negative."""
    zz = np.nan_to_num(z, nan=0.0)
    ii, jj = np.triu_indices(len(z), 1)
    best = int(np.argmax(zz[ii, jj]))
    members = {int(ii[best]), int(jj[best])}
    while len(members) < k:
        remaining = sorted(set(range(len(z))) - members)
        nxt = max(remaining, key=lambda c: sum(zz[c, s] for s in members))
        members.add(nxt)
    return frozenset(members), float(sum(zz[a, b] for a, b in combinations(sorted(members), 2)))


def plugin_mu(z, k):
    vals = np.nan_to_num(z[np.triu_indices(len(z), 1)], nan=0.0)
    return max(float(np.sort(vals)[-math.comb(k, 2):].mean()), 1e-6)


def ml_decode(z, k, mu, max_exhaustive=200_000):
    z = validate_matrix(z)
    if not isinstance(k, int) or not 2 <= k <= len(z):
        raise ValueError("coalition size must be between two and N")
    if not math.isfinite(mu) or mu <= 0 or max_exhaustive < 1:
        raise ValueError("mu must be finite and positive; search budget must be positive")
    if math.comb(len(z), k) <= max_exhaustive:
        combos, sums = _subset_sums(z, k)
        best = int(np.argmax(sums))
        members, value = frozenset(map(int, combos[best])), float(sums[best])
        search = "exhaustive"
    else:
        members, value = _greedy(z, k)
        search = "greedy"
    ll = mu * value - 0.5 * mu**2 * math.comb(k, 2)
    return DecodeResult(k, members, ll, mu, ll, search)


def decode(z, k=None, *, k_max=5, max_exhaustive=200_000):
    """Deployable plug-in decoder: no true k, true members or oracle mu needed."""
    z = validate_matrix(z)
    if k_max < 2 or (k is not None and (not isinstance(k, int) or not 2 <= k <= len(z))):
        raise ValueError("invalid coalition size/search range")
    if not np.isfinite(z[np.triu_indices(len(z), 1)]).any():
        return None
    sizes = [k] if k is not None else range(2, min(k_max, len(z)) + 1)
    results = []
    for size in sizes:
        result = ml_decode(z, size, plugin_mu(z, size), max_exhaustive)
        result.score -= math.log(math.comb(len(z), size)) if k is None else 0
        results.append(result)
    best = max(results, key=lambda r: r.score)
    best.k_scores = {r.k: r.score for r in results}
    return best
