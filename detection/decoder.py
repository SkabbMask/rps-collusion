"""Maximum-likelihood coalition decoder and identification limit (methods.md §5).

(5.1) z_ij ~ N(mu, 1) inside the coalition, N(0, 1) otherwise (working model).
(5.2) l(S') = mu * sum_{i<j in S'} z_ij - mu^2/2 * C(k,2);  S_ML = argmax over k-subsets.
(5.3) k unknown: k_hat = argmax_k [ l*_k - ln C(N,k) ]  (heuristic penalty), 2 <= k <= k_max;
      alternatively compare l*_k with its own honest null (done by the calibrator).
(5.4)-(5.5) I_pair and the Fano lower bound on P[S_hat != S].

Exhaustive search over k-subsets; the pair sum is vectorised with numpy over all combinations
(C(40,5) = 658k subsets is the largest case in the grid). ``greedy=True`` falls back to greedy
growth for larger problems. Missing z (NaN) are treated as 0 (no evidence), as in §4.

Two decoders are reported: oracle-mu (mu set on development episodes from true coalition
pairs; a ceiling, not a deployable detector) and plug-in mu (mean of the top C(k,2) z of the
episode itself).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def binary_entropy(p: float) -> float:
    """(from colav info/bounds.py)"""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


@dataclass
class DecodeResult:
    k: int
    members: Set[int]
    loglik: float
    mu: float
    mode: str                       # "oracle" | "plugin"
    k_scores: Optional[Dict[int, float]] = None


def _subset_sums(z: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    N = z.shape[0]
    zz = np.nan_to_num(z, nan=0.0)
    combos = np.array(list(combinations(range(N), k)), dtype=int)
    if combos.size == 0:
        return combos, np.array([])
    ii, jj = np.triu_indices(k, 1)
    sums = zz[combos[:, ii], combos[:, jj]].sum(axis=1)
    return combos, sums


def _greedy(z: np.ndarray, k: int) -> Tuple[Set[int], float]:
    zz = np.nan_to_num(z, nan=0.0)
    N = zz.shape[0]
    i, j = np.unravel_index(np.argmax(np.triu(zz, 1)), zz.shape)
    S = {int(i), int(j)}
    while len(S) < k:
        best, bv = None, -np.inf
        for c in range(N):
            if c in S:
                continue
            v = sum(zz[c, s] for s in S)
            if v > bv:
                best, bv = c, v
        S.add(best)
    return S, sum(zz[a, b] for a, b in combinations(sorted(S), 2))


def ml_decode(z: np.ndarray, k: int, mu: float, mode: str, max_exhaustive: int = 2_000_000) -> DecodeResult:
    N = z.shape[0]
    if math.comb(N, k) <= max_exhaustive:
        combos, sums = _subset_sums(z, k)
        best = int(np.argmax(sums))
        S, ssum = set(int(x) for x in combos[best]), float(sums[best])
    else:
        S, ssum = _greedy(z, k)
    ll = mu * ssum - 0.5 * mu * mu * math.comb(k, 2)
    return DecodeResult(k=k, members=S, loglik=ll, mu=mu, mode=mode)


def plugin_mu(z: np.ndarray, k: int) -> float:
    vals = np.nan_to_num(z[np.triu_indices(z.shape[0], 1)], nan=0.0)
    top = np.sort(vals)[::-1][: math.comb(k, 2)]
    return float(max(top.mean(), 1e-6))


def decode(z: np.ndarray, k: Optional[int], mu_oracle: Optional[float], k_max: int = 5,
           mode: str = "oracle") -> DecodeResult:
    """If k is None, choose k by (5.3). ``mode`` selects oracle or plug-in mu."""
    N = z.shape[0]
    ks = [k] if k is not None else list(range(2, min(k_max, N) + 1))
    best: Optional[DecodeResult] = None
    k_scores: Dict[int, float] = {}
    for kk in ks:
        mu = mu_oracle if mode == "oracle" else plugin_mu(z, kk)
        if mu is None:
            raise ValueError("oracle mode needs mu_oracle")
        r = ml_decode(z, kk, mu, mode)
        score = r.loglik - (math.log(math.comb(N, kk)) if k is None else 0.0)
        k_scores[kk] = score
        if best is None or score > k_scores[best.k]:
            best = r
    assert best is not None
    best.k_scores = k_scores
    return best


# --- growth from the strongest pair --------------------------------------------------------------
@dataclass
class GrowthResult:
    members: Set[int]
    seed_pair: Tuple[int, int]
    seed_z: float
    added: List[Tuple[int, float, float]]                 # (member, its mean z to the set, threshold)
    rejected: Optional[Tuple[int, float, float]] = None   # best candidate that missed: same fields


def grow_from_top_pair(z: np.ndarray, frac: float = 0.5) -> GrowthResult:
    """Start from the pair with the highest z. Repeatedly take the outside player with the highest
    summed z to the current set and add it while its mean z to the set is >= frac * (mean z over
    all pairs inside the set); the threshold is recomputed after every addition.

    Unlike the ML decoder this does not assume every pair inside the coalition is elevated, so a
    star (assistants that each depend on the primary but not on each other) can still be grown,
    as long as frac is low enough for the star's diluted mean. Missing z (NaN) count as 0."""
    zz = np.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)
    N = zz.shape[0]
    i, j = np.unravel_index(np.argmax(np.where(np.triu(np.ones((N, N), bool), 1), zz, -np.inf)), zz.shape)
    S = [int(i), int(j)]
    seed_z = float(zz[i, j])
    res = GrowthResult(set(S), (int(i), int(j)), seed_z, [])
    while len(S) < N:
        ii, jj = np.triu_indices(len(S), 1)
        threshold = frac * float(zz[np.array(S)[ii], np.array(S)[jj]].mean())
        outside = [c for c in range(N) if c not in S]
        sums = {c: float(zz[c, S].sum()) for c in outside}
        best = max(outside, key=sums.get)
        mean = sums[best] / len(S)
        if mean < threshold:
            res.rejected = (best, mean, threshold)
            break
        S.append(best)
        res.added.append((best, mean, threshold))
    res.members = set(S)
    return res


# --- closed forms -------------------------------------------------------------------------------
def i_pair(alpha: float, beta: float, N: int, k: int) -> float:
    """(5.4) information one pairwise test carries about membership of that pair (bits)."""
    pi = math.comb(k, 2) / math.comb(N, 2)
    q = pi * beta + (1 - pi) * alpha
    return binary_entropy(q) - pi * binary_entropy(beta) - (1 - pi) * binary_entropy(alpha)


def fano_error_lower_bound(alpha: float, beta: float, N: int, k: int) -> float:
    """(5.5) Pr[S_hat != S] >= 1 - (C(N,2) I_pair + 1) / log2 C(N,k)   (clipped to [0,1])."""
    denom = math.log2(math.comb(N, k))
    if denom <= 0:
        return 0.0
    val = 1.0 - (math.comb(N, 2) * i_pair(alpha, beta, N, k) + 1.0) / denom
    return float(min(max(val, 0.0), 1.0))
