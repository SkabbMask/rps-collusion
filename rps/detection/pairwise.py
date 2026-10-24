"""Conditional MI and within-state shuffled-action comparisons.

MI uses nats. Shuffling assumes exchangeability within states, which may fail
for adaptive policies. Adapted from COLAV; see ATTRIBUTION.md."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from rps.observe.features import Symbols


@dataclass(frozen=True)
class PairwiseParams:
    permutations: int = 99
    min_pair_rounds: int = 20
    min_state_rounds: int = 10
    seed: int = 0

    def __post_init__(self):
        if self.permutations < 2 or self.min_pair_rounds < 2 or self.min_state_rounds < 2:
            raise ValueError("permutations and support minima must be at least two")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass
class PairwiseResult:
    gamma: np.ndarray
    z: np.ndarray
    pvalue: np.ndarray
    support: np.ndarray
    permutation_mean: np.ndarray
    permutation_sd: np.ndarray

    @property
    def available(self):
        return bool(np.isfinite(self.z).any())

    @property
    def max_z(self):
        return float(np.nanmax(self.z)) if self.available else float("-inf")


def cmi_from_tables(tables: np.ndarray):
    """(..., S, 3, 3) counts -> CMI in natural-log units."""
    table = np.asarray(tables, dtype=float)
    total = table.sum(axis=(-3, -2, -1))
    ns = table.sum(axis=(-2, -1), keepdims=True)
    ni = table.sum(axis=-1, keepdims=True)
    nj = table.sum(axis=-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(table > 0, table * (np.log(table * ns) - np.log(ni * nj)), 0)
        result = terms.sum(axis=(-3, -2, -1)) / total
    return np.where(total > 0, np.maximum(result, 0), np.nan)


def analyse_pairwise(sym: Symbols, params: PairwiseParams) -> PairwiseResult:
    if not isinstance(sym, Symbols):
        raise TypeError("pairwise detectors require identified Symbols; anonymous views cannot identify coalitions")
    rng = np.random.default_rng(params.seed)
    n = sym.N
    gamma, z, pv, mean, sd = [np.full((n, n), np.nan) for _ in range(5)]
    support = np.zeros((n, n), dtype=int)
    r = params.permutations
    for i, j in combinations(range(n), 2):
        valid = sym.move_defined[:, i] & sym.move_defined[:, j]
        counts = np.bincount(sym.state[valid], minlength=len(sym.state_labels))
        valid &= counts[sym.state] >= params.min_state_rounds
        support[i, j] = support[j, i] = valid.sum()
        if valid.sum() < params.min_pair_rounds:
            continue
        _, state = np.unique(sym.state[valid], return_inverse=True)
        x, y = sym.moves[valid, i], sym.moves[valid, j]
        cells = int(state.max()) + 1
        size = cells * 9
        observed = np.bincount(state * 9 + x * 3 + y, minlength=size).reshape(cells, 3, 3)
        g = float(cmi_from_tables(observed))
        shuffled = np.tile(y, (r, 1))
        for s in range(cells):
            idx = np.flatnonzero(state == s)
            shuffled[:, idx] = rng.permuted(shuffled[:, idx], axis=1)
        codes = state[None, :] * 9 + x[None, :] * 3 + shuffled + np.arange(r)[:, None] * size
        null = cmi_from_tables(np.bincount(codes.ravel(), minlength=r * size).reshape(r, cells, 3, 3))
        mu, sigma = float(null.mean()), float(null.std(ddof=1))
        # Constant strategies contain no information. A finite floor prevents
        # inf/overflow in the downstream Gaussian working likelihood.
        zi = 0.0 if abs(g - mu) <= 1e-12 and sigma <= 1e-12 else (g - mu) / max(sigma, 1e-12)
        for matrix, value in [(gamma, g), (z, zi), (pv, (1 + np.sum(null >= g - 1e-12)) / (r + 1)),
                              (mean, mu), (sd, sigma)]:
            matrix[i, j] = matrix[j, i] = value
    return PairwiseResult(gamma, z, pv, support, mean, sd)
