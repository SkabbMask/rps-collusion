"""Pairwise dependence given the public state (ported from colav detect/pairwise.py, methods.md §2).

* gamma(i;j) = I(M_i; M_j | S) in bits, plug-in counts (Bonjour et al. eqs. 3/6, 12, 13). With
  simultaneous moves and a shared, fully observed state this is symmetric. Counts for a pair use the
  rounds in which BOTH moves are defined.
* Gamma(i;j) = gamma(i;j) - max_{k != i,j} gamma(k;j)   (eq. 4)   directional
* published rule (eq. 5): flag iff Gamma(i;j) >= tau and Gamma(j;i) >= tau, tau = 0.05; outcomes
  NO_PAIR / ONE_PAIR / MULTIPLE_PAIRS / INSUFFICIENT.
* permutation z-score: shuffle j's moves within each state cell, R times. Unlike Gamma, z_ij does not
  subtract the other players' influence, so several members of one coalition do not mask each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from observe.features import MOVE_ALPHABET, Symbols

A = MOVE_ALPHABET


class Outcome(str, Enum):
    NO_PAIR = "no_pair"
    ONE_PAIR = "one_pair"
    MULTIPLE_PAIRS = "multiple_pairs"
    INSUFFICIENT = "insufficient_evidence"


@dataclass
class PairwiseParams:
    R: int = 200                 # permutations
    n_min: int = 20              # minimum visits per state cell (sparser cells are dropped)
    tau: float = 0.05            # published threshold (bits)
    m_min: int = 20              # minimum shared defined rounds for a pair statistic
    seed: int = 0


@dataclass
class PairwiseResult:
    n: int
    gamma: np.ndarray                 # N x N symmetric, NaN where unsupported
    net: np.ndarray                   # N x N directional Gamma(i;j)
    z: np.ndarray                     # N x N symmetric permutation z, NaN where unsupported
    support: np.ndarray               # N x N number of rounds used per pair
    cell_counts: np.ndarray           # visits per state cell
    published_outcome: Outcome = Outcome.NO_PAIR
    published_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def max_z(self) -> float:
        z = self.z[np.triu_indices(self.n, 1)]
        z = z[~np.isnan(z)]
        return float(z.max()) if z.size else float("nan")


def _cmi_from_tables(tables: np.ndarray) -> np.ndarray:
    """tables: (..., S, A, A) counts -> CMI in bits, shape (...)."""
    tables = tables.astype(float)
    n = tables.sum(axis=(-3, -2, -1), keepdims=True)
    ns = tables.sum(axis=(-2, -1), keepdims=True)
    ni = tables.sum(axis=-1, keepdims=True)
    nj = tables.sum(axis=-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = tables * (np.log2(tables * ns) - np.log2(ni * nj))
    term = np.where(tables > 0, term, 0.0)
    out = term.sum(axis=(-3, -2, -1)) / np.squeeze(n, axis=(-3, -2, -1))
    return np.where(np.squeeze(n, axis=(-3, -2, -1)) > 0, out, np.nan)


def _pair_codes(mi: np.ndarray, mj: np.ndarray, s: np.ndarray) -> np.ndarray:
    return s * (A * A) + mi * A + mj


def net_influence(g: np.ndarray) -> np.ndarray:
    """Gamma(i;j) = gamma(i;j) - max_{k != i,j} gamma(k;j). NaN entries are ignored in the max."""
    N = g.shape[0]
    net = np.full((N, N), np.nan)
    for j in range(N):
        col = g[:, j]
        for i in range(N):
            if i == j or np.isnan(col[i]):
                continue
            others = [col[k] for k in range(N) if k not in (i, j) and not np.isnan(col[k])]
            net[i, j] = col[i] - (max(others) if others else 0.0)
    return net


def published_rule(net: np.ndarray, tau: float, cell_counts: np.ndarray, n_min: int,
                   support: np.ndarray, m_min: int) -> Tuple[Outcome, List[Tuple[int, int]]]:
    N = net.shape[0]
    if not (cell_counts >= n_min).any() or not (support[np.triu_indices(N, 1)] >= m_min).any():
        return Outcome.INSUFFICIENT, []
    qual = [(i, j) for i in range(N) for j in range(i + 1, N)
            if support[i, j] >= m_min and not np.isnan(net[i, j]) and not np.isnan(net[j, i])
            and net[i, j] >= tau and net[j, i] >= tau]
    if not qual:
        return Outcome.NO_PAIR, []
    if len(qual) == 1:
        return Outcome.ONE_PAIR, qual
    return Outcome.MULTIPLE_PAIRS, qual


def permutation_z(sym: Symbols, R: int, m_min: int, rng: np.random.Generator,
                  rows: Optional[np.ndarray] = None):
    """z_ij from within-state-cell shuffles of j's moves. Returns (gamma, z, mean, sd, support)."""
    N, S = sym.N, sym.n_states
    sel = np.arange(sym.T) if rows is None else rows
    g, z, mu, sd = (np.full((N, N), np.nan) for _ in range(4))
    sup = np.zeros((N, N), dtype=int)
    moves, state, defined = sym.moves[sel], sym.state[sel], sym.move_defined[sel]
    offsets = (np.arange(R) * (S * A * A))[:, None]
    for i in range(N):
        for j in range(i + 1, N):
            ok = defined[:, i] & defined[:, j]
            n = int(ok.sum())
            sup[i, j] = sup[j, i] = n
            if n < m_min:
                continue
            mi, mj, s = moves[ok, i], moves[ok, j], state[ok]
            tab = np.bincount(_pair_codes(mi, mj, s), minlength=S * A * A).reshape(S, A, A)
            g0 = float(_cmi_from_tables(tab))
            mj_perm = np.tile(mj, (R, 1))
            for c in range(S):
                idx = np.where(s == c)[0]
                if idx.size > 1:
                    mj_perm[:, idx] = rng.permuted(mj_perm[:, idx], axis=1)
            codes = _pair_codes(mi[None, :], mj_perm, s[None, :]) + offsets
            tabs = np.bincount(codes.ravel(), minlength=R * S * A * A).reshape(R, S, A, A)
            gr = _cmi_from_tables(tabs)
            m_, s_ = float(np.mean(gr)), float(np.std(gr, ddof=1)) if R > 1 else 0.0
            g[i, j] = g[j, i] = g0
            mu[i, j] = mu[j, i] = m_
            sd[i, j] = sd[j, i] = s_
            z[i, j] = z[j, i] = (g0 - m_) / s_ if s_ > 0 else (0.0 if g0 <= m_ else np.inf)
    return g, z, mu, sd, sup


def usable_rows(sym: Symbols, n_min: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rounds whose state cell has at least n_min visits (sparse cells are dropped from the pair
    statistics instead of voiding the whole episode). Returns (rows, cell_counts)."""
    anyd = sym.move_defined.any(axis=1)
    cells = np.bincount(sym.state[anyd], minlength=sym.n_states)
    rows = np.where((cells >= n_min)[sym.state] & anyd)[0]
    return rows, cells


def analyse_pairwise(sym: Symbols, p: PairwiseParams, seed_offset: int = 0) -> PairwiseResult:
    rng = np.random.default_rng(p.seed + seed_offset)
    rows, cells = usable_rows(sym, p.n_min)
    g, z, _, _, sup = permutation_z(sym, p.R, p.m_min, rng, rows=rows)
    net = net_influence(g)
    outcome, pairs = published_rule(net, p.tau, cells, p.n_min, sup, p.m_min)
    return PairwiseResult(n=sym.N, gamma=g, net=net, z=z, support=sup, cell_counts=cells,
                          published_outcome=outcome, published_pairs=pairs)


def null_symbols(sym: Symbols, rng: np.random.Generator) -> Symbols:
    """An honest surrogate of the episode: every player's moves are shuffled independently within each
    state cell, which keeps each player's per-state move frequencies and breaks all dependence."""
    moves = sym.moves.copy()
    defined = sym.move_defined.copy()
    for c in range(sym.n_states):
        idx = np.where(sym.state == c)[0]
        for i in range(sym.N):
            p = rng.permutation(idx)
            moves[idx, i], defined[idx, i] = sym.moves[p, i], sym.move_defined[p, i]
    return Symbols(moves, sym.state, defined, sym.n_states, sym.names)
