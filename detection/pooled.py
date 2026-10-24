"""Identity-free test on the histogram view (RPS counterpart of colav detect/pooled.py §3.3).

The market version uses the total correlation of the level-count streams. RPS counts always sum to N,
so instead we test the distribution of per-round count vectors (n_rock, n_paper, n_scissors) against a
multinomial with the pooled per-state move frequencies, with a G statistic and a parametric bootstrap.
Assistant collusion shows up as too few all-equal rounds and too many two-move rounds.

Assumes the players are exchangeable under the null (same move frequencies). Honest players with
different biases (e.g. one LLM, two random players) also break the multinomial, so on mixed line-ups
this statistic flags heterogeneity as well as collusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log
from typing import Dict

import numpy as np

from observe.views import HistogramView


@dataclass
class PooledParams:
    R: int = 200
    seed: int = 0


def _g_statistic(counts: np.ndarray, state: np.ndarray) -> float:
    """G = 2 sum n_obs ln(n_obs / n_exp) over the observed count vectors, per state cell."""
    n = int(counts[0].sum())
    g = 0.0
    for c in np.unique(state):
        rows = counts[state == c]
        p = rows.sum(axis=0) / rows.sum()
        vecs, obs = np.unique(rows, axis=0, return_counts=True)
        for v, o in zip(vecs, obs):
            log_exp = lgamma(n + 1) - sum(lgamma(k + 1) for k in v) + sum(k * log(q) for k, q in zip(v, p) if k)
            g += 2 * o * (log(o / len(rows)) - log_exp)
    return g


def pooled_statistics(hist: HistogramView, p: PooledParams, seed_offset: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(p.seed + seed_offset)
    n_players = hist.sorted_moves.shape[1]
    full = hist.counts.sum(axis=1) == n_players          # rounds with an undefined move are dropped
    counts, state = hist.counts[full], hist.state[full]
    g0 = _g_statistic(counts, state)
    null = np.empty(p.R)
    for r in range(p.R):
        sim = np.zeros_like(counts)
        for c in np.unique(state):
            idx = state == c
            probs = counts[idx].sum(axis=0) / counts[idx].sum()
            sim[idx] = rng.multinomial(n_players, probs, size=int(idx.sum()))
        null[r] = _g_statistic(sim, state)
    m, s = float(null.mean()), float(null.std(ddof=1)) if p.R > 1 else 0.0
    return {"g": g0, "g_null_mean": m, "g_null_sd": s,
            "g_z": (g0 - m) / s if s > 0 else (0.0 if g0 <= m else float("inf")),
            "g_p": float((1 + (null >= g0).sum()) / (1 + p.R)), "rounds": int(full.sum())}
