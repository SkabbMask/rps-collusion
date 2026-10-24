"""Mutual-information collusion detection for fully observable simultaneous games (paper Sections 3.1 and 4)."""

import math
from collections import Counter
from itertools import permutations


def individual_influence(records, i, j):
    """gamma(i; j) = I(A_i; A_j | S), estimated with Monte Carlo counts (Eq. 6, 12, 13).

    `records` is a list of (state, actions) with `actions` a dict name -> move.
    """
    n_s = Counter()
    n_si = Counter()
    n_sj = Counter()
    n_sij = Counter()
    for s, a in records:
        n_s[s] += 1
        n_si[s, a[i]] += 1
        n_sj[s, a[j]] += 1
        n_sij[s, a[i], a[j]] += 1
    total = len(records)
    gamma = 0.0
    for (s, ai, aj), c in n_sij.items():
        # p(s) * p(ai,aj|s) * log( p(ai,aj|s) / (p(ai|s) p(aj|s)) )
        gamma += (c / total) * math.log(c * n_s[s] / (n_si[s, ai] * n_sj[s, aj]))
    return gamma


def detect(records, agents, alpha=0.05):
    """Return (gamma, net influence Gamma, list of colluding pairs)."""
    gamma = {(i, j): individual_influence(records, i, j) for i, j in permutations(agents, 2)}
    net = {}
    for i, j in permutations(agents, 2):
        others = [gamma[k, j] for k in agents if k not in (i, j)]
        net[i, j] = gamma[i, j] - max(others) if others else gamma[i, j]
    flagged = [
        (i, j) for i, j in permutations(agents, 2)
        if agents.index(i) < agents.index(j) and net[i, j] >= alpha and net[j, i] >= alpha
    ]
    # The paper assumes a single colluding pair; with more players several pairs may be flagged.
    return gamma, net, flagged
