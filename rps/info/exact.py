"""Exact small RPS experiment, including the randomized relabelling channel.

Under H0 all moves are iid uniform. H1 picks an ordered (primary, helper)
pair uniformly once per episode, then the helper assists with probability cp
independently per round. All other draws are uniform. Messages are explicitly
observed empty strings under both hypotheses, not unobserved instrumentation.
"""
import itertools
import math
from collections import defaultdict

from .bounds import exact_bounds

VIEWS = ("full", "identified", "relabelled", "histogram", "summaries", "metadata")


def view_keys(rows, m):
    identified = tuple(rows)
    histogram = tuple(tuple(row.count(k) for k in range(3)) for row in rows)
    summaries = []
    for counts in histogram:
        n = sum(counts)
        entropy = -sum(c / n * math.log2(c / n) for c in counts if c)
        summaries.append((round(entropy, 14), max(counts) / n,
                          int(max(counts) == n), int(all(counts))))
    # Canonical column histories are sufficient for an independent uniform
    # permutation in each block: every labelled member of an orbit has the
    # same conditional channel probability, under both H0 and H1.
    blocks = tuple(tuple(sorted(zip(*rows[start:start + m]))) for start in range(0, len(rows), m))
    return {"identified": identified, "full": identified, "metadata": identified,
            "relabelled": blocks, "histogram": histogram, "summaries": tuple(summaries)}


def enumerate_views(n=3, rounds=3, cp=0.5, m=1, alpha=0.05):
    if not isinstance(n, int) or not isinstance(rounds, int) or n < 2 or rounds < 1 or n * rounds > 9:
        raise ValueError("exact enumeration requires N>=2, T>=1 and N*T<=9")
    if not isinstance(m, int) or m < 1 or not math.isfinite(cp) or not 0 <= cp <= 1:
        raise ValueError("need positive integer m and cp in [0,1]")
    pairs = list(itertools.permutations(range(n), 2))
    pmfs = {view: (defaultdict(float), defaultdict(float)) for view in VIEWS}
    p0 = 3.0 ** (-n * rounds)
    for flat in itertools.product(range(3), repeat=n * rounds):
        rows = tuple(tuple(flat[t * n:(t + 1) * n]) for t in range(rounds))
        p1 = 0.0
        for primary, helper in pairs:
            probability = 1.0
            for row in rows:
                assist = row[helper] == (row[primary] - 1) % 3
                probability *= 3.0 ** (-(n - 1)) * ((1 - cp) / 3 + cp * assist)
            p1 += probability / len(pairs)
        for view, key in view_keys(rows, m).items():
            pmfs[view][0][key] += p0
            pmfs[view][1][key] += p1
    return {"model": "iid uniform honest vs one fixed uniformly unknown ordered assisting pair",
            "label": "binary coalition mechanism, balanced prior; not coalition membership",
            "n_players": n, "rounds": rounds, "cp": cp, "relabel_interval": m,
            "messages": "explicit observed silence under both hypotheses; full=metadata=identified",
            "relabel_channel": "fresh independent uniform player permutation at each block boundary",
            "views": {v: {**exact_bounds(*pmfs[v], alpha), "n_view_atoms": len(pmfs[v][0])}
                      for v in VIEWS}}
