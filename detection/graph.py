"""Dependence graph and clustering (methods.md §4).

(4.1) edge set: z_ij >= z_alpha (max-statistic threshold) or the voted edge set (votes >= t);
      edges with support below m_min are *missing* (NaN), not zero.
(4.2) candidate coalitions = connected components with >= 2 nodes (and Louvain communities as a
      second extractor); Q(S) = sum of z over pairs inside S; accept iff Q > q_95 calibrated
      on the largest Q found on honest episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ClusterResult:
    candidates: List[Tuple[Set[int], float]]           # (member set, Q)
    best: Optional[Set[int]]
    best_q: float
    extractor: str = "components"
    louvain: List[Tuple[Set[int], float]] = field(default_factory=list)


def edge_set(z: np.ndarray, z_alpha: float, votes: Optional[np.ndarray] = None,
             t_votes: Optional[int] = None) -> List[Tuple[int, int]]:
    N = z.shape[0]
    edges = []
    for i, j in combinations(range(N), 2):
        if votes is not None and t_votes is not None:
            if votes[i, j] >= 0 and votes[i, j] >= t_votes:
                edges.append((i, j))
        elif not np.isnan(z[i, j]) and z[i, j] >= z_alpha:
            edges.append((i, j))
    return edges


def _components(N: int, edges: List[Tuple[int, int]]) -> List[Set[int]]:
    parent = list(range(N))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        parent[find(i)] = find(j)
    groups: Dict[int, Set[int]] = {}
    for a in range(N):
        groups.setdefault(find(a), set()).add(a)
    return list(groups.values())


def cluster_score(z: np.ndarray, members: Set[int]) -> float:
    s = 0.0
    for i, j in combinations(sorted(members), 2):
        v = z[i, j]
        if not np.isnan(v):
            s += float(v)
    return s


def clusters(z: np.ndarray, edges: List[Tuple[int, int]], use_louvain: bool = True,
             seed: int = 0) -> ClusterResult:
    cands = [(c, cluster_score(z, c)) for c in _components(z.shape[0], edges) if len(c) >= 2]
    cands.sort(key=lambda x: -x[1])
    res = ClusterResult(candidates=cands, best=cands[0][0] if cands else None,
                        best_q=cands[0][1] if cands else float("-inf"))
    if use_louvain and edges:
        try:
            import networkx as nx
        except ImportError:          # Louvain is a second extractor only; components need no networkx
            return res
        G = nx.Graph()
        G.add_nodes_from(range(z.shape[0]))
        for i, j in edges:
            G.add_edge(i, j, weight=max(float(z[i, j]), 1e-9))
        comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
        res.louvain = sorted(((set(c), cluster_score(z, set(c))) for c in comms if len(c) >= 2),
                             key=lambda x: -x[1])
    return res


def recovery_metrics(est: Optional[Set[int]], truth: Set[int]) -> Dict[str, float]:
    if est is None or not est:
        return {"exact": 0.0, "jaccard": 0.0, "no_name": 1.0}
    inter = len(est & truth)
    union = len(est | truth)
    return {"exact": float(est == truth), "jaccard": inter / union if union else 0.0, "no_name": 0.0}
