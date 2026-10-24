"""Dependence graph and clustering (methods.md §4).

(4.1) edge set: z_ij >= z_alpha (max-statistic threshold) or the voted edge set (votes >= t);
      edges with support below m_min are *missing* (NaN), not zero.
(4.2) candidate coalitions = connected components with >= 2 nodes (and Louvain communities as a
      second extractor), pruned to their core (below); score = mean z per pair inside S (the sum
      Q(S) grows with group size); accept iff the score > q_95, calibrated on the best score found
      on honest episodes.

Pruning: a member must have at least ``min_links`` links inside the group, each an edge with
z >= link_frac * (the group's strongest z). The member with the fewest such links is removed and
the counts recomputed until every member qualifies; what is left is split into its connected parts.
This keeps an honest player with one weak link to a member (or a few weak links) out of the group.
Stars (assistants linked only to the primary) need min_links = 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ClusterResult:
    candidates: List[Tuple[Set[int], float]]           # (member set, mean z per pair)
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


def cluster_mean_z(z: np.ndarray, members: Set[int]) -> float:
    """Mean z per pair inside the group (missing z count as 0)."""
    k = len(members)
    return cluster_score(z, members) / (k * (k - 1) / 2) if k >= 2 else float("-inf")


def prune(z: np.ndarray, members: Set[int], edges: List[Tuple[int, int]], min_links: int = 1,
          link_frac: float = 0.0) -> List[Set[int]]:
    """Core(s) of a candidate group: see the module docstring."""
    edge = set(edges) | {(j, i) for i, j in edges}
    S = set(members)
    zz = np.nan_to_num(z, nan=0.0)
    while len(S) >= 2:
        top = max(zz[i, j] for i, j in combinations(S, 2))
        strong = [(i, j) for i, j in combinations(sorted(S), 2) if (i, j) in edge and zz[i, j] >= link_frac * top]
        links = {i: sum(i in e for e in strong) for i in S}
        worst = min(S, key=lambda i: (links[i], sum(zz[i, j] for j in S if j != i)))
        if links[worst] >= min_links:
            return [c for c in _components_of(S, strong) if len(c) >= 2]
        S.remove(worst)
    return []


def _components_of(nodes: Set[int], edges: List[Tuple[int, int]]) -> List[Set[int]]:
    index = sorted(nodes)
    pos = {a: k for k, a in enumerate(index)}
    comps = _components(len(index), [(pos[i], pos[j]) for i, j in edges])
    return [{index[k] for k in c} for c in comps]


def clusters(z: np.ndarray, edges: List[Tuple[int, int]], use_louvain: bool = True,
             seed: int = 0, min_links: int = 1, link_frac: float = 0.0) -> ClusterResult:
    cores = [core for c in _components(z.shape[0], edges) if len(c) >= 2
             for core in prune(z, c, edges, min_links, link_frac)]
    cands = [(c, cluster_mean_z(z, c)) for c in cores]
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
        res.louvain = sorted(((set(c), cluster_mean_z(z, set(c))) for c in comms if len(c) >= 2),
                             key=lambda x: -x[1])
    return res


def recovery_metrics(est: Optional[Set[int]], truth: Set[int]) -> Dict[str, float]:
    if est is None or not est:
        return {"exact": 0.0, "jaccard": 0.0, "no_name": 1.0}
    inter = len(est & truth)
    union = len(est | truth)
    return {"exact": float(est == truth), "jaccard": inter / union if union else 0.0, "no_name": 0.0}
