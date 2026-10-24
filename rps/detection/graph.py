"""The two graph detectors from colav/detect/graph.py, on RPS evidence.

Candidates are connected components or weighted Louvain communities of
the thresholded dependence graph. Q is the sum of supported internal z's.
An independent calibration split sets a cutoff on the *largest* Q found
in an honest episode, accounting for the search over candidate groups.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Candidate:
    members: frozenset[int]
    score: float


def validate_matrix(z):
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[0] != z.shape[1] or len(z) < 2:
        raise ValueError("z must be a square matrix with at least two players")
    if np.isinf(z).any() or not np.allclose(z, z.T, equal_nan=True):
        raise ValueError("z must be symmetric and finite or NaN")
    return z


def edge_set(z, z_alpha):
    z = validate_matrix(z)
    if np.isnan(z_alpha):
        raise ValueError("edge threshold cannot be NaN")
    return [(i, j) for i, j in combinations(range(len(z)), 2)
            if np.isfinite(z[i, j]) and z[i, j] > z_alpha]


def cluster_score(z, members):
    return float(sum(z[i, j] for i, j in combinations(sorted(members), 2) if np.isfinite(z[i, j])))


def clusters(z, edges, *, extractor="components", seed=0):
    z = validate_matrix(z)
    if extractor not in ("components", "louvain"):
        raise ValueError("extractor must be components or louvain")
    graph = nx.Graph()
    graph.add_nodes_from(range(len(z)))
    for i, j in edges:
        if i == j or i not in graph or j not in graph or not np.isfinite(z[i, j]):
            raise ValueError("edge must join two supported, distinct players")
        graph.add_edge(i, j, weight=max(float(z[i, j]), 1e-9))
    if not graph.number_of_edges():
        return []
    groups = (nx.connected_components(graph) if extractor == "components" else
              nx.community.louvain_communities(graph, weight="weight", seed=seed))
    result = [Candidate(frozenset(g), cluster_score(z, g)) for g in groups if len(g) > 1]
    return sorted(result, key=lambda c: (-c.score, tuple(sorted(c.members))))
