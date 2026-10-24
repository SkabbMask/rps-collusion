"""Auditor views of one RPS episode (adapted from colav observe/views.py).

Every view is a deterministic function of the Episode plus, for the relabelled view, a view-seed that
drives the pseudonym permutations. Ground truth (``ep.truth``) is never read here except to *store* the
hidden pseudonym->actor map for the evaluator (``hidden_map``).

Views
-----
1. ``identified``  moves with persistent player ids                    -> IdentifiedView
2. ``relabelled``  ids replaced by fresh pseudonyms every m rounds      -> RelabelledView (blocks)
3. ``histogram``   per-round multiset of moves / move counts            -> HistogramView
4. ``summaries``   per-round number of distinct moves and of winners   -> SummaryView

The market's ``full`` and ``metadata`` views have no counterpart: RPS has no public messages, and the
covert signal to an assistant is part of the ground truth, not of anything an auditor observes.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from rps.game import MOVES, payoffs
from .features import Episode, Symbols, move_counts, symbols


@dataclass
class Block:
    """A stretch of rounds in which pseudonyms are stable (one per address)."""
    index: int
    start: int
    stop: int
    sym: Symbols                 # symbols restricted to the block, columns = addresses
    hidden_map: Dict[int, int]   # address -> actor (ground truth; evaluator only)
    fresh: np.ndarray            # per address: pseudonym is new in this block


@dataclass
class IdentifiedView:
    name: str
    sym: Symbols

    @property
    def blocks(self) -> List[Block]:
        T, N = self.sym.T, self.sym.N
        return [Block(0, 0, T, self.sym, {i: i for i in range(N)}, fresh=np.ones(N, dtype=bool))]


@dataclass
class RelabelledView:
    name: str
    m: int
    blocks: List[Block]
    n_actors: int
    anonymity_entropy_bits: float   # H(actor | address) under the relabelling model

    @property
    def a_eff(self) -> float:
        return 2.0 ** self.anonymity_entropy_bits


@dataclass
class HistogramView:
    name: str
    counts: np.ndarray           # T x 3 move counts n_{b,t}
    sorted_moves: np.ndarray     # T x N move symbols, order destroyed
    state: np.ndarray


@dataclass
class SummaryView:
    name: str
    n_distinct: np.ndarray       # T: 1 = everyone tied, 3 = cycle (everyone scores), 2 = a winning move
    n_winners: np.ndarray        # T: players scoring a point
    state: np.ndarray


def identified_view(ep: Episode) -> IdentifiedView:
    return IdentifiedView("identified", symbols(ep))


def _view_rng(ep: Episode, view_seed: int) -> np.random.Generator:
    h = hashlib.sha256(f"{ep.episode_id}|view|{view_seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def relabelled_view(ep: Episode, m: int, view_seed: int = 0, p_relabel: float = 1.0) -> RelabelledView:
    """Fresh pseudonyms every ``m`` rounds. With ``p_relabel < 1`` each player is relabelled
    independently at each boundary; a kept pseudonym stays linkable across the boundary."""
    base = symbols(ep)
    T, N = base.T, base.N
    rng = _view_rng(ep, view_seed)
    blocks: List[Block] = []
    ent_num, ent_den = 0.0, 0
    prev_perm: Optional[np.ndarray] = None
    for b, start in enumerate(range(0, T, m)):
        stop = min(start + m, T)
        perm = np.empty(N, dtype=int)           # column -> actor
        if prev_perm is None:
            fresh_actor = np.ones(N, dtype=bool)
            perm[:] = rng.permutation(N)
        else:
            fresh_actor = rng.random(N) < p_relabel
            perm[:] = prev_perm
            relabel_cols = np.where(fresh_actor[prev_perm])[0]   # columns whose actor is relabelled
            perm[relabel_cols] = rng.permutation(prev_perm[relabel_cols])
        rows = slice(start, stop)
        sym = Symbols(base.moves[rows][:, perm], base.state[rows], base.move_defined[rows][:, perm],
                      base.n_states, [f"#{c}" for c in range(N)])
        fresh_cols = fresh_actor[perm]
        r = int(fresh_cols.sum())
        # anonymity entropy: a fresh pseudonym is one of the r relabelled actors (log2 r bits); kept: 0
        per_addr = np.where(fresh_cols, math.log2(r) if r > 0 else 0.0, 0.0)
        ent_num += float(per_addr.sum()) * (stop - start)
        ent_den += N * (stop - start)
        blocks.append(Block(b, start, stop, sym, {c: int(perm[c]) for c in range(N)}, fresh=fresh_cols))
        prev_perm = perm
    return RelabelledView(f"relabelled_m{m}", m, blocks, N, ent_num / max(ent_den, 1))


def histogram_view(ep: Episode) -> HistogramView:
    sym = symbols(ep)
    return HistogramView("histogram", move_counts(sym), np.sort(sym.moves, axis=1), sym.state)


def summary_view(ep: Episode) -> SummaryView:
    sym = symbols(ep)
    n_distinct = np.array([len(set(row)) for row in sym.moves])
    n_winners = np.array([sum(payoffs([MOVES[a] for a in row])) for row in sym.moves])
    return SummaryView("summaries", n_distinct, n_winners, sym.state)


def make_view(ep: Episode, spec: Dict[str, Any]):
    kind = spec["kind"]
    if kind == "identified":
        return identified_view(ep)
    if kind == "relabelled":
        return relabelled_view(ep, int(spec["m"]), int(spec.get("view_seed", 0)), float(spec.get("p_relabel", 1.0)))
    if kind == "histogram":
        return histogram_view(ep)
    if kind == "summaries":
        return summary_view(ep)
    raise ValueError(f"unknown view kind {kind!r}")


def view_entropy_bits(view: Any) -> Dict[str, float]:
    """'Data kept' by a view: plug-in entropy of the per-round symbol vector (bits/round) and gzip size
    (bytes), as in Liu et al. 2005. Not evidence, just volume."""
    if isinstance(view, IdentifiedView):
        disc = view.sym.moves
    elif isinstance(view, RelabelledView):
        disc = np.vstack([b.sym.moves for b in view.blocks])
    elif isinstance(view, HistogramView):
        disc = view.counts
    elif isinstance(view, SummaryView):
        disc = np.c_[view.n_distinct, view.n_winners]
    else:
        raise TypeError(type(view))
    raw = json.dumps(np.asarray(disc).tolist()).encode()
    _, counts = np.unique(np.atleast_2d(disc), axis=0, return_counts=True)
    p = counts / counts.sum()
    return {"plugin_entropy_bits_per_round": float(-(p * np.log2(p)).sum()),
            "gzip_bytes": len(gzip.compress(raw)), "raw_bytes": len(raw)}
