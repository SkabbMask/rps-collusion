"""Discretisation of an RPS episode into the symbols the detectors use (adapted from colav observe/features.py).

* move m_{i,t} in {ROCK=0, PAPER=1, SCISSORS=2}: the action itself. Unlike the market's DOWN/SAME/UP moves
  (relative to a seller's previous ask) no round-to-round linkage is needed.
* public state s_t: the ``state`` string logged by the simulation, mapped to 0..S-1.
* counts n_{b,t}: number of players choosing move b in round t (identity-free).
* a player whose LLM reply could not be parsed (random fallback) made no decision that round: undefined.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from rps.game import MOVES

MOVE_ALPHABET = len(MOVES)
MOVE_INDEX = {m: i for i, m in enumerate(MOVES)}


@dataclass
class Episode:
    """One simulation log. ``truth`` (roles, coalitions, signals) is never read by a detector."""
    episode_id: str
    names: List[str]
    rounds: List[Dict[str, Any]]
    truth: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.rounds)

    @property
    def N(self) -> int:
        return len(self.names)

    @staticmethod
    def load(path) -> "Episode":
        path = Path(path)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        meta = records.pop(0)["meta"] if records and "meta" in records[0] else {}
        names = meta.get("names") or list(records[0]["actions"])
        return Episode(path.stem, names, records, truth=meta.get("truth", {}), config=meta.get("config", {}))


@dataclass
class Symbols:
    moves: np.ndarray         # T x N int in {0,1,2}
    state: np.ndarray         # T int in 0..S-1
    move_defined: np.ndarray  # T x N bool
    n_states: int
    names: List[str]

    @property
    def T(self) -> int:
        return self.moves.shape[0]

    @property
    def N(self) -> int:
        return self.moves.shape[1]


def symbols(ep: Episode) -> Symbols:
    cells: Dict[str, int] = {}
    state = np.array([cells.setdefault(str(r.get("state", "-")), len(cells)) for r in ep.rounds], dtype=int)
    moves = np.array([[MOVE_INDEX[r["actions"][n]] for n in ep.names] for r in ep.rounds], dtype=int)
    moves = moves.reshape(ep.T, ep.N)
    defined = np.array([[n not in r.get("fallback", ()) for n in ep.names] for r in ep.rounds], dtype=bool)
    return Symbols(moves, state, defined.reshape(ep.T, ep.N), max(len(cells), 1), list(ep.names))


def move_counts(sym: Symbols) -> np.ndarray:
    """n_{b,t}: T x 3 counts of players per move (undefined moves are not counted)."""
    out = np.zeros((sym.T, MOVE_ALPHABET), dtype=int)
    for b in range(MOVE_ALPHABET):
        out[:, b] = ((sym.moves == b) & sym.move_defined).sum(axis=1)
    return out
