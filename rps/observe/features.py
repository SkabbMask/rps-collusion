"""RPS symbols adapted from colav/observe/features.py; see ATTRIBUTION.md.

Only actions and pre-action public state are observable here. Roles, private
signals, policies, payoffs and ground truth are deliberately never consumed.
Rock/paper/scissors are categories, not ordered numerical price changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rps.game import MOVES

ROCK, PAPER, SCISSORS, MISSING = 0, 1, 2, -1
MOVE_ALPHABET = 3
MOVE_CODES = {move: i for i, move in enumerate(MOVES)}


@dataclass(frozen=True)
class Symbols:
    moves: np.ndarray                 # T x N, missing = -1
    state: np.ndarray                 # T, contiguous categorical state codes
    move_defined: np.ndarray          # T x N boolean validity mask
    names: tuple[str, ...]
    state_labels: tuple[str, ...]
    state_mode: str

    @property
    def T(self) -> int:
        return self.moves.shape[0]

    @property
    def N(self) -> int:
        return self.moves.shape[1]

    def __post_init__(self):
        if self.moves.ndim != 2 or self.moves.shape != self.move_defined.shape:
            raise ValueError("moves and validity mask must be matching T x N arrays")
        if self.T < 1 or self.N < 2 or self.state.shape != (self.T,):
            raise ValueError("need at least one round, two players and one state per round")
        if len(self.names) != self.N or len(set(self.names)) != self.N:
            raise ValueError("player names must be unique and match columns")
        if np.any((self.moves < -1) | (self.moves > 2)):
            raise ValueError("invalid categorical RPS move")
        if not np.array_equal(self.move_defined, self.moves != MISSING):
            raise ValueError("validity mask does not match missing moves")
        if np.any((self.state < 0) | (self.state >= len(self.state_labels))):
            raise ValueError("state code out of range")


def encode_states(values: Sequence) -> tuple[np.ndarray, tuple[str, ...]]:
    """Deterministic factorisation; JSON also handles list/dict public states."""
    labels, index, codes = [], {}, []
    for value in values:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if key not in index:
            index[key] = len(labels)
            labels.append(key)
        codes.append(index[key])
    return np.asarray(codes, dtype=int), tuple(labels)


def symbols(records: Sequence, *, names=None, state_mode="recorded") -> Symbols:
    """Accept upstream JSONL dictionaries OR upstream (state, actions) tuples.

    ``recorded`` trusts the caller's *pre-action public* state, ``none`` ignores
    it, and ``last_round`` reconstructs it in a fixed player order. Missing
    actions are masked; malformed non-missing actions fail loudly.
    """
    if state_mode not in ("recorded", "none", "last_round"):
        raise ValueError(f"unsupported state mode: {state_mode}")
    rows = [r if isinstance(r, Mapping) else {"state": r[0], "actions": r[1]}
            for r in records]
    if not rows:
        raise ValueError("cannot analyse an empty episode")
    if any(not isinstance(r.get("actions"), Mapping) for r in rows):
        raise ValueError("each record needs an actions mapping")
    players = set().union(*(r["actions"] for r in rows))
    names = tuple(sorted(players) if names is None else names)
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("player names must be non-empty strings")
    if not players.issubset(set(names)):
        raise ValueError("records contain a player absent from names")
    order = [r.get("round", i + 1) for i, r in enumerate(rows)]
    if any(b <= a for a, b in zip(order, order[1:])):
        raise ValueError("rounds must be strictly increasing")
    moves = np.full((len(rows), len(names)), MISSING, dtype=int)
    states = []
    for t, row in enumerate(rows):
        for i, name in enumerate(names):
            move = row["actions"].get(name)
            if move is not None:
                if not isinstance(move, str) or move not in MOVE_CODES:
                    raise ValueError(f"invalid move {move!r} for {name} at round {t + 1}")
                moves[t, i] = MOVE_CODES[move]
        if state_mode == "none":
            states.append("-")
        elif state_mode == "last_round":
            states.append(moves[t - 1].tolist() if t else "start")
        else:
            if "state" not in row:
                raise ValueError("recorded mode requires a public state in every record")
            states.append(row["state"])
    state, labels = encode_states(states)
    return Symbols(moves, state, moves != MISSING, names, labels, state_mode)


def move_counts(moves: np.ndarray) -> np.ndarray:
    """T x 3 anonymous counts; missing actions contribute to no category."""
    return np.column_stack([(moves == i).sum(axis=1) for i in range(MOVE_ALPHABET)])


def complete_records(sym: Symbols) -> list:
    """Upstream baseline cannot handle missing actions: use complete rounds."""
    return [(int(sym.state[t]), {n: MOVES[sym.moves[t, i]] for i, n in enumerate(sym.names)})
            for t in np.flatnonzero(sym.move_defined.all(axis=1))]


def load_jsonl(path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows
