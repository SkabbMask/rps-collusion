"""RPS observation views adapted from colav/observe/views.py.

There is no price or valuation analogue in RPS. Only explicitly logged
public ``messages`` enter full/metadata; ``signalled`` remains private.
Relabelled blocks have distinct pseudonyms; they must never be concatenated
as if a column represented one persistent player. Hidden maps are returned
separately for evaluation, never attached to a detector's Symbols object.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import Symbols, encode_states, move_counts, symbols


@dataclass(frozen=True)
class IdentifiedView:
    sym: Symbols
    name: str = "identified"
    messages: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True)
class MetadataView:
    sym: Symbols
    lengths: np.ndarray
    present: np.ndarray
    name: str = "metadata"


def public_messages(records, names):
    """Missing instrumentation is unknown, not evidence of silence.

    An explicit empty string means an observed channel with no message.
    Private prompts, replies, policies and covert signals are never read.
    """
    result = []
    for row in records:
        messages = row.get("messages")
        if not isinstance(messages, dict) or set(messages) != set(names):
            raise ValueError("full/metadata require observed public messages for every player and round")
        if any(not isinstance(messages[n], str) for n in names):
            raise ValueError("unobserved or malformed public message")
        result.append(tuple(messages[n] for n in names))
    return tuple(result)


def full_view(records, **kwargs):
    sym = symbols(records, **kwargs)
    return IdentifiedView(sym, "full", public_messages(records, sym.names))


def metadata_view(records, **kwargs):
    sym = symbols(records, **kwargs)
    messages = public_messages(records, sym.names)
    lengths = np.asarray([[len(text) for text in row] for row in messages], dtype=int)
    return MetadataView(sym, lengths, lengths > 0)


@dataclass(frozen=True)
class Block:
    start: int
    stop: int
    sym: Symbols


@dataclass(frozen=True)
class RelabelledView:
    blocks: tuple[Block, ...]
    n_actors: int
    name: str = "relabelled"


@dataclass(frozen=True)
class HistogramView:
    counts: np.ndarray
    name: str = "histogram"


@dataclass(frozen=True)
class SummaryView:
    move_entropy_bits: np.ndarray
    largest_move_share: np.ndarray
    all_same: np.ndarray
    all_three: np.ndarray
    name: str = "summaries"


def identified_view(records, **kwargs) -> IdentifiedView:
    return IdentifiedView(symbols(records, **kwargs))


def relabelled_view(sym: Symbols, m: int, seed: int = 0):
    """Return (public view, evaluator-only pseudonym maps).

    A recorded state could itself expose persistent identity. Require ``none``
    or rebuild the previous joint action within each block for ``last_round``.
    RPS actions are observed immediately, including the first row of a block;
    unlike auction price changes, they do not require linkage to a past ask.
    """
    if not isinstance(m, int) or m < 1:
        raise ValueError("relabel interval must be a positive integer")
    if sym.state_mode == "recorded":
        raise ValueError("relabelled views require none or last_round to avoid identity leakage in state")
    rng = np.random.default_rng(seed)
    blocks, hidden_maps = [], []
    for b, start in enumerate(range(0, sym.T, m)):
        stop = min(start + m, sym.T)
        perm = rng.permutation(sym.N)
        moves = sym.moves[start:stop, perm].copy()
        names = tuple(f"block{b}:player{i}" for i in range(sym.N))
        values = (["-"] * len(moves) if sym.state_mode == "none" else
                  ["start"] + moves[:-1].tolist())
        state, labels = encode_states(values)
        sub = Symbols(moves, state, moves != -1, names, labels, sym.state_mode)
        blocks.append(Block(start, stop, sub))
        hidden_maps.append({names[i]: sym.names[int(perm[i])] for i in range(sym.N)})
    return RelabelledView(tuple(blocks), sym.N), tuple(hidden_maps)


def histogram_view(sym: Symbols) -> HistogramView:
    return HistogramView(move_counts(sym.moves))


def summary_view(sym: Symbols) -> SummaryView:
    counts = move_counts(sym.moves)
    total = counts.sum(axis=1)
    p = np.divide(counts, total[:, None], out=np.zeros_like(counts, dtype=float), where=total[:, None] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(p > 0, p * np.log2(p), 0).sum(axis=1)
    entropy[total == 0] = np.nan
    largest = p.max(axis=1)
    largest[total == 0] = np.nan
    complete = total == sym.N
    return SummaryView(entropy, largest, complete & ((counts > 0).sum(axis=1) == 1),
                       complete & ((counts > 0).sum(axis=1) == 3))
