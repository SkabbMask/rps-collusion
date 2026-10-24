"""Fixed features of each observation view.

These discard information. A detector on them can lower-bound view information,
but a weak detector does not establish that the view has little information."""
import gzip
import hashlib
import json
from collections import Counter
from itertools import combinations

import numpy as np

from rps.detection.pairwise import cmi_from_tables
from rps.observe.features import move_counts
from rps.observe.views import public_messages, relabelled_view, summary_view


def histogram_features(moves):
    counts = move_counts(moves)
    # Include partial-observation counts; total <= N, never impute missing moves.
    atoms = [(a, b, c) for a in range(moves.shape[1] + 1)
             for b in range(moves.shape[1] - a + 1)
             for c in range(moves.shape[1] - a - b + 1)]
    frequencies = Counter(map(tuple, counts.tolist()))
    return np.asarray([frequencies[x] / len(moves) for x in atoms])


def pair_information(sym):
    result = []
    for i, j in combinations(range(sym.N), 2):
        valid = sym.move_defined[:, i] & sym.move_defined[:, j]
        codes = sym.state[valid] * 9 + sym.moves[valid, i] * 3 + sym.moves[valid, j]
        table = np.bincount(codes, minlength=len(sym.state_labels) * 9).reshape(-1, 3, 3)
        result.append(float(cmi_from_tables(table)) if valid.any() else 0.0)
    return np.asarray(result)


def identified_features(sym):
    marginal = np.asarray([(sym.moves == move).mean(axis=0) for move in (-1, 0, 1, 2)]).ravel()
    joint = []
    for i, j in combinations(range(sym.N), 2):
        # Missing is a fourth category; no private failure flags are consumed.
        joint.extend(np.bincount((sym.moves[:, i] + 1) * 4 + sym.moves[:, j] + 1,
                                 minlength=16) / sym.T)
    return np.r_[marginal, joint, pair_information(sym)]


def project(records, sym, view, m, seed):
    base = None
    if view in ("identified", "full", "metadata"):
        base = identified_features(sym)
        payload = {"names": sym.names, "actions": sym.moves.tolist(),
                   "state": [sym.state_labels[s] for s in sym.state]}
        if view != "identified":
            messages = public_messages(records, sym.names)
            lengths = np.asarray([[len(s) for s in row] for row in messages])
            meta = np.r_[(lengths > 0).mean(axis=0), lengths.mean(axis=0), lengths.std(axis=0)]
            base = np.r_[base, meta]
            if view == "full":
                payload["messages"] = messages
                # Stable token hashing is a lossy text feature map, not full-view MI.
                tokens = np.zeros(32)
                for row in messages:
                    for text in row:
                        for token in text.lower().split():
                            bucket = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little") % 32
                            tokens[bucket] += 1 / sym.T
                base = np.r_[base, tokens]
            else:
                payload["message_lengths"] = lengths.tolist()
                payload["message_present"] = (lengths > 0).tolist()
    elif view == "histogram":
        base = histogram_features(sym.moves)
        payload = {"counts": move_counts(sym.moves).tolist()}
    elif view == "summaries":
        summ = summary_view(sym)
        values = np.column_stack([summ.move_entropy_bits, summ.largest_move_share,
                                  summ.all_same, summ.all_three])
        valid = np.isfinite(values)
        # Explicit missingness feature; undefined is not a zero-entropy observation.
        filled = np.where(valid, values, 0)
        mean = filled.sum(axis=0) / np.maximum(1, valid.sum(axis=0))
        variance = np.where(valid, (filled - mean) ** 2, 0).sum(axis=0) / np.maximum(1, valid.sum(axis=0))
        base = np.r_[mean, np.sqrt(variance), valid.mean(axis=0)]
        payload = {"summaries": [[float(x) if np.isfinite(x) else None for x in row] for row in values]}
    elif view == "relabelled":
        public, _ = relabelled_view(sym, m, seed)
        # Aggregate invariant within-block evidence. Never join pseudonym columns
        # across boundaries or consult the evaluator maps.
        pair = np.asarray([np.sort(pair_information(b.sym)) for b in public.blocks])
        base = np.r_[histogram_features(sym.moves), pair.mean(axis=0), pair.max(axis=0)]
        payload = {"blocks": [{"start": b.start, "stop": b.stop, "names": b.sym.names,
                               "actions": b.sym.moves.tolist(),
                               "state": [b.sym.state_labels[s] for s in b.sym.state]} for b in public.blocks]}
    else:
        raise ValueError(f"unknown view: {view}")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return base, {"raw_bytes": len(raw), "gzip_bytes": len(gzip.compress(raw, mtime=0))}
