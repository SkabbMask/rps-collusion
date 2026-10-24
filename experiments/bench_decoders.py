"""Compare coalition detectors on scripted RPS episodes (random players, rule-based assistants).

Scenarios: a primary A with 1-3 assistants that play the move A beats with probability CP, and two
"roped-in" scenarios where honest player D copies coalition members now and then (D is not part of
the true coalition). Reports, per CP: detection rate / exact recovery / mean Jaccard, and the false
alarm rate on honest episodes (CP = 0).

Usage: python experiments/bench_decoders.py   (a few minutes)
"""
import os
import random
import sys
import zlib
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from detection.decoder import decode, grow_from_top_pair
from detection.graph import _components, cluster_score, clusters, edge_set
from detection.pairwise import PairwiseParams, analyse_pairwise, null_symbols
from observe.features import Episode, symbols
from rps.game import MOVES, assist_move

T, N_NULL, EPISODES, LEVEL = 200, 30, 25, 0.95
CPS = [0.2, 0.35, 0.5]

# name -> (players, assistants, hanger mode, hanger probability)
SCENARIOS = {
    "A + 1 assistant (6 players)": (6, 1, None, 0),
    "A + 2 assistants (6 players)": (6, 2, None, 0),
    "A + 3 assistants (8 players)": (8, 3, None, 0),
    "A + 2 assistants, D copies B 30% (6 players)": (6, 2, "one", 0.3),
    "A + 2 assistants, D copies a member 40% (6 players)": (6, 2, "all", 0.4),
}

GRAPH_VARIANTS = {
    "graph, sum Q (old)": None,
    "graph, mean z": (1, 0.0),
    "graph, mean z, link>=0.25 top": (1, 0.25),
    "graph, mean z, link>=0.5 top": (1, 0.5),
    "graph, mean z, min 2 links": (2, 0.0),
}


def episode(n, n_assist, cp, hanger, h, seed):
    rng = random.Random(seed)
    names = [chr(65 + i) for i in range(n)]
    members = names[:1 + n_assist]
    d = names[1 + n_assist]                      # first honest player after the coalition
    rounds = []
    for _ in range(T):
        acts = {nm: rng.choice(MOVES) for nm in names}
        for a in members[1:]:
            if rng.random() < cp:
                acts[a] = assist_move(acts["A"])
        if hanger and rng.random() < h:
            acts[d] = acts["B"] if hanger == "one" else acts[rng.choice(members)]
        rounds.append({"state": "-", "actions": acts})
    truth = set(range(1 + n_assist)) if cp > 0 else None
    return Episode(f"e{seed}", names, rounds), truth


def old_graph(z, edges):
    cands = [(c, cluster_score(z, c)) for c in _components(len(z), edges) if len(c) >= 2]
    return max(cands, key=lambda x: x[1]) if cands else (None, 0.0)


def new_graph(z, edges, v):
    cr = clusters(z, edges, use_louvain=False, min_links=v[0], link_frac=v[1])
    return cr.best, (cr.best_q if cr.best else 0.0)


def ml(z, n):
    d = decode(z, None, None, k_max=min(5, n), mode="plugin")
    return d.members, d.k_scores[d.k]


def run_episode(ep, p):
    sym = symbols(ep)
    z = analyse_pairwise(sym, p).z
    rng = np.random.default_rng(7)
    nz = [analyse_pairwise(null_symbols(sym, rng), p, seed_offset=k + 1).z for k in range(N_NULL)]
    z_alpha = np.quantile([np.nanmax(x) for x in nz], LEVEL)
    out = {}
    for name, v in GRAPH_VARIANTS.items():
        f = (lambda x: old_graph(x, edge_set(x, z_alpha))) if v is None else (lambda x, v=v: new_graph(x, edge_set(x, z_alpha), v))
        thr = np.quantile([f(x)[1] for x in nz], LEVEL)
        S, score = f(z)
        out[name] = (S, S is not None and score > thr)
    ml_thr = np.quantile([ml(x, sym.N)[1] for x in nz], LEVEL)
    m_set, m_score = ml(z, sym.N)
    out["ML decoder (colav)"] = (m_set, m_score > ml_thr)
    out["growth decoder f=0.25"] = (grow_from_top_pair(z, 0.25).members, np.nanmax(z) >= z_alpha)
    return out


def main():
    p = PairwiseParams(R=100)
    for title, (n, k, hanger, h) in SCENARIOS.items():
        print(f"\n### {title}, T={T}, {EPISODES} episodes per CP")
        stats = defaultdict(lambda: defaultdict(list))
        for cp in CPS + [0.0]:
            for e in range(EPISODES):
                ep, truth = episode(n, k, cp, hanger, h, seed=zlib.crc32(f"{title}|{cp}|{e}".encode()))
                for name, (S, flag) in run_episode(ep, p).items():
                    if cp == 0:
                        stats[name]["fpr"].append(flag)
                        continue
                    stats[name][f"det{cp}"].append(flag)
                    stats[name][f"exact{cp}"].append(bool(flag and S == truth))
                    stats[name][f"jac{cp}"].append(len(S & truth) / len(S | truth) if (flag and S) else 0.0)
        print("detector".ljust(32) + "".join(f"CP={cp}: det/exact/jacc".ljust(26) for cp in CPS) + "false alarm*")
        for name, s in stats.items():
            row = name.ljust(32)
            for cp in CPS:
                row += f"{np.mean(s[f'det{cp}']):.2f} / {np.mean(s[f'exact{cp}']):.2f} / {np.mean(s[f'jac{cp}']):.2f}".ljust(26)
            print(row + f"{np.mean(s['fpr']):.2f}")
        sys.stdout.flush()
    print("\n* CP = 0, but in the roped-in scenarios D still copies B / a member, so 'false alarm' there means "
          "flagging that non-collusive dependence.")


if __name__ == "__main__":
    main()
