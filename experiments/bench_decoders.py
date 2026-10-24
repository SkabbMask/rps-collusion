"""Compare coalition decoders on scripted RPS episodes (random players, rule-based assistants).

Usage: python experiments/bench_decoders.py   (about 1.5 minutes)
"""
import os
import sys
import random
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from detection.decoder import decode, grow_from_top_pair
from detection.graph import clusters, edge_set
from detection.pairwise import PairwiseParams, analyse_pairwise, null_symbols
from observe.features import Episode, symbols
from rps.game import MOVES, assist_move

T, N_NULL, EPISODES, LEVEL = 200, 30, 25, 0.95


def episode(n, n_assist, cp, seed):
    rng = random.Random(seed)
    names = [chr(65 + i) for i in range(n)]
    rounds = []
    for _ in range(T):
        acts = {nm: rng.choice(MOVES) for nm in names}
        for a in names[1:1 + n_assist]:
            if rng.random() < cp:
                acts[a] = assist_move(acts["A"])
        rounds.append({"state": "-", "actions": acts})
    truth = set(range(1 + n_assist)) if n_assist and cp > 0 else None
    return Episode(f"e{seed}", names, rounds), truth


def grow_fixed(z, frac):
    """First version: threshold fixed at frac * z(top pair)."""
    zz = np.nan_to_num(z, nan=0.0)
    N = len(zz)
    i, j = np.unravel_index(np.argmax(np.where(np.triu(np.ones((N, N), bool), 1), zz, -np.inf)), zz.shape)
    S, z0 = [int(i), int(j)], zz[i, j]
    while len(S) < N:
        out = [c for c in range(N) if c not in S]
        best = max(out, key=lambda c: zz[c, S].sum())
        if zz[best, S].mean() < frac * z0:
            break
        S.append(best)
    return set(S)


def ml(z, n):
    d = decode(z, None, None, k_max=min(5, n), mode="plugin")
    return d.members, d.k_scores[d.k]


def run_episode(ep, p):
    sym = symbols(ep)
    z = analyse_pairwise(sym, p).z
    rng = np.random.default_rng(7)
    nz = [analyse_pairwise(null_symbols(sym, rng), p, seed_offset=k + 1).z for k in range(N_NULL)]
    z_alpha = np.quantile([np.nanmax(x) for x in nz], LEVEL)
    q95 = np.quantile([(lambda c: c.best_q if c.best else 0.0)(clusters(x, edge_set(x, z_alpha), use_louvain=False)) for x in nz], LEVEL)
    ml_thr = np.quantile([ml(x, sym.N)[1] for x in nz], LEVEL)
    top = np.nanmax(z)
    cr = clusters(z, edge_set(z, z_alpha), use_louvain=False)
    m_set, m_score = ml(z, sym.N)
    out = {
        "ML decoder (colav)": (m_set, m_score > ml_thr),
        "graph cluster": (cr.best, cr.best is not None and cr.best_q > q95),
    }
    for f in (0.5, 0.25):
        out[f"growth fixed f={f}"] = (grow_fixed(z, f), top >= z_alpha)
        out[f"growth updating f={f}"] = (grow_from_top_pair(z, f).members, top >= z_alpha)
    return out


def main():
    p = PairwiseParams(R=100)
    scenarios = [(6, 1), (6, 2), (8, 3)]
    cps = [0.2, 0.35, 0.5]
    for n, k in scenarios:
        print(f"\n### {n} players, primary + {k} assistant(s), T={T}, {EPISODES} episodes per CP")
        stats = defaultdict(lambda: defaultdict(list))
        for cp in cps + [0.0]:
            for e in range(EPISODES):
                ep, truth = episode(n, k if cp else 0, cp, seed=1000 * n + 100 * k + int(cp * 100) * 7 + e)
                for name, (S, flag) in run_episode(ep, p).items():
                    if cp == 0:
                        stats[name]["fpr"].append(flag)
                    else:
                        stats[name][f"det{cp}"].append(flag)
                        stats[name][f"exact{cp}"].append(bool(flag and S == truth))
                        jac = len(S & truth) / len(S | truth) if (flag and S) else 0.0
                        stats[name][f"jac{cp}"].append(jac)
        hdr = "detector".ljust(24) + "".join(f"  CP={cp}: det/exact/jacc".ljust(30) for cp in cps) + "  false alarm"
        print(hdr)
        for name, s in stats.items():
            row = name.ljust(24)
            for cp in cps:
                row += f"  {np.mean(s[f'det{cp}']):.2f} / {np.mean(s[f'exact{cp}']):.2f} / {np.mean(s[f'jac{cp}']):.2f}".ljust(30)
            print(row + f"  {np.mean(s['fpr']):.2f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
