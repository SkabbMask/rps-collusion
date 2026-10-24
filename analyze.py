"""Run the collusion detectors on a simulation log.

Detectors (identified or relabelled view):
  paper rule      Bonjour et al.: Gamma(i;j) and Gamma(j;i) >= tau (bits) for exactly one pair
  graph clusters  connected components of the edges z_ij >= z_alpha; accepted if Q(S) > q_95
  ML decoder      maximum-likelihood coalition of unknown size k with plug-in mu; accepted above its null
Histogram view (no identities): G test of the per-round move counts against a multinomial.

z_alpha, q_95 and the decoder threshold are calibrated on honest surrogates of the same episode (each
player's moves shuffled independently within state cells), at the 95% level by default.

Usage:
    python analyze.py runs/latest.jsonl
    python analyze.py runs/6p3c.jsonl --view relabelled --m 50
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from detection.decoder import decode
from detection.graph import clusters, edge_set, recovery_metrics
from detection.pairwise import PairwiseParams, analyse_pairwise, null_symbols
from detection.pooled import PooledParams, pooled_statistics
from observe.features import Episode, Symbols
from observe.views import histogram_view, identified_view, relabelled_view


@dataclass
class AnalysisParams:
    pairwise: PairwiseParams = field(default_factory=PairwiseParams)
    pooled: PooledParams = field(default_factory=PooledParams)
    n_null: int = 50          # honest surrogates used for calibration
    level: float = 0.95       # calibration quantile
    k_max: int = 5            # largest coalition the decoder considers


def calibrate(sym: Symbols, ap: AnalysisParams) -> Dict[str, float]:
    """Null quantiles of max z, of the best cluster score and of the decoder score."""
    rng = np.random.default_rng(ap.pairwise.seed + 1)
    zs = [analyse_pairwise(null_symbols(sym, rng), ap.pairwise, seed_offset=k + 1).z for k in range(ap.n_null)]
    max_z = [np.nanmax(z) if not np.isnan(z).all() else 0.0 for z in zs]
    z_alpha = float(np.quantile(max_z, ap.level))
    q, dec = [], []
    for z in zs:
        cr = clusters(z, edge_set(z, z_alpha), use_louvain=False)
        q.append(cr.best_q if cr.best else 0.0)
        d = decode(z, None, None, k_max=min(ap.k_max, sym.N), mode="plugin")
        dec.append(d.k_scores[d.k])
    return {"z_alpha": z_alpha, "q_95": float(np.quantile(q, ap.level)),
            "decoder": float(np.quantile(dec, ap.level))}


def analyse_block(sym: Symbols, ap: AnalysisParams) -> Dict[str, Any]:
    pr = analyse_pairwise(sym, ap.pairwise)
    th = calibrate(sym, ap)
    cr = clusters(pr.z, edge_set(pr.z, th["z_alpha"]))
    d = decode(pr.z, None, None, k_max=min(ap.k_max, sym.N), mode="plugin")
    score = d.k_scores[d.k]
    return {
        "pairwise": pr, "thresholds": th,
        "cluster": cr.best, "cluster_q": cr.best_q,
        "cluster_flag": cr.best is not None and cr.best_q > th["q_95"],
        "louvain": cr.louvain,
        "decoder": d.members, "decoder_score": score, "decoder_flag": score > th["decoder"],
    }


def analyse(ep: Episode, ap: AnalysisParams, view: str = "identified", m: int = 50,
            view_seed: int = 0) -> Dict[str, Any]:
    v = identified_view(ep) if view == "identified" else relabelled_view(ep, m, view_seed)
    blocks = []
    for b in v.blocks:
        res = analyse_block(b.sym, ap)
        res["block"] = b
        blocks.append(res)
    return {"view": v, "blocks": blocks,
            "pooled": pooled_statistics(histogram_view(ep), ap.pooled)}


# ---- report ------------------------------------------------------------------------------------
def _names(ep: Episode, members: Optional[Set[int]], hidden_map: Dict[int, int]) -> str:
    if not members:
        return "-"
    return "{" + ", ".join(sorted(ep.names[hidden_map[c]] for c in members)) + "}"


def _matrix(title: str, M: np.ndarray, labels: List[str], fmt: str) -> None:
    print(title)
    print("      " + "".join(f"{l:>8}" for l in labels))
    for i, l in enumerate(labels):
        cells = ["       -" if i == j or np.isnan(M[i, j]) else format(M[i, j], fmt).rjust(8) for j in range(len(labels))]
        print(f"  {l:<4}" + "".join(cells))


def _recovery(ep: Episode, members: Optional[Set[int]], hidden_map: Dict[int, int]) -> Optional[str]:
    coalitions = ep.truth.get("coalitions")
    if not coalitions:
        return None
    est = {hidden_map[c] for c in members} if members else None
    best = max((recovery_metrics(est, {ep.names.index(n) for n in co}) for co in coalitions),
               key=lambda r: r["jaccard"])
    return f"exact={int(best['exact'])} jaccard={best['jaccard']:.2f}"


def print_report(ep: Episode, res: Dict[str, Any], ap: AnalysisParams) -> None:
    v = res["view"]
    print(f"\nView: {v.name} ({ep.T} rounds, {ep.N} players"
          + (f", {len(v.blocks)} blocks, anonymity {v.anonymity_entropy_bits:.2f} bits)" if v.name != "identified" else ")"))
    if v.name != "identified":
        print("(detectors see pseudonyms only; names below are mapped back for evaluation)")
    for r in res["blocks"]:
        b, pr, th = r["block"], r["pairwise"], r["thresholds"]
        hm = b.hidden_map
        if v.name == "identified":
            labels = ep.names
            _matrix("Individual influence gamma(i;j), bits:", pr.gamma, labels, ".4f")
            _matrix("Net influence Gamma(i;j) (row i on column j):", pr.net, labels, "+.4f")
            _matrix(f"Permutation z (R={ap.pairwise.R}):", pr.z, labels, ".1f")
        else:
            print(f"\n-- block {b.index}: rounds {b.start + 1}-{b.stop}")
        pairs = ", ".join(_names(ep, {i, j}, hm) for i, j in pr.published_pairs)
        print(f"Paper rule (tau={ap.pairwise.tau} bits): {pr.published_outcome.value}" + (f" {pairs}" if pairs else ""))
        print(f"Calibrated on {ap.n_null} honest surrogates ({ap.level:.0%}): z_alpha={th['z_alpha']:.2f}, "
              f"q_95={th['q_95']:.1f}, decoder={th['decoder']:.1f}")
        rec = _recovery(ep, r["cluster"], hm)
        if r["cluster"] is None:
            print("Graph cluster: none (no pair with z >= z_alpha)")
        else:
            print(f"Graph cluster: {_names(ep, r['cluster'], hm)} Q={r['cluster_q']:.1f} -> "
                  f"{'COLLUSION' if r['cluster_flag'] else 'not significant'}" + (f"   [{rec}]" if rec and r["cluster_flag"] else ""))
        if r["louvain"] and len(r["louvain"]) > 1:
            print("  Louvain communities: " + ", ".join(f"{_names(ep, c, hm)} Q={q:.1f}" for c, q in r["louvain"]))
        rec = _recovery(ep, r["decoder"], hm)
        print(f"ML decoder (k unknown): {_names(ep, r['decoder'], hm)} score={r['decoder_score']:.1f} -> "
              f"{'COLLUSION' if r['decoder_flag'] else 'not significant'}" + (f"   [{rec}]" if rec and r["decoder_flag"] else ""))
    pooled = res["pooled"]
    print(f"\nHistogram view (no identities): G={pooled['g']:.1f}, z={pooled['g_z']:.1f}, p={pooled['g_p']:.3f} "
          f"(also sensitive to honest players with different move biases)")
    if ep.truth.get("coalitions"):
        print("Ground truth: " + ", ".join("{" + ", ".join(c) + "}" for c in ep.truth["coalitions"]))


def main():
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("log")
    ap_.add_argument("--view", choices=["identified", "relabelled"], default="identified")
    ap_.add_argument("--m", type=int, default=50, help="rounds per pseudonym block (relabelled view)")
    ap_.add_argument("--view-seed", type=int, default=0)
    ap_.add_argument("--tau", type=float, default=0.05, help="paper threshold on Gamma (bits)")
    ap_.add_argument("--R", type=int, default=200, help="permutations per pair")
    ap_.add_argument("--n-null", type=int, default=50, help="honest surrogates for calibration")
    ap_.add_argument("--level", type=float, default=0.95)
    ap_.add_argument("--k-max", type=int, default=5)
    args = ap_.parse_args()

    ep = Episode.load(args.log)
    ap = AnalysisParams(PairwiseParams(R=args.R, tau=args.tau), PooledParams(R=args.R),
                        n_null=args.n_null, level=args.level, k_max=args.k_max)
    print_report(ep, analyse(ep, ap, args.view, args.m, args.view_seed), ap)


if __name__ == "__main__":
    main()
