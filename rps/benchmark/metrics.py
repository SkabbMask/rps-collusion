"""Episode-level performance; naming metrics count nondetections as failures."""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def wilson(successes, total):
    if not total:
        return (float("nan"), float("nan"))
    p, z = successes / total, 1.959963984540054
    den = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / den
    return max(0.0, min(p, centre - half)), min(1.0, max(p, centre + half))


def naming_metrics(estimated, truth):
    est_groups = {frozenset(g) for g in estimated}
    true_groups = {frozenset(g) for g in truth}
    est = set().union(*est_groups) if est_groups else set()
    real = set().union(*true_groups) if true_groups else set()
    pairs = lambda groups: {frozenset(p) for group in groups for p in combinations(sorted(group), 2)}
    ep, tp = pairs(est_groups), pairs(true_groups)
    intersection = len(ep & tp)
    precision = intersection / len(ep) if ep else 0.0
    recall = intersection / len(tp) if tp else 0.0
    return {"exact_groups": int(est_groups == true_groups), "exact_members": int(est == real),
            "jaccard": len(est & real) / len(est | real) if est | real else 1.0,
            "pair_precision": precision, "pair_recall": recall,
            "pair_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def finite_ranks(values):
    """Order-preserving finite encoding, including abstaining/no-edge -inf."""
    array = np.asarray(values, dtype=float)
    array = np.where(np.isnan(array), -np.inf, array)
    _, ranks = np.unique(array, return_inverse=True)
    return ranks.astype(float)


def binary_metrics(rows):
    y = np.asarray([r["positive"] for r in rows], dtype=int)
    pred = np.asarray([r["flag"] for r in rows], dtype=int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    out = {"n_positive": tp + fn, "n_negative": tn + fp, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
           "precision": precision, "recall": recall, "fpr": fpr,
           "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
           "balanced_accuracy": (recall + 1 - fpr) / 2,
           "recall_ci_low": wilson(tp, tp + fn)[0], "recall_ci_high": wilson(tp, tp + fn)[1],
           "fpr_ci_low": wilson(fp, fp + tn)[0], "fpr_ci_high": wilson(fp, fp + tn)[1],
           "abstain_rate": float(np.mean([r["abstain"] for r in rows])),
           "median_ms": float(np.median([r["seconds"] * 1000 for r in rows]))}
    if len(np.unique(y)) == 2:
        scores = finite_ranks([r["score"] for r in rows])
        out.update(auroc=float(roc_auc_score(y, scores)), average_precision=float(average_precision_score(y, scores)))
    else:
        out.update(auroc=float("nan"), average_precision=float("nan"))
    positives = [r for r in rows if r["positive"]]
    for key in ("exact_groups", "exact_members", "jaccard", "pair_precision", "pair_recall", "pair_f1"):
        out[key] = float(np.mean([r[key] for r in positives])) if positives else float("nan")
    return out
