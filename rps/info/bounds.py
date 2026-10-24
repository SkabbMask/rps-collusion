"""Balanced binary experiment C=honest/coalition; all information is in bits.

Exact bounds require known view distributions. Held-out decision bounds use
independent episodes within each arm and a detector fixed before testing.
"""
import math

from scipy.stats import beta


def h2(p):
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def decision_bounds(tp, n_pos, fp, n_neg, delta=0.05):
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    if n_pos < 1 or n_neg < 1 or not 0 <= tp <= n_pos or not 0 <= fp <= n_neg:
        raise ValueError("need valid counts in both classes")
    t, f = tp / n_pos, fp / n_neg
    lower_t = float(beta.ppf(delta / 2, tp, n_pos - tp + 1)) if tp else 0.0
    upper_f = float(beta.ppf(1 - delta / 2, fp + 1, n_neg - fp)) if fp < n_neg else 1.0
    d = max(0.0, lower_t - upper_f)
    return {"n_positive": n_pos, "n_negative": n_neg, "tp": tp, "fp": fp,
            "tpr": t, "fpr": f, "tpr_lower": lower_t, "fpr_upper": upper_f,
            "tv_plugin_detector": max(0.0, t - f), "tv_lower_confidence": d,
            "mi_decision_bits_plugin": max(0.0, h2((t + f) / 2) - (h2(t) + h2(f)) / 2),
            "mi_view_bits_lower_confidence": 1 - h2((1 - d) / 2),
            "bayes_error_upper_confidence": (1 - d) / 2,
            "delta_per_comparison": delta,
            "kind": "held_out_decision_confidence_lower_bound"}


def exact_bounds(p0, p1, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if any(v < 0 or not math.isfinite(v) for p in (p0, p1) for v in p.values()):
        raise ValueError("PMFs must be finite and nonnegative")
    if any(not math.isclose(sum(p.values()), 1.0, abs_tol=1e-9) for p in (p0, p1)):
        raise ValueError("PMFs must sum to one")
    keys = set(p0) | set(p1)
    tv, mi, kl = 0.0, 0.0, 0.0
    for key in keys:
        a, b = p0.get(key, 0.0), p1.get(key, 0.0)
        tv += abs(a - b) / 2
        mixture = (a + b) / 2
        if a:
            mi += a * math.log2(a / mixture) / 2
        if b:
            mi += b * math.log2(b / mixture) / 2
            kl += b * math.log2(b / a) if a else math.inf
    # Randomized Neyman-Pearson test: order atoms by likelihood ratio.
    budget, power = alpha, 0.0
    for key in sorted(keys, key=lambda k: p1.get(k, 0) / p0[k] if p0.get(k, 0) else math.inf,
                      reverse=True):
        a, b = p0.get(key, 0.0), p1.get(key, 0.0)
        fraction = min(1.0, budget / a) if a else 1.0
        power += fraction * b
        budget = max(0.0, budget - fraction * a)
    tv = min(1.0, tv)
    return {"tv": tv, "bayes_error": (1 - tv) / 2, "mi_bits": max(0.0, mi),
            "kl_coalition_honest_bits": kl, "power_upper_alpha_plus_tv": min(1.0, alpha + tv),
            "optimal_power_at_alpha": min(1.0, power), "alpha": alpha,
            "kind": "exact_enumeration_up_to_float_roundoff"}
