"""Public API, independent null calibration, and decisions for four methods."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from rps.detect import detect as upstream_detect
from rps.observe.features import Symbols, complete_records
from .decoder import decode
from .graph import clusters, edge_set
from .pairwise import PairwiseParams, analyse_pairwise

METHODS = ("bonjour", "graph_components", "graph_louvain", "decoder")
METHOD_LABELS = {"bonjour": "Original Bonjour", "graph_components": "Graph components",
                 "graph_louvain": "Graph Louvain", "decoder": "ML decoder"}


@dataclass(frozen=True)
class DetectionConfig:
    permutations: int = 99
    min_pair_rounds: int = 20
    min_state_rounds: int = 10
    k_max: int = 5
    max_exhaustive: int = 200_000
    seed: int = 20261024
    baseline_alpha: float = 0.05       # nats: unchanged target-repository default

    def __post_init__(self):
        PairwiseParams(self.permutations, self.min_pair_rounds, self.min_state_rounds, self.seed)
        if self.k_max < 2 or self.max_exhaustive < 1:
            raise ValueError("invalid decoder search budget")
        if not math.isfinite(self.baseline_alpha) or self.baseline_alpha <= 0:
            raise ValueError("baseline threshold must be positive and finite")

    def pairwise(self):
        return PairwiseParams(self.permutations, self.min_pair_rounds, self.min_state_rounds, self.seed)


@dataclass
class RawResult:
    score: float
    candidates: list[tuple[frozenset[str], float]]
    abstain: bool
    seconds: float
    details: dict = field(default_factory=dict)


def score_methods(sym: Symbols, cfg: DetectionConfig, edge_threshold: float, *, evidence=None,
                  evidence_seconds=None):
    start = time.perf_counter()
    evidence = evidence if evidence is not None else analyse_pairwise(sym, cfg.pairwise())
    pair_seconds = time.perf_counter() - start if evidence_seconds is None else evidence_seconds
    result = {}
    start = time.perf_counter()
    records = complete_records(sym)
    if len(records) >= 2:
        _, net, pairs = upstream_detect(records, list(sym.names), cfg.baseline_alpha)
        value = max(min(net[i, j], net[j, i]) for i in sym.names for j in sym.names if i < j)
        cands = [(frozenset((i, j)), min(net[i, j], net[j, i])) for i, j in pairs]
        result["bonjour"] = RawResult(value, cands, False, time.perf_counter() - start,
                                      {"complete_rounds": len(records)})
    else:
        result["bonjour"] = RawResult(float("-inf"), [], True, time.perf_counter() - start)
    for extractor in ("components", "louvain"):
        start = time.perf_counter()
        cands = clusters(evidence.z, edge_set(evidence.z, edge_threshold), extractor=extractor, seed=cfg.seed)
        named = [(frozenset(sym.names[i] for i in c.members), c.score) for c in cands]
        result[f"graph_{extractor}"] = RawResult(cands[0].score if cands else float("-inf"), named,
                                                 not evidence.available, pair_seconds + time.perf_counter() - start)
    start = time.perf_counter()
    decoded = decode(evidence.z, k_max=cfg.k_max, max_exhaustive=cfg.max_exhaustive)
    result["decoder"] = RawResult(
        decoded.score if decoded else float("-inf"),
        [(frozenset(sym.names[i] for i in decoded.members), decoded.score)] if decoded else [],
        decoded is None, pair_seconds + time.perf_counter() - start,
        {"k": decoded.k, "mu": decoded.mu, "search": decoded.search, "k_scores": decoded.k_scores} if decoded else {})
    return result, evidence


def rank_threshold(values, alpha):
    """Strict exceedance of ceil((n+1)*(1-alpha))-th order statistic.

    Missing/no-candidate scores count as -inf, NOT dropped. Too few null
    episodes => +inf (never reject), rather than claiming an unattainable FPR.
    For a fixed score and exchangeable episodes this is a marginal guarantee.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    xs = sorted(-math.inf if v is None or math.isnan(v) else float(v) for v in values)
    rank = math.ceil((len(xs) + 1) * (1 - alpha))
    return xs[rank - 1] if 0 < rank <= len(xs) else math.inf


def decisions(raw, calibration):
    out = {}
    for method, result in raw.items():
        threshold = calibration["thresholds"][method]
        accepted = [(s, v) for s, v in result.candidates
                    if (v >= threshold if method == "bonjour" else v > threshold)]
        out[method] = {"flag": bool(accepted) and not result.abstain,
                       "abstain": result.abstain, "score": result.score,
                       "groups": [sorted(s) for s, _ in accepted] if not result.abstain else [],
                       "seconds": result.seconds, "details": result.details}
    return out


def validate_calibration(calibration, sym, cfg):
    if calibration.get("schema_version") != 1 or not calibration.get("frozen"):
        raise ValueError("unrecognised or unfrozen calibration")
    expected = calibration["signature"]
    observed = {"n_players": sym.N, "rounds": sym.T, "state_mode": sym.state_mode}
    if expected != observed or calibration["detector_config"] != asdict(cfg):
        raise ValueError(f"calibration mismatch: expected {expected}, received {observed}; detector settings must also match")


def detect_episode(sym, calibration):
    cfg = DetectionConfig(**calibration["detector_config"])
    validate_calibration(calibration, sym, cfg)
    raw, evidence = score_methods(sym, cfg, calibration["edge_threshold"])
    return decisions(raw, calibration), evidence


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    def restore(value):
        if isinstance(value, dict):
            return {k: restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore(v) for v in value]
        if value == "Infinity":
            return math.inf
        if value == "-Infinity":
            return -math.inf
        return value
    return restore(data)
