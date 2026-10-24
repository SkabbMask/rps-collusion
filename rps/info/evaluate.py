"""Independent honest fit, honest rank calibration, then held-out view bounds."""
import hashlib
import itertools
import json
import csv
from pathlib import Path

import numpy as np

from rps.benchmark.data import seed_for
from rps.detection.pipeline import rank_threshold, write_json
from rps.observe.features import symbols
from .bounds import decision_bounds
from .exact import VIEWS
from .features import project


def evaluate_views(episodes, output, *, m=25, alpha=0.05, delta=0.05, seed=0,
                   allow_repeated_observations=False):
    if not isinstance(m, int) or m < 1 or not 0 < delta < 1:
        raise ValueError("need positive integer m and confidence delta in (0,1)")
    rank_threshold([], alpha)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "information.json").exists():
        raise ValueError("information output already exists; choose a fresh output directory")
    rows, predictions, frozen, unavailable = [], [], [], []
    seen_ids, seen_observations, seen_groups = set(), set(), set()
    seed_splits = {}
    planned_comparisons = 0
    repeated_observations = 0
    source_hash = hashlib.sha256()
    for group_id, items in itertools.groupby(episodes, key=lambda e: e["group"]["id"]):
        if group_id in seen_groups:
            raise ValueError("groups must be contiguous")
        seen_groups.add(group_id)
        buckets = {v: {s: [] for s in ("edge_fit", "calibration", "test")} for v in VIEWS}
        absent, group, phase = {}, None, 0
        positive_conditions = set()
        for ep in items:
            if ep["id"] in seen_ids:
                raise ValueError("duplicate episode ID")
            seen_ids.add(ep["id"])
            source_hash.update((json.dumps(ep, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
            group = ep["group"] if group is None else group
            if ep["group"] != group:
                raise ValueError("inconsistent group settings")
            current = {"edge_fit": 0, "calibration": 1, "test": 2}.get(ep["split"], -1)
            if current < phase or current < 0:
                raise ValueError("require independent edge_fit, calibration, test order")
            phase = current
            if ep.get("seed") is not None:
                seed_key = (group_id, ep["seed"])
                if seed_key in seed_splits and seed_splits[seed_key] != ep["split"]:
                    raise ValueError("episode seeds may not cross fit/calibration/test splits")
                seed_splits[seed_key] = ep["split"]
            if current < 2 and ep["truth"]:
                raise ValueError("fit/calibration must be independently known honest episodes")
            if current == 2 and ep["truth"]:
                positive_conditions.add((ep["scenario"], ep.get("cp")))
            sym = symbols(ep["records"], state_mode=group["state_mode"])
            if (sym.N, sym.T) != (group["n_players"], group["rounds"]):
                raise ValueError("episode shape disagrees with group")
            if any(len(set(g)) != len(g) or len(g) < 2 or not set(g).issubset(sym.names) for g in ep["truth"]):
                raise ValueError("invalid independent truth labels")
            observed = [sym.names, sym.moves.tolist(), sym.state.tolist(),
                        [r.get("messages") for r in ep["records"]]]
            digest = hashlib.sha256(json.dumps(observed, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if digest in seen_observations:
                if not allow_repeated_observations:
                    raise ValueError("duplicate observable episode (possible split leakage)")
                repeated_observations += 1
            seen_observations.add(digest)
            for view in VIEWS:
                if view in absent:
                    continue
                try:
                    features, volume = project(ep["records"], sym, view, m,
                                               seed_for(seed, "view_channel", len(seen_ids)))
                except ValueError as exc:
                    if view not in ("full", "metadata", "relabelled"):
                        raise
                    absent[view] = str(exc)
                    continue
                buckets[view][ep["split"]].append({"id": ep["id"], "x": features, **volume,
                                                   "positive": bool(ep["truth"]),
                                                   "scenario": ep["scenario"], "cp": ep.get("cp")})
        planned_comparisons += len(VIEWS) * len(positive_conditions)
        for view in VIEWS:
            if view in absent:
                unavailable.append({"group": group_id, "view": view, "reason": absent[view],
                                    "policy": "whole group excluded for this view; no class-dependent row dropping"})
                continue
            bucket = buckets[view]
            if len(bucket["edge_fit"]) < 2 or not bucket["calibration"] or not bucket["test"]:
                raise ValueError("each group needs >=2 honest fit, honest calibration, and held-out test episodes")
            xfit = np.asarray([r["x"] for r in bucket["edge_fit"]])
            centre, scale = xfit.mean(axis=0), xfit.std(axis=0)
            scale[scale < 1e-12] = 1.0
            score = lambda x: float(np.mean(((x - centre) / scale) ** 2))
            threshold = rank_threshold([score(r["x"]) for r in bucket["calibration"]], alpha)
            frozen.append({"group": group_id, "view": view, "group_settings": group,
                           "centre": centre, "scale": scale, "threshold": threshold,
                           "fit_ids": [r["id"] for r in bucket["edge_fit"]],
                           "calibration_ids": [r["id"] for r in bucket["calibration"]]})
            tests = []
            for record in bucket["test"]:
                value = score(record["x"])
                item = {k: v for k, v in record.items() if k != "x"}
                item.update(group=group_id, view=view, score=value, flag=value > threshold)
                tests.append(item)
                predictions.append(item)
            honest = [r for r in tests if not r["positive"] and r["scenario"] == "honest"]
            alternatives = sorted({(r["scenario"], r["cp"]) for r in tests if r["positive"]}, key=str)
            if not honest or not alternatives:
                raise ValueError("need matched honest tests and positive tests per group")
            for scenario, cp in alternatives:
                alt = [r for r in tests if r["positive"] and (r["scenario"], r["cp"]) == (scenario, cp)]
                rows.append({"group": group_id, "view": view, "scenario": scenario, "cp": cp,
                             "tp": sum(r["flag"] for r in alt), "n_pos": len(alt),
                             "fp": sum(r["flag"] for r in honest), "n_neg": len(honest),
                             "threshold": threshold, "feature_dim": len(centre),
                             "mean_gzip_bytes": float(np.mean([r["gzip_bytes"] for r in honest + alt])),
                             "mean_raw_bytes": float(np.mean([r["raw_bytes"] for r in honest + alt]))})
        print(f"{group_id}: view analysis complete", flush=True)
    if not rows:
        raise ValueError("no view comparisons")
    # Simultaneous confidence across every reported view/condition comparison.
    # Shared honest test sets need not be independent for this union bound.
    for row in rows:
        row.update(decision_bounds(row.pop("tp"), row.pop("n_pos"), row.pop("fp"), row.pop("n_neg"),
                                   delta=delta / planned_comparisons))
    result = {"schema_version": 1, "label": "binary configured coalition mechanism, not intent or membership",
              "prior": "balanced classes", "alpha": alpha, "family_confidence": 1 - delta,
              "n_planned_comparisons": planned_comparisons,
              "allow_repeated_observations": allow_repeated_observations,
              "repeated_observations": repeated_observations,
              "relabel_interval": m, "seed": seed, "n_episodes": len(seen_ids),
              "dataset_sha256": source_hash.hexdigest(),
              "detector": "honest-fit diagonal standardized squared-distance anomaly; strict independent rank cutoff",
              "feature_version": 1, "comparisons": rows, "unavailable": unavailable,
              "assumptions": ["independent episodes within each arm", "honest population matches fit/calibration/test",
                              "features and detector fixed before test; known honest fit/calibration labels",
                              "TV/MI lower bounds concern the auditor view through a lossy frozen detector",
                              "a zero lower bound does not establish zero information",
                              "gzip/raw bytes measure volume, not I(C;V)",
                              "no power upper bound can be derived from a TV lower bound"]}
    write_json(output / "information.json", result)
    write_json(output / "view_calibration.json", frozen)
    write_json(output / "view_predictions.json", predictions)
    with (output / "view_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return result
