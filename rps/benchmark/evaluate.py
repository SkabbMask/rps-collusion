"""Streaming evaluation: edge fitting -> null calibration -> held-out tests.

The edge threshold is learned BEFORE score calibration on independent null
episodes. Test labels are used for metrics only. Every artifact is generated
from executed episodes; no pre-filled result tables are used.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rps.detection.pipeline import (DetectionConfig, METHODS, decisions, json_safe,
                                    rank_threshold, score_methods, write_json)
from rps.detection.pairwise import analyse_pairwise
from rps.observe.features import symbols
from .metrics import binary_metrics, naming_metrics


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def freeze(group, cfg, edge_values, calibration_scores, ids, alpha):
    n_cal = len(calibration_scores)
    if not edge_values or not n_cal:
        raise ValueError("every group needs independent edge_fit and calibration episodes")
    edge_threshold = rank_threshold(edge_values, alpha)
    thresholds = {m: rank_threshold([r[m].score for r in calibration_scores], alpha) for m in METHODS}
    thresholds["bonjour"] = cfg.baseline_alpha
    return {"schema_version": 1, "frozen": True, "group": group, "target_fpr": alpha,
            "signature": {k: group[k] for k in ("n_players", "rounds", "state_mode")},
            "detector_config": asdict(cfg), "edge_threshold": edge_threshold, "thresholds": thresholds,
            "n_edge_fit": len(edge_values), "n_calibration": n_cal,
            "edge_fit_ids": ids["edge_fit"], "calibration_ids": ids["calibration"],
            "null_scores": {m: [r[m].score for r in calibration_scores] for m in METHODS},
            "note": "Strict order-statistic thresholds. Conditional on fixed edge fit, marginal rank control requires exchangeable calibration and future honest episodes. No joint guarantee for OR-combining methods."}


def summarise(rows):
    groups = sorted({r["group"] for r in rows})
    comparison = []
    for group in groups:
        group_rows = [r for r in rows if r["group"] == group]
        cases = sorted({(r["scenario"], r["cp"] if r["cp"] is not None else -1) for r in group_rows
                        if r["scenario"] != "honest"})
        for method in METHODS:
            null = [r for r in group_rows if r["method"] == method and r["scenario"] == "honest"]
            for scenario, cp in cases:
                alt = [r for r in group_rows if r["method"] == method and r["scenario"] == scenario
                       and (r["cp"] if r["cp"] is not None else -1) == cp]
                # A second honest population is a robustness/FPR test; do not
                # average it together with the matched null reference.
                sample = alt if not any(r["positive"] for r in alt) else null + alt
                comparison.append({"group": group, "scenario": scenario, "cp": cp,
                                   "method": method, **binary_metrics(sample)})
    return comparison


def evaluate(episodes, config, output, *, save_dataset=True, source="generated"):
    output = Path(output)
    if (output / "report.md").exists():
        raise ValueError(f"{output} already has a report; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration").mkdir(exist_ok=True)
    write_json(output / "config.json", config)
    cfg = DetectionConfig(**config.get("detector", {}))
    alpha = config.get("alpha", 0.05)
    rank_threshold([], alpha)  # validate before expensive work
    start = time.perf_counter()
    rows, calibrations, group_counts, example = [], {}, {}, None
    seen_ids, seen_hashes, seen_groups = set(), set(), set()
    dataset_hash = hashlib.sha256()
    writer = gzip.open(output / "episodes.jsonl.gz", "wt", encoding="utf-8") if save_dataset else None
    try:
        for group_id, group_iter in itertools.groupby(episodes, key=lambda e: e["group"]["id"]):
            if group_id in seen_groups:
                raise ValueError("dataset groups must be contiguous")
            if not group_id.replace("_", "").replace("-", "").isalnum():
                raise ValueError("group ids must contain only letters, numbers, underscores or hyphens")
            seen_groups.add(group_id)
            edge_values, cal_scores = [], []
            ids = {"edge_fit": [], "calibration": [], "test": []}
            phase, frozen, group = 0, None, None
            for ep in group_iter:
                if ep["id"] in seen_ids:
                    raise ValueError(f"duplicate episode id: {ep['id']}")
                seen_ids.add(ep["id"])
                group = ep["group"] if group is None else group
                if ep["group"] != group:
                    raise ValueError(f"inconsistent settings inside group {group_id}")
                split = ep["split"]
                current = {"edge_fit": 0, "calibration": 1, "test": 2}.get(split, -1)
                if current < phase or current < 0:
                    raise ValueError("within a group require edge_fit, calibration, then test order")
                phase = current
                sym = symbols(ep["records"], state_mode=group["state_mode"])
                if sym.N != group["n_players"] or sym.T != group["rounds"]:
                    raise ValueError("episode shape disagrees with calibration group")
                # Hash only the observable trajectory; changing private labels
                # cannot disguise a calibration/test duplicate.
                observable = json.dumps([sym.names, sym.moves.tolist(), sym.state.tolist()], separators=(",", ":"))
                digest = hashlib.sha256(observable.encode()).hexdigest()
                if digest in seen_hashes:
                    raise ValueError(f"duplicate observable trajectory (possible leakage): {ep['id']}")
                seen_hashes.add(digest)
                truth = ep["truth"]
                if any(len(set(g)) != len(g) or len(g) < 2 or not set(g).issubset(sym.names) for g in truth):
                    raise ValueError("truth groups must contain distinct known players, with size >= 2")
                if split != "test" and truth:
                    raise ValueError("only independently known honest episodes may calibrate thresholds")
                ids[split].append(ep["id"])
                encoded = json.dumps(ep, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                dataset_hash.update(encoded.encode())
                if writer:
                    writer.write(encoded)
                pair_start = time.perf_counter()
                pairwise = analyse_pairwise(sym, cfg.pairwise())
                pair_time = time.perf_counter() - pair_start
                if split == "edge_fit":
                    edge_values.append(pairwise.max_z)
                    continue
                if not edge_values:
                    raise ValueError("missing edge-fit episodes")
                edge_threshold = rank_threshold(edge_values, alpha)
                raw, _ = score_methods(sym, cfg, edge_threshold, evidence=pairwise, evidence_seconds=pair_time)
                if split == "calibration":
                    cal_scores.append(raw)
                    continue
                if frozen is None:
                    frozen = freeze(group, cfg, edge_values, cal_scores, ids, alpha)
                    write_json(output / "calibration" / f"{group_id}.json", frozen)
                    calibrations[group_id] = frozen
                    print(f"{group_id}: froze {len(edge_values)} edge-fit + {len(cal_scores)} calibration episodes; evaluating test games", flush=True)
                decided = decisions(raw, frozen)
                for method, decision in decided.items():
                    rows.append({"episode_id": ep["id"], "group": group_id,
                                 "scenario": ep["scenario"], "cp": ep.get("cp"), "method": method,
                                 "positive": bool(truth), **decision, **naming_metrics(decision["groups"], truth),
                                 "truth": truth, "n_states": len(sym.state_labels),
                                 "supported_pair_fraction": float(np.isfinite(pairwise.z[np.triu_indices(sym.N, 1)]).mean())})
                if example is None and ep["scenario"] == "star3" and ep.get("cp") == 1 and sym.T == 300:
                    example = {"id": ep["id"], "names": sym.names, "z": pairwise.z, "truth": truth,
                               "edge_threshold": edge_threshold, "decisions": decided}
                    write_json(output / "example_graph.json", example)
                if len(ids["test"]) % 200 == 0:
                    print(f"{group_id}: {len(ids['test'])} test episodes finished", flush=True)
            if frozen is None:
                raise ValueError(f"group {group_id} must contain test episodes")
            group_counts[group_id] = {split: len(values) for split, values in ids.items()}
            print(f"{group_id}: complete ({group_counts[group_id]})", flush=True)
    finally:
        if writer:
            writer.close()
    if not rows:
        raise ValueError("no test results")
    comparisons = summarise(rows)
    write_csv(output / "predictions.csv", rows)
    write_csv(output / "comparison.csv", comparisons)
    threshold_rows = [{"group": g, "method": m, "edge_threshold": c["edge_threshold"],
                       "threshold": c["thresholds"][m], "n_edge_fit": c["n_edge_fit"], "n_calibration": c["n_calibration"]}
                      for g, c in calibrations.items() for m in METHODS]
    write_csv(output / "thresholds.csv", threshold_rows)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "networkx", "matplotlib", "scikit-learn")}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = "archive checkout (see ATTRIBUTION.md)"
    provenance = {"source": source, "base_revision": commit, "python": platform.python_version(),
                  "platform": platform.platform(), "versions": versions, "dataset_sha256": dataset_hash.hexdigest(),
                  "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  "episodes": len(seen_ids), "test_episodes": len(rows) // len(METHODS), "group_counts": group_counts,
                  "evaluation_seconds": time.perf_counter() - start}
    write_json(output / "provenance.json", provenance)
    from .report import generate_report
    generate_report(output, rows, comparisons, calibrations, config, provenance, example)
    return rows, comparisons
