"""Generate traceable Markdown, CSV tables and publication-friendly SVG/PNG plots."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import roc_curve

from rps.detection.graph import edge_set
from rps.detection.pipeline import METHODS, METHOD_LABELS
from .metrics import binary_metrics, finite_ranks, wilson

COLORS = {"bonjour": "#667085", "graph_components": "#007f86", "graph_louvain": "#e49a1b", "decoder": "#7354ad", "growth": "#d4513c"}
STYLE = {"bonjour": "--", "graph_components": "-", "graph_louvain": ":", "decoder": "-.", "growth": "--"}


def _number(x, digits=3):
    return f"{x:.{digits}f}" if x is not None and math.isfinite(float(x)) else "—"


def _rate_ci(rate, lo, hi):
    return f"{rate:.1%} [{lo:.1%}, {hi:.1%}]" if math.isfinite(rate) else "—"


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def _save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def figures(output, rows, comparisons, summary, example):
    methods = tuple(summary)
    dest = output / "figures"
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "white"})
    cells = sorted({c["group"] for c in comparisons if c["scenario"] == "pair" and c["group"].endswith("_none")})
    paths = []
    if cells:
        fig, axes = plt.subplots(math.ceil(len(cells) / 2), 2, figsize=(11, 4 * math.ceil(len(cells) / 2)), squeeze=False)
        for ax, group in zip(axes.ravel(), cells):
            for method in methods:
                values = sorted([c for c in comparisons if c["group"] == group and c["scenario"] == "pair" and c["method"] == method],
                                key=lambda c: c["cp"])
                x, y = [c["cp"] for c in values], [c["recall"] for c in values]
                ax.plot(x, y, STYLE[method], marker="o", markersize=4, color=COLORS[method], label=METHOD_LABELS[method])
                ax.fill_between(x, [c["recall_ci_low"] for c in values], [c["recall_ci_high"] for c in values], color=COLORS[method], alpha=.07)
            ax.set(title=group.replace("_", " · "), xlabel="Probability of a private signal (CP)", ylabel="Held-out detection rate", ylim=(-.03, 1.05))
            ax.grid(alpha=.18)
        for ax in axes.ravel()[len(cells):]:
            ax.set_visible(False)
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle("Single colluding pair: detection power\nShading: 95% Wilson intervals", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .93))
        _save(fig, dest / "power")
        paths.append("power")
    group = "N6_T300_none" if "N6_T300_none" in cells else (cells[0] if cells else None)
    roc_rows = [r for r in rows if r["group"] == group and
                (r["scenario"] == "honest" or r["scenario"] == "pair" and r["cp"] == .25)]
    if roc_rows and {r["positive"] for r in roc_rows} == {True, False}:
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        for method in methods:
            r = [x for x in roc_rows if x["method"] == method]
            fpr, tpr, _ = roc_curve([x["positive"] for x in r], finite_ranks([x["score"] for x in r]))
            ax.plot(fpr, tpr, STYLE[method], color=COLORS[method], label=f"{METHOD_LABELS[method]} (AUC {binary_metrics(r)['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], color="#aaa", linewidth=.8)
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"ROC · {group} · pair CP = 0.25")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=.15)
        _save(fig, dest / "roc")
        paths.append("roc")
    if summary:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        labels = [METHOD_LABELS[m].replace(" ", "\n", 1) for m in methods]
        for index, method in enumerate(methods):
            s = summary[method]
            axes[0].bar(index, s["fpr"], color=COLORS[method])
            axes[0].errorbar(index, s["fpr"], yerr=[[max(0, s["fpr"] - s["fpr_ci_low"])], [max(0, s["fpr_ci_high"] - s["fpr"])]], color="#222", capsize=4)
            axes[1].bar(index, s["median_ms"], color=COLORS[method])
        axes[0].axhline(.05, color="#bc4749", linestyle="--", label="Nominal 5% target for added methods")
        axes[0].set(title="Matched honest test false alarms", ylabel="False-positive rate")
        axes[0].legend(fontsize=8)
        axes[1].set(title="Standalone analysis cost", ylabel="Median milliseconds / episode")
        for ax in axes:
            ax.set_xticks(range(len(methods)), labels)
            ax.grid(axis="y", alpha=.15)
        fig.tight_layout()
        _save(fig, dest / "false_alarms_runtime")
        paths.append("false_alarms_runtime")
    recovery = [c for c in comparisons if c["group"] == "N6_T300_none" and c["cp"] == 1 and c["scenario"] in ("pair", "star3", "two_pairs")]
    if recovery:
        fig, ax = plt.subplots(figsize=(9, 4.7))
        cases = ["pair", "star3", "two_pairs"]
        x = np.arange(3)
        for mi, method in enumerate(methods):
            y = [next((c["exact_groups"] for c in recovery if c["method"] == method and c["scenario"] == case), np.nan) for case in cases]
            ax.bar(x + (mi - (len(methods) - 1) / 2) * (.8 / len(methods)), y, width=.8 / len(methods), color=COLORS[method], label=METHOD_LABELS[method])
        ax.set_xticks(x, ["One pair", "Three-person star", "Two disjoint pairs"])
        ax.set(ylabel="Exact coalition-group recovery", ylim=(0, 1.08), title="Naming the right groups · N = 6, T = 300, CP = 1")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2)
        ax.grid(axis="y", alpha=.15)
        _save(fig, dest / "recovery")
        paths.append("recovery")
    if example:
        z = np.asarray(example["z"], dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
        im = axes[0].imshow(z, cmap="viridis")
        names = example["names"]
        axes[0].set_xticks(range(len(names)), names)
        axes[0].set_yticks(range(len(names)), names)
        axes[0].set_title("Permutation-standardized dependence")
        fig.colorbar(im, ax=axes[0], label="z score", shrink=.75)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(names)))
        graph.add_edges_from(edge_set(z, example["edge_threshold"]))
        truth = set().union(*(set(g) for g in example["truth"]))
        nx.draw_networkx(graph, pos=nx.circular_layout(graph), ax=axes[1], labels=dict(enumerate(names)),
                         node_color=["#007f86" if n in truth else "#aeb7c2" for n in names],
                         node_size=750, font_color="white", edge_color="#7354ad", width=2)
        axes[1].set_title("Thresholded graph\nTeal nodes: known synthetic coalition")
        axes[1].axis("off")
        fig.tight_layout()
        _save(fig, dest / "example_graph")
        paths.append("example_graph")
    return paths


def generate_report(output, rows, comparisons, calibrations, config, provenance, example):
    from .evaluate import write_csv
    methods = tuple(m for m in METHODS if any(r["method"] == m for r in rows))
    core = [r for r in rows if r["group"].endswith("_none") and r["scenario"] in ("honest", "pair", "star3", "two_pairs")]
    # External manifests may use arbitrary group names; preserve a useful
    # report and clearly label that this is the provided test mixture.
    if not core:
        core = [r for r in rows if r["scenario"] != "shared_state_ignored"]
    summary = {m: binary_metrics([r for r in core if r["method"] == m]) for m in methods}
    core_ids = {r["episode_id"] for r in core}
    core_groups = {r["group"] for r in core}
    for method in methods:
        aucs = [c["auroc"] for c in comparisons if c["method"] == method and c["group"] in core_groups
                and c["scenario"] in ("pair", "star3", "two_pairs") and math.isfinite(c["auroc"])]
        summary[method]["macro_auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    write_csv(output / "summary.csv", [{"method": m, **s} for m, s in summary.items()])
    paths = figures(output, rows, comparisons, summary, example)
    header = ["Method", "FPR [95% CI]", "Recall [95% CI]", "Precision", "F1", "Macro AUROC", "Exact groups", "Jaccard", "Median ms"]
    summary_table = markdown_table(header, [[METHOD_LABELS[m], _rate_ci(s["fpr"], s["fpr_ci_low"], s["fpr_ci_high"]),
                      _rate_ci(s["recall"], s["recall_ci_low"], s["recall_ci_high"]), _number(s["precision"]), _number(s["f1"]),
                      _number(s["macro_auroc"]), _number(s["exact_groups"]), _number(s["jaccard"]), _number(s["median_ms"], 2)]
                     for m, s in summary.items()])
    paired = []
    positive_ids = sorted({r["episode_id"] for r in core if r["positive"]})
    by_method = {m: {r["episode_id"]: r for r in core if r["method"] == m} for m in methods}
    rng = np.random.default_rng(config["seed"])
    if positive_ids:
        for method in methods[1:]:
            for metric in ("flag", "exact_groups"):
                differences = np.array([int(by_method[method][i][metric]) - int(by_method["bonjour"][i][metric]) for i in positive_ids])
                boots = np.array([rng.choice(differences, len(differences), replace=True).mean() for _ in range(2000)])
                low, high = np.quantile(boots, [.025, .975])
                paired.append({"method": method, "metric": metric, "difference": float(differences.mean()),
                               "ci_low": float(low), "ci_high": float(high), "episodes": len(differences)})
    write_csv(output / "paired_comparisons.csv", paired)
    paired_table = markdown_table(["Compared with original", "Metric", "Difference [paired bootstrap 95% CI]"],
                   [[METHOD_LABELS[p["method"]], "Recall" if p["metric"] == "flag" else "Exact groups",
                     f"{p['difference']:+.1%} [{p['ci_low']:+.1%}, {p['ci_high']:+.1%}]"] for p in paired])
    selected = [c for c in comparisons if c["group"] in core_groups and c["scenario"] in ("pair", "star3", "two_pairs")]
    detailed = markdown_table(["Group", "Scenario", "CP", "Method", "Recall", "FPR", "AUROC", "Exact groups", "Pair F1"],
               [[c["group"], c["scenario"], c["cp"], METHOD_LABELS[c["method"]], _number(c["recall"]), _number(c["fpr"]),
                 _number(c["auroc"]), _number(c["exact_groups"]), _number(c["pair_f1"])] for c in selected])
    stress_rows = []
    for group in sorted({r["group"] for r in rows}):
        scenarios = sorted({r["scenario"] for r in rows if r["group"] == group and
                            ("_none" not in group or r["scenario"] in ("parity", "shared_state_ignored"))})
        for scenario in scenarios:
            for method in methods:
                subset = [r for r in rows if r["group"] == group and r["scenario"] == scenario and r["method"] == method]
                flags = sum(r["flag"] for r in subset)
                low, high = wilson(flags, len(subset))
                stress_rows.append({"group": group, "scenario": scenario, "method": method, "n": len(subset),
                                    "positive": bool(subset[0]["positive"]), "flag_rate": flags / len(subset),
                                    "ci_low": low, "ci_high": high, "abstain": np.mean([r["abstain"] for r in subset]),
                                    "supported_pairs": np.mean([r["supported_pair_fraction"] for r in subset])})
    write_csv(output / "stress_tests.csv", stress_rows)
    stress_table = markdown_table(["Group", "Scenario", "Method", "n", "Flag rate [95% CI]", "Abstain", "Supported pairs"],
                [[r["group"], r["scenario"], METHOD_LABELS[r["method"]], r["n"], _rate_ci(r["flag_rate"], r["ci_low"], r["ci_high"]),
                  _number(r["abstain"]), _number(r["supported_pairs"])] for r in stress_rows])
    counts_table = markdown_table(["Calibration group", "Edge fit", "Calibration", "Held-out test"],
                                 [[g, c["edge_fit"], c["calibration"], c["test"]] for g, c in provenance["group_counts"].items()])
    best = max(summary, key=lambda m: summary[m]["recall"])
    recovery_best = max(summary, key=lambda m: summary[m]["exact_groups"])
    figures_md = "\n\n".join(f"![{name.replace('_', ' ').title()}](figures/{name}.png)" for name in paths)
    report = f"""# RPS detection results

Source: **{provenance['source']}**. {provenance['episodes']:,} games total;
{provenance['test_episodes']:,} held-out tests. Evaluation: {provenance['evaluation_seconds']:.1f} seconds.
Generated datasets use scripted players, not LLMs. Imported logs use the labels supplied in their manifest.

## Main comparison

This mixture contains {len(core_ids):,} test games. {METHOD_LABELS[best]} has the highest recall
({summary[best]['recall']:.1%}); {METHOD_LABELS[recovery_best]} has the highest exact group recovery
({summary[recovery_best]['exact_groups']:.1%}). These are descriptive comparisons of this mixture.

{summary_table}

Intervals are 95% Wilson intervals. Recovery counts missed detections and abstentions as failures.
Honest games are counted once. Precision and F1 depend on the chosen prevalence.
Macro AUROC averages within-condition AUROCs; scores from different calibrations are not comparable.
Each added method's runtime includes the shared pair-evidence calculation, as if it ran alone.

## Methods and calibration

Methods in this report: {', '.join(METHOD_LABELS[m] for m in methods)}.
All use pairwise action dependence. The graph methods can return several groups;
ML and growth return one. No private signals or true memberships enter their features.

Pair scores use conditional MI, standardized against shuffled moves within each public state.
The settings are {config.get('detector', {}).get('permutations', 99)} shuffles,
a minimum of {config.get('detector', {}).get('min_pair_rounds', 20)} observations per pair and
{config.get('detector', {}).get('min_state_rounds', 10)} per retained state.
Missing evidence remains undefined. The baseline uses complete rounds.

Honest edge-fit games set the graph threshold. Separate honest calibration games set final
score thresholds before testing. Rank cutoffs target a marginal {config.get('alpha', 0.05):.0%} false-positive rate when
calibration and future honest games are exchangeable. They do not guarantee a particular
realized rate for each fitted threshold or for a changed policy population.
Bonjour keeps its original 0.05-nat threshold. Combining methods needs separate calibration.

Growth uses the strongest-pair z for detection and relative average-z expansion for naming.
Its default join fraction is 0.25. ML uses an estimated Gaussian score and a group-size penalty;
that working model does not establish independent pair evidence or a posterior probability.
Sources and revisions are listed in the repository's ATTRIBUTION.md.

{counts_table}

## Scenarios

`pair` has one primary and one assistant. `star3` has one primary and two assistants.
`two_pairs` has two separate primary/assistant pairs. CP is the chance of a private signal;
labels describe the configured mechanism, not an inferred intention. Membership is randomized
per episode. Seed: {config['seed']}. Duplicate IDs and observed trajectories are rejected.

`parity` is a synthetic higher-order case with pairwise-independent coalition actions.
`shared_state_ignored` tests honest players responding to an omitted public cause; it is outside
the matched calibration population. `public_state` includes that cause. `last_round` tests
previous-action conditioning, which can leave too few observations per state.

## Paired comparison with Bonjour

{paired_table}

Intervals use 2,000 paired episode-bootstrap resamples of this fixed test mixture.
They exclude calibration-set uncertainty and are not adjusted for multiple comparisons.

## Figures

{figures_md}

PNG and SVG files are saved in `figures/`. ROC sweeps do not change the frozen decisions.
The example graph is the first N=6, T=300, CP=1 star case, selected by position rather than success.

## Results by condition

{detailed}

Exact groups requires the complete set of groups to match. Exact members compares their union.
Jaccard and pair F1 measure overlap; their denominators include missed detections.

## Stress tests

{stress_table}

For honest cases, the flag rate is a false-positive rate. For coalition cases, it is a detection rate.
Pair rows here pool CP=0.5 and CP=1; use comparison.csv for separate conditions.
Abstention is not evidence of honesty. Pairwise methods can miss higher-order-only coordination,
and omitted public causes can create false alarms. Shuffling may be invalid for adaptive policies.
The recorded model populations do not cover all strategies or settings.

## Files and reproduction

See docs/RUNNING.md in the repository for complete commands. Use a fresh output directory.
To replay these trajectories, pass episodes.jsonl.gz to evaluate.py with --dataset.
Reports can be rebuilt with python -m rps.benchmark.report --results PATH.
Calibration must match the honest policies, state definition, player count and game length.

Predictions, condition metrics, summaries and thresholds are saved as CSV files.
calibration/*.json stores the thresholds; provenance.json records the run settings.
The six-view information experiment is a separate analysis of the observations.

Python: {provenance['python']}. Packages: {provenance['versions']}.
Dataset SHA-256: `{provenance['dataset_sha256']}`.
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def regenerate(directory):
    """Rebuild presentation artifacts without rerunning or changing predictions."""
    import csv
    import json
    from rps.detection.pipeline import load_calibration
    output = Path(directory)
    def read_csv(path):
        def value(text):
            if text in ("True", "False"):
                return text == "True"
            if not text:
                return None
            if text.startswith(("[", "{")):
                return json.loads(text)
            try:
                return float(text)
            except ValueError:
                return text
        with path.open(newline="", encoding="utf-8") as handle:
            return [{k: value(v) for k, v in r.items()} for r in csv.DictReader(handle)]
    rows = read_csv(output / "predictions.csv")
    comparisons = read_csv(output / "comparison.csv")
    calibrations = {p.stem: load_calibration(p) for p in (output / "calibration").glob("*.json")}
    config = json.loads((output / "config.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    example_path = output / "example_graph.json"
    example = load_calibration(example_path) if example_path.exists() else None
    generate_report(output, rows, comparisons, calibrations, config, provenance, example)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild a report from its measured predictions and metrics")
    parser.add_argument("--results", required=True)
    regenerate(parser.parse_args().results)
