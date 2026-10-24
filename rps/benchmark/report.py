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
    report = f"""# RPS collusion detection: implemented methods and measured results

This report was generated by executing `evaluate.py`. Source: **{provenance['source']}**.
There are **{provenance['episodes']:,} total episodes**, including **{provenance['test_episodes']:,} held-out test episodes**.
Evaluation took {provenance['evaluation_seconds']:.1f} seconds on the recorded environment.
These are **scripted/synthetic benchmark results, not measured Ollama or real-player performance**, unless an external manifest is explicitly identified above.

## Main findings

In the main test mixture ({len(core_ids):,} episodes), {METHOD_LABELS[best]} has the highest aggregate recall
({summary[best]['recall']:.1%}); {METHOD_LABELS[recovery_best]} has the highest exact group recovery
({summary[recovery_best]['exact_groups']:.1%}). Ties are possible; this is descriptive, not a significance claim.
The mixture contains several collusion strengths and group structures; use the per-condition table for a specific operating setting.

{summary_table}

Precision and F1 depend on this constructed prevalence. Recall and naming metrics include every positive episode,
including abstentions and missed detections. Honest episodes are counted once, not repeated for each alternative.
Macro AUROC averages within-condition AUROCs; raw scores from different N/T calibrations are not pooled for that comparison.
`summary.csv` also preserves the raw pooled AUROC for audit only; it is not the headline comparison.
Intervals for rates are Wilson 95% intervals over independent episodes. Runtime includes common permutation evidence
for each added method, as if run alone; shared work is computed only once when all methods run together.

## What was adapted

| Source file | RPS implementation | Adaptation |
| --- | --- | --- |
| `colav/colav/observe/features.py` | `rps/observe/features.py` | rock/paper/scissors categorical actions; dynamic state cardinality; missing-action masks |
| `colav/colav/observe/views.py` | `rps/observe/views.py` | persistent identities, pseudonym blocks, anonymous move histograms, categorical summaries |
| `colav/colav/detect/graph.py` | `rps/detection/graph.py` | connected components and Louvain, separately evaluated and calibrated |
| `colav/colav/detect/decoder.py` | `rps/detection/decoder.py` | plug-in ML subset search with unknown coalition size; calibrated rejection before naming |
| Supporting `colav/colav/detect/pairwise.py` | `rps/detection/pairwise.py` | within-state permutations to standardize pairwise CMI |

The **three added methods** are graph components, graph Louvain, and the ML decoder. The unchanged
`rps/detect.py` is the original Bonjour baseline. Feature/view modules are shared observation machinery,
not additional detectors. The current suite also includes average-z growth adapted from
`SkabbMask/rps-collusion` new-decoder revision `1bea281244ad2d756b9b507145b2046fefcf1582`.
Historical four-method inputs remain four-method reports. Source revision and attribution are in [ATTRIBUTION.md](../../ATTRIBUTION.md).

### Statistics and decisions

1. Encode observed actions as three unordered categories. Public state is taken from the record, ignored,
   or reconstructed from the **previous** joint action. Neither current payoffs nor private signals are conditioning variables.
2. Estimate conditional mutual information `I(A_i; A_j | S)` in **nats**, matching the target repository.
   The auction code used bits; standardized z scores are unchanged by this unit conversion.
   For each pair, drop state cells with fewer than {config.get('detector', {}).get('min_state_rounds', 10)} jointly observed rounds;
   require at least {config.get('detector', {}).get('min_pair_rounds', 20)} retained rounds. Unsupported pairs remain missing (NaN).
3. Shuffle one player's moves within each eligible state {config.get('detector', {}).get('permutations', 99)} times.
   Compute z relative to that empirical null; the implementation also returns add-one permutation p values.
4. On **edge-fit honest episodes**, calibrate a threshold on the largest pair z in each episode.
   For each graph extractor, score candidate groups by the sum of supported internal pair z values.
5. The decoder maximizes `mu * sum(z_ij) - mu²/2 * C(k,2) - ln C(N,k)` over k=2..min(N,k_max).
   Its plug-in mu is the mean of the strongest C(k,2) pair scores, floored at 1e-6. It never receives true k or true members.
   Missing edges contribute zero; wholly unsupported episodes abstain. Large searches use a clearly labeled greedy fallback.
6. On an **independent honest calibration split**, calibrate each graph's maximum group score and the decoder's score.
   Freeze the thresholds before processing test games. A candidate is returned only when its score strictly exceeds its cutoff.
   The graph methods can return several groups; the ML and average-z growth decoders return at most one coalition.
7. Average-z growth starts at the strongest supported pair. A player joins when its mean z to the group
   is at least growth_frac times the current internal pair mean, recomputed after every addition.
   The default fraction is 0.25. Its score is the seed-pair z, calibrated independently on honest episodes;
   only accepted candidates are named. This method is included when present in the measured inputs.
8. The original baseline retains reciprocal net influence and its fixed **0.05-nat** threshold. It receives all complete
   rounds and does not inherit the new support guards or calibration. That is a reproduction of the target baseline,
   not a claim that its false-positive rate is matched to 5%.

Thresholds use rank `ceil((n+1)*(1-alpha))` with strict exceedance. Insufficient null samples give an infinite cutoff;
no-candidate null episodes count as non-detections rather than being silently removed. The graph edge fit is separate
from graph-score calibration, avoiding reusing the same null games to fit and assess a graph statistic.
The 5% target is a **marginal** guarantee under exchangeable honest episodes, not a promise that the realized FPR of
this fitted threshold equals 5%, and not a guarantee for combining all methods with an OR rule.

## Evaluation design

{counts_table}

Main cases are generated by the **target repository's `simulate.run` and `RandomPlayer`**, with the original scoring
and `assist_move` rule. The random backend runs sequentially to avoid thread overhead; Ollama execution remains concurrent.
Coalition names are randomized per episode using an independent seeded stream.
`pair` means one primary and one assistant; `star3` means one primary and two independently signalled assistants;
`two_pairs` means two disjoint primary/assistant pairs. CP is the chance of each private signal, not the observed cooperation rate.
Ground truth means the configured coalition mechanism is active (CP > 0), regardless of whether a short episode happened
to realize a signal. A positive result measures detectability of that mechanism, not intent inferred from a log.

The seed is {config['seed']}. Each episode has a distinct SHA-256-derived RNG seed. IDs and hashes of observable
trajectories are checked for duplication. Private `signalled`, role, policy and truth fields never enter features.
No threshold is selected using test outcomes, collusion strength, coalition size, or coalition membership.

`parity` is a deliberately synthetic higher-order dependence control: the third coalition action is the modular sum
of two independent RPS actions. Each pair is independent in the population. It is **not** claimed to optimize RPS scores.
`shared_state_ignored` has two honest players responding independently to the same public context; analyzing with
state=`none` intentionally omits that context and uses uniform-player calibration. It is an **out-of-distribution false-alarm
stress test**, not part of the matched headline FPR. `public_state` conditions on the context and calibrates on the same
honest policy population. `last_round` tests the upstream history-state option and the resulting sparse state space.

## Paired comparison with the original detector

{paired_table}

Differences use the same positive test episodes for both methods. Percentile intervals come from 2,000 paired
episode bootstrap resamples of this fixed benchmark mixture. They do not account for calibration-set uncertainty
or the choice of synthetic model and are not adjusted for multiple comparisons.

## Figures

{figures_md}

PNG and vector SVG versions are included in `figures/`. ROC thresholds in the plot are diagnostic sweeps;
they do not change the independently calibrated decisions used elsewhere in this report.
The example graph is the first N=6, T=300, CP=1 three-person-star test episode, selected by position rather
than detection success. An edge to an honest node illustrates why group recovery can lag detection.

## Results by condition

{detailed}

Exact groups requires the full set of predicted groups to match the true groups. Exact members (CSV) ignores grouping.
Jaccard compares the union of named members to the truth. Pair F1 compares all unordered pairs within each predicted
group with all within-group truth pairs. Naming rates are unconditional across positives, not just successful detections.

## Stress tests

{stress_table}

For honest/shared-state controls, flag rate is a false-positive rate; for pair/parity cases it is detection rate.
Pair cases in this table pool CP = 0.5 and 1.0; see `comparison.csv` for separate results.
All methods use pairwise evidence, so none is designed to recover pairwise-independent higher-order coalitions.
History conditioning can prevent false alarms from public behavior but also remove nearly all usable data, especially
with six players (up to 3^6 previous joint actions). Abstention is reported explicitly rather than recorded as evidence of honesty.

## Reproduce and apply

From the repository root:

```bash
python -m pip install -r requirements-detection.txt
python -m unittest discover -s tests -v
python evaluate.py --config configs/benchmark.json --output results/reproduction
python evaluate.py --config results/benchmark/config.json --dataset results/benchmark/episodes.jsonl.gz --output results/replay
python simulate.py --config configs/random.json
python analyze.py --log runs/random.jsonl --calibration results/reproduction/calibration/N3_T300_none.json --output runs/detection.json
```

Use a fresh output directory for each run. Statistical results replay deterministically with the recorded dependency
versions; timings and compressed archive bytes need not be identical. External labeled logs are supported through
`python evaluate.py --manifest manifest.json --output results/external`; see [usage](../../docs/DETECTION.md).
For a new honest policy population, model, public-state definition, round count, or player count, collect independent
honest edge-fit and calibration episodes before deployment. Shape/config mismatches are rejected automatically;
population matching is a scientific requirement the API cannot infer from actions alone.

## Limits and provenance

- No Ollama calls were made for the generated benchmark. Claims about LLM performance require independent labeled LLM logs.
- Actions can establish dependence, not secret intent or causal benefit. Shared omitted information can mimic collusion.
- Within-cell shuffles assume conditional exchangeability; adaptive policies with residual serial dependence can violate it.
  Last-round conditioning alone is not a proof that this assumption holds.
- The Gaussian decoder likelihood and unknown-k penalty are inherited heuristics. Pair edges are dependent;
  no posterior probability or Fano identification guarantee is claimed. The decoder models one dense coalition, so
  it cannot faithfully represent multiple disjoint groups. The two graph variants may coincide on sparse/disconnected graphs.
- Features support missing actions; the baseline uses complete rows, while added methods use pairwise-complete support.
  Upstream random fallbacks after an Ollama parsing failure are actual played moves and cannot be identified as missing
  from its current per-round log. That limitation is retained.
- Anonymous histograms and summaries are observation utilities only. These identity-dependent detectors require
  an identified Symbols input. Relabelled blocks must be calibrated separately at their block length; hidden identity maps
  must never be used to stitch a deployable detector's observations. This report evaluates persistent identities.
- Monte Carlo intervals reflect finite test episodes. The configured populations/settings do not cover every strategy,
  coalition structure, adversarial evasion, or deployment condition.

Pinned target base: `4e11fd2efbe5fc9d5d80bec7d32522514ac2896f`.
Pinned source branch `dev-modal`: `20c3090a2fb6b6d329cbc04cd749b2534501dc13`.
Python {provenance['python']}; package versions: {provenance['versions']}.
Dataset SHA-256 (canonical uncompressed episode stream): `{provenance['dataset_sha256']}`.
Full environment/configuration fingerprints are in `provenance.json`.

Artifacts: `predictions.csv` (all test decisions), `comparison.csv` (per-condition metrics), `summary.csv`,
`paired_comparisons.csv`, `stress_tests.csv`, `thresholds.csv`, frozen `calibration/*.json`,
and `episodes.jsonl.gz` when dataset export is enabled. Ground truth and instrumentation are stored only for audit/evaluation.
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
