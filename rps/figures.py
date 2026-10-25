"""Export experiment figures from saved results, without simulation or model calls.

Rebuild with: python -m rps.figures --results results/my-run
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"bonjour": "Net influence", "graph_components": "Components",
          "graph_louvain": "Louvain", "decoder": "Likelihood search", "growth": "Average-z growth"}
COLORS = ["#666666", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _has_cp(row):
    try:
        return math.isfinite(float(row.get("cp", "")))
    except (TypeError, ValueError):
        return False


def _wilson(k, n):
    p, z = k/n, 1.959963984540054
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    width = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-width, center+width


def _source(root):
    cfg, prov = _json(root / "config.json"), _json(root / "provenance.json")
    defaults = cfg.get("defaults", {})
    if defaults.get("backend") == "ollama":
        return f"Model experiment: {defaults.get('model', 'Ollama')}"
    source = prov.get("source", _json(root / "information.json").get("source", ""))
    if "random backend" in source or "scripted" in source:
        return "Local scripted benchmark: random players"
    return "Recorded games (see run provenance)"


def _style():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})


class Export:
    def __init__(self, root, inputs, kind, options=None):
        self.root = Path(root)
        self.dest = self.root / "figures"
        self.dest.mkdir(exist_ok=True)
        self.kind = kind
        self.options = options or {}
        files = [Path(p) for p in inputs if Path(p).exists()]
        files.append(Path(__file__))
        self.hashes = {str(p.relative_to(self.root)) if p.is_relative_to(self.root) else "figure_exporter.py":
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        old = _json(self.dest / f"{kind}_manifest.json")
        self.cached = (old.get("inputs_sha256") == self.hashes and
                       old.get("options", {}) == self.options and
                       all((self.dest / p).exists() for p in old.get("files", [])) and
                       bool(old.get("files")))
        self.files, self.captions = [], []

    def save(self, fig, name, caption):
        plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
        for ext in ("png", "svg", "pdf"):
            path = self.dest / f"{name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            self.files.append(path.name)
        plt.close(fig)
        number = len(self.captions)+1
        self.captions.append(f"**Figure {number}. {caption}**\n\n![{name}]({name}.png)\n")

    def finish(self):
        if not self.captions:
            self.captions = ["No applicable figures for the saved results.\n"]
        caption_file = f"{self.kind}_captions.md"
        (self.dest / caption_file).write_text("\n".join(self.captions), encoding="utf-8")
        self.files.append(caption_file)
        (self.dest / f"{self.kind}_manifest.json").write_text(json.dumps({
            "inputs_sha256": self.hashes, "files": self.files, "options": self.options,
            "matplotlib": matplotlib.__version__, "scope": "saved results only; no new experiments"}, indent=2)+"\n")


def detection(root):
    """Performance and per-condition power; support historical method sets."""
    root = Path(root)
    exp = Export(root, [root / p for p in ("predictions.csv", "summary.csv", "comparison.csv",
                                          "thresholds.csv", "config.json", "provenance.json")], "detection")
    if exp.cached:
        return
    _style()
    predictions = _csv(root / "predictions.csv")
    summary = _csv(root / "summary.csv")
    source = _source(root)
    methods = [r["method"] for r in summary]
    # Match the existing report's main mixture, excluding stress conditions.
    main = [r for r in predictions if r["group"].endswith("_none")
            and r["scenario"] in ("honest", "pair", "star3", "two_pairs")]
    # Imported populations may use custom group names; report those as such.
    if not main:
        main = [r for r in predictions if r["scenario"] in ("honest", "pair", "star3", "two_pairs")]
    if main:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, layout="constrained")
        fig.suptitle(source, fontsize=16)
        sizes = None
        fpr_upper = []
        for i, method in enumerate(methods):
            pos = [r for r in main if r["method"] == method and r["positive"] == "True"]
            neg = [r for r in main if r["method"] == method and r["positive"] == "False"]
            sizes = (len(pos), len(neg))
            for ax, title, data, key in zip(axes, ("Detection recall", "Exact group recovery", "False-positive rate"),
                                            (pos, pos, neg), ("flag", "exact_groups", "flag")):
                if data:
                    count = sum(r[key] == "True" if key == "flag" else float(r[key]) == 1 for r in data)
                    v = count/len(data)
                    lo, hi = _wilson(count, len(data))
                    if key == "flag" and data is neg:
                        fpr_upper.append(hi*100)
                    ax.errorbar(v*100, i, xerr=[[(v-lo)*100], [(hi-v)*100]], fmt="o",
                                color=COLORS[i % len(COLORS)], capsize=4)
                ax.set_title(title)
                ax.set_xlim(0, 100)
                ax.set_xlabel("Percent (pointwise 95% Wilson interval)")
                ax.grid(axis="x", alpha=.2)
        axes[0].set_yticks(range(len(methods)), [LABELS.get(m, m) for m in methods])
        axes[0].invert_yaxis()
        axes[-1].set_xlim(0, min(100, max(8, max(fpr_upper, default=8)*1.1)))
        axes[-1].axvline(_json(root / "config.json").get("alpha", .05)*100,
                        color="black", linestyle=":", linewidth=1)
        npos, nneg = sizes
        small = " Tiny sample: case illustration only, not performance validation." if min(npos, nneg) < 20 else ""
        infinite = set()
        if (root / "thresholds.csv").exists():
            for row in _csv(root / "thresholds.csv"):
                if float(row["threshold"]) == math.inf:
                    infinite.add(LABELS.get(row["method"], row["method"]))
        cutoff_note = (" Infinite cutoffs for at least one group: "+", ".join(sorted(infinite))+
                       "; zero flags under these cutoffs do not establish score failure.") if infinite else ""
        exp.save(fig, "paper_performance", f"{source}. Main mixture: {npos} coalition and {nneg} honest test games; "
                 "pointwise Wilson 95% intervals. Recovery counts misses as failures. Methods have different "
                 "threshold procedures; this is not a matched-FPR comparison. Stress controls excluded."+small+cutoff_note)
    cells = _csv(root / "comparison.csv")
    groups = sorted({r["group"] for r in cells if r["scenario"] == "pair" and _has_cp(r)})
    if groups:
        fig, axes = plt.subplots(math.ceil(len(groups)/2), 2, figsize=(12, 3.8*math.ceil(len(groups)/2)),
                                 squeeze=False, layout="constrained")
        fig.suptitle(f"{source}: single assisting pair", fontsize=15)
        for ax, group in zip(axes.flat, groups):
            for i, method in enumerate(methods):
                vals = sorted([r for r in cells if r["group"] == group and r["scenario"] == "pair" and _has_cp(r)
                               and r["method"] == method], key=lambda r: float(r["cp"]))
                x = [float(r["cp"]) for r in vals]
                ax.plot(x, [float(r["recall"])*100 for r in vals], marker=["o", "s", "^", "D", "x"][i % 5],
                        linestyle=["-", "--", ":", "-", "--"][i % 5], color=COLORS[i % 5], label=LABELS.get(method, method))
                ax.fill_between(x, [float(r["recall_ci_low"])*100 for r in vals],
                                [float(r["recall_ci_high"])*100 for r in vals], color=COLORS[i % 5], alpha=.06)
            ax.set(title=group, xlabel="Assistance probability CP", ylabel="Recall (%)", ylim=(-3, 103))
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(groups):]: ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        # Figure.legend outside locations require newer Matplotlib; reserve space explicitly.
        axes.flat[0].legend(handles, labels, fontsize=9, loc="best")
        exp.save(fig, "paper_signal_and_horizon", f"{source}. Single-pair recall by saved condition; shading shows "
                 "pointwise Wilson 95% intervals. Counts are in comparison.csv. Lines connect observations, "
                 "not fitted curves. Groups use separate calibration; tiny test cells are illustrative only.")
    exp.finish()


def information(root, *, include_exact=False):
    root = Path(root)
    exp = Export(root, [root / p for p in ("exact_views.json", "information.json", "config.json", "provenance.json")], "information",
                 options={"include_exact": include_exact})
    if exp.cached: return
    _style()
    exact = _json(root / "exact_views.json")
    if exact and include_exact:
        names = list(exact["views"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
        title = f"Exact synthetic model: N={exact['n_players']}, T={exact['rounds']}, CP={exact['cp']}"
        fig.suptitle(title+" (not model-agent estimates)", fontsize=14)
        for ax, key, label, factor in zip(axes, ("mi_bits", "bayes_error"),
                                          ("Label information (bits)", "Optimal balanced error (%)"), (1, 100)):
            vals = [exact["views"][name][key]*factor for name in names]
            ax.barh(names, vals, color="#0072B2")
            ax.invert_yaxis()
            ax.set_title(label)
            ax.set_xlim(0, max(max(vals)*1.25, .01 if factor == 1 else 50))
            for i, val in enumerate(vals): ax.annotate(f"{val:.4f}" if factor == 1 else f"{val:.2f}",
                                                       (val, i), xytext=(4, 0), textcoords="offset points", va="center")
        exp.save(fig, "paper_exact_views", title+f"; relabel interval {exact['relabel_interval']}. "
                 "Excluded from the paper: inconclusive for its empirical question. "
                 "Balanced priors and empty messages. Exact enumeration up to floating-point rounding; "
                 "no sampling error bars. Equal view values are model-specific. These are not Qwen information estimates.")
    measured = _json(root / "information.json")
    comparisons = measured.get("comparisons", [])
    # One panel per group, with conditions on x; missing views remain missing.
    groups = sorted({r["group"] for r in comparisons})
    for index, group in enumerate(groups, 1):
        selected = [r for r in comparisons if r["group"] == group]
        conditions = sorted({(r["scenario"], r["cp"]) for r in selected}, key=str)
        views = list(dict.fromkeys(r["view"] for r in selected))
        matrix = np.full((len(views), len(conditions)), np.nan)
        for r in selected:
            matrix[views.index(r["view"]), conditions.index((r["scenario"], r["cp"]))] = r["mi_view_bits_lower_confidence"]
        fig, ax = plt.subplots(figsize=(max(7, len(conditions)*.9), 5), layout="constrained")
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_yticks(range(len(views)), views)
        ax.set_xticks(range(len(conditions)), [f"{s}\nCP={cp}" for s, cp in conditions], rotation=35, ha="right")
        ax.set_title(f"{_source(root)}\n{group}: measured information lower bounds")
        for (i, j), value in np.ndenumerate(matrix):
            if np.isfinite(value): ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > .5 else "black")
        fig.colorbar(im, ax=ax, label="Information lower bound (bits)")
        exp.save(fig, f"paper_information_bounds_{index:02d}", f"{_source(root)}; group {group}. "
                 "Detector-based MI lower bounds, not full-view MI estimates. Zero does not establish no information; "
                 "bounds cannot rank true view information. Blank cells are unavailable. See information.json "
                 "for test counts, thresholds, family confidence and assumptions. Infinite calibration cutoffs, "
                 "when present, prevent flags rather than establish that the scores have no useful separation.")
    exp.finish()


def collection(root):
    """One held-out coalition case per group, with actual partner assignments."""
    from rps.benchmark import collect as collector
    from rps.benchmark.collect import game_config, plan
    from rps.game import assist_move
    root = Path(root)
    cfg, manifest = _json(root / "config.json"), _json(root / "manifest.json")
    episodes = {ep["id"]: ep for ep in plan(cfg)}
    cases, seen = [], set()
    for item in manifest["episodes"]:
        if item["split"] == "test" and item["truth"] and item["group"] not in seen:
            cases.append(item)
            seen.add(item["group"])
    exp = Export(root, [root / "config.json", root / "manifest.json", root / "quality.json"]+
                 [Path(collector.__file__), Path(__file__).parent / "game.py"]+
                 [root / item["path"] for item in cases], "collection")
    if exp.cached: return
    _style()
    for index, item in enumerate(cases, 1):
        game, truth = game_config(cfg, episodes[item["id"]])
        if truth != item["truth"]:
            raise ValueError("configured case assignments disagree with saved manifest")
        helper = next(p for p in game["players"] if p.get("partner"))
        h, partner = helper["name"], helper["partner"]
        records = [json.loads(line) for line in (root / item["path"]).read_text().splitlines() if line.strip()]
        active = [h in r["signalled"] for r in records]
        correct = [r["actions"][h] == assist_move(r["actions"][partner]) for r in records]
        followed = sum(a and c for a, c in zip(active, correct))
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 3]}, layout="constrained")
        t = np.arange(1, len(records)+1)
        colors = ["#999999" if not a else "#009E73" if c else "#D55E00" for a, c in zip(active, correct)]
        axes[0].scatter(t, np.zeros(len(t)), c=colors, marker="s", s=max(10, min(200, 3000/len(t))))
        axes[0].set_yticks([])
        axes[0].set_title(f"{_source(root)}: one held-out case\n{h} helps {partner}: {followed}/{sum(active)} signalled rounds", fontsize=13)
        for i, name in enumerate(records[0]["actions"]):
            role = "primary" if name == partner else "assistant" if name == h else "other player"
            axes[1].plot(t, np.cumsum([r["payoffs"][name] for r in records]), label=f"{name}: {role}", color=COLORS[i % 5])
        axes[1].set(xlabel="Round", ylabel="Cumulative payoff")
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=.2)
        exp.save(fig, f"paper_coordination_case_{index:02d}", f"{_source(root)}; {item['id']}. "
                 f"One case, not a performance estimate. {h} followed {followed}/{sum(active)} private signals. "
                 "Green=prescribed action, orange=miss, gray=no signal. Payoffs are not zero-sum; this does not "
                 "demonstrate outsider harm or spontaneous/message-mediated collusion. First held-out coalition "
                 "case per group, selected in manifest order; for multiple helpers the first helper is shown.")
    exp.finish()


def regenerate(root, *, include_exact=False):
    root = Path(root)
    if (root / "predictions.csv").exists(): detection(root)
    if (root / "exact_views.json").exists() or (root / "information.json").exists(): information(root, include_exact=include_exact)
    if (root / "collection.json").exists(): collection(root)
    for stage in ("detection", "information"):
        if (root / stage).is_dir(): regenerate(root / stage, include_exact=include_exact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--include-exact", action="store_true",
                        help="export the optional toy diagnostic, excluded from the paper")
    args = parser.parse_args()
    if not args.results.is_dir(): parser.error("results directory does not exist")
    regenerate(args.results, include_exact=args.include_exact)
    print(f"Figures: {args.results}/figures (plus analysis-stage figures, where applicable)")
