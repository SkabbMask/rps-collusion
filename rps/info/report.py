"""Report exact model limits separately from held-out decision lower bounds."""
from pathlib import Path

from rps.detection.pipeline import write_json
from .exact import VIEWS


def write_report(output, empirical=None, exact=None):
    output = Path(output)
    lines = ["# Information by observation view", "",
             "C means whether the configured coordination mechanism is active, with equal label priors. "
             "It does not identify intent or membership.", "",
             "| View | Auditor observation |", "| --- | --- |",
             "| full | Named categorical actions, pre-action public state, public message content |",
             "| identified | Named categorical actions and pre-action public state |",
             "| relabelled | Action histories under fresh pseudonyms within each block; rebuilt block-local state |",
             "| histogram | Anonymous per-round counts of rock, paper and scissors |",
             "| summaries | Per-round move entropy, largest move share, all-same and all-three indicators |",
             "| metadata | Identified actions/state plus message presence and character length; no content |", "",
             "Private signals, prompts, policy distributions and ground truth are excluded from views. "
             "Missing message instrumentation is unknown; an explicitly recorded empty string is silence.", ""]
    if exact:
        write_json(output / "exact_views.json", exact)
        lines += ["## Exact small model", "",
                  f"N={exact['n_players']}, T={exact['rounds']}, CP={exact['cp']}, relabel interval={exact['relabel_interval']}. "
                  "H0 is independent uniform play. H1 chooses an ordered primary/helper pair uniformly once "
                  "per episode; the helper assists independently with probability CP in each round. "
                  "Both channels have explicit public silence. These are model calculations, not LLM results.", "",
                  "TV = ½Σ|P1−P0|; balanced Bayes error = (1−TV)/2. I(C;V) is Jensen–Shannon divergence "
                  "in bits. Every level-α test has power ≤ min(1, α+TV). The optimal power column is a "
                  "randomized Neyman–Pearson test of the exact view PMFs.", "",
                  "Relabelled enumeration uses column-history orbits, sufficient for the specified uniform "
                  "independent block permutations. Exact up to floating-point roundoff.", "",
                  "| View | TV | Bayes error | I(C;V), bits | Power upper | Optimal power |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for view in VIEWS:
            r = exact["views"][view]
            lines.append(f"| {view} | {r['tv']:.6f} | {r['bayes_error']:.6f} | {r['mi_bits']:.6f} | "
                         f"{r['power_upper_alpha_plus_tv']:.6f} | {r['optimal_power_at_alpha']:.6f} |")
        lines += ["", "Empty messages make full and metadata equal identified in this model. "
                  "At relabel interval 1, relabelled equals histogram.", ""]
    if empirical:
        lines += ["## Bounds from test games", "",
                  f"{empirical['n_episodes']} episodes. Family confidence: {empirical['family_confidence']:.1%}; "
                  f"target honest FPR α={empirical['alpha']}. Relabel interval={empirical['relabel_interval']}.", "",
                  "Each view has fixed features and an anomaly detector fitted on honest games. "
                  "Separate honest games set its threshold before test labels are used.", "",
                  "Simultaneous Clopper–Pearson intervals (Bonferroni across all comparisons and both rate tails) "
                  "give d=max(0, lower(TPR)−upper(FPR)). Then TV(P0^V,P1^V)≥d, "
                  "I(C;V)≥1−h₂((1−d)/2), and Bayes error≤(1−d)/2, at the stated family confidence. "
                  "I(C;D) from sample TPR/FPR is a separate plug-in estimate, not a confidence bound.", "",
                  "These are lower bounds through particular features and detectors. Zero does not prove no information, "
                  "and differing bounds do not rank the views. A TV lower bound cannot give a power upper bound.", "",
                  "| Group | Scenario | CP | View | TPR | FPR | TV lower | MI lower, bits | I(C;D) estimate | Bayes error upper |",
                  "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in empirical["comparisons"]:
            lines.append(f"| {r['group']} | {r['scenario']} | {r['cp']} | {r['view']} | {r['tpr']:.3f} | "
                         f"{r['fpr']:.3f} | {r['tv_lower_confidence']:.4f} | "
                         f"{r['mi_view_bits_lower_confidence']:.4f} | {r['mi_decision_bits_plugin']:.4f} | "
                         f"{r['bayes_error_upper_confidence']:.4f} |")
        lines += ["", "Calibration and feature parameters: `view_calibration.json`. Test decisions and data volume: "
                  "`view_predictions.json`. Machine-readable bounds: `information.json` and `view_comparison.csv`. "
                  "Raw/gzip sizes measure serialized volume; neither is mutual information about C.", "",
                  "Assumptions: independent episodes within each arm, independently known labels, matching "
                  "honest policy populations and unchanged observation mechanisms across splits. Rank control "
                  "is marginal under exchangeability. Reused honest tests across comparisons are handled by "
                  "the family union bound. Message missingness in any episode excludes that view for the whole group.", ""]
        if empirical["unavailable"]:
            lines += ["### Unavailable views", ""]
            for r in empirical["unavailable"]:
                lines.append(f"- {r['group']} / {r['view']}: {r['reason']}.")
            lines.append("")
    lines += ["## Scope", "",
              "The exact model and measured bounds are separate experiments. Neither establishes a general "
              "limit for LLM detection. Count streams are dependent even under honest play, so the "
              "comparison must use matching honest games. See docs/INFORMATION.md for details.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
