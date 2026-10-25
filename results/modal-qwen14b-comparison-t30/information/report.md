# Information by observation view

C means whether the configured coordination mechanism is active, with equal label priors. It does not identify intent or membership.

| View | Auditor observation |
| --- | --- |
| full | Named categorical actions, pre-action public state, public message content |
| identified | Named categorical actions and pre-action public state |
| relabelled | Action histories under fresh pseudonyms within each block; rebuilt block-local state |
| histogram | Anonymous per-round counts of rock, paper and scissors |
| summaries | Per-round move entropy, largest move share, all-same and all-three indicators |
| metadata | Identified actions/state plus message presence and character length; no content |

Private signals, prompts, policy distributions and ground truth are excluded from views. Missing message instrumentation is unknown; an explicitly recorded empty string is silence.

## Bounds from test games

439 episodes. Family confidence: 95.0%; target honest FPR α=0.05. Relabel interval=10.

Each view has fixed features and an anomaly detector fitted on honest games. Separate honest games set its threshold before test labels are used.

Simultaneous Clopper–Pearson intervals (Bonferroni across all comparisons and both rate tails) give d=max(0, lower(TPR)−upper(FPR)). Then TV(P0^V,P1^V)≥d, I(C;V)≥1−h₂((1−d)/2), and Bayes error≤(1−d)/2, at the stated family confidence. I(C;D) from sample TPR/FPR is a separate plug-in estimate, not a confidence bound.

These are lower bounds through particular features and detectors. Zero does not prove no information, and differing bounds do not rank the views. A TV lower bound cannot give a power upper bound.

| Group | Scenario | CP | View | TPR | FPR | TV lower | MI lower, bits | I(C;D) estimate | Bayes error upper |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N3_T30_none | pair | 0.5 | full | 1.000 | 0.020 | 0.8409 | 0.5994 | 0.9290 | 0.0796 |
| N3_T30_none | pair | 1.0 | full | 1.000 | 0.020 | 0.8409 | 0.5994 | 0.9290 | 0.0796 |
| N3_T30_none | pair | 0.5 | identified | 0.630 | 0.110 | 0.2549 | 0.0474 | 0.2254 | 0.3726 |
| N3_T30_none | pair | 1.0 | identified | 1.000 | 0.110 | 0.7127 | 0.4063 | 0.7413 | 0.1436 |
| N3_T30_none | pair | 0.5 | relabelled | 0.180 | 0.030 | 0.0000 | 0.0000 | 0.0474 | 0.5000 |
| N3_T30_none | pair | 1.0 | relabelled | 1.000 | 0.030 | 0.8243 | 0.5707 | 0.9022 | 0.0879 |
| N3_T30_none | pair | 0.5 | histogram | 0.010 | 0.010 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| N3_T30_none | pair | 1.0 | histogram | 0.050 | 0.010 | 0.0000 | 0.0000 | 0.0108 | 0.5000 |
| N3_T30_none | pair | 0.5 | summaries | 0.190 | 0.080 | 0.0000 | 0.0000 | 0.0192 | 0.5000 |
| N3_T30_none | pair | 1.0 | summaries | 0.990 | 0.080 | 0.7306 | 0.4298 | 0.7550 | 0.1347 |
| N3_T30_none | pair | 0.5 | metadata | 1.000 | 0.070 | 0.7655 | 0.4785 | 0.8135 | 0.1173 |
| N3_T30_none | pair | 1.0 | metadata | 1.000 | 0.070 | 0.7655 | 0.4785 | 0.8135 | 0.1173 |

Calibration and feature parameters: `view_calibration.json`. Test decisions and data volume: `view_predictions.json`. Machine-readable bounds: `information.json` and `view_comparison.csv`. Raw/gzip sizes measure serialized volume; neither is mutual information about C.

Assumptions: independent episodes within each arm, independently known labels, matching honest policy populations and unchanged observation mechanisms across splits. Rank control is marginal under exchangeability. Reused honest tests across comparisons are handled by the family union bound. Message missingness in any episode excludes that view for the whole group.

## Scope

These results do not establish a general limit for LLM detection. Count streams are dependent even under honest play, so the comparison must use matching honest games. See docs/INFORMATION.md for details.

## Figures

[Figures and captions](figures/information_captions.md). The optional toy diagnostic is excluded from the paper.
