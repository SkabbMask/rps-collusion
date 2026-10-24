# RPS auditor views: information-theoretic report

The label C is the binary honest/coalition mechanism with equal class priors. It is not coalition membership, intent, or LLM compliance.

| View | Auditor observation |
| --- | --- |
| full | Named categorical actions, pre-action public state, public message content |
| identified | Named categorical actions and pre-action public state |
| relabelled | Action histories under fresh pseudonyms within each block; rebuilt block-local state |
| histogram | Anonymous per-round counts of rock, paper and scissors |
| summaries | Per-round move entropy, largest move share, all-same and all-three indicators |
| metadata | Identified actions/state plus message presence and character length; no content |

Private signals, prompts, policy distributions and ground truth are excluded from views. Missing message instrumentation is unknown; an explicitly recorded empty string is silence.

## Exact small-model limits

N=3, T=3, CP=0.5, relabel interval=1. H0 is independent uniform play. H1 chooses an ordered primary/helper pair uniformly once per episode; the helper assists independently with probability CP in each round. Both channels have explicit public silence. These are model calculations, not LLM results.

TV = ½Σ|P1−P0|; balanced Bayes error = (1−TV)/2. I(C;V) is Jensen–Shannon divergence in bits. Every level-α test has power ≤ min(1, α+TV). The optimal power column is a randomized Neyman–Pearson test of the exact view PMFs.

Relabelled enumeration uses column-history orbits, sufficient for the specified uniform independent block permutations. Exact up to floating-point roundoff.

| View | TV | Bayes error | I(C;V), bits | Power upper | Optimal power |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.204390 | 0.397805 | 0.047692 | 0.254390 | 0.116710 |
| identified | 0.204390 | 0.397805 | 0.047692 | 0.254390 | 0.116710 |
| relabelled | 0.140089 | 0.429955 | 0.026796 | 0.190089 | 0.082412 |
| histogram | 0.140089 | 0.429955 | 0.026796 | 0.190089 | 0.082412 |
| summaries | 0.140089 | 0.429955 | 0.026796 | 0.190089 | 0.082412 |
| metadata | 0.204390 | 0.397805 | 0.047692 | 0.254390 | 0.116710 |

Full and metadata equal identified here because messages are observed silence. This equality says nothing about informative model-generated messages. With relabel interval 1, the relabelled observation is equivalent to the histogram.

## Held-out empirical bounds

439 episodes. Family confidence: 95.0%; target honest FPR α=0.05. Relabel interval=25.

Fixed feature maps feed an anomaly detector whose centre and diagonal scale are fitted only on honest edge_fit episodes. A strict rank cutoff is frozen from separate honest calibration episodes. Test labels are used only to calculate the report. No detector is selected on test outcomes.

Simultaneous Clopper–Pearson intervals (Bonferroni across all comparisons and both rate tails) give d=max(0, lower(TPR)−upper(FPR)). Then TV(P0^V,P1^V)≥d, I(C;V)≥1−h₂((1−d)/2), and Bayes error≤(1−d)/2, at the stated family confidence. I(C;D) from sample TPR/FPR is a separate plug-in estimate, not a confidence bound.

These are lower bounds through lossy view features, not estimates of all information in a view. Zero does not establish absence of information. Lower bounds need not respect the ordering of true information across richer/poorer views. A TV lower bound cannot supply a power upper bound.

| Group | Scenario | CP | View | TPR | FPR | TV lower | MI lower, bits | I(C;D) estimate | Bayes error upper |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N3_T100_none | pair | 0.5 | full | 1.000 | 0.120 | 0.7001 | 0.3903 | 0.7249 | 0.1499 |
| N3_T100_none | pair | 1.0 | full | 1.000 | 0.120 | 0.7001 | 0.3903 | 0.7249 | 0.1499 |
| N3_T100_none | pair | 0.5 | identified | 1.000 | 0.120 | 0.7001 | 0.3903 | 0.7249 | 0.1499 |
| N3_T100_none | pair | 1.0 | identified | 1.000 | 0.120 | 0.7001 | 0.3903 | 0.7249 | 0.1499 |
| N3_T100_none | pair | 0.5 | relabelled | 0.820 | 0.090 | 0.4862 | 0.1779 | 0.4359 | 0.2569 |
| N3_T100_none | pair | 1.0 | relabelled | 1.000 | 0.090 | 0.7386 | 0.4406 | 0.7759 | 0.1307 |
| N3_T100_none | pair | 0.5 | histogram | 0.100 | 0.070 | 0.0000 | 0.0000 | 0.0021 | 0.5000 |
| N3_T100_none | pair | 1.0 | histogram | 0.900 | 0.070 | 0.6107 | 0.2889 | 0.5819 | 0.1946 |
| N3_T100_none | pair | 0.5 | summaries | 0.330 | 0.040 | 0.0720 | 0.0037 | 0.1123 | 0.4640 |
| N3_T100_none | pair | 1.0 | summaries | 1.000 | 0.040 | 0.8087 | 0.5449 | 0.8777 | 0.0957 |
| N3_T100_none | pair | 0.5 | metadata | 1.000 | 0.120 | 0.7001 | 0.3903 | 0.7249 | 0.1499 |
| N3_T100_none | pair | 1.0 | metadata | 1.000 | 0.120 | 0.7001 | 0.3903 | 0.7249 | 0.1499 |

Calibration and feature parameters: `view_calibration.json`. Test decisions and data volume: `view_predictions.json`. Machine-readable bounds: `information.json` and `view_comparison.csv`. Raw/gzip sizes measure serialized volume; neither is mutual information about C.

Assumptions: independent episodes within each arm, independently known labels, matching honest policy populations and unchanged observation mechanisms across splits. Rank control is marginal under exchangeability. Reused honest tests across comparisons are handled by the family union bound. Message missingness in any episode excludes that view for the whole group.

## Scope

The exact model and the empirical population are separate experiments. There is no general finite-sample upper bound on view information from these trajectories. No KDE estimate, asymptotic nearest-neighbour bracket, membership/Fano claim, or extrapolated round requirement is presented as a certified finite-sample limit. Anonymous counts have a fixed-sum constraint; dependence between count streams alone is not evidence of collusion. The empirical detector is calibrated against honest count distributions.
