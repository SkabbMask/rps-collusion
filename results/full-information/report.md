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

6253 episodes. Family confidence: 95.0%; target honest FPR α=0.05. Relabel interval=25.

Fixed feature maps feed an anomaly detector whose centre and diagonal scale are fitted only on honest edge_fit episodes. A strict rank cutoff is frozen from separate honest calibration episodes. Test labels are used only to calculate the report. No detector is selected on test outcomes.

Simultaneous Clopper–Pearson intervals (Bonferroni across all comparisons and both rate tails) give d=max(0, lower(TPR)−upper(FPR)). Then TV(P0^V,P1^V)≥d, I(C;V)≥1−h₂((1−d)/2), and Bayes error≤(1−d)/2, at the stated family confidence. I(C;D) from sample TPR/FPR is a separate plug-in estimate, not a confidence bound.

These are lower bounds through lossy view features, not estimates of all information in a view. Zero does not establish absence of information. Lower bounds need not respect the ordering of true information across richer/poorer views. A TV lower bound cannot supply a power upper bound.

| Group | Scenario | CP | View | TPR | FPR | TV lower | MI lower, bits | I(C;D) estimate | Bayes error upper |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N3_T100_none | pair | 0.1 | full | 0.100 | 0.090 | 0.0000 | 0.0000 | 0.0002 | 0.5000 |
| N3_T100_none | pair | 0.25 | full | 0.530 | 0.090 | 0.1076 | 0.0084 | 0.1762 | 0.4462 |
| N3_T100_none | pair | 0.5 | full | 1.000 | 0.090 | 0.6755 | 0.3603 | 0.7759 | 0.1623 |
| N3_T100_none | pair | 1.0 | full | 1.000 | 0.090 | 0.6755 | 0.3603 | 0.7759 | 0.1623 |
| N3_T100_none | pair | 0.1 | identified | 0.100 | 0.090 | 0.0000 | 0.0000 | 0.0002 | 0.5000 |
| N3_T100_none | pair | 0.25 | identified | 0.530 | 0.090 | 0.1076 | 0.0084 | 0.1762 | 0.4462 |
| N3_T100_none | pair | 0.5 | identified | 1.000 | 0.090 | 0.6755 | 0.3603 | 0.7759 | 0.1623 |
| N3_T100_none | pair | 1.0 | identified | 1.000 | 0.090 | 0.6755 | 0.3603 | 0.7759 | 0.1623 |
| N3_T100_none | pair | 0.1 | relabelled | 0.120 | 0.080 | 0.0000 | 0.0000 | 0.0032 | 0.5000 |
| N3_T100_none | pair | 0.25 | relabelled | 0.230 | 0.080 | 0.0000 | 0.0000 | 0.0321 | 0.5000 |
| N3_T100_none | pair | 0.5 | relabelled | 0.860 | 0.080 | 0.4723 | 0.1675 | 0.5042 | 0.2639 |
| N3_T100_none | pair | 1.0 | relabelled | 1.000 | 0.080 | 0.6894 | 0.3770 | 0.7943 | 0.1553 |
| N3_T100_none | pair | 0.1 | histogram | 0.060 | 0.100 | 0.0000 | 0.0000 | 0.0040 | 0.5000 |
| N3_T100_none | pair | 0.25 | histogram | 0.040 | 0.100 | 0.0000 | 0.0000 | 0.0103 | 0.5000 |
| N3_T100_none | pair | 0.5 | histogram | 0.130 | 0.100 | 0.0000 | 0.0000 | 0.0016 | 0.5000 |
| N3_T100_none | pair | 1.0 | histogram | 0.960 | 0.100 | 0.5842 | 0.2625 | 0.6418 | 0.2079 |
| N3_T100_none | pair | 0.1 | summaries | 0.050 | 0.040 | 0.0000 | 0.0000 | 0.0004 | 0.5000 |
| N3_T100_none | pair | 0.25 | summaries | 0.090 | 0.040 | 0.0000 | 0.0000 | 0.0076 | 0.5000 |
| N3_T100_none | pair | 0.5 | summaries | 0.330 | 0.040 | 0.0100 | 0.0001 | 0.1123 | 0.4950 |
| N3_T100_none | pair | 1.0 | summaries | 1.000 | 0.040 | 0.7495 | 0.4557 | 0.8777 | 0.1253 |
| N3_T100_none | pair | 0.1 | metadata | 0.100 | 0.090 | 0.0000 | 0.0000 | 0.0002 | 0.5000 |
| N3_T100_none | pair | 0.25 | metadata | 0.530 | 0.090 | 0.1076 | 0.0084 | 0.1762 | 0.4462 |
| N3_T100_none | pair | 0.5 | metadata | 1.000 | 0.090 | 0.6755 | 0.3603 | 0.7759 | 0.1623 |
| N3_T100_none | pair | 1.0 | metadata | 1.000 | 0.090 | 0.6755 | 0.3603 | 0.7759 | 0.1623 |
| N3_T300_none | pair | 0.1 | full | 0.190 | 0.010 | 0.0000 | 0.0000 | 0.0779 | 0.5000 |
| N3_T300_none | pair | 0.25 | full | 0.990 | 0.010 | 0.7805 | 0.5008 | 0.9192 | 0.1098 |
| N3_T300_none | pair | 0.5 | full | 1.000 | 0.010 | 0.8038 | 0.5371 | 0.9595 | 0.0981 |
| N3_T300_none | pair | 1.0 | full | 1.000 | 0.010 | 0.8038 | 0.5371 | 0.9595 | 0.0981 |
| N3_T300_none | pair | 0.1 | identified | 0.190 | 0.010 | 0.0000 | 0.0000 | 0.0779 | 0.5000 |
| N3_T300_none | pair | 0.25 | identified | 0.990 | 0.010 | 0.7805 | 0.5008 | 0.9192 | 0.1098 |
| N3_T300_none | pair | 0.5 | identified | 1.000 | 0.010 | 0.8038 | 0.5371 | 0.9595 | 0.0981 |
| N3_T300_none | pair | 1.0 | identified | 1.000 | 0.010 | 0.8038 | 0.5371 | 0.9595 | 0.0981 |
| N3_T300_none | pair | 0.1 | relabelled | 0.050 | 0.040 | 0.0000 | 0.0000 | 0.0004 | 0.5000 |
| N3_T300_none | pair | 0.25 | relabelled | 0.540 | 0.040 | 0.1909 | 0.0264 | 0.2499 | 0.4046 |
| N3_T300_none | pair | 0.5 | relabelled | 1.000 | 0.040 | 0.7495 | 0.4557 | 0.8777 | 0.1253 |
| N3_T300_none | pair | 1.0 | relabelled | 1.000 | 0.040 | 0.7495 | 0.4557 | 0.8777 | 0.1253 |
| N3_T300_none | pair | 0.1 | histogram | 0.060 | 0.030 | 0.0000 | 0.0000 | 0.0038 | 0.5000 |
| N3_T300_none | pair | 0.25 | histogram | 0.180 | 0.030 | 0.0000 | 0.0000 | 0.0474 | 0.5000 |
| N3_T300_none | pair | 0.5 | histogram | 0.710 | 0.030 | 0.3771 | 0.1051 | 0.4191 | 0.3115 |
| N3_T300_none | pair | 1.0 | histogram | 1.000 | 0.030 | 0.7662 | 0.4796 | 0.9022 | 0.1169 |
| N3_T300_none | pair | 0.1 | summaries | 0.190 | 0.060 | 0.0000 | 0.0000 | 0.0291 | 0.5000 |
| N3_T300_none | pair | 0.25 | summaries | 0.550 | 0.060 | 0.1692 | 0.0207 | 0.2272 | 0.4154 |
| N3_T300_none | pair | 0.5 | summaries | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N3_T300_none | pair | 1.0 | summaries | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N3_T300_none | pair | 0.1 | metadata | 0.190 | 0.010 | 0.0000 | 0.0000 | 0.0779 | 0.5000 |
| N3_T300_none | pair | 0.25 | metadata | 0.990 | 0.010 | 0.7805 | 0.5008 | 0.9192 | 0.1098 |
| N3_T300_none | pair | 0.5 | metadata | 1.000 | 0.010 | 0.8038 | 0.5371 | 0.9595 | 0.0981 |
| N3_T300_none | pair | 1.0 | metadata | 1.000 | 0.010 | 0.8038 | 0.5371 | 0.9595 | 0.0981 |
| N6_T100_none | pair | 0.1 | full | 0.040 | 0.020 | 0.0000 | 0.0000 | 0.0025 | 0.5000 |
| N6_T100_none | pair | 0.25 | full | 0.260 | 0.020 | 0.0000 | 0.0000 | 0.1001 | 0.5000 |
| N6_T100_none | pair | 0.5 | full | 0.930 | 0.020 | 0.6606 | 0.3430 | 0.7445 | 0.1697 |
| N6_T100_none | pair | 1.0 | full | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | star3 | 0.25 | full | 0.220 | 0.020 | 0.0000 | 0.0000 | 0.0786 | 0.5000 |
| N6_T100_none | star3 | 0.5 | full | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | star3 | 1.0 | full | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | two_pairs | 0.25 | full | 0.420 | 0.020 | 0.1183 | 0.0101 | 0.1987 | 0.4409 |
| N6_T100_none | two_pairs | 0.5 | full | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | two_pairs | 1.0 | full | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | pair | 0.1 | identified | 0.040 | 0.020 | 0.0000 | 0.0000 | 0.0025 | 0.5000 |
| N6_T100_none | pair | 0.25 | identified | 0.260 | 0.020 | 0.0000 | 0.0000 | 0.1001 | 0.5000 |
| N6_T100_none | pair | 0.5 | identified | 0.930 | 0.020 | 0.6606 | 0.3430 | 0.7445 | 0.1697 |
| N6_T100_none | pair | 1.0 | identified | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | star3 | 0.25 | identified | 0.220 | 0.020 | 0.0000 | 0.0000 | 0.0786 | 0.5000 |
| N6_T100_none | star3 | 0.5 | identified | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | star3 | 1.0 | identified | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | two_pairs | 0.25 | identified | 0.420 | 0.020 | 0.1183 | 0.0101 | 0.1987 | 0.4409 |
| N6_T100_none | two_pairs | 0.5 | identified | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | two_pairs | 1.0 | identified | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | pair | 0.1 | relabelled | 0.020 | 0.030 | 0.0000 | 0.0000 | 0.0007 | 0.5000 |
| N6_T100_none | pair | 0.25 | relabelled | 0.180 | 0.030 | 0.0000 | 0.0000 | 0.0474 | 0.5000 |
| N6_T100_none | pair | 0.5 | relabelled | 0.470 | 0.030 | 0.1439 | 0.0150 | 0.2154 | 0.4281 |
| N6_T100_none | pair | 1.0 | relabelled | 1.000 | 0.030 | 0.7662 | 0.4796 | 0.9022 | 0.1169 |
| N6_T100_none | star3 | 0.25 | relabelled | 0.150 | 0.030 | 0.0000 | 0.0000 | 0.0344 | 0.5000 |
| N6_T100_none | star3 | 0.5 | relabelled | 0.930 | 0.030 | 0.6427 | 0.3229 | 0.7187 | 0.1787 |
| N6_T100_none | star3 | 1.0 | relabelled | 1.000 | 0.030 | 0.7662 | 0.4796 | 0.9022 | 0.1169 |
| N6_T100_none | two_pairs | 0.25 | relabelled | 0.140 | 0.030 | 0.0000 | 0.0000 | 0.0302 | 0.5000 |
| N6_T100_none | two_pairs | 0.5 | relabelled | 0.960 | 0.030 | 0.6885 | 0.3759 | 0.7816 | 0.1558 |
| N6_T100_none | two_pairs | 1.0 | relabelled | 1.000 | 0.030 | 0.7662 | 0.4796 | 0.9022 | 0.1169 |
| N6_T100_none | pair | 0.1 | histogram | 0.060 | 0.030 | 0.0000 | 0.0000 | 0.0038 | 0.5000 |
| N6_T100_none | pair | 0.25 | histogram | 0.010 | 0.030 | 0.0000 | 0.0000 | 0.0038 | 0.5000 |
| N6_T100_none | pair | 0.5 | histogram | 0.020 | 0.030 | 0.0000 | 0.0000 | 0.0007 | 0.5000 |
| N6_T100_none | pair | 1.0 | histogram | 0.000 | 0.030 | 0.0000 | 0.0000 | 0.0152 | 0.5000 |
| N6_T100_none | star3 | 0.25 | histogram | 0.050 | 0.030 | 0.0000 | 0.0000 | 0.0019 | 0.5000 |
| N6_T100_none | star3 | 0.5 | histogram | 0.020 | 0.030 | 0.0000 | 0.0000 | 0.0007 | 0.5000 |
| N6_T100_none | star3 | 1.0 | histogram | 0.240 | 0.030 | 0.0000 | 0.0000 | 0.0763 | 0.5000 |
| N6_T100_none | two_pairs | 0.25 | histogram | 0.020 | 0.030 | 0.0000 | 0.0000 | 0.0007 | 0.5000 |
| N6_T100_none | two_pairs | 0.5 | histogram | 0.030 | 0.030 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| N6_T100_none | two_pairs | 1.0 | histogram | 0.200 | 0.030 | 0.0000 | 0.0000 | 0.0567 | 0.5000 |
| N6_T100_none | pair | 0.1 | summaries | 0.050 | 0.020 | 0.0000 | 0.0000 | 0.0050 | 0.5000 |
| N6_T100_none | pair | 0.25 | summaries | 0.040 | 0.020 | 0.0000 | 0.0000 | 0.0025 | 0.5000 |
| N6_T100_none | pair | 0.5 | summaries | 0.070 | 0.020 | 0.0000 | 0.0000 | 0.0111 | 0.5000 |
| N6_T100_none | pair | 1.0 | summaries | 0.440 | 0.020 | 0.1355 | 0.0133 | 0.2125 | 0.4323 |
| N6_T100_none | star3 | 0.25 | summaries | 0.070 | 0.020 | 0.0000 | 0.0000 | 0.0111 | 0.5000 |
| N6_T100_none | star3 | 0.5 | summaries | 0.050 | 0.020 | 0.0000 | 0.0000 | 0.0050 | 0.5000 |
| N6_T100_none | star3 | 1.0 | summaries | 0.000 | 0.020 | 0.0000 | 0.0000 | 0.0101 | 0.5000 |
| N6_T100_none | two_pairs | 0.25 | summaries | 0.080 | 0.020 | 0.0000 | 0.0000 | 0.0146 | 0.5000 |
| N6_T100_none | two_pairs | 0.5 | summaries | 0.420 | 0.020 | 0.1183 | 0.0101 | 0.1987 | 0.4409 |
| N6_T100_none | two_pairs | 1.0 | summaries | 0.970 | 0.020 | 0.7231 | 0.4199 | 0.8320 | 0.1384 |
| N6_T100_none | pair | 0.1 | metadata | 0.040 | 0.020 | 0.0000 | 0.0000 | 0.0025 | 0.5000 |
| N6_T100_none | pair | 0.25 | metadata | 0.260 | 0.020 | 0.0000 | 0.0000 | 0.1001 | 0.5000 |
| N6_T100_none | pair | 0.5 | metadata | 0.930 | 0.020 | 0.6606 | 0.3430 | 0.7445 | 0.1697 |
| N6_T100_none | pair | 1.0 | metadata | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | star3 | 0.25 | metadata | 0.220 | 0.020 | 0.0000 | 0.0000 | 0.0786 | 0.5000 |
| N6_T100_none | star3 | 0.5 | metadata | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | star3 | 1.0 | metadata | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | two_pairs | 0.25 | metadata | 0.420 | 0.020 | 0.1183 | 0.0101 | 0.1987 | 0.4409 |
| N6_T100_none | two_pairs | 0.5 | metadata | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T100_none | two_pairs | 1.0 | metadata | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T300_none | pair | 0.1 | full | 0.160 | 0.070 | 0.0000 | 0.0000 | 0.0147 | 0.5000 |
| N6_T300_none | pair | 0.25 | full | 0.880 | 0.070 | 0.5120 | 0.1984 | 0.5506 | 0.2440 |
| N6_T300_none | pair | 0.5 | full | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | pair | 1.0 | full | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | parity | 1.0 | full | 0.100 | 0.070 | 0.0000 | 0.0000 | 0.0021 | 0.5000 |
| N6_T300_none | star3 | 0.25 | full | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | star3 | 0.5 | full | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | star3 | 1.0 | full | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 0.25 | full | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 0.5 | full | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 1.0 | full | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | pair | 0.1 | identified | 0.160 | 0.070 | 0.0000 | 0.0000 | 0.0147 | 0.5000 |
| N6_T300_none | pair | 0.25 | identified | 0.880 | 0.070 | 0.5120 | 0.1984 | 0.5506 | 0.2440 |
| N6_T300_none | pair | 0.5 | identified | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | pair | 1.0 | identified | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | parity | 1.0 | identified | 0.100 | 0.070 | 0.0000 | 0.0000 | 0.0021 | 0.5000 |
| N6_T300_none | star3 | 0.25 | identified | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | star3 | 0.5 | identified | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | star3 | 1.0 | identified | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 0.25 | identified | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 0.5 | identified | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 1.0 | identified | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | pair | 0.1 | relabelled | 0.060 | 0.070 | 0.0000 | 0.0000 | 0.0003 | 0.5000 |
| N6_T300_none | pair | 0.25 | relabelled | 0.160 | 0.070 | 0.0000 | 0.0000 | 0.0147 | 0.5000 |
| N6_T300_none | pair | 0.5 | relabelled | 0.910 | 0.070 | 0.5519 | 0.2325 | 0.5985 | 0.2240 |
| N6_T300_none | pair | 1.0 | relabelled | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | parity | 1.0 | relabelled | 0.170 | 0.070 | 0.0000 | 0.0000 | 0.0175 | 0.5000 |
| N6_T300_none | star3 | 0.25 | relabelled | 0.340 | 0.070 | 0.0000 | 0.0000 | 0.0864 | 0.5000 |
| N6_T300_none | star3 | 0.5 | relabelled | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | star3 | 1.0 | relabelled | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 0.25 | relabelled | 0.360 | 0.070 | 0.0000 | 0.0000 | 0.0966 | 0.5000 |
| N6_T300_none | two_pairs | 0.5 | relabelled | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 1.0 | relabelled | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | pair | 0.1 | histogram | 0.010 | 0.020 | 0.0000 | 0.0000 | 0.0012 | 0.5000 |
| N6_T300_none | pair | 0.25 | histogram | 0.000 | 0.020 | 0.0000 | 0.0000 | 0.0101 | 0.5000 |
| N6_T300_none | pair | 0.5 | histogram | 0.000 | 0.020 | 0.0000 | 0.0000 | 0.0101 | 0.5000 |
| N6_T300_none | pair | 1.0 | histogram | 0.010 | 0.020 | 0.0000 | 0.0000 | 0.0012 | 0.5000 |
| N6_T300_none | parity | 1.0 | histogram | 0.150 | 0.020 | 0.0000 | 0.0000 | 0.0439 | 0.5000 |
| N6_T300_none | star3 | 0.25 | histogram | 0.000 | 0.020 | 0.0000 | 0.0000 | 0.0101 | 0.5000 |
| N6_T300_none | star3 | 0.5 | histogram | 0.000 | 0.020 | 0.0000 | 0.0000 | 0.0101 | 0.5000 |
| N6_T300_none | star3 | 1.0 | histogram | 0.610 | 0.020 | 0.2927 | 0.0627 | 0.3457 | 0.3537 |
| N6_T300_none | two_pairs | 0.25 | histogram | 0.000 | 0.020 | 0.0000 | 0.0000 | 0.0101 | 0.5000 |
| N6_T300_none | two_pairs | 0.5 | histogram | 0.040 | 0.020 | 0.0000 | 0.0000 | 0.0025 | 0.5000 |
| N6_T300_none | two_pairs | 1.0 | histogram | 0.860 | 0.020 | 0.5670 | 0.2463 | 0.6267 | 0.2165 |
| N6_T300_none | pair | 0.1 | summaries | 0.040 | 0.020 | 0.0000 | 0.0000 | 0.0025 | 0.5000 |
| N6_T300_none | pair | 0.25 | summaries | 0.100 | 0.020 | 0.0000 | 0.0000 | 0.0222 | 0.5000 |
| N6_T300_none | pair | 0.5 | summaries | 0.320 | 0.020 | 0.0368 | 0.0010 | 0.1348 | 0.4816 |
| N6_T300_none | pair | 1.0 | summaries | 0.860 | 0.020 | 0.5670 | 0.2463 | 0.6267 | 0.2165 |
| N6_T300_none | parity | 1.0 | summaries | 0.030 | 0.020 | 0.0000 | 0.0000 | 0.0007 | 0.5000 |
| N6_T300_none | star3 | 0.25 | summaries | 0.120 | 0.020 | 0.0000 | 0.0000 | 0.0305 | 0.5000 |
| N6_T300_none | star3 | 0.5 | summaries | 0.180 | 0.020 | 0.0000 | 0.0000 | 0.0582 | 0.5000 |
| N6_T300_none | star3 | 1.0 | summaries | 0.050 | 0.020 | 0.0000 | 0.0000 | 0.0050 | 0.5000 |
| N6_T300_none | two_pairs | 0.25 | summaries | 0.200 | 0.020 | 0.0000 | 0.0000 | 0.0682 | 0.5000 |
| N6_T300_none | two_pairs | 0.5 | summaries | 0.790 | 0.020 | 0.4833 | 0.1757 | 0.5323 | 0.2584 |
| N6_T300_none | two_pairs | 1.0 | summaries | 1.000 | 0.020 | 0.7841 | 0.5063 | 0.9290 | 0.1079 |
| N6_T300_none | pair | 0.1 | metadata | 0.160 | 0.070 | 0.0000 | 0.0000 | 0.0147 | 0.5000 |
| N6_T300_none | pair | 0.25 | metadata | 0.880 | 0.070 | 0.5120 | 0.1984 | 0.5506 | 0.2440 |
| N6_T300_none | pair | 0.5 | metadata | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | pair | 1.0 | metadata | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | parity | 1.0 | metadata | 0.100 | 0.070 | 0.0000 | 0.0000 | 0.0021 | 0.5000 |
| N6_T300_none | star3 | 0.25 | metadata | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | star3 | 0.5 | metadata | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | star3 | 1.0 | metadata | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 0.25 | metadata | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 0.5 | metadata | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N6_T300_none | two_pairs | 1.0 | metadata | 1.000 | 0.070 | 0.7037 | 0.3948 | 0.8135 | 0.1482 |
| N3_T300_last_round | pair | 0.5 | full | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N3_T300_last_round | pair | 1.0 | full | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N3_T300_last_round | pair | 0.5 | identified | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N3_T300_last_round | pair | 1.0 | identified | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N3_T300_last_round | pair | 0.5 | relabelled | 0.830 | 0.050 | 0.4797 | 0.1730 | 0.5175 | 0.2601 |
| N3_T300_last_round | pair | 1.0 | relabelled | 1.000 | 0.050 | 0.7336 | 0.4339 | 0.8550 | 0.1332 |
| N3_T300_last_round | pair | 0.5 | histogram | 0.690 | 0.030 | 0.3560 | 0.0934 | 0.3989 | 0.3220 |
| N3_T300_last_round | pair | 1.0 | histogram | 1.000 | 0.030 | 0.7662 | 0.4796 | 0.9022 | 0.1169 |
| N3_T300_last_round | pair | 0.5 | summaries | 0.950 | 0.030 | 0.6726 | 0.3569 | 0.7593 | 0.1637 |
| N3_T300_last_round | pair | 1.0 | summaries | 1.000 | 0.030 | 0.7662 | 0.4796 | 0.9022 | 0.1169 |
| N3_T300_last_round | pair | 0.5 | metadata | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N3_T300_last_round | pair | 1.0 | metadata | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N6_T300_last_round | pair | 0.5 | full | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N6_T300_last_round | pair | 1.0 | full | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N6_T300_last_round | pair | 0.5 | identified | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N6_T300_last_round | pair | 1.0 | identified | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N6_T300_last_round | pair | 0.5 | relabelled | 0.210 | 0.020 | 0.0000 | 0.0000 | 0.0734 | 0.5000 |
| N6_T300_last_round | pair | 1.0 | relabelled | 0.790 | 0.020 | 0.4833 | 0.1757 | 0.5323 | 0.2584 |
| N6_T300_last_round | pair | 0.5 | histogram | 0.000 | 0.070 | 0.0000 | 0.0000 | 0.0359 | 0.5000 |
| N6_T300_last_round | pair | 1.0 | histogram | 0.020 | 0.070 | 0.0000 | 0.0000 | 0.0111 | 0.5000 |
| N6_T300_last_round | pair | 0.5 | summaries | 0.120 | 0.080 | 0.0000 | 0.0000 | 0.0032 | 0.5000 |
| N6_T300_last_round | pair | 1.0 | summaries | 0.760 | 0.080 | 0.3547 | 0.0927 | 0.3828 | 0.3227 |
| N6_T300_last_round | pair | 0.5 | metadata | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N6_T300_last_round | pair | 1.0 | metadata | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N6_T300_public_state | pair | 0.5 | full | 0.550 | 0.060 | 0.1692 | 0.0207 | 0.2272 | 0.4154 |
| N6_T300_public_state | pair | 1.0 | full | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N6_T300_public_state | pair | 0.5 | identified | 0.550 | 0.060 | 0.1692 | 0.0207 | 0.2272 | 0.4154 |
| N6_T300_public_state | pair | 1.0 | identified | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |
| N6_T300_public_state | pair | 0.5 | histogram | 0.170 | 0.080 | 0.0000 | 0.0000 | 0.0136 | 0.5000 |
| N6_T300_public_state | pair | 1.0 | histogram | 1.000 | 0.080 | 0.6894 | 0.3770 | 0.7943 | 0.1553 |
| N6_T300_public_state | pair | 0.5 | summaries | 0.840 | 0.040 | 0.5077 | 0.1949 | 0.5513 | 0.2462 |
| N6_T300_public_state | pair | 1.0 | summaries | 1.000 | 0.040 | 0.7495 | 0.4557 | 0.8777 | 0.1253 |
| N6_T300_public_state | pair | 0.5 | metadata | 0.550 | 0.060 | 0.1692 | 0.0207 | 0.2272 | 0.4154 |
| N6_T300_public_state | pair | 1.0 | metadata | 1.000 | 0.060 | 0.7184 | 0.4136 | 0.8337 | 0.1408 |

Calibration and feature parameters: `view_calibration.json`. Test decisions and data volume: `view_predictions.json`. Machine-readable bounds: `information.json` and `view_comparison.csv`. Raw/gzip sizes measure serialized volume; neither is mutual information about C.

Assumptions: independent episodes within each arm, independently known labels, matching honest policy populations and unchanged observation mechanisms across splits. Rank control is marginal under exchangeability. Reused honest tests across comparisons are handled by the family union bound. Message missingness in any episode excludes that view for the whole group.

### Unavailable views

- N6_T300_public_state / relabelled: relabelled views require none or last_round to avoid identity leakage in state.

## Scope

The exact model and the empirical population are separate experiments. There is no general finite-sample upper bound on view information from these trajectories. No KDE estimate, asymptotic nearest-neighbour bracket, membership/Fano claim, or extrapolated round requirement is presented as a certified finite-sample limit. Anonymous counts have a fixed-sum constraint; dependence between count streams alone is not evidence of collusion. The empirical detector is calibrated against honest count distributions.
