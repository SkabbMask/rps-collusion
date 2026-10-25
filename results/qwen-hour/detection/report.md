# RPS detection results

Source: **external labeled logs**. 6 games total;
2 held-out tests. Evaluation: 0.0 seconds.
Generated datasets use scripted players, not LLMs. Imported logs use the labels supplied in their manifest.

## Main comparison

This mixture contains 2 test games. Original Bonjour has the highest recall
(100.0%); Original Bonjour has the highest exact group recovery
(100.0%). These are descriptive comparisons of this mixture.

| Method | FPR [95% CI] | Recall [95% CI] | Precision | F1 | Macro AUROC | Exact groups | Jaccard | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original Bonjour | 100.0% [20.7%, 100.0%] | 100.0% [20.7%, 100.0%] | 0.500 | 0.667 | 1.000 | 1.000 | 1.000 | 0.14 |
| Graph components | 0.0% [0.0%, 79.3%] | 0.0% [0.0%, 79.3%] | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.85 |
| Graph Louvain | 0.0% [0.0%, 79.3%] | 0.0% [0.0%, 79.3%] | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.81 |
| ML decoder | 0.0% [0.0%, 79.3%] | 0.0% [0.0%, 79.3%] | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.12 |
| Average-z growth | 0.0% [0.0%, 79.3%] | 0.0% [0.0%, 79.3%] | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.86 |

Intervals are 95% Wilson intervals. Recovery counts missed detections and abstentions as failures.
Honest games are counted once. Precision and F1 depend on the chosen prevalence.
Macro AUROC averages within-condition AUROCs; scores from different calibrations are not comparable.
Each added method's runtime includes the shared pair-evidence calculation, as if it ran alone.

## Methods and calibration

Methods in this report: Original Bonjour, Graph components, Graph Louvain, ML decoder, Average-z growth.
All use pairwise action dependence. The graph methods can return several groups;
ML and growth return one. No private signals or true memberships enter their features.

Pair scores use conditional MI, standardized against shuffled moves within each public state.
The settings are 19 shuffles,
a minimum of 10 observations per pair and
10 per retained state.
Missing evidence remains undefined. The baseline uses complete rounds.

Honest edge-fit games set the graph threshold. Separate honest calibration games set final
score thresholds before testing. Rank cutoffs target a marginal 5% false-positive rate when
calibration and future honest games are exchangeable. They do not guarantee a particular
realized rate for each fitted threshold or for a changed policy population.
Bonjour keeps its original 0.05-nat threshold. Combining methods needs separate calibration.

Growth uses the strongest-pair z for detection and relative average-z expansion for naming.
Its default join fraction is 0.25. ML uses an estimated Gaussian score and a group-size penalty;
that working model does not establish independent pair evidence or a posterior probability.
Sources and revisions are listed in the repository's ATTRIBUTION.md.

| Calibration group | Edge fit | Calibration | Held-out test |
| --- | --- | --- | --- |
| N3_T15_none | 2 | 2 | 2 |

## Scenarios

`pair` has one primary and one assistant. `star3` has one primary and two assistants.
`two_pairs` has two separate primary/assistant pairs. CP is the chance of a private signal;
labels describe the configured mechanism, not an inferred intention. Membership is randomized
per episode. Seed: 20261025. Duplicate IDs and observed trajectories are rejected.

`parity` is a synthetic higher-order case with pairwise-independent coalition actions.
`shared_state_ignored` tests honest players responding to an omitted public cause; it is outside
the matched calibration population. `public_state` includes that cause. `last_round` tests
previous-action conditioning, which can leave too few observations per state.

## Paired comparison with Bonjour

| Compared with original | Metric | Difference [paired bootstrap 95% CI] |
| --- | --- | --- |
| Graph components | Recall | -100.0% [-100.0%, -100.0%] |
| Graph components | Exact groups | -100.0% [-100.0%, -100.0%] |
| Graph Louvain | Recall | -100.0% [-100.0%, -100.0%] |
| Graph Louvain | Exact groups | -100.0% [-100.0%, -100.0%] |
| ML decoder | Recall | -100.0% [-100.0%, -100.0%] |
| ML decoder | Exact groups | -100.0% [-100.0%, -100.0%] |
| Average-z growth | Recall | -100.0% [-100.0%, -100.0%] |
| Average-z growth | Exact groups | -100.0% [-100.0%, -100.0%] |

Intervals use 2,000 paired episode-bootstrap resamples of this fixed test mixture.
They exclude calibration-set uncertainty and are not adjusted for multiple comparisons.

## Figures

![Power](figures/power.png)

![False Alarms Runtime](figures/false_alarms_runtime.png)

PNG and SVG files are saved in `figures/`. ROC sweeps do not change the frozen decisions.
The example graph is the first N=6, T=300, CP=1 star case, selected by position rather than success.

## Results by condition

| Group | Scenario | CP | Method | Recall | FPR | AUROC | Exact groups | Pair F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N3_T15_none | pair | 1.0 | Original Bonjour | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| N3_T15_none | pair | 1.0 | Graph components | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 |
| N3_T15_none | pair | 1.0 | Graph Louvain | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 |
| N3_T15_none | pair | 1.0 | ML decoder | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| N3_T15_none | pair | 1.0 | Average-z growth | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

Exact groups requires the complete set of groups to match. Exact members compares their union.
Jaccard and pair F1 measure overlap; their denominators include missed detections.

## Stress tests

| Group | Scenario | Method | n | Flag rate [95% CI] | Abstain | Supported pairs |
| --- | --- | --- | --- | --- | --- | --- |

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

Python: 3.14.8. Packages: {'numpy': '2.5.4', 'scipy': '1.18.1', 'networkx': '3.7', 'matplotlib': '3.11.2', 'scikit-learn': '1.9.1'}.
Dataset SHA-256: `b555d78748d7ba3893cb7870c01df3954bf9312f1a6ff587d077123aed2f11f3`.
