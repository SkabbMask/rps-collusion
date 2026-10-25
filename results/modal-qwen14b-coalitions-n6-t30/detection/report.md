# RPS detection results

Source: **external labeled logs**. 184 games total;
125 held-out tests. Evaluation: 3.0 seconds.
These games were imported from logs (for example model games); labels come from their manifest.

## Main comparison

This mixture contains 125 test games. ML decoder has the highest recall
(93.0%, at 12.0% false positives); Graph components has the
highest exact group recovery (79.0%, at 8.0% false
positives). Compare methods at their false-positive rates; these are descriptive comparisons of this mixture.

| Method | FPR [95% CI] | Recall [95% CI] | Precision | F1 | Macro AUROC | Exact groups | Jaccard | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original Bonjour | 56.0% [37.1%, 73.3%] | 90.0% [82.6%, 94.5%] | 0.865 | 0.882 | 0.851 | 0.440 | 0.722 | 1.60 |
| Graph components | 8.0% [2.2%, 25.0%] | 91.0% [83.8%, 95.2%] | 0.978 | 0.943 | 0.950 | 0.790 | 0.870 | 7.11 |
| Graph Louvain | 8.0% [2.2%, 25.0%] | 91.0% [83.8%, 95.2%] | 0.978 | 0.943 | 0.950 | 0.790 | 0.870 | 7.41 |
| ML decoder | 12.0% [4.2%, 30.0%] | 93.0% [86.3%, 96.6%] | 0.969 | 0.949 | 0.972 | 0.590 | 0.772 | 7.94 |
| Average-z growth | 12.0% [4.2%, 30.0%] | 91.0% [83.8%, 95.2%] | 0.968 | 0.938 | 0.972 | 0.570 | 0.753 | 7.10 |

Intervals are 95% Wilson intervals. Recovery counts missed detections and abstentions as failures.
Honest games are counted once. Precision and F1 depend on the chosen prevalence.
Macro AUROC averages within-condition AUROCs; scores from different calibrations are not comparable.
Each added method's runtime includes the shared pair-evidence calculation, as if it ran alone.

## Methods and calibration

Methods in this report: Original Bonjour, Graph components, Graph Louvain, ML decoder, Average-z growth.
All use pairwise action dependence. The graph methods can return several groups;
ML and growth return one. No private signals or true memberships enter their features.

Pair scores use conditional MI, standardized against shuffled moves within each public state.
The settings are 99 shuffles,
a minimum of 20 observations per pair and
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
| N6_T30_none | 20 | 39 | 125 |

## Scenarios

`pair` has one primary and one assistant. `star3` has one primary and two assistants. `two_pairs` has two separate primary/assistant pairs. CP is the chance of a private signal;
labels describe the configured mechanism, not an inferred intention. Membership is randomized
per episode. Seed: 20261025. Duplicate IDs are rejected; identical trajectories from independent games are allowed (0 occurred).

## Paired comparison with Bonjour

| Compared with original | Metric | Difference [paired bootstrap 95% CI] |
| --- | --- | --- |
| Graph components | Recall | +1.0% [-7.0%, +9.0%] |
| Graph components | Exact groups | +35.0% [+25.0%, +45.0%] |
| Graph Louvain | Recall | +1.0% [-7.0%, +9.0%] |
| Graph Louvain | Exact groups | +35.0% [+26.0%, +45.0%] |
| ML decoder | Recall | +3.0% [-5.0%, +11.0%] |
| ML decoder | Exact groups | +15.0% [-1.0%, +30.0%] |
| Average-z growth | Recall | +1.0% [-7.0%, +9.0%] |
| Average-z growth | Exact groups | +13.0% [-1.0%, +28.0%] |

Intervals use 2,000 paired episode-bootstrap resamples of this fixed test mixture.
They exclude calibration-set uncertainty and are not adjusted for multiple comparisons.

## Figures

![Power](figures/power.png)

![False Alarms Runtime](figures/false_alarms_runtime.png)

PNG and SVG files are saved in `figures/`. ROC sweeps do not change the frozen decisions.


## Results by condition

| Group | Scenario | CP | Method | Recall | FPR | AUROC | Exact groups | Pair F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N6_T30_none | pair | 1.0 | Original Bonjour | 1.000 | 0.560 | 1.000 | 0.880 | 0.960 |
| N6_T30_none | star3 | 0.5 | Original Bonjour | 0.840 | 0.560 | 0.723 | 0.000 | 0.408 |
| N6_T30_none | star3 | 1.0 | Original Bonjour | 0.760 | 0.560 | 0.682 | 0.000 | 0.352 |
| N6_T30_none | two_pairs | 1.0 | Original Bonjour | 1.000 | 0.560 | 1.000 | 0.880 | 0.976 |
| N6_T30_none | pair | 1.0 | Graph components | 1.000 | 0.080 | 1.000 | 1.000 | 1.000 |
| N6_T30_none | star3 | 0.5 | Graph components | 0.640 | 0.080 | 0.799 | 0.240 | 0.433 |
| N6_T30_none | star3 | 1.0 | Graph components | 1.000 | 0.080 | 1.000 | 0.960 | 0.987 |
| N6_T30_none | two_pairs | 1.0 | Graph components | 1.000 | 0.080 | 1.000 | 0.960 | 0.987 |
| N6_T30_none | pair | 1.0 | Graph Louvain | 1.000 | 0.080 | 1.000 | 1.000 | 1.000 |
| N6_T30_none | star3 | 0.5 | Graph Louvain | 0.640 | 0.080 | 0.799 | 0.240 | 0.433 |
| N6_T30_none | star3 | 1.0 | Graph Louvain | 1.000 | 0.080 | 1.000 | 0.960 | 0.976 |
| N6_T30_none | two_pairs | 1.0 | Graph Louvain | 1.000 | 0.080 | 1.000 | 0.960 | 0.987 |
| N6_T30_none | pair | 1.0 | ML decoder | 1.000 | 0.120 | 1.000 | 1.000 | 1.000 |
| N6_T30_none | star3 | 0.5 | ML decoder | 0.720 | 0.120 | 0.888 | 0.360 | 0.527 |
| N6_T30_none | star3 | 1.0 | ML decoder | 1.000 | 0.120 | 1.000 | 1.000 | 1.000 |
| N6_T30_none | two_pairs | 1.0 | ML decoder | 1.000 | 0.120 | 1.000 | 0.000 | 0.667 |
| N6_T30_none | pair | 1.0 | Average-z growth | 1.000 | 0.120 | 1.000 | 1.000 | 1.000 |
| N6_T30_none | star3 | 0.5 | Average-z growth | 0.640 | 0.120 | 0.888 | 0.360 | 0.496 |
| N6_T30_none | star3 | 1.0 | Average-z growth | 1.000 | 0.120 | 1.000 | 0.920 | 0.965 |
| N6_T30_none | two_pairs | 1.0 | Average-z growth | 1.000 | 0.120 | 1.000 | 0.000 | 0.667 |

Exact groups requires the complete set of groups to match. Exact members compares their union.
Jaccard and pair F1 measure overlap; their denominators include missed detections.

## Files and reproduction

See docs/RUNNING.md in the repository for complete commands. Use a fresh output directory.
To replay these trajectories, pass episodes.jsonl.gz to evaluate.py with --dataset.
Reports can be rebuilt with python -m rps.benchmark.report --results PATH.
Calibration must match the honest policies, state definition, player count and game length.

Predictions, condition metrics, summaries and thresholds are saved as CSV files.
calibration/*.json stores the thresholds; provenance.json records the run settings.
The six-view information experiment is a separate analysis of the observations.

Python: 3.12.1. Packages: {'numpy': '2.5.4', 'scipy': '1.18.1', 'networkx': '3.7', 'matplotlib': '3.11.2', 'scikit-learn': '1.9.1'}.
Dataset SHA-256: `21e99825770ac0d390ac47ef3df9f4ba28a773311f899ea583ab0a8bb2a8e6d0`.

## Paper figures

[Figures and captions](figures/detection_captions.md). PNG (300 dpi), SVG and PDF are generated from this run's saved predictions.
