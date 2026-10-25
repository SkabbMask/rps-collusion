# RPS detection results

Source: **external labeled logs**. 439 games total;
300 held-out tests. Evaluation: 3.0 seconds.
These games were imported from logs (for example model games); labels come from their manifest.

## Main comparison

This mixture contains 300 test games. Original Bonjour has the highest recall
(94.5%, at 51.0% false positives); Original Bonjour has the
highest exact group recovery (92.5%, at 51.0% false
positives). Compare methods at their false-positive rates; these are descriptive comparisons of this mixture.

| Method | FPR [95% CI] | Recall [95% CI] | Precision | F1 | Macro AUROC | Exact groups | Jaccard | Median ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original Bonjour | 51.0% [41.3%, 60.6%] | 94.5% [90.4%, 96.9%] | 0.787 | 0.859 | 0.930 | 0.925 | 0.932 | 0.44 |
| Graph components | 3.0% [1.0%, 8.5%] | 80.5% [74.5%, 85.4%] | 0.982 | 0.885 | 0.933 | 0.750 | 0.787 | 1.70 |
| Graph Louvain | 3.0% [1.0%, 8.5%] | 80.5% [74.5%, 85.4%] | 0.982 | 0.885 | 0.933 | 0.750 | 0.787 | 1.88 |
| ML decoder | 3.0% [1.0%, 8.5%] | 80.5% [74.5%, 85.4%] | 0.982 | 0.885 | 0.952 | 0.780 | 0.797 | 2.06 |
| Average-z growth | 3.0% [1.0%, 8.5%] | 80.0% [73.9%, 85.0%] | 0.982 | 0.882 | 0.953 | 0.770 | 0.790 | 1.68 |

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
| N3_T30_none | 40 | 99 | 300 |

## Scenarios

`pair` has one primary and one assistant. CP is the chance of a private signal;
labels describe the configured mechanism, not an inferred intention. Membership is randomized
per episode. Seed: 20261025. Duplicate IDs are rejected; identical trajectories from independent games are allowed (0 occurred).

## Paired comparison with Bonjour

| Compared with original | Metric | Difference [paired bootstrap 95% CI] |
| --- | --- | --- |
| Graph components | Recall | -14.0% [-19.5%, -9.0%] |
| Graph components | Exact groups | -17.5% [-23.0%, -12.5%] |
| Graph Louvain | Recall | -14.0% [-19.5%, -9.0%] |
| Graph Louvain | Exact groups | -17.5% [-23.0%, -12.5%] |
| ML decoder | Recall | -14.0% [-19.0%, -9.0%] |
| ML decoder | Exact groups | -14.5% [-19.5%, -9.5%] |
| Average-z growth | Recall | -14.5% [-19.5%, -9.5%] |
| Average-z growth | Exact groups | -15.5% [-20.5%, -10.5%] |

Intervals use 2,000 paired episode-bootstrap resamples of this fixed test mixture.
They exclude calibration-set uncertainty and are not adjusted for multiple comparisons.

## Figures

![Power](figures/power.png)

![False Alarms Runtime](figures/false_alarms_runtime.png)

PNG and SVG files are saved in `figures/`. ROC sweeps do not change the frozen decisions.


## Results by condition

| Group | Scenario | CP | Method | Recall | FPR | AUROC | Exact groups | Pair F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N3_T30_none | pair | 0.5 | Original Bonjour | 0.890 | 0.510 | 0.859 | 0.850 | 0.850 |
| N3_T30_none | pair | 1.0 | Original Bonjour | 1.000 | 0.510 | 1.000 | 1.000 | 1.000 |
| N3_T30_none | pair | 0.5 | Graph components | 0.610 | 0.030 | 0.867 | 0.560 | 0.585 |
| N3_T30_none | pair | 1.0 | Graph components | 1.000 | 0.030 | 1.000 | 0.940 | 0.970 |
| N3_T30_none | pair | 0.5 | Graph Louvain | 0.610 | 0.030 | 0.867 | 0.560 | 0.585 |
| N3_T30_none | pair | 1.0 | Graph Louvain | 1.000 | 0.030 | 1.000 | 0.940 | 0.970 |
| N3_T30_none | pair | 0.5 | ML decoder | 0.610 | 0.030 | 0.904 | 0.560 | 0.585 |
| N3_T30_none | pair | 1.0 | ML decoder | 1.000 | 0.030 | 1.000 | 1.000 | 1.000 |
| N3_T30_none | pair | 0.5 | Average-z growth | 0.600 | 0.030 | 0.906 | 0.540 | 0.570 |
| N3_T30_none | pair | 1.0 | Average-z growth | 1.000 | 0.030 | 1.000 | 1.000 | 1.000 |

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
Dataset SHA-256: `2fcccb8fa3583e26b5fccc0e9e187fc9065585a8bfab29ded786b4f8fa0fe2ce`.

## Paper figures

[Figures and captions](figures/detection_captions.md). PNG (300 dpi), SVG and PDF are generated from this run's saved predictions.
