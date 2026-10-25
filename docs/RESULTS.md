# Results and replication

## Reference runs

| Directory | Population and role |
| --- | --- |
| `results/full-detection-growth/` | Five-method scripted benchmark used for the main comparison |
| `results/full-information/` | Empirical information-view analysis of the scripted data; historical report also contains the excluded toy diagnostic |
| `results/qwen-hour/` | Six Qwen2.5:14b games; one unassigned and one coalition held-out test |
| `results/full-detection/` | Earlier four-method analysis on the scripted dataset |
| `results/benchmark/` | Earlier four-method benchmark |
| `results/my-first-test/`, `results/information-demo/` | Development checks |

Incomplete local model collections are working data, not published evidence.
Do not infer model identity or completion from a directory name. Consult the
saved configuration, manifest, quality and provenance files.

## Scripted benchmark

The main comparison contains 2,800 coalition and 400 honest test games:
800 coalition/200 honest at N=3, and 2,000 coalition/200 honest at N=6.
Horizons are 100 and 300 rounds. N=6 includes additional coalition structures,
so pooled metrics are weighted toward those cases. Report `comparison.csv`
when discussing a specific N, horizon or coalition type.

| Method | Recall | False-positive rate | Exact group recovery |
| --- | ---: | ---: | ---: |
| Net-influence adapter | 59.29% | 2.50% | 51.32% |
| Components | 79.71% | 2.75% | 73.07% |
| Louvain | 79.71% | 2.75% | 73.29% |
| Likelihood search | 81.71% | 4.00% | 56.29% |
| Average-z growth | 81.50% | 3.75% | 57.75% |

These are exploratory results under the saved calibration. The net-influence
adapter returns every qualifying pair at 0.05 nats; added methods use honest
calibration. The comparison is not at exactly matched FPR. Graph methods can
return several groups; likelihood search and growth return one.

## Qwen case

The six-game check completed with zero parsing fallbacks. In its one held-out
coalition game, helper C followed primary B's private signal in 14/15 rounds.
Payoffs were B=14, C=5 and A=10. Two calibration games force infinite 5% rank
thresholds. The run establishes neither detection performance nor spontaneous
or message-mediated collusion. Do not pool it with scripted results.

## What reproduction means

[RUNNING.md](RUNNING.md) provides both fresh generation and saved-data replay.
Compare `summary.csv` and `comparison.csv` for detection/recovery counts;
`predictions.csv` records each test decision and group estimate. Information
bounds are in `view_comparison.csv`. Timing and figure file bytes can differ
between machines. Floating-point scores may differ across package versions;
record versions and compare numerical values with an appropriate tolerance.

Check a fresh detection output against the saved reference:

```bash
python scripts/compare_results.py results/full-detection-growth \
  results/replication/detection
```

The check compares summary metrics, condition metrics and per-game predictions,
ignoring timing and diagnostic details. It checks group labels and accepts only
small floating-point differences. It exits unsuccessfully on mismatches.

Use the saved `config.json`, not a similarly named current example, when
replaying a reference run. Existing `provenance.json` files identify recorded
versions and dataset hashes. They are not proof that every uncommitted source
change was archived. A publication should cite the released commit and archive
its resolved configuration and environment. Replaying previously inspected
data does not provide an untouched test set.

## Excluded diagnostic

The exact synthetic information model is excluded from the paper because it
was inconclusive for the empirical question. Historical `exact_views.json`
files and old reports are preserved for traceability. Normal information
analysis and paper-figure exports no longer generate it. Its equations remain
a toy-model calculation, not an estimate of information in Qwen games.

For broader limitations, see [PAPER_READINESS_REVIEW.md](PAPER_READINESS_REVIEW.md).
