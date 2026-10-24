# Growth detector validation

Validated on 25 October 2026 using growth source revision
`1bea281244ad2d756b9b507145b2046fefcf1582`. See
[ATTRIBUTION.md](../ATTRIBUTION.md) for the source mapping.

- All 55 tests passed, including reference growth sequences.
- Full evaluation: 6,253 games, 4,300 test games, 21,500 method decisions.
- The original four methods' 17,200 decisions matched the previous full run
  for flags, abstention, scores, groups and recovery metrics. Timing was excluded.
- Single-game analysis agreed with analysis run directly after simulation.
- All 208 information comparisons matched the previous full information run.
- Five-method plots and historical four-method report generation were checked.

## Main comparison

This subset contains 2,800 coalition games and 400 honest games. Stress cases
are separate. The results use scripted players, not LLMs.

| Method | Recall | False positives | Exact group recovery |
| --- | ---: | ---: | ---: |
| Bonjour | 59.29% | 2.50% | 51.32% |
| Graph components | 79.71% | 2.75% | 73.07% |
| Louvain | 79.71% | 2.75% | 73.29% |
| ML decoder | 81.71% | 4.00% | 56.29% |
| Average-z growth | 81.50% | 3.75% | 57.75% |

Growth returns one group and cannot recover two disjoint groups. Its join
fraction was fixed at 0.25. These aggregate results do not establish a best
method for every scenario.

## Reproduce

Follow [RUNNING.md](RUNNING.md), or rerun saved episodes using `--dataset`.
The validation dataset SHA-256 is
`63bece830ad7a898b85fd97886466996a2bc31cd19da41d0bab645397a58ec1f`.
The recorded environment was Python 3.14.8, NumPy 2.4.6, SciPy 1.17.1,
NetworkX 3.4.2, Matplotlib 3.10.9+dfsg1 and scikit-learn 1.9.0. Consult each
run's `provenance.json` for its own versions and timings.
