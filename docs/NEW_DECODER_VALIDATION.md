# new-decoder integration validation

Validated on the local `dev` working tree on 25 October 2026. The added average-z growth algorithm is adapted from teammate revision `1bea281244ad2d756b9b507145b2046fefcf1582`.

- 55 unit/regression tests passed, including fixtures executed from the pinned upstream function.
- Full scripted evaluation completed: 6,253 episodes, 4,300 held-out tests and 21,500 method decisions.
- All 17,200 decisions from the original four methods matched the existing `results/full-detection/predictions.csv` for flags, abstentions, scores, groups, exact-group recovery and pair F1. Runtime was excluded.
- Standalone and simulator-integrated five-method analysis agreed exactly apart from runtime.
- The six-view information stage completed on the same 6,253 episodes. All 208 comparisons matched the existing `results/full-information` report; the opaque recorded-state group excludes relabelling.
- Five-method false-alarm/runtime and recovery plots were visually inspected.
- Historical four-method report regeneration was exercised on a temporary copy; the existing result directories were preserved.

## Main benchmark mixture

This table covers the same main mixture as the report: 2,800 positive and 400 matched honest test episodes. Stress populations are separate. These are scripted/synthetic results, not LLM results.

| Method | Recall | FPR | Exact groups |
| --- | ---: | ---: | ---: |
| bonjour | 59.29% | 2.50% | 51.32% |
| graph_components | 79.71% | 2.75% | 73.07% |
| graph_louvain | 79.71% | 2.75% | 73.29% |
| decoder | 81.71% | 4.00% | 56.29% |
| growth | 81.50% | 3.75% | 57.75% |

Growth has a single-group output, so it cannot represent two disjoint coalitions. The default relative join fraction was fixed at 0.25 before evaluation. These aggregate numbers do not establish superiority across policies or coalition structures.

## Reproduce

```bash
python -m unittest discover -s tests -v
python evaluate.py --config configs/full-information.json --output results/full-detection-growth
python information.py --config configs/full-information.json \
  --dataset results/full-detection-growth/episodes.jsonl.gz \
  --relabel-interval 25 --output results/full-information-growth
```

Complete setup and single-game commands: [RUNNING.md](RUNNING.md).

Canonical full dataset SHA-256: `63bece830ad7a898b85fd97886466996a2bc31cd19da41d0bab645397a58ec1f`.
Validation environment: Python 3.14.8; {'numpy': '2.4.6', 'scipy': '1.17.1', 'networkx': '3.4.2', 'matplotlib': '3.10.9+dfsg1', 'scikit-learn': '1.9.0'}.
Full execution artifacts from this validation are in `/tmp/rps-new-decoder-full-validation`; they are temporary. Reproduce into the project paths above for durable results.

Information execution artifacts are in `/tmp/rps-new-decoder-information-validation`.
