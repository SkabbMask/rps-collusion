# Delivery contents

## Subsequent six-view information extension

`information.py` and `rps/info/` add RPS-adapted full, identified, relabelled,
histogram, summaries and metadata information experiments. Exact small-model
limits are separate from held-out confidence lower bounds. Public messages
are optional; historical message missingness is not treated as silence.
Instructions: `docs/INFORMATION.md`. Executed scripted demonstration:
`results/information-demo/report.md`. The extended suite has 47 passing tests;
the original delivery checks below describe the earlier detector benchmark.

This is a local fork of `SkabbMask/rps-collusion` on branch `feat/rps-colav-detection`,
starting at `4e11fd2efbe5fc9d5d80bec7d32522514ac2896f`. It has not been published to a GitHub account.
The ZIP contains the full updated working tree, including source, tests, datasets, calibration,
the Markdown report, CSV tables, and PNG/SVG figures. Git internals and machine-specific run logs
are excluded. The companion `rps-colav-code.patch` applies the code/documentation changes to a
clean checkout of the pinned target base; it excludes the generated benchmark results.

Start with `README.md`; the completed report is `results/benchmark/report.md`.
The detailed API and external-JSONL instructions are in `docs/DETECTION.md`.

## Checks completed

- 36 unit/regression tests passed; exact command output is in `results/validation.txt`.
- The sequential random-player path exactly reproduces a 12-round fixture generated from the
  pinned upstream threaded implementation, including moves, private signals and payoffs.
- Full evaluation executed 6,253 episodes: 1,953 honest edge-fit/calibration episodes and
  4,300 held-out tests across seven calibration groups. It produced 17,200 method decisions.
- Replaying all 616 episodes of the small dataset reproduced every one of its 840 held-out
  decisions and all scores exactly (timings excluded), with an identical canonical dataset hash.
- External-log manifest import was exercised with 19 edge-fit, 39 calibration, and four held-out
  JSONL logs. This validates the importer; it is not an additional performance estimate.
- Simulator-integrated detection and standalone `analyze.py` gave identical results on the same log.
- A wheel built successfully with `python -m pip wheel --no-deps --no-build-isolation .`.
- All five PNG figures were visually inspected; vector SVG versions are also included.
- The result audit checked disjoint fit/calibration/test IDs, all expected prediction counts,
  and the absence of named groups after rejected detections.

The main results use scripted random players and explicit synthetic stress controls. No
Ollama/LLM experiment was run. Real logs can be evaluated once independently labeled test data
and matching honest calibration data are supplied.

## Run

```bash
python -m pip install -r requirements-detection.txt
python -m unittest discover -s tests -v
python evaluate.py --config configs/benchmark.json --output results/reproduction
```

To apply the code-only patch to a clean checkout of the pinned base, create a branch and run
`git apply /path/to/rps-colav-code.patch`. Generate results with the command above, or use the
completed results in the full ZIP.
