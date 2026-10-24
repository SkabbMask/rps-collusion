# Using and extending RPS collusion detection

## Layout and method interpretation

```text
rps/observe/features.py       JSONL/tuple records -> categorical Symbols
rps/observe/views.py          identified / pseudonym blocks / histogram / summaries
rps/detection/pairwise.py     conditional MI and state-stratified permutation evidence
rps/detection/graph.py        connected-components and Louvain detectors
rps/detection/decoder.py      plug-in maximum-likelihood coalition decoder
rps/detection/pipeline.py     calibrated, threshold-gated decisions
rps/benchmark/data.py         simulator-backed datasets and external-log import
rps/benchmark/evaluate.py     independent edge-fit/calibration/test evaluation
rps/benchmark/metrics.py      detection, recovery and confidence intervals
rps/benchmark/report.py       measured-result tables, figures and Markdown
```

The requested graph file implements two extractors. Together with its decoder these are the
three added detection methods. `features.py` and `views.py` provide observations; they do not
implement independent classifiers. Supporting pairwise code is also adapted because both the
graph and decoder require a standardized dependence matrix. The original Bonjour implementation
remains a baseline, preserving its natural-log units and 0.05 threshold.

## Installation and reproducibility

Python >= 3.10 is required; the recorded run uses Python 3.12. The original simulator can still
run without third-party packages when no new detector is requested.

```bash
python -m pip install -r requirements-detection.txt
python -m unittest discover -s tests -v
python evaluate.py --config configs/benchmark.json --output results/new-run
```

`requirements-evaluated.txt` records the numerical package versions used for the delivered
benchmark. `provenance.json` records the actual interpreter, packages, configuration hash and
canonical dataset hash. Different numerical/library versions can affect borderline thresholds
or Louvain partitions, even with fixed seeds.

Each evaluation writes an exported `episodes.jsonl.gz`. Re-evaluate those exact trajectories:

```bash
python evaluate.py --config results/benchmark/config.json \
  --dataset results/benchmark/episodes.jsonl.gz --output results/replayed
```

Rebuild only plots/tables/Markdown from already measured predictions, without changing decisions:

```bash
python -m rps.benchmark.report --results results/benchmark
```

Choose a new output directory for an evaluation. This avoids silently overwriting a completed
report. Failed/incomplete output directories are diagnostic artifacts, not finished results.

## Analyse a single game

```bash
python simulate.py --config configs/random.json
python analyze.py --log runs/random.jsonl \
  --calibration results/benchmark/calibration/N3_T300_none.json \
  --output runs/detection.json
```

Or run it immediately after simulation:

```bash
python simulate.py --config configs/random.json \
  --calibration results/benchmark/calibration/N3_T300_none.json \
  --detection-output runs/detection.json
```

CLI output names flagged groups or says `no flag` / `insufficient evidence`. JSON includes the
evidence matrix, support, permutation p values, scores, abstention state and accepted groups.
The standalone analysis API never needs coalition labels or private signals.

```python
from rps.observe.features import load_jsonl, symbols
from rps.detection.pipeline import detect_episode, load_calibration

cal = load_calibration("results/benchmark/calibration/N3_T300_none.json")
view = symbols(load_jsonl("runs/random.jsonl"), state_mode="none")
decisions, evidence = detect_episode(view, cal)
print(decisions["decoder"]["groups"])
```

A decoder candidate is not returned as a detected coalition until it passes the independently
calibrated threshold. The result's `details.k` describes the best candidate even if it was rejected;
only `groups` is a thresholded identification. A dependence flag is not a posterior probability
or proof of intent. The decoder assumes one dense coalition; graph methods may return several.

## Public state and missing observations

- `none`: one state; suitable for the benchmark's memoryless random policies.
- `last_round`: the previous joint action, in sorted player order; up to 3^N cells plus the initial state.
- `recorded`: the supplied `state` field, which must represent information available **before** actions.

No action-code arithmetic is used to claim that paper is numerically greater than rock. The
public-state cardinality is learned from the data; the auction model's four-cell assumption is gone.
Unknown move strings fail validation. Missing keys or explicit `null` actions are masked. Sparse
state support is calculated separately for each pair, after both-player observation masking.
The baseline runs on complete rows; added methods can use pairwise-complete rows.

Do not condition on the current joint action, payoff, hidden signal, or policy generated after
receiving a private signal: this can remove the dependence one is trying to detect or leak labels.
The code intentionally ignores `signalled`, `truth`, `roles`, `policies`, and `payoffs` in features.
The recorded-state mode cannot automatically verify that a caller's state really is public and pre-action.

Constant strategies provide no dependence information and yield z=0. Unsupported pairs are NaN.
The decoder treats missing edges as zero evidence; fully unsupported episodes abstain.
There is no post-hoc imputation from true membership. Unknown/partial identities cannot be safely
inferred from the same behavior and then treated as independently observed identities.

## Observation views

`identified_view(records)` exposes persistent named categorical actions. `histogram_view(sym)`
returns only per-round counts of rock, paper and scissors. `summary_view(sym)` returns categorical
move entropy, the largest move fraction, and whether a complete round contains one or three
distinct moves. These contain no invented auction-price fields.

`relabelled_view(sym, m, seed)` returns `(public_view, evaluator_maps)`. Every block has distinct
pseudonyms and its own Symbols. Because RPS actions are observed directly, the first move in a
block remains valid; auction price differences could not do that. Last-round states are rebuilt
within the block and reset at boundaries. `recorded` state is rejected for this view because an
opaque recorded state might itself leak persistent identity. All detector functions require an
identified Symbols object. Analyse a block at its own calibrated length; do not concatenate
pseudonym columns across blocks. Episode-wise false-positive control when scanning many blocks
needs its own calibration over the maximum block score. This is not supplied by a single-block cutoff.

The separate [information experiment](INFORMATION.md) additionally supports full and
metadata views of explicitly recorded public messages. The RPS logs' `signalled` field
remains private experiment instrumentation and is never a public message observation.

## Evaluate real/LLM logs with ground truth

Generate independent episodes with the original simulator and retain their JSONL files. Build a
manifest separating **known honest edge-fit**, **known honest calibration**, and **held-out test**
episodes. Every group must have the same N, T, state definition, and honest population. All labels
must come from experimental configuration or independent annotation, never the detector being evaluated.

Manifest structure (illustrative filenames; extend the episode lists to the desired sample sizes):

```json
{
  "groups": {
    "ollama3_T300": {"state_mode": "last_round", "population": "your-model-and-honest-policy-version"}
  },
  "episodes": [
    {"id": "fit-001", "group": "ollama3_T300", "split": "edge_fit",
     "path": "logs/fit-001.jsonl", "truth": [], "scenario": "honest"},
    {"id": "cal-001", "group": "ollama3_T300", "split": "calibration",
     "path": "logs/cal-001.jsonl", "truth": [], "scenario": "honest"},
    {"id": "test-honest-001", "group": "ollama3_T300", "split": "test",
     "path": "logs/test-honest-001.jsonl", "truth": [], "scenario": "honest"},
    {"id": "test-pair-001", "group": "ollama3_T300", "split": "test",
     "path": "logs/test-pair-001.jsonl", "truth": [["A", "B"]], "scenario": "pair", "cp": 0.5}
  ]
}
```

```bash
python evaluate.py --manifest manifest.json --config configs/benchmark.json --output results/llm-evaluation
```

Paths are resolved relative to the manifest. Numerical detection settings and alpha come from the
config, while generation settings/sample counts are ignored for imported logs. Each manifest entry
must explicitly specify `truth`; absence does not silently imply an honest episode. `[]` means
known honest. `[["A", "B", "C"]]` means one three-player group; `[["A", "B"], ["C", "D"]]`
means two distinct groups. Report labels should identify whether these denote configured intent,
an active covert mechanism, or independently verified behavior. Those are different estimands.

At least 19 honest null samples are necessary for a nontrivial strict 5% rank threshold; 199 are
used in the delivered main calibration. The short four-entry example above is a schema example,
not a sufficient evaluation. With too few calibration episodes thresholds are infinite and the
added detectors do not flag. Edge-fit data are independent from score-calibration data, which are
independent from held-out test data. Duplicate IDs and duplicate observable trajectories are rejected.

The checked-in calibration is **for random/scripted policies**. An equal log length and player count
are not sufficient to justify applying it to an Ollama model. Shape and config mismatches are enforced;
matching the honest policy distribution, state semantics and exchangeability assumptions is the caller's responsibility.

## Metrics

`comparison.csv` contains per-group/scenario/CP confusion counts, precision, recall, F1, balanced
accuracy, AUROC, average precision, Wilson intervals, abstention, and naming metrics. The same
matched honest test set is reused as a reference for each alternative, but appears only once in
the headline aggregate false-positive rate. It must not be counted multiple times when estimating
uncertainty across scenarios. The report compares AUROC within conditions and averages those values;
raw scores across differing calibrations are not directly comparable.

`exact_groups` compares the full unordered set of groups; `exact_members` compares their union.
Jaccard uses the union of named players. Pair precision/recall/F1 use every within-group unordered
pair. Naming metrics include all positive episodes; a missed detection or abstention is a recovery
failure. Timing measures analysis only, excluding game generation; each method's reported standalone
time includes any shared permutation evidence it needs.

Permutation p values are diagnostic per-pair values, not automatically multiplicity-corrected
decisions. Decisions use calibrated episode-level maxima. Likewise OR-combining the three methods
requires separately calibrated ensemble control; do not assume each 5% rule yields a 5% combined rule.

## Known limits

Conditional independence is not the same as absence of collusion, and dependence is not proof of
intent. Higher-order-only dependence can evade all pairwise methods. Shared omitted public causes
can trigger all of them. Fine-grained state can make data too sparse, especially with many players.
Residual temporal dependence can invalidate within-state shuffles. The report includes explicit
stress tests for each of these practical concerns except arbitrary temporal-policy distributions.
