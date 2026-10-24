# Detection methods

For commands, see [RUNNING.md](RUNNING.md). `analyze.py` applies the five
methods below to a saved game. `evaluate.py` calibrates them on honest games
and measures performance on separate test games.

| Method | How it selects a group |
| --- | --- |
| Bonjour | Pairs whose mutual dependence exceeds that of the other players |
| Graph components | Connected groups of unusually dependent pairs |
| Graph Louvain | Communities within the same dependence graph |
| ML decoder | The group with the best estimated likelihood score |
| Average-z growth | Start with the strongest pair and add players with enough average dependence to the group |

The graph methods can return several groups. ML and growth return one.
All use pairwise action dependence, so purely higher-order coordination can
be missed. Dependence can also come from an omitted public cause.

## Scores and thresholds

Pair evidence is conditional mutual information, `I(A_i; A_j | S)`, in nats.
One player's moves are shuffled within each state to estimate the honest
reference mean and standard deviation. The z-score measures the observed
value relative to that reference. By default, each pair needs 20 observations
and each retained state needs 10. Missing pairs remain undefined.

Calibration uses three separate datasets:

1. `edge_fit`: honest games used to set the graph's pair threshold.
2. `calibration`: honest games used to set each method's final score threshold.
3. `test`: honest and coalition games used only to measure results.

Thresholds use rank `ceil((n+1)*(1-alpha))` with strict exceedance. Fewer than
19 calibration games cannot produce a nontrivial 5% cutoff. The 5% target
holds marginally when future honest games and calibration games are
exchangeable; it is not a promise for every fitted threshold or new population.
Combining several methods with an OR rule needs its own calibration.

Bonjour retains the original 0.05-nat rule rather than using this calibration.
The saved permutation p-values describe individual pairs; final decisions use
episode-level thresholds that account for the search across groups.

## Average-z growth

Growth starts with the strongest observed pair. The next player is the one
with the highest mean z to the group. It joins if that mean is at least
`growth_frac` times the group's current internal mean. Recompute after each
addition. The default fraction is 0.25; fix it before evaluating test data.

Missing links contribute zero to these means. An entirely unobserved episode
abstains. The detection score is the seed pair's z, so growth changes group
membership, not the underlying max-pair detection decision. JSON `details`
records additions and the first rejected player; only `groups` lists detections.

Current calibration files use schema 2 and contain all five methods.
Recalibrate after changing detector settings. Four-method files in
`results/benchmark/calibration` are historical and cannot be used directly.

## States and observations

- `none`: ignore state; appropriate for the memoryless scripted players.
- `last_round`: condition on the previous joint action. This can become sparse:
  six players have up to 729 previous-action states.
- `recorded`: use the log's supplied state. It must be public and known before
  the current actions; this is the caller's responsibility.

Do not use current payoffs, private signals or sampled policies as state.
Action features ignore `signalled`, `truth`, `roles`, `policies` and `payoffs`.
Missing or null actions are masked; unknown move strings are rejected.
Bonjour uses complete rounds; the other methods use pairwise-complete rounds.

Named, relabelled, histogram and summary observations are in `rps/observe/`.
Do not join pseudonym columns across blocks as if they were the same player.
The [information experiment](INFORMATION.md) also supports public-message views.
The five coalition detectors themselves require identified actions, or a
single relabelled block with matching calibration.

## Evaluate recorded games

A manifest supplies independent labels and paths to JSONL logs. Paths are
relative to the manifest. This example shows the format; it is too small for
useful calibration or confidence bounds.

```json
{
  "groups": {
    "qwen3_T200": {
      "state_mode": "none",
      "population": "qwen2.5:14b, fixed prompts and message settings"
    }
  },
  "episodes": [
    {"id":"fit-001","group":"qwen3_T200","split":"edge_fit","path":"logs/fit-001.jsonl","truth":[],"scenario":"honest"},
    {"id":"cal-001","group":"qwen3_T200","split":"calibration","path":"logs/cal-001.jsonl","truth":[],"scenario":"honest"},
    {"id":"test-honest-001","group":"qwen3_T200","split":"test","path":"logs/test-honest-001.jsonl","truth":[],"scenario":"honest"},
    {"id":"test-pair-001","group":"qwen3_T200","split":"test","path":"logs/test-pair-001.jsonl","truth":[["A","B"]],"scenario":"pair","cp":0.5}
  ]
}
```

Use `truth: []` only for known honest games. `[["A","B"],["C","D"]]` denotes
two separate pairs. Labels must come from the experiment or independent
annotation, not from the detector. State whether they mean an instructed
agreement, an active signal mechanism, or observed compliance.

```bash
python evaluate.py --manifest manifest.json --config configs/benchmark.json \
  --output results/my-llm-evaluation
```

Keep player count, rounds, models, prompts and honest policy distribution
consistent within a group. Matching shape alone does not make the scripted
calibration valid for LLM logs. Duplicate IDs and observed trajectories are
rejected. Shuffling assumptions can fail for adaptive, time-dependent policies.

## Outputs and code

`predictions.csv` contains test decisions. `comparison.csv` groups results by
scenario and signal probability. `summary.csv` summarizes the main mixture.
`exact_groups` requires the full set of groups to match; `exact_members`
compares their union. Recovery metrics count missed detections as failures.
Precision and F1 depend on the mixture of honest and coalition games.

Code is split between `rps/observe/` (log conversion), `rps/detection/`
(scores and decisions), and `rps/benchmark/` (datasets, evaluation and reports).
Source revisions are listed in [ATTRIBUTION.md](../ATTRIBUTION.md).

For automatic model data collection, see the batch commands in
[RUNNING.md](RUNNING.md#full-qwen-experiment). They record per-game seeds, model
digests, log hashes, signal compliance and parsing failures.

Identical observations are rejected by default as a leakage check. Independent
model games can legitimately repeat, so the collector uses
`--allow-repeated-observations` for analysis and records the repetition count.
This does not permit duplicate IDs or justify analyzing copied logs as fresh data.
