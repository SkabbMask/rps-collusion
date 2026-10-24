# Information bounds across RPS auditor views

This extension adapts COLAV's six-view information experiment to categorical
RPS. The label C is **binary episode-level presence of the configured coalition
mechanism**, under balanced honest/coalition priors. It is not coalition
membership, intent, or LLM compliance. All information units are bits.

## Run

From the repository root, with the detection requirements installed:

```bash
python -m unittest discover -s tests -v
python information.py --output results/my-information
```

The default `configs/information.json` generates 439 scripted episodes: 40
honest fit, 99 honest calibration, 100 honest test, and 100 tests at each of
CP=0.5 and CP=1.0. N=3, T=100. Public channels are explicitly observed and
scripted players remain silent. These are not LLM message results.

```bash
# Exact model only; compare m=1 with --exact-relabel-interval 3.
python information.py --exact-only --exact-cp 0.5 \
  --exact-rounds 3 --exact-relabel-interval 1 --output results/exact-m1

# Replay a dataset. Legacy logs lack observed public messages, so the report
# marks full/metadata unavailable rather than treating missing data as silence.
python information.py --dataset results/benchmark/episodes.jsonl.gz \
  --relabel-interval 25 --output results/information-replay

# Same independently labeled manifest schema as evaluate.py.
python information.py --manifest manifest.json \
  --relabel-interval 25 --output results/information-llm
```

Choose a fresh output directory. See [DETECTION.md](DETECTION.md) for the
manifest schema. Use disjoint IDs, seeds and observable trajectories across
fit/calibration/test. Only independently known honest episodes may fit or
calibrate. Match N, T, public state, models, prompts, messaging protocol and
honest policy population within each group. At least 19 calibration episodes
are necessary for a nontrivial 5% rank cutoff; larger test sets are needed
for useful confidence bounds. One log does not support this experiment.

## Six observations

| View | RPS adaptation |
| --- | --- |
| `full` | Persistent named actions + pre-action public state + public message text |
| `identified` | Persistent named categorical actions + pre-action public state |
| `relabelled` | Fresh pseudonyms every m rounds; linkage only inside blocks |
| `histogram` | Per-round counts of rock, paper and scissors, without player names |
| `summaries` | Move entropy, largest move fraction, all-same and all-three indicators |
| `metadata` | Identified actions/state + message presence and character length, excluding content |

There is no numeric mean of rock/paper/scissors or auction collusion index.
Histograms and summaries omit identity-bearing state. Relabelled last-round
states are rebuilt within blocks; opaque recorded state is rejected for this
view. Hidden identity maps never enter detector features. Anonymous views
cannot name a coalition. Constructors `full_view` and `metadata_view` are in
`rps.observe.views`; existing constructors remain compatible.

## Public message channel

Enable `"emit_messages": true` in player/default settings. An example with
all three players backed by Ollama is included:

```bash
python simulate.py --config configs/ollama-messages.json \
  --url localhost:11434 --log runs/message-episode-001.jsonl
```

Each model emits a `message` string alongside its move/probabilities. Messages
are published after the round, visible in later public histories, and limited
to 512 characters. No same-round public message reaches the assistant. Private
partner signals and system/user prompts are never exposed as messages. This
channel changes the policy population; collect matching honest episodes.

Example additional JSONL field:

```json
{"messages":{"A":"hello","B":"","C":""}}
```

Empty string means observed silence. Missing keys, absent instrumentation or
`null` mean unobserved, including failed responses. Any unobserved message
excludes full/metadata for the **entire group**, avoiding class-dependent
episode selection. Random players with the channel enabled publish silence.
Do not retrofit empty messages into historical logs.

Private `signalled`, `policies`, prompts, roles, truth and payoffs never enter
view features. Truth is used for test labels and calibration validation only.
Define the target mechanism carefully: CP=0 with an assistant prompt disables
the signal, but does not establish absence of a secret agreement in the prompt.

## Exact model limits

Enumeration supports N*T<=9. H0 has iid uniform actions. H1 draws one ordered
primary/helper pair uniformly once per episode. The helper assists with CP
independently per round and otherwise plays uniformly; everyone else remains
uniform. Messages are observed silence under both hypotheses. Relabelling is
an independent uniform permutation at every block boundary. Column-history
orbits are sufficient for this channel, avoiding redundant permutations.

From the exact view PMFs the report computes TV, equal-prior Bayes error
`(1-TV)/2`, `I(C;V)` as Jensen–Shannon divergence, KL(P1||P0), the power upper
bound `min(1,alpha+TV)`, and optimal randomized Neyman–Pearson power at alpha.
These are exact up to floating-point roundoff **for that model**. They do not
bound arbitrary LLM populations or coalition membership. With observed silence,
full=metadata=identified. At m=1, relabelled=histogram.

## Empirical confidence bounds

Fixed, label-free features feed an anomaly detector fitted only on honest
`edge_fit` episodes; a strict rank threshold is frozen from separate honest
`calibration` episodes. Test labels never tune features or thresholds.
Insufficient calibration yields infinite thresholds and no detections.

Identified features use marginal/pair frequencies and conditional MI. Full
adds metadata and stable token hashes; metadata adds presence/length moments.
Histograms use count-vector frequencies, summaries use moments/missingness,
and relabelled features aggregate sorted within-block pair information and
anonymous counts. Pseudonym columns are never stitched across blocks. These
are lossy feature maps, not sufficient statistics of the views.

For held-out TPR t and FPR f, the plug-in decision-information estimate is
`I(C;D)=h2((t+f)/2)-(h2(t)+h2(f))/2`. It is not full-view MI or a confidence
bound. Simultaneous Clopper–Pearson bounds give
`d=max(0,lower(TPR)-upper(FPR))`. With family confidence `1-delta`:

- `TV(P0^V,P1^V) >= d`.
- `I(C;V) >= 1-h2((1-d)/2)`.
- Equal-prior `Bayes_error(V) <= (1-d)/2`.

Bonferroni covers both tails and every view/condition comparison. Reused honest
tests need not be independent across comparisons for this union bound, but
episodes within each arm must be independent. Rounds within an episode are
not independent replicates. Matching honest populations and exchangeability
are needed for the detector's marginal target-FPR interpretation.

Zero lower bounds do not establish zero information. Richer views can have
weaker measured lower bounds because their detectors are imperfect. No power
upper bound is inferred from a TV **lower** bound. Gzip/raw sizes measure
volume, not label information. NN/KDE asymptotic estimates are not presented
as certified finite-sample bounds.

## Outputs and lineage

`report.md`, `exact_views.json`, `information.json`, `view_comparison.csv`,
`view_calibration.json`, `view_predictions.json`, `config.json`, `provenance.json`.
The JSON records the canonical dataset hash and feature version. This follows
COLAV's view hierarchy, exact-TV and conservative decision-information bounds;
the RPS enumeration, message channel and evaluation adapter are new.
See [ATTRIBUTION.md](../ATTRIBUTION.md) for source revisions and notices.
