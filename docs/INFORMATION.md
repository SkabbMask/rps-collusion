# Information views

This experiment asks how much an auditor can learn about whether a configured
coordination mechanism is active. The label is binary: honest or coalition.
It does not represent the coalition's membership or intent. Information is
reported in bits, using equal prior probabilities for the two labels.

## Run

```bash
python information.py --output results/my-information
```

The default runs a small exact calculation and a 439-game scripted experiment
with three players and 100 rounds. For the full scripted or Qwen N=3 and N=6 experiment, see
[RUNNING.md](RUNNING.md).

```bash
# Exact calculation only: three players, three rounds.
python information.py --exact-only --exact-cp 0.5 \
  --exact-relabel-interval 1 --output results/my-exact

# Compare views of an existing dataset.
python information.py --dataset results/full-detection-growth/episodes.jsonl.gz \
  --relabel-interval 25 --output results/my-view-comparison

# Analyze separately labeled recorded games.
python information.py --manifest manifest.json --output results/my-llm-information
```

Use a fresh output directory. The manifest format is in
[DETECTION.md](DETECTION.md). Honest fit, honest calibration and test games
must be separate. Keep models, prompts and observation settings consistent.

## What each view contains

| View | Observation |
| --- | --- |
| `full` | Named actions, public state and public messages |
| `identified` | Named actions and public state |
| `relabelled` | Actions under fresh pseudonyms every m rounds |
| `histogram` | Per-round counts of rock, paper and scissors |
| `summaries` | Move entropy, largest move share, all-same and all-three flags |
| `metadata` | Identified actions/state plus message presence and character length |

RPS actions are categories, so there is no numeric mean price or auction
collusion index. Histogram and summary views omit identity-bearing state.
Relabelled views rebuild previous-round state inside each block; they reject
opaque recorded state that might reveal identities. Hidden identity maps are
kept out of detector features.

## Public messages

Enable `emit_messages` in a player or defaults configuration, or use:

```bash
python simulate.py --config configs/ollama-messages.json \
  --model qwen2.5:14b --url localhost:11434 --log runs/messages.jsonl
```

Each player emits a message alongside its action. Messages are limited to
512 characters, published after the round, and shown in later rounds. The
assistant receives no same-round public messages. Its private partner signal
and prompts are not part of the auditor's message view.

Logs record every player's message, for example:

```json
{"messages":{"A":"hello","B":"","C":""}}
```

An empty string means observed silence. Missing fields or null values mean
unobserved messages, including failed responses. Any unobserved message makes
`full` and `metadata` unavailable for the whole group. Historical logs must not
be assigned empty messages after the fact. Scripted players with an observed
channel remain silent, so their message views add no information.

## Exact calculation

The small model enumerates all action sequences, with `N*T <= 9`. Honest
players act independently and uniformly. Under the alternative, one fixed
primary/helper pair is drawn uniformly; the helper assists with probability
CP each round and otherwise acts uniformly. Messages are empty in both cases.
Fresh player permutations are drawn independently at block boundaries.

The report calculates:

- Total variation (TV), the difference between the two observation distributions.
- Best possible equal-prior classification error, `(1-TV)/2`.
- Mutual information `I(C;V)`, the Jensen–Shannon divergence in bits.
- The power upper bound `min(1, alpha+TV)` and optimal randomized test power.

These values are exact up to floating-point rounding for this model. They are
not bounds for arbitrary LLM behavior. With m=1, relabelled and histogram views
are equivalent. With empty messages, full and metadata equal identified.

## Bounds from recorded games

Each view is reduced to fixed features. An anomaly detector learns a centre
and scale from honest fit games, then gets a threshold from separate honest
calibration games. Test labels are used only to measure performance.

The features include action frequencies and pair dependence, count-vector
frequencies, summary moments, within-block dependence, and message length or
token counts as appropriate. These discard some information from the view.

For true detection and false-positive rates t and f, decision information is
`I(C;D)=h2((t+f)/2)-(h2(t)+h2(f))/2`, where h2 is binary entropy. The table
includes a sample estimate and separate confidence bounds. Exact binomial
intervals give `d=max(0,lower(t)-upper(f))`, which implies:

- `TV(P0^V,P1^V) >= d`.
- `I(C;V) >= 1-h2((1-d)/2)`.
- Equal-prior classification error `<= (1-d)/2`.

Bonferroni adjustment covers both interval tails and all planned comparisons.
The stated confidence requires independent games within each label and a
fixed detector selected before testing. Reused honest tests across comparisons
are allowed. Rounds inside one game are not independent test samples.

A zero lower bound does not prove zero information. Different detectors can
produce different lower bounds even when one view contains another. Do not
rank true view information using these lower bounds, or derive a power upper
bound from a TV lower bound. Compressed file sizes measure volume, not label
information.

## Files

`report.md` is the readable summary. `exact_views.json` contains the exact
calculation; `information.json` and `view_comparison.csv` contain measured
bounds. `view_calibration.json` stores fitted parameters and split IDs;
`view_predictions.json` stores test decisions. `config.json` and
`provenance.json` record settings and source fingerprints.
