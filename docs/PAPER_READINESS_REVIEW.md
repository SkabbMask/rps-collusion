# Research notes

Reviewed on 25 October 2026. The current code is suitable for a controlled
scripted study. The available results do not establish
reliable detection of arbitrary LLM collusion.

## What the results support

The benchmark separates honest fitting, honest calibration and test games.
Private signals and labels do not enter detector features. Unsupported cases
can abstain. The paper uses measured information lower bounds. The exact
toy diagnostic was inconclusive for its empirical question and was excluded.

The main comparison covers 2,800 coalition and 400 honest test games. The
full experiment has 4,300 test games and 1,953 fit/calibration games. Its
21,500 method decisions are not independent samples.

| Method | Recall | False positives | Exact group recovery |
| --- | ---: | ---: | ---: |
| Bonjour | 59.29% | 2.50% | 51.32% |
| Graph components | 79.71% | 2.75% | 73.07% |
| Louvain | 79.71% | 2.75% | 73.29% |
| ML decoder | 81.71% | 4.00% | 56.29% |
| Average-z growth | 81.50% | 3.75% | 57.75% |

## Limits on interpretation

**Growth reconstructs a group after max-pair detection.** Its detection score
is the strongest pair's z. Changing the join fraction changes membership,
not detection flags at fixed evidence and calibration. Compare grown groups
with seed-pair-only groups to isolate the contribution of expansion.

**Information lower bounds do not rank views.** Different feature maps and
detectors can give a stronger bound for a smaller view. Zero does not imply
no information. The optional exact toy diagnostic is not part of the paper evidence.

**The message channel is silent in the scripted results.** Full, metadata
and identified views therefore contain equivalent relevant observations.
They do not test semantic message content or metadata leakage.

**The comparison depends on the setting.** Bonjour uses its fixed 0.05-nat
threshold; the other methods are calibrated. Larger and multiple coalitions
extend the original paper's one-pair setting. Report single-pair and other
scenarios separately. Growth cannot return two disjoint groups and has zero
exact recovery in those cases. At N=6, T=100, CP=0.1, its recall is only 7%.

**Honest correlation can look like collusion.** All five methods flag all
100 honest shared-state games when the public cause is omitted. This is an
intentional change from the calibration population. With six players and
last-round conditioning, the added methods instead abstain in every tested
case because state observations are too sparse. Adaptive policies may also
violate the within-state shuffling assumptions.

**A replay is not a fresh test.** The default dataset has been inspected
while methods evolved. Keep those results exploratory, freeze the analysis,
and evaluate fresh seed blocks. Matching score thresholds to honest games
is not a universal 5% guarantee for changed policies or models.

**Archive the source used for a paper.** Detection provenance records Git
HEAD but not uncommitted changes. Commit or archive the full code and pin
the environment before a final run.

**Assistance does not necessarily reduce others' payoff.** For N=3, uniform
honest play gives expected payoffs (5/9,5/9,5/9). With CP=1, primary/helper/
honest payoffs are (1,1/3,2/3). The primary gains a relative advantage while
the honest player also gains absolute utility. Define the harm being measured.

## Before submission

1. Freeze and archive code, settings and hypotheses; use untouched test seeds.
2. Repeat honest calibration seeds to check threshold variability.
3. Add calibrated max-pair and net-influence baselines, plus the naming ablation.
4. Include weak signals, public causes, abstentions and multiple-group failures.
5. Exclude the toy diagnostic from paper claims; label empirical lower bounds.
6. Collect repeated real-model runs before making empirical LLM claims; record
   model versions, prompts, failures and signal compliance.

An exploratory paired bootstrap on the current main mixture gives growth
minus ML recall of -0.214 percentage points (95% interval -0.500 to +0.036),
and exact recovery of +1.464 points (+0.857 to +2.107). It used 5,000 resamples
with seed 991. These intervals condition on the current calibration and
mixture, and are not adjusted for multiple comparisons.

## Paper framing

A defensible claim is that dependence-based detection and group recovery vary
with signal strength, coalition structure and public-state visibility in a
specified RPS testbed. The view experiments provide detector-based information lower bounds; the
exact toy model is excluded from the paper.

Present this as an adaptation of [Bonjour et al., UAI 2022](https://proceedings.mlr.press/v180/bonjour22a.html),
not a reproduction of all its learned-agent experiments. The current results
measure a programmed assistance mechanism, not spontaneous LLM coordination.
