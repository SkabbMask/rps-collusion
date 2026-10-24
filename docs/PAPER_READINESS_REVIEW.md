# Paper-readiness audit: RPS collusion

Audited 25 October 2026. Scope: current local source, `results/full-detection-growth`, `results/full-information`, exact-model code and existing tests. This review did not change detector code or rerun the main experiment. All 55 tests passed again.

## Verdict

The implementation and calibration design are coherent for a controlled scripted/synthetic benchmark. The available evidence supports a hackathon paper framed as an empirical testbed and a model-specific information analysis. It does not establish reliable detection of real LLM collusion, superiority of growth, or general information-theoretic impossibility under anonymity. Passing tests and reproducibility do not establish those claims.

## Measured results

The headline mixture has 2,800 positive and 400 matched honest test episodes. Across the complete benchmark there are 4,300 held-out episodes, with 1,953 additional honest fit/calibration episodes. The 21,500 method decisions are not independent experimental samples.

| Method | Recall | FPR | Exact group recovery |
| --- | ---: | ---: | ---: |
| bonjour | 59.29% | 2.50% | 51.32% |
| graph_components | 79.71% | 2.75% | 73.07% |
| graph_louvain | 79.71% | 2.75% | 73.29% |
| decoder | 81.71% | 4.00% | 56.29% |
| growth | 81.50% | 3.75% | 57.75% |

Growth minus ML: recall -0.214 percentage points; exploratory paired episode-bootstrap 95% interval approximately [-0.500, +0.036] points. Exact group recovery +1.464 points; corresponding interval approximately [+0.857, +2.107] points. These calculations used 5,000 bootstrap resamples, RNG seed 991, and the same 2,800 positive episodes. They condition on this benchmark and calibration, are unadjusted for multiple comparisons and are not a fresh confirmatory test.

## What is sound

- The public action/state adapters exclude private signal, policy and truth fields from detector features. Relabelled features do not stitch pseudonym columns using hidden maps.
- Graph edge fitting, honest score calibration and test evaluation use separate episode splits. Rank cutoffs are conservative under exchangeable honest episodes, conditional on a fixed scoring rule. This is marginal calibration, not a guarantee about the realized FPR of every fitted cutoff.
- Candidate groups are thresholded before being returned as detections; insufficient support yields explicit abstention.
- Exact enumeration averages over a fixed uniformly unknown primary/helper pair for the whole episode. It computes TV, balanced Bayes error, binary-label mutual information and randomized Neyman–Pearson power for that specified toy model.
- The decision-based information bound has the correct direction. With a fixed detector and independent held-out episodes within each arm, simultaneous binomial intervals give d=max(0,L_TPR-U_FPR), then TV>=d, I(C;V)>=1-h2((1-d)/2), and balanced Bayes error<=(1-d)/2. Bonferroni includes planned view/condition comparisons.
- Unit tests, upstream growth fixtures, seeded datasets and report artifacts provide useful implementation checks.

## Findings that affect paper claims

### 1. Growth changes membership inference; its detection decision is max-pair z

In `rps/detection/pipeline.py`, the growth score is `grown.seed_z`, exactly the largest supported pairwise z. Group expansion never changes that score. Consequently `growth_frac` can change group membership and recovery but cannot change flags, ROC or detection power at fixed evidence/calibration. Describe this as calibrated max-pair detection plus relative-average-z coalition reconstruction. Add a max-z baseline and a seed-pair-only membership ablation to isolate the contribution of expansion. This is a design fact, not a claim that the code is incorrectly implemented.

### 2. Information lower bounds do not rank true view information

The empirical report fits different anomaly detectors to lossy features of each view. A better bound for summaries than histograms can reflect a better detector/feature map, despite summaries being a coarsening of histograms. Neither the plug-in I(C;D) nor a confidence lower bound is the full I(C;V). A zero lower bound does not prove zero information. Do not convert empirical TV lower bounds into power upper bounds.

The exact N=3, T=3, CP=0.5 model gives I(C;identified)=0.047692 bits and I(C;relabelled_m1)=I(C;histogram)=0.026796 bits. These are exact binary-mechanism-label results for that model, not estimates for the 100/300-round empirical populations or arbitrary LLMs.

### 3. Message views have no informative messages in the current results

All scripted public messages are observed silence. Thus full, metadata and identified encode the same relevant observation in this experiment. Six software interfaces do not constitute six distinct empirical information conditions. No conclusion about semantic message content, metadata leakage or covert LLM communication is supported. The optional model-generated public channel has not been evaluated in these benchmark results.

### 4. Scope, baseline fairness and benchmark weighting matter

The original Bonjour implementation uses its fixed 0.05-nat cutoff, whereas added methods are independently calibrated. Compare the current implementations as configured; do not attribute every improvement solely to group extraction. A calibrated net-influence baseline at the same target FPR would strengthen a method comparison.
The original paper principally studies one colluding pair. The main mixture includes three-person stars and two disjoint pairs; those are useful extensions but structurally disadvantage single-pair or single-group outputs. Report per-scenario/CP results and single-pair comparisons, not only the aggregate. Precision/F1 depend on the constructed prevalence.
Growth has zero exact full-group recovery on every two-pair condition because it returns only one group. It cannot represent the correct two-group answer. Low-strength cases remain hard: at N=6, T=100, CP=0.1, growth recall is 7%.

### 5. Honest correlations and sparse state defeat broad deployment claims

All five methods flag 100/100 honest shared-state episodes when the public cause is omitted. This is an intentional out-of-distribution stress population, not a failure of the matched-exchangeability rank theorem. It demonstrates that dependence does not identify secret intent.
For N=6, T=300 with last-round joint state, the four added methods abstain on every tested honest and colluding episode. There are up to 3^6=729 previous-action states. A reported zero FPR there is not evidence of a useful detector.
Within-state shuffles require exchangeability appropriate to the temporal process; arbitrary adaptive LLM play need not satisfy it. Matching episode-level calibration can control final score rejection without making every permutation p-value valid. Model/prompt/history/channel changes require matching null data.

### 6. Code split separation does not undo repeated test-set reuse

The same default seeded dataset has been inspected while methods and feature maps evolved. Re-running the default seed is reproduction, not a new holdout. Even if fitting functions never read test labels, analyst choices informed by prior results make the evaluation exploratory. Freeze code, parameters, comparisons and hypotheses, then generate a fresh seed block; repeat honest calibration seeds to assess threshold variability.

### 7. Provenance does not identify uncommitted detector source

`rps/benchmark/evaluate.py` records git HEAD as `base_revision` but does not record the dirty diff or detector source hashes. Current results list the same base commit as the earlier four-method implementation while growth is in uncommitted files. Before submission, commit/archive the actual source and pin the run environment. The information command has source fingerprints, but the main detection provenance lacks equivalent working-tree capture.

### 8. Configured assistance does not establish economic harm

Exhaustive enumeration of the current N=3 payoff function gives honest expected payoffs (5/9,5/9,5/9). With CP=1 the primary/helper/honest expectations are (1,1/3,2/3). The primary gains a relative advantage while the honest player also gains in absolute payoff. This is assistant coordination in the paper’s non-zero-sum RPS environment; it does not demonstrate that the third player loses absolute utility. Claims about harm need an explicitly defined outcome and counterfactual.

## Minimum changes before a paper submission

1. Freeze and archive/commit the current source; report the precise synthetic mechanism, class label and balanced prior assumptions.
2. Run an untouched test seed block after freezing the analysis. Replicate independent honest calibration seeds rather than treating repeated replays as independent evidence.
3. Add calibrated max-pair and net-influence baselines; isolate seed-pair-only versus grown-group naming at identical detection flags. Compare single- and multiple-group scenarios separately.
4. Put shared-state false alarms, sparse-state abstention, weak-CP behavior and two-pair recovery failures in the main results.
5. Distinguish exact toy-model information from empirical detector-derived lower bounds. Do not rank view information using lower bounds alone.
6. For an AI/LLM empirical claim, collect independent repeated real-model honest/coalition episodes, informative observed messages and matching calibration; report model/version, prompts, seeds, failures and signal compliance. Otherwise label LLM support as an unevaluated extension.

## Suggested defensible conclusion

> In a controlled scripted multi-player RPS testbed, dependence-based detectors identify configured assistant coordination with performance sensitive to coalition structure, signal strength and public-state observability. Relative-average-z growth combines max-pair detection with single-coalition reconstruction. Exact small-model calculations show that identity removal can reduce information about the coordination mechanism, while held-out detector decisions provide information lower bounds for larger episodes. Shared public causes, sparse state and higher-order dependence limit interpretation and generalization.

## Relation to the original paper

[Bonjour et al., UAI 2022](https://proceedings.mlr.press/v180/bonjour22a.html) studies rule-based and learned collusion, limits its main scope to two colluding agents, and uses a 0.05 detection threshold in its experiments. This repository should be presented as an adaptation/extension of that testbed and baseline, not a complete reproduction of all its learning experiments. The RPS payoff rules and manually signalled assistance match the paper’s stated setting.
