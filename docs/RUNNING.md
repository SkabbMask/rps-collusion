# Complete experiment: five detectors and six information views

Run these commands from the repository root. This is the full scripted
experiment, not the smoke configuration. No Ollama server is required.
It evaluates 6,253 episodes (4,300 held-out tests) with **five** methods:
Bonjour, graph components, graph Louvain, ML decoder and average-z growth.

## 1. Install and test

```bash
cd /home/bitchae/aisafety/collusion/rps-collusion-updated/rps-collusion

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-detection.txt
python -m unittest discover -s tests -v
```

## 2. Simulate one game

```bash
python simulate.py \
  --config configs/random.json \
  --log runs/full-test-game-growth.jsonl
```

This prints the original Bonjour result. All five methods are applied in step
5 after the new calibration is generated. Old four-method calibration files,
including `results/benchmark/calibration/*.json`, cannot calibrate the new
growth method and are rejected with instructions to rerun the benchmark.

## 3. Prepare the full benchmark with public-message observations

This creates a separate configuration. `growth_frac=0.25` is the teammate's
analysis default and must be fixed before evaluation, not tuned on test results.

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("configs/benchmark.json").read_text())
config["public_messages"] = True
config.setdefault("detector", {})["growth_frac"] = 0.25
Path("configs/full-information.json").write_text(
    json.dumps(config, indent=2) + "\n"
)
PY
```

## 4. Run the full detection benchmark and generate fresh calibration

```bash
python evaluate.py \
  --config configs/full-information.json \
  --output results/full-detection-growth
```

This executes all five methods, calibrates the added growth detector on
independent honest episodes, and writes 21,500 held-out method decisions.
It also saves all trajectories for the information analysis below. The four
existing methods retain their previous algorithms and thresholds, subject to
the same dependency versions, dataset and configuration.

## 5. Analyze the single game with all five methods

```bash
python analyze.py \
  --log runs/full-test-game-growth.jsonl \
  --calibration results/full-detection-growth/calibration/N3_T300_none.json \
  --output runs/full-test-detection-growth.json
```

For subsequent games, integrated detection is also available:

```bash
python simulate.py \
  --config configs/random.json \
  --log runs/integrated-growth.jsonl \
  --calibration results/full-detection-growth/calibration/N3_T300_none.json \
  --detection-output runs/integrated-growth-detection.json
```

## 6. Evaluate information bounds on the same episodes

```bash
python information.py \
  --config configs/full-information.json \
  --dataset results/full-detection-growth/episodes.jsonl.gz \
  --relabel-interval 25 \
  --output results/full-information-growth
```

Main results:

```text
results/full-detection-growth/report.md
results/full-information-growth/report.md
```

Use fresh output directory names for repeated runs, and change subsequent
commands to match those names. Existing completed reports are not overwritten.
The first stage uses a five-method **detection suite**; the information stage
compares six **observation views** with its own fixed feature detectors. Adding
growth does not change those information-view algorithms or CLI options.

These commands use scripted players. Their public messages are observed
silence, so full and metadata add no information beyond identified actions.
The opaque recorded-state stress group excludes relabelling because its state
cannot safely be anonymized. Exact small-model limits and empirical confidence
lower bounds are separate experiments; see [INFORMATION.md](INFORMATION.md).

## Re-evaluate an existing dataset with the new detector

You can reuse independently split existing trajectories; all five methods are
recalibrated and evaluated. Historical results need not be overwritten:

```bash
python evaluate.py \
  --config results/full-detection/config.json \
  --dataset results/full-detection/episodes.jsonl.gz \
  --output results/full-detection-growth-replay
```

If the saved config predates growth, `growth_frac` defaults to 0.25. The fresh
calibration records it explicitly. There is no need to rerun Ollama or regenerate
scripted episodes just to evaluate the new method. Replaying logs without public
messages still leaves full/metadata information views unavailable.

## Actual three-LLM game with messages

With Ollama running and `llama3.2:latest` available:

```bash
python simulate.py \
  --config configs/ollama-messages.json \
  --url localhost:11434 \
  --log runs/full-ollama-game.jsonl \
  -v
```

This runs three LLM players. The scripted calibration is not valid for this
population. A complete LLM experiment needs independent honest fit/calibration
and honest/coalition test games, with an explicitly labeled manifest:

```bash
python evaluate.py --manifest manifest.json --config configs/full-information.json \
  --output results/llm-detection-growth
python information.py --manifest manifest.json --config configs/full-information.json \
  --relabel-interval 25 --output results/llm-information-growth
```

The repository provides manifest import and analysis, not a single-command LLM
data-collection runner. See [DETECTION.md](DETECTION.md) for the manifest schema
and [INFORMATION.md](INFORMATION.md) for message observation requirements.

## What changed in new-decoder integration

The requested algorithm is `grow_from_top_pair` in upstream
`detection/decoder.py`, pinned at commit
`1bea281244ad2d756b9b507145b2046fefcf1582`. The local method key is `growth`.
It starts from the strongest supported pair, selects the next player with the
largest mean z to the existing group, and joins it when that mean is at least
`growth_frac` times the group's current internal pair mean. The cutoff is
recomputed after each join. Missing edges contribute zero to means; completely
unsupported episodes abstain. Ties use player order. The seed must be supported,
which avoids choosing an unobserved pair over observed negative evidence.

Membership growth has a relative cutoff. An independent honest calibration
still controls episode-level detection: the seed-pair z must strictly exceed
its frozen threshold before the group is reported in `groups`. `details` contains
the diagnostic candidate growth trace, not additional detected groups.
This adapts upstream's strongest-pair acceptance to the local independent
calibration convention rather than its within-episode surrogate calibration.

The separate upstream graph pruning/mean-score changes are not substituted for
the existing components/Louvain methods. This keeps the original four-method
comparison intact and isolates the requested average-z growth addition. Growth
returns one group, so multiple disjoint coalitions remain a limitation.
