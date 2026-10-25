# Qwen and other Ollama models

Every player in `experiment.py` uses the selected model. Each game starts a
new session; fitting, calibration and test games are separate.

## Start locally

Install [Ollama](https://docs.ollama.com/quickstart). If its server is not
already running, start `ollama serve` in another terminal.

```bash
ollama pull qwen2.5:14b
python experiment.py --config configs/qwen-hour.json \
  --model qwen2.5:14b --url localhost:11434 \
  --output results/qwen-check
```

This six-game check is useful for inspecting responses, compliance and output
files. Its two calibration games force infinite cutoffs at 5%, so it cannot
validate the calibrated detectors. The saved `results/qwen-hour` case has one
unassigned and one coalition test game.

## Choose a batch

| Configuration | N | Rounds | Games | Model turns |
| --- | --- | ---: | ---: | ---: |
| `qwen-smoke.json` | 3 | 3 | 6 | 54 |
| `qwen-hour.json` | 3 | 15 | 6 | 270 |
| `qwen-explore-t30.json` | 3 | 30 | 69 | 6,210 |
| `qwen-pilot-t30.json` | 3 | 30 | 119 | 10,710 |
| `qwen-comparison-t30.json` | 3 | 30 | 439 | 39,510 |
| `qwen-pilot.json` | 3 | 100 | 119 | 35,700 |
| `qwen-experiment.json` | 3, 6 | 100, 300 | 4,316 | 4,244,400 |

Paths are under `configs/`. Turns exclude parsing retries. The first two
batches are checks only. The 30-round batches can still have low detection
power. A finite threshold is not evidence of reliable performance.

Inspect a plan without contacting the server:

```bash
python experiment.py --config configs/qwen-pilot-t30.json \
  --model qwen2.5:14b --url localhost:11434 \
  --output results/qwen-pilot-t30 --dry-run
```

Remove `--dry-run` to collect games and run both analyses. Use a fresh output
name when changing the model or settings. A smaller model is a separate
population; do not reuse the 14B calibration:

```bash
ollama pull qwen2.5:1.5b
python experiment.py --config configs/qwen-pilot-t30.json \
  --model qwen2.5:1.5b --url localhost:11434 \
  --output results/qwen-1p5b-pilot-t30
```

## Timing

Progress prints elapsed time and a collection ETA every 30 seconds. It uses
completed rounds in the current session and includes time spent on retries;
analysis runs afterward. Early estimates can change as prompts and load change.

Time two games before choosing a large batch:

```bash
time python experiment.py --config configs/qwen-pilot-t30.json \
  --model qwen2.5:14b --url localhost:11434 \
  --output results/qwen-timing --limit 2 --collect-only
```

These two games contain 180 model turns. Divide that count by elapsed seconds
to estimate aggregate throughput. They are honest games; coalition games may
run at a different speed. At 0.1 turns/s, the 10,710-turn pilot needs about
29 hours 45 minutes before analysis. A one-hour limit does not make it finish:

```bash
timeout --signal=INT --kill-after=30s 1h python experiment.py \
  --config configs/qwen-hour.json --model qwen2.5:14b \
  --url localhost:11434 --output results/qwen-check
```

This Linux command interrupts after an hour and forces termination 30 seconds
later if needed. It can leave an incomplete collection or report. Avoid sharing
the same Ollama server with another experiment when measuring speed.

## Resume and analyse

Repeat the original command to skip completed games. Source, configuration and
model identity must match. An interrupted game restarts at round one; the old
attempt is retained. Do not run two writers against the same output directory.

For a completed collection, analysis requires no model server:

```bash
python experiment.py --output results/qwen-check --analyze-only
```

`--collect-only` skips analysis; `--limit N` collects at most N new games.
The overview is `report.md`, with detection and information reports in their
own subdirectories. Logs, labels, seeds, model identity, hashes and compliance
counts are saved alongside them. Exact toy calculations are no longer included
in normal analysis; old reports may still contain them.

## Response handling and labels

Labels mean an instructed agreement, not verified compliance or harm.
Parsing failures retain a random fallback action and are counted. Transport
failures stop collection. Public messages are observed only when recorded;
missing messages are not replaced with silence.

The collector allows genuinely independent model games with identical observed
trajectories and records repetition counts. It still rejects duplicate IDs and
cross-split seed reuse. See [INFORMATION.md](INFORMATION.md) for view availability
and [MODAL.md](MODAL.md) for optional GPU collection.
