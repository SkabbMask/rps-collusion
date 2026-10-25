# Running experiments

Run these commands from the repository root. Setup is in [README.md](../README.md).
No Ollama server is needed for scripted experiments or analysis of saved logs.

## Full scripted experiment

```bash
python evaluate.py --config configs/full-information.json \
  --output results/replication/detection

python information.py --config configs/full-information.json \
  --dataset results/replication/detection/episodes.jsonl.gz \
  --relabel-interval 25 --output results/replication/information
```

This produces 6,253 episodes, including 4,300 tests. The main comparison uses
2,800 coalition and 400 honest tests across N=3/6 and T=100/300; stress controls
are reported separately. All five detectors run. Scripted public messages are
recorded silence. The exact toy diagnostic is excluded from this workflow.

Outputs:

```text
results/replication/detection/report.md
results/replication/detection/summary.csv
results/replication/detection/comparison.csv
results/replication/detection/predictions.csv
results/replication/detection/calibration/
results/replication/detection/episodes.jsonl.gz
results/replication/information/report.md
results/replication/information/view_comparison.csv
```

For a smaller end-to-end check, substitute `configs/smoke.json` in both commands.
Use new output directories rather than overwriting saved experiments.
[RESULTS.md](RESULTS.md) identifies the reference outputs and comparison limits.

## Replay saved episodes

```bash
python evaluate.py --config results/full-detection-growth/config.json \
  --dataset results/full-detection-growth/episodes.jsonl.gz \
  --output results/replay/detection

python information.py --config results/full-detection-growth/config.json \
  --dataset results/full-detection-growth/episodes.jsonl.gz \
  --relabel-interval 25 --output results/replay/information
```

This uses existing games, refits thresholds and reruns analysis without
inference. It is a replay, not an independent replication.

## Simulate and analyse one game

First run the full benchmark above to produce matching calibration files.
Then:

```bash
python simulate.py --config configs/random.json --log runs/game.jsonl

python analyze.py --log runs/game.jsonl \
  --calibration results/replication/detection/calibration/N3_T300_none.json \
  --output runs/detection.json
```

Calibration must match player count, rounds, state, detector settings and
honest population. Do not apply scripted calibration to model games.
Historical four-method calibration files are incompatible with the current
five-method schema.

## Model experiments

See [QWEN.md](QWEN.md) for Ollama batches, model choices, timing and resume.
See [MODAL.md](MODAL.md) for the optional cloud launcher.

To run one three-model game:

```bash
python simulate.py --config configs/ollama.json \
  --model qwen2.5:14b --url localhost:11434 \
  --rounds 200 --cp 0.5 --log runs/qwen-game.jsonl -v
```

Use `configs/ollama-messages.json` to record public messages. The original
`python simulate.py --rounds 200 --cp 0.5 --model qwen2.5:14b -v` also works,
but its default `config.json` leaves A and C scripted and only B model-backed.

For a server on another machine, pass its address with `--url`. To save local
settings, copy `.env.example` to `.env`; `OLLAMA_URL` sets the default endpoint,
and `PLAYER_A_URL` etc. select individual servers. An explicit `--url` overrides
those endpoints. Keep `.env` out of Git.

## Imported model logs

A manifest supplies independent labels and fit/calibration/test splits.
The schema and requirements are in [DETECTION.md](DETECTION.md).

```bash
python evaluate.py --manifest manifest.json --config configs/full-information.json \
  --output results/imported/detection
python information.py --manifest manifest.json --config configs/full-information.json \
  --relabel-interval 25 --output results/imported/information
```

Models, prompts, horizons and observation settings must match within each
calibration group. Independently collected identical trajectories may require
`--allow-repeated-observations`; copied logs are not independent games.

## Figures

Analysis exports PNG, SVG and PDF figures automatically. To rebuild them from
saved results without rerunning detection or inference:

```bash
python -m rps.figures --results results/full-detection-growth
python -m rps.figures --results results/full-information
python -m rps.figures --results results/qwen-hour
```

See [FIGURES.md](FIGURES.md) for captions and output paths.
