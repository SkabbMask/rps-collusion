# RPS collusion

Run repeated rock–paper–scissors games with random players or local Ollama
models, then test whether their actions show coordination.

## Setup

Requires Python 3.10 or newer. From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-detection.txt
python -m unittest discover -s tests -v
```

## Run Qwen locally

Install [Ollama](https://docs.ollama.com/quickstart) and start it. If needed,
run `ollama serve` in a separate terminal. Then download the model:

```bash
ollama pull qwen2.5:14b
```

The original command still works:

```bash
python simulate.py --rounds 200 --cp 0.5 --model qwen2.5:14b -v
```

With the default `config.json`, only B uses the model; A and C are random.
To run **all three players** on Qwen:

```bash
python simulate.py --config configs/ollama.json \
  --model qwen2.5:14b --url localhost:11434 \
  --rounds 200 --cp 0.5 --log runs/qwen.jsonl -v
```

Use `--model` to select another model installed in Ollama. The full
[run guide](docs/RUNNING.md) covers local and remote servers, public messages,
single-game analysis, and the complete experiment.

## Run a full Qwen experiment

`simulate.py` runs one game. `experiment.py` collects a batch of Qwen games,
then produces both detection and information reports:

```bash
python experiment.py --config configs/qwen-experiment.json \
  --model qwen2.5:14b --url localhost:11434 \
  --output results/qwen-full --dry-run

python experiment.py --config configs/qwen-experiment.json \
  --model qwen2.5:14b --url localhost:11434 \
  --output results/qwen-full
```

The full batch is large: **4,316 games and 4,244,400 model turns**, plus retries.
Use `configs/qwen-smoke.json` for a 54-turn connection/pipeline check, or
`configs/qwen-pilot.json` for a smaller 119-game study. The smoke check is too
small for statistical conclusions.

Rerun the same command to resume completed games. The final overview is
`results/qwen-full/report.md`. See [the batch run guide](docs/RUNNING.md#full-qwen-experiment)
for collection-only, offline analysis and configuration details.

## Run the full scripted experiment

No model server is needed. The included configuration tests 3 and 6 players,
100 and 300 rounds, several coalition structures, and additional stress cases.

```bash
python evaluate.py --config configs/full-information.json \
  --output results/my-run/detection

python information.py --config configs/full-information.json \
  --dataset results/my-run/detection/episodes.jsonl.gz \
  --relabel-interval 25 --output results/my-run/information
```

Open `results/my-run/detection/report.md` and
`results/my-run/information/report.md`. Use a new output directory each time.

## What's included

- Five methods: Bonjour, graph components, Louvain, ML decoder, average-z growth.
- Six observations: full, identified, relabelled, histogram, summaries, metadata.
- Reports of detection, group recovery, false alarms and information bounds.

A is the primary player; B receives A's move with probability `--cp` and tries
to help A score. C plays independently. Add players in the configuration to
try larger games. Everyone gets 0 if all moves match, everyone gets 1 if all
three moves appear, and otherwise players holding the winning move get 1.

The scripted benchmark tests a specified coordination rule. It does not measure
LLM performance. Its public messages are empty, so the message views add no
information. Dependence alone is not proof of intent.

## Documentation

- [Running experiments](docs/RUNNING.md)
- [Detection methods, calibration and log format](docs/DETECTION.md)
- [Information views and bounds](docs/INFORMATION.md)
- [Validation results](docs/NEW_DECODER_VALIDATION.md)
- [Research limitations](docs/PAPER_READINESS_REVIEW.md)
- [Sources and attribution](ATTRIBUTION.md)

Saved results include a [five-method benchmark](results/full-detection-growth/report.md),
[view comparison](results/full-information/report.md), and the older
[four-method benchmark](results/benchmark/report.md). Four-method calibration
files must be regenerated before using the current detection code.
