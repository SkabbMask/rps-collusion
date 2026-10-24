# Running experiments

Commands below assume you are in the repository root. For this checkout:

```bash
cd /home/bitchae/aisafety/collusion/rps-collusion-updated/rps-collusion
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-detection.txt
python -m unittest discover -s tests -v
```

## Qwen or another local model

The model backend is [Ollama](https://docs.ollama.com/quickstart).
Start the server in a separate terminal if it is not already running:

```bash
ollama serve
```

Download [Qwen 2.5 14B](https://ollama.com/library/qwen2.5:14b) and check the
installed model names:

```bash
ollama pull qwen2.5:14b
ollama list
```

The original command uses `config.json`:

```bash
python simulate.py --rounds 200 --cp 0.5 --model qwen2.5:14b -v
```

In that configuration, A and C are random and **only B uses the model**.
The endpoint comes from `.env` or `OLLAMA_URL`, defaulting to localhost.
Add `--url localhost:11434` to force the local server.

For **three model-backed players**, use the included `configs/ollama.json`:

```bash
python simulate.py --config configs/ollama.json \
  --model qwen2.5:14b --url localhost:11434 \
  --rounds 200 --cp 0.5 --log runs/qwen.jsonl -v
```

For another installed model, replace `qwen2.5:14b` with its name from
`ollama list`. For example, if `llama3.2:latest` is installed:

```bash
python simulate.py --config configs/ollama.json \
  --model llama3.2:latest --url localhost:11434 \
  --rounds 200 --cp 0.5 --log runs/llama.jsonl -v
```

Use `--seed` to set the experiment seed and `--log` to keep separate runs.
Each run prints scores, signal compliance and the Bonjour detection result.
Model responses can vary even with a fixed seed.

### Public messages

To record public messages as well as actions:

```bash
python simulate.py --config configs/ollama-messages.json \
  --model qwen2.5:14b --url localhost:11434 \
  --rounds 200 --cp 0.5 --log runs/qwen-messages.jsonl -v
```

Messages are published after each round and shown in later rounds. This is
separate from the private signal sent to B. See [INFORMATION.md](INFORMATION.md)
for the `full` and `metadata` views.

### Another machine

The model must be installed on the machine running the Ollama server:

```bash
python simulate.py --config configs/ollama.json \
  --model qwen2.5:14b --url 192.168.1.50:11434 \
  --rounds 200 --cp 0.5 --log runs/remote-qwen.jsonl -v
```

To save endpoints, copy `.env.example` to `.env` and set `OLLAMA_URL`.
`PLAYER_A_URL`, `PLAYER_B_URL` and `PLAYER_C_URL` can select different servers.
A CLI `--url` overrides them all. Player-specific models go in the `players`
entries in your JSON configuration.

## Full Qwen experiment

Use `experiment.py` to collect honest and coalition games with Qwen, then run
all five detectors and the six-view information analysis. Every player uses
the selected model. Inspect the size first, then start the batch:

```bash
python experiment.py --config configs/qwen-experiment.json \
  --model qwen2.5:14b --url localhost:11434 \
  --output results/qwen-full --dry-run

python experiment.py --config configs/qwen-experiment.json \
  --model qwen2.5:14b --url localhost:11434 \
  --output results/qwen-full
```

The full N=3/N=6 grid has **4,316 games and 4,244,400 model turns**, plus retries.
It does not include the artificial stress generators from the scripted benchmark.
For a connection/pipeline check, use the 54-turn smoke configuration:

```bash
python experiment.py --config configs/qwen-smoke.json \
  --url localhost:11434 --output results/qwen-smoke
```

`configs/qwen-pilot.json` offers a smaller 119-game study. The smoke check has
too little calibration data for research conclusions. Edit a copy of a config
before collecting data; `--dry-run` recalculates the size.

Rerun the same command to resume. Read `results/qwen-full/report.md` for links
to both analyses and model-compliance counts. For collection-only runs, limits,
offline analysis and label details, see [QWEN.md](QWEN.md).

## Full scripted experiment

This uses random players and programmed assistants. No Ollama server is needed.
The configuration includes N=3 and N=6, T=100 and T=300, plus stress cases.
There are 6,253 episodes: 1,953 for fitting/calibration and 4,300 held-out tests.
All five detection methods run automatically.

### 1. Simulate one game

```bash
python simulate.py --config configs/random.json \
  --log runs/test-game.jsonl
```

### 2. Run the benchmark

`configs/full-information.json` already includes public-message observations
and `growth_frac=0.25`; no configuration-generation step is needed.

```bash
python evaluate.py --config configs/full-information.json \
  --output results/my-run/detection
```

This saves the report, figures, predictions, calibration files and dataset.
For a smaller check, use `configs/smoke.json` instead.

### 3. Analyze the game with all five methods

```bash
python analyze.py --log runs/test-game.jsonl \
  --calibration results/my-run/detection/calibration/N3_T300_none.json \
  --output runs/test-detection.json
```

Or combine simulation and analysis in one command after calibration exists:

```bash
python simulate.py --config configs/random.json \
  --log runs/next-game.jsonl \
  --calibration results/my-run/detection/calibration/N3_T300_none.json \
  --detection-output runs/next-detection.json
```

Calibration must match the number of players, rounds, state definition,
detector settings and honest player population. These files are for scripted
players; do not use them to claim calibrated LLM detection. Older four-method
files, including `results/benchmark/calibration`, cannot be used with the
current five-method code.

### 4. Compare information views

```bash
python information.py --config configs/full-information.json \
  --dataset results/my-run/detection/episodes.jsonl.gz \
  --relabel-interval 25 --output results/my-run/information
```

Read:

```text
results/my-run/detection/report.md
results/my-run/information/report.md
```

Use a fresh `results/my-run` name for each complete experiment. Scripted
players send empty public messages. The recorded-state stress case cannot
use relabelling because its state may expose identities.

## Reuse existing data

Recalibrate and rerun all five detectors on saved episodes:

```bash
python evaluate.py --config results/full-detection/config.json \
  --dataset results/full-detection/episodes.jsonl.gz \
  --output results/my-replay
```

This does not call Ollama or regenerate games. If the old configuration has
no `growth_frac`, it defaults to 0.25.

## Evaluate a collection of LLM games

One game lets you inspect behavior. A detection or information experiment
needs separate honest fit/calibration games and honest/coalition test games.
List their files and known labels in a manifest, using the schema in
[DETECTION.md](DETECTION.md):

```bash
python evaluate.py --manifest manifest.json --config configs/full-information.json \
  --output results/my-llm-run/detection
python information.py --manifest manifest.json --config configs/full-information.json \
  --relabel-interval 25 --output results/my-llm-run/information
```

Models, prompts, game length and message settings must match within each
calibration group. Use the batch runner above to collect compatible games, or supply independently
collected logs in this format.
