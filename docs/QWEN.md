# Qwen batch experiments

Install dependencies and start Ollama as shown in [RUNNING.md](RUNNING.md).

The single-game JSON files above are not batch experiments. Use `experiment.py`
to collect independent Qwen games and run both analyses automatically. Every
player uses the selected model, including the honest players.

First inspect the plan without contacting Ollama:

```bash
python experiment.py --config configs/qwen-experiment.json \
  --model qwen2.5:14b --url localhost:11434 \
  --output results/qwen-full --dry-run
```

Then start the run:

```bash
python experiment.py --config configs/qwen-experiment.json \
  --model qwen2.5:14b --url localhost:11434 \
  --output results/qwen-full
```

The full configuration collects 4,316 games across N=3 and N=6, T=100 and
T=300: one pair, a three-player coalition, and two pairs where applicable.
It needs **4,244,400 model turns**, before retries. The artificial parity and
public-state stress models remain part of the separate scripted benchmark.

Two smaller configurations are available:

| Configuration | Games | Model turns | Purpose |
| --- | ---: | ---: | --- |
| `configs/qwen-smoke.json` | 6 | 54 | Check the server, collection and report pipeline |
| `configs/qwen-pilot.json` | 119 | 35,700 | Smaller N=3, T=100 study |
| `configs/qwen-experiment.json` | 4,316 | 4,244,400 | Full Qwen grid |

For example, check the pipeline first:

```bash
python experiment.py --config configs/qwen-smoke.json \
  --url localhost:11434 --output results/qwen-smoke
```

The smoke configuration has only two honest calibration games, so the added
detectors receive infinite acceptance thresholds. Use it as a connection and
pipeline check, not a performance study. Copy a configuration and edit sample sizes, models or rounds
before starting a larger run; `--dry-run` shows the resulting size.

Progress shows elapsed time and estimated collection time remaining every 30
seconds and after each game. The estimate starts after the first completed
round and uses model turns per second measured in the current session,
including retries. On resume it estimates the remaining games from fresh
measurements. Speed can change with player count, prompts and model load;
the detection and information analyses run afterward and are not included.
A running process needs to be restarted to display the new progress output.

### Output and resume

The run writes:

```text
results/qwen-full/report.md               # overview and model compliance
results/qwen-full/detection/report.md     # five detectors
results/qwen-full/information/report.md   # six information views
results/qwen-full/manifest.json           # logs, labels, splits and seeds
results/qwen-full/quality.json            # parsing failures and compliance
results/qwen-full/logs/                   # game logs and metadata
```

Run the same command again to resume. Saved games are checked by hash and
skipped; an interrupted game restarts from its first round, with its old
partial log kept in `attempts/`. Settings, collection source and model weights
must match the saved run. Do not update the model during collection.

To collect at most five new games and then stop:

```bash
python experiment.py --config configs/qwen-experiment.json \
  --url localhost:11434 --output results/qwen-full --limit 5
```

To collect without analysis, add `--collect-only`. Once all games exist,
analysis can run without a model server:

```bash
python experiment.py --output results/qwen-full --analyze-only
```

### Labels and failed responses

Honest fit/calibration/test games contain only honest-role prompts. Coalition
games add assistant prompts and private signals. Labels mean an instructed
agreement with CP>0; they do not guarantee the model complied. Role assignments
and seeds are set before responses are seen. Parsing failures are retained as
played random fallback moves and counted, not resampled until successful.
Missing messages can make full/metadata views unavailable for that group.

Separately collected games may legitimately have identical actions, especially
if the model always chooses one move. The batch analysis allows these repeated
observations and records their count. The lower-level commands still reject
repeated observations by default; use `--allow-repeated-observations` only when
independent collection is established. Duplicate IDs and reused cross-split
seeds remain errors. Repeated output is not evidence of independent sampling
by itself.

