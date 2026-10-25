# RPS collusion detection

A testbed for detecting coordination in repeated rock–paper–scissors games.
It runs scripted players or Ollama models and compares five detection methods:
net influence, connected components, Louvain, likelihood search and average-z
growth.

## Install

Use Python 3.10 or newer. The experiment code is on the `dev` branch:

```bash
git clone --branch dev https://github.com/SkabbMask/rps-collusion.git
cd rps-collusion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-detection.txt
python -m unittest discover -s tests -v
```

For the numerical package versions recorded with the scripted benchmark, use
Python 3.14.8 and `requirements-reproduction.txt`. Saved runs record their own
versions in `provenance.json`.

## Reproduce the scripted benchmark

No model server is needed. The configuration covers 3 and 6 players, 100 and
300 rounds, several coalition structures and stress controls.

```bash
python evaluate.py --config configs/full-information.json \
  --output results/replication/detection

python information.py --config configs/full-information.json \
  --dataset results/replication/detection/episodes.jsonl.gz \
  --relabel-interval 25 --output results/replication/information
```

Read `results/replication/detection/report.md` and
`results/replication/information/report.md`. Both commands save tables and
figures. Use a fresh output name when repeating the experiment.

The saved five-method benchmark is in
[results/full-detection-growth](results/full-detection-growth/report.md).
The exact synthetic information diagnostic was inconclusive for the paper's
empirical question and was excluded. It is not run or plotted by default.

## Run Qwen

Install [Ollama](https://docs.ollama.com/quickstart), start its server, then:

```bash
ollama pull qwen2.5:14b

python experiment.py --config configs/qwen-hour.json \
  --model qwen2.5:14b --url localhost:11434 \
  --output results/qwen-check
```

This is a six-game behavior check, not a detector-performance study. The
completed `qwen-hour` example had only one held-out game per condition.
For larger batches, smaller models, timing and resume, see [QWEN.md](docs/QWEN.md).
The full model grid requires 4,244,400 model turns before retries; inspect it
with `--dry-run` before starting it.

## One scripted game

```bash
python simulate.py --config configs/random.json --log runs/game.jsonl
```

`random.json` specifies one game. It is not the full benchmark configuration.
For a single game with all players using a model, use `configs/ollama.json`.
The legacy default `config.json` is a hybrid: only B uses Ollama.

## Results and scope

Keep scripted experiments and model experiments separate. The scripted results
measure a programmed assistance rule. Model labels record an instructed
agreement; they do not establish spontaneous collusion or harmful intent.
The six observation views compare what an auditor can infer from actions,
identity linkage and public messages. A zero information lower bound does not
show that a view contains no information.

## Documentation

- [Run and replay experiments](docs/RUNNING.md)
- [Saved results and replication checks](docs/RESULTS.md)
- [Qwen and other Ollama models](docs/QWEN.md)
- [Modal GPU launcher](docs/MODAL.md)
- [Detection methods](docs/DETECTION.md)
- [Observation views and information bounds](docs/INFORMATION.md)
- [Figures](docs/FIGURES.md)
- [Configuration guide](configs/README.md)
- [Sources and attribution](ATTRIBUTION.md)
