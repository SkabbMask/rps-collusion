# Project contents

Start with [README.md](README.md) and [the run guide](docs/RUNNING.md).

| File or directory | Purpose |
| --- | --- |
| `experiment.py` | Collect and resume a model batch, then run both analyses |
| `simulate.py` | Run a game with random players or Ollama models |
| `analyze.py` | Apply five detectors to a saved game |
| `evaluate.py` | Fit thresholds, evaluate games and write reports |
| `information.py` | Compare the six information views |
| `configs/` | Example game and experiment settings |
| `rps/` | Game rules, players, observations, detectors and reporting |
| `tests/` | Unit tests and reference fixtures |
| `results/` | Saved experiments, datasets and reports |

The suite covers game rules, detectors, information bounds and batch collection. See [validation](docs/NEW_DECODER_VALIDATION.md)
for the measured comparison and [research notes](docs/PAPER_READINESS_REVIEW.md)
for limits on paper claims. Source revisions are listed in [ATTRIBUTION.md](ATTRIBUTION.md).
