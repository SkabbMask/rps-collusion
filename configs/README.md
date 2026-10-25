# Configurations

Run commands from the repository root. Single-game settings are consumed by
`simulate.py`; batch settings are consumed by `evaluate.py`, `information.py`
or `experiment.py`.

| Configuration | Runner | Purpose |
| --- | --- | --- |
| `random.json` | `simulate.py` | One three-player scripted game, 300 rounds, CP=0.5 |
| `ollama.json` | `simulate.py` | Three Ollama players, 200 rounds |
| `ollama-messages.json` | `simulate.py` | Model game with public-message recording |
| `full-information.json` | `evaluate.py` | Reference five-method scripted grid, observed silent messages |
| `benchmark.json` | `evaluate.py` | Same main grid without public-message recording |
| `smoke.json` | `evaluate.py` | Smaller scripted grid for checking the pipeline |
| `information.json` | `information.py` | Smaller three-player information-view experiment |
| `qwen-smoke.json`, `qwen-hour.json` | `experiment.py` | Six-game connection/behavior checks; too little calibration for performance claims |
| `qwen-explore-t30.json` | `experiment.py` | 69 three-player games, 30 rounds |
| `qwen-pilot-t30.json` | `experiment.py` | 119 three-player games, 30 rounds |
| `qwen-comparison-t30.json` | `experiment.py` | 439 three-player games, 30 rounds |
| `qwen-pilot.json` | `experiment.py` | 119 three-player games, 100 rounds |
| `qwen-experiment.json` | `experiment.py` | Full N=3/N=6 model grid; 4,244,400 model turns |

The repository-root `config.json` is the legacy hybrid game: only B uses
Ollama. A `--model` override does not turn scripted players into model players.

For model batches, `--dry-run` shows counts without inference. The honest
fitting and calibration counts set threshold resolution; `n_test` is the
number of repetitions per condition. Never change a live collection's
configuration to shorten it. Start a new output directory instead.

Replay a saved experiment with its own `config.json`. Example files can evolve;
a result's resolved configuration is the reference for that run.
