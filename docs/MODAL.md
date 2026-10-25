# Run on Modal

Install the project locally first; see [RUNNING.md](RUNNING.md).


`modal_experiment.py` runs Ollama and the experiment together on one L4 GPU.
It keeps games on the `rps-results` volume and caches weights on
`rps-ollama-models`. It sends no inference requests to your local Ollama.
Games remain sequential, with concurrent player calls within each round.
This launcher has been checked locally but has not been tested on a paid GPU.

The runner uses Ollama's `/api/chat` protocol. OpenAI-compatible endpoints
cannot be supplied through `--url`.

### Setup and connection check

```bash
python -m pip install 'modal>=1.6.2,<2'
modal setup

modal run modal_experiment.py::run \
  --config configs/qwen-smoke.json \
  --model qwen2.5:14b --run-name modal-qwen14b-smoke \
  --max-seconds 1800
```

The image pins Ollama 0.5.7 to match the initial local experiments. The first
run downloads the model. The GPU is billed while the function runs, including
download and analysis; storage is billed separately. See
[Modal GPU resources](https://modal.com/docs/guide/gpu) and
[current pricing](https://modal.com/pricing).

### Time two games, then run the pilot

```bash
modal run modal_experiment.py::run \
  --config configs/qwen-pilot-t30.json \
  --model qwen2.5:14b --run-name modal-qwen14b-pilot-t30 \
  --limit 2 --collect-only --max-seconds 3600
```

Read the collection ETA in the logs. If the measured throughput is 3 turns/s,
the 119-game pilot needs about one hour of collection; at 1 turn/s it needs
about three hours. Choose the time limit from the measured speed, allowing
startup and analysis time. For example, a four-hour limit is:

```bash
modal run --detach modal_experiment.py::run \
  --config configs/qwen-pilot-t30.json \
  --model qwen2.5:14b --run-name modal-qwen14b-pilot-t30 \
  --max-seconds 14400
```

To run the smaller model on Modal instead, use `--model qwen2.5:1.5b` and
`--run-name modal-qwen15b-pilot-t30`. Change the config and run name for the
exploratory or larger study. Do not run two jobs writing the same run name.

`--max-seconds` limits each invocation, including model startup and analysis,
with a small cleanup allowance afterward. The maximum is 82,800 seconds
(23 hours). If it expires, completed games are retained and the interrupted
game restarts on resume. Repeat the same command with unchanged source,
settings and model to continue. Detached runs survive terminal disconnection;
use the Modal dashboard link printed by the CLI to view logs or stop a run.

### Download the results

After the function has exited and committed its files:

```bash
modal volume get rps-results /modal-qwen14b-pilot-t30 \
  results/modal-qwen14b-pilot-t30
```

The downloaded folder contains `report.md`, `detection/report.md`,
`information/report.md`, logs, configuration and labels. See
[Modal volume commands](https://modal.com/docs/cli/latest/volume).
If collection finished but analysis was interrupted, rerun the Modal command
with `--analyze-only`; it reuses saved games without model inference. Use the
same config, model and run name.

