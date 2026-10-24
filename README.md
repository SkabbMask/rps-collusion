# N-player RPS collusion simulation

## Run the complete experiment

Follow [the complete step-by-step run guide](docs/RUNNING.md): environment setup,
tests, one game, the full 6,253-episode **five-detector** benchmark, fresh
calibration, single-game analysis, six-view information bounds and Ollama runs.

The `new-decoder` average-z growth method is now the fifth detector; the
previous four remain available for direct comparison. Existing four-method
calibration files must be regenerated. The guide uses fresh output paths
`results/full-detection-growth` and `results/full-information-growth`.
See [integration validation and measured comparison](docs/NEW_DECODER_VALIDATION.md).

## Six-view information-theoretic report

The information experiment compares `full`, `identified`, `relabelled`,
`histogram`, `summaries` and `metadata` observations adapted to RPS. It reports
exact small-model TV, Bayes error, mutual information and power limits, plus
held-out confidence lower bounds for view information. Public-message logging
is optional; unknown messages are never silently interpreted as silence.

```bash
python -m pip install -r requirements-detection.txt
python information.py --output results/my-information
```

See [information experiment instructions](docs/INFORMATION.md) and
[executed scripted demonstration](results/information-demo/report.md).
The demonstration uses observed public silence, not LLM messages.

## Added: three COLAV detection methods and reproducible evaluation

This branch starts from `SkabbMask/rps-collusion` revision
`4e11fd2efbe5fc9d5d80bec7d32522514ac2896f` and adapts the requested methods from
`lee-chaeeun/apart-29-ml4gang`, branch `dev-modal`, revision
`20c3090a2fb6b6d329cbc04cd749b2534501dc13`.

The three additions are **graph connected components**, **graph Louvain communities**, and
the **plug-in maximum-likelihood coalition decoder**. They share a permutation-calibrated
conditional-mutual-information evidence layer. The original `rps/detect.py` remains unchanged
and is evaluated as the baseline method. Average-z growth from the teammate's
`new-decoder` branch is the fifth method. Observation helpers now work with categorical
RPS actions and public game histories. See [attribution](ATTRIBUTION.md) and
[method/API/data documentation](docs/DETECTION.md).

```bash
python -m pip install -r requirements-detection.txt
python -m unittest discover -s tests -v

# No Ollama server needed for this reproducible scripted benchmark.
python evaluate.py --config configs/benchmark.json --output results/reproduction

# Play and analyse one game using the freshly generated five-method calibration.
python simulate.py --config configs/random.json
python analyze.py --log runs/random.jsonl \
  --calibration results/reproduction/calibration/N3_T300_none.json \
  --output runs/detection.json
```

Historical four-method results: [Markdown report](results/benchmark/report.md),
[per-condition table](results/benchmark/comparison.csv),
[all predictions](results/benchmark/predictions.csv), and
[figures](results/benchmark/figures/). The report documents actual executed experiments,
confidence intervals, coalition recovery, false alarms, abstentions, runtime and limitations.
The benchmark uses the original simulator's random/scripted players, plus clearly labeled
synthetic stress models. It does **not** measure LLM performance.

Use `--config configs/smoke.json` for a smaller pipeline check; choose a fresh output directory.
Use `--manifest` to evaluate independently labeled recorded games, or `--dataset` to replay
the included `results/benchmark/episodes.jsonl.gz` without regenerating games.
The generated honest calibration must not be assumed valid for a different policy population.

## Original game

Reproduces (and extends to N players) the iterated Rock Paper Scissors experiment from Bonjour et al., *Information-Theoretic Approach
to Detect Collusion in Multi-Agent Games* (UAI 2022), with each player backed by an Ollama model.

- **A** (`primary`): the primary colluder.
- **B** (`assistant`): with probability `collusion_probability` (CP), B is covertly told A's move and asked
  to play the move that guarantees A a point. Whether the LLM actually follows the signal is reported.
- **C** (`honest`): plays on its own.

Any number of players works: add entries to `players` in `config.json`. The scoring rules generalize
directly: everyone plays the same move -> 0 for all; rock, paper and scissors all appear -> 1 for all;
exactly two moves appear -> 1 for each player holding the winning move. Several colluding pairs can be
configured (each `assistant` names its `partner`), and every pair whose net influences both exceed α is
reported. With more players the "all three moves appear" draw happens more often, so collusion pays off
less. Detection measures how players' actions depend on each other, not scores; its finite-sample power
can still change with player count, coalition structure, policies and state sparsity.

After the run, collusion is detected using the individual influence γ(i;j) = I(Aᵢ;Aⱼ|S) and the net
influence Γ(i;j) = γ(i;j) − max_k γ(k;j). A pair is flagged when both Γ values are ≥ α (default 0.05).

## Run

    cp .env.example .env    # set OLLAMA_URL to the machine running Ollama, e.g. 192.168.1.50:11434
    python simulate.py
    python simulate.py --rounds 200 --cp 0.5 --model qwen2.5:14b -v

**`.env`** (not committed) holds the machine-specific endpoints: `OLLAMA_URL` for every player and
optionally `PLAYER_<NAME>_URL` for a single one. A URL may be `ip:port` (`/api/chat` is appended) or a
full `http://ip:port/path`. Real environment variables override the file. Without either, players use
`http://localhost:11434`.

**[config.json](config.json)** (committed) holds the experiment settings. Each player can override any key
in `defaults`: `model`, `options` (Ollama options, e.g. temperature), `think`, `history_window`,
`system_prompt` (may use `{name}`/`{partner}`), `timeout`, and `backend` (`ollama` or `random`; `random`
gives the paper's uniform/rule-based players as a non-LLM baseline).

CLI flags (`--rounds`, `--cp`, `--alpha`, `--model`, `--url`, `--state`, `--seed`, `--log`) override both.

`state` sets what detection conditions on: `none` (a single state, as in the paper's manual experiment) or
`last_round` (the previous joint action; this removes correlation that comes from the shared history,
but needs many more rounds). Per-round records go to the JSONL file set by `log`.
