# N-player RPS collusion simulation

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
less, but detection is unaffected because it measures how players' actions depend on each other, not scores.

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
