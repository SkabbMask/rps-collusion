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

## Detection

After a run, [analyze.py](analyze.py) runs these detectors on the log. They come from the market project
(`colav`) and are adapted to RPS in [detection/](detection/) and [observe/](observe/). To analyse a saved
log again without rerunning the game:

    python analyze.py runs/latest.jsonl
    python analyze.py runs/latest.jsonl --view relabelled --m 50   # pseudonyms change every 50 rounds

- **Paper rule** (Bonjour et al.): γ(i;j) = I(Aᵢ;Aⱼ|S) in bits, net influence Γ(i;j) = γ(i;j) − max_k γ(k;j),
  and a pair is flagged when both Γ ≥ τ = 0.05 (`alpha` in the config). Because Γ subtracts the strongest
  *other* influence, two assistants of the same primary mask each other, so it finds at most one pair.
- **Permutation z**: for each pair, γ is compared with R shuffles of one player's moves within state
  cells. This doesn't subtract other players' influence, so coalitions larger than two stay visible.
- **Graph clusters**: connected components of the pairs with z ≥ z_α, scored by Q = the sum of z inside.
- **ML decoder**: the most likely coalition of unknown size k (plug-in μ, penalised by ln C(N,k)).
- **Histogram view** (no identities): a G test of the per-round move counts against a multinomial. It also
  reacts to honest players with different move biases, so treat it as a weak signal on mixed line-ups.

z_α, the cluster threshold q_95 and the decoder threshold are calibrated on 50 honest surrogates of the same
log (each player's moves shuffled independently within state cells, which removes any dependence), so
each is a 5% false-alarm threshold for that episode. The relabelled view runs everything per block of
m rounds. Ground truth (the coalitions) is written to the log's first line and is used only to score the
result. The optional `detection` block in the config sets `R`, `n_null`, `level` and `k_max`.

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
