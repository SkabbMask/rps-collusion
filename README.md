# N-player RPS collusion simulation

Reproduces (and extends to N players) the iterated Rock Paper Scissors experiment from Bonjour et al., *Information-Theoretic Approach
to Detect Collusion in Multi-Agent Games* (UAI 2022), with each player backed by an Ollama model.

- **A** (`primary`): the primary colluder.
- **B** (`assistant`): with probability `collusion_probability` (CP), B is covertly told A's move and asked
  to play the move that guarantees A a point. Whether the LLM actually follows the signal is reported.
- **C** (`honest`): plays on its own.

Any number of players works: add entries to `players` in `config.json`. The scoring rules generalize
directly: everyone plays the same move -> 0 for all; rock, paper and scissors all appear -> 1 for all;
exactly two moves appear -> 1 for each player holding the winning move. Several assistants can share a
primary, and several coalitions can be configured (each `assistant` names its `partner`). With more
players the "all three moves appear" draw happens more often, so collusion pays off less, but detection is unaffected because it measures how players' actions depend on each other, not scores.

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
- **Graph clusters**: connected components of the pairs with z ≥ z_α, then pruned: a member must have at
  least `min_links` links inside the group, each with z ≥ `link_frac` × the group's strongest z
  (defaults 1 and 0.25). The weakest member is dropped and the counts recomputed until all qualify. This
  keeps out an honest player who is only weakly linked to the group. A group is scored by its mean z per
  pair (the sum would grow with group size) and accepted above the honest q_95 of the same score.
- **Growth decoder**: start from the pair with the highest z, then repeatedly take the outside player with
  the highest summed z to the set and add it while its mean z to the set is ≥ `frac` × the mean z over
  all pairs inside the set, recomputed after every addition (`--frac`, default 0.25). Flagged when
  z(top pair) ≥ z_α. Assistants of one primary form a star (each depends on the primary, not on each
  other), so the k-th assistant's mean is only about z/k: with m assistants, `frac` must be at most
  about 1/m.
- **Histogram view** (no identities): a G test of the per-round move counts against a multinomial. It also
  reacts to honest players with different move biases, so treat it as a weak signal on mixed line-ups.

z_α and the cluster threshold q_95 (with the same pruning) are calibrated on 50 honest surrogates of the same
log (each player's moves shuffled independently within state cells, which removes any dependence), so
each is a 5% false-alarm threshold for that episode. The relabelled view runs everything per block of
m rounds. Ground truth (the coalitions) is written to the log's first line and is used only to score the
result. The optional `detection` block in the config sets `R`, `n_null`, `level`, `frac`, `min_links` and
`link_frac` (also `--min-links` / `--link-frac` on analyze.py).

## Comparing the methods

Apart from the histogram test, all methods start from the same pairwise statistics and differ in how they
turn them into a verdict.

| Method | Finds | Good at | Weak at |
|---|---|---|---|
| Paper rule (Γ ≥ τ) | at most one pair | a single colluding pair, as in the paper | coalitions of 3+: members mask each other, often nothing is found |
| Graph clusters | any connected group | stars and mixed shapes; medium collusion | an honest player as strongly linked as a colluder is kept |
| Growth decoder | one group grown from the top pair | naming the exact coalition, especially larger ones | `frac` must suit the coalition shape (≈ 1/number of assistants) |
| colav ML decoder (unused) | most likely k-subset | coalitions where every pair is linked (cliques) | stars: members linked only to the primary are dropped |
| Histogram G test | yes/no, no names | needs no identities | weak; fooled by honest players with different biases |

**Benchmark** ([experiments/bench_decoders.py](experiments/bench_decoders.py)): random players with
rule-based assistants, 200 rounds, 25 episodes per cell. The two "roped-in" scenarios add an honest player
D who copies assistant B's move 30% of the time (one link), or a random coalition member's 40% of the time
(weak links to all). D is not part of the true coalition. Each cell is the fraction of episodes in which
the method named the exact coalition, at CP = 0.2 / 0.35 / 0.5.

| Coalition | colav ML decoder | Growth, frac 0.25 | Graph, unpruned | Graph, link ≥ 0.25 top | Graph, link ≥ 0.5 top |
|---|---|---|---|---|---|
| A + 1 assistant | 0.48 / 1.00 / 1.00 | 0.48 / 1.00 / 1.00 | 0.48 / 1.00 / 1.00 | 0.48 / 1.00 / 1.00 | 0.48 / 1.00 / 1.00 |
| A + 2 assistants | 0.16 / 0.60 / 0.96 | 0.28 / 0.92 / 1.00 | 0.28 / 1.00 / 0.96 | **0.28 / 1.00 / 1.00** | 0.24 / 0.88 / 1.00 |
| A + 3 assistants | 0.08 / 0.68 / 0.96 | 0.28 / 0.92 / 1.00 | 0.28 / 0.96 / 0.84 | **0.28 / 1.00 / 1.00** | 0.24 / 0.76 / 0.96 |
| A + 2, D copies B | 0.00 / 0.32 / 0.84 | 0.00 / 0.28 / 0.84 | 0.00 / 0.08 / 0.00 | 0.00 / 0.08 / 0.44 | **0.00 / 0.32 / 0.92** |
| A + 2, D weakly linked to all | 0.24 / 0.52 / 0.92 | 0.20 / 0.56 / 0.92 | 0.36 / 0.40 / 0.52 | 0.36 / 0.64 / **0.96** | 0.32 / **0.76** / **0.96** |

- **Pruning** at 0.25 × the strongest link is a free improvement: it is never worse than the unpruned graph
  beyond noise, and it keeps a weakly linked honest player out once the collusion is clear (D linked to
  all: 0.52 → 0.96 at CP = 0.5). At 0.5 it also removes D when D copies one member, but it starts dropping
  real assistants in larger stars (A + 3: 0.76 at CP = 0.35).
- **Requiring 2 links** per member (not in the table) breaks pairs and stars: assistants are linked only
  to the primary. It found nothing at all for A + 1 and at most 0.84 elsewhere, so `min_links` stays 1.
- **Mean z instead of the sum** changed nothing in these scenarios (one coalition each), but it stops a
  large, weakly linked group from outscoring a small, strong one.
- **D copying B 30% of the time is as dependent on B as an assistant at CP ≈ 0.3.** No pairwise
  method can tell it apart at CP ≤ 0.35; pruning only helps once the real collusion is clearly stronger.
- At CP = 0.2 no method names a coalition of 3+ reliably; they mostly find its strongest pair. Detection
  itself (flagging that collusion happens) is about equal across methods. Differences of about ±0.1 are
  within noise at 25 episodes.

The scripted assistants always follow the signal. In the LLM runs (`llama3.2` assistants, which followed
72–77% of signals), with 200 rounds at CP = 0.5:

| Run | Paper rule | Graph clusters | Growth, frac 0.25 |
|---|---|---|---|
| 3 players, A + B | {A, B} | {A, B} | {A, B} |
| 6 players, A + B + C | nothing | {A, B, C} | {A, B, C} |

On honest episodes all calibrated methods raised false alarms at the intended rate of about 5% (checked
with 100 surrogates). With the default 50 surrogates the threshold itself is noisier; use `--n-null 100`
for tighter control.

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
