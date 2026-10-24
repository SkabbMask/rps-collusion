"""Run an N-player iterated RPS collusion simulation between LLM players and detect collusion.

Experiment settings live in config.json; machine-specific endpoints live in .env; CLI flags override both.

Usage:
    python simulate.py
    python simulate.py --config config.json --rounds 200 --cp 0.5 -v
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from rps.detect import detect
from rps.game import assist_move, payoffs
from rps.ollama import OllamaError
from rps.players import build_player

def state_of(history, mode):
    if mode == "none":
        return "-"
    if mode == "last_round":
        return "|".join(history[-1]["actions"].values()) if history else "start"
    raise ValueError(f"unknown state mode: {mode}")


def run(cfg):
    rng = random.Random(cfg.get("seed"))
    defaults = cfg.get("defaults", {})
    names = [p["name"] for p in cfg["players"]]
    players = [build_player(p, defaults, names, seed=rng.randrange(2**31)) for p in cfg["players"]]
    for client in {(c.url, c.model): c for c in (getattr(p, "client", None) for p in players) if c}.values():
        client.check()
    by_name = {p.name: p for p in players}
    # Assistants move after the other players so that, when collusion is active, they can receive their
    # partner's move over the covert channel. Everyone else moves simultaneously.
    assistants = [p for p in players if p.role == "assistant" and p.partner in by_name]
    others = [p for p in players if p not in assistants]
    if any(by_name[a.partner].role == "assistant" for a in assistants):
        raise ValueError("an assistant's partner must not itself be an assistant")
    cp = cfg.get("collusion_probability", 0.0)
    state_mode = cfg.get("state", "none")

    history, records = [], []
    log = None
    if cfg.get("log"):
        os.makedirs(os.path.dirname(cfg["log"]) or ".", exist_ok=True)
        log = open(cfg["log"], "w")
    pool = ThreadPoolExecutor(max_workers=len(players))
    try:
        for t in range(cfg["rounds"]):
            state = state_of(history, state_mode)
            futures = {p.name: pool.submit(p.act, history) for p in others}
            actions = {n: f.result() for n, f in futures.items()}
            signalled = [a.name for a in assistants if rng.random() < cp]
            futures = {
                a.name: pool.submit(a.act, history, actions[a.partner] if a.name in signalled else None)
                for a in assistants
            }
            actions.update({n: f.result() for n, f in futures.items()})
            actions = {n: actions[n] for n in names}
            pay = dict(zip(names, payoffs(list(actions.values()))))
            policies = {p.name: p.last_policy for p in players if getattr(p, "last_policy", None)}
            entry = {"round": t + 1, "state": state, "actions": actions, "payoffs": pay, "signalled": signalled,
                     "policies": policies}
            history.append(entry)
            records.append((state, actions))
            if log:
                log.write(json.dumps(entry) + "\n")
                log.flush()
            if cfg.get("verbose"):
                print(f"[{t + 1:4d}] {actions} {pay}{f'  (signal to {signalled})' if signalled else ''}", file=sys.stderr)
    finally:
        pool.shutdown()
        if log:
            log.close()
    return players, history, records


def report(players, history, records, alpha):
    names = [p.name for p in players]
    totals = {n: sum(r["payoffs"][n] for r in history) for n in names}
    print(f"Rounds: {len(history)}")
    print("Win rate: " + ", ".join(f"{n}={totals[n] / len(history):.3f}" for n in names))

    for a in players:
        signalled = [r for r in history if a.name in r["signalled"]]
        if signalled:
            ok = sum(r["actions"][a.name] == assist_move(r["actions"][a.partner]) for r in signalled)
            print(f"Signals sent to {a.name}: {len(signalled)}, followed: {ok} ({ok / len(signalled):.1%})")
    for p in players:
        if getattr(p, "parse_failures", 0):
            print(f"{p.name}: {p.parse_failures} unparseable replies (fell back to random)")

    gamma, net, pairs = detect(records, names, alpha)
    print("\nIndividual influence gamma(i;j):")
    for (i, j), g in gamma.items():
        print(f"  gamma({i};{j}) = {g:.4f}")
    print("Net influence Gamma(i;j):")
    for (i, j), g in net.items():
        print(f"  Gamma({i};{j}) = {g:+.4f}")
    found = ", ".join(f"{i}-{j}" for i, j in pairs)
    print(f"\nalpha = {alpha}: " + (f"collusion detected between {found}" if pairs else "no collusion detected"))


def load_env(path):
    """Read KEY=VALUE lines from a dotenv file (if present); real environment variables take precedence."""
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    env[key.strip()] = val.strip().strip("'\"")
    return {**env, **os.environ}


def apply_env(cfg, env):
    """OLLAMA_URL sets the endpoint for every player; PLAYER_<NAME>_URL overrides it for one player."""
    cfg.setdefault("defaults", {})["url"] = env.get("OLLAMA_URL", "http://localhost:11434")
    for p in cfg["players"]:
        url = env.get(f"PLAYER_{p['name'].upper()}_URL")
        if url:
            p["url"] = url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json", help="experiment settings")
    ap.add_argument("--env", default=".env", help="dotenv file with Ollama endpoints (ignored if missing)")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--cp", type=float, help="collusion probability (chance the covert signal is sent)")
    ap.add_argument("--alpha", type=float, help="collusion threshold")
    ap.add_argument("--url", help="override the Ollama URL for every player (ip:port or full URL)")
    ap.add_argument("--model", help="override the model for every player")
    ap.add_argument("--state", choices=["none", "last_round"], help="state used for conditioning in detection")
    ap.add_argument("--log", help="write per-round JSONL records here")
    ap.add_argument("--seed", type=int)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    apply_env(cfg, load_env(args.env))
    for key, val in [("rounds", args.rounds), ("collusion_probability", args.cp), ("alpha", args.alpha),
                     ("state", args.state), ("log", args.log), ("seed", args.seed)]:
        if val is not None:
            cfg[key] = val
    cfg["verbose"] = args.verbose or cfg.get("verbose", False)
    for key, val in [("url", args.url), ("model", args.model)]:
        if val is not None:
            for p in cfg["players"]:
                p[key] = val

    try:
        players, history, records = run(cfg)
    except OllamaError as e:
        sys.exit(f"Ollama error: {e}")
    report(players, history, records, cfg.get("alpha", 0.05))


if __name__ == "__main__":
    main()
