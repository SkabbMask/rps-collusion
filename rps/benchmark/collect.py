"""Collect independent Ollama games and save a resumable labeled manifest."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import urllib.request

from simulate import run
from rps.detection.pipeline import DetectionConfig, write_json
from rps.game import assist_move
from rps.observe.features import load_jsonl, symbols
from rps.ollama import normalize_url
from .data import seed_for
from .progress import GameProgress


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_config(cfg):
    for key in ("n_edge_fit", "n_calibration", "n_test", "seed"):
        if type(cfg.get(key)) is not int or cfg[key] < (2 if key == "n_edge_fit" else 0 if key == "seed" else 1):
            raise ValueError(f"invalid {key}")
    for key, minimum in (("players", 3), ("rounds", 2)):
        if (not cfg.get(key) or len(set(cfg[key])) != len(cfg[key])
                or any(type(n) is not int or n < minimum for n in cfg[key])):
            raise ValueError(f"{key} must contain distinct integers >= {minimum}")
    if cfg.get("state", "none") not in ("none", "last_round"):
        raise ValueError("collection state must be none or last_round")
    defaults = cfg.get("defaults", {})
    if defaults.get("backend") != "ollama" or not defaults.get("model"):
        raise ValueError("batch collection requires an Ollama model for every player")
    if "seed" in defaults.get("options", {}):
        raise ValueError("set the top-level seed; player seeds must differ between games")
    if not 0 < cfg.get("alpha", .05) < 1:
        raise ValueError("alpha must be in (0,1)")
    DetectionConfig(**cfg.get("detector", {}))
    scenarios = cfg.get("scenarios", {"pair": cfg.get("collusion_probabilities", [])})
    if not scenarios or "pair" not in scenarios or set(scenarios) - {"pair", "star3", "two_pairs"}:
        raise ValueError("scenarios must include pair and may include star3 and two_pairs")
    for probabilities in scenarios.values():
        if (not probabilities or len(set(probabilities)) != len(probabilities)
                or any(not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 < p <= 1
                       for p in probabilities)):
            raise ValueError("scenario probabilities must be distinct values in (0,1]")
    if scenarios["pair"] != cfg.get("collusion_probabilities"):
        raise ValueError("collusion_probabilities must match scenarios.pair")


def plan(cfg):
    validate_config(cfg)
    scenarios = cfg.get("scenarios", {"pair": cfg["collusion_probabilities"]})
    for n in cfg["players"]:
        for t in cfg["rounds"]:
            group = f"N{n}_T{t}_{cfg.get('state', 'none')}"
            cases = [("honest", 0.0)] + [(s, cp) for s, cps in scenarios.items()
                                         if s == "pair" or n >= 6 for cp in cps]
            for split, count in (("edge_fit", cfg["n_edge_fit"]), ("calibration", cfg["n_calibration"]),
                                 ("test", cfg["n_test"])):
                for scenario, cp in cases if split == "test" else [("honest", 0.0)]:
                    for rep in range(count):
                        episode_id = f"{group}/{split}/{scenario}/cp{cp:g}/rep{rep}"
                        yield {"id": episode_id, "group": group, "split": split, "scenario": scenario,
                               "cp": cp, "seed": seed_for(cfg["seed"], episode_id), "n": n, "t": t}


def game_config(cfg, episode):
    names = [chr(65 + i) if i < 26 else f"P{i}" for i in range(episode["n"])]
    shuffled = random.Random(seed_for(episode["seed"], "membership")).sample(names, len(names))
    links, truth = {}, []
    if episode["scenario"] == "pair":
        truth = [shuffled[:2]]
        links = {shuffled[1]: shuffled[0]}
    elif episode["scenario"] == "star3":
        truth = [shuffled[:3]]
        links = {shuffled[1]: shuffled[0], shuffled[2]: shuffled[0]}
    elif episode["scenario"] == "two_pairs":
        truth = [shuffled[:2], shuffled[2:4]]
        links = {g[1]: g[0] for g in truth}
    players = [{"name": name, "role": "assistant" if name in links else "honest",
                **({"partner": links[name]} if name in links else {})} for name in names]
    return {"rounds": episode["t"], "state": cfg.get("state", "none"), "seed": episode["seed"],
            "collusion_probability": episode["cp"], "defaults": copy.deepcopy(cfg["defaults"]),
            "players": players}, [sorted(g) for g in truth]


def model_identity(defaults):
    endpoint = normalize_url(defaults["url"])
    if not endpoint.endswith("/api/chat"):
        raise ValueError("batch collection requires an Ollama /api/chat endpoint")
    with urllib.request.urlopen(endpoint[:-4] + "tags", timeout=10) as response:
        models = json.load(response)["models"]
    for model in models:
        if defaults["model"] in (model["name"], model["name"].removesuffix(":latest")):
            if not model.get("digest"):
                raise ValueError("Ollama did not return a model digest")
            return {"name": model["name"], "digest": model["digest"]}
    raise ValueError(f"model {defaults['model']} is not installed on the selected server")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "simulate.py", root / "rps/players.py",
             root / "rps/ollama.py", root / "rps/game.py", root / "rps/benchmark/data.py"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def manifest_for(cfg, episodes, identity):
    population = f"ollama:{identity['name']}:{identity['digest']}:{digest(cfg['defaults'])[:12]}"
    groups, entries = {}, []
    for ep in episodes:
        groups[ep["group"]] = {"state_mode": cfg.get("state", "none"), "population": population}
        _, truth = game_config(cfg, ep)
        entries.append({k: ep[k] for k in ("id", "group", "split", "scenario", "cp", "seed")}
                       | {"path": f"logs/{ep['id']}.jsonl", "truth": truth})
    return {"groups": groups, "episodes": entries,
            "truth_semantics": "instructed assistant agreement with CP>0; not measured compliance",
            "model": identity}


def completed(root, episode, fingerprint):
    log = root / "logs" / f"{episode['id']}.jsonl"
    meta = log.with_suffix(".meta.json")
    if not meta.exists():
        return False
    data = json.loads(meta.read_text())
    if (data["id"] != episode["id"] or data["seed"] != episode["seed"]
            or data["collection_fingerprint"] != fingerprint or not log.exists()
            or hashlib.sha256(log.read_bytes()).hexdigest() != data["log_sha256"]):
        raise ValueError(f"completed game changed: {episode['id']}")
    return True


def collect(cfg, output, *, limit=None):
    episodes = list(plan(cfg))
    root = Path(output)
    source = source_hashes()
    fingerprint = digest({"config": cfg, "source": source})
    state_file = root / "collection.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        if state["fingerprint"] != fingerprint:
            # This version only adds progress reporting to the previous collector.
            previous_source = dict(source)
            previous_source["rps/benchmark/collect.py"] = (
                "20e3f4081c91773edb6d9835d19d79cb7644f197d71804fe233b851951bde409")
            if (state["source_sha256"] != previous_source or state["config"] != cfg
                    or state["fingerprint"] != digest({"config": cfg, "source": previous_source})):
                raise ValueError("collection settings or source changed; use a new output directory")
            fingerprint = state["fingerprint"]
    elif root.exists() and any(root.iterdir()):
        raise ValueError("output is not an existing collection or an empty directory")
    identity = model_identity(cfg["defaults"])
    if state_file.exists():
        if state["model"] != identity:
            raise ValueError("model weights changed; use a new output directory")
    else:
        state = {"schema_version": 1, "fingerprint": fingerprint, "config": cfg,
                 "source_sha256": source, "model": identity, "n_planned": len(episodes)}
        atomic_json(state_file, state)
    atomic_json(root / "config.json", cfg)
    atomic_json(root / "manifest.json", manifest_for(cfg, episodes, identity))
    done = {ep["id"] for ep in episodes if completed(root, ep, fingerprint)}
    count = 0
    progress = GameProgress(sum(ep["n"] * ep["t"] for ep in episodes if ep["id"] not in done))
    for index, ep in enumerate(episodes, 1):
        if ep["id"] in done:
            continue
        if limit is not None and count >= limit:
            break
        log = root / "logs" / f"{ep['id']}.jsonl"
        partial = log.with_suffix(".partial.jsonl")
        # Keep interrupted attempts rather than treating them as completed games.
        for candidate in (log, partial):
            if candidate.exists():
                attempts = root / "attempts" / ep["id"]
                attempts.mkdir(parents=True, exist_ok=True)
                candidate.replace(attempts / f"attempt-{len(list(attempts.iterdir())) + 1}.jsonl")
        game, _ = game_config(cfg, ep)
        game["log"] = str(partial)
        print(f"[{index}/{len(episodes)}] {ep['id']}", flush=True)
        with progress.watch(partial, ep["n"], ep["t"]):
            players, history, _ = run(game)
        progress.finish(ep["n"] * ep["t"])
        observed = symbols(load_jsonl(partial), state_mode=cfg.get("state", "none"))
        if observed.T != ep["t"] or observed.N != ep["n"]:
            raise ValueError("simulator returned an incomplete game")
        partial.replace(log)
        signals = sum(len(r["signalled"]) for r in history)
        followed = sum(r["actions"][p.name] == assist_move(r["actions"][p.partner])
                       for p in players if p.partner for r in history if p.name in r["signalled"])
        atomic_json(log.with_suffix(".meta.json"), {
            "id": ep["id"], "seed": ep["seed"], "collection_fingerprint": fingerprint,
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            "parse_failures": {p.name: p.parse_failures for p in players},
            "signals": signals, "signals_followed": followed})
        done.add(ep["id"])
        atomic_json(root / "progress.json", {"complete": len(done), "planned": len(episodes)})
        count += 1
    n_complete = len(done)
    atomic_json(root / "progress.json", {"complete": n_complete, "planned": len(episodes)})
    metadata = [json.loads((root / "logs" / f"{ep['id']}.meta.json").read_text())
                for ep in episodes if ep["id"] in done]
    signals = sum(m["signals"] for m in metadata)
    followed = sum(m["signals_followed"] for m in metadata)
    failures = sum(sum(m["parse_failures"].values()) for m in metadata)
    atomic_json(root / "quality.json", {"games": n_complete, "parse_failures": failures,
                "signals": signals, "signals_followed": followed,
                "signal_compliance": followed / signals if signals else None})
    print(f"Saved {n_complete}/{len(episodes)} games. Manifest: {root / 'manifest.json'}", flush=True)
    print(f"Random fallbacks after parsing failures: {failures}. Signals followed: {followed}/{signals}.", flush=True)
    return n_complete == len(episodes)
