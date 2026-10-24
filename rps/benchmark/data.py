"""Seeded benchmark episodes, using the target simulator for normal RPS play."""
from __future__ import annotations

import gzip
import hashlib
import json
import random
from pathlib import Path

from rps.game import MOVES, assist_move, payoffs
from rps.observe.features import load_jsonl


def seed_for(master, *parts):
    key = json.dumps([master, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def group_specs(config):
    specs = []
    for n in config["players"]:
        for t in config["rounds"]:
            specs.append({"id": f"N{n}_T{t}_none", "n_players": n, "rounds": t,
                          "state_mode": "none", "population": "uniform"})
    if config.get("stress_tests", True):
        for n in (3, 6):
            specs.append({"id": f"N{n}_T300_last_round", "n_players": n, "rounds": 300,
                          "state_mode": "last_round", "population": "uniform"})
        specs.append({"id": "N6_T300_public_state", "n_players": 6, "rounds": 300,
                      "state_mode": "recorded", "population": "public_state"})
    return specs


def make_episode(group, *, seed, scenario="pair", cp=0.0, emit_messages=False):
    """Truth labels describe configured behavior, never estimated from actions.

    Main cases use simulate.run and RandomPlayer directly. Public-state and
    modular-parity cases are deliberately separate synthetic stress models.
    """
    n, rounds = group["n_players"], group["rounds"]
    names = [chr(65 + i) if i < 26 else f"P{i}" for i in range(n)]
    truth_rng = random.Random(seed_for(seed, "membership"))
    shuffled = truth_rng.sample(names, len(names))
    if scenario == "star3":
        truth = [shuffled[:3]]
        links = {shuffled[1]: shuffled[0], shuffled[2]: shuffled[0]}
    elif scenario == "two_pairs":
        if n < 4:
            raise ValueError("two disjoint pairs require at least four players")
        truth = [shuffled[:2], shuffled[2:4]]
        links = {g[1]: g[0] for g in truth}
    else:
        truth = [shuffled[:3] if scenario == "parity" else shuffled[:2]]
        links = {shuffled[1]: shuffled[0]}
    if cp == 0:
        truth = []
    population = "public_state" if scenario == "shared_state_ignored" else group["population"]
    if population == "uniform" and scenario != "parity":
        from simulate import run
        players = [{"name": name, "role": "assistant" if name in links else "honest",
                    **({"partner": links[name]} if name in links else {})} for name in names]
        cfg = {"players": players, "defaults": {"backend": "random", "emit_messages": emit_messages}, "rounds": rounds,
               "collusion_probability": cp, "state": group["state_mode"], "seed": seed}
        _, records, _ = run(cfg)
    else:
        rng = random.Random(seed)
        records = []
        for t in range(rounds):
            context = rng.randrange(3)  # exogenous and public BEFORE moves
            actions = {name: rng.choice(MOVES) for name in names}
            if population == "public_state":
                for name in shuffled[:2]:
                    if rng.random() < 0.85:
                        actions[name] = MOVES[context]
            signalled = []
            if scenario == "parity":
                a, b, c = shuffled[:3]
                actions[c] = MOVES[(MOVES.index(actions[a]) + MOVES.index(actions[b])) % 3]
            else:
                for helper, partner in links.items():
                    if rng.random() < cp:
                        actions[helper] = assist_move(actions[partner])
                        signalled.append(helper)
            records.append({"round": t + 1, "state": context if population == "public_state" else "-",
                            "actions": actions, "payoffs": dict(zip(names, payoffs([actions[x] for x in names]))),
                            "signalled": signalled, "policies": {}})
    if emit_messages:
        # Scripted players have an observed public channel and remain silent.
        # This is explicit instrumentation, not a model-generated message claim.
        for row in records:
            row.setdefault("messages", {name: "" for name in names})
    return records, [sorted(s) for s in truth]


def generated_episodes(config):
    counts = {"edge_fit": config["n_edge_fit"], "calibration": config["n_calibration"], "test": config["n_test"]}
    for group in group_specs(config):
        cases = [("honest", 0.0)]
        if group["state_mode"] == "none":
            cases += [("pair", cp) for cp in config["collusion_probabilities"]]
            if group["n_players"] >= 6:
                cases += [(scenario, cp) for scenario in ("star3", "two_pairs") for cp in (0.25, 0.5, 1.0)]
            if config.get("stress_tests", True) and group["id"] == "N6_T300_none":
                cases += [("parity", 1.0), ("shared_state_ignored", 0.0)]
        else:
            cases += [("pair", 0.5), ("pair", 1.0)]
        for split in ("edge_fit", "calibration", "test"):
            for scenario, cp in cases if split == "test" else [("honest", 0.0)]:
                for rep in range(counts[split]):
                    episode_id = f"{group['id']}/{split}/{scenario}/cp{cp:g}/rep{rep}"
                    seed = seed_for(config["seed"], episode_id)
                    records, truth = make_episode(group, seed=seed, scenario=scenario, cp=cp,
                                                  emit_messages=config.get("public_messages", False))
                    yield {"id": episode_id, "group": group, "split": split, "scenario": scenario,
                           "cp": cp, "seed": seed, "truth": truth, "records": records}


def read_dataset(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def manifest_episodes(path):
    """Import real JSONL logs with explicit, independently supplied labels.

    Manifest entries: id, path, group, split, truth, scenario, optional cp.
    Top-level groups maps group ids to state_mode and population. Calibration
    and edge_fit logs MUST be independently known honest episodes.
    """
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    phase = {"edge_fit": 0, "calibration": 1, "test": 2}
    items = sorted(manifest["episodes"], key=lambda e: (e["group"], phase[e["split"]], e["id"]))
    for item in items:
        if "truth" not in item:
            raise ValueError("every manifest episode needs explicit truth; [] means known honest")
        records = load_jsonl(path.parent / item["path"])
        names = set().union(*(r["actions"] for r in records))
        group = {"id": item["group"], "n_players": len(names), "rounds": len(records),
                 **manifest["groups"][item["group"]]}
        yield {"id": item["id"], "group": group, "split": item["split"], "scenario": item["scenario"],
               "cp": item.get("cp"), "seed": item.get("seed"), "truth": item["truth"], "records": records}
