"""Collect a batch of Ollama games, then run detection and information analysis."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from rps.benchmark.collect import collect, completed, manifest_for, plan
from simulate import load_env


def analyze(output, cfg):
    root = Path(__file__).resolve().parent
    output = Path(output).resolve()
    state = json.loads((output / "collection.json").read_text())
    if (cfg != state["config"] or json.loads((output / "config.json").read_text()) != cfg
            or json.loads((output / "manifest.json").read_text()) != manifest_for(cfg, list(plan(cfg)), state["model"])):
        raise ValueError("saved configuration or labels changed; restore them before analysis")
    if not all(completed(output, ep, state["fingerprint"]) for ep in plan(cfg)):
        raise ValueError("collection is incomplete; resume it before analysis")
    for stage in ("detection", "information"):
        destination = output / stage
        if (destination / "report.md").exists():
            print(f"Using completed report: {destination / 'report.md'}", flush=True)
            continue
        attempts = output / "analysis-attempts"
        attempts.mkdir(exist_ok=True)
        temporary = attempts / f"{stage}-{len(list(attempts.iterdir())) + 1}"
        if stage == "detection":
            command = [sys.executable, str(root / "evaluate.py"), "--manifest", str(output / "manifest.json")]
        else:
            command = [sys.executable, str(root / "information.py"), "--dataset",
                       str(output / "detection/episodes.jsonl.gz"), "--relabel-interval",
                       str(cfg.get("relabel_interval", 25))]
        subprocess.run([*command, "--allow-repeated-observations", "--config", str(output / "config.json"),
                        "--output", str(temporary)],
                       cwd=root, check=True)
        temporary.replace(destination)
    quality = json.loads((output / "quality.json").read_text())
    (output / "report.md").write_text(
        f"# Model experiment\n\nModel: {state['model']['name']}\n\n"
        f"Completed games: {quality['games']}. Random fallbacks after parsing failures: {quality['parse_failures']}. "
        f"Signals followed: {quality['signals_followed']}/{quality['signals']}.\n\n"
        "- [Detection results](detection/report.md)\n- [Information views](information/report.md)\n\n"
        "Labels describe instructed coordination, not whether a model complied. See quality.json and "
        "the per-game metadata for parsing failures and signal compliance.\n", encoding="utf-8")
    print(f"Report: {output / 'report.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="batch settings; defaults to configs/qwen-experiment.json for a new run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", help="override the model for every player")
    parser.add_argument("--url", help="Ollama server for every player")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without contacting Ollama")
    parser.add_argument("--limit", type=int, help="collect at most this many new games, then stop")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    state_file = Path(args.output) / "collection.json"
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
    elif state_file.exists():
        cfg = copy.deepcopy(json.loads(state_file.read_text())["config"])
    else:
        cfg = json.loads(Path("configs/qwen-experiment.json").read_text())
    defaults = cfg.setdefault("defaults", {})
    defaults["url"] = args.url or defaults.get("url") or load_env(args.env).get("OLLAMA_URL", "http://localhost:11434")
    if args.model:
        defaults["model"] = args.model
    episodes = list(plan(cfg))
    turns = sum(ep["n"] * ep["t"] for ep in episodes)
    print(f"Model: {defaults['model']}. Games: {len(episodes):,}. Model turns: {turns:,} plus retries.", flush=True)
    if args.dry_run:
        return
    if args.analyze_only:
        if not state_file.exists() or cfg != json.loads(state_file.read_text())["config"]:
            parser.error("analysis requires an existing collection with matching settings")
        analyze(args.output, cfg)
    elif collect(cfg, args.output, limit=args.limit) and not args.collect_only:
        analyze(args.output, cfg)


if __name__ == "__main__":
    main()
