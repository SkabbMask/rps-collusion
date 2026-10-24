"""Compare six RPS auditor views using exact and held-out information bounds."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from rps.benchmark.data import generated_episodes, manifest_episodes, read_dataset
from rps.detection.pipeline import write_json
from rps.info.evaluate import evaluate_views
from rps.info.exact import enumerate_views
from rps.info.report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/information.json")
    parser.add_argument("--output", default="results/information")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", help="exported benchmark episodes.jsonl[.gz]")
    source.add_argument("--manifest", help="independently labeled logs; same schema as evaluate.py")
    parser.add_argument("--relabel-interval", type=int, default=25)
    parser.add_argument("--delta", type=float, default=0.05, help="family confidence error probability")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--allow-repeated-observations", action="store_true",
                        help="allow identical outcomes from independently collected games; IDs must remain unique")
    parser.add_argument("--exact-rounds", type=int, default=3)
    parser.add_argument("--exact-cp", type=float, default=0.5)
    parser.add_argument("--exact-relabel-interval", type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("choose a fresh/empty output directory")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    alpha = cfg.get("alpha", 0.05)
    exact = enumerate_views(rounds=args.exact_rounds, cp=args.exact_cp,
                            m=args.exact_relabel_interval, alpha=alpha)
    empirical = None
    if not args.exact_only:
        if args.dataset:
            episodes, source_name = read_dataset(args.dataset), "replayed dataset"
        elif args.manifest:
            episodes, source_name = manifest_episodes(args.manifest), "external labeled logs"
        else:
            for key in ("n_edge_fit", "n_calibration", "n_test"):
                if not isinstance(cfg.get(key), int) or cfg[key] < 1:
                    parser.error(f"{key} must be positive")
            if cfg["n_edge_fit"] < 2 or any(n < 2 for n in cfg["players"]) or any(t < 2 for t in cfg["rounds"]):
                parser.error("need >=2 fit episodes, players and rounds")
            if any(not 0 < p <= 1 for p in cfg["collusion_probabilities"]):
                parser.error("alternative CP values must be in (0,1]")
            episodes, source_name = generated_episodes(cfg), "scripted RPS; optional explicitly observed public silence"
        empirical = evaluate_views(episodes, output, m=args.relabel_interval, alpha=alpha,
                                   delta=args.delta, seed=cfg["seed"],
                                   allow_repeated_observations=args.allow_repeated_observations)
        empirical["source"] = source_name
        write_json(output / "information.json", empirical)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", cfg)
    root = Path(__file__).resolve().parent
    sources = [Path(__file__), *sorted((root / "rps/info").glob("*.py")),
               root / "rps/observe/views.py", root / "rps/observe/features.py",
               root / "rps/benchmark/data.py", root / "rps/players.py", root / "simulate.py"]
    write_json(output / "provenance.json", {"arguments": vars(args), "python": platform.python_version(),
               "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
               "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
               "information_feature_version": 1})
    write_report(output, empirical, exact)
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
