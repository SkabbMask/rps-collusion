"""Generate or import RPS episodes, evaluate five detectors, and write the report."""
import argparse
import json
from pathlib import Path

from rps.benchmark.data import generated_episodes, manifest_episodes, read_dataset
from rps.benchmark.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.json")
    parser.add_argument("--output", default="results/benchmark")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", help="replay a previously exported episodes.jsonl[.gz]")
    source.add_argument("--manifest", help="manifest of labeled upstream JSONL logs and honest calibration logs")
    parser.add_argument("--no-save-dataset", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key in ("n_edge_fit", "n_calibration", "n_test"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if any(not 0 < p <= 1 for p in cfg["collusion_probabilities"]):
        parser.error("alternative collusion probabilities must be in (0,1]")
    if any(n < 2 for n in cfg["players"]) or any(t < 2 for t in cfg["rounds"]):
        parser.error("need at least two players and rounds")
    if args.dataset:
        episodes, source_name = read_dataset(args.dataset), "replayed dataset"
    elif args.manifest:
        episodes, source_name = manifest_episodes(args.manifest), "external labeled logs"
    else:
        episodes, source_name = generated_episodes(cfg), "target simulator random backend + explicit synthetic stress models"
    evaluate(episodes, cfg, args.output, save_dataset=not args.no_save_dataset, source=source_name)
    print(f"Report: {args.output}/report.md")


if __name__ == "__main__":
    main()
