"""Apply all five detectors to one upstream JSONL log using fresh calibration."""
import argparse
from dataclasses import asdict

from rps.detection.pipeline import METHOD_LABELS, detect_episode, load_calibration, write_json
from rps.observe.features import load_jsonl, symbols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--calibration", required=True, help="matching N, T, state mode, settings AND honest population")
    parser.add_argument("--output", default="runs/detection.json")
    args = parser.parse_args()
    cal = load_calibration(args.calibration)
    sym = symbols(load_jsonl(args.log), state_mode=cal["signature"]["state_mode"])
    results, evidence = detect_episode(sym, cal)
    write_json(args.output, {"players": sym.names, "rounds": sym.T, "results": results,
                             "pairwise": asdict(evidence), "calibration_group": cal["group"],
                             "interpretation": "Statistical action dependence, not proof of intent. Honest-population match must be checked by the caller."})
    for method, value in results.items():
        outcome = "insufficient evidence" if value["abstain"] else str(value["groups"]) if value["flag"] else "no flag"
        print(f"{METHOD_LABELS[method]}: {outcome}")
    print(args.output)


if __name__ == "__main__":
    main()
