"""Compare detection outputs while ignoring machine-dependent timing."""
import argparse
import csv
import math
from pathlib import Path


TABLES = {
    "summary.csv": ("method",),
    "comparison.csv": ("group", "scenario", "cp", "method"),
    "predictions.csv": ("episode_id", "method"),
}
IGNORE = {"seconds", "median_ms", "details"}
JSON_COLUMNS = {"groups", "truth"}


def same(left, right):
    try:
        a, b = float(left), float(right)
    except ValueError:
        return left == right
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10)


def load(path, keys):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"duplicate key in {path}: {key}")
        result[key] = row
    return result


def compare(reference, candidate):
    import json
    errors = []
    for name, keys in TABLES.items():
        ref = load(Path(reference) / name, keys)
        actual = load(Path(candidate) / name, keys)
        if ref.keys() != actual.keys():
            errors.append(f"{name}: row keys differ")
        for key in ref.keys() & actual.keys():
            expected, observed = ref[key], actual[key]
            columns = set(expected) - IGNORE
            if not columns.issubset(observed):
                errors.append(f"{name}: missing columns for {key}")
                continue
            for column in sorted(columns):
                a, b = expected[column], observed[column]
                if column in JSON_COLUMNS:
                    equal = json.loads(a) == json.loads(b)
                else:
                    equal = same(a, b)
                if not equal:
                    errors.append(f"{name}: {key}, {column}: {a!r} != {b!r}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    errors = compare(args.reference, args.candidate)
    if errors:
        for error in errors[:10]:
            print(error)
        raise SystemExit(f"Comparison failed: {len(errors)} differences")
    print("Summary, condition tables and per-game predictions match (timing excluded).")


if __name__ == "__main__":
    main()
