import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "compare_results", Path(__file__).parents[1] / "scripts/compare_results.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class TestReplicationCheck(unittest.TestCase):
    def test_timing_is_ignored_but_changed_decisions_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            ref, candidate = [Path(folder) / name for name in ("ref", "candidate")]
            for root, timing in ((ref, "1"), (candidate, "20")):
                root.mkdir()
                for name, keys in checker.TABLES.items():
                    with (root / name).open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=[*keys, "flag", "score", "seconds"])
                        writer.writeheader()
                        writer.writerow({**dict.fromkeys(keys, "example"), "flag": "True",
                                         "score": "0.5", "seconds": timing})
            self.assertEqual(checker.compare(ref, candidate), [])
            path = candidate / "predictions.csv"
            path.write_text(path.read_text().replace("True", "False"))
            self.assertEqual(len(checker.compare(ref, candidate)), 1)

    def test_float_tolerance_and_special_values(self):
        self.assertTrue(checker.same("nan", "nan"))
        self.assertTrue(checker.same("-inf", "-inf"))
        self.assertTrue(checker.same("0.5", "0.50000000001"))
        self.assertFalse(checker.same("0.5", "0.6"))


if __name__ == "__main__":
    unittest.main()
