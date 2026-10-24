import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rps.benchmark.data import make_episode, seed_for
from rps.benchmark.evaluate import evaluate
from rps.benchmark.metrics import binary_metrics, naming_metrics, wilson
from rps.game import assist_move, payoffs
from simulate import run


class TestSimulator(unittest.TestCase):
    def test_fast_path_exactly_matches_pinned_upstream_fixture(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/upstream_seed7.json").read_text())
        _, history, _ = run(fixture["config"])
        self.assertEqual(history, fixture["history"])

    def test_assistant_guarantees_primary_point(self):
        for primary in ("rock", "paper", "scissors"):
            for third in ("rock", "paper", "scissors"):
                self.assertEqual(payoffs([primary, assist_move(primary), third])[0], 1)

    def test_reproducible_generation_and_random_members(self):
        group = {"n_players": 6, "rounds": 50, "population": "uniform", "state_mode": "none"}
        a, truth = make_episode(group, seed=7, scenario="star3", cp=1)
        b, truth2 = make_episode(group, seed=7, scenario="star3", cp=1)
        self.assertEqual((a, truth), (b, truth2))
        self.assertEqual(len(truth[0]), 3)
        self.assertNotEqual(seed_for(1, "test", 0), seed_for(1, "calibration", 0))

    def test_modular_parity_has_full_joint_dependence(self):
        group = {"n_players": 6, "rounds": 100, "population": "uniform", "state_mode": "none"}
        rows, truth = make_episode(group, seed=3, scenario="parity", cp=1)
        self.assertEqual(len(truth[0]), 3)
        self.assertEqual(len(rows), 100)
        names = truth[0]
        codes = {m: i for i, m in enumerate(("rock", "paper", "scissors"))}
        valid_relations = []
        for c in names:
            a, b = [n for n in names if n != c]
            valid_relations.append(all(codes[r["actions"][c]] == (codes[r["actions"][a]] + codes[r["actions"][b]]) % 3 for r in rows))
        self.assertTrue(any(valid_relations))


class TestMetrics(unittest.TestCase):
    def test_wilson_endpoints_contain_observed_rate(self):
        self.assertEqual(wilson(0, 100)[0], 0)
        self.assertEqual(wilson(100, 100)[1], 1)
        self.assertGreater(wilson(0, 100)[1], 0)

    def test_exact_groups_different_from_union(self):
        m = naming_metrics([["A", "B", "C", "D"]], [["A", "B"], ["C", "D"]])
        self.assertEqual(m["exact_members"], 1)
        self.assertEqual(m["exact_groups"], 0)
        self.assertEqual(m["jaccard"], 1)
        self.assertAlmostEqual(m["pair_precision"], 1 / 3)

    def test_nondetection_counts_as_naming_failure(self):
        m = naming_metrics([], [["A", "B"]])
        self.assertEqual(m["exact_groups"], 0)
        self.assertEqual(m["jaccard"], 0)
        self.assertEqual(m["pair_f1"], 0)

    def test_confusion_and_negative_infinite_scores(self):
        rows = []
        for positive, flag, score in [(False, False, -np.inf), (False, True, 1), (True, True, 2), (True, False, 0)]:
            rows.append({"positive": positive, "flag": flag, "score": score, "abstain": False, "seconds": .01,
                         **naming_metrics([["A", "B"]] if flag else [], [["A", "B"]] if positive else [])})
        m = binary_metrics(rows)
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["auroc"], .75)
        self.assertEqual(m["exact_groups"], .5)


class TestLeakageGuards(unittest.TestCase):
    def setUp(self):
        self.group = {"id": "test", "n_players": 3, "rounds": 30, "population": "uniform", "state_mode": "none"}
        records, _ = make_episode(self.group, seed=8, cp=0)
        self.ep = {"id": "one", "group": self.group, "split": "edge_fit", "truth": [],
                   "scenario": "honest", "cp": 0, "records": records}
        self.cfg = {"seed": 7, "detector": {"permutations": 5}}

    def test_relabeling_truth_cannot_hide_duplicate_calibration_trajectory(self):
        duplicate = copy.deepcopy(self.ep)
        duplicate.update(id="two", split="calibration")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "duplicate observable trajectory"):
                evaluate(iter([self.ep, duplicate]), self.cfg, tmp, save_dataset=False)

    def test_collusion_labels_cannot_calibrate(self):
        self.ep["truth"] = [["A", "B"]]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "honest"):
                evaluate(iter([self.ep]), self.cfg, tmp, save_dataset=False)


if __name__ == "__main__":
    unittest.main()
