import json
import math
from pathlib import Path
import unittest

import numpy as np

from rps.detection.growth import grow_from_top_pair
from rps.detection.pipeline import (DetectionConfig, METHODS, decisions, score_methods,
                                    validate_calibration, load_calibration)
from test_detection import sym_from_array


class TestAverageZGrowth(unittest.TestCase):
    def test_star_does_not_require_every_pair_to_have_a_threshold_edge(self):
        z = np.zeros((5, 5))
        z[0, 1] = z[1, 0] = z[0, 2] = z[2, 0] = 10
        np.fill_diagonal(z, np.nan)
        result = grow_from_top_pair(z)
        self.assertEqual(result.members, frozenset([0, 1, 2]))
        self.assertEqual(result.seed_pair, (0, 1))
        self.assertEqual(result.added, [(2, 5, 2.5)])
        self.assertAlmostEqual(result.rejected[2], (20 / 3) * .25)

    def test_threshold_is_recomputed_after_each_join(self):
        z = np.zeros((5, 5))
        z[0, 1] = z[1, 0] = 10
        z[0, 2] = z[2, 0] = z[1, 2] = z[2, 1] = 3
        for i in (0, 1, 2):
            z[i, 3] = z[3, i] = 1.5
        np.fill_diagonal(z, np.nan)
        result = grow_from_top_pair(z)
        self.assertEqual(result.members, frozenset([0, 1, 2, 3]))
        self.assertAlmostEqual(result.added[1][2], 4 / 3)

    def test_equality_is_accepted_and_ties_use_player_order(self):
        z = np.array([[np.nan, 8, 2, 2], [8, np.nan, 2, 2],
                      [2, 2, np.nan, 2], [2, 2, 2, np.nan]])
        result = grow_from_top_pair(z)
        self.assertEqual(result.added[0], (2, 2, 2))
        self.assertEqual(result.members, frozenset(range(4)))

    def test_missing_and_negative_seeds_and_input_validation(self):
        self.assertIsNone(grow_from_top_pair(np.full((3, 3), np.nan)))
        z = np.full((3, 3), np.nan)
        z[1, 2] = z[2, 1] = -2
        result = grow_from_top_pair(z)
        self.assertEqual(result.seed_pair, (1, 2))
        self.assertEqual(result.seed_z, -2)
        for frac in (-1, 1.01, math.inf, math.nan):
            with self.assertRaises(ValueError):
                grow_from_top_pair(z, frac)
            with self.assertRaises(ValueError):
                DetectionConfig(growth_frac=frac)
        for invalid in (np.zeros((2, 3)), np.array([[0, 1], [2, 0]]), np.full((2, 2), math.inf)):
            with self.assertRaises(ValueError):
                grow_from_top_pair(invalid)

    def test_matches_pinned_teammate_fixtures(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/teammate_growth.json").read_text())
        for case in fixture["cases"]:
            result = grow_from_top_pair(np.asarray(case["z"], dtype=float), case["frac"])
            self.assertEqual(sorted(result.members), case["members"])
            self.assertEqual(list(result.seed_pair), case["seed_pair"])
            np.testing.assert_allclose(result.added, case["added"])
            if case["rejected"] is None:
                self.assertIsNone(result.rejected)
            else:
                np.testing.assert_allclose(result.rejected, case["rejected"])

    def test_episode_calibration_gates_growth_naming(self):
        rng = np.random.default_rng(12)
        moves = rng.integers(0, 3, (100, 4))
        moves[:, 1] = (moves[:, 0] - 1) % 3
        raw, _ = score_methods(sym_from_array(moves), DetectionConfig(permutations=19), 3)
        self.assertEqual(set(raw), set(METHODS))
        cal = {"thresholds": {m: math.inf for m in METHODS}}
        self.assertEqual(decisions(raw, cal)["growth"]["groups"], [])
        cal["thresholds"]["growth"] = raw["growth"].score - 1
        accepted = decisions(raw, cal)["growth"]
        self.assertTrue(accepted["flag"])
        self.assertEqual(accepted["details"]["seed_pair"], ["0", "1"])

    def test_unsupported_episodes_abstain(self):
        sym = sym_from_array(np.zeros((5, 3), dtype=int))
        raw, _ = score_methods(sym, DetectionConfig(), 3)
        self.assertTrue(raw["growth"].abstain)
        self.assertEqual(raw["growth"].candidates, [])

    def test_old_four_method_calibration_has_actionable_error(self):
        old = load_calibration(Path(__file__).parents[1] / "results/benchmark/calibration/N3_T300_none.json")
        with self.assertRaisesRegex(ValueError, "rerun evaluate.py"):
            validate_calibration(old, sym_from_array(np.zeros((300, 3), dtype=int)), DetectionConfig())


if __name__ == "__main__":
    unittest.main()
