import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rps.detect import individual_influence
from rps.detection.decoder import decode, ml_decode
from rps.detection.graph import clusters, edge_set
from rps.detection.pairwise import PairwiseParams, analyse_pairwise, cmi_from_tables
from rps.detection.pipeline import (DetectionConfig, RawResult, decisions, load_calibration,
                                    rank_threshold, score_methods, validate_calibration, write_json)
from rps.game import MOVES
from rps.observe.features import symbols


def sym_from_array(moves, states=None):
    states = ["-"] * len(moves) if states is None else states
    return symbols([{"state": state, "actions": {str(i): None if x == -1 else MOVES[x] for i, x in enumerate(row)}}
                    for state, row in zip(states, moves)])


class TestPairwise(unittest.TestCase):
    def test_higher_order_dependence_has_zero_pairwise_mi(self):
        moves = np.array([[a, b, (a + b) % 3] for _ in range(30) for a in range(3) for b in range(3)])
        result = analyse_pairwise(sym_from_array(moves), PairwiseParams(39, 20, 10))
        np.testing.assert_allclose(result.gamma[np.triu_indices(3, 1)], 0, atol=1e-12)
        self.assertTrue((result.pvalue[np.triu_indices(3, 1)] == 1).all())

    def test_anonymous_view_is_not_silently_used_as_identified(self):
        from rps.observe.views import histogram_view
        view = histogram_view(sym_from_array(np.zeros((30, 3), dtype=int)))
        with self.assertRaises(TypeError):
            analyse_pairwise(view, PairwiseParams())

    def test_cmi_exact_nats_and_matches_original(self):
        table = np.zeros((1, 3, 3))
        table[0] = np.eye(3) * 30
        self.assertAlmostEqual(float(cmi_from_tables(table)), math.log(3))
        table[:] = 10
        self.assertAlmostEqual(float(cmi_from_tables(table)), 0)
        rng = np.random.default_rng(42)
        moves = rng.integers(0, 3, (300, 3))
        states = rng.integers(0, 3, 300)
        sym = sym_from_array(moves, states.tolist())
        r = analyse_pairwise(sym, PairwiseParams(39, 20, 2))
        records = [(int(s), {str(i): MOVES[x] for i, x in enumerate(row)}) for s, row in zip(states, moves)]
        self.assertAlmostEqual(r.gamma[0, 1], individual_influence(records, "0", "1"), places=12)

    def test_public_state_removes_pure_common_cause(self):
        states = np.tile(np.arange(3), 100)
        moves = np.column_stack([states, states, states])
        r = analyse_pairwise(sym_from_array(moves, states.tolist()), PairwiseParams(39, 20, 10))
        self.assertEqual(r.z[0, 1], 0)
        self.assertEqual(r.pvalue[0, 1], 1)
        unconditional = analyse_pairwise(sym_from_array(moves), PairwiseParams(39, 20, 10))
        self.assertGreater(unconditional.z[0, 1], 10)

    def test_permutations_recover_assistance_not_just_copying(self):
        rng = np.random.default_rng(12)
        moves = rng.integers(0, 3, (300, 3))
        moves[:, 1] = (moves[:, 0] - 1) % 3
        r = analyse_pairwise(sym_from_array(moves), PairwiseParams(99, 20, 10, 4))
        self.assertGreater(r.z[0, 1], 10)
        self.assertEqual(r.pvalue[0, 1], 0.01)
        np.testing.assert_allclose(r.z, r.z.T)

    def test_missing_pair_support_and_sparse_states_abstain(self):
        rng = np.random.default_rng(12)
        moves = rng.integers(0, 3, (30, 3))
        moves[10:, 1] = -1
        r = analyse_pairwise(sym_from_array(moves), PairwiseParams(19, 20, 2))
        self.assertTrue(np.isnan(r.z[0, 1]))
        self.assertTrue(np.isfinite(r.z[0, 2]))
        r = analyse_pairwise(sym_from_array(moves, list(range(30))), PairwiseParams(19, 20, 2))
        self.assertFalse(r.available)

    def test_pairwise_counts_cells_using_joint_validity(self):
        # A state has many total rows but only one jointly observed pair row.
        moves = np.zeros((40, 2), dtype=int)
        moves[1:20, 1] = -1
        r = analyse_pairwise(sym_from_array(moves, [0] * 20 + [1] * 20), PairwiseParams(19, 20, 10))
        self.assertEqual(r.support[0, 1], 20)

    def test_deterministic_seed(self):
        moves = np.random.default_rng(8).integers(0, 3, (100, 3))
        a = analyse_pairwise(sym_from_array(moves), PairwiseParams())
        b = analyse_pairwise(sym_from_array(moves), PairwiseParams())
        np.testing.assert_array_equal(a.z, b.z)


class TestGraphDecoder(unittest.TestCase):
    def setUp(self):
        self.z = np.zeros((6, 6))
        self.z[0, 1] = self.z[1, 0] = 12
        self.z[3, 4] = self.z[4, 3] = 10
        np.fill_diagonal(self.z, np.nan)

    def test_graphs_find_disjoint_groups(self):
        for extractor in ("components", "louvain"):
            candidates = clusters(self.z, edge_set(self.z, 3), extractor=extractor)
            self.assertEqual({c.members for c in candidates}, {frozenset([0, 1]), frozenset([3, 4])})

    def test_strict_edge_cutoff_and_missing_edges(self):
        self.assertEqual(edge_set(self.z, 12), [])
        self.z[0, 1] = self.z[1, 0] = np.nan
        self.assertEqual(edge_set(self.z, 3), [(3, 4)])

    def test_decoder_finds_unknown_coalition(self):
        z = np.zeros((6, 6))
        for i in [1, 3, 4]:
            for j in [1, 3, 4]:
                if i != j:
                    z[i, j] = 10
        np.fill_diagonal(z, np.nan)
        r = decode(z)
        self.assertEqual(r.members, frozenset([1, 3, 4]))
        self.assertEqual(r.k, 3)
        self.assertEqual(r.search, "exhaustive")

    def test_greedy_with_negative_edges_does_not_select_diagonal(self):
        z = -np.ones((6, 6))
        np.fill_diagonal(z, np.nan)
        r = ml_decode(z, 2, 1, max_exhaustive=1)
        self.assertEqual(len(r.members), 2)
        self.assertEqual(r.loglik, -1.5)
        self.assertEqual(r.search, "greedy")

    def test_validation_and_no_supported_pairs(self):
        self.assertIsNone(decode(np.full((3, 3), np.nan)))
        for k in (1, 7):
            with self.assertRaises(ValueError):
                decode(self.z, k=k)
        with self.assertRaises(ValueError):
            decode(self.z, k_max=1)
        with self.assertRaises(ValueError):
            ml_decode(self.z, 2, -1)
        with self.assertRaises(ValueError):
            decode(np.array([[np.nan, np.inf], [np.inf, np.nan]]))


class TestCalibration(unittest.TestCase):
    def test_finite_sample_order_and_ties(self):
        self.assertEqual(rank_threshold(list(range(19)), .05), 18)
        self.assertEqual(rank_threshold(list(range(20)), .05), 19)
        self.assertEqual(rank_threshold(list(range(199)), .05), 189)
        self.assertEqual(rank_threshold(list(range(18)), .05), math.inf)
        self.assertEqual(rank_threshold([None] * 20, .05), -math.inf)

    def test_threshold_gates_naming(self):
        from rps.detection.pipeline import METHODS
        raw = {m: RawResult(4, [(frozenset(["A", "B"]), 4)], False, 0) for m in METHODS}
        cal = {"thresholds": {m: 4 for m in METHODS}}
        results = decisions(raw, cal)
        self.assertTrue(results["bonjour"]["flag"])  # upstream rule is >=
        for m in METHODS[1:]:
            self.assertFalse(results[m]["flag"])
            self.assertEqual(results[m]["groups"], [])

    def test_matching_shape_and_settings_required(self):
        cfg = DetectionConfig()
        sym = sym_from_array(np.zeros((30, 3), dtype=int))
        cal = {"schema_version": 1, "frozen": True, "detector_config": asdict(cfg),
               "signature": {"n_players": 3, "rounds": 30, "state_mode": "recorded"}}
        validate_calibration(cal, sym, cfg)
        cal["signature"]["rounds"] = 300
        with self.assertRaises(ValueError):
            validate_calibration(cal, sym, cfg)

    def test_json_is_standard_and_infinities_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cal.json"
            write_json(p, {"high": math.inf, "low": -math.inf, "missing": math.nan})
            self.assertNotIn(": Infinity", p.read_text())
            result = load_calibration(p)
            self.assertEqual(result, {"high": math.inf, "low": -math.inf, "missing": None})


if __name__ == "__main__":
    unittest.main()
