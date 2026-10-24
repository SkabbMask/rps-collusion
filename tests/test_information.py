import copy
import itertools
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from rps.benchmark.data import generated_episodes
from rps.info.bounds import decision_bounds, exact_bounds
from rps.info.evaluate import evaluate_views
from rps.info.exact import enumerate_views
from rps.info.features import project
from rps.observe.features import symbols
from rps.observe.views import full_view, metadata_view
from rps.players import OllamaPlayer, parse_policy


class TestInformationMath(unittest.TestCase):
    def test_exact_identical_and_disjoint_distributions(self):
        same = exact_bounds({0: 1}, {0: 1})
        self.assertEqual(same["tv"], 0)
        self.assertEqual(same["mi_bits"], 0)
        self.assertEqual(same["bayes_error"], 0.5)
        self.assertAlmostEqual(same["optimal_power_at_alpha"], 0.05)
        separate = exact_bounds({0: 1}, {1: 1})
        self.assertEqual(separate["tv"], 1)
        self.assertEqual(separate["mi_bits"], 1)
        self.assertEqual(separate["optimal_power_at_alpha"], 1)

    def test_confidence_limits_are_conservative_at_endpoints(self):
        weak = decision_bounds(0, 20, 0, 20)
        self.assertEqual(weak["mi_view_bits_lower_confidence"], 0)
        strong = decision_bounds(200, 200, 0, 200)
        self.assertGreater(strong["mi_view_bits_lower_confidence"], 0.7)
        self.assertLess(strong["tv_lower_confidence"], 1)
        self.assertLess(decision_bounds(200, 200, 0, 200, 0.001)["tv_lower_confidence"],
                        strong["tv_lower_confidence"])

    def test_rps_data_processing_and_zero_cp(self):
        result = enumerate_views(rounds=2, cp=0.8, m=1)["views"]
        for key in ("tv", "mi_bits", "kl_coalition_honest_bits"):
            self.assertGreaterEqual(result["identified"][key] + 1e-12, result["relabelled"][key])
            self.assertAlmostEqual(result["relabelled"][key], result["histogram"][key])
            self.assertGreaterEqual(result["histogram"][key] + 1e-12, result["summaries"][key])
            self.assertAlmostEqual(result["full"][key], result["metadata"][key])
        for value in enumerate_views(rounds=2, cp=0)["views"].values():
            self.assertAlmostEqual(value["tv"], 0)
            self.assertAlmostEqual(value["mi_bits"], 0)

    def test_relabelled_orbit_matches_explicit_random_channel(self):
        # Independent enumeration of every per-round player permutation.
        p0, p1 = defaultdict(float), defaultdict(float)
        for flat in itertools.product(range(3), repeat=4):
            rows = (flat[:2], flat[2:])
            alt = 0.0
            for a, b in ((0, 1), (1, 0)):
                alt += np.prod([(1 / 3) * (0.5 / 3 + 0.5 * (r[b] == (r[a] - 1) % 3))
                                for r in rows]) / 2
            for flip in itertools.product((False, True), repeat=2):
                key = tuple(tuple(reversed(r)) if f else r for r, f in zip(rows, flip))
                p0[key] += (1 / 81) / 4
                p1[key] += alt / 4
        brute = exact_bounds(p0, p1)
        orbit = enumerate_views(n=2, rounds=2, m=1)["views"]["relabelled"]
        for key in ("tv", "mi_bits", "optimal_power_at_alpha"):
            self.assertAlmostEqual(brute[key], orbit[key])


class TestMessageViews(unittest.TestCase):
    def setUp(self):
        self.rows = [{"round": t + 1, "actions": {"A": "rock", "B": "paper"},
                      "messages": {"A": "hello", "B": ""}} for t in range(4)]

    def test_content_and_metadata_have_different_access(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["messages"]["A"] = "world"  # same length, different content
        self.assertNotEqual(full_view(self.rows, state_mode="none").messages,
                            full_view(changed, state_mode="none").messages)
        np.testing.assert_array_equal(metadata_view(self.rows, state_mode="none").lengths,
                                      metadata_view(changed, state_mode="none").lengths)
        for view in ("identified", "metadata", "histogram", "summaries", "relabelled"):
            a, _ = project(self.rows, symbols(self.rows, state_mode="none"), view, 2, 0)
            b, _ = project(changed, symbols(changed, state_mode="none"), view, 2, 0)
            np.testing.assert_array_equal(a, b)

    def test_private_instrumentation_never_enters_projections(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            row.update(signalled=["A"], policies={"A": "secret"}, truth=[["A", "B"]],
                       prompts={"A": "private"}, payoffs={"A": 999})
        for view in ("full", "identified", "metadata", "histogram", "summaries", "relabelled"):
            a, volume_a = project(self.rows, symbols(self.rows, state_mode="none"), view, 2, 0)
            b, volume_b = project(changed, symbols(changed, state_mode="none"), view, 2, 0)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(volume_a, volume_b)

    def test_unknown_is_not_silence(self):
        self.rows[0].pop("messages")
        for constructor in (full_view, metadata_view):
            with self.assertRaisesRegex(ValueError, "observed public messages"):
                constructor(self.rows, state_mode="none")

    def test_ollama_public_message_schema_and_log(self):
        class Client:
            def chat(self, messages, schema):
                self.messages, self.schema = messages, schema
                return '{"rock":1,"paper":0,"scissors":0,"message":"hello"}'
        client = Client()
        player = OllamaPlayer("A", "honest", None, client, ["A", "B"], emit_messages=True)
        self.assertEqual(player.act([]), "rock")
        self.assertEqual(player.last_message, "hello")
        self.assertIn("message", client.schema["required"])
        self.assertIsNone(parse_policy('{"rock":1e309,"paper":0,"scissors":0}'))


class TestInformationEvaluation(unittest.TestCase):
    def episodes(self, messages=True):
        return list(generated_episodes({"seed": 51, "players": [3], "rounds": [30],
                    "collusion_probabilities": [1.0], "n_edge_fit": 5, "n_calibration": 19,
                    "n_test": 8, "stress_tests": False, "public_messages": messages}))

    def test_six_views_and_frozen_test_independence(self):
        episodes = self.episodes()
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_views(iter(episodes), Path(tmp) / "one", m=5)
            changed = copy.deepcopy(episodes)
            for ep in changed:
                if ep["split"] == "test":
                    ep["truth"] = [] if ep["truth"] else [["A", "B"]]
                    ep["scenario"] = "pair" if ep["truth"] else "honest"
            evaluate_views(iter(changed), Path(tmp) / "two", m=5)
            self.assertEqual((Path(tmp) / "one/view_calibration.json").read_text(),
                             (Path(tmp) / "two/view_calibration.json").read_text())
            self.assertEqual(len(result["comparisons"]), 6)
            self.assertFalse(result["unavailable"])
            self.assertAlmostEqual(result["comparisons"][0]["delta_per_comparison"], 0.05 / 6)

    def test_missing_messages_excludes_entire_view(self):
        episodes = self.episodes()
        episodes[-1]["records"][0].pop("messages")
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_views(iter(episodes), tmp, m=5)
            self.assertEqual({r["view"] for r in result["unavailable"]}, {"full", "metadata"})
            self.assertEqual(len(result["comparisons"]), 4)
            self.assertEqual(result["n_planned_comparisons"], 6)
            self.assertAlmostEqual(result["comparisons"][0]["delta_per_comparison"], 0.05 / 6)

    def test_leakage_rejected(self):
        for variant in ("truth", "duplicate", "seed"):
            episodes = self.episodes()
            if variant == "truth":
                episodes[0]["truth"] = [["A", "B"]]
            elif variant == "duplicate":
                episodes[-1]["records"] = episodes[0]["records"]
            else:
                episodes[-1]["seed"] = episodes[0]["seed"]
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                evaluate_views(iter(episodes), tmp, m=5)


if __name__ == "__main__":
    unittest.main()
