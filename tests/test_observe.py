import copy
import unittest

import numpy as np

from rps.observe.features import complete_records, symbols
from rps.observe.views import histogram_view, relabelled_view, summary_view


class TestObservations(unittest.TestCase):
    def setUp(self):
        self.rows = [{"round": 1, "state": "start", "actions": {"B": "paper", "A": "rock", "C": "scissors"}},
                     {"round": 2, "state": ["rock", "paper", "scissors"], "actions": {"C": "rock", "B": "scissors", "A": "paper"}}]

    def test_categorical_actions_order_and_first_round(self):
        sym = symbols(self.rows)
        self.assertEqual(sym.names, ("A", "B", "C"))
        np.testing.assert_array_equal(sym.moves, [[0, 1, 2], [1, 2, 0]])
        self.assertTrue(sym.move_defined.all())
        self.assertEqual(len(sym.state_labels), 2)

    def test_upstream_tuple_input(self):
        sym = symbols([(r["state"], r["actions"]) for r in self.rows])
        self.assertEqual(len(complete_records(sym)), 2)

    def test_last_round_uses_past_not_current_action(self):
        before = symbols(self.rows, state_mode="last_round")
        altered = copy.deepcopy(self.rows)
        altered[1]["actions"]["A"] = "scissors"
        after = symbols(altered, state_mode="last_round")
        self.assertEqual(before.state_labels, after.state_labels)

    def test_hidden_fields_do_not_leak(self):
        before = symbols(self.rows)
        altered = copy.deepcopy(self.rows)
        for row in altered:
            row.update(truth=["A", "B"], signalled=["B"], roles={"B": "assistant"},
                       payoffs={"A": 10000}, policies={"B": {"rock": 1}})
        after = symbols(altered)
        np.testing.assert_array_equal(before.moves, after.moves)
        np.testing.assert_array_equal(before.state, after.state)

    def test_missing_actions_masked_and_invalid_rejected(self):
        self.rows[1]["actions"]["A"] = None
        sym = symbols(self.rows)
        self.assertEqual(sym.moves[1, 0], -1)
        self.assertFalse(sym.move_defined[1, 0])
        self.assertEqual(len(complete_records(sym)), 1)
        self.rows[1]["actions"]["A"] = "lizard"
        with self.assertRaises(ValueError):
            symbols(self.rows)

    def test_arbitrary_public_state_cardinality(self):
        rows = [{"state": f"public{i}", "actions": {"A": "rock", "B": "paper"}} for i in range(37)]
        self.assertEqual(len(symbols(rows).state_labels), 37)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            symbols([])
        with self.assertRaises(ValueError):
            symbols(self.rows, names=["A", "B"])
        with self.assertRaises(ValueError):
            symbols(self.rows, names=["A", "B", "C", "A"])
        with self.assertRaises(ValueError):
            symbols(list(reversed(self.rows)))
        with self.assertRaises(ValueError):
            symbols([{"actions": {"A": "rock", "B": "paper"}}])

    def test_histogram_is_identity_free(self):
        sym = symbols(self.rows, state_mode="none")
        np.testing.assert_array_equal(histogram_view(sym).counts, [[1, 1, 1], [1, 1, 1]])
        summaries = summary_view(sym)
        np.testing.assert_allclose(summaries.move_entropy_bits, np.log2(3))
        self.assertTrue(summaries.all_three.all())

    def test_relabelled_blocks_cannot_be_stitched(self):
        sym = symbols(self.rows, state_mode="none")
        view, hidden = relabelled_view(sym, m=1, seed=12)
        self.assertFalse(set(view.blocks[0].sym.names) & set(view.blocks[1].sym.names))
        for block, mapping in zip(view.blocks, hidden):
            for column, label in enumerate(block.sym.names):
                original = sym.names.index(mapping[label])
                np.testing.assert_array_equal(block.sym.moves[:, column], sym.moves[block.start:block.stop, original])
            self.assertTrue(block.sym.move_defined.all())
        with self.assertRaises(ValueError):
            relabelled_view(symbols(self.rows), 1)
        with self.assertRaises(ValueError):
            relabelled_view(sym, 0)


if __name__ == "__main__":
    unittest.main()
