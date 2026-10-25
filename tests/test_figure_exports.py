import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rps.figures import detection, information
from rps.info.exact import enumerate_views


class TestFigureExports(unittest.TestCase):
    def test_saved_custom_model_groups_historical_methods_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"defaults": {"backend": "ollama", "model": "stub"}}))
            (root / "summary.csv").write_text("method\nbonjour\ndecoder\n")
            (root / "thresholds.csv").write_text("group,method,threshold\ncustom_model_group,decoder,inf\n")
            # A recorded manifest may omit CP; it still gets performance figures.
            (root / "comparison.csv").write_text("group,scenario,cp,method\ncustom_model_group,pair,,bonjour\n")
            with (root / "predictions.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["episode_id", "group", "scenario", "method", "positive", "flag", "exact_groups"])
                writer.writeheader()
                for m in ("bonjour", "decoder"):
                    for positive in (False, True):
                        writer.writerow({"episode_id": str(positive), "group": "custom_model_group",
                                         "scenario": "pair" if positive else "honest", "method": m,
                                         "positive": positive, "flag": positive, "exact_groups": 1})
            original = (root / "predictions.csv").read_bytes()
            detection(root)
            captions = (root / "figures/detection_captions.md").read_text()
            self.assertIn("Model experiment: stub", captions)
            self.assertIn("1 coalition and 1 honest", captions)
            self.assertIn("Tiny sample", captions)
            self.assertIn("Infinite cutoffs", captions)
            svg = root / "figures/paper_performance.svg"
            self.assertIn("Model experiment: stub", svg.read_text())
            before = svg.stat().st_mtime_ns
            detection(root)
            self.assertEqual(before, svg.stat().st_mtime_ns)
            svg.unlink()  # Missing exports are repaired even with unchanged inputs.
            detection(root)
            self.assertTrue(svg.exists())
            self.assertEqual(original, (root / "predictions.csv").read_bytes())
            manifest = json.loads((root / "figures/detection_manifest.json").read_text())
            self.assertEqual(manifest["inputs_sha256"]["predictions.csv"], hashlib.sha256(original).hexdigest())

    def test_exact_and_empirical_figures_remain_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "exact_views.json").write_text(json.dumps(enumerate_views(rounds=1)))
            (root / "config.json").write_text(json.dumps({"defaults": {"backend": "ollama", "model": "stub"}}))
            (root / "information.json").write_text(json.dumps({"comparisons": [
                {"group": "custom", "scenario": "pair", "cp": 1, "view": "identified",
                 "mi_view_bits_lower_confidence": 0}]}))
            information(root)
            self.assertFalse((root / "figures/paper_exact_views.svg").exists())
            self.assertNotIn("Exact synthetic model", (root / "figures/information_captions.md").read_text())
            information(root, include_exact=True)
            captions = (root / "figures/information_captions.md").read_text()
            self.assertIn("Exact synthetic model", captions)
            self.assertIn("not Qwen information estimates", captions)
            self.assertIn("Zero does not establish no information", captions)
            self.assertIn("Model experiment: stub", captions)
            for stem in ("paper_exact_views", "paper_information_bounds_01"):
                for ext in ("png", "svg", "pdf"):
                    self.assertGreater((root / f"figures/{stem}.{ext}").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
