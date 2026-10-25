import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from experiment import analyze
from rps.benchmark.collect import collect, game_config, plan, source_hashes
from rps.benchmark.data import manifest_episodes


def settings():
    return {"seed": 20261025, "players": [3], "rounds": [20],
            "collusion_probabilities": [1.0], "scenarios": {"pair": [1.0]},
            "n_edge_fit": 3, "n_calibration": 19, "n_test": 2, "alpha": .05,
            "relabel_interval": 5, "state": "none", "detector": {"permutations": 5},
            "defaults": {"backend": "ollama", "model": "stub", "url": "http://localhost:11434",
                         "strategy": "mixed", "emit_messages": True}}


class TestModelExperiment(unittest.TestCase):
    def setUp(self):
        self.identity = patch("rps.benchmark.collect.model_identity", return_value={"name": "stub", "digest": "v1"})
        self.check = patch("rps.ollama.OllamaClient.check")
        # Constant output deliberately creates legitimate repeated observations.
        self.chat = patch("rps.ollama.OllamaClient.chat", return_value=
                          '{"rock":1,"paper":0,"scissors":0,"message":"hello"}')
        self.identity.start()
        self.check.start()
        self.calls = self.chat.start()
        self.addCleanup(self.identity.stop)
        self.addCleanup(self.check.stop)
        self.addCleanup(self.chat.stop)

    def test_documented_t30_plan_sizes(self):
        for name, count in (("explore", 69), ("pilot", 119), ("comparison", 439)):
            cfg = json.loads((Path(__file__).parents[1] / f"configs/qwen-{name}-t30.json").read_text())
            episodes = list(plan(cfg))
            self.assertEqual(len(episodes), count)
            self.assertEqual(sum(ep["n"] * ep["t"] for ep in episodes), count * 90)
            self.assertGreaterEqual(cfg["n_edge_fit"], 19)
            self.assertGreaterEqual(cfg["n_calibration"], 19)

    def test_full_plan_sizes_and_roles(self):
        cfg = json.loads((Path(__file__).parents[1] / "configs/qwen-experiment.json").read_text())
        episodes = list(plan(cfg))
        self.assertEqual(len(episodes), 4316)
        self.assertEqual(sum(e["n"] * e["t"] for e in episodes), 4244400)
        self.assertEqual(len({e["seed"] for e in episodes}), len(episodes))
        for e in (episodes[0], next(e for e in episodes if e["scenario"] == "star3"),
                  next(e for e in episodes if e["scenario"] == "two_pairs")):
            game, truth = game_config(cfg, e)
            self.assertEqual(game["defaults"]["backend"], "ollama")
            expected = 0 if e["scenario"] == "honest" else 2
            self.assertEqual(sum(p["role"] == "assistant" for p in game["players"]), expected)
            if not expected:
                self.assertEqual(truth, [])

    def test_resume_skips_completed_games_and_preserves_seeds(self):
        cfg = settings()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            self.assertFalse(collect(cfg, folder, limit=2))
            first_log = next((Path(folder) / "logs").rglob("*.jsonl"))
            original = first_log.read_bytes()
            self.assertTrue(collect(cfg, folder))
            self.assertEqual(first_log.read_bytes(), original)
            n = len(list(plan(cfg)))
            self.assertEqual(self.calls.call_count, n * 3 * 20)
            imported = list(manifest_episodes(Path(folder) / "manifest.json"))
            self.assertEqual({e["seed"] for e in imported}, {e["seed"] for e in plan(cfg)})
            metadata = [json.loads(p.read_text()) for p in (Path(folder) / "logs").rglob("*.meta.json")]
            self.assertTrue(any(m["signals"] > 0 and m["signals_followed"] == 0 for m in metadata))
            self.assertEqual(len(metadata), n)

    def test_resume_before_progress_update(self):
        cfg = settings()
        previous = source_hashes()
        previous["rps/benchmark/collect.py"] = "20e3f4081c91773edb6d9835d19d79cb7644f197d71804fe233b851951bde409"
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            with patch("rps.benchmark.collect.source_hashes", return_value=previous):
                collect(cfg, folder, limit=1)
            collect(cfg, folder, limit=1)
            self.assertEqual(json.loads((Path(folder) / "progress.json").read_text())["complete"], 2)

    def test_changed_settings_or_weights_are_rejected(self):
        cfg = settings()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            collect(cfg, folder, limit=1)
            changed = copy.deepcopy(cfg)
            changed["defaults"]["model"] = "another-model"
            with self.assertRaisesRegex(ValueError, "settings or source changed"):
                collect(changed, folder)
            with patch("rps.benchmark.collect.model_identity", return_value={"name": "stub", "digest": "v2"}):
                with self.assertRaisesRegex(ValueError, "weights changed"):
                    collect(cfg, folder)

    def test_modified_completed_log_is_rejected(self):
        cfg = settings()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            collect(cfg, folder, limit=1)
            log = next((Path(folder) / "logs").rglob("*.jsonl"))
            log.write_text(log.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "completed game changed"):
                collect(cfg, folder)

    def test_interrupted_game_is_restarted_and_attempt_is_kept(self):
        cfg = settings()
        def fail(game):
            path = Path(game["log"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"round":1}\n')
            raise RuntimeError("connection lost")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            with patch("rps.benchmark.collect.run", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "connection lost"):
                    collect(cfg, folder, limit=1)
            collect(cfg, folder, limit=1)
            self.assertEqual(len(list((Path(folder) / "attempts").rglob("*.jsonl"))), 1)
            self.assertEqual(json.loads((Path(folder) / "progress.json").read_text())["complete"], 1)

    def test_all_stages_with_stub_model_and_repeated_observations(self):
        cfg = settings()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            collect(cfg, folder, limit=1)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyze(folder, cfg)
            self.assertTrue(collect(cfg, folder))
            analyze(folder, cfg)
            root = Path(folder)
            self.assertTrue((root / "detection/report.md").exists())
            self.assertTrue((root / "information/report.md").exists())
            self.assertTrue((root / "report.md").exists())
            self.assertTrue((root / "figures/paper_coordination_case_01.png").exists())
            self.assertTrue((root / "detection/figures/paper_performance.pdf").exists())
            self.assertFalse((root / "information/exact_views.json").exists())
            self.assertFalse((root / "information/figures/paper_exact_views.svg").exists())
            self.assertTrue((root / "information/figures/paper_information_bounds_01.svg").exists())
            self.assertIn("Model experiment: stub", (root / "figures/collection_captions.md").read_text())
            self.assertIn("not a performance estimate", (root / "figures/collection_captions.md").read_text())
            provenance = json.loads((root / "detection/provenance.json").read_text())
            self.assertGreater(provenance["repeated_observations"], 0)
            info = json.loads((root / "information/information.json").read_text())
            self.assertEqual(len(info["comparisons"]), 6)
            self.assertFalse(info["unavailable"])
            # Finished analysis can be reused without running either stage again.
            with patch("experiment.subprocess.run") as subprocess:
                analyze(folder, cfg)
                subprocess.assert_not_called()
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["episodes"][0]["truth"] = [["A", "B"]]
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "labels changed"):
                analyze(folder, cfg)

    def test_parse_failures_are_recorded_without_resampling_games(self):
        cfg = settings()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            with patch("rps.ollama.OllamaClient.chat", return_value="invalid") as chat:
                self.assertFalse(collect(cfg, folder, limit=1))
            self.assertEqual(chat.call_count, 20 * 3 * 3)  # Three attempts per played move.
            quality = json.loads((Path(folder) / "quality.json").read_text())
            self.assertEqual(quality["games"], 1)
            self.assertEqual(quality["parse_failures"], 60)

    def test_invalid_batch_config_does_not_contact_server(self):
        cfg = settings()
        cfg["defaults"]["backend"] = "random"
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            collect(cfg, folder)
        self.assertEqual(self.calls.call_count, 0)


class TestProgress(unittest.TestCase):
    def test_estimate_uses_turns_and_elapsed_time(self):
        from rps.benchmark.progress import GameProgress, duration
        output = io.StringIO()
        with patch("rps.benchmark.progress.time.monotonic", side_effect=[0, 0, 120, 150]), redirect_stdout(output):
            progress = GameProgress(900)
            progress.report()
            progress.report(300)
            progress.done = 300
            progress.report(150)
        self.assertIn("estimating after first round", output.getvalue())
        self.assertIn("Estimated remaining: 0h 04m", output.getvalue())
        self.assertIn("Estimated remaining: 0h 02m | Turns: 450/900", output.getvalue())
        self.assertEqual(duration(90061), "1d 1h 01m")

    def test_heartbeat_reads_completed_rounds_and_stops(self):
        from rps.benchmark.progress import GameProgress
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "game.jsonl"
            log.write_bytes(b'{}\n{}\n{"partial":')
            progress = GameProgress(300, interval=.001)
            from threading import Event
            reported = Event()
            with patch.object(progress, "report", side_effect=lambda *args: reported.set() if args else None) as report:
                with progress.watch(log, 3, 100):
                    self.assertTrue(reported.wait(2))
                report.assert_any_call(6, " | Round: 2/100")


if __name__ == "__main__":
    unittest.main()
