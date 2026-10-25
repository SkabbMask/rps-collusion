"""Exercise the launcher without installing Modal or starting paid jobs."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch


def launcher():
    modal = MagicMock()
    modal.App.return_value.function.side_effect = lambda **kwargs: lambda f: f
    spec = importlib.util.spec_from_file_location(
        "modal_launcher_test", Path(__file__).parents[1] / "modal_experiment.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modal": modal}):
        spec.loader.exec_module(module)
    return module


class TestModalLauncher(unittest.TestCase):
    def test_collection_command_and_volume_commit(self):
        module = launcher()
        response = io.BytesIO(b'{"models":[{"name":"qwen2.5:14b"}]}')
        with (patch.object(Path, "is_file", return_value=True),
              patch("subprocess.Popen") as server,
              patch("subprocess.run") as run,
              patch("urllib.request.urlopen", return_value=response)):
            server.return_value.poll.return_value = None
            module.run(limit=2, collect_only=True, max_seconds=60)
        command = run.call_args.args[0]
        self.assertIn("/results/qwen14b-pilot-t30", command)
        self.assertIn("--collect-only", command)
        self.assertEqual(command[-3:], ["--limit", "2", "--collect-only"])
        self.assertLessEqual(run.call_args.kwargs["timeout"], 60)
        server.return_value.terminate.assert_called_once()
        module.results.commit.assert_called()
        module.models.commit.assert_called()

    def test_timeout_commits_partial_results(self):
        module = launcher()
        with (patch.object(Path, "is_file", return_value=True),
              patch("subprocess.Popen") as server,
              patch("subprocess.run", side_effect=subprocess.TimeoutExpired("experiment", 1)),
              patch("urllib.request.urlopen", return_value=io.BytesIO(
                  b'{"models":[{"name":"qwen2.5:14b"}]}'))):
            server.return_value.poll.return_value = None
            with self.assertRaises(subprocess.TimeoutExpired):
                module.run(max_seconds=1)
        module.results.commit.assert_called()
        server.return_value.terminate.assert_called_once()

    def test_offline_analysis_does_not_start_ollama(self):
        module = launcher()
        with (patch.object(Path, "is_file", return_value=True),
              patch("subprocess.Popen") as server, patch("subprocess.run") as run):
            module.run(analyze_only=True)
        server.assert_not_called()
        self.assertIn("--analyze-only", run.call_args.args[0])

    def test_invalid_run_name_is_rejected_before_launch(self):
        module = launcher()
        with patch("subprocess.Popen") as server, self.assertRaises(ValueError):
            module.run(run_name="../other")
        server.assert_not_called()
