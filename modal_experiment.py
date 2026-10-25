"""Run an Ollama experiment on Modal; see docs/RUNNING.md."""
from pathlib import Path
import os
import re
import modal

ROOT = Path(__file__).resolve().parent
# The GPU is fixed when the function is defined, so it is chosen at launch: RPS_GPU=A100 modal run ...
GPU = os.environ.get("RPS_GPU", "L4")
app = modal.App("rps-experiments")
results = modal.Volume.from_name("rps-results", create_if_missing=True)
models = modal.Volume.from_name("rps-ollama-models", create_if_missing=True)

# Match the local Ollama version used for the initial experiments.
image = (
    modal.Image.from_registry("ollama/ollama:0.5.7", add_python="3.12")
    .entrypoint([])
    .pip_install_from_requirements(str(ROOT / "requirements-detection.txt"))
    .env({"OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_MODELS": "/models",
          "OLLAMA_NUM_PARALLEL": "3", "OLLAMA_MAX_LOADED_MODELS": "1",
          "MPLCONFIGDIR": "/tmp/matplotlib"})
)
for filename in ("experiment.py", "simulate.py", "evaluate.py", "information.py"):
    image = image.add_local_file(ROOT / filename, f"/repo/{filename}")
for directory in ("rps", "configs"):
    image = image.add_local_dir(ROOT / directory, f"/repo/{directory}",
                                ignore=["**/__pycache__/**", "**/*.pyc"])


@app.function(image=image, gpu=GPU, cpu=4, memory=16384,
              volumes={"/results": results, "/models": models},
              timeout=24 * 60 * 60, max_containers=16)
def run(config: str = "configs/qwen-pilot-t30.json", model: str = "qwen2.5:14b",
        run_name: str = "qwen14b-pilot-t30", limit: int = 0,
        max_seconds: int = 7200, collect_only: bool = False,
        analyze_only: bool = False, workers: int = 1, shard: str = "0/1"):
    """Run or resume a saved batch. max_seconds includes model startup and analysis."""
    import json
    import subprocess
    import sys
    import time
    import urllib.request

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        raise ValueError("run-name must contain only letters, digits, underscores and hyphens")
    config_path = (Path("/repo") / config).resolve()
    if not config_path.is_relative_to(Path("/repo/configs")) or not config_path.is_file():
        raise ValueError("config must be a JSON file under configs/")
    if limit < 0 or not 0 < max_seconds <= 23 * 60 * 60 or not 1 <= workers <= 32:
        raise ValueError("limit must be nonnegative; max-seconds must be 1..82800; workers 1..32")
    if collect_only and analyze_only:
        raise ValueError("choose collect-only or analyze-only")
    deadline = time.monotonic() + max_seconds

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("run time limit reached; repeat the command to resume")
        return seconds

    command = [sys.executable, "experiment.py", "--config", str(config_path),
               "--model", model, "--url", "localhost:11434",
               "--output", f"/results/{run_name}"]
    if limit:
        command += ["--limit", str(limit)]
    if workers > 1:
        command += ["--workers", str(workers)]
    if shard != "0/1":
        command += ["--shard", shard]
    if collect_only:
        command += ["--collect-only"]
    if analyze_only:
        command += ["--analyze-only"]
    server = None
    try:
        if not analyze_only:
            # One slot per concurrent player call in each of `workers` games.
            players = max(json.loads(config_path.read_text())["players"])
            server = subprocess.Popen(["ollama", "serve"], env={
                **os.environ, "OLLAMA_NUM_PARALLEL": str(players * workers)})
            for attempt in range(120):
                remaining()
                if server.poll() is not None:
                    raise RuntimeError("Ollama exited during startup")
                try:
                    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as response:
                        installed = {m["name"] for m in json.load(response)["models"]}
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise RuntimeError("Ollama did not start within two minutes")
            # Never update an already cached tag during resume; the collector pins its digest.
            if model not in installed and model + ":latest" not in installed:
                subprocess.run(["ollama", "pull", model], check=True, timeout=remaining())
                models.commit()
        subprocess.run(command, cwd="/repo", check=True, timeout=remaining())
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        results.commit()
        models.commit()
    return f"Saved to volume rps-results: /{run_name}"


@app.local_entrypoint()
def sharded(config: str = "configs/qwen-pilot-t30.json", model: str = "qwen2.5:14b",
            run_name: str = "qwen14b-pilot-t30", shards: int = 8, max_seconds: int = 7200):
    """Split one collection across `shards` containers, then merge counts and analyse in one."""
    kwargs = dict(config=config, model=model, run_name=run_name, max_seconds=max_seconds)
    # Shards write disjoint game files; only identical setup files overlap on the volume.
    calls = [run.spawn(**kwargs, collect_only=True, shard=f"{i}/{shards}") for i in range(shards)]
    for call in calls:
        print(call.get())
    # A plain resume finds every game complete, rewrites quality.json from all of them and analyses.
    print(run.remote(**kwargs))
