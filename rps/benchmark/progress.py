"""Wall-clock collection estimates from completed model turns."""
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
import time


def duration(seconds):
    minutes = int(max(0, seconds) // 60)
    days, minutes = divmod(minutes, 1440)
    hours, minutes = divmod(minutes, 60)
    return (f"{days}d " if days else "") + f"{hours}h {minutes:02d}m"


class GameProgress:
    def __init__(self, total_turns, interval=30):
        self.total = total_turns
        self.done = 0
        self.started = time.monotonic()
        self.interval = interval

    def report(self, current_turns=0, detail=""):
        elapsed = time.monotonic() - self.started
        turns = min(self.total, self.done + current_turns)
        remaining = elapsed * (self.total - turns) / turns if turns else None
        estimate = duration(remaining) if remaining is not None else "estimating after first round"
        print(f"  Elapsed: {duration(elapsed)} | Estimated remaining: {estimate}"
              f" | Turns: {turns:,}/{self.total:,}{detail}", flush=True)

    @contextmanager
    def watch(self, log, players, rounds):
        stop = Event()

        def heartbeat():
            while not stop.wait(self.interval):
                try:
                    # Only newline-terminated rows represent completed rounds.
                    with Path(log).open("rb") as stream:
                        completed = min(rounds, sum(row.endswith(b"\n") for row in stream))
                except FileNotFoundError:
                    completed = 0
                self.report(completed * players, f" | Round: {completed}/{rounds}")

        self.report()
        thread = Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()

    def finish(self, turns):
        self.done += turns
        self.report()
