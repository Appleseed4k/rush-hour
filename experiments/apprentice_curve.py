"""Pretraining curves of AND-OR's apprentices, cached for notebooks/summary.ipynb.

Every saved checkpoint of each run (the network before training and after every epoch of every expert-iteration
round) plays surrender.py's test puzzles - 20 pool puzzles at each of people's lengths 6, 10, 13 and 15 - RUNS
times each with no giving up, up to 10x the puzzle's length (at least 50 moves). The attempts are written in
surrender.py's cache format (puzzle: [moves if solved, else None]), so people's giving up can be applied afterwards
at any scale (surrender.predict).

    python apprentice_curve.py            # about a minute per checkpoint; finished ones are skipped
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import paths, surrender as fs  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "apprentice_curve")
RUNS_PER_PUZZLE = 10
APPRENTICES = {  # name: folder of round{r}_epoch{e}.pt checkpoints with a manifest.json
    "protected": os.path.join(paths.MODELS, "checkpoints"),  # phase 1: gamma 0.07, protect_movers on
    "unprotected": os.path.join(paths.APPRENTICE, "gamma0.07_free"),  # gamma 0.07, protect_movers off
}


def main():
    os.makedirs(OUT, exist_ok=True)
    puzzles = fs.puzzle_set("test")
    for name, folder in APPRENTICES.items():
        for m in json.load(open(os.path.join(folder, "manifest.json"))):
            path = os.path.join(OUT, f"{name}_{m['file'][:-3]}.json")
            if os.path.exists(path):
                continue
            t = time.time()
            runs = fs.attempts(os.path.join(folder, m["file"]), puzzles, RUNS_PER_PUZZLE)
            with open(path, "w") as f:
                json.dump(runs, f)
            solved = sum(x is not None for v in runs.values() for x in v) / sum(len(v) for v in runs.values())
            print(f"{name} {m['file']}: solved {solved:.0%} ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
