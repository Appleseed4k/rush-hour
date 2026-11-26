"""The fitted salience (experiments/fit_salience.py) on the teaching tests: seed 0's test puzzles (9-26 moves;
untaught, no giving up, a cap of 10 x optimal), the start (8 attempts per puzzle) and the 8 end-of-session students
of twomode_m10_agency1 (1 each), as the two-mode student with and without salience (setup of
experiments/stuck_planning.py).

    python experiments/salience_test.py     # writes models/summary/salience/test.csv
"""
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solve_rate as sr  # noqa: E402
from rushhour import paths, sessions as se  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "salience", "test.csv")
VARIANTS = {"two-mode": None, "salience: lam 0.5, never fades": (0.5, None),
            "salience: lam 1, half-life 32": (1.0, 32)}
RUNS = {"start": 8, "after two-mode session": 1}


def run(processes=8):
    config = json.load(open(os.path.join(paths.SESSIONS, sr.RUNS["two-mode"], "config.json")))
    _, tests = se.sample_puzzles(config)
    solver = se.two_mode_solver(config)
    w_flat = {"start": solver.flat_at(1), "after two-mode session": solver.flat_at(sr.END // se.multiplier(config) + 1)}
    jobs = []
    for key, file in sr.networks().items():
        if key[0] not in RUNS:
            continue
        for name, salience in VARIANTS.items():
            settings = sr.play_settings(config, "two-mode", w_flat[key[0]]) | {"salience": salience}
            for i, (d, state) in enumerate(tests):
                for r in range(RUNS[key[0]]):
                    jobs.append((key, file, name, settings, state, d, se.step_cap(d, config), 1000 * i + r,
                                 config["memory"]))
    with Pool(processes) as pool:
        rows = list(tqdm(pool.imap(sr.job, jobs, chunksize=4), total=len(jobs)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).fillna({c: 0 for c in ("moves intuition", "moves flat", "moves and-or")}).to_csv(OUT, index=False)


if __name__ == "__main__":
    run()
