"""Can the two-mode student plan its way out of a loop? (p2.6's follow-up: it loops on long puzzles, but people return
to earlier boards as often and still solve more.) The same setup as experiments/solve_rate.py - seed 0's test puzzles
(9-26 moves; untaught, no giving up, a cap of 10 x optimal), the start and the 8 end-of-session students of
twomode_m10_agency1 - with solve_two_mode's `stuck` rule: on a board the attempt has been on before, the modes are
drawn with other odds (more AND-OR deliberation, or more flat planning).

    python experiments/stuck_planning.py      # writes models/summary/stuck_planning/attempts.csv
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

OUT = os.path.join(paths.MODELS, "summary", "stuck_planning")
VARIANTS = {"two-mode": None,
            "stuck: AND-OR 0.5": {"w_ao": 0.5, "w_flat": 0.0},
            "stuck: AND-OR 1": {"w_ao": 1.0, "w_flat": 0.0},
            "stuck: flat 0.5": {"w_ao": 0.0, "w_flat": 0.5},
            "stuck: flat 1": {"w_ao": 0.0, "w_flat": 1.0},
            "stuck: AND-OR 0.5, flat 0.5": {"w_ao": 0.5, "w_flat": 0.5}}
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
        for name, stuck in VARIANTS.items():
            settings = sr.play_settings(config, "two-mode", w_flat[key[0]]) | {"stuck": stuck}
            for i, (d, state) in enumerate(tests):
                for r in range(RUNS[key[0]]):
                    jobs.append((key, file, name, settings, state, d, se.step_cap(d, config), 1000 * i + r,
                                 config["memory"]))
    with Pool(processes) as pool:
        rows = list(tqdm(pool.imap(sr.job, jobs, chunksize=4), total=len(jobs)))
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows).fillna({c: 0 for c in ("moves intuition", "moves flat", "moves and-or")})
    df.to_csv(os.path.join(OUT, "attempts.csv"), index=False)
    return df


if __name__ == "__main__":
    run()
