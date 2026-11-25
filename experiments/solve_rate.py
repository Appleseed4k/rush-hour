"""Why the two-mode student's test solve rate stays flat (p2.6, section 5): the test puzzles of the replication's
seed 0 (9-26 moves; untaught, no giving up, a cap of 10 x optimal) played by three kinds of network, each with the
two-mode student's play and with parts of it switched off.

Networks: the shared start (initial.pt), and the 8 end-of-session students (after 720 puzzles) of the no-teacher
runs twomode_m10_agency1 and intuition_m10_agency1.
Plays:
- two-mode: as the tests play it (AND-OR 18%, the opening, flat planning at its share then, soft memory);
- no AND-OR: the same without deliberation (w_ao 0, no opening);
- no flat: the same without flat planning;
- intuition, soft memory: neither (a policy sample, with the two-mode student's soft memory);
- intuition, hard memory: phase 1's play (a policy sample, never back to one of the last 10 boards if avoidable).
Each attempt records whether it was solved, its moves, the distinct boards it visited, and its moves by mode.

    python experiments/solve_rate.py           # plays (cached in models/summary/solve_rate/attempts.csv)
"""
import json
import os
import random
import sys
from multiprocessing import Pool

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import paths, sessions as se, student as st  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "solve_rate")
RUNS = {"two-mode": "twomode_m10_agency1", "intuition": "intuition_m10_agency1"}
END = 720
RUNS_PER_PUZZLE = {"start": 16}  # the one start network: as many attempts per puzzle as the 8 students'
DEFAULT_RUNS = 2
PLAYS = ["two-mode", "no AND-OR", "no flat", "intuition, soft memory", "intuition, hard memory"]
_AGENT = {}


def networks():
    first = os.path.join(paths.SESSIONS, RUNS["two-mode"])
    out = {("start", 0): os.path.join(first, "initial.pt")}
    for trained, run in RUNS.items():
        for s in range(8):
            out[(f"after {trained} session", s)] = os.path.join(paths.SESSIONS, run, "no_teacher",
                                                                 f"student{s:02d}_puzzle{END}.pt")
    return out


def play_settings(config, play, w_flat):
    base = {k: v for k, v in config["two_mode"].items() if k not in ("flat_start", "flat_end", "ramp")}
    base["w_flat"] = w_flat
    if play == "no AND-OR":
        base |= {"w_ao": 0.0, "opening": False}
    elif play == "no flat":
        base["w_flat"] = 0.0
    elif play == "intuition, soft memory":
        base |= {"w_ao": 0.0, "opening": False, "w_flat": 0.0}
    elif play == "intuition, hard memory":
        return None
    return base


def job(args):
    key, file, play, settings, state, distance, cap, seed, memory = args
    if file not in _AGENT:
        torch.set_num_threads(1)
        _AGENT.clear()
        _AGENT[file] = st.PolicyAgent.load(file)
    agent = _AGENT[file]
    if settings is None:
        r = st.solve_intuition(agent, state, cap, memory, 1.0, rng=random.Random(seed))
        boards, n, sources = len(set(r["boards"])), len(r["moves"]), {"intuition": len(r["moves"])}
    else:
        random.seed(seed)
        trace, src = [], []
        r = st.solve_two_mode(agent, state, cap, rng=random.Random(seed), trace=trace, sources=src, **settings)
        boards, n = len({state} | {b for _, b in trace}), r["moves"]
        sources = {s: src.count(s) for s in set(src)}
    return {"network": key[0], "student": key[1], "play": play, "distance": distance, "seed": seed,
            "solved": r["solved"], "moves": n, "boards": boards, **{f"moves {k}": v for k, v in sources.items()}}


def run(processes=4):
    config = json.load(open(os.path.join(paths.SESSIONS, RUNS["two-mode"], "config.json")))
    _, tests = se.sample_puzzles(config)
    solver = se.two_mode_solver(config)
    w_flat = {"start": solver.flat_at(1)} | {f"after {t} session": solver.flat_at(END // se.multiplier(config) + 1)
                                             for t in RUNS}
    jobs = []
    for key, file in networks().items():
        for play in PLAYS:
            settings = play_settings(config, play, w_flat[key[0]])
            for i, (d, state) in enumerate(tests):
                for r in range(RUNS_PER_PUZZLE.get(key[0], DEFAULT_RUNS)):
                    jobs.append((key, file, play, settings, state, d, se.step_cap(d, config), 1000 * i + r,
                                 config["memory"]))
    with Pool(processes) as pool:
        rows = list(tqdm(pool.imap(job, jobs, chunksize=4), total=len(jobs)))
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame(rows).fillna({c: 0 for c in ("moves intuition", "moves flat", "moves and-or")})
    df.to_csv(os.path.join(OUT, "attempts.csv"), index=False)
    return df


if __name__ == "__main__":
    run()
