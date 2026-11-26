"""Does a loop-aware bayes4 step in when the student goes round in circles? (bayes4's `loops` option: a move back onto
a board of this attempt counts as a misstep in its belief about the puzzle.) A replay without learning: the start
network (paths.START) plays every third puzzle of the replication's seed 0 session (240 puzzles of 9-26 moves, in the
session's order) with the session's giving up (people's hazard x 0.40) and cap, as the two-mode student (with and
without salience, experiments/fit_salience.py) and as phase 1's intuition student, with no teacher, bayes4 and bayes4 with loops; one teacher per sequence, kept across
its puzzles; REPS sequences each with other draws.

Per attempt: solved, gave up, moves, moves shown, the move of the first shown one, and the returns before it.

    python experiments/bayes4_loops.py      # writes models/summary/bayes4_loops/attempts.csv
"""
import json
import os
import random
import sys
from multiprocessing import Pool

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import paths, sessions as se, student as st  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "bayes4_loops", "attempts.csv")
RUN = "twomode_m10_agency1"
TEACHERS = ["none", "bayes4", "bayes4:0.45,loops=1"]
STUDENTS = ["two-mode", "two-mode, salient", "intuition"]
SALIENCE = (0.5, None)  # experiments/fit_salience.py
REPS = 3


def sequence(args):
    student, teacher_spec, rep = args
    torch.set_num_threads(1)
    config = json.load(open(os.path.join(paths.SESSIONS, RUN, "config.json")))
    session, _ = se.sample_puzzles(config)
    puzzles = session[::3]
    rule = se.surrender_rule(config, puzzles)
    agent = st.PolicyAgent.load(paths.START)
    teacher = se.make_teacher(teacher_spec)
    rows = []
    for k, (d, s0) in enumerate(puzzles):
        rng = random.Random(10_000 * rep + k)
        cap = se.step_cap(d, config)
        gives_up = rule.for_puzzle(s0, 0)
        if teacher is not None:
            teacher.start(s0, rng)
        if student.startswith("two-mode"):
            random.seed(rng.random())
            trace = []
            r = st.solve_two_mode(agent, s0, cap, w_ao=0.18, w_flat=0.06, opening=True, rng=rng, trace=trace,
                                  surrender=gives_up, teacher=teacher,
                                  salience=SALIENCE if "salient" in student else None)
            boards, moves = [s0] + [b for _, b in trace], [m for m, _ in trace]
        else:
            r = st.solve_intuition(agent, s0, cap, config["memory"], 1.0, teacher=teacher, rng=rng,
                                   surrender=gives_up)
            boards, moves = r["boards"], r["moves"]
        if teacher is not None:
            teacher.end_attempt({"moves": moves, "solved": r["solved"], "surrendered": r["surrendered"]})
        taught = list(r["taught"])
        first = next((i for i, t in enumerate(taught) if t), None)
        seen, returns = set(), []
        for b in boards[1:]:
            returns.append(b in seen or b == boards[0])
            seen.add(b)
        rows.append({"student": student, "teacher": se.teacher_name(teacher_spec), "rep": rep, "puzzle": k,
                     "distance": d, "solved": bool(r["solved"]), "gave up": bool(r["surrendered"]),
                     "moves": len(moves), "shown": sum(taught), "first shown": first,
                     "returns before": sum(returns[:first]) if first is not None else sum(returns)})
    return rows


if __name__ == "__main__":
    only = sys.argv[1:]  # students to play (default: all); their rows replace those saved
    jobs = [(s, t, rep) for s in (only or STUDENTS) for t in TEACHERS for rep in range(REPS)]
    with Pool(8) as pool:
        rows = [x for part in pool.imap_unordered(sequence, jobs) for x in part]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.DataFrame(rows)
    if only and os.path.exists(OUT):
        old = pd.read_csv(OUT)
        df = pd.concat([old[~old.student.isin(only)], df])
    df.to_csv(OUT, index=False)
