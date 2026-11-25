"""How people and the student loop on people's own layouts (p2.6, section 7): every attempt of people's on their 70
layouts (6-15 moves), and the start network (paths.START) playing each layout 10 times as the two-mode student, by
intuition with the soft memory, and by intuition with phase 1's hard memory (cap max(50, 10 x optimal), no giving up).

Per attempt: its moves, returns (moves to a board the attempt has been on before), returns to a board already visited
twice or more, returns after which the student leaves the way it left last time ("same exit"), and the most visits
to one board. People's boards are identified by chaining each move's "board age" (human_profile/moves.csv.gz).

    python experiments/loops.py      # writes models/summary/solve_rate/loops.csv
"""
import collections
import os
import random
import sys
import zlib

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import human_data as hd, paths, student as st  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "solve_rate", "loops.csv")
RUNS = 10
PLAYS = ["soft (two-mode)", "soft (intuition)", "hard (intuition)"]


def counts(ids):
    """Loop counts of one attempt, given its boards as ids (the start first)."""
    visits, exit_of = collections.Counter(), {}
    out = {"returns": 0, "third+": 0, "exits": 0, "same exit": 0}
    for p, b in enumerate(ids):
        if visits[b]:
            out["returns"] += 1
            out["third+"] += visits[b] >= 2
            if p + 1 < len(ids) and b in exit_of:
                out["exits"] += 1
                out["same exit"] += ids[p + 1] == exit_of[b]
        visits[b] += 1
        if p + 1 < len(ids):
            exit_of[b] = ids[p + 1]
    return out | {"moves": len(ids) - 1, "max visits": max(visits.values())}


def people():
    m = pd.read_csv(os.path.join(paths.MODELS, "summary", "human_profile", "moves.csv.gz"), dtype={"who": str})
    m = m[m.source == "people"].sort_values(["who", "instance", "attempt", "move"])
    rows = []
    for (_, instance, _), g in m.groupby(["who", "instance", "attempt"], sort=False):
        ids = [0]
        for age in g["board age"].values:
            ids.append(ids[len(ids) - int(age)] if not np.isnan(age) else len(ids))
        rows.append(counts(ids) | {"who": "people", "instance": instance, "optimal": g["optimal length"].iloc[0],
                                   "solved": bool(g.solved.iloc[0])})
    return rows, sorted(m.instance.unique())


def student(instances):
    agent = st.PolicyAgent.load(paths.START)
    rows = []
    for play in PLAYS:
        for instance in instances:
            start, d = hd.puzzle(instance), hd.distance(instance)
            cap = max(50, 10 * d)
            for run in range(RUNS):
                rng = random.Random(zlib.crc32(f"{play}|{instance}|{run}".encode()))
                if play.startswith("hard"):
                    r = st.solve_intuition(agent, start, cap, 10, 1.0, rng=rng)
                    boards = r["boards"]
                else:
                    random.seed(rng.random())
                    trace = []
                    kw = (dict(w_ao=0.18, w_flat=0.06, opening=True) if "two-mode" in play
                          else dict(w_ao=0.0, w_flat=0.0, opening=False))
                    r = st.solve_two_mode(agent, start, cap, rng=rng, trace=trace, **kw)
                    boards = [start] + [b for _, b in trace]
                ids = {}
                rows.append(counts([ids.setdefault(b, len(ids)) for b in boards])
                            | {"who": play, "instance": instance, "optimal": d, "solved": bool(r["solved"])})
    return rows


if __name__ == "__main__":
    torch.set_num_threads(8)
    rows, instances = people()
    rows += student(instances)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
