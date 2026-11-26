"""Salience of repeated returns (solve_two_mode's `salience`), fitted to how people repeat boards on their own layouts.

The soft memory (beta 2, half-life 1; experiments/fit_memory.py) matches people's returns to earlier boards up to about
8 x optimal, but in longer attempts the student keeps going round the same boards: on people's layouts it returns to
a board it has already visited twice 54% of the time at 8-10 x optimal, people 37% (experiments/loops.py). Salience
adds to the soft memory a trace `lam` per return to a board, fading by half every `half_life` moves (or never), that
adds up over returns - a board's first return is penalized as before.

The start network (paths.START) plays people's 70 layouts as the two-mode student (AND-OR 18%, flat planning at its
start share, the opening), RUNS times each, cap max(50, 10 x optimal), no giving up, for each (lam, half_life) and
without salience. Per attempt: loops.counts (returns, returns to a board already visited twice, same exits, most
visits) and the returns' ages (for fit_memory's misfit on matched attempts).

    python experiments/fit_salience.py         # writes models/summary/salience/fit.csv, then prints the summary
    python experiments/fit_salience.py show    # the summary only
"""
import os
import random
import sys
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fit_memory as fm  # noqa: E402
import loops  # noqa: E402
from rushhour import human_data as hd, paths, student as st  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "salience", "fit.csv")
RUNS = 10
LAMS, HALVES = (0.5, 1, 2, 4), (8, 32, None)
CONDITIONS = [("none", None)] + [(f"lam {l:g}, half-life {h if h else 'inf'}", (l, h)) for l in LAMS for h in HALVES]
LENGTHS = [0, 1.5, 3, 5, 8, 10.01]
_AGENT = {}


def job(args):
    name, salience, instance, run = args
    if "agent" not in _AGENT:
        torch.set_num_threads(1)
        _AGENT["agent"] = st.PolicyAgent.load(paths.START)
    rng = random.Random(zlib.crc32(f"{instance}|{run}".encode()))  # the same draws for every condition
    start, d = hd.puzzle(instance), hd.distance(instance)
    random.seed(rng.random())
    trace = []
    r = st.solve_two_mode(_AGENT["agent"], start, max(50, 10 * d), w_ao=0.18, w_flat=0.06, opening=True, rng=rng,
                          trace=trace, salience=salience)
    boards = [start] + [b for _, b in trace]
    ids = {}
    ages = [a for a in fm.ages(boards) if a is not None]
    return (loops.counts([ids.setdefault(b, len(ids)) for b in boards])
            | {"condition": name, "instance": instance, "optimal": d, "solved": bool(r["solved"]),
               "ages": " ".join(map(str, ages))})


def run(processes=8):
    instances = sorted(pd.read_csv(fm.PROFILE, usecols=["source", "instance"]).query("source == 'people'").instance
                       .unique())
    jobs = [(name, sal, inst, r) for name, sal in CONDITIONS for inst in instances for r in range(RUNS)]
    with Pool(processes) as pool:
        rows = list(tqdm(pool.imap(job, jobs, chunksize=4), total=len(jobs)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)


def loop_stats(df):
    """Per condition (or who) and attempt length: returns / move, the share of returns to a board visited twice
    already, the most visits to one board, and attempts."""
    df = df.assign(length=pd.cut(df.moves / df.optimal, LENGTHS, labels=["<1.5", "1.5-3", "3-5", "5-8", "8-10"]))
    g = df.groupby(["who", "length"], observed=True)
    return pd.DataFrame({"attempts": g.size(), "returns / move": g.returns.sum() / g.moves.sum(),
                         "third+": g["third+"].sum() / g.returns.sum(), "max visits": g["max visits"].mean()})


def summary():
    sims = pd.read_csv(OUT)
    people = pd.read_csv(loops.OUT).query("who == 'people'")
    target, _, _ = fm.people_target()
    matched = fm.summarize(sims.assign(moves=sims.moves), target)[["misfit", "solved"]]
    stats = loop_stats(pd.concat([people, sims.rename(columns={"condition": "who"})]))
    want = stats.loc["people"]
    rows = []
    for name, _ in CONDITIONS:
        s = stats.loc[name]
        long = ["5-8", "8-10"]
        rows.append({"condition": name, "matched misfit": matched.loc[name, "misfit"],
                     "solved": sims[sims.condition == name].solved.mean(),
                     **{f"third+ {l}": s.loc[l, "third+"] for l in ["3-5", "5-8", "8-10"]},
                     **{f"max visits {l}": s.loc[l, "max visits"] for l in long},
                     "long misfit": float(sum(((s.loc[l, c] - want.loc[l, c]) / want.loc[l, c]) ** 2
                                              for l in long for c in ("third+", "max visits")))})
    out = pd.DataFrame(rows).set_index("condition")
    ppl = {"matched misfit": np.nan, "solved": people.solved.mean(),
           **{f"third+ {l}": want.loc[l, "third+"] for l in ["3-5", "5-8", "8-10"]},
           **{f"max visits {l}": want.loc[l, "max visits"] for l in ["5-8", "8-10"]}, "long misfit": 0.0}
    return pd.concat([pd.DataFrame([ppl], index=["people"]), out])


if __name__ == "__main__":
    if "show" not in sys.argv:
        run()
    pd.set_option("display.width", 220)
    print(summary().round(3).to_string())
