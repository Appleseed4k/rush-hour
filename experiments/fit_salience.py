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

Is the extra parameter needed (`refit`)? The soft memory alone, refitted: beta x half-life over BETAS x HALF_LIVES,
without salience, the same draws, against the same targets. Salience earns its parameter only if no soft memory
matches it on the long misfit while keeping the matched misfit and the held-out bins (1.5-3 and 3-5 x optimal, used by
neither fit) as good.

    python experiments/fit_salience.py refit        # writes models/summary/salience/refit.csv, then the summary
    python experiments/fit_salience.py refit show

Both refitted together (`joint`): the refit found the soft memory's (2, 1) stale for this student, and salience was
fitted on top of it. Each soft memory of JOINT_BETAS x JOINT_HALVES, with salience lam of JOINT_LAMS (never fading,
the simplest) or without (from refit.csv). Scored by the shape of looping across attempt lengths: chi2 sums, over the
4 bins 1.5-10 x optimal and both statistics (third+ and max visits), ((model - people) / people's SE)^2, so thin bins
weigh less. Fixed in advance: the extra parameter is justified if the best model with salience lowers chi2 by more
than 3.84 (one parameter's likelihood-ratio cut) against the best without, with a matched misfit no worse than 0.01
above it.

    python experiments/fit_salience.py joint        # writes models/summary/salience/joint.csv, then the summary
    python experiments/fit_salience.py joint show
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
REFIT_OUT = os.path.join(paths.MODELS, "summary", "salience", "refit.csv")
BETAS, HALF_LIVES = (1, 2, 3, 4, 6, 8), (0.5, 1, 2, 4, 8)
REFIT = [(f"soft beta {b:g}, half-life {h:g}", (b, h)) for b in BETAS for h in HALF_LIVES]
JOINT_OUT = os.path.join(paths.MODELS, "summary", "salience", "joint.csv")
JOINT_BETAS, JOINT_HALVES, JOINT_LAMS = (1, 2, 3, 4), (1, 2, 4), (0.25, 0.5, 1)
JOINT = [(f"soft beta {b:g}, half-life {h:g}, lam {l:g}", (b, h), (l, None))
         for b in JOINT_BETAS for h in JOINT_HALVES for l in JOINT_LAMS]
SHAPE_BINS = ["1.5-3", "3-5", "5-8", "8-10"]
_AGENT = {}


def job(args):
    name, salience, instance, run, *soft = args
    soft = soft[0] if soft else fm.SOFT
    if "agent" not in _AGENT:
        torch.set_num_threads(1)
        _AGENT["agent"] = st.PolicyAgent.load(paths.START)
    rng = random.Random(zlib.crc32(f"{instance}|{run}".encode()))  # the same draws for every condition
    start, d = hd.puzzle(instance), hd.distance(instance)
    random.seed(rng.random())
    trace = []
    r = st.solve_two_mode(_AGENT["agent"], start, max(50, 10 * d), w_ao=0.18, w_flat=0.06, opening=True, rng=rng,
                          trace=trace, salience=salience, soft_memory=soft)
    boards = [start] + [b for _, b in trace]
    ids = {}
    ages = [a for a in fm.ages(boards) if a is not None]
    return (loops.counts([ids.setdefault(b, len(ids)) for b in boards])
            | {"condition": name, "instance": instance, "optimal": d, "solved": bool(r["solved"]),
               "ages": " ".join(map(str, ages))})


def run(processes=8, refit=False, joint=False):
    instances = sorted(pd.read_csv(fm.PROFILE, usecols=["source", "instance"]).query("source == 'people'").instance
                       .unique())
    if joint:
        jobs = [(name, sal, inst, r, soft) for name, soft, sal in JOINT for inst in instances for r in range(RUNS)]
    elif refit:
        jobs = [(name, None, inst, r, soft) for name, soft in REFIT for inst in instances for r in range(RUNS)]
    else:
        jobs = [(name, sal, inst, r) for name, sal in CONDITIONS for inst in instances for r in range(RUNS)]
    with Pool(processes) as pool:
        rows = list(tqdm(pool.imap(job, jobs, chunksize=4), total=len(jobs)))
    out = JOINT_OUT if joint else REFIT_OUT if refit else OUT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def loop_stats(df):
    """Per condition (or who) and attempt length: returns / move, the share of returns to a board visited twice
    already, the most visits to one board, and attempts."""
    df = df.assign(length=pd.cut(df.moves / df.optimal, LENGTHS, labels=["<1.5", "1.5-3", "3-5", "5-8", "8-10"]))
    g = df.groupby(["who", "length"], observed=True)
    return pd.DataFrame({"attempts": g.size(), "returns / move": g.returns.sum() / g.moves.sum(),
                         "third+": g["third+"].sum() / g.returns.sum(), "max visits": g["max visits"].mean()})


def summary(refit=False):
    """Per condition: the matched misfit (fit_memory's revisit ages at 1.5-3 x optimal, the soft memory's own
    target), the share of returns to a board visited twice already and the most visits by attempt length, the long
    misfit (5-10 x optimal, salience's target) and the held-out misfit (1.5-5 x optimal, neither's target). With
    `refit`, the soft-memory grid beside the soft memory as fitted and the fitted salience."""
    sims = pd.read_csv(OUT)
    conditions = [c for c, _ in CONDITIONS]
    if refit:
        keep = ["none", "lam 0.5, half-life inf"]
        sims = pd.concat([sims[sims.condition.isin(keep)], pd.read_csv(REFIT_OUT)])
        conditions = keep + [c for c, _ in REFIT]
    people = pd.read_csv(loops.OUT).query("who == 'people'")
    target, _, _ = fm.people_target()
    matched = fm.summarize(sims.assign(moves=sims.moves), target)[["misfit", "solved"]]
    stats = loop_stats(pd.concat([people, sims.rename(columns={"condition": "who"})]))
    want = stats.loc["people"]
    rows = []
    def misfit(s, bins):
        return float(sum(((s.loc[l, c] - want.loc[l, c]) / want.loc[l, c]) ** 2
                         for l in bins for c in ("third+", "max visits")))

    shown = ["1.5-3", "3-5", "5-8", "8-10"] if refit else ["3-5", "5-8", "8-10"]
    for name in conditions:
        s = stats.loc[name]
        long = ["5-8", "8-10"]
        rows.append({"condition": name, "matched misfit": matched.loc[name, "misfit"],
                     "solved": sims[sims.condition == name].solved.mean(),
                     **{f"third+ {l}": s.loc[l, "third+"] for l in shown},
                     **{f"max visits {l}": s.loc[l, "max visits"] for l in long},
                     "long misfit": misfit(s, long),
                     **({"held-out misfit": misfit(s, ["1.5-3", "3-5"])} if refit else {})})
    out = pd.DataFrame(rows).set_index("condition")
    ppl = {"matched misfit": np.nan, "solved": people.solved.mean(),
           **{f"third+ {l}": want.loc[l, "third+"] for l in shown},
           **{f"max visits {l}": want.loc[l, "max visits"] for l in ["5-8", "8-10"]}, "long misfit": 0.0,
           **({"held-out misfit": 0.0} if refit else {})}
    return pd.concat([pd.DataFrame([ppl], index=["people"]), out])


def people_se(people, reps=1000):
    """People's SE of third+ and max visits per attempt length, by resampling attempts (people's subjects aren't in
    loops.csv, so this understates it)."""
    rng = np.random.default_rng(0)
    p = people.assign(who="people")
    draws = [loop_stats(p.iloc[rng.integers(0, len(p), len(p))]).loc["people"] for _ in range(reps)]
    return pd.concat(draws).groupby(level=0)[["third+", "max visits"]].std()


def joint_summary():
    """Per soft memory x salience: parameters, matched misfit, solved, third+ and max visits by length, and chi2 of
    the shape across SHAPE_BINS (all; the short 1.5-5 and long 5-10 x optimal parts)."""
    base = [c for c, _ in REFIT if any(c == f"soft beta {b:g}, half-life {h:g}" for b in JOINT_BETAS
                                       for h in JOINT_HALVES)]
    refit = pd.read_csv(REFIT_OUT)
    sims = pd.concat([refit[refit.condition.isin(base)], pd.read_csv(JOINT_OUT)])
    conditions = base + [c for c, _, _ in JOINT]
    people = pd.read_csv(loops.OUT).query("who == 'people'")
    target, _, _ = fm.people_target()
    matched = fm.summarize(sims, target)["misfit"]
    stats = loop_stats(pd.concat([people, sims.rename(columns={"condition": "who"})]))
    want, se = stats.loc["people"], people_se(people)

    def chi2(s, bins):
        return float(sum(((s.loc[l, c] - want.loc[l, c]) / se.loc[l, c]) ** 2
                         for l in bins for c in ("third+", "max visits")))

    rows = []
    for name in conditions:
        s = stats.loc[name]
        rows.append({"condition": name, "parameters": 3 if "lam" in name else 2, "matched misfit": matched[name],
                     "solved": sims[sims.condition == name].solved.mean(),
                     **{f"third+ {l}": s.loc[l, "third+"] for l in SHAPE_BINS},
                     **{f"max visits {l}": s.loc[l, "max visits"] for l in SHAPE_BINS},
                     "chi2": chi2(s, SHAPE_BINS), "chi2 short": chi2(s, SHAPE_BINS[:2]),
                     "chi2 long": chi2(s, SHAPE_BINS[2:])})
    out = pd.DataFrame(rows).set_index("condition").sort_values("chi2")
    ppl = pd.DataFrame([{**{f"third+ {l}": want.loc[l, "third+"] for l in SHAPE_BINS},
                         **{f"max visits {l}": want.loc[l, "max visits"] for l in SHAPE_BINS}}], index=["people"])
    ses = pd.DataFrame([{**{f"third+ {l}": se.loc[l, "third+"] for l in SHAPE_BINS},
                         **{f"max visits {l}": se.loc[l, "max visits"] for l in SHAPE_BINS}}], index=["people SE"])
    return pd.concat([ppl, ses, out])


if __name__ == "__main__":
    refit, joint = "refit" in sys.argv, "joint" in sys.argv
    if "show" not in sys.argv:
        run(refit=refit, joint=joint)
    pd.set_option("display.width", 260)
    print((joint_summary() if joint else summary(refit)).round(3).to_string())
