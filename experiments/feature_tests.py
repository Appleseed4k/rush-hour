"""Are the opening rule and the flat-planning schedule needed? (MODEL.md, "Why each feature is there", group 3: fitted,
but never compared with the student without them.) Each is tested against the student without it - the simpler model
refitted where it has a value to refit - with the decision rules fixed here before the runs.

The student: spec.STUDENT's two-mode student with its memory (soft beta 1, half-life 4; salience 0.5), in sessions on
people's puzzles (experiments/planner_session.py --lean, the calibrated protocol at people's lengths), seeds 0 and 1;
its runs lean_people_all_m10_ao_b2_joint[_seed1] are the baseline.

**The flat schedule** (6% -> 18% of moves over the first 30 trials: 3 settings) against a constant flat share (1
setting) at 6%, 12% and 18%, the best of them kept (runs ..._joint_flat<6|12|18>[_seed1]). Scored on the session's
learning curve: chi2 over the 4 stages of trials (1-9, 10-27, 28-45, 46-72) of moves / optimal on solved puzzles and
the solve rate, each gap from people over people's SE (resampling people), the model the mean of its two seeds. The
schedule is needed if the best constant's chi2 is more than 5.99 above it (two parameters' likelihood-ratio cut).

**The opening** (an attempt's first move a deliberation, its second the same pass's second move) against none (runs
..._joint_noopen[_seed1]). It acts on two moves per attempt, which a session's statistics barely see, so it is also
scored where it acts: the start network plays people's 70 layouts (10 runs each, same draws) with and without it, and
its first and second moves' share optimal is compared with people's in trials 1-9 (z over the two shares' SEs, the
model's included). The opening is needed if, without it, chi2 of those first two moves is more than 3.84 above it,
or the learning curve's chi2 is.

**Both removed** (added after the two tests, rule fixed before its runs): a constant 12% and no opening (runs
..._joint_noopen_flat12[_seed1]). It may replace the student if its learning curve's chi2 is no more than 3.84 above
the student's, its snapshots' misfits are no worse than the worse of the student's two seeds, and its first two moves'
chi2 is no more than 3.84 above the student's.

    python experiments/feature_tests.py sessions      # the 8 sessions and their snapshots (skips those saved)
    python experiments/feature_tests.py first-moves   # writes models/summary/feature_tests/first_moves.csv
    python experiments/feature_tests.py show          # the summaries
"""
import json
import os
import random
import subprocess
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
from rushhour import board, human_data as hd, paths, student as st  # noqa: E402
from rushhour.student import spec  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "feature_tests")
BASE = "lean_people_all_m10_ao_b2_joint"
MEMORY = ["--soft-memory", *map(str, spec.TWO_MODE["soft_memory"]), "--salience", str(spec.TWO_MODE["salience"][0]),
          "0"]
CONSTANTS = (0.06, 0.12, 0.18)
VARIANTS = ({"noopen": ["--no-opening"]} | {f"flat{round(c * 100)}": ["--flat-start", str(c), "--flat-end", str(c)]
                                            for c in CONSTANTS}
            | {"noopen_flat12": ["--no-opening", "--flat-start", "0.12", "--flat-end", "0.12"]})
# the start network's first moves: (name, opening, flat share)
FIRST = [("with the opening", True, 0.06), ("without", False, 0.06), ("without, flat 12%", False, 0.12)]
STAGES = [(1, 9), (10, 27), (28, 45), (46, 72)]
RUNS = 10
_AGENT = {}


def run_name(variant, seed):
    return BASE + (f"_{variant}" if variant else "") + ("_seed1" if seed == 1 else "")


def sessions():
    env = os.environ | {"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"}
    py = sys.executable
    for variant, flags in VARIANTS.items():
        for seed in (0, 1):
            run = run_name(variant, seed)
            if not os.path.exists(os.path.join(paths.SESSIONS, run, "no_teacher", "sessions.json")):
                subprocess.run([py, "experiments/planner_session.py", "--lean", "--run", run, *MEMORY, *flags,
                                "--seed", str(seed)], check=True, env=env, cwd=paths.ROOT)
            if not os.path.exists(os.path.join(paths.MODELS, "summary", f"lean_snapshots_{run}.csv")):
                subprocess.run([py, "experiments/fit_memory.py", "lean-snapshots", "--run", run], check=True, env=env,
                               cwd=paths.ROOT)


# -- the learning curve

def people_curve(reps=1000):
    """People's solve rate and moves / optimal (solved) per stage, means over people, and their SEs (resampling
    people)."""
    tp = hd.trials()
    rows = []
    for lo, hi in STAGES:
        g = tp[tp.trial.between(lo, hi)]
        solved = g.groupby("subject").solved.mean()
        w = g[g.solved]
        ratio = (w.moves / w.distance).groupby(w.subject).mean()
        rng = np.random.default_rng(lo)
        bs = [(solved.iloc[i].mean(), ratio.iloc[j].mean()) for i, j in
              ((rng.integers(0, len(solved), len(solved)), rng.integers(0, len(ratio), len(ratio)))
               for _ in range(reps))]
        rows.append({"stage": f"{lo}-{hi}", "solved": solved.mean(), "ratio": ratio.mean(),
                     "solved se": np.std([b[0] for b in bs]), "ratio se": np.std([b[1] for b in bs])})
    return pd.DataFrame(rows).set_index("stage")


def session_curve(run):
    d = json.load(open(os.path.join(paths.SESSIONS, run, "no_teacher", "sessions.json")))
    recs = [e for r in d.values() for e in r]
    rows = []
    for lo, hi in STAGES:
        es = [e for e in recs if lo <= e["trial"] <= hi]
        won = [e for e in es if e["solved"]]
        rows.append({"stage": f"{lo}-{hi}", "solved": np.mean([e["solved"] for e in es]),
                     "ratio": np.mean([e["moves"] / e["distance"] for e in won])})
    return pd.DataFrame(rows).set_index("stage")


def snapshots(run):
    out, _ = fm.summarize_learning(pd.read_csv(os.path.join(paths.MODELS, "summary", f"lean_snapshots_{run}.csv")),
                                   fm.SNAP_STAGES)
    out = out[[r.snapshot == fm.SNAP_OF[r.stage] for r in out.itertuples()]]
    return out["skill misfit"].mean(), out["revisit misfit"].mean()


def curve_summary():
    """Per variant (the student as it is: 'schedule, opening'): the curve per seed, chi2 of the seeds' mean against
    people, the gain, and the snapshots' misfits."""
    people = people_curve()
    rows = []
    for variant in [None, *VARIANTS]:
        curves = [session_curve(run_name(variant, s)) for s in (0, 1)]
        mean = sum(curves) / 2
        chi2 = float((((mean.ratio - people.ratio) / people["ratio se"]) ** 2).sum()
                     + (((mean.solved - people.solved) / people["solved se"]) ** 2).sum())
        snaps = [snapshots(run_name(variant, s)) for s in (0, 1)]
        rows.append({"variant": variant or "schedule, opening", "chi2": chi2,
                     **{f"m/o {k}": v for k, v in mean.ratio.items()},
                     **{f"solved {k}": v for k, v in mean.solved.items()},
                     "gain": [c.ratio.iloc[0] - c.ratio.iloc[-1] for c in curves],
                     "skill misfit": np.mean([x[0] for x in snaps]), "revisit misfit": np.mean([x[1] for x in snaps])})
    out = pd.DataFrame(rows).set_index("variant")
    ppl = {"chi2": 0.0, **{f"m/o {k}": v for k, v in people.ratio.items()},
           **{f"solved {k}": v for k, v in people.solved.items()}, "gain": people.ratio.iloc[0] - people.ratio.iloc[-1]}
    ses = {**{f"m/o {k}": v for k, v in people["ratio se"].items()},
           **{f"solved {k}": v for k, v in people["solved se"].items()}}
    return pd.concat([pd.DataFrame([ppl, ses], index=["people", "people SE"]), out])


# -- the opening where it acts

def first_job(args):
    name, opening, w_flat, instance, r = args
    if "agent" not in _AGENT:
        torch.set_num_threads(1)
        _AGENT["agent"] = st.PolicyAgent.load(paths.START)
    rng = random.Random(zlib.crc32(f"{instance}|{r}".encode()))
    start, d = hd.puzzle(instance), hd.distance(instance)
    random.seed(rng.random())
    trace = []
    kw = {k: v for k, v in spec.TWO_MODE.items() if k in ("w_ao", "soft_memory", "salience")}
    st.solve_two_mode(_AGENT["agent"], start, max(50, 10 * d), w_flat=w_flat, opening=opening,
                      rng=rng, trace=trace, **kw)
    if instance not in _AGENT:
        _AGENT[instance] = board.multi_bfs(start)
    dist = _AGENT[instance]
    boards = [start] + [b for _, b in trace]
    return [{"student": name, "instance": instance, "run": r, "move": i, "distance": dist[b0],
             "optimal": dist[b1] < dist[b0]} for i, (b0, b1) in enumerate(zip(boards, boards[1:])) if i < 6]


def first_moves(processes=8):
    instances = sorted(pd.read_csv(fm.PROFILE, usecols=["source", "instance"]).query("source == 'people'").instance
                       .unique())
    jobs = [(*c, i, r) for c in FIRST for i in instances for r in range(RUNS)]
    with Pool(processes) as pool:
        rows = [x for part in tqdm(pool.imap(first_job, jobs, chunksize=4), total=len(jobs)) for x in part]
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "first_moves.csv"), index=False)


def first_summary():
    """Share optimal of moves 1, 2 and 3-6 of an attempt (2+ moves from the goal): people in trials 1-9, the start
    network with and without the opening; chi2 of moves 1-2 against people."""
    mv = fm._people((1, 9))
    mv = mv[mv.distance >= 2].assign(optimal=lambda d: d.after < d.distance)
    sims = pd.read_csv(os.path.join(OUT, "first_moves.csv")).query("distance >= 2")
    which = {"move 1": lambda d: d.move == 0, "move 2": lambda d: d.move == 1, "moves 3-6": lambda d: d.move.between(2, 5)}
    rows = []
    for who, d in [("people", mv)] + [(c, sims[sims.student == c]) for c, _, _ in FIRST]:
        row = {"who": who}
        for name, f in which.items():
            x = d[f(d)].optimal
            row[name], row[f"{name} se"], row[f"{name} n"] = x.mean(), x.std() / np.sqrt(len(x)), len(x)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("who")
    p = out.loc["people"]
    out["chi2 moves 1-2"] = [sum((out.loc[w, m] - p[m]) ** 2 / (out.loc[w, f"{m} se"] ** 2 + p[f"{m} se"] ** 2)
                                 for m in ("move 1", "move 2")) if w != "people" else 0.0 for w in out.index]
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    if "sessions" in sys.argv:
        sessions()
    if "first-moves" in sys.argv:
        first_moves()
    if "show" in sys.argv:
        print(curve_summary().round(3).to_string())
        print()
        print(first_summary().round(3).to_string())
