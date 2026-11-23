"""The session student trained on pool puzzles, judged on people's layouts.

The start: round 2 epoch 5 of AND-OR's apprentice played by intuition, which on people's own 70 layouts matches their
first attempts (planner_session.py --fit-strong; giving up at 0.35 x people's rates) and their skill by distance in
trials 1-9. The session's puzzles come from the pool and need not resemble people's layouts: on pool puzzles of the
same lengths a network is 8-15 points better 7+ moves from the goal than on people's layouts, so the pool is only what
the student learns from. Learning is measured by snapshots (after 18, 36 and 54 of people's trials) playing people's
layouts by intuition, each against the stage of people's session it sits in the middle of (fit_memory.
summarize_learning: the chance of an optimal move by distance to the goal and moves / optimal, with the attempts
weighted by how often people at that stage played each layout).

The grid: what the session trains on (people's lengths; harder, as the teaching runs' bins; harder still) x the
multiplier (simulated puzzles per human trial, the learning-rate parameter phase 1 fitted on pool puzzles), the rest
as the fitted student (spec.STUDENT, full hindsight). Tests on pool puzzles are kept small: they are not the measure.

    python experiments/session_fit.py run             # plays the sessions (models/sessions/r2e5_<bins>_m<m>)
    python experiments/session_fit.py fit             # snapshots on people's layouts: models/summary/session_fit.csv
    python experiments/session_fit.py pool            # why: start networks by intuition on the pool's test puzzles vs
                                                      # people's layouts (models/summary/pool_shape_profile.csv)
    python experiments/session_fit.py smoke           # a tiny session and fit, to check the pipeline
"""
import multiprocessing as mp
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import fit_memory as fm
import forward
from rushhour import human_data as hd, paths, surrender as sr

INIT = os.path.join(paths.APPRENTICE, "gamma0.07_free", "round2_epoch05.pt")
SCALE = 0.35  # giving up, fitted with INIT against people's first attempts
BINS = {"people": ["6-6", "10-10", "13-13", "15-15"],
        "harder": ["9-11", "14-16", "19-21", "24-26"],
        "hardest": ["14-16", "19-21", "24-26", "29-31"]}
MULTIPLIERS = (5, 10, 20)
TRIALS = fm.SNAP_TRIALS  # snapshots after this many human trials (18 per bin x 4 = 72 in all)
RUNS = 2  # attempts per snapshot and layout (the start: x 8, it stands for all 8 students)


def name(bins, m):
    return f"r2e5_{bins}_m{m}"


def session_args(bins, m, run=None, smoke=False):
    args = [sys.executable, "-m", "rushhour.sessions", run or name(bins, m), "--fitted", "--init", INIT,
            "--surrender-scale", str(SCALE), "--teachers", "none", "--session-bins", *BINS[bins],
            "--test-bins", *BINS[bins], "--test-per-bin", "5", "--test-runs", "1",
            "--multiplier", str(m), "--block", str(m)]
    if smoke:
        args += ["--students", "2", "--per-bin", "2", "--test-every", "2", "--overwrite"]
    return args


def networks(run, multiplier, trials=TRIALS):
    """{"0": the run's initial network, "<trials> studentNN": each student's snapshot after that many trials}."""
    folder = os.path.join(paths.SESSIONS, run)
    nets = {"0": os.path.join(folder, "initial.pt")}
    for n in trials:
        for s in range(32):
            f = os.path.join(folder, "no_teacher", f"student{s:02d}_puzzle{multiplier * n:03d}.pt")
            if os.path.exists(f):
                nets[f"{n} student{s:02d}"] = f
    return nets


def evaluate(run, multiplier, instances, trials=TRIALS, runs=RUNS, start=True):
    """Each snapshot (and, with `start`, the initial network) playing people's layouts by intuition: job rows."""
    nets = networks(run, multiplier, trials)
    weights = {}
    for k, f in nets.items():
        with open(f, "rb") as fh:
            weights[k] = fh.read()
    jobs = []
    for n in ((0,) if start else ()) + tuple(trials):
        c = {"name": f"after {n} trials", "snapshot": n, "kind": "intuition", "kw": {"k": 1}}
        names = ["0"] if n == 0 else [k for k in nets if k.split()[0] == str(n)]
        for i in instances:
            for k in names:
                for r in range(runs * (8 if n == 0 else 1)):
                    jobs.append((c, i, f"{k}|{r}", k))
    random.Random(0).shuffle(jobs)
    with mp.Pool(initializer=forward._init, initargs=(weights,)) as pool:
        return pd.DataFrame(pool.map(fm.job_snapshot, jobs, chunksize=4))


POOL_NETS = ["start", "round 2 epoch 2", "round 2 epoch 5"]
_POOL = {}


def _pool_init(weights):
    """forward._init, with fit_memory.job reading boards and distances from the pool's test puzzles."""
    import types
    forward._init(weights)
    _POOL.update(sr.puzzle_set("test"))
    fm.hd = types.SimpleNamespace(puzzle=lambda n: _POOL[n][1], distance=lambda n: _POOL[n][0])


def profile(df):
    """Unweighted over attempts: the chance a move is optimal per distance bin, moves / optimal (solved), solved."""
    rows = []
    for c, d in df.groupby("condition", sort=False):
        s_ = d[d.solved]
        rows.append({"condition": c, **{b: d[f"opt {b}"].sum() / d[f"n {b}"].sum() for b in fm.DBINS},
                     "moves / optimal": (s_.moves / s_.optimal).mean(), "solved": d.solved.mean()})
    return pd.DataFrame(rows)


def pool_profile(runs=8):
    """Start networks by intuition on the pool's 80 test puzzles (people's lengths, none of their boards), against
    the same networks on people's layouts (strong_shape_profile.csv, planner_session.py --fit-strong's profile)."""
    import planner_session as ps
    weights = {}
    for n in POOL_NETS:
        with open(ps.NETS[n], "rb") as f:
            weights[n] = f.read()
    conds = [{"name": f"intuition, {n}", "kind": "intuition", "snapshot": 0, "net": n, "kw": {"k": 1}}
             for n in POOL_NETS]
    jobs = [(c, i, f"{r}", c["net"]) for c in conds for i in sr.puzzle_set("test") for r in range(runs)]
    random.Random(0).shuffle(jobs)
    with mp.Pool(initializer=_pool_init, initargs=(weights,)) as pool:
        df = pd.DataFrame(pool.map(fm.job_snapshot, jobs, chunksize=4))
    df.to_csv(os.path.join(paths.MODELS, "summary", "pool_shape_profile.csv"), index=False)
    human = pd.read_csv(os.path.join(paths.MODELS, "summary", "strong_shape_profile.csv"))
    human = human[human.condition.isin(df.condition.unique())]
    return pd.concat([profile(df).assign(puzzles="pool"), profile(human).assign(puzzles="people's")])


def paired(df):
    """summarize_learning per stage, each stage against its own snapshot (the start for trials 1-9)."""
    out, people = fm.summarize_learning(df, fm.SNAP_STAGES)
    return out[[r.snapshot == fm.SNAP_OF[r.stage] for r in out.itertuples()]], people


COLS = ["stage", "condition", "optimal 2-3", "optimal 4-6", "optimal 7-9", "optimal 10+", "moves / optimal",
        "skill misfit", "solved"]

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "fit"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    instances = sorted(hd.trials().instance.unique())
    if what == "smoke":
        subprocess.run(session_args("harder", 2, run="r2e5_smoke", smoke=True), cwd=root, check=True)
        df = evaluate("r2e5_smoke", 2, instances[:6], trials=(2, 4), runs=1)
        print(df.groupby("snapshot")[["moves", "solved"]].mean())
        sys.exit()
    if what == "pool":
        print(pool_profile().sort_values(["condition", "puzzles"]).round(3).to_string(index=False))
        sys.exit()
    if what == "run":
        for m in MULTIPLIERS:
            for bins in BINS:
                print(name(bins, m), flush=True)
                subprocess.run(session_args(bins, m), cwd=root, check=True)
        sys.exit()
    if what == "fit":
        frames, start = [], None
        for m in MULTIPLIERS:
            for bins in BINS:
                run = name(bins, m)
                if not os.path.exists(os.path.join(paths.SESSIONS, run, "no_teacher", "sessions.json")):
                    print("not played yet:", run)
                    continue
                df = evaluate(run, m, instances, start=start is None)
                if start is None:  # every run starts from the same network: play it once
                    start = df[df.snapshot == 0]
                    df = df[df.snapshot != 0]
                frames.append(pd.concat([start, df]).assign(run=run, bins=bins, multiplier=m))
                print(run, "evaluated", flush=True)
        everything = pd.concat(frames)
        everything.to_csv(os.path.join(paths.MODELS, "summary", "session_fit.csv"), index=False)
        rows = []
        for run, d in everything.groupby("run", sort=False):
            out, people = paired(d)
            rows.append(out.assign(run=run))
        table = pd.concat(rows)
        pd.set_option("display.width", 220)
        print(people[["optimal 2-3", "optimal 4-6", "optimal 7-9", "optimal 10+", "moves / optimal"]].round(3)
              .to_string())
        print(table[["run"] + COLS].round(3).to_string(index=False))
        print(table.groupby("run", sort=False)["skill misfit"].sum().round(3).sort_values().to_string())
