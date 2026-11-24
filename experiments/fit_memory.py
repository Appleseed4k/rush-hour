"""Fits the student's memory of where it has been to people's revisits.

The session student never moves back to one of its last 10 boards (a hard rule, MODEL.md). People do: on solved
attempts of 1.5-3 x optimal (so efficiency doesn't drive it), 5.6% of their moves undo the last one and about 4%
return to a board seen 3 or more moves earlier (notebooks/p2.3_human_likeness.ipynb). The candidate replacement is a soft,
fading memory: a move to a board the attempt last had `age` moves ago (2 = an undo) has its logit lowered by
beta x 0.5^(age / half_life) (forward.solve_forward's soft_memory).

The fit: the start network playing by intuition (the session student) on the 70 human layouts, RUNS runs each, cap
max(50, 10 x optimal), over a grid of (beta, half_life), plus the hard rule and no memory. Each condition's share of
moves that return to a board by age (2, 3-4, 5-8, 9-16, 17+), on the same matched attempts, against people's; misfit
= sum over age bins of (model - people)^2 / people.

The planner (`planner`): the forward planner has a second memory, novelty in its rollout score (lam x the car
placements a rollout reaches that the attempt hasn't had recently, fading with novelty_half_life), which steers it
away from earlier boards and placements whatever the board memory allows. With the board memory fixed at the fit
above (beta 2, half-life 1), a grid of lam and the novelty half-life is fitted to people's returns to a board AND
of the moved car to a placement it had before, by age (the misfit sums both).

    python experiments/fit_memory.py           # writes models/summary/memory_fit.csv (one row per attempt)
Shared lapse rate (`gamma`): the planner that mixes 4 policy rollouts with 4 AND-OR passes (judged by the value head,
the soft board memory) over a grid of the rollouts' lapse rate (each simulated move past the first continues with
probability 1 - gamma, at most 20) crossed with AND-OR's. The diagonal is one shared gamma. Fitted to people's skill
(chance of an optimal move by distance to the goal; moves / optimal on solved attempts) and revisits (as above).

Learning over the session (`learning`): van Opheusden et al. fit their model to successive blocks of play and found
expertise as deeper search. Here the same mixed planner runs over a finer grid of one shared gamma (rollouts capped at
40 moves), and each stage of people's session (trials 1-10, 11-20, 21-35, 36+) gets its own best gamma: the model's
attempts are weighted by how often people at that stage played each layout, and the misfit is the gamma fit's (skill
+ revisits) against that stage's people.

    python experiments/fit_memory.py planner   # writes models/summary/memory_fit_planner.csv
    python experiments/fit_memory.py gamma     # writes models/summary/gamma_fit.csv
Judgement or depth (`snapshots`): the start network can't reach later people's mid-puzzle skill at any gamma. The
session student's own learning changes its judgement: the calibrated session (ao_people_all_m10, 8 students, x10,
full hindsight; snapshots in the archive) saved each student's network after 18, 36 and 54 of people's trials. Each
snapshot plays the 70 layouts by intuition (as in the session) and in the mixed planner over a coarse gamma grid, and
is fitted to the stage of people's session it sits in the middle of (trials 10-27, 28-45, 46+; the start network to
1-9, its planner runs taken from gamma_learning.csv).

    python experiments/fit_memory.py learning  # writes models/summary/gamma_learning.csv
    python experiments/fit_memory.py lean      # writes models/summary/lean_profile.csv: the lean student
                                               # (forward.solve_lean) with and without AND-OR, against people
    python experiments/fit_memory.py lean-snapshots --run lean_people_all_m10_ao   # lean_snapshots_<run>.csv: a
                                               # lean session's snapshots, as it played them, per stage
    python experiments/fit_memory.py shape     # writes models/summary/shape_profile.csv: the lean student with goal
                                               # in sight / a sharper intuition, by distance, against trials 1-9
    python experiments/fit_memory.py snapshots # writes models/summary/gamma_snapshots.csv
    python experiments/fit_memory.py snapshots --run planner_people_all_m10  # gamma_snapshots_<run>.csv: a run's
                                               # own snapshots (experiments/planner_session.py), start not rerun
"""
import glob
import multiprocessing as mp
import os
import random
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import board, human_data as hd, paths  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "memory_fit.csv")
PROFILE = os.path.join(paths.MODELS, "summary", "human_profile", "moves.csv.gz")
RUNS = 20
EDGES, BINS = [0, 2, 4, 8, 16, 10_000], ["2", "3-4", "5-8", "9-16", "17+"]
BETAS, HALVES = (0.5, 1, 2, 4, 8), (1, 2, 4, 8, 16, 32)
CONDITIONS = ([{"name": "hard, last 10 boards", "kw": {"k": 1}}, {"name": "no memory", "kw": {"k": 1, "memory": 1}}]
              + [{"name": f"beta {b}, half-life {h}", "beta": b, "half": h,
                  "kw": {"k": 1, "soft_memory": (b, h)}} for b in BETAS for h in HALVES])
MATCHED = (1.5, 3.0)
SOFT = (2, 1)  # the board memory fitted above
PLANNER_RUNS = 10
LAMS, NOVELTY_HALVES = (2, 4, 8, 16), (2, 5, 20, 80)
PLANNER = ([{"name": "value only, soft memory", "lam": 0, "kw": {"k": 8, "depth": 4, "evaluation": "value",
                                                                 "soft_memory": SOFT}},
            {"name": "lam 16, novelty half-life 80, hard rule", "lam": 16, "nov": 80,
             "kw": {"k": 8, "depth": 4, "evaluation": "weighted", "lam": 16, "half_life": 80}}]
           + [{"name": f"lam {lam}, novelty half-life {h}", "lam": lam, "nov": h,
               "kw": {"k": 8, "depth": 4, "evaluation": "weighted", "lam": lam, "half_life": h, "soft_memory": SOFT}}
              for lam in LAMS for h in NOVELTY_HALVES])


GAMMA_RUNS = 10
GAMMA_GRID = (0.03, 0.07, 0.15, 0.3)
GAMMA = [{"name": f"forward gamma {gf}, backward gamma {gb}", "gamma forward": gf, "gamma backward": gb,
          "kw": {"k": 4, "depth": 20, "backward": 4, "evaluation": "value", "soft_memory": SOFT, "gamma": gb,
                 "rollout_gamma": gf}} for gf in GAMMA_GRID for gb in GAMMA_GRID]
DBINS, DEDGES = ["2-3", "4-6", "7-9", "10+"], [1, 3, 6, 9, 1000]
LEARN_RUNS = 10
LEARN_GRID = (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)
LEARN = [{"name": f"gamma {g}", "gamma forward": g, "gamma backward": g,
          "kw": {"k": 4, "depth": 40, "backward": 4, "evaluation": "value", "soft_memory": SOFT, "gamma": g,
                 "rollout_gamma": g}} for g in LEARN_GRID]
UNDO_RUNS = 10
_MIX = {"k": 4, "depth": 20, "backward": 4, "evaluation": "value", "soft_memory": SOFT, "gamma": 0.03,
        "rollout_gamma": 0.03}
UNDO = [{"name": "mixed (gamma 0.03)", "kw": _MIX},
        {"name": "candidate memory x1", "kw": {**_MIX, "candidate_memory": 1}},
        {"name": "candidate memory x4", "kw": {**_MIX, "candidate_memory": 4}},
        {"name": "candidate memory x16", "kw": {**_MIX, "candidate_memory": 16}},
        {"name": "commit to plan", "kw": {**_MIX, "commit": "plan"}},
        {"name": "plan stays a candidate", "kw": {**_MIX, "commit": "candidate"}},
        {"name": "commit + candidate memory x4", "kw": {**_MIX, "commit": "plan", "candidate_memory": 4}}]
ARCHIVE_RUN = os.path.join(os.path.dirname(os.path.dirname(paths.ROOT)), "rush-hour-archive", "models", "sessions",
                           "ao_people_all_m10", "no_teacher")
SNAP_TRIALS = (18, 36, 54)  # x 10 = the simulated puzzles of the saved snapshots
SNAP_STAGES = {"1-9": (1, 9), "10-27": (10, 27), "28-45": (28, 45), "46+": (46, 1000)}
SNAP_OF = {"1-9": 0, "10-27": 18, "28-45": 36, "46+": 54}  # the snapshot (trials played) for each stage
SNAP_GRID = (0.01, 0.03, 0.1, 0.3)
SNAP_RUNS = {"intuition": 2, "mixed": 1}  # per student and layout
SNAPSHOT = ([{"name": f"after {n} trials, intuition", "snapshot": n, "kind": "intuition", "kw": {"k": 1}}
             for n in (0,) + SNAP_TRIALS]
            + [{"name": f"after {n} trials, gamma {g}", "snapshot": n, "kind": "mixed", "gamma forward": g,
                "gamma backward": g, "kw": LEARN[LEARN_GRID.index(g)]["kw"]} for n in SNAP_TRIALS for g in SNAP_GRID])
LEAN_RUNS = 10
SHAPE_RUNS = 6
LEAN = [{"name": name, "kw": {"lean": True, **kw}} for name, kw in [
    ("intuition", {"w_flat": 0, "w_ao": 0}),
    ("intuition + flat", {"w_flat": 0.08, "w_ao": 0}),
    ("intuition + flat + AND-OR", {"w_flat": 0.08, "w_ao": 0.10}),
    ("intuition + flat + AND-OR, opening", {"w_flat": 0.08, "w_ao": 0.10, "opening": True}),
    ("intuition + AND-OR", {"w_flat": 0, "w_ao": 0.10}),
    ("later: intuition + more flat", {"w_flat": 0.18, "w_ao": 0}),
    ("later: intuition + more flat + AND-OR", {"w_flat": 0.18, "w_ao": 0.10}),
]]
STAGES = {"1-10": (1, 10), "11-20": (11, 20), "21-35": (21, 35), "36+": (36, 1000)}
_DIST = {}


def placement_ages(boards):
    """For each move, how many moves ago the moved car last stood where it lands (None if never)."""
    last, out = {f: 0 for f in boards[0]}, []
    for t, (b0, b1) in enumerate(zip(boards, boards[1:]), 1):
        moved = next(f for f in b1 if f not in b0)
        out.append(t - last[moved] if moved in last else None)
        for f in b1:
            last[f] = t
    return out


def ages(boards):
    """For each move, how many moves ago the attempt last had the board it leads to (None if never)."""
    last, out = {boards[0]: 0}, []
    for t, b in enumerate(boards[1:], 1):
        out.append(t - last[b] if b in last else None)
        last[b] = t
    return out


def job(args):
    cond, instance, run = args
    rng = random.Random(zlib.crc32(f"{cond['name']}|{instance}|{run}".encode()))
    start = hd.puzzle(instance)
    distance = hd.distance(instance)
    trace, sources = [], []
    kw = dict(cond["kw"])
    solve = forward.solve_lean if kw.pop("lean", False) else forward.solve_forward
    out = solve(forward._W["agent"], start, max(50, 10 * distance), rng=rng, trace=trace, sources=sources, **kw)
    boards = [start] + [b for _, b in trace]
    if instance not in _DIST:
        _DIST[instance] = board.multi_bfs(start)
    dist = _DIST[instance]
    d0 = np.array([dist[b] for b in boards[:-1]])
    better = np.array([dist[b1] < dist[b0] for b0, b1 in zip(boards, boards[1:])])
    idx = np.digitize(d0, np.array(DEDGES[1:-1]) + 0.5) if len(d0) else d0
    keep = d0 > 1
    totals = {f"n {b}": int(((idx == i) & keep).sum()) for i, b in enumerate(DBINS)}
    optimal = {f"opt {b}": int(((idx == i) & keep & better).sum()) for i, b in enumerate(DBINS)}
    every = ages(boards)
    by_source = {}
    for src in ("forward", "backward", "commit", "intuition", "flat", "and-or"):
        by_source[f"{src} moves"] = sum(x == src for x in sources)
        by_source[f"{src} undos"] = sum(x == src and a_ == 2 for x, a_ in zip(sources, every))
        mine = np.array([x == src for x in sources[:len(d0)]], bool)
        by_source[f"{src} scored"] = int((mine & (d0 > 1)).sum())
        by_source[f"{src} optimal"] = int((mine & (d0 > 1) & better).sum())
    a = [x for x in every if x is not None]
    pa = [x for x in placement_ages(boards) if x is not None]
    return {"condition": cond["name"], "beta": cond.get("beta"), "half-life": cond.get("half"),
            "lam": cond.get("lam"), "novelty half-life": cond.get("nov"),
            "instance": instance, "run": run, "optimal": distance, "moves": len(trace),
            "solved": out["solved"], "ages": " ".join(map(str, a)), "placement ages": " ".join(map(str, pa)),
            "gamma forward": cond.get("gamma forward"), "gamma backward": cond.get("gamma backward"),
            "chose backward": out.get("backward"), "snapshot": cond.get("snapshot"), "kind": cond.get("kind"),
            **totals, **optimal, **by_source}


def binned(df, column="ages", prefix=""):
    """The attempts' revisits counted per age bin (right-closed, as pd.cut bins people's)."""
    counts = np.array([np.histogram([int(x) for x in str(a).split()] if isinstance(a, str) and a else [],
                                    np.array(EDGES) + 0.5)[0] for a in df[column]])
    return df.assign(**{prefix + b: counts[:, i] for i, b in enumerate(BINS)})


def _init(weights):
    forward._init_one(weights)


def _people(stage=None):
    mv = pd.read_csv(PROFILE, dtype={"who": str})
    mv = mv[mv.source == "people"]
    return mv if stage is None else mv[mv.trial.between(*stage)]


def people_target(stage=None):
    """People's share of moves returning to a board, by age bin, on matched attempts (and their move count); `stage`
    = (first, last) trial keeps that part of the session."""
    mv = _people(stage)
    keys = ["who", "instance", "attempt"]
    att = mv.groupby(keys).agg(n=("move", "size"), L=("optimal length", "first"), solved=("solved", "first"))
    ok = att[(att.n / att.L).between(*MATCHED) & att.solved].index
    m = mv.set_index(keys).loc[ok]
    share = pd.cut(m["board age"], EDGES, labels=BINS).value_counts(sort=False) / len(m)
    placed = pd.cut(m["placement age"], EDGES, labels=BINS).value_counts(sort=False) / len(m)
    placed.index = ["car " + b for b in BINS]
    return share, len(m), placed


def summarize(df, target):
    """Per condition: the matched share by age, the misfit, and how the student plays (all attempts)."""
    df = binned(df).assign(ratio=df.moves / df.optimal)
    m = df[df.ratio.between(*MATCHED) & df.solved]
    g = m.groupby("condition", sort=False)
    share = g[BINS].sum().div(g.moves.sum(), axis=0)
    out = share.copy()
    out["misfit"] = ((share - target) ** 2 / target).sum(axis=1)
    out["matched attempts"] = g.size()
    a = df.groupby("condition", sort=False)
    out["solved"] = a.solved.mean()
    out["moves / optimal (solved)"] = df[df.solved].groupby("condition", sort=False).ratio.mean()
    return out


def summarize_planner(df):
    """summarize() over both board and car-placement returns: the misfit sums the two."""
    target, _, placed = people_target()
    df = binned(binned(df), "placement ages", "car ").assign(ratio=df.moves / df.optimal)
    m = df[df.ratio.between(*MATCHED) & df.solved]
    g = m.groupby("condition", sort=False)
    cols = BINS + list(placed.index)
    share = g[cols].sum().div(g.moves.sum(), axis=0)
    want = pd.concat([target, placed])
    out = share.copy()
    out["board misfit"] = ((share[BINS] - target) ** 2 / target).sum(axis=1)
    out["car misfit"] = ((share[list(placed.index)] - placed) ** 2 / placed).sum(axis=1)
    out["misfit"] = out["board misfit"] + out["car misfit"]
    out["matched attempts"] = g.size()
    out["solved"] = df.groupby("condition", sort=False).solved.mean()
    out["moves / optimal (solved)"] = df[df.solved].groupby("condition", sort=False).ratio.mean()
    return out, want


def people_skill(stage=None):
    """People's chance of an optimal move by distance bin (all moves, distance 1 left out) and moves / optimal on
    solved attempts."""
    mv = _people(stage)
    d = mv[mv.distance > 1]
    opt = (d.after < d.distance).groupby(pd.cut(d.distance, DEDGES, labels=DBINS), observed=True).mean()
    keys = ["who", "instance", "attempt"]
    att = mv.groupby(keys).agg(n=("move", "size"), L=("optimal length", "first"), solved=("solved", "first"))
    return opt, (att.n / att.L)[att.solved].mean()


def summarize_gamma(df):
    """Per condition: skill (optimal-move chance by distance, moves / optimal) and revisits against people's, with
    a misfit for each and their sum."""
    rev, want = summarize_planner(df)
    opt_p, ratio_p = people_skill()
    g = df.groupby("condition", sort=False)
    opt = pd.DataFrame({b: g[f"opt {b}"].sum() / g[f"n {b}"].sum() for b in DBINS})
    ratio = (df.moves / df.optimal)[df.solved].groupby(df.condition[df.solved], sort=False).mean()
    out = opt.add_prefix("optimal ")
    out["moves / optimal"] = ratio
    out["skill misfit"] = ((opt - opt_p.values) ** 2 / opt_p.values).sum(axis=1) + (ratio - ratio_p) ** 2 / ratio_p
    out["revisit misfit"] = rev["misfit"]
    out["misfit"] = out["skill misfit"] + out["revisit misfit"]
    out["solved"] = g.solved.mean()
    out["backward chosen"] = g["chose backward"].sum() / g.moves.sum()
    out["gamma forward"] = g["gamma forward"].first()
    out["gamma backward"] = g["gamma backward"].first()
    people = pd.Series({**{f"optimal {b}": v for b, v in zip(DBINS, opt_p.values)}, "moves / optimal": ratio_p})
    return out, people


def summarize_undo(df):
    """The gamma fit's summary plus, over all attempts, the undo rate and where undos come from."""
    out, people = summarize_gamma(df)
    rev, want = summarize_planner(df)
    g = df.groupby("condition", sort=False)
    out = out.drop(columns=["gamma forward", "gamma backward"])
    out["undo (matched)"] = rev["2"]
    out["board return (matched)"] = rev[BINS].sum(axis=1)
    out["car return (matched)"] = rev[["car " + b for b in BINS]].sum(axis=1)
    for src in ("forward", "backward", "commit"):
        out[f"{src} share"] = g[f"{src} moves"].sum() / g.moves.sum()
        out[f"{src} undo rate"] = g[f"{src} undos"].sum() / g[f"{src} moves"].sum().replace(0, np.nan)
    people["undo (matched)"] = want["2"]
    people["board return (matched)"] = want[BINS].sum()
    people["car return (matched)"] = want[["car " + b for b in BINS]].sum()
    return out, people


def snapshot_networks(folder=ARCHIVE_RUN):
    """{name: path}: the start network and every student's snapshot in a session run's teacher folder (default:
    the calibrated session's, in the archive)."""
    nets = {"0": paths.START}
    for n in SNAP_TRIALS:
        for f in sorted(glob.glob(os.path.join(folder, f"student*_puzzle{10 * n:03d}.pt"))):
            nets[f"{n} {os.path.basename(f)[:9]}"] = f
    return nets


def job_snapshot(args):
    """job() with the network named in args[3] (loaded by forward._init)."""
    forward._W["agent"] = forward._W["agents"][args[3]]
    return job(args[:3])


def summarize_lean(df):
    """The gamma fit's summary (skill and revisits against people) plus, per mode, its share of moves, the chance its
    moves are optimal (2+ from the goal) and its undo rate."""
    out, people = summarize_gamma(df)
    rev, want = summarize_planner(df)
    out = out.drop(columns=["gamma forward", "gamma backward", "backward chosen"])
    out["undo (matched)"] = rev["2"]
    out["board return (matched)"] = rev[BINS].sum(axis=1)
    people["undo (matched)"] = want["2"]
    people["board return (matched)"] = want[BINS].sum()
    g = df.groupby("condition", sort=False)
    for src in ("intuition", "flat", "and-or"):
        out[f"{src} share"] = g[f"{src} moves"].sum() / g.moves.sum()
        out[f"{src} optimal"] = g[f"{src} optimal"].sum() / g[f"{src} scored"].sum().replace(0, np.nan)
        out[f"{src} undo"] = g[f"{src} undos"].sum() / g[f"{src} moves"].sum().replace(0, np.nan)
    return out, people


def summarize_learning(df, stages=None):
    """Per stage of people's session x gamma: the gamma fit's misfits, with the model's attempts weighted by how
    often people at that stage played each layout. Returns (table, people's targets per stage)."""
    stages = stages or STAGES
    df = binned(binned(df), "placement ages", "car ").assign(ratio=df.moves / df.optimal)
    mv = _people()
    first = mv.groupby(["who", "instance", "attempt"]).agg(trial=("trial", "first"), instance_=("instance", "first"))
    rows, people = [], {}
    for stage, (lo, hi) in stages.items():
        mix = first[first.trial.between(lo, hi)].instance_.value_counts()
        w = df.instance.map(mix).fillna(0) / df.groupby(["condition", "instance"]).run.transform("size")
        target, _, placed = people_target((lo, hi))
        opt_p, ratio_p = people_skill((lo, hi))
        people[stage] = pd.Series({**{f"optimal {b}": v for b, v in zip(DBINS, opt_p.values)},
                                   "moves / optimal": ratio_p, **target.add_prefix("board ").to_dict(),
                                   **placed.to_dict()})
        for cond, d in df.assign(w=w).groupby("condition", sort=False):
            opt = pd.Series({b: (d.w * d[f"opt {b}"]).sum() / (d.w * d[f"n {b}"]).sum() for b in DBINS})
            s_ = d[d.solved]
            ratio = (s_.w * s_.ratio).sum() / s_.w.sum()
            m = d[d.ratio.between(*MATCHED) & d.solved]
            board_ = (m[BINS].mul(m.w, axis=0).sum() / (m.w * m.moves).sum())
            car = (m[list(placed.index)].mul(m.w, axis=0).sum() / (m.w * m.moves).sum())
            skill = ((opt - opt_p.values) ** 2 / opt_p.values).sum() + (ratio - ratio_p) ** 2 / ratio_p
            revisit = ((board_ - target) ** 2 / target).sum() + ((car - placed) ** 2 / placed).sum()
            rows.append({"stage": stage, "condition": cond, "gamma": d["gamma forward"].iloc[0],
                         "snapshot": d["snapshot"].iloc[0] if "snapshot" in d else 0,
                         "kind": d["kind"].iloc[0] if "kind" in d else "mixed",
                         **{f"optimal {b}": opt[b] for b in DBINS}, "moves / optimal": ratio,
                         **{f"board {b}": board_[b] for b in BINS}, "skill misfit": skill,
                         "revisit misfit": revisit, "misfit": skill + revisit,
                         "solved": (d.w * d.solved).sum() / d.w.sum()})
    return pd.DataFrame(rows), pd.DataFrame(people).T


if __name__ == "__main__":
    instances = sorted(hd.trials().instance.unique())
    if sys.argv[1:2] == ["lean-snapshots"]:
        import json
        run = sys.argv[sys.argv.index("--run") + 1]
        with open(os.path.join(paths.SESSIONS, run, "config.json")) as f:
            lean = json.load(f)["lean"]
        solver = forward.LeanSolver(**lean)
        nets = snapshot_networks(os.path.join(paths.SESSIONS, run, "no_teacher"))
        with open(os.path.join(paths.SESSIONS, run, "config.json")) as f:
            nets["0"] = json.load(f)["init"]  # the run's own start (paths.START for the current one)
        weights = {}
        for name, f in nets.items():
            with open(f, "rb") as fh:
                weights[name] = fh.read()
        jobs = []
        for n in (0,) + SNAP_TRIALS:
            c = {"name": f"after {n} trials, lean", "snapshot": n, "kind": "lean",
                 "kw": {"lean": True, "w_flat": solver.flat_at(n + 1), "w_ao": lean["w_ao"],
                        "opening": lean["opening"],
                        **({"ao_order": lean["ao_order"]} if "ao_order" in lean else {})}}
            names = ["0"] if n == 0 else [m for m in nets if m.split()[0] == str(n)]
            for i in instances:
                for sname in names:
                    for r in range(16 if n == 0 else 2):
                        jobs.append((c, i, f"{sname}|{r}", sname))
        random.Random(0).shuffle(jobs)
        with mp.Pool(initializer=forward._init, initargs=(weights,)) as pool:
            df = pd.DataFrame(pool.map(job_snapshot, jobs, chunksize=4))
        df.to_csv(os.path.join(paths.MODELS, "summary", f"lean_snapshots_{run}.csv"), index=False)
        out, people = summarize_learning(df, SNAP_STAGES)
        paired = out[[r.snapshot == SNAP_OF[r.stage] for r in out.itertuples()]]
        print(people[["optimal 2-3", "optimal 4-6", "optimal 7-9", "optimal 10+", "moves / optimal",
                      "board 2"]].round(3).to_string())
        print(paired[["stage", "condition", "optimal 2-3", "optimal 4-6", "optimal 7-9", "optimal 10+",
                      "moves / optimal", "board 2", "skill misfit", "revisit misfit", "misfit",
                      "solved"]].round(3).to_string(index=False))
        sys.exit()
    if sys.argv[1:2] == ["shape"]:
        import planner_session as ps
        weights = {}
        for name, f in ps.NETS.items():
            with open(f, "rb") as fh:
                weights[name] = fh.read()
        conds = [{"name": name, "kind": "lean", "snapshot": 0, "kw": {k: v for k, v in kw.items() if k != "net"},
                  "net": kw["net"]} for name, kw in ps.SHAPE.items()]
        jobs = [(c, i, f"{r}", c["net"]) for c in conds for i in instances for r in range(SHAPE_RUNS)]
        random.Random(0).shuffle(jobs)
        with mp.Pool(initializer=forward._init, initargs=(weights,)) as pool:
            df = pd.DataFrame(pool.map(job_snapshot, jobs, chunksize=4))
        df.to_csv(os.path.join(paths.MODELS, "summary", "shape_profile.csv"), index=False)
        out, people = summarize_learning(df, {"1-9": SNAP_STAGES["1-9"]})
        print(people[["optimal 2-3", "optimal 4-6", "optimal 7-9", "optimal 10+", "moves / optimal",
                      "board 2"]].round(3).to_string())
        print(out[["condition", "optimal 2-3", "optimal 4-6", "optimal 7-9", "optimal 10+", "moves / optimal",
                   "board 2", "skill misfit", "solved"]].round(3).to_string(index=False))
        sys.exit()
    if sys.argv[1:2] == ["lean"]:
        with open(paths.START, "rb") as f:
            weights = f.read()
        jobs = [(c, i, r) for c in LEAN for i in instances for r in range(LEAN_RUNS)]
        random.Random(0).shuffle(jobs)
        with mp.Pool(initializer=_init, initargs=(weights,)) as pool:
            df = pd.DataFrame(pool.map(job, jobs, chunksize=2))
        df.to_csv(os.path.join(paths.MODELS, "summary", "lean_profile.csv"), index=False)
        out, people = summarize_lean(df)
        print(pd.concat([people.to_frame("people").T, out]).round(3).T.to_string())
        sys.exit()
    if sys.argv[1:2] in (["undo"], ["undo-smoke"]):
        with open(paths.START, "rb") as f:
            weights = f.read()
        smoke = sys.argv[1] == "undo-smoke"
        jobs = [(c, i, r) for c in UNDO for i in (instances[:6] if smoke else instances)
                for r in range(2 if smoke else UNDO_RUNS)]
        random.Random(0).shuffle(jobs)
        with mp.Pool(4 if smoke else None, initializer=_init, initargs=(weights,)) as pool:
            df = pd.DataFrame(pool.map(job, jobs, chunksize=1))
        if not smoke:
            df.to_csv(os.path.join(paths.MODELS, "summary", "undo_fit.csv"), index=False)
        out, people = summarize_undo(df)
        print(pd.concat([people.to_frame("people").T, out]).round(3).T.to_string())
        sys.exit()
    if sys.argv[1:2] in (["snapshots"], ["snapshots-smoke"]):
        smoke = sys.argv[1] == "snapshots-smoke"
        run = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else None
        nets = snapshot_networks(os.path.join(paths.SESSIONS, run, "no_teacher") if run else ARCHIVE_RUN)
        assert len(nets) == 1 + 8 * len(SNAP_TRIALS), f"{len(nets)} networks"
        weights = {}
        for name, f in nets.items():
            with open(f, "rb") as fh:
                weights[name] = fh.read()
        print(len(nets), "networks", flush=True)
        jobs = []
        for c in SNAPSHOT:
            names = ["0"] if c["snapshot"] == 0 else [m for m in nets if m.split()[0] == str(c["snapshot"])]
            for i in (instances[:4] if smoke else instances):
                for sname in names:
                    for r in range(1 if smoke else SNAP_RUNS[c["kind"]] * (8 if c["snapshot"] == 0 else 1)):
                        jobs.append((c, i, f"{sname}|{r}", sname))
        if smoke:
            jobs = [j for j in jobs if j[0]["kind"] == "intuition" or j[0]["gamma forward"] == 0.1][:200]
        if run:  # the start network is shared with the archive's run: only the run's own snapshots
            jobs = [j for j in jobs if j[0]["snapshot"] != 0]
        random.Random(0).shuffle(jobs)
        rows = []
        with mp.Pool(4 if smoke else None, initializer=forward._init, initargs=(weights,)) as pool:
            for n, row in enumerate(pool.imap_unordered(job_snapshot, jobs, chunksize=1), 1):
                rows.append(row)
                if n % 500 == 0:
                    print(n, "of", len(jobs), flush=True)
        df = pd.DataFrame(rows)
        if not smoke:
            df.to_csv(os.path.join(paths.MODELS, "summary", f"gamma_snapshots{'_' + run if run else ''}.csv"),
                      index=False)
        out, people = summarize_learning(df, SNAP_STAGES)
        print(people.round(3).to_string())
        print(out[["stage", "condition", "misfit", "skill misfit", "revisit misfit", "optimal 7-9",
                   "moves / optimal", "solved"]].round(3).to_string())
        sys.exit()
    if sys.argv[1:2] == ["learning"]:
        with open(paths.START, "rb") as f:
            weights = f.read()
        jobs = [(c, i, r) for c in LEARN for i in instances for r in range(LEARN_RUNS)]
        random.Random(0).shuffle(jobs)
        rows = []
        with mp.Pool(initializer=_init, initargs=(weights,)) as pool:
            for n, row in enumerate(pool.imap_unordered(job, jobs, chunksize=1), 1):
                rows.append(row)
                if n % 500 == 0:
                    print(n, "of", len(jobs), flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(paths.MODELS, "summary", "gamma_learning.csv"), index=False)
        out, people = summarize_learning(df)
        print(people.round(3).to_string())
        print(out.pivot(index="gamma", columns="stage", values="misfit")[list(STAGES)].round(3).to_string())
        sys.exit()
    if sys.argv[1:2] == ["gamma"]:
        with open(paths.START, "rb") as f:
            weights = f.read()
        jobs = [(c, i, r) for c in GAMMA for i in instances for r in range(GAMMA_RUNS)]
        random.Random(0).shuffle(jobs)
        with mp.Pool(initializer=_init, initargs=(weights,)) as pool:
            df = pd.DataFrame(pool.map(job, jobs, chunksize=1))
        df.to_csv(os.path.join(paths.MODELS, "summary", "gamma_fit.csv"), index=False)
        out, people = summarize_gamma(df)
        print("people", people.round(3).to_dict())
        print(out.sort_values("misfit").round(3).to_string())
        sys.exit()
    if sys.argv[1:2] == ["planner"]:
        with open(paths.START, "rb") as f:
            weights = f.read()
        jobs = [(c, i, r) for c in PLANNER for i in instances for r in range(PLANNER_RUNS)]
        random.Random(0).shuffle(jobs)
        with mp.Pool(initializer=_init, initargs=(weights,)) as pool:
            df = pd.DataFrame(pool.map(job, jobs, chunksize=2))
        df.to_csv(OUT.replace(".csv", "_planner.csv"), index=False)
        out, want = summarize_planner(df)
        print("people", want.round(4).to_dict())
        print(out.sort_values("misfit").round(4).to_string())
        sys.exit()
    with open(paths.START, "rb") as f:
        weights = f.read()
    jobs = [(c, i, r) for c in CONDITIONS for i in instances for r in range(RUNS)]
    random.Random(0).shuffle(jobs)
    with mp.Pool(initializer=_init, initargs=(weights,)) as pool:
        df = pd.DataFrame(pool.map(job, jobs, chunksize=4))
    df.to_csv(OUT, index=False)
    target, n, _ = people_target()
    print("people", target.round(4).to_dict(), n, "moves")
    print(summarize(df, target).sort_values("misfit").round(4).to_string())
