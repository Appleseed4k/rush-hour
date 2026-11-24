"""Do people shift from intuition to flat planning over the session? A move-by-move mixture fit.

Every move people made 2 or more moves from the goal gets two probabilities from the same network:
- intuition: the policy's softmax over the legal moves, each lowered by the soft board memory (beta 2, half-life 1: a
  move to a board the attempt had `age` moves ago loses beta x 0.5^(age / half-life));
- flat planning (forward.solve_flat without the check): the policy's `consider` most likely moves (memory included),
  one chosen by a softmax over -(the value head's moves to go from the board it leads to) / tau - the memory penalty.

A move is modelled as (1 - eps) x (w x P_flat + (1 - w) x P_intuition) + eps / legal moves: w is the share of moves
made by flat planning, eps a lapse. (consider, tau, eps) are fitted over all moves; w per stage of the session (trials
1-9, 10-27, 28-45, 46+), per stage x fast / slow move (over 2.5 x the person's median time), and per person early
(trials 1-27) and late (28+). Stage intervals: a bootstrap over people.

Two versions of "the same network": the start network for every stage, or the stage's own network from the calibrated
intuition session (ao_people_all_m10 in the archive; snapshots after 18, 36, 54 trials, the 8 students' probabilities
averaged) - so that a rising w isn't just people's intuition improving.

    python experiments/mode_mixture.py   # writes models/summary/mode_mixture/{moves.csv.gz, fit.csv, people.csv}

Backward reasoning (`andor`): a third component, the two-mode student's backward mode - one AND-OR pass from the board
(the network ordering its options with Gumbel noise, lapse rate 0.07 - the dest head of the start of the time,
paths.DEST_START, in the saved fit; the policy head since 2026-11-24), scored by its first move. That
probability has no closed form, so it is estimated by sampling passes from every board people played on (100 with the
start network; 25 per student with a stage's 8 snapshots). Each move is scored as the first move of a fresh pass (plan
following - later steps of a plan made earlier - is not modelled). The three-way weights (intuition, flat, AND-OR) are
fitted per stage, per distance to the goal, per fast / slow move and per person, with (consider, tau) from the two-way
fit and eps refitted.

    python experiments/mode_mixture.py andor   # adds andor_counts.npz, fit3.csv, people3.csv

The two-mode switch (`loops`): the two-mode student turns to an AND-OR plan when a move leads back to a board the
attempt had within the last 10 moves (a loop) and follows it for a few moves. People's loops are observable (the
human profile's board age), so their moves are split by how long ago their last loop was - on the board a loop just
led to (1 move after), 2-3 moves after, or elsewhere (4+ moves after one, or none yet) - and by the loop's kind (an
undo, age 2, or a longer return, 3-10) - and the three-way weights are refitted per group, and per group x distance to
the goal and x move speed (after a loop, boards and moves differ in both).

    python experiments/mode_mixture.py loops   # writes loops.csv

Plan, then execute (`slow`): if a slow move is a deliberation that produces a backward plan, the quick moves after it
carry the plan out. Every move of an attempt is slow if it took over 2.5 x the person's median time (all their moves);
mixture moves are split into the slow moves themselves, fast moves 1 or 2-3 moves after the attempt's last slow move,
and other fast moves (4+ after one, or none yet), and by whether that slow move opened the attempt (move 0, the 2.9 s
first look) or came mid-attempt; the three-way weights are refitted per group and per group x distance to the goal.

    python experiments/mode_mixture.py slow    # writes slow.csv
"""
import multiprocessing as mp
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import board, human_data as hd, paths  # noqa: E402
from rushhour.and_or import solve as and_or_solve  # noqa: E402
from rushhour.student import PolicyAgent  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_memory as fm  # noqa: E402
import human_profile as hp  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "mode_mixture")
BETA, HALF = 2.0, 1.0
CONSIDER, TAU, EPS = (2, 3, 5, None), (0.25, 0.5, 1, 2, 4), (0.01, 0.03, 0.1)
W = np.linspace(0, 1, 51)
SLOW = 2.5


def records():
    """One row per human move 2+ moves from the goal: its context, the legal options (child boards), the memory
    penalty of each and which was played."""
    dist = {}
    rows, parents, children = [], [], []
    for who, trial, instance, attempt, boards, moves, solved, seconds in hp.people_attempts():
        if trial is None:
            continue
        if instance not in dist:
            dist[instance] = board.multi_bfs(boards[0])
        d = dist[instance]
        orient = board.car_orientations(boards[0])
        last = {boards[0]: 0}
        for t, (b, played) in enumerate(zip(boards, moves)):
            if d[b] > 1:
                options = list(board.neighbors(b, orient))
                chosen = [m for m, _ in options].index(played)
                pen = [0.0 if c not in last else BETA * 0.5 ** ((t + 1 - last[c]) / HALF) for _, c in options]
                rows.append({"who": who, "trial": trial, "instance": instance, "attempt": attempt, "move": t,
                             "distance": d[b], "seconds": seconds[t], "chosen": chosen, "n": len(options)})
                parents.append((b, [m for m, _ in options]))
                children.append(([c for _, c in options], np.array(pen, dtype=np.float32)))
            last[boards[t + 1]] = t + 1
    return pd.DataFrame(rows), parents, children


def bundle(parents, children):
    """The unique boards of these moves, encoded once for every network."""
    encoder = PolicyAgent.load(paths.START).encoder
    uniq = {}
    for b, _ in parents:
        uniq.setdefault(b, len(uniq))
    for cs, _ in children:
        for c in cs:
            uniq.setdefault(c, len(uniq))
    enc = np.stack([encoder.encode_numpy(b) for b in uniq])
    is_parent = np.zeros(len(uniq), bool)
    is_parent[[uniq[b] for b, _ in parents]] = True
    return {"uniq": uniq, "enc": enc, "is_parent": is_parent, "parents": parents, "children": children}


def evaluate(path, bun, device):
    """Per move: (policy logits of its options, moves to go after each option) from the network at `path`."""
    agent = PolicyAgent.load(path)
    net = agent.net.to(device).eval()
    uniq, enc, is_parent = bun["uniq"], bun["enc"], bun["is_parent"]
    value = np.zeros(len(enc), np.float32)
    policy = {}
    with torch.no_grad():
        for i in range(0, len(enc), 4096):
            x = torch.from_numpy(enc[i:i + 4096]).float().to(device)
            p, v = net.heads(x)
            value[i:i + 4096] = (10 * v).cpu().numpy()
            pc = p.cpu().numpy()
            for j in np.nonzero(is_parent[i:i + 4096])[0]:
                policy[i + j] = pc[j]
    logits, ends = [], []
    for (b, moves), (cs, _) in zip(bun["parents"], bun["children"]):
        p = policy[uniq[b]]
        logits.append(np.array([p[agent.action_id(b, *m)] for m in moves], np.float32))
        ends.append(value[[uniq[c] for c in cs]])
    return logits, ends


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def probabilities(logits, ends, pens, chosen):
    """P(played move) under intuition, and under flat planning for every (consider, tau)."""
    p_int = np.empty(len(chosen))
    p_flat = {(m, t): np.empty(len(chosen)) for m in CONSIDER for t in TAU}
    for k, (lg, en, pen, c) in enumerate(zip(logits, ends, pens, chosen)):
        p_int[k] = softmax(lg - pen)[c]
        order = np.argsort(-(lg - pen), kind="stable")
        for m in CONSIDER:
            pool = order if m is None else order[:m]
            inside = np.nonzero(pool == c)[0]
            for t in TAU:
                p_flat[(m, t)][k] = softmax(-en[pool] / t - pen[pool])[inside[0]] if len(inside) else 0.0
    return p_int, p_flat


def loglik(p_int, p_flat, n, w, eps):
    """Summed log likelihood for every w in `w` (an array)."""
    mix = np.outer(w, p_flat) + np.outer(1 - w, p_int)
    return np.log((1 - eps) * mix + eps / n).sum(axis=1)


def fit(df, p_int, p_flat):
    """(consider, tau, eps) over all moves with w free per stage; returns the best and its per-stage w."""
    best = None
    for key, pf in p_flat.items():
        for eps in EPS:
            total, ws = 0.0, {}
            for st, idx in df.groupby("stage", observed=True).indices.items():
                ll = loglik(p_int[idx], pf[idx], df.n.values[idx], W, eps)
                total += ll.max()
                ws[st] = W[ll.argmax()]
            if best is None or total > best[0]:
                best = (total, key, eps, ws)
    return best


def stage_table(df, p_int, pf, eps, rng, boots=200):
    """Per stage (and fast / slow): w, its bootstrap 90% interval over people, and log likelihood per move for
    intuition only, flat only and the mix."""
    rows = []
    groups = [("all", df.index.values)] + [(s, df.index[df.speed == s].values) for s in ("fast", "slow")]
    for label, sub in groups:
        d = df.loc[sub]
        for st, g in d.groupby("stage", observed=True):
            idx = g.index.values
            ll = loglik(p_int[idx], pf[idx], df.n.values[idx], W, eps)
            people = g.who.unique()
            by_who = {w: g.index[g.who == w].values for w in people}
            boot = []
            for _ in range(boots):
                pick = np.concatenate([by_who[w] for w in rng.choice(people, len(people))])
                boot.append(W[loglik(p_int[pick], pf[pick], df.n.values[pick], W, eps).argmax()])
            rows.append({"moves": label, "stage": st, "w": W[ll.argmax()], "w lo": np.percentile(boot, 5),
                         "w hi": np.percentile(boot, 95), "n moves": len(idx), "people": len(people),
                         "ll intuition": ll[0] / len(idx), "ll flat": ll[-1] / len(idx), "ll mix": ll.max() / len(idx)})
    return pd.DataFrame(rows)


def per_person(df, p_int, pf, eps, min_moves=150):
    """w early (trials 1-27) and late (28+) for each person with at least `min_moves` moves in both."""
    rows = []
    for who, g in df.groupby("who"):
        early, late = g.index[g.trial <= 27].values, g.index[g.trial >= 28].values
        if len(early) < min_moves or len(late) < min_moves:
            continue
        we = W[loglik(p_int[early], pf[early], df.n.values[early], W, eps).argmax()]
        wl = W[loglik(p_int[late], pf[late], df.n.values[late], W, eps).argmax()]
        rows.append({"who": who, "early moves": len(early), "late moves": len(late), "w early": we, "w late": wl})
    return pd.DataFrame(rows)


N_START, N_STUDENT, AO_GAMMA = 100, 25, 0.07
STEP = 0.02
SIMPLEX = np.array([(a, b) for a in np.arange(0, 1 + 1e-9, STEP) for b in np.arange(0, 1 - a + 1e-9, STEP)],
                   dtype=np.float32)  # (w flat, w AND-OR); w intuition = 1 - both
_AO = {}


def _ao_job(job):
    """AND-OR first-move counts for a chunk of boards: [(board, legal moves, passes)] with the network at `path`."""
    path, chunk, seed = job
    if _AO.get("path") != path:
        torch.set_num_threads(1)
        agent = PolicyAgent.load(path)
        agent.temperature = 1.0
        _AO.update(path=path, agent=agent)
    agent = _AO["agent"]
    random.seed(seed)
    out = []
    for b, moves, n in chunk:
        idx = {m: i for i, m in enumerate(moves)}
        c = np.zeros(len(moves), np.int32)
        for _ in range(n):
            att = and_or_solve(b, heuristic=agent.heuristic, gamma=AO_GAMMA)
            plan = att if type(att) is list else att[1]
            if plan and plan[0] in idx:
                c[idx[plan[0]]] += 1
        out.append(c)
    return out


def andor_counts(parents, df):
    """Per move: AND-OR first-move counts over its legal moves and the passes run - with the start network (every
    move), and pooled over the stage's 8 snapshots (moves in stages after the first)."""
    cache = os.path.join(OUT, "andor_counts.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return list(z["start"]), list(z["stage"]), z["n_start"], z["n_stage"]
    jobs, where = [], []

    def add(path, ids, n, tag):
        uniq = {}
        for k in ids:
            uniq.setdefault(parents[k][0], []).append(k)
        items = [(b, parents[ks[0]][1], n) for b, ks in uniq.items()]
        for i in range(0, len(items), 40):
            jobs.append((path, items[i:i + 40], len(jobs)))
            where.append((tag, [uniq[b] for b, _, _ in items[i:i + 40]]))

    add(paths.START, range(len(parents)), N_START, "start")
    snaps = fm.snapshot_networks()
    for st, n in fm.SNAP_OF.items():
        if n == 0:
            continue
        ids = np.nonzero((df.stage == st).values)[0]
        for name, f in snaps.items():
            if name.split()[0] == str(n):
                add(f, ids, N_STUDENT, "stage")
    start = [None] * len(parents)
    stage = [np.zeros(len(parents[k][1]), np.int32) for k in range(len(parents))]
    with mp.Pool() as pool:
        for i, res in enumerate(pool.imap(_ao_job, jobs)):
            tag, groups = where[i]
            for c, ks in zip(res, groups):
                for k in ks:
                    if tag == "start":
                        start[k] = c
                    else:
                        stage[k] = stage[k] + c
            if i % 200 == 0:
                print(f"AND-OR: {i} of {len(jobs)} chunks", flush=True)
    n_start = np.full(len(parents), N_START)
    n_stage = np.where((df.stage == list(fm.SNAP_OF)[0]).values, N_START, 8 * N_STUDENT)
    for k in np.nonzero(n_stage == N_START)[0]:
        stage[k] = start[k]
    np.savez(cache, start=np.array(start, dtype=object), stage=np.array(stage, dtype=object), n_start=n_start,
             n_stage=n_stage)
    return start, stage, n_start, n_stage


def ll3(pi, pf, pa, n, eps):
    """Summed log likelihood of the moves for every (w flat, w AND-OR) on SIMPLEX."""
    wf, wa = SIMPLEX[:, :1], SIMPLEX[:, 1:]
    f32 = lambda x: x[None].astype(np.float32)  # noqa: E731
    mix = (1 - wf - wa) * f32(pi) + wf * f32(pf) + wa * f32(pa)
    return np.log((1 - eps) * mix + eps / n[None]).sum(axis=1)


def three_way(df, pi, pf, pa, eps, rng, boots=100):
    """Weights (intuition, flat, AND-OR) with 90% people-bootstrap intervals, per stage, distance and speed; and the
    log likelihood per move of intuition alone, intuition + flat, intuition + AND-OR and all three."""
    n = df.n.values
    one_flat = np.abs(SIMPLEX[:, 1]) < 1e-6
    one_ao = np.abs(SIMPLEX[:, 0]) < 1e-6
    rows = []
    for by in ("stage", "dist", "speed"):
        for level, g in df.groupby(by, observed=True):
            idx = g.index.values
            ll = ll3(pi[idx], pf[idx], pa[idx], n[idx], eps)
            k = ll.argmax()
            people = g.who.unique()
            by_who = {w: g.index[g.who == w].values for w in people}
            boot = []
            for _ in range(boots):
                pick = np.concatenate([by_who[w] for w in rng.choice(people, len(people))])
                boot.append(SIMPLEX[ll3(pi[pick], pf[pick], pa[pick], n[pick], eps).argmax()])
            boot = np.array(boot)
            wf, wa = SIMPLEX[k]
            rows.append({"by": by, "level": level, "moves": len(idx), "people": len(people),
                         "w intuition": 1 - wf - wa, "w flat": wf, "w AND-OR": wa,
                         "w flat lo": np.percentile(boot[:, 0], 5), "w flat hi": np.percentile(boot[:, 0], 95),
                         "w AND-OR lo": np.percentile(boot[:, 1], 5), "w AND-OR hi": np.percentile(boot[:, 1], 95),
                         "w intuition lo": np.percentile(1 - boot.sum(1), 5),
                         "w intuition hi": np.percentile(1 - boot.sum(1), 95),
                         "ll intuition": ll[0] / len(idx), "ll + flat": ll[one_ao].max() / len(idx),
                         "ll + AND-OR": ll[one_flat].max() / len(idx), "ll all three": ll[k] / len(idx)})
    return pd.DataFrame(rows)


def people3(df, pi, pf, pa, eps, min_moves=150):
    rows = []
    for who, g in df.groupby("who"):
        early, late = g.index[g.trial <= 27].values, g.index[g.trial >= 28].values
        if len(early) < min_moves or len(late) < min_moves:
            continue
        e = SIMPLEX[ll3(pi[early], pf[early], pa[early], df.n.values[early], eps).argmax()]
        l_ = SIMPLEX[ll3(pi[late], pf[late], pa[late], df.n.values[late], eps).argmax()]
        rows.append({"who": who, "w flat early": e[0], "w AND-OR early": e[1], "w flat late": l_[0],
                     "w AND-OR late": l_[1]})
    return pd.DataFrame(rows)


def main_andor():
    df, parents, children = records()
    df["stage"] = pd.cut(df.trial, [0, 9, 27, 45, 1000], labels=list(fm.SNAP_STAGES))
    df["speed"] = np.where(df.seconds > SLOW * df.groupby("who").seconds.transform("median"), "slow", "fast")
    df["dist"] = pd.cut(df.distance, [1, 3, 6, 9, 99], labels=["2-3", "4-6", "7-9", "10+"])
    saved = pd.read_csv(os.path.join(OUT, "moves.csv.gz"), dtype={"who": str})
    assert len(saved) == len(df) and (saved.who.values == df.who.values).all() and \
        (saved.move.values == df.move.values).all(), "moves.csv.gz doesn't match the replayed moves"
    start, stage, n_start, n_stage = andor_counts(parents, df)
    chosen = df.chosen.values
    pa = {"start network": np.array([c[k] for c, k in zip(start, chosen)]) / n_start,
          "stage's network": np.array([c[k] for c, k in zip(stage, chosen)]) / n_stage}
    df["p AND-OR (start network)"] = pa["start network"]
    df["p AND-OR (stage's network)"] = pa["stage's network"]
    rng = np.random.default_rng(0)
    tables, people = [], []
    for version in ("start network", "stage's network"):
        pi = saved[f"p intuition ({version})"].values
        pf = saved[f"p flat ({version})"].values
        eps = max(EPS, key=lambda e: ll3(pi, pf, pa[version], df.n.values, e).max())
        t = three_way(df, pi, pf, pa[version], eps, rng).assign(version=version, eps=eps)
        tables.append(t)
        people.append(people3(df, pi, pf, pa[version], eps).assign(version=version))
        print(f"== {version} (eps {eps}); AND-OR gives the played move {pa[version].mean():.3f} on average "
              f"(intuition {pi.mean():.3f}, flat {pf.mean():.3f})")
        print(t.drop(columns=["version"]).round(3).to_string(index=False), flush=True)
        pp = people[-1]
        print(f"per person ({len(pp)}): AND-OR early {pp['w AND-OR early'].mean():.2f} late "
              f"{pp['w AND-OR late'].mean():.2f}; flat early {pp['w flat early'].mean():.2f} late "
              f"{pp['w flat late'].mean():.2f}", flush=True)
    pd.concat(tables).to_csv(os.path.join(OUT, "fit3.csv"), index=False)
    pd.concat(people).to_csv(os.path.join(OUT, "people3.csv"), index=False)
    saved["p AND-OR (start network)"] = pa["start network"]
    saved["p AND-OR (stage's network)"] = pa["stage's network"]
    saved.to_csv(os.path.join(OUT, "moves.csv.gz"), index=False)
    print("saved to", OUT)


LOOP_WINDOW = 10


def loop_groups(df):
    """Per mixture move: moves since the attempt's last loop ("1", "2-3", "elsewhere") and that loop's kind."""
    prof = pd.read_csv(os.path.join(paths.MODELS, "summary", "human_profile", "moves.csv.gz"), dtype={"who": str})
    prof = prof[prof.source == "people"].sort_values(["who", "instance", "attempt", "move"])
    keys = ["who", "instance", "attempt"]
    since, kind = [], []
    for _, g in prof.groupby(keys, sort=False):
        last, last_kind = None, None
        for t, age in zip(g.move.values, g["board age"].values):
            since.append(None if last is None else t - last)
            kind.append(last_kind)
            if age == age and age <= LOOP_WINDOW:  # move t led back to a recent board
                last, last_kind = t, "undo" if age == 2 else "longer"
    prof = prof.assign(since=since, kind=kind)
    out = df.merge(prof[keys + ["move", "since", "kind"]], on=keys + ["move"], how="left", validate="1:1")
    out["after loop"] = np.select([out.since == 1, out.since.isin([2, 3])], ["1", "2-3"], "elsewhere")
    out.loc[out["after loop"] == "elsewhere", "kind"] = None
    return out


def weights(df, pi, pf, pa, eps, groups, rng, boots=100, first="after loop"):
    """Three-way weights with 90% people-bootstrap intervals for each group (name -> row indices)."""
    n = df.n.values
    rows = []
    for name, idx in groups.items():
        if len(idx) < 200:
            continue
        g = df.iloc[idx]
        ll = ll3(pi[idx], pf[idx], pa[idx], n[idx], eps)
        wf, wa = SIMPLEX[ll.argmax()]
        by_who = {w: idx[(g.who == w).values] for w in g.who.unique()}
        people = list(by_who)
        boot = np.array([SIMPLEX[ll3(pi[k], pf[k], pa[k], n[k], eps).argmax()]
                         for k in (np.concatenate([by_who[w] for w in rng.choice(people, len(people))])
                                   for _ in range(boots))])
        rows.append({**dict(zip((first, "breakdown", "level"), name)), "moves": len(idx), "people": len(people),
                     "w intuition": 1 - wf - wa, "w flat": wf, "w AND-OR": wa,
                     "w AND-OR lo": np.percentile(boot[:, 1], 5), "w AND-OR hi": np.percentile(boot[:, 1], 95),
                     "w flat lo": np.percentile(boot[:, 0], 5), "w flat hi": np.percentile(boot[:, 0], 95),
                     "mean p AND-OR / p intuition": np.mean(pa[idx]) / np.mean(pi[idx]),
                     "slow": (g.speed == "slow").mean()})
    return pd.DataFrame(rows)


def main_loops():
    saved = pd.read_csv(os.path.join(OUT, "moves.csv.gz"), dtype={"who": str})
    df = loop_groups(saved)
    df["dist"] = pd.cut(df.distance, [1, 3, 6, 9, 99], labels=["2-3", "4-6", "7-9", "10+"]).astype(str)
    fit3 = pd.read_csv(os.path.join(OUT, "fit3.csv"))
    rng = np.random.default_rng(0)
    tables = []
    for version in ("start network", "stage's network"):
        pi = df[f"p intuition ({version})"].values
        pf = df[f"p flat ({version})"].values
        pa = df[f"p AND-OR ({version})"].values
        eps = fit3[fit3.version == version].eps.iloc[0]
        groups = {}
        for a, g in df.groupby("after loop"):
            groups[(a, "all", "all")] = g.index.values
            for by in ("dist", "speed", "kind"):
                if by == "kind" and a == "elsewhere":
                    continue
                for level, h in g.groupby(by):
                    groups[(a, by, level)] = h.index.values
        t = weights(df, pi, pf, pa, eps, groups, rng).assign(version=version, eps=eps)
        tables.append(t)
        print(f"== {version} (eps {eps})")
        print(t.drop(columns=["version"]).round(3).to_string(index=False), flush=True)
    pd.concat(tables).to_csv(os.path.join(OUT, "loops.csv"), index=False)
    print(df["after loop"].value_counts().to_dict())
    print("saved to", OUT)


def slow_groups(df):
    """Per mixture move: "slow" (itself), "1" / "2-3" (a fast move that many moves after the attempt's last slow
    move), "other fast"; and whether that slow move opened the attempt."""
    prof = pd.read_csv(os.path.join(paths.MODELS, "summary", "human_profile", "moves.csv.gz"), dtype={"who": str})
    prof = prof[(prof.source == "people") & prof.seconds.notna()].sort_values(["who", "instance", "attempt", "move"])
    prof["is slow"] = prof.seconds > SLOW * prof.groupby("who").seconds.transform("median")
    keys = ["who", "instance", "attempt"]
    since, opener = [], []
    for _, g in prof.groupby(keys, sort=False):
        last = None
        for t, slow in zip(g.move.values, g["is slow"].values):
            since.append(None if last is None else t - last)
            opener.append(None if last is None else ("opening" if last == 0 else "mid-attempt"))
            if slow:
                last = t
    prof = prof.assign(since=since, opener=opener)
    out = df.merge(prof[keys + ["move", "since", "opener", "is slow"]], on=keys + ["move"], how="left",
                   validate="1:1")
    out["after slow"] = np.select([out["is slow"] == True, out.since == 1, out.since.isin([2, 3])],  # noqa: E712
                                  ["slow", "1", "2-3"], "other fast")
    out.loc[out["after slow"].isin(["slow", "other fast"]), "opener"] = None
    return out


def main_slow():
    saved = pd.read_csv(os.path.join(OUT, "moves.csv.gz"), dtype={"who": str})
    df = slow_groups(saved)
    df["dist"] = pd.cut(df.distance, [1, 3, 6, 9, 99], labels=["2-3", "4-6", "7-9", "10+"]).astype(str)
    fit3 = pd.read_csv(os.path.join(OUT, "fit3.csv"))
    rng = np.random.default_rng(0)
    tables = []
    for version in ("start network", "stage's network"):
        pi = df[f"p intuition ({version})"].values
        pf = df[f"p flat ({version})"].values
        pa = df[f"p AND-OR ({version})"].values
        eps = fit3[fit3.version == version].eps.iloc[0]
        groups = {}
        for a, g in df.groupby("after slow"):
            groups[(a, "all", "all")] = g.index.values
            for by in ("dist", "opener"):
                if by == "opener" and a in ("slow", "other fast"):
                    continue
                for level, h in g.groupby(by):
                    groups[(a, by, level)] = h.index.values
        t = weights(df, pi, pf, pa, eps, groups, rng, first="after slow").assign(version=version, eps=eps)
        tables.append(t)
        print(f"== {version} (eps {eps})")
        print(t.drop(columns=["version"]).round(3).to_string(index=False), flush=True)
    pd.concat(tables).to_csv(os.path.join(OUT, "slow.csv"), index=False)
    print(df["after slow"].value_counts().to_dict())
    print("saved to", OUT)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, parents, children = records()
    df["stage"] = pd.cut(df.trial, [0, 9, 27, 45, 1000], labels=list(fm.SNAP_STAGES))
    df["speed"] = np.where(df.seconds > SLOW * df.groupby("who").seconds.transform("median"), "slow", "fast")
    pens = [p for _, p in children]
    print(f"{len(df)} moves from {df.who.nunique()} people", flush=True)

    # the start network for every stage
    logits, ends = evaluate(paths.START, bundle(parents, children), device)
    p_int, p_flat = probabilities(logits, ends, pens, df.chosen.values)
    # each stage's own network: the calibrated intuition session's snapshots, 8 students averaged
    snaps = fm.snapshot_networks()
    s_int, s_flat = p_int.copy(), {k: v.copy() for k, v in p_flat.items()}
    for st, n in fm.SNAP_OF.items():
        if n == 0:
            continue
        idx = np.nonzero((df.stage == st).values)[0]
        sel = lambda xs: [xs[i] for i in idx]  # noqa: E731
        acc_i, acc_f = np.zeros(len(idx)), {k: np.zeros(len(idx)) for k in p_flat}
        files = [f for name, f in snaps.items() if name.split()[0] == str(n)]
        bun = bundle([parents[i] for i in idx], [children[i] for i in idx])
        for f in files:
            lg, en = evaluate(f, bun, device)
            pi, pf = probabilities(lg, en, sel(pens), df.chosen.values[idx])
            acc_i += pi / len(files)
            for k in pf:
                acc_f[k] += pf[k] / len(files)
        s_int[idx] = acc_i
        for k in acc_f:
            s_flat[k][idx] = acc_f[k]
        print(f"stage {st}: {len(files)} snapshots after {n} trials", flush=True)

    rng = np.random.default_rng(0)
    tables, people = [], []
    moves = df.drop(columns=["chosen"]).copy()
    for version, pi, pfs in [("start network", p_int, p_flat), ("stage's network", s_int, s_flat)]:
        total, key, eps, ws = fit(df, pi, pfs)
        print(f"{version}: consider {key[0]}, tau {key[1]}, eps {eps}; w by stage {ws}", flush=True)
        t = stage_table(df, pi, pfs[key], eps, rng).assign(version=version, consider=key[0], tau=key[1], eps=eps)
        tables.append(t)
        people.append(per_person(df, pi, pfs[key], eps).assign(version=version))
        moves[f"p intuition ({version})"] = pi
        moves[f"p flat ({version})"] = pfs[key]
        print(t.round(3).to_string(index=False), flush=True)
        pp = people[-1]
        print(f"per person ({len(pp)}): w early {pp['w early'].mean():.2f}, late {pp['w late'].mean():.2f}; "
              f"rose for {(pp['w late'] > pp['w early']).sum()}, fell for {(pp['w late'] < pp['w early']).sum()}",
              flush=True)
    pd.concat(tables).to_csv(os.path.join(OUT, "fit.csv"), index=False)
    pd.concat(people).to_csv(os.path.join(OUT, "people.csv"), index=False)
    moves.to_csv(os.path.join(OUT, "moves.csv.gz"), index=False)
    print("saved to", OUT)


if __name__ == "__main__":
    {"andor": main_andor, "loops": main_loops, "slow": main_slow}.get(sys.argv[1] if sys.argv[1:] else "", main)()
