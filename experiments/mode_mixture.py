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
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import board, human_data as hd, paths  # noqa: E402
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
            p, _, v = net.heads(x)
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
    main()
