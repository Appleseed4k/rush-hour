"""How good is the dest head's relevance map, and does it sharpen with training? For every saved epoch of the AND-OR
apprentice (and the previous start, for reference), on the apprentice's held-out test puzzles.

A feature is (car, position): a car standing somewhere it isn't now. The dest head scores every such feature on a
board (labels._destinations); breadth-first search gives the truth: the optimal-route DAG from the board (every
next board one move closer), and for each feature the fewest moves k at which any optimal route first puts that car
there (None if no optimal route does). The head's training target is 0.8^(k-1) for the route it saw.

Measures, per board, then averaged:
- on-route AUC: do features on some optimal route score above those on none (0.5 = chance)?
- next-move AUC: do features an optimal next move reaches (k = 1) score above the rest?
- far AUC: do features an optimal route first reaches 3 or more moves on score above off-route ones?
- precision: of the n top-scored features (n = the board's on-route features), the share on a route;
- spearman: rank correlation of the scores with the true target 0.8^(k-1) (0 off-route);
- policy optimal: the share of boards where the policy head's top legal move is optimal (the proposals).

Boards: each puzzle's start, and boards reached from it by 3 and by 8 random legal moves (2 each) - off the route,
as a student's boards mostly are.

    python experiments/dest_quality.py               # writes models/summary/dest_quality.csv
    python experiments/dest_quality.py --session models/sessions/aostart_free_m10_s2_agency1
                                                     # over a session's snapshots: dest_quality_session.csv
"""
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import board, paths  # noqa: E402
from rushhour.student import PolicyAgent  # noqa: E402
from rushhour.student.labels import DEST_DECAY, _destinations, _move_index  # noqa: E402

FOLDER = os.path.join(paths.APPRENTICE, "gamma0.07_free")
OUT = os.path.join(paths.MODELS, "summary", "dest_quality.csv")
WANDER = {"start": [0], "3 random moves": [3, 3], "8 random moves": [8, 8]}


def test_puzzles():
    config = json.load(open(os.path.join(FOLDER, "config.json")))
    return board.sample_by_distance(board.load_puzzle_pool(), config["bins"], 20, seed=config["seed"])


def wander(state, n, rng):
    for _ in range(n):
        state = rng.choice(list(board.neighbors(state, board.car_orientations(state))))[1]
    return state


def truth(b, dist):
    """{(car, positions): fewest moves at which an optimal route from `b` first puts that car there}, and the
    optimal next moves."""
    orient = board.car_orientations(b)
    here = set(b)
    first, layer, t = {}, {b}, 0
    best_moves = [m for m, s in board.neighbors(b, orient) if dist[s] == dist[b] - 1]
    while layer and dist[next(iter(layer))] > 0:
        t += 1
        nxt = set()
        for s in layer:
            for _, s2 in board.neighbors(s, orient):
                if dist[s2] == dist[s] - 1:
                    nxt.add(s2)
        for s in nxt:
            for entry in s:
                if entry not in here:
                    first.setdefault(entry, t)
        layer = nxt
    return first, best_moves


def auc(scores, positive):
    positive = np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(scores)
    return (r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boards():
    rng = random.Random(0)
    out = []
    for d, s in test_puzzles():
        dist = board.multi_bfs(s)
        for kind, steps in WANDER.items():
            for n in steps:
                b = wander(s, n, rng)
                if dist[b] == 0:
                    continue
                first, best = truth(b, dist)
                out.append({"kind": kind, "puzzle distance": d, "board": b, "distance": dist[b], "first": first,
                            "best": best})
    return out


def measure(agent, items):
    rows = []
    n_cols, max_slide = 6, agent.max_slide
    for it in items:
        b = it["board"]
        sd = dict(b)
        with torch.no_grad():
            policy, dest = agent._evaluate(b)
        feats = list(_destinations(b, (6, 6), max_slide))
        scores = np.array([dest[_move_index(sd, car, dr, st, n_cols, max_slide)].item() for car, dr, st, _ in feats])
        k = [it["first"].get((car, moved)) for car, _, _, moved in feats]
        on = np.array([x is not None for x in k])
        target = np.array([DEST_DECAY ** (x - 1) if x is not None else 0.0 for x in k])
        legal = [m for m, _ in board.neighbors(b, board.car_orientations(b))]
        top = max(legal, key=lambda m: policy[_move_index(sd, *m, n_cols, max_slide)].item())
        n = int(on.sum())
        rows.append({"kind": it["kind"], "distance": it["distance"], "features": len(feats), "on route": n,
                     "on-route AUC": auc(scores, on), "next-move AUC": auc(scores, target == 1.0),
                     "far AUC": auc(scores[(~on) | (target <= DEST_DECAY ** 2)],
                                    on[(~on) | (target <= DEST_DECAY ** 2)]),
                     "precision": on[np.argsort(-scores)[:n]].mean() if n else np.nan,
                     "spearman": spearmanr(scores, target).statistic if n else np.nan,
                     "policy optimal": float(top in it["best"])})
    return pd.DataFrame(rows)


MEASURES = ["on-route AUC", "next-move AUC", "far AUC", "precision", "spearman", "policy optimal"]


def session(run, items):
    """The same over a session: each teacher's students at every saved snapshot (student*_puzzle<n>.pt), averaged
    over students, from the start (puzzle 0)."""
    start = measure(PolicyAgent.load(paths.START), items)[MEASURES].mean()
    rows = []
    for teacher in sorted(os.listdir(run)):
        snaps = glob.glob(os.path.join(run, teacher, "student*_puzzle*.pt"))
        if not snaps:
            continue
        rows.append({"teacher": teacher, "puzzle": 0, **start.to_dict()})
        for n in sorted({int(f.rsplit("puzzle", 1)[1][:-3]) for f in snaps}):
            per = [measure(PolicyAgent.load(f), items)[MEASURES].mean()
                   for f in sorted(glob.glob(os.path.join(run, teacher, f"student*_puzzle{n}.pt")))]
            rows.append({"teacher": teacher, "puzzle": n, **pd.concat(per, axis=1).mean(axis=1).to_dict(),
                         "students": len(per)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    items = boards()
    if "--session" in sys.argv:  # --session <run folder>: models/summary/dest_quality_session.csv
        df = session(sys.argv[sys.argv.index("--session") + 1], items)
        df.to_csv(OUT.replace(".csv", "_session.csv"), index=False)
        print(df.round(3).to_string(index=False))
        sys.exit()
    print(f"{len(items)} boards from {len(test_puzzles())} held-out puzzles")
    manifest = json.load(open(os.path.join(FOLDER, "manifest.json")))
    nets = [(m["file"][:-3], m["round"], m["epoch"], os.path.join(FOLDER, m["file"])) for m in manifest]
    nets.append(("previous start", None, None, paths.PREVIOUS_START))
    out = []
    for name, rnd, ep, path in nets:
        agent = PolicyAgent.load(path)
        df = measure(agent, items)
        for kind, g in [("all", df), *df.groupby("kind")]:
            row = {"network": name, "round": rnd, "epoch": ep, "boards": kind}
            row |= g[MEASURES].mean().to_dict()
            row["base rate"] = (g["on route"] / g["features"]).mean()
            out.append(row)
        print(name, {k: round(float(v), 3) for k, v in out[-4].items() if isinstance(v, (float, np.floating))})
    pd.DataFrame(out).to_csv(OUT, index=False)
    print("wrote", OUT)
