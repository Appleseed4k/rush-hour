"""Which of the rules' concepts does the student's policy network represent, where, and does that change as it learns?
(Concept discovery, steps 1-2: record the network's features; check that known concepts can be read from them.)

Boards: every board people played 2+ moves from the goal (rule_mining's cache: 18,716 boards over the 70 layouts).
Concepts, per car on a board, read from the features at the car's anchor cell (where the policy head reads its
moves): in the blocking chain; red's blocker (depth 1) rather than deeper; has a move that clears the cells it must
vacate; has a move that frees the car above it; has a move that shrinks the chain; has an optimal move. Per board,
from the features averaged over the cells (as the value head reads them): the true distance; the chain's size.

Features: the policy network's 96 numbers per cell after the stem, after lane block 1 and after lane block 2.
Controls: the raw input along the car's row and column (the 9 input channels of its 11 lane cells: what a readout
gets without the network), and the same layers of the same architecture with random weights (what the structure alone
makes readable). Probes: logistic regression (binary concepts; AUC) or ridge regression (board measures; R^2),
standardized features, 5-fold cross-validation grouped by layout - every score is on layouts the probe never saw.

    python experiments/concept_probes.py            # probes.csv: the start network, every layer, with the controls
    python experiments/concept_probes.py session    # session.csv: block 2 of the start and of the session's snapshots
    python experiments/concept_probes.py strong     # strong.csv, strong_play.csv: stronger networks (longer training,
                                                    # expert-iteration rounds 2-3), each depth with random weights
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rule_mining as rm
from rushhour import board, human_data as hd, paths
from rushhour.student import PolicyAgent
from rushhour.student.network import car_anchor

OUT = os.path.join(paths.MODELS, "summary", "concepts")
SESSION = os.path.join(paths.SESSIONS, "lean_people_all_m10_ao_b2")
CAR_CONCEPTS = ["in chain", "red's blocker (not deeper)", "can clear", "can free the car above", "can shrink the chain",
                "has an optimal move"]
BOARD_CONCEPTS = ["distance", "chain size"]


def labels():
    """(boards, layout per board, car rows: board index, car, anchor cell, concepts; board rows: distance, chain size)."""
    moves, extra = rm._people_cached()
    seen, boards, layout = {}, [], []
    for inst, (b, _, _) in zip(moves.instance, extra):
        if b not in seen:
            seen[b] = len(boards)
            boards.append(b)
            layout.append(inst)
    dist = {}
    cars, per_board = [], []
    for i, (b, inst) in enumerate(zip(boards, layout)):
        if inst not in dist:
            dist[inst] = board.multi_bfs(hd.puzzle(inst))
        d = dist[inst]
        orient = board.car_orientations(b)
        depth, _ = rm.chain(b)
        tags = {}
        for (car, _, _), nb, m in rm.move_masks(b, orient):
            t = tags.setdefault(car, {"clears": False, "frees": False, "chain_fewer": False, "optimal": False})
            t["clears"] |= bool(m & rm.BIT["clears"])
            t["frees"] |= bool(m & rm.BIT["frees"])
            t["chain_fewer"] |= bool(m & rm.BIT["chain_fewer"])
            t["optimal"] |= d[nb] < d[b]
        for car, _ in b:
            if car == "red":
                continue
            t = tags.get(car, {"clears": False, "frees": False, "chain_fewer": False, "optimal": False})
            dep = depth.get(car)
            cars.append({"board": i, "layout": inst, "car": car, "anchor": car_anchor(b, car),
                         "in chain": dep is not None, "red's blocker (not deeper)": dep == 1 if dep else np.nan,
                         "can clear": t["clears"] if dep else np.nan, "can free the car above": t["frees"],
                         "can shrink the chain": t["chain_fewer"], "has an optimal move": t["optimal"]})
        per_board.append({"board": i, "layout": inst, "distance": d[b], "chain size": len(depth) - 1})
    return boards, pd.DataFrame(cars), pd.DataFrame(per_board)


def layers(net, x):
    """{layer: (n, 96, 6, 6)} after the stem and after each lane block."""
    with torch.no_grad():
        x = torch.cat([x, net.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        h = F.relu(net.stem(x))
        out = {"stem": h}
        for k, block in enumerate(net.blocks, 1):
            h = block(h)
            out[f"block {k}"] = h
    return out


def features(net, enc, cars, n_boards):
    """{layer: (car features (n_cars, 96), board features (n_boards, 96))}."""
    net.eval()
    rows, cols = cars.anchor.map(lambda a: a[0]).values, cars.anchor.map(lambda a: a[1]).values
    acc = {}
    for k in range(0, n_boards, 2048):
        for name, h in layers(net, enc[k:k + 2048]).items():
            acc.setdefault(name, []).append(h.numpy())
    out = {}
    for name, parts in acc.items():
        h = np.concatenate(parts)
        out[name] = (h[cars.board.values, :, rows, cols], h.mean((2, 3)))
    return out


def lane_input(enc, cars):
    """The 9 input channels (with coordinates) of the car's row and column cells: (n_cars, 9 * 11)."""
    coords = PolicyAgent().net.coords.numpy()
    x = np.concatenate([enc.numpy(), np.broadcast_to(coords, (len(enc),) + coords.shape)], axis=1)
    feats = []
    for b, (r, c) in zip(cars.board.values, cars.anchor.values):
        row = x[b, :, r, :]
        col = np.delete(x[b, :, :, c], r, axis=1)
        feats.append(np.concatenate([row.ravel(), col.ravel()]))
    return np.array(feats)


def probe(X, y, groups, binary=True):
    ok = ~pd.isna(y)
    X, y, groups = X[ok], np.asarray(y[ok]).astype(float), np.asarray(groups)[ok]
    scores, pred = [], np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        if binary:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0))
            m.fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        else:
            m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
    return roc_auc_score(y, pred) if binary else r2_score(y, pred)


def run(nets, cars, per_board, enc, controls=True):
    rows = []
    sources = {}
    for name, net in nets.items():
        for layer, (cf, bf) in features(net, enc, cars, len(per_board)).items():
            sources[(name, layer)] = (cf, bf)
    if controls:
        sources[("raw input", "the car's row and column")] = (lane_input(enc, cars), None)
    for (name, layer), (cf, bf) in sources.items():
        for c in CAR_CONCEPTS:
            rows.append({"network": name, "layer": layer, "concept": c, "level": "car", "metric": "AUC",
                         "score": probe(cf, cars[c].values, cars.layout.values),
                         "base rate": float(np.nanmean(cars[c].values.astype(float)))})
        if bf is not None:
            for c in BOARD_CONCEPTS:
                rows.append({"network": name, "layer": layer, "concept": c, "level": "board", "metric": "R2",
                             "score": probe(bf, per_board[c].values, per_board.layout.values, binary=False)})
        print(name, layer, "done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(8)
    boards, cars, per_board = labels()
    ref = PolicyAgent.load(paths.START)
    enc = torch.from_numpy(np.stack([ref.encoder.encode_numpy(b) for b in boards])).float()
    print(f"{len(boards)} boards, {len(cars)} cars", flush=True)
    pd.set_option("display.width", 220)
    if sys.argv[1:2] == ["strong"]:
        import random as _random
        import multiprocessing as mp
        import fit_memory as fm
        import forward
        A = paths.APPRENTICE
        files = {"2 blocks, round 1 epoch 3 (the start)": paths.START,
                 "2 blocks, round 1 epoch 11": os.path.join(A, "gamma0.07_free_novalue_b2", "round1_epoch11.pt"),
                 "2 blocks, after the session (72 trials)": os.path.join(SESSION, "no_teacher", "student00_puzzle720.pt"),
                 "4 blocks, round 2 (expert iteration)": os.path.join(A, "gamma0.07_free", "round2_epoch02.pt"),
                 "4 blocks, round 3 (expert iteration)": os.path.join(A, "gamma0.07_free", "round3_epoch01.pt")}
        nets = {k: PolicyAgent.load(f).net for k, f in files.items()}
        torch.manual_seed(0)
        nets["2 blocks, random weights"] = PolicyAgent(blocks=2).net
        nets["4 blocks, random weights"] = PolicyAgent(blocks=4).net
        df = run(nets, cars, per_board, enc)
        df["last"] = df.apply(lambda r: r.layer == f"block {len(nets[r.network].blocks)}" if r.network in nets
                              else True, axis=1)
        df.to_csv(os.path.join(OUT, "strong.csv"), index=False)
        last = df[df["last"]]
        print(last.pivot_table(index="concept", columns="network", values="score").round(3).to_string())
        # how well each plays by intuition on people's layouts (their first trials' stage)
        weights = {k: open(f, "rb").read() for k, f in files.items()}
        conds = [{"name": k, "kind": "intuition", "snapshot": 0, "net": k, "kw": {"k": 1, "soft_memory": (2, 1)}}
                 for k in files]
        jobs = [(c, i, f"{r}", c["net"]) for c in conds for i in sorted(hd.trials().instance.unique()) for r in range(4)]
        _random.Random(0).shuffle(jobs)
        with mp.Pool(initializer=forward._init, initargs=(weights,)) as pool:
            play = pd.DataFrame(pool.map(fm.job_snapshot, jobs, chunksize=4))
        o, people = fm.summarize_learning(play, {"1-9": (1, 9)})
        o.to_csv(os.path.join(OUT, "strong_play.csv"), index=False)
        print(o[["condition", "optimal 2-3", "optimal 4-6", "optimal 7-9", "optimal 10+", "moves / optimal",
                 "solved"]].round(3).to_string(index=False))
        print(people[["optimal 2-3", "optimal 4-6", "optimal 7-9", "optimal 10+", "moves / optimal"]].round(3))
        sys.exit()
    if sys.argv[1:2] == ["session"]:
        config = json.load(open(os.path.join(SESSION, "config.json")))
        nets = {"start": PolicyAgent.load(config["init"]).net}
        for n in (18, 36, 54):
            for k in range(3):
                f = os.path.join(SESSION, "no_teacher", f"student{k:02d}_puzzle{10 * n:03d}.pt")
                nets[f"after {n} trials, student {k}"] = PolicyAgent.load(f).net
        df = run(nets, cars, per_board, enc, controls=False)
        df = df[df.layer == f"block {ref.blocks}"]
        df.to_csv(os.path.join(OUT, "session.csv"), index=False)
        df["trials"] = df.network.map(lambda s: 0 if s == "start" else int(s.split()[1]))
        print(df.groupby(["concept", "trials"]).score.mean().unstack().round(3).to_string())
        sys.exit()
    torch.manual_seed(0)
    nets = {"student's start": ref.net, "random weights": PolicyAgent(blocks=ref.blocks).net}
    df = run(nets, cars, per_board, enc)
    df.to_csv(os.path.join(OUT, "probes.csv"), index=False)
    print(df.pivot_table(index="concept", columns=["network", "layer"], values="score").round(3).to_string())
    print(cars[CAR_CONCEPTS].astype(float).mean().round(3).to_string())
