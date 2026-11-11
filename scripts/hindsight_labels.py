"""Hindsight label variants scored against exact distances, on attempts by the starting network.

Run from scripts/:  python hindsight_labels.py [per_bin] [processes]   (the notebook used 150, 8)
Plays `per_bin` session puzzles per bin of teach_shortcut_agency1 (its settings: temperature 1, memory 10, cap
max(50, 10 d), giving up at 0.4x people's rates), then labels every solved attempt with each variant:

- own:        the path played with loops cut out (exact repeats only), as before shortcut labels;
- walk:       shortest routes through the moves actually played (W = 1 below);
- W = w:      shortcut labels, noticing a one-move connection between two visited boards only when the student
              was at them within w moves of each other; W = inf is today's "all" labels;
- path:       today's "path" labels (boards on shortest routes from the start, no window);
- near D:     shortcut labels only on boards at most D moves (by the label) from the goal.

Scores per label: is each target move optimal, label distance / true distance, and whether the board lies off
every optimal path from the start (coverage that a fully shown solution never gives)."""
import json, math, os, random, sys
from collections import deque
from multiprocessing import Pool

import numpy as np

import rl
import session_experiment as se
import teach_experiment as te
from lib import bfs, car_orientations, multi_bfs, neighbors

RUN = "teach_shortcut_agency1"
INF = math.inf
VARIANTS = [("own", None), ("walk (W=1)", dict(window=1)), ("W=3", dict(window=3)), ("W=5", dict(window=5)),
            ("W=10", dict(window=10)), ("W=20", dict(window=20)), ("W=inf (all)", dict(window=INF)),
            ("path", dict(window=INF, path=True)), ("near 5", dict(window=INF, near=5)),
            ("near 10", dict(window=INF, near=10)), ("W=10, near 10", dict(window=10, near=10))]

_AGENT = {}


def _init(init):
    import torch
    torch.set_num_threads(1)
    _AGENT["a"] = rl.PolicyAgent.load(init)


def play(job):
    i, d, state, cap, seed, edges, rates, scale = job
    rule = rl.SurrenderRule(edges, rates, {state: d}, scale)
    rng = random.Random(seed)
    r = rl.solve_intuition(_AGENT["a"], state, cap, 10, 1.0, rng=rng, surrender=rule.for_puzzle(state))
    if not r["solved"]:
        return {"i": i, "d": d, "solved": False, "moves": len(r["moves"])}
    dist = multi_bfs(state)
    start = bfs([state], car_orientations(state))  # moves from the start
    out = {"i": i, "d": d, "solved": True, "moves": len(r["moves"]), "visited": len(set(r["boards"]))}
    for name, opts in VARIANTS:
        out[name] = score(labels(r, opts), dist, start, d)
    return out


def labels(r, opts):
    """{board: (label distance, set of target moves)} for a solved attempt."""
    boards, moves = r["boards"], r["moves"]
    if opts is None:  # own: loops cut out
        kept_b, kept_m = rl.erase_loops(boards, moves)
        n = len(kept_m)
        return {b: (n - t, {m}) for t, (b, m) in enumerate(zip(kept_b[:-1], kept_m))}
    window = opts["window"]
    times = {}
    for t, b in enumerate(boards):
        times.setdefault(b, []).append(t)
    orient = car_orientations(boards[0])
    visited = set(boards)

    def gap(a, b):
        return min(abs(x - y) for x in times[a] for y in times[b])

    adj = {b: [(m, n) for m, n in neighbors(b, orient) if n in visited and gap(b, n) <= window] for b in visited}
    goal = boards[-1]
    dist, queue = {goal: 0}, deque([goal])
    while queue:
        b = queue.popleft()
        for _, n in adj[b]:  # moves are reversible, so adjacency is symmetric
            if n not in dist:
                dist[n] = dist[b] + 1
                queue.append(n)
    if opts.get("path"):
        keep, frontier = set(), {boards[0]} if dist.get(boards[0], 0) > 0 else set()
        while frontier:
            keep |= frontier
            frontier = {n for b in frontier for _, n in adj[b] if dist.get(n) == dist[b] - 1 and dist[n] > 0}
    else:
        keep = {b for b in visited if dist.get(b, 0) > 0}
    near = opts.get("near", INF)
    return {b: (dist[b], {m for m, n in adj[b] if dist.get(n) == dist[b] - 1}) for b in keep if dist[b] <= near}


def score(lab, dist, start, d0):
    rows = []
    for b, (k, targets) in lab.items():
        true = dist[b]
        orient = car_orientations(b)
        nxt = dict(neighbors(b, orient))
        opt = np.mean([dist.get(nxt[m]) == true - 1 for m in targets])
        off = start[b] + true != d0
        rows.append((opt, k / true, off))
    return rows


def summarise(results, n_puzzles):
    table = []
    for name, _ in VARIANTS:
        rows = [x for r in results if r["solved"] for x in r[name]]
        if not rows:
            continue
        opt, ratio, off = (np.array(c) for c in zip(*rows))
        table.append({"labels": name, "labels per puzzle": len(rows) / n_puzzles,
                      "off-path labels per puzzle": off.sum() / n_puzzles,
                      "targets optimal": opt.mean(), "distance / true": ratio.mean(),
                      "off-path: targets optimal": opt[off].mean() if off.any() else np.nan,
                      "off-path: distance / true": ratio[off].mean() if off.any() else np.nan,
                      "on-path: targets optimal": opt[~off].mean()})
    return table


if __name__ == "__main__":
    per_bin = int(sys.argv[1]) if len(sys.argv) > 1 else 75
    processes = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    config = json.load(open(os.path.join(se.OUT, RUN, "config.json")))
    session, _ = se.sample_puzzles(config)
    by_bin = {}
    for d, s in session:
        by_bin.setdefault(next(b for b in config["session_bins"] if int(b.split("-")[0]) <= d <= int(b.split("-")[1])),
                          []).append((d, s))
    puzzles = [p for b in config["session_bins"] for p in by_bin[b][:per_bin]]
    edges, rates = config["surrender_table"]
    jobs = [(i, d, s, te.step_cap(d, config), 17 + i, edges, rates, config["surrender_scale"])
            for i, (d, s) in enumerate(puzzles)]
    with Pool(processes, initializer=_init, initargs=(config["init"],)) as pool:
        results = pool.map(play, jobs, chunksize=4)
    solved = [r for r in results if r["solved"]]
    print(f"{len(puzzles)} puzzles, {len(solved)} solved ({len(solved) / len(puzzles):.0%}); "
          f"moves per solved attempt {np.mean([r['moves'] for r in solved]):.1f}, "
          f"distinct boards {np.mean([r['visited'] for r in solved]):.1f}")
    table = summarise(results, len(puzzles))
    out = os.environ.get("LABEL_OUT", os.path.join(se.OUT, "hindsight_labels.json"))
    json.dump({"per_bin": per_bin, "n": len(puzzles), "solved": len(solved), "table": table,
               "by_bin": {b: summarise([r for r in results if int(b.split("-")[0]) <= r["d"] <= int(b.split("-")[1])],
                                       sum(int(b.split("-")[0]) <= r["d"] <= int(b.split("-")[1]) for r in results))
                          for b in config["session_bins"]}},
              open(out, "w"), indent=1, default=float)
    import pandas as pd
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(pd.DataFrame(table).round(3).to_string(index=False))
