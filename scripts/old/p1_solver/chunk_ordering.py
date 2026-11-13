"""Chunks as ordering knowledge: does recognizing a familiar sub-plan, and
trying it first, bring and_or toward the oracle ordering of search_control?

Mining: lapse-free and_or passes over training puzzles, recording the plan
tree each pass actually builds (the OrNodes and actions of its successful
branch). Two sources:
- "opt": the search ordered by search_control's oracle (how early each car
  reaches a destination on an optimal solution) - plans an expert would form;
- "self": the search's own random ordering - what the solver already does.
Every OrNode subtree of up to 6 moves seen in >= 3 training puzzles becomes a
template (chunk_library), including single moves.

Evaluation: held-out puzzles, where each OrNode orders its candidate moves by
ChunkLibrary.rank (recognized first, largest/most-supported template first,
rest random) instead of randomly. Lapse checks are charged exactly as before;
recognition is free. Libraries are learned from the first n training
puzzles for several n, as a learning curve."""
import argparse
import collections
import random
import statistics
import time
from multiprocessing import Pool

import and_or as ao
import chunk_library as cl
import chunks
from lib import multi_bfs, parse_puzzle_line
from search_control import load_puzzles, oracle_heuristic

_or_solve = ao.OrNode.solve
_and_solve = ao.AndNode.solve
_lapse_move = ao.lapse_move

# --- mining: record the plan tree of each lapse-free pass -------------------

STACK, ROOTS, COUNT = [], [], [0]
RECORDING = [False]
MAX_CHECKS = 5_000


class _Exhausted(Exception):
    pass


def _recording_or_solve(self):
    if not RECORDING[0]:
        return _or_solve(self)
    COUNT[0] += 1
    if COUNT[0] > MAX_CHECKS:
        raise _Exhausted
    frame = {"node": self, "attempts": []}
    parent = STACK[-1] if STACK else None
    STACK.append(frame)
    try:
        result = _or_solve(self)
    finally:
        STACK.pop()
    frame["ok"] = result is not None
    if parent is not None:
        parent["children"].append(frame)
    return result


def _recording_and_solve(self):
    if not RECORDING[0]:
        return _and_solve(self)
    frame = {"node": self, "children": []}
    parent = STACK[-1] if STACK else None
    STACK.append(frame)
    try:
        result = _and_solve(self)
    finally:
        STACK.pop()
    frame["ok"] = result is not None
    (parent["attempts"] if parent is not None else ROOTS).append(frame)
    return result


def _plan_children(and_frame):
    """The OrNode frames of a successful AndNode's winning blocker order: its
    last k OrNode calls, k = its blocker count (it returns right after)."""
    k = len(and_frame["node"].blockers())
    return and_frame["children"][len(and_frame["children"]) - k:] if k else []


def _tree(or_frame):
    chosen = next(a for a in reversed(or_frame["attempts"]) if a["ok"])
    node = or_frame["node"]
    return {"car": node.car_name, "state": node.state, "collisions": node.collisions,
            "action": (chosen["node"].direction, chosen["node"].steps),
            "children": [_tree(c) for c in _plan_children(chosen)]}


def capture(state, heuristic, max_moves=200):
    """Annotated plan trees (chunks.py format) of the lapse-free passes that
    drive red out from `state`; a pass with no plan plays a random move."""
    forest, moves = [], 0
    while ao._red_steps(state) > 0 and moves < max_moves:
        STACK.clear()
        ROOTS.clear()
        COUNT[0] = 0
        RECORDING[0] = True
        try:
            result = ao._solve_pass(state, heuristic, 0.0)
        except _Exhausted:
            result = None
        finally:
            RECORDING[0] = False
        root = next((r for r in reversed(ROOTS) if r["ok"]), None)
        if result is None or root is None:
            move = random.choice(ao.legal_moves(state))
            state, step = ao.AndNode(state, *move).apply(state), [move]
        else:
            step = result if type(result) is list else result[1]
            tree = {"car": "red", "state": root["node"].state, "collisions": frozenset(),
                    "action": (root["node"].direction, root["node"].steps),
                    "children": [_tree(c) for c in _plan_children(root)]}
            chunks.annotate(tree)
            forest.append(tree)
            for move in step:
                state = ao.AndNode(state, *move).apply(state)
        moves += len(step)
    return forest


def mine_one(job):
    idx, state, source, seed = job
    random.seed(seed)
    heuristic = oracle_heuristic(multi_bfs(state)) if source == "opt" else None
    return idx, capture(state, heuristic)


# --- evaluation: chunk-ordered search ----------------------------------------

LIBRARY = [None]


def _random_lapse_move(state, heuristic=None):
    return _lapse_move(state, None)


def _ordered_or_solve(self):
    """OrNode.solve with candidates ordered by LIBRARY.rank instead of randomly."""
    if LIBRARY[0] is None or self.parent_direction is None:
        return _or_solve(self)
    key = (self.car_name, self.collisions)
    if key in self.visited:
        return None
    next_visited = self.visited | {key}
    if random.random() < self.gamma:
        move = ao.lapse_move(self.state, self.heuristic)
        if move is None:
            return None
        raise ao.GammaLapse(ao.AndNode(self.state, *move).apply(self.state), [move])
    for direction, steps in LIBRARY[0].rank(self, self.directions(), next_visited):
        result = ao.AndNode(self.state, self.car_name, direction, steps, next_visited, self.protected,
                            self.heuristic, self.gamma, self.library).solve()
        if result is not None:
            return result
    return None


def _init_eval(libraries):
    ao.lapse_move = _random_lapse_move
    ao.OrNode.solve = _ordered_or_solve
    EVAL_LIBRARIES.update(libraries)


EVAL_LIBRARIES = {}


def run_one(job):
    name, state, distances, gamma, max_steps, seed = job
    random.seed(seed)
    LIBRARY[0] = EVAL_LIBRARIES.get(name)
    heuristic = oracle_heuristic(distances) if name == "oracle" else None
    moves = 0
    while moves < max_steps and ao._red_steps(state) > 0:
        attempt = ao.solve(state, heuristic=heuristic, gamma=gamma)
        step = attempt if type(attempt) is list else attempt[1]
        if not step:
            break
        for move in step:
            state = ao.AndNode(state, *move).apply(state)
        moves += len(step)
    return ao._red_steps(state) <= 0, moves


def sample_train(path, bins, per_bin, exclude, seed):
    by_bin = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            d, state = parse_puzzle_line(line)[:2]
            if state in exclude:
                continue
            for lo, hi in bins:
                if lo <= d <= hi:
                    by_bin[(lo, hi)].append(state)
    rng = random.Random(seed + 1)
    picked = [s for b in bins for s in rng.sample(by_bin[b], min(per_bin, len(by_bin[b])))]
    rng.shuffle(picked)
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="../data/rush_nw.txt")
    ap.add_argument("--train-per-bin", type=int, default=200, help="training puzzles per distance bin")
    ap.add_argument("--sizes", type=int, nargs="+", default=[30, 100, 300, 1000],
                    help="training puzzles each 'opt' library is learned from (learning curve)")
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--per-bin", type=int, default=40, help="held-out puzzles per distance bin")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.07, 0.2])
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bins = [(5, 8), (9, 12), (13, 16), (17, 20), (21, 30)]
    test = load_puzzles(args.pool, bins, args.per_bin, args.seed)
    exclude = {s for b in bins for s in test[b]}
    train = sample_train(args.pool, bins, args.train_per_bin, exclude, args.seed)
    n_train = min(max(args.sizes), len(train))
    train = train[:n_train]

    with Pool() as pool:
        forests = {}
        for source in ("opt", "self"):
            t0 = time.time()
            jobs = [(i, s, source, hash((args.seed, source, i)) & 0xFFFFFFFF) for i, s in enumerate(train)]
            forests[source] = dict(pool.map(mine_one, jobs, chunksize=2))
            trees = sum(len(f) for f in forests[source].values())
            print(f"mined {source}: {trees} plan trees from {n_train} training puzzles ({time.time() - t0:.0f}s)",
                  flush=True)

    libraries = {}
    for n in args.sizes:
        split = [(i, forests["opt"][i]) for i in range(min(n, n_train))]
        libraries[f"opt@{n}"] = cl.ChunkLibrary(cl.flat_templates(split, 1, 6, args.min_support), min_moves=1)
    split = [(i, forests["self"][i]) for i in range(n_train)]
    libraries[f"self@{n_train}"] = cl.ChunkLibrary(cl.flat_templates(split, 1, 6, args.min_support), min_moves=1)
    print(", ".join(f"{k}: {v.size} templates" for k, v in libraries.items()), flush=True)

    configs = ["base", "oracle"] + list(libraries)
    with Pool(initializer=_init_eval, initargs=(libraries,)) as pool:
        distances = {b: pool.map(multi_bfs, test[b]) for b in bins}
        for gamma in args.gammas:
            print(f"\n=== gamma = {gamma}, {args.per_bin} held-out puzzles x {args.runs} runs per bin, "
                  f"cap {args.max_steps} moves: solved (moves/solve) ===", flush=True)
            print(f"{'config':<10} " + " ".join(f"{f'{lo}-{hi}':>14}" for lo, hi in bins), flush=True)
            for name in configs:
                row = []
                for b in bins:
                    jobs = [(name, s, distances[b][i] if name == "oracle" else None, gamma, args.max_steps,
                             hash((args.seed, b, i, r)) & 0xFFFFFFFF)
                            for i, s in enumerate(test[b]) for r in range(args.runs)]
                    res = pool.map(run_one, jobs, chunksize=2)
                    lengths = [m for ok, m in res if ok]
                    row.append(f"{len(lengths) / len(res):>5.0%} ({statistics.mean(lengths) if lengths else 0:>5.1f})")
                print(f"{name:<10} " + " ".join(f"{c:>14}" for c in row), flush=True)


ao.OrNode.solve = _recording_or_solve
ao.AndNode.solve = _recording_and_solve

if __name__ == "__main__":
    main()
