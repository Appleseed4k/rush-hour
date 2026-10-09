"""Does ordering or bounding the AND-OR search bring passes within the lapse
budget? Solve rate of and_or.solve at fixed gamma under:

- ordering: a `heuristic` that tries candidates best-first instead of in
  random order - "local" (fewest blockers, then fewest steps: try the move
  that's already free) or "oracle" (where the car sits in the states nearest
  the goal, from the puzzle's full BFS - a ceiling for any learned ordering).
  Ties are broken at random. Lapse moves stay uniformly random either way.
- bounding: "depth<D>" never opens a subgoal more than D OrNodes deep (no
  check spent on it); "cap<N>" gives a pass up after N lapse checks, handled
  exactly like a lapse (completed moves are kept, plus one random move).

Wraps and_or from outside; and_or itself is unchanged."""
import argparse
import collections
import random
import statistics
import time
from multiprocessing import Pool

import and_or as ao
from lib import car_orientations, multi_bfs, neighbors, parse_puzzle_line

CONFIG = {"depth": None, "cap": None}
STATS = collections.Counter()
DEPTH = [0]
PASS_CHECKS = [0]
IN_LAPSE = [False]
_or_solve = ao.OrNode.solve
_lapse_move = ao.lapse_move
_legal_moves = ao.legal_moves
_solve_pass = ao._solve_pass


def _random_lapse_move(state, heuristic=None):
    STATS["lapse_moves"] += 1  # lapses and give-ups both play one
    IN_LAPSE[0] = True
    try:
        return _lapse_move(state, None)
    finally:
        IN_LAPSE[0] = False


def _flagging_legal_moves(state):
    if not IN_LAPSE[0]:
        STATS["fallback_pass"] = 1  # _solve_pass found no plan
    return _legal_moves(state)


def _bounded_or_solve(self):
    if (self.car_name, self.collisions) in self.visited:
        return None
    if CONFIG["depth"] is not None and DEPTH[0] >= CONFIG["depth"]:
        return None
    if CONFIG["cap"] is not None and PASS_CHECKS[0] >= CONFIG["cap"]:
        STATS["giveups"] += 1
        move = ao.lapse_move(self.state)
        if move is None:
            return None
        raise ao.GammaLapse(ao.AndNode(self.state, *move).apply(self.state), [move])
    PASS_CHECKS[0] += 1
    STATS["checks"] += 1
    DEPTH[0] += 1
    try:
        return _or_solve(self)
    finally:
        DEPTH[0] -= 1


def _counting_solve_pass(*args, **kwargs):
    PASS_CHECKS[0] = 0
    DEPTH[0] = 0
    return _solve_pass(*args, **kwargs)


def install():
    """Patches and_or for this experiment (call in the parent and every worker)."""
    ao.lapse_move = _random_lapse_move
    ao.legal_moves = _flagging_legal_moves
    ao.OrNode.solve = _bounded_or_solve
    ao._solve_pass = _counting_solve_pass


def local_heuristic(state, actions):
    """Fewest blockers first, then fewest steps; random among ties."""
    return {a: -10 * len(ao.AndNode(state, *a).blockers()) - a[2] + random.random() for a in actions}


def oracle_heuristic(distances, max_states=20_000):
    """Scores a candidate by how early its car reaches that destination along
    an optimal solution from the current board (every state on any shortest
    path to the goal, from the puzzle's full BFS `distances`); destinations
    off every optimal path score worst. Cached per board."""
    cache = {}

    def first_reached(state):
        if state not in cache:
            first = {}
            layer, depth, seen = {state}, 0, 0
            while layer and seen < max_states:
                nxt = set()
                for s in layer:
                    for car in s:
                        first.setdefault(car, depth)
                    for _, t in neighbors(s, orientations):
                        if distances.get(t) == distances[s] - 1:
                            nxt.add(t)
                seen += len(layer)
                layer, depth = nxt, depth + 1
            cache[state] = first
        return cache[state]

    orientations = None

    def heuristic(state, actions):
        nonlocal orientations
        orientations = orientations or car_orientations(state)
        if state not in distances:  # mid-plan boards are always reachable, but be safe
            return {a: random.random() for a in actions}
        first = first_reached(state)
        scores = {}
        for a in actions:
            moved = dict(ao.AndNode(state, *a).apply(state))[a[0]]
            scores[a] = -first.get((a[0], moved), 1 << 10) + random.random()
        return scores
    return heuristic


def parse_config(name):
    """"base", or "+"-joined parts: local / oracle / depth<D> / cap<N>."""
    order, depth, cap = None, None, None
    for part in ([] if name == "base" else name.split("+")):
        if part in ("local", "oracle"):
            order = part
        elif part.startswith("depth"):
            depth = int(part[5:])
        elif part.startswith("cap"):
            cap = int(part[3:])
        else:
            raise ValueError(name)
    return order, depth, cap


def run_one(job):
    name, state, values, gamma, max_steps, seed = job
    order, CONFIG["depth"], CONFIG["cap"] = parse_config(name)
    heuristic = {"local": local_heuristic, "oracle": oracle_heuristic(values) if values else None}.get(order)
    random.seed(seed)
    STATS.clear()
    moves, passes, plan_passes = 0, 0, 0
    while moves < max_steps and ao._red_steps(state) > 0:
        STATS["fallback_pass"] = 0
        before = STATS["lapse_moves"]
        attempt = ao.solve(state, heuristic=heuristic, gamma=gamma)
        step = attempt if type(attempt) is list else attempt[1]
        if not step:
            break
        passes += 1
        # A pass ends in a lapse, a give-up, a random fallback, or a plan.
        plan_passes += STATS["lapse_moves"] == before and not STATS["fallback_pass"]
        for move in step:
            state = ao.AndNode(state, *move).apply(state)
        moves += len(step)
    return ao._red_steps(state) <= 0, moves, passes, plan_passes, STATS["checks"], STATS["giveups"]


def load_puzzles(path, bins, per_bin, seed):
    by_bin = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            d = int(line.split(maxsplit=1)[0])
            for lo, hi in bins:
                if lo <= d <= hi:
                    by_bin[(lo, hi)].append(line)
    rng = random.Random(seed)
    return {b: [parse_puzzle_line(l)[1] for l in rng.sample(by_bin[b], min(per_bin, len(by_bin[b])))] for b in bins}


def _values(state):
    return multi_bfs(state)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="../data/rush_nw.txt")
    ap.add_argument("--configs", nargs="+",
                    default=["base", "local", "oracle", "depth3", "depth5", "cap10", "cap30", "local+cap30",
                             "oracle+cap30"])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.07, 0.2])
    ap.add_argument("--per-bin", type=int, default=40, help="puzzles per distance bin")
    ap.add_argument("--runs", type=int, default=3, help="attempts per puzzle")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bins = [(5, 8), (9, 12), (13, 16), (17, 20), (21, 30)]
    puzzles = load_puzzles(args.pool, bins, args.per_bin, args.seed)
    need_oracle = any("oracle" in c for c in args.configs)
    install()
    with Pool(initializer=install) as pool:
        values = {}
        if need_oracle:
            for b in bins:
                values[b] = pool.map(_values, puzzles[b])
        for gamma in args.gammas:
            print(f"\n=== gamma = {gamma} (budget {ao.or_budget(gamma)}), {args.per_bin} puzzles x {args.runs} runs "
                  f"per bin, cap {args.max_steps} moves ===", flush=True)
            print(f"{'dist':>5} | {'config':<12} | {'solved':>6} | {'moves/solve':>11} | {'checks/pass':>11} | "
                  f"{'plan passes':>11} | {'give-ups/try':>12} | secs", flush=True)
            for b in bins:
                for name in args.configs:
                    jobs = [(name, s, values[b][i] if "oracle" in name else None, gamma, args.max_steps,
                             hash((args.seed, b, i, r)) & 0xFFFFFFFF)
                            for i, s in enumerate(puzzles[b]) for r in range(args.runs)]
                    t0 = time.time()
                    res = pool.map(run_one, jobs, chunksize=2)
                    lengths = [m for ok, m, *_ in res if ok]
                    passes = sum(r[2] for r in res) or 1
                    print(f"{b[0]:>2}-{b[1]:<2} | {name:<12} | {len(lengths) / len(res):>6.0%} | "
                          f"{statistics.mean(lengths) if lengths else float('nan'):>11.1f} | "
                          f"{sum(r[4] for r in res) / passes:>11.1f} | {sum(r[3] for r in res) / passes:>11.0%} | "
                          f"{statistics.mean(r[5] for r in res):>12.1f} | {time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
