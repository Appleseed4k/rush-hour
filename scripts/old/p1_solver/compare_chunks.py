"""Solve rate of and_or.solve at fixed gamma on held-out puzzles, with no chunk
library vs. the flat (chunks.py) and nested (nested_chunks.py) libraries
learned on the training split, consulted inside the search via
chunk_library.ChunkLibrary."""
import argparse
import collections
import random
import statistics
import time
from multiprocessing import Pool

import and_or as ao
import chunk_library
import chunks

# Lapses fired and lapse checks made (OrNode.solve calls) during one run.
# "chunks/try" counts recognitions, including ones inside branches the search
# later abandons.
COUNTS = collections.Counter()
_lapse_move = ao.lapse_move
_or_solve = ao.OrNode.solve


def _counting_lapse_move(state, heuristic=None):
    COUNTS["lapses"] += 1
    return _lapse_move(state, heuristic)


def _counting_or_solve(self):
    COUNTS["checks"] += 1
    return _or_solve(self)


ao.lapse_move = _counting_lapse_move
ao.OrNode.solve = _counting_or_solve

LIBRARIES = {}


PROTECT_MOVERS = [ao.PROTECT_MOVERS]


def _init(libraries, protect_movers=ao.PROTECT_MOVERS):
    LIBRARIES.update(libraries)
    PROTECT_MOVERS[0] = protect_movers


def run_one(args):
    kind, state, gamma, max_steps, seed = args
    random.seed(seed)
    library = LIBRARIES.get(kind)
    if library is not None:
        library.reset_stats()
    COUNTS.clear()
    moves = []
    while len(moves) < max_steps and ao._red_steps(state) > 0:
        attempt = ao.solve(state, gamma=gamma, library=library, protect_movers=PROTECT_MOVERS[0])
        step_moves = attempt if type(attempt) is list else attempt[1]
        if not step_moves:
            break
        for move in step_moves:
            state = ao.AndNode(state, *move).apply(state)
        moves.extend(step_moves)
    solved = ao._red_steps(state) <= 0
    recognized = sum(library.uses.values()) if library is not None else 0
    return solved, len(moves), COUNTS["lapses"], COUNTS["checks"], recognized


def distance_bin(distance, edges):
    lo = max((e for e in edges if e <= distance), default=0)
    hi = min((e for e in edges if e > distance), default=None)
    return f"{lo}-{hi - 1}" if hi is not None else f"{lo}+"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="../data/rush_nw.txt")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--max-expansions", type=int, default=1_000)
    ap.add_argument("--cache", help="chunks.solve_and_split cache (pickle)")
    ap.add_argument("--max-chunks", type=int, default=500, help="nested library size")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.07, 0.2])
    ap.add_argument("--libraries", nargs="+", default=["none", "flat", "nested"])
    ap.add_argument("--protect-movers", action=argparse.BooleanOptionalAction, default=ao.PROTECT_MOVERS,
                    help="and_or.solve's protect_movers, for both mining and solving "
                         "(default: and_or.PROTECT_MOVERS)")
    ap.add_argument("--runs", type=int, default=5, help="attempts per held-out puzzle")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--bins", type=int, nargs="+", default=[1, 9, 13, 17], help="distance bin edges")
    args = ap.parse_args()

    entries, train, test = chunks.solve_and_split(args.pool, args.count, args.seed, args.max_expansions,
                                                  args.test_frac, args.cache, args.protect_movers)
    libraries = {}
    if "flat" in args.libraries:
        libraries["flat"] = chunk_library.build(train, "flat")
    if "nested" in args.libraries:
        libraries["nested"] = chunk_library.build(train, "nested", max_chunks=args.max_chunks)
    print(", ".join(f"{k}: {v.size} templates" for k, v in libraries.items()))

    puzzles = collections.defaultdict(list)
    for idx, _ in test:
        distance, state, _line = entries[idx]
        puzzles[distance_bin(distance, args.bins)].append((idx, state))
    bins = sorted(puzzles, key=lambda b: int(b.split("-")[0].rstrip("+")))
    print("held-out puzzles per distance bin: " + ", ".join(f"{b}: {len(puzzles[b])}" for b in bins))

    with Pool(initializer=_init, initargs=(libraries, args.protect_movers)) as pool:
        for gamma in args.gammas:
            print(f"\n=== gamma = {gamma} (budget {ao.or_budget(gamma)}), {args.runs} runs per puzzle, "
                  f"cap {args.max_steps} moves ===")
            print(f"{'dist':>5} | {'library':<7} | {'solved':>6} | {'moves/solve':>11} | {'lapses/try':>10} | "
                  f"{'checks/try':>10} | {'chunks/try':>10} | secs")
            for b in bins:
                for kind in args.libraries:
                    jobs = [(kind, state, gamma, args.max_steps, hash((args.seed, idx, r)) & 0xFFFFFFFF)
                            for idx, state in puzzles[b] for r in range(args.runs)]
                    t0 = time.time()
                    results = pool.map(run_one, jobs, chunksize=4)
                    lengths = [n for ok, n, *_ in results if ok]
                    print(f"{b:>5} | {kind:<7} | {len(lengths) / len(results):>6.1%} | "
                          f"{statistics.mean(lengths) if lengths else float('nan'):>11.1f} | "
                          f"{statistics.mean(r[2] for r in results):>10.1f} | "
                          f"{statistics.mean(r[3] for r in results):>10.1f} | "
                          f"{statistics.mean(r[4] for r in results):>10.1f} | {time.time() - t0:.0f}",
                          flush=True)


if __name__ == "__main__":
    main()
