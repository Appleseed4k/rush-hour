"""Solve rate / path length / lapses of and_or.solve at different execution
horizons (whole plan vs. k moves per plan, with or without the paper's root
check) and of paper_model.solve_paper, on the same puzzles."""
import argparse
import collections
import random
import statistics
import time
from multiprocessing import Pool

import and_or as ao
import original_model as om
import paper_model as pm
from lib import parse_puzzle_line


# Lapses encountered by and_or's search: every lapse, OrNode or root, plays the
# move lapse_move returns, so wrapping it counts them. `random_moves` then tells
# which of them actually got played (a horizon can cut a lapse's move off), by
# identity: the move object lapse_move hands out is the one carried in the plan.
LAPSES = [0]
LAST_LAPSE_MOVE = [None]
_lapse_move = ao.lapse_move


def _counting_lapse_move(state, heuristic=None):
    LAPSES[0] += 1
    LAST_LAPSE_MOVE[0] = _lapse_move(state, heuristic)
    return LAST_LAPSE_MOVE[0]


ao.lapse_move = _counting_lapse_move


def parse_model(name):
    """"current" = and_or.solve as is; "h<k>" plays k moves per plan before
    replanning; "+root" adds the paper's root stopping check; "+pm" / "+np"
    force protect_movers on / off (default: and_or.PROTECT_MOVERS), e.g.
    "h1+root+np";
    "paper" = paper_model's one-move distribution (my reading of the paper);
    "original" / "original-nosame" = the archive's own rule (original_model),
    replanning after every move, with / without its same-car exclusion."""
    if name in ("paper", "original", "original-nosame"):
        return None
    base, *flags = name.split("+")
    horizon = None if base == "current" else int(base.removeprefix("h"))
    kwargs = {"horizon": horizon, "root_check": "root" in flags}
    if "pm" in flags or "np" in flags:
        kwargs["protect_movers"] = "pm" in flags
    return kwargs


def solve_and_or(state, gamma, max_steps, **solve_kwargs):
    """The notebook's loop (rush_hour_solver.ipynb cell 4): and_or.solve passes
    until red exits. Returns (moves or None, lapses, random_moves)."""
    moves, random_moves = [], 0
    LAPSES[0] = 0
    while len(moves) < max_steps:
        if ao._red_steps(state) <= 0:
            return moves, LAPSES[0], random_moves
        LAST_LAPSE_MOVE[0] = None
        attempt = ao.solve(state, gamma=gamma, **solve_kwargs)
        step_moves = attempt if type(attempt) is list else attempt[1]
        if not step_moves:
            return None, LAPSES[0], random_moves
        random_moves += any(m is LAST_LAPSE_MOVE[0] for m in step_moves)
        for move in step_moves:
            state = ao.AndNode(state, *move).apply(state)
        moves.extend(step_moves)
    return None, LAPSES[0], random_moves


def solve_original(state, gamma, max_steps, rng, exclude_same_car=True):
    """original_model's one-step distribution sampled move by move (given the
    car moved last, as in the archive's gamma_only_model), until red exits.
    Returns (moves or None, lapses, capped): a lapse is the stop mass firing,
    which plays a uniformly random legal move; `capped` counts steps whose tree
    exceeded original_model's visit cap (played as a random move, so also
    counted as lapses)."""
    moves, lapses, capped, prev_car = [], 0, 0, None
    while len(moves) < max_steps:
        if ao._red_steps(state) <= 0:
            return moves, lapses, capped
        try:
            probs, stop, _ = om.action_distribution(state, gamma, prev_car, exclude_same_car)
        except om.TreeTooLarge:  # counted as a lapse, and reported
            capped += 1
            move = rng.choice(ao.legal_moves(state))
            lapses += 1
            state = ao.AndNode(state, *move).apply(state)
            moves.append(move)
            prev_car = move[0]
            continue
        legal = list(probs)
        if rng.random() < stop:
            move, lapses = rng.choice(legal), lapses + 1
        else:
            weights = [max(probs[m] - stop / len(legal), 0.0) for m in legal]
            move = rng.choices(legal, weights=weights)[0] if sum(weights) > 0 else rng.choice(legal)
        state = ao.AndNode(state, *move).apply(state)
        moves.append(move)
        prev_car = move[0]
    return (moves, lapses, capped) if ao._red_steps(state) <= 0 else (None, lapses, capped)


def run_one(args):
    model, state, gamma, max_steps, seed = args
    rng = random.Random(seed)
    random.seed(seed)  # and_or draws from the global `random`
    kwargs, capped = parse_model(model), 0
    if model in ("original", "original-nosame"):
        moves, lapses, capped = solve_original(state, gamma, max_steps, rng, exclude_same_car=model == "original")
        random_moves = lapses
    elif kwargs is not None:
        moves, lapses, random_moves = solve_and_or(state, gamma, max_steps, **kwargs)
    else:
        moves, lapses = pm.solve_paper(state, gamma, max_steps, rng)
        random_moves = lapses
    return moves is not None, len(moves) if moves is not None else max_steps, lapses, random_moves, capped


def load_pool(path, distances, per_distance, seed):
    pool = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            if int(line.split(maxsplit=1)[0]) in distances:
                pool[int(line.split(maxsplit=1)[0])].append(line)
    rng = random.Random(seed)
    return {d: [parse_puzzle_line(l)[1] for l in rng.sample(pool[d], per_distance)] for d in distances}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 9, 12, 14, 16, 20])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.07, 0.2])
    ap.add_argument("--puzzles", type=int, default=20, help="puzzles per distance")
    ap.add_argument("--runs", type=int, default=5, help="attempts per puzzle")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default="../data/rush_nw.txt")
    ap.add_argument("--models", nargs="+", default=["current", "h1", "h1+root", "paper"],
                    help="see parse_model: current, h<k>[+root], paper")
    args = ap.parse_args()

    puzzles = load_pool(args.pool, set(args.distances), args.puzzles, args.seed)
    with Pool() as pool:
        for gamma in args.gammas:
            print(f"\n=== gamma = {gamma} ({args.puzzles} puzzles x {args.runs} runs per distance, "
                  f"cap {args.max_steps} moves) ===")
            print(f"{'dist':>4} | {'model':<8} | {'solved':>7} | {'moves/solve':>11} | {'moves/opt':>9} | "
                  f"{'median':>6} | {'lapses/try':>10} | {'random/move':>11} | {'capped':>6} | secs")
            for dist in args.distances:
                for model in args.models:
                    jobs = [(model, state, gamma, args.max_steps, hash((args.seed, dist, i, r)) & 0xFFFFFFFF)
                            for i, state in enumerate(puzzles[dist]) for r in range(args.runs)]
                    t0 = time.time()
                    results = pool.map(run_one, jobs, chunksize=4)
                    secs = time.time() - t0
                    lengths = [n for ok, n, *_ in results if ok]
                    lapses = statistics.mean(r[2] for r in results)
                    random_share = sum(r[3] for r in results) / sum(r[1] for r in results)
                    capped = sum(r[4] for r in results) / sum(r[1] for r in results)
                    print(f"{dist:>4} | {model:<8} | {len(lengths) / len(results):>7.1%} | "
                          f"{statistics.mean(lengths) if lengths else float('nan'):>11.1f} | "
                          f"{statistics.mean(lengths) / dist if lengths else float('nan'):>9.2f} | "
                          f"{statistics.median(lengths) if lengths else float('nan'):>6.0f} | "
                          f"{lapses:>10.1f} | {random_share:>11.1%} | {capped:>6.1%} | {secs:.0f}", flush=True)


if __name__ == "__main__":
    main()
