"""How much does the apprentice need breadth-first-search labels? Trains
rl.PolicyAgent on the same training puzzles with teachers a person could
have, and compares it with exact BFS labels.

Teachers:
- bfs: exact labels (rl.label_puzzles) - every optimal move, the distance
  to the goal, and how soon an optimal solution puts each car on each
  destination.
- traces: the solver's own solved AND-OR attempts (lapse probability
  GAMMA). Per board, the attempt that finished soonest from it supplies the
  labels: the move it played, the moves it still took (distance), and how
  many moves later it had each car on each destination (dest targets).
- erased: the same attempts with loops cut out first - whenever an attempt
  returns to a board it saw before, the moves in between are dropped.
- exit<k>: expert iteration. Round k's attempts come from the human-like
  solver (AND-OR ordered by round k-1's network), loop-erased, and the
  network keeps training on them.

Each network is scored on held-out puzzles: agreement (its top legal move is
optimal, on BFS-labelled states) and the human-like solver's solve rate at
GAMMA."""
import argparse
import io
import random
import statistics
import time
from multiprocessing import Pool

import torch

import and_or as ao
import rl

BINS = [(1, 8), (9, 12), (13, 16), (17, 20), (21, 30)]
WORKER = {}


# --- trace labels --------------------------------------------------------------------

def _attempt(state, gamma, heuristic, max_steps):
    """One AND-OR attempt: the boards visited (start first) and moves played,
    or None if red doesn't exit within max_steps."""
    boards, moves = [state], []
    while len(moves) < max_steps:
        att = ao.solve(state, heuristic=heuristic, gamma=gamma)
        step = att if type(att) is list else att[1]
        if not step:
            return None
        for move in step:
            state = ao.AndNode(state, *move).apply(state)
            boards.append(state)
            moves.append(move)
        if type(att) is list:
            return boards, moves
    return None


def _init_worker(weights):
    torch.set_num_threads(1)
    WORKER.clear()
    if weights is not None:
        agent = rl.PolicyAgent()
        agent.net.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu"))
        WORKER["agent"] = agent


def trace_puzzle(job):
    """(examples, solved attempts) for one puzzle: `n` attempts, loop-erased
    if `erase`, by plain AND-OR or (in a worker given weights) the human-like
    solver."""
    state, n, gamma, erase, seed, max_steps = job
    random.seed(seed)
    agent = WORKER.get("agent")
    heuristic = agent.heuristic if agent is not None else None
    attempts = []
    for _ in range(n):
        result = _attempt(state, gamma, heuristic, max_steps)
        if result is not None:
            attempts.append(rl.erase_loops(*result) if erase else result)
    return rl.attempt_examples(attempts), len(attempts)


def solve_puzzle(job):
    state, gamma, seed, max_steps, distance = job
    random.seed(seed)
    agent = WORKER.get("agent")
    moves = rl.solve_and_or(state, max_steps, heuristic=agent.heuristic if agent else None, gamma=gamma)
    return moves is not None, (len(moves) / distance if moves else None)


def _weights(agent):
    buffer = io.BytesIO()
    torch.save({k: v.cpu() for k, v in agent.net.state_dict().items()}, buffer)
    return buffer.getvalue()


# --- experiment ------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="../data/rush_nw.txt")
    ap.add_argument("--train-per-bin", type=int, default=200)
    ap.add_argument("--val-per-bin", type=int, default=20)
    ap.add_argument("--test-per-bin", type=int, default=20)
    ap.add_argument("--states-per-puzzle", type=int, default=50, help="BFS labels per puzzle")
    ap.add_argument("--attempts", type=int, default=4, help="AND-OR attempts per training puzzle")
    ap.add_argument("--gamma", type=float, default=0.07)
    ap.add_argument("--exit-rounds", type=int, default=2, help="expert-iteration rounds after 'erased'")
    ap.add_argument("--solve-runs", type=int, default=2, help="human-like solver attempts per test puzzle")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    pool_entries = rl.load_puzzle_pool(args.pool)
    test = rl.sample_by_distance(pool_entries, BINS, args.test_per_bin, seed=args.seed)
    val = rl.sample_by_distance(pool_entries, BINS, args.val_per_bin, seed=args.seed + 1,
                                exclude=[s for _, s in test])
    train = rl.sample_by_distance(pool_entries, BINS, args.train_per_bin, seed=args.seed + 2,
                                  exclude=[s for _, s in test + val])
    val_ex = [e for ex in rl.label_puzzles([s for _, s in val], args.states_per_puzzle, seed=1) for e in ex]
    test_ex = [e for ex in rl.label_puzzles([s for _, s in test], args.states_per_puzzle, seed=2) for e in ex]
    log(f"{len(train)} training puzzles; validation/test scored on BFS labels of "
        f"{len(val)}/{len(test)} other puzzles")

    def score(name, agent, n_states, solved_attempts=None):
        agent.to("cpu")
        m = rl.evaluate_agreement(agent, test_ex)
        weights = _weights(agent)
        jobs = [(s, args.gamma, hash((args.seed, i, r)) & 0xFFFFFFFF, args.max_steps, d)
                for i, (d, s) in enumerate(test) for r in range(args.solve_runs)]
        with Pool(initializer=_init_worker, initargs=(weights,)) as p:
            res = p.map(solve_puzzle, jobs, chunksize=2)
        by_bin = []
        for b_i, (lo, hi) in enumerate(BINS):
            chunk = res[b_i * args.test_per_bin * args.solve_runs:(b_i + 1) * args.test_per_bin * args.solve_runs]
            by_bin.append(f"{sum(ok for ok, _ in chunk) / len(chunk):.0%}")
        extra = f", {solved_attempts} solved attempts" if solved_attempts is not None else ""
        log(f"  {name:<10} {n_states:>6} training boards{extra} | agreement {m['agreement']:.0%} "
            f"(chance {m['chance']:.0%}) | human-like solver solved by distance {' / '.join(by_bin)}")
        agent.to(args.device)

    # Reference rows: plain AND-OR needs no network.
    with Pool(initializer=_init_worker, initargs=(None,)) as p:
        res = p.map(solve_puzzle, [(s, args.gamma, hash((args.seed, i, r)) & 0xFFFFFFFF, args.max_steps, d)
                                   for i, (d, s) in enumerate(test) for r in range(args.solve_runs)], chunksize=2)
    per = args.test_per_bin * args.solve_runs
    log(f"  {'plain AND-OR':<10} (no network) | solved by distance "
        + " / ".join(f"{sum(ok for ok, _ in res[i * per:(i + 1) * per]) / per:.0%}" for i in range(len(BINS))))

    def fit(agent, examples):
        t0 = time.time()
        rl.train_policy(agent, examples, val_ex, epochs=args.epochs)
        return time.time() - t0

    t0 = time.time()
    bfs_ex = [e for ex in rl.label_puzzles([s for _, s in train], args.states_per_puzzle, seed=0) for e in ex]
    torch.manual_seed(args.seed)
    agent = rl.PolicyAgent(device=args.device)
    fit(agent, bfs_ex)
    score("bfs", agent, len(bfs_ex))
    log(f"    ({time.time() - t0:.0f}s)")

    def traces(erase, weights, round_seed):
        jobs = [(s, args.attempts, args.gamma, erase, hash((args.seed, round_seed, i)) & 0xFFFFFFFF, args.max_steps)
                for i, (_, s) in enumerate(train)]
        with Pool(initializer=_init_worker, initargs=(weights,)) as p:
            out = p.map(trace_puzzle, jobs, chunksize=2)
        return [e for ex, _ in out for e in ex], sum(n for _, n in out)

    for name, erase in (("traces", False), ("erased", True)):
        t0 = time.time()
        examples, solved = traces(erase, None, name)
        torch.manual_seed(args.seed)
        agent = rl.PolicyAgent(device=args.device)
        fit(agent, examples)
        score(name, agent, len(examples), solved)
        log(f"    ({time.time() - t0:.0f}s)")

    for k in range(2, args.exit_rounds + 2):
        t0 = time.time()
        examples, solved = traces(True, _weights(agent), f"exit{k}")
        fit(agent, examples)
        score(f"exit{k}", agent, len(examples), solved)
        log(f"    ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
