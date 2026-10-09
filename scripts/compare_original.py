"""One-step action distributions of the original archive's AND-OR model
(original_model.py), my reading of the paper (paper_model.py) and and_or.solve,
compared on the same boards: states visited by rollouts of and_or.solve on
9-car puzzles from the pool, with the car moved last as the original's
same-car rule needs it."""
import argparse
import collections
import random
import statistics

import and_or as ao
import original_model as om
import paper_model as pm
from lib import parse_puzzle_line


def sample_states(path, distances, n_puzzles, n_moves, gamma, seed):
    """[(state, prev_car)] along `n_moves`-long one-move-at-a-time rollouts."""
    rng = random.Random(seed)
    pool = [l for l in open(path) if int(l.split(maxsplit=1)[0]) in distances]
    out = []
    random.seed(seed)
    for line in rng.sample(pool, n_puzzles * 3):
        _, state = parse_puzzle_line(line)
        if len(state) != 9:
            continue
        prev = None
        for _ in range(n_moves):
            if ao._red_steps(state) <= 0:
                break
            out.append((state, prev))
            attempt = ao.solve(state, gamma=gamma, horizon=1)
            moves = attempt if type(attempt) is list else attempt[1]
            if not moves:
                break
            state, prev = ao.AndNode(state, *moves[0]).apply(state), moves[0][0]
        if len({s for s, _ in out}) >= n_puzzles * n_moves:
            break
    return out


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def paper_dist(state, gamma):
    """paper_model's distribution over legal moves, lapse spread uniformly."""
    lapse, leaves, _ = pm.action_distribution(state, gamma)
    legal = ao.legal_moves(state)
    return {m: lapse / len(legal) + leaves.get(m, 0.0) for m in legal}, lapse


def solver_dist(state, gamma, n):
    counts = collections.Counter()
    for _ in range(n):
        attempt = ao.solve(state, gamma=gamma, horizon=1)
        moves = attempt if type(attempt) is list else attempt[1]
        counts[moves[0] if moves else None] += 1
    return {m: c / n for m, c in counts.items() if m is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.07)
    ap.add_argument("--puzzles", type=int, default=120)
    ap.add_argument("--moves", type=int, default=15)
    ap.add_argument("--solver-states", type=int, default=250)
    ap.add_argument("--solver-samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default="../data/rush_nw.txt")
    args = ap.parse_args()
    g = args.gamma

    states = sample_states(args.pool, {5, 9, 12, 14}, args.puzzles, args.moves, g, args.seed)
    print(f"{len(states)} states from rollouts of and_or.solve (gamma={g}, horizon=1), 9-car pool puzzles\n")

    rows = collections.defaultdict(list)
    fb_only_p = fb_only_o = quirk_changes = stray = 0
    for state, prev in states:
        o, o_stop, info = om.action_distribution(state, g, prev)
        o_nq, _, _ = om.action_distribution(state, g, prev, faithful=False)
        o_nox, _, _ = om.action_distribution(state, g, prev, exclude_same_car=False)
        p, p_lapse = paper_dist(state, g)
        stray += info["stray_leaf_mass"] > 0
        rows["tv(original, no-quirks)"].append(tv(o, o_nq))
        rows["tv(original, no same-car rule)"].append(tv(o, o_nox))
        rows["tv(original, paper_model)"].append(tv(o, p))
        rows["stop: original"].append(o_stop)
        rows["lapse: paper_model"].append(p_lapse)
        rows["cycle mass in original"].append(info["cycle_mass"])
        quirk_changes += tv(o, o_nq) > 1e-9
        _, leaves_p, fb = pm.action_distribution(state, g)
        fb_only_p += fb and info["n_leaves"] > 0
        fb_only_o += (not fb) and info["n_leaves"] == 0

    print(f"{'quantity':<34}{'mean':>8}{'median':>8}{'p90':>8}{'share >0.05':>13}")
    for name, xs in rows.items():
        xs = sorted(xs)
        print(f"{name:<34}{statistics.mean(xs):>8.3f}{statistics.median(xs):>8.3f}{xs[int(.9 * len(xs))]:>8.3f}"
              f"{sum(x > 0.05 for x in xs) / len(xs):>13.1%}")
    n = len(states)
    print(f"\nstates where the sum-to-zero quirk changes the original's distribution: {quirk_changes / n:.1%}")
    print(f"states where paper_model needs red's fallback but the original has leaves: {fb_only_p / n:.1%}")
    print(f"states where the original has no leaves but paper_model does:              {fb_only_o / n:.1%}")
    print(f"states where the original's tree has a leaf outside the legal set:         {stray / n:.1%}")

    sub = random.Random(1).sample(states, min(args.solver_states, n))
    d_solver_o, d_solver_p = [], []
    for state, prev in sub:
        s = solver_dist(state, g, args.solver_samples)
        o, _, _ = om.action_distribution(state, g, prev)
        p, _ = paper_dist(state, g)
        d_solver_o.append(tv(s, o))
        d_solver_p.append(tv(s, p))
    print(f"\nand_or.solve first move ({args.solver_samples} samples x {len(sub)} states), same gamma:")
    print(f"  tv(solver, original)    mean {statistics.mean(d_solver_o):.3f}  median {statistics.median(d_solver_o):.3f}")
    print(f"  tv(solver, paper_model) mean {statistics.mean(d_solver_p):.3f}  median {statistics.median(d_solver_p):.3f}")


if __name__ == "__main__":
    main()
